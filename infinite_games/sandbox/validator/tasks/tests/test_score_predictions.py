import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import bittensor as bt
import pandas as pd
import pytest
import requests
import torch
from bittensor.core.metagraph import MetagraphMixin
from bittensor_wallet import Wallet
from freezegun import freeze_time

from infinite_games.sandbox.validator.db.client import DatabaseClient
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.models.event import EventsModel
from infinite_games.sandbox.validator.models.miner import MinersModel
from infinite_games.sandbox.validator.models.prediction import PredictionsModel
from infinite_games.sandbox.validator.tasks.score_predictions import ScorePredictions
from infinite_games.sandbox.validator.utils.logger.logger import InfiniteGamesLogger

CURRENT_DIR = Path(__file__).parent


class TestScorePredictions:
    @pytest.fixture(scope="function", autouse=True)
    async def setup_test_dir(self):
        test_dir = CURRENT_DIR / "test_dir"
        test_dir.mkdir(exist_ok=True)
        yield test_dir
        for file in test_dir.iterdir():
            file.unlink()
        test_dir.rmdir()

    @pytest.fixture(scope="function")
    async def db_client(self):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        logger = MagicMock(spec=InfiniteGamesLogger)

        db_client = DatabaseClient(db_path, logger)

        await db_client.migrate()

        return db_client

    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        return DatabaseOperations(db_client=db_client)

    @pytest.fixture
    def bt_wallet(self):
        hotkey_mock = MagicMock()
        hotkey_mock.sign = MagicMock(side_effect=lambda x: x.encode("utf-8"))
        hotkey_mock.ss58_address = "hotkey2"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)
        bt_wallet.hotkey.ss58_address = "hotkey2"

        return bt_wallet

    @pytest.fixture
    def score_predictions_task(
        self, db_operations: DatabaseOperations, setup_test_dir, bt_wallet: Wallet
    ):
        api_client = IfGamesClient(
            env="test", logger=MagicMock(spec=InfiniteGamesLogger), bt_wallet=bt_wallet
        )
        metagraph = MagicMock(spec=MetagraphMixin)
        config = MagicMock(spec=bt.Config)
        config.neuron = MagicMock()
        config.netuid = 155
        subtensor = MagicMock(spec=bt.Subtensor)

        # Mock metagraph attributes
        metagraph.uids = [1, 2, 3]
        metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
        metagraph.n = 3

        # Mock subtensor methods
        subtensor.min_allowed_weights.return_value = 1  # Set minimum allowed weights
        subtensor.max_weight_limit.return_value = 10  # Set maximum weight limit
        subtensor.network = "mock"

        config.neuron.full_path = setup_test_dir
        with freeze_time("2024-12-27 07:00:00"):
            return ScorePredictions(
                interval_seconds=60.0,
                db_operations=db_operations,
                api_client=api_client,
                metagraph=metagraph,
                config=config,
                wallet=bt_wallet,
                subtensor=subtensor,
            )

    def test_init(self, score_predictions_task: ScorePredictions):
        unit = score_predictions_task

        assert isinstance(unit, ScorePredictions)
        assert unit.current_uids == [1, 2, 3]
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3"]
        assert unit.n_hotkeys == 3
        assert unit.wallet.hotkey.ss58_address == "hotkey2"
        assert unit.vali_uid == 1
        assert unit.is_test is True
        assert unit.state_file == Path(CURRENT_DIR, "test_dir/state.pt")

    def test_load_save_state(self, score_predictions_task: ScorePredictions):
        unit = score_predictions_task

        # init from missing file
        assert unit.state_file.exists() is False

        assert unit.state is not None
        assert unit.state["step"] == 0
        assert unit.state["hotkeys"] == unit.current_hotkeys
        assert isinstance(unit.state["scores"], torch.Tensor)
        assert unit.state["scores"].tolist() == [0.0, 0.0, 0.0]
        assert unit.state["average_scores"].tolist() == [0.0, 0.0, 0.0]
        assert unit.state["previous_average_scores"].tolist() == [0.0, 0.0, 0.0]
        assert unit.state["scoring_iterations"] == 0
        assert unit.state["latest_reset_date"] == datetime(
            2024, 12, 26, 0, 0, 0, 0, tzinfo=timezone.utc
        )

        # adjust and save state
        unit.state["scoring_iterations"] = 10
        unit.state["scores"] = torch.Tensor([1.0, 2.0, 3.0])
        unit.state["latest_reset_date"] = datetime(2024, 12, 27, 0, 0, 0, 0, tzinfo=timezone.utc)

        unit.save_state()
        unit.state = None
        unit.load_state()
        assert unit.state["scoring_iterations"] == 10
        assert isinstance(unit.state["scores"], torch.Tensor)
        assert unit.state["scores"].tolist() == [1.0, 2.0, 3.0]
        assert unit.state["latest_reset_date"] == datetime(
            2024, 12, 27, 0, 0, 0, 0, tzinfo=timezone.utc
        )

    def test_minutes_since_epoch(self, score_predictions_task: ScorePredictions):
        unit = score_predictions_task

        assert unit.minutes_since_epoch(datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc)) == 519841

    def test_align_to_interval(self, score_predictions_task: ScorePredictions):
        unit = score_predictions_task

        assert unit.align_to_interval(519841) == 519840
        assert unit.align_to_interval(520079) == 519840
        assert unit.align_to_interval(520080) == 520080

    @pytest.mark.parametrize(
        "predictions, context, miner_reg_date, expected_rema_brier_score, expected_rema_prediction",
        [
            # Case 0: Miner registered before event and not scored -> 0 Brier score
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [],
                        "interval_agg_prediction": [],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 0,
                },
                datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                0.0,
                -999,
            ),
            # Case 1: Miner registered after event, neutral score
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [],
                        "interval_agg_prediction": [],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 0,
                },
                datetime(2024, 12, 27, 1, 0, 0, 0, timezone.utc),
                0.75,
                0.5,
            ),
            # Case 2: Predictions with valid intervals
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [519840, 520080, 520320],
                        "interval_agg_prediction": [0.8, 0.6, 0.4],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 519840,
                },
                datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                0.8933525,  # Expected Brier score (example calculation)
                0.6992802,  # Weighted average prediction
            ),
            # Case 3: Missing predictions for some intervals
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [519840, 520320],
                        "interval_agg_prediction": [1.0, 1.0],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 519840,
                },
                datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                0.6517926,  # Example with missing intervals handled
                0.6517926,
            ),
            # Case 4: Predictions with exaggerated values
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [519840, 520080, 520320],
                        "interval_agg_prediction": [100.0, 299.0, 1000.0],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 519840,
                },
                datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                1.0,
                1.0,
            ),
        ],
    )
    def test_process_miner_event_score(
        self,
        score_predictions_task,
        predictions,
        context,
        miner_reg_date,
        expected_rema_brier_score,
        expected_rema_prediction,
    ):
        unit = score_predictions_task

        # Mock event
        event = EventsModel(
            unique_event_id="unique_event_id",
            event_id="event_id",
            market_type="market_type",
            event_type="event_type",
            registered_date=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
            description="description",
            outcome="1",
            status=3,
            metadata="metadata",
            created_at=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
        )

        # Add miner's registration mock
        unit.miners_last_reg = pd.DataFrame(
            {
                "uid": ["miner_1"],
                "registered_date": [miner_reg_date],
            }
        )

        # Call the function
        result = unit.process_miner_event_score(event, predictions, context)

        assert result["rema_brier_score"] == pytest.approx(expected_rema_brier_score, rel=1e-6)
        assert result["rema_prediction"] == pytest.approx(expected_rema_prediction, rel=1e-6)

    @pytest.mark.parametrize(
        "predictions, miners_last_reg, expected_scores",
        [
            # Case 0: No miners and no predictions
            (
                [],  # Empty predictions
                pd.DataFrame(columns=["uid", "hotkey", "registered_date"]),
                pd.DataFrame(
                    columns=["miner_uid", "hotkey", "rema_brier_score", "rema_prediction"]
                ),
            ),
            # Case 1: One miner with valid predictions
            (
                [
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=519840,
                        interval_agg_prediction=0.8,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=520080,
                        interval_agg_prediction=0.6,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=520320,
                        interval_agg_prediction=0.4,
                    ),
                ],
                pd.DataFrame(
                    {
                        "uid": [1],
                        "hotkey": ["hotkey_1"],
                        "registered_date": [datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc)],
                    }
                ),
                pd.DataFrame(
                    {
                        "miner_uid": [1],
                        "hotkey": ["hotkey_1"],
                        "rema_brier_score": [0.8933525],
                        "rema_prediction": [0.6992802],
                    }
                ),
            ),
            # Case 2: Multiple miners with mixed predictions
            (
                [
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=519840,
                        interval_agg_prediction=1.0,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=520080,
                        interval_agg_prediction=1.0,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=520320,
                        interval_agg_prediction=1.0,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="2",
                        interval_start_minutes=519840,
                        interval_agg_prediction=0.4,
                    ),
                ],
                pd.DataFrame(
                    {
                        "uid": [1, 2],
                        "hotkey": ["hotkey_1", "hotkey_2"],
                        "registered_date": [
                            datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                            datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "miner_uid": [1, 2],
                        "hotkey": ["hotkey_1", "hotkey_2"],
                        "rema_brier_score": [
                            1.0,
                            0.36742,
                        ],
                        "rema_prediction": [
                            1.0,
                            0.229638,
                        ],
                    }
                ),
            ),
        ],
    )
    def test_score_predictions_method(
        self, score_predictions_task, predictions, miners_last_reg, expected_scores
    ):
        unit = score_predictions_task

        event = EventsModel(
            unique_event_id="unique_event_id",
            event_id="event_id",
            market_type="market_type",
            event_type="event_type",
            registered_date=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
            cutoff=datetime(2024, 12, 27, 12, 0, 0, 0, timezone.utc),
            description="description",
            outcome="1",
            status=3,
            metadata="metadata",
            created_at=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
        )

        # Mock miners_last_reg in the task
        unit.miners_last_reg = miners_last_reg

        # Call the method
        result = unit.score_predictions(event, predictions)

        # Compare the resulting DataFrame to the expected DataFrame
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected_scores.reset_index(drop=True),
            check_exact=False,
        )

    @pytest.mark.parametrize(
        "scores, expected_normalized_scores",
        [
            # Case 0: Normal case with valid scores
            (
                pd.DataFrame({"rema_brier_score": [0.8, 0.6, 0.4]}),
                pd.DataFrame(
                    {
                        "rema_brier_score": [0.8, 0.6, 0.4],
                        "normalized_score": [0.997521, 0.0024726, 6.128982e-6],
                    }
                ),
            ),
            # Case 1: All scores are zero
            (
                pd.DataFrame({"rema_brier_score": [0.0, 0.0, 0.0]}),
                pd.DataFrame(
                    {
                        "rema_brier_score": [0.0, 0.0, 0.0],
                        "normalized_score": [1 / 3, 1 / 3, 1 / 3],
                    }
                ),
            ),
            # Case 2: Another normal case
            (
                pd.DataFrame({"rema_brier_score": [1.0, 0.99, 0.98]}),
                pd.DataFrame(
                    {
                        "rema_brier_score": [1.0, 0.99, 0.98],
                        "normalized_score": [0.4367518, 0.3235537, 0.2396944],
                    }
                ),
            ),
            # Case 3: Another normal case - normalized same as above
            (
                pd.DataFrame({"rema_brier_score": [0.02, 0.01, 0.0]}),
                pd.DataFrame(
                    {
                        "rema_brier_score": [0.02, 0.01, 0.0],
                        "normalized_score": [0.4367518, 0.3235537, 0.2396944],
                    }
                ),
            ),
        ],
    )
    def test_normalize_scores(self, score_predictions_task, scores, expected_normalized_scores):
        unit = score_predictions_task

        result = unit.normalize_scores(scores)

        # Compare the resulting DataFrame to the expected DataFrame
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected_normalized_scores.reset_index(drop=True),
            check_like=True,  # Ignore column order
        )

    @pytest.mark.parametrize(
        "norm_scores, current_uids, current_hotkeys, state, scoring_iterations, expected_updated_scores",
        [
            # Case 0: Normal case with matching miners
            (
                pd.DataFrame(
                    {"miner_uid": [1, 2], "hotkey": ["hk1", "hk2"], "normalized_score": [0.6, 0.4]}
                ),
                torch.tensor([1, 2], dtype=torch.int32),
                ["hk1", "hk2"],
                {
                    "miner_uids": torch.tensor([1, 2], dtype=torch.int32),
                    "hotkeys": ["hk1", "hk2"],
                    "scores": torch.tensor([0.5, 0.3], dtype=torch.float32),
                    "average_scores": torch.tensor([0.5, 0.3], dtype=torch.float32),
                    "previous_average_scores": torch.tensor([0.4, 0.2], dtype=torch.float32),
                },
                1,
                pd.DataFrame(
                    {
                        "miner_uid": [1, 2],
                        "hotkey": ["hk1", "hk2"],
                        "normalized_score": [0.6, 0.4],
                        "eff_scores": [0.52, 0.32],
                        "average_scores": [0.55, 0.35],
                        "previous_average_scores": [0.4, 0.2],
                    }
                ),
            ),
            # Case 1: Miners not in current state
            (
                pd.DataFrame(
                    {"miner_uid": [1, 3], "hotkey": ["hk1", "hk3"], "normalized_score": [0.7, 0.2]}
                ),
                torch.tensor([1, 2], dtype=torch.int32),
                ["hk1", "hk2"],
                {
                    "miner_uids": torch.tensor([1, 2], dtype=torch.int32),
                    "hotkeys": ["hk1", "hk2"],
                    "scores": torch.tensor([0.6, 0.4], dtype=torch.float32),
                    "average_scores": torch.tensor([0.6, 0.4], dtype=torch.float32),
                    "previous_average_scores": torch.tensor([0.5, 0.3], dtype=torch.float32),
                },
                1,
                pd.DataFrame(
                    {
                        "miner_uid": [1],
                        "hotkey": ["hk1"],
                        "normalized_score": [0.7],
                        "eff_scores": [0.62],
                        "average_scores": [0.65],
                        "previous_average_scores": [0.5],
                    }
                ),
            ),
            # Case 2: New miners with no previous state
            (
                pd.DataFrame(
                    {
                        "miner_uid": [1, 3, 4],
                        "hotkey": ["hk1", "hk3", "hk4"],
                        "normalized_score": [1.0, 0.8, 0.2],
                    }
                ),
                torch.tensor([1, 3, 4], dtype=torch.int32),
                ["hk1", "hk3", "hk4"],
                {
                    "miner_uids": torch.tensor(
                        [
                            1,
                        ],
                        dtype=torch.int32,
                    ),
                    "hotkeys": ["hk1"],
                    "scores": torch.tensor([1.0], dtype=torch.float32),
                    "average_scores": torch.tensor([1.0], dtype=torch.float32),
                    "previous_average_scores": torch.tensor([1.0], dtype=torch.float32),
                },
                1,
                pd.DataFrame(
                    {
                        "miner_uid": [1, 3, 4],
                        "hotkey": ["hk1", "hk3", "hk4"],
                        "normalized_score": [1.0, 0.8, 0.2],
                        "eff_scores": [1.0, 0.92, 0.68],
                        "average_scores": [1.0, 0.9, 0.6],
                        "previous_average_scores": [1.0, 1.0, 1.0],
                    }
                ),
            ),
        ],
    )
    def test_update_daily_scores(
        self,
        score_predictions_task,
        norm_scores,
        current_uids,
        current_hotkeys,
        state,
        scoring_iterations,
        expected_updated_scores,
    ):
        # Prepare the task object
        unit = score_predictions_task
        unit.current_uids = current_uids
        unit.current_hotkeys = current_hotkeys
        unit.state = state
        unit.state["scoring_iterations"] = scoring_iterations
        unit.metagraph_lite_sync = MagicMock()

        # Call the method
        result = unit.update_daily_scores(norm_scores)

        # Compare the resulting DataFrame to the expected DataFrame
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected_updated_scores.reset_index(drop=True),
            check_like=True,  # Ignore column order
        )

    @pytest.mark.parametrize(
        "norm_scores_extended, metagraph_data, state, scoring_iterations, expected_state, expected_return",
        [
            # Case 0: Normal case with aligned hotkeys and uids
            (
                pd.DataFrame(
                    {
                        "miner_uid": [1, 2],
                        "hotkey": ["hk1", "hk2"],
                        "eff_scores": [0.6, 0.4],
                        "average_scores": [0.55, 0.35],
                        "previous_average_scores": [0.5, 0.3],
                    }
                ),
                {
                    "uids": torch.tensor([1, 2], dtype=torch.long),
                    "hotkeys": ["hk1", "hk2"],
                },
                {
                    "scores": torch.tensor([], dtype=torch.float32),
                    "average_scores": torch.tensor([], dtype=torch.float32),
                    "previous_average_scores": torch.tensor([], dtype=torch.float32),
                    "miner_uids": torch.tensor([], dtype=torch.long),
                    "hotkeys": [],
                },
                0,
                {
                    "scores": torch.tensor([0.6, 0.4], dtype=torch.float32),
                    "average_scores": torch.tensor([0.55, 0.35], dtype=torch.float32),
                    "previous_average_scores": torch.tensor([0.5, 0.3], dtype=torch.float32),
                    "miner_uids": torch.tensor([1, 2], dtype=torch.long),
                    "hotkeys": ["hk1", "hk2"],
                },
                pd.DataFrame(
                    {
                        "miner_uid": [1, 2],
                        "hotkey": ["hk1", "hk2"],
                        "eff_scores": [0.6, 0.4],
                        "average_scores": [0.55, 0.35],
                        "previous_average_scores": [0.5, 0.3],
                    }
                ),
            ),
            # Case 1: New hotkeys added, with missing previous state
            (
                pd.DataFrame(
                    {
                        "miner_uid": [1],
                        "hotkey": ["hk1"],
                        "eff_scores": [0.6],
                        "average_scores": [0.55],
                        "previous_average_scores": [0.5],
                    }
                ),
                {
                    "uids": torch.tensor([1, 2], dtype=torch.long),
                    "hotkeys": ["hk1", "hk2"],
                },
                {
                    "scores": torch.tensor([0.6], dtype=torch.float32),
                    "average_scores": torch.tensor([0.55], dtype=torch.float32),
                    "previous_average_scores": torch.tensor([0.5], dtype=torch.float32),
                    "miner_uids": torch.tensor([1], dtype=torch.long),
                    "hotkeys": ["hk1"],
                },
                1,
                {
                    "scores": torch.tensor([0.6, 0.0], dtype=torch.float32),
                    "average_scores": torch.tensor([0.55, 0.55], dtype=torch.float32),
                    "previous_average_scores": torch.tensor([0.5, 0.5], dtype=torch.float32),
                    "miner_uids": torch.tensor([1, 2], dtype=torch.long),
                    "hotkeys": ["hk1", "hk2"],
                },
                pd.DataFrame(
                    {
                        "miner_uid": [1, 2],
                        "hotkey": ["hk1", "hk2"],
                        "eff_scores": [0.6, 0.0],
                        "average_scores": [0.55, 0.55],
                        "previous_average_scores": [0.5, 0.5],
                    }
                ),
            ),
            # Case 2: Duplicates in normalized scores
            (
                pd.DataFrame(
                    {
                        "miner_uid": [1, 1],
                        "hotkey": ["hk1", "hk1"],
                        "eff_scores": [0.6, 0.6],
                        "average_scores": [0.55, 0.55],
                        "previous_average_scores": [0.5, 0.5],
                    }
                ),
                {
                    "uids": torch.tensor([1], dtype=torch.long),
                    "hotkeys": ["hk1"],
                },
                {
                    "scores": torch.tensor([], dtype=torch.float32),
                    "average_scores": torch.tensor([], dtype=torch.float32),
                    "previous_average_scores": torch.tensor([], dtype=torch.float32),
                    "miner_uids": torch.tensor([], dtype=torch.long),
                    "hotkeys": [],
                },
                0,
                {
                    "scores": torch.tensor([0.6], dtype=torch.float32),
                    "average_scores": torch.tensor([0.55], dtype=torch.float32),
                    "previous_average_scores": torch.tensor([0.5], dtype=torch.float32),
                    "miner_uids": torch.tensor([1], dtype=torch.long),
                    "hotkeys": ["hk1"],
                },
                pd.DataFrame(
                    {
                        "miner_uid": [1],
                        "hotkey": ["hk1"],
                        "eff_scores": [0.6],
                        "average_scores": [0.55],
                        "previous_average_scores": [0.5],
                    }
                ),
            ),
        ],
    )
    def test_update_state(
        self,
        score_predictions_task,
        norm_scores_extended,
        metagraph_data,
        state,
        scoring_iterations,
        expected_state,
        expected_return,
    ):
        # Mock the metagraph
        unit = score_predictions_task
        unit.metagraph.hotkeys = metagraph_data["hotkeys"]
        unit.metagraph.uids = metagraph_data["uids"]

        # Set initial state
        score_predictions_task.state = state
        score_predictions_task.state["scoring_iterations"] = scoring_iterations

        # Call the method
        result = score_predictions_task.update_state(norm_scores_extended)

        # Verify the state
        assert torch.equal(score_predictions_task.state["scores"], expected_state["scores"])
        assert torch.equal(
            score_predictions_task.state["average_scores"], expected_state["average_scores"]
        )
        assert torch.equal(
            score_predictions_task.state["previous_average_scores"],
            expected_state["previous_average_scores"],
        )
        assert torch.equal(score_predictions_task.state["miner_uids"], expected_state["miner_uids"])
        assert score_predictions_task.state["hotkeys"] == expected_state["hotkeys"]

        # Compare the returned DataFrame to the expected DataFrame
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected_return.reset_index(drop=True),
            check_like=True,  # Ignore column order
        )

    @pytest.mark.parametrize(
        "state, expected_logs",
        [
            # Case 0: Normal case with proper weights
            (
                {
                    "scores": torch.tensor([0.6, 0.4], dtype=torch.float32),
                    "miner_uids": torch.tensor([1, 2], dtype=torch.long),
                    "hotkeys": ["hk1", "hk2"],
                },
                {  # Expected logs
                    "top_10_weights": [0.6 / (0.6 + 0.4), 0.4 / (0.6 + 0.4)],
                    "top_10_uids": [1, 2],
                    "bottom_10_weights": [0.4 / (0.6 + 0.4), 0.6 / (0.6 + 0.4)],
                    "bottom_10_uids": [2, 1],
                },
            ),
            # Case 1: Failed processing weights
            (
                {
                    "scores": torch.tensor([0.7, 0.3], dtype=torch.float32),
                    "miner_uids": torch.tensor([3, 4], dtype=torch.long),
                    "hotkeys": ["hk3", "hk4"],
                },
                {  # Expected logs when processing fails
                    "error_message": "Failed to process the weights - received None.",
                },
            ),
        ],
    )
    def test_set_weights(self, score_predictions_task, state, expected_logs):
        # Mock dependencies
        unit = score_predictions_task
        unit.state = state

        # Mock bt.utils.weight_utils.process_weights_for_netuid
        def mock_process_weights(uids, weights, metagraph, netuid, subtensor):
            if torch.equal(uids, torch.tensor([3, 4])):
                return None, None  # Simulate processing failure
            return uids, weights  # Return processed uids and weights for valid cases

        # mock unit.subtensor.set_weights to return True, "success"
        unit.subtensor.set_weights = MagicMock(
            spec=bt.Subtensor.set_weights, return_value=(True, "success")
        )

        with (
            patch("infinite_games.sandbox.validator.tasks.score_predictions.logger") as mock_logger,
            patch("infinite_games.sandbox.validator.tasks.score_predictions.bt") as bt_mock,
        ):
            # Mock bittensor
            bt_mock.utils.weight_utils.process_weights_for_netuid = mock_process_weights

            # Mock logger
            mock_logger.debug = MagicMock(spec=logging.Logger.debug)
            mock_logger.error = MagicMock(spec=logging.Logger.error)
            mock_logger.warning = MagicMock(spec=logging.Logger.warning)

            # Call the method
            success, msg = unit.set_weights()

            debug_calls = mock_logger.debug.call_args_list
            warning_calls = mock_logger.warning.call_args_list
            error_calls = mock_logger.error.call_args_list

            # Check logs
            if "error_message" in expected_logs:
                assert debug_calls[0].args[0] == "Top 10 and bottom 10 weights."
                assert success is False
                assert msg == "Failed to process the weights."
                assert warning_calls == []
                assert error_calls[0].args[0] == expected_logs["error_message"]
            else:
                assert success
                assert msg == "success"
                assert error_calls == []
                assert warning_calls == []
                assert debug_calls[1].args[0] == "Weights set successfully."
                assert debug_calls[0].args[0] == "Top 10 and bottom 10 weights."
                assert debug_calls[0].kwargs["extra"]["top_10_weights"] == pytest.approx(
                    expected_logs["top_10_weights"], rel=1e-6
                )
                assert debug_calls[0].kwargs["extra"]["top_10_uids"] == expected_logs["top_10_uids"]
                assert debug_calls[0].kwargs["extra"]["bottom_10_weights"] == pytest.approx(
                    expected_logs["bottom_10_weights"], rel=1e-6
                )
                assert (
                    debug_calls[0].kwargs["extra"]["bottom_10_uids"]
                    == expected_logs["bottom_10_uids"]
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "final_scores, expected_body, post_scores_side_effect, expected_logs",
        [
            # Case 0: Successful export
            (
                pd.DataFrame(
                    {
                        "miner_uid": [1, 2],
                        "hotkey": ["hk1", "hk2"],
                        "rema_brier_score": [0.6, 0.4],
                        "rema_prediction": [0.55, 0.45],
                        "eff_scores": [0.7, 0.3],
                    }
                ),
                {  # Expected body
                    "results": [
                        {
                            "miner_uid": 1,
                            "miner_hotkey": "hk1",
                            "miner_score": 0.6,
                            "prediction": 0.55,
                            "miner_effective_score": 0.7,
                            "event_id": "unique_event_id",
                            "provider_type": "market_type",
                            "title": "Event Description",
                            "description": "Event Description",
                            "category": "event",
                            "start_date": "2024-12-25T12:00:00+00:00",
                            "end_date": "2024-12-26T12:00:00+00:00",
                            "resolve_date": "2024-12-26T12:00:00+00:00",
                            "settle_date": "2024-12-27T12:00:00+00:00",
                            "answer": 1.0,
                            "validator_hotkey": "validator_hotkey",
                            "validator_uid": 1,
                            "metadata": "metadata",
                            "spec_version": "1.0.0",
                        },
                        {
                            "miner_uid": 2,
                            "miner_hotkey": "hk2",
                            "miner_score": 0.4,
                            "prediction": 0.45,
                            "miner_effective_score": 0.3,
                            "event_id": "unique_event_id",
                            "provider_type": "market_type",
                            "title": "Event Description",
                            "description": "Event Description",
                            "category": "event",
                            "start_date": "2024-12-25T12:00:00+00:00",
                            "end_date": "2024-12-26T12:00:00+00:00",
                            "resolve_date": "2024-12-26T12:00:00+00:00",
                            "settle_date": "2024-12-27T12:00:00+00:00",
                            "answer": 1.0,
                            "validator_hotkey": "validator_hotkey",
                            "validator_uid": 1,
                            "metadata": "metadata",
                            "spec_version": "1.0.0",
                        },
                    ]
                },
                None,  # No exception
                [],  # No error logs
            ),
            # Case 1: RequestException raised during post_scores
            (
                pd.DataFrame(
                    {
                        "miner_uid": [1],
                        "hotkey": ["hk1"],
                        "rema_brier_score": [0.5],
                        "rema_prediction": [0.5],
                        "eff_scores": [0.5],
                    }
                ),
                {},  # Body is irrelevant since post_scores fails
                requests.exceptions.RequestException,  # Simulate failure
                [
                    "Retrying export scores.",
                    "Retrying export scores.",
                    "Failed to export scores.",
                ],  # Expected error logs
            ),
        ],
    )
    @patch("backoff.on_exception", lambda *args, **kwargs: lambda func: func)
    async def test_export_scores(
        self,
        score_predictions_task,
        final_scores,
        expected_body,
        post_scores_side_effect,
        expected_logs,
    ):
        # TODO: speed up this test
        # Mock dependencies
        unit = score_predictions_task
        unit.spec_version = "1.0.0"
        unit.wallet.get_hotkey = MagicMock(return_value=MagicMock(ss58_address="validator_hotkey"))

        event = EventsModel(
            unique_event_id="unique_event_id",
            event_id="event_id",
            market_type="market_type",
            event_type="event_type",
            starts=datetime(2024, 12, 25, 12, 0, 0, 0, timezone.utc),
            resolve_date=datetime(2024, 12, 26, 12, 0, 0, 0, timezone.utc),
            cutoff=datetime(2024, 12, 27, 12, 0, 0, 0, timezone.utc),
            description="Event Description",
            outcome="1",
            status=3,
            metadata="metadata",
            created_at=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
        )

        unit.api_client.post_scores = AsyncMock(side_effect=post_scores_side_effect)

        with patch(
            "infinite_games.sandbox.validator.tasks.score_predictions.logger"
        ) as mock_logger:
            # Mock logger methods
            mock_logger.warning = MagicMock()
            mock_logger.error = MagicMock()

            # Call the method
            if post_scores_side_effect:
                with pytest.raises(post_scores_side_effect):
                    await unit.export_scores(event, final_scores)
            else:
                await unit.export_scores(event, final_scores)

            # Validate the API call
            if not post_scores_side_effect:
                unit.api_client.post_scores.assert_awaited_once_with(scores=expected_body)

            # Check logs
            if expected_logs:
                warning_calls = mock_logger.warning.call_args_list
                assert len(warning_calls) == 2
                assert warning_calls[0].args[0] == expected_logs[0]
                assert warning_calls[1].args[0] == expected_logs[1]

                error_calls = mock_logger.error.call_args_list
                assert len(error_calls) == 1
                assert error_calls[0].args[0] == expected_logs[2]
            else:
                mock_logger.error.assert_not_called()

    @pytest.mark.parametrize(
        "state, n_hotkeys, current_time, reset_interval, expected_state, expected_logs",
        [
            # Case 0: No reset needed (within interval)
            (
                {
                    "latest_reset_date": datetime(2024, 12, 26, 0, 0, 0, tzinfo=timezone.utc),
                    "average_scores": torch.tensor([0.6, 0.4], dtype=torch.float32),
                    "scoring_iterations": 10,
                },
                2,
                datetime(2024, 12, 26, 23, 59, 0, tzinfo=timezone.utc),
                86400,  # Reset interval: 1 day
                {  # Expected state: unchanged
                    "latest_reset_date": datetime(2024, 12, 26, 0, 0, 0, tzinfo=timezone.utc),
                    "average_scores": torch.tensor([0.6, 0.4], dtype=torch.float32),
                    "scoring_iterations": 10,
                },
                [],  # No logs
            ),
            # Case 1: Reset needed (scores not zero)
            (
                {
                    "latest_reset_date": datetime(2024, 12, 25, 0, 0, 0, tzinfo=timezone.utc),
                    "average_scores": torch.tensor([0.6, 0.4], dtype=torch.float32),
                    "scoring_iterations": 10,
                },
                2,
                datetime(2024, 12, 26, 1, 0, 0, tzinfo=timezone.utc),
                86400,  # Reset interval: 1 day
                {  # Expected state: reset
                    "latest_reset_date": datetime(2024, 12, 26, 0, 0, 0, tzinfo=timezone.utc),
                    "average_scores": torch.tensor([0.0, 0.0], dtype=torch.float32),
                    "scoring_iterations": 0,
                },
                ["Resetting daily scores."],  # Debug log
            ),
            # Case 2: Reset needed (scores are zero)
            (
                {
                    "latest_reset_date": datetime(2024, 12, 25, 0, 0, 0, tzinfo=timezone.utc),
                    "average_scores": torch.tensor([0.0, 0.0], dtype=torch.float32),
                    "scoring_iterations": 10,
                },
                2,
                datetime(2024, 12, 26, 1, 0, 0, tzinfo=timezone.utc),
                86400,  # Reset interval: 1 day
                {  # Expected state: unchanged
                    "latest_reset_date": datetime(2024, 12, 25, 0, 0, 0, tzinfo=timezone.utc),
                    "average_scores": torch.tensor([0.0, 0.0], dtype=torch.float32),
                    "scoring_iterations": 10,
                },
                ["Reset daily scores: average scores are 0, not resetting."],  # Error log
            ),
        ],
    )
    @patch("infinite_games.sandbox.validator.tasks.score_predictions.logger")
    def test_check_reset_daily_scores(
        self,
        mock_logger,
        score_predictions_task,
        state,
        n_hotkeys,
        current_time,
        reset_interval,
        expected_state,
        expected_logs,
    ):
        # Mock dependencies
        unit = score_predictions_task
        unit.state = state
        init_latest_reset = unit.state["latest_reset_date"]
        unit.n_hotkeys = n_hotkeys

        # Mock datetime.now
        with patch(
            "infinite_games.sandbox.validator.tasks.score_predictions.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = current_time
            mock_datetime.now.timezone = timezone.utc

            # Mock save_state
            unit.save_state = MagicMock()

            # Call the method
            unit.check_reset_daily_scores()

            # Validate the entire state
            assert (
                unit.state["latest_reset_date"] == expected_state["latest_reset_date"]
            ), "Mismatch in latest_reset_date"
            assert torch.equal(
                unit.state["average_scores"], expected_state["average_scores"]
            ), "Mismatch in average_scores"
            assert (
                unit.state["scoring_iterations"] == expected_state["scoring_iterations"]
            ), "Mismatch in scoring_iterations"

            # Validate logs
            if expected_logs:
                if "Resetting daily scores." in expected_logs:
                    expected_seconds_since_reset = (
                        current_time - init_latest_reset
                    ).total_seconds()

                    mock_logger.debug.assert_called_once_with(
                        "Resetting daily scores.",
                        extra={"seconds_since_reset": expected_seconds_since_reset},
                    )
                if "Reset daily scores: average scores are 0, not resetting." in expected_logs:
                    mock_logger.error.assert_called_once_with(
                        "Reset daily scores: average scores are 0, not resetting."
                    )
            else:
                mock_logger.debug.assert_not_called()
                mock_logger.error.assert_not_called()

            # Validate save_state call
            if "Resetting daily scores." in expected_logs:
                unit.save_state.assert_called_once()
            else:
                unit.save_state.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "predictions, expected_logs",
        [
            # Case 0: Event has no cutoff date
            (
                [],
                [
                    {
                        "level": "error",
                        "message": "Event has no cutoff date.",
                        "extra": {"event_id": "unique_event_id"},
                    }
                ],
            ),
            # Case 1: No predictions for event
            (
                [],
                [
                    {
                        "level": "warning",
                        "message": "There are no predictions for a settled event.",
                        "extra": {
                            "event_id": "unique_event_id",
                            "event_cutoff": datetime(2024, 12, 26, 12, 0, 0, 0, timezone.utc),
                        },
                    }
                ],
            ),
            # Case 2: Normal flow
            (
                [
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=0,
                        interval_agg_prediction=0.6,
                        interval_count=10,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="2",
                        interval_start_minutes=0,
                        interval_agg_prediction=0.4,
                        interval_count=10,
                    ),
                ],
                [
                    {
                        "level": "debug",
                        "message": "Scores calculated, sample below.",
                    },
                ],
            ),
        ],
    )
    @patch("infinite_games.sandbox.validator.tasks.score_predictions.logger")
    async def test_score_event(
        self, mock_logger, score_predictions_task, predictions, expected_logs
    ):
        event = EventsModel(
            unique_event_id="unique_event_id",
            event_id="event_id",
            market_type="market_type",
            event_type="event_type",
            starts=datetime(2024, 12, 25, 12, 0, 0, 0, timezone.utc),
            resolve_date=datetime(2024, 12, 26, 12, 0, 0, 0, timezone.utc),
            cutoff=datetime(2024, 12, 27, 12, 0, 0, 0, timezone.utc),
            description="Event Description",
            outcome="1",
            status=3,
            metadata="metadata",
            created_at=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
        )

        if expected_logs[0]["level"] == "error":
            event.cutoff = None

        # Mock dependencies
        unit = score_predictions_task

        # Mock methods
        unit.db_operations.get_predictions_for_scoring = AsyncMock(return_value=predictions)
        unit.db_operations.mark_event_as_processed = MagicMock()
        unit.db_operations.mark_event_as_exported = MagicMock()
        unit.save_state = MagicMock()
        unit.set_weights = MagicMock(return_value=(True, "success"))
        unit.export_scores = AsyncMock()

        unit.score_predictions = MagicMock(
            return_value=pd.DataFrame(
                {
                    "miner_uid": [1, 2],
                    "rema_brier_score": [0.6, 0.4],
                    "rema_prediction": [0.55, 0.45],
                }
            )
        )
        unit.normalize_scores = MagicMock(
            return_value=pd.DataFrame({"miner_uid": [1, 2], "normalized_score": [0.6, 0.4]})
        )
        unit.update_daily_scores = MagicMock(
            return_value=pd.DataFrame({"miner_uid": [1, 2], "eff_scores": [0.7, 0.3]})
        )
        unit.update_state = MagicMock(
            return_value=pd.DataFrame({"miner_uid": [1, 2], "aligned_scores": [0.7, 0.3]})
        )
        unit.check_reset_daily_scores = MagicMock()

        # Call the method
        await unit.score_event(event, predictions)

        # Check logs
        error_calls = mock_logger.error.call_args_list
        warning_calls = mock_logger.warning.call_args_list
        debug_calls = mock_logger.debug.call_args_list
        for i, log in enumerate(expected_logs):
            if log["level"] == "error":
                assert len(error_calls) == 1
                assert len(warning_calls) == 1  # there are no predictions
                assert len(debug_calls) == 0
                assert error_calls[0].args[0] == log["message"]
                assert error_calls[0].kwargs["extra"]["event_id"] == log["extra"]["event_id"]
            elif log["level"] == "warning":
                assert len(warning_calls) == 1
                assert len(error_calls) == 0
                assert len(debug_calls) == 0
                assert warning_calls[0].args[0] == log["message"]
                assert warning_calls[0].kwargs["extra"]["event_id"] == log["extra"]["event_id"]
                assert (
                    warning_calls[0].kwargs["extra"]["event_cutoff"] == log["extra"]["event_cutoff"]
                )
            elif log["level"] == "debug":
                assert len(debug_calls) == 1
                assert len(error_calls) == 0
                assert len(warning_calls) == 0
                assert debug_calls[i].args[0] == expected_logs[i]["message"]

        # Assertions for normal flow
        if predictions:
            unit.db_operations.mark_event_as_processed.assert_called_once_with(
                unique_event_id=event.unique_event_id
            )
            unit.save_state.assert_called_once()
            unit.set_weights.assert_called_once()
            unit.export_scores.assert_awaited_once_with(
                event=event, final_scores=unit.update_state.return_value
            )
            unit.db_operations.mark_event_as_exported.assert_called_once_with(
                unique_event_id=event.unique_event_id
            )
            unit.check_reset_daily_scores.assert_called_once()
        else:
            unit.db_operations.mark_event_as_processed.assert_not_called()
            unit.save_state.assert_not_called()
            unit.set_weights.assert_not_called()
            unit.export_scores.assert_not_called()
            unit.db_operations.mark_event_as_exported.assert_not_called()
            unit.check_reset_daily_scores.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "miners_last_reg_rows, events_to_score, expected_logs",
        [
            # Case 0: No events to score
            (
                [
                    MinersModel(
                        miner_hotkey="hk1",
                        miner_uid="1",
                        registered_date=datetime(2024, 12, 1, 0, 0, 0, tzinfo=timezone.utc),
                    ),
                ],
                [],
                [
                    {"level": "debug", "message": "No events to score."},
                ],
            ),
            # Case 1: Events to score found
            (
                [
                    MinersModel(
                        miner_hotkey="hk1",
                        miner_uid="1",
                        registered_date=datetime(2024, 12, 1, 0, 0, 0, tzinfo=timezone.utc),
                    ),
                ],
                [
                    EventsModel(
                        unique_event_id="event_1",
                        event_id="event_id_1",
                        market_type="binary",
                        event_type="sports",
                        registered_date=datetime(2024, 12, 20, 0, 0, 0, tzinfo=timezone.utc),
                        description="Event 1 Description",
                        starts=datetime(2024, 12, 21, 12, 0, 0, tzinfo=timezone.utc),
                        resolve_date=None,
                        outcome=None,
                        local_updated_at=None,
                        status=3,
                        metadata="metadata",
                        processed=False,
                        exported=False,
                        created_at=datetime(2024, 12, 20, 0, 0, 0, tzinfo=timezone.utc),
                        cutoff=datetime(2024, 12, 27, 12, 0, 0, tzinfo=timezone.utc),
                        end_date=None,
                    ),
                    EventsModel(
                        unique_event_id="event_2",
                        event_id="event_id_2",
                        market_type="binary",
                        event_type="sports",
                        registered_date=datetime(2024, 12, 20, 0, 0, 0, tzinfo=timezone.utc),
                        description="Event 2 Description",
                        starts=datetime(2024, 12, 21, 12, 0, 0, tzinfo=timezone.utc),
                        resolve_date=None,
                        outcome=None,
                        local_updated_at=None,
                        status=3,
                        metadata="metadata",
                        processed=False,
                        exported=False,
                        created_at=datetime(2024, 12, 20, 0, 0, 0, tzinfo=timezone.utc),
                        cutoff=datetime(2024, 12, 27, 12, 0, 0, tzinfo=timezone.utc),
                        end_date=None,
                    ),
                ],
                [
                    {
                        "level": "debug",
                        "message": "Found events to score.",
                        "extra": {"n_events": 2},
                    },
                    {
                        "level": "debug",
                        "message": "All events scored, weights set, scores exported.",
                    },
                ],
            ),
        ],
    )
    @patch("infinite_games.sandbox.validator.tasks.score_predictions.logger")
    async def test_run(
        self,
        mock_logger,
        score_predictions_task,
        miners_last_reg_rows,
        events_to_score,
        expected_logs,
    ):
        # Mock dependencies
        unit = score_predictions_task

        # Mock database operations
        unit.db_operations.get_miners_last_registration = AsyncMock(
            return_value=miners_last_reg_rows
        )
        unit.db_operations.get_events_for_scoring = AsyncMock(return_value=events_to_score)

        # Mock helper methods
        unit.score_event = AsyncMock()

        # Call the method
        await unit.run()

        # Validate the logger calls
        for log in expected_logs:
            if log["level"] == "debug":
                if "extra" in log:
                    mock_logger.debug.assert_any_call(log["message"], extra=log["extra"])
                else:
                    mock_logger.debug.assert_any_call(log["message"])

        # Additional assertions for events
        if events_to_score:
            assert unit.score_event.call_count == len(events_to_score)
            for event in events_to_score:
                unit.score_event.assert_any_await(event)
        else:
            unit.score_event.assert_not_called()