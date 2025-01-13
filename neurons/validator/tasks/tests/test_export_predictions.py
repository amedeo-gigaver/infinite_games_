import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.prediction import PredictionExportedStatus
from neurons.validator.tasks.export_predictions import ExportPredictions
from neurons.validator.utils.common.interval import (
    get_interval_iso_datetime,
    get_interval_start_minutes,
)
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestExportPredictions:
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
        hotkey_mock.ss58_address = "ss58_address"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)

        return bt_wallet

    @pytest.fixture
    def export_predictions_task(self, db_operations: DatabaseOperations, bt_wallet: Wallet):
        mocked_logger = MagicMock(spec=InfiniteGamesLogger)

        api_client = IfGamesClient(env="test", logger=mocked_logger, bt_wallet=bt_wallet)

        return ExportPredictions(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
            batch_size=1,
            validator_uid=0,
            validator_hotkey="validator_hotkey_test",
            logger=mocked_logger,
        )

    def test_parse_predictions_for_exporting(self, export_predictions_task: ExportPredictions):
        # Test with a single prediction
        predictions = [
            (
                1,  # ROWID (unused in function)
                "event123",  # unique_event_id
                "miner_key_1",  # miner_hotkey
                "miner_uid_1",  # miner_uid
                "weather",  # event_type
                0.75,  # predicted_outcome
                120,  # interval_start_minutes
                0.8,  # interval_agg_prediction
                5,  # interval_count
            )
        ]

        result = export_predictions_task.parse_predictions_for_exporting(predictions)

        assert "submissions" in result
        assert "events" in result
        assert result["events"] is None
        assert len(result["submissions"]) == 1

        submission = result["submissions"][0]
        assert submission["unique_event_id"] == "event123"
        assert submission["provider_type"] == "weather"
        assert submission["prediction"] == 0.75
        assert submission["interval_start_minutes"] == 120
        assert submission["interval_agg_prediction"] == 0.8
        assert submission["interval_agg_count"] == 5
        assert submission["miner_hotkey"] == "miner_key_1"
        assert submission["miner_uid"] == "miner_uid_1"
        assert submission["validator_hotkey"] == "validator_hotkey_test"
        assert submission["validator_uid"] == 0
        assert submission["title"] is None
        assert submission["outcome"] is None

        # Verify datetime calculation
        expected_datetime = datetime(2024, 1, 1, 2, 0, 0, 0, tzinfo=timezone.utc).isoformat()
        assert submission["interval_datetime"] == expected_datetime

    def test_parse_predictions_for_exporting_multiple_predictions(
        self, export_predictions_task: ExportPredictions
    ):
        # Test with multiple predictions
        predictions = [
            (0, "event1", "miner1", "uid1", "weather", 0.75, 120, 0.8, 5),
            (0, "event2", "miner2", "uid2", "sports", 0.25, 240, 0.3, 3),
        ]

        result = export_predictions_task.parse_predictions_for_exporting(predictions)

        assert len(result["submissions"]) == 2
        assert result["submissions"][0]["unique_event_id"] == "event1"
        assert result["submissions"][1]["unique_event_id"] == "event2"

    async def test_run(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
        export_predictions_task: ExportPredictions,
    ):
        """Test the run method when there are predictions to export."""

        # Mock API client
        export_predictions_task.api_client = AsyncMock(spec=IfGamesClient)

        events = [
            (
                "unique_event_id_1",
                "event_1",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                "outcome1",
                "status1",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
            (
                "unique_event_id_2",
                "event_2",
                "truncated_market2",
                "market_2",
                "desc2",
                "2024-12-02",
                "2024-12-03",
                "outcome2",
                "status2",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
            (
                "unique_event_id_3",
                "event_3",
                "truncated_market3",
                "market_3",
                "desc3",
                "2024-12-02",
                "2024-12-03",
                "outcome3",
                "status3",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
        ]

        current_interval_minutes = get_interval_start_minutes()
        previous_interval_minutes = current_interval_minutes - 1

        predictions = [
            (
                "unique_event_id_1",
                "neuronHotkey_1",
                "neuronUid_1",
                "1",
                previous_interval_minutes,
                "1",
                1,
                "1",
            ),
            (
                "unique_event_id_2",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                previous_interval_minutes,
                "1",
                1,
                "1",
            ),
            (
                "unique_event_id_3",
                "neuronHotkey_3",
                "neuronUid_3",
                "1",
                current_interval_minutes,
                "1",
                1,
                "1",
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Act
        await export_predictions_task.run()

        # Assert
        assert export_predictions_task.api_client.post_predictions.call_count == 2

        mock_calls = export_predictions_task.api_client.post_predictions.mock_calls
        first_call = mock_calls[0]
        second_call = mock_calls[1]

        assert first_call == (
            "__call__",
            {
                "predictions": {
                    "events": None,
                    "submissions": [
                        {
                            "unique_event_id": "unique_event_id_1",
                            "provider_type": "market_1",
                            "prediction": "1",
                            "interval_start_minutes": previous_interval_minutes,
                            "interval_agg_prediction": 1.0,
                            "interval_agg_count": 1,
                            "interval_datetime": get_interval_iso_datetime(
                                previous_interval_minutes
                            ),
                            "miner_hotkey": "neuronHotkey_1",
                            "miner_uid": "neuronUid_1",
                            "validator_hotkey": "validator_hotkey_test",
                            "validator_uid": 0,
                            "title": None,
                            "outcome": None,
                        }
                    ],
                }
            },
        )
        assert second_call == (
            "__call__",
            {
                "predictions": {
                    "events": None,
                    "submissions": [
                        {
                            "unique_event_id": "unique_event_id_2",
                            "provider_type": "market_2",
                            "prediction": "1",
                            "interval_start_minutes": previous_interval_minutes,
                            "interval_agg_prediction": 1.0,
                            "interval_agg_count": 1,
                            "interval_datetime": get_interval_iso_datetime(
                                previous_interval_minutes
                            ),
                            "miner_hotkey": "neuronHotkey_2",
                            "miner_uid": "neuronUid_2",
                            "validator_hotkey": "validator_hotkey_test",
                            "validator_uid": 0,
                            "title": None,
                            "outcome": None,
                        }
                    ],
                }
            },
        )

        result = await db_client.many(
            """
                SELECT exported FROM predictions
            """
        )

        assert len(result) == 3
        assert result[0][0] == PredictionExportedStatus.EXPORTED
        assert result[1][0] == PredictionExportedStatus.EXPORTED
        assert result[2][0] == PredictionExportedStatus.NOT_EXPORTED

    async def test_run_no_predictions(self, export_predictions_task: ExportPredictions):
        # Mock API client
        export_predictions_task.api_client = AsyncMock(spec=IfGamesClient)

        # Act
        await export_predictions_task.run()

        # Assert
        export_predictions_task.api_client.post_predictions.assert_not_called()
