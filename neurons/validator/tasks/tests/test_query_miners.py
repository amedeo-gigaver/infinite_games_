import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import numpy as np
import pytest
import torch
from bittensor.core.chain_data import AxonInfo
from bittensor.core.dendrite import DendriteMixin
from bittensor.core.metagraph import MetagraphMixin

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventStatus
from neurons.validator.models.event_prediction_synapse import EventPredictionSynapse
from neurons.validator.tasks.query_miners import QueryMiners
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestQueryMiners:
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
    def query_miners_task(self, db_operations: DatabaseOperations):
        dendrite = MagicMock(spec=DendriteMixin)
        metagraph = MagicMock(spec=MetagraphMixin)
        logger = MagicMock(spec=InfiniteGamesLogger)

        return QueryMiners(
            interval_seconds=60.0,
            db_operations=db_operations,
            dendrite=dendrite,
            metagraph=metagraph,
            logger=logger,
        )

    def test_get_axons(
        self,
        query_miners_task: QueryMiners,
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = np.array([1, 2, 3])  # UIDs in the metagraph
        query_miners_task.metagraph.axons = {
            1: MagicMock(is_serving=True),  # Valid axon
            2: MagicMock(is_serving=False),  # Not serving
            3: None,  # No axon at this UID
        }

        # Call the method
        result = query_miners_task.get_axons()

        # Assertions
        # Only UID 1 should be included because it's serving
        assert result == {1: query_miners_task.metagraph.axons[1]}

    def test_get_axons_empty_metagraph(
        self,
        query_miners_task: QueryMiners,
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = np.array([])  # No UIDs in the metagraph

        # Call the method
        result = query_miners_task.get_axons()

        # Assertions
        # No axons should be included
        assert result == {}

    def test_make_predictions_synapse(self, query_miners_task: QueryMiners):
        events_from_db = [
            (
                "event1",  # event_id
                "ifgames",  # market_type
                "Test match",  # description
                "2012-12-02T14:30:00+00:00",  # cutoff
                "2012-12-02T14:30:00+00:00",  # resolve_date
                "2012-12-03T14:30:00+00:00",  # end_date
                json.dumps(
                    {
                        "market_type": "acled",
                    }
                ),  # metadata
            ),
            (
                "event2",  # event_id
                "ifgames",  # market_type
                "Test match 2",  # description
                "2012-12-02T14:30:00+00:00",  # cutoff
                "2012-12-02T14:30:00+00:00",  # resolve_date
                "2012-12-03T14:30:00+00:00",  # end_date
                json.dumps(
                    {
                        "market_type": "azuro",
                    }
                ),  # metadata
            ),
        ]

        synapse = query_miners_task.make_predictions_synapse(events=events_from_db)

        assert isinstance(synapse, EventPredictionSynapse)
        assert len(synapse.events) == 2

        # Assert event 1
        expected_key = "ifgames-event1"

        assert expected_key in synapse.events

        event_data = synapse.events[expected_key]

        assert event_data["event_id"] == "event1"
        assert event_data["market_type"] == "acled"
        assert event_data["description"] == "Test match"
        assert event_data["cutoff"] == 1354458600
        assert event_data["probability"] is None
        assert event_data["miner_answered"] is False
        assert event_data["starts"] is None
        assert event_data["resolve_date"] == 1354458600
        assert event_data["end_date"] == 1354545000

        # Assert event 2
        expected_key = "ifgames-event2"

        assert expected_key in synapse.events

        event_data = synapse.events[expected_key]

        assert event_data["event_id"] == "event2"
        assert event_data["market_type"] == "azuro"
        assert event_data["description"] == "Test match 2"
        assert event_data["cutoff"] == 1354458600
        assert event_data["probability"] is None
        assert event_data["miner_answered"] is False
        assert event_data["starts"] == 1354458600
        assert event_data["resolve_date"] == 1354458600
        assert event_data["end_date"] == 1354545000

    def test_make_predictions_synapse_empty_events(self, query_miners_task: QueryMiners):
        """Test handling of empty events"""
        events = []

        result = query_miners_task.make_predictions_synapse(events=events)

        assert isinstance(result, EventPredictionSynapse)
        assert len(result.events) == 0

    def test_parse_neuron_predictions(self, query_miners_task: QueryMiners):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = np.array([1, 2])
        query_miners_task.metagraph.axons = {
            1: MagicMock(hotkey="hotkey1"),
            2: MagicMock(hotkey="hotkey2"),
        }

        interval_start_minutes = 12345
        block = 54321
        uid = 2

        neuron_predictions = EventPredictionSynapse(
            events={
                "ifgames-event1": {
                    "event_id": "event1",
                    "market_type": "acled",
                    "probability": 0.5,
                    "miner_answered": True,
                    "description": "Test match",
                    "cutoff": 1354458600,
                    "starts": None,
                    "resolve_date": 1354458600,
                    "end_date": 1354545000,
                },
                "ifgames-event2": {
                    "event_id": "event2",
                    "market_type": "azuro",
                    "probability": 0.75,
                    "miner_answered": False,
                    "description": "Test match 2",
                    "cutoff": 1354458600,
                    "starts": 1354458600,
                    "resolve_date": 1354458600,
                    "end_date": 1354545000,
                },
                "ifgames-event3": {
                    "event_id": "event3",
                    "market_type": "azuro",
                    # None predictions are dropped
                    "probability": None,
                    "miner_answered": False,
                    "description": "Test match 2",
                    "cutoff": 1354458600,
                    "starts": 1354458600,
                    "resolve_date": 1354458600,
                    "end_date": 1354545000,
                },
            }
        )

        result = query_miners_task.parse_neuron_predictions(
            block=block,
            interval_start_minutes=interval_start_minutes,
            uid=uid,
            neuron_predictions=neuron_predictions,
        )

        assert result == [
            (
                "ifgames-event1",
                query_miners_task.metagraph.axons[uid].hotkey,
                uid,
                0.5,
                interval_start_minutes,
                0.5,
                block,
                0.5,
            ),
            (
                "ifgames-event2",
                query_miners_task.metagraph.axons[uid].hotkey,
                uid,
                0.75,
                interval_start_minutes,
                0.75,
                block,
                0.75,
            ),
        ]

    async def test_query_neurons(self, query_miners_task: QueryMiners):
        synapse = EventPredictionSynapse(events={})

        axons_by_uid = {"1": MagicMock(spec=AxonInfo), "50": MagicMock(spec=AxonInfo)}

        query_miners_task.dendrite.forward = AsyncMock(return_value=[synapse, synapse])

        response = await query_miners_task.query_neurons(axons_by_uid=axons_by_uid, synapse=synapse)

        # Assertions
        assert len(response) == 2
        assert response["1"] == synapse
        assert response["50"] == synapse

    async def test_store_miners(self, db_client: DatabaseClient, query_miners_task: QueryMiners):
        block = 12345
        axons = {"uid_1": MagicMock(hotkey="hotkey_1", ip="ip_1")}

        await query_miners_task.store_miners(block=block, axons=axons)

        # Assertions
        result = await db_client.many(
            """
                SELECT
                    miner_uid,
                    miner_hotkey,
                    node_ip,
                    registered_date,
                    last_updated,
                    blocktime
                FROM
                    miners
            """
        )

        assert len(result) == 1
        assert result[0] == ("uid_1", "hotkey_1", "ip_1", "2024-01-01T00:00:00", ANY, block)

        # Conflicting insert
        new_block = 23456
        new_axons = {
            "uid_1": MagicMock(hotkey="hotkey_1", ip="ip_2"),
            "uid_2": MagicMock(hotkey="hotkey_2", ip="ip_3"),
        }

        # Wait 1s so last updated is different
        await asyncio.sleep(1)

        await query_miners_task.store_miners(block=new_block, axons=new_axons)

        # Assertions
        result_2 = await db_client.many(
            """
                SELECT
                    miner_uid,
                    miner_hotkey,
                    node_ip,
                    registered_date,
                    last_updated,
                    blocktime
                FROM
                    miners
            """
        )

        assert len(result_2) == 2
        assert result_2[0] == ("uid_1", "hotkey_1", "ip_2", "2024-01-01T00:00:00", ANY, new_block)
        assert result_2[1] == ("uid_2", "hotkey_2", "ip_3", ANY, ANY, new_block)

        # Check that last update has been updated
        assert result[0][4] != result_2[0][4]
        # Check that the registered_date is not anymore 2024-01-01T00:00:00
        assert result_2[1][4] != "2024-01-01T00:00:00"

    async def test_store_predictions(self, query_miners_task: QueryMiners):
        # Set up mocks
        query_miners_task.db_operations.upsert_predictions = AsyncMock()
        query_miners_task.metagraph.axons = {
            "uid_1": MagicMock(hotkey="hotkey_1"),
            "uid_2": MagicMock(hotkey="hotkey_2"),
        }

        interval_start_minutes = 12345
        block = 54321

        synapse = EventPredictionSynapse(
            events={
                "acled-event1": {
                    "event_id": "event1",
                    "market_type": "acled",
                    "probability": 0.5,
                    "miner_answered": True,
                    "description": "Test match",
                    "cutoff": 1354458600,
                    "starts": None,
                    "resolve_date": 1354458600,
                    "end_date": 1354545000,
                },
                "azuro-event2": {
                    "event_id": "event2",
                    "market_type": "azuro",
                    "probability": 0.75,
                    "miner_answered": False,
                    "description": "Test match 2",
                    "cutoff": 1354458600,
                    "starts": 1354458600,
                    "resolve_date": 1354458600,
                    "end_date": 1354545000,
                },
            }
        )

        neurons_predictions = {"uid_1": synapse, "uid_2": synapse}

        await query_miners_task.store_predictions(
            block=block,
            interval_start_minutes=interval_start_minutes,
            neurons_predictions=neurons_predictions,
        )

        # Assertions
        assert query_miners_task.db_operations.upsert_predictions.await_count == 2

        assert query_miners_task.db_operations.upsert_predictions.await_args_list == [
            call(
                predictions=[
                    (
                        "acled-event1",
                        "hotkey_1",
                        "uid_1",
                        0.5,
                        interval_start_minutes,
                        0.5,
                        block,
                        0.5,
                    ),
                    (
                        "azuro-event2",
                        "hotkey_1",
                        "uid_1",
                        0.75,
                        interval_start_minutes,
                        0.75,
                        block,
                        0.75,
                    ),
                ]
            ),
            call(
                predictions=[
                    (
                        "acled-event1",
                        "hotkey_2",
                        "uid_2",
                        0.5,
                        interval_start_minutes,
                        0.5,
                        block,
                        0.5,
                    ),
                    (
                        "azuro-event2",
                        "hotkey_2",
                        "uid_2",
                        0.75,
                        interval_start_minutes,
                        0.75,
                        block,
                        0.75,
                    ),
                ]
            ),
        ]

    async def test_run(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
        query_miners_task: QueryMiners,
    ):
        # Set events to query & predict
        cutoff_future = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()

        events = [
            (
                "unique1",
                "event1",
                "ifgames",
                "sports",
                "desc1",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                EventStatus.PENDING,
                json.dumps({"market_type": "sports"}),
                "2012-12-02T14:30:00+00:00",
                cutoff_future,
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique2",
                "event2",
                "ifgames",
                "acled",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                EventStatus.PENDING,
                json.dumps({"market_type": "acled"}),
                "2012-12-02T14:30:00+00:00",
                cutoff_future,
                "2000-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        # Set up the bittensor mocks
        block = 101.0

        query_miners_task.metagraph.uids = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        query_miners_task.metagraph.block = torch.nn.Parameter(torch.tensor(block))
        query_miners_task.metagraph.axons = {
            1: MagicMock(is_serving=True, hotkey="hotkey_1", ip="ip_1"),
            2: MagicMock(is_serving=True, hotkey="hotkey_2", ip="ip_2"),
        }

        async def forward(
            axons: list[AxonInfo], synapse: EventPredictionSynapse, deserialize: bool, timeout: int
        ):
            # Add a fake probability to each event in synapse.events
            for _, event in synapse.events.items():
                event["probability"] = 0.8

            # Build responses
            responses = [synapse for _ in axons]

            return responses

        query_miners_task.dendrite.forward = forward

        # Configure mock to return our test time when now() is called
        mocked_interval_start_minutes = 530160

        with patch(
            "neurons.validator.tasks.query_miners.get_interval_start_minutes"
        ) as mock_get_interval_start_minutes:
            mock_get_interval_start_minutes.return_value = mocked_interval_start_minutes

            # Run the task
            await query_miners_task.run()

        # Assertions
        predictions = await db_client.many(
            """
                SELECT * FROM predictions
            """
        )

        assert len(predictions) == 4

        assert predictions == [
            (
                # unique_event_id
                "ifgames-event1",
                # minerHotkey
                "hotkey_1",
                # minerUid
                "1",
                # predictedOutcome
                "0.8",
                # canOverwrite
                None,
                # outcome
                None,
                # interval_start_minutes
                mocked_interval_start_minutes,
                # interval_agg_prediction
                0.8,
                # interval_count
                1,
                # submitted
                ANY,
                # blocktime
                block,
                # exported
                0,
            ),
            (
                "ifgames-event2",
                "hotkey_1",
                "1",
                "0.8",
                None,
                None,
                mocked_interval_start_minutes,
                0.8,
                1,
                ANY,
                block,
                0,
            ),
            (
                "ifgames-event1",
                "hotkey_2",
                "2",
                "0.8",
                None,
                None,
                mocked_interval_start_minutes,
                0.8,
                1,
                ANY,
                block,
                0,
            ),
            (
                "ifgames-event2",
                "hotkey_2",
                "2",
                "0.8",
                None,
                None,
                mocked_interval_start_minutes,
                0.8,
                1,
                ANY,
                block,
                0,
            ),
        ]

        miners = await db_client.many(
            """
                SELECT
                    miner_uid,
                    miner_hotkey
                FROM
                    miners
                """
        )

        assert len(miners) == 2
        assert miners[0] == ("1", "hotkey_1")
        assert miners[1] == ("2", "hotkey_2")

    async def test_run_no_events_to_predict(self, query_miners_task: QueryMiners):
        # Set mocks
        query_miners_task.make_predictions_synapse = MagicMock()

        # Run the task
        await query_miners_task.run()

        # Assertions
        query_miners_task.make_predictions_synapse.assert_not_called()

    async def test_run_no_axons_to_query(
        self, db_operations: DatabaseOperations, query_miners_task: QueryMiners
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = torch.nn.Parameter(torch.tensor([]))
        query_miners_task.metagraph.block = torch.nn.Parameter(torch.tensor(99.0))

        query_miners_task.query_neurons = MagicMock()
        query_miners_task.make_predictions_synapse = MagicMock()

        # Set events to query & predict
        events = [
            (
                "unique1",
                "event1",
                "ifgames",
                "sports",
                "desc1",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                EventStatus.PENDING,
                json.dumps({"market_type": "sports"}),
                "2012-12-02T14:30:00+00:00",
                (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat(),
                "2000-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        # Run the task
        await query_miners_task.run()

        # Assertions
        query_miners_task.make_predictions_synapse.assert_called_once()
        query_miners_task.query_neurons.assert_not_called()
