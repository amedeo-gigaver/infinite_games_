import json
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from bittensor.core.chain_data import AxonInfo
from bittensor.core.dendrite import DendriteMixin
from bittensor.core.metagraph import MetagraphMixin

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.models.event import EventStatus
from infinite_games.sandbox.validator.models.events_prediction_synapse import (
    EventsPredictionSynapse,
)
from infinite_games.sandbox.validator.tasks.query_miners import QueryMiners
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger


class TestQueryMiners:
    @pytest.fixture(scope="function")
    async def db_client(self):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        logger = MagicMock(spec=AbstractLogger)

        db_client = Client(db_path, logger)

        await db_client.migrate()

        return db_client

    @pytest.fixture
    def db_operations(self, db_client: Client):
        return DatabaseOperations(db_client=db_client)

    @pytest.fixture
    def query_miners_task(self, db_operations: DatabaseOperations):
        dendrite = MagicMock(spec=DendriteMixin)
        metagraph = MagicMock(spec=MetagraphMixin)

        return QueryMiners(
            interval_seconds=60.0,
            db_operations=db_operations,
            dendrite=dendrite,
            metagraph=metagraph,
        )

    def test_get_axons(
        self,
        query_miners_task: QueryMiners,
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = [1, 2, 3]  # UIDs in the metagraph
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
        query_miners_task.metagraph.uids = []  # No UIDs in the metagraph

        # Call the method
        result = query_miners_task.get_axons()

        # Assertions
        # No axons should be included
        assert result == {}

    @pytest.mark.parametrize(
        "test_time,expected",
        [
            # Test exact epoch start
            (datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 0),
            # Test multiple days later
            (datetime(2025, 1, 3, 4, 0, 0, tzinfo=timezone.utc), 530160),
            # Test with non-zero minutes and seconds (should round down)
            (datetime(2024, 1, 1, 4, 45, 30, tzinfo=timezone.utc), 240),
        ],
    )
    def test_get_interval_start_minutes(self, query_miners_task: QueryMiners, test_time, expected):
        with patch("infinite_games.sandbox.validator.tasks.query_miners.datetime") as mock_datetime:
            # Configure mock to return our test time when now() is called
            mock_datetime.now.return_value = test_time

            result = query_miners_task.get_interval_start_minutes()

            assert result == expected

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

        assert isinstance(synapse, EventsPredictionSynapse)
        assert len(synapse.events) == 2

        # Assert event 1
        expected_key = "acled-event1"

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
        expected_key = "azuro-event2"

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

        assert isinstance(result, EventsPredictionSynapse)
        assert len(result.events) == 0

    def test_parse_neuron_predictions(self, query_miners_task: QueryMiners):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = [1, 2]
        query_miners_task.metagraph.axons = {
            1: MagicMock(hotkey="hotkey1"),
            2: MagicMock(hotkey="hotkey2"),
        }

        interval_start_minutes = 12345
        block = 54321
        uid = 2

        neuron_predictions = EventsPredictionSynapse(
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

        result = query_miners_task.parse_neuron_predictions(
            block=block,
            interval_start_minutes=interval_start_minutes,
            uid=uid,
            neuron_predictions=neuron_predictions,
        )

        assert result == [
            (
                "acled-event1",
                query_miners_task.metagraph.axons[uid].hotkey,
                uid,
                0.5,
                interval_start_minutes,
                0.5,
                block,
                0.5,
            ),
            (
                "azuro-event2",
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
        synapse = EventsPredictionSynapse(events={})

        axons_by_uid = {"1": MagicMock(spec=AxonInfo), "50": MagicMock(spec=AxonInfo)}

        query_miners_task.dendrite.forward = AsyncMock(return_value=[synapse, synapse])

        response = await query_miners_task.query_neurons(axons_by_uid=axons_by_uid, synapse=synapse)

        # Assertions
        assert len(response) == 2
        assert response["1"] == synapse
        assert response["50"] == synapse

    async def test_store_predictions(self, query_miners_task: QueryMiners):
        # Set up mocks
        query_miners_task.db_operations.upsert_predictions = AsyncMock()
        query_miners_task.metagraph.axons = {
            "uid_1": MagicMock(hotkey="hotkey_1"),
            "uid_2": MagicMock(hotkey="hotkey_2"),
        }

        interval_start_minutes = 12345
        block = 54321

        synapse = EventsPredictionSynapse(
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
        self, db_client: Client, db_operations: DatabaseOperations, query_miners_task: QueryMiners
    ):
        # Set events to query & predict
        cutoff_future = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()

        events = [
            (
                "unique1",
                "event1",
                "ifgames",
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
        block = 101

        query_miners_task.metagraph.block = block
        query_miners_task.metagraph.uids = [1, 2]
        query_miners_task.metagraph.axons = {
            1: MagicMock(is_serving=True, hotkey="hotkey_1"),
            2: MagicMock(is_serving=True, hotkey="hotkey_2"),
        }

        async def forward(
            axons: list[AxonInfo], synapse: EventsPredictionSynapse, deserialize: bool, timeout: int
        ):
            # Add a fake probability to each event in synapse.events
            for _, event in synapse.events.items():
                event["probability"] = 0.8

            # Build responses
            responses = [synapse for _ in axons]

            return responses

        query_miners_task.dendrite.forward = forward

        # Configure mock to return our test time when now() is called
        with patch("infinite_games.sandbox.validator.tasks.query_miners.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 3, 4, 0, 0, tzinfo=timezone.utc)

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
                "sports-event1",
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
                530160,
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
                "acled-event2",
                "hotkey_1",
                "1",
                "0.8",
                None,
                None,
                530160,
                0.8,
                1,
                ANY,
                block,
                0,
            ),
            (
                "sports-event1",
                "hotkey_2",
                "2",
                "0.8",
                None,
                None,
                530160,
                0.8,
                1,
                ANY,
                block,
                0,
            ),
            (
                "acled-event2",
                "hotkey_2",
                "2",
                "0.8",
                None,
                None,
                530160,
                0.8,
                1,
                ANY,
                block,
                0,
            ),
        ]

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
        query_miners_task.metagraph.uids = []
        query_miners_task.metagraph.block = 99
        query_miners_task.query_neurons = MagicMock()
        query_miners_task.make_predictions_synapse = MagicMock()

        # Set events to query & predict
        events = [
            (
                "unique1",
                "event1",
                "ifgames",
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