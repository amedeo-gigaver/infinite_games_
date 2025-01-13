import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from aioresponses import aioresponses
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.event import EventStatus
from neurons.validator.tasks.resolve_events import ResolveEvents
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestResolveEventsTask:
    @pytest.fixture
    def db_operations_mock(self):
        db_operations = AsyncMock(spec=DatabaseOperations)

        return db_operations

    @pytest.fixture
    def api_client_mock(self):
        api_client = AsyncMock(spec=IfGamesClient)

        return api_client

    @pytest.fixture
    def bt_wallet(self):
        hotkey_mock = MagicMock()
        hotkey_mock.sign = MagicMock(side_effect=lambda x: x.encode("utf-8"))
        hotkey_mock.ss58_address = "ss58_address"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)

        return bt_wallet

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
    def logger_mock(self):
        return MagicMock(spec=InfiniteGamesLogger)

    @pytest.fixture
    def resolve_events_task(
        self, db_operations: DatabaseOperations, bt_wallet: Wallet, logger_mock: InfiniteGamesLogger
    ):
        api_client = IfGamesClient(env="test", logger=logger_mock, bt_wallet=bt_wallet)

        return ResolveEvents(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
            logger=logger_mock,
        )

    async def test_run_no_pending_events(
        self,
        db_operations_mock: DatabaseOperations,
        bt_wallet: Wallet,
        logger_mock: InfiniteGamesLogger,
    ):
        """Test the run method when there are no pending events."""
        # Arrange
        api_client = IfGamesClient(env="test", logger=logger_mock, bt_wallet=bt_wallet)

        resolve_events_task = ResolveEvents(
            interval_seconds=60.0,
            db_operations=db_operations_mock,
            api_client=api_client,
            logger=logger_mock,
        )
        db_operations_mock.get_pending_events.return_value = []

        # Act
        await resolve_events_task.run()

        # Assert
        db_operations_mock.resolve_event.assert_not_called()

    async def test_resolve_events(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
        resolve_events_task: ResolveEvents,
    ):
        """Test that task resolves correctly pending events."""
        # Arrange
        events = [
            (
                "unique_1",
                "event_id_1",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.DISCARDED,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_2",
                "event_id_2",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_3",
                "event_id_3",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.SETTLED,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_4",
                "event_id_4",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        mock_response_event_2 = {
            "event_id": events[1][1],
            "answer": None,
            "resolved_at": None,
        }

        mock_response_event_4 = {
            "event_id": events[3][1],
            "answer": 1,  # Event is settled now
            "resolved_at": 1234567890,
        }

        # Mock API response
        with aioresponses() as mocked:
            mocked.get(
                f"/api/v2/events/{mock_response_event_2['event_id']}",
                status=200,
                body=json.dumps(mock_response_event_2).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/{mock_response_event_4['event_id']}",
                status=200,
                body=json.dumps(mock_response_event_4).encode("utf-8"),
            )

            # Act
            await resolve_events_task.run()

        # Assert
        response = await db_client.many(
            """
                SELECT status, outcome, resolved_at from events
            """
        )

        assert len(response) == 4
        assert response[0][0] == str(EventStatus.DISCARDED.value)
        assert response[1][0] == str(EventStatus.PENDING.value)
        assert response[2][0] == str(EventStatus.SETTLED.value)

        # Event 4 is settled
        assert response[3][0] == str(EventStatus.SETTLED.value)
        assert response[3][1] == str(mock_response_event_4["answer"])
        assert (
            response[3][2]
            == datetime.fromtimestamp(
                mock_response_event_4["resolved_at"], timezone.utc
            ).isoformat()
        )

    async def test_resolve_events_404_410_errors(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
        resolve_events_task: ResolveEvents,
    ):
        """Test that task handles 404 & 410 on get event."""
        # Arrange
        events = [
            (
                "unique_1",
                "event_id_1",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_2",
                "event_id_2",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_3",
                "event_id_3",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        mock_response_event_1 = {"detail": "EVENT_NOT_FOUND"}

        mock_response_event_2 = {"detail": "EVENT_NO_LONGER_ACTIVE"}

        mock_response_event_3 = {
            "event_id": events[2][1],
            "answer": 1,  # Event is settled now,
            "resolved_at": 1234567890,
        }

        # Mock API response
        with aioresponses() as mocked:
            mocked.get(
                f"/api/v2/events/{events[0][1]}",
                status=404,
                body=json.dumps(mock_response_event_1).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/{events[1][1]}",
                status=410,
                body=json.dumps(mock_response_event_2).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/{events[2][1]}",
                status=200,
                body=json.dumps(mock_response_event_3).encode("utf-8"),
            )

            # Act
            await resolve_events_task.run()

        # Assert
        response = await db_client.many(
            """
                SELECT event_id, status, outcome, resolved_at from events
            """
        )

        # Only one event left, the others are deleted
        assert len(response) == 1

        # Resolved event status
        assert response[0][0] == mock_response_event_3["event_id"]
        assert response[0][1] == str(EventStatus.SETTLED.value)
        assert response[0][2] == str(mock_response_event_3["answer"])
        assert (
            response[0][3]
            == datetime.fromtimestamp(
                mock_response_event_3["resolved_at"], timezone.utc
            ).isoformat()
        )
