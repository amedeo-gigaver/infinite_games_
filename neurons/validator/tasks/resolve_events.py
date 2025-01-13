from datetime import datetime, timezone

import aiohttp

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class ResolveEvents(AbstractTask):
    interval: float
    api_client: IfGamesClient
    db_operations: DatabaseOperations
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        api_client: IfGamesClient,
        logger: InfiniteGamesLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate api_client
        if not isinstance(api_client, IfGamesClient):
            raise TypeError("api_client must be an instance of IfGamesClient.")

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.api_client = api_client
        self.logger = logger

    @property
    def name(self):
        return "resolve-events"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Read pending events from db
        pending_events = await self.db_operations.get_pending_events()

        for event in pending_events:
            event_id = event[0]

            try:
                # Query
                event = await self.api_client.get_event(event_id=event_id)

                resolved = True if event.get("answer") is not None else False

                # Mark resolved
                if resolved:
                    outcome = event.get("answer")

                    resolved_at = datetime.fromtimestamp(
                        event.get("resolved_at"), tz=timezone.utc
                    ).isoformat()

                    await self.db_operations.resolve_event(
                        event_id=event_id,
                        outcome=outcome,
                        resolved_at=resolved_at,
                    )

                    self.logger.debug("Event resolved", extra={"event_id": event_id})

            except aiohttp.ClientResponseError as error:
                # Clear deleted events
                if error.status in [404, 410]:
                    await self.db_operations.delete_event(event_id=event_id)

                    self.logger.debug(
                        "Event deleted",
                        extra={"event_id": event_id, "request_status": error.status},
                    )

                    continue

                raise error
