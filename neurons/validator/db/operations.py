from typing import Iterable

from neurons.validator.db.client import DatabaseClient
from neurons.validator.models.event import EVENTS_FIELDS, EventsModel, EventStatus
from neurons.validator.models.miner import MINERS_FIELDS, MinersModel
from neurons.validator.models.prediction import (
    PREDICTION_FIELDS,
    PredictionExportedStatus,
    PredictionsModel,
)
from neurons.validator.utils.logger.logger import db_logger


class DatabaseOperations:
    __db_client: DatabaseClient

    def __init__(self, db_client: DatabaseClient):
        if not isinstance(db_client, DatabaseClient):
            raise ValueError("Invalid db_client arg")

        self.__db_client = db_client

    async def delete_event(self, event_id: str) -> Iterable[tuple[str]]:
        return await self.__db_client.delete(
            """
                DELETE FROM events WHERE event_id = ? RETURNING event_id
            """,
            [event_id],
        )

    async def get_events_to_predict(self) -> Iterable[tuple[str]]:
        return await self.__db_client.many(
            """
                SELECT
                    event_id,
                    market_type,
                    description,
                    cutoff,
                    resolve_date,
                    end_date,
                    metadata
                FROM
                    events
                WHERE
                    status = ?
                    AND datetime(CURRENT_TIMESTAMP) < datetime(cutoff)
            """,
            parameters=[EventStatus.PENDING],
        )

    async def get_last_event_from(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(created_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_miners_count(self) -> int:
        row = await self.__db_client.one(
            """
                SELECT COUNT(*) FROM miners
            """
        )

        return row[0]

    async def get_pending_events(self) -> Iterable[tuple[str]]:
        # TODO limit to events that passed cutoff
        return await self.__db_client.many(
            """
                SELECT event_id FROM events WHERE status = ?
            """,
            parameters=[EventStatus.PENDING],
        )

    async def get_predictions_to_export(self, current_interval_minutes: int, batch_size: int):
        return await self.__db_client.many(
            """
                SELECT
                    p.ROWID,
                    p.unique_event_id,
                    p.minerHotkey,
                    p.minerUid,
                    e.event_type,
                    p.predictedOutcome,
                    p.interval_start_minutes,
                    p.interval_agg_prediction,
                    p.interval_count
                FROM
                    predictions p
                JOIN
                    events e ON e.unique_event_id = p.unique_event_id
                WHERE
                    p.exported = ?
                    AND p.interval_start_minutes < ?
                ORDER BY
                    p.ROWID ASC
                LIMIT
                    ?
            """,
            [PredictionExportedStatus.NOT_EXPORTED, current_interval_minutes, batch_size],
        )

    async def mark_predictions_as_exported(self, ids: list[str]):
        placeholders = ", ".join(["?"] * len(ids))

        return await self.__db_client.update(
            f"""
                UPDATE
                    predictions
                SET
                    exported = ?
                WHERE
                    ROWID IN ({placeholders})
                RETURNING
                    ROWID
            """,
            [PredictionExportedStatus.EXPORTED] + ids,
        )

    async def resolve_event(
        self, event_id: str, outcome: str, resolved_at: str
    ) -> Iterable[tuple[str]]:
        return await self.__db_client.update(
            """
                UPDATE
                    events
                SET
                    status = ?,
                    outcome = ?,
                    resolved_at = ?,
                    local_updated_at = CURRENT_TIMESTAMP
                WHERE
                    event_id = ?
                RETURNING
                    event_id
            """,
            [EventStatus.SETTLED, outcome, resolved_at, event_id],
        )

    async def upsert_events(self, events: list[list[any]]) -> None:
        return await self.__db_client.insert_many(
            """
                INSERT INTO events
                    (
                        unique_event_id,
                        event_id,
                        market_type,
                        event_type,
                        description,
                        starts,
                        resolve_date,
                        outcome,
                        status,
                        metadata,
                        created_at,
                        cutoff,
                        end_date,
                        registered_date,
                        local_updated_at
                    )
                VALUES
                    (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP
                    )
                ON CONFLICT
                    (unique_event_id)
                DO NOTHING
            """,
            events,
        )

    async def upsert_miners(self, miners: list[list[any]]) -> None:
        return await self.__db_client.insert_many(
            """
                INSERT INTO miners
                    (
                        miner_uid,
                        miner_hotkey,
                        node_ip,
                        registered_date,
                        last_updated,
                        blocktime,
                        blocklisted
                    )
                VALUES
                    (
                        ?,
                        ?,
                        ?,
                        ?,
                        CURRENT_TIMESTAMP,
                        ?,
                        FALSE
                    )
                ON CONFLICT
                    (miner_hotkey, miner_uid)
                DO UPDATE
                    set node_ip = ?,
                    last_updated = CURRENT_TIMESTAMP,
                    blocktime = ?
            """,
            miners,
        )

    async def upsert_predictions(self, predictions: list[list[any]]):
        return await self.__db_client.insert_many(
            """
                INSERT INTO predictions (
                    unique_event_id,
                    minerHotkey,
                    minerUid,
                    predictedOutcome,
                    interval_start_minutes,
                    interval_agg_prediction,
                    blocktime,
                    interval_count,
                    submitted
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    1,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT(unique_event_id,  interval_start_minutes, minerUid)
                DO UPDATE SET
                    interval_agg_prediction = (interval_agg_prediction * interval_count + ?) / (interval_count + 1),
                    interval_count = interval_count + 1
            """,
            predictions,
        )

    async def upsert_pydantic_events(self, events: list[EventsModel]) -> None:
        """Same as upsert_events but with pydantic models"""

        fields_to_insert = [
            field_name
            for field_name in EVENTS_FIELDS
            if field_name not in ("registered_date", "local_updated_at")
        ]
        placeholders = ", ".join(["?"] * len(fields_to_insert))
        columns = ", ".join(fields_to_insert)

        # Convert each event into a tuple of values in the same order as fields_to_insert
        event_tuples = [
            tuple(getattr(event, field_name) for field_name in fields_to_insert) for event in events
        ]

        sql = f"""
                INSERT INTO events
                    ({columns}, registered_date, local_updated_at)
                VALUES
                    ({placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT
                    (unique_event_id)
                DO NOTHING
        """
        return await self.__db_client.insert_many(
            sql=sql,
            parameters=event_tuples,
        )

    async def get_events_for_scoring(self) -> list[EventsModel]:
        """
        Returns all events that were recently resolved and need to be scored
        """

        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(EVENTS_FIELDS)}
                FROM events
                WHERE status = ?
                    AND outcome IS NOT NULL
                    AND processed = false
            """,
            parameters=[EventStatus.SETTLED],
            use_row_factory=True,
        )

        events = []
        for row in rows:
            try:
                event = EventsModel(**dict(row))
                events.append(event)
            except Exception:
                db_logger.exception("Error parsing event", extra={"row": row[0]})

        return events

    async def get_predictions_for_scoring(self, event_id: str) -> list[PredictionsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(PREDICTION_FIELDS)}
                FROM predictions
                WHERE unique_event_id = ?
            """,
            parameters=(event_id,),
            use_row_factory=True,
        )

        predictions = []
        for row in rows:
            try:
                prediction = PredictionsModel(**dict(row))
                predictions.append(prediction)
            except Exception:
                db_logger.exception("Error parsing prediction", extra={"row": row[0]})

        return predictions

    async def get_miners_last_registration(self) -> list:
        rows = await self.__db_client.many(
            f"""
                WITH ranked AS (
                    SELECT
                        {', '.join(MINERS_FIELDS)},
                        ROW_NUMBER() OVER (
                            PARTITION BY miner_uid
                            ORDER BY registered_date DESC
                        ) AS rn
                    FROM miners t
                )
                SELECT
                    {', '.join(MINERS_FIELDS)}
                FROM ranked
                WHERE rn = 1
                ORDER BY miner_uid
            """,
            use_row_factory=True,
        )
        miners = []
        for row in rows:
            try:
                miner = MinersModel(**dict(row))
                miners.append(miner)
            except Exception:
                db_logger.exception("Error parsing miner", extra={"row": row[0]})

        return miners

    async def mark_event_as_processed(self, unique_event_id: str) -> None:
        return await self.__db_client.update(
            """
                UPDATE events
                SET processed = true
                WHERE unique_event_id = ?
            """,
            parameters=(unique_event_id,),
        )

    async def mark_event_as_exported(self, unique_event_id: str) -> None:
        return await self.__db_client.update(
            """
                UPDATE events
                SET exported = true
                WHERE unique_event_id = ?
            """,
            parameters=(unique_event_id,),
        )
