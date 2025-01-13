from datetime import datetime
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class EventStatus(IntEnum):
    """Generic event status"""

    DISCARDED = 1
    PENDING = 2
    SETTLED = 3
    # In case of errors
    NOT_IMPLEMENTED = 4


class EventsModel(BaseModel):
    """Events model: keep it 1:1 with the DB table"""

    model_config = {"arbitrary_types_allowed": True}
    unique_event_id: str
    event_id: str
    market_type: str
    event_type: str
    registered_date: Optional[datetime] = None
    description: str
    starts: Optional[datetime] = None
    resolve_date: Optional[datetime] = None
    outcome: Optional[str] = None
    local_updated_at: Optional[datetime] = None
    status: EventStatus
    metadata: str
    processed: Optional[bool] = False
    exported: Optional[bool] = False
    created_at: Optional[datetime] = None
    cutoff: Optional[datetime] = None
    end_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    event_type: Optional[str] = None

    @property
    def primary_key(self):
        return [
            "unique_event_id",
        ]

    @field_validator("exported", mode="before")
    def parse_exported_as_bool(cls, v: Any) -> bool:
        # If the DB returns an integer, convert it to boolean
        if isinstance(v, int):
            return bool(v)
        return v

    @field_validator("processed", mode="before")
    def parse_processed_as_bool(cls, v: Any) -> bool:
        # Similarly, ensure processed is boolean (in case DB returns int)
        if isinstance(v, int):
            return bool(v)
        return v

    @field_validator("status", mode="before")
    def parse_status_as_enum(cls, v: Any) -> EventStatus:
        # Ensure status is an EventStatus enum
        # Convert ints or strings into the enum if possible
        try:
            if isinstance(v, EventStatus):
                return v
            return EventStatus(int(v))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid status: {v}")


EVENTS_FIELDS = EventsModel.model_fields.keys()
