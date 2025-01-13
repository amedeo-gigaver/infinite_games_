from datetime import datetime
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class PredictionExportedStatus(IntEnum):
    NOT_EXPORTED = 0
    EXPORTED = 1


class PredictionsModel(BaseModel):
    unique_event_id: str
    minerHotkey: Optional[str] = None
    minerUid: str
    predictedOutcome: Optional[str] = None
    canOverwrite: Optional[Any] = None
    outcome: Optional[str] = None
    interval_start_minutes: int
    interval_agg_prediction: float
    interval_count: int = 1
    submitted: Optional[datetime] = None
    blocktime: Optional[int] = None
    exported: Optional[bool] = False
    model_config = {"arbitrary_types_allowed": True}

    @property
    def primary_key(self):
        return [
            "unique_event_id",
            "interval_start_minutes",
            "minerUid",
        ]

    @field_validator("exported", mode="before")
    def parse_exported_as_bool(cls, v: Any) -> bool:
        # If the DB returns an integer, convert it to boolean
        if isinstance(v, int):
            return bool(v)
        return v


PREDICTION_FIELDS = PredictionsModel.model_fields.keys()
