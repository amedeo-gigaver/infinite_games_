from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator


class MinersModel(BaseModel):
    miner_hotkey: str
    miner_uid: str
    node_ip: Optional[str] = None
    registered_date: datetime
    last_updated: Optional[datetime] = None
    blocktime: Optional[int] = None
    blocklisted: bool = False
    model_config = {"arbitrary_types_allowed": True}

    @property
    def primary_key(self):
        return [
            "miner_hotkey",
            "miner_uid",
        ]

    @field_validator("blocklisted", mode="before")
    def parse_blocklisted_as_bool(cls, v: bool) -> bool:
        # If the DB returns an integer, convert it to boolean
        if isinstance(v, int):
            return bool(v)
        return v


MINERS_FIELDS = MinersModel.model_fields.keys()
