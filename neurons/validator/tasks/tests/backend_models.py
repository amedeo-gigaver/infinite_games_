# THIS SHOULD BE KEPT IN SYNC WITH BACKEND MODELS
# https://github.com/infinite-mech/infinite_games_events/blob/develop/event_generator/models/provider_events.py

import datetime
import typing

from pydantic import BaseModel, ConfigDict, Field, conlist


class MinerEventResult(BaseModel):
    event_id: str
    provider_type: str
    title: typing.Optional[str]
    description: typing.Optional[str]
    category: str
    start_date: typing.Optional[datetime.datetime]
    end_date: typing.Optional[datetime.datetime]
    resolve_date: typing.Optional[datetime.datetime]
    settle_date: datetime.datetime
    prediction: float
    answer: float = Field(..., ge=0, le=1)  # noqa
    miner_hotkey: str
    miner_uid: int
    miner_score: float
    miner_effective_score: float
    validator_hotkey: str
    validator_uid: int
    metadata: typing.Dict[str, typing.Any]
    spec_version: typing.Optional[str] = "0.0.0"

    model_config = ConfigDict(from_attributes=True)


class MinerEventResultItems(BaseModel):
    results: conlist(MinerEventResult, min_length=1)  # type: ignore
