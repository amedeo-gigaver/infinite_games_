import json
from datetime import datetime, timezone
from typing import Iterable

from bittensor.core.chain_data import AxonInfo
from bittensor.core.dendrite import DendriteMixin
from bittensor.core.metagraph import MetagraphMixin

from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.models.events_prediction_synapse import (
    EventsPredictionSynapse,
)
from infinite_games.sandbox.validator.scheduler.task import AbstractTask

AxonInfoByUidType = dict[int, AxonInfo]
SynapseResponseByUidType = dict[int, EventsPredictionSynapse]

CLUSTER_EPOCH_2024 = datetime(2024, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES = 60 * 4


class QueryMiners(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    dendrite: DendriteMixin
    metagraph: MetagraphMixin

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        dendrite: DendriteMixin,
        metagraph: MetagraphMixin,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate dendrite
        if not isinstance(dendrite, DendriteMixin):
            raise TypeError("dendrite must be an instance of DendriteMixin.")

        # Validate metagraph
        if not isinstance(metagraph, MetagraphMixin):
            raise TypeError("metagraph must be an instance of MetagraphMixin.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.dendrite = dendrite
        self.metagraph = metagraph

    @property
    def name(self):
        return "query-miners"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Get events to predict
        events = await self.db_operations.get_events_to_predict()

        if not len(events):
            return

        # Build synapse
        synapse = self.make_predictions_synapse(events)

        # Sync metagraph & store the current block
        self.metagraph.sync(lite=True)
        block = self.metagraph.block

        # Get axons to query
        axons = self.get_axons()

        if not len(axons):
            return

        # keep record of neurons
        # https://linear.app/infinite-games/issue/INF-210

        # Query neurons
        predictions_synapses: SynapseResponseByUidType = await self.query_neurons(
            axons_by_uid=axons, synapse=synapse
        )

        interval_start_minutes = self.get_interval_start_minutes()

        # Store predictions
        await self.store_predictions(
            block=block,
            interval_start_minutes=interval_start_minutes,
            neurons_predictions=predictions_synapses,
        )

    def get_axons(self) -> AxonInfoByUidType:
        axons: AxonInfoByUidType = {}

        for uid in self.metagraph.uids:
            axon = self.metagraph.axons[uid]

            if axon is not None and axon.is_serving is True:
                axons[uid] = axon

        return axons

    def get_interval_start_minutes(self):
        now = datetime.now(timezone.utc)

        minutes_since_epoch = int((now - CLUSTER_EPOCH_2024).total_seconds()) // 60

        interval_start_minutes = minutes_since_epoch - (
            minutes_since_epoch % (CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        )

        return interval_start_minutes

    def make_predictions_synapse(self, events: Iterable[tuple[any]]) -> EventsPredictionSynapse:
        compiled_events = {}

        for event in events:
            event_id = event[0]
            description = event[2]
            cutoff = int(datetime.fromisoformat(event[3]).timestamp()) if event[3] else None
            resolve_date = int(datetime.fromisoformat(event[4]).timestamp()) if event[4] else None
            end_date = int(datetime.fromisoformat(event[5]).timestamp()) if event[5] else None
            metadata = {**json.loads(event[6])}
            market_type = (metadata.get("market_type", event[1])).lower()

            compiled_events[f"{market_type}-{event_id}"] = {
                "event_id": event_id,
                "market_type": market_type,
                "probability": None,
                "miner_answered": False,
                "description": description,
                "cutoff": cutoff,
                "starts": cutoff if (cutoff and market_type == "azuro") else None,
                "resolve_date": resolve_date,
                "end_date": end_date,
            }

        return EventsPredictionSynapse(events=compiled_events)

    def parse_neuron_predictions(
        self,
        block: int,
        interval_start_minutes: int,
        uid: int,
        neuron_predictions: EventsPredictionSynapse,
    ):
        axon_hotkey = self.metagraph.axons[uid].hotkey

        predictions_to_insert = []

        # Iterate over all event predictions
        for unique_event_id, event_prediction in neuron_predictions.events.items():
            answer = event_prediction.get("probability")

            prediction = (
                unique_event_id,
                axon_hotkey,
                uid,
                answer,
                interval_start_minutes,
                answer,
                block,
                answer,
            )

            predictions_to_insert.append(prediction)

        return predictions_to_insert

    async def query_neurons(
        self, axons_by_uid: AxonInfoByUidType, synapse: EventsPredictionSynapse
    ):
        timeout = 120

        axons_list = list(axons_by_uid.values())

        # Use forward directly to make it async
        responses: list[EventsPredictionSynapse] = await self.dendrite.forward(
            # Send the query to selected miner axons in the network.
            axons=axons_list,
            synapse=synapse,
            # Do not deserialize the response so that we have access to the raw response.
            deserialize=False,
            timeout=timeout,
        )

        responses_by_uid: SynapseResponseByUidType = {}
        axons_uids = list(axons_by_uid.keys())

        for uid, response in zip(axons_uids, responses):
            responses_by_uid[uid] = response

        return responses_by_uid

    async def store_predictions(
        self, block: int, interval_start_minutes: int, neurons_predictions: SynapseResponseByUidType
    ):
        # For each neuron predictions
        for uid, neuron_predictions in neurons_predictions.items():
            # Parse neuron predictions for insert
            parsed_neuron_predictions_for_insertion = self.parse_neuron_predictions(
                block=block,
                interval_start_minutes=interval_start_minutes,
                uid=uid,
                neuron_predictions=neuron_predictions,
            )

            # Batch upsert neuron predictions
            await self.db_operations.upsert_predictions(
                predictions=parsed_neuron_predictions_for_insertion
            )