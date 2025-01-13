import json
import time
from datetime import datetime
from typing import Iterable

from bittensor.core.chain_data import AxonInfo
from bittensor.core.dendrite import DendriteMixin
from bittensor.core.metagraph import MetagraphMixin

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event_prediction_synapse import EventPredictionSynapse
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.converters import torch_or_numpy_to_int
from neurons.validator.utils.common.interval import get_interval_start_minutes
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

AxonInfoByUidType = dict[int, AxonInfo]
SynapseResponseByUidType = dict[int, EventPredictionSynapse]


class QueryMiners(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    dendrite: DendriteMixin
    metagraph: MetagraphMixin
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        dendrite: DendriteMixin,
        metagraph: MetagraphMixin,
        logger: InfiniteGamesLogger,
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

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.logger = logger

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
        block = torch_or_numpy_to_int(self.metagraph.block)

        # Get axons to query
        axons = self.get_axons()

        if not len(axons):
            return

        # Store miners
        await self.store_miners(block=block, axons=axons)

        # Query neurons
        predictions_synapses: SynapseResponseByUidType = await self.query_neurons(
            axons_by_uid=axons, synapse=synapse
        )

        interval_start_minutes = get_interval_start_minutes()

        # Store predictions
        await self.store_predictions(
            block=block,
            interval_start_minutes=interval_start_minutes,
            neurons_predictions=predictions_synapses,
        )

    def get_axons(self) -> AxonInfoByUidType:
        axons: AxonInfoByUidType = {}

        for uid in self.metagraph.uids:
            int_uid = torch_or_numpy_to_int(uid)

            axon = self.metagraph.axons[int_uid]

            if axon is not None and axon.is_serving is True:
                axons[int_uid] = axon

        return axons

    def make_predictions_synapse(self, events: Iterable[tuple[any]]) -> EventPredictionSynapse:
        compiled_events = {}

        for event in events:
            event_id = event[0]
            truncated_market_type = event[1]
            description = event[2]
            cutoff = int(datetime.fromisoformat(event[3]).timestamp()) if event[3] else None
            resolve_date = int(datetime.fromisoformat(event[4]).timestamp()) if event[4] else None
            end_date = int(datetime.fromisoformat(event[5]).timestamp()) if event[5] else None
            metadata = {**json.loads(event[6])}
            market_type = (metadata.get("market_type", event[1])).lower()

            compiled_events[f"{truncated_market_type}-{event_id}"] = {
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

        return EventPredictionSynapse(events=compiled_events)

    def parse_neuron_predictions(
        self,
        block: int,
        interval_start_minutes: int,
        uid: int,
        neuron_predictions: EventPredictionSynapse,
    ):
        axon_hotkey = self.metagraph.axons[uid].hotkey

        predictions_to_insert = []

        # Iterate over all event predictions
        for unique_event_id, event_prediction in neuron_predictions.events.items():
            answer = event_prediction.get("probability")

            # Drop null predictions
            if answer is None:
                continue

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

    async def query_neurons(self, axons_by_uid: AxonInfoByUidType, synapse: EventPredictionSynapse):
        timeout = 120

        axons_list = list(axons_by_uid.values())

        start_time = time.time()

        # Use forward directly to make it async
        responses: list[EventPredictionSynapse] = await self.dendrite.forward(
            # Send the query to selected miner axons in the network.
            axons=axons_list,
            synapse=synapse,
            # Do not deserialize the response so that we have access to the raw response.
            deserialize=False,
            timeout=timeout,
        )

        elapsed_time_ms = round((time.time() - start_time) * 1000)

        self.logger.debug(
            "Miners queried",
            extra={"miners_count": len(axons_list), "elapsed_time_ms": elapsed_time_ms},
        )

        responses_by_uid: SynapseResponseByUidType = {}
        axons_uids = list(axons_by_uid.keys())

        for uid, response in zip(axons_uids, responses):
            responses_by_uid[uid] = response

        return responses_by_uid

    async def store_miners(self, block: int, axons: AxonInfoByUidType):
        miners_count_in_db = await self.db_operations.get_miners_count()

        registered_date = (
            datetime.now().isoformat()
            if miners_count_in_db > 0
            else datetime(year=2024, month=1, day=1).isoformat()
        )

        miners = [
            (uid, axon.hotkey, axon.ip, registered_date, block, axon.ip, block)
            for uid, axon in axons.items()
        ]

        await self.db_operations.upsert_miners(miners=miners)

        self.logger.debug("Miners stored", extra={"miners_count": len(miners)})

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

            if len(parsed_neuron_predictions_for_insertion) > 0:
                # Batch upsert neuron predictions
                await self.db_operations.upsert_predictions(
                    predictions=parsed_neuron_predictions_for_insertion
                )

                self.logger.debug(
                    "Predictions stored",
                    extra={
                        "neuron_uid": uid,
                        "predictions_count": len(parsed_neuron_predictions_for_insertion),
                    },
                )
