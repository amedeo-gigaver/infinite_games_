from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import Mock

import torch

from infinite_games.events.base import ProviderEvent
from infinite_games.protocol import EventPredictionSynapse
from neurons.miner import Miner


def after(**kwargs):
    return datetime.now(timezone.utc) + timedelta(**kwargs)


def before(**kwargs):
    return datetime.now(timezone.utc) - timedelta(**kwargs)


def fake_synapse_response(events: List[ProviderEvent]):
    uid_responses = []
    for uid in range(0, 256):
        synapse = EventPredictionSynapse()
        synapse.init(events)
        uid_responses.append(synapse)

    return uid_responses


class MockMiner(Miner):
    def __init__(self):
        self.metagraph = Mock()
        self.subtensor = Mock()
        self.subtensor.network = "mock"
        self.metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
        self.metagraph.validator_permit = [True, True, False]
        self.metagraph.S = torch.tensor([10000.0, 2000.0, 5.1])

    async def blacklist(self, synapse):
        return await super().blacklist(synapse)
