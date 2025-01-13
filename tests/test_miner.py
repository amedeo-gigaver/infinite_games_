from unittest.mock import Mock

import pytest
import torch

from neurons.miner import Miner
from tests.utils import MockMiner


class TestMinerNeuronTestCase:
    @pytest.mark.asyncio
    async def test_miner_blacklisting(self):
        mock_miner = MockMiner()

        synapse = Mock()
        synapse.dendrite = Mock()
        synapse.dendrite.hotkey = None

        result, message = await mock_miner.blacklist(synapse)
        assert result is False
        assert message == "Blacklisting disabled in testnet"

        mock_miner.subtensor.network = "Impersonate mainnet"
        result, message = await mock_miner.blacklist(synapse)
        assert result is True
        assert message == "Hotkey not provided"

        synapse.dendrite.hotkey = "fake_hotkey"
        result, message = await mock_miner.blacklist(synapse)
        assert result is True
        assert message == "Unrecognized hotkey"

        synapse.dendrite.hotkey = "hotkey3"
        result, message = await mock_miner.blacklist(synapse)
        assert result is True
        assert message == "Non-validator hotkey"

        synapse.dendrite.hotkey = "hotkey2"
        result, message = await mock_miner.blacklist(synapse)
        assert result is True
        assert message == "Low stake"

        synapse.dendrite.hotkey = "hotkey1"
        result, message = await mock_miner.blacklist(synapse)
        assert result is False
        assert message == "Hotkey recognized!"

        # Test a code error
        mock_miner.metagraph = Mock()
        mock_miner.metagraph.S = None
        result, message = await mock_miner.blacklist(synapse)
        assert result is True
        assert message == "Failed to validate hotkey"
