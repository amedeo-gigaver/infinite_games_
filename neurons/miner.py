# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# -- DO NOT TOUCH BELOW - ENV SET --
# flake8: noqa: E402
import os
import sys

# Force torch - must be set before importing bittensor
os.environ["USE_TORCH"] = "1"

# Add the parent directory of the script to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
# -- DO NOT TOUCH ABOVE --

import random
import time
import typing
from datetime import datetime

import bittensor as bt

import infinite_games

# import base miner class which takes care of most of the boilerplate
from infinite_games.base.miner import BaseMinerNeuron
from infinite_games.events.azuro import AzuroProviderIntegration
from infinite_games.events.polymarket import PolymarketProviderIntegration
from infinite_games.utils.miner_cache import (
    MarketType,
    MinerCache,
    MinerCacheObject,
    MinerCacheStatus,
)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

VAL_MIN_STAKE = 10000
DEV_MINER_UID = 93

if os.getenv("OPENAI_KEY"):
    from llm.forecasting import Forecaster


async def _calculate_next_try(market: MinerCacheObject) -> (int, int):
    """
    This function calculates the next point in time that you want to run a probability recalculation for
    your event. It also calculates the number of (remaining) retries that should be done in this event.

    This implementation only assigns a recalculation at the cutoff point of every event.

    Args:
        market (MinerCacheObject): The object that contains all the information for the event.

    Returns:
        (int, int): The number of retries left, the timestamp of next probability calculation

    """
    return market.event.retries - 1, market.event.cutoff


class Miner(BaseMinerNeuron):
    """
    Miner neuron class. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    prev_emission = None

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.providers_set = False
        self.azuro = None
        self.polymarket = None
        self.cache = MinerCache()
        self.cache.initialize_cache()
        self.llm = Forecaster() if os.getenv("OPENAI_KEY") else None
        self.is_testnet = self.metagraph.network == "test"
        bt.logging.info(
            "Miner {} initialized on network: {}: testnet {}".format(
                self.uid, self.metagraph.network, self.is_testnet
            )
        )

    async def initialize_providers(self):
        self.azuro = await AzuroProviderIntegration()._ainit()
        self.polymarket = await PolymarketProviderIntegration()._ainit()

    async def _generate_prediction(self, market: MinerCacheObject) -> None:
        if self.is_testnet and self.uid != DEV_MINER_UID:
            # in testnet, we just assign a random probability; do not make real API calls
            # but keep real calls for the dev miner for testing purposes
            market.event.probability = random.random()
            return
        try:
            llm_prediction = None
            # Polymarket
            if market.event.market_type == MarketType.POLYMARKET and self.polymarket is not None:
                x = await self.polymarket.get_event_by_id(market.event.event_id)
                market.event.probability = x["tokens"][0]["price"]

            # Azuro
            elif market.event.market_type == MarketType.AZURO and self.azuro is not None:
                x = await self.azuro.get_event_by_id(market.event.event_id)
                market.event.probability = 1.0 / float(x["outcome"]["currentOdds"])

            else:
                # LLM
                llm_prediction = (await self.llm.get_prediction(market)) if self.llm else None
                if llm_prediction is not None:
                    market.event.probability = llm_prediction
                else:
                    market.event.probability = 0

            bt.logging.info(
                "({}) Calculate {} prob to {} event {}, retries left: {}".format(
                    "No LLM" if llm_prediction is None else "LLM",
                    market.event.probability,
                    market.event.market_type.name,
                    market.event.event_id,
                    market.event.retries,
                )
            )
        except Exception as e:
            bt.logging.error("Failed to assign, probability, {}".format(repr(e)), exc_info=True)

    async def forward(
        self, synapse: infinite_games.protocol.EventPredictionSynapse
    ) -> infinite_games.protocol.EventPredictionSynapse:
        """
        Processes the incoming synapse and attaches the response to the synapse.
        """
        start_time = time.time()
        if not self.providers_set:
            self.providers_set = True
            await self.initialize_providers()

        today = datetime.now()
        bt.logging.info("[{}] Incoming Events {}".format(today, len(synapse.events.items())))

        for cid, market in synapse.events.items():
            try:
                cached_market: typing.Optional[MinerCacheObject] = await self.cache.get(cid)
                if cached_market is not None:
                    if cached_market.status == MinerCacheStatus.COMPLETED:
                        # Check IF it is time for a re-calculation of the probability.
                        if cached_market.event.retries > 0 and cached_market.event.next_try <= int(
                            today.timestamp()
                        ):
                            # Set the stored object in a rerun state.
                            cached_market.set_for_rerun()

                            # After this re-run, set the next.
                            (
                                cached_market.event.retries,
                                cached_market.event.next_try,
                            ) = await _calculate_next_try(cached_market)

                            await self.cache.add(cid, self._generate_prediction, cached_market)
                        else:
                            market["probability"] = cached_market.event.probability
                            bt.logging.info(
                                "Assign cache {} prob to {} event {}".format(
                                    cached_market.event.probability,
                                    cached_market.event.market_type.name,
                                    cached_market.event.event_id,
                                )
                            )
                else:
                    new_market = MinerCacheObject.init_from_market(market)
                    new_market.event.retries, new_market.event.next_try = await _calculate_next_try(
                        new_market
                    )
                    await self.cache.add(cid, self._generate_prediction, new_market)

                market["miner_answered"] = True
            except Exception as e:
                bt.logging.error(
                    "Failed to process event {} {}".format(cid, repr(e)), exc_info=True
                )

        bt.logging.info(f"Miner answered in {time.time() - start_time:.2f} seconds")
        return synapse

    async def blacklist(
        self, synapse: infinite_games.protocol.EventPredictionSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.EventPredictionSynapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        try:
            if self.subtensor.network in ["test", "mock", "local"]:
                return False, "Blacklisting disabled in testnet"

            # Check if the hotkey is provided
            if not synapse.dendrite.hotkey:
                return True, "Hotkey not provided"

            # Check if the hotkey is recognized
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                bt.logging.warning(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
                return True, "Unrecognized hotkey"

            # Check if the hotkey is a validator
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

            stake = self.metagraph.S[uid].item()
            bt.logging.debug(f"Validator {synapse.dendrite.hotkey} has stake {stake}")
            if stake < VAL_MIN_STAKE:
                bt.logging.warning(
                    f"Blacklisting a request from hotkey {synapse.dendrite.hotkey} with stake {stake}"
                )
                return True, "Low stake"

            bt.logging.debug(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
            return False, "Hotkey recognized!"
        except Exception as e:
            bt.logging.error(f"Failed to validate hotkey {synapse.dendrite.hotkey}: {e}")
            return True, "Failed to validate hotkey"

    async def priority(self, synapse: infinite_games.protocol.EventPredictionSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.EventPredictionSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # Get the caller index.
        prirority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.debug(f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority)
        return prirority

    def save_state(self):
        pass


bt.debug(True)
# bt.trace(true)

# This is the main function, which runs the miner.
if __name__ == "__main__":
    start_time = time.time()
    with Miner() as miner:
        while True:
            bt.logging.debug(f"Miner running for {time.time() - start_time:.1f} seconds")
            time.sleep(5)
