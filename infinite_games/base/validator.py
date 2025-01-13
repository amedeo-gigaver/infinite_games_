# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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


import asyncio
import base64
import copy
import json
import os
import pathlib
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from traceback import print_exception
from typing import List

import backoff
import bittensor as bt
import requests
import torch
import wandb

from infinite_games import __deprecated_version__
from infinite_games.base.neuron import BaseNeuron
from infinite_games.events.base import CLUSTER_EPOCH_2024

TENSOR_DEBUG_SLICE = 12


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    def __init__(self, config=None):
        bt.logging.info("BaseValidatorNeuron __init__ start")
        start_time = time.time()
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        self.average_scores: torch.Tensor = torch.zeros((self.metagraph.n))
        # previous day.
        self.previous_average_scores: torch.Tensor = torch.zeros((self.metagraph.n))
        self.latest_reset_date: datetime = None
        self.scoring_iterations = 0

        # Init sync with the network. Updates the metagraph.
        self.sync(False)

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

        self.USER_ID = 1610011
        self.GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY")

        netrc_path = pathlib.Path.home() / ".netrc"
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is not None:
            bt.logging.info("WANDB_API_KEY is set")
        bt.logging.info("~/.netrc exists:", netrc_path.exists())

        if wandb_api_key is None and not netrc_path.exists():
            bt.logging.warning("WANDB_API_KEY not found in environment variables.")

        if wandb_api_key:
            wandb.init(
                project=f"ig-{self.config.netuid}-validators",
                # entity="infinitegames",
                config={
                    "hotkey": self.wallet.hotkey.ss58_address,
                },
                name=f"validator-{self.uid}-{__deprecated_version__}",
                resume=None,
                dir=self.config.neuron.full_path,
                reinit=True,
            )
        end_time = time.time()
        bt.logging.info(
            f"BaseValidatorNeuron __init__ completed in {end_time - start_time:.2f} seconds"
        )

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {repr(e)}")
                bt.logging.error(traceback.format_exc())
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {repr(e)}")
            bt.logging.error(traceback.format_exc())
            pass

    async def concurrent_forward(self):
        coroutines = [self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync(False)

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")
                start_time = time.time()

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                try:
                    self.sync()
                except Exception as e:
                    bt.logging.error(f"Error during sync: {repr(e)}")
                    bt.logging.error(traceback.format_exc())

                self.step += 1
                end_time = time.time()
                bt.logging.info(
                    f"Validation loop iteration completed in {end_time - start_time:.2f} seconds"
                )

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {err}")
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            # self.wandb_run.finish()
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        start_time = time.time()

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nn.functional.normalize(self.scores, p=1, dim=0)
        bt.logging.trace("raw_weights", raw_weights)
        bt.logging.trace("top10 values", raw_weights.sort()[0])
        bt.logging.trace("top10 uids", raw_weights.sort()[1])

        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids.to("cpu"),
            weights=raw_weights.to("cpu"),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.trace("processed_weights", processed_weights)
        bt.logging.trace("processed_weight_uids", processed_weight_uids)
        bt.logging.info(f"current weight version {self.spec_version}")

        # Set the weights on chain via our subtensor connection.
        set_weight = False
        # Sync the metagraph.
        i = 1
        while (not set_weight) and i < 5:
            try:
                self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.config.netuid,
                    uids=processed_weight_uids,
                    weights=processed_weights,
                    wait_for_finalization=False,
                    version_key=self.spec_version,
                )
                set_weight = True
            except Exception as e:
                bt.logging.error(f"Failed to set weights on attempt {i}: {repr(e)}")
                bt.logging.error(traceback.format_exc())
                i += 1
                time.sleep(4)

        if set_weight:
            bt.logging.info(f"Set weights: {processed_weights}")
        else:
            bt.logging.error("Failed to set weights after multiple attempts.")
        end_time = time.time()
        bt.logging.info(f"set_weights completed in {end_time - start_time:.2f} seconds")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")
        start_time = time.time()

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)
        # Sync the metagraph.

        synced = False
        # Sync the metagraph.
        while not synced:
            try:
                self.metagraph.sync(subtensor=self.subtensor)
                synced = True
            except Exception as e:
                bt.logging.error(f"Try to resync metagraph: {repr(e)}")
                bt.logging.error(traceback.format_exc())
                time.sleep(4)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        end_time = time.time()
        bt.logging.info(f"resync_metagraph completed in {end_time - start_time:.2f} seconds")

    def reset_daily_average_scores(self):
        """Current daily average scores are fixed and saved as previous day results for further moving average calculation"""
        RESET_INTERVAL_SECONDS = 60 * 60 * 24
        # if we dont have reset date, make sure it resets
        latest_reset_date = self.latest_reset_date or (
            datetime.now(timezone.utc) - timedelta(seconds=RESET_INTERVAL_SECONDS + 1)
        )
        latest_reset_ts = latest_reset_date.timestamp()
        now_ts = datetime.now().timestamp()
        # bt.logging.info(f'{now_ts} - {latest_reset_ts}, {now_ts - latest_reset_ts} > {RESET_HOURS}')
        if now_ts - latest_reset_ts > RESET_INTERVAL_SECONDS:
            if datetime.now(timezone.utc).hour > 10:
                bt.logging.info(f"Resetting daily scores: {datetime.now(timezone.utc)}")
                if self.average_scores is None:
                    bt.logging.error("Do not have average scores to set for previous day!")
                else:
                    all_uids = [uid for uid in range(self.metagraph.n.item())]
                    bt.logging.debug(f"Daily average total: {self.average_scores}")
                    self.send_average_scores(
                        miner_scores=list(
                            zip(all_uids, self.average_scores.tolist(), self.scores.tolist())
                        )
                    )
                    self.previous_average_scores = self.average_scores.clone().detach()
                    self.scoring_iterations = 0
                    self.average_scores = torch.zeros(self.metagraph.n.item())
                    bt.logging.info("Daily scores reset, previous day scores saved.")
                self.latest_reset_date = datetime.now()

    def update_scores(self, rewards: torch.FloatTensor, uids: torch.LongTensor):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""
        start_time = time.time()
        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)
        total_neurons = self.metagraph.n.item()
        all_zeros = torch.zeros(total_neurons)
        bt.logging.info(f"Total neurons: {total_neurons}")

        if len(self.scores) < total_neurons:
            # extend score shape in case we have new miners
            self.scores = torch.cat([self.scores, all_zeros])[:total_neurons]
        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_scores: torch.FloatTensor = self.scores.scatter(
            0, uids.clone().detach(), rewards
        ).to(self.device)
        bt.logging.debug(
            f"Scattered scores[-{TENSOR_DEBUG_SLICE}:]:"
            f" {torch.round(scattered_scores, decimals=3)[-TENSOR_DEBUG_SLICE:]} {len(scattered_scores)}"
        )

        alpha: float = self.config.neuron.moving_average_alpha

        if len(self.average_scores) < total_neurons:
            # extend score shape in case we have new miners
            self.average_scores = torch.cat([self.average_scores, all_zeros])[:total_neurons]

        if len(self.previous_average_scores) < total_neurons:
            # extend score shape in case we have new miners
            self.previous_average_scores = torch.cat([self.previous_average_scores, all_zeros])[
                :total_neurons
            ]

        zero_scattered_rewards = torch.zeros(total_neurons).scatter(
            0, uids.clone().detach(), rewards
        )

        bt.logging.debug(
            f"Scattered rewards[-{TENSOR_DEBUG_SLICE}:]:"
            f" {torch.round(zero_scattered_rewards, decimals=3)[-TENSOR_DEBUG_SLICE:]}"
        )
        bt.logging.debug(
            f"Average total[-{TENSOR_DEBUG_SLICE}:]:"
            f" {torch.round(self.average_scores, decimals=3)[-TENSOR_DEBUG_SLICE:]}"
        )
        bt.logging.debug(f"Daily iteration: {self.scoring_iterations + 1}")

        self.average_scores = (
            self.average_scores * self.scoring_iterations + zero_scattered_rewards
        ) / (self.scoring_iterations + 1)
        bt.logging.debug(
            f"New Average total[-{TENSOR_DEBUG_SLICE}:]:"
            f" {torch.round(self.average_scores, decimals=3)[-TENSOR_DEBUG_SLICE:]}"
        )

        alpha = 0.8
        if (
            self.previous_average_scores is not None
            and torch.count_nonzero(self.previous_average_scores).item() != 0
        ):
            bt.logging.info("Recalculate moving average based on previous day")
            self.scores: torch.FloatTensor = alpha * self.average_scores + (
                1 - alpha
            ) * self.previous_average_scores.to(self.device)
        else:
            bt.logging.info("No daily average available yet, prefer scores for moving average")
            self.scores: torch.FloatTensor = alpha * scattered_scores + (
                1 - alpha
            ) * self.scores.to(self.device)
        bt.logging.debug(
            f"Updated moving avg scores[-{TENSOR_DEBUG_SLICE}:]:"
            f" {self.scores[-TENSOR_DEBUG_SLICE:]}"
        )
        self.scoring_iterations += 1
        end_time = time.time()
        bt.logging.info(f"update_scores completed in {end_time - start_time:.2f} seconds")

    @backoff.on_exception(backoff.expo, Exception, max_tries=6)
    def send_average_scores(self, miner_scores=None):
        if not self.GRAFANA_API_KEY:
            return
        miner_logs = ""
        measurement = os.environ.get("AVERAGE_MEASUREMENT_NAME", "miners_average_scores")
        if miner_scores:
            for miner_id, score, total_weight in miner_scores:
                # bt.logging.debug(f'Miner {miner_id} {score} {old_weight} -> {total_weight}')
                miner_logs += f"{measurement},source={miner_id},vali={self.wallet.hotkey.ss58_address} metric={score},weight={total_weight}\n"

        body = f"""
        {miner_logs}
        """

        try:
            response = requests.post(
                "https://influx-prod-24-prod-eu-west-2.grafana.net/api/v1/push/influx/write",
                headers={
                    "Content-Type": "text/plain",
                },
                data=str(body),
                auth=(self.USER_ID, self.GRAFANA_API_KEY),
            )

            status_code = response.status_code
            if status_code != 204:
                bt.logging.error(f'*** Error sending logs! {response.content.decode("utf8")}')
            else:
                bt.logging.debug("*** Grafana logs sent")
        except Exception as e:
            bt.logging.error(f"Error sending average scores: {repr(e)}")
            bt.logging.error(traceback.format_exc())

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=6,
        on_backoff=lambda details: bt.logging.warning(f"Backing off send_interval_data: {details}"),
        on_giveup=lambda details: bt.logging.error(f"Giving up send_interval_data: {details}"),
    )
    def send_interval_data(self, miner_data):
        try:
            v_uid = self.metagraph.hotkeys.index(self.wallet.get_hotkey().ss58_address)
            body = {
                "submissions": [
                    {
                        "unique_event_id": unique_event_id,
                        "provider_type": market_type,
                        "title": None,
                        "prediction": agg_prediction,
                        "outcome": None,
                        "interval_start_minutes": interval_minutes,
                        "interval_agg_prediction": agg_prediction,
                        "interval_agg_count": count,
                        "interval_datetime": (
                            CLUSTER_EPOCH_2024 + timedelta(minutes=interval_minutes)
                        ).isoformat(),
                        "miner_hotkey": self.metagraph.hotkeys[int(miner_uid)],
                        "miner_uid": int(miner_uid),
                        "validator_hotkey": self.wallet.get_hotkey().ss58_address,
                        "validator_uid": int(v_uid),
                    }
                    for miner_uid, unique_event_id, market_type, interval_minutes, agg_prediction, count in miner_data
                ],
                "events": None,
            }
            hk = self.wallet.get_hotkey()
            signed = base64.b64encode(hk.sign(json.dumps(body))).decode("utf-8")
            res = requests.post(
                f"{self.base_api_url}/api/v1/validators/data",
                headers={
                    "Authorization": f"Bearer {signed}",
                    "Validator": self.wallet.get_hotkey().ss58_address,
                },
                json=body,
            )
            res.raise_for_status()
        except Exception as e:
            bt.logging.error(f"Error sending interval data: {repr(e)}", exc_info=True)
            raise e

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")
        start_time = time.time()
        # Save the state of the validator to file.
        try:
            torch.save(
                {
                    "step": self.step,
                    "scores": self.scores,
                    "hotkeys": self.hotkeys,
                    "average_scores": self.average_scores,
                    "previous_average_scores": self.previous_average_scores,
                    "scoring_iterations": self.scoring_iterations,
                    "latest_reset_date": self.latest_reset_date,
                },
                self.config.neuron.full_path + "/state.pt",
            )
        except Exception as e:
            bt.logging.error(f"Error saving state: {repr(e)}")
            bt.logging.error(traceback.format_exc())
        end_time = time.time()
        bt.logging.info(f"State saved in {end_time - start_time:.2f} seconds")

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        start_time = time.time()
        # Load the state of the validator from file.
        try:
            state = torch.load(self.config.neuron.full_path + "/state.pt")
            self.step = state["step"]
            self.scores = state["scores"]
            self.hotkeys = state["hotkeys"]
            if state.get("average_scores") is not None:
                self.average_scores = state["average_scores"]
            if state.get("previous_average_scores") is not None:
                self.previous_average_scores = state["previous_average_scores"]
            if state.get("scoring_iterations") is not None:
                self.scoring_iterations = state["scoring_iterations"]
            if state.get("latest_reset_date"):
                self.latest_reset_date = state["latest_reset_date"]
                bt.logging.info(f"Latest score reset date: {self.latest_reset_date}")
        except FileNotFoundError:
            bt.logging.info(
                f"Neuron state not found in {self.config.neuron.full_path + '/state.pt'}, skip.."
            )
        except Exception as e:
            bt.logging.error(f"Error loading state: {repr(e)}")
            bt.logging.error(traceback.format_exc())
        end_time = time.time()
        bt.logging.info(f"State loaded in {end_time - start_time:.2f} seconds")
