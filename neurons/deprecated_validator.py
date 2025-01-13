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

import asyncio
import base64
import json
import math
import sqlite3
import time
import traceback
from datetime import datetime, timedelta, timezone

import bittensor as bt
import pandas as pd
import requests
import torch

import infinite_games

# import base validator class which takes care of most of the boilerplate
from infinite_games import __deprecated_spec_version__
from infinite_games.base.validator import TENSOR_DEBUG_SLICE, BaseValidatorNeuron
from infinite_games.events.base import (
    CLUSTER_EPOCH_2024,
    CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES,
    EventAggregator,
    EventStatus,
    ProviderEvent,
    ProviderIntegration,
)
from infinite_games.events.ifgames import IFGamesProviderIntegration
from infinite_games.utils.misc import split_chunks
from infinite_games.utils.query import query_miners
from infinite_games.utils.uids import get_miner_data_by_uid

# The minimum time between querying miners in seconds - avoid spamming miners
MIN_TIME_BETWEEN_QUERY_MINERS = 300


class Validator(BaseValidatorNeuron):
    """
    Validator neuron class.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron.
    The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph,
    logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you
    need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average
    of the scores of the miners and using them to set weights at the end of each epoch.
    Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, integrations, db_path="validator.db", config=None):
        bt.logging.info("Validator __init__ start")
        start_time = time.time()
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        self.active_markets = {}
        self.blocktime = 0
        self.event_provider = None
        self.SEND_LOGS_INTERVAL = 60 * 60
        self.SEND_MINER_LOGS_INTERVAL = 60 * 60 * 4
        self.integrations = integrations
        self.db_path = db_path
        self.last_log_block = 0
        self.is_test = self.subtensor.network in ["test", "mock", "local"]
        self.base_api_url = "https://stage.ifgames.win" if self.is_test else "https://ifgames.win"
        if self.is_test:
            log_msg = (
                f"Using provider in test mode with base url: {self.base_api_url};"
                f" Subtensor network: {self.subtensor.network}"
            )
            bt.logging.info(log_msg)
        end_time = time.time()

        self.last_query_time = end_time
        bt.logging.info(f"Validator __init__ completed in {end_time - start_time:.2f} seconds")

    async def initialize_provider(self):
        start_time = time.time()
        if not self.event_provider:
            try:
                self.event_provider: EventAggregator = await EventAggregator.create(
                    state_path=self.config.neuron.full_path + "/events-v2.pickle",
                    integrations=self.integrations,
                    db_path=self.db_path,
                )
                self.event_provider.load_state()
            except Exception as e:
                bt.logging.error(f"Error initializing EventAggregator: {repr(e)}")
                bt.logging.error(traceback.format_exc())
                return
            # self.event_provider.migrate_pickle_to_sql()
            self.event_provider.on_event_updated_hook(self.on_event_update)
            if os.getenv("VALIDATOR_WATCH_EVENTS_DISABLED", "0") == "0":
                # watch for existing registered events
                self.loop.create_task(self.event_provider.watch_events())
                # pull new markets
                self.loop.create_task(self.event_provider.collect_events())
            bt.logging.info(
                f'TARGET_MONITOR_HOTKEY: {os.environ.get("TARGET_MONITOR_HOTKEY", "None")}'
            )
            bt.logging.info(f'GRAFANA_API_KEY: {os.environ.get("GRAFANA_API_KEY", "None")}')
            # if self.wallet.hotkey.ss58_address == os.environ.get('TARGET_MONITOR_HOTKEY'):
            self.loop.create_task(self.track_interval_stats())
            bt.logging.debug("Provider initialized..")
        end_time = time.time()
        bt.logging.info(f"initialize_provider completed in {end_time - start_time:.2f} seconds")

    async def send_interval_stats(self, overwrite_now=None):
        start_time = time.time()
        now = datetime.now(timezone.utc)
        # overwrite_now for testing, freeze_date does not work well with get_event_loop
        # https://github.com/spulec/freezegun/issues/529
        if overwrite_now:
            now = overwrite_now
        minutes_since_epoch = int((now - CLUSTER_EPOCH_2024).total_seconds()) // 60
        # previous interval from current one filled already, sending it.
        previous_interval_start_minutes = (
            minutes_since_epoch
            - (minutes_since_epoch % CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
            - CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
        )
        previous_m20_interval_start_minutes = (
            minutes_since_epoch
            - (minutes_since_epoch % CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
            - (CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 20)
        )
        interval_date_begin = (
            CLUSTER_EPOCH_2024 + timedelta(minutes=previous_m20_interval_start_minutes)
        ).isoformat()
        interval_date_end = (
            CLUSTER_EPOCH_2024 + timedelta(minutes=previous_interval_start_minutes)
        ).isoformat()
        bt.logging.debug(
            f"Sending interval data between: {interval_date_begin=} -> {interval_date_end=}"
        )
        metrics = []
        predictions_data = self.event_provider.get_all_non_exported_event_predictions(
            interval_minutes_begin=previous_m20_interval_start_minutes,
            interval_minutes_end=previous_interval_start_minutes,
        )

        bt.logging.debug(f"Loaded {len(predictions_data)} submissions..")
        for (
            metadata,
            unique_event_id,
            _,
            uid,
            _,
            interval_minutes,
            agg_prediction,
            count,
            _,
            _,
        ) in predictions_data:
            market_type = unique_event_id.split("-")[0]
            if metadata:
                md = json.loads(metadata)
                market_type = md.get("market_type", market_type)
            metrics.append(
                [uid, unique_event_id, market_type, interval_minutes, agg_prediction, count]
            )
        if not metrics:
            bt.logging.info("no new submission to send skip..")

        if metrics and len(metrics) > 0:
            bt.logging.info(f"Total submission to export: {len(metrics)}")
            chunk_metrics = split_chunks(list(metrics), 15000)
            try:
                async for metrics in chunk_metrics:
                    self.send_interval_data(miner_data=metrics)
                    bt.logging.info(f"chunk submissions processed {len(metrics)}")

                await asyncio.sleep(4)
                self.event_provider.mark_submissions_as_exported(
                    interval_minutes_begin=previous_m20_interval_start_minutes,
                    interval_minutes_end=previous_interval_start_minutes,
                )
            except Exception as e:
                bt.logging.error(f"Error sending interval data: {repr(e)}", exc_info=True)

        end_time = time.time()
        bt.logging.info(f"send_interval_stats completed in {end_time - start_time:.2f} seconds")

    async def track_interval_stats(self):
        bt.logging.info("Scheduling sending interval stats.")

        while True:
            await self.send_interval_stats()
            bt.logging.info(
                f"Waiting for next {self.SEND_MINER_LOGS_INTERVAL} seconds to schedule interval logs.."
            )
            await asyncio.sleep(self.SEND_MINER_LOGS_INTERVAL)

    def on_event_update(self, pe: ProviderEvent):
        """Hook called whenever we have settling events. Event removed when we return True"""
        if pe.status == EventStatus.SETTLED:
            start_time = time.time()
            market_type = pe.metadata.get("market_type", pe.market_type)
            event_text = f"{market_type} {pe.event_id}"
            bt.logging.info(
                f"Settled event: {event_text} {pe.description[:100]} answer: {pe.answer}"
            )
            miner_uids = torch.tensor([uid for uid in range(self.metagraph.n.item())])
            correct_ans = pe.answer
            if correct_ans is None:
                bt.logging.info(f"Unknown answer for event, discarding : {pe}")
                return True
            predictions = self.event_provider.get_event_predictions(pe)
            # predictions = pe.miner_predictions
            if not predictions:
                bt.logging.warning(f"No predictions for {pe} skipping..")
                return True
            integration: ProviderIntegration = self.event_provider.get_integration(pe)
            # if not integration:
            #     bt.logging.error(f'no integration found for event {pe}. will skip this event!')
            #     return True
            cutoff = integration.latest_submit_date(pe)
            bt.logging.info(
                f"Miners to update: {len(miner_uids)} submissions: {len(predictions.keys())} from {self.metagraph.n.item()}"
            )
            bt.logging.info(
                f"Register: {pe.registered_date} cutoff: {cutoff} tz: {cutoff.tzinfo}, resolve: {pe.resolve_date}"
            )

            # we take either now or cutoff time (event can be settled earlier)
            cutoff_minutes_since_epoch = int((cutoff - CLUSTER_EPOCH_2024).total_seconds()) // 60
            cutoff_interval_start_minutes = cutoff_minutes_since_epoch - (
                cutoff_minutes_since_epoch % CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
            )
            now = datetime.now(timezone.utc)
            now_minutes_since_epoch = int((now - CLUSTER_EPOCH_2024).total_seconds()) // 60
            now_interval_start_minutes = now_minutes_since_epoch - (
                now_minutes_since_epoch % CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
            )
            bt.logging.info(
                f"Comparing cutoff to now: {cutoff=}: {cutoff_interval_start_minutes}, {now=} {now_interval_start_minutes}"
            )
            effective_finish_start_minutes = min(
                cutoff_interval_start_minutes, now_interval_start_minutes
            )
            start_minutes_since_epoch = (
                int((pe.registered_date - CLUSTER_EPOCH_2024).total_seconds()) // 60
            )
            start_interval_start_minutes = start_minutes_since_epoch - (
                start_minutes_since_epoch % CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
            )
            total_intervals = (
                effective_finish_start_minutes - start_interval_start_minutes
            ) // CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
            first_n_intervals = 1
            bt.logging.info(
                f"{integration.__class__.__name__} {first_n_intervals=} intervals: {pe.registered_date=} {effective_finish_start_minutes=} {pe.resolve_date=} {cutoff=}  total={total_intervals}"
            )
            scores = []
            market_type = pe.metadata.get("market_type", pe.market_type)
            for uid in miner_uids:
                miner_data = get_miner_data_by_uid(self.db_path, int(uid))
                miner_reg_time = datetime.fromisoformat(miner_data["registered_date"]).replace(
                    tzinfo=timezone.utc
                )
                bt.logging.debug(f"miner {uid=} reg time: {miner_reg_time}")
                prediction_intervals = predictions.get(uid.item())
                if market_type == "azuro":
                    # if miner registered after the cutoff.
                    if miner_reg_time >= cutoff:
                        bt.logging.info("new miner assign: 1/2")
                        ans = 1 / 2
                    # if we had a chance to submit, but did not submit anything
                    elif miner_reg_time < cutoff and not prediction_intervals:
                        scores.append(0)
                        continue
                    else:
                        if 0 in prediction_intervals:
                            ans = prediction_intervals[0]["interval_agg_prediction"]
                        else:
                            # fallback if we have intervals assigned for azuro, take last
                            max_interval = max(prediction_intervals.keys())
                            ans = prediction_intervals[max_interval]["interval_agg_prediction"]

                    if ans is None:
                        scores.append(0)
                        continue
                    ans = max(0, min(1, ans))  # Clamp the answer
                    brier_score = 1 - ((ans - correct_ans) ** 2)
                    scores.append(brier_score)
                    bt.logging.debug(
                        f"settled answer for {uid=} for {pe.event_id=} {ans=} {brier_score=}"
                    )
                else:
                    # if miner is registered before the event is streamed
                    if miner_reg_time < pe.registered_date and not prediction_intervals:
                        scores.append(0)
                        continue
                    mk = []

                    weights_sum = 0

                    for interval_start_minutes in range(
                        start_interval_start_minutes,
                        effective_finish_start_minutes,
                        CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES,
                    ):
                        interval_data = (prediction_intervals or {}).get(
                            interval_start_minutes, {"interval_agg_prediction": None}
                        )
                        ans: float = interval_data["interval_agg_prediction"]
                        interval_start_date = CLUSTER_EPOCH_2024 + timedelta(
                            minutes=interval_start_minutes
                        )
                        interval_end_date = CLUSTER_EPOCH_2024 + timedelta(
                            minutes=interval_start_minutes + CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
                        )
                        if miner_reg_time > interval_end_date:
                            ans = 1 / 2

                        current_interval_no = (
                            interval_start_minutes - start_interval_start_minutes
                        ) // CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
                        if current_interval_no + 1 <= first_n_intervals:
                            wk = 1
                        else:
                            wk = math.exp(
                                -(total_intervals / (total_intervals - current_interval_no)) + 1
                            )
                        weights_sum += wk
                        # bt.logging.info(f'answer for {uid=} {interval_start_minutes=} {ans=} total={total_intervals} curr={current_interval_no} {wk=} ')
                        if ans is None:
                            mk.append(0)
                            continue
                        ans = max(0, min(1, ans))  # Clamp the answer
                        brier_score = 1 - ((ans - correct_ans) ** 2)
                        mk.append(wk * brier_score)

                        bt.logging.debug(
                            f"{pe} answer for {uid=} {interval_start_minutes=} {interval_start_date=} {ans=} total={total_intervals} curr={current_interval_no} {wk=} {brier_score=}"
                        )
                    if weights_sum < 0.01:
                        range_list = range(
                            start_interval_start_minutes,
                            effective_finish_start_minutes,
                            CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES,
                        )
                        bt.logging.error(f"Weight WK is zero for event {uid} {pe}  {range_list}")
                    final_avg_score = sum(mk) / weights_sum if weights_sum > 0 else 0
                    bt.logging.info(
                        f"final avg answer for intervals={len(range(start_interval_start_minutes, effective_finish_start_minutes, CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES))} {uid=} {final_avg_score=}"
                    )

                    scores.append(final_avg_score)
            brier_scores = torch.FloatTensor(scores)
            bt.logging.info(
                f"Brier scores[-{TENSOR_DEBUG_SLICE}:]:"
                f" {torch.round(brier_scores, decimals=3)[-TENSOR_DEBUG_SLICE:]}"
            )
            scores = torch.FloatTensor(scores)
            if all(score.item() <= 0.0 for score in scores):
                # bt.logging.info('All effective scores zero for this event!')
                pass
            else:
                alpha = 0
                beta = 1
                non_zeros = scores != 0
                scores[non_zeros] = alpha * scores[non_zeros] + (
                    beta * torch.exp(30 * scores[non_zeros])
                )
            bt.logging.info(
                f"Expd scores[-{TENSOR_DEBUG_SLICE}:]:"
                f" {torch.round(scores, decimals=3)[-TENSOR_DEBUG_SLICE:]}"
            )
            scores = torch.nn.functional.normalize(scores, p=1, dim=0)
            bt.logging.info(
                f"Normalized scores[-{TENSOR_DEBUG_SLICE}:]:"
                f" {torch.round(scores, decimals=3)[-TENSOR_DEBUG_SLICE:]}"
            )
            self.update_scores(scores, miner_uids)
            self.export_scores(p_event=pe, miner_score_data=zip(miner_uids, brier_scores, scores))
            end_time = time.time()
            bt.logging.info(
                f"on_event_update for event {pe.event_id} completed in {end_time - start_time:.2f} seconds"
            )
            return True
        elif pe.status == EventStatus.DISCARDED:
            bt.logging.info(f"Canceled event: {pe} removing from registry!")
            self.event_provider.remove_event(pe)

        return False

    def export_scores(self, p_event: ProviderEvent, miner_score_data):
        start_time = time.time()
        if os.environ.get("ENV") != "pytest":
            try:
                v_uid = self.metagraph.hotkeys.index(self.wallet.get_hotkey().ss58_address)
                body = {
                    "results": [
                        {
                            "event_id": p_event.event_id,
                            "provider_type": p_event.market_type,
                            "title": p_event.description[:50],
                            "description": p_event.description,
                            "category": "event",
                            "start_date": p_event.starts.isoformat() if p_event.starts else None,
                            "end_date": (
                                p_event.resolve_date.isoformat() if p_event.resolve_date else None
                            ),
                            "resolve_date": (
                                p_event.resolve_date.isoformat() if p_event.resolve_date else None
                            ),
                            "settle_date": datetime.now(tz=timezone.utc).isoformat(),
                            "prediction": 0.0,
                            "answer": float(p_event.answer),
                            "miner_hotkey": self.metagraph.hotkeys[miner_uid],
                            "miner_uid": int(miner_uid),
                            "miner_score": float(score),
                            "miner_effective_score": float(effective_score),
                            "validator_hotkey": self.wallet.get_hotkey().ss58_address,
                            "validator_uid": int(v_uid),
                            "metadata": p_event.metadata,
                            "spec_version": str(__deprecated_spec_version__) or "0",
                        }
                        for miner_uid, score, effective_score in miner_score_data
                    ]
                }
                hk = self.wallet.get_hotkey()
                signed = base64.b64encode(hk.sign(json.dumps(body))).decode("utf-8")
                res = requests.post(
                    f"{self.base_api_url}/api/v1/validators/results",
                    headers={
                        "Authorization": f"Bearer {signed}",
                        "Validator": self.wallet.get_hotkey().ss58_address,
                    },
                    json=body,
                )
                if not res.status_code == 200:
                    bt.logging.warning(
                        f"Error processing scores for event {p_event}: {res.status_code=} {res.content=}"
                    )
                else:
                    bt.logging.info(f"Scores processed {res.status_code=} {res.content=}")
                    self.event_provider.mark_event_as_exported(p_event)
                time.sleep(1)
            except Exception as e:
                bt.logging.error(f"Error exporting scores for event {p_event.event_id}: {repr(e)}")
                bt.logging.error(traceback.format_exc())
        else:
            bt.logging.info("Skip export scores in test")
        bt.logging.info(f"export_scores completed in {time.time() - start_time:.2f} seconds")

    async def forward(self):
        """
        The forward function is called by the validator every time step.

        """
        try:
            start_time = time.time()
            await self.initialize_provider()
            self.reset_daily_average_scores()
            self.print_info()
            block_start = self.block
            miner_uids = infinite_games.utils.uids.get_all_uids(self)
            # Create synapse object to send to the miner.
            synapse = infinite_games.protocol.EventPredictionSynapse()
            events_available_for_submission = self.event_provider.get_events_for_submission()
            bt.logging.info(f"Event for submission: {len(events_available_for_submission)}")
            synapse.init(events_available_for_submission)
            bt.logging.info(f"Axons: {len(self.metagraph.axons)}")
            for axon in self.metagraph.axons:
                bt.logging.trace(f"IP: {axon.ip}, hotkey id: {axon.hotkey}")
            self.event_provider.sync_miners(
                [(uid, self.metagraph.axons[uid]) for uid in range(self.metagraph.n.item())],
                block_start,
            )
            bt.logging.info("Querying miners..")
            # The dendrite client queries the network.
            responses = query_miners(
                self.dendrite, [self.metagraph.axons[uid] for uid in miner_uids], synapse
            )
            now = datetime.now(timezone.utc)
            minutes_since_epoch = int((now - CLUSTER_EPOCH_2024).total_seconds()) // 60
            interval_start_minutes = minutes_since_epoch - (
                minutes_since_epoch % (CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
            )
            bt.logging.info(f"Forward pre-loop done in {time.time() - start_time:.2f} seconds")

        except Exception as e:
            bt.logging.error(f"Error in validator forward pre-loop: {repr(e)}", exc_info=True)
            raise e

        loop_msg = (
            f"Forward loop: processing {len(miner_uids)} miners for interval {interval_start_minutes} - "
            f"{len(responses)} responses received."
        )
        bt.logging.info(loop_msg)

        err_count = 0
        start_time = time.time()
        provider_events = []
        minerUids = []
        answers = []
        details = []
        resp_logs = 0
        no_updates_count = 0
        current_uid = -1
        for uid, resp in zip(miner_uids, responses):
            for unique_event_id, event_data in resp.events.items():
                try:
                    score = event_data.get("probability")
                    miner_answered = event_data.get("miner_answered")
                    if miner_answered is False and score is None:
                        no_updates_count += 1

                    provider_event = self.event_provider.get_registered_event(unique_event_id)
                    answers.append(score)
                    minerUids.append(uid.item())
                    provider_events.append(provider_event)

                    if (
                        resp_logs <= 5
                        and uid.item() != current_uid
                        and score is not None
                        and score > 0
                    ):
                        resp_logs += 1
                        bt.logging.debug(
                            f"Sample of valid responses from 5 miners: {uid.item()} "
                            f"{unique_event_id=} {score=} >v1.3.2_{miner_answered=}"
                        )
                        current_uid = uid.item()

                    if not provider_event:
                        details.append("non-registered-event")
                        bt.logging.trace(
                            f"Miner submission for non registered event detected  {uid=} {unique_event_id=}"
                        )
                        continue
                    if score is None:
                        details.append("no-prediction")
                        bt.logging.trace(
                            f"uid: {uid.item()} no prediction for {unique_event_id} sent, skip.."
                        )
                        continue
                    integration = self.event_provider.integrations.get(provider_event.market_type)
                    if not integration:
                        details.append("no-integration")
                        bt.logging.error(
                            f"No integration found to register miner submission {uid=} {unique_event_id=} {score=}"
                        )
                        continue
                    if integration.available_for_submission(provider_event):
                        details.append("valid")
                        bt.logging.trace(
                            f"Submission {uid=} for {interval_start_minutes} {unique_event_id}"
                        )
                    else:
                        details.append("not-open")
                        bt.logging.trace(
                            f"Submission received, but this event is not open for submissions miner {uid=} {unique_event_id=} {score=}"
                        )
                        continue

                except Exception as e:
                    # don't log all errors tracebacks, just first 5 - to avoid spamming logs
                    if err_count < 5:
                        bt.logging.error(
                            f"Error processing miner prediction for uid {uid}: {repr(e)}",
                            exc_info=True,
                        )
                    else:
                        bt.logging.error(
                            f"Error processing miner prediction for uid {uid}: {repr(e)}"
                        )
                    err_count += 1

        if no_updates_count > 0:
            bt.logging.warning(
                f"Count predictions which were not updated from the init value: {no_updates_count}"
            )

        if len(minerUids) > 0:
            try:
                miners_payload = pd.DataFrame()
                miners_payload["provider_event"] = provider_events
                miners_payload["minerUid"] = minerUids
                miners_payload["blocktime"] = block_start
                miners_payload["interval_start_minutes"] = interval_start_minutes
                miners_payload["answer"] = answers
                miners_payload["details"] = details

                self.event_provider.miner_batch_update_predictions(miners_payload)

                elapsed = time.time() - start_time
                bt.logging.info(f"Miners submissions uploaded to DB in {elapsed:.2f} seconds")

            except Exception as e:
                bt.logging.error(f"Error creating miners payload: {repr(e)}", exc_info=True)
                raise e

        else:
            bt.logging.info("No miner submissions received")

        self.blocktime += 1
        try:
            since_last_query = time.time() - self.last_query_time
            if (
                since_last_query < MIN_TIME_BETWEEN_QUERY_MINERS
                and os.environ.get("PYTEST_CURRENT_TEST") is None
            ):
                wait_time = MIN_TIME_BETWEEN_QUERY_MINERS - since_last_query
                bt.logging.info(
                    f"Wait {wait_time:.2f} seconds before query miners again, not enough time elapsed"
                )
                await asyncio.sleep(wait_time)
        except Exception as e:
            bt.logging.error(f"Error in validator forward post-loop: {repr(e)}", exc_info=True)
            raise e
        finally:
            self.last_query_time = time.time()

    def save_state(self):
        super().save_state()
        try:
            self.event_provider.save_state()
        except Exception as e:
            bt.logging.error(f"Error saving event provider state: {repr(e)}")
            bt.logging.error(traceback.format_exc())


# The main function parses the configuration and runs the validator.
bt.debug(True)
if "trace" in ("".join(sys.argv)):
    bt.trace(True)


if __name__ == "__main__":
    version = sys.version
    version_info = sys.version_info
    bt.logging.debug(f"Subnet version {__deprecated_spec_version__}")
    bt.logging.debug(f"Python version {version} {version_info}")
    bt.logging.debug(f"Bittensor version  {bt.__version__}")
    bt.logging.debug(f"SQLite version  {sqlite3.sqlite_version}")
    major, minor, patch = sqlite3.sqlite_version.split(".")
    if int(major) < 3 or int(minor) < 35:
        bt.logging.error(
            f"**** Please install SQLite version 3.35 or higher, current: {sqlite3.sqlite_version}"
        )
        exit(1)
    # if bt.__version__ != "7.0.2":
    #     bt.logging.error(f'**** Please install bittensor==7.0.2 version , current: {bt.__version__}')
    #     exit(1)

    v = Validator(
        integrations=[
            # AzuroProviderIntegration(),
            # PolymarketProviderIntegration(),
            IFGamesProviderIntegration()
        ]
    )
    v.run_in_background_thread()
    time.sleep(2)
    v.thread.join()
