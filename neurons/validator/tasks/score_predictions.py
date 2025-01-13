import copy
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import backoff
import bittensor as bt
import numpy as np
import pandas as pd
import torch
from bittensor.core.metagraph import MetagraphMixin
from bittensor_wallet.wallet import Wallet

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.event import EventsModel
from neurons.validator.models.prediction import PredictionsModel
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.converters import pydantic_models_to_dataframe
from neurons.validator.utils.logger.logger import logger
from neurons.validator.version import __spec_version__ as spec_version

# The base time epoch for clustering intervals.
SCORING_REFERENCE_DATE = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Intervals are grouped in 4-hour blocks (240 minutes).
AGGREGATION_INTERVAL_LENGTH_MINUTES = 60 * 4  # 240

RESET_INTERVAL_SECONDS = 60 * 60 * 24  # 24 hours

BLOCK_DURATION = 12  # 12 seconds block duration from bittensor

EXP_FACTOR_K = 30
NEURON_MOVING_AVERAGE_ALPHA = 0.8


class ScorePredictions(AbstractTask):
    interval: float
    page_size: int
    api_client: IfGamesClient
    db_operations: DatabaseOperations
    config: bt.Config

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        api_client: IfGamesClient,
        metagraph: MetagraphMixin,
        config: bt.Config,
        subtensor: bt.Subtensor,
        wallet: Wallet,  # type: ignore
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # get current hotkeys and uids
        # regularly update these during and after each event scoring
        self.metagraph = metagraph
        self.current_hotkeys = None
        self.n_hotkeys = None
        self.current_uids = None
        self.metagraph_lite_sync()

        self.config = config
        self.subtensor = subtensor
        self.wallet = wallet
        self.vali_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.weights_rate_limit = self.subtensor.weights_rate_limit(self.config.netuid)
        self.last_set_weights_at = round(time.time())

        self.spec_version = spec_version

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.api_client = api_client
        self.miners_last_reg = None

        # keep the same path for backwards compatibility
        # /home/vscode/.bittensor/miners/validator/default/netuid155/validator/state.pt
        self.state_file = Path(
            self.config.logging.logging_dir,
            self.config.wallet.name,
            self.config.wallet.hotkey,
            f"netuid{self.config.netuid}",
            "validator",  # hardcoded to validator, state.pt used only for validator
            "state.pt",
        ).expanduser()

        self.errors_count = 0

        # load last object state
        self.state = None
        self.load_state()

    @property
    def name(self):
        return "score-predictions"

    @property
    def interval_seconds(self):
        return self.interval

    def metagraph_lite_sync(self):
        # sync the metagraph in lite mode
        self.metagraph.sync(lite=True)
        #  WARNING! hotkeys is a list[str] and uids is a torch.tensor
        self.current_hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.n_hotkeys = len(self.current_hotkeys)
        self.current_uids = copy.deepcopy(self.metagraph.uids)

    def save_state(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state, self.state_file)
        except Exception:
            logger.exception("Failed to save state.")
        logger.debug("State saved.")

    def _create_new_state(self) -> dict:
        """Create a new state if the state file does not exist."""
        return {
            "step": 0,  # Unused but kept for compatibility
            "hotkeys": self.current_hotkeys,
            "scores": torch.zeros((self.n_hotkeys), dtype=torch.float32),
            "average_scores": torch.zeros((self.n_hotkeys), dtype=torch.float32),
            "previous_average_scores": torch.zeros((self.n_hotkeys), dtype=torch.float32),
            "scoring_iterations": 0,  # TODO: Change this to a list for new miners
            # reset the state to the previous day midnight
            # it will be updated to the current day midnight after the first event
            "latest_reset_date": (
                datetime.now(timezone.utc) - timedelta(seconds=RESET_INTERVAL_SECONDS)
            ).replace(hour=0, minute=0, second=0, microsecond=0),
            "miner_uids": self.current_uids,
        }

    def _load_existing_state(self) -> dict:
        """Load the state from the state file."""
        return torch.load(self.state_file)

    def _realign_loaded_state(self, state: dict) -> dict:
        """Realign the state to match the current hotkeys and UIDs using pandas."""
        # Convert current hotkeys and UIDs to a DataFrame
        current_df = pd.DataFrame(
            {
                "hotkey": self.current_hotkeys,
                "miner_uid": self.current_uids.tolist(),
            }
        )

        # Convert existing state to a DataFrame
        # ignore miner_uids - they might not be present in the old state file
        # we keep only the current hotkeys and their associated uids
        state_df = pd.DataFrame(
            {
                "hotkey": state["hotkeys"],
                "scores": state["scores"].tolist(),
                "average_scores": state["average_scores"].tolist(),
                "previous_average_scores": state["previous_average_scores"].tolist(),
            }
        )

        # Perform an inner join on hotkeys to realign
        aligned_df = pd.merge(
            current_df,
            state_df,
            on=[
                "hotkey",
            ],
            how="left",
        )

        # Fill missing agerage values with quantile 20
        aligned_df = aligned_df.infer_objects(copy=False)
        aligned_df["scores"] = aligned_df["scores"].fillna(0.0)
        avg_quantile_20 = aligned_df["average_scores"].dropna().quantile(0.2)
        prev_avg_quantile_20 = aligned_df["previous_average_scores"].dropna().quantile(0.2)
        aligned_df["average_scores"] = aligned_df["average_scores"].fillna(avg_quantile_20)
        aligned_df["previous_average_scores"] = aligned_df["previous_average_scores"].fillna(
            prev_avg_quantile_20
        )

        # Convert back to torch tensors
        aligned_state = {
            "hotkeys": aligned_df["hotkey"].tolist(),
            "miner_uids": torch.tensor(aligned_df["miner_uid"].tolist(), dtype=torch.long),
            "scores": torch.tensor(aligned_df["scores"].tolist(), dtype=torch.float32),
            "average_scores": torch.tensor(
                aligned_df["average_scores"].tolist(), dtype=torch.float32
            ),
            "previous_average_scores": torch.tensor(
                aligned_df["previous_average_scores"].tolist(), dtype=torch.float32
            ),
        }

        # Update the state with the aligned state
        # there are more keys in the state, but they are not updated here
        updated_state = copy.deepcopy(state)
        updated_state.update(aligned_state)

        return updated_state

    def load_state(self) -> dict:
        if not self.state_file.exists():
            self.errors_count += 1
            logger.error(
                "State file does not exist, creating a new state.",
                extra={"state_file_path": str(self.state_file)},
            )
            self.state = self._create_new_state()  # new state will be aligned
            return self.state

        try:
            logger.debug("State file found - loading state.")
            state = self._load_existing_state()
            self.state = self._realign_loaded_state(state)
            logger.debug(
                "State loaded and realigned.",
                extra={
                    "state_file_path": str(self.state_file),
                },
            )
        except Exception:
            self.errors_count += 1
            logger.exception(
                "Failed to load state - creating a new state.",
                extra={"state_file_path": str(self.state_file)},
            )
            self.state = self._create_new_state()

        return self.state

    def minutes_since_epoch(self, dt: datetime) -> int:
        """Convert a given datetime to the 'minutes since the reference date'."""
        return int((dt - SCORING_REFERENCE_DATE).total_seconds()) // 60

    def align_to_interval(self, minutes_since: int) -> int:
        """
        Align a given number of minutes_since_epoch down to
        the nearest AGGREGATION_INTERVAL_LENGTH_MINUTES boundary.
        """
        return minutes_since - (minutes_since % AGGREGATION_INTERVAL_LENGTH_MINUTES)

    def process_miner_event_score(
        self, event: EventsModel, miner_predictions: pd.DataFrame, context: dict
    ) -> float:
        # Removed the special treatment for azuro events, made no sense

        # outcome is text in DB :|
        outcome = float(event.outcome)

        # if miners could have predicted but didn't, gets a score of 0
        miner_reg_date = self.miners_last_reg[
            self.miners_last_reg["miner_uid"] == context["miner_uid"]
        ]["registered_date"].iloc[
            0
        ]  # DB primary key ensures only one row
        miner_reg_date = miner_reg_date.to_pydatetime().replace(tzinfo=timezone.utc)
        if miner_reg_date < event.registered_date and miner_predictions.empty:
            logger.debug(
                "Miner did not predict for an event, gets 0.0 rema_brier_score.",
                extra={
                    "miner_uid": context["miner_uid"],
                    "unique_event_id": event.unique_event_id,
                    "miner_reg_date": miner_reg_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_reg_date": event.registered_date.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            return {
                "rema_brier_score": 0.0,
                "rema_prediction": -999,
            }

        weights_brier_sum = 0
        weights_pred_sum = 0
        weights_sum = 0
        miner_reg_date_since_epoch = self.minutes_since_epoch(miner_reg_date)

        for interval_idx in range(context["n_intervals"]):
            interval_start = (
                context["registered_date_start_minutes"]
                + interval_idx * AGGREGATION_INTERVAL_LENGTH_MINUTES
            )
            interval_end = interval_start + AGGREGATION_INTERVAL_LENGTH_MINUTES

            if miner_reg_date_since_epoch > interval_end:
                # if miner registered after the interval, gets a neutral score
                ans = 0.5
            elif (
                miner_predictions.empty
                or interval_start not in miner_predictions["interval_start_minutes"].values
            ):
                # if miner should have answered but didn't, assume wrong prediction
                # and will get a m1_brier_score of 0
                ans = 1 - outcome
            else:
                agg_predictions = miner_predictions[
                    miner_predictions["interval_start_minutes"] == interval_start
                ]["interval_agg_prediction"]

                # DB primary key for predictions ensures maximum one prediction
                # clip the prediction between 0 and 1
                ans = agg_predictions.iloc[0].clip(0, 1)

            m1_brier_score = 1 - ((ans - outcome) ** 2)

            # reverse exponential MA (rema): oldest interval gets the highest weight = 1
            wk = math.exp(-(context["n_intervals"] / (context["n_intervals"] - interval_idx)) + 1)
            weights_sum += wk
            weights_pred_sum += wk * ans
            weights_brier_sum += wk * m1_brier_score

        if weights_sum < 0.01:
            self.errors_count += 1
            logger.debug(
                "Weights sum is too low for an event-miner, assigning 0.0 rema_brier_score.",
                extra={
                    "miner_uid": context["miner_uid"],
                    "unique_event_id": event.unique_event_id,
                    "weights_sum": weights_sum,
                    "n_intervals": context["n_intervals"],
                    "miner_reg_date": miner_reg_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_reg_date": event.registered_date.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            rema_brier_score = 0.0
            rema_prediction = -99
        else:
            rema_brier_score = weights_brier_sum / weights_sum
            rema_prediction = weights_pred_sum / weights_sum

        remas = {
            "rema_brier_score": rema_brier_score,
            "rema_prediction": rema_prediction,
        }
        return remas

    def score_predictions(
        self, event: EventsModel, predictions: list[PredictionsModel]
    ) -> pd.DataFrame:
        # Convert cutoff and now to minutes since epoch, then align them to the interval start
        effective_cutoff_minutes = self.minutes_since_epoch(event.cutoff)
        effective_cutoff_start_minutes = self.align_to_interval(effective_cutoff_minutes)

        # Determine when we started, based on registered_date
        registered_date_minutes = self.minutes_since_epoch(event.registered_date)
        registered_date_start_minutes = self.align_to_interval(registered_date_minutes)

        n_intervals = (
            effective_cutoff_start_minutes - registered_date_start_minutes
        ) // AGGREGATION_INTERVAL_LENGTH_MINUTES
        scores_ls = []

        for miner in self.miners_last_reg.itertuples():
            miner_uid = int(miner.miner_uid)
            miner_hotkey = miner.miner_hotkey
            miner_predictions = pydantic_models_to_dataframe(
                [p for p in predictions if int(p.minerUid) == miner_uid]
            )
            context = {
                "miner_uid": miner_uid,
                "registered_date_start_minutes": registered_date_start_minutes,
                "n_intervals": n_intervals,
            }

            try:
                miner_remas = self.process_miner_event_score(event, miner_predictions, context)

                scores_ls.append(
                    {
                        "miner_uid": miner_uid,
                        "hotkey": miner_hotkey,
                        "rema_brier_score": miner_remas["rema_brier_score"],
                        "rema_prediction": miner_remas["rema_prediction"],
                    },
                )
            except Exception:
                # avoid impacting other miners if one miner fails
                self.errors_count += 1
                logger.exception(
                    "Failed to process miner predictions.",
                    extra={
                        "miner_uid": miner_uid,
                        "miner_hotkey": miner_hotkey,
                        "unique_event_id": event.unique_event_id,
                    },
                )

        scores_df = pd.DataFrame(scores_ls)

        if scores_df.empty:
            self.errors_count += 1
            logger.error(
                "No scores calculated for an event.",
                extra={"unique_event_id": event.unique_event_id},
            )
            return pd.DataFrame(
                columns=["miner_uid", "hotkey", "rema_brier_score", "rema_prediction"]
            )

        # if any score is outside the range [0, 1] or None, log an error
        if scores_df["rema_brier_score"].isnull().any():
            self.errors_count += 1
            logger.error(
                "Some Brier scores are None for an event.",
                extra={"unique_event_id": event.unique_event_id},
            )
        if (scores_df["rema_brier_score"] < 0).any() or (scores_df["rema_brier_score"] > 1).any():
            self.errors_count += 1
            logger.error(
                "Scores outside the range [0, 1] for an event.",
                extra={"unique_event_id": event.unique_event_id},
            )

        return scores_df

    def normalize_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        processed_scores = scores.copy()

        processed_scores["normalized_score"] = np.exp(
            EXP_FACTOR_K * processed_scores["rema_brier_score"]
        )
        logger.debug(
            "Scores exponentiation sample.",
            extra={"processed_scores": processed_scores.head(n=5).to_dict(orient="index")},
        )

        # Normalize the scores - sum cannot be 0, all elements are >= 1
        processed_scores["normalized_score"] /= processed_scores["normalized_score"].sum()
        logger.debug(
            "Scores normalization sample.",
            extra={"processed_scores": processed_scores.head(n=5).to_dict(orient="index")},
        )

        return processed_scores

    def update_daily_scores(self, norm_scores: pd.DataFrame) -> pd.DataFrame:
        self.metagraph_lite_sync()
        # if the miners are not anymore in the current uid - hotkeys, remove them
        # TODO: add new miners to each event immediately? = right join
        miners_df = pd.DataFrame(
            {
                "miner_uid": self.current_uids.tolist(),
                "hotkey": self.current_hotkeys,
            }
        )
        norm_scores_ext = pd.merge(norm_scores, miners_df, on=["miner_uid", "hotkey"], how="inner")

        # TODO: remove this after we confirm the bug is fixed
        logger.debug(
            "State info before the daily update.",
            extra={
                "miner_uid_length": len(self.state["miner_uids"].tolist()),
                "hotkey_length": len(self.state["hotkeys"]),
                "eff_scores_length": len(self.state["scores"].tolist()),
                "average_scores_length": len(self.state["average_scores"].tolist()),
                "previous_average_scores_length": len(
                    self.state["previous_average_scores"].tolist()
                ),
            },
        )

        state_df = pd.DataFrame(
            {
                "miner_uid": self.state["miner_uids"].tolist(),
                "hotkey": self.state["hotkeys"],
                "eff_scores": self.state["scores"].tolist(),
                "average_scores": self.state["average_scores"].tolist(),
                "previous_average_scores": self.state["previous_average_scores"].tolist(),
            }
        )

        norm_scores_ext = pd.merge(
            norm_scores_ext, state_df, on=["miner_uid", "hotkey"], how="left"
        )

        # miner can be new -> not in the state file which can be hours old
        # calculate the 80th (lowest) percentile = quantile 0.2 for existing scores
        avg_quantile_20 = norm_scores_ext["average_scores"].dropna().quantile(0.2)
        prev_avg_quantile_20 = norm_scores_ext["previous_average_scores"].dropna().quantile(0.2)

        # fill the NAs with the quantile 20
        norm_scores_ext = norm_scores_ext.infer_objects(copy=False)
        norm_scores_ext["average_scores"] = norm_scores_ext["average_scores"].fillna(
            avg_quantile_20
        )
        norm_scores_ext["previous_average_scores"] = norm_scores_ext[
            "previous_average_scores"
        ].fillna(prev_avg_quantile_20)

        # update the daily average
        norm_scores_ext["average_scores"] = (
            norm_scores_ext["average_scores"] * self.state["scoring_iterations"]
            + norm_scores_ext["normalized_score"]
        ).div(self.state["scoring_iterations"] + 1)

        # update the moving average - we could be missing the state file
        if (norm_scores_ext["previous_average_scores"] == 0).all():
            logger.warning("Missing the first iteration for moving average.")
            # TODO: kept it like the original code, but should be revisited
            norm_scores_ext["eff_scores"] = (
                NEURON_MOVING_AVERAGE_ALPHA * norm_scores_ext["average_scores"]
                + (1 - NEURON_MOVING_AVERAGE_ALPHA) * norm_scores_ext["eff_scores"]
            )
        else:
            norm_scores_ext["eff_scores"] = (
                NEURON_MOVING_AVERAGE_ALPHA * norm_scores_ext["average_scores"]
                + (1 - NEURON_MOVING_AVERAGE_ALPHA) * norm_scores_ext["previous_average_scores"]
            )

        if norm_scores_ext.isnull().any().any():
            self.errors_count += 1
            logger.error(
                "Some entries in norm_scores are None.",
                extra={
                    "norm_scores_ext": norm_scores_ext[norm_scores_ext.isnull().any(axis=1)]
                    .head(n=5)
                    .to_dict(orient="index")
                },
            )
            norm_scores_ext.fillna(0.0, inplace=True)

        return norm_scores_ext

    def update_state(self, norm_scores_extended: pd.DataFrame):
        """
        Update the state to the last scores and current hotkeys & uids
        """

        self.metagraph_lite_sync()
        self.state["scoring_iterations"] += 1

        # realign the scores to the current hotkeys - right join:
        # - new hotkeys get effective score 0 right after registration
        #  but quantile 20 for the averages (p80 below median)
        # - remove the hotkeys that are not in the current hotkeys
        # this seems to be redundant to update_daily_scores,
        # but miner can be new and without predictions for the event
        norm_scores_aligned = norm_scores_extended.merge(
            pd.DataFrame(
                {
                    "miner_uid": self.current_uids.tolist(),
                    "hotkey": self.current_hotkeys,
                }
            ),
            on=["miner_uid", "hotkey"],
            how="right",
        )
        norm_scores_aligned = norm_scores_aligned.infer_objects(copy=False)
        norm_scores_aligned["eff_scores"] = norm_scores_aligned["eff_scores"].fillna(0.0)
        avg_quantile_20 = norm_scores_aligned["average_scores"].dropna().quantile(0.2)
        prev_avg_quantile_20 = norm_scores_aligned["previous_average_scores"].dropna().quantile(0.2)
        norm_scores_aligned["average_scores"] = norm_scores_aligned["average_scores"].fillna(
            avg_quantile_20
        )
        norm_scores_aligned["previous_average_scores"] = norm_scores_aligned[
            "previous_average_scores"
        ].fillna(prev_avg_quantile_20)

        # if there are duplicates, log error
        if norm_scores_aligned.duplicated(subset=["miner_uid", "hotkey"]).any():
            duplicated_rows = norm_scores_aligned[
                norm_scores_aligned.duplicated(subset=["miner_uid", "hotkey"], keep=False)
            ]
            self.errors_count += 1
            logger.error(
                "Duplicated miner_uid-hotkey pairs in the normalized scores.",
                extra={"norm_scores_extended": duplicated_rows.to_dict(orient="index")},
            )
            norm_scores_aligned.drop_duplicates(subset=["miner_uid", "hotkey"], inplace=True)

        self.state["scores"] = torch.tensor(
            norm_scores_aligned["eff_scores"].values, dtype=torch.float32
        )
        self.state["average_scores"] = torch.tensor(
            norm_scores_aligned["average_scores"].values, dtype=torch.float32
        )
        self.state["previous_average_scores"] = torch.tensor(
            norm_scores_aligned["previous_average_scores"].values, dtype=torch.float32
        )
        self.state["miner_uids"] = torch.tensor(
            norm_scores_aligned["miner_uid"].values, dtype=torch.long
        )
        self.state["hotkeys"] = norm_scores_aligned["hotkey"].tolist()

        # this is what we will export to the ClickHouse DB and used to set the weights
        return norm_scores_aligned

    def set_weights(self):
        # do not attempt to set weights more often than the rate limit
        blocks_since_last_attempt = (
            round(time.time()) - self.last_set_weights_at
        ) // BLOCK_DURATION
        if blocks_since_last_attempt < self.weights_rate_limit:
            logger.debug(
                "Not setting the weights - not enough blocks passed.",
                extra={
                    "last_set_weights_at": self.last_set_weights_at,
                    "blocks_since_last_attempt": blocks_since_last_attempt,
                    "weights_rate_limit": self.weights_rate_limit,
                },
            )
            return True, "Not enough blocks passed."
        else:
            logger.debug(
                "Attempting to set the weights - enough blocks passed.",
                extra={
                    "last_set_weights_at": self.last_set_weights_at,
                    "blocks_since_last_attempt": blocks_since_last_attempt,
                    "weights_rate_limit": self.weights_rate_limit,
                },
            )
            self.last_set_weights_at = round(time.time())

        # re-normalize the scores for the weights - should be already normalized
        raw_weights = torch.nn.functional.normalize(self.state["scores"], p=1, dim=0)

        # if all raw weights are 0, log an error - only if the state.pt was just recreated
        if torch.all(raw_weights == 0):
            self.errors_count += 1
            logger.error(
                "All raw weights are 0, not setting the weights.",
                extra={"raw_weights[:10]": raw_weights.tolist()[:10]},
            )
            return False, "All raw weights are 0."

        if not torch.equal(raw_weights, self.state["scores"]):
            logger.warning(
                "Scores normalized for the weights.",
                extra={
                    "scores[:10]": self.state["scores"].tolist()[:10],
                    "weights[:10]": raw_weights.tolist()[:10],
                },
            )

        # this is only for logging purposes
        sorted_indices = torch.argsort(raw_weights, descending=True)
        sorted_indices_rev = sorted_indices.flip(dims=[0])
        logger.debug(
            "Top 10 and bottom 10 weights.",
            extra={
                "top_10_weights": raw_weights[sorted_indices][:10].tolist(),
                "top_10_uids": self.state["miner_uids"][sorted_indices][:10].tolist(),
                "bottom_10_weights": raw_weights[sorted_indices_rev][:10].tolist(),
                "bottom_10_uids": self.state["miner_uids"][sorted_indices_rev][:10].tolist(),
            },
        )

        # set the weights in the metagraph
        processed_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.state["miner_uids"].to("cpu"),
            weights=raw_weights.to("cpu"),
            metagraph=self.metagraph,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
        )

        if processed_uids is None or processed_weights is None:
            self.errors_count += 1
            logger.error(
                "Failed to process the weights - received None.",
                extra={
                    "processed_uids[:10]": (
                        processed_uids.tolist()[:10] if processed_uids is not None else None
                    ),
                    "processed_weights[:10]": (
                        processed_weights.tolist()[:10] if processed_weights is not None else None
                    ),
                },
            )
            return False, "Failed to process the weights."

        # process_weights excludes the zero weights
        mask = raw_weights != 0
        if not torch.equal(processed_uids, self.state["miner_uids"][mask]):
            self.errors_count += 1
            logger.error(
                "Processed UIDs do not match the original UIDs.",
                extra={
                    "processed_uids[:10]": processed_uids.tolist()[:10],
                    "original_uids[:10]": self.state["miner_uids"].tolist()[:10],
                },
            )
            return False, "Processed UIDs do not match the original UIDs."

        if not torch.equal(processed_weights, raw_weights[mask]):
            logger.warning(
                "Processed weights do not match the original weights.",
                extra={
                    "processed_weights[:10]": processed_weights.tolist()[:10],
                    "original_weights[:10]": raw_weights.tolist()[:10],
                },
            )

        successful, sw_msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=processed_uids,
            weights=processed_weights,
            version_key=self.spec_version,
            wait_for_inclusion=False,
            wait_for_finalization=False,  # can take more than 5 mins when True
            max_retries=5,
        )

        if not successful:
            self.errors_count += 1
            logger.error(
                "Failed to set the weights.",
                extra={
                    "fail_msg": sw_msg,
                    "processed_uids[:10]": processed_uids.tolist()[:10],
                    "processed_weights[:10]": processed_weights.tolist()[:10],
                },
            )
        else:
            logger.debug(
                "Weights set successfully.", extra={"last_set_weights_at": self.last_set_weights_at}
            )

        return successful, sw_msg

    @backoff.on_exception(
        backoff.expo,
        Exception,  # TODO: specify the exception
        max_tries=3,
        factor=2,
        max_time=60,
        on_backoff=lambda details: logger.warning(
            "Retrying export scores.",
            extra={"details": str(details.get("exception", "unknown exception"))},
        ),
        on_giveup=lambda details: logger.error(
            "Failed to export scores.",
            extra={"details": str(details.get("exception", "unknown exception"))},
        ),
    )
    async def export_scores(self, event: EventsModel, final_scores: pd.DataFrame):
        body = {
            "results": [
                {
                    "event_id": event.event_id,  # awful: backend reconstructs unique_event_id
                    "provider_type": event.market_type,
                    "title": event.description[:50],  # as in the original
                    "description": event.description,
                    "category": "event",  # as in the original
                    "start_date": event.starts.isoformat() if event.starts else None,
                    "end_date": event.resolve_date.isoformat() if event.resolve_date else None,
                    "resolve_date": event.resolve_date.isoformat() if event.resolve_date else None,
                    "settle_date": event.cutoff.isoformat(),  # as in the original
                    "prediction": row.rema_prediction,
                    "answer": float(event.outcome),
                    "miner_hotkey": row.hotkey,
                    "miner_uid": row.miner_uid,
                    "miner_score": row.rema_brier_score,
                    "miner_effective_score": row.eff_scores,
                    "validator_hotkey": self.wallet.get_hotkey().ss58_address,
                    "validator_uid": int(self.vali_uid),
                    "metadata": json.loads(event.metadata),
                    "spec_version": str(self.spec_version),
                }
                for _, row in final_scores.iterrows()
            ]
        }

        await self.api_client.post_scores(scores=body)
        logger.debug(
            "Scores exported.",
            extra={
                "event_id": event.unique_event_id,
                "len_scores": len(final_scores),
            },
        )
        return body  # for testing

    def check_reset_daily_scores(self):
        now_dt = datetime.now(timezone.utc)
        if self.state["latest_reset_date"].tzinfo is None:
            self.state["latest_reset_date"] = self.state["latest_reset_date"].replace(
                tzinfo=timezone.utc
            )
        seconds_since_reset = (now_dt - self.state["latest_reset_date"]).total_seconds()

        # change previous logic - adjust to reset to every midnight
        if seconds_since_reset <= RESET_INTERVAL_SECONDS:
            logger.debug(
                "Not resetting daily scores.", extra={"seconds_since_reset": seconds_since_reset}
            )
            return

        today_midnight = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        # if all average scores are 0 - either new state file or recently reset - error
        if (self.state["average_scores"] == 0).all():
            self.errors_count += 1
            logger.error("Reset daily scores: average scores are 0, not resetting.")
        else:
            logger.debug(
                "Resetting daily scores.",
                extra={
                    "seconds_since_reset": seconds_since_reset,
                    "latest_reset_date": self.state["latest_reset_date"].isoformat(),
                    "new_latest_reset_date": today_midnight.isoformat(),
                },
            )
            self.state["previous_average_scores"] = self.state["average_scores"]
            self.state["average_scores"] = torch.zeros((self.n_hotkeys), dtype=torch.float32)
            self.state["latest_reset_date"] = today_midnight
            self.state["scoring_iterations"] = 0
            self.save_state()

    def set_right_cutoff(self, event: EventsModel):
        # TODO: cleanup this after we have resolve date for all events in DB
        # https://linear.app/infinite-games/issue/INF-203/events-solved-before-cutoff

        # Ensure all dates are converted to UTC
        def to_utc(input_dt: datetime) -> datetime:
            return (
                input_dt.astimezone(timezone.utc)
                if input_dt.tzinfo
                else input_dt.replace(tzinfo=timezone.utc)
            )

        # Convert event dates to UTC if they exist
        if event.resolved_at:
            event.resolved_at = to_utc(event.resolved_at)
        if event.cutoff:
            event.cutoff = to_utc(event.cutoff)
        if event.registered_date:  # used in the scoring
            event.registered_date = to_utc(event.registered_date)

        # Get the right cutoff date
        if event.resolved_at and event.cutoff:
            effective_cutoff = min(event.cutoff, event.resolved_at)
        elif event.cutoff:
            effective_cutoff = min(event.cutoff, datetime.now(timezone.utc))
        elif event.resolved_at:
            effective_cutoff = min(event.resolved_at, datetime.now(timezone.utc))
        else:
            effective_cutoff = datetime.now(timezone.utc)

        # dirty: mutate the event object
        event.cutoff = effective_cutoff
        event.resolved_at = effective_cutoff
        return event

    async def score_event(self, event: EventsModel):
        if not event.cutoff:
            self.errors_count += 1
            logger.error("Event has no cutoff date.", extra={"event_id": event.unique_event_id})

        event_id = event.unique_event_id
        event = self.set_right_cutoff(event)

        predictions = await self.db_operations.get_predictions_for_scoring(event_id)

        if not predictions:
            logger.warning(
                "There are no predictions for a settled event.",
                extra={"event_id": event_id, "event_cutoff": event.cutoff.isoformat()},
            )
            return

        scores = self.score_predictions(event=event, predictions=predictions)
        logger.debug(
            "Scores calculated, sample below.",
            extra={"scores": scores.head(n=5).to_dict(orient="index")},
        )

        norm_scores = self.normalize_scores(scores=scores)

        norm_scores_extended = self.update_daily_scores(norm_scores=norm_scores)

        norm_scores_aligned = self.update_state(norm_scores_extended=norm_scores_extended)

        self.save_state()

        await self.db_operations.mark_event_as_processed(unique_event_id=event.unique_event_id)

        await self.export_scores(event=event, final_scores=norm_scores_aligned)

        await self.db_operations.mark_event_as_exported(unique_event_id=event.unique_event_id)

    async def run(self):
        miners_last_reg_rows = await self.db_operations.get_miners_last_registration()

        if not miners_last_reg_rows:
            logger.error("No miners found in the DB, skipping scoring!")
            return

        self.miners_last_reg = pydantic_models_to_dataframe(miners_last_reg_rows)
        # for some reason, someone decided to store the miner_uid as a string in the DB
        self.miners_last_reg["miner_uid"] = self.miners_last_reg["miner_uid"].astype(
            pd.Int64Dtype()
        )

        # TODO: separate the event loop from the set_weights and reset daily scores
        events_to_score = await self.db_operations.get_events_for_scoring()
        if not events_to_score:
            logger.debug("No events to score.")
        else:
            logger.debug("Found events to score.", extra={"n_events": len(events_to_score)})

            for event in events_to_score:
                await self.score_event(event)

            logger.debug(
                "Events scoring loop finished. Confirm that errors count in logs is 0!",
                extra={"errors_count_in_logs": self.errors_count},
            )

        success, sw_msg = self.set_weights()
        # seems redundant, but could be different reasons from self.set_weights
        if not success:
            self.errors_count += 1
            logger.error(
                "Failed to set the weights!",
                extra={"fail_msg": sw_msg},
            )

        self.check_reset_daily_scores()

        logger.debug(
            "Score Predictions run finished. Resetting errors count.",
            extra={"errors_count_in_logs": self.errors_count},
        )
        self.errors_count = 0
