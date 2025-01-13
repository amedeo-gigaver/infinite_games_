import sqlite3
import sys

from bittensor import Dendrite, Subtensor
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.scheduler.tasks_scheduler import TasksScheduler
from neurons.validator.tasks.export_predictions import ExportPredictions
from neurons.validator.tasks.pull_events import PullEvents
from neurons.validator.tasks.query_miners import QueryMiners
from neurons.validator.tasks.resolve_events import ResolveEvents
from neurons.validator.tasks.score_predictions import ScorePredictions
from neurons.validator.utils.config import get_config
from neurons.validator.utils.env import assert_requirements
from neurons.validator.utils.logger.logger import logger


async def main():
    # Assert system requirements
    assert_requirements()

    # Start session id
    logger.start_session()

    # Set dependencies
    config, ifgames_env, db_path = get_config()

    # Bittensor stuff
    bt_netuid = config.get("netuid")
    bt_network = config.get("subtensor").get("network")
    bt_wallet = Wallet(config=config)
    bt_dendrite = Dendrite(wallet=bt_wallet)
    bt_subtensor = Subtensor(config=config)
    bt_metagraph = bt_subtensor.metagraph(netuid=bt_netuid, lite=True)

    validator_hotkey = bt_wallet.hotkey.ss58_address
    validator_uid = bt_metagraph.hotkeys.index(validator_hotkey)

    # Components
    db_client = DatabaseClient(db_path=db_path, logger=logger)
    db_operations = DatabaseOperations(db_client=db_client)
    api_client = IfGamesClient(env=ifgames_env, logger=logger, bt_wallet=bt_wallet)

    # Migrate db
    await db_client.migrate()

    # Tasks
    pull_events_task = PullEvents(
        interval_seconds=50.0, page_size=50, db_operations=db_operations, api_client=api_client
    )

    resolve_events_task = ResolveEvents(
        interval_seconds=1800.0, db_operations=db_operations, api_client=api_client, logger=logger
    )

    query_miners_task = QueryMiners(
        interval_seconds=180.0,
        db_operations=db_operations,
        dendrite=bt_dendrite,
        metagraph=bt_metagraph,
        logger=logger,
    )

    export_predictions_task = ExportPredictions(
        interval_seconds=180.0,
        db_operations=db_operations,
        api_client=api_client,
        batch_size=50,
        validator_uid=validator_uid,
        validator_hotkey=validator_hotkey,
        logger=logger,
    )

    # TODO: add the logger to the ScorePredictions object
    score_predictions_task = ScorePredictions(
        interval_seconds=300.0,
        db_operations=db_operations,
        api_client=api_client,
        metagraph=bt_metagraph,
        config=config,
        subtensor=bt_subtensor,
        wallet=bt_wallet,
    )

    # Add tasks to scheduler
    scheduler = TasksScheduler(logger=logger)

    scheduler.add(task=pull_events_task)
    scheduler.add(task=resolve_events_task)
    scheduler.add(task=query_miners_task)
    scheduler.add(task=export_predictions_task)
    scheduler.add(task=score_predictions_task)

    logger.info(
        "Validator started",
        extra={
            "validator_uid": validator_uid,
            "validator_hotkey": validator_hotkey,
            "bt_network": bt_network,
            "bt_netuid": bt_netuid,
            "ifgames_env": ifgames_env,
            "db_path": db_path,
            "python": sys.version,
            "sqlite": sqlite3.sqlite_version,
        },
    )

    # Start scheduler
    await scheduler.start()
