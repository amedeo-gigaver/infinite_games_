import argparse
from typing import Literal

from bittensor.core.config import Config
from bittensor.core.subtensor import Subtensor
from bittensor.utils.btlogging import LoggingMachine
from bittensor_wallet.wallet import Wallet

IfgamesEnvType = Literal["test", "prod"]

VALID_NETWORK_CONFIGS = [
    {"subtensor.network": "finney", "netuid": 6, "ifgames.env": None},
    {"subtensor.network": "test", "netuid": 155, "ifgames.env": None},
    {"subtensor.network": "local", "netuid": 6, "ifgames.env": "prod"},
    {"subtensor.network": "local", "netuid": 155, "ifgames.env": "test"},
]


def get_config():
    # Build parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--netuid", type=int, help="Subnet netuid", choices=[6, 155], required=False
    )
    parser.add_argument(
        "--ifgames.env", type=str, help="IFGames env", choices=["prod", "test"], required=False
    )
    Subtensor.add_args(parser=parser)
    LoggingMachine.add_args(parser=parser)
    Wallet.add_args(parser=parser)

    # Validate args
    args = parser.parse_args()
    netuid = args.__getattribute__("netuid")
    network = args.__getattribute__("subtensor.network")
    ifgames_env = args.__getattribute__("ifgames.env")

    # Validate network config
    if not any(
        [
            netuid == config["netuid"]
            and network == config["subtensor.network"]
            and ifgames_env == config["ifgames.env"]
            for config in VALID_NETWORK_CONFIGS
        ]
    ):
        raise ValueError(
            (
                f"Invalid netuid {netuid}, subtensor.network '{network}' and ifgames.env '{ifgames_env}' combination.\n"
                f"Valid combinations are:\n"
                f"{chr(10).join(map(str, VALID_NETWORK_CONFIGS))}"
            )
        )

    config = Config(parser=parser, strict=True)
    env = "prod" if netuid == 6 else "test"
    db_path = "validator.db" if env == "prod" else "validator_test.db"

    return config, env, db_path
