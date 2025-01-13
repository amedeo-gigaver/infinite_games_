import asyncio
import os
import shutil

import bittensor as bt
from bittensor_wallet.mock.wallet_mock import get_mock_wallet
from pytest import fixture

from tests.bittensor_mocks import MockSubtensor

bt.debug(True)

os.environ["NEST_ASYNCIO"] = "0"

# @fixture
# def event_loop():
#     loop = asyncio.get_event_loop()
#     yield loop
#     loop.close()


@fixture(scope="session")
def netuid():
    try:
        netuid = int(os.environ.get("NETUID"))
        if netuid < 255:
            raise Exception("Please set netuid higher than 255")
    except ValueError:
        raise Exception("Environment variable NETUID should be set as integer > 255")
    return netuid


# this is mock that forces to update blocks, but waiting
# for this fix https://github.com/opentensor/bittensor/pull/2138
# class AutoBlockMockSubtensor(MockSubtensor):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def get_current_block(self) -> int:
#         self.do_block_step()
#         return super().get_current_block()


@fixture(scope="session")
def test_network(netuid):
    w = get_mock_wallet()
    s = MockSubtensor(_mock=True)
    s.create_subnet(netuid)
    uid = s.force_register_neuron(netuid, w.hotkey.ss58_address, w.coldkey.ss58_address, 20, 20)
    print(f"Main neuron {w.hotkey.ss58_address} registered")
    assert uid is not None, f"Failed to register {w} in {netuid}"
    remaining_wallets = [get_mock_wallet() for uid in range(1, 256)]
    for w in remaining_wallets:
        uid = s.force_register_neuron(netuid, w.hotkey.ss58_address, w.coldkey.ss58_address, 20, 20)
    print("All subnet neurons registered")

    yield w, s


@fixture
def mock_network(monkeypatch, test_network, netuid):
    w, s = test_network
    monkeypatch.setattr("infinite_games.base.neuron.get_wallet", lambda a: w)
    monkeypatch.setattr("infinite_games.base.neuron.get_subtensor", lambda a: s)
    neuron_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            "~/.bittensor/miners",  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            "default",
            "default",
            netuid,
            "validator",
        )
    )
    print("Reset neuron path: ", neuron_path)
    try:
        shutil.rmtree(neuron_path)
        os.remove("test.db")
    except FileNotFoundError:
        pass
    yield w, s


@fixture
def disable_event_updates():
    os.environ["VALIDATOR_WATCH_EVENTS_DISABLED"] = "1"
    yield
    os.environ["VALIDATOR_WATCH_EVENTS_DISABLED"] = "0"


@fixture
def mock_miner_reg_time(monkeypatch):
    reg_time = "2024-01-01 00:00"
    monkeypatch.setattr(
        "neurons.deprecated_validator.get_miner_data_by_uid",
        lambda validator, _: {"registered_date": reg_time},
    )

    yield reg_time


@fixture(autouse=True, scope="function")
def clean_db():
    # Setup: clean the sqlite DB
    try:
        os.remove("test.db")
    except FileNotFoundError:
        pass
    yield
    # Teardown: clean the sqlite DB again after the test
    try:
        os.remove("test.db")
    except FileNotFoundError:
        pass


@fixture(autouse=True, scope="function")
async def cleanup_event_loop():
    yield
    # Cleanup tasks after the test completes
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@fixture(autouse=True)
def mock_sleep(monkeypatch):
    # Mock asyncio.sleep
    async def no_sleep_async(_):
        pass

    # Mock time.sleep
    def no_sleep(_):
        pass

    monkeypatch.setattr("asyncio.sleep", no_sleep_async)
    monkeypatch.setattr("time.sleep", no_sleep)
