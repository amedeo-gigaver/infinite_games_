# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import os
import random
import sqlite3
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import bittensor as bt
import pytest
from freezegun import freeze_time

from infinite_games.events.base import (
    CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES,
    EventStatus,
    ProviderEvent,
)
from neurons.deprecated_validator import Validator
from tests.providers import (
    MockAzuroProviderIntegration,
    MockIFGamesProviderIntegration,
    MockPolymarketProviderIntegration,
)
from tests.utils import after, fake_synapse_response


@pytest.mark.asyncio
class TestTemplateValidatorNeuronTestCase:
    async def next_run(self, v: Validator):
        """Imitate Validator.run with 1 step"""

        bt.logging.info(
            f"Running validator {v.axon} on network: {v.config.subtensor.chain_endpoint} with netuid: {v.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {v.block}")

        bt.logging.info(f"step({v.step}) block({v.block})")

        await v.forward()
        # Sync metagraph and potentially set weights.
        v.sync()

        v.step += 1

    # def test_validator_clean_run(self, mock_network, caplog):
    #     wallet, subtensor = mock_network
    #     v = Validator(integrations=[
    #         AzuroProviderIntegration(),
    #         PolymarketProviderIntegration()
    #     ], db_path='test.db')

    #     # await v.forward()

    #     v.run_in_background_thread()

    #     sleep(30)

    #     # check from simple outputs for now.
    #     assert 'Provider initialized..' in caplog.text, 'Event Provider has to be initialized!'
    #     assert 'Event for submission: 0' in caplog.text, 'There should be some events!'
    #     # assert 'EventAggregator Start watcher...' in caplog.text, 'Event watcher should be started'
    #     assert 'Processed miner responses.' not in caplog.text, 'There should not be any miner submissions!'
    #     v.stop_run_thread()
    #     assert v.event_provider
    #     assert v.event_provider.integrations
    #     assert len(v.event_provider.registered_events) > 0, "There should be at least one registered event"

    async def test_validator_acled_events(
        self, mock_miner_reg_time, mock_network, caplog, monkeypatch
    ):
        wallet, subtensor = mock_network
        provider = MockIFGamesProviderIntegration()
        v = Validator(integrations=[provider], db_path="test.db")

        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()

            test_event = await provider.get_single_event("dbcba93a-fe3b-4092-b918-8231b23f2faa")
            assert v.event_provider.register_or_update_event(test_event) is True
            assert v.event_provider.integrations

            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events["ifgames-dbcba93a-fe3b-4092-b918-8231b23f2faa"][
                "probability"
            ] = 1
            mock_response[4].events["ifgames-dbcba93a-fe3b-4092-b918-8231b23f2faa"][
                "probability"
            ] = 1
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)
        for window in range(1, 42):
            window_time = initial_date + timedelta(
                minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * window
            )
            with freeze_time(window_time, tick=True):
                await self.next_run(v)

        # based on providers.py hardcode values
        settle_date = initial_date + timedelta(days=7)
        with freeze_time(settle_date, tick=True):
            test_event = await provider.get_single_event("dbcba93a-fe3b-4092-b918-8231b23f2faa")

            assert test_event.market_type == "ifgames"
            test_event.status = EventStatus.SETTLED
            test_event.answer = 1
            # not a new event
            is_new = v.event_provider.register_or_update_event(test_event)
            assert is_new is False

            await self.next_run(v)

        actual_event = v.event_provider.get_event(f"{test_event.market_type}-{test_event.event_id}")
        assert actual_event.event_id == test_event.event_id
        assert actual_event.market_type == test_event.market_type
        assert actual_event.status == EventStatus.SETTLED
        assert actual_event.answer == 1

        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.4, 0.4)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (
            0.5,
            0.5,
        )

        await provider.close()

    async def test_validator_settled_event_scores_polymarket_aggregation_interval(
        self, mock_miner_reg_time, mock_network, caplog, monkeypatch, disable_event_updates
    ):
        wallet, subtensor = mock_network
        v = Validator(
            integrations=[
                MockAzuroProviderIntegration(max_pending_events=6),
                MockPolymarketProviderIntegration(),
                MockIFGamesProviderIntegration(),
            ],
            db_path="test.db",
        )

        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()

            test_event = ProviderEvent(
                "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                datetime.now(timezone.utc),
                "polymarket",
                "Test event 1",
                None,
                after(hours=12),
                None,
                datetime.now(timezone.utc),
                EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
                },
            )
            assert v.event_provider.register_or_update_event(test_event) is True
            # assert v.event_provider.registered_events.get(f'{test_event.market_type}-{test_event.event_id}')
            # assert len(v.event_provider.registered_events) == 1
            assert v.event_provider.integrations

            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

            mock_response[3].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.7
            mock_response[4].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.5

            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.5
            mock_response[4].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.9

            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(second_window, tick=True):
            await self.next_run(v)
        third_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(third_window, tick=True):
            await self.next_run(v)

        fourth_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 3)
        with freeze_time(fourth_window, tick=True):
            await self.next_run(v)

        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.register_or_update_event(test_event)
        assert round(v.scores[2].item(), 1) == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective  score -> moving average
        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.011, 0.789)
        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (
            0.014,
            0.986,
        )

    async def test_validator_polymarket_pricing_events(
        self, mock_miner_reg_time, mock_network, caplog, monkeypatch
    ):
        wallet, subtensor = mock_network
        iggames_provider = MockIFGamesProviderIntegration()
        v = Validator(integrations=[iggames_provider], db_path="test.db")

        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()

            test_event = await iggames_provider.get_single_event(
                "cbcba93a-fe3b-4092-b918-8231b23f2faa"
            )
            assert v.event_provider.register_or_update_event(test_event) is True
            assert v.event_provider.integrations

            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events["ifgames-cbcba93a-fe3b-4092-b918-8231b23f2faa"][
                "probability"
            ] = 1
            mock_response[4].events["ifgames-cbcba93a-fe3b-4092-b918-8231b23f2faa"][
                "probability"
            ] = 1
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)
        for window in range(1, 42):
            window_time = initial_date + timedelta(
                minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * window
            )
            with freeze_time(window_time, tick=True):
                await self.next_run(v)

        # based on providers.py hardcode values
        settle_date = initial_date + timedelta(days=7)
        with freeze_time(settle_date, tick=True):
            test_event = await iggames_provider.get_single_event(
                "cbcba93a-fe3b-4092-b918-8231b23f2faa"
            )

            test_event.status = EventStatus.SETTLED
            test_event.answer = 1
            v.event_provider.register_or_update_event(test_event)

            await self.next_run(v)

        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.4, 0.4)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (
            0.5,
            0.5,
        )

    async def test_validator_settled_event_scores_polymarket_short(
        self, mock_miner_reg_time, mock_network, caplog, monkeypatch, disable_event_updates
    ):
        wallet, subtensor = mock_network
        v = Validator(
            integrations=[
                MockAzuroProviderIntegration(max_pending_events=6),
                MockPolymarketProviderIntegration(),
            ],
            db_path="test.db",
        )

        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()
            test_event = ProviderEvent(
                "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                datetime.now(timezone.utc),
                "polymarket",
                "Test event 1",
                None,
                after(hours=12),
                None,
                datetime.now(timezone.utc),
                EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
                },
            )
            assert v.event_provider.register_or_update_event(test_event) is True
            v.event_provider.get_events(
                statuses=[EventStatus.PENDING, EventStatus.SETTLED], processed=False
            )
            # assert v.event_provider.registered_events.get(f'{test_event.market_type}-{test_event.event_id}')
            # assert len(v.event_provider.registered_events) == 1
            assert v.event_provider.integrations
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.7
            mock_response[4].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.9
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(second_window, tick=True):
            await self.next_run(v)
        third_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(third_window, tick=True):
            await self.next_run(v)

        fourth_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 3)
        with freeze_time(fourth_window, tick=True):
            await self.next_run(v)

        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.register_or_update_event(test_event)
        assert round(v.scores[2].item(), 1) == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective  score -> moving average
        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.0665, 0.7335)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (
            0.083,
            0.917,
        )

    async def test_validator_settled_event_scores_polymarket_longer_settle_date(
        self, mock_miner_reg_time, mock_network, caplog, monkeypatch, disable_event_updates
    ):
        wallet, subtensor = mock_network
        v = Validator(
            integrations=[
                MockAzuroProviderIntegration(max_pending_events=6),
                MockPolymarketProviderIntegration(),
            ],
            db_path="test.db",
        )

        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()

            test_event = ProviderEvent(
                "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                datetime.now(timezone.utc),
                "polymarket",
                "Test event 1",
                None,
                after(days=12),
                None,
                datetime.now(timezone.utc),
                EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
                },
            )
            assert v.event_provider.register_or_update_event(test_event) is True
            # assert v.event_provider.registered_events.get(f'{test_event.market_type}-{test_event.event_id}')
            # assert len(v.event_provider.registered_events) == 1
            assert v.event_provider.integrations
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.7
            mock_response[4].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.9
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(second_window, tick=True):
            await self.next_run(v)
        third_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(third_window, tick=True):
            await self.next_run(v)

        fourth_window = initial_date + timedelta(
            minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 10
        )
        with freeze_time(fourth_window, tick=True):
            await self.next_run(v)

        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.register_or_update_event(test_event)
        assert round(v.scores[2].item(), 1) == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective  score -> moving average
        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.3386, 0.4614)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (
            0.423,
            0.577,
        )

    async def test_validator_settled_event_scores_polymarket_earlier_settle_date(
        self, mock_miner_reg_time, mock_network, caplog, monkeypatch, disable_event_updates
    ):
        wallet, subtensor = mock_network
        v = Validator(
            integrations=[
                MockAzuroProviderIntegration(max_pending_events=6),
                MockPolymarketProviderIntegration(),
            ],
            db_path="test.db",
        )

        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()
            test_event = ProviderEvent(
                "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                datetime.now(timezone.utc),
                "polymarket",
                "Test event 1",
                None,
                after(days=12),
                None,
                datetime.now(timezone.utc),
                EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
                },
            )
            assert v.event_provider.register_or_update_event(test_event) is True
            # assert v.event_provider.registered_events.get(f'{test_event.market_type}-{test_event.event_id}')
            # assert len(v.event_provider.registered_events) == 1
            assert v.event_provider.integrations
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.7
            mock_response[4].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.9
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(second_window, tick=True):
            await self.next_run(v)
        third_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(third_window, tick=True):
            await self.next_run(v)

        tenth_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 10)
        with freeze_time(tenth_window, tick=True):
            await self.next_run(v)

        eleventh_window = initial_date + timedelta(
            minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 11
        )
        with freeze_time(eleventh_window, tick=True):
            test_event.status = EventStatus.SETTLED
            test_event.answer = 1
            v.event_provider.register_or_update_event(test_event)
        assert round(v.scores[2].item(), 1) == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective  score -> moving average
        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.1697, 0.6303)

        assert round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3) == (
            0.0,
            0.0,
        )

    async def test_validator_settled_event_scores_azuro(
        self, mock_miner_reg_time, mock_network, caplog, monkeypatch, disable_event_updates
    ):
        wallet, subtensor = mock_network
        v = Validator(
            integrations=[
                MockAzuroProviderIntegration(max_pending_events=6),
                MockPolymarketProviderIntegration(),
                MockIFGamesProviderIntegration(),
            ],
            db_path="test.db",
        )

        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()
            test_event = ProviderEvent(
                "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                datetime.now(timezone.utc),
                "ifgames",
                "Test event 1",
                after(hours=12),
                None,
                None,
                datetime.now(timezone.utc),
                EventStatus.PENDING,
                {},
                {
                    "market_type": "azuro",
                    "cutoff": 1722462600,
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
                },
            )
            assert v.event_provider.register_or_update_event(test_event) is True
            # assert v.event_provider.registered_events.get(f'{test_event.market_type}-{test_event.event_id}')
            # assert len(v.event_provider.registered_events) == 1
            assert v.event_provider.integrations
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.7
            mock_response[4].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.9
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.register_or_update_event(test_event)
        assert round(v.scores[2].item(), 1) == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective  score -> moving average
        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.0665, 0.7335)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (
            0.083,
            0.917,
        )

    async def test_validator_settled_event_scores_new_regged_miner_azuro(
        self,
        mock_miner_reg_time,
        mock_network,
        caplog,
        monkeypatch,
        disable_event_updates,
    ):
        wallet, subtensor = mock_network
        v = Validator(
            integrations=[
                # MockAzuroProviderIntegration(max_pending_events=6),
                # MockPolymarketProviderIntegration(),
                MockIFGamesProviderIntegration()
            ],
            db_path="test.db",
        )

        reg_time = "2028-12-31 00:00"
        monkeypatch.setattr(
            "neurons.deprecated_validator.get_miner_data_by_uid",
            lambda validator, uid: {
                "registered_date": reg_time if uid == 1 else "2024-01-01 00:00"
            },
        )
        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()
            test_event = ProviderEvent(
                "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                datetime.now(timezone.utc),
                "ifgames",
                "Test event 1",
                after(hours=12),
                None,
                None,
                datetime.now(timezone.utc),
                EventStatus.PENDING,
                {},
                {
                    "market_type": "azuro"
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
                },
            )
            assert v.event_provider.register_or_update_event(test_event) is True
            # assert v.event_provider.registered_events.get(f'{test_event.market_type}-{test_event.event_id}')
            # assert len(v.event_provider.registered_events) == 1
            assert v.event_provider.integrations
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.7
            mock_response[4].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.9
            mock_response[5].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.0
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.register_or_update_event(test_event)
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective  score -> moving average
        assert (
            round(v.scores[1].item(), 4),
            round(v.scores[2].item(), 4),
            round(v.scores[3].item(), 4),
            round(v.scores[4].item(), 4),
        ) == (0.0005, 0.000, 0.0665, 0.733)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (
            0.083,
            0.916,
        )

    async def test_validator_settled_event_scores_distribution(
        self, mock_network, caplog, monkeypatch, disable_event_updates, mock_miner_reg_time
    ):
        wallet, subtensor = mock_network
        v = Validator(
            integrations=[
                MockAzuroProviderIntegration(max_pending_events=6),
                MockPolymarketProviderIntegration(),
                MockIFGamesProviderIntegration(),
            ],
            db_path="test.db",
        )
        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()
            test_event = ProviderEvent(
                "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                datetime.now(timezone.utc),
                "polymarket",
                "Test event 1",
                None,
                after(hours=12),
                None,
                datetime.now(timezone.utc),
                EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
                },
            )
            assert v.event_provider.register_or_update_event(test_event) is True
            assert v.event_provider.integrations
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[2].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.0
            mock_response[3].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.1
            mock_response[4].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.2
            mock_response[5].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.3
            mock_response[6].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.4
            mock_response[7].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.5
            mock_response[8].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.6
            mock_response[9].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.7
            mock_response[10].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.8
            mock_response[11].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.9
            mock_response[12].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 1.0
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[2].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.0
            mock_response[3].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.1
            mock_response[4].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.2
            mock_response[5].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.3
            mock_response[6].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.4
            mock_response[7].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.5
            mock_response[8].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.6
            mock_response[9].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.7
            mock_response[10].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.8
            mock_response[11].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 0.9
            mock_response[12].events[f"{test_event.market_type}-{test_event.event_id}"][
                "probability"
            ] = 1.0
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)
        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(second_window, tick=True):
            await self.next_run(v)
        third_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(third_window, tick=True):
            await self.next_run(v)
        fourth_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 3)
        with freeze_time(fourth_window, tick=True):
            await self.next_run(v)
        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.register_or_update_event(test_event)
        assert (
            round(v.scores[2].item(), 3),
            round(v.scores[3].item(), 3),
            round(v.scores[4].item(), 3),
            round(v.scores[5].item(), 3),
            round(v.scores[6].item(), 3),
            round(v.scores[6].item(), 3),
            round(v.scores[8].item(), 3),
            round(v.scores[9].item(), 3),
            round(v.scores[10].item(), 3),
            round(v.scores[11].item(), 3),
            round(v.scores[12].item(), 3),
        ) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003, 0.025, 0.114, 0.28, 0.378)

        # async def test_send_interval_data(self, mock_network, caplog, monkeypatch, disable_event_updates):
        #     wallet, subtensor = mock_network
        #     v = Validator(integrations=[
        #         MockAzuroProviderIntegration(max_pending_events=6),
        #         MockPolymarketProviderIntegration()
        #     ], db_path='test.db')
        #     miner_data = []
        #     for miner in range(0, 255):
        #         for event in range(200):

        #             miner_data.append(
        #                 [
        #                     miner,
        #                     f'polymarket-idevent' + f'{miner}',
        #                     'polymarket',
        #                     100000,
        #                     0.9,
        #                     23
        #                  ]
        #             )
        #     v.send_interval_data(miner_data)

    def get_all_records_table(self, table_name):
        with sqlite3.connect("test.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM {table_name}",
            )
            return cursor.fetchall()

    async def test_miner_predict_to_db(
        self, mock_miner_reg_time, mock_network, caplog, monkeypatch, disable_event_updates
    ):
        wallet, subtensor = mock_network
        v = Validator(
            integrations=[
                MockAzuroProviderIntegration(max_pending_events=6),
                MockPolymarketProviderIntegration(),
                MockIFGamesProviderIntegration(),
            ],
            db_path="test.db",
        )

        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date, tick=True):
            await v.initialize_provider()

            test_event = ProviderEvent(
                "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                datetime.now(timezone.utc),
                "ifgames",
                "Test event 1",
                None,
                after(hours=12),
                None,
                datetime.now(timezone.utc),
                EventStatus.PENDING,
                {},
                {},
            )

            test_event_2 = deepcopy(test_event)
            test_event_2.market_type = "polymarket"

            test_event_3 = deepcopy(test_event)
            test_event_3.market_type = "azuro"
            test_event_3.starts = after(hours=12)
            test_event_3.resolve_date = None
            test_event_3.metadata = {
                "market_type": "azuro",
            }

            all_events = [test_event, test_event_2, test_event_3]
            for event in all_events:
                assert v.event_provider.register_or_update_event(event) is True
            assert v.event_provider.integrations

            # no run, just register events
            actual_events = self.get_all_records_table("events")
            actual_miners = self.get_all_records_table("miners")
            actual_predictions = self.get_all_records_table("predictions")

            assert len(actual_events) == 3
            assert set([event[0] for event in actual_events]) == {
                f"{event.market_type}-{event.event_id}" for event in all_events
            }
            assert len(actual_miners) == 0
            assert len(actual_predictions) == 0

            # first run, register miners but None predictions
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

            actual_events = self.get_all_records_table("events")
            actual_miners = self.get_all_records_table("miners")
            actual_predictions = self.get_all_records_table("predictions")

            assert len(actual_events) == 3
            assert set([event[0] for event in actual_events]) == {
                f"{event.market_type}-{event.event_id}" for event in all_events
            }

            assert len(actual_miners) == 256
            assert set([miner[1] for miner in actual_miners]) == set(str(i) for i in range(256))

            # second run, set the predictions first time
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            for event in all_events:
                for miner_uid in range(1, 253):
                    mock_response[miner_uid].events[f"{event.market_type}-{event.event_id}"][
                        "probability"
                    ] = 0.5
                mock_response[0].events[f"{event.market_type}-{event.event_id}"][
                    "probability"
                ] = 0.7
                mock_response[253].events[f"{event.market_type}-{event.event_id}"][
                    "probability"
                ] = None
                mock_response[254].events[f"{event.market_type}-{event.event_id}"][
                    "probability"
                ] = None
                mock_response[255].events[f"{event.market_type}-{event.event_id}"][
                    "probability"
                ] = 0.3
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

            actual_events = self.get_all_records_table("events")
            actual_miners = self.get_all_records_table("miners")
            actual_predictions = self.get_all_records_table("predictions")

            assert len(actual_events) == 3
            assert set([event[0] for event in actual_events]) == {
                f"{event.market_type}-{event.event_id}" for event in all_events
            }

            assert len(actual_miners) == 256
            assert set([miner[1] for miner in actual_miners]) == set(str(i) for i in range(256))
            assert len(actual_predictions) == 762  # missing 2 miners None * 3 events
            prediction_counter = Counter([pred[7] for pred in actual_predictions])
            count_counter = Counter([pred[8] for pred in actual_predictions])
            assert prediction_counter[0.5] == 756
            assert prediction_counter[0.7] == 3
            assert prediction_counter[0.3] == 3
            assert count_counter == {1: 762}

            # third run, set or update the predictions
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            for event in all_events:
                for miner_uid in range(1, 253):
                    mock_response[miner_uid].events[f"{event.market_type}-{event.event_id}"][
                        "probability"
                    ] = 0.1
                mock_response[0].events[f"{event.market_type}-{event.event_id}"][
                    "probability"
                ] = 0.9
                mock_response[253].events[f"{event.market_type}-{event.event_id}"][
                    "probability"
                ] = 0.5
                mock_response[254].events[f"{event.market_type}-{event.event_id}"][
                    "probability"
                ] = None
                mock_response[255].events[f"{event.market_type}-{event.event_id}"][
                    "probability"
                ] = 0.4
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

            actual_events = self.get_all_records_table("events")
            actual_miners = self.get_all_records_table("miners")
            actual_predictions = self.get_all_records_table("predictions")

            assert len(actual_miners) == 256
            assert set([miner[1] for miner in actual_miners]) == set(str(i) for i in range(256))
            assert len(actual_predictions) == 765  # missing 1 miner None * 3 events
            prediction_counter = Counter([pred[7] for pred in actual_predictions])
            count_counter = Counter([pred[8] for pred in actual_predictions])
            assert prediction_counter[0.3] == 756  # updated with average
            assert prediction_counter[0.8] == 3
            assert prediction_counter[0.35] == 3
            assert prediction_counter[0.5] == 3
            assert count_counter[1] == 3  # the one which had None in the previous run
            assert count_counter[2] == 762

    @pytest.mark.asyncio
    async def test_submissions_export(
        self, mock_network, caplog, monkeypatch, disable_event_updates
    ):
        n_miners = 3
        initial_date = datetime(year=2024, month=2, day=3)
        with freeze_time(initial_date, tick=True):
            wallet, subtensor = mock_network
            v = Validator(
                integrations=[
                    MockAzuroProviderIntegration(max_pending_events=6),
                    MockPolymarketProviderIntegration(),
                    MockIFGamesProviderIntegration(),
                ],
                db_path="test.db",
            )

            await v.initialize_provider()

            test_event = ProviderEvent(
                "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                datetime.now(timezone.utc),
                "ifgames",
                "Test event 1",
                None,
                after(hours=12),
                None,
                datetime.now(timezone.utc),
                EventStatus.PENDING,
                {},
                {},
            )

            test_event_2 = deepcopy(test_event)
            test_event_2.market_type = "polymarket"

            all_events = [test_event, test_event_2]
            batch_size = n_miners * len(all_events)

            for event in all_events:
                assert v.event_provider.register_or_update_event(event) is True
            assert v.event_provider.integrations

            # set some predictions
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            for event in all_events:
                for miner_uid in range(n_miners):
                    mock_response[miner_uid].events[f"{event.market_type}-{event.event_id}"][
                        "probability"
                    ] = (5 * random.random())
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

        # move to the next window and set some predictions again
        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(second_window, tick=True):
            # set some predictions again
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            for event in all_events:
                for miner_uid in range(n_miners):
                    mock_response[miner_uid].events[f"{event.market_type}-{event.event_id}"][
                        "probability"
                    ] = random.random()
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

            # no predictions should be exported
            not_exported = v.event_provider.get_all_non_exported_event_predictions(
                interval_minutes_begin=47520,
                interval_minutes_end=48000,
            )
            assert len(not_exported) == 2 * batch_size

            # now we export the predictions for the first window
            monkeypatch.setattr(v, "send_interval_data", lambda *args, **kwargs: None)
            await v.send_interval_stats(datetime.now(timezone.utc))

            not_exported = v.event_provider.get_all_non_exported_event_predictions(
                interval_minutes_begin=47520,
                interval_minutes_end=48000,
            )
            assert len(not_exported) == batch_size

            # now we export the predictions for the second window
            await v.send_interval_stats(datetime.now(timezone.utc) + timedelta(minutes=240))

            not_exported = v.event_provider.get_all_non_exported_event_predictions(
                interval_minutes_begin=47520,
                interval_minutes_end=48000,
            )
            assert len(not_exported) == 0

        # we insert some older predictions
        previous_window = initial_date - timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(previous_window, tick=True):
            for event in all_events:
                for miner_uid in range(n_miners):
                    mock_response[miner_uid].events[f"{event.market_type}-{event.event_id}"][
                        "probability"
                    ] = random.random()
            monkeypatch.setattr(
                "neurons.deprecated_validator.query_miners", lambda a, b, c: mock_response
            )
            await self.next_run(v)

            # the older predictions are not exported
            not_exported = v.event_provider.get_all_non_exported_event_predictions(
                interval_minutes_begin=47000,
                interval_minutes_end=48000,
            )
            assert len(not_exported) == batch_size

            # now we export the older predictions
            await v.send_interval_stats(
                initial_date.astimezone(timezone.utc) + timedelta(minutes=240)
            )
            not_exported = v.event_provider.get_all_non_exported_event_predictions(
                interval_minutes_begin=47000,
                interval_minutes_end=48000,
            )
            assert len(not_exported) == 0
