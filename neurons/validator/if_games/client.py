import asyncio
import base64
import json
import time

import aiohttp
import aiohttp.typedefs
from bittensor_wallet import Wallet

from neurons.validator.utils.config import IfgamesEnvType
from neurons.validator.utils.git import commit_short_hash
from neurons.validator.utils.logger.logger import InfiniteGamesLogger
from neurons.validator.version import __version__


class IfGamesClient:
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: aiohttp.typedefs.LooseHeaders
    __logger: InfiniteGamesLogger
    __bt_wallet: Wallet

    def __init__(self, env: IfgamesEnvType, logger: InfiniteGamesLogger, bt_wallet: Wallet) -> None:
        # Validate env
        if not isinstance(env, str):
            raise TypeError("env must be an instance of str.")

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        # Validate bt_wallet
        if not isinstance(bt_wallet, Wallet):
            raise TypeError("bt_wallet must be an instance of Wallet.")

        self.__logger = logger
        self.__base_url = "https://ifgames.win" if env == "prod" else "https://stage.ifgames.win"
        self.__timeout = aiohttp.ClientTimeout(total=90)  # In seconds
        self.__headers = {
            "Validator-Version": __version__,
            "Validator-Hash": commit_short_hash,
        }
        self.__bt_wallet = bt_wallet

    def create_session(self, other_headers: dict = None) -> aiohttp.ClientSession:
        headers = self.__headers.copy()
        if other_headers:
            headers.update(other_headers)

        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_end.append(self.on_request_end)
        trace_config.on_request_exception.append(self.on_request_exception)

        return aiohttp.ClientSession(
            base_url=self.__base_url,
            timeout=self.__timeout,
            headers=headers,
            trace_configs=[trace_config],
        )

    async def on_request_start(self, _, trace_config_ctx, __):
        trace_config_ctx.start_time = time.time()

    async def on_request_end(self, _, trace_config_ctx, params: aiohttp.TraceRequestEndParams):
        elapsed_time_ms = round((time.time() - trace_config_ctx.start_time) * 1000)

        response_status = params.response.status

        extra = {
            "response_status": response_status,
            "method": params.method,
            "url": str(params.url),
            "elapsed_time_ms": elapsed_time_ms,
        }

        if response_status >= 400:
            # Add message if error
            response_message = await params.response.text()
            extra["response_message"] = response_message

            self.__logger.error("Http request failed", extra=extra)
        else:
            self.__logger.debug("Http request finished", extra=extra)

    async def on_request_exception(
        self, _, trace_config_ctx, params: aiohttp.TraceRequestExceptionParams
    ):
        exception = params.exception

        # Ignore cancelled exceptions
        if isinstance(exception, asyncio.exceptions.CancelledError):
            return

        elapsed_time_ms = round((time.time() - trace_config_ctx.start_time) * 1000)

        extra = {
            "method": params.method,
            "url": str(params.url),
            "elapsed_time_ms": elapsed_time_ms,
        }

        self.__logger.exception("Http request exception", extra=extra)

    def make_auth_headers(self, body: any) -> dict[str, str]:
        hot_key = self.__bt_wallet.get_hotkey()
        signed = base64.b64encode(hot_key.sign(json.dumps(body))).decode("utf-8")

        return {
            "Authorization": f"Bearer {signed}",
            "Validator": hot_key.ss58_address,
        }

    async def get_events(self, from_date: int, offset: int, limit: int):
        # Check that all parameters are provided
        if from_date is None or offset is None or limit is None:
            raise ValueError("Invalid parameters")

        async with self.create_session() as session:
            path = f"/api/v2/events?from_date={from_date}&offset={offset}&limit={limit}"

            async with session.get(path) as response:
                response.raise_for_status()

                return await response.json()

    async def get_event(self, event_id: str):
        if not isinstance(event_id, str):
            raise ValueError("Invalid parameter")

        async with self.create_session() as session:
            path = f"/api/v2/events/{event_id}"

            async with session.get(path) as response:
                response.raise_for_status()

                return await response.json()

    async def post_predictions(self, predictions: dict[any]):
        if not isinstance(predictions, dict):
            raise ValueError("Invalid parameter")

        assert len(predictions) > 0

        auth_headers = self.make_auth_headers(body=predictions)

        async with self.create_session(other_headers=auth_headers) as session:
            path = "/api/v1/validators/data"

            async with session.post(path, json=predictions) as response:
                response.raise_for_status()

                return await response.json()

    async def post_scores(self, scores: dict):
        if not isinstance(scores, dict):
            raise ValueError("Invalid parameter")

        assert len(scores) > 0

        auth_headers = self.make_auth_headers(body=scores)

        async with self.create_session(other_headers=auth_headers) as session:
            path = "/api/v1/validators/results"

            async with session.post(path, json=scores) as response:
                response.raise_for_status()

                return await response.json()
