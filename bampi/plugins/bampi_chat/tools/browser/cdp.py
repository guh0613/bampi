from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import inspect
import itertools
import json
from typing import Any

import aiohttp

from .errors import BrowserError, CdpError


EventCallback = Callable[[str, dict[str, Any], str | None], Awaitable[None] | None]


class CdpClient:
    """Small multiplexed Chrome DevTools Protocol client.

    A single browser websocket carries browser commands and flattened target
    sessions. Responses are matched by command id while events are fanned out
    to lightweight listeners.
    """

    def __init__(self, http: aiohttp.ClientSession, websocket: aiohttp.ClientWebSocketResponse) -> None:
        self._http = http
        self._websocket = websocket
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._listeners: dict[int, EventCallback] = {}
        self._listener_ids = itertools.count(1)
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._reader = asyncio.create_task(self._read_loop(), name="bampi-browser-cdp-reader")

    @classmethod
    async def connect(cls, websocket_url: str, *, timeout: float) -> "CdpClient":
        http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        try:
            websocket = await http.ws_connect(
                websocket_url,
                heartbeat=30.0,
                max_msg_size=0,
                timeout=aiohttp.ClientWSTimeout(ws_close=5.0),
            )
        except Exception:
            await http.close()
            raise
        return cls(http, websocket)

    @property
    def closed(self) -> bool:
        return self._closed or self._websocket.closed

    def add_listener(self, callback: EventCallback) -> Callable[[], None]:
        listener_id = next(self._listener_ids)
        self._listeners[listener_id] = callback

        def remove() -> None:
            self._listeners.pop(listener_id, None)

        return remove

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        if self.closed:
            raise BrowserError("Chromium DevTools connection is closed.")
        command_id = next(self._ids)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[command_id] = future
        payload: dict[str, Any] = {"id": command_id, "method": method}
        if params:
            payload["params"] = params
        if session_id:
            payload["sessionId"] = session_id
        try:
            async with self._send_lock:
                await self._websocket.send_str(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            message = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            self._pending.pop(command_id, None)
            raise BrowserError(f"CDP {method} timed out after {timeout:g}s.") from exc
        except BaseException:
            self._pending.pop(command_id, None)
            raise
        error = message.get("error")
        if isinstance(error, dict):
            raise CdpError(method, str(error.get("message") or "unknown error"), error.get("code"))
        result = message.get("result")
        return result if isinstance(result, dict) else {}

    async def wait_for_event(
        self,
        method: str,
        *,
        session_id: str | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        def listener(event_method: str, params: dict[str, Any], event_session: str | None) -> None:
            if event_method != method or (session_id is not None and event_session != session_id):
                return
            if predicate is not None and not predicate(params):
                return
            if not future.done():
                future.set_result(params)

        remove = self.add_listener(listener)
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            remove()

    async def _read_loop(self) -> None:
        failure: BaseException | None = None
        try:
            async for raw in self._websocket:
                if raw.type not in {aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY}:
                    if raw.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                        break
                    continue
                try:
                    data = json.loads(raw.data)
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                    continue
                if not isinstance(data, dict):
                    continue
                command_id = data.get("id")
                if isinstance(command_id, int):
                    future = self._pending.pop(command_id, None)
                    if future is not None and not future.done():
                        future.set_result(data)
                    continue
                method = data.get("method")
                if not isinstance(method, str):
                    continue
                params = data.get("params") if isinstance(data.get("params"), dict) else {}
                session_id = data.get("sessionId") if isinstance(data.get("sessionId"), str) else None
                for callback in tuple(self._listeners.values()):
                    try:
                        result = callback(method, params, session_id)
                        if inspect.isawaitable(result):
                            asyncio.create_task(result)
                    except Exception:
                        continue
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            failure = exc
        finally:
            self._closed = True
            exception = BrowserError(
                f"Chromium DevTools connection closed{f': {failure}' if failure else '.'}"
            )
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(exception)
            self._pending.clear()

    async def close(self) -> None:
        if self._closed and self._http.closed:
            return
        self._closed = True
        if not self._websocket.closed:
            with suppress(Exception):
                await self._websocket.close()
        if not self._reader.done():
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
        await self._http.close()
