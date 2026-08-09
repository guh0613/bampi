"""Headless-Chromium HTML-to-PNG rendering.

A single browser process is shared across renders. It is launched lazily on
first use and shut down after an idle period, so nothing keeps Chromium
resident between bursts of work.

Two capture modes are offered:

``render``
    One PNG of the whole document, sized to its content. Suited to pages that
    *are* the image.

``capture_elements``
    One PNG per CSS selector, from a single page load. Suited to pages that
    carry several independent fragments — rendering them together amortises the
    navigation and asset parsing across all of them.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
from pathlib import Path
from typing import Sequence
import json
import uuid

from nonebot import logger

from .errors import BrowserError
from .launcher import (
    LaunchedChromium,
    base_launch_args,
    launch_chromium_process,
    resolve_chromium,
)

_SETTLE_JS = """
(async () => {
    if (%(wait_ready)s) {
        const deadline = Date.now() + %(ready_timeout_ms)d;
        while (document.body.getAttribute('data-ready') !== 'true') {
            if (Date.now() > deadline) {
                throw new Error(
                    'page never signalled data-ready: '
                    + (document.body.getAttribute('data-error') || 'no reason given')
                );
            }
            await new Promise((resolve) => setTimeout(resolve, 25));
        }
    }
    if (document.fonts && document.fonts.ready) {
        await document.fonts.ready;
    }
    await new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve))
    );
    return true;
})()
"""

_DOCUMENT_BOX_JS = """
(() => {
    const doc = document.documentElement;
    return {
        x: 0,
        y: 0,
        width: Math.ceil(Math.max(doc.scrollWidth, document.body.scrollWidth)),
        height: Math.ceil(Math.max(doc.scrollHeight, document.body.scrollHeight)),
    };
})()
"""

_ELEMENT_BOXES_JS = """
((selectors) => selectors.map((selector) => {
    const node = document.querySelector(selector);
    if (!node) return null;
    const rect = node.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
        x: rect.x + window.scrollX,
        y: rect.y + window.scrollY,
        width: Math.ceil(rect.width),
        height: Math.ceil(rect.height),
    };
}))(%(selectors)s)
"""

_RENDER_EXTRA_ARGS = (
    "--hide-scrollbars",
    "--force-color-profile=srgb",
    "--allow-file-access-from-files",
    "--disable-lazy-loading",
)


class HtmlImageRenderer:
    """Render self-contained HTML documents to PNG via headless Chromium."""

    def __init__(
        self,
        *,
        work_dir: Path,
        executable_path: str = "",
        scale: int = 2,
        idle_ttl_seconds: int = 180,
        launch_timeout: float = 60.0,
        render_timeout: float = 30.0,
        log_label: str = "html render",
    ) -> None:
        self._work_dir = work_dir
        self._profile_dir = work_dir / "chromium-profile"
        self._pages_dir = work_dir / "pages"
        self._executable_path = executable_path.strip()
        self._scale = max(1, scale)
        self._idle_ttl_seconds = max(0, idle_ttl_seconds)
        self._launch_timeout = launch_timeout
        self._render_timeout = render_timeout
        self._log_label = log_label
        self._lock = asyncio.Lock()
        self._launched: LaunchedChromium | None = None
        self._session_id: str | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def scale(self) -> int:
        return self._scale

    async def render(self, html: str, *, viewport_width: int) -> bytes:
        """Render *html* and return PNG bytes sized to the full content."""
        results = await self._run(
            html,
            viewport_width=viewport_width,
            selectors=None,
            wait_for_ready=False,
        )
        return results[0]

    async def capture_elements(
        self,
        html: str,
        *,
        viewport_width: int,
        selectors: Sequence[str],
        wait_for_ready: bool = False,
    ) -> list[bytes]:
        """Render *html* once and return one PNG per entry in *selectors*.

        ``wait_for_ready`` makes the renderer block until the page sets
        ``document.body[data-ready="true"]``, for documents whose layout is
        produced by scripts rather than by markup.
        """
        if not selectors:
            return []
        return await self._run(
            html,
            viewport_width=viewport_width,
            selectors=list(selectors),
            wait_for_ready=wait_for_ready,
        )

    async def shutdown(self) -> None:
        self._closed = True
        async with self._lock:
            self._cancel_idle_locked()
            await self._close_locked()

    # ------------------------------------------------------------------ #
    # Internals (all called with self._lock held unless noted)            #
    # ------------------------------------------------------------------ #

    async def _run(
        self,
        html: str,
        *,
        viewport_width: int,
        selectors: list[str] | None,
        wait_for_ready: bool,
    ) -> list[bytes]:
        if self._closed:
            raise BrowserError("HtmlImageRenderer has been shut down.")
        async with self._lock:
            self._cancel_idle_locked()
            try:
                try:
                    await self._ensure_started_locked()
                    return await self._render_locked(
                        html, viewport_width, selectors, wait_for_ready
                    )
                except BrowserError:
                    # The browser process may have died between renders;
                    # relaunch once before giving up.
                    await self._close_locked()
                    await self._ensure_started_locked()
                    return await self._render_locked(
                        html, viewport_width, selectors, wait_for_ready
                    )
            finally:
                self._schedule_idle_locked()

    @property
    def _running(self) -> bool:
        return (
            self._launched is not None
            and not self._launched.client.closed
            and self._launched.process.returncode is None
            and self._session_id is not None
        )

    async def _ensure_started_locked(self) -> None:
        if self._running:
            return
        await self._close_locked()
        executable = await resolve_chromium(
            executable_path=self._executable_path or None,
            install_timeout=self._launch_timeout * 5,
        )
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        args = base_launch_args(
            executable,
            self._profile_dir,
            headless=True,
            window_width=1280,
            window_height=960,
            extra_args=_RENDER_EXTRA_ARGS,
        )
        launched = await launch_chromium_process(
            args,
            executable=executable,
            profile_dir=self._profile_dir,
            launch_timeout=self._launch_timeout,
        )
        try:
            targets = (await launched.client.call("Target.getTargets")).get(
                "targetInfos", []
            )
            target_id = next(
                (
                    str(info.get("targetId"))
                    for info in targets
                    if isinstance(info, dict) and info.get("type") == "page"
                ),
                None,
            )
            if target_id is None:
                created = await launched.client.call(
                    "Target.createTarget", {"url": "about:blank"}
                )
                target_id = str(created.get("targetId") or "")
            attached = await launched.client.call(
                "Target.attachToTarget", {"targetId": target_id, "flatten": True}
            )
            session_id = str(attached.get("sessionId") or "")
            if not session_id:
                raise BrowserError("Could not attach to the Chromium render page.")
            await launched.client.call("Page.enable", session_id=session_id)
            await launched.client.call("Runtime.enable", session_id=session_id)
        except BaseException:
            with suppress(Exception):
                await launched.close()
            raise
        self._launched = launched
        self._session_id = session_id
        logger.info(f"{self._log_label} browser started ({executable})")

    async def _render_locked(
        self,
        html: str,
        viewport_width: int,
        selectors: list[str] | None,
        wait_for_ready: bool,
    ) -> list[bytes]:
        assert self._launched is not None and self._session_id is not None
        client = self._launched.client
        session_id = self._session_id

        await client.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": viewport_width,
                "height": 720,
                "deviceScaleFactor": self._scale,
                "mobile": False,
            },
            session_id=session_id,
        )

        # Pages live under the work dir so relative and file:// asset references
        # inside the document resolve against a stable location.
        page_path = self._pages_dir / f"render-{uuid.uuid4().hex}.html"
        page_path.write_text(html, encoding="utf-8")
        try:
            load_waiter = asyncio.create_task(
                client.wait_for_event(
                    "Page.loadEventFired",
                    session_id=session_id,
                    timeout=self._render_timeout,
                )
            )
            try:
                await client.call(
                    "Page.navigate",
                    {"url": page_path.resolve().as_uri()},
                    session_id=session_id,
                    timeout=self._render_timeout,
                )
                await load_waiter
            except BaseException:
                load_waiter.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await load_waiter
                raise

            settle_timeout = self._render_timeout if not wait_for_ready else max(
                self._render_timeout, 45.0
            )
            await self._evaluate(
                _SETTLE_JS
                % {
                    "wait_ready": "true" if wait_for_ready else "false",
                    "ready_timeout_ms": int(settle_timeout * 1000),
                },
                session_id=session_id,
                timeout=settle_timeout + 5.0,
            )

            if selectors is None:
                box = await self._evaluate(
                    _DOCUMENT_BOX_JS, session_id=session_id, await_promise=False
                )
                if not box or not box.get("height"):
                    raise BrowserError("Rendered document reported no content height.")
                box["width"] = viewport_width
                boxes = [box]
            else:
                boxes = await self._evaluate(
                    _ELEMENT_BOXES_JS % {"selectors": json.dumps(selectors)},
                    session_id=session_id,
                    await_promise=False,
                )
                if not isinstance(boxes, list) or len(boxes) != len(selectors):
                    raise BrowserError("Element measurement returned an unusable result.")
                missing = [
                    selector
                    for selector, box in zip(selectors, boxes)
                    if not isinstance(box, dict)
                ]
                if missing:
                    raise BrowserError(f"Selectors matched nothing: {missing}")

            shots: list[bytes] = []
            for box in boxes:
                shots.append(
                    await self._capture(box, session_id=session_id)
                )
            return shots
        finally:
            with suppress(OSError):
                page_path.unlink()

    async def _capture(self, box: dict, *, session_id: str) -> bytes:
        screenshot = await self._launched.client.call(  # type: ignore[union-attr]
            "Page.captureScreenshot",
            {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": True,
                "clip": {
                    "x": box["x"],
                    "y": box["y"],
                    "width": box["width"],
                    "height": box["height"],
                    "scale": 1,
                },
            },
            session_id=session_id,
            timeout=max(self._render_timeout, 30.0),
        )
        data = screenshot.get("data")
        if not isinstance(data, str) or not data:
            raise BrowserError("Chromium returned an empty screenshot.")
        return base64.b64decode(data)

    async def _evaluate(
        self,
        expression: str,
        *,
        session_id: str,
        await_promise: bool = True,
        timeout: float | None = None,
    ):
        assert self._launched is not None
        result = await self._launched.client.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True,
            },
            session_id=session_id,
            timeout=timeout if timeout is not None else self._render_timeout,
        )
        exception = result.get("exceptionDetails")
        if exception:
            text = (exception.get("exception") or {}).get("description") or exception.get(
                "text"
            )
            raise BrowserError(f"Page script failed: {text}")
        return (result.get("result") or {}).get("value")

    async def _close_locked(self) -> None:
        launched, self._launched = self._launched, None
        self._session_id = None
        if launched is not None:
            with suppress(Exception):
                await launched.close()

    def _cancel_idle_locked(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None

    def _schedule_idle_locked(self) -> None:
        if self._closed or self._idle_ttl_seconds <= 0 or self._launched is None:
            return
        self._idle_task = asyncio.create_task(self._expire_after_idle())

    async def _expire_after_idle(self) -> None:
        try:
            await asyncio.sleep(self._idle_ttl_seconds)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if self._idle_task is not asyncio.current_task():
                return
            self._idle_task = None
            await self._close_locked()
            logger.debug(f"{self._log_label} browser closed after idle timeout")
