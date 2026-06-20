from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Any

from .bridge import LocalhostBridgeManager
from .config import BrowserConfig
from .errors import BrowserError, CommandError
from .launcher import LaunchedChromium, launch_chromium
from .models import PageState, RecordingState
from .policy import NavigationPolicy


class BrowserRuntime:
    def __init__(
        self,
        workspace_dir: Path,
        config: BrowserConfig,
        *,
        container_root: str | None,
        container_name: str | None,
        bridge_localhost: bool,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.config = config
        self.container_root = container_root
        self.bridge = LocalhostBridgeManager(container_name)
        self.policy = NavigationPolicy(
            workspace_dir,
            container_root=container_root,
            bridge=self.bridge,
            bridge_localhost=bridge_localhost,
            config=config,
        )
        self.launched: LaunchedChromium | None = None
        self.pages: dict[str, PageState] = {}
        self.target_to_page: dict[str, str] = {}
        self.session_to_page: dict[str, str] = {}
        self.frame_sessions: dict[str, str] = {}
        self.execution_contexts: dict[str, tuple[str, int]] = {}
        self.downloads: dict[str, str] = {}
        self.download_dir = self.workspace_dir / "outbox" / "browser" / "downloads"
        self._page_number = 1
        self._listener_remove = None
        self.recording: RecordingState | None = None
        self.lock = asyncio.Lock()
        self.last_used_at = time.monotonic()

    @property
    def client(self):
        if self.launched is None or self.launched.client.closed:
            raise BrowserError("Browser runtime is not running.")
        return self.launched.client

    @property
    def running(self) -> bool:
        return self.launched is not None and not self.launched.client.closed and self.launched.process.returncode is None

    async def start(self) -> None:
        if self.running:
            return
        await self.close()
        self.launched = await launch_chromium(self.workspace_dir, self.config)
        self._listener_remove = self.client.add_listener(self._on_event)
        await self.client.call("Target.setDiscoverTargets", {"discover": True})
        await self.client.call(
            "Target.setAutoAttach",
            {"autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True},
        )
        targets = (await self.client.call("Target.getTargets")).get("targetInfos", [])
        for info in targets:
            if isinstance(info, dict) and info.get("type") == "page":
                await self._attach_page(info)
        if not self.pages:
            await self.create_page()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        with suppress(Exception):
            await self.client.call(
                "Browser.setDownloadBehavior",
                {"behavior": "allowAndName", "downloadPath": str(self.download_dir), "eventsEnabled": True},
            )

    async def _attach_page(self, info: dict[str, Any]) -> PageState:
        target_id = str(info.get("targetId") or "")
        existing_id = self.target_to_page.get(target_id)
        if existing_id and existing_id in self.pages:
            return self.pages[existing_id]
        attached = await self.client.call("Target.attachToTarget", {"targetId": target_id, "flatten": True})
        session_id = str(attached.get("sessionId") or "")
        return await self._register_page(info, session_id)

    async def _register_page(self, info: dict[str, Any], session_id: str) -> PageState:
        target_id = str(info.get("targetId") or "")
        existing_id = self.target_to_page.get(target_id)
        if existing_id and existing_id in self.pages:
            page = self.pages[existing_id]
            if session_id:
                page.session_id = session_id
                self.session_to_page[session_id] = page.page_id
            return page
        page_id = f"p{self._page_number}"
        self._page_number += 1
        page = PageState(
            page_id=page_id,
            target_id=target_id,
            session_id=session_id,
            url=str(info.get("url") or "about:blank"),
            title=str(info.get("title") or ""),
        )
        page.session_generations[session_id] = 0
        self.pages[page_id] = page
        self.target_to_page[target_id] = page_id
        if session_id:
            self.session_to_page[session_id] = page_id
            await self._enable_session(session_id)
        return page

    async def _enable_session(self, session_id: str) -> None:
        for method in ("Page.enable", "Runtime.enable", "DOM.enable", "Accessibility.enable", "Network.enable", "Log.enable"):
            with suppress(Exception):
                await self.client.call(method, session_id=session_id)
        with suppress(Exception):
            await self.client.call(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
                session_id=session_id,
            )
        if self.config.block_images:
            with suppress(Exception):
                await self.client.call(
                    "Network.setBlockedURLs",
                    {"urls": ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg", "*.avif"]},
                    session_id=session_id,
                )

    async def _on_event(self, method: str, params: dict[str, Any], session_id: str | None) -> None:
        if method == "Target.attachedToTarget":
            info = params.get("targetInfo") if isinstance(params.get("targetInfo"), dict) else {}
            attached_session = str(params.get("sessionId") or "")
            target_type = info.get("type")
            if target_type == "page":
                await self._register_page(info, attached_session)
            elif target_type == "iframe" and attached_session:
                target_id = str(info.get("targetId") or "")
                if target_id:
                    self.frame_sessions[target_id] = attached_session
                parent_page_id = self.session_to_page.get(session_id or "")
                if parent_page_id:
                    self.session_to_page[attached_session] = parent_page_id
                    self.pages[parent_page_id].session_generations.setdefault(attached_session, 0)
                await self._enable_session(attached_session)
            return
        if method == "Target.targetInfoChanged":
            info = params.get("targetInfo") if isinstance(params.get("targetInfo"), dict) else {}
            page_id = self.target_to_page.get(str(info.get("targetId") or ""))
            if page_id and page_id in self.pages:
                self.pages[page_id].url = str(info.get("url") or self.pages[page_id].url)
                self.pages[page_id].title = str(info.get("title") or self.pages[page_id].title)
            return
        if method == "Target.targetDestroyed":
            target_id = str(params.get("targetId") or "")
            page_id = self.target_to_page.pop(target_id, None)
            if page_id:
                page = self.pages.pop(page_id, None)
                if page:
                    self.session_to_page.pop(page.session_id, None)
            self.frame_sessions.pop(target_id, None)
            return
        if method == "Runtime.executionContextCreated":
            context = params.get("context") if isinstance(params.get("context"), dict) else {}
            aux = context.get("auxData") if isinstance(context.get("auxData"), dict) else {}
            frame_id = aux.get("frameId")
            context_id = context.get("id")
            if isinstance(frame_id, str) and isinstance(context_id, int) and session_id:
                self.execution_contexts[frame_id] = (session_id, context_id)
            return
        if method == "Browser.downloadWillBegin":
            guid = str(params.get("guid") or "")
            suggested = Path(str(params.get("suggestedFilename") or guid)).name
            if guid:
                self.downloads[guid] = suggested
            return
        if method == "Browser.downloadProgress" and params.get("state") == "completed":
            guid = str(params.get("guid") or "")
            suggested = self.downloads.get(guid)
            source = self.download_dir / guid
            if suggested and source.exists():
                target = self.download_dir / suggested
                sequence = 1
                while target.exists():
                    target = self.download_dir / f"{Path(suggested).stem}-{sequence}{Path(suggested).suffix}"
                    sequence += 1
                with suppress(OSError):
                    source.rename(target)
            return
        page_id = self.session_to_page.get(session_id or "")
        if not page_id:
            return
        page = self.pages.get(page_id)
        if page is None:
            return
        if method == "Page.frameNavigated":
            frame = params.get("frame") if isinstance(params.get("frame"), dict) else {}
            if session_id:
                page.session_generations[session_id] = page.session_generations.get(session_id, 0) + 1
                page.refs = {key: ref for key, ref in page.refs.items() if ref.session_id != session_id}
            if session_id == page.session_id and not frame.get("parentId"):
                page.main_frame_id = str(frame.get("id") or "") or None
                page.url = str(frame.get("url") or page.url)
                page.document_generation += 1
                page.refs.clear()
        elif method == "Page.javascriptDialogOpening":
            page.dialog = {
                "type": params.get("type"),
                "message": params.get("message"),
                "defaultPrompt": params.get("defaultPrompt"),
            }
        elif method == "Page.javascriptDialogClosed":
            page.dialog = None
        elif method == "Runtime.consoleAPICalled":
            values = []
            for arg in params.get("args", []):
                if isinstance(arg, dict):
                    values.append(str(arg.get("value", arg.get("description", ""))))
            page.console.append(f"{params.get('type', 'log')}: {' '.join(values)}")
        elif method in {"Runtime.exceptionThrown", "Log.entryAdded"}:
            if method == "Runtime.exceptionThrown":
                details = params.get("exceptionDetails", {})
                text = details.get("text") or details.get("exception", {}).get("description") or "JavaScript exception"
            else:
                text = params.get("entry", {}).get("text") or "Log error"
            page.errors.append(str(text))
        elif method == "Network.requestWillBeSent":
            request = params.get("request", {})
            request_id = params.get("requestId")
            if isinstance(request_id, str):
                page.network_inflight.add(request_id)
            page.last_network_activity = time.monotonic()
            page.network.append(
                {"kind": "request", "method": request.get("method"), "url": request.get("url"), "type": params.get("type")}
            )
        elif method == "Network.responseReceived":
            response = params.get("response", {})
            page.network.append(
                {"kind": "response", "status": response.get("status"), "url": response.get("url"), "type": params.get("type")}
            )
        elif method in {"Network.loadingFinished", "Network.loadingFailed"}:
            request_id = params.get("requestId")
            if isinstance(request_id, str):
                page.network_inflight.discard(request_id)
            page.last_network_activity = time.monotonic()

    async def create_page(self, url: str = "about:blank") -> PageState:
        if len(self.pages) >= self.config.max_pages:
            raise CommandError(f"The browser already has the configured maximum of {self.config.max_pages} tabs.")
        result = await self.client.call("Target.createTarget", {"url": url})
        target_id = str(result.get("targetId") or "")
        for _ in range(100):
            page_id = self.target_to_page.get(target_id)
            if page_id and page_id in self.pages:
                return self.pages[page_id]
            await asyncio.sleep(0.02)
        info = (await self.client.call("Target.getTargetInfo", {"targetId": target_id})).get("targetInfo", {})
        return await self._attach_page(info)

    def get_page(self, page_id: str | None = None) -> PageState:
        if page_id:
            page = self.pages.get(page_id)
            if page is None:
                raise CommandError(f"Unknown tab {page_id}. Use `tabs` to list open tabs.")
            return page
        if not self.pages:
            raise CommandError("No browser tab is open. Use `open URL` first.")
        return next(reversed(self.pages.values()))

    async def close_page(self, page: PageState) -> None:
        await self.client.call("Target.closeTarget", {"targetId": page.target_id})
        self.pages.pop(page.page_id, None)
        self.target_to_page.pop(page.target_id, None)
        self.session_to_page.pop(page.session_id, None)

    async def refresh_page_info(self, page: PageState) -> None:
        with suppress(Exception):
            info = (await self.client.call("Target.getTargetInfo", {"targetId": page.target_id})).get("targetInfo", {})
            page.url = str(info.get("url") or page.url)
            page.title = str(info.get("title") or page.title)

    async def close(self, *, clear_profile: bool = False) -> None:
        if self.recording is not None:
            recording, self.recording = self.recording, None
            recording.task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await recording.task
        if self._listener_remove:
            self._listener_remove()
            self._listener_remove = None
        launched, self.launched = self.launched, None
        self.pages.clear()
        self.target_to_page.clear()
        self.session_to_page.clear()
        self.frame_sessions.clear()
        self.execution_contexts.clear()
        self.downloads.clear()
        await self.bridge.close()
        if launched:
            await launched.close()
        if clear_profile:
            shutil.rmtree(self.workspace_dir / ".browser" / "chromium-profile", ignore_errors=True)


@dataclass(slots=True)
class _PoolEntry:
    runtime: BrowserRuntime
    idle_task: asyncio.Task[None] | None = None


class BrowserRuntimePool:
    def __init__(self) -> None:
        self._entries: dict[str, _PoolEntry] = {}
        self._clients: dict[str, set[int]] = {}
        self._lock = asyncio.Lock()

    def register(self, workspace_dir: Path, client_id: int) -> None:
        key = str(workspace_dir.resolve())
        self._clients.setdefault(key, set()).add(client_id)

    async def get(
        self,
        workspace_dir: Path,
        config: BrowserConfig,
        *,
        container_root: str | None,
        container_name: str | None,
        bridge_localhost: bool,
    ) -> BrowserRuntime:
        key = str(workspace_dir.resolve())
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _PoolEntry(
                    BrowserRuntime(
                        workspace_dir.resolve(), config,
                        container_root=container_root,
                        container_name=container_name,
                        bridge_localhost=bridge_localhost,
                    )
                )
                self._entries[key] = entry
            if entry.idle_task:
                entry.idle_task.cancel()
                entry.idle_task = None
            runtime = entry.runtime
        async with runtime.lock:
            await runtime.start()
        runtime.last_used_at = time.monotonic()
        return runtime

    async def release(self, workspace_dir: Path) -> None:
        key = str(workspace_dir.resolve())
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.runtime.last_used_at = time.monotonic()
            ttl = entry.runtime.config.idle_ttl_seconds
            if entry.idle_task:
                entry.idle_task.cancel()
            if ttl > 0:
                entry.idle_task = asyncio.create_task(self._expire(key, entry, ttl))

    async def _expire(self, key: str, entry: _PoolEntry, ttl: int) -> None:
        try:
            await asyncio.sleep(ttl)
            async with self._lock:
                if self._entries.get(key) is not entry:
                    return
                if entry.runtime.recording is not None:
                    entry.idle_task = asyncio.create_task(self._expire(key, entry, ttl))
                    return
                self._entries.pop(key, None)
            async with entry.runtime.lock:
                await entry.runtime.close()
        except asyncio.CancelledError:
            return

    async def reset(self, workspace_dir: Path, *, clear_profile: bool) -> bool:
        key = str(workspace_dir.resolve())
        async with self._lock:
            entry = self._entries.pop(key, None)
            if entry and entry.idle_task:
                entry.idle_task.cancel()
        if entry:
            async with entry.runtime.lock:
                await entry.runtime.close(clear_profile=clear_profile)
            return True
        if clear_profile:
            shutil.rmtree(workspace_dir / ".browser" / "chromium-profile", ignore_errors=True)
        return False

    async def unregister(self, workspace_dir: Path, client_id: int) -> None:
        key = str(workspace_dir.resolve())
        async with self._lock:
            clients = self._clients.get(key)
            if clients is not None:
                clients.discard(client_id)
                if clients:
                    return
                self._clients.pop(key, None)
            entry = self._entries.pop(key, None)
            if entry and entry.idle_task:
                entry.idle_task.cancel()
        if entry:
            async with entry.runtime.lock:
                await entry.runtime.close()


runtime_pool = BrowserRuntimePool()
