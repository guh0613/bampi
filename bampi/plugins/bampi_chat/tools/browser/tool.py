from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bampy.agent.cancellation import CancellationError, CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import ImageContent, TextContent

from .commands import BrowserCommandDispatcher
from .config import BrowserConfig
from .runtime import runtime_pool


class BrowserToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(
        min_length=1,
        description=(
            "Browser command. Core syntax:\n"
            "  open URL | goto URL — navigate to a page\n"
            "  snapshot [--interactive] [--depth N] [--scope TARGET] — capture page state with @eN refs\n"
            "  click @eN | dblclick | hover | focus — interact with snapshot element refs\n"
            "  fill @eN \"text\" | type @eN \"text\" | press KEY — form input\n"
            "  select | check | uncheck — dropdowns and checkboxes\n"
            "  wait 3 — wait N seconds\n"
            "  wait --load domcontentloaded|load|networkidle — wait for page load state (value required)\n"
            "  wait --url GLOB | wait --text TEXT | wait --fn \"JS\" | wait TARGET [--state visible|hidden|detached]\n"
            "  extract [TARGET] | eval \"JS\" | get attr|value|count — inspection\n"
            "  scroll up|down|left|right|top|bottom|TARGET [AMOUNT]\n"
            "  screenshot [PATH] [--target TARGET] [--full] [--annotate] [--jpeg]\n"
            "  pdf [PATH] | record start|stop | tabs | tab PAGE_ID | close | reload | back | forward\n"
            "  drag SOURCE TARGET | upload TARGET PATH\n"
            "  batch [--continue] followed by one command per line (max 32)\n"
            "TARGET: @eN (from snapshot), css=..., text=..., label=..., placeholder=..., testid=..., role=button[name=Submit].\n"
            "Use `help` for full syntax reference."
        ),
    )


class BrowserTool:
    name = "browser"
    label = "browser"
    description = (
        "Control a headless automated Chromium browser via CDP and accessibility-tree refs. "
        "Covers navigation, snapshot, forms, drag/drop, screenshots, recording, and batch. "
        "Because it runs headless without human indicators, sites with bot detection (search engines, CAPTCHAs, "
        "login walls) will often block access — navigate directly to content pages instead. "
        "Typical flow: `open URL` → `snapshot` (returns @eN element refs) → interact (`click @e1`, `fill @e2 \"text\"`) → `screenshot`. "
        "Batch multiple steps with `batch` + one command per line."
    )
    parameters = BrowserToolInput

    def __init__(
        self,
        workspace_dir: str,
        *,
        container_root: str | None = None,
        container_name: str | None = None,
        bridge_localhost: bool = False,
        executable_path: str | None = None,
        auto_install: bool = True,
        cache_dir: str | None = None,
        install_timeout: float = 300.0,
        headless: bool = True,
        block_images: bool = False,
        launch_timeout: float = 45.0,
        action_timeout: float = 20.0,
        idle_ttl_seconds: int = 300,
        max_pages: int = 6,
        inline_image_max_bytes: int = 1_000_000,
        viewport_width: int = 1440,
        viewport_height: int = 1000,
        batch_max_commands: int = 32,
        batch_timeout: float = 120.0,
        recording_fps: int = 10,
        recording_max_seconds: int = 600,
        allow_private_network: bool = False,
    ) -> None:
        self._workspace_dir = Path(workspace_dir).resolve()
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        (self._workspace_dir / "outbox" / "browser").mkdir(parents=True, exist_ok=True)
        self._container_root = container_root
        self._container_name = container_name
        self._bridge_localhost = bridge_localhost
        self._config = BrowserConfig(
            executable_path=executable_path,
            auto_install=auto_install,
            cache_dir=cache_dir,
            install_timeout=install_timeout,
            headless=headless,
            block_images=block_images,
            launch_timeout=launch_timeout,
            action_timeout=action_timeout,
            idle_ttl_seconds=idle_ttl_seconds,
            max_pages=max_pages,
            inline_image_max_bytes=inline_image_max_bytes,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            batch_max_commands=batch_max_commands,
            batch_timeout=batch_timeout,
            recording_fps=recording_fps,
            recording_max_seconds=recording_max_seconds,
            allow_private_network=allow_private_network,
        )
        self._active_page_id: str | None = None
        self._closed = False
        runtime_pool.register(self._workspace_dir, id(self))

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id
        arguments = params if isinstance(params, BrowserToolInput) else BrowserToolInput.model_validate(params)
        if cancellation:
            cancellation.raise_if_cancelled()
        runtime = await runtime_pool.get(
            self._workspace_dir,
            self._config,
            container_root=self._container_root,
            container_name=self._container_name,
            bridge_localhost=self._bridge_localhost,
        )

        async def emit(text: str) -> None:
            if on_update is None:
                return
            result = on_update(AgentToolResult(content=[TextContent(text=text)]))
            if asyncio.iscoroutine(result):
                await result

        dispatcher = BrowserCommandDispatcher(
            runtime,
            active_page_id=self._active_page_id,
            cancellation=cancellation,
        )
        task: asyncio.Task | None = None
        cancel_task: asyncio.Task | None = None
        try:
            async with runtime.lock:
                task = asyncio.create_task(dispatcher.execute(arguments.command, on_update=emit))
                if cancellation is not None:
                    cancel_task = asyncio.create_task(cancellation.wait())
                    done, _ = await asyncio.wait({task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
                    if cancel_task in done and task not in done:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                        raise CancellationError(cancellation.reason or "Browser operation cancelled")
                output = await task
                self._active_page_id = dispatcher.active_page_id
        finally:
            if cancel_task is not None:
                cancel_task.cancel()
            await runtime_pool.release(self._workspace_dir)
        blocks: list[TextContent | ImageContent] = [TextContent(text=output.text)]
        if output.image_data is not None and output.image_mime_type:
            import base64
            blocks.append(ImageContent(data=base64.b64encode(output.image_data).decode("ascii"), mime_type=output.image_mime_type))
        return AgentToolResult(content=blocks)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await runtime_pool.unregister(self._workspace_dir, id(self))
