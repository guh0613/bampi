from __future__ import annotations

import asyncio
import base64
import importlib
import json
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import PydanticUndefined

from bampy.agent.cancellation import CancellationError, CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import ImageContent, TextContent

from .workspace import ensure_workspace_dirs, host_to_container_path, resolve_workspace_path, to_workspace_relative

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page


BrowserAction = Literal[
    "open",
    "goto",
    "click",
    "type",
    "press",
    "wait",
    "extract",
    "screenshot",
    "evaluate",
    "pages",
    "switch",
    "close_page",
    "reload",
    "back",
    "forward",
    "scroll",
    "reset",
]
LoadState = Literal["domcontentloaded", "load", "networkidle"]
WaitUntilState = Literal["commit", "domcontentloaded", "load", "networkidle"]
SelectorState = Literal["attached", "detached", "hidden", "visible"]
MouseButton = Literal["left", "middle", "right"]
ScreenshotFormat = Literal["png", "jpeg"]
ExtractFormat = Literal["text", "html"]
ScrollTarget = Literal["top", "bottom"]

_DEFAULT_TEXT_PREVIEW_CHARS = 1_500
_DEFAULT_EXTRACT_CHARS = 4_000


@dataclass(slots=True)
class _BrowserApi:
    async_camoufox_cls: Any
    playwright_error: type[BaseException]
    playwright_timeout_error: type[BaseException]


@dataclass(slots=True)
class _BrowserRuntime:
    manager: Any
    context: "BrowserContext"
    profile_dir: Path
    launched_at: float
    last_used_at: float


class BrowserToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: BrowserAction = Field(description="Browser action to run.")
    page_id: str | None = Field(default=None, description="Target page id. Uses the active page when omitted.")
    url: str | None = Field(default=None, description="Navigation URL for open/goto.")
    selector: str | None = Field(default=None, description="CSS selector for click/type/wait/extract/screenshot.")
    text: str | None = Field(default=None, description="Text to fill or type into an element.")
    keys: str | None = Field(default=None, description="Keyboard shortcut or key sequence, for example Enter or Meta+L.")
    script: str | None = Field(default=None, description="JavaScript expression or function body for evaluate.")
    argument: Any | None = Field(default=None, description="Optional JSON-serializable argument passed to evaluate().")
    path: str | None = Field(default=None, description="Screenshot output path inside the workspace. Defaults to outbox/browser/...")
    timeout_ms: int | None = Field(default=None, ge=1, le=120_000, description="Per-action timeout in milliseconds.")
    duration_ms: int | None = Field(default=None, ge=0, le=120_000, description="Sleep duration for wait or post-scroll settling.")
    wait_until: WaitUntilState = Field(default="domcontentloaded", description="Navigation wait target for open/goto/reload/back/forward.")
    load_state: LoadState | None = Field(default=None, description="Optional load state to wait for after an action, or the target state for wait.")
    state: SelectorState = Field(default="visible", description="Selector wait state used by wait and element interactions.")
    url_contains: str | None = Field(default=None, description="Substring to wait for in the current page URL.")
    click_count: int = Field(default=1, ge=1, le=3, description="How many times to click.")
    button: MouseButton = Field(default="left", description="Mouse button used for click.")
    clear_first: bool = Field(default=True, description="Whether type should clear the field before entering text.")
    full_page: bool = Field(default=False, description="Whether page screenshots should capture the full page.")
    return_image: bool = Field(
        default=True,
        description="Whether screenshot should also return an inline image result when the file is small enough.",
    )
    screenshot_format: ScreenshotFormat = Field(default="png", description="Screenshot file format.")
    quality: int | None = Field(default=None, ge=1, le=100, description="JPEG quality. Ignored for PNG.")
    extract_format: ExtractFormat = Field(default="text", description="Whether extract returns rendered text or full HTML.")
    max_chars: int = Field(default=_DEFAULT_EXTRACT_CHARS, ge=200, le=40_000, description="Maximum characters returned by extract/evaluate.")
    delta_x: int = Field(default=0, ge=-20_000, le=20_000, description="Horizontal scroll distance for scroll.")
    delta_y: int = Field(default=800, ge=-20_000, le=20_000, description="Vertical scroll distance for scroll.")
    scroll_to: ScrollTarget | None = Field(default=None, description="Scroll directly to the top or bottom of the page.")
    clear_profile: bool = Field(default=False, description="When reset is used, also delete the persisted browser profile.")

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls_for_defaulted_non_nullable_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        for name, field in cls.model_fields.items():
            if normalized.get(name) is not None:
                continue
            if name not in normalized:
                continue
            if field.default in (None, PydanticUndefined):
                continue
            normalized.pop(name, None)
        return normalized

    @model_validator(mode="after")
    def _validate_action_requirements(self) -> "BrowserToolInput":
        if self.action == "open" and not self.url:
            raise ValueError("open requires url")
        if self.action == "goto" and not self.url:
            raise ValueError("goto requires url")
        if self.action == "click" and not self.selector:
            raise ValueError("click requires selector")
        if self.action == "type" and (not self.selector or self.text is None):
            raise ValueError("type requires selector and text")
        if self.action == "press" and not self.keys:
            raise ValueError("press requires keys")
        if self.action == "evaluate" and not self.script:
            raise ValueError("evaluate requires script")
        if self.action in {"switch", "close_page"} and not self.page_id:
            raise ValueError(f"{self.action} requires page_id")
        if self.action == "wait" and not any(
            (
                self.selector,
                self.load_state,
                self.url_contains,
                self.duration_ms is not None,
            )
        ):
            raise ValueError("wait requires selector, load_state, url_contains, or duration_ms")
        if self.action == "scroll" and self.scroll_to is None and self.delta_x == 0 and self.delta_y == 0:
            raise ValueError("scroll requires scroll_to or a non-zero delta_x/delta_y")
        if self.quality is not None and self.screenshot_format != "jpeg":
            raise ValueError("quality can only be used with screenshot_format=jpeg")
        return self


def _import_browser_api() -> _BrowserApi:
    try:
        camoufox_module = importlib.import_module("camoufox.async_api")
        playwright_module = importlib.import_module("playwright.async_api")
    except ImportError as exc:
        raise RuntimeError(
            "Camoufox browser support is unavailable. "
            "Install it with `uv add camoufox`, then download the browser with "
            "`uv run python -m camoufox fetch`."
        ) from exc

    return _BrowserApi(
        async_camoufox_cls=camoufox_module.AsyncCamoufox,
        playwright_error=getattr(playwright_module, "Error", Exception),
        playwright_timeout_error=getattr(playwright_module, "TimeoutError", TimeoutError),
    )


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    text = value.strip()
    if len(text) <= limit:
        return text, False
    return text[: max(0, limit - 3)].rstrip() + "...", True


def _serialize_result(value: Any, *, limit: int) -> tuple[str, bool]:
    if isinstance(value, str):
        return _truncate_text(value, limit)
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        text = str(value)
    return _truncate_text(text, limit)


def _now_timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


class BrowserTool:
    name = "browser"
    label = "browser"
    description = (
        "Control a real Camoufox browser with persistent state. "
        "Open pages, click/type/wait on rendered DOM, run JavaScript, manage tabs, and save screenshots."
    )
    parameters = BrowserToolInput

    def __init__(
        self,
        workspace_dir: str,
        *,
        container_root: str | None = None,
        headless: bool = True,
        block_images: bool = False,
        launch_timeout: float = 45.0,
        action_timeout: float = 20.0,
        idle_ttl_seconds: int = 300,
        max_pages: int = 6,
        inline_image_max_bytes: int = 1_000_000,
    ) -> None:
        self._workspace_dir = str(ensure_workspace_dirs(workspace_dir))
        self._container_root = container_root
        self._headless = headless
        self._block_images = block_images
        self._launch_timeout = launch_timeout
        self._action_timeout = action_timeout
        self._idle_ttl_seconds = idle_ttl_seconds
        self._max_pages = max_pages
        self._inline_image_max_bytes = inline_image_max_bytes

        self._lock = asyncio.Lock()
        self._runtime: _BrowserRuntime | None = None
        self._pages: dict[str, "Page"] = {}
        self._active_page_id: str | None = None
        self._page_sequence = 1

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        arguments = BrowserToolInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
        )

        async with self._lock:
            if cancellation is not None:
                cancellation.raise_if_cancelled()

            await self._close_if_idle_locked()
            result = await self._execute_locked(arguments)

            if cancellation is not None and cancellation.cancelled:
                raise CancellationError(cancellation.reason or "Browser action aborted")
            return result

    async def close(self) -> None:
        async with self._lock:
            await self._close_locked(clear_profile=False)

    async def _execute_locked(self, arguments: BrowserToolInput) -> AgentToolResult:
        action = arguments.action

        if action == "pages":
            return await self._pages_result_locked()
        if action == "reset":
            return await self._reset_result_locked(clear_profile=arguments.clear_profile)

        if action == "open":
            return await self._open_page_locked(arguments)
        if action == "goto":
            page_id, page = await self._page_for_navigation_locked(arguments.page_id)
            await page.goto(arguments.url, wait_until=arguments.wait_until, timeout=self._timeout_ms(arguments))
            return AgentToolResult(content=[TextContent(text=await self._page_summary_text(page_id, page, include_preview=True))])

        page_id, page = await self._require_page_locked(arguments.page_id)

        if action == "click":
            locator = page.locator(arguments.selector).first
            await locator.click(
                button=arguments.button,
                click_count=arguments.click_count,
                timeout=self._timeout_ms(arguments),
            )
            if arguments.load_state is not None:
                await page.wait_for_load_state(arguments.load_state, timeout=self._timeout_ms(arguments))
            return AgentToolResult(content=[TextContent(text=await self._page_summary_text(page_id, page, include_preview=True))])

        if action == "type":
            locator = page.locator(arguments.selector).first
            if arguments.clear_first:
                await locator.fill(arguments.text, timeout=self._timeout_ms(arguments))
            else:
                await locator.click(timeout=self._timeout_ms(arguments))
                await page.keyboard.type(arguments.text)
            return AgentToolResult(content=[TextContent(text=f"Typed into {arguments.selector} on {page_id}.")])

        if action == "press":
            if arguments.selector:
                await page.locator(arguments.selector).first.press(arguments.keys, timeout=self._timeout_ms(arguments))
            else:
                await page.keyboard.press(arguments.keys)
            if arguments.load_state is not None:
                await page.wait_for_load_state(arguments.load_state, timeout=self._timeout_ms(arguments))
            return AgentToolResult(content=[TextContent(text=f"Pressed {arguments.keys} on {page_id}.")])

        if action == "wait":
            return await self._wait_result_locked(page_id, page, arguments)

        if action == "extract":
            return await self._extract_result_locked(page_id, page, arguments)

        if action == "screenshot":
            return await self._screenshot_result_locked(page_id, page, arguments)

        if action == "evaluate":
            value = await page.evaluate(arguments.script, arguments.argument)
            serialized, truncated = _serialize_result(value, limit=arguments.max_chars)
            message = f"Evaluate result on {page_id}:\n{serialized}"
            if truncated:
                message += "\n\n[Result truncated]"
            return AgentToolResult(content=[TextContent(text=message)])

        if action == "switch":
            self._active_page_id = page_id
            return AgentToolResult(content=[TextContent(text=await self._page_summary_text(page_id, page, include_preview=True))])

        if action == "close_page":
            await page.close()
            self._pages.pop(page_id, None)
            if self._active_page_id == page_id:
                self._active_page_id = next(reversed(self._pages), None)
            if not self._pages:
                await self._close_locked(clear_profile=False)
                return AgentToolResult(content=[TextContent(text=f"Closed {page_id}. No pages remain; browser session released.")])
            return AgentToolResult(content=[TextContent(text=await self._pages_text_locked(prefix=f"Closed {page_id}."))])

        if action == "reload":
            await page.reload(wait_until=arguments.wait_until, timeout=self._timeout_ms(arguments))
            return AgentToolResult(content=[TextContent(text=await self._page_summary_text(page_id, page, include_preview=True))])

        if action == "back":
            await page.go_back(wait_until=arguments.wait_until, timeout=self._timeout_ms(arguments))
            return AgentToolResult(content=[TextContent(text=await self._page_summary_text(page_id, page, include_preview=True))])

        if action == "forward":
            await page.go_forward(wait_until=arguments.wait_until, timeout=self._timeout_ms(arguments))
            return AgentToolResult(content=[TextContent(text=await self._page_summary_text(page_id, page, include_preview=True))])

        if action == "scroll":
            if arguments.scroll_to == "top":
                await page.evaluate("window.scrollTo(0, 0)")
            elif arguments.scroll_to == "bottom":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                await page.evaluate("([x, y]) => window.scrollBy(x, y)", [arguments.delta_x, arguments.delta_y])
            if arguments.duration_ms:
                await asyncio.sleep(arguments.duration_ms / 1000)
            return AgentToolResult(content=[TextContent(text=await self._page_summary_text(page_id, page, include_preview=True))])

        raise RuntimeError(f"Unsupported browser action: {action}")

    async def _pages_result_locked(self) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=await self._pages_text_locked())])

    async def _reset_result_locked(self, *, clear_profile: bool) -> AgentToolResult:
        closed = await self._close_locked(clear_profile=clear_profile)
        if not closed and not clear_profile:
            return AgentToolResult(content=[TextContent(text="Browser session is already idle.")])
        message = "Browser session reset."
        if clear_profile:
            message += " Persistent profile data was also removed."
        return AgentToolResult(content=[TextContent(text=message)])

    async def _open_page_locked(self, arguments: BrowserToolInput) -> AgentToolResult:
        context = await self._ensure_context_locked()
        self._sync_pages_from_context_locked(context)
        if len(self._pages) >= self._max_pages:
            raise RuntimeError(
                f"Browser already has {len(self._pages)} open pages, which reaches the configured limit "
                f"of {self._max_pages}. Close a page or reset the browser first."
            )

        page = await context.new_page()
        page_id = self._register_page_locked(page)
        self._active_page_id = page_id
        await page.goto(arguments.url, wait_until=arguments.wait_until, timeout=self._timeout_ms(arguments))
        return AgentToolResult(content=[TextContent(text=await self._page_summary_text(page_id, page, include_preview=True))])

    async def _wait_result_locked(self, page_id: str, page: "Page", arguments: BrowserToolInput) -> AgentToolResult:
        timeout_ms = self._timeout_ms(arguments)
        if arguments.selector:
            await page.locator(arguments.selector).first.wait_for(
                state=arguments.state,
                timeout=timeout_ms,
            )
            return AgentToolResult(content=[TextContent(text=f"Selector {arguments.selector} reached state {arguments.state} on {page_id}.")])
        if arguments.load_state:
            await page.wait_for_load_state(arguments.load_state, timeout=timeout_ms)
            return AgentToolResult(content=[TextContent(text=f"Page {page_id} reached load state {arguments.load_state}.")])
        if arguments.url_contains:
            await self._wait_for_url_contains(page, arguments.url_contains, timeout_ms)
            return AgentToolResult(content=[TextContent(text=f"Page {page_id} URL now contains {arguments.url_contains}.")])
        await asyncio.sleep((arguments.duration_ms or 0) / 1000)
        return AgentToolResult(content=[TextContent(text=f"Waited {arguments.duration_ms or 0}ms on {page_id}.")])

    async def _extract_result_locked(
        self,
        page_id: str,
        page: "Page",
        arguments: BrowserToolInput,
    ) -> AgentToolResult:
        if arguments.selector:
            locator = page.locator(arguments.selector).first
            if arguments.extract_format == "html":
                value = await locator.evaluate("(el) => el.outerHTML", timeout=self._timeout_ms(arguments))
            else:
                value = await locator.inner_text(timeout=self._timeout_ms(arguments))
            serialized, truncated = _truncate_text(value or "", arguments.max_chars)
            lines = [
                f"Extracted {arguments.extract_format} from {arguments.selector} on {page_id}:",
                serialized or "(empty)",
            ]
            if truncated:
                lines.append("")
                lines.append("[Extract truncated]")
            return AgentToolResult(content=[TextContent(text="\n".join(lines))])

        if arguments.extract_format == "html":
            value = await page.content()
        else:
            value = await page.evaluate(
                """
                (limit) => {
                    const body = document.body;
                    const text = body ? (body.innerText || '').trim() : '';
                    return text.slice(0, limit * 2);
                }
                """,
                arguments.max_chars,
            )

        serialized, truncated = _truncate_text(value or "", arguments.max_chars)
        lines = [
            await self._page_summary_text(page_id, page, include_preview=False),
            "",
            f"{arguments.extract_format.upper()} CONTENT:",
            serialized or "(empty)",
        ]
        if truncated:
            lines.append("")
            lines.append("[Extract truncated]")
        return AgentToolResult(content=[TextContent(text="\n".join(lines))])

    async def _screenshot_result_locked(
        self,
        page_id: str,
        page: "Page",
        arguments: BrowserToolInput,
    ) -> AgentToolResult:
        host_path = self._resolve_screenshot_path(arguments.path, page_id, arguments.screenshot_format)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        timeout_ms = self._timeout_ms(arguments)

        screenshot_kwargs: dict[str, Any] = {
            "path": str(host_path),
            "type": arguments.screenshot_format,
            "timeout": timeout_ms,
        }
        if arguments.screenshot_format == "jpeg" and arguments.quality is not None:
            screenshot_kwargs["quality"] = arguments.quality

        if arguments.selector:
            await page.locator(arguments.selector).first.screenshot(**screenshot_kwargs)
        else:
            await page.screenshot(
                **screenshot_kwargs,
                full_page=arguments.full_page,
            )

        blocks: list[Any] = []
        relative_path = to_workspace_relative(self._workspace_dir, host_path)
        lines = [
            f"Saved screenshot for {page_id}.",
            f"Workspace path: {relative_path}",
        ]
        if self._container_root is not None:
            container_path = host_to_container_path(self._workspace_dir, host_path, self._container_root)
            lines.append(f"Container path: {container_path}")

        data = host_path.read_bytes()
        if arguments.return_image:
            if len(data) <= self._inline_image_max_bytes:
                blocks.append(
                    ImageContent(
                        data=base64.b64encode(data).decode("ascii"),
                        mime_type="image/jpeg" if arguments.screenshot_format == "jpeg" else "image/png",
                    )
                )
            else:
                lines.append(
                    f"Inline image skipped because the screenshot is {len(data)} bytes, "
                    f"above the inline limit of {self._inline_image_max_bytes} bytes."
                )

        blocks.insert(0, TextContent(text="\n".join(lines)))
        return AgentToolResult(content=blocks)

    async def _ensure_context_locked(self) -> "BrowserContext":
        if self._runtime is not None:
            self._runtime.last_used_at = time.monotonic()
            self._sync_pages_from_context_locked(self._runtime.context)
            return self._runtime.context

        api = _import_browser_api()
        profile_dir = self._profile_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)

        firefox_user_prefs = {
            "dom.webnotifications.enabled": False,
            "media.autoplay.default": 5,
        }
        manager = api.async_camoufox_cls(
            persistent_context=True,
            user_data_dir=str(profile_dir),
            headless=self._headless,
            block_images=self._block_images,
            block_webrtc=True,
            enable_cache=False,
            firefox_user_prefs=firefox_user_prefs,
            timeout=int(self._launch_timeout * 1000),
        )

        try:
            context = await asyncio.wait_for(
                manager.__aenter__(),
                timeout=max(self._launch_timeout + 5.0, 10.0),
            )
        except Exception as exc:
            try:
                await manager.__aexit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass
            raise RuntimeError(self._launch_error_message(exc)) from exc

        self._runtime = _BrowserRuntime(
            manager=manager,
            context=context,
            profile_dir=profile_dir,
            launched_at=time.monotonic(),
            last_used_at=time.monotonic(),
        )
        self._sync_pages_from_context_locked(context)
        return context

    async def _close_if_idle_locked(self) -> None:
        runtime = self._runtime
        if runtime is None or self._idle_ttl_seconds <= 0:
            return
        if time.monotonic() - runtime.last_used_at < self._idle_ttl_seconds:
            return
        await self._close_locked(clear_profile=False)

    async def _close_locked(self, *, clear_profile: bool) -> bool:
        runtime = self._runtime
        self._runtime = None
        self._pages.clear()
        self._active_page_id = None

        if runtime is None:
            if clear_profile:
                shutil.rmtree(self._profile_dir(), ignore_errors=True)
            return False

        try:
            await runtime.manager.__aexit__(None, None, None)
        finally:
            if clear_profile:
                shutil.rmtree(runtime.profile_dir, ignore_errors=True)
        return True

    def _sync_pages_from_context_locked(self, context: "BrowserContext") -> None:
        current_pages = [page for page in context.pages if not page.is_closed()]
        current_ids = {id(page) for page in current_pages}

        stale_ids = [
            page_id
            for page_id, page in self._pages.items()
            if page.is_closed() or id(page) not in current_ids
        ]
        for page_id in stale_ids:
            self._pages.pop(page_id, None)
            if self._active_page_id == page_id:
                self._active_page_id = None

        for page in current_pages:
            self._register_page_locked(page)

        if self._active_page_id not in self._pages:
            self._active_page_id = next(reversed(self._pages), None)

        if self._runtime is not None:
            self._runtime.last_used_at = time.monotonic()

    def _register_page_locked(self, page: "Page") -> str:
        for existing_id, existing_page in self._pages.items():
            if existing_page is page:
                return existing_id
        page_id = f"p{self._page_sequence}"
        self._page_sequence += 1
        self._pages[page_id] = page
        return page_id

    async def _page_for_navigation_locked(self, page_id: str | None) -> tuple[str, "Page"]:
        context = await self._ensure_context_locked()
        self._sync_pages_from_context_locked(context)
        if page_id:
            return await self._require_page_locked(page_id)
        if self._active_page_id in self._pages:
            return self._active_page_id, self._pages[self._active_page_id]
        if self._pages:
            current_id = next(reversed(self._pages))
            self._active_page_id = current_id
            return current_id, self._pages[current_id]
        if len(self._pages) >= self._max_pages:
            raise RuntimeError(
                f"Browser already has {len(self._pages)} open pages, which reaches the configured limit "
                f"of {self._max_pages}. Close a page or reset the browser first."
            )
        page = await context.new_page()
        current_id = self._register_page_locked(page)
        self._active_page_id = current_id
        return current_id, page

    async def _require_page_locked(self, page_id: str | None) -> tuple[str, "Page"]:
        context = await self._ensure_context_locked()
        self._sync_pages_from_context_locked(context)
        effective_page_id = page_id or self._active_page_id
        if not effective_page_id or effective_page_id not in self._pages:
            raise RuntimeError(
                "No active browser page is available. Use `browser` with action `open` or `goto` first."
            )
        self._active_page_id = effective_page_id
        return effective_page_id, self._pages[effective_page_id]

    async def _pages_text_locked(self, *, prefix: str | None = None) -> str:
        if self._runtime is None or not self._pages:
            return "Browser session is not started."

        lines: list[str] = []
        if prefix:
            lines.append(prefix)
            lines.append("")
        lines.append(f"Open pages ({len(self._pages)}/{self._max_pages}):")
        for page_id, page in self._pages.items():
            marker = " [active]" if page_id == self._active_page_id else ""
            title = await self._safe_page_title(page)
            lines.append(f"- {page_id}{marker}: {title or '(untitled)'}")
            lines.append(f"  URL: {page.url or '(blank)'}")
        return "\n".join(lines)

    async def _page_summary_text(
        self,
        page_id: str,
        page: "Page",
        *,
        include_preview: bool,
    ) -> str:
        title = await self._safe_page_title(page)
        ready_state = await self._safe_ready_state(page)
        lines = [
            f"Page ID: {page_id}",
            f"Title: {title or '(untitled)'}",
            f"URL: {page.url or '(blank)'}",
            f"Ready state: {ready_state or 'unknown'}",
        ]
        if include_preview:
            preview = await self._page_text_preview(page, _DEFAULT_TEXT_PREVIEW_CHARS)
            if preview:
                lines.append("")
                lines.append("Text preview:")
                lines.append(preview)
        return "\n".join(lines)

    async def _safe_page_title(self, page: "Page") -> str:
        try:
            return await page.title()
        except Exception:
            return ""

    async def _safe_ready_state(self, page: "Page") -> str:
        try:
            value = await page.evaluate("document.readyState")
        except Exception:
            return ""
        return str(value or "")

    async def _page_text_preview(self, page: "Page", limit: int) -> str:
        try:
            text = await page.evaluate(
                """
                (limit) => {
                    const body = document.body;
                    const text = body ? (body.innerText || '').trim() : '';
                    return text.slice(0, limit * 2);
                }
                """,
                limit,
            )
        except Exception:
            return ""
        preview, _ = _truncate_text(text or "", limit)
        return preview

    async def _wait_for_url_contains(self, page: "Page", needle: str, timeout_ms: int) -> None:
        deadline = time.monotonic() + (timeout_ms / 1000)
        while time.monotonic() < deadline:
            if needle in (page.url or ""):
                return
            await asyncio.sleep(0.1)
        raise RuntimeError(f"Timed out waiting for the page URL to contain {needle!r}.")

    def _timeout_ms(self, arguments: BrowserToolInput) -> int:
        return arguments.timeout_ms if arguments.timeout_ms is not None else int(self._action_timeout * 1000)

    def _profile_dir(self) -> Path:
        return Path(self._workspace_dir) / ".browser" / "camoufox-profile"

    def _resolve_screenshot_path(
        self,
        raw_path: str | None,
        page_id: str,
        screenshot_format: ScreenshotFormat,
    ) -> Path:
        if raw_path:
            return resolve_workspace_path(
                self._workspace_dir,
                raw_path,
                container_root=self._container_root,
            )
        relative = Path("outbox") / "browser" / f"{_now_timestamp_slug()}-{page_id}.{screenshot_format}"
        return (Path(self._workspace_dir) / relative).resolve()

    def _launch_error_message(self, exc: Exception) -> str:
        text = str(exc).strip() or exc.__class__.__name__
        lowered = text.lower()
        if "executable" in lowered or "browser" in lowered or "camoufox" in lowered:
            return (
                f"Failed to launch Camoufox: {text}\n\n"
                "If this is the first time using the browser tool, run "
                "`uv run python -m camoufox fetch` to download the patched Firefox build."
            )
        return f"Failed to launch Camoufox: {text}"
