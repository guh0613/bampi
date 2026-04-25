from __future__ import annotations

import asyncio
import base64
import binascii
from contextlib import suppress
import importlib
import json
import os
import platform
from pathlib import PurePosixPath
import signal
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import PydanticUndefined

from bampy.agent.cancellation import CancellationError, CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import ImageContent, TextContent

from .workspace import (
    container_to_host_path,
    ensure_workspace_dirs,
    host_to_container_path,
    resolve_workspace_path,
    to_workspace_relative,
)

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page


BrowserAction = Literal[
    "open",
    "goto",
    "click",
    "type",
    "press",
    "wait",
    "observe",
    "extract",
    "screenshot",
    "screenshot_ref",
    "evaluate",
    "click_ref",
    "save_image",
    "extract_image",
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
_DEFAULT_OBSERVE_ITEMS = 40
_LOCAL_BROWSER_HOSTS = {"localhost", "127.0.0.1", "::1"}
_CJK_FONT_FALLBACKS = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Songti SC",
    "Noto Sans SC",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "Microsoft JhengHei",
]
_OBSERVE_SCRIPT = """
(limit) => {
  const toText = (value, max = 80) => String(value || '').replace(/\\s+/g, ' ').trim().slice(0, max);
  const className = (el) => {
    const raw = el.className;
    if (!raw) return '';
    if (typeof raw === 'string') return toText(raw, 120);
    return toText(raw.baseVal || raw.toString?.() || '', 120);
  };
  const rectOf = (el) => {
    const r = el.getBoundingClientRect();
    return {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)};
  };
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const style = getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none' && Number(style.opacity || '1') > 0.01;
  };
  const cssPath = (el) => {
    if (el.id && !/\\s/.test(el.id)) return `#${CSS.escape(el.id)}`;
    const parts = [];
    for (let node = el; node && node.nodeType === Node.ELEMENT_NODE && node !== document.body; node = node.parentElement) {
      let part = node.tagName.toLowerCase();
      if (node.classList && node.classList.length) {
        part += '.' + Array.from(node.classList).slice(0, 2).map((item) => CSS.escape(item)).join('.');
      }
      const parent = node.parentElement;
      if (parent) {
        const siblings = Array.from(parent.children).filter((item) => item.tagName === node.tagName);
        if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(node) + 1})`;
      }
      parts.unshift(part);
    }
    return parts.length ? parts.join(' > ') : el.tagName.toLowerCase();
  };
  const describe = (el, kind) => {
    const src = el.currentSrc || el.src || '';
    return {
      kind,
      tag: el.tagName.toLowerCase(),
      selector: cssPath(el),
      text: toText(el.innerText || el.textContent || el.getAttribute('aria-label') || el.getAttribute('title') || el.value, 120),
      aria: toText(el.getAttribute('aria-label') || el.getAttribute('title') || '', 80),
      role: toText(el.getAttribute('role') || '', 40),
      type: toText(el.getAttribute('type') || '', 40),
      id: toText(el.id || '', 80),
      class: className(el),
      href: toText(el.href || '', 160),
      src: toText(src, 180),
      rect: rectOf(el),
    };
  };
  const clickableSelector = [
    'a[href]',
    'button',
    '[role="button"]',
    '[onclick]',
    '[tabindex]:not([tabindex="-1"])',
    'input[type="button"]',
    'input[type="submit"]',
    'input[type="checkbox"]',
    'input[type="radio"]',
    '[class*="avatar" i]',
    '[class*="login" i]',
    '[class*="qrcode" i]',
    '[class*="qr-" i]'
  ].join(',');
  const all = Array.from(document.querySelectorAll(clickableSelector)).filter(visible);
  const seen = new Set();
  const unique = [];
  for (const el of all) {
    if (seen.has(el)) continue;
    seen.add(el);
    unique.push(el);
  }
  const inputs = Array.from(document.querySelectorAll('input, textarea, select'))
    .filter(visible)
    .slice(0, limit)
    .map((el) => describe(el, 'input'));
  const images = Array.from(document.querySelectorAll('img, canvas, svg'))
    .filter(visible)
    .sort((a, b) => {
      const ar = a.getBoundingClientRect();
      const br = b.getBoundingClientRect();
      return (br.width * br.height) - (ar.width * ar.height);
    })
    .slice(0, limit)
    .map((el) => describe(el, 'img'));
  return {
    url: location.href,
    title: document.title || '',
    text: toText(document.body ? document.body.innerText : '', 700),
    elements: unique.slice(0, limit).map((el) => describe(el, 'element')),
    inputs,
    images,
  };
}
"""
_SAVE_IMAGE_SCRIPT = """
async ({selector, imageIndex}) => {
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const style = getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none' && Number(style.opacity || '1') > 0.01;
  };
  const images = selector
    ? [document.querySelector(selector)].filter(Boolean)
    : Array.from(document.querySelectorAll('img, canvas, svg')).filter(visible).sort((a, b) => {
        const ar = a.getBoundingClientRect();
        const br = b.getBoundingClientRect();
        return (br.width * br.height) - (ar.width * ar.height);
      });
  const el = images[imageIndex || 0];
  if (!el) return null;
  if (el instanceof HTMLCanvasElement) {
    return {dataUrl: el.toDataURL('image/png'), src: 'canvas', width: el.width, height: el.height};
  }
  if (el instanceof SVGElement && !(el instanceof SVGImageElement)) {
    const xml = new XMLSerializer().serializeToString(el);
    const encoded = btoa(unescape(encodeURIComponent(xml)));
    const box = el.getBoundingClientRect();
    return {dataUrl: `data:image/svg+xml;base64,${encoded}`, src: 'inline-svg', width: Math.round(box.width), height: Math.round(box.height)};
  }
  const img = el;
  const src = img.currentSrc || img.src || '';
  if (src.startsWith('data:')) return {dataUrl: src, src, width: img.naturalWidth || img.width, height: img.naturalHeight || img.height};
  try {
    const response = await fetch(src, {credentials: 'include'});
    const blob = await response.blob();
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
    return {dataUrl, src, width: img.naturalWidth || img.width, height: img.naturalHeight || img.height};
  } catch (fetchError) {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    return {dataUrl: canvas.toDataURL('image/png'), src, width: canvas.width, height: canvas.height};
  }
}
"""
_DOCKER_PORT_BRIDGE_SCRIPT = """
import os
import select
import socket
import sys

port = int(sys.argv[1])
sock = socket.create_connection(("127.0.0.1", port))
sock.setblocking(False)
stdin_fd = sys.stdin.fileno()
stdout_fd = sys.stdout.fileno()
sock_fd = sock.fileno()
stdin_open = True

while True:
    read_fds = [sock_fd]
    if stdin_open:
        read_fds.append(stdin_fd)
    ready, _, _ = select.select(read_fds, [], [])
    if stdin_open and stdin_fd in ready:
        data = os.read(stdin_fd, 65536)
        if not data:
            stdin_open = False
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        else:
            sock.sendall(data)
    if sock_fd in ready:
        data = sock.recv(65536)
        if not data:
            break
        os.write(stdout_fd, data)

sock.close()
""".strip()


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


@dataclass(slots=True)
class _LocalPortBridge:
    target_port: int
    listen_port: int
    server: asyncio.AbstractServer


class BrowserToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: BrowserAction = Field(description="Browser action to run.")
    page_id: str | None = Field(default=None, description="Target page id. Uses the active page when omitted.")
    frame_id: str | None = Field(default=None, description="Target frame id from observe, for example f0 or f1.")
    frame_url_contains: str | None = Field(default=None, description="Target the first frame whose URL contains this substring.")
    ref: str | None = Field(default=None, description="Element/image ref returned by observe, for example f0:e1 or f1:img2.")
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
    max_items: int = Field(default=_DEFAULT_OBSERVE_ITEMS, ge=1, le=200, description="Maximum observed elements/images per frame.")
    image_index: int = Field(default=0, ge=0, le=500, description="Visible image index for save_image/extract_image when no selector/ref is supplied.")
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
        if self.action == "click_ref" and not self.ref:
            raise ValueError("click_ref requires ref")
        if self.action == "type" and (not self.selector or self.text is None):
            raise ValueError("type requires selector and text")
        if self.action == "press" and not self.keys:
            raise ValueError("press requires keys")
        if self.action == "evaluate" and not self.script:
            raise ValueError("evaluate requires script")
        if self.action == "screenshot_ref" and not self.ref:
            raise ValueError("screenshot_ref requires ref")
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


def _is_local_browser_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    return hostname.lower() in _LOCAL_BROWSER_HOSTS


def _default_port_for_scheme(scheme: str) -> int:
    if scheme == "https":
        return 443
    return 80


def _terminate_subprocess_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()


def _host_camoufox_os() -> Literal["windows", "macos", "linux"]:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return "linux"


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
        container_name: str | None = None,
        bridge_localhost: bool = False,
        headless: bool = True,
        block_images: bool = False,
        launch_timeout: float = 45.0,
        action_timeout: float = 20.0,
        idle_ttl_seconds: int = 300,
        max_pages: int = 6,
        inline_image_max_bytes: int = 1_000_000,
    ) -> None:
        self._workspace_dir = str(ensure_workspace_dirs(workspace_dir))
        self._container_root = PurePosixPath(container_root).as_posix() if container_root else None
        self._container_name = container_name
        self._bridge_localhost = bridge_localhost and bool(container_name)
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
        self._local_port_bridges: dict[int, _LocalPortBridge] = {}

    def _workspace_file_url(self, relative_path: str) -> str:
        if self._container_root is None:
            return ""
        visible_path = PurePosixPath(self._container_root) / PurePosixPath(relative_path)
        return f"file://{quote(visible_path.as_posix(), safe='/')}"

    def _display_url(self, url: str) -> str:
        if not url:
            return url

        parsed = urlsplit(url)
        if parsed.scheme != "file":
            return url

        raw_path = unquote(parsed.path or "")
        if not raw_path:
            return url

        host_path = Path(raw_path)
        try:
            relative_path = to_workspace_relative(self._workspace_dir, host_path)
        except ValueError:
            return url

        workspace_url = self._workspace_file_url(relative_path)
        return workspace_url or url

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
            return await self._navigate_page_locked(page_id, page, arguments)

        page_id, page = await self._require_page_locked(arguments.page_id)

        if action == "click":
            target = await self._frame_for_arguments_locked(page, arguments)
            locator = target.locator(arguments.selector).first
            try:
                await locator.click(
                    button=arguments.button,
                    click_count=arguments.click_count,
                    timeout=self._timeout_ms(arguments),
                )
            except Exception as exc:
                raise RuntimeError(await self._browser_failure_text(page_id, page, arguments, exc)) from exc
            if arguments.load_state is not None:
                await page.wait_for_load_state(arguments.load_state, timeout=self._timeout_ms(arguments))
            return AgentToolResult(content=[TextContent(text=await self._page_summary_text(page_id, page, include_preview=True))])

        if action == "click_ref":
            frame_id, target, item = await self._resolve_ref_locked(page, arguments.ref or "", max_items=arguments.max_items)
            try:
                await target.locator(item["selector"]).first.click(
                    button=arguments.button,
                    click_count=arguments.click_count,
                    timeout=self._timeout_ms(arguments),
                )
            except Exception as exc:
                raise RuntimeError(await self._browser_failure_text(page_id, page, arguments, exc)) from exc
            if arguments.load_state is not None:
                await page.wait_for_load_state(arguments.load_state, timeout=self._timeout_ms(arguments))
            return AgentToolResult(
                content=[
                    TextContent(
                        text=(
                            f"Clicked {arguments.ref} in {frame_id} "
                            f"({item.get('tag', 'element')} {item.get('text') or item.get('aria') or item.get('class') or ''}).\n\n"
                            + await self._page_summary_text(page_id, page, include_preview=True)
                        )
                    )
                ]
            )

        if action == "type":
            target = await self._frame_for_arguments_locked(page, arguments)
            locator = target.locator(arguments.selector).first
            try:
                if arguments.clear_first:
                    await locator.fill(arguments.text, timeout=self._timeout_ms(arguments))
                else:
                    await locator.click(timeout=self._timeout_ms(arguments))
                    await page.keyboard.type(arguments.text)
            except Exception as exc:
                raise RuntimeError(await self._browser_failure_text(page_id, page, arguments, exc)) from exc
            return AgentToolResult(content=[TextContent(text=f"Typed into {arguments.selector} on {page_id}.")])

        if action == "press":
            target = await self._frame_for_arguments_locked(page, arguments)
            try:
                if arguments.selector:
                    await target.locator(arguments.selector).first.press(arguments.keys, timeout=self._timeout_ms(arguments))
                else:
                    await page.keyboard.press(arguments.keys)
            except Exception as exc:
                raise RuntimeError(await self._browser_failure_text(page_id, page, arguments, exc)) from exc
            if arguments.load_state is not None:
                await page.wait_for_load_state(arguments.load_state, timeout=self._timeout_ms(arguments))
            return AgentToolResult(content=[TextContent(text=f"Pressed {arguments.keys} on {page_id}.")])

        if action == "wait":
            return await self._wait_result_locked(page_id, page, arguments)

        if action == "observe":
            return await self._observe_result_locked(page_id, page, arguments)

        if action == "extract":
            return await self._extract_result_locked(page_id, page, arguments)

        if action == "screenshot":
            return await self._screenshot_result_locked(page_id, page, arguments)

        if action == "screenshot_ref":
            frame_id, target, item = await self._resolve_ref_locked(page, arguments.ref or "", max_items=arguments.max_items)
            host_path = self._resolve_screenshot_path(arguments.path, page_id, arguments.screenshot_format)
            return await self._screenshot_element_result_locked(
                page_id,
                target.locator(item["selector"]).first,
                arguments,
                host_path=host_path,
                label=f"{arguments.ref} in {frame_id}",
            )

        if action == "evaluate":
            target = await self._frame_for_arguments_locked(page, arguments)
            value = await target.evaluate(arguments.script, arguments.argument)
            serialized, truncated = _serialize_result(value, limit=arguments.max_chars)
            message = f"Evaluate result on {page_id}:\n{serialized}"
            if truncated:
                message += "\n\n[Result truncated]"
            return AgentToolResult(content=[TextContent(text=message)])

        if action in {"save_image", "extract_image"}:
            return await self._save_image_result_locked(page_id, page, arguments)

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
        return await self._navigate_page_locked(page_id, page, arguments)

    async def _navigate_page_locked(
        self,
        page_id: str,
        page: "Page",
        arguments: BrowserToolInput,
    ) -> AgentToolResult:
        requested_url = arguments.url or ""
        resolved_url, notes = await self._rewrite_navigation_url_locked(requested_url)
        await page.goto(resolved_url, wait_until=arguments.wait_until, timeout=self._timeout_ms(arguments))

        summary = await self._page_summary_text(page_id, page, include_preview=True)
        display_resolved_url = self._display_url(resolved_url)
        if resolved_url == requested_url and not notes:
            return AgentToolResult(content=[TextContent(text=summary)])

        lines = [f"Requested URL: {requested_url}"]
        if display_resolved_url != requested_url:
            lines.append(f"Resolved URL: {display_resolved_url}")
        lines.extend(notes)
        lines.extend(["", summary])
        return AgentToolResult(content=[TextContent(text="\n".join(lines))])

    async def _rewrite_navigation_url_locked(self, raw_url: str) -> tuple[str, list[str]]:
        file_url, file_notes = self._rewrite_file_url(raw_url)
        if file_url != raw_url or file_notes:
            return file_url, file_notes
        return await self._rewrite_localhost_url_locked(raw_url)

    def _rewrite_file_url(self, raw_url: str) -> tuple[str, list[str]]:
        parsed = urlsplit(raw_url)
        if parsed.scheme != "file":
            return raw_url, []

        raw_path = unquote(parsed.path or "")
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            relative_parts = [unquote(parsed.netloc)]
            stripped_path = raw_path.lstrip("/")
            if stripped_path:
                relative_parts.append(stripped_path)
            relative_path = PurePosixPath(*relative_parts).as_posix()
            try:
                workspace_path = resolve_workspace_path(
                    self._workspace_dir,
                    relative_path,
                    container_root=self._container_root,
                )
            except ValueError:
                workspace_path = None
            else:
                workspace_url = self._workspace_file_url(
                    to_workspace_relative(self._workspace_dir, workspace_path)
                )
                return workspace_path.as_uri(), [
                    (
                        "Mapped the workspace-relative file URL to the host workspace so the browser can read it."
                        if not workspace_url
                        else f"Workspace URL: {workspace_url}"
                    ),
                ]

        mapped_path: Path | None = None
        if self._container_root:
            mapped_path = container_to_host_path(self._workspace_dir, raw_path, self._container_root)
        if mapped_path is not None:
            relative_path = to_workspace_relative(self._workspace_dir, mapped_path)
            workspace_url = self._workspace_file_url(relative_path)
            return mapped_path.as_uri(), [
                (
                    "Mapped the workspace file URL to the host workspace so the browser can read it."
                    if not workspace_url
                    else f"Workspace URL: {workspace_url}"
                ),
            ]

        if raw_path and Path(raw_path).is_absolute() and self._container_root and not Path(raw_path).exists():
            raise RuntimeError(
                "Cannot open that file:// URL directly because the browser runs on the host, "
                "but the path appears to exist only inside the docker sandbox. "
                f"Move the file into `{self._container_root}` first so it can be mapped into the workspace."
            )
        return raw_url, []

    async def _rewrite_localhost_url_locked(self, raw_url: str) -> tuple[str, list[str]]:
        parsed = urlsplit(raw_url)
        if not (
            self._bridge_localhost
            and parsed.scheme in {"http", "https"}
            and _is_local_browser_host(parsed.hostname)
        ):
            return raw_url, []

        target_port = parsed.port or _default_port_for_scheme(parsed.scheme)
        bridge = await self._ensure_local_port_bridge_locked(target_port)
        resolved = urlunsplit(
            (
                parsed.scheme,
                f"{parsed.hostname}:{bridge.listen_port}",
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
        return resolved, [
            f"Bridged docker-local port {target_port} through host port {bridge.listen_port} so the browser can reach the sandbox service.",
        ]

    async def _wait_result_locked(self, page_id: str, page: "Page", arguments: BrowserToolInput) -> AgentToolResult:
        timeout_ms = self._timeout_ms(arguments)
        target = await self._frame_for_arguments_locked(page, arguments)
        if arguments.selector:
            try:
                await target.locator(arguments.selector).first.wait_for(
                    state=arguments.state,
                    timeout=timeout_ms,
                )
            except Exception as exc:
                raise RuntimeError(await self._browser_failure_text(page_id, page, arguments, exc)) from exc
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
            target = await self._frame_for_arguments_locked(page, arguments)
            locator = target.locator(arguments.selector).first
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
            target = await self._frame_for_arguments_locked(page, arguments)
            value = await target.content()
        else:
            target = await self._frame_for_arguments_locked(page, arguments)
            value = await target.evaluate(
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
            target = await self._frame_for_arguments_locked(page, arguments)
            await target.locator(arguments.selector).first.screenshot(**screenshot_kwargs)
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
                preview = await self._try_screenshot_preview_locked(page, arguments, host_path, selector=arguments.selector)
                if preview is not None:
                    preview_path, preview_data = preview
                    lines.append(
                        f"Inline preview generated because the original screenshot is {len(data)} bytes, "
                        f"above the inline limit of {self._inline_image_max_bytes} bytes."
                    )
                    lines.append(f"Preview path: {to_workspace_relative(self._workspace_dir, preview_path)}")
                    if len(preview_data) <= self._inline_image_max_bytes:
                        blocks.append(ImageContent(data=base64.b64encode(preview_data).decode("ascii"), mime_type="image/jpeg"))
                    else:
                        lines.append(
                            f"Preview image is still {len(preview_data)} bytes, so inline image was skipped."
                        )
                else:
                    lines.append(
                        f"Inline image skipped because the screenshot is {len(data)} bytes, "
                        f"above the inline limit of {self._inline_image_max_bytes} bytes."
                    )

        blocks.insert(0, TextContent(text="\n".join(lines)))
        return AgentToolResult(content=blocks)

    async def _screenshot_element_result_locked(
        self,
        page_id: str,
        locator: Any,
        arguments: BrowserToolInput,
        *,
        host_path: Path,
        label: str,
    ) -> AgentToolResult:
        host_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_kwargs: dict[str, Any] = {
            "path": str(host_path),
            "type": arguments.screenshot_format,
            "timeout": self._timeout_ms(arguments),
        }
        if arguments.screenshot_format == "jpeg" and arguments.quality is not None:
            screenshot_kwargs["quality"] = arguments.quality

        await locator.screenshot(**screenshot_kwargs)

        relative_path = to_workspace_relative(self._workspace_dir, host_path)
        lines = [
            f"Saved screenshot for {label} on {page_id}.",
            f"Workspace path: {relative_path}",
        ]
        if self._container_root is not None:
            lines.append(f"Container path: {host_to_container_path(self._workspace_dir, host_path, self._container_root)}")

        blocks: list[Any] = [TextContent(text="\n".join(lines))]
        data = host_path.read_bytes()
        if arguments.return_image and len(data) <= self._inline_image_max_bytes:
            blocks.append(
                ImageContent(
                    data=base64.b64encode(data).decode("ascii"),
                    mime_type="image/jpeg" if arguments.screenshot_format == "jpeg" else "image/png",
                )
            )
        elif arguments.return_image:
            preview_path = host_path.with_name(f"{host_path.stem}-preview.jpeg")
            preview_kwargs: dict[str, Any] = {
                "path": str(preview_path),
                "type": "jpeg",
                "quality": 60,
                "timeout": self._timeout_ms(arguments),
                "scale": "css",
            }
            with suppress(Exception):
                await locator.screenshot(**preview_kwargs)
                preview_data = preview_path.read_bytes()
                lines.append(
                    f"Inline preview generated because the original screenshot is {len(data)} bytes, "
                    f"above the inline limit of {self._inline_image_max_bytes} bytes."
                )
                lines.append(f"Preview path: {to_workspace_relative(self._workspace_dir, preview_path)}")
                blocks[0] = TextContent(text="\n".join(lines))
                if len(preview_data) <= self._inline_image_max_bytes:
                    blocks.append(ImageContent(data=base64.b64encode(preview_data).decode("ascii"), mime_type="image/jpeg"))
        return AgentToolResult(content=blocks)

    async def _try_screenshot_preview_locked(
        self,
        page: "Page",
        arguments: BrowserToolInput,
        original_path: Path,
        *,
        selector: str | None,
    ) -> tuple[Path, bytes] | None:
        preview_path = original_path.with_name(f"{original_path.stem}-preview.jpeg")
        preview_kwargs: dict[str, Any] = {
            "path": str(preview_path),
            "type": "jpeg",
            "quality": 60,
            "timeout": self._timeout_ms(arguments),
            "scale": "css",
        }
        try:
            if selector:
                target = await self._frame_for_arguments_locked(page, arguments)
                await target.locator(selector).first.screenshot(**preview_kwargs)
            else:
                await page.screenshot(**preview_kwargs, full_page=False)
        except Exception:
            return None
        return preview_path, preview_path.read_bytes()

    async def _observe_result_locked(
        self,
        page_id: str,
        page: "Page",
        arguments: BrowserToolInput,
    ) -> AgentToolResult:
        observation = await self._collect_observation_locked(page, max_items=arguments.max_items)
        text = await self._format_observation_text(page_id, page, observation, max_chars=arguments.max_chars)
        return AgentToolResult(content=[TextContent(text=text)])

    async def _collect_observation_locked(self, page: "Page", *, max_items: int) -> list[dict[str, Any]]:
        frames = self._frame_entries(page)
        observed: list[dict[str, Any]] = []
        for frame_id, frame in frames:
            try:
                data = await frame.evaluate(_OBSERVE_SCRIPT, max_items)
            except Exception as exc:
                data = {
                    "url": getattr(frame, "url", ""),
                    "title": "",
                    "text": "",
                    "error": str(exc).strip() or exc.__class__.__name__,
                    "elements": [],
                    "inputs": [],
                    "images": [],
                }
            if not isinstance(data, dict):
                data = {
                    "url": getattr(frame, "url", ""),
                    "title": "",
                    "text": str(data or ""),
                    "elements": [],
                    "inputs": [],
                    "images": [],
                }
            for group_name, prefix in (("elements", "e"), ("inputs", "input"), ("images", "img")):
                for index, item in enumerate(data.get(group_name) or [], start=1):
                    if isinstance(item, dict):
                        item["ref"] = f"{frame_id}:{prefix}{index}"
            observed.append({"frame_id": frame_id, "frame": frame, "data": data})
        return observed

    async def _format_observation_text(
        self,
        page_id: str,
        page: "Page",
        observation: list[dict[str, Any]],
        *,
        max_chars: int,
    ) -> str:
        title = await self._safe_page_title(page)
        lines = [
            f"Page ID: {page_id}",
            f"Title: {title or '(untitled)'}",
            f"URL: {self._display_url(page.url or '(blank)')}",
            "",
            "Frames:",
        ]
        for entry in observation:
            data = entry["data"]
            frame_id = entry["frame_id"]
            frame_url = self._display_url(str(data.get("url") or getattr(entry["frame"], "url", "")))
            marker = " [main]" if frame_id == "f0" else ""
            lines.append(f"- {frame_id}{marker}: {frame_url or '(blank)'}")
            if data.get("error"):
                lines.append(f"  observe error: {data['error']}")

        for entry in observation:
            data = entry["data"]
            frame_id = entry["frame_id"]
            text = str(data.get("text") or "")
            if text:
                lines.extend(["", f"Visible text in {frame_id}:", text])
            self._append_observed_items(lines, f"Clickable elements in {frame_id}", data.get("elements") or [])
            self._append_observed_items(lines, f"Inputs in {frame_id}", data.get("inputs") or [])
            self._append_observed_items(lines, f"Images in {frame_id}", data.get("images") or [])

        rendered = "\n".join(lines)
        truncated, was_truncated = _truncate_text(rendered, max_chars)
        if was_truncated:
            truncated += "\n\n[Observation truncated]"
        return truncated

    def _append_observed_items(self, lines: list[str], heading: str, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        lines.extend(["", f"{heading}:"])
        for item in items:
            rect = item.get("rect") or {}
            label = item.get("text") or item.get("aria") or item.get("id") or item.get("class") or item.get("src") or item.get("href") or ""
            label = str(label).strip()
            tag = item.get("tag") or "element"
            role = f" role={item['role']}" if item.get("role") else ""
            item_type = f" type={item['type']}" if item.get("type") else ""
            lines.append(
                f"- {item.get('ref')}: <{tag}>{role}{item_type} "
                f"{json.dumps(label, ensure_ascii=False)} "
                f"rect=({rect.get('x', 0)},{rect.get('y', 0)},{rect.get('w', 0)},{rect.get('h', 0)}) "
                f"selector={item.get('selector') or ''}"
            )

    async def _resolve_ref_locked(
        self,
        page: "Page",
        ref: str,
        *,
        max_items: int,
    ) -> tuple[str, Any, dict[str, Any]]:
        observation = await self._collect_observation_locked(page, max_items=max_items)
        for entry in observation:
            for group_name in ("elements", "inputs", "images"):
                for item in entry["data"].get(group_name) or []:
                    if item.get("ref") == ref:
                        selector = item.get("selector")
                        if not selector:
                            raise RuntimeError(f"Ref {ref} did not include a usable selector.")
                        return str(entry["frame_id"]), entry["frame"], item
        available: list[str] = []
        for entry in observation:
            for group_name in ("elements", "inputs", "images"):
                available.extend(str(item.get("ref")) for item in entry["data"].get(group_name) or [] if item.get("ref"))
        hint = ", ".join(available[:40]) or "none"
        raise RuntimeError(f"Ref {ref!r} was not found in the current page. Available refs: {hint}")

    async def _frame_for_arguments_locked(self, page: "Page", arguments: BrowserToolInput) -> Any:
        return self._frame_by_hint(page, frame_id=arguments.frame_id, frame_url_contains=arguments.frame_url_contains)

    def _frame_entries(self, page: "Page") -> list[tuple[str, Any]]:
        frames = getattr(page, "frames", None)
        if frames:
            return [(f"f{index}", frame) for index, frame in enumerate(list(frames))]
        return [("f0", page)]

    def _frame_by_hint(self, page: "Page", *, frame_id: str | None, frame_url_contains: str | None) -> Any:
        entries = self._frame_entries(page)
        if frame_id:
            for current_id, frame in entries:
                if current_id == frame_id:
                    return frame
            raise RuntimeError(f"Frame {frame_id!r} was not found. Use observe to list current frames.")
        if frame_url_contains:
            for _, frame in entries:
                if frame_url_contains in str(getattr(frame, "url", "")):
                    return frame
            raise RuntimeError(f"No frame URL contains {frame_url_contains!r}. Use observe to list current frames.")
        return entries[0][1]

    async def _browser_failure_text(
        self,
        page_id: str,
        page: "Page",
        arguments: BrowserToolInput,
        exc: Exception,
    ) -> str:
        lines = [
            f"Browser {arguments.action} failed on {page_id}: {str(exc).strip() or exc.__class__.__name__}",
        ]
        if arguments.selector:
            lines.append(f"Selector: {arguments.selector}")
        if arguments.ref:
            lines.append(f"Ref: {arguments.ref}")
        lines.extend(["", "Current page candidates:"])
        try:
            observation = await self._collect_observation_locked(page, max_items=min(arguments.max_items, 12))
            compact = await self._format_observation_text(page_id, page, observation, max_chars=2_500)
            lines.append(compact)
        except Exception as observe_exc:
            lines.append(f"(Could not observe page after failure: {observe_exc})")
        return "\n".join(lines)

    async def _save_image_result_locked(
        self,
        page_id: str,
        page: "Page",
        arguments: BrowserToolInput,
    ) -> AgentToolResult:
        target = await self._frame_for_arguments_locked(page, arguments)
        selector = arguments.selector
        image_ref = arguments.ref
        if image_ref:
            _, target, item = await self._resolve_ref_locked(page, image_ref, max_items=arguments.max_items)
            selector = str(item.get("selector") or "")

        result = await target.evaluate(
            _SAVE_IMAGE_SCRIPT,
            {"selector": selector, "imageIndex": arguments.image_index},
        )
        if not result:
            raise RuntimeError(await self._browser_failure_text(page_id, page, arguments, RuntimeError("No matching visible image found.")))

        data_url = str(result.get("dataUrl") or "")
        mime_type, payload = self._decode_data_url(data_url)
        extension = "jpg" if mime_type == "image/jpeg" else "svg" if mime_type == "image/svg+xml" else "png"
        host_path = self._resolve_image_path(arguments.path, page_id, extension)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_bytes(payload)

        relative_path = to_workspace_relative(self._workspace_dir, host_path)
        lines = [
            f"Saved image from {image_ref or selector or f'visible image #{arguments.image_index}'} on {page_id}.",
            f"Workspace path: {relative_path}",
            f"MIME type: {mime_type}",
            f"Bytes: {len(payload)}",
        ]
        width = result.get("width")
        height = result.get("height")
        if width or height:
            lines.append(f"Dimensions: {width or '?'}x{height or '?'}")
        if result.get("src"):
            src_preview, _ = _truncate_text(str(result["src"]), 180)
            lines.append(f"Source: {src_preview}")
        if self._container_root is not None:
            lines.append(f"Container path: {host_to_container_path(self._workspace_dir, host_path, self._container_root)}")

        blocks: list[Any] = [TextContent(text="\n".join(lines))]
        if len(payload) <= self._inline_image_max_bytes:
            blocks.append(ImageContent(data=base64.b64encode(payload).decode("ascii"), mime_type=mime_type))
        return AgentToolResult(content=blocks)

    def _decode_data_url(self, data_url: str) -> tuple[str, bytes]:
        if not data_url.startswith("data:") or "," not in data_url:
            raise RuntimeError("Image extraction did not return a data URL.")
        header, encoded = data_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "image/png"
        try:
            if ";base64" in header:
                payload = base64.b64decode(encoded)
            else:
                payload = encoded.encode("utf-8")
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError("Image data URL could not be decoded.") from exc
        return mime_type, payload

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
            "font.language.group": "zh-CN",
            "font.name-list.sans-serif.x-western": ", ".join(_CJK_FONT_FALLBACKS),
            "font.name-list.sans-serif.zh-CN": ", ".join(_CJK_FONT_FALLBACKS),
            "font.name-list.serif.zh-CN": "Songti SC, Noto Serif CJK SC, Noto Serif SC, " + ", ".join(_CJK_FONT_FALLBACKS),
            "font.name.sans-serif.x-western": _CJK_FONT_FALLBACKS[0],
            "font.name.sans-serif.zh-CN": _CJK_FONT_FALLBACKS[0],
            "font.name.serif.zh-CN": "Songti SC",
            "intl.accept_languages": "zh-CN, zh, en-US, en",
            "media.autoplay.default": 5,
        }
        manager = api.async_camoufox_cls(
            persistent_context=True,
            user_data_dir=str(profile_dir),
            headless=self._headless,
            block_images=self._block_images,
            block_webrtc=True,
            enable_cache=False,
            fonts=_CJK_FONT_FALLBACKS,
            firefox_user_prefs=firefox_user_prefs,
            locale=["zh-CN", "zh", "en-US", "en"],
            os=_host_camoufox_os(),
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

    async def _ensure_local_port_bridge_locked(self, target_port: int) -> _LocalPortBridge:
        existing = self._local_port_bridges.get(target_port)
        if existing is not None:
            return existing

        server = await asyncio.start_server(
            lambda reader, writer: self._handle_local_port_bridge_connection(target_port, reader, writer),
            host="127.0.0.1",
            port=0,
        )
        sockets = list(server.sockets or [])
        if not sockets:
            server.close()
            await server.wait_closed()
            raise RuntimeError(f"Failed to create a local port bridge for docker-local port {target_port}.")

        listen_port = int(sockets[0].getsockname()[1])
        bridge = _LocalPortBridge(
            target_port=target_port,
            listen_port=listen_port,
            server=server,
        )
        self._local_port_bridges[target_port] = bridge
        return bridge

    async def _handle_local_port_bridge_connection(
        self,
        target_port: int,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if not self._container_name:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            return

        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            self._container_name,
            "python3",
            "-c",
            _DOCKER_PORT_BRIDGE_SCRIPT,
            str(target_port),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        async def client_to_process() -> None:
            try:
                while True:
                    chunk = await reader.read(65_536)
                    if not chunk:
                        break
                    if process.stdin is None:
                        break
                    process.stdin.write(chunk)
                    await process.stdin.drain()
            finally:
                if process.stdin is not None and not process.stdin.is_closing():
                    process.stdin.close()
                    wait_closed = getattr(process.stdin, "wait_closed", None)
                    if callable(wait_closed):
                        with suppress(Exception):
                            await wait_closed()

        async def process_to_client() -> None:
            if process.stdout is None:
                return
            while True:
                chunk = await process.stdout.read(65_536)
                if not chunk:
                    return
                writer.write(chunk)
                await writer.drain()

        async def drain_stderr() -> None:
            if process.stderr is None:
                return
            while True:
                chunk = await process.stderr.read(65_536)
                if not chunk:
                    return

        client_task = asyncio.create_task(client_to_process())
        process_task = asyncio.create_task(process_to_client())
        stderr_task = asyncio.create_task(drain_stderr())

        try:
            done, _ = await asyncio.wait({client_task, process_task}, return_when=asyncio.FIRST_COMPLETED)
            if process_task in done:
                with suppress(asyncio.CancelledError):
                    client_task.cancel()
                    await client_task
            else:
                await process_task
        finally:
            if process.returncode is None:
                _terminate_subprocess_group(process)
            with suppress(Exception):
                await process.wait()
            with suppress(asyncio.CancelledError):
                stderr_task.cancel()
                await stderr_task
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _close_local_port_bridges_locked(self) -> None:
        bridges = list(self._local_port_bridges.values())
        self._local_port_bridges.clear()
        for bridge in bridges:
            bridge.server.close()
            await bridge.server.wait_closed()

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
        await self._close_local_port_bridges_locked()

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
            lines.append(f"  URL: {self._display_url(page.url or '(blank)')}")
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
            f"URL: {self._display_url(page.url or '(blank)')}",
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

    def _resolve_image_path(
        self,
        raw_path: str | None,
        page_id: str,
        extension: str,
    ) -> Path:
        if raw_path:
            return resolve_workspace_path(
                self._workspace_dir,
                raw_path,
                container_root=self._container_root,
            )
        relative = Path("outbox") / "browser" / f"{_now_timestamp_slug()}-{page_id}-image.{extension}"
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
