from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from bampi.plugins.bampi_chat.tools.browser import BrowserTool, BrowserToolInput, _BrowserRuntime, _LocalPortBridge


class FakeManager:
    def __init__(self) -> None:
        self.exited = False

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class FakeKeyboard:
    def __init__(self) -> None:
        self.typed: list[str] = []
        self.pressed: list[str] = []

    async def type(self, text: str) -> None:
        self.typed.append(text)

    async def press(self, keys: str) -> None:
        self.pressed.append(keys)


class FakeLocator:
    def __init__(self, page: "FakePage", selector: str) -> None:
        self._page = page
        self._selector = selector

    @property
    def first(self) -> "FakeLocator":
        return self

    async def click(self, **kwargs) -> None:
        self._page.actions.append(("click", self._selector, kwargs))

    async def fill(self, text: str, timeout: int | None = None) -> None:
        self._page.fills[self._selector] = (text, timeout)

    async def press(self, keys: str, timeout: int | None = None) -> None:
        self._page.actions.append(("press", self._selector, {"keys": keys, "timeout": timeout}))

    async def wait_for(self, state: str = "visible", timeout: int | None = None) -> None:
        self._page.actions.append(("wait_for", self._selector, {"state": state, "timeout": timeout}))

    async def inner_text(self, timeout: int | None = None) -> str:
        del timeout
        return f"text for {self._selector}"

    async def evaluate(self, script: str, timeout: int | None = None) -> str:
        del script, timeout
        return f"<div>{self._selector}</div>"

    async def screenshot(self, *, path: str, **kwargs) -> None:
        del kwargs
        Path(path).write_bytes(b"fake-locator-image")


class FakePage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self._title = "Blank"
        self.closed = False
        self.actions: list[tuple[str, str, object]] = []
        self.fills: dict[str, tuple[str, int | None]] = {}
        self.keyboard = FakeKeyboard()

    async def goto(self, url: str, wait_until: str, timeout: int) -> None:
        self.url = url
        self._title = "Example Domain"
        self.actions.append(("goto", url, {"wait_until": wait_until, "timeout": timeout}))

    async def title(self) -> str:
        return self._title

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self, selector)

    async def wait_for_load_state(self, state: str, timeout: int | None = None) -> None:
        self.actions.append(("wait_for_load_state", state, {"timeout": timeout}))

    async def evaluate(self, script: str, arg=None):
        if script == "document.readyState":
            return "complete"
        if "document.body" in script:
            return "Hello from the fake page.\nRendered content is here."
        return {"script": script, "arg": arg}

    async def content(self) -> str:
        return "<html><body>Hello from the fake page.</body></html>"

    async def screenshot(self, *, path: str, **kwargs) -> None:
        del kwargs
        Path(path).write_bytes(b"fake-page-image")

    async def reload(self, *, wait_until: str, timeout: int) -> None:
        self.actions.append(("reload", wait_until, {"timeout": timeout}))

    async def go_back(self, *, wait_until: str, timeout: int) -> None:
        self.actions.append(("back", wait_until, {"timeout": timeout}))

    async def go_forward(self, *, wait_until: str, timeout: int) -> None:
        self.actions.append(("forward", wait_until, {"timeout": timeout}))

    async def close(self) -> None:
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed


class FakeContext:
    def __init__(self) -> None:
        self.pages: list[FakePage] = []

    async def new_page(self) -> FakePage:
        page = FakePage()
        self.pages.append(page)
        return page


def test_browser_tool_input_validation():
    with pytest.raises(ValidationError, match="open requires url"):
        BrowserToolInput(action="open")
    with pytest.raises(ValidationError, match="wait requires selector"):
        BrowserToolInput(action="wait")


def test_browser_tool_input_accepts_null_for_defaulted_non_nullable_fields():
    arguments = BrowserToolInput.model_validate(
        {
            "action": "open",
            "url": "https://example.com",
            "wait_until": None,
            "state": None,
            "click_count": None,
            "return_image": None,
        }
    )

    assert arguments.wait_until == "domcontentloaded"
    assert arguments.state == "visible"
    assert arguments.click_count == 1
    assert arguments.return_image is True


@pytest.mark.asyncio
async def test_browser_tool_open_and_screenshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tool = BrowserTool(str(tmp_path), container_root="/workspace")
    context = FakeContext()
    manager = FakeManager()

    async def fake_ensure_context_locked() -> FakeContext:
        if tool._runtime is None:
            tool._runtime = _BrowserRuntime(
                manager=manager,
                context=context,
                profile_dir=tmp_path / ".browser" / "camoufox-profile",
                launched_at=time.monotonic(),
                last_used_at=time.monotonic(),
            )
        return context

    monkeypatch.setattr(tool, "_ensure_context_locked", fake_ensure_context_locked)

    open_result = await tool.execute("call-1", {"action": "open", "url": "https://example.com"})
    assert "Page ID: p1" in open_result.content[0].text
    assert "https://example.com" in open_result.content[0].text

    screenshot_result = await tool.execute(
        "call-2",
        {"action": "screenshot", "page_id": "p1", "return_image": False},
    )
    assert "Workspace path: outbox/browser/" in screenshot_result.content[0].text

    output_files = list((tmp_path / "outbox" / "browser").iterdir())
    assert len(output_files) == 1
    assert output_files[0].read_bytes() == b"fake-page-image"


@pytest.mark.asyncio
async def test_browser_tool_pages_and_reset_clear_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tool = BrowserTool(str(tmp_path))
    context = FakeContext()
    manager = FakeManager()
    profile_dir = tmp_path / ".browser" / "camoufox-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "state.txt").write_text("persisted", encoding="utf-8")

    async def fake_ensure_context_locked() -> FakeContext:
        if tool._runtime is None:
            tool._runtime = _BrowserRuntime(
                manager=manager,
                context=context,
                profile_dir=profile_dir,
                launched_at=time.monotonic(),
                last_used_at=time.monotonic(),
            )
        return context

    monkeypatch.setattr(tool, "_ensure_context_locked", fake_ensure_context_locked)

    await tool.execute("call-1", {"action": "open", "url": "https://example.com"})
    pages_result = await tool.execute("call-2", {"action": "pages"})
    assert "Open pages (1/6)" in pages_result.content[0].text
    assert "p1 [active]" in pages_result.content[0].text

    reset_result = await tool.execute("call-3", {"action": "reset", "clear_profile": True})
    assert "Browser session reset." in reset_result.content[0].text
    assert manager.exited is True
    assert not profile_dir.exists()


@pytest.mark.asyncio
async def test_browser_tool_maps_container_workspace_file_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tool = BrowserTool(str(tmp_path), container_root="/workspace/group-demo")
    context = FakeContext()
    manager = FakeManager()

    async def fake_ensure_context_locked() -> FakeContext:
        if tool._runtime is None:
            tool._runtime = _BrowserRuntime(
                manager=manager,
                context=context,
                profile_dir=tmp_path / ".browser" / "camoufox-profile",
                launched_at=time.monotonic(),
                last_used_at=time.monotonic(),
            )
        return context

    monkeypatch.setattr(tool, "_ensure_context_locked", fake_ensure_context_locked)

    result = await tool.execute(
        "call-1",
        {"action": "open", "url": "file:///workspace/group-demo/math_solution.html"},
    )

    assert "Requested URL: file:///workspace/group-demo/math_solution.html" in result.content[0].text
    assert "Resolved URL:" in result.content[0].text
    assert context.pages[0].actions[0][1] == (tmp_path / "math_solution.html").resolve().as_uri()


@pytest.mark.asyncio
async def test_browser_tool_bridges_docker_localhost_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tool = BrowserTool(
        str(tmp_path),
        container_root="/workspace/group-demo",
        container_name="bampi-sandbox",
        bridge_localhost=True,
    )
    context = FakeContext()
    manager = FakeManager()

    async def fake_ensure_context_locked() -> FakeContext:
        if tool._runtime is None:
            tool._runtime = _BrowserRuntime(
                manager=manager,
                context=context,
                profile_dir=tmp_path / ".browser" / "camoufox-profile",
                launched_at=time.monotonic(),
                last_used_at=time.monotonic(),
            )
        return context

    async def fake_bridge(target_port: int) -> _LocalPortBridge:
        assert target_port == 8889
        return _LocalPortBridge(target_port=8889, listen_port=40123, server=None)

    monkeypatch.setattr(tool, "_ensure_context_locked", fake_ensure_context_locked)
    monkeypatch.setattr(tool, "_ensure_local_port_bridge_locked", fake_bridge)

    result = await tool.execute(
        "call-1",
        {"action": "open", "url": "http://localhost:8889/math_render.html"},
    )

    assert "Requested URL: http://localhost:8889/math_render.html" in result.content[0].text
    assert "Bridged docker-local port 8889 through host port 40123" in result.content[0].text
    assert context.pages[0].actions[0][1] == "http://localhost:40123/math_render.html"
