from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bampi.plugins.bampi_chat.tools.browser import BrowserTool, BrowserToolInput
from bampi.plugins.bampi_chat.tools.browser.artifacts import ArtifactManager
from bampi.plugins.bampi_chat.tools.browser.commands import BrowserCommandDispatcher, HELP_TEXT, _split
from bampi.plugins.bampi_chat.tools.browser.config import BrowserConfig
from bampi.plugins.bampi_chat.tools.browser.errors import BrowserLaunchError, CommandError
from bampi.plugins.bampi_chat.tools.browser.interaction import InteractionEngine, ResolvedElement
from bampi.plugins.bampi_chat.tools.browser.installer import (
    _select_download,
    chrome_binary_in,
    default_cache_dir,
    find_cached_chrome,
    platform_key,
)
from bampi.plugins.bampi_chat.tools.browser.launcher import chromium_launch_args
from bampi.plugins.bampi_chat.tools.browser.models import PageState
from bampi.plugins.bampi_chat.tools.browser.policy import NavigationPolicy
from bampi.plugins.bampi_chat.tools.browser.stealth import (
    apply_stealth_to_session,
    apply_window_geometry,
    blocked_url_patterns,
    build_stealth_identity,
)


def test_browser_tool_schema_is_one_lightweight_command_field() -> None:
    schema = BrowserToolInput.model_json_schema()

    assert set(schema["properties"]) == {"command"}
    assert schema["required"] == ["command"]
    assert "snapshot" in schema["properties"]["command"]["description"]
    assert "batch" in schema["properties"]["command"]["description"]
    with pytest.raises(ValidationError):
        BrowserToolInput.model_validate({"command": "snapshot", "action": "observe"})


def test_browser_tool_description_exposes_common_capabilities_without_help() -> None:
    description = BrowserTool.description

    for capability in ("navigation", "snapshot", "forms", "drag/drop", "screenshots", "recording", "batch"):
        assert capability in description


@pytest.mark.asyncio
async def test_interaction_box_supports_border_box_for_element_screenshots() -> None:
    class FakeClient:
        async def call(self, method, params=None, *, session_id=None, timeout=20.0):
            del params, session_id, timeout
            if method == "DOM.getBoxModel":
                return {
                    "model": {
                        "content": [24, 20, 776, 20, 776, 180, 24, 180],
                        "border": [0, 0, 800, 0, 800, 200, 0, 200],
                    }
                }
            return {}

    interaction = InteractionEngine(SimpleNamespace(client=FakeClient()))
    element = ResolvedElement("session", 1, "object", "css=body")

    assert await interaction.box(element) == (24, 20, 752, 160)
    assert await interaction.box(element, box_type="border") == (0, 0, 800, 200)


@pytest.mark.asyncio
async def test_target_screenshot_captures_element_border_box(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.capture_params = None

        async def call(self, method, params=None, *, session_id=None, timeout=20.0):
            del session_id, timeout
            if method == "Page.captureScreenshot":
                self.capture_params = params
                return {"data": base64.b64encode(b"png").decode("ascii")}
            return {}

    class FakeInteraction:
        def __init__(self) -> None:
            self.box_type = None

        async def resolve(self, page, target):
            del page, target
            return object()

        async def box(self, element, *, box_type="content"):
            del element
            self.box_type = box_type
            return 0, 0, 800, 200

    client = FakeClient()
    interaction = FakeInteraction()
    runtime = SimpleNamespace(
        client=client,
        config=SimpleNamespace(action_timeout=20.0, inline_image_max_bytes=1_000_000),
        workspace_dir=tmp_path,
        container_root=None,
    )
    artifacts = ArtifactManager(runtime, interaction)
    page = PageState(page_id="p1", target_id="target", session_id="session")

    await artifacts.screenshot(
        page,
        path="body.png",
        target="css=body",
        full_page=True,
        jpeg=False,
        quality=85,
        inline=False,
        annotate=False,
    )

    assert interaction.box_type == "border"
    assert client.capture_params["clip"] == {
        "x": 0,
        "y": 0,
        "width": 800,
        "height": 200,
        "scale": 1,
    }


def test_browser_command_uses_shell_quoting_without_shell_execution() -> None:
    assert _split('fill @e2 "hello world"') == ["fill", "@e2", "hello world"]
    assert _split("eval 'document.title'") == ["eval", "document.title"]
    with pytest.raises(CommandError, match="quoting"):
        _split('fill @e1 "unterminated')


def test_chrome_for_testing_metadata_and_cache_resolution(tmp_path: Path) -> None:
    metadata = {
        "channels": {
            "Stable": {
                "version": "150.0.1.2",
                "downloads": {
                    "chrome": [
                        {"platform": "linux64", "url": "https://example.test/linux.zip"},
                        {"platform": "mac-arm64", "url": "https://example.test/mac.zip"},
                    ]
                },
            }
        }
    }

    assert _select_download(metadata, "mac-arm64") == ("150.0.1.2", "https://example.test/mac.zip")
    with pytest.raises(BrowserLaunchError):
        _select_download({}, "mac-arm64")

    cache = tmp_path / "browsers"
    key = platform_key()
    relative = {
        "linux64": Path("chrome-linux64/chrome"),
        "mac-arm64": Path("chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"),
        "mac-x64": Path("chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"),
        "win64": Path("chrome-win64/chrome.exe"),
    }[key]
    older = cache / "chrome-149.0.0.1" / relative
    newer = cache / "chrome-150.0.0.1" / relative
    for binary in (older, newer):
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("binary", encoding="utf-8")
        binary.chmod(0o755)

    assert chrome_binary_in(cache / "chrome-150.0.0.1", key) == newer
    assert find_cached_chrome(cache) == newer


def test_chrome_for_testing_default_cache_is_project_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BAMPI_BROWSER_CACHE_DIR", raising=False)

    assert default_cache_dir() == tmp_path / ".bampi" / "browser" / "chrome-for-testing"


def test_stealth_identity_reduces_low_entropy_ua_and_keeps_coherent_geometry(tmp_path: Path) -> None:
    identity = build_stealth_identity(
        tmp_path,
        {"product": "HeadlessChrome/150.0.1.2"},
        platform_name="linux",
        env={"TZ": "Asia/Shanghai", "LANG": "zh_CN.UTF-8"},
        viewport_width=1440,
        viewport_height=1000,
    )

    assert "HeadlessChrome" not in identity.user_agent
    assert "Chrome/150.0.0.0" in identity.user_agent
    assert "150.0.1.2" not in identity.user_agent
    assert identity.user_agent_metadata["platform"] == "Linux"
    assert identity.locale == "zh_CN"
    assert identity.languages == ("zh-CN", "zh")
    assert identity.timezone_id == "Asia/Shanghai"
    assert identity.geolocation is not None
    assert identity.screen_width >= identity.window_width >= identity.viewport_width
    assert identity.screen_height >= identity.window_height > identity.viewport_height
    assert identity.window_left + identity.window_width <= identity.screen_width
    assert identity.window_top + identity.window_height <= identity.screen_height
    assert {"brand": "Google Chrome", "version": "150.0.1.2"} in identity.full_version_list
    assert identity.user_agent_metadata["fullVersion"] == "150.0.1.2"


def test_chromium_launch_args_enable_default_stealth(tmp_path: Path) -> None:
    config = BrowserConfig(headless=True)
    identity = build_stealth_identity(
        tmp_path,
        viewport_width=config.viewport_width,
        viewport_height=config.viewport_height,
    )
    args = chromium_launch_args("/opt/chrome", tmp_path / "profile", tmp_path, config)

    assert args[0] == "/opt/chrome"
    assert "--headless=new" in args
    assert "--disable-blink-features=AutomationControlled" in args
    assert any(arg.startswith("--lang=") for arg in args)
    assert f"--window-size={identity.window_width},{identity.window_height}" in args
    assert args[-1] == "about:blank"


def test_blocked_url_patterns_combine_stealth_and_image_blocks() -> None:
    patterns = blocked_url_patterns(BrowserConfig(block_images=True))

    assert "*.png" in patterns
    assert "*://*.fingerprint.com/*" in patterns
    assert len(patterns) == len(set(patterns))


@pytest.mark.asyncio
async def test_apply_stealth_to_session_installs_cdp_overrides(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object], str | None]] = []

        async def call(self, method: str, params=None, *, session_id: str | None = None, timeout: float = 20.0):
            del timeout
            self.calls.append((method, params or {}, session_id))
            return {}

    client = FakeClient()
    identity = build_stealth_identity(tmp_path, {"product": "HeadlessChrome/150.0.1.2"})

    await apply_stealth_to_session(client, "session-1", identity, BrowserConfig(block_images=True))

    methods = [method for method, _, _ in client.calls]
    assert "Emulation.setAutomationOverride" in methods
    assert "Emulation.setUserAgentOverride" in methods
    assert "Emulation.setTimezoneOverride" in methods
    assert "Network.setBlockedURLs" in methods
    assert "Page.addScriptToEvaluateOnNewDocument" not in methods
    assert "Runtime.evaluate" not in methods
    ua_call = next(params for method, params, _ in client.calls if method == "Emulation.setUserAgentOverride")
    assert "HeadlessChrome" not in str(ua_call["userAgent"])
    assert "Chrome/150.0.0.0" in str(ua_call["userAgent"])
    blocked = next(params for method, params, _ in client.calls if method == "Network.setBlockedURLs")
    assert "*.png" in blocked["urls"]
    assert all(session_id == "session-1" for _, _, session_id in client.calls)


@pytest.mark.asyncio
async def test_apply_window_geometry_uses_native_browser_bounds(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call(self, method: str, params=None, **kwargs):
            assert not kwargs
            self.calls.append((method, params or {}))
            return {"windowId": 42} if method == "Browser.getWindowForTarget" else {}

    identity = build_stealth_identity(
        tmp_path,
        {"product": "Chrome/150.0.1.2"},
        viewport_width=1440,
        viewport_height=1000,
    )
    client = FakeClient()

    assert await apply_window_geometry(client, "target-1", identity) is True
    assert client.calls == [
        ("Browser.getWindowForTarget", {"targetId": "target-1"}),
        (
            "Browser.setWindowBounds",
            {"windowId": 42, "bounds": identity.window_bounds_params},
        ),
    ]


@pytest.mark.asyncio
async def test_batch_default_failure_is_a_tool_error() -> None:
    class _Dispatcher(BrowserCommandDispatcher):
        async def _single(self, command: str):
            if command == "fail":
                raise CommandError("expected failure")
            return SimpleNamespace(text=command, image_data=None, image_mime_type=None)

    runtime = SimpleNamespace(config=SimpleNamespace(batch_max_commands=32, batch_timeout=10.0))
    dispatcher = _Dispatcher(runtime, active_page_id=None, cancellation=None)

    with pytest.raises(CommandError, match="step 2/3"):
        await dispatcher.execute("batch\nok\nfail\nskipped")

    result = await dispatcher.execute("batch --continue\nok\nfail\ncontinued")
    assert "with 1 failure" in result.text
    assert "continued" in result.text


@pytest.mark.asyncio
async def test_help_is_available_without_starting_chromium() -> None:
    dispatcher = BrowserCommandDispatcher(
        SimpleNamespace(),
        active_page_id=None,
        cancellation=None,
    )

    result = await dispatcher.execute("help")

    assert result.text == HELP_TEXT
    assert "open URL" in result.text
    assert "batch" in result.text


@pytest.mark.asyncio
async def test_navigation_policy_maps_only_workspace_files(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text("<title>ok</title>", encoding="utf-8")
    bridge = SimpleNamespace()
    policy = NavigationPolicy(
        tmp_path,
        container_root="/workspace",
        bridge=bridge,
        bridge_localhost=False,
        config=BrowserConfig(),
    )

    resolved, notes = await policy.resolve("file:///workspace/page.html")

    assert resolved == page.resolve().as_uri()
    assert notes == ["workspace file: page.html"]
    with pytest.raises(CommandError, match="restricted"):
        await policy.resolve(Path("/etc/hosts").as_uri())


@pytest.mark.asyncio
async def test_navigation_policy_allows_project_localhost_without_dns(tmp_path: Path) -> None:
    policy = NavigationPolicy(
        tmp_path,
        container_root=None,
        bridge=SimpleNamespace(),
        bridge_localhost=False,
        config=BrowserConfig(),
    )

    resolved, notes = await policy.resolve("http://127.0.0.1:3000/app")

    assert resolved == "http://127.0.0.1:3000/app"
    assert notes == []


@pytest.mark.asyncio
async def test_navigation_policy_blocks_direct_metadata_address(tmp_path: Path) -> None:
    policy = NavigationPolicy(
        tmp_path,
        container_root=None,
        bridge=SimpleNamespace(),
        bridge_localhost=False,
        config=BrowserConfig(),
    )

    with pytest.raises(CommandError, match="blocked"):
        await policy.resolve("http://169.254.169.254/latest/meta-data")
