from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bampi.plugins.bampi_chat.tools.browser import BrowserTool, BrowserToolInput
from bampi.plugins.bampi_chat.tools.browser.commands import BrowserCommandDispatcher, HELP_TEXT, _split
from bampi.plugins.bampi_chat.tools.browser.config import BrowserConfig
from bampi.plugins.bampi_chat.tools.browser.errors import BrowserLaunchError, CommandError
from bampi.plugins.bampi_chat.tools.browser.installer import (
    _select_download,
    chrome_binary_in,
    default_cache_dir,
    find_cached_chrome,
    platform_key,
)
from bampi.plugins.bampi_chat.tools.browser.policy import NavigationPolicy


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
