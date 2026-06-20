from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bampi.plugins.bampi_chat.tools.browser import BrowserTool, BrowserToolInput
from bampi.plugins.bampi_chat.tools.browser.commands import BrowserCommandDispatcher, HELP_TEXT, _split
from bampi.plugins.bampi_chat.tools.browser.config import BrowserConfig
from bampi.plugins.bampi_chat.tools.browser.errors import CommandError
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
