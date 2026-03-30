from __future__ import annotations

from pathlib import Path

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.prompt import build_system_prompt
from bampi.plugins.bampi_chat.tools.files import WorkspaceEditTool, WorkspaceReadTool, WorkspaceWriteTool
from bampi.plugins.bampi_chat.tools.safe_bash import SafeBashTool
from bampi.plugins.bampi_chat.tools.web_search import parse_duckduckgo_results


@pytest.mark.asyncio
async def test_workspace_tools_restrict_path_escape(tmp_path: Path):
    tool = WorkspaceWriteTool(str(tmp_path))
    with pytest.raises(ValueError, match="escapes workspace"):
        await tool.execute("call-1", {"path": "../oops.txt", "content": "x"})


@pytest.mark.asyncio
async def test_workspace_tools_round_trip(tmp_path: Path):
    writer = WorkspaceWriteTool(str(tmp_path))
    reader = WorkspaceReadTool(str(tmp_path))
    editor = WorkspaceEditTool(str(tmp_path))

    await writer.execute("call-1", {"path": "notes.txt", "content": "alpha\nbeta\n"})
    await editor.execute("call-2", {"path": "notes.txt", "old_text": "beta", "new_text": "BETA"})
    result = await reader.execute("call-3", {"path": "notes.txt"})

    assert "BETA" in result.content[0].text


@pytest.mark.asyncio
async def test_workspace_tools_accept_container_absolute_path(tmp_path: Path):
    writer = WorkspaceWriteTool(str(tmp_path), container_root="/workspace")
    reader = WorkspaceReadTool(str(tmp_path), container_root="/workspace")

    await writer.execute("call-1", {"path": "/workspace/report.txt", "content": "from container path\n"})
    result = await reader.execute("call-2", {"path": "/workspace/report.txt"})

    assert result.content[0].text == "from container path\n"
    assert (tmp_path / "report.txt").read_text(encoding="utf-8") == "from container path\n"


def test_bampi_chat_defaults_use_docker_sandbox():
    config = BampiChatConfig()

    assert config.bampi_bash_mode == "docker"
    assert config.bampi_bash_container_name == "bampi-sandbox"
    assert config.bampi_bash_container_workdir == "/workspace"
    assert config.bampi_bash_container_shell == "/bin/bash"


def test_system_prompt_mentions_docker_workspace():
    prompt = build_system_prompt(BampiChatConfig(), ["bash", "read"])

    assert "Docker 容器" in prompt
    assert "/workspace" in prompt


def test_safe_bash_tool_uses_container_bash_shell(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHELL", "/bin/zsh")
    tool = SafeBashTool(
        workspace_dir="/tmp/workspace",
        mode="docker",
        container_name="bampi-sandbox",
        container_workdir="/workspace",
        container_shell="/bin/bash",
        default_timeout=30.0,
    )

    assert tool._docker_command("pwd")[-3:] == ["/bin/bash", "-lc", "pwd"]
    assert tool._local_command("pwd") == ["/bin/zsh", "-lc", "pwd"]


def test_parse_duckduckgo_results_extracts_links():
    html = """
    <html>
      <body>
        <a href="https://example.com/a">Result A</a>
        <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb">Result B</a>
        <a href="https://duckduckgo.com/about">Ignore Me</a>
      </body>
    </html>
    """
    results = parse_duckduckgo_results(html, 5)
    assert [item.title for item in results] == ["Result A", "Result B"]
    assert [item.url for item in results] == ["https://example.com/a", "https://example.com/b"]
