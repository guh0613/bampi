from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat import prompt as prompt_module
from bampi.plugins.bampi_chat.prompt import build_system_prompt
from bampi.plugins.bampi_chat.tools import create_agent_tools
from bampi.plugins.bampi_chat.tools.files import WorkspaceEditTool, WorkspacePatchTool, WorkspaceReadTool, WorkspaceWriteTool
from bampi.plugins.bampi_chat.tools.safe_bash import SafeBashTool
from bampi.plugins.bampi_chat.tools import web_search as web_search_module


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
    await editor.execute("call-2", {"path": "notes.txt", "edits": [{"old_text": "beta", "new_text": "BETA"}]})
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


@pytest.mark.asyncio
async def test_workspace_patch_tool_accepts_container_absolute_paths(tmp_path: Path):
    file_path = tmp_path / "report.txt"
    file_path.write_text("alpha\nbeta\n", encoding="utf-8")
    patcher = WorkspacePatchTool(str(tmp_path), container_root="/workspace")

    await patcher.execute(
        "call-1",
        {
            "patch": (
                "--- /workspace/report.txt\n"
                "+++ /workspace/report.txt\n"
                "@@ -1,2 +1,2 @@\n"
                " alpha\n"
                "-beta\n"
                "+BETA\n"
            )
        },
    )

    assert file_path.read_text(encoding="utf-8") == "alpha\nBETA\n"


def test_bampi_chat_defaults_use_docker_sandbox():
    config = BampiChatConfig()

    assert config.bampi_bash_mode == "docker"
    assert config.bampi_bash_container_name == "bampi-sandbox"
    assert config.bampi_bash_container_workdir == "/workspace"
    assert config.bampi_bash_container_shell == "/bin/bash"
    assert config.bampi_group_whitelist == []


def test_bampi_chat_group_whitelist_normalizes_entries():
    config = BampiChatConfig(bampi_group_whitelist=[" 1001 ", 1002, "", "  "])

    assert config.bampi_group_whitelist == ["1001", "1002"]


def test_system_prompt_mentions_docker_workspace():
    prompt = build_system_prompt(BampiChatConfig(), ["bash", "read"])

    assert "/workspace" in prompt
    assert "常用开发环境" in prompt
    assert "Noto Sans CJK SC" in prompt
    assert "WenQuanYi Zen Hei" in prompt


def test_system_prompt_mentions_browser_tool():
    prompt = build_system_prompt(BampiChatConfig(), ["browser", "web_search"])

    assert "browser" in prompt
    assert "outbox/browser/" in prompt
    assert "只有 `outbox/` 根目录的新文件会自动发回群里" in prompt
    assert "path` 设为 `outbox/xxx.png`" in prompt
    assert "点击 ref 用 `click_ref`" in prompt


def test_system_prompt_mentions_background_bash_sessions():
    prompt = build_system_prompt(BampiChatConfig(), ["bash"])

    assert "后台会话" in prompt
    assert "start" in prompt
    assert "notify_on_exit" in prompt


def test_system_prompt_uses_utc_plus_8_time_to_minute(monkeypatch: pytest.MonkeyPatch):
    class _FakeDatetime:
        @classmethod
        def now(cls, tz):
            assert tz.utcoffset(None) == timedelta(hours=8)
            return datetime(2026, 4, 1, 0, 30, tzinfo=tz)

    monkeypatch.setattr(prompt_module, "datetime", _FakeDatetime)

    prompt = build_system_prompt(BampiChatConfig(), ["bash", "read"])

    assert "当前时间(UTC+8): 2026-04-01 00:30" in prompt


def test_create_agent_tools_includes_browser_by_default(tmp_path: Path):
    tools = create_agent_tools(BampiChatConfig(), str(tmp_path), container_root="/workspace")
    tool_names = [tool.name for tool in tools]

    assert "browser" in tool_names
    assert "ls" not in tool_names


@pytest.mark.asyncio
async def test_create_agent_tools_web_search_uses_dedicated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}
        text = ""

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"Search answer"}}]}'
            yield "data: [DONE]"

    class _FakeStreamContext:
        async def __aenter__(self) -> _FakeResponse:
            return _FakeResponse()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class _FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            state["client_kwargs"] = kwargs

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, object]) -> _FakeStreamContext:
            state["request"] = {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
            }
            return _FakeStreamContext()

    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", _FakeAsyncClient)

    tools = create_agent_tools(
        BampiChatConfig(
            bampi_api_key="model-secret",
            bampi_base_url="https://model.example.com",
            bampi_web_search_api_key="search-secret",
            bampi_web_search_base_url="https://search.example.com",
        ),
        str(tmp_path),
        container_root="/workspace",
    )
    web_search_tool = next(tool for tool in tools if getattr(tool, "name", None) == "web_search")

    result = await web_search_tool.execute("call-1", {"query": "最新模型信息"})

    assert result.content[0].text == "Search answer"
    assert state["client_kwargs"] == {"timeout": 15.0}
    assert state["request"] == {
        "method": "POST",
        "url": "https://search.example.com/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer search-secret",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": web_search_module.DEFAULT_WEB_SEARCH_USER_AGENT,
        },
        "json": {
            "model": web_search_module.DEFAULT_WEB_SEARCH_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "最新模型信息",
                },
            ],
        },
    }


def test_safe_bash_tool_uses_container_bash_shell(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHELL", "/bin/zsh")
    tool = SafeBashTool(
        workspace_dir="/tmp/workspace",
        mode="docker",
        container_name="bampi-sandbox",
        container_workdir="/workspace",
        visible_workspace_root="/workspace",
        container_shell="/bin/bash",
        default_timeout=30.0,
    )

    assert tool._docker_command("pwd")[-3:] == ["/bin/bash", "-lc", "pwd"]
    assert tool._local_command("pwd") == ["/bin/zsh", "-lc", "pwd"]


@pytest.mark.asyncio
async def test_safe_bash_background_session_lifecycle(tmp_path: Path):
    tool = SafeBashTool(
        workspace_dir=str(tmp_path),
        mode="local",
        container_name="bampi-sandbox",
        container_workdir="/workspace",
        visible_workspace_root="/workspace",
        container_shell="/bin/bash",
        default_timeout=30.0,
    )

    start_result = await tool.execute(
        "call-1",
        {
            "action": "start",
            "command": "python3 -u -c 'import time; print(\"ready\", flush=True); time.sleep(60)'",
        },
    )
    session_id = start_result.details["session_id"]
    assert session_id == "term-1"

    logs_result = None
    for _ in range(40):
        logs_result = await tool.execute("call-2", {"action": "logs", "session_id": session_id})
        if "ready" in logs_result.content[0].text:
            break
        await asyncio.sleep(0.05)

    assert logs_result is not None
    assert "ready" in logs_result.content[0].text

    status_result = await tool.execute("call-3", {"action": "status", "session_id": session_id})
    assert f"Background session `{session_id}` is running." in status_result.content[0].text
    assert "Working directory: /workspace" in status_result.content[0].text

    stop_result = await tool.execute("call-4", {"action": "stop", "session_id": session_id})
    assert f"Background session `{session_id}` stopped." in stop_result.content[0].text


@pytest.mark.asyncio
async def test_safe_bash_background_session_accepts_input(tmp_path: Path):
    tool = SafeBashTool(
        workspace_dir=str(tmp_path),
        mode="local",
        container_name="bampi-sandbox",
        container_workdir="/workspace",
        visible_workspace_root="/workspace",
        container_shell="/bin/bash",
        default_timeout=30.0,
    )

    start_result = await tool.execute(
        "call-1",
        {
            "action": "start",
            "command": (
                "python3 -u -c 'import sys; print(\"boot\", flush=True); "
                "line = sys.stdin.readline().strip(); print(f\"echo:{line}\", flush=True)'"
            ),
        },
    )
    session_id = start_result.details["session_id"]

    await tool.execute("call-2", {"action": "input", "session_id": session_id, "stdin": "hello\n"})

    logs_result = None
    for _ in range(40):
        logs_result = await tool.execute("call-3", {"action": "logs", "session_id": session_id})
        if "echo:hello" in logs_result.content[0].text:
            break
        await asyncio.sleep(0.05)

    assert logs_result is not None
    assert "echo:hello" in logs_result.content[0].text


@pytest.mark.asyncio
async def test_safe_bash_background_session_notifies_exit_listener(tmp_path: Path):
    tool = SafeBashTool(
        workspace_dir=str(tmp_path),
        mode="local",
        container_name="bampi-sandbox",
        container_workdir="/workspace",
        visible_workspace_root="/workspace",
        container_shell="/bin/bash",
        default_timeout=30.0,
    )
    events = []
    tool.add_exit_listener(lambda event: events.append(event))

    start_result = await tool.execute(
        "call-1",
        {
            "action": "start",
            "command": "python3 -u -c 'print(\"done\", flush=True)'",
            "notify_on_exit": True,
        },
    )

    try:
        for _ in range(40):
            if events:
                break
            await asyncio.sleep(0.05)

        assert start_result.details["notify_on_exit"] is True
        assert len(events) == 1
        assert events[0].session_id == start_result.details["session_id"]
        assert events[0].notify_on_exit is True
        assert "done" in events[0].output_text
    finally:
        await tool.close()


@pytest.mark.asyncio
async def test_safe_bash_rewrites_visible_workspace_root_and_sanitizes_output(tmp_path: Path):
    tool = SafeBashTool(
        workspace_dir=str(tmp_path),
        mode="local",
        container_name="bampi-sandbox",
        container_workdir="/workspace/group-1001",
        visible_workspace_root="/workspace",
        container_shell="/bin/bash",
        default_timeout=30.0,
    )

    result = await tool.execute("call-1", {"action": "run", "command": "pwd"})

    assert result.content[0].text.strip() == "/workspace"


def test_web_search_normalize_base_url_adds_v1():
    assert web_search_module._normalize_base_url("https://api.example.com") == "https://api.example.com/v1"
    assert web_search_module._normalize_base_url("https://api.example.com/v1/") == "https://api.example.com/v1"


@pytest.mark.asyncio
async def test_web_search_tool_uses_chat_completions(monkeypatch: pytest.MonkeyPatch):
    state: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}
        text = ""

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            lines = [
                'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
                'data: {"choices":[{"delta":{"content":"<think>Thinking about your request\\n[WebSearch] example query\\nbrowse_page {\\"url\\":\\"https://example.com/page\\",\\"instructions\\":\\"Summarize\\"}\\nCalling web_search tool to fetch current details.\\n</think>Summary: latest info\\nSources:\\n- Example | https://example.com"}}]}',
                "data: [DONE]",
            ]
            for line in lines:
                yield line

    class _FakeStreamContext:
        async def __aenter__(self) -> _FakeResponse:
            state["stream_entered"] = True
            return _FakeResponse()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            state["stream_exited"] = True

    class _FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            state["client_kwargs"] = kwargs

        async def __aenter__(self) -> "_FakeAsyncClient":
            state["client_entered"] = True
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            state["client_exited"] = True

        def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, object]) -> _FakeStreamContext:
            state["request"] = {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
            }
            return _FakeStreamContext()

    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", _FakeAsyncClient)

    tool = web_search_module.create_web_search_tool(
        12.0,
        base_url="https://api.example.com",
        api_key="secret",
    )
    result = await tool.execute("call-1", {"query": "最新模型信息"})

    assert result.content[0].text == "Summary: latest info\nSources:\n- Example | https://example.com"
    assert state["client_kwargs"] == {
        "timeout": 12.0,
    }
    assert state["request"] == {
        "method": "POST",
        "url": "https://api.example.com/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": web_search_module.DEFAULT_WEB_SEARCH_USER_AGENT,
        },
        "json": {
            "model": web_search_module.DEFAULT_WEB_SEARCH_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "最新模型信息",
                },
            ],
        },
    }
    assert state["client_entered"] is True
    assert state["client_exited"] is True
    assert state["stream_entered"] is True
    assert state["stream_exited"] is True


@pytest.mark.asyncio
async def test_web_search_reads_sse_and_compacts_thinking_trace():
    class _FakeResponse:
        async def aiter_lines(self):
            lines = [
                'data: {"choices":[{"delta":{"content":"<think>Thinking...\\n[WebSearch] OpenAI latest models 2026\\n- Planning to search for \\"OpenAI latest models\\"\\nbrowse_page {\\"url\\":\\"https://developers.openai.com/api/docs/models\\",\\"instructions\\":\\"Summarize\\"}\\nCalling web_search tool to fetch current OpenAI model details.\\n</think>"}}]}',
                'data: {"choices":[{"delta":{"content":"Answer with source"}}]}',
                "data: [DONE]",
            ]
            for line in lines:
                yield line

    text = await web_search_module._read_sse_text(_FakeResponse())

    assert web_search_module._compact_response_text(text) == "Answer with source"


@pytest.mark.asyncio
async def test_web_search_tool_reports_configuration_errors():
    tool = web_search_module.create_web_search_tool(
        12.0,
        base_url="",
        api_key="secret",
    )

    result = await tool.execute("call-1", {"query": "最新模型信息"})

    assert result.content[0].text == (
        "Web search failed for: 最新模型信息\n"
        "Error: web_search is not configured: bampi_web_search_base_url is empty"
    )
