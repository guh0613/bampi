from __future__ import annotations

import importlib
import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from bampy.ai import AssistantMessage, TextContent

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.handler import ResponseDispatchResult
from bampi.plugins.bampi_chat.prompt import build_system_prompt
from bampi.plugins.bampi_chat.schedule_manager import ScheduleManager
from bampi.plugins.bampi_chat.tools import create_agent_tools

handler_module = importlib.import_module("bampi.plugins.bampi_chat.handler")
schedule_manager_module = importlib.import_module("bampi.plugins.bampi_chat.schedule_manager")


@dataclass
class FakeSharedSession:
    messages: list[object]
    prompt_calls: list[tuple[object, str]]
    session_manager: object

    def __init__(self) -> None:
        self.messages = [AssistantMessage(content=[TextContent(text="定时任务执行完成。")])]
        self.prompt_calls = []
        self.session_manager = SimpleNamespace(leaf_id="leaf-1")

    async def prompt(self, user_message, source: str) -> None:  # noqa: ANN001
        self.prompt_calls.append((user_message, source))


class FakeGroupSessionManager:
    def __init__(self, workspace_dir: Path) -> None:
        self._workspace_dir = workspace_dir
        self.active_status = SimpleNamespace(is_active=False)

    def workspace_dir_for_group(self, group_id: str) -> str:
        path = self._workspace_dir / group_id
        (path / "outbox").mkdir(parents=True, exist_ok=True)
        return str(path)

    async def inspect_interaction(self, group_id: str):  # noqa: ANN001
        return self.active_status


def test_system_prompt_mentions_schedule_tool():
    prompt = build_system_prompt(BampiChatConfig(), ["schedule"])

    assert "schedule" in prompt
    assert "定时" in prompt


def test_create_agent_tools_includes_schedule_when_manager_is_provided(tmp_path: Path):
    fake_group_manager = FakeGroupSessionManager(tmp_path / "workspace")
    schedule_manager = ScheduleManager(
        config=BampiChatConfig(bampi_schedule_dir=str(tmp_path / "schedules")),
        group_session_manager=fake_group_manager,
    )

    tools = create_agent_tools(
        BampiChatConfig(),
        str(tmp_path / "group-workspace"),
        container_root="/workspace",
        bash_workdir="/workspace/group-1001",
        group_id="1001",
        schedule_manager=schedule_manager,
    )

    assert "schedule" in [tool.name for tool in tools]


@pytest.mark.asyncio
async def test_schedule_manager_runs_task_in_shared_session_and_marks_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_group_manager = FakeGroupSessionManager(tmp_path / "workspace")
    manager = ScheduleManager(
        config=BampiChatConfig(bampi_schedule_dir=str(tmp_path / "schedules")),
        group_session_manager=fake_group_manager,
    )

    sent_payloads: list[dict[str, object]] = []
    bot_calls: list[tuple[str, dict[str, object]]] = []

    async def fake_send_background_agent_response(**kwargs):  # noqa: ANN003
        sent_payloads.append(kwargs)
        return ResponseDispatchResult(delivered=True, rollback_context=False)

    class FakeBot:
        async def call_api(self, action: str, **params: object) -> dict[str, object]:
            bot_calls.append((action, params))
            return {}

    monkeypatch.setattr(schedule_manager_module, "get_bots", lambda: {"bot": FakeBot()})
    monkeypatch.setattr(
        handler_module,
        "send_background_agent_response",
        fake_send_background_agent_response,
    )

    record = await manager.create_task(
        group_id="1001",
        name="morning-report",
        prompt="读取 workspace 中的日报模板并生成今天的简报。",
        trigger_type="date",
        timezone="Asia/Shanghai",
        run_at="2099-01-01 09:00",
        cron=None,
        replace_existing=False,
    )

    managed = SimpleNamespace(
        lock=asyncio.Lock(),
        session=FakeSharedSession(),
        last_used_at=0.0,
    )
    pending = schedule_manager_module.PendingScheduledRun(
        task_id=record.task_id,
        trigger_source="manual",
        scheduled_for="2099-01-01T09:00:00+08:00",
    )

    await manager._run_task(managed, pending)
    updated = await manager.get_task(group_id="1001", task_ref=record.task_id)

    assert updated.state == "completed"
    assert updated.run_count == 1
    assert updated.last_run_status == "success"
    assert updated.is_running is False
    session = managed.session
    assert isinstance(session, FakeSharedSession)
    assert len(session.prompt_calls) == 1
    user_message, source = session.prompt_calls[0]
    assert source == "scheduled_task"
    assert "scheduled_task_prompt:" in user_message.content[0].text
    assert "读取 workspace 中的日报模板" in user_message.content[0].text
    assert len(sent_payloads) == 1
    assert "text_prefix" not in sent_payloads[0]
    assert len(bot_calls) == 1
    assert bot_calls[0][0] == "send_group_msg"


@pytest.mark.asyncio
async def test_schedule_manager_list_hides_completed_one_time_tasks_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_group_manager = FakeGroupSessionManager(tmp_path / "workspace")
    manager = ScheduleManager(
        config=BampiChatConfig(bampi_schedule_dir=str(tmp_path / "schedules")),
        group_session_manager=fake_group_manager,
    )

    async def fake_send_background_agent_response(**kwargs):  # noqa: ANN003
        return ResponseDispatchResult(delivered=True, rollback_context=False)

    class FakeBot:
        async def call_api(self, action: str, **params: object) -> dict[str, object]:
            return {}

    monkeypatch.setattr(schedule_manager_module, "get_bots", lambda: {"bot": FakeBot()})
    monkeypatch.setattr(
        handler_module,
        "send_background_agent_response",
        fake_send_background_agent_response,
    )

    one_time = await manager.create_task(
        group_id="1001",
        name="one-shot",
        prompt="执行一次的任务。",
        trigger_type="date",
        timezone="Asia/Shanghai",
        run_at="2099-01-01 09:00",
        cron=None,
        replace_existing=False,
    )
    recurring = await manager.create_task(
        group_id="1001",
        name="weekly-report",
        prompt="每周生成一次报告。",
        trigger_type="cron",
        timezone="Asia/Shanghai",
        run_at=None,
        cron="0 9 * * 1",
        replace_existing=False,
    )

    managed = SimpleNamespace(
        lock=asyncio.Lock(),
        session=FakeSharedSession(),
        last_used_at=0.0,
    )
    pending = schedule_manager_module.PendingScheduledRun(
        task_id=one_time.task_id,
        trigger_source="manual",
        scheduled_for="2099-01-01T09:00:00+08:00",
    )

    await manager._run_task(managed, pending)

    active_records = await manager.list_tasks(group_id="1001")
    all_records = await manager.list_tasks(group_id="1001", include_inactive=True)

    assert [record.task_id for record in active_records] == [recurring.task_id]
    assert {record.task_id for record in all_records} == {one_time.task_id, recurring.task_id}
    assert "No active scheduled tasks in this group." == manager.render_task_list([])


@pytest.mark.asyncio
async def test_schedule_manager_sends_queue_notice_when_group_is_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_group_manager = FakeGroupSessionManager(tmp_path / "workspace")
    fake_group_manager.active_status = SimpleNamespace(is_active=True)
    manager = ScheduleManager(
        config=BampiChatConfig(bampi_schedule_dir=str(tmp_path / "schedules")),
        group_session_manager=fake_group_manager,
    )

    bot_calls: list[tuple[str, dict[str, object]]] = []

    class FakeBot:
        async def call_api(self, action: str, **params: object) -> dict[str, object]:
            bot_calls.append((action, params))
            return {}

    async def fake_drain_group_queue(group_id: str) -> None:  # noqa: ANN001
        return None

    monkeypatch.setattr(schedule_manager_module, "get_bots", lambda: {"bot": FakeBot()})
    monkeypatch.setattr(manager, "_drain_group_queue", fake_drain_group_queue)

    record = await manager.create_task(
        group_id="1001",
        name="ping-later",
        prompt="15分钟后提醒一下用户。",
        trigger_type="date",
        timezone="Asia/Shanghai",
        run_at="2099-01-01 09:00",
        cron=None,
        replace_existing=False,
    )

    await manager._enqueue_task_run(record.task_id, trigger_source="schedule")

    assert len(bot_calls) == 1
    assert bot_calls[0][0] == "send_group_msg"
    assert "排队等空闲后再开始" in str(bot_calls[0][1]["message"])
