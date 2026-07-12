from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from bampy.ai import AssistantMessage, ImageContent, TextContent
from bampy.ai.types import StopReason, TextDeltaEvent

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat import handler as handler_module
from bampi.plugins.bampi_chat.handler import (
    CLEAR_NO_CONTEXT_MESSAGE,
    CLEARED_SESSION_MESSAGE,
    COMPACT_FORBIDDEN_MESSAGE,
    IncomingMedia,
    LiveProgressReporter,
    ResponseDispatchResult,
    STOPPED_BACKGROUND_SESSION_MESSAGE,
    STOPPED_SESSION_MESSAGE,
    TriggerDecision,
    build_user_message,
    collect_incoming_media,
    describe_tool_progress,
    format_tool_progress_message,
    is_clear_command,
    is_compact_command,
    is_stop_command,
    matched_prefix,
    prepare_group_file_upload,
    reply_target_for_event,
    send_agent_response,
    should_respond,
    strip_streamed_prefix,
)


@dataclass
class FakeReplySender:
    user_id: int


@dataclass
class FakeReply:
    sender: object
    message: object | None = None


@dataclass
class FakeSender:
    user_id: int
    nickname: str = ""
    card: str = ""


@dataclass
class FakeEvent:
    text: str
    to_me: bool = False
    reply: object | None = None
    message: object | None = None

    def get_plaintext(self) -> str:
        return self.text


@dataclass
class FakeGroupEvent:
    group_id: int
    user_id: int
    message_id: int
    message: Message = field(default_factory=Message)
    sender: object = field(default_factory=lambda: FakeSender(user_id=42, nickname="tester"))
    reply: object | None = None

    def get_plaintext(self) -> str:
        return self.message.extract_plain_text()


class FakeBot:
    def __init__(self, responder=None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._responder = responder

    async def call_api(self, action: str, **params: object) -> dict[str, object]:
        self.calls.append((action, params))
        if self._responder is not None:
            response = self._responder(action, params)
            if asyncio.iscoroutine(response):
                return await response
            return response
        return {}


class FakeMatcher:
    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send(self, message: object) -> None:
        self.sent.append(message)


class FakeSession:
    def __init__(self) -> None:
        self.listener = None

    def subscribe(self, listener):
        self.listener = listener

        def unsubscribe() -> None:
            self.listener = None

        return unsubscribe


class FakeGroupSessionManager:
    def __init__(self, workspace_dir: str) -> None:
        self.workspace_dir = workspace_dir
        self.released_group_ids: list[str] = []

    def workspace_dir_for_group(self, group_id: str) -> str:
        return self.workspace_dir

    async def inspect_interaction(self, group_id: str):
        return SimpleNamespace(is_active=False, is_streaming=False)

    async def release(self, group_id: str) -> None:
        self.released_group_ids.append(group_id)


def test_should_respond_when_to_me():
    config = BampiChatConfig()
    decision = should_respond(FakeEvent("hello", to_me=True), bot_self_id="42", config=config, random_value=1.0)
    assert decision == TriggerDecision(True, reason="to_me", direct=True, cleaned_text="hello")


def test_should_respond_when_reply_to_bot():
    config = BampiChatConfig()
    event = FakeEvent("follow up", reply=FakeReply(sender=FakeReplySender(user_id=42)))
    decision = should_respond(event, bot_self_id="42", config=config, random_value=1.0)
    assert decision.should_respond is True
    assert decision.reason == "to_me"


def test_should_strip_trigger_prefix():
    config = BampiChatConfig(bampi_trigger_prefix=["/bot", "小帮"])
    decision = should_respond(FakeEvent("/bot   帮我写个脚本"), bot_self_id="42", config=config, random_value=1.0)
    assert decision.cleaned_text == "帮我写个脚本"


def test_should_match_keyword():
    config = BampiChatConfig(bampi_trigger_keywords=["帮我"])
    decision = should_respond(FakeEvent("你可以帮我看看吗"), bot_self_id="42", config=config, random_value=1.0)
    assert decision.should_respond is True
    assert decision.reason == "keyword"


def test_should_not_random_reply_when_probability_misses():
    config = BampiChatConfig(bampi_random_reply_prob=0.1)
    decision = should_respond(FakeEvent("just chatting"), bot_self_id="42", config=config, random_value=0.5)
    assert decision.should_respond is False


def test_should_respond_when_bot_mentioned_mid_message():
    config = BampiChatConfig()
    message = Message([MessageSegment.text("大家觉得 "), MessageSegment.at(42), MessageSegment.text(" 怎么看")])
    decision = should_respond(FakeEvent("", message=message), bot_self_id="42", config=config, random_value=1.0)
    assert decision.should_respond is True
    assert decision.reason == "mention"
    assert decision.direct is True
    assert decision.cleaned_text == "大家觉得 @42 怎么看"


def test_should_respond_renders_at_and_face_segments():
    config = BampiChatConfig()
    message = Message(
        [
            MessageSegment.text("帮我提醒 "),
            MessageSegment.at(10001),
            MessageSegment.text(" 开会"),
            MessageSegment.face(179),
        ]
    )
    decision = should_respond(
        FakeEvent("", to_me=True, message=message),
        bot_self_id="42",
        config=config,
        random_value=1.0,
        resolve_name={"10001": "张三"}.get,
    )
    assert decision.cleaned_text == "帮我提醒 @张三(10001) 开会[表情:doge]"


def test_matched_prefix_returns_first_match():
    assert matched_prefix("@bot hello", ["@bot", "/bot"]) == "@bot"


def test_is_stop_command_accepts_normalized_command():
    assert is_stop_command("/stop") is True
    assert is_stop_command("  /STOP  ") is True
    assert is_stop_command("/stop now") is False


def test_is_clear_command_accepts_aliases():
    assert is_clear_command("/clear") is True
    assert is_clear_command(" /NEW ") is True
    assert is_clear_command("/clear now") is False


def test_is_compact_command_accepts_normalized_command():
    assert is_compact_command("/compact") is True
    assert is_compact_command(" /COMPACT ") is True
    assert is_compact_command("/compact now") is False


def test_format_tool_progress_message_uses_emoji_style():
    message = format_tool_progress_message("read", {"path": "README.md"})

    assert message == "📖 正在读取：README.md"
    assert "进度：" not in message
    assert "`" not in message


def test_describe_tool_progress_browser_single_command():
    description = describe_tool_progress(
        "browser",
        {"command": "open https://example.com"},
    )

    assert description == "正在操作浏览器：open https://example.com"


def test_describe_tool_progress_browser_batch_shows_step_count():
    command = (
        "batch\n"
        "open file://outbox/render.html\n"
        'wait css=body[data-ready="true"]\n'
        "screenshot outbox/md-render.png --target css=body --full\n"
        "close"
    )

    description = describe_tool_progress("browser", {"command": command})

    assert description == "正在操作浏览器（批量 4 步）"


def test_format_tool_progress_message_marks_skill_loads():
    message = format_tool_progress_message(
        "read",
        {"path": ".agents/builtin-skills/docx/SKILL.md"},
    )

    assert message == "🧩 正在加载 skill：docx"


def test_format_tool_progress_message_marks_skill_resources():
    message = format_tool_progress_message(
        "read",
        {"path": ".agents/builtin-skills/docx/references/guide.md"},
    )

    assert message == "🧩 正在读取 skill 资料：docx/references/guide.md"


@pytest.mark.asyncio
async def test_live_progress_reporter_does_not_send_delayed_ack():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(bampi_live_progress_enabled=True)
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    await asyncio.sleep(0.02)
    await reporter.prepare_final_reply()
    await reporter.close()

    assert bot.calls == []


@pytest.mark.asyncio
async def test_live_progress_reporter_sends_threshold_compaction_notice_even_without_live_progress():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(
        bampi_live_progress_enabled=False,
        bampi_threshold_compaction_notice_enabled=True,
    )
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None
    session.listener(SimpleNamespace(type="auto_compaction_start", reason="threshold"))
    await reporter.prepare_final_reply()
    await reporter.close()

    assert len(bot.calls) == 1
    action, params = bot.calls[0]
    assert action == "send_group_msg"
    assert params["group_id"] == 1001
    assert str(params["message"]) == "[CQ:reply,id=99]🧹 上下文长度接近上限，正在自动压缩前文，完成后继续。"


@pytest.mark.asyncio
async def test_live_progress_reporter_sends_emoji_tool_update():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(bampi_live_progress_enabled=True)
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None
    session.listener(SimpleNamespace(type="tool_execution_start", tool_name="grep", args={"pattern": "TODO"}, tool_call_id="tc1"))
    await reporter.prepare_final_reply()
    await reporter.close()

    assert len(bot.calls) == 1
    action, params = bot.calls[0]
    assert action == "send_group_msg"
    assert params["group_id"] == 1001
    assert str(params["message"]) == "[CQ:reply,id=99]🔍 正在搜索：TODO"


@pytest.mark.asyncio
async def test_live_progress_reporter_can_announce_skill_loading():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(bampi_live_progress_enabled=True)
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    reporter.announce_skill_loading(["docx", "skill-creator"])
    await reporter.prepare_final_reply()
    await reporter.close()

    assert len(bot.calls) == 1
    _, params = bot.calls[0]
    assert str(params["message"]) == "[CQ:reply,id=99]🧩 正在加载 skills：docx, skill-creator"


@pytest.mark.asyncio
async def test_handle_skill_command_installs_from_message_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bot = FakeBot()
    matcher = FakeMatcher()
    manager = FakeGroupSessionManager(str(tmp_path / "workspace"))
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig()
    install_calls: list[str] = []

    async def fake_collect_incoming_media(bot, event, config, workspace_dir):  # noqa: ANN001
        return IncomingMedia(saved_paths=["inbox/skill-pack.zip"], reply_saved_paths=["inbox/other-skill.md"])

    def fake_install_skills_from_source(source: str, **kwargs):  # noqa: ANN003
        install_calls.append(source)
        if source.endswith(".zip"):
            return SimpleNamespace(installed_names=["docx"], replaced_names=[], diagnostics=[])
        return SimpleNamespace(installed_names=["skill-creator"], replaced_names=[], diagnostics=[])

    monkeypatch.setattr(handler_module, "collect_incoming_media", fake_collect_incoming_media)
    monkeypatch.setattr(handler_module, "install_skills_from_source", fake_install_skills_from_source)

    handled = await handler_module._handle_skill_command(
        bot=bot,
        event=event,
        command_text="/skill install",
        group_id="1001",
        matcher=matcher,
        session_manager=manager,
        config=config,
    )

    assert handled is True
    assert install_calls == ["inbox/skill-pack.zip", "inbox/other-skill.md"]
    assert manager.released_group_ids == ["1001"]
    assert matcher.sent == [
        "已安装 2 个 skill：docx, skill-creator\n"
        "使用方法：在消息开头写 `/skill-name` 即可调用。"
    ]


@pytest.mark.asyncio
async def test_handle_skill_command_rejects_local_path_argument(tmp_path: Path):
    bot = FakeBot()
    matcher = FakeMatcher()
    manager = FakeGroupSessionManager(str(tmp_path / "workspace"))
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig()

    handled = await handler_module._handle_skill_command(
        bot=bot,
        event=event,
        command_text="/skill install inbox/skill-pack.zip",
        group_id="1001",
        matcher=matcher,
        session_manager=manager,
        config=config,
    )

    assert handled is True
    assert matcher.sent == [
        "URL 无效。\n"
        "请发送或引用 skill 文件后执行 `/skill install`，"
        "或使用 `/skill install <url>`。"
    ]


@pytest.mark.asyncio
async def test_live_progress_reporter_allows_unlimited_tool_updates_when_limit_zero():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(
        bampi_live_progress_enabled=True,
        bampi_live_progress_max_tool_updates=0,
    )
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None
    for i, (name, args) in enumerate([
        ("write", {"path": "a.py"}),
        ("bash", {"command": "python3 a.py"}),
        ("find", {"pattern": "**/*.py", "path": "/workspace"}),
    ]):
        session.listener(SimpleNamespace(type="tool_execution_start", tool_name=name, args=args, tool_call_id=f"tc{i}"))
        session.listener(SimpleNamespace(type="tool_execution_end", tool_name=name, tool_call_id=f"tc{i}", is_error=False, result=None))
    await reporter.prepare_final_reply()
    await reporter.close()

    assert len(bot.calls) == 3
    assert [call[0] for call in bot.calls] == ["send_group_msg", "send_group_msg", "send_group_msg"]


@pytest.mark.asyncio
async def test_live_progress_reporter_recalls_failed_tool_update_after_min_visible_delay():
    call_times: dict[str, list[float]] = {}

    def responder(action: str, params: dict[str, object]) -> dict[str, object]:
        call_times.setdefault(action, []).append(time.monotonic())
        if action == "send_group_msg":
            return {"message_id": 5566}
        return {}

    bot = FakeBot(responder=responder)
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(
        bampi_live_progress_enabled=True,
        bampi_live_progress_error_recall_min_visible_seconds=0.05,
    )
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None
    session.listener(SimpleNamespace(type="tool_execution_start", tool_name="bash", args={"command": "make test"}, tool_call_id="tc1"))
    session.listener(SimpleNamespace(type="tool_execution_end", tool_name="bash", tool_call_id="tc1", is_error=True, result=None))
    await reporter.prepare_final_reply()
    await asyncio.sleep(0.08)
    await reporter.close()

    assert [action for action, _ in bot.calls] == ["send_group_msg", "delete_msg"]
    assert bot.calls[1][1] == {"message_id": 5566}
    assert "正在执行命令：make test" in str(bot.calls[0][1]["message"])
    assert call_times["delete_msg"][0] - call_times["send_group_msg"][0] >= 0.045


@pytest.mark.asyncio
async def test_live_progress_reporter_uses_text_delta_without_snapshot_desync():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(
        bampi_live_progress_enabled=True,
        bampi_live_text_stream_enabled=True,
        bampi_live_text_stream_min_chars=999,
        bampi_live_text_stream_force_chars=9999,
    )
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None

    session.listener(SimpleNamespace(type="message_start", message=AssistantMessage(content=[])))
    first_partial = AssistantMessage(content=[TextContent(text="让我先看看 inbox 目录里有什么文件，然后解读一下内容。")])
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=first_partial,
            assistant_message_event=TextDeltaEvent(
                content_index=0,
                delta="让我先看看 inbox 目录里有什么文件，然后解读一下内容。",
                partial=first_partial,
            ),
        )
    )
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=first_partial,
            assistant_message_event=SimpleNamespace(type="toolcall_start"),
        )
    )
    session.listener(SimpleNamespace(type="message_end", message=first_partial))
    session.listener(SimpleNamespace(type="tool_execution_start", tool_name="find", args={"pattern": "*", "path": "inbox"}, tool_call_id="tc1"))
    session.listener(SimpleNamespace(type="tool_execution_end", tool_name="find", tool_call_id="tc1", is_error=False, result=None))

    session.listener(SimpleNamespace(type="message_start", message=AssistantMessage(content=[])))
    second_partial = AssistantMessage(content=[TextContent(text="实验已完成。")])
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=second_partial,
            assistant_message_event=TextDeltaEvent(
                content_index=0,
                delta="实验已完成。",
                partial=second_partial,
            ),
        )
    )
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=AssistantMessage(content=[TextContent(text="实验已完成。")]),
            assistant_message_event=SimpleNamespace(type="text_end"),
        )
    )
    session.listener(SimpleNamespace(type="message_end", message=second_partial))
    await reporter.prepare_final_reply()
    await reporter.close()

    assert [str(call[1]["message"]) for call in bot.calls] == [
        "[CQ:reply,id=99]让我先看看 inbox 目录里有什么文件，然后解读一下内容。",
        "🔎 正在查找：*",
        "实验已完成。",
    ]
    assert reporter.streamed_text == "实验已完成。"


@pytest.mark.asyncio
async def test_live_progress_reporter_flushes_pending_text_on_message_end():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(
        bampi_live_progress_enabled=True,
        bampi_live_text_stream_enabled=True,
        bampi_live_text_stream_min_chars=999,
        bampi_live_text_stream_force_chars=9999,
    )
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None
    partial = AssistantMessage(content=[TextContent(text="这是最终答案。")])
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=partial,
            assistant_message_event=TextDeltaEvent(
                content_index=0,
                delta="这是最终答案。",
                partial=partial,
            ),
        )
    )
    session.listener(SimpleNamespace(type="message_end", message=partial))
    await reporter.prepare_final_reply()
    await reporter.close()

    assert len(bot.calls) == 1
    assert str(bot.calls[0][1]["message"]) == "[CQ:reply,id=99]这是最终答案。"


@pytest.mark.asyncio
async def test_live_progress_reporter_emits_whole_message_once_at_end():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(
        bampi_live_progress_enabled=True,
        bampi_live_text_stream_enabled=True,
    )
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None
    session.listener(SimpleNamespace(type="message_start", message=AssistantMessage(content=[])))

    first = AssistantMessage(content=[TextContent(text="第一句。")])
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=first,
            assistant_message_event=TextDeltaEvent(
                content_index=0,
                delta="第一句。",
                partial=first,
            ),
        )
    )
    second = AssistantMessage(content=[TextContent(text="第一句。第二句。")])
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=second,
            assistant_message_event=TextDeltaEvent(
                content_index=0,
                delta="第二句。",
                partial=second,
            ),
        )
    )
    session.listener(SimpleNamespace(type="message_end", message=second))
    await reporter.prepare_final_reply()
    await reporter.close()

    assert len(bot.calls) == 1
    assert str(bot.calls[0][1]["message"]) == "[CQ:reply,id=99]第一句。第二句。"


@pytest.mark.asyncio
async def test_live_progress_reporter_ignores_snapshot_updates_after_text_delta():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(
        bampi_live_progress_enabled=True,
        bampi_live_text_stream_enabled=True,
        bampi_live_text_stream_min_chars=999,
        bampi_live_text_stream_force_chars=9999,
    )
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None
    session.listener(SimpleNamespace(type="message_start", message=AssistantMessage(content=[])))

    first_partial = AssistantMessage(content=[TextContent(text="我来")])
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=first_partial,
            assistant_message_event=TextDeltaEvent(
                content_index=0,
                delta="我来",
                partial=first_partial,
            ),
        )
    )
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=AssistantMessage(content=[TextContent(text="我来先读取实验要求文件看看具体内容。")]),
            assistant_message_event=SimpleNamespace(type="text_end"),
        )
    )
    full_partial = AssistantMessage(content=[TextContent(text="我来先读取实验要求文件看看具体内容。")])
    session.listener(
        SimpleNamespace(
            type="message_update",
            message=full_partial,
            assistant_message_event=TextDeltaEvent(
                content_index=0,
                delta="先读取实验要求文件看看具体内容。",
                partial=full_partial,
            ),
        )
    )
    session.listener(SimpleNamespace(type="message_end", message=full_partial))
    await reporter.prepare_final_reply()
    await reporter.close()

    assert len(bot.calls) == 1
    assert str(bot.calls[0][1]["message"]) == "[CQ:reply,id=99]我来先读取实验要求文件看看具体内容。"


def test_strip_streamed_prefix_removes_only_exact_prefix():
    full_text = "画好了！\n\n- 主体是经典的心形参数方程\n- 配了金色线条"
    streamed_text = "画好了！\n\n- 主体是经典的心形参数方程\n"

    assert strip_streamed_prefix(full_text, streamed_text) == "- 配了金色线条"


def test_strip_streamed_prefix_keeps_full_text_when_prefix_mismatches():
    full_text = "完整回复内容"
    streamed_text = "不匹配的前缀"

    assert strip_streamed_prefix(full_text, streamed_text) == full_text


def test_memory_tool_progress_hides_internal_arguments():
    cases = {
        "memory_search": ({"query": "nginx 证书"}, "正在搜索记忆"),
        "memory_time_search": (
            {
                "start_time": "2026-05-05T00:00:00+08:00",
                "end_time": "2026-05-05T23:59:59+08:00",
            },
            "正在搜索记忆",
        ),
        "memory_open": ({"archive_id": 123}, "正在查看记忆"),
        "memory_manage": ({"action": "add", "content": "喜欢 Rust"}, "正在记录记忆"),
    }

    for tool_name, (payload, expected) in cases.items():
        assert describe_tool_progress(tool_name, payload) == expected


def test_build_user_message_marks_media_only_message():
    event = FakeGroupEvent(
        group_id=1001,
        user_id=42,
        message_id=99,
        sender=FakeSender(user_id=42, nickname="Alice"),
    )

    message = build_user_message(
        event,
        "",
        IncomingMedia(saved_paths=["inbox/report.txt"]),
    )

    assert message.content[0].text.startswith("sender_name: Alice(42)")
    assert "message_text: (无纯文本内容；本条消息仅包含媒体/文件)" in message.content[0].text
    assert "workspace_attachments:\n- inbox/report.txt" in message.content[0].text
    assert "group_id:" not in message.content[0].text
    assert "sender_id:" not in message.content[0].text
    assert "sender_user_id:" not in message.content[0].text


def test_build_user_message_separates_reply_media_context():
    event = FakeGroupEvent(
        group_id=1001,
        user_id=42,
        message_id=99,
        sender=FakeSender(user_id=42, nickname="Alice"),
        reply=FakeReply(
            sender=FakeSender(user_id=7, nickname="Bob"),
            message=Message("原始说明"),
        ),
    )
    media = IncomingMedia(
        inline_images=[ImageContent(data="YQ==", mime_type="image/png")],
        saved_paths=["inbox/current.txt"],
        notes=["当前备注"],
        reply_inline_images=[ImageContent(data="Yg==", mime_type="image/jpeg")],
        reply_saved_paths=["inbox/reply.pdf"],
        reply_notes=["回复备注"],
    )

    message = build_user_message(event, "帮我看看", media)
    text_block = message.content[0].text

    assert "reply_to_name: Bob" in text_block
    assert "reply_message: 原始说明" in text_block
    assert "reply_to_user_id:" not in text_block
    assert "inline_image_count: 1" in text_block
    assert "workspace_attachments:\n- inbox/current.txt" in text_block
    assert "media_notes:\n- 当前备注" in text_block
    assert "reply_inline_image_count: 1" in text_block
    assert "reply_workspace_attachments:\n- inbox/reply.pdf" in text_block
    assert "reply_media_notes:\n- 回复备注" in text_block
    assert len(message.content) == 3
    assert message.content[1].mime_type == "image/png"
    assert message.content[2].mime_type == "image/jpeg"


def test_build_user_message_renders_reply_at_and_face_segments():
    event = FakeGroupEvent(
        group_id=1001,
        user_id=42,
        message_id=99,
        sender=FakeSender(user_id=42, nickname="Alice"),
        reply=FakeReply(
            sender=FakeSender(user_id=7, nickname="Bob"),
            message=Message(
                [
                    MessageSegment.at(10001),
                    MessageSegment.text(" 说得对"),
                    MessageSegment.face(14),
                ]
            ),
        ),
    )

    message = build_user_message(
        event,
        "帮我看看",
        IncomingMedia(),
        resolve_name={"10001": "李四"}.get,
    )

    assert "reply_message: @李四(10001) 说得对[表情:微笑]" in message.content[0].text


@pytest.mark.asyncio
async def test_collect_incoming_media_includes_reply_image_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def fake_download(url: str, *, timeout: float, max_bytes: int) -> tuple[bytes, str]:
        assert timeout > 0
        assert max_bytes > 0
        if url.endswith("reply.png"):
            return b"reply-image", "image/png"
        if url.endswith("reply.txt"):
            return b"reply-file", "text/plain"
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(handler_module, "download_url", fake_download)

    def responder(action: str, params: dict[str, object]) -> dict[str, object]:
        assert action == "get_group_file_url"
        assert params["group_id"] == 1001
        assert params["file_id"] == "file-1"
        return {"url": "https://example.com/reply.txt"}

    bot = FakeBot(responder=responder)
    event = FakeGroupEvent(
        group_id=1001,
        user_id=42,
        message_id=99,
        message=Message("帮我看这个"),
        sender=FakeSender(user_id=42, nickname="Alice"),
        reply=FakeReply(
            sender=FakeSender(user_id=7, nickname="Bob"),
            message=Message(
                [
                    MessageSegment("image", {"url": "https://example.com/reply.png"}),
                    MessageSegment("file", {"file_id": "file-1", "file": "reply.txt"}),
                ]
            ),
        ),
    )
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_max_inline_image_size=1024,
    )

    media = await collect_incoming_media(bot, event, config, str(tmp_path / "workspace"))

    assert media.inline_images == []
    assert media.saved_paths == []
    assert len(media.reply_inline_images) == 1
    assert media.reply_inline_images[0].mime_type == "image/png"
    assert media.reply_inline_images[0].data == base64.b64encode(b"reply-image").decode("ascii")
    assert len(media.reply_saved_paths) == 1
    saved_path = tmp_path / "workspace" / media.reply_saved_paths[0]
    assert saved_path.name.startswith("reply-")
    assert saved_path.suffix == ".txt"
    assert saved_path.read_text(encoding="utf-8") == "reply-file"
    assert media.reply_notes == []
    assert bot.calls == [
        (
            "get_group_file_url",
            {"group_id": 1001, "file_id": "file-1"},
        )
    ]


@pytest.mark.asyncio
async def test_collect_incoming_media_preserves_zip_name_from_segment_file_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_download(url: str, *, timeout: float, max_bytes: int) -> tuple[bytes, str]:
        assert url == "https://example.com/archive"
        return b"PK\x03\x04zip-bytes", "application/octet-stream"

    monkeypatch.setattr(handler_module, "download_url", fake_download)

    def responder(action: str, params: dict[str, object]) -> dict[str, object]:
        assert action == "get_group_file_url"
        return {"url": "https://example.com/archive"}

    bot = FakeBot(responder=responder)
    event = FakeGroupEvent(
        group_id=1001,
        user_id=42,
        message_id=99,
        message=Message("帮我看看"),
        sender=FakeSender(user_id=42, nickname="Alice"),
        reply=FakeReply(
            sender=FakeSender(user_id=7, nickname="Bob"),
            message=Message(
                [
                    MessageSegment("file", {"file_id": "file-zip", "file": "ark.zip"}),
                ]
            ),
        ),
    )
    config = BampiChatConfig(bampi_workspace_dir=str(tmp_path / "workspace"))

    media = await collect_incoming_media(bot, event, config, str(tmp_path / "workspace"))

    assert media.reply_saved_paths
    saved_path = tmp_path / "workspace" / media.reply_saved_paths[0]
    assert saved_path.name.startswith("ark-")
    assert saved_path.suffix == ".zip"
    assert saved_path.read_bytes() == b"PK\x03\x04zip-bytes"


@pytest.mark.asyncio
async def test_prepare_group_file_upload_stages_file_for_napcat(tmp_path: Path):
    source = tmp_path / "outbox" / "report.txt"
    source.parent.mkdir(parents=True)
    source.write_text("hello", encoding="utf-8")

    config = BampiChatConfig(
        bampi_group_file_upload_host_dir=str(tmp_path / "qq-temp"),
        bampi_group_file_upload_container_dir="/app/.config/QQ/temp",
    )

    prepared = await prepare_group_file_upload(source, config)

    assert prepared.file_uri.startswith("file:///app/.config/QQ/temp/")
    assert len(prepared.cleanup_paths) == 1
    staged_path = prepared.cleanup_paths[0]
    assert staged_path.parent == tmp_path / "qq-temp"
    assert staged_path.read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_send_agent_response_uploads_file_with_uri_and_cleans_up(tmp_path: Path):
    workspace = tmp_path / "workspace"
    outbox = workspace / "outbox"
    outbox.mkdir(parents=True)
    result_file = outbox / "report.txt"
    result_file.write_text("answer", encoding="utf-8")

    config = BampiChatConfig(
        bampi_workspace_dir=str(workspace),
        bampi_reply_with_quote=False,
        bampi_at_sender=False,
        bampi_group_file_upload_host_dir=str(tmp_path / "qq-temp"),
        bampi_group_file_upload_container_dir="/app/.config/QQ/temp",
    )
    bot = FakeBot()
    matcher = FakeMatcher()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    assistant_message = AssistantMessage(content=[TextContent(text="结果见附件")])

    result = await send_agent_response(
        bot=bot,
        event=event,
        matcher=matcher,
        config=config,
        workspace_dir=str(workspace),
        assistant_message=assistant_message,
        outbox_before={},
    )

    assert result.delivered is True
    assert len(matcher.sent) == 1
    assert len(bot.calls) == 1
    action, params = bot.calls[0]
    assert action == "upload_group_file"
    assert params["group_id"] == 1001
    assert params["name"] == "report.txt"
    assert str(params["file"]).startswith("file:///app/.config/QQ/temp/")
    assert not result_file.exists()
    staged_dir = tmp_path / "qq-temp"
    assert staged_dir.exists()
    assert list(staged_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_send_agent_response_uses_streamed_text_prefix_without_truncation(tmp_path: Path):
    workspace = tmp_path / "workspace"
    outbox = workspace / "outbox"
    outbox.mkdir(parents=True)

    config = BampiChatConfig(
        bampi_workspace_dir=str(workspace),
        bampi_reply_with_quote=False,
        bampi_at_sender=False,
    )
    bot = FakeBot()
    matcher = FakeMatcher()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    assistant_message = AssistantMessage(
        content=[
            TextContent(
                text="画好了！\n\n- 主体是经典的心形参数方程\n- 配了金色线条"
            )
        ]
    )

    result = await send_agent_response(
        bot=bot,
        event=event,
        matcher=matcher,
        config=config,
        workspace_dir=str(workspace),
        assistant_message=assistant_message,
        outbox_before={},
        streamed_text="画好了！\n\n- 主体是经典的心形参数方程\n",
        streamed_any_text=True,
    )

    assert result.delivered is True
    assert len(matcher.sent) == 1
    assert str(matcher.sent[0]) == "- 配了金色线条"


@pytest.mark.asyncio
async def test_send_agent_response_inlines_small_image_and_cleans_up(tmp_path: Path):
    workspace = tmp_path / "workspace"
    outbox = workspace / "outbox"
    outbox.mkdir(parents=True)
    image_file = outbox / "plot.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\nsmall-image")

    config = BampiChatConfig(
        bampi_workspace_dir=str(workspace),
        bampi_reply_with_quote=False,
        bampi_at_sender=False,
        bampi_max_inline_image_size=1024,
        bampi_group_file_upload_host_dir=str(tmp_path / "qq-temp"),
        bampi_group_file_upload_container_dir="/app/.config/QQ/temp",
    )
    bot = FakeBot()
    matcher = FakeMatcher()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    assistant_message = AssistantMessage(content=[TextContent(text="给你一张图")])

    result = await send_agent_response(
        bot=bot,
        event=event,
        matcher=matcher,
        config=config,
        workspace_dir=str(workspace),
        assistant_message=assistant_message,
        outbox_before={},
    )

    assert result.delivered is True
    assert bot.calls == []
    assert len(matcher.sent) == 1
    message = matcher.sent[0]
    segments = list(message)
    image_segments = [segment for segment in segments if segment.type == "image"]
    assert len(image_segments) == 1
    assert image_segments[0].data["file"].startswith("base64://")
    assert not image_file.exists()


@pytest.mark.asyncio
async def test_send_agent_response_stages_large_image_for_napcat(tmp_path: Path):
    workspace = tmp_path / "workspace"
    outbox = workspace / "outbox"
    outbox.mkdir(parents=True)
    image_file = outbox / "plot.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\nlarge-image")

    staging_dir = tmp_path / "qq-temp"
    config = BampiChatConfig(
        bampi_workspace_dir=str(workspace),
        bampi_reply_with_quote=False,
        bampi_at_sender=False,
        bampi_max_inline_image_size=1,
        bampi_group_file_upload_host_dir=str(staging_dir),
        bampi_group_file_upload_container_dir="/app/.config/QQ/temp",
    )
    bot = FakeBot()
    matcher = FakeMatcher()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    assistant_message = AssistantMessage(content=[TextContent(text="给你一张大图")])

    result = await send_agent_response(
        bot=bot,
        event=event,
        matcher=matcher,
        config=config,
        workspace_dir=str(workspace),
        assistant_message=assistant_message,
        outbox_before={},
    )

    assert result.delivered is True
    assert bot.calls == []
    assert len(matcher.sent) == 1
    message = matcher.sent[0]
    segments = list(message)
    image_segments = [segment for segment in segments if segment.type == "image"]
    assert len(image_segments) == 1
    assert str(image_segments[0].data["file"]).startswith("file:///app/.config/QQ/temp/")
    assert not str(image_segments[0].data["file"]).startswith("file:///Users/")
    assert not image_file.exists()
    assert staging_dir.exists()
    assert list(staging_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_send_agent_response_skips_aborted_reply(tmp_path: Path):
    workspace = tmp_path / "workspace"
    outbox = workspace / "outbox"
    outbox.mkdir(parents=True)

    config = BampiChatConfig(
        bampi_workspace_dir=str(workspace),
        bampi_reply_with_quote=False,
        bampi_at_sender=False,
    )
    bot = FakeBot()
    matcher = FakeMatcher()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    assistant_message = AssistantMessage(
        content=[TextContent(text="这段内容不会被发送")],
        stop_reason=StopReason.ABORTED,
        error_message="stopped by session owner",
    )

    result = await send_agent_response(
        bot=bot,
        event=event,
        matcher=matcher,
        config=config,
        workspace_dir=str(workspace),
        assistant_message=assistant_message,
        outbox_before={},
    )

    assert result.delivered is False
    assert result.rollback_context is True
    assert matcher.sent == []
    assert bot.calls == []


@pytest.mark.asyncio
async def test_register_handlers_clears_context_with_clear_command(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)

    class FakeSessionManagerForClear:
        def __init__(self) -> None:
            self.cleared_groups: list[str] = []

        def workspace_dir_for_group(self, group_id: str) -> str:
            return "."

        async def inspect_interaction(self, group_id: str):
            return SimpleNamespace(
                is_active=False,
                is_streaming=False,
                has_running_background=False,
                managed=None,
                active_user_id=None,
            )

        async def clear_context(self, group_id: str) -> bool:
            self.cleared_groups.append(group_id)
            return True

    session_manager = FakeSessionManagerForClear()
    config = BampiChatConfig()
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 42
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("/clear"))
    matcher = FakeMatcher()

    await handler(bot, event, matcher)

    assert session_manager.cleared_groups == ["1001"]
    assert matcher.sent == [CLEARED_SESSION_MESSAGE]


@pytest.mark.asyncio
async def test_register_handlers_reports_no_context_for_new_command(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)

    class FakeSessionManagerForNew:
        def workspace_dir_for_group(self, group_id: str) -> str:
            return "."

        async def inspect_interaction(self, group_id: str):
            return SimpleNamespace(
                is_active=False,
                is_streaming=False,
                has_running_background=False,
                managed=None,
                active_user_id=None,
            )

        async def clear_context(self, group_id: str) -> bool:
            return False

    session_manager = FakeSessionManagerForNew()
    config = BampiChatConfig()
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 42
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("/new"))
    matcher = FakeMatcher()

    await handler(bot, event, matcher)

    assert matcher.sent == [CLEAR_NO_CONTEXT_MESSAGE]


@pytest.mark.asyncio
async def test_register_handlers_rejects_compact_for_non_superuser(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)
    monkeypatch.setattr(handler_module, "is_nonebot_superuser", lambda user_id: False)

    class FakeSessionManagerForCompact:
        def workspace_dir_for_group(self, group_id: str) -> str:
            return "."

    session_manager = FakeSessionManagerForCompact()
    config = BampiChatConfig()
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 42
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("/compact"))
    matcher = FakeMatcher()

    await handler(bot, event, matcher)

    assert matcher.sent == [COMPACT_FORBIDDEN_MESSAGE]


@pytest.mark.asyncio
async def test_register_handlers_runs_manual_compact_for_superuser(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)
    monkeypatch.setattr(handler_module, "is_nonebot_superuser", lambda user_id: True)

    class FakeManagedSessionRuntime:
        def __init__(self) -> None:
            self.compact_calls = 0

        async def compact(self):
            self.compact_calls += 1
            return SimpleNamespace(tokens_before=1200, tokens_after=800)

    class FakeSessionManagerForCompact:
        def __init__(self) -> None:
            self.managed = SimpleNamespace(
                session=FakeManagedSessionRuntime(),
                lock=asyncio.Lock(),
            )
            self.complete_calls = 0

        def workspace_dir_for_group(self, group_id: str) -> str:
            return "."

        async def inspect_interaction(self, group_id: str):
            return SimpleNamespace(
                is_active=False,
                is_streaming=False,
                has_running_background=False,
                managed=None,
                active_user_id=None,
            )

        async def has_context(self, group_id: str) -> bool:
            return True

        async def get_or_create(self, group_id: str):
            return self.managed

        async def complete_interaction(self, group_id: str) -> None:
            self.complete_calls += 1

    session_manager = FakeSessionManagerForCompact()
    config = BampiChatConfig()
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 42
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("/compact"))
    matcher = FakeMatcher()

    await handler(bot, event, matcher)

    assert session_manager.managed.session.compact_calls == 1
    assert session_manager.complete_calls == 1
    assert matcher.sent == ["已完成上下文压缩，约减少 400 tokens。"]


@pytest.mark.asyncio
async def test_register_handlers_stop_cancels_waiting_background_session(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)
    monkeypatch.setattr(handler_module, "is_nonebot_superuser", lambda user_id: False)

    class FakeSessionManagerForStop:
        def __init__(self) -> None:
            self.stop_calls: list[tuple[str, str]] = []

        def workspace_dir_for_group(self, group_id: str) -> str:
            return "."

        async def inspect_interaction(self, group_id: str):
            return SimpleNamespace(
                is_active=True,
                active_user_id="42",
                is_streaming=False,
                has_running_background=True,
                background_owner_user_ids=frozenset({"42"}),
                managed=SimpleNamespace(),
            )

        async def stop_interaction(self, group_id: str, *, reason: str):
            self.stop_calls.append((group_id, reason))
            return SimpleNamespace(
                aborted_streaming=False,
                stopped_background_sessions=True,
                stopped_background_session_ids=["term-1"],
            )

    session_manager = FakeSessionManagerForStop()
    config = BampiChatConfig()
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 42
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("/stop"))
    matcher = FakeMatcher()

    await handler(bot, event, matcher)

    assert session_manager.stop_calls == [("1001", "stopped by session owner")]
    assert matcher.sent == [STOPPED_BACKGROUND_SESSION_MESSAGE]


@pytest.mark.asyncio
async def test_register_handlers_superuser_can_force_stop_other_users_session(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)
    monkeypatch.setattr(handler_module, "is_nonebot_superuser", lambda user_id: True)

    class FakeSessionManagerForForceStop:
        def __init__(self) -> None:
            self.stop_calls: list[tuple[str, str]] = []

        def workspace_dir_for_group(self, group_id: str) -> str:
            return "."

        async def inspect_interaction(self, group_id: str):
            return SimpleNamespace(
                is_active=True,
                active_user_id="7",
                is_streaming=True,
                has_running_background=False,
                background_owner_user_ids=frozenset(),
                managed=SimpleNamespace(),
            )

        async def stop_interaction(self, group_id: str, *, reason: str):
            self.stop_calls.append((group_id, reason))
            return SimpleNamespace(
                aborted_streaming=True,
                stopped_background_sessions=False,
                stopped_background_session_ids=[],
            )

    session_manager = FakeSessionManagerForForceStop()
    config = BampiChatConfig()
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 42
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("/stop"))
    matcher = FakeMatcher()

    await handler(bot, event, matcher)

    assert session_manager.stop_calls == [("1001", "stopped by superuser")]
    assert matcher.sent == ["已强制" + STOPPED_SESSION_MESSAGE.removeprefix("已")]


@pytest.mark.asyncio
async def test_register_handlers_does_not_steer_owner_without_trigger(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    async def unexpected_collect_incoming_media(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("collect_incoming_media should not be called without trigger")

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)
    monkeypatch.setattr(handler_module, "collect_incoming_media", unexpected_collect_incoming_media)

    class FakeManagedSessionRuntime:
        def __init__(self) -> None:
            self.is_processing = True
            self.messages = [AssistantMessage(content=[TextContent(text="processing")])]
            self.session_manager = SimpleNamespace(leaf_id=None)
            self.steer_calls: list[object] = []

        def steer(self, user_message) -> None:
            self.steer_calls.append(user_message)

    class FakeSessionManagerForActiveOwner:
        def __init__(self) -> None:
            self.workspace_dir = "."
            self.managed = SimpleNamespace(
                session=FakeManagedSessionRuntime(),
                lock=asyncio.Lock(),
                last_used_at=0.0,
            )
            self.active_user_id = "42"

        def workspace_dir_for_group(self, group_id: str) -> str:
            return self.workspace_dir

        async def inspect_interaction(self, group_id: str):
            return SimpleNamespace(
                is_active=True,
                active_user_id=self.active_user_id,
                is_streaming=self.managed.session.is_processing,
                managed=self.managed,
            )

        async def reserve_interaction(self, group_id: str, user_id: str):
            raise AssertionError("reserve_interaction should not be called without trigger")

    session_manager = FakeSessionManagerForActiveOwner()
    config = BampiChatConfig()
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 99

    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("随便说一句"))
    event.to_me = False
    matcher = FakeMatcher()
    await handler(bot, event, matcher)

    assert session_manager.managed.session.steer_calls == []
    assert matcher.sent == []


@pytest.mark.asyncio
async def test_register_handlers_allows_owner_to_steer_when_trigger_matches(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)
    monkeypatch.setattr(handler_module, "collect_incoming_media", lambda *args, **kwargs: asyncio.sleep(0, result=IncomingMedia()))

    class FakeManagedSessionRuntime:
        def __init__(self) -> None:
            self.is_processing = True
            self.messages = [AssistantMessage(content=[TextContent(text="processing")])]
            self.session_manager = SimpleNamespace(leaf_id=None)
            self.steer_calls: list[object] = []

        def steer(self, user_message) -> None:
            self.steer_calls.append(user_message)

    class FakeSessionManagerForActiveOwner:
        def __init__(self) -> None:
            self.workspace_dir = "."
            self.managed = SimpleNamespace(
                session=FakeManagedSessionRuntime(),
                lock=asyncio.Lock(),
                last_used_at=0.0,
            )
            self.active_user_id = "42"

        def workspace_dir_for_group(self, group_id: str) -> str:
            return self.workspace_dir

        async def inspect_interaction(self, group_id: str):
            return SimpleNamespace(
                is_active=True,
                active_user_id=self.active_user_id,
                is_streaming=self.managed.session.is_processing,
                managed=self.managed,
            )

        async def reserve_interaction(self, group_id: str, user_id: str):
            raise AssertionError("reserve_interaction should not be called for active owner steer")

    session_manager = FakeSessionManagerForActiveOwner()
    config = BampiChatConfig()
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 99

    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("继续看这个"))
    event.to_me = True
    matcher = FakeMatcher()
    await handler(bot, event, matcher)

    assert len(session_manager.managed.session.steer_calls) == 1
    assert matcher.sent == []


@pytest.mark.asyncio
async def test_register_handlers_clears_owner_after_successful_turn(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)
    monkeypatch.setattr(handler_module, "collect_incoming_media", lambda *args, **kwargs: asyncio.sleep(0, result=IncomingMedia()))
    monkeypatch.setattr(handler_module, "snapshot_outbox", lambda workspace_dir: {})
    monkeypatch.setattr(
        handler_module,
        "send_agent_response",
        lambda **kwargs: asyncio.sleep(0, result=ResponseDispatchResult(delivered=True)),
    )

    class FakeManagedSessionRuntime:
        def __init__(self) -> None:
            self.is_processing = False
            self.messages = [AssistantMessage(content=[TextContent(text="ok")])]
            self.session_manager = SimpleNamespace(leaf_id=None)

        async def prompt(self, user_message, *, source: str) -> None:
            self.is_processing = True
            await asyncio.sleep(0)
            self.is_processing = False

        def subscribe(self, listener):
            def unsubscribe() -> None:
                return None

            return unsubscribe

    class FakeSessionManagerForHandler:
        def __init__(self) -> None:
            self.workspace_dir = "."
            self.managed = SimpleNamespace(
                session=FakeManagedSessionRuntime(),
                lock=asyncio.Lock(),
                last_used_at=0.0,
            )
            self.active_user_id: str | None = None
            self.complete_calls = 0

        def workspace_dir_for_group(self, group_id: str) -> str:
            return self.workspace_dir

        async def inspect_interaction(self, group_id: str):
            return SimpleNamespace(
                is_active=self.active_user_id is not None,
                active_user_id=self.active_user_id,
                is_streaming=self.managed.session.is_processing,
                managed=self.managed,
            )

        async def reserve_interaction(self, group_id: str, user_id: str):
            if self.active_user_id is None:
                self.active_user_id = user_id
                return SimpleNamespace(action="start", managed=self.managed, active_user_id=user_id)
            if self.active_user_id == user_id and self.managed.session.is_processing:
                return SimpleNamespace(action="steer", managed=self.managed, active_user_id=user_id)
            return SimpleNamespace(action="busy", managed=self.managed, active_user_id=self.active_user_id)

        async def complete_interaction(self, group_id: str) -> None:
            self.complete_calls += 1
            self.active_user_id = None

    session_manager = FakeSessionManagerForHandler()
    config = BampiChatConfig()
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 42

    first_event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("第一条"))
    first_event.to_me = True
    first_matcher = FakeMatcher()
    await handler(bot, first_event, first_matcher)

    assert session_manager.active_user_id is None
    assert session_manager.complete_calls == 1
    assert first_matcher.sent == []

    second_event = FakeGroupEvent(group_id=1001, user_id=42, message_id=100, message=Message("第二条"))
    second_event.to_me = True
    second_matcher = FakeMatcher()
    await handler(bot, second_event, second_matcher)

    assert session_manager.complete_calls == 2
    assert second_matcher.sent == []


@pytest.mark.asyncio
async def test_register_handlers_rejects_group_outside_whitelist(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)

    class FakeSessionManagerForWhitelist:
        def __init__(self) -> None:
            self.workspace_dir_calls: list[str] = []
            self.inspect_calls: list[str] = []
            self.reserve_calls: list[tuple[str, str]] = []

        def workspace_dir_for_group(self, group_id: str) -> str:
            self.workspace_dir_calls.append(group_id)
            return "."

        async def inspect_interaction(self, group_id: str):
            self.inspect_calls.append(group_id)
            return SimpleNamespace(is_active=False, is_streaming=False, managed=None, active_user_id=None)

        async def reserve_interaction(self, group_id: str, user_id: str):
            self.reserve_calls.append((group_id, user_id))
            raise AssertionError("unexpected reserve_interaction call")

    session_manager = FakeSessionManagerForWhitelist()
    config = BampiChatConfig(bampi_group_whitelist=["1002"])
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 42

    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("@bot 帮我写个脚本"))
    event.to_me = True
    matcher = FakeMatcher()
    await handler(bot, event, matcher)

    assert matcher.sent == []
    assert bot.calls == []
    assert session_manager.workspace_dir_calls == []
    assert session_manager.inspect_calls == []
    assert session_manager.reserve_calls == []


@pytest.mark.asyncio
async def test_register_handlers_accepts_group_inside_whitelist(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class CapturingMatcherRegistration:
        def handle(self):
            def decorator(func):
                captured["handler"] = func
                return func

            return decorator

    monkeypatch.setattr(handler_module, "on_message", lambda **kwargs: CapturingMatcherRegistration())
    monkeypatch.setattr(handler_module, "GroupMessageEvent", FakeGroupEvent)
    monkeypatch.setattr(handler_module, "collect_incoming_media", lambda *args, **kwargs: asyncio.sleep(0, result=IncomingMedia()))
    monkeypatch.setattr(handler_module, "snapshot_outbox", lambda workspace_dir: {})
    monkeypatch.setattr(
        handler_module,
        "send_agent_response",
        lambda **kwargs: asyncio.sleep(0, result=ResponseDispatchResult(delivered=True)),
    )

    class FakeManagedSessionRuntime:
        def __init__(self) -> None:
            self.is_processing = False
            self.messages = [AssistantMessage(content=[TextContent(text="ok")])]
            self.session_manager = SimpleNamespace(leaf_id=None)

        async def prompt(self, user_message, *, source: str) -> None:
            self.is_processing = True
            await asyncio.sleep(0)
            self.is_processing = False

        def subscribe(self, listener):
            def unsubscribe() -> None:
                return None

            return unsubscribe

    class FakeSessionManagerForAllowedGroup:
        def __init__(self) -> None:
            self.workspace_dir = "."
            self.workspace_dir_calls: list[str] = []
            self.managed = SimpleNamespace(
                session=FakeManagedSessionRuntime(),
                lock=asyncio.Lock(),
                last_used_at=0.0,
            )
            self.active_user_id: str | None = None
            self.complete_calls = 0

        def workspace_dir_for_group(self, group_id: str) -> str:
            self.workspace_dir_calls.append(group_id)
            return self.workspace_dir

        async def inspect_interaction(self, group_id: str):
            return SimpleNamespace(
                is_active=self.active_user_id is not None,
                active_user_id=self.active_user_id,
                is_streaming=self.managed.session.is_processing,
                managed=self.managed,
            )

        async def reserve_interaction(self, group_id: str, user_id: str):
            if self.active_user_id is None:
                self.active_user_id = user_id
                return SimpleNamespace(action="start", managed=self.managed, active_user_id=user_id)
            if self.active_user_id == user_id and self.managed.session.is_processing:
                return SimpleNamespace(action="steer", managed=self.managed, active_user_id=user_id)
            return SimpleNamespace(action="busy", managed=self.managed, active_user_id=self.active_user_id)

        async def complete_interaction(self, group_id: str) -> None:
            self.complete_calls += 1
            self.active_user_id = None

    session_manager = FakeSessionManagerForAllowedGroup()
    config = BampiChatConfig(bampi_group_whitelist=["1001"])
    handler_module.register_handlers(config, session_manager)
    handler = captured["handler"]

    bot = FakeBot()
    bot.self_id = 42

    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99, message=Message("第一条"))
    event.to_me = True
    matcher = FakeMatcher()
    await handler(bot, event, matcher)

    assert session_manager.workspace_dir_calls == ["1001"]
    assert session_manager.complete_calls == 1
    assert matcher.sent == []


class FakeAgentSessionForBackgroundExit:
    def __init__(self, *, processing: bool) -> None:
        self._processing = processing
        self.steered: list[object] = []
        self.followed_up: list[object] = []
        self.continued = 0
        self.messages: list[object] = []

    @property
    def is_processing(self) -> bool:
        return self._processing

    def steer(self, message) -> None:  # noqa: ANN001
        self.steered.append(message)

    def follow_up(self, message) -> None:  # noqa: ANN001
        self.followed_up.append(message)

    def has_queued_messages(self) -> bool:
        return bool(self.steered or self.followed_up) and self.continued == 0

    async def continue_(self) -> None:
        self.continued += 1
        self.messages.append(AssistantMessage(content=[TextContent(text="后台任务已处理完成")]))


def _make_background_exit_event(session_id: str = "term-1"):
    from bampi.plugins.bampi_chat.tools.safe_bash import BackgroundSessionExitEvent

    return BackgroundSessionExitEvent(
        session_id=session_id,
        command="uv run pytest",
        cwd_display="/workspace",
        returncode=0,
        log_path="/workspace/.bampi/logs/term-1.log",
        output_text="all tests passed",
        notify_on_exit=True,
        total_output_bytes=16,
    )


@pytest.mark.asyncio
async def test_background_exit_handler_steers_into_active_turn(tmp_path: Path):
    session = FakeAgentSessionForBackgroundExit(processing=True)
    managed = SimpleNamespace(
        group_id="1001",
        session=session,
        background_task_context={},
        lock=asyncio.Lock(),
        last_used_at=time.monotonic(),
    )
    session_manager = SimpleNamespace(workspace_dir_for_group=lambda group_id: str(tmp_path))

    handler = handler_module.create_background_exit_handler(BampiChatConfig(), session_manager)
    await handler(managed, _make_background_exit_event())

    assert len(session.steered) == 1
    steered_text = session.steered[0].content[0].text
    assert "term-1" in steered_text
    assert "all tests passed" in steered_text
    assert session.continued == 0


@pytest.mark.asyncio
async def test_background_exit_handler_runs_resume_turn_when_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    session = FakeAgentSessionForBackgroundExit(processing=False)
    origin = handler_module.BackgroundTaskOrigin(bot_self_id="42", user_id=7, reply_message_id=99)
    managed = SimpleNamespace(
        group_id="1001",
        session=session,
        background_task_context={"term-1": origin},
        lock=asyncio.Lock(),
        last_used_at=time.monotonic(),
    )
    session_manager = SimpleNamespace(workspace_dir_for_group=lambda group_id: str(tmp_path))
    fake_bot = SimpleNamespace(self_id=42)
    monkeypatch.setattr(handler_module, "get_bot", lambda *args, **kwargs: fake_bot)
    sent: list[dict[str, object]] = []

    async def fake_send_agent_response_to_target(**kwargs):  # noqa: ANN003
        sent.append(kwargs)
        return ResponseDispatchResult(delivered=True)

    monkeypatch.setattr(
        handler_module,
        "send_agent_response_to_target",
        fake_send_agent_response_to_target,
    )

    handler = handler_module.create_background_exit_handler(BampiChatConfig(), session_manager)
    await handler(managed, _make_background_exit_event())

    assert len(session.followed_up) == 1
    assert session.continued == 1
    assert len(sent) == 1
    assert sent[0]["bot"] is fake_bot
    target = sent[0]["target"]
    assert target.group_id == 1001
    assert target.user_id == 7
    assert target.reply_message_id == 99
    assistant_message = sent[0]["assistant_message"]
    assert "后台任务已处理完成" in assistant_message.content[0].text


def test_group_reaction_buffer_dedupes_and_caps(monkeypatch: pytest.MonkeyPatch):
    now = 1000.0
    monkeypatch.setattr(handler_module.time, "monotonic", lambda: now)
    buffer = handler_module.GroupReactionBuffer(ttl_seconds=60.0, max_per_group=2)

    buffer.add("g", dedupe_key="m1:u1", note="旧的")
    buffer.add("g", dedupe_key="m1:u1", note="新的")
    buffer.add("g", dedupe_key="m2:u1", note="第二条")
    buffer.add("g", dedupe_key="m3:u1", note="第三条")

    assert buffer.drain("g") == ["第二条", "第三条"]
    assert buffer.drain("g") == []


def test_group_reaction_buffer_expires_notes(monkeypatch: pytest.MonkeyPatch):
    now = 1000.0
    monkeypatch.setattr(handler_module.time, "monotonic", lambda: now)
    buffer = handler_module.GroupReactionBuffer(ttl_seconds=60.0)
    buffer.add("g", dedupe_key="m1:u1", note="会过期")
    now += 61.0
    assert buffer.drain("g") == []


def test_build_poke_user_message_includes_action_and_reactions():
    message = handler_module.build_poke_user_message(
        sender_name="张三",
        user_id="10001",
        action="拍了拍",
        suffix="的头",
        reaction_notes=["李四(2) 给你的消息「好」贴了表情 [表情:赞]"],
    )
    text = message.content[0].text
    assert "sender_name: 张三(10001)" in text
    assert "message_text: (拍了拍你的头)" in text
    assert "recent_reactions:\n- 李四(2) 给你的消息「好」贴了表情 [表情:赞]" in text


def test_build_user_message_includes_reaction_notes():
    event = FakeGroupEvent(
        group_id=1001,
        user_id=42,
        message_id=99,
        sender=FakeSender(user_id=42, nickname="Alice"),
    )
    message = build_user_message(
        event,
        "继续",
        IncomingMedia(),
        reaction_notes=["Bob(7) 给你的消息「进度如何」贴了表情 👍"],
    )
    assert "recent_reactions:\n- Bob(7) 给你的消息「进度如何」贴了表情 👍" in message.content[0].text


class FakeReactionBot:
    def __init__(self, *, self_id: int, msg_sender_id: int, message_segments: object) -> None:
        self.self_id = self_id
        self._msg_sender_id = msg_sender_id
        self._message_segments = message_segments
        self.api_calls: list[tuple[str, dict[str, object]]] = []

    async def call_api(self, action: str, **params: object) -> dict[str, object]:
        self.api_calls.append((action, params))
        if action == "get_msg":
            return {
                "sender": {"user_id": self._msg_sender_id},
                "message": self._message_segments,
            }
        return {}

    async def get_group_member_info(self, *, group_id: int, user_id: int) -> dict[str, object]:
        return {"card": "", "nickname": "张三"}


@pytest.mark.asyncio
async def test_build_reaction_note_for_bot_message():
    bot = FakeReactionBot(
        self_id=99,
        msg_sender_id=99,
        message_segments=[{"type": "text", "data": {"text": "这是 bot 的回复内容"}}],
    )
    note = await handler_module.build_reaction_note(
        bot=bot,
        group_id="1001",
        user_id="42",
        message_id=555,
        likes=[{"emoji_id": "76", "count": 1}],
        cache=handler_module.MentionNameCache(),
    )
    assert note == "张三(42) 给你的消息「这是 bot 的回复内容」贴了表情 [表情:赞]"


@pytest.mark.asyncio
async def test_build_reaction_note_ignores_non_bot_message():
    bot = FakeReactionBot(
        self_id=99,
        msg_sender_id=7,
        message_segments=[{"type": "text", "data": {"text": "群友的消息"}}],
    )
    note = await handler_module.build_reaction_note(
        bot=bot,
        group_id="1001",
        user_id="42",
        message_id=555,
        likes=[{"emoji_id": "76", "count": 1}],
        cache=handler_module.MentionNameCache(),
    )
    assert note is None


class FakePokeAgentSession:
    def __init__(self, reply_text: str | None = "戳我干嘛，有事直接说。") -> None:
        self.prompt_calls: list[tuple[object, str]] = []
        self.messages: list[object] = []
        self._reply_text = reply_text

    def subscribe(self, listener):
        def unsubscribe() -> None:
            return None

        return unsubscribe

    async def prompt(self, user_message, source: str = "") -> None:
        self.prompt_calls.append((user_message, source))
        content = [TextContent(text=self._reply_text)] if self._reply_text else []
        self.messages.append(AssistantMessage(content=content))


class FakePokeSessionManager:
    def __init__(self, workspace_dir: str, action: str = "start") -> None:
        self.workspace_dir = workspace_dir
        self.action = action
        self.session = FakePokeAgentSession()
        self.managed = SimpleNamespace(
            session=self.session,
            lock=asyncio.Lock(),
            last_used_at=0.0,
        )
        self.completed: list[str] = []
        self.turn_contexts: list[dict[str, object]] = []

    def workspace_dir_for_group(self, group_id: str) -> str:
        return self.workspace_dir

    async def reserve_interaction(self, group_id: str, user_id: str):
        return SimpleNamespace(action=self.action, managed=self.managed)

    async def complete_interaction(self, group_id: str) -> None:
        self.completed.append(group_id)

    def set_qq_turn_context(self, group_id: str, *, bot_self_id: str, user_id: str, message_id: object) -> None:
        self.turn_contexts.append(
            {
                "group_id": group_id,
                "bot_self_id": bot_self_id,
                "user_id": user_id,
                "message_id": message_id,
            }
        )


@pytest.mark.asyncio
async def test_run_poke_reply_turn_prompts_and_replies(tmp_path: Path):
    config = BampiChatConfig(
        bampi_live_progress_enabled=False,
        bampi_live_text_stream_enabled=False,
        bampi_threshold_compaction_notice_enabled=False,
    )
    session_manager = FakePokeSessionManager(str(tmp_path))
    bot = FakeBot()
    bot.self_id = 99

    await handler_module.run_poke_reply_turn(
        bot=bot,
        config=config,
        session_manager=session_manager,
        group_id="1001",
        user_id="42",
        sender_name="张三",
        action="戳了戳",
        suffix="",
        reaction_notes=None,
    )

    assert len(session_manager.session.prompt_calls) == 1
    prompt_message, source = session_manager.session.prompt_calls[0]
    assert source == "qq_group"
    assert "sender_name: 张三(42)" in prompt_message.content[0].text
    assert "message_text: (戳了戳你)" in prompt_message.content[0].text

    send_calls = [call for call in bot.calls if call[0] == "send_group_msg"]
    assert len(send_calls) == 1
    sent_message = send_calls[0][1]["message"]
    assert "戳我干嘛" in str(sent_message)
    assert session_manager.completed == ["1001"]
    assert session_manager.turn_contexts == [
        {"group_id": "1001", "bot_self_id": "99", "user_id": "42", "message_id": None}
    ]


@pytest.mark.asyncio
async def test_run_poke_reply_turn_stays_silent_on_empty_reply(tmp_path: Path):
    config = BampiChatConfig(
        bampi_live_progress_enabled=False,
        bampi_live_text_stream_enabled=False,
        bampi_threshold_compaction_notice_enabled=False,
    )
    session_manager = FakePokeSessionManager(str(tmp_path))
    session_manager.session._reply_text = None
    bot = FakeBot()
    bot.self_id = 99

    await handler_module.run_poke_reply_turn(
        bot=bot,
        config=config,
        session_manager=session_manager,
        group_id="1001",
        user_id="42",
        sender_name="张三",
        action="戳了戳",
        suffix="",
    )

    assert [call for call in bot.calls if call[0] == "send_group_msg"] == []
    assert session_manager.completed == ["1001"]


class CapturingMatcherRegistrationForNotices:
    def __init__(self, store: list) -> None:
        self._store = store

    def handle(self):
        def decorator(func):
            self._store.append(func)
            return func

        return decorator


def _register_with_captured_notices(monkeypatch: pytest.MonkeyPatch, config, session_manager):
    notice_handlers: list = []
    monkeypatch.setattr(
        handler_module,
        "on_message",
        lambda **kwargs: CapturingMatcherRegistrationForNotices([]),
    )
    monkeypatch.setattr(
        handler_module,
        "on_notice",
        lambda **kwargs: CapturingMatcherRegistrationForNotices(notice_handlers),
    )
    handler_module.register_handlers(config, session_manager)
    assert len(notice_handlers) == 2
    return notice_handlers


@pytest.mark.asyncio
async def test_poke_notice_triggers_reply_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = BampiChatConfig()
    session_manager = FakePokeSessionManager(str(tmp_path))
    session_manager.inspect_interaction = lambda group_id: asyncio.sleep(
        0, result=SimpleNamespace(is_active=False)
    )
    poke_handler, _ = _register_with_captured_notices(monkeypatch, config, session_manager)

    runs: list[dict[str, object]] = []

    async def fake_run_poke_reply_turn(**kwargs):  # noqa: ANN003
        runs.append(kwargs)

    monkeypatch.setattr(handler_module, "run_poke_reply_turn", fake_run_poke_reply_turn)

    async def fake_resolve(bot, *, group_id, user_id, cache):
        return "张三"

    monkeypatch.setattr(handler_module, "resolve_member_display_name", fake_resolve)

    bot = FakeBot()
    bot.self_id = 99
    event = SimpleNamespace(
        group_id=1001,
        user_id=42,
        target_id=99,
        raw_info=[{"type": "nor", "txt": "拍了拍"}, {"type": "nor", "txt": "的头"}],
    )
    await poke_handler(bot, event)

    assert len(runs) == 1
    assert runs[0]["group_id"] == "1001"
    assert runs[0]["user_id"] == "42"
    assert runs[0]["sender_name"] == "张三"
    assert runs[0]["action"] == "拍了拍"
    assert runs[0]["suffix"] == "的头"


@pytest.mark.asyncio
async def test_poke_notice_ignores_pokes_not_targeting_bot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = BampiChatConfig()
    session_manager = FakePokeSessionManager(str(tmp_path))
    poke_handler, _ = _register_with_captured_notices(monkeypatch, config, session_manager)

    runs: list[dict[str, object]] = []

    async def fake_run_poke_reply_turn(**kwargs):  # noqa: ANN003
        runs.append(kwargs)

    monkeypatch.setattr(handler_module, "run_poke_reply_turn", fake_run_poke_reply_turn)

    bot = FakeBot()
    bot.self_id = 99
    event = SimpleNamespace(group_id=1001, user_id=42, target_id=7, raw_info=None)
    await poke_handler(bot, event)

    assert runs == []


@pytest.mark.asyncio
async def test_reaction_notice_buffers_note_for_bot_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = BampiChatConfig()
    session_manager = FakePokeSessionManager(str(tmp_path))

    class RecordingReactionBuffer:
        instances: list["RecordingReactionBuffer"] = []

        def __init__(self, **kwargs) -> None:
            self.added: list[tuple[str, str, str]] = []
            RecordingReactionBuffer.instances.append(self)

        def add(self, group_id: str, *, dedupe_key: str, note: str) -> None:
            self.added.append((group_id, dedupe_key, note))

        def drain(self, group_id: str) -> list[str]:
            return []

    monkeypatch.setattr(handler_module, "GroupReactionBuffer", RecordingReactionBuffer)
    _, reaction_handler = _register_with_captured_notices(monkeypatch, config, session_manager)
    buffer = RecordingReactionBuffer.instances[-1]

    bot = FakeReactionBot(
        self_id=99,
        msg_sender_id=99,
        message_segments=[{"type": "text", "data": {"text": "bot 的回复"}}],
    )
    event = SimpleNamespace(
        notice_type="group_msg_emoji_like",
        group_id=1001,
        user_id=42,
        message_id=555,
        likes=[{"emoji_id": "76", "count": 1}],
        is_add=True,
    )
    await reaction_handler(bot, event)

    assert len(buffer.added) == 1
    group_id, dedupe_key, note = buffer.added[0]
    assert group_id == "1001"
    assert dedupe_key == "555:42"
    assert "张三(42)" in note
    assert "「bot 的回复」" in note
    assert "[表情:赞]" in note


@pytest.mark.asyncio
async def test_reaction_notice_ignores_removed_reactions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = BampiChatConfig()
    session_manager = FakePokeSessionManager(str(tmp_path))

    class RecordingReactionBuffer:
        instances: list["RecordingReactionBuffer"] = []

        def __init__(self, **kwargs) -> None:
            self.added: list[tuple[str, str, str]] = []
            RecordingReactionBuffer.instances.append(self)

        def add(self, group_id: str, *, dedupe_key: str, note: str) -> None:
            self.added.append((group_id, dedupe_key, note))

        def drain(self, group_id: str) -> list[str]:
            return []

    monkeypatch.setattr(handler_module, "GroupReactionBuffer", RecordingReactionBuffer)
    _, reaction_handler = _register_with_captured_notices(monkeypatch, config, session_manager)
    buffer = RecordingReactionBuffer.instances[-1]

    bot = FakeReactionBot(self_id=99, msg_sender_id=99, message_segments=[])
    event = SimpleNamespace(
        notice_type="group_msg_emoji_like",
        group_id=1001,
        user_id=42,
        message_id=555,
        likes=[{"emoji_id": "76", "count": 1}],
        is_add=False,
    )
    await reaction_handler(bot, event)

    assert buffer.added == []


@pytest.mark.asyncio
async def test_live_progress_reporter_stays_silent_for_qq_react_tool():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(bampi_live_progress_enabled=True)
    reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None
    session.listener(
        SimpleNamespace(
            type="tool_execution_start",
            tool_name="qq_react",
            tool_call_id="call-1",
            args={"action": "emoji", "emoji": "赞"},
        )
    )
    session.listener(
        SimpleNamespace(
            type="tool_execution_end",
            tool_name="qq_react",
            tool_call_id="call-1",
            is_error=False,
        )
    )
    await reporter.prepare_final_reply()
    await reporter.close()

    assert bot.calls == []
