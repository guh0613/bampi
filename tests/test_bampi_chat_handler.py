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
    IncomingMedia,
    LiveProgressReporter,
    ResponseDispatchResult,
    TriggerDecision,
    build_user_message,
    collect_incoming_media,
    format_tool_progress_message,
    is_stop_command,
    matched_prefix,
    prepare_group_file_upload,
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


def test_matched_prefix_returns_first_match():
    assert matched_prefix("@bot hello", ["@bot", "/bot"]) == "@bot"


def test_is_stop_command_accepts_normalized_command():
    assert is_stop_command("/stop") is True
    assert is_stop_command("  /STOP  ") is True
    assert is_stop_command("/stop now") is False


def test_format_tool_progress_message_uses_emoji_style():
    message = format_tool_progress_message("read", {"path": "README.md"})

    assert message == "📖 正在读取：README.md"
    assert "进度：" not in message
    assert "`" not in message


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
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
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
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
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
    assert str(params["message"]) == "[CQ:reply,id=99]上下文有点长，我先整理一下前面的聊天记录，再继续。"


@pytest.mark.asyncio
async def test_live_progress_reporter_sends_emoji_tool_update():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(bampi_live_progress_enabled=True)
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
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
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
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
        f"安装目录：{(Path(manager.workspace_dir_for_group('1001')) / '.agents' / 'skills').resolve().as_posix()}\n"
        "显式调用：在普通消息最开头写 `/skill-name`。\n"
        "当前群会话已刷新；其他现有会话会在下次重建后看到新 skill。"
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
        "你发送的url有误。\n"
        "请直接发送或引用 skill 压缩包/Markdown 文件后执行 `/skill install`，"
        "或使用 `/skill install https://...`。"
    ]


@pytest.mark.asyncio
async def test_live_progress_reporter_allows_unlimited_tool_updates_when_limit_zero():
    bot = FakeBot()
    event = FakeGroupEvent(group_id=1001, user_id=42, message_id=99)
    config = BampiChatConfig(
        bampi_live_progress_enabled=True,
        bampi_live_progress_max_tool_updates=0,
    )
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
    session = FakeSession()

    reporter.start(session)
    assert session.listener is not None
    for i, (name, args) in enumerate([
        ("write", {"path": "a.py"}),
        ("bash", {"command": "python3 a.py"}),
        ("ls", {"path": "/workspace/outbox"}),
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
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
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
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
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
    session.listener(SimpleNamespace(type="tool_execution_start", tool_name="ls", args={"path": "inbox"}, tool_call_id="tc1"))
    session.listener(SimpleNamespace(type="tool_execution_end", tool_name="ls", tool_call_id="tc1", is_error=False, result=None))

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
        "📂 正在查看目录：inbox",
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
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
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
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
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
    reporter = LiveProgressReporter(bot=bot, event=event, config=config)
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

    assert message.content[0].text.startswith("group_id: 1001")
    assert "message_text: (无纯文本内容；本条消息仅包含媒体/文件)" in message.content[0].text
    assert "workspace_attachments:\n- inbox/report.txt" in message.content[0].text


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
    assert "inline_image_count: 1" in text_block
    assert "workspace_attachments:\n- inbox/current.txt" in text_block
    assert "media_notes:\n- 当前备注" in text_block
    assert "reply_inline_image_count: 1" in text_block
    assert "reply_workspace_attachments:\n- inbox/reply.pdf" in text_block
    assert "reply_media_notes:\n- 回复备注" in text_block
    assert len(message.content) == 3
    assert message.content[1].mime_type == "image/png"
    assert message.content[2].mime_type == "image/jpeg"


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
