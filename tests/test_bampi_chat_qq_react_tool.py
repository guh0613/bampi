from __future__ import annotations

from pathlib import Path

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.message_render import resolve_reaction_emoji_id
from bampi.plugins.bampi_chat.tools import create_agent_tools
from bampi.plugins.bampi_chat.tools import qq_react as qq_react_module
from bampi.plugins.bampi_chat.tools.qq_react import (
    NO_CONTEXT_MESSAGE,
    NO_MESSAGE_TARGET_MESSAGE,
    QQReactTool,
    QQReactToolInput,
    QQTurnContext,
)


class FakeApiBot:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_api(self, action: str, **params: object) -> dict[str, object]:
        self.calls.append((action, params))
        if self.fail:
            raise RuntimeError("napcat unavailable")
        return {}


def make_tool(context: QQTurnContext | None, bot: FakeApiBot, monkeypatch: pytest.MonkeyPatch) -> QQReactTool:
    monkeypatch.setattr(qq_react_module, "get_bot", lambda self_id=None: bot)
    return QQReactTool(context_provider=lambda: context)


def test_resolve_reaction_emoji_id_variants():
    assert resolve_reaction_emoji_id("赞") == 76
    assert resolve_reaction_emoji_id("doge") == 179
    assert resolve_reaction_emoji_id("DOGE") == 179
    assert resolve_reaction_emoji_id("666") == 356
    assert resolve_reaction_emoji_id("76") == 76
    assert resolve_reaction_emoji_id("👍") == 128077
    assert resolve_reaction_emoji_id("[表情:赞]") == 76
    assert resolve_reaction_emoji_id("不存在的表情名") is None
    assert resolve_reaction_emoji_id("") is None


def test_input_requires_emoji_for_emoji_action():
    with pytest.raises(ValueError):
        QQReactToolInput(action="emoji")
    with pytest.raises(ValueError):
        QQReactToolInput(action="poke", user_id="not-a-number")
    assert QQReactToolInput(action="poke").user_id is None


@pytest.mark.asyncio
async def test_emoji_action_reacts_to_turn_message(monkeypatch: pytest.MonkeyPatch):
    bot = FakeApiBot()
    context = QQTurnContext(bot_self_id="99", group_id="1001", user_id="42", message_id=555)
    tool = make_tool(context, bot, monkeypatch)

    result = await tool.execute("call-1", {"action": "emoji", "emoji": "赞"})

    assert bot.calls == [("set_msg_emoji_like", {"message_id": 555, "emoji_id": 76, "set": True})]
    assert "已给消息贴上表情 [表情:赞]" in result.content[0].text


@pytest.mark.asyncio
async def test_emoji_action_accepts_unicode_emoji(monkeypatch: pytest.MonkeyPatch):
    bot = FakeApiBot()
    context = QQTurnContext(bot_self_id="99", group_id="1001", user_id="42", message_id=555)
    tool = make_tool(context, bot, monkeypatch)

    await tool.execute("call-1", {"action": "emoji", "emoji": "👍"})

    assert bot.calls == [("set_msg_emoji_like", {"message_id": 555, "emoji_id": 128077, "set": True})]


@pytest.mark.asyncio
async def test_poke_action_defaults_to_turn_sender(monkeypatch: pytest.MonkeyPatch):
    bot = FakeApiBot()
    context = QQTurnContext(bot_self_id="99", group_id="1001", user_id="42", message_id=None)
    tool = make_tool(context, bot, monkeypatch)

    result = await tool.execute("call-1", {"action": "poke"})

    assert bot.calls == [("group_poke", {"group_id": 1001, "user_id": 42})]
    assert "已戳一戳 42" in result.content[0].text


@pytest.mark.asyncio
async def test_poke_action_accepts_explicit_target(monkeypatch: pytest.MonkeyPatch):
    bot = FakeApiBot()
    context = QQTurnContext(bot_self_id="99", group_id="1001", user_id="42", message_id=None)
    tool = make_tool(context, bot, monkeypatch)

    await tool.execute("call-1", {"action": "poke", "user_id": "10001"})

    assert bot.calls == [("group_poke", {"group_id": 1001, "user_id": 10001})]


@pytest.mark.asyncio
async def test_gracefully_handles_missing_context(monkeypatch: pytest.MonkeyPatch):
    bot = FakeApiBot()
    tool = make_tool(None, bot, monkeypatch)

    result = await tool.execute("call-1", {"action": "emoji", "emoji": "赞"})

    assert bot.calls == []
    assert result.content[0].text == NO_CONTEXT_MESSAGE


@pytest.mark.asyncio
async def test_emoji_action_requires_message_target(monkeypatch: pytest.MonkeyPatch):
    bot = FakeApiBot()
    context = QQTurnContext(bot_self_id="99", group_id="1001", user_id="42", message_id=None)
    tool = make_tool(context, bot, monkeypatch)

    result = await tool.execute("call-1", {"action": "emoji", "emoji": "赞"})

    assert bot.calls == []
    assert result.content[0].text == NO_MESSAGE_TARGET_MESSAGE


@pytest.mark.asyncio
async def test_emoji_action_reports_unknown_emoji(monkeypatch: pytest.MonkeyPatch):
    bot = FakeApiBot()
    context = QQTurnContext(bot_self_id="99", group_id="1001", user_id="42", message_id=555)
    tool = make_tool(context, bot, monkeypatch)

    result = await tool.execute("call-1", {"action": "emoji", "emoji": "不存在的表情名"})

    assert bot.calls == []
    assert "无法识别表情" in result.content[0].text


@pytest.mark.asyncio
async def test_api_failure_returns_readable_error(monkeypatch: pytest.MonkeyPatch):
    bot = FakeApiBot(fail=True)
    context = QQTurnContext(bot_self_id="99", group_id="1001", user_id="42", message_id=555)
    tool = make_tool(context, bot, monkeypatch)

    result = await tool.execute("call-1", {"action": "emoji", "emoji": "赞"})

    assert "贴表情失败" in result.content[0].text


def test_create_agent_tools_registers_qq_react_with_provider(tmp_path: Path):
    config = BampiChatConfig(bampi_bash_mode="local")
    context = QQTurnContext(bot_self_id="99", group_id="1001", user_id="42", message_id=1)

    tools = create_agent_tools(
        config,
        str(tmp_path),
        group_id="1001",
        qq_turn_context_provider=lambda: context,
    )
    assert "qq_react" in [tool.name for tool in tools]

    without_provider = create_agent_tools(config, str(tmp_path), group_id="1001")
    assert "qq_react" not in [tool.name for tool in without_provider]

    disabled_config = BampiChatConfig(bampi_bash_mode="local", bampi_qq_react_tool_enabled=False)
    disabled = create_agent_tools(
        disabled_config,
        str(tmp_path),
        group_id="1001",
        qq_turn_context_provider=lambda: context,
    )
    assert "qq_react" not in [tool.name for tool in disabled]
