"""qq_react 工具：让模型对当前轮次做轻量 QQ 互动（贴表情 / 戳一戳）。

设计上不向模型暴露 message_id：贴表情固定作用于触发本轮对话的那条
消息（即时反应场景），目标信息由 handler 在每轮开始时写入
`QQTurnContext`，工具执行时通过 provider 读取。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from nonebot import get_bot, logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import TextContent

from ..message_render import describe_reaction_emoji, resolve_reaction_emoji_id


@dataclass(slots=True)
class QQTurnContext:
    """当前轮次的 QQ 消息上下文，供 qq_react 定位互动目标。"""

    bot_self_id: str
    group_id: str
    user_id: str
    message_id: int | None = None


QQTurnContextProvider = Callable[[], "QQTurnContext | None"]

QQReactAction = Literal["emoji", "poke"]

NO_CONTEXT_MESSAGE = "当前轮次没有可互动的群消息上下文（例如定时任务或后台恢复轮次），跳过。"
NO_MESSAGE_TARGET_MESSAGE = "本轮不是由群消息触发（例如戳一戳），没有可贴表情的目标消息。"
NO_BOT_MESSAGE = "当前没有已连接的 QQ Bot，无法执行互动。"


class QQReactToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: QQReactAction = Field(
        description="`emoji` 给触发本轮对话的那条消息贴表情；`poke` 戳一戳群成员。"
    )
    emoji: str | None = Field(
        default=None,
        description="action=emoji 必填：表情名（如 赞、doge、笑哭）、emoji 字符（如 👍）或 QQ 表情 ID。",
        max_length=32,
    )
    user_id: str | None = Field(
        default=None,
        description="action=poke 可选：要戳的群成员 QQ 号，默认戳本轮发消息的群友。",
        max_length=20,
    )

    @model_validator(mode="after")
    def _validate_action_requirements(self) -> "QQReactToolInput":
        if self.action == "emoji" and not (self.emoji or "").strip():
            raise ValueError("emoji action requires emoji")
        if self.user_id is not None and not self.user_id.strip().isdigit():
            raise ValueError("user_id must be a QQ number")
        return self


class QQReactTool:
    name = "qq_react"
    label = "qq_react"
    description = (
        "对触发本轮对话的那条消息贴表情（表情回应），或戳一戳群成员。"
        "适合用轻量互动代替整句文字回复，例如表示收到、赞同或回应戳一戳。"
    )
    parameters = QQReactToolInput

    def __init__(self, *, context_provider: QQTurnContextProvider) -> None:
        self._context_provider = context_provider

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        arguments = QQReactToolInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
        )

        context = self._context_provider()
        if context is None:
            return _text_result(NO_CONTEXT_MESSAGE)

        try:
            bot = get_bot(context.bot_self_id)
        except Exception:
            logger.warning(f"qq_react could not resolve bot self_id={context.bot_self_id}")
            return _text_result(NO_BOT_MESSAGE)

        if arguments.action == "emoji":
            return await self._react_with_emoji(bot, context, arguments.emoji or "")
        return await self._poke(bot, context, arguments.user_id)

    async def _react_with_emoji(self, bot: Any, context: QQTurnContext, emoji: str) -> AgentToolResult:
        if context.message_id is None:
            return _text_result(NO_MESSAGE_TARGET_MESSAGE)
        emoji_id = resolve_reaction_emoji_id(emoji)
        if emoji_id is None:
            return _text_result(
                f"无法识别表情 {emoji!r}。可用表情名（如 赞、doge、笑哭）、emoji 字符（如 👍）或 QQ 表情 ID。"
            )
        try:
            await bot.call_api(
                "set_msg_emoji_like",
                message_id=context.message_id,
                emoji_id=emoji_id,
                set=True,
            )
        except Exception as exc:
            logger.warning(
                f"qq_react set_msg_emoji_like failed group_id={context.group_id} "
                f"message_id={context.message_id} emoji_id={emoji_id} error={exc}"
            )
            return _text_result(f"贴表情失败：{exc}")
        rendered = describe_reaction_emoji(emoji_id)
        return _text_result(
            f"已给消息贴上表情 {rendered}。",
            details={"action": "emoji", "message_id": context.message_id, "emoji_id": emoji_id},
        )

    async def _poke(self, bot: Any, context: QQTurnContext, user_id: str | None) -> AgentToolResult:
        target = (user_id or context.user_id or "").strip()
        if not target.isdigit():
            return _text_result(f"无法识别要戳的群成员 {target!r}，请提供 QQ 号。")
        try:
            await bot.call_api(
                "group_poke",
                group_id=int(context.group_id),
                user_id=int(target),
            )
        except Exception as exc:
            logger.warning(
                f"qq_react group_poke failed group_id={context.group_id} "
                f"target={target} error={exc}"
            )
            return _text_result(f"戳一戳失败：{exc}")
        return _text_result(
            f"已戳一戳 {target}。",
            details={"action": "poke", "group_id": context.group_id, "user_id": target},
        )


def _text_result(text: str, *, details: Any = None) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)], details=details)
