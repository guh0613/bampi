"""戳一戳与表情回应：非消息触发的 agent 回合。"""

from __future__ import annotations

import time
from typing import Any

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot

from bampy.ai import TextContent, UserMessage

from ..config import BampiChatConfig
from ..feedback import assess_failure, build_reply_failure_message
from ..message_render import (
    MentionNameCache,
    describe_reaction_emojis,
    render_message_text,
    resolve_member_display_name,
)
from ..session_manager import GroupSessionManager
from .background import BackgroundTaskOrigin, subscribe_background_task_origins
from .outbound import (
    GroupReplyTarget,
    _send_group_message_via_bot,
    build_group_reply_message,
    find_last_assistant_message,
    send_agent_response_to_target,
    snapshot_outbox,
)
from .progress import LiveProgressReporter
from .utils import normalize_text, update_qq_turn_context


class ThinkingReactionIndicator:
    """A best-effort transient reaction shown while one message turn is running."""

    def __init__(
        self,
        *,
        bot: Bot,
        group_id: str,
        message_id: int,
        emoji_id: int,
    ) -> None:
        self._bot = bot
        self._group_id = group_id
        self._message_id = message_id
        self._emoji_id = emoji_id
        self._cleanup_required = False

    async def show(self) -> None:
        # Mark cleanup necessary before the request: if the API applies the
        # reaction but the response is lost, the finally path still removes it.
        self._cleanup_required = True
        try:
            await self._set_reaction(set_reaction=True)
        except Exception as exc:
            logger.warning(
                f"bampi_chat failed to show thinking reaction group_id={self._group_id} "
                f"message_id={self._message_id} emoji_id={self._emoji_id} error={exc}"
            )

    async def close(self) -> None:
        if not self._cleanup_required:
            return
        try:
            await self._set_reaction(set_reaction=False)
        except Exception as exc:
            logger.warning(
                f"bampi_chat failed to clear thinking reaction group_id={self._group_id} "
                f"message_id={self._message_id} emoji_id={self._emoji_id} error={exc}"
            )
        else:
            self._cleanup_required = False

    async def _set_reaction(self, *, set_reaction: bool) -> None:
        await self._bot.call_api(
            "set_msg_emoji_like",
            message_id=self._message_id,
            emoji_id=self._emoji_id,
            set=set_reaction,
        )


def build_poke_user_message(
    *,
    sender_name: str,
    user_id: str,
    action: str,
    suffix: str,
    reaction_notes: list[str] | None = None,
) -> UserMessage:
    lines = [
        f"sender_name: {sender_name}({user_id})",
        f"message_text: ({action}你{suffix})",
    ]
    if reaction_notes:
        lines.append("recent_reactions:")
        lines.extend(f"- {note}" for note in reaction_notes)
    return UserMessage(content=[TextContent(text="\n".join(lines))])


async def build_reaction_note(
    *,
    bot: Bot,
    group_id: str,
    user_id: str,
    message_id: Any,
    likes: Any,
    cache: MentionNameCache,
) -> str | None:
    """为贴在 bot 消息上的表情回应生成上下文说明；非 bot 消息返回 None。"""
    emoji_text = describe_reaction_emojis(likes)
    if not emoji_text:
        return None
    try:
        info = await bot.call_api("get_msg", message_id=int(message_id))
    except Exception as exc:
        logger.debug(
            f"bampi_chat reaction get_msg failed group_id={group_id} "
            f"message_id={message_id} error={exc}"
        )
        return None
    if not isinstance(info, dict):
        return None
    sender = info.get("sender")
    sender_id = sender.get("user_id") if isinstance(sender, dict) else None
    if sender_id is None or str(sender_id) != str(bot.self_id):
        return None
    preview = normalize_text(render_message_text(info.get("message")))
    if len(preview) > 40:
        preview = preview[:39] + "…"
    reactor = await resolve_member_display_name(bot, group_id=group_id, user_id=user_id, cache=cache)
    reactor_label = f"{reactor}({user_id})" if reactor else user_id
    quoted = f"「{preview}」" if preview else ""
    return f"{reactor_label} 给你的消息{quoted}贴了表情 {emoji_text}"


async def run_poke_reply_turn(
    *,
    bot: Bot,
    config: BampiChatConfig,
    session_manager: GroupSessionManager,
    group_id: str,
    user_id: str,
    sender_name: str,
    action: str,
    suffix: str,
    reaction_notes: list[str] | None = None,
) -> None:
    """戳一戳 bot 时以完整 agent 交互回应（无引用目标，仅按配置 @ 发起者）。"""
    try:
        reservation = await session_manager.reserve_interaction(group_id, user_id)
    except Exception:
        logger.exception("bampi_chat failed to reserve session for poke")
        return
    if reservation.action != "start":
        logger.info(
            f"bampi_chat skipped poke reservation group_id={group_id} "
            f"user_id={user_id} action={reservation.action}"
        )
        return
    managed = reservation.managed
    workspace_dir = session_manager.workspace_dir_for_group(group_id)
    target = GroupReplyTarget(group_id=int(group_id), user_id=int(user_id))
    try:
        user_message = build_poke_user_message(
            sender_name=sender_name,
            user_id=user_id,
            action=action,
            suffix=suffix,
            reaction_notes=reaction_notes,
        )
        prepare_memory_turn = getattr(session_manager, "prepare_memory_for_user_turn", None)
        if callable(prepare_memory_turn):
            prepare_memory_turn(managed, user_id=user_id, nickname=sender_name, message=user_message)
        update_qq_turn_context(
            session_manager,
            group_id,
            bot_self_id=str(bot.self_id),
            user_id=user_id,
            message_id=None,
        )
        outbox_before = snapshot_outbox(workspace_dir)
        async with managed.lock:
            managed.last_used_at = time.monotonic()
            unsubscribe_background_origin = subscribe_background_task_origins(
                managed,
                origin=BackgroundTaskOrigin(bot_self_id=str(bot.self_id), user_id=int(user_id)),
            )
            reporter = LiveProgressReporter(bot=bot, target=target, config=config)
            reporter.start(managed.session)
            try:
                try:
                    await managed.session.prompt(user_message, source="qq_group")
                except Exception as exc:
                    logger.exception("bampi_chat poke prompt failed")
                    await _send_group_message_via_bot(
                        bot=bot,
                        target=target,
                        message=build_group_reply_message(
                            config=config,
                            target=target,
                            text=build_reply_failure_message(assess_failure(str(exc))),
                        ),
                    )
                    return
                managed.last_used_at = time.monotonic()
                await reporter.prepare_final_reply()
                assistant_message = find_last_assistant_message(managed.session.messages)
                await send_agent_response_to_target(
                    bot=bot,
                    target=target,
                    config=config,
                    workspace_dir=workspace_dir,
                    assistant_message=assistant_message,
                    outbox_before=outbox_before,
                    intermediate_text_sent=reporter.intermediate_text_sent,
                    quote_reply=not reporter.visible_update_sent,
                    log_label="poke",
                    failure_message_builder=build_reply_failure_message,
                    empty_reply_text=None,
                )
            finally:
                unsubscribe_background_origin()
                await reporter.close()
    except Exception:
        logger.exception("bampi_chat failed to deliver poke reply")
    finally:
        await session_manager.complete_interaction(group_id)
