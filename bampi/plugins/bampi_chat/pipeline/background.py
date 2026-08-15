"""后台任务回调：notify_on_exit 会话结束后把结果带回群会话。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from nonebot import get_bot, logger
from nonebot.adapters.onebot.v11 import Bot

from bampy.ai import TextContent, UserMessage

from ..config import BampiChatConfig
from ..feedback import assess_failure, build_background_failure_message
from ..session_manager import GroupSessionManager, ManagedGroupSession
from ..tools.safe_bash import BackgroundSessionExitEvent
from .outbound import (
    GroupReplyTarget,
    _send_group_message_via_bot,
    build_group_reply_message,
    find_last_assistant_message,
    send_agent_response_to_target,
    snapshot_outbox,
)
from .utils import clear_qq_turn_context


@dataclass(slots=True)
class BackgroundTaskOrigin:
    """Reply metadata for a notify_on_exit session: who asked, and where."""

    bot_self_id: str
    user_id: int
    reply_message_id: int | None = None


def subscribe_background_task_origins(
    managed: ManagedGroupSession,
    *,
    origin: BackgroundTaskOrigin,
) -> Callable[[], None]:
    """Record the current turn's origin for notify_on_exit sessions it starts.

    This metadata only improves the eventual notification (reply/at target);
    exit delivery itself is driven by the tool-level exit event and works
    without it.
    """

    def _listener(session_event: Any) -> None:
        if getattr(session_event, "type", None) != "tool_execution_end":
            return
        if getattr(session_event, "tool_name", "") != "bash":
            return
        if bool(getattr(session_event, "is_error", False)):
            return
        result = getattr(session_event, "result", None)
        details = getattr(result, "details", None)
        if not isinstance(details, dict):
            return
        session_id = str(details.get("session_id", "")).strip()
        if not session_id or not bool(details.get("notify_on_exit")):
            return
        managed.background_task_context[session_id] = origin

    return managed.session.subscribe(_listener)


def build_background_resume_follow_up_message(
    exit_event: BackgroundSessionExitEvent,
) -> UserMessage:
    lines = [
        "系统通知：你之前启动的后台终端命令已经结束，请继续基于结果完成任务。",
        f"background_session_id: {exit_event.session_id}",
        f"command: {exit_event.command}",
        f"exit_code: {exit_event.returncode}",
        f"working_directory: {exit_event.cwd_display}",
    ]
    if exit_event.log_path:
        lines.append(f"log_path: {exit_event.log_path}")
    lines.extend(
        [
            "captured_output:",
            exit_event.output_text or "(no output)",
            "",
            "如需更多上下文，你仍然可以继续使用 bash 的 status/logs 查看这个后台会话。",
        ]
    )
    return UserMessage(content=[TextContent(text="\n".join(lines))])


def _resolve_background_bot(origin: BackgroundTaskOrigin | None) -> Bot | None:
    if origin is not None and origin.bot_self_id:
        try:
            return get_bot(origin.bot_self_id)
        except (KeyError, ValueError):
            pass
    try:
        return get_bot()
    except (KeyError, ValueError):
        return None


def create_background_exit_handler(
    config: BampiChatConfig,
    session_manager: GroupSessionManager,
) -> Callable[[ManagedGroupSession, BackgroundSessionExitEvent], Awaitable[None]]:
    """Deliver a notify_on_exit session result back into the conversation.

    The group session stays interactive while background tasks run. On exit,
    the result is steered into the turn currently in flight, or — when the
    session is idle — a resume turn runs and its reply is sent to the group.
    """

    async def _handle(managed: ManagedGroupSession, exit_event: BackgroundSessionExitEvent) -> None:
        group_id = str(managed.group_id)
        session = managed.session
        origin = managed.background_task_context.get(exit_event.session_id)
        if not isinstance(origin, BackgroundTaskOrigin):
            origin = None
        follow_up_message = build_background_resume_follow_up_message(exit_event)

        queued_as_steer = False
        if session.is_processing:
            session.steer(follow_up_message)
            queued_as_steer = True
            if session.is_processing:
                # The in-flight turn will pick this up and its handler
                # delivers the combined reply.
                logger.info(
                    f"bampi_chat steered background result into active turn "
                    f"group_id={group_id} session_id={exit_event.session_id} "
                    f"exit_code={exit_event.returncode}"
                )
                return
            # The turn ended while we queued; drive the leftover message below.

        target = GroupReplyTarget(
            group_id=int(group_id),
            user_id=origin.user_id if origin else None,
            reply_message_id=origin.reply_message_id if origin else None,
        )
        workspace_dir = session_manager.workspace_dir_for_group(group_id)
        async with managed.lock:
            if not queued_as_steer:
                session.follow_up(follow_up_message)
            if not session.has_queued_messages():
                # An interleaving turn consumed (and delivered) the result
                # while we waited for the lock.
                return
            managed.last_used_at = time.monotonic()
            clear_qq_turn_context(session_manager, group_id)
            outbox_before = snapshot_outbox(workspace_dir)
            logger.info(
                f"bampi_chat auto-resume start group_id={group_id} "
                f"session_id={exit_event.session_id} "
                f"exit_code={exit_event.returncode}"
            )
            bot = _resolve_background_bot(origin)
            try:
                await session.continue_()
            except Exception as exc:
                logger.exception(
                    f"bampi_chat auto-resume failed group_id={group_id} "
                    f"session_id={exit_event.session_id}"
                )
                if bot is not None:
                    await _send_group_message_via_bot(
                        bot=bot,
                        target=target,
                        message=build_group_reply_message(
                            config=config,
                            target=target,
                            text=build_background_failure_message(assess_failure(str(exc))),
                        ),
                    )
                return

            if bot is None:
                logger.error(
                    f"bampi_chat auto-resume finished but no bot is connected "
                    f"group_id={group_id} session_id={exit_event.session_id}"
                )
                return
            assistant_message = find_last_assistant_message(session.messages)
            await send_agent_response_to_target(
                bot=bot,
                target=target,
                config=config,
                workspace_dir=workspace_dir,
                assistant_message=assistant_message,
                outbox_before=outbox_before,
            )

    return _handle
