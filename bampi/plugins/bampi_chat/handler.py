"""bampi_chat 的 NoneBot 入口：注册群消息/戳一戳/表情回应 matcher 并编排完整回合。

处理链路的各环节位于 pipeline 子包：
trigger（触发判定）→ inbound（入站收集）→ progress（进度汇报）→ outbound（出站投递），
poke（戳一戳回合）与 background（后台任务回调）为旁路入口。
"""

from __future__ import annotations

import random
import time

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, NoticeEvent, PokeNotifyEvent
from nonebot.matcher import Matcher
from nonebot.plugin import on_message, on_notice

from .config import BampiChatConfig
from .feedback import assess_failure, build_reply_failure_message
from .message_render import (
    MentionNameCache,
    collect_mention_names,
    extract_poke_action_texts,
    resolve_member_display_name,
    resolve_reaction_emoji_id,
)
from .pipeline.background import (
    BackgroundTaskOrigin,
    create_background_exit_handler,
    subscribe_background_task_origins,
)
from .pipeline.inbound import build_user_message, collect_incoming_context, display_name
from .pipeline.outbound import (
    find_last_assistant_message,
    reply_target_for_event,
    send_agent_response,
    snapshot_outbox,
)
from .pipeline.poke import ThinkingReactionIndicator, build_reaction_note, run_poke_reply_turn
from .pipeline.progress import LiveProgressReporter
from .pipeline.trigger import (
    ACTIVE_SESSION_WINDING_DOWN_MESSAGE,
    BACKGROUND_TASKS_RUNNING_MESSAGE,
    CLEAR_NO_CONTEXT_MESSAGE,
    CLEARED_SESSION_MESSAGE,
    COMPACT_FORBIDDEN_MESSAGE,
    COMPACT_NO_CONTEXT_MESSAGE,
    STOP_NO_ACTIVE_MESSAGE,
    STOP_NOT_OWNER_MESSAGE,
    GroupRateLimiter,
    GroupReactionBuffer,
    build_stop_success_message,
    interaction_busy_message,
    is_clear_command,
    is_compact_command,
    is_group_allowed,
    is_nonebot_superuser,
    is_stop_command,
    should_respond,
)
from .pipeline.utils import log_preview, normalize_text, summarize_segments, update_qq_turn_context
from .session_manager import GroupSessionManager


def register_handlers(config: BampiChatConfig, session_manager: GroupSessionManager) -> None:
    limiter = GroupRateLimiter(
        config.bampi_rate_limit,
        config.bampi_rate_limit_window_seconds,
    )
    mention_cache = MentionNameCache()
    reaction_buffer = GroupReactionBuffer()
    thinking_reaction_emoji_id = None
    if config.bampi_thinking_reaction_enabled:
        thinking_reaction_emoji_id = resolve_reaction_emoji_id(
            config.bampi_thinking_reaction_emoji
        )
        if thinking_reaction_emoji_id is None:
            logger.warning(
                "bampi_chat thinking reaction disabled because emoji could not be resolved "
                f"emoji={config.bampi_thinking_reaction_emoji!r}"
            )
    set_background_notify_handler = getattr(session_manager, "set_background_notify_handler", None)
    if callable(set_background_notify_handler):
        set_background_notify_handler(create_background_exit_handler(config, session_manager))
    matcher = on_message(priority=10, block=False)

    @matcher.handle()
    async def _handle_group_message(bot: Bot, event: GroupMessageEvent, matcher: Matcher) -> None:
        if not isinstance(event, GroupMessageEvent):
            return

        group_id = str(event.group_id)
        user_id = str(event.user_id)
        if not is_group_allowed(group_id, config):
            logger.info(
                f"bampi_chat ignored unauthorized group group_id={group_id} "
                f"user_id={user_id} "
                f"message_id={event.message_id}"
            )
            return

        raw_text = normalize_text(event.get_plaintext())
        workspace_dir = session_manager.workspace_dir_for_group(group_id)
        logger.info(
            f"bampi_chat received group_id={event.group_id} "
            f"user_id={event.user_id} "
            f"message_id={event.message_id} "
            f"to_me={getattr(event, 'to_me', False)} "
            f"segments={summarize_segments(event.message)} "
            f"text={log_preview(raw_text)!r}"
        )

        if is_clear_command(raw_text):
            status = await session_manager.inspect_interaction(group_id)
            if status.is_active:
                await matcher.send(interaction_busy_message(status, requester_user_id=user_id))
                return
            if status.has_running_background:
                await matcher.send(BACKGROUND_TASKS_RUNNING_MESSAGE)
                return
            cleared = await session_manager.clear_context(group_id)
            await matcher.send(CLEARED_SESSION_MESSAGE if cleared else CLEAR_NO_CONTEXT_MESSAGE)
            return

        if is_compact_command(raw_text):
            if not is_nonebot_superuser(user_id):
                await matcher.send(COMPACT_FORBIDDEN_MESSAGE)
                return
            status = await session_manager.inspect_interaction(group_id)
            if status.is_active:
                await matcher.send(interaction_busy_message(status, requester_user_id=user_id))
                return
            if not await session_manager.has_context(group_id):
                await matcher.send(COMPACT_NO_CONTEXT_MESSAGE)
                return
            try:
                managed = await session_manager.get_or_create(group_id)
                async with managed.lock:
                    result = await managed.session.compact()
            except Exception:
                logger.exception("bampi_chat manual compaction failed")
                await matcher.send("上下文压缩失败，请稍后重试。")
                return
            finally:
                await session_manager.complete_interaction(group_id)

            if result is None:
                await matcher.send(COMPACT_NO_CONTEXT_MESSAGE)
                return

            saved_tokens = result.tokens_before - result.tokens_after
            await matcher.send(
                f"已完成上下文压缩，约减少 {saved_tokens} tokens。"
            )
            return

        if is_stop_command(raw_text):
            status = await session_manager.inspect_interaction(group_id)
            if not status.is_active and not status.has_running_background:
                await matcher.send(STOP_NO_ACTIVE_MESSAGE)
                return
            requester_is_superuser = is_nonebot_superuser(user_id)
            requester_is_owner = (
                status.active_user_id == user_id
                or user_id in status.background_owner_user_ids
            )
            if not requester_is_owner and not requester_is_superuser:
                await matcher.send(STOP_NOT_OWNER_MESSAGE)
                return
            force_stop = requester_is_superuser and not requester_is_owner
            if status.managed is None:
                await matcher.send(ACTIVE_SESSION_WINDING_DOWN_MESSAGE)
                return

            stop_reason = "stopped by superuser" if force_stop else "stopped by session owner"
            stop_result = await session_manager.stop_interaction(
                group_id,
                reason=stop_reason,
            )
            if not stop_result.aborted_streaming and not stop_result.stopped_background_sessions:
                await matcher.send(ACTIVE_SESSION_WINDING_DOWN_MESSAGE)
                return
            logger.info(
                f"bampi_chat stop requested group_id={group_id} "
                f"user_id={user_id} "
                f"message_id={event.message_id} "
                f"force_stop={force_stop} "
                f"aborted_streaming={stop_result.aborted_streaming} "
                f"stopped_background_sessions={stop_result.stopped_background_sessions} "
                f"stopped_background_session_ids={stop_result.stopped_background_session_ids}"
            )
            await matcher.send(
                build_stop_success_message(
                    force=force_stop,
                    aborted_streaming=stop_result.aborted_streaming,
                    stopped_background_sessions=stop_result.stopped_background_sessions,
                )
            )
            return

        active_status = await session_manager.inspect_interaction(group_id)
        try:
            mention_names = await collect_mention_names(
                bot,
                group_id=group_id,
                messages=(event.message, getattr(event.reply, "message", None)),
                cache=mention_cache,
            )
        except Exception:
            logger.exception("bampi_chat failed to collect mention names")
            mention_names = {}
        decision = should_respond(
            event,
            bot_self_id=str(bot.self_id),
            config=config,
            random_value=random.random(),
            resolve_name=mention_names.get,
        )
        if not decision.should_respond:
            logger.info(
                f"bampi_chat ignored group_id={event.group_id} "
                f"message_id={event.message_id} "
                f"reason=no_trigger "
                f"text={log_preview(raw_text)!r}"
            )
            return

        if active_status.is_active and active_status.active_user_id == user_id and active_status.is_streaming:
            logger.info(
                f"bampi_chat owner follow-up accepted group_id={group_id} "
                f"user_id={user_id} "
                f"message_id={event.message_id} "
                f"reason={decision.reason}"
            )
            try:
                media, forwards = await collect_incoming_context(
                    bot,
                    event,
                    config,
                    workspace_dir,
                )
            except Exception:
                logger.exception("bampi_chat failed to collect follow-up message context")
                await matcher.send("⚠️ 获取消息里的附件或合并转发失败，请重发一次。")
                return
            user_message = build_user_message(
                event,
                decision.cleaned_text,
                media,
                forwards=forwards,
                resolve_name=mention_names.get,
                reaction_notes=reaction_buffer.drain(group_id) or None,
            )
            prepare_memory_turn = getattr(session_manager, "prepare_memory_for_user_turn", None)
            if active_status.managed is not None and callable(prepare_memory_turn):
                prepare_memory_turn(
                    active_status.managed,
                    user_id=user_id,
                    nickname=display_name(event.sender),
                    message=user_message,
                )
            update_qq_turn_context(
                session_manager,
                group_id,
                bot_self_id=str(bot.self_id),
                user_id=user_id,
                message_id=int(event.message_id),
            )
            active_status.managed.session.steer(user_message)
            return

        logger.info(
            f"bampi_chat triggered group_id={group_id} "
            f"message_id={event.message_id} "
            f"reason={decision.reason} "
            f"direct={decision.direct} "
            f"cleaned_text={log_preview(decision.cleaned_text)!r}"
        )

        if active_status.is_active:
            logger.info(
                f"bampi_chat rejected concurrent trigger group_id={group_id} "
                f"user_id={user_id} "
                f"message_id={event.message_id} "
                f"active_user_id={active_status.active_user_id} "
                f"is_streaming={active_status.is_streaming}"
            )
            if decision.direct:
                await matcher.send(interaction_busy_message(active_status, requester_user_id=user_id))
            return

        if not limiter.allow(group_id):
            logger.warning(
                f"bampi_chat rate limited group_id={group_id} "
                f"message_id={event.message_id} "
                f"direct={decision.direct}"
            )
            if decision.direct:
                await matcher.send("当前繁忙，请稍后重试。")
            return

        try:
            reservation = await session_manager.reserve_interaction(group_id, user_id)
        except Exception:
            logger.exception("bampi_chat failed to create or restore group session")
            await matcher.send("会话启动失败，请稍后重试。")
            return

        if reservation.action == "busy":
            logger.info(
                f"bampi_chat rejected reservation group_id={group_id} "
                f"user_id={user_id} "
                f"message_id={event.message_id} "
                f"active_user_id={reservation.active_user_id}"
            )
            await matcher.send(interaction_busy_message(await session_manager.inspect_interaction(group_id), requester_user_id=user_id))
            return

        managed = reservation.managed
        thinking_indicator = (
            ThinkingReactionIndicator(
                bot=bot,
                group_id=group_id,
                message_id=int(event.message_id),
                emoji_id=thinking_reaction_emoji_id,
            )
            if reservation.action == "start" and thinking_reaction_emoji_id is not None
            else None
        )
        try:
            if thinking_indicator is not None:
                await thinking_indicator.show()
            media, forwards = await collect_incoming_context(
                bot,
                event,
                config,
                workspace_dir,
            )
            logger.info(
                f"bampi_chat message context collected group_id={group_id} "
                f"message_id={event.message_id} "
                f"inline_images={len(media.inline_images)} "
                f"saved_paths={media.saved_paths} "
                f"reply_inline_images={len(media.reply_inline_images)} "
                f"reply_saved_paths={media.reply_saved_paths} "
                f"forward_roots={len(forwards.current)} "
                f"reply_forward_roots={len(forwards.reply)} "
                f"notes={media.notes} "
                f"reply_notes={media.reply_notes}"
            )
            user_message = build_user_message(
                event,
                decision.cleaned_text,
                media,
                forwards=forwards,
                resolve_name=mention_names.get,
                reaction_notes=reaction_buffer.drain(group_id) or None,
            )
            prepare_memory_turn = getattr(session_manager, "prepare_memory_for_user_turn", None)
            if callable(prepare_memory_turn):
                prepare_memory_turn(
                    managed,
                    user_id=user_id,
                    nickname=display_name(event.sender),
                    message=user_message,
                )
            update_qq_turn_context(
                session_manager,
                group_id,
                bot_self_id=str(bot.self_id),
                user_id=user_id,
                message_id=int(event.message_id),
            )

            if reservation.action == "steer":
                managed.session.steer(user_message)
                logger.info(
                    f"bampi_chat queued steer group_id={group_id} "
                    f"user_id={user_id} "
                    f"message_id={event.message_id} "
                    f"content_blocks={len(user_message.content)}"
                )
                return

            outbox_before = snapshot_outbox(workspace_dir)
            logger.info(
                f"bampi_chat session ready group_id={group_id} "
                f"message_id={event.message_id} "
                f"session_message_count={len(managed.session.messages)}"
            )

            async with managed.lock:
                managed.last_used_at = time.monotonic()
                started_at = time.monotonic()

                unsubscribe_background_origin = subscribe_background_task_origins(
                    managed,
                    origin=BackgroundTaskOrigin(
                        bot_self_id=str(bot.self_id),
                        user_id=int(event.user_id),
                        reply_message_id=int(event.message_id),
                    ),
                )
                reporter = LiveProgressReporter(bot=bot, target=reply_target_for_event(event), config=config)
                reporter.start(managed.session)
                try:
                    logger.info(
                        f"bampi_chat prompt start group_id={group_id} "
                        f"message_id={event.message_id} "
                        f"content_blocks={len(user_message.content)}"
                    )
                    try:
                        await managed.session.prompt(user_message, source="qq_group")
                    except Exception as exc:
                        logger.exception("bampi_chat session prompt failed")
                        await matcher.send(build_reply_failure_message(assess_failure(str(exc))))
                        return

                    managed.last_used_at = time.monotonic()
                    logger.info(
                        f"bampi_chat prompt finished group_id={group_id} "
                        f"message_id={event.message_id} "
                        f"duration={time.monotonic() - started_at:.2f}s "
                        f"total_messages={len(managed.session.messages)}"
                    )
                    await reporter.prepare_final_reply()
                    assistant_message = find_last_assistant_message(managed.session.messages)
                    result = await send_agent_response(
                        bot=bot,
                        event=event,
                        matcher=matcher,
                        config=config,
                        workspace_dir=workspace_dir,
                        assistant_message=assistant_message,
                        outbox_before=outbox_before,
                        intermediate_text_sent=reporter.intermediate_text_sent,
                        quote_reply=not reporter.visible_update_sent,
                    )
                finally:
                    unsubscribe_background_origin()
                    await reporter.close()
        except Exception:
            logger.exception("bampi_chat failed while preparing or delivering interaction")
            await matcher.send("消息处理异常，请稍后重试。")
            return
        finally:
            if reservation.action == "start":
                try:
                    if thinking_indicator is not None:
                        await thinking_indicator.close()
                finally:
                    await session_manager.complete_interaction(group_id)

    poke_matcher = on_notice(priority=10, block=False)

    @poke_matcher.handle()
    async def _handle_group_poke(bot: Bot, event: PokeNotifyEvent) -> None:
        if not config.bampi_enabled or not config.bampi_poke_reply_enabled:
            return
        if event.group_id is None:
            return
        if str(event.target_id) != str(bot.self_id) or str(event.user_id) == str(bot.self_id):
            return
        group_id = str(event.group_id)
        if not is_group_allowed(group_id, config):
            return
        user_id = str(event.user_id)
        status = await session_manager.inspect_interaction(group_id)
        if status.is_active:
            logger.info(
                f"bampi_chat ignored poke during active interaction group_id={group_id} "
                f"user_id={user_id}"
            )
            return
        if not limiter.allow(group_id):
            logger.info(f"bampi_chat poke rate limited group_id={group_id} user_id={user_id}")
            return
        action, suffix = extract_poke_action_texts(getattr(event, "raw_info", None))
        sender_name = (
            await resolve_member_display_name(bot, group_id=group_id, user_id=user_id, cache=mention_cache)
            or "unknown-user"
        )
        logger.info(
            f"bampi_chat poke triggered group_id={group_id} user_id={user_id} "
            f"action={action!r} suffix={suffix!r}"
        )
        await run_poke_reply_turn(
            bot=bot,
            config=config,
            session_manager=session_manager,
            group_id=group_id,
            user_id=user_id,
            sender_name=sender_name,
            action=action,
            suffix=suffix,
            reaction_notes=reaction_buffer.drain(group_id) or None,
        )

    reaction_matcher = on_notice(priority=10, block=False)

    @reaction_matcher.handle()
    async def _handle_group_emoji_reaction(bot: Bot, event: NoticeEvent) -> None:
        if not config.bampi_enabled or not config.bampi_reaction_context_enabled:
            return
        if getattr(event, "notice_type", "") != "group_msg_emoji_like":
            return
        if getattr(event, "is_add", True) is False:
            return
        group_id = str(getattr(event, "group_id", "") or "")
        user_id = str(getattr(event, "user_id", "") or "")
        message_id = getattr(event, "message_id", None)
        if not group_id or not user_id or message_id is None:
            return
        if user_id == str(bot.self_id) or not is_group_allowed(group_id, config):
            return
        try:
            note = await build_reaction_note(
                bot=bot,
                group_id=group_id,
                user_id=user_id,
                message_id=message_id,
                likes=getattr(event, "likes", None),
                cache=mention_cache,
            )
        except Exception:
            logger.exception("bampi_chat failed to build reaction note")
            return
        if note is None:
            return
        reaction_buffer.add(group_id, dedupe_key=f"{message_id}:{user_id}", note=note)
        logger.info(
            f"bampi_chat buffered reaction group_id={group_id} user_id={user_id} "
            f"message_id={message_id} note={log_preview(note)!r}"
        )
