from __future__ import annotations

import asyncio
import base64
import mimetypes
import random
import re
import shutil
import time
import uuid
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from nonebot import get_bot, get_driver, logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment, NoticeEvent, PokeNotifyEvent
from nonebot.matcher import Matcher
from nonebot.plugin import on_message, on_notice

from bampy.ai import ImageContent, TextContent, UserMessage
from bampy.ai.types import AssistantMessage, StopReason
from bampy.app import AgentSession

from .config import BampiChatConfig
from .feedback import (
    THRESHOLD_COMPACTION_NOTICE,
    FailureAssessment,
    assess_failure,
    build_background_failure_message,
    build_reply_failure_message,
)
from .forward_messages import (
    ForwardContext,
    collect_forward_context,
    iter_forward_nodes,
)
from .skills import describe_skill_resource_path
from .message_compose import (
    append_composed_text,
    compose_options_from_config,
)
from .rich_render import (
    ImagePart,
    TextPart,
    build_delivery_plan,
    rich_render_options_from_config,
)
from .rich_render.service import get_renderer as get_rich_renderer
from .message_render import (
    MentionNameCache,
    NameResolver,
    collect_mention_names,
    describe_reaction_emojis,
    extract_poke_action_texts,
    iter_segments,
    message_mentions_user,
    render_event_text,
    render_message_text,
    resolve_reaction_emoji_id,
    resolve_member_display_name,
    segment_data,
    segment_type,
)
from .session_manager import GroupSessionManager, ManagedGroupSession
from .timeutil import resolve_timezone
from .tools.safe_bash import BackgroundSessionExitEvent
from .tools.workspace import ensure_workspace_dirs, is_image_file


@dataclass(slots=True)
class TriggerDecision:
    should_respond: bool
    reason: str = ""
    direct: bool = False
    cleaned_text: str = ""


@dataclass(slots=True)
class IncomingMedia:
    inline_images: list[ImageContent] = field(default_factory=list)
    saved_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reply_inline_images: list[ImageContent] = field(default_factory=list)
    reply_saved_paths: list[str] = field(default_factory=list)
    reply_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ForwardMediaBudget:
    remaining_items: int
    remaining_bytes: int
    seen_sources: set[str] = field(default_factory=set)
    limit_noted: bool = False

    def claim(self, source_key: str) -> bool:
        if source_key and source_key in self.seen_sources:
            return False
        if self.remaining_items <= 0 or self.remaining_bytes <= 0:
            return False
        if source_key:
            self.seen_sources.add(source_key)
        self.remaining_items -= 1
        return True

    def consume(self, byte_count: int) -> None:
        self.remaining_bytes = max(0, self.remaining_bytes - max(0, byte_count))


@dataclass(slots=True)
class ResponseDispatchResult:
    delivered: bool
    rollback_context: bool = False


@dataclass(slots=True)
class ProgressMessage:
    text: str
    quote: bool = False
    tool_call_id: str | None = None
    parse_outbound_markup: bool = False


@dataclass(slots=True)
class ToolProgressNotice:
    message_id: int | None = None
    sent_at: float = 0.0
    finished: bool = False
    should_recall: bool = False
    send_failed: bool = False


@dataclass(slots=True)
class PreparedGroupFileUpload:
    file_uri: str
    cleanup_paths: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class PreparedOutboundImage:
    source: str | bytes
    cleanup_paths: list[Path] = field(default_factory=list)


TOOL_PROGRESS_EMOJIS: dict[str, str] = {
    "skill": "🧩",
    "read": "📖",
    "find": "🔎",
    "grep": "🔍",
    "bash": "💻",
    "write": "📝",
    "edit": "🛠️",
    "patch": "🩹",
    "web_ask": "🌐",
    "web_search": "🌐",
    "browser": "🧭",
    "service": "🚀",
    "schedule": "⏰",
    "memory_search": "🧠",
    "memory_time_search": "🧠",
    "memory_open": "🧠",
    "memory_manage": "🧠",
}

# 本身就是即时轻量互动的工具，群里不播报"正在调用"
SILENT_PROGRESS_TOOLS = frozenset({"qq_react"})

STOP_COMMAND = "/stop"
CLEAR_COMMANDS = {"/clear", "/new"}
COMPACT_COMMAND = "/compact"
ACTIVE_SESSION_BUSY_MESSAGE = "当前群有进行中的会话。如需中止，请让发起者发送 /stop。"
ACTIVE_SESSION_WINDING_DOWN_MESSAGE = "当前会话正在收尾，请稍候。"
STOP_NO_ACTIVE_MESSAGE = "当前没有进行中的会话或后台任务。"
STOP_NOT_OWNER_MESSAGE = "当前会话非你发起，无法停止。如需中止，请让发起者发送 /stop。"
STOPPED_SESSION_MESSAGE = "已停止当前会话。"
STOPPED_BACKGROUND_SESSION_MESSAGE = "已终止后台任务。"
STOPPED_SESSION_AND_BACKGROUND_MESSAGE = "已停止当前会话并终止后台任务。"
FORCE_STOP_PREFIX = "已强制"
BACKGROUND_TASKS_RUNNING_MESSAGE = "仍有后台任务在运行，发送 /stop 终止后再试。"
CLEARED_SESSION_MESSAGE = "已清空对话上下文。"
CLEAR_NO_CONTEXT_MESSAGE = "当前没有可清空的上下文。"
COMPACT_NO_CONTEXT_MESSAGE = "当前没有可压缩的上下文。"
COMPACT_FORBIDDEN_MESSAGE = "权限不足，仅管理员可使用 /compact。"


@dataclass(slots=True)
class GroupReplyTarget:
    group_id: int
    user_id: int | None = None
    reply_message_id: int | None = None


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


def reply_target_for_event(event: GroupMessageEvent) -> GroupReplyTarget:
    return GroupReplyTarget(
        group_id=int(event.group_id),
        user_id=int(event.user_id),
        reply_message_id=int(event.message_id),
    )


def update_qq_turn_context(
    session_manager: GroupSessionManager,
    group_id: str,
    *,
    bot_self_id: str,
    user_id: str,
    message_id: int | None,
) -> None:
    """记录本轮触发消息，供 qq_react 工具定位贴表情/戳一戳目标。"""
    setter = getattr(session_manager, "set_qq_turn_context", None)
    if callable(setter):
        setter(group_id, bot_self_id=bot_self_id, user_id=user_id, message_id=message_id)


def clear_qq_turn_context(session_manager: GroupSessionManager, group_id: str) -> None:
    """清除轮次上下文；定时/后台恢复轮次没有可互动的触发消息。"""
    clearer = getattr(session_manager, "clear_qq_turn_context", None)
    if callable(clearer):
        clearer(group_id)


class PlaintextEvent(Protocol):
    to_me: bool
    reply: Any

    def get_plaintext(self) -> str: ...


class GroupRateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self._limit = limit
        self._window = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def allow(self, group_id: str) -> bool:
        if self._limit <= 0:
            return True
        now = time.monotonic()
        bucket = self._buckets.setdefault(group_id, deque())
        while bucket and now - bucket[0] >= self._window:
            bucket.popleft()
        if len(bucket) >= self._limit:
            return False
        bucket.append(now)
        return True


class GroupReactionBuffer:
    """暂存贴表情（表情回应）事件，随下一轮交互作为上下文交给模型。"""

    def __init__(self, *, ttl_seconds: float = 30 * 60, max_per_group: int = 10) -> None:
        self._ttl = ttl_seconds
        self._max_per_group = max_per_group
        self._buckets: dict[str, OrderedDict[str, tuple[float, str]]] = {}

    def add(self, group_id: str, *, dedupe_key: str, note: str) -> None:
        bucket = self._buckets.setdefault(group_id, OrderedDict())
        bucket.pop(dedupe_key, None)
        bucket[dedupe_key] = (time.monotonic(), note)
        self._prune(bucket)

    def drain(self, group_id: str) -> list[str]:
        bucket = self._buckets.pop(group_id, None)
        if not bucket:
            return []
        now = time.monotonic()
        return [note for created_at, note in bucket.values() if now - created_at < self._ttl]

    def _prune(self, bucket: "OrderedDict[str, tuple[float, str]]") -> None:
        now = time.monotonic()
        for key in [key for key, (created_at, _) in bucket.items() if now - created_at >= self._ttl]:
            del bucket[key]
        while len(bucket) > self._max_per_group:
            bucket.popitem(last=False)


def is_group_allowed(group_id: str, config: BampiChatConfig) -> bool:
    whitelist = config.bampi_group_whitelist
    return not whitelist or group_id in whitelist


def log_preview(text: str | None, *, limit: int = 160) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def summarize_segments(message: Message) -> str:
    counts = Counter(segment.type for segment in message)
    if not counts:
        return "empty"
    return ",".join(f"{segment_type}:{counts[segment_type]}" for segment_type in sorted(counts))


def extract_api_message_id(response: Any) -> int | None:
    candidate: Any = None
    if isinstance(response, dict):
        candidate = response.get("message_id")
        if candidate is None:
            nested = response.get("data")
            if isinstance(nested, dict):
                candidate = nested.get("message_id")
    elif response is not None:
        candidate = getattr(response, "message_id", response)

    if candidate is None:
        return None
    try:
        return int(candidate)
    except (TypeError, ValueError):
        logger.warning(f"bampi_chat got non-numeric message_id from api: {candidate!r}")
        return None


class LiveProgressReporter:
    def __init__(
        self,
        *,
        bot: Bot,
        target: GroupReplyTarget,
        config: BampiChatConfig,
    ) -> None:
        self._bot = bot
        self._target = target
        self._config = config
        self._live_progress_enabled = config.bampi_live_progress_enabled
        self._compaction_notice_enabled = config.bampi_threshold_compaction_notice_enabled
        self._enabled = self._live_progress_enabled or self._compaction_notice_enabled
        self._queue: asyncio.Queue[ProgressMessage | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._closed = False
        self._visible_update_sent = False
        self._compaction_notice_sent = False
        self._tool_updates_sent = 0
        self._streamed_text = ""
        self._streamed_any_text = False
        self._last_seen_text = ""
        self._pending_text = ""
        self._tool_notices: dict[str, ToolProgressNotice] = {}
        self._recall_tasks: set[asyncio.Task[None]] = set()
        self._last_text_flush_at = 0.0

    @property
    def streamed_text(self) -> str:
        return self._streamed_text

    @property
    def streamed_any_text(self) -> bool:
        return self._streamed_any_text

    def start(self, session: AgentSession) -> None:
        if not self._enabled:
            return
        self._worker = asyncio.create_task(self._run_sender())
        self._unsubscribe = session.subscribe(self._handle_event)

    async def prepare_final_reply(self) -> None:
        if not self._enabled:
            return
        self._flush_pending_text(force=True)
        await self._queue.join()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._worker is not None:
            await self._queue.join()
            self._queue.put_nowait(None)
            await self._worker
            self._worker = None

    async def _run_sender(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                message = Message()
                if item.quote and self._target.reply_message_id is not None:
                    message += MessageSegment.reply(self._target.reply_message_id)
                if item.parse_outbound_markup:
                    append_composed_text(
                        message,
                        item.text,
                        options=compose_options_from_config(self._config),
                    )
                else:
                    message += MessageSegment.text(item.text)
                response = await self._bot.call_api(
                    "send_group_msg",
                    group_id=self._target.group_id,
                    message=message,
                )
                if item.tool_call_id:
                    self._mark_tool_notice_sent(
                        item.tool_call_id,
                        extract_api_message_id(response),
                    )
            except Exception:
                if item is not None and item.tool_call_id:
                    self._mark_tool_notice_send_failed(item.tool_call_id)
                logger.exception(
                    f"bampi_chat failed to send live progress "
                    f"group_id={self._target.group_id} "
                    f"message_id={self._target.reply_message_id}"
                )
            finally:
                self._queue.task_done()

    def _handle_event(self, event: Any) -> None:
        if self._closed or not self._enabled:
            return
        event_type = getattr(event, "type", None)
        if event_type == "auto_compaction_start":
            self._handle_auto_compaction_start(event)
            return
        if not self._live_progress_enabled:
            return
        if event_type == "tool_execution_start":
            self._handle_tool_start(event)
            return
        if event_type == "tool_execution_end":
            self._handle_tool_end(event)
            return
        if event_type == "message_start" and self._config.bampi_live_text_stream_enabled:
            self._handle_message_start()
            return
        if event_type == "message_end" and self._config.bampi_live_text_stream_enabled:
            self._handle_message_end()
            return
        if event_type == "message_update" and self._config.bampi_live_text_stream_enabled:
            self._handle_message_update(event)

    def _handle_auto_compaction_start(self, event: Any) -> None:
        if not self._compaction_notice_enabled:
            return
        if getattr(event, "reason", "") != "threshold":
            return
        if self._compaction_notice_sent:
            return
        self._compaction_notice_sent = True
        self._enqueue(THRESHOLD_COMPACTION_NOTICE)

    def _handle_message_start(self) -> None:
        self._last_seen_text = ""
        self._pending_text = ""
        self._streamed_text = ""
        self._streamed_any_text = False

    def _handle_tool_start(self, event: Any) -> None:
        if getattr(event, "tool_name", "") in SILENT_PROGRESS_TOOLS:
            return
        limit = self._config.bampi_live_progress_max_tool_updates
        if limit > 0 and self._tool_updates_sent >= limit:
            return
        if self._config.bampi_live_text_stream_enabled:
            self._flush_pending_text(force=True)

        tool_call_id = getattr(event, "tool_call_id", "")
        self._tool_updates_sent += 1
        progress_msg = format_tool_progress_message(
            getattr(event, "tool_name", ""),
            getattr(event, "args", None),
        )
        if tool_call_id:
            self._tool_notices[tool_call_id] = ToolProgressNotice()
        self._enqueue(progress_msg, tool_call_id=tool_call_id or None)

    def _handle_tool_end(self, event: Any) -> None:
        tool_call_id = getattr(event, "tool_call_id", "")
        if not tool_call_id:
            return
        notice = self._tool_notices.get(tool_call_id)
        if notice is None:
            return
        notice.finished = True
        notice.should_recall = bool(getattr(event, "is_error", False))
        self._finalize_tool_notice(tool_call_id)

    def _handle_message_update(self, event: Any) -> None:
        message = getattr(event, "message", None)
        current_text = extract_text_blocks(message)
        if not current_text:
            return

        delta = self._extract_snapshot_delta(current_text)
        if not delta:
            return

        self._pending_text += delta

    def _handle_message_end(self) -> None:
        self._flush_pending_text(force=True)

    def _flush_pending_text(self, *, force: bool = False) -> None:
        if self._closed or not self._pending_text.strip():
            return

        normalized_length = len(normalize_text(self._pending_text))
        now = time.monotonic()
        min_chars = max(1, self._config.bampi_live_text_stream_min_chars)
        force_chars = max(min_chars, self._config.bampi_live_text_stream_force_chars)
        min_interval = max(0.0, self._config.bampi_live_text_stream_min_interval_seconds)

        if not force:
            if normalized_length < min_chars:
                return
            if normalized_length < force_chars and now - self._last_text_flush_at < min_interval:
                return

        payload = self._pending_text
        self._pending_text = ""
        self._streamed_text += payload
        self._streamed_any_text = True
        self._last_text_flush_at = now
        self._enqueue(
            payload,
            preserve_whitespace=True,
            parse_outbound_markup=True,
        )

    def _extract_snapshot_delta(self, current_text: str) -> str:
        if not self._last_seen_text:
            self._last_seen_text = current_text
            return current_text
        if current_text == self._last_seen_text:
            return ""
        if current_text.startswith(self._last_seen_text):
            delta = current_text[len(self._last_seen_text) :]
            self._last_seen_text = current_text
            return delta
        if self._last_seen_text.startswith(current_text):
            return ""

        prefix_len = longest_common_prefix_len(self._last_seen_text, current_text)
        logger.warning(
            f"bampi_chat live text stream desynced "
            f"group_id={self._target.group_id} "
            f"message_id={self._target.reply_message_id} "
            f"last_seen={log_preview(self._last_seen_text)!r} "
            f"current={log_preview(current_text)!r} "
            f"common_prefix={prefix_len}"
        )
        delta = current_text[prefix_len:]
        self._last_seen_text = current_text
        return delta

    def _enqueue(
        self,
        text: str,
        *,
        preserve_whitespace: bool = False,
        tool_call_id: str | None = None,
        parse_outbound_markup: bool = False,
    ) -> None:
        if self._closed:
            return
        if not text.strip():
            return
        payload = text if preserve_whitespace else text.strip()
        quote = not self._visible_update_sent
        if quote:
            self._visible_update_sent = True
        self._queue.put_nowait(
            ProgressMessage(
                text=payload,
                quote=quote,
                tool_call_id=tool_call_id,
                parse_outbound_markup=parse_outbound_markup,
            )
        )

    def _mark_tool_notice_sent(self, tool_call_id: str, message_id: int | None) -> None:
        notice = self._tool_notices.get(tool_call_id)
        if notice is None:
            return
        notice.message_id = message_id
        notice.sent_at = time.monotonic()
        self._finalize_tool_notice(tool_call_id)

    def _mark_tool_notice_send_failed(self, tool_call_id: str) -> None:
        notice = self._tool_notices.get(tool_call_id)
        if notice is None:
            return
        notice.send_failed = True
        self._finalize_tool_notice(tool_call_id)

    def _finalize_tool_notice(self, tool_call_id: str) -> None:
        notice = self._tool_notices.get(tool_call_id)
        if notice is None or not notice.finished:
            return
        if notice.sent_at <= 0 and not notice.send_failed:
            return
        if notice.should_recall and notice.message_id is not None:
            self._schedule_tool_notice_recall(
                tool_call_id=tool_call_id,
                message_id=notice.message_id,
                sent_at=notice.sent_at,
            )
        elif notice.should_recall and notice.sent_at > 0 and not notice.send_failed:
            logger.warning(
                f"bampi_chat cannot recall tool progress without message_id "
                f"group_id={self._target.group_id} "
                f"message_id={self._target.reply_message_id} "
                f"tool_call_id={tool_call_id}"
            )
        self._tool_notices.pop(tool_call_id, None)

    def _schedule_tool_notice_recall(
        self,
        *,
        tool_call_id: str,
        message_id: int,
        sent_at: float,
    ) -> None:
        task = asyncio.create_task(
            self._recall_tool_notice(
                tool_call_id=tool_call_id,
                message_id=message_id,
                sent_at=sent_at,
            )
        )
        self._recall_tasks.add(task)

        def _cleanup(done: asyncio.Task[None]) -> None:
            self._recall_tasks.discard(done)
            if done.cancelled():
                return
            exc = done.exception()
            if exc is not None:
                logger.error(
                    f"bampi_chat tool progress recall task failed "
                    f"group_id={self._target.group_id} "
                    f"message_id={self._target.reply_message_id} "
                    f"tool_call_id={tool_call_id} "
                    f"error={exc!r}"
                )

        task.add_done_callback(_cleanup)

    async def _recall_tool_notice(
        self,
        *,
        tool_call_id: str,
        message_id: int,
        sent_at: float,
    ) -> None:
        min_visible = max(
            0.0,
            self._config.bampi_live_progress_error_recall_min_visible_seconds,
        )
        remaining = sent_at + min_visible - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        await self._bot.call_api("delete_msg", message_id=message_id)
        logger.info(
            f"bampi_chat recalled failed tool progress "
            f"group_id={self._target.group_id} "
            f"message_id={self._target.reply_message_id} "
            f"tool_call_id={tool_call_id} "
            f"recalled_message_id={message_id}"
        )


def format_tool_progress_message(tool_name: str, args: Any) -> str:
    if tool_name == "read":
        payload = args if isinstance(args, dict) else {}
        path = payload.get("path") or payload.get("file_path")
        skill_resource = describe_skill_resource_path(str(path) if path is not None else None)
        if skill_resource is not None:
            return f"{TOOL_PROGRESS_EMOJIS['skill']} {format_skill_resource_progress(skill_resource)}"

    description = describe_tool_progress(tool_name, args)
    emoji = TOOL_PROGRESS_EMOJIS.get(tool_name, "🛠️")
    return f"{emoji} {description}"


def render_tool_progress_value(value: Any, fallback: str, *, limit: int = 80) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    return log_preview(text, limit=limit)


def longest_common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def format_skill_resource_progress(skill_resource: tuple[str, str]) -> str:
    skill_name, relative_path = skill_resource
    normalized = relative_path.strip("/")
    if not normalized or normalized == "SKILL.md":
        return f"正在加载 skill：{skill_name}"
    return f"正在读取 skill 资料：{skill_name}/{normalized}"


def describe_tool_progress(tool_name: str, args: Any) -> str:
    payload = args if isinstance(args, dict) else {}
    if tool_name == "read":
        path = render_tool_progress_value(
            payload.get("path") or payload.get("file_path"),
            "目标文件",
        )
        return f"正在读取：{path}"
    if tool_name == "find":
        pattern = render_tool_progress_value(
            payload.get("pattern") or payload.get("name") or payload.get("query"),
            "目标文件",
        )
        return f"正在查找：{pattern}"
    if tool_name == "grep":
        pattern = render_tool_progress_value(payload.get("pattern") or payload.get("query"), "关键词")
        return f"正在搜索：{pattern}"
    if tool_name == "bash":
        action = render_tool_progress_value(payload.get("action"), "run")
        command = render_tool_progress_value(payload.get("command") or payload.get("cmd"), "当前命令")
        session_id = render_tool_progress_value(payload.get("session_id"), "会话")
        if action == "start":
            return f"正在启动后台终端：{command}"
        if action == "logs":
            return f"正在查看后台终端日志：{session_id}"
        if action == "status":
            return f"正在查看后台终端状态：{session_id}"
        if action == "input":
            return f"正在向后台终端发送输入：{session_id}"
        if action == "stop":
            return f"正在停止后台终端：{session_id}"
        if action == "list":
            return "正在查看后台终端列表"
        if command:
            return f"正在执行命令：{command}"
        return "正在执行命令"
    if tool_name == "write":
        path = render_tool_progress_value(payload.get("path") or payload.get("file_path"), "目标文件")
        return f"正在写入：{path}"
    if tool_name == "edit":
        path = render_tool_progress_value(payload.get("path") or payload.get("file_path"), "目标文件")
        return f"正在修改：{path}"
    if tool_name == "patch":
        return "正在应用补丁"
    if tool_name in ("web_ask", "web_search"):
        query = render_tool_progress_value(payload.get("query") or payload.get("q"), "查询内容")
        return f"正在搜索网页：{query}"
    if tool_name == "browser":
        raw_command = str(payload.get("command") or "").strip()
        lines = [line.strip() for line in raw_command.splitlines() if line.strip()]
        if lines and lines[0].split()[0].lower() == "batch":
            step_count = len(lines) - 1
            return f"正在操作浏览器（批量 {step_count} 步）"
        command = render_tool_progress_value(payload.get("command"), "操作网页")
        first_line = command.splitlines()[0] if command else "操作网页"
        return f"正在操作浏览器：{first_line}"
    if tool_name == "service":
        action = render_tool_progress_value(payload.get("action"), "status")
        service_ref = render_tool_progress_value(
            payload.get("service") or payload.get("name"),
            "服务",
        )
        command = render_tool_progress_value(payload.get("command"), "当前服务命令")
        if action == "start":
            return f"正在启动对外服务：{command}"
        if action == "list":
            return "正在查看服务列表"
        if action == "logs":
            return f"正在查看服务日志：{service_ref}"
        if action == "stop":
            return f"正在停止服务：{service_ref}"
        return f"正在查看服务状态：{service_ref}"
    if tool_name == "schedule":
        action = render_tool_progress_value(payload.get("action"), "status")
        task_ref = render_tool_progress_value(
            payload.get("task") or payload.get("name"),
            "定时任务",
        )
        trigger_type = render_tool_progress_value(payload.get("trigger_type"), "date")
        run_at = render_tool_progress_value(payload.get("run_at"), "")
        cron = render_tool_progress_value(payload.get("cron"), "")
        if action == "create":
            if trigger_type == "cron" and cron:
                return f"正在创建定时任务：{task_ref}（cron {cron}）"
            if run_at:
                return f"正在创建定时任务：{task_ref}（{run_at}）"
            return f"正在创建定时任务：{task_ref}"
        if action == "list":
            return "正在查看定时任务列表"
        if action == "pause":
            return f"正在暂停定时任务：{task_ref}"
        if action == "resume":
            return f"正在恢复定时任务：{task_ref}"
        if action == "cancel":
            return f"正在取消定时任务：{task_ref}"
        if action == "run_now":
            return f"正在立即执行定时任务：{task_ref}"
        return f"正在查看定时任务：{task_ref}"
    if tool_name == "memory_search":
        return "正在搜索记忆"
    if tool_name == "memory_time_search":
        return "正在搜索记忆"
    if tool_name == "memory_open":
        return "正在查看记忆"
    if tool_name == "memory_manage":
        payload = args if isinstance(args, dict) else {}
        action = payload.get("action", "")
        if action == "add":
            return "正在记录记忆"
        if action == "update":
            return "正在更新记忆"
        if action == "delete":
            return "正在删除记忆"
        return "正在编辑记忆"
    display_name = render_tool_progress_value(tool_name, "unknown", limit=40)
    return f"正在执行工具：{display_name}"


def is_stop_command(text: str) -> bool:
    return normalize_text(text).lower() == STOP_COMMAND


def is_clear_command(text: str) -> bool:
    return normalize_text(text).lower() in CLEAR_COMMANDS


def is_compact_command(text: str) -> bool:
    return normalize_text(text).lower() == COMPACT_COMMAND


def is_nonebot_superuser(user_id: str | int) -> bool:
    try:
        driver = get_driver()
    except ValueError:
        return False

    configured = getattr(driver.config, "superusers", None) or set()
    return str(user_id) in {str(item) for item in configured}


def interaction_busy_message(status: Any, *, requester_user_id: str | None = None) -> str:
    active_user_id = getattr(status, "active_user_id", None)
    if requester_user_id is not None and active_user_id == requester_user_id:
        return ACTIVE_SESSION_WINDING_DOWN_MESSAGE
    if not bool(getattr(status, "is_streaming", False)):
        return ACTIVE_SESSION_WINDING_DOWN_MESSAGE
    return ACTIVE_SESSION_BUSY_MESSAGE


def build_stop_success_message(
    *,
    force: bool,
    aborted_streaming: bool,
    stopped_background_sessions: bool,
) -> str:
    if aborted_streaming and stopped_background_sessions:
        message = STOPPED_SESSION_AND_BACKGROUND_MESSAGE
    elif stopped_background_sessions:
        message = STOPPED_BACKGROUND_SESSION_MESSAGE
    else:
        message = STOPPED_SESSION_MESSAGE
    if force:
        message = FORCE_STOP_PREFIX + message.removeprefix("已")
    return message


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
                        streamed_text=reporter.streamed_text,
                        streamed_any_text=reporter.streamed_any_text,
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


def should_respond(
    event: PlaintextEvent,
    *,
    bot_self_id: str,
    config: BampiChatConfig,
    random_value: float,
    resolve_name: NameResolver | None = None,
) -> TriggerDecision:
    if not config.bampi_enabled:
        return TriggerDecision(False)

    text = normalize_text(render_event_text(event, resolve_name=resolve_name))
    reply_to_bot = is_reply_to_bot(event.reply, bot_self_id)

    if bool(getattr(event, "to_me", False)) or reply_to_bot:
        return TriggerDecision(True, reason="to_me", direct=True, cleaned_text=text)

    if message_mentions_user(getattr(event, "message", None), bot_self_id):
        return TriggerDecision(True, reason="mention", direct=True, cleaned_text=text)

    prefix = matched_prefix(text, config.bampi_trigger_prefix)
    if prefix is not None:
        return TriggerDecision(
            True,
            reason="prefix",
            direct=True,
            cleaned_text=normalize_text(text[len(prefix) :]),
        )

    if text and any(keyword in text for keyword in config.bampi_trigger_keywords):
        return TriggerDecision(True, reason="keyword", direct=True, cleaned_text=text)

    if text and config.bampi_random_reply_prob > 0 and random_value < config.bampi_random_reply_prob:
        return TriggerDecision(True, reason="random", direct=False, cleaned_text=text)

    return TriggerDecision(False)


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def extract_message_text(message: Any, *, resolve_name: NameResolver | None = None) -> str:
    if message is None:
        return ""
    try:
        return normalize_text(render_message_text(message, resolve_name=resolve_name))
    except Exception:
        logger.warning("bampi_chat failed to render message text")
        return normalize_text(str(message))


def extract_segment_filename(segment: Any) -> str | None:
    data = segment_data(segment)
    for key in ("name", "file", "file_name", "filename"):
        raw_value = data.get(key)
        if raw_value is None:
            continue
        filename = sanitize_filename(str(raw_value))
        if filename:
            return filename
    return None


def sanitize_filename(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text:
        return None

    text = text.replace("\\", "/")
    candidate = Path(text).name.strip()
    if not candidate or candidate in {".", ".."}:
        return None

    candidate = re.sub(r"[\x00-\x1f]", "", candidate)
    candidate = candidate.strip().strip(".")
    return candidate or None


def infer_filename_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    return sanitize_filename(path)


def infer_extension_from_content(data: bytes) -> str:
    if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06") or data.startswith(b"PK\x07\x08"):
        return ".zip"
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"\x1f\x8b\x08"):
        return ".gz"
    return ""


def resolve_inbox_preferred_name(
    *,
    preferred_name: str | None,
    download_url: str | None = None,
    mime_type: str | None = None,
    content: bytes | None = None,
) -> str | None:
    filename = sanitize_filename(preferred_name) or infer_filename_from_url(download_url)
    if filename:
        return filename

    extension = mimetypes.guess_extension(mime_type or "") or ""
    if not extension and content:
        extension = infer_extension_from_content(content)
    if not extension:
        return None
    return f"attachment{extension}"


def matched_prefix(text: str, prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        if text.startswith(prefix):
            return prefix
    return None


def is_reply_to_bot(reply: Any, bot_self_id: str) -> bool:
    if reply is None or getattr(reply, "sender", None) is None:
        return False
    sender_id = getattr(reply.sender, "user_id", None)
    return sender_id is not None and str(sender_id) == str(bot_self_id)


def display_name(sender: Any) -> str:
    card = getattr(sender, "card", "") or ""
    nickname = getattr(sender, "nickname", "") or ""
    return card.strip() or nickname.strip() or "unknown-user"


def build_user_message(
    event: GroupMessageEvent,
    cleaned_text: str,
    media: IncomingMedia,
    *,
    forwards: ForwardContext | None = None,
    resolve_name: NameResolver | None = None,
    reaction_notes: list[str] | None = None,
) -> UserMessage:
    forwards = forwards or ForwardContext()
    sender_name = display_name(event.sender)
    if cleaned_text:
        body = cleaned_text
    elif forwards.has_current:
        body = "(无附言；本条消息包含合并转发，请结合 forwarded_messages 理解)"
    elif media.inline_images or media.saved_paths:
        body = "(无纯文本内容；本条消息仅包含媒体/文件)"
    elif forwards.has_reply:
        body = "(无附言；请结合回复引用中的合并转发理解)"
    elif media.reply_inline_images or media.reply_saved_paths:
        body = "(无纯文本内容；请结合回复引用内容理解)"
    else:
        body = "(无纯文本内容)"

    lines = [
        f"sender_name: {sender_name}({event.user_id})",
        f"message_text: {body}",
    ]

    if event.reply is not None and getattr(event.reply, "sender", None) is not None:
        reply_name = display_name(event.reply.sender)
        reply_text = extract_message_text(getattr(event.reply, "message", None), resolve_name=resolve_name)
        lines.append(f"reply_to_name: {reply_name}")
        if reply_text:
            lines.append(f"reply_message: {reply_text}")

    if forwards.current_render.text:
        lines.append("forwarded_messages:")
        lines.extend(f"  {line}" for line in forwards.current_render.text.splitlines())
    if forwards.reply_render.text:
        lines.append("reply_forwarded_messages:")
        lines.extend(f"  {line}" for line in forwards.reply_render.text.splitlines())

    if media.inline_images:
        lines.append(f"inline_image_count: {len(media.inline_images)}")
    if media.saved_paths:
        lines.append("workspace_attachments:")
        lines.extend(f"- {path}" for path in media.saved_paths)
    if media.notes:
        lines.append("media_notes:")
        lines.extend(f"- {note}" for note in media.notes)
    if media.reply_inline_images:
        lines.append(f"reply_inline_image_count: {len(media.reply_inline_images)}")
    if media.reply_saved_paths:
        lines.append("reply_workspace_attachments:")
        lines.extend(f"- {path}" for path in media.reply_saved_paths)
    if media.reply_notes:
        lines.append("reply_media_notes:")
        lines.extend(f"- {note}" for note in media.reply_notes)
    if reaction_notes:
        lines.append("recent_reactions:")
        lines.extend(f"- {note}" for note in reaction_notes)

    content: list[TextContent | ImageContent] = [TextContent(text="\n".join(lines))]
    content.extend(media.inline_images)
    content.extend(media.reply_inline_images)
    return UserMessage(content=content)


async def collect_incoming_context(
    bot: Bot,
    event: GroupMessageEvent,
    config: BampiChatConfig,
    workspace_dir: str,
) -> tuple[IncomingMedia, ForwardContext]:
    """Collect ordinary media and lazily expand merged-forward messages."""
    reply_message = getattr(event.reply, "message", None)
    forwards = await collect_forward_context(
        bot,
        message=event.message,
        reply_message=reply_message,
        enabled=config.bampi_forward_enabled,
        max_depth=config.bampi_forward_max_depth,
        max_nodes=config.bampi_forward_max_nodes,
        max_roots=config.bampi_forward_max_roots,
        max_api_calls=config.bampi_forward_max_api_calls,
        max_text_chars=config.bampi_forward_max_text_chars,
        timeout_seconds=config.bampi_forward_resolve_timeout_seconds,
        timezone=resolve_timezone(config.bampi_schedule_timezone),
    )
    media = await collect_incoming_media(
        bot,
        event,
        config,
        workspace_dir,
        forwards=forwards,
    )
    await _persist_truncated_forward_transcripts(
        media=media,
        forwards=forwards,
        workspace_dir=workspace_dir,
    )
    return media, forwards


async def _persist_truncated_forward_transcripts(
    *,
    media: IncomingMedia,
    forwards: ForwardContext,
    workspace_dir: str,
) -> None:
    targets = (
        (
            forwards.current_render,
            media.saved_paths,
            media.notes,
            "forwarded-messages.txt",
            "合并转发",
        ),
        (
            forwards.reply_render,
            media.reply_saved_paths,
            media.reply_notes,
            "reply-forwarded-messages.txt",
            "回复引用中的合并转发",
        ),
    )
    for rendered, saved_paths, notes, preferred_name, label in targets:
        if not rendered.truncated or not rendered.full_text:
            continue
        try:
            saved = await save_bytes_to_inbox(
                workspace_dir,
                rendered.full_text.encode("utf-8"),
                preferred_name=preferred_name,
                mime_type="text/plain",
            )
        except Exception as exc:
            logger.warning(
                f"bampi_chat failed to save full forward transcript error={exc!r}"
            )
            notes.append(f"{label}内容较长，预览已截断，完整转录保存失败。")
            continue
        saved_paths.append(saved)
        notes.append(f"{label}内容较长，完整转录已保存到 {saved}")


async def collect_incoming_media(
    bot: Bot,
    event: GroupMessageEvent,
    config: BampiChatConfig,
    workspace_dir: str,
    *,
    forwards: ForwardContext | None = None,
) -> IncomingMedia:
    ensure_workspace_dirs(workspace_dir)
    media = IncomingMedia()

    await _collect_media_from_message(
        bot=bot,
        event=event,
        message=event.message,
        media=media,
        config=config,
        workspace_dir=workspace_dir,
        from_reply=False,
    )

    reply_message = getattr(event.reply, "message", None)
    if reply_message is not None:
        await _collect_media_from_message(
            bot=bot,
            event=event,
            message=reply_message,
            media=media,
            config=config,
            workspace_dir=workspace_dir,
            from_reply=True,
        )

    if forwards is not None and (forwards.has_current or forwards.has_reply):
        budget = _ForwardMediaBudget(
            remaining_items=config.bampi_forward_max_media_items,
            remaining_bytes=config.bampi_forward_max_total_media_bytes,
        )
        for from_reply, resolved in (
            (False, forwards.current),
            (True, forwards.reply),
        ):
            for node in iter_forward_nodes(resolved):
                await _collect_media_from_message(
                    bot=bot,
                    event=event,
                    message=node.segments,
                    media=media,
                    config=config,
                    workspace_dir=workspace_dir,
                    from_reply=from_reply,
                    allow_group_file_lookup=False,
                    forward_budget=budget,
                )

    return media


def _media_targets(
    media: IncomingMedia,
    *,
    from_reply: bool,
) -> tuple[list[ImageContent], list[str], list[str]]:
    if from_reply:
        return media.reply_inline_images, media.reply_saved_paths, media.reply_notes
    return media.inline_images, media.saved_paths, media.notes


async def _collect_media_from_message(
    *,
    bot: Bot,
    event: GroupMessageEvent,
    message: Any,
    media: IncomingMedia,
    config: BampiChatConfig,
    workspace_dir: str,
    from_reply: bool,
    allow_group_file_lookup: bool = True,
    forward_budget: _ForwardMediaBudget | None = None,
) -> None:
    source = "reply" if from_reply else "message"
    if forward_budget is not None:
        source = f"forward_{source}"
    if message is None:
        return

    for segment in iter_segments(message):
        seg_type = segment_type(segment)
        if seg_type not in {"image", "file"}:
            continue
        data = segment_data(segment)
        source_key = _media_source_key(seg_type, data)
        if forward_budget is not None:
            direct_url = str(data.get("url") or "").strip()
            if direct_url and urlparse(direct_url).scheme.lower() not in {
                "http",
                "https",
            }:
                if not source_key or source_key not in forward_budget.seen_sources:
                    if source_key:
                        forward_budget.seen_sources.add(source_key)
                    _, _, notes = _media_targets(media, from_reply=from_reply)
                    notes.append(
                        "合并转发中的媒体 URL 不是可下载的 HTTP(S) 地址，已跳过。"
                    )
                continue
            if source_key and source_key in forward_budget.seen_sources:
                continue
            if not forward_budget.claim(source_key):
                _note_forward_media_limit(media, from_reply=from_reply, budget=forward_budget)
                continue

        logger.info(
            f"bampi_chat processing {seg_type} segment "
            f"group_id={event.group_id} "
            f"message_id={event.message_id} "
            f"source={source}"
        )
        max_download_bytes = (
            forward_budget.remaining_bytes if forward_budget is not None else None
        )
        if seg_type == "image":
            consumed = await _handle_image_segment(
                segment,
                media,
                config,
                workspace_dir,
                from_reply=from_reply,
                max_download_bytes=max_download_bytes,
            )
        else:
            consumed = await _handle_file_segment(
                bot,
                event,
                segment,
                media,
                config,
                workspace_dir,
                from_reply=from_reply,
                allow_group_file_lookup=allow_group_file_lookup,
                max_download_bytes=max_download_bytes,
            )
        if forward_budget is not None:
            forward_budget.consume(consumed)


def _media_source_key(seg_type: str, data: dict[str, Any]) -> str:
    source = (
        data.get("url")
        or data.get("file_id")
        or data.get("id")
        or data.get("file")
    )
    return f"{seg_type}:{source}" if source else ""


def _note_forward_media_limit(
    media: IncomingMedia,
    *,
    from_reply: bool,
    budget: _ForwardMediaBudget,
) -> None:
    if budget.limit_noted:
        return
    budget.limit_noted = True
    _, _, notes = _media_targets(media, from_reply=from_reply)
    notes.append("合并转发中的媒体较多或总大小超过限制，仅处理了前一部分。")


async def _handle_image_segment(
    segment: Any,
    media: IncomingMedia,
    config: BampiChatConfig,
    workspace_dir: str,
    *,
    from_reply: bool,
    max_download_bytes: int | None = None,
) -> int:
    inline_images, saved_paths, notes = _media_targets(media, from_reply=from_reply)
    data = segment_data(segment)
    url = str(data.get("url") or "").strip()
    if not url:
        logger.warning("bampi_chat image segment missing download url")
        if from_reply:
            notes.append("回复引用里有图片，但适配器未提供可下载 URL。")
        else:
            notes.append("收到图片，但适配器未提供可下载 URL。")
        return 0

    download_limit = config.bampi_max_download_size
    if max_download_bytes is not None:
        download_limit = min(download_limit, max_download_bytes)
    if download_limit <= 0:
        notes.append("图片未下载：已达到本轮合并转发媒体大小上限。")
        return 0

    try:
        content, mime_type = await download_url(
            url,
            timeout=config.bampi_web_search_timeout,
            max_bytes=download_limit,
        )
    except Exception as exc:
        logger.warning(f"bampi_chat failed to download image: {exc}")
        if from_reply:
            notes.append(f"下载回复引用图片失败: {exc}")
        else:
            notes.append(f"下载图片失败: {exc}")
        return 0

    mime_type = mime_type or guess_mime_type(url, default="image/png")
    if len(content) <= config.bampi_max_inline_image_size:
        logger.info(f"bampi_chat inlined image mime_type={mime_type} size={len(content)}")
        inline_images.append(
            ImageContent(
                data=base64.b64encode(content).decode("ascii"),
                mime_type=mime_type,
            )
        )
        return len(content)

    saved = await save_bytes_to_inbox(
        workspace_dir,
        content,
        preferred_name=data.get("file"),
        mime_type=mime_type,
    )
    logger.info(f"bampi_chat saved oversized image path={saved} size={len(content)}")
    saved_paths.append(saved)
    if from_reply:
        notes.append(f"回复引用中的图片过大，已保存到 {saved}")
    else:
        notes.append(f"图片过大，已保存到 {saved}")
    return len(content)


async def _handle_file_segment(
    bot: Bot,
    event: GroupMessageEvent,
    segment: Any,
    media: IncomingMedia,
    config: BampiChatConfig,
    workspace_dir: str,
    *,
    from_reply: bool,
    allow_group_file_lookup: bool = True,
    max_download_bytes: int | None = None,
) -> int:
    data = segment_data(segment)
    file_id = data.get("id") or data.get("file_id")
    direct_url = str(data.get("url", "")).strip()
    if not direct_url and (not file_id or not allow_group_file_lookup):
        logger.warning("bampi_chat file segment missing usable download url")
        _, _, notes = _media_targets(media, from_reply=from_reply)
        if from_reply:
            notes.append("回复引用里有文件，但缺少可下载 URL。")
        else:
            notes.append("收到文件，但缺少可下载 URL。")
        return 0

    _, saved_paths, notes = _media_targets(media, from_reply=from_reply)
    download_limit = config.bampi_max_download_size
    if max_download_bytes is not None:
        download_limit = min(download_limit, max_download_bytes)
    if download_limit <= 0:
        notes.append("文件未下载：已达到本轮合并转发媒体大小上限。")
        return 0

    try:
        if direct_url:
            url = direct_url
        else:
            info = await bot.call_api(
                "get_group_file_url",
                group_id=event.group_id,
                file_id=file_id,
            )
            url = str(info.get("url", ""))
        if not url:
            raise RuntimeError("empty file url")
        content, mime_type = await download_url(
            url,
            timeout=config.bampi_web_search_timeout,
            max_bytes=download_limit,
        )
        preferred_name = resolve_inbox_preferred_name(
            preferred_name=extract_segment_filename(segment) or file_id,
            download_url=url,
            mime_type=mime_type,
            content=content,
        )
        saved = await save_bytes_to_inbox(
            workspace_dir,
            content,
            preferred_name=preferred_name,
            mime_type=mime_type,
        )
        logger.info(
            f"bampi_chat saved group file file_id={file_id} "
            f"preferred_name={preferred_name!r} path={saved} size={len(content)}"
        )
        saved_paths.append(saved)
        return len(content)
    except Exception as exc:
        logger.warning(f"bampi_chat failed to download group file file_id={file_id}: {exc}")
        if from_reply:
            notes.append(f"下载回复引用文件失败: {exc}")
        else:
            notes.append(f"下载群文件失败: {exc}")
        return 0


async def download_url(url: str, *, timeout: float, max_bytes: int) -> tuple[bytes, str]:
    def _download() -> tuple[bytes, str]:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; BampiBot/0.1)"})
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"download exceeds limit: {max_bytes} bytes")
        return data, content_type

    return await asyncio.to_thread(_download)


def guess_mime_type(filename: str | None, *, default: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename or "")
    return mime_type or default


async def save_bytes_to_inbox(
    workspace_dir: str,
    data: bytes,
    *,
    preferred_name: str | None,
    mime_type: str | None,
) -> str:
    inbox = ensure_workspace_dirs(workspace_dir) / "inbox"
    clean_name = sanitize_filename(preferred_name)
    suffix = "".join(Path(clean_name or "").suffixes)
    if not suffix:
        suffix = mimetypes.guess_extension(mime_type or "") or ""
    if not suffix:
        suffix = infer_extension_from_content(data)

    stem = ""
    if clean_name:
        stem = Path(clean_name).name
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        stem = stem.strip().strip(".")

    unique = uuid.uuid4().hex[:12]
    if stem:
        filename = f"{stem}-{unique}{suffix}"
    else:
        filename = f"{unique}{suffix}"
    path = inbox / filename
    await asyncio.to_thread(path.write_bytes, data)
    return f"inbox/{filename}"


def posix_path_to_file_uri(path: PurePosixPath | str) -> str:
    normalized = PurePosixPath(path).as_posix()
    return f"file://{quote(normalized, safe='/')}"


async def prepare_group_file_upload(
    path: Path,
    config: BampiChatConfig,
) -> PreparedGroupFileUpload:
    host_dir = config.bampi_group_file_upload_host_dir.strip()
    container_dir = config.bampi_group_file_upload_container_dir.strip()

    if host_dir and container_dir:
        staged_path: Path | None = None
        try:
            if not container_dir.startswith("/"):
                raise ValueError("container upload dir must be an absolute POSIX path")

            staging_dir = Path(host_dir).expanduser()
            if not staging_dir.is_absolute():
                staging_dir = staging_dir.resolve()
            await asyncio.to_thread(staging_dir.mkdir, parents=True, exist_ok=True)

            staged_name = f"{uuid.uuid4().hex[:12]}-{path.name}"
            staged_path = staging_dir / staged_name
            await asyncio.to_thread(shutil.copy2, path, staged_path)

            container_path = PurePosixPath(container_dir) / staged_name
            file_uri = posix_path_to_file_uri(container_path)
            logger.info(
                f"bampi_chat staged group upload source={path} "
                f"staged={staged_path} file_uri={file_uri}"
            )
            return PreparedGroupFileUpload(file_uri=file_uri, cleanup_paths=[staged_path])
        except Exception as exc:
            if staged_path is not None:
                try:
                    staged_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(f"bampi_chat failed to cleanup staged upload file: {staged_path}")
            logger.warning(
                f"bampi_chat failed to stage group upload path={path} "
                f"host_dir={host_dir!r} container_dir={container_dir!r}: {exc}. "
                f"Falling back to local file URI."
            )

    file_uri = path.resolve().as_uri()
    logger.info(f"bampi_chat using local group upload file_uri={file_uri}")
    return PreparedGroupFileUpload(file_uri=file_uri)


async def prepare_outbound_image(
    path: Path,
    config: BampiChatConfig,
) -> PreparedOutboundImage:
    file_size = await asyncio.to_thread(lambda: path.stat().st_size)
    if file_size <= config.bampi_max_inline_image_size:
        data = await asyncio.to_thread(path.read_bytes)
        logger.info(
            f"bampi_chat prepared inline image source={path} size={len(data)}"
        )
        return PreparedOutboundImage(source=data)

    prepared = await prepare_group_file_upload(path, config)
    if prepared.cleanup_paths:
        logger.info(
            f"bampi_chat prepared staged image source={path} file_uri={prepared.file_uri}"
        )
        return PreparedOutboundImage(
            source=prepared.file_uri,
            cleanup_paths=prepared.cleanup_paths,
        )

    data = await asyncio.to_thread(path.read_bytes)
    logger.warning(
        f"bampi_chat image staging unavailable, falling back to inline base64 "
        f"source={path} size={len(data)}"
    )
    return PreparedOutboundImage(source=data)


def snapshot_outbox(workspace_dir: str) -> dict[str, float]:
    outbox = ensure_workspace_dirs(workspace_dir) / "outbox"
    snapshot: dict[str, float] = {}
    for path in outbox.iterdir():
        if path.is_file():
            snapshot[path.name] = path.stat().st_mtime
    return snapshot


def find_last_assistant_message(messages: list[Any]) -> AssistantMessage | None:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            return message
    return None


def extract_text_blocks(message: AssistantMessage | None) -> str:
    if message is None:
        return ""
    if isinstance(message.content, str):
        return message.content.strip()

    parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def strip_streamed_prefix(full_text: str, streamed_text: str) -> str:
    if not streamed_text:
        return full_text
    if full_text.startswith(streamed_text):
        return full_text[len(streamed_text) :]
    logger.warning(
        f"bampi_chat live text prefix mismatch "
        f"streamed={log_preview(streamed_text)!r} "
        f"full={log_preview(full_text)!r}"
    )
    return full_text


def collect_outbox_files(
    workspace_dir: str,
    *,
    before: dict[str, float],
    text: str,
) -> list[Path]:
    outbox = ensure_workspace_dirs(workspace_dir) / "outbox"
    candidates: dict[str, Path] = {}

    for path in outbox.iterdir():
        if not path.is_file():
            continue
        previous = before.get(path.name)
        if previous is None or path.stat().st_mtime > previous:
            candidates[path.name] = path

    pattern = re.compile(r"(?P<path>(?:outbox/|/workspace(?:/[^/\s`'\"()]+)*/outbox/)[^\s`'\"()]+)")
    for match in pattern.finditer(text):
        raw = match.group("path")
        normalized = raw
        if normalized.startswith("/workspace/"):
            _, _, suffix = normalized.partition("/outbox/")
            normalized = f"outbox/{suffix}" if suffix else normalized
        if normalized.startswith("outbox/"):
            path = outbox / normalized.removeprefix("outbox/")
            if path.is_file():
                candidates[path.name] = path

    return sorted(candidates.values(), key=lambda item: item.name.lower())


def build_group_reply_message(
    *,
    config: BampiChatConfig,
    target: GroupReplyTarget,
    text: str,
    parse_outbound_markup: bool = False,
) -> Message:
    message = Message()
    if config.bampi_reply_with_quote and target.reply_message_id is not None:
        message += MessageSegment.reply(target.reply_message_id)
    if text:
        if parse_outbound_markup:
            append_composed_text(
                message,
                text,
                options=compose_options_from_config(config),
            )
        else:
            message += MessageSegment.text(text)
    return message


async def append_reply_body(
    message: Message,
    text: str,
    config: BampiChatConfig,
) -> int:
    """Append *text* to *message*, rendering layout-bearing blocks as images.

    Code, formulas and tables become image segments placed where they appeared
    in the reply; everything else goes through the normal outbound markup
    parser. Returns the number of rendered blocks.

    If rendering is disabled or fails, the reply is appended as plain Markdown
    text, so no path here can lose content.
    """
    if not text:
        return 0

    options = rich_render_options_from_config(config)
    renderer = get_rich_renderer(config) if options.enabled else None
    plan = await build_delivery_plan(text, renderer=renderer, options=options)

    compose_options = compose_options_from_config(config)
    rendered = 0
    for part in plan:
        if isinstance(part, TextPart):
            append_composed_text(message, part.text, options=compose_options)
        elif isinstance(part, ImagePart):
            message += MessageSegment.image(part.png)
            rendered += 1
    return rendered


async def _send_group_message_via_bot(
    *,
    bot: Bot,
    target: GroupReplyTarget,
    message: Message,
) -> None:
    await bot.call_api(
        "send_group_msg",
        group_id=target.group_id,
        message=message,
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
                    streamed_text=reporter.streamed_text,
                    streamed_any_text=reporter.streamed_any_text,
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


async def send_agent_response_to_target(
    *,
    bot: Bot,
    target: GroupReplyTarget,
    config: BampiChatConfig,
    workspace_dir: str,
    assistant_message: AssistantMessage | None,
    outbox_before: dict[str, float],
    streamed_text: str = "",
    streamed_any_text: bool = False,
    text_prefix: str = "",
    log_label: str = "auto-resume",
    failure_message_builder: Callable[[FailureAssessment], str] = build_background_failure_message,
    empty_reply_text: str | None = "后续处理完成，无新内容可回复。可以发送新消息继续。",
) -> ResponseDispatchResult:
    full_text = extract_text_blocks(assistant_message)
    text = strip_streamed_prefix(full_text, streamed_text)
    text = text.lstrip()
    if text_prefix and text:
        text = f"{text_prefix}{text}"
    files = collect_outbox_files(workspace_dir, before=outbox_before, text=full_text)
    stop_reason = getattr(assistant_message, "stop_reason", None)
    error_message = normalize_text(getattr(assistant_message, "error_message", None))
    logger.info(
        f"bampi_chat preparing {log_label} reply group_id={target.group_id} "
        f"reply_message_id={target.reply_message_id} "
        f"text={log_preview(text)!r} "
        f"files={[path.name for path in files]} "
        f"stop_reason={stop_reason} "
        f"error={log_preview(error_message)!r}"
    )

    if stop_reason in {StopReason.ABORTED, "aborted"}:
        logger.info(
            f"bampi_chat skipped aborted {log_label} reply group_id={target.group_id} "
            f"reply_message_id={target.reply_message_id}"
        )
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    if not text and not files:
        if error_message:
            logger.warning(
                f"bampi_chat {log_label} returned no deliverable content "
                f"group_id={target.group_id} "
                f"reply_message_id={target.reply_message_id} "
                f"stop_reason={stop_reason} "
                f"error={log_preview(error_message)!r}"
            )
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=config,
                    target=target,
                    text=failure_message_builder(assess_failure(error_message)),
                ),
            )
            return ResponseDispatchResult(delivered=False, rollback_context=True)
        if streamed_any_text:
            logger.info(
                f"bampi_chat {log_label} fully covered by live stream "
                f"group_id={target.group_id} "
                f"reply_message_id={target.reply_message_id}"
            )
            return ResponseDispatchResult(delivered=True, rollback_context=False)

        logger.warning(
            f"bampi_chat {log_label} returned empty content "
            f"group_id={target.group_id} "
            f"reply_message_id={target.reply_message_id}"
        )
        if empty_reply_text:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=config,
                    target=target,
                    text=empty_reply_text,
                ),
            )
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    message = build_group_reply_message(config=config, target=target, text="")
    rendered_block_count = await append_reply_body(message, text, config)

    uploaded_files: list[Path] = []
    staged_upload_files: list[Path] = []
    failed_artifacts: list[str] = []
    sent_image_count = 0
    try:
        for path in files:
            if not is_image_file(path):
                continue
            try:
                prepared_image = await prepare_outbound_image(path, config)
                staged_upload_files.extend(prepared_image.cleanup_paths)
                message += MessageSegment.image(prepared_image.source)
                sent_image_count += 1
            except Exception:
                logger.exception(f"failed to prepare {log_label} outbox image: {path}")
                failed_artifacts.append(path.name)

        if message:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=message,
            )
            logger.info(
                f"bampi_chat {log_label} reply sent group_id={target.group_id} "
                f"reply_message_id={target.reply_message_id} "
                f"has_text={bool(text)} "
                f"rich_block_count={rendered_block_count} "
                f"image_count={sent_image_count}"
            )

        for path in files:
            if is_image_file(path):
                if path.name in failed_artifacts:
                    continue
                uploaded_files.append(path)
                continue
            try:
                upload = await prepare_group_file_upload(path, config)
                staged_upload_files.extend(upload.cleanup_paths)
                await bot.call_api(
                    "upload_group_file",
                    group_id=target.group_id,
                    file=upload.file_uri,
                    name=path.name,
                )
                logger.info(f"bampi_chat {log_label} uploaded outbox file group_id={target.group_id} path={path}")
                uploaded_files.append(path)
            except Exception:
                logger.exception(f"failed to upload {log_label} outbox file: {path}")
                failed_artifacts.append(path.name)
        if failed_artifacts:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=config,
                    target=target,
                    text=f"有 {len(failed_artifacts)} 个文件已生成但发送失败。",
                ),
            )
    finally:
        for path in uploaded_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup {log_label} outbox file: {path}")
        for staged_path in staged_upload_files:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup {log_label} staged upload file: {staged_path}")
    return ResponseDispatchResult(delivered=True, rollback_context=False)


async def send_agent_response(
    *,
    bot: Bot,
    event: GroupMessageEvent,
    matcher: Matcher,
    config: BampiChatConfig,
    workspace_dir: str,
    assistant_message: AssistantMessage | None,
    outbox_before: dict[str, float],
    streamed_text: str = "",
    streamed_any_text: bool = False,
) -> ResponseDispatchResult:
    full_text = extract_text_blocks(assistant_message)
    text = strip_streamed_prefix(full_text, streamed_text)
    text = text.lstrip()
    files = collect_outbox_files(workspace_dir, before=outbox_before, text=full_text)
    stop_reason = getattr(assistant_message, "stop_reason", None)
    error_message = normalize_text(getattr(assistant_message, "error_message", None))
    logger.info(
        f"bampi_chat preparing reply group_id={event.group_id} "
        f"message_id={event.message_id} "
        f"text={log_preview(text)!r} "
        f"files={[path.name for path in files]} "
        f"stop_reason={stop_reason} "
        f"error={log_preview(error_message)!r}"
    )

    if stop_reason in {StopReason.ABORTED, "aborted"}:
        logger.info(
            f"bampi_chat skipped aborted reply group_id={event.group_id} "
            f"message_id={event.message_id}"
        )
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    if not text and not files:
        if error_message:
            logger.warning(
                f"bampi_chat assistant returned no deliverable content "
                f"group_id={event.group_id} "
                f"message_id={event.message_id} "
                f"stop_reason={stop_reason} "
                f"error={log_preview(error_message)!r}"
            )
            await matcher.send(build_reply_failure_message(assess_failure(error_message)))
            return ResponseDispatchResult(delivered=False, rollback_context=True)
        if streamed_any_text:
            logger.info(
                f"bampi_chat final reply fully covered by live stream "
                f"group_id={event.group_id} "
                f"message_id={event.message_id}"
            )
            return ResponseDispatchResult(delivered=True, rollback_context=False)

        logger.warning(
            f"bampi_chat assistant returned empty content "
            f"group_id={event.group_id} "
            f"message_id={event.message_id}"
        )
        await matcher.send("⚠️ 这次没有生成可发送的内容，请换个说法再试一次。")
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    message = Message()
    if config.bampi_reply_with_quote:
        message += MessageSegment.reply(event.message_id)
    rendered_block_count = await append_reply_body(message, text, config)

    uploaded_files: list[Path] = []
    staged_upload_files: list[Path] = []
    failed_artifacts: list[str] = []
    sent_image_count = 0
    try:
        for path in files:
            if not is_image_file(path):
                continue
            try:
                prepared_image = await prepare_outbound_image(path, config)
                staged_upload_files.extend(prepared_image.cleanup_paths)
                message += MessageSegment.image(prepared_image.source)
                sent_image_count += 1
            except Exception:
                logger.exception(f"failed to prepare outbox image: {path}")
                failed_artifacts.append(path.name)

        if message:
            await matcher.send(message)
            logger.info(
                f"bampi_chat reply sent group_id={event.group_id} "
                f"message_id={event.message_id} "
                f"has_text={bool(text)} "
                f"rich_block_count={rendered_block_count} "
                f"image_count={sent_image_count}"
            )
        else:
            logger.warning(
                f"bampi_chat reply skipped because message was empty "
                f"group_id={event.group_id} "
                f"message_id={event.message_id}"
            )

        for path in files:
            if is_image_file(path):
                if path.name in failed_artifacts:
                    continue
                uploaded_files.append(path)
                continue
            try:
                upload = await prepare_group_file_upload(path, config)
                staged_upload_files.extend(upload.cleanup_paths)
                await bot.call_api(
                    "upload_group_file",
                    group_id=event.group_id,
                    file=upload.file_uri,
                    name=path.name,
                )
                logger.info(f"bampi_chat uploaded outbox file group_id={event.group_id} path={path}")
                uploaded_files.append(path)
            except Exception:
                logger.exception(f"failed to upload outbox file: {path}")
                failed_artifacts.append(path.name)
        if failed_artifacts:
            await matcher.send(
                f"有 {len(failed_artifacts)} 个文件已生成但发送失败。"
            )
    finally:
        for path in uploaded_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup outbox file: {path}")
        for staged_path in staged_upload_files:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup staged upload file: {staged_path}")
    return ResponseDispatchResult(delivered=True, rollback_context=False)
