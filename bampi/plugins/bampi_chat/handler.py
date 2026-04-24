from __future__ import annotations

import asyncio
import base64
import mimetypes
import random
import re
import shutil
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from nonebot import get_driver, logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.matcher import Matcher
from nonebot.plugin import on_message

from bampy.ai import ImageContent, TextContent, UserMessage
from bampy.ai.types import AssistantMessage, StopReason
from bampy.app import AgentSession

from .config import BampiChatConfig
from .skills import (
    ExplicitSkillResolution,
    build_explicit_skill_payload_text,
    describe_skill_resource_path,
    format_skill_details,
    format_skill_help,
    format_skill_list,
    install_skills_from_source,
    load_chat_skills,
    parse_skill_command,
    resolve_explicit_skills,
    strip_explicit_skill_mentions,
)
from .session_manager import BackgroundWaitReminderEvent, GroupSessionManager
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
class ResponseDispatchResult:
    delivered: bool
    rollback_context: bool = False


@dataclass(slots=True)
class ProgressMessage:
    text: str
    quote: bool = False
    tool_call_id: str | None = None


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
    "web_search": "🌐",
    "browser": "🧭",
    "service": "🚀",
    "schedule": "⏰",
}

STOP_COMMAND = "/stop"
CLEAR_COMMANDS = {"/clear", "/new"}
COMPACT_COMMAND = "/compact"
ACTIVE_SESSION_BUSY_MESSAGE = "当前群里已有进行中的会话，只有发起者可以继续跟进；如需中止，请让发起者发送 /stop。"
ACTIVE_SESSION_WINDING_DOWN_MESSAGE = "当前会话正在收尾，请稍等结果发出。"
ACTIVE_SESSION_BACKGROUND_WAIT_MESSAGE = "当前会话正在后台等待命令结果，完成后会自动继续，请稍等。"
STOP_NO_ACTIVE_MESSAGE = "当前没有你发起的进行中会话。"
STOP_NOT_OWNER_MESSAGE = "当前会话不是你发起的，不能由你停止；如需中止，请让发起者发送 /stop。"
STOPPED_SESSION_MESSAGE = "已停止你发起的当前会话。"
STOPPED_WAITING_SESSION_MESSAGE = "已停止你发起的当前会话，并取消等待中的后台命令。"
FORCE_STOPPED_SESSION_MESSAGE = "已强制停止当前群会话。"
FORCE_STOPPED_WAITING_SESSION_MESSAGE = "已强制停止当前群会话，并取消等待中的后台命令。"
CLEARED_SESSION_MESSAGE = "已清空当前群的对话上下文。"
CLEAR_NO_CONTEXT_MESSAGE = "当前群还没有可清空的上下文。"
COMPACT_NO_CONTEXT_MESSAGE = "当前群还没有可压缩的上下文。"
COMPACT_FORBIDDEN_MESSAGE = "只有 NoneBot superuser 可以使用 /compact。"


@dataclass(slots=True)
class GroupReplyTarget:
    group_id: int
    user_id: int | None = None
    reply_message_id: int | None = None


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
        event: GroupMessageEvent,
        config: BampiChatConfig,
    ) -> None:
        self._bot = bot
        self._event = event
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
                if item.quote:
                    message += MessageSegment.reply(self._event.message_id)
                message += MessageSegment.text(item.text)
                response = await self._bot.call_api(
                    "send_group_msg",
                    group_id=self._event.group_id,
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
                    f"group_id={self._event.group_id} "
                    f"message_id={self._event.message_id}"
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
        self._enqueue("上下文有点长，我先整理一下前面的聊天记录，再继续。")

    def _handle_message_start(self) -> None:
        self._last_seen_text = ""
        self._pending_text = ""
        self._streamed_text = ""
        self._streamed_any_text = False

    def announce_skill_loading(self, skill_names: list[str]) -> None:
        if not self._live_progress_enabled or not skill_names:
            return
        limit = self._config.bampi_live_progress_max_tool_updates
        if limit > 0 and self._tool_updates_sent >= limit:
            return
        if self._config.bampi_live_text_stream_enabled:
            self._flush_pending_text(force=True)
        self._tool_updates_sent += 1
        self._enqueue(format_skill_load_message(skill_names))

    def _handle_tool_start(self, event: Any) -> None:
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
        self._enqueue(payload, preserve_whitespace=True)

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
            f"group_id={self._event.group_id} "
            f"message_id={self._event.message_id} "
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
                f"group_id={self._event.group_id} "
                f"message_id={self._event.message_id} "
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
                    f"group_id={self._event.group_id} "
                    f"message_id={self._event.message_id} "
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
            f"group_id={self._event.group_id} "
            f"message_id={self._event.message_id} "
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


def format_skill_load_message(skill_names: list[str]) -> str:
    unique_names = list(dict.fromkeys(name for name in skill_names if name))
    if not unique_names:
        return f"{TOOL_PROGRESS_EMOJIS['skill']} 正在加载 skill"
    if len(unique_names) == 1:
        return f"{TOOL_PROGRESS_EMOJIS['skill']} 正在加载 skill：{unique_names[0]}"
    return f"{TOOL_PROGRESS_EMOJIS['skill']} 正在加载 skills：{', '.join(unique_names)}"


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
    if tool_name == "web_search":
        query = render_tool_progress_value(payload.get("query") or payload.get("q"), "查询内容")
        return f"正在搜索网页：{query}"
    if tool_name == "browser":
        action = render_tool_progress_value(payload.get("action"), "操作")
        url = render_tool_progress_value(payload.get("url"), "")
        selector = render_tool_progress_value(payload.get("selector"), "")
        if action in {"open", "goto"} and url:
            return f"正在浏览网页（{action}）：{url}"
        if action in {"click", "type", "wait", "extract", "screenshot"} and selector:
            return f"正在操作网页（{action}）：{selector}"
        if action in {"pages", "switch", "close_page", "reload", "back", "forward", "scroll", "reset"}:
            return f"正在操作浏览器：{action}"
        return f"正在操作浏览器：{action}"
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
    if bool(getattr(status, "is_waiting_background", False)):
        return ACTIVE_SESSION_BACKGROUND_WAIT_MESSAGE
    active_user_id = getattr(status, "active_user_id", None)
    if requester_user_id is not None and active_user_id == requester_user_id:
        return ACTIVE_SESSION_WINDING_DOWN_MESSAGE
    if not bool(getattr(status, "is_streaming", False)):
        return ACTIVE_SESSION_WINDING_DOWN_MESSAGE
    return ACTIVE_SESSION_BUSY_MESSAGE


def build_stop_success_message(*, force: bool, stopped_background_waits: bool) -> str:
    if force:
        return (
            FORCE_STOPPED_WAITING_SESSION_MESSAGE
            if stopped_background_waits
            else FORCE_STOPPED_SESSION_MESSAGE
        )
    return (
        STOPPED_WAITING_SESSION_MESSAGE
        if stopped_background_waits
        else STOPPED_SESSION_MESSAGE
    )


def _format_skill_diagnostics(diagnostics: list[Any]) -> str:
    if not diagnostics:
        return ""

    lines = ["Skill 诊断："]
    for diagnostic in diagnostics[:5]:
        path = getattr(diagnostic, "path", "")
        message = getattr(diagnostic, "message", "")
        lines.append(f"- {message} ({path})")
    if len(diagnostics) > 5:
        lines.append(f"- 其余 {len(diagnostics) - 5} 条已省略")
    return "\n".join(lines)


def _format_missing_skills_message(names: list[str]) -> str:
    missing = ", ".join(names)
    return (
        f"这些 skill 不存在或当前不可用：{missing}\n"
        "先用 `/skills` 查看已安装 skill，"
        "或发送/引用 skill 文件后执行 `/skill install`，也可以用 `/skill install https://...` 安装。"
    )


async def _handle_skill_command(
    *,
    bot: Bot,
    event: GroupMessageEvent,
    command_text: str,
    group_id: str,
    matcher: Matcher,
    session_manager: GroupSessionManager,
    config: BampiChatConfig,
) -> bool:
    command = parse_skill_command(command_text)
    if command is None:
        return False

    workspace_dir = session_manager.workspace_dir_for_group(group_id)

    if command.action == "help":
        await matcher.send(format_skill_help())
        return True

    if command.action == "list":
        loaded = load_chat_skills(workspace_dir)
        message = format_skill_list(loaded.skills, workspace_dir=workspace_dir)
        diagnostics = _format_skill_diagnostics(loaded.diagnostics)
        if diagnostics:
            message = f"{message}\n\n{diagnostics}"
        await matcher.send(message)
        return True

    if command.action == "show":
        if not command.argument:
            await matcher.send("用法：`/skill show <name>`")
            return True

        loaded = load_chat_skills(workspace_dir)
        by_name = {skill.name.lower(): skill for skill in loaded.skills}
        skill = by_name.get(command.argument.lower())
        if skill is None:
            await matcher.send(_format_missing_skills_message([command.argument]))
            return True

        message = format_skill_details(skill, workspace_dir=workspace_dir)
        diagnostics = _format_skill_diagnostics(loaded.diagnostics)
        if diagnostics:
            message = f"{message}\n\n{diagnostics}"
        await matcher.send(message)
        return True

    if command.action == "install":
        status = await session_manager.inspect_interaction(group_id)
        if status.is_active:
            message = (
                ACTIVE_SESSION_WINDING_DOWN_MESSAGE
                if not status.is_streaming
                else ACTIVE_SESSION_BUSY_MESSAGE
            )
            await matcher.send(message)
            return True

        install_sources: list[str] = []
        if command.argument:
            parsed = urlparse(command.argument)
            if parsed.scheme in {"http", "https"}:
                install_sources.append(command.argument)
            else:
                await matcher.send(
                    "你发送的url有误。\n"
                    "请直接发送或引用 skill 压缩包/Markdown 文件后执行 `/skill install`，"
                    "或使用 `/skill install https://...`。"
                )
                return True
        else:
            try:
                media = await collect_incoming_media(bot, event, config, workspace_dir)
            except Exception:
                logger.exception("bampi_chat failed to collect media for skill installation")
                await matcher.send("读取这次安装消息里的附件失败了，可以重新发送或重新引用一次。")
                return True

            install_sources.extend(media.saved_paths)
            install_sources.extend(media.reply_saved_paths)
            if not install_sources:
                await matcher.send(
                    "没有找到可安装的 skill 文件。\n"
                    "请直接发送或引用一个 zip/tar/Markdown skill 文件，再执行 `/skill install`；"
                    "也可以使用 `/skill install https://...`。"
                )
                return True

        installed_names: list[str] = []
        replaced_names: list[str] = []
        collected_diagnostics: list[Any] = []
        try:
            for source in install_sources:
                result = install_skills_from_source(
                    source,
                    workspace_dir=workspace_dir,
                    force=command.force,
                    max_bytes=config.bampi_max_download_size,
                    timeout=config.bampi_web_search_timeout,
                )
                installed_names.extend(result.installed_names)
                replaced_names.extend(result.replaced_names)
                collected_diagnostics.extend(result.diagnostics)
        except Exception as exc:
            await matcher.send(f"安装 skill 失败：{exc}")
            return True

        await session_manager.release(group_id)

        lines = [
            f"已安装 {len(installed_names)} 个 skill：{', '.join(installed_names)}",
            f"安装目录：{Path(workspace_dir, '.agents/skills').resolve().as_posix()}",
            "显式调用：在普通消息最开头写 `/skill-name`。",
            "当前群会话已刷新；其他现有会话会在下次重建后看到新 skill。",
        ]
        if replaced_names:
            lines.append(f"已覆盖：{', '.join(replaced_names)}")
        diagnostics = _format_skill_diagnostics(collected_diagnostics)
        if diagnostics:
            lines.append("")
            lines.append(diagnostics)
        await matcher.send("\n".join(lines))
        return True

    await matcher.send(format_skill_help())
    return True


def register_handlers(config: BampiChatConfig, session_manager: GroupSessionManager) -> None:
    limiter = GroupRateLimiter(
        config.bampi_rate_limit,
        config.bampi_rate_limit_window_seconds,
    )
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

        original_text = (event.get_plaintext() or "").strip()
        raw_text = normalize_text(original_text)
        workspace_dir = session_manager.workspace_dir_for_group(group_id)
        logger.info(
            f"bampi_chat received group_id={event.group_id} "
            f"user_id={event.user_id} "
            f"message_id={event.message_id} "
            f"to_me={getattr(event, 'to_me', False)} "
            f"segments={summarize_segments(event.message)} "
            f"text={log_preview(raw_text)!r}"
        )

        if await _handle_skill_command(
            bot=bot,
            event=event,
            command_text=original_text,
            group_id=group_id,
            matcher=matcher,
            session_manager=session_manager,
            config=config,
        ):
            return

        if is_clear_command(raw_text):
            status = await session_manager.inspect_interaction(group_id)
            if status.is_active:
                await matcher.send(interaction_busy_message(status, requester_user_id=user_id))
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
                await matcher.send("这次上下文压缩失败了，请检查模型配置或稍后再试。")
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
            if not status.is_active:
                await matcher.send(STOP_NO_ACTIVE_MESSAGE)
                return
            requester_is_superuser = is_nonebot_superuser(user_id)
            force_stop = requester_is_superuser and status.active_user_id != user_id
            if status.active_user_id != user_id and not requester_is_superuser:
                await matcher.send(STOP_NOT_OWNER_MESSAGE)
                return
            if status.managed is None:
                await matcher.send(ACTIVE_SESSION_WINDING_DOWN_MESSAGE)
                return

            stop_reason = "stopped by superuser" if force_stop else "stopped by session owner"
            stop_result = await session_manager.stop_interaction(
                group_id,
                reason=stop_reason,
            )
            if not stop_result.aborted_streaming and not stop_result.stopped_background_waits:
                await matcher.send(ACTIVE_SESSION_WINDING_DOWN_MESSAGE)
                return
            logger.info(
                f"bampi_chat stop requested group_id={group_id} "
                f"user_id={user_id} "
                f"message_id={event.message_id} "
                f"force_stop={force_stop} "
                f"aborted_streaming={stop_result.aborted_streaming} "
                f"stopped_background_waits={stop_result.stopped_background_waits} "
                f"stopped_background_session_ids={stop_result.stopped_background_session_ids}"
            )
            await matcher.send(
                build_stop_success_message(
                    force=force_stop,
                    stopped_background_waits=stop_result.stopped_background_waits,
                )
            )
            return

        active_status = await session_manager.inspect_interaction(group_id)
        decision = should_respond(
            event,
            bot_self_id=str(bot.self_id),
            config=config,
            random_value=random.random(),
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
                media = await collect_incoming_media(bot, event, config, workspace_dir)
            except Exception:
                logger.exception("bampi_chat failed to collect follow-up media")
                await matcher.send("这条跟进消息处理失败了，你可以重新发一次。")
                return
            explicit_skills = resolve_explicit_skills(
                decision.cleaned_text,
                workspace_dir=workspace_dir,
            )
            if explicit_skills.missing_names:
                await matcher.send(_format_missing_skills_message(explicit_skills.missing_names))
                return
            user_message = build_user_message(
                event,
                decision.cleaned_text,
                media,
                workspace_dir=workspace_dir,
                explicit_skills=explicit_skills,
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
                await matcher.send("这会儿有点忙，稍后再戳我一下。")
            return

        try:
            reservation = await session_manager.reserve_interaction(group_id, user_id)
        except Exception:
            logger.exception("bampi_chat failed to create or restore group session")
            await matcher.send("agent 会话初始化失败了，先检查模型配置、API Key 或会话目录。")
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
        try:
            media = await collect_incoming_media(bot, event, config, workspace_dir)
            logger.info(
                f"bampi_chat media collected group_id={group_id} "
                f"message_id={event.message_id} "
                f"inline_images={len(media.inline_images)} "
                f"saved_paths={media.saved_paths} "
                f"reply_inline_images={len(media.reply_inline_images)} "
                f"reply_saved_paths={media.reply_saved_paths} "
                f"notes={media.notes} "
                f"reply_notes={media.reply_notes}"
            )
            explicit_skills = resolve_explicit_skills(
                decision.cleaned_text,
                workspace_dir=workspace_dir,
            )
            if explicit_skills.missing_names:
                await matcher.send(_format_missing_skills_message(explicit_skills.missing_names))
                return
            user_message = build_user_message(
                event,
                decision.cleaned_text,
                media,
                workspace_dir=workspace_dir,
                explicit_skills=explicit_skills,
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
                background_resume_waits: dict[str, str | None] = {}

                def _capture_background_wait(event: Any) -> None:
                    if getattr(event, "type", None) != "tool_execution_end":
                        return
                    if getattr(event, "tool_name", "") != "bash":
                        return
                    if bool(getattr(event, "is_error", False)):
                        return
                    result = getattr(event, "result", None)
                    details = getattr(result, "details", None)
                    if not isinstance(details, dict):
                        return
                    session_id = str(details.get("session_id", "")).strip()
                    if not session_id or not bool(details.get("notify_on_exit")):
                        return
                    command = details.get("command")
                    background_resume_waits[session_id] = str(command).strip() if command is not None else None

                unsubscribe_background_wait = managed.session.subscribe(_capture_background_wait)
                reporter = LiveProgressReporter(bot=bot, event=event, config=config)
                reporter.start(managed.session)
                if explicit_skills.skills:
                    reporter.announce_skill_loading([skill.name for skill in explicit_skills.skills])
                try:
                    logger.info(
                        f"bampi_chat prompt start group_id={group_id} "
                        f"message_id={event.message_id} "
                        f"content_blocks={len(user_message.content)}"
                    )
                    try:
                        await managed.session.prompt(user_message, source="qq_group")
                    except Exception:
                        logger.exception("bampi_chat session prompt failed")
                        await matcher.send("这次调用 agent 失败了，检查一下模型配置、网络或工具环境。")
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
                    for session_id, command in sorted(background_resume_waits.items()):
                        await session_manager.register_background_wait(
                            group_id,
                            session_id,
                            owner_user_id=user_id,
                            callback=_create_background_resume_callback(
                                bot=bot,
                                config=config,
                                managed=managed,
                                group_id=group_id,
                                user_id=int(event.user_id),
                                reply_message_id=event.message_id,
                                workspace_dir=workspace_dir,
                            ),
                            command=command,
                            reminder_after_seconds=config.bampi_background_wait_reminder_seconds,
                            reminder_callback=_create_background_wait_reminder_callback(
                                bot=bot,
                                config=config,
                                group_id=group_id,
                                user_id=int(event.user_id),
                                reply_message_id=event.message_id,
                            ),
                        )
                finally:
                    unsubscribe_background_wait()
                    await reporter.close()
        except Exception:
            logger.exception("bampi_chat failed while preparing or delivering interaction")
            await matcher.send("处理这条消息时出了点问题，请稍后再试一次。")
            return
        finally:
            if reservation.action == "start":
                await session_manager.complete_interaction(group_id)


def should_respond(
    event: PlaintextEvent,
    *,
    bot_self_id: str,
    config: BampiChatConfig,
    random_value: float,
) -> TriggerDecision:
    if not config.bampi_enabled:
        return TriggerDecision(False)

    text = normalize_text(event.get_plaintext())
    reply_to_bot = is_reply_to_bot(event.reply, bot_self_id)

    if bool(getattr(event, "to_me", False)) or reply_to_bot:
        return TriggerDecision(True, reason="to_me", direct=True, cleaned_text=text)

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

    if text and strip_explicit_skill_mentions(text) != text:
        return TriggerDecision(True, reason="skill", direct=True, cleaned_text=text)

    if text and config.bampi_random_reply_prob > 0 and random_value < config.bampi_random_reply_prob:
        return TriggerDecision(True, reason="random", direct=False, cleaned_text=text)

    return TriggerDecision(False)


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def extract_message_plaintext(message: Any) -> str:
    if message is None:
        return ""
    extractor = getattr(message, "extract_plain_text", None)
    if callable(extractor):
        try:
            return normalize_text(extractor())
        except Exception:
            logger.warning("bampi_chat failed to extract plain text from message object")
    return normalize_text(str(message))


def extract_segment_filename(segment: MessageSegment) -> str | None:
    for key in ("name", "file", "file_name", "filename"):
        raw_value = segment.data.get(key)
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
    workspace_dir: str | None = None,
    explicit_skills: ExplicitSkillResolution | None = None,
) -> UserMessage:
    effective_text = explicit_skills.cleaned_text if explicit_skills is not None else cleaned_text
    sender_name = display_name(event.sender)
    if effective_text:
        body = effective_text
    elif media.inline_images or media.saved_paths:
        body = "(无纯文本内容；本条消息仅包含媒体/文件)"
    elif media.reply_inline_images or media.reply_saved_paths:
        body = "(无纯文本内容；请结合回复引用内容理解)"
    else:
        body = "(无纯文本内容)"

    lines = [
        f"sender_name: {sender_name}",
        f"message_text: {body}",
    ]

    if event.reply is not None and getattr(event.reply, "sender", None) is not None:
        reply_name = display_name(event.reply.sender)
        reply_text = extract_message_plaintext(getattr(event.reply, "message", None))
        lines.append(f"reply_to_name: {reply_name}")
        if reply_text:
            lines.append(f"reply_message: {reply_text}")

    if explicit_skills is not None and explicit_skills.skills:
        lines.append("requested_skills:")
        lines.extend(f"- {skill.name}" for skill in explicit_skills.skills)

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

    content: list[TextContent | ImageContent] = [TextContent(text="\n".join(lines))]
    if explicit_skills is not None and explicit_skills.skills:
        if workspace_dir is None:
            raise ValueError("workspace_dir is required when explicit_skills are provided")
        payload = build_explicit_skill_payload_text(
            explicit_skills.skills,
            workspace_dir=workspace_dir,
        )
        if payload:
            content.append(TextContent(text=payload))
    content.extend(media.inline_images)
    content.extend(media.reply_inline_images)
    return UserMessage(content=content)


async def collect_incoming_media(
    bot: Bot,
    event: GroupMessageEvent,
    config: BampiChatConfig,
    workspace_dir: str,
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
) -> None:
    source = "reply" if from_reply else "message"
    if message is None:
        return

    for segment in message:
        if segment.type == "image":
            logger.info(
                f"bampi_chat processing image segment "
                f"group_id={event.group_id} "
                f"message_id={event.message_id} "
                f"source={source}"
            )
            await _handle_image_segment(
                segment,
                media,
                config,
                workspace_dir,
                from_reply=from_reply,
            )
        elif segment.type == "file":
            logger.info(
                f"bampi_chat processing file segment "
                f"group_id={event.group_id} "
                f"message_id={event.message_id} "
                f"source={source}"
            )
            await _handle_file_segment(
                bot,
                event,
                segment,
                media,
                config,
                workspace_dir,
                from_reply=from_reply,
            )


async def _handle_image_segment(
    segment: MessageSegment,
    media: IncomingMedia,
    config: BampiChatConfig,
    workspace_dir: str,
    *,
    from_reply: bool,
) -> None:
    inline_images, saved_paths, notes = _media_targets(media, from_reply=from_reply)
    url = segment.data.get("url")
    if not url:
        logger.warning("bampi_chat image segment missing download url")
        if from_reply:
            notes.append("回复引用里有图片，但适配器未提供可下载 URL。")
        else:
            notes.append("收到图片，但适配器未提供可下载 URL。")
        return

    try:
        content, mime_type = await download_url(url, timeout=config.bampi_web_search_timeout, max_bytes=config.bampi_max_download_size)
    except Exception as exc:
        logger.warning(f"bampi_chat failed to download image: {exc}")
        if from_reply:
            notes.append(f"下载回复引用图片失败: {exc}")
        else:
            notes.append(f"下载图片失败: {exc}")
        return

    mime_type = mime_type or guess_mime_type(url, default="image/png")
    if len(content) <= config.bampi_max_inline_image_size:
        logger.info(f"bampi_chat inlined image mime_type={mime_type} size={len(content)}")
        inline_images.append(
            ImageContent(
                data=base64.b64encode(content).decode("ascii"),
                mime_type=mime_type,
            )
        )
        return

    saved = await save_bytes_to_inbox(workspace_dir, content, preferred_name=segment.data.get("file"), mime_type=mime_type)
    logger.info(f"bampi_chat saved oversized image path={saved} size={len(content)}")
    saved_paths.append(saved)
    if from_reply:
        notes.append(f"回复引用中的图片过大，已保存到 {saved}")
    else:
        notes.append(f"图片过大，已保存到 {saved}")


async def _handle_file_segment(
    bot: Bot,
    event: GroupMessageEvent,
    segment: MessageSegment,
    media: IncomingMedia,
    config: BampiChatConfig,
    workspace_dir: str,
    *,
    from_reply: bool,
) -> None:
    file_id = segment.data.get("id") or segment.data.get("file_id")
    direct_url = str(segment.data.get("url", "")).strip()
    if not file_id and not direct_url:
        logger.warning("bampi_chat file segment missing file_id")
        _, _, notes = _media_targets(media, from_reply=from_reply)
        if from_reply:
            notes.append("回复引用里有文件，但缺少可下载标识。")
        else:
            notes.append("收到文件，但缺少 file_id。")
        return

    _, saved_paths, notes = _media_targets(media, from_reply=from_reply)
    try:
        if file_id:
            info = await bot.call_api(
                "get_group_file_url",
                group_id=event.group_id,
                file_id=file_id,
            )
            url = str(info.get("url", ""))
        else:
            url = direct_url
        if not url:
            raise RuntimeError("empty file url")
        content, mime_type = await download_url(url, timeout=config.bampi_web_search_timeout, max_bytes=config.bampi_max_download_size)
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
    except Exception as exc:
        logger.warning(f"bampi_chat failed to download group file file_id={file_id}: {exc}")
        if from_reply:
            notes.append(f"下载回复引用文件失败: {exc}")
        else:
            notes.append(f"下载群文件失败: {exc}")


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
) -> Message:
    message = Message()
    if config.bampi_reply_with_quote and target.reply_message_id is not None:
        message += MessageSegment.reply(target.reply_message_id)
    if config.bampi_at_sender and target.user_id is not None:
        message += MessageSegment.at(target.user_id)
    if text:
        message += MessageSegment.text(text)
    return message


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


def _create_background_wait_reminder_callback(
    *,
    bot: Bot,
    config: BampiChatConfig,
    group_id: str,
    user_id: int,
    reply_message_id: int,
) -> Callable[[BackgroundWaitReminderEvent], Awaitable[None]]:
    target = GroupReplyTarget(
        group_id=int(group_id),
        user_id=user_id,
        reply_message_id=reply_message_id,
    )

    async def _remind(event: BackgroundWaitReminderEvent) -> None:
        minutes = max(1, int(round(event.waited_seconds / 60)))
        command_hint = ""
        if event.command:
            command_hint = f"\n命令：{log_preview(event.command, limit=120)}"
        await _send_group_message_via_bot(
            bot=bot,
            target=target,
            message=build_group_reply_message(
                config=config,
                target=target,
                text=(
                    f"你之前启动的后台终端 `{event.session_id}` 已运行超过 {minutes} 分钟，"
                    "如果它看起来卡住了，可以发送 /stop 停止当前群会话。"
                    f"{command_hint}"
                ),
            ),
        )

    return _remind


def build_background_resume_follow_up_message(
    exit_event: BackgroundSessionExitEvent,
) -> UserMessage:
    lines = [
        "系统通知：你此前要求等待的后台终端命令已经结束，请继续基于结果完成任务。",
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


def _create_background_resume_callback(
    *,
    bot: Bot,
    config: BampiChatConfig,
    managed: Any,
    group_id: str,
    user_id: int,
    reply_message_id: int,
    workspace_dir: str,
) -> Callable[[BackgroundSessionExitEvent], Awaitable[None]]:
    target = GroupReplyTarget(
        group_id=int(group_id),
        user_id=user_id,
        reply_message_id=reply_message_id,
    )

    async def _resume(exit_event: BackgroundSessionExitEvent) -> None:
        async with managed.lock:
            managed.last_used_at = time.monotonic()
            outbox_before = snapshot_outbox(workspace_dir)
            logger.info(
                f"bampi_chat auto-resume start group_id={group_id} "
                f"session_id={exit_event.session_id} "
                f"exit_code={exit_event.returncode}"
            )
            try:
                managed.session.follow_up(
                    build_background_resume_follow_up_message(exit_event)
                )
                await managed.session.continue_()
            except Exception:
                logger.exception(
                    f"bampi_chat auto-resume failed group_id={group_id} "
                    f"session_id={exit_event.session_id}"
                )
                failure_message = build_group_reply_message(
                    config=config,
                    target=target,
                    text=(
                        f"后台终端 `{exit_event.session_id}` 已结束，但自动续跑失败了。"
                        "你可以直接发一句话让我继续处理。"
                    ),
                )
                await _send_group_message_via_bot(
                    bot=bot,
                    target=target,
                    message=failure_message,
                )
                return

            resume_message = build_background_resume_follow_up_message(exit_event)
            assistant_message = find_last_assistant_message(managed.session.messages)
            result = await send_background_agent_response(
                bot=bot,
                target=target,
                config=config,
                workspace_dir=workspace_dir,
                assistant_message=assistant_message,
                outbox_before=outbox_before,
            )

    return _resume


async def send_background_agent_response(
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
        f"bampi_chat preparing auto-resume reply group_id={target.group_id} "
        f"reply_message_id={target.reply_message_id} "
        f"text={log_preview(text)!r} "
        f"files={[path.name for path in files]} "
        f"stop_reason={stop_reason} "
        f"error={log_preview(error_message)!r}"
    )

    if stop_reason in {StopReason.ABORTED, "aborted"}:
        logger.info(
            f"bampi_chat skipped aborted auto-resume reply group_id={target.group_id} "
            f"reply_message_id={target.reply_message_id}"
        )
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    if not text and not files:
        if error_message:
            logger.warning(
                f"bampi_chat auto-resume returned no deliverable content "
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
                    text="这次自动续跑失败了，请检查 API Key、模型配置或上游服务。",
                ),
            )
            return ResponseDispatchResult(delivered=False, rollback_context=True)
        if streamed_any_text:
            logger.info(
                f"bampi_chat auto-resume fully covered by live stream "
                f"group_id={target.group_id} "
                f"reply_message_id={target.reply_message_id}"
            )
            return ResponseDispatchResult(delivered=True, rollback_context=False)

        logger.warning(
            f"bampi_chat auto-resume returned empty content "
            f"group_id={target.group_id} "
            f"reply_message_id={target.reply_message_id}"
        )
        await _send_group_message_via_bot(
            bot=bot,
            target=target,
            message=build_group_reply_message(
                config=config,
                target=target,
                text="这次自动续跑没有生成可发送的内容，你可以直接发一句话让我继续。",
            ),
        )
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    message = build_group_reply_message(
        config=config,
        target=target,
        text=text,
    )

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
                logger.exception(f"failed to prepare auto-resume outbox image: {path}")
                failed_artifacts.append(path.name)

        if message:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=message,
            )
            logger.info(
                f"bampi_chat auto-resume reply sent group_id={target.group_id} "
                f"reply_message_id={target.reply_message_id} "
                f"has_text={bool(text)} "
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
                logger.info(f"bampi_chat auto-resume uploaded outbox file group_id={target.group_id} path={path}")
                uploaded_files.append(path)
            except Exception:
                logger.exception(f"failed to upload auto-resume outbox file: {path}")
                failed_artifacts.append(path.name)
        if failed_artifacts:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=config,
                    target=target,
                    text=f"这些产物已经生成，但自动发送失败了：{', '.join(failed_artifacts)}",
                ),
            )
    finally:
        for path in uploaded_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup auto-resume outbox file: {path}")
        for staged_path in staged_upload_files:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup auto-resume staged upload file: {staged_path}")
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
            await matcher.send("这次调用模型失败了，请检查 API Key、模型配置或上游服务。")
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
        await matcher.send("这次没有生成可发送的内容，你可以换个说法再试一次。")
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    message = Message()
    if config.bampi_reply_with_quote:
        message += MessageSegment.reply(event.message_id)
    if config.bampi_at_sender:
        message += MessageSegment.at(event.user_id)
    if text:
        message += MessageSegment.text(text)

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
                f"这些产物已经生成，但自动发送失败了：{', '.join(failed_artifacts)}"
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
