"""进度汇报：会话事件 → 群里的实时进度消息（工具调用、流式文本、压缩提示）。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment

from bampy.ai.types import AssistantMessage
from bampy.app import AgentSession

from ..config import BampiChatConfig
from ..feedback import THRESHOLD_COMPACTION_NOTICE
from ..skills import describe_skill_resource_path
from .outbound import (
    GroupReplyTarget,
    append_reply_body,
    assistant_message_has_tool_call,
    extract_text_blocks,
)
from .utils import extract_api_message_id, log_preview


@dataclass(slots=True)
class ProgressMessage:
    text: str
    quote: bool = False
    tool_call_id: str | None = None
    assistant_text: bool = False


@dataclass(slots=True)
class ToolProgressNotice:
    message_id: int | None = None
    sent_at: float = 0.0
    finished: bool = False
    should_recall: bool = False
    send_failed: bool = False


@dataclass(slots=True)
class DigestEntry:
    """A buffered progress step: ``label`` names the action (emoji included),
    ``detail`` carries its argument preview. Entries sharing a label are
    aggregated into one digest line with their details joined."""

    label: str
    detail: str = ""
    tool_call_id: str | None = None


DIGEST_HEADER = "⏳ 仍在处理中："
DIGEST_MAX_LINES = 6
DIGEST_MAX_GROUP_DETAILS = 3


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


SILENT_PROGRESS_TOOLS = frozenset({"qq_react"})


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
        self._intermediate_text_sent = False
        self._last_seen_text = ""
        self._pending_text = ""
        self._tool_notices: dict[str, ToolProgressNotice] = {}
        self._recall_tasks: set[asyncio.Task[None]] = set()
        self._digest_mode = False
        self._visible_progress_count = 0
        self._digest_buffer: list[DigestEntry] = []
        self._digest_task: asyncio.Task[None] | None = None

    @property
    def intermediate_text_sent(self) -> bool:
        """Whether a completed assistant tool turn reached the group."""
        return self._intermediate_text_sent

    @property
    def visible_update_sent(self) -> bool:
        """Whether a progress message already claimed the reply quote."""
        return self._visible_update_sent

    def start(self, session: AgentSession) -> None:
        if not self._enabled:
            return
        self._worker = asyncio.create_task(self._run_sender())
        self._unsubscribe = session.subscribe(self._handle_event)

    async def prepare_final_reply(self) -> None:
        if not self._enabled:
            return
        # The final reply supersedes any progress still waiting in the digest
        # buffer, so pending digest updates are discarded rather than flushed.
        self._discard_pending_digest()
        # Terminal assistant text is deliberately left to the final delivery
        # path, where complete Markdown can be rendered without racing the
        # progress sender. Only already-enqueued intermediate updates need to
        # drain before that reply is sent.
        await self._queue.join()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._discard_pending_digest()
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
                if item.assistant_text:
                    await append_reply_body(message, item.text, self._config)
                else:
                    message += MessageSegment.text(item.text)
                response = await self._bot.call_api(
                    "send_group_msg",
                    group_id=self._target.group_id,
                    message=message,
                )
                if item.assistant_text:
                    self._intermediate_text_sent = True
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
            self._handle_message_start(event)
            return
        if event_type == "message_end" and self._config.bampi_live_text_stream_enabled:
            self._handle_message_end(event)
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

    def _handle_message_start(self, event: Any) -> None:
        if not isinstance(getattr(event, "message", None), AssistantMessage):
            return
        self._last_seen_text = ""
        self._pending_text = ""

    def _handle_tool_start(self, event: Any) -> None:
        if getattr(event, "tool_name", "") in SILENT_PROGRESS_TOOLS:
            return
        tool_call_id = getattr(event, "tool_call_id", "")
        if self._digest_mode:
            if self._config.bampi_live_text_stream_enabled:
                self._flush_pending_text()
            self._append_digest_entry(
                make_tool_digest_entry(
                    format_tool_progress_message(
                        getattr(event, "tool_name", ""),
                        getattr(event, "args", None),
                    ),
                    tool_call_id=tool_call_id or None,
                )
            )
            return
        limit = self._config.bampi_live_progress_max_tool_updates
        if limit > 0 and self._tool_updates_sent >= limit:
            return
        if self._config.bampi_live_text_stream_enabled:
            self._flush_pending_text()

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
            # Digest entries mirror the recall-on-failure behaviour of live
            # updates: failed steps buffered but not yet flushed are dropped.
            if bool(getattr(event, "is_error", False)):
                self._drop_digest_entries(tool_call_id)
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

    def _handle_message_end(self, event: Any) -> None:
        message = getattr(event, "message", None)
        if not isinstance(message, AssistantMessage):
            return
        if not assistant_message_has_tool_call(message):
            # This is the terminal assistant message. Do not send it through
            # the progress channel: the final dispatcher needs the complete
            # text to render code, formulas and tables atomically.
            self._pending_text = ""
            return

        # A message containing a tool call is an intermediate turn. Use the
        # completed snapshot rather than accumulated deltas so it is emitted
        # exactly once before tool execution starts.
        self._pending_text = extract_text_blocks(message)
        self._last_seen_text = self._pending_text
        self._flush_pending_text()

    def _flush_pending_text(self) -> None:
        if self._closed or not self._pending_text.strip():
            return

        payload = self._pending_text
        self._pending_text = ""
        # Intermediate assistant text reads as ordinary conversation, so it
        # always goes out directly and never enters the digest buffer.
        self._enqueue(
            payload,
            preserve_whitespace=True,
            assistant_text=True,
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
        assistant_text: bool = False,
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
                assistant_text=assistant_text,
            )
        )
        # Assistant text is ordinary conversation rather than a progress
        # notice, so it does not count toward the digest-mode switch.
        if not assistant_text:
            self._register_visible_progress()

    def _register_visible_progress(self) -> None:
        if self._digest_mode:
            return
        threshold = self._config.bampi_live_progress_digest_after_updates
        if threshold <= 0:
            return
        self._visible_progress_count += 1
        if self._visible_progress_count >= threshold:
            self._digest_mode = True
            logger.info(
                f"bampi_chat live progress switched to digest mode "
                f"group_id={self._target.group_id} "
                f"message_id={self._target.reply_message_id} "
                f"after_updates={self._visible_progress_count}"
            )

    def _append_digest_entry(self, entry: DigestEntry) -> None:
        if self._closed:
            return
        self._digest_buffer.append(entry)
        self._schedule_digest_flush()

    def _drop_digest_entries(self, tool_call_id: str) -> None:
        if not self._digest_buffer:
            return
        self._digest_buffer = [
            entry for entry in self._digest_buffer if entry.tool_call_id != tool_call_id
        ]

    def _schedule_digest_flush(self) -> None:
        if self._digest_task is not None and not self._digest_task.done():
            return
        task = asyncio.create_task(self._flush_digest_after_interval())
        self._digest_task = task

        def _cleanup(done: asyncio.Task[None]) -> None:
            if self._digest_task is done:
                self._digest_task = None
            if done.cancelled():
                return
            exc = done.exception()
            if exc is not None:
                logger.error(
                    f"bampi_chat digest flush task failed "
                    f"group_id={self._target.group_id} "
                    f"message_id={self._target.reply_message_id} "
                    f"error={exc!r}"
                )

        task.add_done_callback(_cleanup)

    async def _flush_digest_after_interval(self) -> None:
        await asyncio.sleep(self._config.bampi_live_progress_digest_interval_seconds)
        if self._closed or not self._digest_buffer:
            return
        entries = list(self._digest_buffer)
        self._digest_buffer.clear()
        self._enqueue(build_digest_text(entries))

    def _discard_pending_digest(self) -> None:
        self._digest_buffer.clear()
        task = self._digest_task
        self._digest_task = None
        if task is not None and not task.done():
            task.cancel()

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


def make_tool_digest_entry(message: str, *, tool_call_id: str | None = None) -> DigestEntry:
    # Tool progress messages follow the "emoji label：detail" convention, so
    # the first full-width colon splits action from argument. Messages without
    # a detail part ("正在搜索记忆") become label-only entries.
    label, sep, detail = message.partition("：")
    if not sep:
        return DigestEntry(label=message, tool_call_id=tool_call_id)
    return DigestEntry(label=label, detail=detail, tool_call_id=tool_call_id)


def build_digest_text(entries: list[DigestEntry]) -> str:
    groups: list[tuple[str, list[str], int]] = []
    for entry in entries:
        if groups and groups[-1][0] == entry.label:
            _, details, count = groups[-1]
            if entry.detail and entry.detail not in details:
                details.append(entry.detail)
            groups[-1] = (entry.label, details, count + 1)
        else:
            details = [entry.detail] if entry.detail else []
            groups.append((entry.label, details, 1))
    lines = [_render_digest_group(label, details, count) for label, details, count in groups]
    hidden = len(lines) - DIGEST_MAX_LINES
    if hidden > 0:
        lines = lines[:DIGEST_MAX_LINES]
        lines.append(f"……以及另外 {hidden} 项")
    return "\n".join([DIGEST_HEADER, *lines])


def _render_digest_group(label: str, details: list[str], count: int) -> str:
    if not details:
        return f"{label} ×{count}" if count > 1 else label
    if len(details) == 1:
        line = f"{label}：{details[0]}"
        return f"{line} ×{count}" if count > 1 else line
    shown = details[:DIGEST_MAX_GROUP_DETAILS]
    line = f"{label}：{'、'.join(shown)}"
    if len(details) > DIGEST_MAX_GROUP_DETAILS:
        line += f" 等 {len(details)} 项"
    return line


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
