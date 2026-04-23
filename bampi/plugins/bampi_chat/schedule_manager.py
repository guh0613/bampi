from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.job import Job
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from nonebot import get_bots, logger

from bampy.ai import TextContent, UserMessage

from .config import BampiChatConfig

if TYPE_CHECKING:
    from .session_manager import GroupSessionManager


ScheduleAction = Literal["create", "list", "status", "pause", "resume", "cancel", "run_now"]
TaskTriggerType = Literal["date", "cron"]
TaskState = Literal["scheduled", "paused", "completed", "cancelled"]
TaskRunStatus = Literal["success", "failed"]

_REGISTRY_VERSION = 1
_IMMEDIATE_RESUME_DELAY = timedelta(seconds=1)
_SCHEDULE_ACTOR_PREFIX = "__schedule__:"


@dataclass(slots=True)
class PendingScheduledRun:
    task_id: str
    trigger_source: str
    scheduled_for: str
    enqueued_at: float = field(default_factory=time.monotonic)


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now_utc().replace(microsecond=0).isoformat()


def _trim_text(value: str, *, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[: max(0, limit - 3)].rstrip() + "...", True


def _parse_timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value.strip())
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(f"invalid timezone: {value}") from exc


def _parse_run_at(value: str, *, timezone: str) -> datetime:
    text = value.strip()
    if not text:
        raise RuntimeError("run_at must not be empty")
    candidate = text
    if "T" not in candidate and " " in candidate:
        candidate = candidate.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise RuntimeError(
            "run_at must be an ISO datetime like `2026-04-23T09:00` or `2026-04-23 09:00`"
        ) from exc

    tz = _parse_timezone(timezone)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    else:
        parsed = parsed.astimezone(tz)
    return parsed.replace(second=0, microsecond=0)


def _format_local_timestamp(value: str | None, *, timezone: str) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.astimezone(_parse_timezone(timezone)).strftime("%Y-%m-%d %H:%M %Z")


@dataclass(slots=True)
class ScheduledTaskRecord:
    task_id: str
    group_id: str
    name: str
    prompt: str
    trigger_type: TaskTriggerType
    trigger: dict[str, Any]
    timezone: str
    state: TaskState
    created_at: str
    updated_at: str
    next_run_at: str | None = None
    last_run_started_at: str | None = None
    last_run_finished_at: str | None = None
    last_run_status: TaskRunStatus | None = None
    last_error: str | None = None
    last_result_preview: str | None = None
    run_count: int = 0
    is_running: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScheduledTaskRecord":
        payload = dict(value)
        payload.setdefault("next_run_at", None)
        payload.setdefault("last_run_started_at", None)
        payload.setdefault("last_run_finished_at", None)
        payload.setdefault("last_run_status", None)
        payload.setdefault("last_error", None)
        payload.setdefault("last_result_preview", None)
        payload.setdefault("run_count", 0)
        payload.setdefault("is_running", False)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_active(self) -> bool:
        return self.state in {"scheduled", "paused"}


class ScheduleManager:
    def __init__(
        self,
        *,
        config: BampiChatConfig,
        group_session_manager: "GroupSessionManager",
    ) -> None:
        self._config = config
        self._group_session_manager = group_session_manager
        self._runtime_root = Path(config.bampi_schedule_dir).resolve()
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._registry_path = (self._runtime_root / "registry.json").resolve()
        self._lock = asyncio.Lock()
        self._scheduler = AsyncIOScheduler(
            timezone=_parse_timezone(config.bampi_schedule_timezone),
        )
        self._started = False
        self._tasks: dict[str, ScheduledTaskRecord] = {}
        self._next_task_sequence = 1
        self._background_runs: set[asyncio.Task[None]] = set()
        self._pending_runs_by_group: dict[str, deque[PendingScheduledRun]] = {}
        self._group_worker_tasks: dict[str, asyncio.Task[None]] = {}
        self._load_registry()

    async def start(self) -> None:
        due_immediately: list[str] = []
        async with self._lock:
            if self._started:
                return
            self._scheduler.start()
            self._started = True
            due_immediately = await self._restore_jobs_locked()
        for task_id in due_immediately:
            await self._enqueue_task_run(task_id, trigger_source="startup_due")
        logger.info(
            f"bampi_chat schedule manager started runtime_dir={self._runtime_root} "
            f"task_count={len(self._tasks)}"
        )

    async def close(self) -> None:
        async with self._lock:
            if not self._started:
                return
            self._started = False
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                logger.exception("bampi_chat failed to shutdown scheduler cleanly")
            await self._save_registry_locked()

        pending = list(self._background_runs)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._group_worker_tasks.clear()
        self._pending_runs_by_group.clear()
        logger.info("bampi_chat schedule manager stopped")

    async def create_task(
        self,
        *,
        group_id: str,
        name: str | None,
        prompt: str,
        trigger_type: TaskTriggerType,
        timezone: str | None,
        run_at: str | None,
        cron: str | None,
        replace_existing: bool,
    ) -> ScheduledTaskRecord:
        effective_timezone = (timezone or self._config.bampi_schedule_timezone).strip()
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise RuntimeError("prompt must not be empty")

        now = _now_utc()
        async with self._lock:
            normalized_name = (name or "").strip()
            duplicates = [
                task
                for task in self._tasks.values()
                if task.group_id == group_id and task.name == normalized_name and task.is_active
            ] if normalized_name else []
            active_tasks = [
                task
                for task in self._tasks.values()
                if task.group_id == group_id and task.is_active and task not in duplicates
            ]
            if (
                self._config.bampi_schedule_max_active_tasks_per_group > 0
                and len(active_tasks) >= self._config.bampi_schedule_max_active_tasks_per_group
            ):
                raise RuntimeError(
                    f"this group already has {len(active_tasks)} active scheduled tasks, "
                    f"which reaches the configured limit of "
                    f"{self._config.bampi_schedule_max_active_tasks_per_group}"
                )

            if normalized_name:
                if duplicates and not replace_existing:
                    raise RuntimeError(
                        f"a scheduled task named `{normalized_name}` already exists in this group"
                    )
                for duplicate in duplicates:
                    await self._cancel_task_locked(duplicate)

            trigger_payload = self._validate_trigger_payload(
                trigger_type=trigger_type,
                timezone=effective_timezone,
                run_at=run_at,
                cron=cron,
                now=now,
            )

            task_id = f"task-{self._next_task_sequence}"
            self._next_task_sequence += 1
            task_name = normalized_name or task_id
            record = ScheduledTaskRecord(
                task_id=task_id,
                group_id=group_id,
                name=task_name,
                prompt=normalized_prompt,
                trigger_type=trigger_type,
                trigger=trigger_payload,
                timezone=effective_timezone,
                state="scheduled",
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )
            self._tasks[task_id] = record
            if self._started:
                self._add_or_replace_job_locked(record)
            else:
                record.next_run_at = self._preview_next_run_locked(record)
            await self._save_registry_locked()
            return ScheduledTaskRecord.from_dict(record.to_dict())

    async def list_tasks(self, *, group_id: str) -> list[ScheduledTaskRecord]:
        async with self._lock:
            records = sorted(
                (task for task in self._tasks.values() if task.group_id == group_id),
                key=lambda item: (item.created_at, item.task_id),
                reverse=True,
            )
            if self._started:
                for record in records:
                    self._sync_next_run_locked(record)
                await self._save_registry_locked()
            return [ScheduledTaskRecord.from_dict(record.to_dict()) for record in records]

    async def get_task(self, *, group_id: str, task_ref: str) -> ScheduledTaskRecord:
        async with self._lock:
            record = self._resolve_task_ref_locked(group_id, task_ref)
            if self._started:
                self._sync_next_run_locked(record)
                await self._save_registry_locked()
            return ScheduledTaskRecord.from_dict(record.to_dict())

    async def pause_task(self, *, group_id: str, task_ref: str) -> ScheduledTaskRecord:
        async with self._lock:
            record = self._resolve_task_ref_locked(group_id, task_ref)
            if record.state == "cancelled":
                raise RuntimeError("cancelled tasks cannot be paused")
            if record.state == "completed":
                raise RuntimeError("completed one-time tasks cannot be paused")
            record.state = "paused"
            record.updated_at = _now_iso()
            record.next_run_at = None
            self._remove_pending_runs_locked(group_id, record.task_id)
            if self._started and self._scheduler.get_job(record.task_id) is not None:
                self._scheduler.pause_job(record.task_id)
            await self._save_registry_locked()
            return ScheduledTaskRecord.from_dict(record.to_dict())

    async def resume_task(self, *, group_id: str, task_ref: str) -> ScheduledTaskRecord:
        due_immediately = False
        async with self._lock:
            record = self._resolve_task_ref_locked(group_id, task_ref)
            if record.state == "cancelled":
                raise RuntimeError("cancelled tasks cannot be resumed")
            if record.state == "completed":
                raise RuntimeError("completed one-time tasks cannot be resumed")
            record.state = "scheduled"
            record.updated_at = _now_iso()
            due_immediately = self._is_immediate_date_due(record)
            if self._started and not due_immediately:
                self._add_or_replace_job_locked(record)
            else:
                record.next_run_at = self._preview_next_run_locked(record)
            await self._save_registry_locked()
            snapshot = ScheduledTaskRecord.from_dict(record.to_dict())
        if due_immediately:
            await self._enqueue_task_run(snapshot.task_id, trigger_source="resume_due")
        return snapshot

    async def cancel_task(self, *, group_id: str, task_ref: str) -> ScheduledTaskRecord:
        async with self._lock:
            record = self._resolve_task_ref_locked(group_id, task_ref)
            await self._cancel_task_locked(record)
            await self._save_registry_locked()
            return ScheduledTaskRecord.from_dict(record.to_dict())

    async def run_task_now(self, *, group_id: str, task_ref: str) -> ScheduledTaskRecord:
        async with self._lock:
            record = self._resolve_task_ref_locked(group_id, task_ref)
            if record.state == "cancelled":
                raise RuntimeError("cancelled tasks cannot be run manually")
            if record.is_running or self._is_task_already_queued_locked(record.group_id, record.task_id):
                raise RuntimeError(f"task `{record.task_id}` is already running")
            snapshot = ScheduledTaskRecord.from_dict(record.to_dict())
        await self._enqueue_task_run(snapshot.task_id, trigger_source="manual")
        return snapshot

    def render_task_summary(self, record: ScheduledTaskRecord, *, max_chars: int = 4_000) -> str:
        prompt_preview, trimmed = _trim_text(record.prompt, limit=max_chars // 2)
        lines = [
            f"Task: {record.task_id}",
            f"Name: {record.name}",
            f"State: {record.state}",
            f"Trigger: {self._render_trigger(record)}",
            f"Timezone: {record.timezone}",
            f"Next run: {_format_local_timestamp(record.next_run_at, timezone=record.timezone)}",
            f"Run count: {record.run_count}",
            f"Currently running: {'yes' if record.is_running else 'no'}",
            f"Created at: {_format_local_timestamp(record.created_at, timezone=record.timezone)}",
        ]
        if record.last_run_started_at:
            lines.append(
                f"Last run started: {_format_local_timestamp(record.last_run_started_at, timezone=record.timezone)}"
            )
        if record.last_run_finished_at:
            lines.append(
                f"Last run finished: {_format_local_timestamp(record.last_run_finished_at, timezone=record.timezone)}"
            )
        if record.last_run_status:
            lines.append(f"Last run status: {record.last_run_status}")
        if record.last_error:
            lines.append(f"Last error: {record.last_error}")
        if record.last_result_preview:
            lines.append(f"Last result preview: {record.last_result_preview}")
        lines.append("")
        lines.append("Prompt:")
        lines.append(prompt_preview)
        if trimmed:
            lines.append("(prompt truncated)")
        return "\n".join(lines)

    @staticmethod
    def render_task_list(records: list[ScheduledTaskRecord]) -> str:
        if not records:
            return "No scheduled tasks in this group."
        lines = ["Scheduled tasks in this group:"]
        for record in records:
            next_run = _format_local_timestamp(record.next_run_at, timezone=record.timezone)
            lines.append(
                f"- {record.task_id} ({record.name}): {record.state}, trigger={ScheduleManager._render_trigger(record)}, next={next_run}"
            )
        return "\n".join(lines)

    @staticmethod
    def task_details(record: ScheduledTaskRecord) -> dict[str, Any]:
        return {
            "task_id": record.task_id,
            "name": record.name,
            "state": record.state,
            "trigger_type": record.trigger_type,
            "trigger": dict(record.trigger),
            "timezone": record.timezone,
            "next_run_at": record.next_run_at,
            "run_count": record.run_count,
            "is_running": record.is_running,
        }

    async def _enqueue_task_run(self, task_id: str, *, trigger_source: str) -> None:
        worker_to_start: tuple[str, asyncio.Task[None]] | None = None
        notify_queued = False
        queued_name = task_id
        queued_group_id = ""
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            if trigger_source != "manual" and record.state != "scheduled":
                return
            if record.is_running or self._is_task_already_queued_locked(record.group_id, task_id):
                logger.info(
                    f"bampi_chat skipped duplicate scheduled queue entry "
                    f"task_id={task_id} group_id={record.group_id}"
                )
                return
            queue = self._pending_runs_by_group.setdefault(record.group_id, deque())
            notify_queued = bool(queue)
            queued_name = record.name
            queued_group_id = record.group_id
            queue.append(
                PendingScheduledRun(
                    task_id=task_id,
                    trigger_source=trigger_source,
                    scheduled_for=_now_iso(),
                )
            )
            worker = self._group_worker_tasks.get(record.group_id)
            if worker is None or worker.done():
                worker = asyncio.create_task(
                    self._drain_group_queue(record.group_id),
                    name=f"bampi-chat-schedule-queue-{record.group_id}",
                )
                self._group_worker_tasks[record.group_id] = worker
                worker_to_start = (record.group_id, worker)

        if not notify_queued:
            try:
                status = await self._group_session_manager.inspect_interaction(queued_group_id)
                notify_queued = status.is_active
            except Exception:
                logger.exception(
                    f"bampi_chat failed to inspect interaction for queued scheduled task "
                    f"group_id={queued_group_id} task_id={task_id}"
                )

        if notify_queued:
            await self._notify_task_queued(
                task_id,
                task_name=queued_name,
            )

        if worker_to_start is not None:
            group_id, worker = worker_to_start
            self._background_runs.add(worker)

            def _cleanup(done: asyncio.Task[None], *, current_group_id: str) -> None:
                self._background_runs.discard(done)
                existing = self._group_worker_tasks.get(current_group_id)
                if existing is done:
                    self._group_worker_tasks.pop(current_group_id, None)
                if done.cancelled():
                    return
                exc = done.exception()
                if exc is not None:
                    logger.error(
                        f"bampi_chat scheduled queue worker failed "
                        f"group_id={current_group_id} error={exc!r}"
                    )

            worker.add_done_callback(lambda done, current_group_id=group_id: _cleanup(done, current_group_id=current_group_id))

    async def _drain_group_queue(self, group_id: str) -> None:
        while True:
            async with self._lock:
                queue = self._pending_runs_by_group.get(group_id)
                if not queue:
                    self._pending_runs_by_group.pop(group_id, None)
                    return
                pending = queue[0]
                record = self._tasks.get(pending.task_id)
                if record is None or record.state == "cancelled":
                    queue.popleft()
                    continue

            status = await self._group_session_manager.inspect_interaction(group_id)
            if status.is_active:
                await asyncio.sleep(max(1.0, self._config.bampi_schedule_tick_seconds))
                continue

            try:
                reservation = await self._group_session_manager.reserve_interaction(
                    group_id,
                    self._schedule_actor_id(pending.task_id),
                )
            except Exception:
                logger.exception(
                    f"bampi_chat failed to reserve interaction for scheduled task "
                    f"group_id={group_id} task_id={pending.task_id}"
                )
                await asyncio.sleep(max(1.0, self._config.bampi_schedule_tick_seconds))
                continue

            if reservation.action != "start":
                await asyncio.sleep(max(1.0, self._config.bampi_schedule_tick_seconds))
                continue

            async with self._lock:
                queue = self._pending_runs_by_group.get(group_id)
                current = queue.popleft() if queue else pending
                if queue is not None and not queue:
                    self._pending_runs_by_group.pop(group_id, None)

            try:
                await self._run_task(
                    reservation.managed,
                    current,
                )
            finally:
                await self._group_session_manager.complete_interaction(group_id)

    async def _run_task(self, managed: Any, pending: PendingScheduledRun) -> None:
        task_id = pending.task_id
        task_name = task_id
        try:
            from .handler import (
                GroupReplyTarget,
                _send_group_message_via_bot,
                build_group_reply_message,
                extract_text_blocks,
                find_last_assistant_message,
                log_preview,
                send_background_agent_response,
                snapshot_outbox,
            )

            async with self._lock:
                record = self._tasks.get(task_id)
                if record is None:
                    return
                if record.state == "cancelled":
                    return
                if pending.trigger_source != "manual" and record.state != "scheduled":
                    return
                if record.is_running:
                    return
                record.is_running = True
                record.updated_at = _now_iso()
                record.last_run_started_at = record.updated_at
                record.last_run_finished_at = None
                record.last_run_status = None
                record.last_error = None
                record.next_run_at = None
                task_name = record.name
                snapshot = ScheduledTaskRecord.from_dict(record.to_dict())
                await self._save_registry_locked()

            bot = self._resolve_bot()
            if bot is None:
                raise RuntimeError("no connected bot is available to deliver scheduled task output")

            target = GroupReplyTarget(group_id=int(snapshot.group_id))
            scheduled_for_text = _format_local_timestamp(pending.scheduled_for, timezone=snapshot.timezone)
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=self._config,
                    target=target,
                    text=f"⏰ 正在执行定时任务 `{snapshot.name}` ...",
                ),
            )

            workspace_dir = self._group_session_manager.workspace_dir_for_group(snapshot.group_id)
            outbox_before = snapshot_outbox(workspace_dir)
            async with managed.lock:
                managed.last_used_at = time.monotonic()
                user_message = self._build_execution_message(
                    snapshot,
                    scheduled_for=pending.scheduled_for,
                )
                try:
                    await managed.session.prompt(user_message, source="scheduled_task")
                except Exception:
                    raise

                assistant_message = find_last_assistant_message(managed.session.messages)
                result_text = extract_text_blocks(assistant_message)
                result = await send_background_agent_response(
                    bot=bot,
                    target=target,
                    config=self._config,
                    workspace_dir=workspace_dir,
                    assistant_message=assistant_message,
                    outbox_before=outbox_before,
                )

            last_status: TaskRunStatus = "success"
            error_message: str | None = None
            if result.rollback_context:
                last_status = "failed"
                error_message = "scheduled task execution produced no deliverable result"

            preview = (
                log_preview(result_text, limit=240)
                if result_text
                else f"started from scheduled trigger at {scheduled_for_text}"
            )
            await self._finish_task_run(
                task_id,
                last_run_status=last_status,
                last_error=error_message,
                last_result_preview=preview,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            from .handler import normalize_text

            error_message = normalize_text(str(exc)) or exc.__class__.__name__
            logger.exception(
                f"bampi_chat scheduled task failed task_id={task_id} "
                f"name={task_name} trigger_source={pending.trigger_source}"
            )
            await self._notify_task_failure(task_id, task_name=task_name, error_message=error_message)
            await self._finish_task_run(
                task_id,
                last_run_status="failed",
                last_error=error_message,
                last_result_preview=None,
            )

    async def _finish_task_run(
        self,
        task_id: str,
        *,
        last_run_status: TaskRunStatus,
        last_error: str | None,
        last_result_preview: str | None,
    ) -> None:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            record.run_count += 1
            record.updated_at = _now_iso()
            record.last_run_finished_at = record.updated_at
            record.last_run_status = last_run_status
            record.last_error = last_error
            record.last_result_preview = last_result_preview
            record.is_running = False
            if record.state == "cancelled":
                record.next_run_at = None
            elif record.trigger_type == "date":
                record.state = "completed"
                self._remove_job_locked(record.task_id)
                record.next_run_at = None
            else:
                self._sync_next_run_locked(record)
            await self._save_registry_locked()

    async def _notify_task_failure(
        self,
        task_id: str,
        *,
        task_name: str,
        error_message: str,
    ) -> None:
        from .handler import (
            GroupReplyTarget,
            _send_group_message_via_bot,
            build_group_reply_message,
        )

        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            group_id = record.group_id

        bot = self._resolve_bot()
        if bot is None:
            return

        target = GroupReplyTarget(group_id=int(group_id))
        try:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=self._config,
                    target=target,
                    text=f"定时任务 `{task_name}` 执行失败：{error_message}",
                ),
            )
        except Exception:
            logger.exception(
                f"bampi_chat failed to deliver scheduled task failure notice "
                f"task_id={task_id}"
            )

    async def _notify_task_queued(
        self,
        task_id: str,
        *,
        task_name: str,
    ) -> None:
        from .handler import (
            GroupReplyTarget,
            _send_group_message_via_bot,
            build_group_reply_message,
        )

        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            group_id = record.group_id

        bot = self._resolve_bot()
        if bot is None:
            return

        target = GroupReplyTarget(group_id=int(group_id))
        try:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=self._config,
                    target=target,
                    text=(
                        f"⏰ 定时任务 `{task_name}` 已触发，"
                        "但当前群里有进行中的会话，我会排队等空闲后再开始。"
                    ),
                ),
            )
        except Exception:
            logger.exception(
                f"bampi_chat failed to deliver scheduled task queue notice "
                f"task_id={task_id}"
            )

    def _build_execution_message(
        self,
        record: ScheduledTaskRecord,
        *,
        scheduled_for: str,
    ) -> UserMessage:
        display_name = record.name if record.name != record.task_id else "定时任务"
        lines = [
            f"sender_name: scheduler",
            f"message_text: 用户设定的定时任务「{display_name}」到期，请按以下内容执行。",
            "scheduled_task_prompt:",
            record.prompt,
        ]
        return UserMessage(content=[TextContent(text="\n".join(lines))])

    def _is_task_already_queued_locked(self, group_id: str, task_id: str) -> bool:
        queue = self._pending_runs_by_group.get(group_id)
        if not queue:
            return False
        return any(item.task_id == task_id for item in queue)

    @staticmethod
    def _schedule_actor_id(task_id: str) -> str:
        return f"{_SCHEDULE_ACTOR_PREFIX}{task_id}"

    def _validate_trigger_payload(
        self,
        *,
        trigger_type: TaskTriggerType,
        timezone: str,
        run_at: str | None,
        cron: str | None,
        now: datetime,
    ) -> dict[str, Any]:
        _parse_timezone(timezone)
        if trigger_type == "date":
            if not run_at:
                raise RuntimeError("date tasks require run_at")
            parsed = _parse_run_at(run_at, timezone=timezone)
            if parsed.astimezone(UTC) <= now:
                raise RuntimeError("run_at must be later than the current time")
            return {"run_at": parsed.isoformat()}
        if trigger_type == "cron":
            expression = (cron or "").strip()
            if not expression:
                raise RuntimeError("cron tasks require cron")
            try:
                CronTrigger.from_crontab(expression, timezone=_parse_timezone(timezone))
            except ValueError as exc:
                raise RuntimeError(f"invalid cron expression: {expression}") from exc
            return {"cron": expression}
        raise RuntimeError(f"unsupported trigger type: {trigger_type}")

    async def _restore_jobs_locked(self) -> list[str]:
        due_immediately: list[str] = []
        for record in self._tasks.values():
            record.is_running = False
            if record.state in {"cancelled", "completed"}:
                record.next_run_at = None
                continue
            if record.state == "paused":
                self._add_or_replace_job_locked(record)
                if self._scheduler.get_job(record.task_id) is not None:
                    self._scheduler.pause_job(record.task_id)
                record.next_run_at = None
                continue
            if self._is_immediate_date_due(record):
                due_immediately.append(record.task_id)
                record.next_run_at = _now_utc().astimezone(_parse_timezone(record.timezone)).isoformat()
                continue
            self._add_or_replace_job_locked(record)
        await self._save_registry_locked()
        return due_immediately

    def _add_or_replace_job_locked(self, record: ScheduledTaskRecord) -> None:
        trigger = self._build_trigger(record)
        self._scheduler.add_job(
            self._scheduled_job_callback,
            id=record.task_id,
            kwargs={"task_id": record.task_id},
            trigger=trigger,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._sync_next_run_locked(record)

    def _build_trigger(self, record: ScheduledTaskRecord) -> DateTrigger | CronTrigger:
        timezone = _parse_timezone(record.timezone)
        if record.trigger_type == "date":
            run_at = _parse_run_at(record.trigger.get("run_at", ""), timezone=record.timezone)
            if run_at.astimezone(UTC) <= _now_utc():
                run_at = _now_utc().astimezone(timezone) + _IMMEDIATE_RESUME_DELAY
            return DateTrigger(run_date=run_at, timezone=timezone)
        return CronTrigger.from_crontab(record.trigger.get("cron", ""), timezone=timezone)

    async def _scheduled_job_callback(self, task_id: str) -> None:
        await self._enqueue_task_run(task_id, trigger_source="schedule")

    def _sync_next_run_locked(self, record: ScheduledTaskRecord) -> None:
        if record.state != "scheduled":
            record.next_run_at = None
            return
        job = self._scheduler.get_job(record.task_id)
        record.next_run_at = self._job_next_run(job, timezone=record.timezone)

    def _preview_next_run_locked(self, record: ScheduledTaskRecord) -> str | None:
        trigger = self._build_trigger(record)
        next_run = trigger.get_next_fire_time(None, _now_utc().astimezone(_parse_timezone(record.timezone)))
        if next_run is None:
            return None
        return next_run.astimezone(_parse_timezone(record.timezone)).replace(second=0, microsecond=0).isoformat()

    @staticmethod
    def _job_next_run(job: Job | None, *, timezone: str) -> str | None:
        if job is None or job.next_run_time is None:
            return None
        return (
            job.next_run_time.astimezone(_parse_timezone(timezone))
            .replace(second=0, microsecond=0)
            .isoformat()
        )

    def _is_immediate_date_due(self, record: ScheduledTaskRecord) -> bool:
        if record.trigger_type != "date" or record.state != "scheduled":
            return False
        run_at = _parse_run_at(record.trigger.get("run_at", ""), timezone=record.timezone)
        return run_at.astimezone(UTC) <= _now_utc()

    async def _cancel_task_locked(self, record: ScheduledTaskRecord) -> None:
        record.state = "cancelled"
        record.updated_at = _now_iso()
        record.next_run_at = None
        self._remove_pending_runs_locked(record.group_id, record.task_id)
        self._remove_job_locked(record.task_id)

    def _remove_pending_runs_locked(self, group_id: str, task_id: str) -> None:
        queue = self._pending_runs_by_group.get(group_id)
        if not queue:
            return
        filtered = deque(item for item in queue if item.task_id != task_id)
        if filtered:
            self._pending_runs_by_group[group_id] = filtered
            return
        self._pending_runs_by_group.pop(group_id, None)

    def _remove_job_locked(self, task_id: str) -> None:
        try:
            self._scheduler.remove_job(task_id)
        except Exception:
            pass

    def _resolve_task_ref_locked(self, group_id: str, task_ref: str) -> ScheduledTaskRecord:
        ref = (task_ref or "").strip()
        if not ref:
            raise RuntimeError("task reference must not be empty")

        record = self._tasks.get(ref)
        if record is not None and record.group_id == group_id:
            return record

        matches = [
            candidate
            for candidate in self._tasks.values()
            if candidate.group_id == group_id and candidate.name == ref
        ]
        if not matches:
            raise RuntimeError(f"scheduled task `{ref}` was not found in this group")
        matches.sort(key=lambda candidate: (candidate.created_at, candidate.task_id), reverse=True)
        return matches[0]

    @staticmethod
    def _render_trigger(record: ScheduledTaskRecord) -> str:
        if record.trigger_type == "date":
            return f"date:{_format_local_timestamp(record.trigger.get('run_at'), timezone=record.timezone)}"
        return f"cron:{record.trigger.get('cron', '')}"

    def _resolve_bot(self) -> Any | None:
        bots = get_bots()
        if not bots:
            return None
        return next(iter(bots.values()))

    def _load_registry(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception(
                f"bampi_chat failed to load schedule registry path={self._registry_path}"
            )
            return

        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            return
        self._next_task_sequence = max(1, int(payload.get("next_task_sequence", 1) or 1))
        for item in tasks:
            if not isinstance(item, dict):
                continue
            record = ScheduledTaskRecord.from_dict(item)
            self._tasks[record.task_id] = record

    async def _save_registry_locked(self) -> None:
        payload = {
            "version": _REGISTRY_VERSION,
            "next_task_sequence": self._next_task_sequence,
            "tasks": [record.to_dict() for record in self._tasks.values()],
        }
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
