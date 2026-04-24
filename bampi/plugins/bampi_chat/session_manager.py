from __future__ import annotations

import asyncio
import inspect
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from nonebot import get_driver, logger

from bampy.ai import Model, ModelCost, SimpleStreamOptions, get_model
from bampy.app import AgentSession, SessionManager

from .config import BampiChatConfig
from .prompt import build_system_prompt
from .service_manager import ServiceManager
from .skills import build_prompt_skills, load_chat_skills
from .tools import create_agent_tools
from .tools.safe_bash import BackgroundSessionExitEvent, SafeBashTool
from .tools.workspace import (
    reset_workspace_files,
    resolve_group_container_workspace,
    resolve_group_workspace_dir,
)

BackgroundWaitCallback = Callable[[BackgroundSessionExitEvent], Awaitable[None] | None]
BackgroundWaitReminderCallback = Callable[
    ["BackgroundWaitReminderEvent"],
    Awaitable[None] | None,
]

_API_KEY_ENV_BY_API: dict[str, str] = {
    "anthropic-messages": "ANTHROPIC_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
    "ollama-responses": "OLLAMA_API_KEY",
    "openai-completions": "OPENAI_API_KEY",
    "openai-responses": "OPENAI_API_KEY",
}


@dataclass(slots=True)
class ManagedGroupSession:
    group_id: str
    session: AgentSession
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used_at: float = field(default_factory=time.monotonic)
    idle_reset_task: asyncio.Task[None] | None = None
    active_user_id: str | None = None
    pending_background_waits: dict[str, "PendingBackgroundWait"] = field(default_factory=dict)
    background_listener_unsubscribes: list[Callable[[], None]] = field(default_factory=list)


@dataclass(slots=True)
class PendingBackgroundWait:
    session_id: str
    owner_user_id: str | None
    callback: BackgroundWaitCallback
    command: str | None = None
    registered_at: float = field(default_factory=time.monotonic)
    reminder_callback: BackgroundWaitReminderCallback | None = None
    reminder_task: asyncio.Task[None] | None = None
    cancelled: bool = False


@dataclass(slots=True)
class BackgroundWaitReminderEvent:
    group_id: str
    session_id: str
    owner_user_id: str | None
    command: str | None
    waited_seconds: float


@dataclass(slots=True)
class StopInteractionResult:
    managed: ManagedGroupSession | None = None
    aborted_streaming: bool = False
    stopped_background_waits: bool = False
    stopped_background_session_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GroupInteractionStatus:
    managed: ManagedGroupSession | None = None
    active_user_id: str | None = None
    is_streaming: bool = False
    pending_background_wait_count: int = 0

    @property
    def is_waiting_background(self) -> bool:
        return self.pending_background_wait_count > 0

    @property
    def is_active(self) -> bool:
        return self.active_user_id is not None or self.is_waiting_background


@dataclass(slots=True)
class InteractionReservation:
    managed: ManagedGroupSession
    action: Literal["start", "steer", "busy"]
    active_user_id: str | None = None


class GroupSessionManager:
    def __init__(self, config: BampiChatConfig) -> None:
        self._config = config
        workspace_root = Path(config.bampi_workspace_dir).resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        self._workspace_root_dir = str(workspace_root)
        self._session_dir = Path(config.bampi_session_dir).resolve()
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ManagedGroupSession] = {}
        self._guard = asyncio.Lock()
        self._service_manager: ServiceManager | None = None
        self._schedule_manager = None
        if config.bampi_service_enabled and config.bampi_bash_mode == "docker":
            self._service_manager = ServiceManager.from_config(config)
        logger.info(
            f"bampi_chat session manager initialized "
            f"workspace_root_dir={self._workspace_root_dir} "
            f"session_dir={self._session_dir} "
            f"bash_mode={config.bampi_bash_mode} "
            f"bash_container={config.bampi_bash_container_name} "
            f"bash_workdir={config.bampi_bash_container_workdir} "
            f"service_enabled={config.bampi_service_enabled} "
            f"service_port_range={config.bampi_service_port_range} "
            f"service_public_host={config.bampi_service_public_host or '<unset>'} "
            f"idle_ttl={config.bampi_session_idle_ttl_seconds}s"
        )

    @property
    def workspace_dir(self) -> str:
        return self._workspace_root_dir

    def workspace_dir_for_group(self, group_id: str) -> str:
        return str(resolve_group_workspace_dir(self._workspace_root_dir, group_id))

    def container_workspace_dir_for_group(self, group_id: str) -> str:
        return resolve_group_container_workspace(
            self._config.bampi_bash_container_workdir,
            group_id,
            workspace_root_dir=self._workspace_root_dir,
        )

    def attach_schedule_manager(self, manager: object) -> None:
        self._schedule_manager = manager

    async def get_or_create(self, group_id: str) -> ManagedGroupSession:
        await self.close_idle()
        async with self._guard:
            return await self._get_or_create_locked(group_id)

    async def inspect_interaction(self, group_id: str) -> GroupInteractionStatus:
        await self.close_idle()
        async with self._guard:
            managed = self._sessions.get(group_id)
            if managed is None:
                return GroupInteractionStatus()
            owner_user_id = managed.active_user_id or self._background_wait_owner(managed)
            return GroupInteractionStatus(
                managed=managed,
                active_user_id=owner_user_id,
                is_streaming=managed.session.is_processing,
                pending_background_wait_count=len(managed.pending_background_waits),
            )

    async def reserve_interaction(self, group_id: str, user_id: str) -> InteractionReservation:
        await self.close_idle()
        async with self._guard:
            managed = await self._get_or_create_locked(group_id)
            self._touch_session(managed)
            if managed.pending_background_waits:
                owner_user_id = managed.active_user_id or self._background_wait_owner(managed)
                logger.info(
                    f"bampi_chat rejected interaction while waiting for background sessions "
                    f"group_id={group_id} user_id={user_id} "
                    f"pending={list(managed.pending_background_waits)}"
                )
                return InteractionReservation(
                    managed=managed,
                    action="busy",
                    active_user_id=owner_user_id,
                )
            if managed.active_user_id is None and not managed.lock.locked():
                managed.active_user_id = user_id
                logger.info(
                    f"bampi_chat reserved interaction group_id={group_id} "
                    f"user_id={user_id} action=start"
                )
                return InteractionReservation(
                    managed=managed,
                    action="start",
                    active_user_id=user_id,
                )

            action: Literal["steer", "busy"] = "busy"
            if managed.active_user_id == user_id and managed.session.is_processing:
                action = "steer"

            logger.info(
                f"bampi_chat inspected interaction group_id={group_id} "
                f"user_id={user_id} action={action} "
                f"active_user_id={managed.active_user_id} "
                f"is_streaming={managed.session.is_processing}"
            )
            return InteractionReservation(
                managed=managed,
                action=action,
                active_user_id=managed.active_user_id,
            )

    async def complete_interaction(self, group_id: str) -> None:
        async with self._guard:
            managed = self._sessions.get(group_id)
            if managed is None:
                return
            managed.active_user_id = None
            managed.last_used_at = time.monotonic()
            if not managed.pending_background_waits:
                self._schedule_idle_reset_locked(managed)

    async def register_background_wait(
        self,
        group_id: str,
        session_id: str,
        *,
        owner_user_id: str | None,
        callback: BackgroundWaitCallback,
        command: str | None = None,
        reminder_after_seconds: float | None = None,
        reminder_callback: BackgroundWaitReminderCallback | None = None,
    ) -> bool:
        async with self._guard:
            managed = self._sessions.get(group_id)
            if managed is None:
                return False
            existing = managed.pending_background_waits.get(session_id)
            if existing is not None:
                self._cancel_background_wait(existing)
            pending = PendingBackgroundWait(
                session_id=session_id,
                owner_user_id=owner_user_id,
                callback=callback,
                command=command,
                reminder_callback=reminder_callback,
            )
            managed.pending_background_waits[session_id] = pending
            if reminder_callback is not None and (reminder_after_seconds or 0) > 0:
                pending.reminder_task = asyncio.create_task(
                    self._run_background_wait_reminder(
                        group_id=group_id,
                        pending=pending,
                        after_seconds=float(reminder_after_seconds or 0),
                    ),
                    name=f"bampi-chat-background-reminder-{group_id}-{session_id}",
                )
            self._cancel_idle_reset_task(managed)
            logger.info(
                f"bampi_chat registered background wait group_id={group_id} "
                f"session_id={session_id} owner_user_id={owner_user_id} "
                f"reminder_after_seconds={reminder_after_seconds}"
            )
            return True

    async def stop_interaction(self, group_id: str, *, reason: str) -> StopInteractionResult:
        async with self._guard:
            managed = self._sessions.get(group_id)
            if managed is None:
                return StopInteractionResult()
            pending_waits = list(managed.pending_background_waits.values())
            for pending in pending_waits:
                self._cancel_background_wait(pending)
            managed.pending_background_waits.clear()
            should_schedule_idle = bool(pending_waits) and not managed.session.is_processing
            if should_schedule_idle:
                managed.active_user_id = None
                managed.last_used_at = time.monotonic()
                self._schedule_idle_reset_locked(managed)

        stopped_background_session_ids = await self._stop_background_sessions(
            managed.session,
            [pending.session_id for pending in pending_waits],
        )
        aborted_streaming = False
        if managed.session.is_processing:
            managed.session.clear_all_queues()
            managed.session.abort(reason)
            aborted_streaming = True

        return StopInteractionResult(
            managed=managed,
            aborted_streaming=aborted_streaming,
            stopped_background_waits=bool(pending_waits),
            stopped_background_session_ids=stopped_background_session_ids,
        )

    async def has_context(self, group_id: str) -> bool:
        async with self._guard:
            managed = self._sessions.get(group_id)
            if managed is not None and managed.session.messages:
                return True
        return self.session_file_for_group(group_id).exists()

    async def clear_context(self, group_id: str) -> bool:
        async with self._guard:
            managed = self._sessions.pop(group_id, None)

        if managed is not None:
            await self._dispose_session(
                managed,
                reason="clear_context",
                clear_history=True,
                clear_workspace=False,
            )
            return True

        session_file = self.session_file_for_group(group_id)
        existed = session_file.exists()
        if existed:
            try:
                session_file.unlink(missing_ok=True)
                logger.info(
                    f"bampi_chat cleared persisted session history group_id={group_id} "
                    f"session_file={session_file}"
                )
            except OSError:
                logger.warning(
                    f"bampi_chat failed to clear persisted session history group_id={group_id} "
                    f"session_file={session_file}"
                )
                return False
        return existed

    async def close_idle(self) -> None:
        ttl = self._config.bampi_session_idle_ttl_seconds
        if ttl <= 0:
            return

        stale_ids: list[str] = []
        now = time.monotonic()
        async with self._guard:
            for group_id, managed in self._sessions.items():
                if managed.lock.locked():
                    continue
                if managed.pending_background_waits:
                    continue
                if now - managed.last_used_at >= ttl:
                    stale_ids.append(group_id)

            stale_sessions = [self._sessions.pop(group_id) for group_id in stale_ids]

        for managed in stale_sessions:
            await self._dispose_session(
                managed,
                reason="idle_timeout",
                clear_history=True,
                clear_workspace=False,
            )

    async def release(self, group_id: str) -> None:
        async with self._guard:
            managed = self._sessions.pop(group_id, None)
        if managed is not None:
            await self._dispose_session(managed, reason="release", clear_history=False)

    async def close_all(self) -> None:
        async with self._guard:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        logger.info(f"bampi_chat closing all sessions count={len(sessions)}")
        for managed in sessions:
            await self._dispose_session(managed, reason="shutdown", clear_history=False)

    def _build_session(self, group_id: str) -> AgentSession:
        return self._create_agent_session(
            group_id,
            persist=True,
            session_file=str((self._session_dir / f"group-{group_id}.jsonl").resolve()),
            include_schedule=True,
        )

    def _create_agent_session(
        self,
        group_id: str,
        *,
        persist: bool,
        session_file: str | None,
        include_schedule: bool,
        system_prompt_suffix: str | None = None,
    ) -> AgentSession:
        workspace_dir = self.workspace_dir_for_group(group_id)
        container_workspace_dir = self.container_workspace_dir_for_group(group_id)
        model_workspace_root = container_workspace_dir
        model = self._build_model()
        tools = create_agent_tools(
            self._config,
            workspace_dir,
            container_root=model_workspace_root,
            bash_workdir=container_workspace_dir,
            group_id=group_id,
            service_manager=self._service_manager,
            schedule_manager=self._schedule_manager,
            include_schedule=include_schedule,
        )
        tool_names = [tool.name for tool in tools]
        loaded_skills = load_chat_skills(workspace_dir)
        system_prompt = build_system_prompt(
            self._config,
            tool_names,
            skills=build_prompt_skills(loaded_skills.skills, workspace_dir=workspace_dir),
            prompt_cwd=model_workspace_root,
            append_system_prompt=system_prompt_suffix,
        )
        stream_options = SimpleStreamOptions(api_key=self._config.bampi_api_key or None)

        session_manager = (
            SessionManager(
                workspace_dir,
                session_file=session_file,
                persist=True,
            )
            if persist and session_file
            else SessionManager.in_memory(workspace_dir)
        )

        logger.info(
            f"bampi_chat building session group_id={group_id} "
            f"workspace_dir={workspace_dir} "
            f"container_workspace_dir={container_workspace_dir} "
            f"provider={model.provider} "
            f"api={model.api} "
            f"model={model.id} "
            f"session_file={session_file or '<in-memory>'} "
            f"bash_mode={self._config.bampi_bash_mode} "
            f"bash_container={self._config.bampi_bash_container_name} "
            f"bash_workdir={container_workspace_dir} "
            f"tools={tool_names} "
            f"skills={[skill.name for skill in loaded_skills.skills]}"
        )
        for diagnostic in loaded_skills.diagnostics:
            logger.warning(
                f"bampi_chat skill diagnostic group_id={group_id} "
                f"type={diagnostic.type} "
                f"path={diagnostic.path} "
                f"message={diagnostic.message}"
            )

        return AgentSession(
            cwd=workspace_dir,
            model=model,
            thinking_level=self._config.bampi_thinking_level,
            tools=tools,
            session_manager=session_manager,
            custom_system_prompt=system_prompt,
            augment_custom_system_prompt=False,
            stream_options=stream_options,
            get_api_key=self._resolve_api_key,
            max_turns=self._config.bampi_max_turns,
        )

    async def create_ephemeral_session(
        self,
        group_id: str,
        *,
        include_schedule: bool = False,
        system_prompt_suffix: str | None = None,
        reason: str = "ephemeral",
    ) -> AgentSession:
        session = self._create_agent_session(
            group_id,
            persist=False,
            session_file=None,
            include_schedule=include_schedule,
            system_prompt_suffix=system_prompt_suffix,
        )
        self._attach_session_debug_logging(session, f"{group_id}:{reason}")
        await session.start()
        return session

    async def close_ephemeral_session(self, session: AgentSession) -> None:
        await self._close_session_tools(session)
        await session.close()

    async def _get_or_create_locked(self, group_id: str) -> ManagedGroupSession:
        managed = self._sessions.get(group_id)
        if managed is not None:
            self._touch_session(managed)
            logger.info(
                f"bampi_chat reusing session group_id={group_id} "
                f"message_count={len(managed.session.messages)}"
            )
            return managed

        session = self._build_session(group_id)
        self._attach_session_debug_logging(session, group_id)
        await session.start()
        managed = ManagedGroupSession(group_id=group_id, session=session)
        self._attach_background_wait_listeners(managed)
        self._touch_session(managed)
        self._sessions[group_id] = managed
        logger.info(
            f"bampi_chat created session group_id={group_id} "
            f"restored_message_count={len(session.messages)}"
        )
        return managed

    def _attach_session_debug_logging(self, session: AgentSession, group_id: str) -> None:
        def _listener(event: Any) -> None:
            event_type = getattr(event, "type", None)
            if event_type == "tool_execution_start":
                logger.info(
                    f"bampi_chat tool start group_id={group_id} "
                    f"tool={getattr(event, 'tool_name', '')} "
                    f"tool_call_id={getattr(event, 'tool_call_id', '')} "
                    f"args={self._truncate_text(repr(getattr(event, 'args', None)))}"
                )
            elif event_type == "tool_execution_end":
                result = getattr(event, "result", None)
                logger.info(
                    f"bampi_chat tool end group_id={group_id} "
                    f"tool={getattr(event, 'tool_name', '')} "
                    f"tool_call_id={getattr(event, 'tool_call_id', '')} "
                    f"is_error={getattr(event, 'is_error', False)} "
                    f"content={self._summarize_tool_result(result)}"
                )

        session.subscribe(_listener)

    @staticmethod
    def _summarize_tool_result(result: Any) -> str:
        if result is None:
            return "None"
        content = getattr(result, "content", None)
        if not isinstance(content, list) or not content:
            return "[]"

        parts: list[str] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    parts.append(f"text:{GroupSessionManager._truncate_text(text)}")
            elif block_type == "image":
                mime_type = getattr(block, "mime_type", "")
                parts.append(f"image:{mime_type or 'unknown'}")
            else:
                parts.append(str(block_type or type(block).__name__))
        return "[" + ", ".join(parts) + "]"

    @staticmethod
    def _truncate_text(text: str, limit: int = 240) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3]}..."

    def _touch_session(self, managed: ManagedGroupSession) -> None:
        now = time.monotonic()
        managed.last_used_at = now
        self._cancel_idle_reset_task(managed)

    def _cancel_background_wait(self, pending: PendingBackgroundWait) -> None:
        pending.cancelled = True
        task = pending.reminder_task
        pending.reminder_task = None
        if task is None:
            return
        if task is asyncio.current_task():
            return
        task.cancel()

    @staticmethod
    def _background_wait_owner(managed: ManagedGroupSession) -> str | None:
        for pending in managed.pending_background_waits.values():
            if pending.owner_user_id is not None:
                return pending.owner_user_id
        return None

    async def _run_background_wait_reminder(
        self,
        *,
        group_id: str,
        pending: PendingBackgroundWait,
        after_seconds: float,
    ) -> None:
        try:
            await asyncio.sleep(after_seconds)
            if pending.cancelled or pending.reminder_callback is None:
                return
            async with self._guard:
                managed = self._sessions.get(group_id)
                if managed is None:
                    return
                current = managed.pending_background_waits.get(pending.session_id)
                if current is not pending or pending.cancelled:
                    return
            reminder = BackgroundWaitReminderEvent(
                group_id=group_id,
                session_id=pending.session_id,
                owner_user_id=pending.owner_user_id,
                command=pending.command,
                waited_seconds=after_seconds,
            )
            maybe = pending.reminder_callback(reminder)
            if inspect.isawaitable(maybe):
                await maybe
        except asyncio.CancelledError:
            logger.debug(
                f"bampi_chat background wait reminder cancelled "
                f"group_id={group_id} session_id={pending.session_id}"
            )
        except Exception:
            logger.exception(
                f"bampi_chat background wait reminder failed "
                f"group_id={group_id} session_id={pending.session_id}"
            )

    def _schedule_idle_reset_locked(self, managed: ManagedGroupSession) -> None:
        idle_ttl = self._config.bampi_session_idle_ttl_seconds
        if idle_ttl <= 0:
            return
        self._cancel_idle_reset_task(managed)
        scheduled_at = managed.last_used_at
        managed.idle_reset_task = asyncio.create_task(
            self._run_idle_reset(managed.group_id, scheduled_at),
            name=f"bampi-chat-idle-reset-{managed.group_id}",
        )

    def _cancel_idle_reset_task(self, managed: ManagedGroupSession) -> None:
        task = managed.idle_reset_task
        if task is None:
            return
        managed.idle_reset_task = None
        if task is asyncio.current_task():
            return
        task.cancel()

    async def _run_idle_reset(self, group_id: str, scheduled_at: float) -> None:
        idle_ttl = self._config.bampi_session_idle_ttl_seconds
        if idle_ttl <= 0:
            return

        try:
            remaining = max(0.0, scheduled_at + idle_ttl - time.monotonic())
            if remaining > 0:
                await asyncio.sleep(remaining)

            async with self._guard:
                managed = self._sessions.get(group_id)
                if managed is None or managed.last_used_at != scheduled_at:
                    return
                if managed.pending_background_waits:
                    return
                self._sessions.pop(group_id, None)

            async with managed.lock:
                await self._dispose_session(
                    managed,
                    reason="idle_timeout",
                    clear_history=True,
                    clear_workspace=False,
                )
        except asyncio.CancelledError:
            logger.debug(f"bampi_chat idle reset cancelled group_id={group_id}")
        except Exception:
            logger.exception(f"bampi_chat idle reset failed group_id={group_id}")
        finally:
            async with self._guard:
                managed = self._sessions.get(group_id)
                current_task = asyncio.current_task()
                if managed is not None and managed.idle_reset_task is current_task:
                    managed.idle_reset_task = None

    async def _dispose_session(
        self,
        managed: ManagedGroupSession,
        *,
        reason: str,
        clear_history: bool = True,
        clear_workspace: bool = True,
    ) -> None:
        self._cancel_idle_reset_task(managed)
        managed.active_user_id = None
        pending_wait_ids = list(managed.pending_background_waits)
        for pending in managed.pending_background_waits.values():
            self._cancel_background_wait(pending)
        managed.pending_background_waits.clear()
        for unsubscribe in managed.background_listener_unsubscribes:
            try:
                unsubscribe()
            except Exception:
                logger.exception(
                    f"bampi_chat failed to unsubscribe background listener "
                    f"group_id={managed.group_id}"
                )
        managed.background_listener_unsubscribes.clear()
        session_file = managed.session.session_manager.session_file
        logger.info(
            f"bampi_chat disposing session group_id={managed.group_id} "
            f"reason={reason} "
            f"clear_history={clear_history} "
            f"clear_workspace={clear_workspace} "
            f"pending_background_waits={pending_wait_ids}"
        )
        await self._close_session_tools(managed.session)
        await managed.session.close()
        if clear_history and session_file:
            path = Path(session_file)
            try:
                path.unlink(missing_ok=True)
                logger.info(
                    f"bampi_chat cleared session history group_id={managed.group_id} "
                    f"session_file={path}"
                )
            except OSError:
                logger.warning(
                    f"bampi_chat failed to clear session history group_id={managed.group_id} "
                    f"session_file={path}"
                )
        if clear_history and clear_workspace:
            async with self._guard:
                if managed.group_id in self._sessions:
                    return
                try:
                    workspace_dir = self.workspace_dir_for_group(managed.group_id)
                    reset_workspace_files(workspace_dir)
                    logger.info(
                        f"bampi_chat reset workspace files group_id={managed.group_id} "
                        f"workspace_dir={workspace_dir}"
                    )
                except OSError:
                    logger.warning(
                        f"bampi_chat failed to reset workspace files group_id={managed.group_id} "
                        f"workspace_dir={self.workspace_dir_for_group(managed.group_id)}"
                    )

    def session_file_for_group(self, group_id: str) -> Path:
        return (self._session_dir / f"group-{group_id}.jsonl").resolve()

    def _attach_background_wait_listeners(self, managed: ManagedGroupSession) -> None:
        for tool in managed.session.get_all_tools():
            if not isinstance(tool, SafeBashTool):
                continue
            unsubscribe = tool.add_exit_listener(
                lambda event, group_id=managed.group_id: self._handle_background_session_exit(group_id, event)
            )
            managed.background_listener_unsubscribes.append(unsubscribe)

    async def _handle_background_session_exit(
        self,
        group_id: str,
        event: BackgroundSessionExitEvent,
    ) -> None:
        async with self._guard:
            managed = self._sessions.get(group_id)
            if managed is None:
                return
            pending = managed.pending_background_waits.get(event.session_id)
        if pending is None:
            return

        async def _run_callback() -> None:
            try:
                if pending.cancelled:
                    return
                maybe = pending.callback(event)
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception:
                logger.exception(
                    f"bampi_chat background wait callback failed group_id={group_id} "
                    f"session_id={event.session_id}"
                )
            finally:
                async with self._guard:
                    managed = self._sessions.get(group_id)
                    if managed is not None:
                        self._cancel_background_wait(pending)
                        managed.pending_background_waits.pop(event.session_id, None)
                        if managed.active_user_id is None and not managed.lock.locked():
                            managed.last_used_at = time.monotonic()
                            self._schedule_idle_reset_locked(managed)

        asyncio.create_task(
            _run_callback(),
            name=f"bampi-chat-background-wait-{group_id}-{event.session_id}",
        )

    async def _close_session_tools(self, session: AgentSession) -> None:
        for tool in session.get_all_tools():
            close = getattr(tool, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    f"bampi_chat failed to close tool "
                    f"tool={getattr(tool, 'name', type(tool).__name__)}"
                )

    async def _stop_background_sessions(
        self,
        session: AgentSession,
        session_ids: list[str],
    ) -> list[str]:
        if not session_ids:
            return []

        stopped: list[str] = []
        remaining = set(session_ids)
        for tool in session.get_all_tools():
            if not isinstance(tool, SafeBashTool):
                continue
            for session_id in list(remaining):
                try:
                    await tool.stop_session(session_id)
                    stopped.append(session_id)
                    remaining.discard(session_id)
                except Exception:
                    logger.exception(
                        f"bampi_chat failed to stop background session "
                        f"session_id={session_id}"
                    )
        return stopped

    def _build_model(self) -> Model:
        model = get_model(
            self._config.bampi_model_id,
            provider=self._config.bampi_model_provider,
        )
        if model is None:
            model = self._build_custom_model()
        return self._apply_model_overrides(model)

    def _build_custom_model(self) -> Model:
        provider = self._config.bampi_model_provider
        model_id = self._config.bampi_model_id
        if not provider or not model_id:
            raise RuntimeError("Custom model requires non-empty provider and model_id")

        api = self._resolve_model_api(provider)
        logger.warning(
            f"bampi_chat using custom model "
            f"provider={provider} model={model_id} api={api}"
        )
        return Model(
            id=model_id,
            name=model_id,
            api=api,
            provider=provider,
            base_url=self._config.bampi_base_url,
            reasoning=False,
            input_types=["text", "image"],
            context_window=128_000,
            max_tokens=16_384,
            cost=ModelCost(),
        )

    def _apply_model_overrides(self, model: Model) -> Model:
        updates: dict[str, Any] = {}
        api = self._config.bampi_model_api
        if api != "auto" and api != model.api:
            logger.warning(
                f"bampi_chat overriding model api "
                f"provider={model.provider} model={model.id} from={model.api} to={api}"
            )
            updates["api"] = api
        if self._config.bampi_base_url:
            updates["base_url"] = self._config.bampi_base_url
        if not updates:
            return model
        return model.model_copy(update=updates)

    def _resolve_model_api(self, provider: str) -> str:
        configured_api = self._config.bampi_model_api
        if configured_api != "auto":
            return configured_api

        provider_key = provider.strip().lower().replace("_", "-")
        if provider_key in _API_KEY_ENV_BY_API:
            return provider_key
        if provider_key in {"anthropic", "claude"} or "anthropic" in provider_key:
            return "anthropic-messages"
        if (
            provider_key in {"google", "gemini"}
            or "google" in provider_key
            or "gemini" in provider_key
        ):
            return "google-genai"
        if provider_key == "openai":
            return "openai-responses"
        if provider_key == "ollama" or "ollama" in provider_key:
            return "ollama-responses"
        return "openai-completions"

    async def _resolve_api_key(self, provider: str) -> str | None:
        if self._config.bampi_api_key:
            logger.info(f"bampi_chat resolved api key provider={provider} source=config")
            return self._config.bampi_api_key

        env_keys = self._candidate_api_key_env_keys(provider)
        for env_key in env_keys:
            config_value = self._resolve_nonebot_config_value(env_key.lower())
            if config_value is not None:
                logger.info(
                    f"bampi_chat resolved api key provider={provider} "
                    f"source=nonebot_config key={env_key.lower()}"
                )
                return config_value
            env_value = os.environ.get(env_key, "") or None
            if env_value is not None:
                logger.info(
                    f"bampi_chat resolved api key provider={provider} "
                    f"source=env env={env_key}"
                )
                return env_value

        logger.warning(
            f"bampi_chat api key missing provider={provider} "
            f"candidates={env_keys}"
        )
        return None

    def _candidate_api_key_env_keys(self, provider: str) -> list[str]:
        candidates: list[str] = []

        normalized_provider = re.sub(
            r"[^A-Z0-9]+",
            "_",
            provider.strip().upper(),
        ).strip("_")
        if normalized_provider:
            candidates.append(f"{normalized_provider}_API_KEY")

        api = self._resolve_model_api(provider)
        api_env_key = _API_KEY_ENV_BY_API.get(api)
        if api_env_key:
            candidates.append(api_env_key)

        deduped: list[str] = []
        for env_key in candidates:
            if env_key not in deduped:
                deduped.append(env_key)
        return deduped

    @staticmethod
    def _resolve_nonebot_config_value(key: str) -> str | None:
        try:
            driver = get_driver()
        except ValueError:
            return None

        value = getattr(driver.config, key, None)
        if value is None:
            return None
        text = str(value).strip()
        return text or None
