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

from bampy.agent.messages import clone_message
from bampy.ai import Model, ModelCost, SimpleStreamOptions, get_model
from bampy.app import AgentSession, BeforeAgentStartEventResult, ExtensionAPI, SessionManager

from .config import BampiChatConfig
from .memory import MemoryManager, MemoryParticipant, MemoryUserTurn
from .prompt import build_system_prompt
from .service_manager import ServiceManager
from .skills import build_prompt_skills, load_chat_skills
from .tools import create_agent_tools
from .tools.safe_bash import BackgroundSessionExitEvent, SafeBashTool
from .tools.workspace import (
    cleanup_stale_group_workspaces,
    reset_workspace_files,
    resolve_group_container_workspace,
    resolve_group_workspace_dir,
)

# Invoked with (managed, exit_event) whenever a notify_on_exit background
# session finishes. The handler owns delivery: steering the result into a live
# turn or driving a resume turn and replying to the group.
BackgroundNotifyHandler = Callable[
    ["ManagedGroupSession", BackgroundSessionExitEvent],
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
    # Handler-owned reply metadata (who started a notify_on_exit session),
    # keyed by bash session id. Purely cosmetic: exit delivery works without it.
    background_task_context: dict[str, Any] = field(default_factory=dict)
    # Session ids whose exit notification must be swallowed (e.g. /stop).
    suppressed_background_session_ids: set[str] = field(default_factory=set)
    background_listener_unsubscribes: list[Callable[[], None]] = field(default_factory=list)
    memory_user_turns: list[MemoryUserTurn] = field(default_factory=list)
    memory_participants: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryTurnState:
    current_user_id: str
    current_nickname: str
    participants: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class StopInteractionResult:
    managed: ManagedGroupSession | None = None
    aborted_streaming: bool = False
    stopped_background_sessions: bool = False
    stopped_background_session_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GroupInteractionStatus:
    managed: ManagedGroupSession | None = None
    active_user_id: str | None = None
    is_streaming: bool = False
    running_background_session_ids: list[str] = field(default_factory=list)
    background_owner_user_ids: frozenset[str] = frozenset()

    @property
    def has_running_background(self) -> bool:
        return bool(self.running_background_session_ids)

    @property
    def is_active(self) -> bool:
        return self.active_user_id is not None


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
        self._memory_manager = MemoryManager.from_config(config) if config.bampi_memory_enabled else None
        self._background_archive_tasks: set[asyncio.Task[None]] = set()
        self._background_notify_handler: BackgroundNotifyHandler | None = None
        self._background_notify_tasks: set[asyncio.Task[None]] = set()
        self._workspace_cleanup_task: asyncio.Task[None] | None = None
        self._workspace_cleanup_lock = asyncio.Lock()
        self._memory_turn_states: dict[str, MemoryTurnState] = {}
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
            f"memory_enabled={config.bampi_memory_enabled} "
            f"memory_storage_mode={config.bampi_memory_storage_mode} "
            f"service_port_range={config.bampi_service_port_range} "
            f"service_public_host={config.bampi_service_public_host or '<unset>'} "
            f"idle_ttl={config.bampi_session_idle_ttl_seconds}s "
            f"workspace_cleanup_enabled={config.bampi_workspace_cleanup_enabled} "
            f"workspace_cleanup_ttl={config.bampi_workspace_cleanup_ttl_seconds}s "
            f"workspace_cleanup_interval={config.bampi_workspace_cleanup_interval_seconds}s"
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

    @property
    def memory_manager(self) -> MemoryManager | None:
        return self._memory_manager

    async def start_memory_tasks(self) -> None:
        if self._memory_manager is None:
            return
        model = self._build_memory_model()
        api_key = await self._resolve_memory_api_key(
            model.provider,
            configured_api=model.api,
        )
        self._memory_manager.start_background_tasks(model=model, api_key=api_key)

    def close_memory_tasks(self) -> None:
        if self._memory_manager is None:
            return
        self._memory_manager.close_background_tasks()

    def start_workspace_cleanup_tasks(self) -> None:
        if not self._config.bampi_workspace_cleanup_enabled:
            return
        if self._workspace_cleanup_task is not None and not self._workspace_cleanup_task.done():
            return
        self._workspace_cleanup_task = asyncio.create_task(
            self._run_workspace_cleanup_loop(),
            name="bampi-chat-workspace-cleanup",
        )

    async def close_workspace_cleanup_tasks(self) -> None:
        task = self._workspace_cleanup_task
        self._workspace_cleanup_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def run_workspace_cleanup_once(self) -> None:
        if not self._config.bampi_workspace_cleanup_enabled:
            return
        if self._workspace_cleanup_lock.locked():
            return
        async with self._guard:
            active_workspace_dirs = {
                self.workspace_dir_for_group(group_id)
                for group_id in self._sessions
            }
        async with self._workspace_cleanup_lock:
            ttl = max(0, self._config.bampi_workspace_cleanup_ttl_seconds)
            results = await asyncio.to_thread(
                cleanup_stale_group_workspaces,
                self._workspace_root_dir,
                ttl_seconds=ttl,
                skip_workspace_dirs=active_workspace_dirs,
            )
        deleted_files = sum(result.deleted_files for result in results)
        deleted_dirs = sum(result.deleted_dirs for result in results)
        errors = sum(len(result.errors) for result in results)
        if deleted_files or deleted_dirs or errors:
            samples = [
                sample
                for result in results
                for sample in result.deleted_samples[:5]
            ][:10]
            logger.info(
                f"bampi_chat workspace cleanup finished "
                f"workspaces={len(results)} "
                f"deleted_files={deleted_files} "
                f"deleted_dirs={deleted_dirs} "
                f"errors={errors} "
                f"samples={samples}"
            )

    async def _run_workspace_cleanup_loop(self) -> None:
        interval = max(60.0, float(self._config.bampi_workspace_cleanup_interval_seconds))
        try:
            while True:
                try:
                    await self.run_workspace_cleanup_once()
                except Exception:
                    logger.exception("bampi_chat workspace cleanup run failed")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("bampi_chat workspace cleanup loop cancelled")
            raise

    def prepare_memory_for_user_turn(
        self,
        managed: ManagedGroupSession,
        *,
        user_id: str,
        nickname: str,
        message: Any,
    ) -> None:
        if self._memory_manager is None:
            return
        normalized_user_id = str(user_id).strip()
        if not normalized_user_id:
            return
        normalized_nickname = str(nickname).strip()
        timestamp = getattr(message, "timestamp", None)
        try:
            timestamp_value = float(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            timestamp_value = None
        managed.memory_user_turns.append(
            MemoryUserTurn(
                user_id=normalized_user_id,
                nickname=normalized_nickname,
                timestamp=timestamp_value,
            )
        )
        managed.memory_participants[normalized_user_id] = normalized_nickname
        self._memory_turn_states[managed.group_id] = MemoryTurnState(
            current_user_id=normalized_user_id,
            current_nickname=normalized_nickname,
            participants=dict(managed.memory_participants),
        )

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
            owner_user_ids = {
                str(owner)
                for owner in (
                    getattr(context, "user_id", None)
                    for context in managed.background_task_context.values()
                )
                if owner is not None
            }
            return GroupInteractionStatus(
                managed=managed,
                active_user_id=managed.active_user_id,
                is_streaming=managed.session.is_processing,
                running_background_session_ids=self._running_notify_session_ids(managed),
                background_owner_user_ids=frozenset(owner_user_ids),
            )

    async def reserve_interaction(self, group_id: str, user_id: str) -> InteractionReservation:
        await self.close_idle()
        async with self._guard:
            managed = await self._get_or_create_locked(group_id)
            self._touch_session(managed)
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
            if not self._has_running_notify_sessions(managed):
                self._schedule_idle_reset_locked(managed)

    def set_background_notify_handler(self, handler: BackgroundNotifyHandler) -> None:
        self._background_notify_handler = handler

    async def stop_interaction(self, group_id: str, *, reason: str) -> StopInteractionResult:
        async with self._guard:
            managed = self._sessions.get(group_id)
            if managed is None:
                return StopInteractionResult()
            running_session_ids = self._running_notify_session_ids(managed)
            # Suppress exit notifications for sessions we are about to kill:
            # the user asked for silence, not an auto-resume of the task.
            managed.suppressed_background_session_ids.update(running_session_ids)
            for session_id in running_session_ids:
                managed.background_task_context.pop(session_id, None)

        stopped_background_session_ids = await self._stop_background_sessions(
            managed.session,
            running_session_ids,
        )
        aborted_streaming = False
        if managed.session.is_processing:
            managed.session.clear_all_queues()
            managed.session.abort(reason)
            aborted_streaming = True

        async with self._guard:
            if (
                self._sessions.get(group_id) is managed
                and managed.active_user_id is None
                and not managed.lock.locked()
                and not self._has_running_notify_sessions(managed)
            ):
                managed.last_used_at = time.monotonic()
                self._schedule_idle_reset_locked(managed)

        return StopInteractionResult(
            managed=managed,
            aborted_streaming=aborted_streaming,
            stopped_background_sessions=bool(stopped_background_session_ids),
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
            self._schedule_archive_persisted_session_if_needed(
                group_id,
                session_file=session_file,
                reason="clear_context",
            )
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
                if self._has_running_notify_sessions(managed):
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
        await self.close_workspace_cleanup_tasks()
        async with self._guard:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        logger.info(f"bampi_chat closing all sessions count={len(sessions)}")
        for managed in sessions:
            await self._dispose_session(managed, reason="shutdown", clear_history=False)
        await self._cancel_background_notify_tasks()
        await self._cancel_background_archive_tasks()

    async def wait_for_background_archives(self) -> None:
        while True:
            tasks = [task for task in self._background_archive_tasks if not task.done()]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    def _memory_current_user(self, group_id: str) -> tuple[str, str] | None:
        state = self._memory_turn_states.get(group_id)
        if state is None or not state.current_user_id:
            return None
        return (state.current_user_id, state.current_nickname)

    def _build_memory_extension(self, group_id: str):
        api = ExtensionAPI("bampi_chat_memory")

        def _on_before_agent_start(event: Any, _ctx: Any) -> BeforeAgentStartEventResult | None:
            manager = self._memory_manager
            state = self._memory_turn_states.get(group_id)
            if manager is None or state is None:
                return None
            participants = [
                MemoryParticipant(user_id=user_id, nickname=nickname)
                for user_id, nickname in state.participants.items()
                if user_id
            ]
            context = manager.get_memory_context_for_turn(
                group_id=group_id,
                current_user_id=state.current_user_id,
                current_nickname=state.current_nickname,
                session_participants=participants,
            )
            if not context.strip():
                return None
            return BeforeAgentStartEventResult(
                system_prompt=f"{event.system_prompt}\n\n## 记忆上下文\n{context.strip()}"
            )

        api.on("before_agent_start", _on_before_agent_start)
        return api._build_extension()

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
            supports_images="image" in model.input_types,
            container_root=model_workspace_root,
            bash_workdir=container_workspace_dir,
            group_id=group_id,
            memory_manager=self._memory_manager,
            memory_current_user_provider=lambda group_id=group_id: self._memory_current_user(group_id),
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
            f"input_types={model.input_types} "
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

        extensions = []
        if self._memory_manager is not None:
            extensions.append(self._build_memory_extension(group_id))

        return AgentSession(
            cwd=workspace_dir,
            model=model,
            thinking_level=self._config.bampi_thinking_level,
            tools=tools,
            extensions=extensions,
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
        self._attach_background_exit_listeners(managed)
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

    @staticmethod
    def _running_notify_session_ids(managed: ManagedGroupSession) -> list[str]:
        session_ids: list[str] = []
        for tool in managed.session.get_all_tools():
            if isinstance(tool, SafeBashTool):
                session_ids.extend(tool.running_notify_session_ids())
        return session_ids

    @classmethod
    def _has_running_notify_sessions(cls, managed: ManagedGroupSession) -> bool:
        return bool(cls._running_notify_session_ids(managed))

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
                if self._has_running_notify_sessions(managed):
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
        running_notify_ids = self._running_notify_session_ids(managed)
        managed.background_task_context.clear()
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
            f"running_notify_sessions={running_notify_ids}"
        )
        self._schedule_archive_session_if_needed(
            managed,
            reason=reason,
            clear_history=clear_history,
        )
        await self._close_session_tools(managed.session)
        await managed.session.close()
        self._memory_turn_states.pop(managed.group_id, None)
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

    def _schedule_archive_session_if_needed(
        self,
        managed: ManagedGroupSession,
        *,
        reason: str,
        clear_history: bool,
    ) -> None:
        manager = self._memory_manager
        if manager is None:
            return
        if reason == "shutdown" or not clear_history:
            return
        messages = [clone_message(message) for message in managed.session.messages]
        if not messages:
            return
        user_turns = list(managed.memory_user_turns)
        try:
            model = self._build_memory_model()
        except Exception:
            logger.exception(
                f"bampi_chat failed to build memory model for archive "
                f"group_id={managed.group_id}"
            )
            return
        self._schedule_archive_snapshot(
            manager=manager,
            group_id=managed.group_id,
            messages=messages,
            user_turns=user_turns,
            model=model,
            api_key=self._config.bampi_memory_api_key or None,
            reason=reason,
        )

    def _schedule_archive_persisted_session_if_needed(
        self,
        group_id: str,
        *,
        session_file: Path,
        reason: str,
    ) -> None:
        manager = self._memory_manager
        if manager is None:
            return
        try:
            persisted_session = SessionManager(
                self.workspace_dir_for_group(group_id),
                session_file=str(session_file),
                persist=True,
            )
            context = persisted_session.build_session_context()
            messages = [clone_message(message) for message in context.messages]
        except Exception:
            logger.exception(
                f"bampi_chat failed to load persisted session for memory archive "
                f"group_id={group_id} session_file={session_file}"
            )
            return
        if not messages:
            return
        try:
            model = self._build_memory_model()
        except Exception:
            logger.exception(
                f"bampi_chat failed to build memory model for persisted archive "
                f"group_id={group_id} session_file={session_file}"
            )
            return
        self._schedule_archive_snapshot(
            manager=manager,
            group_id=group_id,
            messages=messages,
            user_turns=[],
            model=model,
            api_key=self._config.bampi_memory_api_key or None,
            reason=reason,
        )

    def _schedule_archive_snapshot(
        self,
        *,
        manager: MemoryManager,
        group_id: str,
        messages: list[Any],
        user_turns: list[MemoryUserTurn],
        model: Any,
        api_key: str | None,
        reason: str,
    ) -> None:
        task = asyncio.create_task(
            self._archive_session_snapshot(
                manager=manager,
                group_id=group_id,
                messages=messages,
                user_turns=user_turns,
                model=model,
                api_key=api_key,
                reason=reason,
            ),
            name=f"bampi-chat-memory-archive-{group_id}",
        )
        self._background_archive_tasks.add(task)
        task.add_done_callback(self._background_archive_tasks.discard)

    async def _archive_session_snapshot(
        self,
        *,
        manager: MemoryManager,
        group_id: str,
        messages: list[Any],
        user_turns: list[MemoryUserTurn],
        model: Any,
        api_key: str | None,
        reason: str,
    ) -> None:
        try:
            if api_key is None and model is not None:
                provider = str(getattr(model, "provider", "")).strip()
                if provider:
                    model_api = str(getattr(model, "api", "")).strip() or None
                    api_key = await self._resolve_memory_api_key(
                        provider,
                        configured_api=model_api,
                    )
            archive_id = await manager.archive_session_async(
                group_id=group_id,
                messages=messages,
                user_turns=user_turns,
                model=model,
                api_key=api_key,
            )
        except Exception:
            logger.exception(
                f"bampi_chat failed to archive memory session group_id={group_id} "
                f"reason={reason}"
            )
            return
        if archive_id is None:
            logger.info(
                f"bampi_chat skipped memory archive group_id={group_id} "
                f"reason={reason} message_count={len(messages)}"
            )
            return
        logger.info(
            f"bampi_chat archived memory session group_id={group_id} "
            f"reason={reason} archive_id={archive_id}"
        )

    async def _cancel_background_archive_tasks(self) -> None:
        tasks = list(self._background_archive_tasks)
        if not tasks:
            return
        logger.info(f"bampi_chat cancelling background memory archives count={len(tasks)}")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._background_archive_tasks.clear()

    def session_file_for_group(self, group_id: str) -> Path:
        return (self._session_dir / f"group-{group_id}.jsonl").resolve()

    def _attach_background_exit_listeners(self, managed: ManagedGroupSession) -> None:
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
        if not event.notify_on_exit:
            return
        async with self._guard:
            managed = self._sessions.get(group_id)
            if managed is None:
                return
            if event.session_id in managed.suppressed_background_session_ids:
                managed.suppressed_background_session_ids.discard(event.session_id)
                managed.background_task_context.pop(event.session_id, None)
                logger.info(
                    f"bampi_chat suppressed background exit notification "
                    f"group_id={group_id} session_id={event.session_id}"
                )
                return
        handler = self._background_notify_handler
        if handler is None:
            logger.warning(
                f"bampi_chat background session exited but no notify handler is registered "
                f"group_id={group_id} session_id={event.session_id}"
            )
            return
        task = asyncio.create_task(
            self._run_background_notify(managed, event, handler),
            name=f"bampi-chat-background-notify-{group_id}-{event.session_id}",
        )
        self._background_notify_tasks.add(task)
        task.add_done_callback(self._background_notify_tasks.discard)

    async def _run_background_notify(
        self,
        managed: ManagedGroupSession,
        event: BackgroundSessionExitEvent,
        handler: BackgroundNotifyHandler,
    ) -> None:
        try:
            maybe = handler(managed, event)
            if inspect.isawaitable(maybe):
                await maybe
        except Exception:
            logger.exception(
                f"bampi_chat background notify handler failed "
                f"group_id={managed.group_id} session_id={event.session_id}"
            )
        finally:
            managed.background_task_context.pop(event.session_id, None)
            async with self._guard:
                if (
                    self._sessions.get(managed.group_id) is managed
                    and managed.active_user_id is None
                    and not managed.lock.locked()
                    and not self._has_running_notify_sessions(managed)
                ):
                    managed.last_used_at = time.monotonic()
                    self._schedule_idle_reset_locked(managed)

    async def _cancel_background_notify_tasks(self) -> None:
        tasks = [task for task in self._background_notify_tasks if not task.done()]
        if not tasks:
            return
        logger.info(f"bampi_chat cancelling background notify tasks count={len(tasks)}")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._background_notify_tasks.clear()

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
        return self._build_model_from_spec(
            provider=self._config.bampi_model_provider,
            model_id=self._config.bampi_model_id,
            model_api=self._config.bampi_model_api,
            base_url=self._config.bampi_base_url,
        )

    def _build_memory_model(self) -> Model:
        cfg = self._config
        provider = cfg.bampi_memory_model_provider or cfg.bampi_model_provider
        model_id = cfg.bampi_memory_model_id or cfg.bampi_model_id
        base_url = cfg.bampi_memory_base_url or cfg.bampi_base_url
        memory_identity_overridden = bool(
            cfg.bampi_memory_model_provider or cfg.bampi_memory_model_id
        )
        if cfg.bampi_memory_model_api != "auto":
            model_api = cfg.bampi_memory_model_api
        elif memory_identity_overridden:
            # Independent memory model: infer API from its own provider/registry.
            model_api = "auto"
        else:
            model_api = cfg.bampi_model_api
        return self._build_model_from_spec(
            provider=provider,
            model_id=model_id,
            model_api=model_api,
            base_url=base_url,
        )

    def _build_model_from_spec(
        self,
        *,
        provider: str,
        model_id: str,
        model_api: str,
        base_url: str,
    ) -> Model:
        model = get_model(model_id, provider=provider)
        if model is None:
            model = self._build_custom_model(
                provider=provider,
                model_id=model_id,
                model_api=model_api,
                base_url=base_url,
            )
        return self._apply_model_overrides(
            model,
            model_api=model_api,
            base_url=base_url,
        )

    def _build_custom_model(
        self,
        *,
        provider: str | None = None,
        model_id: str | None = None,
        model_api: str | None = None,
        base_url: str | None = None,
    ) -> Model:
        resolved_provider = provider if provider is not None else self._config.bampi_model_provider
        resolved_model_id = model_id if model_id is not None else self._config.bampi_model_id
        resolved_model_api = model_api if model_api is not None else self._config.bampi_model_api
        resolved_base_url = base_url if base_url is not None else self._config.bampi_base_url
        if not resolved_provider or not resolved_model_id:
            raise RuntimeError("Custom model requires non-empty provider and model_id")

        api = self._resolve_model_api(resolved_provider, configured_api=resolved_model_api)
        logger.warning(
            f"bampi_chat using custom model "
            f"provider={resolved_provider} model={resolved_model_id} api={api}"
        )
        return Model(
            id=resolved_model_id,
            name=resolved_model_id,
            api=api,
            provider=resolved_provider,
            base_url=resolved_base_url,
            reasoning=False,
            input_types=list(self._config.bampi_model_input_types or ["text"]),
            context_window=128_000,
            max_tokens=16_384,
            cost=ModelCost(),
        )

    def _apply_model_overrides(
        self,
        model: Model,
        *,
        model_api: str | None = None,
        base_url: str | None = None,
    ) -> Model:
        updates: dict[str, Any] = {}
        api = self._config.bampi_model_api if model_api is None else model_api
        resolved_base_url = self._config.bampi_base_url if base_url is None else base_url
        if api != "auto" and api != model.api:
            logger.warning(
                f"bampi_chat overriding model api "
                f"provider={model.provider} model={model.id} from={model.api} to={api}"
            )
            updates["api"] = api
        if resolved_base_url:
            updates["base_url"] = resolved_base_url
        configured_input_types = self._config.bampi_model_input_types
        if configured_input_types is not None and configured_input_types != model.input_types:
            logger.warning(
                f"bampi_chat overriding model input types "
                f"provider={model.provider} model={model.id} "
                f"from={model.input_types} to={configured_input_types}"
            )
            updates["input_types"] = list(configured_input_types)
        if not updates:
            return model
        return model.model_copy(update=updates)

    def _resolve_model_api(
        self,
        provider: str,
        *,
        configured_api: str | None = None,
    ) -> str:
        api = self._config.bampi_model_api if configured_api is None else configured_api
        if api != "auto":
            return api

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

    async def _resolve_memory_api_key(
        self,
        provider: str,
        *,
        configured_api: str | None = None,
    ) -> str | None:
        if self._config.bampi_memory_api_key:
            logger.info(
                f"bampi_chat resolved memory api key provider={provider} source=memory_config"
            )
            return self._config.bampi_memory_api_key
        return await self._resolve_api_key(provider, configured_api=configured_api)

    async def _resolve_api_key(
        self,
        provider: str,
        *,
        configured_api: str | None = None,
    ) -> str | None:
        if self._config.bampi_api_key:
            logger.info(f"bampi_chat resolved api key provider={provider} source=config")
            return self._config.bampi_api_key

        env_keys = self._candidate_api_key_env_keys(
            provider,
            configured_api=configured_api,
        )
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

    def _candidate_api_key_env_keys(
        self,
        provider: str,
        *,
        configured_api: str | None = None,
    ) -> list[str]:
        candidates: list[str] = []

        normalized_provider = re.sub(
            r"[^A-Z0-9]+",
            "_",
            provider.strip().upper(),
        ).strip("_")
        if normalized_provider:
            candidates.append(f"{normalized_provider}_API_KEY")

        api = self._resolve_model_api(provider, configured_api=configured_api)
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
