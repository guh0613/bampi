from __future__ import annotations

import asyncio
import inspect
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nonebot import get_driver, logger

from bampy.ai import Model, ModelCost, SimpleStreamOptions, get_model
from bampy.app import AgentSession, SessionManager

from .config import BampiChatConfig
from .prompt import build_system_prompt
from .skills import build_prompt_skills, load_chat_skills
from .tools import create_agent_tools
from .tools.workspace import (
    reset_workspace_files,
    resolve_group_container_workspace,
    resolve_group_workspace_dir,
)


@dataclass(slots=True)
class ManagedGroupSession:
    group_id: str
    session: AgentSession
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used_at: float = field(default_factory=time.monotonic)
    idle_reset_task: asyncio.Task[None] | None = None
    active_user_id: str | None = None


@dataclass(slots=True)
class GroupInteractionStatus:
    managed: ManagedGroupSession | None = None
    active_user_id: str | None = None
    is_streaming: bool = False

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
        logger.info(
            f"bampi_chat session manager initialized "
            f"workspace_root_dir={self._workspace_root_dir} "
            f"session_dir={self._session_dir} "
            f"bash_mode={config.bampi_bash_mode} "
            f"bash_container={config.bampi_bash_container_name} "
            f"bash_workdir={config.bampi_bash_container_workdir} "
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
            return GroupInteractionStatus(
                managed=managed,
                active_user_id=managed.active_user_id,
                is_streaming=managed.session.is_processing,
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
            self._schedule_idle_reset_locked(managed)

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
                if now - managed.last_used_at >= ttl:
                    stale_ids.append(group_id)

            stale_sessions = [self._sessions.pop(group_id) for group_id in stale_ids]

        for managed in stale_sessions:
            await self._dispose_session(managed, reason="idle_timeout")

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
        workspace_dir = self.workspace_dir_for_group(group_id)
        container_workspace_dir = self.container_workspace_dir_for_group(group_id)
        model_workspace_root = self._config.bampi_bash_container_workdir
        model = self._build_model()
        tools = create_agent_tools(
            self._config,
            workspace_dir,
            container_root=model_workspace_root,
            bash_workdir=container_workspace_dir,
        )
        tool_names = [tool.name for tool in tools]
        loaded_skills = load_chat_skills(workspace_dir)
        system_prompt = build_system_prompt(
            self._config,
            tool_names,
            skills=build_prompt_skills(loaded_skills.skills, workspace_dir=workspace_dir),
            prompt_cwd=model_workspace_root,
        )
        stream_options = SimpleStreamOptions(api_key=self._config.bampi_api_key or None)

        session_file = str((self._session_dir / f"group-{group_id}.jsonl").resolve())
        session_manager = SessionManager(
            workspace_dir,
            session_file=session_file,
            persist=True,
        )

        logger.info(
            f"bampi_chat building session group_id={group_id} "
            f"workspace_dir={workspace_dir} "
            f"container_workspace_dir={container_workspace_dir} "
            f"provider={model.provider} "
            f"model={model.id} "
            f"session_file={session_file} "
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
                self._sessions.pop(group_id, None)

            async with managed.lock:
                await self._dispose_session(managed, reason="idle_timeout")
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
    ) -> None:
        self._cancel_idle_reset_task(managed)
        managed.active_user_id = None
        session_file = managed.session.session_manager.session_file
        logger.info(
            f"bampi_chat disposing session group_id={managed.group_id} "
            f"reason={reason} "
            f"clear_history={clear_history}"
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
        if clear_history:
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

    def _build_model(self) -> Model:
        model = get_model(self._config.bampi_model_id, provider=self._config.bampi_model_provider)
        if model is None:
            model = self._build_custom_model()
        if self._config.bampi_base_url:
            model = model.model_copy(update={"base_url": self._config.bampi_base_url})
        return model

    def _build_custom_model(self) -> Model:
        provider = self._config.bampi_model_provider
        model_id = self._config.bampi_model_id

        if provider in {"openai", "ollama"}:
            api = "openai-responses" if provider == "openai" else "ollama-responses"
            logger.warning(
                f"bampi_chat using custom {provider}-compatible model "
                f"provider={provider} model={model_id}"
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

        raise RuntimeError(
            f"Unsupported model: provider={provider}, model={model_id}"
        )

    async def _resolve_api_key(self, provider: str) -> str | None:
        if self._config.bampi_api_key:
            logger.info(f"bampi_chat resolved api key provider={provider} source=config")
            return self._config.bampi_api_key

        env_key = {
            "openai": "OPENAI_API_KEY",
            "ollama": "OLLAMA_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }.get(provider)
        if env_key is None:
            logger.warning(f"bampi_chat no api key mapping for provider={provider}")
            return None
        config_value = self._resolve_nonebot_config_value(env_key.lower())
        if config_value is not None:
            logger.info(
                f"bampi_chat resolved api key provider={provider} "
                f"source=nonebot_config key={env_key.lower()}"
            )
            return config_value
        env_value = os.environ.get(env_key, "") or None
        if env_value is None:
            logger.warning(f"bampi_chat api key missing provider={provider} env={env_key}")
            return None
        logger.info(f"bampi_chat resolved api key provider={provider} source=env env={env_key}")
        return env_value

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
