from __future__ import annotations

import asyncio
import inspect
import os
import re
import signal
import tempfile
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import PydanticUndefined

from bampy.agent.cancellation import CancellationError, CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import TextContent
from bampy.app.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, serialize_truncation, truncate_tail

from ..config import BashMode


BashAction = Literal["run", "start", "status", "logs", "input", "stop", "list"]

_SESSION_BUFFER_BYTES = DEFAULT_MAX_BYTES * 2


class SafeBashToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: BashAction = Field(default="run", description="`run` for one-shot commands, or manage background sessions.")
    command: str | None = Field(default=None, description="Shell command to execute.")
    timeout: float | None = Field(default=None, gt=0, description="Timeout in seconds for one-shot commands.")
    session_id: str | None = Field(default=None, description="Background session id for status/logs/input/stop.")
    stdin: str | None = Field(default=None, description="Text to send to a background session stdin.")
    max_chars: int = Field(default=4_000, ge=200, le=40_000, description="Maximum characters returned for logs/status.")

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls_for_defaulted_non_nullable_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        for name, field in cls.model_fields.items():
            if normalized.get(name) is not None:
                continue
            if name not in normalized:
                continue
            if field.default in (None, PydanticUndefined):
                continue
            normalized.pop(name, None)
        return normalized

    @model_validator(mode="after")
    def _validate_action_requirements(self) -> "SafeBashToolInput":
        if self.action in {"run", "start"} and not self.command:
            raise ValueError(f"{self.action} requires command")
        if self.action in {"status", "logs", "input", "stop"} and not self.session_id:
            raise ValueError(f"{self.action} requires session_id")
        if self.action == "input" and self.stdin is None:
            raise ValueError("input requires stdin")
        return self


@dataclass(slots=True)
class _BackgroundShellSession:
    session_id: str
    command: str
    process: asyncio.subprocess.Process
    cwd_display: str
    started_at: float
    rolling_chunks: deque[bytes] = field(default_factory=deque)
    rolling_bytes: int = 0
    total_output_bytes: int = 0
    log_handle: Any | None = None
    log_path: str | None = None
    stdout_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None
    watch_task: asyncio.Task[None] | None = None


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()


async def _maybe_notify_update(
    callback: AgentToolUpdateCallback | None,
    result: AgentToolResult,
) -> None:
    if callback is None:
        return
    maybe = callback(result)
    if inspect.isawaitable(maybe):
        await maybe


def _docker_failure(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "no such container",
            "is not running",
            "cannot connect to the docker daemon",
            "permission denied while trying to connect",
        )
    )


def _trim_text(value: str, *, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[: max(0, limit - 3)].rstrip() + "...", True


class SafeBashTool:
    name = "bash"
    label = "bash"
    description = (
        "Execute shell commands in the configured workspace. "
        "Supports one-shot commands and managed background sessions for servers or watch tasks. "
        f"Foreground output is truncated to the last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB."
    )
    parameters = SafeBashToolInput

    def __init__(
        self,
        *,
        workspace_dir: str,
        mode: BashMode,
        container_name: str,
        container_workdir: str,
        visible_workspace_root: str | None,
        container_shell: str,
        default_timeout: float,
    ) -> None:
        self._workspace_dir = str(Path(workspace_dir).resolve())
        self._mode = mode
        self._container_name = container_name
        self._container_workdir = container_workdir
        self._visible_workspace_root = visible_workspace_root or container_workdir or "/workspace"
        self._container_shell = container_shell
        self._default_timeout = default_timeout
        self._session_lock = asyncio.Lock()
        self._sessions: dict[str, _BackgroundShellSession] = {}
        self._session_sequence = 1

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id
        arguments = SafeBashToolInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params)
        )

        if cancellation is not None:
            cancellation.raise_if_cancelled()

        if arguments.action == "run":
            return await self._execute_run(arguments, cancellation, on_update)
        if arguments.action == "start":
            return await self._start_background_session(arguments)
        if arguments.action == "status":
            return await self._session_status(arguments.session_id, max_chars=arguments.max_chars)
        if arguments.action == "logs":
            return await self._session_logs(arguments.session_id, max_chars=arguments.max_chars)
        if arguments.action == "input":
            return await self._session_input(arguments.session_id, arguments.stdin or "")
        if arguments.action == "stop":
            return await self._stop_background_session(arguments.session_id)
        if arguments.action == "list":
            return await self._list_background_sessions()

        raise RuntimeError(f"Unsupported bash action: {arguments.action}")

    async def close(self) -> None:
        async with self._session_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(*(self._terminate_session(session) for session in sessions), return_exceptions=True)

    async def _execute_run(
        self,
        arguments: SafeBashToolInput,
        cancellation: CancellationToken | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        if self._mode == "docker":
            return await self._run_docker_bash(arguments, cancellation, on_update)
        if self._mode == "local":
            return await self._run_bash(
                arguments,
                self._local_command(arguments.command or ""),
                self._workspace_dir,
                cancellation,
                on_update,
            )

        try:
            return await self._run_bash(
                arguments,
                self._docker_command(arguments.command or ""),
                None,
                cancellation,
                on_update,
            )
        except RuntimeError as exc:
            if not _docker_failure(str(exc)):
                raise
        except FileNotFoundError:
            pass

        return await self._run_bash(
            arguments,
            self._local_command(arguments.command or ""),
            self._workspace_dir,
            cancellation,
            on_update,
        )

    async def _run_docker_bash(
        self,
        arguments: SafeBashToolInput,
        cancellation: CancellationToken | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        try:
            return await self._run_bash(
                arguments,
                self._docker_command(arguments.command or ""),
                None,
                cancellation,
                on_update,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(self._docker_start_hint("Docker CLI not found on the host")) from exc
        except RuntimeError as exc:
            if _docker_failure(str(exc)):
                raise RuntimeError(self._docker_start_hint(str(exc))) from exc
            raise

    async def _start_background_session(self, arguments: SafeBashToolInput) -> AgentToolResult:
        command = arguments.command or ""
        session = await self._create_background_session(command)
        lines = [
            f"Started background bash session `{session.session_id}`.",
            f"Command: {session.command}",
            f"Working directory: {session.cwd_display}",
            f"Log path: {session.log_path}",
            "",
            "Use `bash` with `action=status`, `logs`, `input`, `stop`, or `list` to manage it.",
        ]
        return AgentToolResult(
            content=[TextContent(text="\n".join(lines))],
            details={
                "session_id": session.session_id,
                "full_output_path": session.log_path,
            },
        )

    async def _session_status(self, session_id: str | None, *, max_chars: int) -> AgentToolResult:
        session = await self._require_session(session_id)
        return AgentToolResult(
            content=[TextContent(text=self._format_session_summary(session, include_output=True, max_chars=max_chars))],
            details={
                "session_id": session.session_id,
                "full_output_path": session.log_path,
                "returncode": session.process.returncode,
            },
        )

    async def _session_logs(self, session_id: str | None, *, max_chars: int) -> AgentToolResult:
        session = await self._require_session(session_id)
        return AgentToolResult(
            content=[TextContent(text=self._format_session_logs(session, max_chars=max_chars))],
            details={
                "session_id": session.session_id,
                "full_output_path": session.log_path,
                "returncode": session.process.returncode,
            },
        )

    async def _session_input(self, session_id: str | None, text: str) -> AgentToolResult:
        session = await self._require_session(session_id)
        if session.process.returncode is not None:
            raise RuntimeError(
                f"Background session `{session.session_id}` has already exited with code {session.process.returncode}."
            )
        writer = session.process.stdin
        if writer is None:
            raise RuntimeError(f"Background session `{session.session_id}` does not accept stdin.")
        writer.write(text.encode("utf-8"))
        await writer.drain()
        return AgentToolResult(
            content=[TextContent(text=f"Sent {len(text)} characters to background session `{session.session_id}`.")],
            details={"session_id": session.session_id},
        )

    async def _stop_background_session(self, session_id: str | None) -> AgentToolResult:
        session = await self._require_session(session_id)
        if session.process.returncode is None:
            _kill_process_group(session.process)
        try:
            await self._await_session_exit(session)
        except asyncio.TimeoutError:
            if session.process.returncode is None:
                session.process.kill()
            await session.process.wait()
        lines = [
            f"Background session `{session.session_id}` stopped.",
            f"Command: {session.command}",
            f"Exit code: {session.process.returncode}",
        ]
        if session.log_path:
            lines.append(f"Log path: {session.log_path}")
        return AgentToolResult(
            content=[TextContent(text="\n".join(lines))],
            details={
                "session_id": session.session_id,
                "full_output_path": session.log_path,
                "returncode": session.process.returncode,
            },
        )

    async def _list_background_sessions(self) -> AgentToolResult:
        async with self._session_lock:
            sessions = list(self._sessions.values())

        if not sessions:
            return AgentToolResult(content=[TextContent(text="No background bash sessions.")])

        lines = ["Background bash sessions:"]
        for session in sessions:
            state = "running" if session.process.returncode is None else f"exited ({session.process.returncode})"
            lines.append(f"- {session.session_id}: {state}")
            lines.append(f"  Command: {session.command}")
        return AgentToolResult(content=[TextContent(text="\n".join(lines))])

    async def _create_background_session(self, command: str) -> _BackgroundShellSession:
        if self._mode == "docker":
            return await self._create_background_session_with_command(
                command,
                self._docker_command(command),
                self._visible_workspace_root,
            )
        if self._mode == "local":
            return await self._create_background_session_with_command(
                command,
                self._local_command(command),
                self._visible_workspace_root,
            )

        try:
            return await self._create_background_session_with_command(
                command,
                self._docker_command(command),
                self._visible_workspace_root,
            )
        except RuntimeError as exc:
            if not _docker_failure(str(exc)):
                raise
        except FileNotFoundError:
            pass

        return await self._create_background_session_with_command(
            command,
            self._local_command(command),
            self._visible_workspace_root,
        )

    async def _create_background_session_with_command(
        self,
        original_command: str,
        command: Sequence[str],
        cwd_display: str,
    ) -> _BackgroundShellSession:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=None if command[:2] == ["docker", "exec"] else self._workspace_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            if command[:1] == ["docker"]:
                raise RuntimeError(self._docker_start_hint("Docker CLI not found on the host")) from exc
            raise

        async with self._session_lock:
            session_id = f"term-{self._session_sequence}"
            self._session_sequence += 1
            log_handle = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix="bampi-bash-session-",
                suffix=".log",
                delete=False,
            )
            session = _BackgroundShellSession(
                session_id=session_id,
                command=original_command,
                process=process,
                cwd_display=cwd_display,
                started_at=time.monotonic(),
                log_handle=log_handle,
                log_path=log_handle.name,
            )
            session.stdout_task = asyncio.create_task(self._read_background_stream(session, process.stdout))
            session.stderr_task = asyncio.create_task(self._read_background_stream(session, process.stderr))
            session.watch_task = asyncio.create_task(self._watch_background_session(session))
            self._sessions[session_id] = session
        return session

    async def _watch_background_session(self, session: _BackgroundShellSession) -> None:
        try:
            await session.process.wait()
            await asyncio.gather(
                session.stdout_task or asyncio.sleep(0),
                session.stderr_task or asyncio.sleep(0),
                return_exceptions=True,
            )
        finally:
            if session.log_handle is not None:
                session.log_handle.flush()
                session.log_handle.close()
                session.log_handle = None

    async def _read_background_stream(
        self,
        session: _BackgroundShellSession,
        stream: asyncio.StreamReader | None,
    ) -> None:
        if stream is None:
            return

        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            session.total_output_bytes += len(chunk)
            if session.log_handle is not None:
                session.log_handle.write(chunk)
                session.log_handle.flush()
            session.rolling_chunks.append(chunk)
            session.rolling_bytes += len(chunk)
            while session.rolling_bytes > _SESSION_BUFFER_BYTES and session.rolling_chunks:
                removed = session.rolling_chunks.popleft()
                session.rolling_bytes -= len(removed)

    async def _require_session(self, session_id: str | None) -> _BackgroundShellSession:
        async with self._session_lock:
            session = self._sessions.get(session_id or "")
        if session is None:
            raise RuntimeError(f"Background session `{session_id}` was not found.")
        return session

    async def _await_session_exit(self, session: _BackgroundShellSession) -> None:
        if session.watch_task is None:
            await session.process.wait()
            return
        await asyncio.wait_for(session.watch_task, timeout=10.0)

    async def _terminate_session(self, session: _BackgroundShellSession) -> None:
        if session.process.returncode is None:
            _kill_process_group(session.process)
        try:
            await self._await_session_exit(session)
        except Exception:
            if session.process.returncode is None:
                session.process.kill()
                await session.process.wait()
            if session.log_handle is not None:
                session.log_handle.flush()
                session.log_handle.close()
                session.log_handle = None

    def _format_session_summary(
        self,
        session: _BackgroundShellSession,
        *,
        include_output: bool,
        max_chars: int,
    ) -> str:
        state = "running" if session.process.returncode is None else f"exited ({session.process.returncode})"
        lines = [
            f"Background session `{session.session_id}` is {state}.",
            f"Command: {session.command}",
            f"Working directory: {session.cwd_display}",
        ]
        if session.log_path:
            lines.append(f"Log path: {session.log_path}")
        if include_output:
            lines.append("")
            lines.append(self._render_session_output(session, max_chars=max_chars))
        return "\n".join(lines)

    def _format_session_logs(self, session: _BackgroundShellSession, *, max_chars: int) -> str:
        lines = [
            f"Logs for background session `{session.session_id}`:",
            f"Command: {session.command}",
        ]
        state = "running" if session.process.returncode is None else f"exited ({session.process.returncode})"
        lines.append(f"State: {state}")
        if session.log_path:
            lines.append(f"Log path: {session.log_path}")
        lines.append("")
        lines.append(self._render_session_output(session, max_chars=max_chars))
        return "\n".join(lines)

    def _render_session_output(self, session: _BackgroundShellSession, *, max_chars: int) -> str:
        text = self._sanitize_workspace_paths(
            b"".join(session.rolling_chunks).decode("utf-8", errors="replace")
        )
        truncation = truncate_tail(text)
        output = truncation.content or "(no output yet)"
        output, trimmed = _trim_text(output, limit=max_chars)

        notes: list[str] = []
        if truncation.truncated and session.log_path:
            notes.append(f"Tail only. Full log: {session.log_path}")
        if trimmed:
            notes.append(f"Displayed text trimmed to {max_chars} characters.")
        if notes:
            output += "\n\n[" + " | ".join(notes) + "]"
        return output

    def _docker_start_hint(self, detail: str) -> str:
        message = detail.strip() or "Unable to execute the bash command in the sandbox container."
        return (
            f"{message}\n\n"
            f"Ensure container `{self._container_name}` is running and exposing "
            f"`{self._visible_workspace_root}` as the workspace.\n"
            f"Suggested command: docker compose up -d {self._container_name}"
        )

    def _docker_command(self, command: str) -> list[str]:
        command = self._rewrite_visible_workspace_root(
            command,
            actual_root=self._container_workdir,
        )
        return [
            "docker",
            "exec",
            "-i",
            "-w",
            self._container_workdir,
            self._container_name,
            self._container_shell,
            "-lc",
            command,
        ]

    def _local_command(self, command: str) -> list[str]:
        command = self._rewrite_visible_workspace_root(
            command,
            actual_root=self._workspace_dir,
        )
        return [os.environ.get("SHELL") or "/bin/bash", "-lc", command]

    def _rewrite_visible_workspace_root(self, command: str, *, actual_root: str) -> str:
        visible_root = self._visible_workspace_root
        if not command or not visible_root or visible_root == actual_root:
            return command
        pattern = re.compile(rf"(?<![A-Za-z0-9._-]){re.escape(visible_root)}(?=(?:/|\b))")
        return pattern.sub(actual_root, command)

    def _sanitize_workspace_paths(self, text: str) -> str:
        visible_root = self._visible_workspace_root
        if not text or not visible_root:
            return text

        sanitized = text
        roots = sorted(
            {self._workspace_dir, self._container_workdir} - {visible_root, ""},
            key=len,
            reverse=True,
        )
        for root in roots:
            sanitized = sanitized.replace(root, visible_root)
        return sanitized

    async def _run_bash(
        self,
        arguments: SafeBashToolInput,
        command: Sequence[str],
        cwd: str | None,
        cancellation: CancellationToken | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
            start_new_session=True,
        )

        rolling_chunks: deque[bytes] = deque()
        rolling_bytes = 0
        max_rolling_bytes = DEFAULT_MAX_BYTES * 2
        initial_chunks: list[bytes] = []
        temp_handle = None
        temp_path: str | None = None
        total_bytes = 0

        def handle_chunk(data: bytes) -> None:
            nonlocal rolling_bytes, temp_handle, temp_path, total_bytes
            total_bytes += len(data)
            if temp_handle is None and total_bytes > DEFAULT_MAX_BYTES:
                temp_handle = tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix="bampi-bash-",
                    suffix=".log",
                    delete=False,
                )
                temp_path = temp_handle.name
                for chunk in initial_chunks:
                    temp_handle.write(chunk)

            if temp_handle is None:
                initial_chunks.append(data)
            else:
                temp_handle.write(data)

            rolling_chunks.append(data)
            rolling_bytes += len(data)
            while rolling_bytes > max_rolling_bytes and rolling_chunks:
                removed = rolling_chunks.popleft()
                rolling_bytes -= len(removed)

        async def read_stream(stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    return
                handle_chunk(chunk)
                truncation = truncate_tail(b"".join(rolling_chunks).decode("utf-8", errors="replace"))
                truncated_text = self._sanitize_workspace_paths(truncation.content or "")
                await _maybe_notify_update(
                    on_update,
                    AgentToolResult(
                        content=[TextContent(text=truncated_text)],
                        details={
                            "truncation": serialize_truncation(truncation) if truncation.truncated else None,
                            "full_output_path": temp_path,
                        },
                    ),
                )

        remove_cancel = None
        if cancellation is not None:
            remove_cancel = cancellation.add_callback(lambda _reason: _kill_process_group(process))

        stdout_task = asyncio.create_task(read_stream(process.stdout))
        stderr_task = asyncio.create_task(read_stream(process.stderr))
        timeout = arguments.timeout if arguments.timeout is not None else self._default_timeout

        try:
            if timeout and timeout > 0:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            else:
                await process.wait()
            await asyncio.gather(stdout_task, stderr_task)
        except asyncio.TimeoutError as exc:
            _kill_process_group(process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            text = self._sanitize_workspace_paths(
                b"".join(rolling_chunks).decode("utf-8", errors="replace")
            )
            if text:
                text += "\n\n"
            raise RuntimeError(f"{text}Command timed out after {timeout} seconds") from exc
        finally:
            if remove_cancel is not None:
                remove_cancel()
            if temp_handle is not None:
                temp_handle.flush()
                temp_handle.close()

        if cancellation is not None and cancellation.cancelled:
            raise CancellationError(cancellation.reason or "Command aborted")

        full_text = self._sanitize_workspace_paths(
            b"".join(rolling_chunks).decode("utf-8", errors="replace")
        )
        truncation = truncate_tail(full_text)
        output = truncation.content or "(no output)"
        details: dict[str, object] | None = None

        if truncation.truncated:
            details = {
                "truncation": serialize_truncation(truncation),
                "full_output_path": temp_path,
            }
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                output += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line}. "
                    f"Full output: {temp_path}]"
                )
            elif truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {temp_path}]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {temp_path}]"
                )

        if process.returncode not in (0, None):
            if output:
                output += "\n\n"
            output += f"Command exited with code {process.returncode}"
            raise RuntimeError(output)

        return AgentToolResult(content=[TextContent(text=output)], details=details)
