from __future__ import annotations

import asyncio
import inspect
import os
import signal
import tempfile
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bampy.agent.cancellation import CancellationError, CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import TextContent
from bampy.app.tools import BashToolInput
from bampy.app.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, serialize_truncation, truncate_tail

from ..config import BashMode


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


class SafeBashTool:
    name = "bash"
    label = "bash"
    description = (
        "Execute a shell command in the configured workspace. "
        f"Output is truncated to the last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB."
    )
    parameters = BashToolInput

    def __init__(
        self,
        *,
        workspace_dir: str,
        mode: BashMode,
        container_name: str,
        container_workdir: str,
        container_shell: str,
        default_timeout: float,
    ) -> None:
        self._workspace_dir = str(Path(workspace_dir).resolve())
        self._mode = mode
        self._container_name = container_name
        self._container_workdir = container_workdir
        self._container_shell = container_shell
        self._default_timeout = default_timeout

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id
        arguments = BashToolInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params)
        )
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        if self._mode == "docker":
            return await self._run_docker_bash(arguments, cancellation, on_update)
        if self._mode == "local":
            return await self._run_bash(arguments, self._local_command(arguments.command), self._workspace_dir, cancellation, on_update)

        try:
            return await self._run_bash(arguments, self._docker_command(arguments.command), None, cancellation, on_update)
        except RuntimeError as exc:
            if not _docker_failure(str(exc)):
                raise
        except FileNotFoundError:
            pass

        return await self._run_bash(arguments, self._local_command(arguments.command), self._workspace_dir, cancellation, on_update)

    async def _run_docker_bash(
        self,
        arguments: BashToolInput,
        cancellation: CancellationToken | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        try:
            return await self._run_bash(arguments, self._docker_command(arguments.command), None, cancellation, on_update)
        except FileNotFoundError as exc:
            raise RuntimeError(self._docker_start_hint("Docker CLI not found on the host")) from exc
        except RuntimeError as exc:
            if _docker_failure(str(exc)):
                raise RuntimeError(self._docker_start_hint(str(exc))) from exc
            raise

    def _docker_start_hint(self, detail: str) -> str:
        message = detail.strip() or "Unable to execute the bash command in the sandbox container."
        return (
            f"{message}\n\n"
            f"Ensure container `{self._container_name}` is running and exposing "
            f"`{self._container_workdir}` as the workspace.\n"
            f"Suggested command: docker compose up -d {self._container_name}"
        )

    def _docker_command(self, command: str) -> list[str]:
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
        return [os.environ.get("SHELL") or "/bin/bash", "-lc", command]

    async def _run_bash(
        self,
        arguments: BashToolInput,
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
                await _maybe_notify_update(
                    on_update,
                    AgentToolResult(
                        content=[TextContent(text=truncation.content or "")],
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
            text = b"".join(rolling_chunks).decode("utf-8", errors="replace")
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

        full_text = b"".join(rolling_chunks).decode("utf-8", errors="replace")
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
