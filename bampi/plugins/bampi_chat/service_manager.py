from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from nonebot import logger

from .config import BampiChatConfig

ServiceStatus = Literal["starting", "running", "stopped", "exited", "failed", "unknown"]
ServiceProtocol = Literal["tcp"]

_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_LOG_TAIL_BYTES = 128 * 1024
_REGISTRY_VERSION = 1


@dataclass(slots=True)
class ManagedServiceRecord:
    service_id: str
    group_id: str
    name: str
    command: str
    port: int
    protocol: ServiceProtocol
    status: ServiceStatus
    workdir: str
    pid: int | None
    created_at: str
    updated_at: str
    log_path: str
    pid_file: str
    exit_code_file: str
    public_host: str = ""
    started_at: str | None = None
    stopped_at: str | None = None
    exit_code: int | None = None
    startup_error: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ManagedServiceRecord":
        payload = dict(value)
        payload.setdefault("protocol", "tcp")
        payload.setdefault("public_host", "")
        payload.setdefault("started_at", None)
        payload.setdefault("stopped_at", None)
        payload.setdefault("exit_code", None)
        payload.setdefault("startup_error", None)
        payload.setdefault("pid", None)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def address(self) -> str:
        host = (self.public_host or "").strip() or "<configure-public-host>"
        return f"{host}:{self.port}"

    @property
    def is_active(self) -> bool:
        return self.status in {"starting", "running"}


@dataclass(slots=True)
class ServiceStartResult:
    record: ManagedServiceRecord
    ready: bool
    startup_log_excerpt: str


def parse_service_port_range(value: str | None) -> list[int]:
    text = (value or "").strip()
    if not text:
        return []

    ports: list[int] = []
    seen: set[int] = set()
    for chunk in text.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = [segment.strip() for segment in part.split("-", 1)]
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"invalid port range: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for port in values:
            if port < 1 or port > 65535:
                raise ValueError(f"port out of range: {port}")
            if port in seen:
                continue
            seen.add(port)
            ports.append(port)
    return ports


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _trim_text(value: str, *, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[: max(0, limit - 3)].rstrip() + "...", True


def _quote_env_value(value: str) -> str:
    return shlex.quote(value)


def _group_runtime_name(group_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", (group_id or "").strip()).strip("._-")
    return f"group-{sanitized or 'default'}"


class ServiceManager:
    def __init__(
        self,
        *,
        workspace_root: str,
        visible_container_root: str,
        container_name: str,
        container_shell: str,
        port_range: str,
        public_host: str,
        startup_timeout: float,
        stop_timeout: float,
        max_active_services_per_group: int,
    ) -> None:
        self._workspace_root = Path(workspace_root).resolve()
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        self._visible_container_root = PurePosixPath(visible_container_root).as_posix()
        self._container_name = container_name.strip()
        self._container_shell = container_shell.strip()
        self._public_host = public_host.strip()
        self._startup_timeout = max(0.0, startup_timeout)
        self._stop_timeout = max(1.0, stop_timeout)
        self._max_active_services_per_group = max(0, max_active_services_per_group)
        self._port_pool = parse_service_port_range(port_range)
        if not self._port_pool:
            raise ValueError("service port range must not be empty")

        self._runtime_root = (self._workspace_root / ".bampi-services").resolve()
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._registry_path = (self._runtime_root / "registry.json").resolve()
        self._lock = asyncio.Lock()

        self._services: dict[str, ManagedServiceRecord] = {}
        self._next_service_sequence = 1
        self._load_registry()

    @classmethod
    def from_config(cls, config: BampiChatConfig) -> "ServiceManager":
        return cls(
            workspace_root=config.bampi_workspace_dir,
            visible_container_root=config.bampi_bash_container_workdir,
            container_name=config.bampi_bash_container_name,
            container_shell=config.bampi_bash_container_shell,
            port_range=config.bampi_service_port_range,
            public_host=config.bampi_service_public_host,
            startup_timeout=config.bampi_service_startup_timeout,
            stop_timeout=config.bampi_service_stop_timeout,
            max_active_services_per_group=config.bampi_service_max_active_services_per_group,
        )

    @property
    def port_pool(self) -> list[int]:
        return list(self._port_pool)

    @property
    def public_host(self) -> str:
        return self._public_host

    async def list_services(
        self,
        *,
        group_id: str,
        include_stopped: bool = False,
    ) -> list[ManagedServiceRecord]:
        async with self._lock:
            records = sorted(
                (service for service in self._services.values() if service.group_id == group_id),
                key=lambda service: (service.created_at, service.service_id),
            )
            for service in records:
                await self._refresh_service_state_locked(service)
            if not include_stopped:
                records = [service for service in records if service.is_active]
            await self._save_registry_locked()
            return [ManagedServiceRecord.from_dict(service.to_dict()) for service in records]

    async def get_service(
        self,
        *,
        group_id: str,
        service_ref: str,
    ) -> ManagedServiceRecord:
        async with self._lock:
            service = self._resolve_service_ref_locked(group_id, service_ref)
            await self._refresh_service_state_locked(service)
            await self._save_registry_locked()
            return ManagedServiceRecord.from_dict(service.to_dict())

    async def stop_service(
        self,
        *,
        group_id: str,
        service_ref: str,
    ) -> ManagedServiceRecord:
        async with self._lock:
            service = self._resolve_service_ref_locked(group_id, service_ref)
            await self._refresh_service_state_locked(service)
            if service.pid is not None and service.is_active:
                await self._terminate_service_process(service.pid)
                await self._wait_for_service_exit(service.pid, timeout=self._stop_timeout)
            service.status = "stopped"
            service.updated_at = _now_iso()
            service.stopped_at = service.updated_at
            service.exit_code = self._read_optional_int(Path(service.exit_code_file))
            service.pid = self._read_optional_int(Path(service.pid_file))
            await self._save_registry_locked()
            return ManagedServiceRecord.from_dict(service.to_dict())

    async def read_logs(
        self,
        *,
        group_id: str,
        service_ref: str,
        max_chars: int,
    ) -> ManagedServiceRecord:
        async with self._lock:
            service = self._resolve_service_ref_locked(group_id, service_ref)
            await self._refresh_service_state_locked(service)
            await self._save_registry_locked()
            return ManagedServiceRecord.from_dict(service.to_dict())

    def render_service_summary(
        self,
        service: ManagedServiceRecord,
        *,
        include_recent_logs: bool = False,
        max_chars: int = 4_000,
    ) -> str:
        lines = [
            f"Service: {service.service_id}",
            f"Name: {service.name}",
            f"Status: {service.status}",
            f"Address: {service.address}",
            f"Port: {service.port}/{service.protocol}",
            f"Working directory: {service.workdir}",
            f"Command: {service.command}",
            f"Created at: {service.created_at}",
        ]
        if service.started_at:
            lines.append(f"Started at: {service.started_at}")
        if service.stopped_at:
            lines.append(f"Stopped at: {service.stopped_at}")
        if service.pid is not None:
            lines.append(f"PID: {service.pid}")
        if service.exit_code is not None:
            lines.append(f"Exit code: {service.exit_code}")
        if service.startup_error:
            lines.append(f"Startup error: {service.startup_error}")

        if include_recent_logs:
            lines.append("")
            lines.append("Recent logs:")
            lines.append(self.read_log_text(service, max_chars=max_chars) or "(no logs yet)")
        return "\n".join(lines)

    def read_log_text(self, service: ManagedServiceRecord, *, max_chars: int) -> str:
        return self._read_log_tail(Path(service.log_path), max_chars=max_chars)

    async def start_service(
        self,
        *,
        group_id: str,
        workspace_dir: str,
        visible_workspace_root: str,
        actual_container_workdir: str,
        command: str,
        name: str | None,
        cwd: str | None,
        preferred_port: int | None,
        replace_existing: bool,
        env: dict[str, str],
        startup_timeout: float | None,
    ) -> ServiceStartResult:
        normalized_name = (name or "").strip()
        if not command.strip():
            raise RuntimeError("service command must not be empty")

        workspace_root = Path(workspace_dir).resolve()
        if not workspace_root.exists():
            raise RuntimeError(f"workspace does not exist: {workspace_root}")

        async with self._lock:
            await self._refresh_active_services_locked()
            active_services = [
                service
                for service in self._services.values()
                if service.group_id == group_id and service.is_active
            ]
            if (
                self._max_active_services_per_group > 0
                and len(active_services) >= self._max_active_services_per_group
            ):
                raise RuntimeError(
                    f"this group already has {len(active_services)} active services, "
                    f"which reaches the configured limit of {self._max_active_services_per_group}"
                )

            if normalized_name:
                duplicates = [
                    service
                    for service in self._services.values()
                    if service.group_id == group_id and service.name == normalized_name
                ]
                if duplicates and not replace_existing:
                    active_duplicates = [service for service in duplicates if service.is_active]
                    if active_duplicates:
                        raise RuntimeError(
                            f"a service named `{normalized_name}` is already running in this group"
                        )
                for duplicate in duplicates:
                    if duplicate.is_active:
                        await self._terminate_service_process(duplicate.pid)
                        await self._wait_for_service_exit(duplicate.pid, timeout=self._stop_timeout)
                    self._services.pop(duplicate.service_id, None)

            port = self._allocate_port_locked(preferred_port=preferred_port)
            service_id = f"svc-{self._next_service_sequence}"
            self._next_service_sequence += 1
            service_name = normalized_name or service_id
            runtime_dir = self._runtime_root / _group_runtime_name(group_id) / service_id
            runtime_dir.mkdir(parents=True, exist_ok=True)

            log_path = (runtime_dir / "service.log").resolve()
            pid_file = (runtime_dir / "service.pid").resolve()
            exit_code_file = (runtime_dir / "exit_code").resolve()

            for path in (log_path, pid_file, exit_code_file):
                path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")
            pid_file.unlink(missing_ok=True)
            exit_code_file.unlink(missing_ok=True)

            visible_cwd, actual_cwd = self._resolve_workdir(
                workspace_dir=str(workspace_root),
                visible_workspace_root=visible_workspace_root,
                actual_container_workdir=actual_container_workdir,
                cwd=cwd,
            )

            now = _now_iso()
            record = ManagedServiceRecord(
                service_id=service_id,
                group_id=group_id,
                name=service_name,
                command=command.strip(),
                port=port,
                protocol="tcp",
                status="starting",
                workdir=visible_cwd,
                pid=None,
                created_at=now,
                updated_at=now,
                log_path=str(log_path),
                pid_file=str(pid_file),
                exit_code_file=str(exit_code_file),
                public_host=self._public_host,
                started_at=now,
            )
            self._services[service_id] = record
            await self._save_registry_locked()

            try:
                await self._launch_service_process(
                    actual_cwd=actual_cwd,
                    runtime_dir=runtime_dir,
                    command=record.command,
                    port=record.port,
                    env=env,
                )
                record.pid = await self._wait_for_pid_file(pid_file)
                effective_timeout = self._startup_timeout if startup_timeout is None else max(0.0, startup_timeout)
                ready = await self._wait_for_ready(record, timeout=effective_timeout)
                record.status = "running" if ready else "starting"
                record.exit_code = self._read_optional_int(exit_code_file)
                record.startup_error = None
                record.updated_at = _now_iso()
                await self._save_registry_locked()
                startup_log = self._read_log_tail(log_path, max_chars=2_000)
                return ServiceStartResult(
                    record=ManagedServiceRecord.from_dict(record.to_dict()),
                    ready=ready,
                    startup_log_excerpt=startup_log,
                )
            except Exception as exc:
                await self._refresh_service_state_locked(record)
                record.status = "failed"
                record.startup_error = str(exc).strip() or "service start failed"
                record.updated_at = _now_iso()
                await self._save_registry_locked()
                raise RuntimeError(
                    f"{record.startup_error}\n\n"
                    f"Recent logs:\n{self._read_log_tail(log_path, max_chars=2_000) or '(no logs yet)'}"
                ) from exc

    async def _refresh_active_services_locked(self) -> None:
        for service in self._services.values():
            if service.status in {"starting", "running", "unknown"}:
                await self._refresh_service_state_locked(service)
        await self._save_registry_locked()

    async def _refresh_service_state_locked(self, service: ManagedServiceRecord) -> None:
        pid = self._read_optional_int(Path(service.pid_file))
        exit_code = self._read_optional_int(Path(service.exit_code_file))
        is_running = False
        if pid is not None:
            is_running = await self._is_pid_running(pid)

        service.pid = pid
        service.exit_code = exit_code
        service.updated_at = _now_iso()
        if is_running:
            if service.status != "running":
                service.status = "running"
            return

        if service.status == "stopped":
            return
        if exit_code is not None:
            service.status = "exited" if exit_code == 0 else "failed"
            service.stopped_at = service.updated_at
            return
        if service.started_at is not None:
            service.status = "unknown"

    def _resolve_service_ref_locked(
        self,
        group_id: str,
        service_ref: str,
    ) -> ManagedServiceRecord:
        ref = (service_ref or "").strip()
        if not ref:
            raise RuntimeError("service reference must not be empty")

        service = self._services.get(ref)
        if service is not None and service.group_id == group_id:
            return service

        matches = [
            candidate
            for candidate in self._services.values()
            if candidate.group_id == group_id and candidate.name == ref
        ]
        if not matches:
            raise RuntimeError(f"service `{ref}` was not found in this group")
        matches.sort(key=lambda candidate: (candidate.created_at, candidate.service_id), reverse=True)
        return matches[0]

    def _allocate_port_locked(self, *, preferred_port: int | None) -> int:
        in_use = {
            service.port
            for service in self._services.values()
            if service.is_active
        }
        if preferred_port is not None:
            if preferred_port not in self._port_pool:
                raise RuntimeError(
                    f"preferred port {preferred_port} is outside the configured service pool"
                )
            if preferred_port in in_use:
                raise RuntimeError(f"preferred port {preferred_port} is already in use")
            return preferred_port

        for port in self._port_pool:
            if port not in in_use:
                return port
        raise RuntimeError("no free service ports remain in the configured pool")

    def _resolve_workdir(
        self,
        *,
        workspace_dir: str,
        visible_workspace_root: str,
        actual_container_workdir: str,
        cwd: str | None,
    ) -> tuple[str, str]:
        host_workspace = Path(workspace_dir).resolve()
        requested = (cwd or "").strip()
        if not requested or requested in {".", visible_workspace_root}:
            return visible_workspace_root, actual_container_workdir

        text = requested
        try:
            relative_to_visible = PurePosixPath(text).relative_to(PurePosixPath(visible_workspace_root))
        except ValueError:
            candidate = (host_workspace / text).resolve()
        else:
            candidate = (host_workspace / Path(relative_to_visible.as_posix())).resolve()
        try:
            relative = candidate.relative_to(host_workspace)
        except ValueError as exc:
            raise RuntimeError(f"service cwd escapes the workspace: {cwd}") from exc

        visible_path = PurePosixPath(visible_workspace_root) / PurePosixPath(relative.as_posix())
        actual_path = PurePosixPath(actual_container_workdir) / PurePosixPath(relative.as_posix())
        return visible_path.as_posix(), actual_path.as_posix()

    async def _launch_service_process(
        self,
        *,
        actual_cwd: str,
        runtime_dir: Path,
        command: str,
        port: int,
        env: dict[str, str],
    ) -> None:
        container_runtime_dir = (
            PurePosixPath(self._visible_container_root)
            / ".bampi-services"
            / PurePosixPath(runtime_dir.relative_to(self._runtime_root).as_posix())
        ).as_posix()
        log_file = (PurePosixPath(container_runtime_dir) / "service.log").as_posix()
        pid_file = (PurePosixPath(container_runtime_dir) / "service.pid").as_posix()
        exit_code_file = (PurePosixPath(container_runtime_dir) / "exit_code").as_posix()

        env_exports = {
            "PORT": str(port),
            "SERVICE_PORT": str(port),
            "BAMPI_SERVICE_PORT": str(port),
            "HOST": "0.0.0.0",
            "LISTEN_HOST": "0.0.0.0",
            "BAMPI_SERVICE_HOST": "0.0.0.0",
            "BAMPI_PUBLIC_HOST": self._public_host,
            "BAMPI_PUBLIC_PORT": str(port),
        }
        for key, value in env.items():
            normalized_key = key.strip()
            if not _ENV_KEY_PATTERN.match(normalized_key):
                raise RuntimeError(f"invalid environment variable name: {key!r}")
            env_exports[normalized_key] = str(value)

        export_lines = " ".join(
            f"export {key}={_quote_env_value(value)};"
            for key, value in sorted(env_exports.items())
        )
        launch_script = " ".join(
            [
                f"mkdir -p {shlex.quote(container_runtime_dir)};",
                f"cd {shlex.quote(actual_cwd)};",
                f"rm -f {shlex.quote(pid_file)} {shlex.quote(exit_code_file)};",
                f": > {shlex.quote(log_file)};",
                f"printf '%s' $$ > {shlex.quote(pid_file)};",
                export_lines,
                f"{command} >> {shlex.quote(log_file)} 2>&1;",
                'status=$?;',
                f"printf '%s' \"$status\" > {shlex.quote(exit_code_file)};",
                'exit "$status";',
            ]
        )
        await self._run_command(
            [
                "docker",
                "exec",
                "-d",
                "-w",
                actual_cwd,
                self._container_name,
                self._container_shell,
                "-lc",
                launch_script,
            ],
            check=True,
        )

    async def _wait_for_pid_file(self, pid_file: Path) -> int:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            pid = self._read_optional_int(pid_file)
            if pid is not None and pid > 0:
                return pid
            await asyncio.sleep(0.1)
        raise RuntimeError("service pid file was not written in time")

    async def _wait_for_ready(self, service: ManagedServiceRecord, *, timeout: float) -> bool:
        if timeout <= 0:
            return await self._is_pid_running(service.pid)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await self._refresh_service_state_locked(service)
            if service.status in {"failed", "exited"}:
                raise RuntimeError(f"service exited during startup with code {service.exit_code}")
            if await self._is_host_port_ready(service.port):
                return True
            await asyncio.sleep(0.25)
        raise RuntimeError(
            f"service did not become reachable on port {service.port} within {timeout:.1f}s"
        )

    async def _wait_for_service_exit(self, pid: int | None, *, timeout: float) -> None:
        if pid is None:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not await self._is_pid_running(pid):
                return
            await asyncio.sleep(0.2)

        await self._run_container_shell(
            f"kill -KILL -- -{pid} >/dev/null 2>&1 || kill -KILL -- {pid} >/dev/null 2>&1 || true",
            check=False,
        )

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not await self._is_pid_running(pid):
                return
            await asyncio.sleep(0.2)
        raise RuntimeError(f"service pid {pid} did not stop in time")

    async def _terminate_service_process(self, pid: int | None) -> None:
        if pid is None:
            return
        await self._run_container_shell(
            f"kill -TERM -- -{pid} >/dev/null 2>&1 || kill -TERM -- {pid} >/dev/null 2>&1 || true",
            check=False,
        )

    async def _is_pid_running(self, pid: int | None) -> bool:
        if pid is None or pid <= 0:
            return False
        try:
            await self._run_container_shell(
                f"kill -0 -- {pid} >/dev/null 2>&1",
                check=True,
            )
            return True
        except RuntimeError:
            return False

    async def _is_host_port_ready(self, port: int) -> bool:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port),
                timeout=0.5,
            )
        except Exception:
            return False

        writer.close()
        wait_closed = getattr(writer, "wait_closed", None)
        if callable(wait_closed):
            try:
                await writer.wait_closed()
            except Exception:
                pass
        del reader
        return True

    async def _run_container_shell(self, script: str, *, check: bool) -> str:
        return await self._run_command(
            [
                "docker",
                "exec",
                self._container_name,
                self._container_shell,
                "-lc",
                script,
            ],
            check=check,
        )

    async def _run_command(self, command: list[str], *, check: bool) -> str:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=os.environ.copy(),
        )
        stdout, _ = await process.communicate()
        output = stdout.decode("utf-8", errors="replace").strip()
        if check and process.returncode != 0:
            raise RuntimeError(output or f"command failed with exit code {process.returncode}")
        return output

    def _read_log_tail(self, path: Path, *, max_chars: int) -> str:
        if not path.exists():
            return ""

        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _DEFAULT_LOG_TAIL_BYTES), os.SEEK_SET)
            data = handle.read()
        text = data.decode("utf-8", errors="replace")
        trimmed, truncated = _trim_text(text, limit=max_chars)
        if truncated:
            return f"{trimmed}\n\n[Showing the most recent {max_chars} characters.]"
        return trimmed

    @staticmethod
    def _read_optional_int(path: Path) -> int | None:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _load_registry(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception(
                f"bampi_chat failed to read service registry path={self._registry_path}"
            )
            return

        if payload.get("version") != _REGISTRY_VERSION:
            logger.warning(
                f"bampi_chat ignoring unsupported service registry version "
                f"path={self._registry_path} version={payload.get('version')}"
            )
            return

        self._next_service_sequence = int(payload.get("next_service_sequence", 1) or 1)
        self._services = {}
        for raw_service in payload.get("services", []):
            try:
                record = ManagedServiceRecord.from_dict(dict(raw_service))
            except Exception:
                logger.exception(
                    f"bampi_chat failed to load service registry record path={self._registry_path}"
                )
                continue
            self._services[record.service_id] = record

    async def _save_registry_locked(self) -> None:
        payload = {
            "version": _REGISTRY_VERSION,
            "next_service_sequence": self._next_service_sequence,
            "services": [
                service.to_dict()
                for service in sorted(
                    self._services.values(),
                    key=lambda record: (record.created_at, record.service_id),
                )
            ],
        }
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._registry_path.parent,
            prefix="registry-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(self._registry_path)
