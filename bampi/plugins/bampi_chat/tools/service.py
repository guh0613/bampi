from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import PydanticUndefined

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import TextContent

from ..service_manager import ManagedServiceRecord, ServiceManager

ServiceAction = Literal["start", "list", "status", "logs", "stop"]


class ServiceToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ServiceAction = Field(description="Service management action to run.")
    service: str | None = Field(default=None, description="Service id or exact service name for status/logs/stop.")
    name: str | None = Field(default=None, description="Optional stable name for a new service.")
    command: str | None = Field(default=None, description="Foreground command used to start the service.")
    cwd: str | None = Field(default=None, description="Optional working directory inside the current workspace.")
    preferred_port: int | None = Field(default=None, ge=1, le=65535, description="Preferred port from the managed pool.")
    replace_existing: bool = Field(
        default=False,
        description="Whether start should replace an existing service with the same name in this group.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Optional environment variables added when the service starts.",
    )
    include_stopped: bool = Field(
        default=False,
        description="Whether list should also include stopped or exited services for this group.",
    )
    startup_timeout: float | None = Field(
        default=None,
        ge=0,
        le=300,
        description="Seconds to wait for the service port to become reachable. Use 0 to skip the reachability wait.",
    )
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
    def _validate_action_requirements(self) -> "ServiceToolInput":
        if self.action == "start" and not self.command:
            raise ValueError("start requires command")
        if self.action in {"status", "logs", "stop"} and not self.service:
            raise ValueError(f"{self.action} requires service")
        return self


class ServiceTool:
    name = "service"
    label = "service"
    description = (
        "Start and manage long-lived TCP services in the docker sandbox with a persisted port pool. "
        "Use it for externally reachable servers instead of raw background bash sessions."
    )
    parameters = ServiceToolInput

    def __init__(
        self,
        *,
        manager: ServiceManager,
        group_id: str,
        workspace_dir: str,
        visible_workspace_root: str,
        actual_container_workdir: str,
    ) -> None:
        self._manager = manager
        self._group_id = group_id
        self._workspace_dir = workspace_dir
        self._visible_workspace_root = visible_workspace_root
        self._actual_container_workdir = actual_container_workdir

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        arguments = ServiceToolInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
        )

        if arguments.action == "start":
            started = await self._manager.start_service(
                group_id=self._group_id,
                workspace_dir=self._workspace_dir,
                visible_workspace_root=self._visible_workspace_root,
                actual_container_workdir=self._actual_container_workdir,
                command=arguments.command or "",
                name=arguments.name,
                cwd=arguments.cwd,
                preferred_port=arguments.preferred_port,
                replace_existing=arguments.replace_existing,
                env=arguments.env,
                startup_timeout=arguments.startup_timeout,
            )
            record = started.record
            lines = [
                f"Started service `{record.service_id}`.",
                f"Name: {record.name}",
                f"Address: {record.address}",
                f"Port: {record.port}/{record.protocol}",
                f"Working directory: {record.workdir}",
                f"Command: {record.command}",
                (
                    "Ready check: passed."
                    if started.ready
                    else "Ready check: skipped; use `service` with `action=status` if you need a fresh check."
                ),
                "",
                "Use `service` with `action=status`, `logs`, `stop`, or `list` to manage it later.",
            ]
            return AgentToolResult(
                content=[TextContent(text="\n".join(lines))],
                details=self._service_details(record),
            )

        if arguments.action == "list":
            services = await self._manager.list_services(
                group_id=self._group_id,
                include_stopped=arguments.include_stopped,
            )
            if not services:
                return AgentToolResult(content=[TextContent(text="No managed services in this group.")])
            lines = ["Managed services in this group:"]
            for service in services:
                lines.extend(self._list_entry_lines(service))
            return AgentToolResult(
                content=[TextContent(text="\n".join(lines))],
                details={"services": [self._service_details(service) for service in services]},
            )

        if arguments.action == "status":
            service = await self._manager.get_service(
                group_id=self._group_id,
                service_ref=arguments.service or "",
            )
            return AgentToolResult(
                content=[
                    TextContent(
                        text=self._manager.render_service_summary(
                            service,
                            include_recent_logs=True,
                            max_chars=arguments.max_chars,
                        )
                    )
                ],
                details=self._service_details(service),
            )

        if arguments.action == "logs":
            service = await self._manager.read_logs(
                group_id=self._group_id,
                service_ref=arguments.service or "",
                max_chars=arguments.max_chars,
            )
            log_text = self._manager.read_log_text(service, max_chars=arguments.max_chars) or "(no logs yet)"
            return AgentToolResult(
                content=[
                    TextContent(
                        text=(
                            f"Logs for service `{service.service_id}` ({service.name}):\n"
                            f"{log_text}"
                        )
                    )
                ],
                details=self._service_details(service),
            )

        if arguments.action == "stop":
            service = await self._manager.stop_service(
                group_id=self._group_id,
                service_ref=arguments.service or "",
            )
            lines = [
                f"Stopped service `{service.service_id}`.",
                f"Name: {service.name}",
                f"Status: {service.status}",
                f"Address: {service.address}",
            ]
            if service.exit_code is not None:
                lines.append(f"Exit code: {service.exit_code}")
            return AgentToolResult(
                content=[TextContent(text="\n".join(lines))],
                details=self._service_details(service),
            )

        raise RuntimeError(f"Unsupported service action: {arguments.action}")

    @staticmethod
    def _service_details(service: ManagedServiceRecord) -> dict[str, Any]:
        return {
            "service_id": service.service_id,
            "name": service.name,
            "status": service.status,
            "port": service.port,
            "protocol": service.protocol,
            "address": service.address,
        }

    @staticmethod
    def _list_entry_lines(service: ManagedServiceRecord) -> list[str]:
        lines = [
            f"- {service.service_id} ({service.name}): {service.status} at {service.address}",
            f"  Port: {service.port}/{service.protocol}",
            f"  Workdir: {service.workdir}",
        ]
        if service.exit_code is not None:
            lines.append(f"  Exit code: {service.exit_code}")
        return lines
