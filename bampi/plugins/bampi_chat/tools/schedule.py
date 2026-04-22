from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import PydanticUndefined

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import TextContent

from ..schedule_manager import ScheduleManager

ScheduleAction = Literal["create", "list", "status", "pause", "resume", "cancel", "run_now"]
ScheduleTriggerType = Literal["date", "cron"]


class ScheduleToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ScheduleAction = Field(description="Schedule management action to run.")
    task: str | None = Field(
        default=None,
        description="Task id or exact task name for status/pause/resume/cancel/run_now.",
    )
    name: str | None = Field(
        default=None,
        description="Optional stable task name for create. Useful for later status/cancel.",
        max_length=120,
    )
    prompt: str | None = Field(
        default=None,
        description=(
            "A sufficiently self-contained task prompt for the later scheduled group conversation. "
            "It should still work even if the exact current turn context is no longer obvious."
        ),
        max_length=12_000,
    )
    trigger_type: ScheduleTriggerType | None = Field(
        default=None,
        description="Trigger type for create: `date` for one-time, `cron` for recurring.",
    )
    run_at: str | None = Field(
        default=None,
        description="Local execution time for `date`, e.g. `2026-04-23 09:00` or ISO datetime.",
    )
    cron: str | None = Field(
        default=None,
        description="Five-field crontab expression for `cron`, e.g. `0 9 * * 1-5`.",
        max_length=128,
    )
    timezone: str | None = Field(
        default=None,
        description="IANA timezone like `Asia/Shanghai`. Defaults to the configured timezone.",
        max_length=80,
    )
    replace_existing: bool = Field(
        default=False,
        description="Whether create should replace an existing active task with the same name in this group.",
    )
    max_chars: int = Field(
        default=4_000,
        ge=200,
        le=40_000,
        description="Maximum characters returned by status output.",
    )

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
    def _validate_action_requirements(self) -> "ScheduleToolInput":
        if self.action == "create":
            if not self.prompt:
                raise ValueError("create requires prompt")
            if self.trigger_type is None:
                raise ValueError("create requires trigger_type")
            if self.trigger_type == "date" and not self.run_at:
                raise ValueError("date create requires run_at")
            if self.trigger_type == "cron" and not self.cron:
                raise ValueError("cron create requires cron")
        if self.action in {"status", "pause", "resume", "cancel", "run_now"} and not self.task:
            raise ValueError(f"{self.action} requires task")
        return self


class ScheduleTool:
    name = "schedule"
    label = "schedule"
    description = (
        "Create and manage scheduled tasks for future execution. "
        "When triggered, a scheduled task starts a normal group conversation turn in the shared session. "
        "Use a prompt that remains understandable later."
    )
    parameters = ScheduleToolInput

    def __init__(
        self,
        *,
        manager: ScheduleManager,
        group_id: str,
    ) -> None:
        self._manager = manager
        self._group_id = group_id

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

        arguments = ScheduleToolInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
        )

        if arguments.action == "create":
            record = await self._manager.create_task(
                group_id=self._group_id,
                name=arguments.name,
                prompt=arguments.prompt or "",
                trigger_type=arguments.trigger_type or "date",
                timezone=arguments.timezone,
                run_at=arguments.run_at,
                cron=arguments.cron,
                replace_existing=arguments.replace_existing,
            )
            lines = [
                f"Created scheduled task `{record.task_id}`.",
                f"Name: {record.name}",
                f"Trigger: {self._manager._render_trigger(record)}",
                f"Timezone: {record.timezone}",
                f"Next run: {record.next_run_at or '-'}",
                "",
                "Use `schedule` with `status`, `list`, `pause`, `resume`, `cancel`, or `run_now` to manage it later.",
            ]
            return AgentToolResult(
                content=[TextContent(text="\n".join(lines))],
                details=self._manager.task_details(record),
            )

        if arguments.action == "list":
            records = await self._manager.list_tasks(group_id=self._group_id)
            return AgentToolResult(
                content=[TextContent(text=self._manager.render_task_list(records))],
                details={"tasks": [self._manager.task_details(record) for record in records]},
            )

        if arguments.action == "status":
            record = await self._manager.get_task(group_id=self._group_id, task_ref=arguments.task or "")
            return AgentToolResult(
                content=[TextContent(text=self._manager.render_task_summary(record, max_chars=arguments.max_chars))],
                details=self._manager.task_details(record),
            )

        if arguments.action == "pause":
            record = await self._manager.pause_task(group_id=self._group_id, task_ref=arguments.task or "")
            return AgentToolResult(
                content=[TextContent(text=f"Paused scheduled task `{record.task_id}` ({record.name}).")],
                details=self._manager.task_details(record),
            )

        if arguments.action == "resume":
            record = await self._manager.resume_task(group_id=self._group_id, task_ref=arguments.task or "")
            return AgentToolResult(
                content=[
                    TextContent(
                        text=(
                            f"Resumed scheduled task `{record.task_id}` ({record.name}).\n"
                            f"Next run: {record.next_run_at or '-'}"
                        )
                    )
                ],
                details=self._manager.task_details(record),
            )

        if arguments.action == "cancel":
            record = await self._manager.cancel_task(group_id=self._group_id, task_ref=arguments.task or "")
            return AgentToolResult(
                content=[TextContent(text=f"Cancelled scheduled task `{record.task_id}` ({record.name}).")],
                details=self._manager.task_details(record),
            )

        if arguments.action == "run_now":
            record = await self._manager.run_task_now(group_id=self._group_id, task_ref=arguments.task or "")
            return AgentToolResult(
                content=[
                    TextContent(
                        text=(
                            f"Triggered scheduled task `{record.task_id}` ({record.name}) to run now "
                            "in a fresh isolated session."
                        )
                    )
                ],
                details=self._manager.task_details(record),
            )

        raise RuntimeError(f"Unsupported schedule action: {arguments.action}")
