from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import PydanticUndefined

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.ai.types import TextContent

from ..memory import (
    MemoryManager,
    opened_archive_to_dict,
    render_search_results,
    search_hit_to_dict,
)


class MemorySearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description=(
            "Short search-engine style content query. Use content words only, for example "
            "`nginx 配置 证书`; do not include temporal/meta words like `上周`, `之前`, `上次`, "
            "`那个`, `聊过`, or `聊天记录`. For pure time-range recall, use memory_time_search."
        ),
    )
    user_id: str | None = Field(default=None, description="Optional QQ user id to restrict participation.")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum candidate archives to return.")

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


class MemoryTimeSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_time: str | None = Field(
        default=None,
        description=(
            "Optional ISO datetime lower bound for the requested chat time range. "
            "Use the current UTC+8 time from the system prompt to resolve relative dates."
        ),
    )
    end_time: str | None = Field(
        default=None,
        description=(
            "Optional ISO datetime upper bound for the requested chat time range. "
            "Use the current UTC+8 time from the system prompt to resolve relative dates."
        ),
    )
    user_id: str | None = Field(default=None, description="Optional QQ user id to restrict participation.")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum archives to return.")

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
    def _require_time_bound(self) -> Self:
        if not (self.start_time or self.end_time):
            raise ValueError("start_time or end_time is required")
        return self


class MemoryOpenInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    archive_id: int = Field(description="Archive id returned by memory_search or memory_time_search.")
    mode: Literal["compact", "transcript", "tools", "full"] = Field(
        default="compact",
        description="How much context to open. Start with compact; use tools/full only when needed.",
    )
    around_message_id: int | None = Field(default=None, description="Optional message id to open around.")
    before: int = Field(default=8, ge=0, le=50, description="Messages before around_message_id.")
    after: int = Field(default=8, ge=0, le=50, description="Messages after around_message_id.")
    include_tool_results: bool = Field(
        default=False,
        description="Whether to include fuller tool result text instead of previews.",
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


class MemoryManageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add", "update", "delete"] = Field(
        description="Profile edit action. Use delete when the user asks to forget or invalidate a memory."
    )
    user_id: str | None = Field(
        default=None,
        description="Optional QQ user id. Omit to edit the current speaker's profile.",
    )
    content: str = Field(
        min_length=1,
        description="Memory content to add/update, or the memory description to delete/forget.",
    )


class MemorySearchTool:
    name = "memory_search"
    label = "memory_search"
    description = (
        "Semantically search this group's archived historical conversations by content keywords. "
        "Use it when the user refers to a topic, entity, file, URL, person, or technical detail from "
        "past chat. The query must contain content words only. It returns candidate archives, not full "
        "transcripts."
    )
    parameters = MemorySearchInput

    def __init__(self, *, manager: MemoryManager, group_id: str) -> None:
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

        arguments = MemorySearchInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
        )
        hits = self._manager.search(
            group_id=self._group_id,
            query=arguments.query,
            user_id=arguments.user_id,
            max_results=arguments.max_results,
        )
        return AgentToolResult(
            content=[TextContent(text=render_search_results(hits))],
            details={"archives": [search_hit_to_dict(hit) for hit in hits]},
        )


class MemoryTimeSearchTool:
    name = "memory_time_search"
    label = "memory_time_search"
    description = (
        "Search this group's archived historical conversations by time range. Use it when the user asks "
        "what was discussed during a period, such as last week, yesterday, this morning, or a specific "
        "date/time range. Provide ISO datetimes resolved from the current UTC+8 system time. It returns "
        "matching archives, not full transcripts."
    )
    parameters = MemoryTimeSearchInput

    def __init__(self, *, manager: MemoryManager, group_id: str) -> None:
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

        arguments = MemoryTimeSearchInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
        )
        hits = self._manager.time_search(
            group_id=self._group_id,
            start_time=arguments.start_time,
            end_time=arguments.end_time,
            user_id=arguments.user_id,
            max_results=arguments.max_results,
        )
        return AgentToolResult(
            content=[TextContent(text=render_search_results(hits))],
            details={"archives": [search_hit_to_dict(hit) for hit in hits]},
        )


class MemoryOpenTool:
    name = "memory_open"
    label = "memory_open"
    description = (
        "Open one archived conversation returned by memory_search or memory_time_search. "
        "Use compact first; use transcript/tools/full only if details are needed."
    )
    parameters = MemoryOpenInput

    def __init__(self, *, manager: MemoryManager, group_id: str) -> None:
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

        arguments = MemoryOpenInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
        )
        opened = self._manager.open_archive(
            group_id=self._group_id,
            archive_id=arguments.archive_id,
            mode=arguments.mode,
            around_message_id=arguments.around_message_id,
            before=arguments.before,
            after=arguments.after,
            include_tool_results=arguments.include_tool_results,
        )
        if opened is None:
            return AgentToolResult(
                content=[TextContent(text=f"没有找到 archive_id={arguments.archive_id} 的历史会话。")],
                details={"archive": None},
            )
        return AgentToolResult(
            content=[TextContent(text=opened.text)],
            details={"archive": opened_archive_to_dict(opened)},
        )


class MemoryManageTool:
    name = "memory_manage"
    label = "memory_manage"
    description = (
        "Add, update, or delete pending profile memories for group members. Use it only when "
        "the user explicitly asks you to remember/forget something, or shares durable preference "
        "or background information. Omit user_id to target the current speaker."
    )
    parameters = MemoryManageInput

    def __init__(
        self,
        *,
        manager: MemoryManager,
        group_id: str,
        current_user_provider: Callable[[], tuple[str, str] | None] | None = None,
    ) -> None:
        self._manager = manager
        self._group_id = group_id
        self._current_user_provider = current_user_provider

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

        arguments = MemoryManageInput.model_validate(
            params.model_dump() if hasattr(params, "model_dump") else dict(params or {})
        )
        target_user_id = (arguments.user_id or "").strip()
        target_nickname = ""
        current_user = self._current_user_provider() if self._current_user_provider is not None else None
        if current_user is not None:
            current_user_id, current_nickname = current_user
            if not target_user_id:
                target_user_id = current_user_id
                target_nickname = current_nickname
            elif target_user_id == current_user_id:
                target_nickname = current_nickname
        if not target_user_id:
            return AgentToolResult(
                content=[TextContent(text="无法确定要编辑的用户；请提供 user_id。")],
                details={"edit": None},
            )

        edit = self._manager.add_profile_edit(
            group_id=self._group_id,
            user_id=target_user_id,
            edit_type=arguments.action,
            content=arguments.content,
            nickname=target_nickname,
        )
        action_label = {
            "add": "已记录",
            "update": "已更新",
            "delete": "已标记忘记",
        }[arguments.action]
        return AgentToolResult(
            content=[
                TextContent(
                    text=f"{action_label} user_id={target_user_id} 的画像补充：{arguments.content}"
                )
            ],
            details={"edit": asdict(edit)},
        )
