from __future__ import annotations

from pathlib import Path
from typing import Any

from bampy.agent.cancellation import CancellationToken
from bampy.agent.types import AgentToolResult, AgentToolUpdateCallback
from bampy.app.tools import (
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_patch_tool,
    create_read_tool,
    create_write_tool,
)

from .workspace import resolve_workspace_path, to_workspace_relative


class _WorkspaceToolMixin:
    def __init__(self, workspace_dir: str, container_root: str | None = None) -> None:
        self._workspace_dir = str(Path(workspace_dir).resolve())
        self._container_root = container_root

    def _safe_path(self, value: str | None) -> str:
        resolved = resolve_workspace_path(
            self._workspace_dir,
            value or ".",
            container_root=self._container_root,
        )
        return to_workspace_relative(self._workspace_dir, resolved)


class WorkspaceReadTool(_WorkspaceToolMixin):
    def __init__(self, workspace_dir: str, container_root: str | None = None) -> None:
        super().__init__(workspace_dir, container_root=container_root)
        self._delegate = create_read_tool(self._workspace_dir)
        self.name = self._delegate.name
        self.label = self._delegate.label
        self.description = self._delegate.description
        self.parameters = self._delegate.parameters

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        payload = dict(params.model_dump() if hasattr(params, "model_dump") else dict(params))
        payload["path"] = self._safe_path(payload.get("path"))
        return await self._delegate.execute(tool_call_id, payload, cancellation=cancellation, on_update=on_update)


class WorkspaceWriteTool(_WorkspaceToolMixin):
    def __init__(self, workspace_dir: str, container_root: str | None = None) -> None:
        super().__init__(workspace_dir, container_root=container_root)
        self._delegate = create_write_tool(self._workspace_dir)
        self.name = self._delegate.name
        self.label = self._delegate.label
        self.description = self._delegate.description
        self.parameters = self._delegate.parameters

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        payload = dict(params.model_dump() if hasattr(params, "model_dump") else dict(params))
        payload["path"] = self._safe_path(payload.get("path"))
        return await self._delegate.execute(tool_call_id, payload, cancellation=cancellation, on_update=on_update)


class WorkspaceEditTool(_WorkspaceToolMixin):
    def __init__(self, workspace_dir: str, container_root: str | None = None) -> None:
        super().__init__(workspace_dir, container_root=container_root)
        self._delegate = create_edit_tool(self._workspace_dir)
        self.name = self._delegate.name
        self.label = self._delegate.label
        self.description = self._delegate.description
        self.parameters = self._delegate.parameters

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        payload = dict(params.model_dump() if hasattr(params, "model_dump") else dict(params))
        payload["path"] = self._safe_path(payload.get("path"))
        return await self._delegate.execute(tool_call_id, payload, cancellation=cancellation, on_update=on_update)


class WorkspacePatchTool(_WorkspaceToolMixin):
    def __init__(self, workspace_dir: str, container_root: str | None = None) -> None:
        super().__init__(workspace_dir, container_root=container_root)
        self._delegate = create_patch_tool(self._workspace_dir, container_root=container_root)
        self.name = self._delegate.name
        self.label = self._delegate.label
        self.description = self._delegate.description
        self.parameters = self._delegate.parameters

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        payload = dict(params.model_dump() if hasattr(params, "model_dump") else dict(params))
        return await self._delegate.execute(tool_call_id, payload, cancellation=cancellation, on_update=on_update)


class WorkspaceFindTool(_WorkspaceToolMixin):
    def __init__(self, workspace_dir: str, container_root: str | None = None) -> None:
        super().__init__(workspace_dir, container_root=container_root)
        self._delegate = create_find_tool(self._workspace_dir)
        self.name = self._delegate.name
        self.label = self._delegate.label
        self.description = self._delegate.description
        self.parameters = self._delegate.parameters

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        payload = dict(params.model_dump() if hasattr(params, "model_dump") else dict(params))
        payload["path"] = self._safe_path(payload.get("path"))
        return await self._delegate.execute(tool_call_id, payload, cancellation=cancellation, on_update=on_update)


class WorkspaceGrepTool(_WorkspaceToolMixin):
    def __init__(self, workspace_dir: str, container_root: str | None = None) -> None:
        super().__init__(workspace_dir, container_root=container_root)
        self._delegate = create_grep_tool(self._workspace_dir)
        self.name = self._delegate.name
        self.label = self._delegate.label
        self.description = self._delegate.description
        self.parameters = self._delegate.parameters

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        payload = dict(params.model_dump() if hasattr(params, "model_dump") else dict(params or {}))
        if payload.get("path") is not None:
            payload["path"] = self._safe_path(payload.get("path"))
        return await self._delegate.execute(tool_call_id, payload, cancellation=cancellation, on_update=on_update)
