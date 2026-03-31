from __future__ import annotations

from .files import WorkspaceEditTool, WorkspaceFindTool, WorkspaceGrepTool, WorkspaceLsTool, WorkspaceReadTool, WorkspaceWriteTool
from .safe_bash import SafeBashTool
from .web_search import create_web_search_tool


def create_agent_tools(config, workspace_dir: str, *, container_root: str | None = None) -> list[object]:
    effective_container_root = container_root or config.bampi_bash_container_workdir
    return [
        WorkspaceReadTool(workspace_dir, container_root=effective_container_root),
        WorkspaceLsTool(workspace_dir, container_root=effective_container_root),
        WorkspaceFindTool(workspace_dir, container_root=effective_container_root),
        WorkspaceGrepTool(workspace_dir, container_root=effective_container_root),
        SafeBashTool(
            workspace_dir=workspace_dir,
            mode=config.bampi_bash_mode,
            container_name=config.bampi_bash_container_name,
            container_workdir=effective_container_root,
            container_shell=config.bampi_bash_container_shell,
            default_timeout=config.bampi_bash_timeout,
        ),
        WorkspaceEditTool(workspace_dir, container_root=effective_container_root),
        WorkspaceWriteTool(workspace_dir, container_root=effective_container_root),
        create_web_search_tool(config.bampi_web_search_timeout),
    ]
