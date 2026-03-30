from __future__ import annotations

from .files import WorkspaceEditTool, WorkspaceFindTool, WorkspaceGrepTool, WorkspaceLsTool, WorkspaceReadTool, WorkspaceWriteTool
from .safe_bash import SafeBashTool
from .web_search import create_web_search_tool


def create_agent_tools(config, workspace_dir: str) -> list[object]:
    return [
        WorkspaceReadTool(workspace_dir, container_root=config.bampi_bash_container_workdir),
        WorkspaceLsTool(workspace_dir, container_root=config.bampi_bash_container_workdir),
        WorkspaceFindTool(workspace_dir, container_root=config.bampi_bash_container_workdir),
        WorkspaceGrepTool(workspace_dir, container_root=config.bampi_bash_container_workdir),
        SafeBashTool(
            workspace_dir=workspace_dir,
            mode=config.bampi_bash_mode,
            container_name=config.bampi_bash_container_name,
            container_workdir=config.bampi_bash_container_workdir,
            container_shell=config.bampi_bash_container_shell,
            default_timeout=config.bampi_bash_timeout,
        ),
        WorkspaceEditTool(workspace_dir, container_root=config.bampi_bash_container_workdir),
        WorkspaceWriteTool(workspace_dir, container_root=config.bampi_bash_container_workdir),
        create_web_search_tool(config.bampi_web_search_timeout),
    ]
