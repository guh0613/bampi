from __future__ import annotations

from .browser import BrowserTool
from .files import WorkspaceEditTool, WorkspaceFindTool, WorkspaceGrepTool, WorkspaceLsTool, WorkspaceReadTool, WorkspaceWriteTool

from .safe_bash import SafeBashTool
from .web_search import create_web_search_tool


def create_agent_tools(config, workspace_dir: str, *, container_root: str | None = None) -> list[object]:
    effective_container_root = container_root or config.bampi_bash_container_workdir
    tools: list[object] = [
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
        create_web_search_tool(
            config.bampi_web_search_timeout,
            base_url=config.bampi_web_search_base_url,
            api_key=config.bampi_web_search_api_key,
        ),
    ]
    if config.bampi_browser_enabled:
        tools.append(
            BrowserTool(
                workspace_dir,
                container_root=effective_container_root,
                container_name=config.bampi_bash_container_name if config.bampi_bash_mode == "docker" else None,
                bridge_localhost=config.bampi_bash_mode == "docker",
                headless=config.bampi_browser_headless,
                block_images=config.bampi_browser_block_images,
                launch_timeout=config.bampi_browser_launch_timeout,
                action_timeout=config.bampi_browser_action_timeout,
                idle_ttl_seconds=config.bampi_browser_idle_ttl_seconds,
                max_pages=config.bampi_browser_max_pages,
                inline_image_max_bytes=config.bampi_browser_inline_image_max_bytes,
            )
        )
    return tools
