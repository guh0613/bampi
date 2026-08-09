from __future__ import annotations

from pathlib import Path

from bampi.browser import (
    LaunchedChromium,
    base_launch_args,
    launch_chromium_process,
)
from bampi.browser import find_chromium as find_chromium  # noqa: PLC0414  (re-export)
from bampi.browser import resolve_chromium as _resolve_chromium

from .config import BrowserConfig
from .stealth import build_stealth_identity, stealth_launch_args

__all__ = ["LaunchedChromium", "chromium_launch_args", "find_chromium", "launch_chromium", "resolve_chromium"]


async def resolve_chromium(config: BrowserConfig) -> str:
    return await _resolve_chromium(
        executable_path=config.executable_path,
        cache_dir=config.cache_dir,
        auto_install=config.auto_install,
        install_timeout=config.install_timeout,
    )


def chromium_launch_args(executable: str, profile_dir: Path, workspace_dir: Path, config: BrowserConfig) -> list[str]:
    identity = None
    if config.stealth:
        identity = build_stealth_identity(
            workspace_dir,
            viewport_width=config.viewport_width,
            viewport_height=config.viewport_height,
        )
    window_width = identity.window_width if identity is not None else config.viewport_width
    window_height = identity.window_height if identity is not None else config.viewport_height
    return base_launch_args(
        executable,
        profile_dir,
        headless=config.headless,
        window_width=window_width,
        window_height=window_height,
        extra_args=stealth_launch_args(identity) if identity is not None else (),
    )


async def launch_chromium(workspace_dir: Path, config: BrowserConfig) -> LaunchedChromium:
    executable = await resolve_chromium(config)
    profile_dir = workspace_dir / ".browser" / "chromium-profile"
    args = chromium_launch_args(executable, profile_dir, workspace_dir, config)
    return await launch_chromium_process(
        args,
        executable=executable,
        profile_dir=profile_dir,
        launch_timeout=config.launch_timeout,
    )
