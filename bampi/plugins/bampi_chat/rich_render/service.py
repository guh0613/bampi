"""Process-wide lifecycle for the rich-block renderer.

One renderer is shared by every group. It owns a headless Chromium that is
launched on first use and released after an idle period, so a bot that never
receives a code block never starts a browser.
"""

from __future__ import annotations

from pathlib import Path

from nonebot import logger

from .renderer import RichBlockRenderer, missing_assets

_renderer: RichBlockRenderer | None = None
_disabled_reason: str | None = None


def get_renderer(config: object) -> RichBlockRenderer | None:
    """Return the shared renderer, or ``None`` if rendering is unavailable.

    ``None`` is a normal outcome, not an error: callers fall back to sending
    the reply's original Markdown as text.
    """
    global _renderer, _disabled_reason

    if not getattr(config, "bampi_rich_render_enabled", True):
        return None
    if _disabled_reason is not None:
        return None
    if _renderer is not None:
        return _renderer

    absent = missing_assets()
    if absent:
        _disabled_reason = f"missing vendored assets: {absent}"
        logger.error(f"bampi_chat rich render disabled — {_disabled_reason}")
        return None

    work_dir = Path(
        getattr(config, "bampi_rich_render_dir", "data/bampi/rich-render")
    ).expanduser()
    _renderer = RichBlockRenderer(
        work_dir=work_dir,
        executable_path=str(getattr(config, "bampi_browser_executable_path", "") or ""),
        scale=int(getattr(config, "bampi_rich_render_scale", 2) or 2),
        idle_ttl_seconds=int(
            getattr(config, "bampi_rich_render_idle_ttl_seconds", 180) or 0
        ),
        render_timeout=float(getattr(config, "bampi_rich_render_timeout", 25.0)),
    )
    logger.info(f"bampi_chat rich render ready work_dir={work_dir}")
    return _renderer


async def shutdown_renderer() -> None:
    global _renderer
    renderer, _renderer = _renderer, None
    if renderer is not None:
        logger.info("bampi_chat shutting down rich block renderer")
        await renderer.shutdown()
