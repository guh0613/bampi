"""Shared headless-Chromium primitives used across bampi plugins.

This package hosts the transport-level pieces of the browser stack:
executable discovery (with Chrome for Testing auto-install), process
launching, and a small CDP client. Higher-level behaviour (stealth,
navigation policy, page management) lives with each consumer.
"""

from .cdp import CdpClient
from .errors import BrowserError, BrowserLaunchError, CdpError
from .html_renderer import HtmlImageRenderer
from .installer import default_cache_dir, ensure_chrome_for_testing, find_cached_chrome
from .launcher import (
    LaunchedChromium,
    base_launch_args,
    find_chromium,
    launch_chromium_process,
    resolve_chromium,
)

__all__ = [
    "BrowserError",
    "BrowserLaunchError",
    "CdpClient",
    "CdpError",
    "HtmlImageRenderer",
    "LaunchedChromium",
    "base_launch_args",
    "default_cache_dir",
    "ensure_chrome_for_testing",
    "find_cached_chrome",
    "find_chromium",
    "launch_chromium_process",
    "resolve_chromium",
]
