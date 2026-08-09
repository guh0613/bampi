from __future__ import annotations

from bampi.browser.errors import BrowserError, BrowserLaunchError, CdpError

__all__ = ["BrowserError", "BrowserLaunchError", "CdpError", "CommandError", "StaleRefError"]


class CommandError(BrowserError):
    pass


class StaleRefError(CommandError):
    pass
