from __future__ import annotations


class BrowserError(RuntimeError):
    """Base exception for browser transport failures."""


class BrowserLaunchError(BrowserError):
    pass


class CdpError(BrowserError):
    def __init__(self, method: str, message: str, code: int | None = None) -> None:
        prefix = f"CDP {method} failed"
        if code is not None:
            prefix += f" ({code})"
        super().__init__(f"{prefix}: {message}")
        self.method = method
        self.code = code
