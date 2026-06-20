from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BrowserConfig:
    executable_path: str | None = None
    auto_install: bool = True
    cache_dir: str | None = None
    install_timeout: float = 300.0
    headless: bool = True
    block_images: bool = False
    launch_timeout: float = 45.0
    action_timeout: float = 20.0
    idle_ttl_seconds: int = 300
    max_pages: int = 6
    inline_image_max_bytes: int = 1_000_000
    viewport_width: int = 1440
    viewport_height: int = 1000
    batch_max_commands: int = 32
    batch_timeout: float = 120.0
    recording_fps: int = 10
    recording_max_seconds: int = 600
    allow_private_network: bool = False

    @property
    def action_timeout_ms(self) -> int:
        return max(1, int(self.action_timeout * 1000))
