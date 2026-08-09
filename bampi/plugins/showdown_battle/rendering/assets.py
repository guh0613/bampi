"""Bundled static assets for battle rendering."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

PIXEL_FONT_FAMILY = "Fusion Pixel"
_PIXEL_FONT_FILE = _ASSETS_DIR / "fonts" / "fusion-pixel-12px-proportional-zh_hans.otf.woff2"


@lru_cache(maxsize=1)
def pixel_font_url() -> str:
    """``file://`` URL of the bundled CJK pixel font.

    The render browser is launched with ``--allow-file-access-from-files``
    so @font-face can reference the font directly without inlining ~660 KiB
    of base64 into every rendered document.
    """
    return _PIXEL_FONT_FILE.resolve().as_uri()
