"""Render rich Markdown blocks to PNGs for inline delivery in QQ messages.

All fragments of one reply are laid out on a single page and captured
element by element, so a reply containing five code blocks costs one
navigation and one asset parse rather than five.

Every asset — Shiki, KaTeX, the Maple Mono faces — is vendored under
``assets/`` and referenced by ``file://`` URL. This path runs on ordinary
replies, so a CDN round-trip would be both latency and a failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from jinja2 import Environment, FileSystemLoader, select_autoescape
from nonebot import logger

from bampi.browser import BrowserError, HtmlImageRenderer

from .segments import BlockKind, RichBlock, parse_table

_ASSETS_DIR = (Path(__file__).resolve().parent / "assets").resolve()
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

SHIKI_THEME = "one-dark-pro"

# Outer frame widths. Code is wide enough for roughly 84 columns, which is
# where real code stops wrapping constantly; paper fragments stay narrow
# because a table or formula scaled to bubble width reads better tall.
CODE_FRAME_WIDTH = 760
PAPER_FRAME_WIDTH = 560
CODE_FONT_SIZE = 13.0
GUTTER_WIDTH = 24

# A bare display formula is roughly 5:1, past the ratio at which QQ
# centre-crops an image preview. Vertical padding buys the ratio back.
MATH_FRAME_PADDING = 72


@dataclass(frozen=True, slots=True)
class RenderedBlock:
    """A rich block paired with the PNG it produced."""

    block: RichBlock
    png: bytes


class RichBlockRenderer:
    """Turn :class:`RichBlock` values into PNG bytes."""

    def __init__(
        self,
        *,
        work_dir: Path,
        executable_path: str = "",
        scale: int = 2,
        idle_ttl_seconds: int = 180,
        render_timeout: float = 30.0,
    ) -> None:
        self._browser = HtmlImageRenderer(
            work_dir=work_dir,
            executable_path=executable_path,
            scale=scale,
            idle_ttl_seconds=idle_ttl_seconds,
            render_timeout=render_timeout,
            log_label="bampi_chat rich block render",
        )
        self._env = Environment(
            loader=FileSystemLoader(_TEMPLATES_DIR),
            autoescape=select_autoescape(default=True, default_for_string=True),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def render(self, blocks: list[RichBlock]) -> list[bytes]:
        """Render *blocks* in order, returning one PNG each.

        Raises :class:`BrowserError` if the page fails; callers are expected to
        fall back to sending the original Markdown as text.
        """
        if not blocks:
            return []

        html = self._build_html(blocks)
        selectors = [f"#block-{index}" for index in range(len(blocks))]
        pngs = await self._browser.capture_elements(
            html,
            viewport_width=CODE_FRAME_WIDTH + 40,
            selectors=selectors,
            wait_for_ready=True,
        )
        logger.info(
            f"bampi_chat rendered {len(pngs)} rich blocks "
            f"kinds={[block.kind.value for block in blocks]}"
        )
        return pngs

    async def shutdown(self) -> None:
        await self._browser.shutdown()

    # ------------------------------------------------------------------ #

    def _build_html(self, blocks: list[RichBlock]) -> str:
        view: list[dict[str, object]] = []
        payload: list[dict[str, object]] = []

        for index, block in enumerate(blocks):
            entry: dict[str, object] = {
                "kind": block.kind.value,
                "dom_id": f"block-{index}",
            }
            if block.kind is BlockKind.TABLE:
                header, rows = parse_table(block.content)
                entry["header"] = header
                entry["rows"] = rows
            view.append(entry)
            # Code and TeX go through JSON rather than the template so that
            # neither Jinja escaping nor HTML parsing can touch the source.
            payload.append(
                {
                    "index": index,
                    "content": block.content,
                    "language": block.language,
                }
            )

        template = self._env.get_template("blocks.html.j2")
        return template.render(
            blocks=view,
            payload_json=_script_json(payload),
            theme_json=_script_json(SHIKI_THEME),
            katex_css_url=_asset_uri("katex/katex.min.css"),
            katex_js_url=_asset_uri("katex/katex.min.js"),
            shiki_js_url=_asset_uri("shiki.min.js"),
            font_regular_url=_asset_uri("fonts/maple-400-normal.woff2"),
            font_bold_url=_asset_uri("fonts/maple-700-normal.woff2"),
            font_italic_url=_asset_uri("fonts/maple-400-italic.woff2"),
            code_width=CODE_FRAME_WIDTH,
            paper_width=PAPER_FRAME_WIDTH,
            code_font_size=CODE_FONT_SIZE,
            gutter_width=GUTTER_WIDTH,
            math_padding=MATH_FRAME_PADDING,
        )


def _script_json(value: object) -> str:
    """Serialise *value* for embedding inside a ``<script>`` block.

    The template marks the result safe, so escaping ``</`` here is what stops a
    code sample containing ``</script>`` from closing the tag early.
    """
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _asset_uri(relative: str) -> str:
    path = (_ASSETS_DIR / relative).resolve()
    if not path.is_file():
        raise BrowserError(f"missing vendored render asset: {relative}")
    return path.as_uri()


def missing_assets() -> list[str]:
    """Return the vendored assets that are absent, for startup diagnostics."""
    required = (
        "shiki.min.js",
        "katex/katex.min.css",
        "katex/katex.min.js",
        "fonts/maple-400-normal.woff2",
        "fonts/maple-700-normal.woff2",
        "fonts/maple-400-italic.woff2",
    )
    return [name for name in required if not (_ASSETS_DIR / name).is_file()]
