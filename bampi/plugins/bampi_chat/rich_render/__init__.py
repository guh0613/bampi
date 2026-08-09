"""Render layout-bearing Markdown blocks as images for QQ delivery.

QQ shows Markdown as literal characters, so fenced code, display formulas and
tables arrive as noise. This package finds exactly those blocks in a reply and
renders them, leaving the surrounding prose to be sent as ordinary text.
"""

from .delivery import (
    DeliveryPart,
    ImagePart,
    RichRenderOptions,
    TextPart,
    build_delivery_plan,
    plan_segments,
    rich_render_options_from_config,
)
from .inline import render_inline
from .renderer import (
    CODE_FRAME_WIDTH,
    PAPER_FRAME_WIDTH,
    RenderedBlock,
    RichBlockRenderer,
    missing_assets,
)
from .segments import (
    BlockKind,
    RichBlock,
    Segment,
    TextSegment,
    normalize_language,
    parse_table,
    split_segments,
)

__all__ = [
    "BlockKind",
    "CODE_FRAME_WIDTH",
    "PAPER_FRAME_WIDTH",
    "DeliveryPart",
    "ImagePart",
    "RenderedBlock",
    "RichBlock",
    "RichBlockRenderer",
    "RichRenderOptions",
    "Segment",
    "TextPart",
    "TextSegment",
    "build_delivery_plan",
    "missing_assets",
    "normalize_language",
    "parse_table",
    "plan_segments",
    "render_inline",
    "rich_render_options_from_config",
    "split_segments",
]
