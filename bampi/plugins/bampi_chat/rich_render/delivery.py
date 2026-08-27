"""Turn a reply into an ordered mix of text runs and rendered images.

The caller gets back a plan it can walk to build an OneBot message. Rendering
is best-effort in the strongest sense: any failure degrades to the original
Markdown, and no path here can drop reply content.
"""

from __future__ import annotations

from dataclasses import dataclass

from nonebot import logger

from .mathtext import convert_inline_math, normalize_math_delimiters
from .renderer import RichBlockRenderer
from .segments import BlockKind, RichBlock, Segment, TextSegment, split_segments


@dataclass(frozen=True, slots=True)
class TextPart:
    """Text to send through the normal outbound markup parser."""

    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    """A rendered block, as PNG bytes ready for an image segment."""

    png: bytes
    kind: BlockKind


DeliveryPart = TextPart | ImagePart


@dataclass(frozen=True, slots=True)
class RichRenderOptions:
    enabled: bool = True
    code: bool = True
    math: bool = True
    table: bool = True
    # Rewrite inline ``$...$`` spans as Unicode text where that is faithful.
    math_unicode: bool = True
    # All parts go into a single QQ message, so this cap is only a fuse
    # against pathological replies, not a taste judgement: a math-heavy
    # derivation legitimately carries a dozen display formulas.
    max_blocks: int = 20

    def wants(self, kind: BlockKind) -> bool:
        if kind is BlockKind.CODE:
            return self.code
        if kind is BlockKind.MATH:
            return self.math
        return self.table


def rich_render_options_from_config(config: object) -> RichRenderOptions:
    return RichRenderOptions(
        enabled=bool(getattr(config, "bampi_rich_render_enabled", True)),
        code=bool(getattr(config, "bampi_rich_render_code", True)),
        math=bool(getattr(config, "bampi_rich_render_math", True)),
        table=bool(getattr(config, "bampi_rich_render_table", True)),
        math_unicode=bool(getattr(config, "bampi_rich_render_math_unicode", True)),
        max_blocks=int(getattr(config, "bampi_rich_render_max_blocks", 20) or 20),
    )


def plan_segments(text: str, options: RichRenderOptions) -> list[Segment]:
    """Segment *text*, demoting any block the options exclude back to text.

    Returns a list whose ``source`` values still concatenate to *text*.
    """
    if not text or not options.enabled:
        return [TextSegment(text)] if text else []

    segments = split_segments(text)
    selected = [
        segment
        for segment in segments
        if isinstance(segment, RichBlock) and options.wants(segment.kind)
    ]
    if not selected:
        return [TextSegment(text)] if text else []

    if len(selected) > options.max_blocks:
        logger.info(
            f"bampi_chat rich render skipped: {len(selected)} blocks exceeds "
            f"max_blocks={options.max_blocks}"
        )
        return [TextSegment(text)]

    # Demote excluded kinds so downstream only ever sees blocks it will render.
    demoted: list[Segment] = []
    for segment in segments:
        if isinstance(segment, RichBlock) and not options.wants(segment.kind):
            demoted.append(TextSegment(segment.source))
        else:
            demoted.append(segment)
    return _merge_adjacent_text(demoted)


async def build_delivery_plan(
    text: str,
    *,
    renderer: RichBlockRenderer | None,
    options: RichRenderOptions,
) -> list[DeliveryPart]:
    """Render the rich blocks in *text* and return the interleaved plan.

    Math delimiters are normalised first, so every display-math dialect the
    model may emit reaches the block scanner in the one form it recognises;
    inline ``$...$`` spans in outgoing text are rewritten as Unicode where
    that can be done faithfully. On any rendering failure the whole reply
    falls back to a single text part holding the Markdown, so the content
    still reaches the chat.
    """
    if not text:
        return []

    if options.enabled:
        text = normalize_math_delimiters(text)

    def finish(value: str) -> str:
        if options.enabled and options.math_unicode:
            return convert_inline_math(value)
        return value

    segments = plan_segments(text, options)
    blocks = [segment for segment in segments if isinstance(segment, RichBlock)]
    if not blocks or renderer is None:
        return [TextPart(finish(text))]

    try:
        pngs = await renderer.render(blocks)
    except Exception:
        logger.exception(
            "bampi_chat rich block rendering failed; falling back to Markdown text"
        )
        return [TextPart(finish(text))]

    if len(pngs) != len(blocks):
        logger.error(
            f"bampi_chat rich render returned {len(pngs)} images for "
            f"{len(blocks)} blocks; falling back to Markdown text"
        )
        return [TextPart(finish(text))]

    parts: list[DeliveryPart] = []
    rendered = iter(pngs)
    for segment in segments:
        if isinstance(segment, RichBlock):
            parts.append(ImagePart(png=next(rendered), kind=segment.kind))
            continue
        # Blank runs between blocks would otherwise become stray empty lines
        # around images, which QQ already spaces out on its own.
        if segment.is_blank:
            continue
        parts.append(TextPart(finish(segment.source.strip("\n"))))

    return [
        part
        for part in parts
        if not (isinstance(part, TextPart) and not part.text.strip())
    ]


def _merge_adjacent_text(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if (
            isinstance(segment, TextSegment)
            and merged
            and isinstance(merged[-1], TextSegment)
        ):
            merged[-1] = TextSegment(merged[-1].source + segment.source)
            continue
        merged.append(segment)
    return merged
