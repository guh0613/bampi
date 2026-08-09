from __future__ import annotations

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.rich_render import (
    BlockKind,
    ImagePart,
    RichBlock,
    RichRenderOptions,
    TextPart,
    TextSegment,
    build_delivery_plan,
    normalize_language,
    parse_table,
    plan_segments,
    rich_render_options_from_config,
    split_segments,
)

CODE_REPLY = "看这段：\n\n```python\nprint(1)\n```\n\n就这样。"


def _kinds(segments) -> list[str]:
    return [
        segment.kind.value if isinstance(segment, RichBlock) else "text"
        for segment in segments
    ]


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "",
        "纯文本没有任何块",
        CODE_REPLY,
        "a\n$$x^2$$\nb",
        "a\n$$\n\\sum_i x_i\n$$\nb",
        "a\n\\[ x = 1 \\]\nb",
        "| a | b |\n|---|---|\n| 1 | 2 |\n后续文字",
        "```js\n未闭合的围栏",
        "行内 $x$ 公式不该被切走",
        "文字 | 带竖线 | 但不是表格",
        "```\n\n```",
        "~~~\ncode\n~~~",
        "```python\nprint('```')\n```",
    ],
)
def test_split_segments_reconstructs_input_exactly(text: str):
    """The hard invariant: segmentation can never drop or alter reply content."""
    segments = split_segments(text)
    assert "".join(segment.source for segment in segments) == text


def test_split_segments_finds_fenced_code_with_language():
    segments = split_segments(CODE_REPLY)
    assert _kinds(segments) == ["text", "code", "text"]
    block = segments[1]
    assert isinstance(block, RichBlock)
    assert block.language == "python"
    assert block.content == "print(1)"


def test_split_segments_keeps_inline_math_as_text():
    segments = split_segments("能量是 $E=mc^2$ 对吧")
    assert _kinds(segments) == ["text"]


def test_split_segments_display_math_strips_delimiters():
    segments = split_segments("看：\n$$\n\\alpha + \\beta\n$$\n完")
    assert _kinds(segments) == ["text", "math", "text"]
    assert segments[1].content == "\\alpha + \\beta"


def test_split_segments_requires_table_delimiter_row():
    # A pipe-laden sentence is not a table without the |---| row beneath it.
    assert _kinds(split_segments("苹果 | 香蕉 | 梨")) == ["text"]


def test_split_segments_unclosed_math_stays_text():
    """An unclosed $$ must not swallow the rest of the reply."""
    text = "开头\n$$\n\\alpha\n还有很多正文"
    assert _kinds(split_segments(text)) == ["text"]


def test_split_segments_ignores_empty_block():
    assert _kinds(split_segments("```\n\n```")) == ["text"]


def test_split_segments_handles_multiple_blocks():
    text = "一\n\n```py\nx=1\n```\n\n二\n\n| a |\n|---|\n| 1 |\n\n三"
    assert _kinds(split_segments(text)) == ["text", "code", "text", "table", "text"]


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ("python", "python"),
        ("py", "python"),
        ("JS", "javascript"),
        ("ts", "typescript"),
        ("c++", "cpp"),
        ("python title=foo.py", "python"),
        ("brainfuck", ""),
        ("", ""),
    ],
)
def test_normalize_language(info: str, expected: str):
    assert normalize_language(info) == expected


def test_parse_table_pads_ragged_rows():
    header, rows = parse_table("| a | b | c |\n|---|---|---|\n| 1 | 2 |")
    assert header == ["a", "b", "c"]
    assert rows == [["1", "2", ""]]


def test_parse_table_respects_escaped_pipe():
    header, rows = parse_table("| a | b |\n|---|---|\n| x \\| y | 2 |")
    assert header == ["a", "b"]
    assert rows == [["x | y", "2"]]


# --------------------------------------------------------------------------- #
# Delivery planning
# --------------------------------------------------------------------------- #


def test_plan_segments_demotes_disabled_kinds():
    text = "一\n\n```py\nx=1\n```\n\n二\n\n$$x^2$$\n尾巴"
    options = RichRenderOptions(code=False, math=True)
    segments = plan_segments(text, options)
    # The code fence is folded back into surrounding text; only math survives.
    assert _kinds(segments) == ["text", "math", "text"]
    assert "```py" in segments[0].source
    assert "".join(segment.source for segment in segments) == text


def test_plan_segments_disabled_returns_single_text():
    segments = plan_segments(CODE_REPLY, RichRenderOptions(enabled=False))
    assert segments == [TextSegment(CODE_REPLY)]


def test_plan_segments_over_max_blocks_falls_back_to_text():
    text = "\n\n".join(f"```py\nx={index}\n```" for index in range(5))
    segments = plan_segments(text, RichRenderOptions(max_blocks=3))
    assert segments == [TextSegment(text)]


def test_rich_render_options_from_config_defaults():
    options = rich_render_options_from_config(BampiChatConfig())
    assert options.enabled is True
    assert options.code is True
    assert options.math is True
    assert options.table is True


class _StubRenderer:
    """Stands in for the browser-backed renderer."""

    def __init__(self, *, fail: bool = False, count: int | None = None) -> None:
        self.fail = fail
        self.count = count
        self.calls: list[list[RichBlock]] = []

    async def render(self, blocks: list[RichBlock]) -> list[bytes]:
        self.calls.append(blocks)
        if self.fail:
            raise RuntimeError("boom")
        total = len(blocks) if self.count is None else self.count
        return [f"png{index}".encode() for index in range(total)]


async def test_build_delivery_plan_interleaves_text_and_images():
    renderer = _StubRenderer()
    plan = await build_delivery_plan(
        CODE_REPLY, renderer=renderer, options=RichRenderOptions()
    )
    assert [type(part).__name__ for part in plan] == [
        "TextPart",
        "ImagePart",
        "TextPart",
    ]
    assert plan[0] == TextPart("看这段：")
    assert isinstance(plan[1], ImagePart)
    assert plan[1].kind is BlockKind.CODE
    assert plan[2] == TextPart("就这样。")


async def test_build_delivery_plan_falls_back_to_markdown_when_render_fails():
    renderer = _StubRenderer(fail=True)
    plan = await build_delivery_plan(
        CODE_REPLY, renderer=renderer, options=RichRenderOptions()
    )
    assert plan == [TextPart(CODE_REPLY)]


async def test_build_delivery_plan_falls_back_on_image_count_mismatch():
    """A short render result must not silently shift images onto wrong blocks."""
    renderer = _StubRenderer(count=0)
    plan = await build_delivery_plan(
        CODE_REPLY, renderer=renderer, options=RichRenderOptions()
    )
    assert plan == [TextPart(CODE_REPLY)]


async def test_build_delivery_plan_without_renderer_sends_text():
    plan = await build_delivery_plan(
        CODE_REPLY, renderer=None, options=RichRenderOptions()
    )
    assert plan == [TextPart(CODE_REPLY)]


async def test_build_delivery_plan_plain_text_untouched():
    renderer = _StubRenderer()
    plan = await build_delivery_plan(
        "就是一句普通的话", renderer=renderer, options=RichRenderOptions()
    )
    assert plan == [TextPart("就是一句普通的话")]
    assert renderer.calls == []


async def test_build_delivery_plan_drops_blank_runs_between_blocks():
    text = "```py\nx=1\n```\n\n```py\ny=2\n```"
    renderer = _StubRenderer()
    plan = await build_delivery_plan(
        text, renderer=renderer, options=RichRenderOptions()
    )
    assert [type(part).__name__ for part in plan] == ["ImagePart", "ImagePart"]
