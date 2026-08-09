"""Split a reply into plain-text runs and renderable rich blocks.

QQ shows Markdown as literal characters, so structures that carry meaning
through layout — fenced code, display formulas, GFM tables — arrive as noise.
This module locates exactly those structures and leaves everything else alone,
so the caller can render them to images and interleave the results with the
surrounding text.

Two invariants hold for every input:

* concatenating the ``source`` of every returned segment reproduces the input
  byte for byte, so nothing can be silently dropped;
* anything not recognised stays a text segment rather than being guessed at.

The scanner is deliberately narrow. It is not a Markdown implementation and
does not try to become one: only block-level constructs whose visual form is
the point are recognised, because those are the only ones an image improves.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class BlockKind(str, Enum):
    CODE = "code"
    MATH = "math"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A run of text to be sent as-is."""

    source: str

    @property
    def is_blank(self) -> bool:
        return not self.source.strip()


@dataclass(frozen=True, slots=True)
class RichBlock:
    """A block whose meaning depends on layout, to be rendered as an image."""

    kind: BlockKind
    source: str
    """The original Markdown, retained verbatim for the text fallback."""
    content: str
    """The payload: code body, TeX source, or the table's own Markdown."""
    language: str = ""

    @property
    def is_blank(self) -> bool:
        return not self.content.strip()


Segment = TextSegment | RichBlock

# ``~~~`` fences are accepted alongside backticks; CommonMark allows up to three
# leading spaces of indentation before the opening fence.
_FENCE_RE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$")

# A display formula must own its line(s); ``$$`` appearing mid-sentence is
# inline usage and is left as text.
_MATH_OPEN_RE = re.compile(r"^ {0,3}(?:\$\$|\\\[)")

# One cell of a GFM delimiter row: ---, :---, ---:, :---:
_TABLE_DELIM_CELL_RE = re.compile(r"^:?-+:?$")

_LANGUAGE_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "yml": "yaml",
    "md": "markdown",
    "c++": "cpp",
    "rs": "rust",
    "golang": "go",
    "": "",
}

# Languages present in the vendored Shiki bundle. Anything else still renders,
# but as plain monospace rather than highlighted, so it is normalised to "".
SUPPORTED_LANGUAGES = frozenset(
    {
        "jsx",
        "tsx",
        "javascript",
        "typescript",
        "python",
        "bash",
        "json",
        "yaml",
        "html",
        "css",
        "sql",
        "go",
        "rust",
        "java",
        "c",
        "cpp",
        "markdown",
        "diff",
    }
)


def normalize_language(info: str) -> str:
    """Map a fence info string to a bundled Shiki language id, or ``""``."""
    token = info.strip().split()[0].lower() if info.strip() else ""
    token = token.strip("{}.")
    token = _LANGUAGE_ALIASES.get(token, token)
    return token if token in SUPPORTED_LANGUAGES else ""


def split_segments(text: str) -> list[Segment]:
    """Split *text* into ordered text runs and rich blocks.

    Adjacent text is merged, so the result alternates between the two kinds
    wherever a rich block was found.
    """
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    segments: list[Segment] = []
    pending: list[str] = []
    index = 0

    def flush_text() -> None:
        if pending:
            segments.append(TextSegment("".join(pending)))
            pending.clear()

    while index < len(lines):
        for scan in (_scan_fenced_code, _scan_display_math, _scan_table):
            found = scan(lines, index)
            if found is not None:
                block, next_index = found
                if block.is_blank:
                    # An empty construct carries no layout worth an image.
                    pending.extend(lines[index:next_index])
                else:
                    flush_text()
                    segments.append(block)
                index = next_index
                break
        else:
            pending.append(lines[index])
            index += 1

    flush_text()
    return segments


def _scan_fenced_code(lines: list[str], start: int) -> tuple[RichBlock, int] | None:
    match = _FENCE_RE.match(lines[start].rstrip("\n"))
    if match is None:
        return None

    fence = match.group("fence")
    marker = fence[0]
    info = match.group("info")
    # A backtick fence's info string may not itself contain backticks.
    if marker == "`" and "`" in info:
        return None

    body: list[str] = []
    index = start + 1
    closed = False
    while index < len(lines):
        stripped = lines[index].rstrip("\n")
        closing = re.match(rf"^ {{0,3}}{re.escape(marker)}{{{len(fence)},}} *$", stripped)
        if closing:
            index += 1
            closed = True
            break
        body.append(lines[index])
        index += 1

    # An unclosed fence is almost always a truncated reply; treating it as a
    # block anyway keeps the visible result closer to the intent than dumping
    # the raw backticks into the chat.
    content = "".join(body)
    if not closed and not content.strip():
        return None

    return (
        RichBlock(
            kind=BlockKind.CODE,
            source="".join(lines[start:index]),
            content=content.rstrip("\n"),
            language=normalize_language(info),
        ),
        index,
    )


def _scan_display_math(lines: list[str], start: int) -> tuple[RichBlock, int] | None:
    opening = lines[start].rstrip("\n")
    if _MATH_OPEN_RE.match(opening) is None:
        return None

    is_bracket = opening.lstrip().startswith("\\[")
    open_token = "\\[" if is_bracket else "$$"
    close_token = "\\]" if is_bracket else "$$"

    remainder = opening.lstrip()[len(open_token) :]
    closing_at = remainder.find(close_token)
    if closing_at >= 0:
        # Single-line form: $$ ... $$
        trailing = remainder[closing_at + len(close_token) :]
        if trailing.strip():
            return None
        return (
            RichBlock(
                kind=BlockKind.MATH,
                source=lines[start],
                content=remainder[:closing_at].strip(),
            ),
            start + 1,
        )

    body = [remainder] if remainder.strip() else []
    index = start + 1
    while index < len(lines):
        stripped = lines[index].rstrip("\n")
        position = stripped.find(close_token)
        if position >= 0:
            if stripped[position + len(close_token) :].strip():
                return None
            if stripped[:position].strip():
                body.append(stripped[:position])
            index += 1
            return (
                RichBlock(
                    kind=BlockKind.MATH,
                    source="".join(lines[start:index]),
                    content="\n".join(part.strip() for part in body).strip(),
                ),
                index,
            )
        body.append(stripped)
        index += 1

    # Never closed — leave it as text rather than swallowing the rest of the
    # reply into a formula.
    return None


def _is_table_delimiter_row(line: str) -> bool:
    """Whether *line* is a GFM delimiter row such as ``|---|:--:|``.

    Written as a scan rather than one regex because a single-column table
    (``|---|``) is legal, and expressing "one or more cells" alongside the
    optional outer pipes is where regex versions of this quietly go wrong.
    """
    stripped = line.strip()
    if not stripped or len(line) - len(line.lstrip(" ")) > 3:
        return False
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = stripped.split("|")
    if not cells:
        return False
    return all(_TABLE_DELIM_CELL_RE.match(cell.strip()) for cell in cells)


def _scan_table(lines: list[str], start: int) -> tuple[RichBlock, int] | None:
    header = lines[start].rstrip("\n")
    if "|" not in header or not header.strip():
        return None
    if start + 1 >= len(lines):
        return None
    if not _is_table_delimiter_row(lines[start + 1].rstrip("\n")):
        return None

    index = start + 2
    while index < len(lines):
        row = lines[index].rstrip("\n")
        if "|" not in row or not row.strip():
            break
        index += 1

    return (
        RichBlock(
            kind=BlockKind.TABLE,
            source="".join(lines[start:index]),
            content="".join(lines[start:index]).rstrip("\n"),
        ),
        index,
    )


def parse_table(markdown: str) -> tuple[list[str], list[list[str]]]:
    """Parse a GFM table into ``(header_cells, body_rows)``.

    Cell contents are returned raw; escaping is the renderer's concern.
    """
    rows = [line for line in markdown.split("\n") if line.strip()]
    if len(rows) < 2:
        return [], []

    def cells(line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|") and not stripped.endswith("\\|"):
            stripped = stripped[:-1]
        # Split on unescaped pipes only.
        parts = re.split(r"(?<!\\)\|", stripped)
        return [part.strip().replace("\\|", "|") for part in parts]

    header = cells(rows[0])
    body = [cells(row) for row in rows[2:]]
    # Pad or trim so every row matches the header width; a ragged table is a
    # model slip, not a reason to fail the whole render.
    width = len(header)
    normalized = [
        (row + [""] * width)[:width] if len(row) != width else row for row in body
    ]
    return header, normalized
