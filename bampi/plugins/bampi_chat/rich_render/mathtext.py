r"""Normalize math delimiters and say simple inline TeX in Unicode.

Models write mathematics in whichever dialect their training favours:
``$$...$$``, ``\[...\]``, ``\(...\)`` — often glued to the surrounding prose.
QQ renders none of them, and the block scanner in :mod:`.segments`
deliberately recognises only delimiters that own their lines. The two passes
here close that gap, one before segmentation and one after it:

* :func:`normalize_math_delimiters` rewrites every display-math dialect into
  the block form the scanner accepts and turns ``\(...\)`` into ``$...$``,
  so downstream code sees one notation regardless of the model;
* :func:`convert_inline_math` rewrites inline ``$...$`` spans as plain
  Unicode where :func:`.tex_unicode.tex_to_unicode` can do so faithfully,
  because readable text beats raw TeX and no image can sit inside a QQ
  sentence.

Both passes skip fenced code, inline code and Markdown links, and neither
can drop content: anything unpaired, empty or inexpressible is emitted
verbatim.
"""

from __future__ import annotations

import re

from ..message_compose import find_markdown_protected_ranges
from .tex_unicode import tex_to_unicode

__all__ = ["convert_inline_math", "normalize_math_delimiters"]


# Openers of the three math dialects. ``$`` alone is not one: it is handled
# by the inline pass, where the currency heuristics live.
_DELIM_TOKEN_RE = re.compile(r"\\\(|\\\[|(?<!\\)\$\$")
_DISPLAY_CLOSE_RE = re.compile(r"(?<!\\)\$\$")

# A blank line inside a candidate pairing almost always means the delimiters
# do not belong together (a stray opener picking up a closer paragraphs
# away), so such a pairing is refused.
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")

# An inline opener: a single ``$`` that is neither escaped nor part of ``$$``.
_INLINE_OPEN_RE = re.compile(r"(?<![\\$])\$(?!\$)")
_INLINE_CLOSE_RE = re.compile(r"(?<!\\)\$")


def normalize_math_delimiters(text: str) -> str:
    r"""Rewrite ``\(...\)`` as ``$...$`` and display math into block form.

    ``\[...\]`` pairs and ``$$...$$`` pairs that do not already own their
    lines are re-anchored so that the opening and closing delimiters do,
    which is the only form :func:`.segments.split_segments` recognises.
    Unpaired or empty delimiters are left exactly as written.
    """
    if not text or _DELIM_TOKEN_RE.search(text) is None:
        return text

    protected = find_markdown_protected_ranges(text)
    out: list[str] = []
    pos = 0
    while True:
        match = _DELIM_TOKEN_RE.search(text, pos)
        if match is None:
            out.append(text[pos:])
            break
        skip_to = _protected_end(protected, match.start())
        if skip_to is not None:
            out.append(text[pos:skip_to])
            pos = skip_to
            continue
        out.append(text[pos : match.start()])
        replacement, pos = _rewrite_delimiter(text, match, out)
        out.append(replacement)
    return "".join(out)


def convert_inline_math(text: str) -> str:
    r"""Rewrite inline ``$...$`` spans as Unicode where that is faithful.

    A span whose TeX has no complete Unicode form — the all-or-nothing
    verdict of :func:`tex_to_unicode` — is kept verbatim, delimiters and
    all: half-translating a formula would be worse than showing its source.
    Currency never converts: the opener must hug its content, the closer
    must not open a number, and a span may not cross a line break.
    """
    if not text or "$" not in text:
        return text

    protected = find_markdown_protected_ranges(text)
    out: list[str] = []
    pos = 0
    while True:
        match = _INLINE_OPEN_RE.search(text, pos)
        if match is None:
            out.append(text[pos:])
            break
        skip_to = _protected_end(protected, match.start())
        if skip_to is not None:
            out.append(text[pos:skip_to])
            pos = skip_to
            continue
        out.append(text[pos : match.start()])
        replacement, pos = _convert_span(text, match.start(), match.end())
        out.append(replacement)
    return "".join(out)


# --------------------------------------------------------------------------- #


def _protected_end(protected: list[tuple[int, int]], index: int) -> int | None:
    """The end of the protected range covering *index*, or ``None``."""
    for start, end in protected:
        if start > index:
            return None
        if index < end:
            return end
    return None


def _last_char(parts: list[str]) -> str:
    for part in reversed(parts):
        if part:
            return part[-1]
    return ""


def _rewrite_delimiter(
    text: str, match: re.Match[str], out: list[str]
) -> tuple[str, int]:
    """Rewrite the construct *match* opens, returning ``(text, next_index)``."""
    token = match.group()
    end = match.end()

    if token == "\\(":
        closer = text.find("\\)", end)
        if closer < 0:
            return token, end
        content = text[end:closer]
        if not content.strip() or _BLANK_LINE_RE.search(content):
            return token, end
        # Wrapped inline math is re-joined: a formula is one line by nature,
        # and the inline pass only pairs delimiters within a single line.
        return f"${' '.join(content.split())}$", closer + 2

    if token == "\\[":
        closer = text.find("\\]", end)
        if closer < 0:
            return token, end
        content = text[end:closer].strip()
        if not content or _BLANK_LINE_RE.search(content):
            return token, end
        return _display_block(text, out, content, closer + 2)

    # token == "$$"
    closer_match = _DISPLAY_CLOSE_RE.search(text, end)
    if closer_match is None:
        return token, end
    inner = text[end : closer_match.start()]
    content = inner.strip()
    if not content or _BLANK_LINE_RE.search(inner):
        return token, end
    resume = closer_match.end()
    if _is_block_form(text, match.start(), resume):
        return text[match.start() : resume], resume
    return _display_block(text, out, content, resume)


def _is_block_form(text: str, start: int, resume: int) -> bool:
    """Whether a ``$$ ... $$`` occurrence already owns its lines."""
    line_start = text.rfind("\n", 0, start) + 1
    before = text[line_start:start]
    if before.strip() or len(before) > 3:
        return False
    line_end = text.find("\n", resume)
    after = text[resume:] if line_end < 0 else text[resume:line_end]
    return not after.strip()


def _display_block(
    text: str, out: list[str], content: str, resume: int
) -> tuple[str, int]:
    """Emit *content* as a ``$$`` block owning its lines.

    Newlines are inserted only where the surrounding text does not already
    provide them; spaces hugging the block are dropped so the inserted
    breaks do not leave stray indentation behind.
    """
    if out and out[-1]:
        out[-1] = out[-1].rstrip(" \t")
    last = _last_char(out)
    prefix = "" if not last or last == "\n" else "\n"

    while resume < len(text) and text[resume] in " \t":
        resume += 1
    following = text[resume : resume + 1]
    suffix = "" if not following or following == "\n" else "\n"

    block = f"$$\n{content}\n$$" if "\n" in content else f"$${content}$$"
    return f"{prefix}{block}{suffix}", resume


def _convert_span(text: str, start: int, end: int) -> tuple[str, int]:
    """Convert one candidate ``$...$`` span, or leave the ``$`` alone.

    The pairing heuristics mirror :func:`.inline._math_span`: a lone ``$``
    is far more often currency than a formula, so the delimiters must hug
    their content and the closer may not open a number.
    """
    following = text[end : end + 1]
    if not following or following.isspace():
        return "$", end

    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)

    for candidate in _INLINE_CLOSE_RE.finditer(text, end, line_end):
        if text[candidate.start() - 1].isspace():
            continue
        after = text[candidate.end() : candidate.end() + 1]
        if after.isdigit() or after == "$":
            continue
        rendered = tex_to_unicode(text[end : candidate.start()])
        if rendered is None:
            # A genuine formula with no Unicode form: keep its source whole.
            return text[start : candidate.end()], candidate.end()
        return rendered, candidate.end()

    return "$", end
