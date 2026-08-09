"""Convert inline Markdown to HTML for use inside rendered blocks.

Table cells routinely carry emphasis, inline code and short formulas — the
whole reason a table was chosen over prose is that its cells are dense. Passing
those cells through as literal text would put ``**bold**`` and ``$x$`` into an
image, which is exactly the noise this package exists to remove.

Scope is deliberately inline-only and small: emphasis, strikethrough, code
spans, links, inline math and explicit line breaks. Block constructs cannot
occur inside a cell, and anything unrecognised — including an unclosed
delimiter — is emitted as literal text, so a cell can never lose content.

This is not a CommonMark implementation. The known simplifications are:

* ``***both***`` is not split into nested emphasis; it degrades to literal
  asterisks around an emphasised run;
* link destinations are dropped, since an image cannot be clicked — only the
  label is shown, styled as a link.
"""

from __future__ import annotations

from html import escape
import re

__all__ = ["render_inline"]


# Only ASCII punctuation may be backslash-escaped; ``\alpha`` in a cell has to
# survive as ``\alpha`` rather than losing its backslash. Math is tried before
# the escape rule so that ``\(`` opens a formula rather than escaping a
# parenthesis — the ambiguity is inherent, and formulas are why cells contain
# backslashes in the first place.
_TOKEN_RE = re.compile(
    r"(?P<math>\$|\\\()"
    r"|(?P<escape>\\[!-/:-@\[-`{-~])"
    r"|(?P<br><br\s*/?>)"
    r"|(?P<code>`+)"
    r"|(?P<link>\[)"
    r"|(?P<delim>\*\*|__|~~|\*|_)"
)

_LINK_RE = re.compile(
    r"\[(?P<label>(?:[^\[\]\\]|\\.)*)\]"
    r"\((?:<[^>]*>|[^\s()]*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)"
)

_MATH_CLOSE_RE = re.compile(r"(?<!\\)\$")

_TAGS = {"**": "strong", "__": "strong", "*": "em", "_": "em", "~~": "s"}


def render_inline(text: str) -> str:
    """Render *text* as an HTML fragment, escaping everything not recognised."""
    if not text:
        return ""
    return _convert(text)


# --------------------------------------------------------------------------- #


def _convert(text: str) -> str:
    parts: list[str] = []
    position = 0
    while position < len(text):
        match = _TOKEN_RE.search(text, position)
        if match is None:
            parts.append(_escape_text(text[position:]))
            break
        if match.start() > position:
            parts.append(_escape_text(text[position : match.start()]))
        html, position = _dispatch(text, match)
        parts.append(html)
    return "".join(parts)


def _dispatch(text: str, match: re.Match[str]) -> tuple[str, int]:
    """Render the construct *match* opens, returning ``(html, next_index)``."""
    kind = match.lastgroup
    start, end = match.span()
    if kind == "escape":
        return _escape_text(match.group()[1]), end
    if kind == "br":
        return "<br>", end
    if kind == "code":
        return _code_span(text, start, end)
    if kind == "math":
        return _math_span(text, start, end)
    if kind == "link":
        return _link(text, start)
    return _emphasis(text, start, end)


def _code_span(text: str, start: int, end: int) -> tuple[str, int]:
    """A run of *n* backticks closed by a run of exactly *n* backticks."""
    length = end - start
    closing = re.compile(r"(?<!`)`{%d}(?!`)" % length).search(text, end)
    if closing is None:
        return _escape_text(text[start:end]), end

    content = text[end : closing.start()].replace("\n", " ")
    # CommonMark: one space of padding on both sides is stripped, which is how
    # a span containing a literal backtick is written.
    if len(content) > 2 and content[0] == " " and content[-1] == " " and content.strip():
        content = content[1:-1]
    return f"<code>{_escape_text(content)}</code>", closing.end()


def _math_span(text: str, start: int, end: int) -> tuple[str, int]:
    r"""Inline math, written ``$...$`` or ``\(...\)``.

    The TeX is handed to the page as an attribute rather than as markup: KaTeX
    runs in the browser, where the fonts are, and attribute text cannot be
    mistaken for HTML on the way there.
    """
    if text[start:end] == "$":
        # A lone "$" is far more often currency than a formula, so require the
        # delimiters to hug their content and the closer not to open a number.
        if end >= len(text) or text[end].isspace() or text[end] == "$":
            return _escape_text("$"), end
        for candidate in _MATH_CLOSE_RE.finditer(text, end):
            if text[candidate.start() - 1].isspace():
                continue
            following = text[candidate.end() : candidate.end() + 1]
            if following.isdigit():
                continue
            return _tex(text[end : candidate.start()]), candidate.end()
        return _escape_text("$"), end

    closing = text.find(r"\)", end)
    if closing < 0:
        return _escape_text(text[start:end]), end
    return _tex(text[end:closing]), closing + 2


def _link(text: str, start: int) -> tuple[str, int]:
    match = _LINK_RE.match(text, start)
    if match is None:
        return _escape_text("["), start + 1
    label = _convert(match.group("label"))
    return f'<span class="link">{label}</span>', match.end()


def _emphasis(text: str, start: int, end: int) -> tuple[str, int]:
    """``*em*``, ``**strong**`` and ``~~strike~~``, resolved non-greedily."""
    marker = text[start:end]
    literal = (_escape_text(marker), end)

    # An opener hugs its content on the right, and an underscore may not sit
    # inside a word — ``snake_case_name`` is an identifier, not emphasis.
    if end >= len(text) or text[end].isspace():
        return literal
    if marker[0] == "_" and start > 0 and _is_word_char(text[start - 1]):
        return literal

    closing = _find_closer(text, marker, end)
    if closing < 0:
        return literal

    tag = _TAGS[marker]
    return f"<{tag}>{_convert(text[end:closing])}</{tag}>", closing + len(marker)


def _find_closer(text: str, marker: str, search_from: int) -> int:
    """Index of the delimiter run that closes *marker*, or ``-1``."""
    char = marker[0]
    length = len(marker)
    position = search_from
    while True:
        position = text.find(marker, position)
        if position < 0:
            return -1
        run_end = position + length
        # Part of a longer run of the same character: it belongs to a different
        # delimiter, as the inner "**" does in "*a **b** c*".
        in_longer_run = (position > 0 and text[position - 1] == char) or (
            run_end < len(text) and text[run_end] == char
        )
        preceding = text[position - 1]
        following = text[run_end : run_end + 1]
        if (
            not in_longer_run
            and position > search_from
            and preceding != "\\"
            and not preceding.isspace()
            and not (char == "_" and _is_word_char(following))
        ):
            return position
        position = run_end


def _is_word_char(char: str) -> bool:
    return bool(char) and (char.isalnum() or char == "_")


def _escape_text(text: str) -> str:
    return escape(text, quote=False)


def _tex(source: str) -> str:
    stripped = source.strip()
    if not stripped:
        return ""
    return f'<span class="tex" data-tex="{escape(stripped, quote=True)}"></span>'
