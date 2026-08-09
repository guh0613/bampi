"""Rewrite simple TeX as plain Unicode, or refuse.

Most inline formulas in a chat reply are small — a symbol, a subscript, a
ratio — and Unicode says them perfectly well: ``\\nabla \\cdot \\mathbf{E} =
\\rho / \\varepsilon_0`` is ``∇·E = ρ/ε₀``. Text that reads correctly beats an
image every time: it wraps, it can be copied, and it costs no render.

So the question this module answers is not "is this formula complicated" — any
score for that would be a pile of magic numbers — but "can I say the whole
thing in Unicode without lying about it". The answer is all-or-nothing per
formula: one construct that cannot be expressed makes the whole conversion
fail, and the caller decides what to do with the formula instead.

What cannot be expressed is, in practice, exactly what is genuinely
two-dimensional: matrices, cases, stacked limits that Unicode has no
sub/superscript glyphs for. Linear structures survive, because a fraction
written ``a/b`` is not a lie as long as the parentheses are put back.
"""

from __future__ import annotations

__all__ = ["tex_to_unicode"]


def _table(source: str, target: str) -> dict[str, str]:
    if len(source) != len(target):  # pragma: no cover - guards a typo at import
        raise ValueError("table halves must align")
    return dict(zip(source, target))


# Unicode's superscript and subscript repertoires are famously incomplete —
# there is no superscript "q" and no subscript "b" — so these tables are the
# real boundary of what this module can express.
_SUPERSCRIPT = {
    **_table("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"),
    **_table("+-=()", "⁺⁻⁼⁽⁾"),
    **_table("abcdefghijklmnoprstuvwxyz", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"),
    **_table("ABDEGHIJKLMNOPRTUVW", "ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ"),
}
_SUBSCRIPT = {
    **_table("0123456789", "₀₁₂₃₄₅₆₇₈₉"),
    **_table("+-=()", "₊₋₌₍₎"),
    **_table("aehijklmnoprstuvx", "ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"),
    **_table("βγρφχ", "ᵦᵧᵨᵩᵪ"),
}

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ",
    "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "ϕ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
}

_SYMBOLS = {
    **_GREEK,
    # Operators and relations.
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓",
    "ast": "∗", "star": "⋆", "bullet": "•", "circ": "∘", "oplus": "⊕",
    "ominus": "⊖", "otimes": "⊗", "odot": "⊙", "setminus": "∖",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "ll": "≪", "gg": "≫", "approx": "≈", "sim": "∼", "simeq": "≃",
    "cong": "≅", "equiv": "≡", "propto": "∝", "doteq": "≐",
    "subset": "⊂", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇",
    "in": "∈", "notin": "∉", "ni": "∋", "cup": "∪", "cap": "∩",
    "land": "∧", "lor": "∨", "wedge": "∧", "vee": "∨", "neg": "¬",
    "perp": "⊥", "parallel": "∥", "angle": "∠", "triangle": "△",
    # Arrows.
    "to": "→", "rightarrow": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔",
    "mapsto": "↦", "uparrow": "↑", "downarrow": "↓",
    # Big operators. Their limits become sub/superscripts, which is where a
    # formula usually fails: "∞" has no superscript form.
    "sum": "∑", "prod": "∏", "coprod": "∐",
    "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮", "oiint": "∯",
    # Letterlike and misc.
    "infty": "∞", "partial": "∂", "nabla": "∇", "emptyset": "∅",
    "varnothing": "∅", "forall": "∀", "exists": "∃", "nexists": "∄",
    "hbar": "ℏ", "ell": "ℓ", "Re": "ℜ", "Im": "ℑ", "aleph": "ℵ",
    "degree": "°", "prime": "′", "dagger": "†", "surd": "√",
    "ldots": "…", "dots": "…", "cdots": "⋯", "vdots": "⋮", "ddots": "⋱",
    "quad": " ", "qquad": "  ", "colon": ":",
    "lbrace": "{", "rbrace": "}", "langle": "⟨", "rangle": "⟩",
    "lceil": "⌈", "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋",
    "vert": "|", "Vert": "‖", "lVert": "‖", "rVert": "‖", "|": "‖",
    # Escaped literals.
    "{": "{", "}": "}", "$": "$", "%": "%", "&": "&", "#": "#", "_": "_",
    " ": " ", ",": " ", ";": " ", ":": " ", "!": "", "/": "",
}

# Upright multi-letter names. Rendering them as their own letters is exactly
# what the TeX means.
_FUNCTIONS = frozenset(
    """arccos arcsin arctan arg cos cosh cot coth csc deg det dim exp gcd hom
    inf ker lg lim liminf limsup ln log max min sec sin sinh sup tan tanh"""
    .split()
)

# Font switches carry no meaning a chat message can keep, so their argument is
# simply unwrapped — except where Unicode has the letter outright.
_UNWRAPPING = frozenset(
    {"mathbf", "mathrm", "mathit", "mathsf", "mathtt", "boldsymbol", "bm",
     "text", "textrm", "textbf", "textit", "operatorname", "displaystyle"}
)
_BLACKBOARD = _table("CHNPQRZ", "ℂℍℕℙℚℝℤ")
_SCRIPT = _table("BEFHILMPR", "ℬℰℱℋℐℒℳ℘ℛ")

_ACCENTS = {
    "hat": "̂", "widehat": "̂", "bar": "̄", "overline": "̄",
    "vec": "⃗", "dot": "̇", "ddot": "̈", "tilde": "̃",
}

# A part of a fraction or a radicand needs parentheses when it is not a single
# term; these are the characters that make it more than one.
_NEEDS_PARENS = set(" +-*/±∓·×÷=<>≤≥≠→↔∈∪∩")

# How long a sub/superscript may be before an unmappable one is refused rather
# than left in "x_max" form.
_SCRIPT_FALLBACK_MAX = 4


class _Unsupported(Exception):
    """Raised as soon as a construct has no faithful Unicode form."""


def tex_to_unicode(tex: str) -> str | None:
    """Return *tex* as Unicode text, or ``None`` if it cannot be expressed.

    ``None`` is a verdict, not an error: it means the formula genuinely needs
    typesetting, and the caller should render or pass it through untouched.
    """
    if not tex.strip():
        return None
    scanner = _Scanner(tex)
    try:
        rendered = _render_sequence(scanner, closing=False)
    except _Unsupported:
        return None
    if not scanner.at_end:
        return None
    text = " ".join(rendered.split())
    return text or None


# --------------------------------------------------------------------------- #


class _Scanner:
    """A cursor over TeX source, handing out one construct at a time."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.index = 0

    @property
    def at_end(self) -> bool:
        return self.index >= len(self.source)

    def peek(self) -> str:
        return self.source[self.index] if not self.at_end else ""

    def take(self) -> str:
        char = self.source[self.index]
        self.index += 1
        return char

    def take_command(self) -> str:
        """Consume a ``\\name`` and return ``name``.

        A control symbol such as ``\\,`` is a single character; a control word
        swallows the whitespace that separates it from the next token.
        """
        self.index += 1  # the backslash
        if self.at_end:
            raise _Unsupported("trailing backslash")
        first = self.take()
        if not first.isalpha():
            return first
        name = first
        while not self.at_end and self.peek().isalpha():
            name += self.take()
        while not self.at_end and self.peek() == " ":
            self.index += 1
        return name

    def skip_spaces(self) -> None:
        while not self.at_end and self.peek().isspace():
            self.index += 1


def _render_sequence(scanner: _Scanner, *, closing: bool) -> str:
    """Render tokens until end of input, or until the group's ``}``."""
    out: list[str] = []
    while not scanner.at_end:
        char = scanner.peek()
        if char == "}":
            if not closing:
                raise _Unsupported("unbalanced brace")
            scanner.take()
            return "".join(out)
        if char == "{":
            scanner.take()
            out.append(_render_sequence(scanner, closing=True))
            continue
        if char in "^_":
            scanner.take()
            out.append(_script(scanner, superscript=char == "^"))
            continue
        if char == "\\":
            out.append(_render_command(scanner))
            continue
        if char == "&" or char == "$":
            # An alignment tab or a stray delimiter means the caller handed us
            # something that is not one formula.
            raise _Unsupported(f"unsupported character: {char}")
        if char == "'":
            scanner.take()
            out.append("′")
            continue
        if char == "~":
            scanner.take()
            out.append(" ")
            continue
        out.append(scanner.take())
    if closing:
        raise _Unsupported("unclosed group")
    return "".join(out)


def _render_command(scanner: _Scanner) -> str:
    name = scanner.take_command()

    if name in _ACCENTS:
        argument = _render_argument(scanner)
        if len(argument) != 1:
            raise _Unsupported(f"\\{name} over more than one character")
        return argument + _ACCENTS[name]

    if name in _UNWRAPPING:
        return _render_argument(scanner)

    if name in ("mathbb", "mathcal", "mathscr"):
        argument = _render_argument(scanner)
        table = _BLACKBOARD if name == "mathbb" else _SCRIPT
        return "".join(table.get(char, char) for char in argument)

    if name in ("frac", "dfrac", "tfrac", "cfrac"):
        numerator = _render_argument(scanner)
        denominator = _render_argument(scanner)
        return f"{_parenthesize(numerator)}/{_parenthesize(denominator)}"

    if name == "sqrt":
        root = _optional_argument(scanner)
        radicand = _parenthesize(_render_argument(scanner))
        if not root:
            return "√" + radicand
        if root == "3":
            return "∛" + radicand
        if root == "4":
            return "∜" + radicand
        raise _Unsupported(f"\\sqrt[{root}]")

    if name in ("left", "right", "big", "Big", "bigg", "Bigg", "middle"):
        scanner.skip_spaces()
        if scanner.at_end:
            raise _Unsupported(f"\\{name} without a delimiter")
        if scanner.peek() == "\\":
            return _SYMBOLS.get(scanner.take_command(), "")
        delimiter = scanner.take()
        return "" if delimiter == "." else delimiter

    if name in _FUNCTIONS:
        return name

    if name in _SYMBOLS:
        return _SYMBOLS[name]

    raise _Unsupported(f"\\{name}")


def _render_argument(scanner: _Scanner) -> str:
    """Render one argument: a braced group, a command, or a single character."""
    scanner.skip_spaces()
    if scanner.at_end:
        raise _Unsupported("missing argument")
    if scanner.peek() == "{":
        scanner.take()
        return _render_sequence(scanner, closing=True)
    if scanner.peek() == "\\":
        return _render_command(scanner)
    return scanner.take()


def _optional_argument(scanner: _Scanner) -> str:
    """Render a ``[...]`` argument if one is present."""
    scanner.skip_spaces()
    if scanner.peek() != "[":
        return ""
    scanner.take()
    depth = 1
    start = scanner.index
    while not scanner.at_end and depth:
        char = scanner.take()
        depth += (char == "[") - (char == "]")
    if depth:
        raise _Unsupported("unclosed optional argument")
    return scanner.source[start : scanner.index - 1].strip()


def _script(scanner: _Scanner, *, superscript: bool) -> str:
    """Render a ``^``/``_`` argument, raised or lowered if Unicode allows."""
    body = _render_argument(scanner)
    if not body:
        raise _Unsupported("empty script")

    table = _SUPERSCRIPT if superscript else _SUBSCRIPT
    if all(char in table for char in body):
        return "".join(table[char] for char in body)

    # Unicode is missing the glyph — there is no subscript "b", and no
    # superscript "∞" at all. A short, plain script still reads correctly in
    # its written form ("T_c", "x^n"), so keep that rather than give up; a
    # long or structured one would not, so refuse and let it be typeset.
    marker = "^" if superscript else "_"
    if len(body) <= _SCRIPT_FALLBACK_MAX and not (
        set(body) & _NEEDS_PARENS or any(char in body for char in "^_/()")
    ):
        return marker + body
    raise _Unsupported(f"script has no Unicode form: {marker}{body}")


def _parenthesize(text: str) -> str:
    """Wrap *text* if reading it linearly would otherwise change its meaning."""
    stripped = text.strip()
    if not stripped:
        raise _Unsupported("empty fraction part")
    if len(stripped) == 1 or not (set(stripped) & _NEEDS_PARENS):
        return stripped
    if stripped.startswith("(") and stripped.endswith(")"):
        return stripped
    return f"({stripped})"
