from __future__ import annotations

from typing import Optional, Tuple

from ..text_assets import STATUS_TEXT


def simplify_ident(ident: str) -> str:
    """Strip player prefix from a full showdown ident."""
    return ident.split(": ", 1)[-1]


def parse_hp_and_status(chunk: str) -> Tuple[str, Optional[str]]:
    """Split a showdown HP chunk into hp text and translated status."""
    if not chunk:
        return "", None
    hp_part, *status_part = chunk.split(" ", 1)
    status_key = status_part[0].strip() if status_part else ""
    status_text = STATUS_TEXT.get(status_key, status_key) if status_key else None
    return hp_part, status_text or None


def slot_token_from_ident(ident: str) -> str:
    """Extract the raw slot token (e.g. p1a) from a showdown ident."""
    token = (ident or "").split(":", 1)[0]
    return token.strip().lower()


def position_index_from_slot(slot: str) -> Optional[int]:
    """Map a slot token (p1a/p1b/…) to a 1-based position index."""
    slot = (slot or "").strip().lower()
    if not slot.startswith("p"):
        return None
    if len(slot) == 2 and slot[1].isdigit():
        # Singles sometimes report as p1/p2 without letter – treat as first position.
        return 1
    if len(slot) >= 3 and slot[1].isdigit():
        letter = slot[2]
        if letter.isalpha():
            return ord(letter.lower()) - ord("a") + 1
    return None


def position_index_from_ident(ident: str) -> Optional[int]:
    """Derive the 1-based position index from a showdown ident."""
    return position_index_from_slot(slot_token_from_ident(ident))
