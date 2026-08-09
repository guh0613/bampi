"""Canonical held-item data (names and sprite-sheet indices) from Showdown."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .bridge import ShowdownRuntime
from .move_data import to_id


@dataclass(frozen=True, slots=True)
class ItemEntry:
    item_id: str
    name: str
    spritenum: int | None


class ItemDataRepository:
    """Lazy-loaded index of held items keyed by id or English name."""

    def __init__(self, runtime: ShowdownRuntime) -> None:
        self._runtime = runtime
        self._by_id: dict[str, ItemEntry] = {}
        self._loaded = False
        self._lock = RLock()
        self._load_lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def warm_up(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            payload = await asyncio.to_thread(self._runtime.load_item_payload)
            self._build_index(payload)

    def get(self, token: str) -> ItemEntry | None:
        """Look up an item by id (``choiceband``) or name (``Choice Band``)."""
        key = to_id(token or "")
        if not key:
            return None
        with self._lock:
            return self._by_id.get(key)

    def _build_index(self, payload: dict[str, dict[str, Any]]) -> None:
        by_id: dict[str, ItemEntry] = {}
        for item_id, data in (payload.get("items") or {}).items():
            name = str(data.get("name") or item_id)
            spritenum = data.get("spritenum")
            entry = ItemEntry(
                item_id=item_id,
                name=name,
                spritenum=spritenum if isinstance(spritenum, int) else None,
            )
            by_id[item_id] = entry
            name_key = to_id(name)
            if name_key:
                by_id.setdefault(name_key, entry)
        with self._lock:
            self._by_id = by_id
            self._loaded = True
