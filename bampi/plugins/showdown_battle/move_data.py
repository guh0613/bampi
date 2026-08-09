from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .bridge import ShowdownRuntime


_ID_ALLOWED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


def to_id(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch in _ID_ALLOWED)


@dataclass(frozen=True, slots=True)
class MoveEntry:
    move_id: str
    data: dict[str, Any]
    text: dict[str, Any]


class MoveDataRepository:
    def __init__(self, runtime: ShowdownRuntime) -> None:
        self._runtime = runtime
        self._by_id: dict[str, MoveEntry] = {}
        self._name_map: dict[str, str] = {}
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
            payload = await asyncio.to_thread(self._runtime.load_move_payload)
            self._build_index(payload)

    def get(self, move_id: str) -> MoveEntry | None:
        with self._lock:
            direct = self._by_id.get(move_id)
            if direct:
                return direct
            mapped = self._name_map.get(to_id(move_id))
            return self._by_id.get(mapped) if mapped else None

    def search(self, query: str) -> MoveEntry | None:
        normalized = query.strip()
        if not normalized:
            return None
        return self.get(normalized)

    def _build_index(self, payload: dict[str, dict[str, Any]]) -> None:
        moves: dict[str, dict[str, Any]] = payload.get("moves", {})
        moves_text: dict[str, dict[str, Any]] = payload.get("movesText", {})
        by_id: dict[str, MoveEntry] = {}
        name_map: dict[str, str] = {}
        for move_id, move_data in moves.items():
            entry = MoveEntry(
                move_id=move_id,
                data=move_data,
                text=moves_text.get(move_id, {}),
            )
            by_id[move_id] = entry
            for candidate in (move_id, move_data.get("name")):
                if not candidate:
                    continue
                key = to_id(str(candidate))
                if key:
                    name_map.setdefault(key, move_id)
        with self._lock:
            self._by_id = by_id
            self._name_map = name_map
            self._loaded = True
