"""Async Pokémon sprite retrieval with on-disk caching.

Sprites come from the public Pokémon Showdown sprite CDN (gen5 pixel art,
front and back views). Downloads are cached under the configured sprite
cache directory and exposed as ``data:`` URIs for inline template embedding.
"""

from __future__ import annotations

import asyncio
import base64
import io
import unicodedata
from pathlib import Path
from typing import Dict, Optional

import httpx
from nonebot import logger


def to_slug(text: str) -> str:
    """Canonical alphanumeric identifier used for cache file names."""
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized.lower() if ch.isalnum())


def slug_candidates(text: str) -> list[str]:
    """Possible Showdown sprite ids for a species token.

    Showdown keeps a hyphen between base species and forme
    (``landorus-therian``) but strips punctuation inside names (``hooh``,
    ``mrmime``). Without full dex data we try the hyphen-preserving variant
    first and fall back to the fully collapsed one.
    """
    normalized = unicodedata.normalize("NFKD", (text or "").lower())
    hyphenated = "".join(
        ch for ch in normalized.replace(" ", "-") if ch.isalnum() or ch == "-"
    ).strip("-")
    while "--" in hyphenated:
        hyphenated = hyphenated.replace("--", "-")
    collapsed = to_slug(text)
    candidates = []
    if hyphenated and "-" in hyphenated:
        candidates.append(hyphenated)
    if collapsed:
        candidates.append(collapsed)
    return candidates


_SPRITE_URL_SETS: Dict[str, tuple[str, ...]] = {
    "front": (
        "https://play.pokemonshowdown.com/sprites/gen5/{slug}.png",
        "https://play.pokemonshowdown.com/sprites/gen5ani/{slug}.gif",
    ),
    "back": (
        "https://play.pokemonshowdown.com/sprites/gen5-back/{slug}.png",
        "https://play.pokemonshowdown.com/sprites/gen5ani-back/{slug}.gif",
    ),
}

# Item icons ship as one sprite sheet of 24x24 tiles, 16 per row, indexed by
# each item's ``spritenum`` (the standalone per-item icon files on the CDN
# are incomplete for recent generations, so the sheet is the reliable source).
_ITEM_SHEET_URL = "https://play.pokemonshowdown.com/sprites/itemicons-sheet.png"
_ITEM_ICON_SIZE = 24
_ITEM_SHEET_COLUMNS = 16


class SpriteStore:
    """Fetch and cache gen5 sprites, exposing them as data URIs."""

    def __init__(self, cache_dir: Path, *, download_timeout: float = 6.0) -> None:
        self._cache_dir = cache_dir
        self._download_timeout = download_timeout
        self._memory: Dict[str, Optional[str]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def get_data_uri(self, species: str, *, back: bool = False) -> Optional[str]:
        slug = to_slug(species)
        if not slug:
            return None
        view = "back" if back else "front"
        key = f"{view}/{slug}"
        cached = self._memory.get(key)
        if key in self._memory:
            return cached
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._memory:
                return self._memory[key]
            data = await self._load(species, slug, view)
            uri = (
                f"data:image/png;base64,{base64.b64encode(data).decode()}"
                if data
                else None
            )
            self._memory[key] = uri
            return uri

    async def get_item_icon_data_uri(self, spritenum: Optional[int]) -> Optional[str]:
        """Data URI for a held-item icon cut from Showdown's item sheet."""
        if spritenum is None or spritenum < 0:
            return None
        key = f"item/{spritenum}"
        if key in self._memory:
            return self._memory[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._memory:
                return self._memory[key]
            data = await self._load_item_icon(spritenum)
            uri = (
                f"data:image/png;base64,{base64.b64encode(data).decode()}"
                if data
                else None
            )
            self._memory[key] = uri
            return uri

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ------------------------------------------------------------------ #

    def _cache_path(self, slug: str, view: str) -> Path:
        return self._cache_dir / f"gen5-{view}" / f"{slug}.png"

    async def _load(self, species: str, slug: str, view: str) -> Optional[bytes]:
        path = self._cache_path(slug, view)
        if path.exists():
            try:
                return path.read_bytes()
            except OSError:
                logger.debug(f"SpriteStore failed to read cached sprite: {path}")
        data = await self._download(species, view)
        if data is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError:
            logger.debug(f"SpriteStore failed to persist sprite cache: {path}")
        return data

    async def _download(self, species: str, view: str) -> Optional[bytes]:
        client = await self._get_client()
        for slug in slug_candidates(species):
            for template in _SPRITE_URL_SETS[view]:
                url = template.format(slug=slug)
                try:
                    response = await client.get(url)
                except httpx.HTTPError:
                    logger.debug(f"SpriteStore failed to fetch {url}")
                    continue
                if response.status_code != 200 or not response.content:
                    continue
                if url.endswith(".gif"):
                    png = await asyncio.to_thread(_gif_to_png, response.content)
                    if png is None:
                        logger.debug(f"SpriteStore failed to decode gif sprite: {url}")
                        continue
                    return png
                return response.content
        logger.debug(f"SpriteStore found no {view} sprite for {species!r}")
        return None

    async def _load_item_icon(self, spritenum: int) -> Optional[bytes]:
        path = self._cache_dir / "itemicons" / f"{spritenum}.png"
        if path.exists():
            try:
                return path.read_bytes()
            except OSError:
                logger.debug(f"SpriteStore failed to read cached item icon: {path}")
        sheet = await self._load_item_sheet()
        if sheet is None:
            return None
        tile = await asyncio.to_thread(_crop_item_icon, sheet, spritenum)
        if tile is None:
            logger.debug(f"SpriteStore found no item icon for spritenum {spritenum}")
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tile)
        except OSError:
            logger.debug(f"SpriteStore failed to persist item icon cache: {path}")
        return tile

    async def _load_item_sheet(self) -> Optional[bytes]:
        path = self._cache_dir / "itemicons-sheet.png"
        if path.exists():
            try:
                return path.read_bytes()
            except OSError:
                logger.debug(f"SpriteStore failed to read cached item sheet: {path}")
        client = await self._get_client()
        try:
            response = await client.get(_ITEM_SHEET_URL)
        except httpx.HTTPError:
            logger.debug(f"SpriteStore failed to fetch {_ITEM_SHEET_URL}")
            return None
        if response.status_code != 200 or not response.content:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        except OSError:
            logger.debug(f"SpriteStore failed to persist item sheet cache: {path}")
        return response.content

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=self._download_timeout,
                    follow_redirects=True,
                )
            return self._client


def _crop_item_icon(sheet: bytes, spritenum: int) -> Optional[bytes]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(sheet)) as image:
            left = (spritenum % _ITEM_SHEET_COLUMNS) * _ITEM_ICON_SIZE
            top = (spritenum // _ITEM_SHEET_COLUMNS) * _ITEM_ICON_SIZE
            if left + _ITEM_ICON_SIZE > image.width or top + _ITEM_ICON_SIZE > (
                image.height
            ):
                return None
            tile = image.convert("RGBA").crop(
                (left, top, left + _ITEM_ICON_SIZE, top + _ITEM_ICON_SIZE)
            )
            buffer = io.BytesIO()
            tile.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:
        return None


def _gif_to_png(content: bytes) -> Optional[bytes]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as gif:
            buffer = io.BytesIO()
            gif.convert("RGBA").save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:
        return None
