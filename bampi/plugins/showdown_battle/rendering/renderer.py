"""Public battle-visual API: builds template contexts and drives the browser."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape
from nonebot import logger

from bampi.browser import HtmlImageRenderer

from ..battle.state import BattleState, PlayerSlot
from ..item_data import ItemDataRepository
from ..translations import TranslationService
from . import context as ctx
from .assets import pixel_font_url
from .models import PokemonBattleStatus, TeamPreviewPokemon
from .sprites import SpriteStore

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class PokemonBattleRenderer:
    """Render battle information as GBA-style pixel-art images."""

    VIEWPORT_WIDTH = 880

    def __init__(
        self,
        *,
        sprite_cache_dir: Path,
        browser_work_dir: Path,
        sprite_download_timeout: float = 6.0,
        browser_executable: str = "",
        render_scale: int = 2,
        render_idle_ttl_seconds: int = 180,
        item_repository: Optional[ItemDataRepository] = None,
    ) -> None:
        self._items = item_repository
        self._sprites = SpriteStore(
            sprite_cache_dir, download_timeout=sprite_download_timeout
        )
        self._browser = HtmlImageRenderer(
            work_dir=browser_work_dir,
            executable_path=browser_executable,
            scale=render_scale,
            idle_ttl_seconds=render_idle_ttl_seconds,
            log_label="showdown battle render",
        )
        self._env = Environment(
            loader=FileSystemLoader(_TEMPLATES_DIR),
            autoescape=select_autoescape(default=True, default_for_string=True),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def render_turn_log(
        self,
        *,
        format_name: str,
        format_id: str,
        players: Dict[str, PlayerSlot],
        state: BattleState,
        translator: TranslationService,
        lines: Sequence[str],
        items: Optional[Mapping[str, str]] = None,
    ) -> Optional[str]:
        try:
            return await self._render_turn_log(
                format_name=format_name,
                format_id=format_id,
                players=players,
                state=state,
                translator=translator,
                lines=lines,
                items=items,
            )
        except Exception as exc:
            logger.exception(
                f"PokemonBattleRenderer failed to render battle log: {exc}"
            )
            return None

    async def render_team_preview(
        self,
        *,
        format_name: str,
        format_id: str,
        players: Dict[str, PlayerSlot],
        preview: Dict[str, List[TeamPreviewPokemon]],
    ) -> Optional[str]:
        try:
            return await self._render_team_preview(
                format_name=format_name,
                format_id=format_id,
                players=players,
                preview=preview,
            )
        except Exception as exc:
            logger.exception(
                f"PokemonBattleRenderer failed to render team preview: {exc}"
            )
            return None

    async def render_status_panel(
        self,
        *,
        format_name: str,
        format_id: str,
        players: Dict[str, PlayerSlot],
        state: BattleState,
        translator: TranslationService,
        items: Optional[Mapping[str, str]] = None,
    ) -> Optional[str]:
        try:
            return await self._render_status_panel(
                format_name=format_name,
                format_id=format_id,
                players=players,
                state=state,
                translator=translator,
                items=items,
            )
        except Exception as exc:
            logger.exception(
                f"PokemonBattleRenderer failed to render status panel: {exc}"
            )
            return None

    async def shutdown(self) -> None:
        await self._browser.shutdown()
        await self._sprites.close()

    # ------------------------------------------------------------------ #
    # Context assembly                                                    #
    # ------------------------------------------------------------------ #

    def _base_context(
        self,
        *,
        format_name: str,
        format_id: str,
        players: Dict[str, PlayerSlot],
        turn_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        p1 = players.get("p1")
        p2 = players.get("p2")
        return {
            "font_url": pixel_font_url(),
            "width": self.VIEWPORT_WIDTH,
            "format_name": format_name,
            "format_id": format_id,
            "turn_number": turn_number or None,
            "p1_name": p1.display_name if p1 else "Player 1",
            "p2_name": p2.display_name if p2 else "Player 2",
        }

    async def _item_view(
        self,
        token: Optional[str],
        translator: TranslationService,
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve a raw item token into ``(label, icon data URI)``."""
        if not token:
            return None, None
        entry = self._items.get(token) if self._items is not None else None
        icon = None
        if entry is not None:
            icon = await self._sprites.get_item_icon_data_uri(entry.spritenum)
        label = (
            translator.translate_item(token)
            or (entry.name if entry is not None else None)
            or token
        )
        return label, icon

    async def _mon_view(
        self,
        status: PokemonBattleStatus,
        *,
        back: bool,
        translator: TranslationService,
        with_sprite: bool = True,
        items: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        sprite = None
        if with_sprite:
            sprite = await self._sprites.get_data_uri(
                status.species or status.ident, back=back
            )
            if sprite is None and back:
                sprite = await self._sprites.get_data_uri(
                    status.species or status.ident, back=False
                )
        percent = max(0, min(100, round(status.hp_ratio * 100)))
        item_label, item_icon = await self._item_view(
            ctx.resolve_item_token(items, status), translator
        )
        return {
            "name": status.name,
            "sprite": sprite,
            "hp_text": status.hp_text,
            "hp_percent": percent,
            "hp_class": ctx.hp_bar_class(status.hp_ratio),
            "status": status.status,
            "status_class": ctx.status_class(status.status),
            "tera": status.tera_type,
            "boosts": status.boosts,
            "volatiles": status.volatiles,
            "fainted": status.fainted,
            "position_label": status.position_label,
            "item": item_label,
            "item_icon": item_icon,
        }

    async def _mon_views(
        self,
        statuses: Sequence[PokemonBattleStatus],
        *,
        back: bool,
        translator: TranslationService,
        with_sprite: bool = True,
        items: Optional[Mapping[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        return list(
            await asyncio.gather(
                *(
                    self._mon_view(
                        status,
                        back=back,
                        translator=translator,
                        with_sprite=with_sprite,
                        items=items,
                    )
                    for status in statuses
                )
            )
        )

    async def _render_turn_log(
        self,
        *,
        format_name: str,
        format_id: str,
        players: Dict[str, PlayerSlot],
        state: BattleState,
        translator: TranslationService,
        lines: Sequence[str],
        items: Optional[Mapping[str, str]] = None,
    ) -> str:
        banner, log_lines = ctx.sanitize_log_lines(lines)
        active_map = ctx.collect_active_statuses(state, translator)
        sides: Dict[str, Any] = {}
        for side in ("p1", "p2"):
            sides[side] = {
                "active": await self._mon_views(
                    active_map.get(side) or [],
                    back=(side == "p1"),
                    translator=translator,
                    items=items,
                ),
                "conditions": ctx.side_condition_labels(state, translator, side),
            }
        template_context = self._base_context(
            format_name=format_name,
            format_id=format_id,
            players=players,
            turn_number=state.turn_number,
        )
        template_context.update(
            {
                "scene_classes": ctx.scene_classes(state),
                "field_chips": ctx.field_chips(state, translator),
                "sides": sides,
                "banner": banner,
                "log_lines": log_lines,
                "ball_rows": ctx.build_ball_rows(state),
            }
        )
        return await self._screenshot("battle_turn.html.j2", template_context)

    async def _render_team_preview(
        self,
        *,
        format_name: str,
        format_id: str,
        players: Dict[str, PlayerSlot],
        preview: Dict[str, List[TeamPreviewPokemon]],
    ) -> str:
        teams: List[Dict[str, Any]] = []
        for side in ("p1", "p2"):
            entries = ctx.normalize_preview_entries(preview.get(side) or [])
            views: List[Dict[str, Any]] = []
            sprites = await asyncio.gather(
                *(
                    self._sprites.get_data_uri(entry.species or entry.ident)
                    for entry in entries
                )
            )
            for entry, sprite in zip(entries, sprites):
                gender_symbol = (entry.gender or "").strip() or None
                gender_class = {"♂": "m", "♀": "f"}.get(gender_symbol or "", "m")
                views.append(
                    {
                        "name": entry.display_name,
                        "sprite": sprite,
                        "level": entry.level,
                        "gender_symbol": gender_symbol,
                        "gender_class": gender_class,
                    }
                )
            player = players.get(side)
            teams.append(
                {
                    "key": side,
                    "player_name": player.display_name
                    if player
                    else ("Player 1" if side == "p1" else "Player 2"),
                    "entries": views,
                }
            )
        template_context = self._base_context(
            format_name=format_name, format_id=format_id, players=players
        )
        template_context["teams"] = teams
        return await self._screenshot("team_preview.html.j2", template_context)

    async def _render_status_panel(
        self,
        *,
        format_name: str,
        format_id: str,
        players: Dict[str, PlayerSlot],
        state: BattleState,
        translator: TranslationService,
        items: Optional[Mapping[str, str]] = None,
    ) -> str:
        active_map = ctx.collect_active_statuses(state, translator)
        ball_rows = ctx.build_ball_rows(state)
        sides: List[Dict[str, Any]] = []
        for side in ("p1", "p2"):
            bench = ctx.collect_bench_statuses(state, side, translator)
            player = players.get(side)
            sides.append(
                {
                    "key": side,
                    "player_name": player.display_name
                    if player
                    else ("Player 1" if side == "p1" else "Player 2"),
                    "active": await self._mon_views(
                        active_map.get(side) or [],
                        back=False,
                        translator=translator,
                        items=items,
                    ),
                    "bench": await self._mon_views(
                        bench,
                        back=False,
                        translator=translator,
                        with_sprite=False,
                        items=items,
                    ),
                    "conditions": ctx.side_condition_labels(state, translator, side),
                    "balls": ball_rows.get(side) or [],
                }
            )
        template_context = self._base_context(
            format_name=format_name,
            format_id=format_id,
            players=players,
            turn_number=state.turn_number,
        )
        template_context.update(
            {
                "field_chips": ctx.field_chips(state, translator),
                "sides": sides,
                "ball_rows": ball_rows,
            }
        )
        return await self._screenshot("status_panel.html.j2", template_context)

    # ------------------------------------------------------------------ #

    async def _screenshot(
        self, template_name: str, template_context: Dict[str, Any]
    ) -> str:
        html = self._env.get_template(template_name).render(template_context)
        png = await self._browser.render(html, viewport_width=self.VIEWPORT_WIDTH)
        return f"base64://{base64.b64encode(png).decode()}"
