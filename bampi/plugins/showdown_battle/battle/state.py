from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ..text_assets import (
    BOOST_ORDER,
    FIELD_TEXT,
    SIDE_CONDITION_TEXT,
    STAT_TEXT,
    VOLATILE_TEXT,
    WEATHER_TEXT,
)
from .utils import (
    parse_hp_and_status,
    position_index_from_ident,
    position_index_from_slot,
    simplify_ident,
    slot_token_from_ident,
)

if TYPE_CHECKING:
    from ..translations import TranslationService


@dataclass
class PlayerSlot:
    side: str
    user_id: str
    display_name: str
    team_pack: Optional[str] = None
    team_raw: Optional[str] = None
    is_ai: bool = False


@dataclass
class PokemonState:
    ident: str
    side: str
    name: str = ""
    details: str = ""
    hp: str = ""
    status: Optional[str] = None
    active: bool = False
    fainted: bool = False
    boosts: Dict[str, int] = field(default_factory=dict)
    volatiles: set[str] = field(default_factory=set)
    tera_type: Optional[str] = None
    position_index: Optional[int] = None

    def apply_condition(self, chunk: str) -> Tuple[str, Optional[str]]:
        hp_text, status_text = parse_hp_and_status(chunk)
        if hp_text:
            self.hp = hp_text
            if hp_text.startswith("0") or hp_text.lower() == "0/0":
                self.fainted = True
        if status_text:
            self.status = status_text
            if status_text == "倒下":
                self.fainted = True
        elif "fnt" in (chunk or ""):
            self.status = "倒下"
            self.fainted = True
        return hp_text, status_text


@dataclass
class FieldState:
    weather: Optional[str] = None
    terrain: Optional[str] = None
    field_effects: Dict[str, int] = field(default_factory=dict)
    side_conditions: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {"p1": {}, "p2": {}}
    )

    def reset(self) -> None:
        self.weather = None
        self.terrain = None
        self.field_effects.clear()
        for side in self.side_conditions.values():
            side.clear()


@dataclass
class BattleState:
    pokemon: Dict[str, PokemonState] = field(default_factory=dict)
    field_state: FieldState = field(default_factory=FieldState)
    side_rosters: Dict[str, List[str]] = field(default_factory=dict)
    team_sizes: Dict[str, int] = field(default_factory=dict)
    turn_number: int = 0

    def side_from_ident(self, ident: str) -> Optional[str]:
        ident = ident or ""
        token = ident.split(":", 1)[0].strip()
        if token.startswith("p1"):
            return "p1"
        if token.startswith("p2"):
            return "p2"
        return None

    def get_or_create_pokemon(
        self, ident: str, translator: "TranslationService"
    ) -> PokemonState:
        ident = ident or ""
        state = self.pokemon.get(ident)
        if state:
            if state.position_index is None:
                state.position_index = position_index_from_ident(ident)
            return state
        side = self.side_from_ident(ident) or ""
        base = simplify_ident(ident)
        translated = translator.translate_species(base)
        state = PokemonState(
            ident=ident,
            side=side,
            name=translated or base or ident,
            position_index=position_index_from_ident(ident),
        )
        self.pokemon[ident] = state
        return state

    def _adopt_preview_ident(self, ident: str) -> None:
        """Rebind a team-preview ident (``p1: X``) to its battle ident (``p1a: X``).

        Team-preview requests reference Pokémon without a slot letter while
        battle events use lettered slots. When the lettered ident first shows
        up, migrate the previously registered preview entry so the roster keeps
        a single entry (and its original team order) per Pokémon.
        """
        if not ident or ident in self.pokemon:
            return
        slot = slot_token_from_ident(ident)
        if len(slot) < 3:
            return
        preview_ident = f"{slot[:2]}: {simplify_ident(ident)}"
        state = self.pokemon.pop(preview_ident, None)
        if state is None:
            return
        state.ident = ident
        self.pokemon[ident] = state
        for roster in self.side_rosters.values():
            for index, entry in enumerate(roster):
                if entry == preview_ident:
                    roster[index] = ident

    def register_pokemon(
        self, ident: str, details: str, translator: "TranslationService"
    ) -> PokemonState:
        self._adopt_preview_ident(ident)
        state = self.get_or_create_pokemon(ident, translator)
        state.details = details or state.details
        base = simplify_ident(ident)
        translated_details = translator.translate_details(details) if details else ""
        if translated_details:
            state.name = translated_details.split(",", 1)[0]
        elif base:
            state.name = translator.translate_species(base) or base
        state.side = self.side_from_ident(ident) or state.side
        side_key = state.side or "unknown"
        roster = self.side_rosters.setdefault(side_key, [])
        if ident not in roster:
            roster.append(ident)
        if side_key != "unknown":
            unknown_roster = self.side_rosters.get("unknown")
            if unknown_roster and ident in unknown_roster:
                unknown_roster.remove(ident)
                if not unknown_roster:
                    self.side_rosters.pop("unknown", None)
        return state

    def apply_condition(self, ident: str, condition: str) -> Tuple[str, Optional[str]]:
        if not ident:
            return "", None
        state = self.pokemon.get(ident)
        if not state:
            return "", None
        return state.apply_condition(condition)

    def update_boost(
        self, ident: str, stat: str, amount: str, *, increase: bool
    ) -> None:
        state = self.pokemon.get(ident)
        if not state:
            return
        try:
            delta = int(amount)
        except ValueError:
            delta = 1
        current = state.boosts.get(stat, 0)
        state.boosts[stat] = current + delta if increase else current - delta

    def clear_boosts(self, ident: Optional[str] = None) -> None:
        if ident:
            state = self.pokemon.get(ident)
            if state:
                state.boosts.clear()
            return
        for state in self.pokemon.values():
            state.boosts.clear()

    def set_active_pokemon(self, side: str, ident: str) -> None:
        """Mark the given ident as active without clearing other slots.

        In doubles, each side may have multiple active Pokémon (e.g. p1a, p1b).
        We only demote previous active in the same slot (p1a or p1b) while
        preserving the other slot's active flag.
        """
        slot_token = slot_token_from_ident(ident)
        target_index = position_index_from_slot(slot_token)
        if not slot_token:
            # Fallback to side-wide behavior if slot cannot be determined
            fallback_index = position_index_from_ident(ident)
            target_state: Optional[PokemonState] = None
            for state in self.pokemon.values():
                if state.side != side:
                    continue
                state.active = state.ident == ident
                if state.ident == ident:
                    state.position_index = fallback_index
                    target_state = state
            if target_state and fallback_index:
                self._ensure_unique_position(target_state, fallback_index)
            return
        target_state: Optional[PokemonState] = None
        for state in self.pokemon.values():
            st_slot = slot_token_from_ident(state.ident)
            if st_slot != slot_token:
                continue
            state.active = state.ident == ident
            if state.ident == ident:
                state.position_index = target_index
                target_state = state
        if target_state and target_index:
            self._ensure_unique_position(target_state, target_index)

    def _ensure_unique_position(
        self, target: PokemonState, position_index: int
    ) -> None:
        """Ensure no other Pokémon on the same side claims the same slot index."""
        for state in self.pokemon.values():
            if state is target:
                continue
            if state.side != target.side:
                continue
            if state.position_index == position_index:
                state.position_index = None

    def swap_positions(self, ident: str, new_position: Optional[int]) -> None:
        """Update a Pokémon's tracked position index after a |swap| event."""
        if not ident:
            return
        state = self.pokemon.get(ident)
        if not state:
            return
        side = state.side or self.side_from_ident(ident)
        old_position = state.position_index
        state.active = True
        state.position_index = new_position
        if side and new_position:
            for other in self.pokemon.values():
                if other is state:
                    continue
                if other.side != side:
                    continue
                if other.position_index == new_position:
                    other.position_index = old_position
                    break

    def format_boosts(self, state: PokemonState) -> str:
        if not state.boosts:
            return ""
        ordered: List[str] = []
        for stat in BOOST_ORDER:
            value = state.boosts.get(stat)
            if not value:
                continue
            stat_name = STAT_TEXT.get(stat, stat)
            symbol = "+" if value > 0 else ""
            ordered.append(f"{stat_name}{symbol}{value}")
        remaining = [
            (stat, value)
            for stat, value in state.boosts.items()
            if stat not in BOOST_ORDER and value
        ]
        for stat, value in remaining:
            stat_name = STAT_TEXT.get(stat, stat)
            symbol = "+" if value > 0 else ""
            ordered.append(f"{stat_name}{symbol}{value}")
        return "、".join(ordered)

    def format_volatiles(
        self, state: PokemonState, translator: "TranslationService"
    ) -> str:
        if not state.volatiles:
            return ""
        translated = []
        for effect in sorted(state.volatiles):
            translated.append(VOLATILE_TEXT.get(effect, translator.translate(effect)))
        return "、".join(filter(None, translated))

    def format_active_pokemon_line(
        self, state: PokemonState, translator: "TranslationService"
    ) -> str:
        base_name = simplify_ident(state.ident)
        name = state.name or translator.translate_species(base_name) or base_name
        pos_index = state.position_index or position_index_from_ident(state.ident)
        if pos_index and pos_index < 1:
            pos_index = None
        label_prefix = f"位置{pos_index} " if pos_index else ""
        if state.fainted or state.status == "倒下":
            line = f"{label_prefix}{name} 已倒下"
        else:
            hp_text = state.hp or "未知HP"
            line = f"{label_prefix}{name} HP {hp_text}"
            if state.status:
                line += f"（{state.status}）"
        extras: List[str] = []
        boost_text = self.format_boosts(state)
        if boost_text:
            extras.append(f"能力 {boost_text}")
        volatile_text = self.format_volatiles(state, translator)
        if volatile_text:
            extras.append(f"效果 {volatile_text}")
        if extras:
            line += f"；{'；'.join(extras)}"
        return line

    def format_field_summary(self, translator: "TranslationService") -> Optional[str]:
        segments: List[str] = []
        if self.field_state.weather:
            weather_name = WEATHER_TEXT.get(
                self.field_state.weather, translator.translate(self.field_state.weather)
            )
            segments.append(f"天气：{weather_name}")
        if self.field_state.terrain:
            terrain_name = FIELD_TEXT.get(
                self.field_state.terrain, translator.translate(self.field_state.terrain)
            )
            segments.append(f"场地：{terrain_name}")
        other_effects = [
            FIELD_TEXT.get(effect, translator.translate(effect))
            for effect in sorted(self.field_state.field_effects.keys())
        ]
        other_effects = [name for name in other_effects if name]
        if other_effects:
            segments.append(f"领域：{'、'.join(other_effects)}")
        if not segments:
            return None
        return "；".join(segments)

    def format_side_summary(
        self,
        side: str,
        players: Dict[str, PlayerSlot],
        translator: "TranslationService",
    ) -> List[str]:
        player = players.get(side)
        display = player.display_name if player else side
        lines = [f"{display}："]
        roster = self.side_rosters.get(side)
        if roster:
            states = [self.pokemon.get(ident) for ident in roster]
            states = [state for state in states if state]
        else:
            states = [state for state in self.pokemon.values() if state.side == side]
            states.sort(key=lambda st: st.ident)
        if not states:
            lines.append("  暂无宝可梦信息")
            return lines

        def sort_key(st: PokemonState) -> Tuple[int, str]:
            index = st.position_index or position_index_from_ident(st.ident) or 99
            return (index, st.ident)

        active_states = [state for state in states if state.active]
        active_states.sort(key=sort_key)
        if active_states:
            for state in active_states:
                lines.append(
                    f"  在场：{self.format_active_pokemon_line(state, translator)}"
                )
        else:
            lines.append("  无在场宝可梦")
        bench_states = [
            state for state in states if not state.active and not state.fainted
        ]
        bench_states.sort(key=lambda st: st.ident)
        if bench_states:
            bench_names = []
            for bench in bench_states:
                base_name = simplify_ident(bench.ident)
                bench_names.append(
                    bench.name
                    or translator.translate_species(base_name)
                    or base_name
                    or bench.ident
                )
            lines.append("  后备：" + "、".join(bench_names))
        side_conditions = self.field_state.side_conditions.get(side, {})
        if side_conditions:
            condition_names = []
            for effect_key, layers in side_conditions.items():
                name = SIDE_CONDITION_TEXT.get(
                    effect_key, translator.translate(effect_key)
                )
                if layers > 1:
                    name = f"{name}×{layers}"
                condition_names.append(name)
            if condition_names:
                lines.append(f"  场地：{'、'.join(condition_names)}")
        return lines

    def build_status_report(
        self, players: Dict[str, PlayerSlot], translator: "TranslationService"
    ) -> str:
        lines = ["【对战状态】"]
        if self.turn_number:
            lines.append(f"回合：{self.turn_number}")
        field_line = self.format_field_summary(translator)
        if field_line:
            lines.append(field_line)
        for side in ("p1", "p2"):
            lines.extend(self.format_side_summary(side, players, translator))
        return "\n".join(lines)
