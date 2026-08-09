"""Pure view-model construction: BattleState -> template-friendly data.

Everything here is synchronous and side-effect free so it can be unit
tested without a browser or network access.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..battle.state import BattleState, PokemonState
from ..battle.utils import (
    position_index_from_ident,
    simplify_ident,
    slot_token_from_ident,
)
from ..text_assets import FIELD_TEXT, SIDE_CONDITION_TEXT, WEATHER_TEXT
from ..translations import TranslationService
from .models import PokemonBattleStatus, TeamPreviewPokemon

# Chinese status labels (from STATUS_TEXT) mapped to CSS badge classes.
STATUS_CLASS = {
    "灼伤": "brn",
    "麻痹": "par",
    "睡眠": "slp",
    "剧毒": "tox",
    "中毒": "psn",
    "冰冻": "frz",
    "瞌睡": "slp",
    "倒下": "fnt",
}

# Showdown weather ids mapped to scene CSS classes.
WEATHER_CLASS = {
    "RainDance": "rain",
    "HeavyRain": "rain",
    "SunnyDay": "sun",
    "HarshSunshine": "sun",
    "Sandstorm": "sand",
    "Hail": "hail",
    "Snow": "hail",
    "StrongWinds": "wind",
}

TERRAIN_CLASS = {
    "Electric Terrain": "electric",
    "Grassy Terrain": "grassy",
    "Misty Terrain": "misty",
    "Psychic Terrain": "psychic",
}


def item_label_key(side: str, name: str) -> str:
    """Normalized lookup key for held-item entries: ``side|nickname``."""
    return f"{(side or '').strip().lower()}|{(name or '').strip().lower()}"


def resolve_item_token(
    items: Optional[Mapping[str, str]],
    status: PokemonBattleStatus,
) -> Optional[str]:
    """Find the held-item token for a Pokémon, matching nickname then species.

    Nicknames come from battle idents; species is the fallback for sources
    that only know the species (e.g. open team sheets).
    """
    if not items:
        return None
    name = simplify_ident(status.ident or "")
    token = items.get(item_label_key(status.side, name))
    if token is not None:
        return token
    return items.get(item_label_key(status.side, status.species))


def build_ball_rows(battle_state: BattleState) -> Dict[str, List[str]]:
    """Poké-ball indicators per side: ``alive`` / ``fainted`` / ``unknown``.

    Slots follow the public roster order for Pokémon that entered battle
    (lettered idents); remaining slots up to the announced team size are
    rendered as ``unknown``. Team-preview-only entries (letterless idents)
    are excluded because e.g. VGC six-pick-four rosters exceed the actual
    battle team.
    """
    rows: Dict[str, List[str]] = {}
    for side in ("p1", "p2"):
        entries: List[str] = []
        seen: set[str] = set()
        for ident in battle_state.side_rosters.get(side) or []:
            state = battle_state.pokemon.get(ident)
            if state is None:
                continue
            if len(slot_token_from_ident(ident)) < 3:
                continue
            name_key = simplify_ident(ident).strip().lower()
            if name_key in seen:
                continue
            seen.add(name_key)
            entries.append("fainted" if state.fainted else "alive")
        total = battle_state.team_sizes.get(side) or len(entries)
        entries = entries[:total]
        entries.extend("unknown" for _ in range(total - len(entries)))
        rows[side] = entries
    return rows


def calculate_hp_ratio(hp_text: str) -> float:
    text = (hp_text or "").strip()
    if not text:
        return 1.0
    if "/" in text:
        current, _, total = text.partition("/")
        try:
            current_value = float(current)
            total_value = float(total)
            if total_value <= 0:
                return 0.0 if current_value <= 0 else 1.0
            return max(0.0, min(current_value / total_value, 1.0))
        except ValueError:
            return 0.0
    if text.endswith("%"):
        try:
            return max(0.0, min(float(text[:-1]) / 100.0, 1.0))
        except ValueError:
            return 0.0
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return max(0.0, min(value / 100.0, 1.0))


def hp_bar_class(ratio: float) -> str:
    if ratio >= 0.5:
        return "hp-high"
    if ratio >= 0.2:
        return "hp-mid"
    return "hp-low"


def status_class(status: Optional[str]) -> str:
    return STATUS_CLASS.get((status or "").strip(), "generic")


def build_status_from_state(
    battle_state: BattleState,
    pokemon: PokemonState,
    translator: TranslationService,
) -> PokemonBattleStatus:
    side = pokemon.side or battle_state.side_from_ident(pokemon.ident or "") or ""
    species_token = pokemon.details.split(",", 1)[0].strip() or simplify_ident(
        pokemon.ident or ""
    )
    name = (
        pokemon.name
        or translator.translate_species(species_token)
        or species_token
        or "未知宝可梦"
    )
    raw_hp_text = (pokemon.hp or "").strip()
    normalized_status = (pokemon.status or "").strip()
    status_text: Optional[str] = normalized_status or None
    base_ratio = calculate_hp_ratio(raw_hp_text) if raw_hp_text else 1.0
    is_fainted = (
        bool(pokemon.fainted) or normalized_status == "倒下" or base_ratio <= 0
    )
    if is_fainted:
        if "/" in raw_hp_text:
            _, _, total = raw_hp_text.partition("/")
            total = total.strip()
            hp_text = f"0/{total}" if total else "0/0"
        elif raw_hp_text:
            hp_text = "0"
        else:
            hp_text = "0/0"
        ratio = 0.0
        status_text = "倒下"
    else:
        hp_text = raw_hp_text
        ratio = base_ratio
    boosts = battle_state.format_boosts(pokemon)
    volatiles = battle_state.format_volatiles(pokemon, translator)
    tera_type = (pokemon.tera_type or "").strip()
    pos_index = pokemon.position_index or position_index_from_ident(pokemon.ident or "")
    if pos_index is not None and pos_index < 1:
        pos_index = None
    return PokemonBattleStatus(
        ident=pokemon.ident,
        side=side or "",
        species=species_token or pokemon.name or pokemon.ident,
        name=name,
        hp_text=hp_text or "未知",
        hp_ratio=ratio,
        status=status_text,
        boosts=boosts.strip() or None,
        volatiles=volatiles.strip() or None,
        tera_type=tera_type or None,
        fainted=is_fainted,
        active=bool(pokemon.active),
        position_index=pos_index,
    )


def collect_active_statuses(
    battle_state: BattleState, translator: TranslationService
) -> Dict[str, List[PokemonBattleStatus]]:
    active: Dict[str, List[PokemonBattleStatus]] = {"p1": [], "p2": []}
    for pokemon in battle_state.pokemon.values():
        side = pokemon.side or battle_state.side_from_ident(pokemon.ident or "")
        if side not in active or not pokemon.active:
            continue
        active[side].append(build_status_from_state(battle_state, pokemon, translator))
    for side in active:
        normalized: Dict[str, PokemonBattleStatus] = {}
        entries = active[side]
        # If lettered slots (p1a/p1b/…) exist for this side, ignore
        # letterless entries (plain p1/p2) to prevent duplicates.
        has_lettered = any(
            len(slot_token_from_ident(st.ident)) >= 3 for st in entries
        )
        if has_lettered:
            entries = [
                st for st in entries if len(slot_token_from_ident(st.ident)) >= 3
            ]
        for status in entries:
            ident_token = (status.ident or "").strip()
            slot = ident_token.split(":", 1)[0].strip() if ident_token else ""
            if len(slot) == 2 and slot.startswith("p") and slot[1].isdigit():
                slot = f"{slot}a"
            simplified = simplify_ident(ident_token) if ident_token else ""
            simplified = simplified or status.name or status.slug
            key = "|".join(
                part
                for part in (
                    (status.side or "").lower(),
                    slot.lower(),
                    simplified.lower(),
                )
                if part
            )
            normalized[key] = status
        ordered = list(normalized.values())
        ordered.sort(
            key=lambda status: (
                status.position_index if status.position_index is not None else 99,
                status.ident or status.name,
            )
        )
        active[side] = ordered
    return active


def collect_bench_statuses(
    battle_state: BattleState,
    side: str,
    translator: TranslationService,
) -> List[PokemonBattleStatus]:
    roster = battle_state.side_rosters.get(side)
    if roster:
        states = [battle_state.pokemon.get(ident) for ident in roster]
        states = [state for state in states if state is not None]
    else:
        states = [
            state for state in battle_state.pokemon.values() if state.side == side
        ]
        states.sort(key=lambda st: st.ident)
    bench = [state for state in states if not state.active]
    bench.sort(key=lambda st: (st.fainted, st.ident))
    return [
        build_status_from_state(battle_state, state, translator) for state in bench
    ]


def side_condition_labels(
    battle_state: BattleState,
    translator: TranslationService,
    side: str,
) -> List[str]:
    conditions = battle_state.field_state.side_conditions.get(side, {})
    labels: List[str] = []
    for effect_key, layers in conditions.items():
        label = SIDE_CONDITION_TEXT.get(effect_key, translator.translate(effect_key))
        if not label:
            continue
        if layers > 1:
            label = f"{label}×{layers}"
        labels.append(label)
    return labels


def field_chips(
    battle_state: BattleState, translator: TranslationService
) -> List[Dict[str, str]]:
    chips: List[Dict[str, str]] = []
    weather = battle_state.field_state.weather
    if weather and weather != "none":
        name = WEATHER_TEXT.get(weather, translator.translate(weather))
        if name:
            chips.append({"kind": "weather", "label": "天气", "value": name})
    terrain = battle_state.field_state.terrain
    if terrain:
        name = FIELD_TEXT.get(terrain, translator.translate(terrain))
        if name:
            chips.append({"kind": "terrain", "label": "场地", "value": name})
    for effect_key, layers in sorted(battle_state.field_state.field_effects.items()):
        label = FIELD_TEXT.get(effect_key, translator.translate(effect_key))
        if not label:
            continue
        if layers and layers > 1:
            label = f"{label}×{layers}"
        chips.append({"kind": "room", "label": "领域", "value": label})
    return chips


def scene_classes(battle_state: BattleState) -> str:
    classes = ["scene"]
    weather = battle_state.field_state.weather
    if weather and weather in WEATHER_CLASS:
        classes.append(f"weather-{WEATHER_CLASS[weather]}")
    terrain = battle_state.field_state.terrain
    if terrain and terrain in TERRAIN_CLASS:
        classes.append(f"terrain-{TERRAIN_CLASS[terrain]}")
    return " ".join(classes)


def sanitize_log_lines(lines: Sequence[str]) -> Tuple[str, List[str]]:
    """Drop redundant turn markers and split off a leading banner line."""
    sanitized: List[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        normalized = stripped.strip("-— ").replace(" ", "")
        if (
            normalized.startswith("第")
            and "回合" in normalized
            and normalized.endswith("回合")
        ):
            continue
        sanitized.append(stripped)
    usable = sanitized or ["暂无新的战况日志。"]

    banner = ""
    if usable and usable[0].startswith(("——", "--", "—", "-")):
        candidate = usable[0].strip("-— ").strip()
        usable = usable[1:]
        if candidate:
            banner = candidate
        if not usable:
            usable = ["战斗即将开始，请稍候。"]
    return banner, usable


def normalize_preview_entries(
    entries: Sequence[object],
) -> List[TeamPreviewPokemon]:
    normalized: List[TeamPreviewPokemon] = []
    for entry in entries:
        if isinstance(entry, TeamPreviewPokemon):
            normalized.append(entry)
        else:
            text = str(entry)
            normalized.append(
                TeamPreviewPokemon(
                    ident=text,
                    species=text,
                    display_name=text or "未知宝可梦",
                )
            )
    if not normalized:
        normalized.append(
            TeamPreviewPokemon(
                ident="placeholder",
                species="unknown",
                display_name="暂无队伍信息",
            )
        )
    return normalized
