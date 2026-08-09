from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal


STAT_IDS = ("hp", "atk", "def", "spa", "spd", "spe")


@dataclass(frozen=True, slots=True)
class TeamFormatRules:
    format_id: str
    min_team_size: int
    max_team_size: int
    picked_team_size: int | None
    max_move_count: int
    min_level: int
    max_level: int
    default_level: int
    stat_value_limit: int
    stat_total_limit: int | None
    uses_stat_points: bool
    supports_tera: bool


@dataclass(slots=True)
class EditablePokemonSet:
    name: str = ""
    species: str = ""
    item: str = ""
    ability: str = ""
    moves: list[str] = field(default_factory=list)
    nature: str = ""
    gender: str = ""
    evs: dict[str, int] = field(default_factory=lambda: {stat: 0 for stat in STAT_IDS})
    ivs: dict[str, int] = field(default_factory=lambda: {stat: 31 for stat in STAT_IDS})
    level: int = 100
    shiny: bool = False
    happiness: int = 255
    pokeball: str = ""
    hp_type: str = ""
    dynamax_level: int = 10
    gigantamax: bool = False
    tera_type: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EditablePokemonSet:
        def stat_table(key: str, default: int) -> dict[str, int]:
            raw = payload.get(key)
            return {
                stat: (
                    int(raw[stat])
                    if isinstance(raw, dict) and isinstance(raw.get(stat), (int, float))
                    else default
                )
                for stat in STAT_IDS
            }

        raw_moves = payload.get("moves")
        moves = (
            [str(move).strip() for move in raw_moves if str(move).strip()]
            if isinstance(raw_moves, list)
            else []
        )
        return cls(
            name=str(payload.get("name") or "").strip(),
            species=str(payload.get("species") or "").strip(),
            item=str(payload.get("item") or "").strip(),
            ability=str(payload.get("ability") or "").strip(),
            moves=moves,
            nature=str(payload.get("nature") or "").strip(),
            gender=str(payload.get("gender") or "").strip(),
            evs=stat_table("evs", 0),
            ivs=stat_table("ivs", 31),
            level=int(payload.get("level") or 100),
            shiny=bool(payload.get("shiny", False)),
            happiness=int(
                payload["happiness"]
                if isinstance(payload.get("happiness"), (int, float))
                else 255
            ),
            pokeball=str(payload.get("pokeball") or "").strip(),
            hp_type=str(payload.get("hpType") or "").strip(),
            dynamax_level=int(
                payload["dynamaxLevel"]
                if isinstance(payload.get("dynamaxLevel"), (int, float))
                else 10
            ),
            gigantamax=bool(payload.get("gigantamax", False)),
            tera_type=str(payload.get("teraType") or "").strip(),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "species": self.species,
            "item": self.item,
            "ability": self.ability,
            "moves": list(self.moves),
            "nature": self.nature,
            "gender": self.gender,
            "evs": dict(self.evs),
            "ivs": dict(self.ivs),
            "level": self.level,
        }
        if self.shiny:
            payload["shiny"] = True
        if self.happiness != 255:
            payload["happiness"] = self.happiness
        if self.pokeball:
            payload["pokeball"] = self.pokeball
        if self.hp_type:
            payload["hpType"] = self.hp_type
        if self.dynamax_level != 10:
            payload["dynamaxLevel"] = self.dynamax_level
        if self.gigantamax:
            payload["gigantamax"] = True
        if self.tera_type:
            payload["teraType"] = self.tera_type
        return payload

    def clone(self) -> EditablePokemonSet:
        return EditablePokemonSet.from_payload(copy.deepcopy(self.to_payload()))


@dataclass(slots=True)
class TeamDraft:
    user_id: str
    format_id: str
    rules: TeamFormatRules
    sets: list[EditablePokemonSet]
    team_name: str | None = None
    original_updated_at: float | None = None
    dirty: bool = False


EditorStep = Literal[
    "dashboard",
    "member",
    "add_method",
    "add_recommend_species",
    "add_recommend_set",
    "add_paste",
    "add_manual_species",
    "replace_recommend_species",
    "replace_recommend_set",
    "replace_paste",
    "field_species",
    "field_item",
    "field_ability",
    "field_moves",
    "field_nature",
    "field_evs",
    "field_ivs",
    "field_tera",
    "field_advanced",
    "field_nickname",
    "field_level",
    "field_gender",
    "field_shiny",
    "field_happiness",
    "field_pokeball",
    "field_hp_type",
    "field_dynamax_level",
    "field_gigantamax",
    "delete_confirm",
    "reorder",
    "save_name",
    "save_overwrite",
    "discard_confirm",
]


@dataclass(slots=True)
class TeamEditorState:
    draft: TeamDraft
    step: EditorStep = "dashboard"
    selected_index: int | None = None
    prompt: str = ""
    pending_species: str | None = None
    pending_set_names: tuple[str, ...] = ()
    pending_name: str | None = None
    pending_updated_at: float | None = None
    pending_team_text: str | None = None
    pending_packed: str | None = None
    pending_warnings: tuple[str, ...] = ()
    discard_from_step: EditorStep | None = None
    discard_from_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class EditorResponse:
    message: str
    status: Literal["active", "saved", "cancelled"] = "active"


@dataclass(frozen=True, slots=True)
class TeamCatalog:
    rules: TeamFormatRules
    species: tuple[str, ...]
    items: tuple[str, ...]
    natures: tuple[str, ...]
    types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpeciesEditorOptions:
    species: str
    abilities: tuple[str, ...]
    moves: tuple[str, ...]


__all__ = [
    "EditablePokemonSet",
    "EditorResponse",
    "EditorStep",
    "SpeciesEditorOptions",
    "STAT_IDS",
    "TeamCatalog",
    "TeamDraft",
    "TeamEditorState",
    "TeamFormatRules",
]
