from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .sprites import to_slug


@dataclass
class TeamPreviewPokemon:
    ident: str
    species: str
    display_name: str
    level: Optional[int] = None
    gender: Optional[str] = None
    item: Optional[str] = None
    ability: Optional[str] = None
    tera_type: Optional[str] = None

    @property
    def slug(self) -> str:
        return to_slug(self.species or self.ident)


@dataclass
class PokemonBattleStatus:
    ident: str
    side: str
    species: str
    name: str
    hp_text: str
    hp_ratio: float
    status: Optional[str]
    boosts: Optional[str] = None
    volatiles: Optional[str] = None
    tera_type: Optional[str] = None
    fainted: bool = False
    active: bool = False
    position_index: Optional[int] = None

    @property
    def slug(self) -> str:
        return to_slug(self.species or self.ident)

    @property
    def position_label(self) -> Optional[str]:
        if not self.position_index or self.position_index < 1:
            return None
        return f"位置{self.position_index}"
