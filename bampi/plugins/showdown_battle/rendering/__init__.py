"""HTML-based battle visual rendering for the showdown battle plugin.

Battle scenes are expressed as Jinja2 templates (GBA-inspired pixel look),
rendered to PNG through a dedicated headless Chromium managed on top of the
shared :mod:`bampi.browser` primitives.
"""

from .context import item_label_key
from .models import PokemonBattleStatus, TeamPreviewPokemon
from .renderer import PokemonBattleRenderer

__all__ = [
    "PokemonBattleRenderer",
    "PokemonBattleStatus",
    "TeamPreviewPokemon",
    "item_label_key",
]
