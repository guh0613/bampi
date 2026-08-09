from __future__ import annotations

from .sources import RecommendedSetOption
from ..translations import TranslationService


def format_quick_build_selections(
    selections: tuple[tuple[str, str], ...],
    translator: TranslationService,
) -> str:
    return "、".join(
        f"{translator.translate_species(species)}〔{set_name}〕"
        for species, set_name in selections
    )


def format_recommended_set_option(
    index: int,
    option: RecommendedSetOption,
    translator: TranslationService,
) -> list[str]:
    lines = [f"{index}. {option.name}"]
    metadata: list[str] = []
    if option.item:
        metadata.append(f"道具：{translator.translate_item(option.item)}")
    if option.ability:
        metadata.append(f"特性：{translator.translate_ability(option.ability)}")
    if option.nature:
        metadata.append(f"性格：{translator.translate(option.nature)}")
    if option.tera_type:
        metadata.append(f"太晶：{translator.translate_type(option.tera_type)}")
    if metadata:
        lines.append("  " + "；".join(metadata))
    if option.moves:
        moves = " / ".join(translator.translate_move(move) for move in option.moves)
        lines.append(f"  招式：{moves}")
    if option.evs:
        stat_names = {
            "hp": "HP",
            "atk": "攻击",
            "def": "防御",
            "spa": "特攻",
            "spd": "特防",
            "spe": "速度",
        }
        evs = " / ".join(
            f"{value} {stat_names.get(stat, stat)}" for stat, value in option.evs
        )
        lines.append(f"  努力值：{evs}")
    return lines


__all__ = ["format_quick_build_selections", "format_recommended_set_option"]
