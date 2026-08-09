from __future__ import annotations

from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from ..move_data import MoveDataRepository
from .utils import parse_hp_and_status, simplify_ident

if TYPE_CHECKING:
    from ..translations import TranslationService


class PromptBuilder:
    """Generate player-facing prompts based on battle requests."""

    def __init__(
        self,
        translator: "TranslationService",
        move_repository: MoveDataRepository,
    ) -> None:
        self._translator = translator
        self._move_repository = move_repository

    def build_request_prompt(self, request: dict) -> Optional[str]:
        if request.get("wait"):
            return None
        if request.get("teamPreview"):
            return self.build_team_preview_prompt(request)
        if request.get("forceSwitch"):
            return self.build_switch_prompt(request)
        if request.get("active"):
            return self.build_move_prompt(request)
        return None

    def build_team_preview_prompt(self, request: dict) -> str:
        side = request.get("side", {})
        pokemon = side.get("pokemon", [])
        max_team_size = (
            request.get("maxChosenTeamSize")
            or request.get("maxTeamSize")
            or request.get("maxTeam")
        )
        names: List[str] = []
        for idx, mon in enumerate(pokemon, start=1):
            ident = mon.get("ident") or ""
            if ident:
                base_name = simplify_ident(ident)
                translated = self._translator.translate_species(base_name)
            else:
                translated = self._translator.translate_details(mon.get("details", ""))
            names.append(f"{idx}. {translated or '未知宝可梦'}")

        selection_size = (
            max_team_size
            if isinstance(max_team_size, int) and max_team_size > 0
            else len(pokemon)
        )
        if pokemon and selection_size < len(pokemon):
            example_indexes = list(range(1, selection_size + 1))
            example = self._team_order_example(example_indexes)
            rule_line = (
                f"本规则为 {len(pokemon)} 选 {selection_size}："
                f"只填写 {selection_size} 个互不重复的编号，不是给全部成员排序。"
            )
            lead_count = sum(bool(mon.get("active")) for mon in pokemon)
            if lead_count == 1:
                rule_line += "第一个编号是首发，其余为替补顺序。"
            elif lead_count == 2:
                rule_line += "前两个编号是首发，其余为替补顺序。"
            elif lead_count > 2:
                rule_line += f"前 {lead_count} 个编号是首发，其余为替补顺序。"
            instruction = (
                f"发送：team 使用默认选择；或 team {example}（示例）"
                f"选择 {selection_size} 只并指定出场顺序。"
            )
            return "【队伍预览】\n" + "\n".join(names) + f"\n{rule_line}\n{instruction}"

        example_indexes = list(range(1, len(pokemon) + 1))
        example = self._team_order_example(example_indexes)
        instruction = "发送：team 使用默认顺序。"
        if example:
            instruction = f"发送：team 使用默认顺序；或 team {example} 指定顺序。"
        return "【队伍预览】\n" + "\n".join(names) + f"\n{instruction}"

    def build_switch_prompt(self, request: dict) -> str:
        available = self._build_switch_option_lines(request)
        tips = "\n".join(available) if available else "暂无可用替换"
        return (
            "【需换人】\n"
            + tips
            + "\n发送：switch <编号> 完成换人；双打可用 switch1/switch2 指定位置，多个操作用分号隔开：switch1 3; switch2 4。或输入 forfeit 认输。"
        )

    def build_move_prompt(self, request: dict) -> str:
        active = request.get("active") or []
        if not active:
            return "暂不需要行动"
        sections: List[str] = ["【行动选择】"]

        # Try to resolve active Pokémon display names for doubles headers
        slot_names: Dict[int, str] = {}
        side_info = request.get("side") or {}
        side_pokemon = side_info.get("pokemon") or []
        if side_pokemon:
            # Collect active mons with their inferred slot order (1-based)
            from .utils import position_index_from_ident, simplify_ident as _simp

            actives = []
            for mon in side_pokemon:
                if not mon.get("active"):
                    continue
                ident = mon.get("ident") or ""
                pos_index = position_index_from_ident(ident) or 0
                base = _simp(ident) if ident else ""
                display = (
                    self._translator.translate_details(mon.get("details", ""))
                    or (self._translator.translate_species(base) if base else "")
                    or base
                    or mon.get("species")
                    or "未知宝可梦"
                )
                actives.append((pos_index, display))
            # Sort by inferred slot and map
            actives.sort(key=lambda t: t[0] if t[0] > 0 else 99)
            for idx, (_, name) in enumerate(actives, start=1):
                slot_names[idx] = name

        available_modifiers: dict[str, int] = {}
        for pos, actor in enumerate(active, start=1):
            moves = actor.get("moves", [])
            move_lines = []
            for idx, move in enumerate(moves, start=1):
                status = "(不可用)" if move.get("disabled") else ""
                move_name = self._translator.translate_move(move.get("move"))
                move_lines.append(
                    f"{idx}. {move_name} PP {move.get('pp')}/{move.get('maxpp')} {status}".strip()
                )
            header_name = slot_names.get(pos)
            if header_name:
                header = f"— 位置{pos}：{header_name} —"
            else:
                header = f"— 位置{pos} —"
            extras: List[str] = []
            if actor.get("canTerastallize"):
                tera_type = self._translator.translate_type(actor["canTerastallize"])
                extras.append(f"可太晶化：{tera_type}")
                available_modifiers.setdefault("tera", pos)
            if actor.get("canZMove"):
                extras.append("可使用 Z 招式")
                available_modifiers.setdefault("zmove", pos)
            if actor.get("canMegaEvo"):
                extras.append("可 Mega 进化")
                available_modifiers.setdefault("mega", pos)
            if actor.get("canUltraBurst"):
                extras.append("可究极爆发")
                available_modifiers.setdefault("ultra", pos)
            if actor.get("canDynamax") or actor.get("maxMoves"):
                extras.append("可极巨化")
                available_modifiers.setdefault("dynamax", pos)
            if extras:
                header += "（" + "，".join(extras) + "）"
            sections.append(header)
            sections.extend(move_lines)
        switch_options = self._build_switch_option_lines(request)
        if switch_options:
            sections.append("【可换入队友】")
            sections.extend(switch_options)
        sections.append(
            "发送：单打用 move <编号>；双打用 move1 <编号> [目标位置]；"
            "move2 <编号> [目标位置]。可与 switch1/switch2 组合，"
            "多个操作用逗号隔开。例如：move1 1 1, move2 2。"
        )
        unique_modifiers = tuple(available_modifiers)
        if unique_modifiers:
            labels = {
                "tera": "太晶化",
                "mega": "Mega 进化",
                "zmove": "Z 招式",
                "ultra": "究极爆发",
                "dynamax": "极巨化",
            }
            modifier_text = "、".join(
                f"{modifier}（{labels[modifier]}）" for modifier in unique_modifiers
            )
            example_modifier = unique_modifiers[0]
            example_actor = available_modifiers[example_modifier]
            example = (
                f"move{example_actor} 1 1 {example_modifier}"
                if len(active) > 1
                else f"move 1 {example_modifier}"
            )
            sections.append(
                f"特殊机制：当前可用 {modifier_text}；"
                f"在对应 move 指令末尾添加关键字，例如 {example}。"
                "仅在该位置提示可用时使用，每回合最多选择一种一次性机制。"
            )
        sections.append(
            "其他：switch <编号> 换人；forfeit 认输；check <编号> 查看招式详情；发送“状态”查看当前在场状态。"
        )
        return "\n".join(sections)

    @staticmethod
    def _team_order_example(indexes: Sequence[int]) -> str:
        if not indexes:
            return ""
        if max(indexes) >= 10:
            return ",".join(str(index) for index in indexes)
        return "".join(str(index) for index in indexes)

    def _build_switch_option_lines(self, request: dict) -> List[str]:
        side_info = request.get("side") or {}
        pokemon = side_info.get("pokemon") or []
        available: List[str] = []
        for idx, mon in enumerate(pokemon, start=1):
            if mon.get("active"):
                continue
            condition = str(mon.get("condition") or "").strip()
            if "fnt" in condition.lower():
                continue
            ident = mon.get("ident") or ""
            details = mon.get("details") or ""
            base_name = simplify_ident(ident) if ident else ""
            name = (
                self._translator.translate_details(details)
                or (self._translator.translate_species(base_name) if base_name else "")
                or details
                or base_name
                or "未知宝可梦"
            )
            condition_display = condition or "状态未知"
            available.append(f"{idx}. {name} [{condition_display}]")
        return available

    def build_full_team_summary(self, request: dict) -> Optional[str]:
        return self._build_team_summary(request, title="【队伍详情】")

    def build_random_team_summary(self, request: dict) -> Optional[str]:
        return self._build_team_summary(request, title="【随机队伍详情】")

    def _build_team_summary(self, request: dict, *, title: str) -> Optional[str]:
        side_info = request.get("side") or {}
        pokemon_list = side_info.get("pokemon") or []
        if not pokemon_list:
            return None
        lines = [title]
        for idx, mon in enumerate(pokemon_list, start=1):
            ident = mon.get("ident") or ""
            details = mon.get("details") or ""
            translated_details = self._translator.translate_details(details)
            base_name = simplify_ident(ident)
            fallback_name = (
                self._translator.translate_species(base_name) if base_name else ""
            )
            name = (
                translated_details
                or fallback_name
                or details
                or base_name
                or ident
                or "未知宝可梦"
            )
            condition = mon.get("condition") or ""
            hp_text, status_text = parse_hp_and_status(condition)
            condition_bits: List[str] = []
            if hp_text:
                condition_bits.append(f"HP {hp_text}")
            if status_text:
                condition_bits.append(status_text)
            header = f"{idx}. {name}"
            if condition_bits:
                header += f"（{'，'.join(condition_bits)}）"
            lines.append(header)

            item = mon.get("item") or ""
            ability = mon.get("baseAbility") or mon.get("ability") or ""
            tera_type = mon.get("teraType") or ""
            nature = mon.get("nature") or ""

            if item:
                item_display = self._translator.translate_item(item) or item
            else:
                item_display = "无"
            if ability:
                ability_display = self._translator.translate_ability(ability) or ability
            else:
                ability_display = "未知"
            meta_parts: List[str] = [
                f"道具：{item_display}",
                f"特性：{ability_display}",
            ]
            if tera_type:
                tera_display = self._translator.translate_type(tera_type) or tera_type
                meta_parts.append(f"太晶：{tera_display}")
            if nature:
                nature_display = self._translator.translate(nature) or nature
                meta_parts.append(f"性格：{nature_display}")
            if meta_parts:
                lines.append("  " + "  ".join(meta_parts))

            move_tokens: Sequence[object] = mon.get("moves") or []
            move_names: List[str] = []
            for token in move_tokens:
                move_id = ""
                move_label = ""
                if isinstance(token, str):
                    move_id = token
                elif isinstance(token, dict):
                    move_id = token.get("id") or token.get("move") or ""
                    move_label = token.get("move") or ""
                if move_id:
                    entry = self._move_repository.get(move_id)
                    english_name = entry.data.get("name") if entry else ""
                    english_name = english_name or move_id
                else:
                    english_name = move_label
                translated_move = (
                    self._translator.translate_move(english_name)
                    if english_name
                    else ""
                )
                move_display = translated_move or english_name
                if move_display:
                    move_names.append(move_display)
            if move_names:
                lines.append("  招式：" + " / ".join(move_names))
        return "\n".join(lines)
