from __future__ import annotations

from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from ..text_assets import (
    EFFECT_CAUSE_TEXT,
    FIELD_TEXT,
    SIDE_CONDITION_TEXT,
    STAT_TEXT,
    STATUS_TEXT,
    TERRAIN_EFFECTS,
    VOLATILE_TEXT,
    WEATHER_TEXT,
)
from .state import BattleState, PlayerSlot
from .utils import (
    parse_hp_and_status,
    position_index_from_ident,
    position_index_from_slot,
    simplify_ident,
    slot_token_from_ident,
)

if TYPE_CHECKING:
    from ..translations import TranslationService


class BattleEventFormatter:
    """Translate raw showdown log lines into localized battle narration."""

    def __init__(
        self,
        *,
        state: BattleState,
        translator: "TranslationService",
        players: Dict[str, PlayerSlot],
    ) -> None:
        self._state = state
        self._translator = translator
        self._players = players
        self._handlers = {
            "teamsize": self._handle_teamsize,
            "turn": self._handle_turn,
            "switch": self._handle_switch,
            "drag": self._handle_switch,
            "detailschange": self._handle_details_change,
            "move": self._handle_move,
            "-damage": self._handle_damage,
            "-heal": self._handle_heal,
            "-sethp": self._handle_sethp,
            "faint": self._handle_faint,
            "-status": self._handle_status,
            "-curestatus": self._handle_cure_status,
            "-cureteam": self._handle_cure_team,
            "-boost": self._handle_boost,
            "-unboost": self._handle_unboost,
            "-lower": self._handle_unboost,
            "-clearboost": self._handle_clear_boost,
            "-clearallboost": self._handle_clear_all_boost,
            "-weather": self._handle_weather,
            "-fieldstart": self._handle_field_start,
            "-fieldend": self._handle_field_end,
            "-sidestart": self._handle_side_start,
            "-sideend": self._handle_side_end,
            "-ability": self._handle_ability,
            "-item": self._handle_item_gain,
            "-enditem": self._handle_item_loss,
            "-terastallize": self._handle_terastallize,
            "-mega": self._handle_mega,
            "-primal": self._handle_primal,
            "-formechange": self._handle_forme_change,
            "-start": self._handle_start,
            "-end": self._handle_end,
            "-crit": self._handle_crit,
            "-supereffective": self._handle_supereffective,
            "-resisted": self._handle_resisted,
            "-immune": self._handle_immune,
            "-fail": self._handle_fail,
            "cant": self._handle_cant,
            "-activate": self._handle_activate,
            "-message": self._handle_message,
            "-hint": self._handle_hint,
            "-fieldactivate": self._handle_field_activate,
            "swap": self._handle_swap,
        }

    def format(self, line: str) -> Optional[str]:
        parts = line.split("|")
        if len(parts) < 2:
            return None
        cmd = parts[1]
        handler = self._handlers.get(cmd)
        if not handler:
            return None
        return handler(parts)

    def _handle_teamsize(self, parts: Sequence[str]) -> Optional[str]:
        """Track roster sizes announced at battle start (silent update)."""
        if len(parts) < 4:
            return None
        side = parts[2].strip().lower()
        try:
            size = int(parts[3])
        except ValueError:
            return None
        if side in {"p1", "p2"} and size > 0:
            self._state.team_sizes[side] = size
        return None

    def _describe_effect(self, effect: str) -> str:
        """Translate a raw ``[from]`` effect token into localized text.

        Tokens are either categorized (``item: Leftovers``), bare condition
        names (``Stealth Rock``, ``Sandstorm``), status ids (``psn``) or
        simulator-internal causes (``recoil``).
        """
        effect = (effect or "").strip()
        if not effect:
            return ""
        if ":" in effect:
            prefix, _, name = effect.partition(":")
            name = name.strip()
            kind = prefix.strip().lower()
            if kind == "item":
                return f"道具 {self._translator.translate_item(name) or name}"
            if kind == "ability":
                return f"特性 {self._translator.translate_ability(name) or name}"
            if kind == "move":
                return f"招式 {self._translator.translate_move(name) or name}"
        for table in (
            STATUS_TEXT,
            EFFECT_CAUSE_TEXT,
            SIDE_CONDITION_TEXT,
            WEATHER_TEXT,
            FIELD_TEXT,
            VOLATILE_TEXT,
        ):
            if effect in table:
                return table[effect]
        translated = (
            self._translator.translate_move(effect)
            or self._translator.translate_item(effect)
            or self._translator.translate_ability(effect)
        )
        return translated or self._translator.translate(effect)

    def _plain_pokemon_name(self, ident: str) -> str:
        """Pokémon display name without the owner prefix."""
        ident = (ident or "").strip()
        if not ident:
            return ""
        state = self._state.pokemon.get(ident)
        if state and state.name:
            return state.name
        base = simplify_ident(ident)
        return self._translator.translate_species(base) or base

    def _extract_cause(self, tokens: Sequence[str]) -> Optional[str]:
        """Build a localized cause phrase from ``[from]``/``[of]`` tags."""
        effect = ""
        source_ident = ""
        for token in tokens:
            token = (token or "").strip()
            if token.startswith("[from]"):
                effect = token[len("[from]"):].strip()
            elif token.startswith("[of]"):
                source_ident = token[len("[of]"):].strip()
        if not effect:
            return None
        display = self._describe_effect(effect)
        if not display:
            return None
        if source_ident:
            source_name = self._plain_pokemon_name(source_ident)
            if source_name:
                return f"{source_name}的{display}"
        return display

    def _handle_turn(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        try:
            self._state.turn_number = int(parts[2])
        except ValueError:
            pass
        return f"—— 第 {parts[2]} 回合 ——"

    def _handle_switch(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        details = parts[3]
        hp_chunk = parts[4] if len(parts) > 4 else ""
        pokemon = self._format_pokemon_name(ident)
        detail_text = self._translator.translate_details(details)
        detail_segment = f" ({detail_text})" if detail_text else ""
        state = self._state.register_pokemon(ident, details, self._translator)
        state.active = True
        state.position_index = position_index_from_ident(ident)
        if hp_chunk:
            state.apply_condition(hp_chunk)
        side = self._state.side_from_ident(ident)
        if side:
            self._state.set_active_pokemon(side, ident)
        bits = [f"{pokemon} 上场{detail_segment}"]
        hp_text, status_text = parse_hp_and_status(hp_chunk)
        if hp_text:
            bits.append(hp_text)
        if status_text and status_text != "倒下":
            bits.append(status_text)
        return " ".join(bits).strip()

    def _handle_details_change(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        details = parts[3]
        state = self._state.get_or_create_pokemon(ident, self._translator)
        state.details = details
        species = details.split(",", 1)[0].strip()
        if species:
            state.name = self._translator.translate_species(species) or species
        if len(parts) > 4 and parts[4] and not parts[4].startswith("["):
            state.apply_condition(parts[4])
        return None

    def _handle_move(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        move = self._translator.translate_move(parts[3])
        target_raw = ""
        extra_tokens: List[str] = []
        for token in parts[4:]:
            if not target_raw and token and not token.startswith("["):
                target_raw = token
            else:
                extra_tokens.append(token)
        attacker = self._format_pokemon_name(ident)
        target = self._format_pokemon_name(target_raw) if target_raw else ""
        attacker_state = self._state.get_or_create_pokemon(ident, self._translator)
        attacker_state.active = True
        if attacker_state.position_index is None:
            attacker_state.position_index = position_index_from_ident(ident)
        tags = self._extract_effect_tags(extra_tokens)
        if target:
            base = f"{attacker} 使用了 {move}，目标 {target}"
        else:
            base = f"{attacker} 使用了 {move}"
        if tags:
            base += f"（{'，'.join(tags)}）"
        return base

    def _handle_swap(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        position_token = parts[3]
        pokemon = self._format_pokemon_name(ident)
        state = self._state.get_or_create_pokemon(ident, self._translator)
        state.active = True
        if position_token.isdigit():
            new_index = int(position_token) + 1
        else:
            new_index = position_index_from_slot(slot_token_from_ident(position_token))
        if len(parts) > 4 and parts[4]:
            state.apply_condition(parts[4])
        self._state.swap_positions(ident, new_index)
        if new_index:
            return f"{pokemon} 调整至位置{new_index}"
        return f"{pokemon} 调整站位"

    def _handle_damage(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        hp_text, status_text = parse_hp_and_status(parts[3])
        pokemon = self._format_pokemon_name(ident)
        state = self._state.get_or_create_pokemon(ident, self._translator)
        if state.position_index is None:
            state.position_index = position_index_from_ident(ident)
        state.apply_condition(parts[3])
        cause = self._extract_cause(parts[4:])
        if cause:
            message = f"{pokemon} 受到{cause}的伤害，HP {hp_text}"
        else:
            message = f"{pokemon} HP {hp_text}"
        if status_text:
            message += f"（{status_text}）"
        return message

    def _handle_heal(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        hp_text, status_text = parse_hp_and_status(parts[3])
        pokemon = self._format_pokemon_name(ident)
        state = self._state.get_or_create_pokemon(ident, self._translator)
        if state.position_index is None:
            state.position_index = position_index_from_ident(ident)
        state.apply_condition(parts[3])
        cause = self._extract_cause(parts[4:])
        if cause:
            message = f"{pokemon} 通过{cause}回复了体力，HP {hp_text}"
        else:
            message = f"{pokemon} 恢复至 {hp_text}"
        if status_text:
            message += f"（{status_text}）"
        return message

    def _handle_faint(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        ident = parts[2]
        pokemon = self._format_pokemon_name(ident)
        state = self._state.get_or_create_pokemon(ident, self._translator)
        state.hp = "0"
        state.status = "倒下"
        state.fainted = True
        if state.position_index is None:
            state.position_index = position_index_from_ident(ident)
        return f"{pokemon} 倒下了"

    def _handle_sethp(self, parts: Sequence[str]) -> Optional[str]:
        """Handle explicit HP set events (e.g. recoil rounding or effects that set HP).

        Showdown sometimes emits `|-sethp|` lines to directly set the current HP
        rather than as incremental damage/heal. Treat it the same as a damage/heal
        update for our internal state and narration.
        """
        if len(parts) < 4:
            return None
        ident = parts[2]
        hp_text, status_text = parse_hp_and_status(parts[3])
        pokemon = self._format_pokemon_name(ident)
        state = self._state.get_or_create_pokemon(ident, self._translator)
        if state.position_index is None:
            state.position_index = position_index_from_ident(ident)
        state.apply_condition(parts[3])
        message = f"{pokemon} HP 设为 {hp_text}"
        cause = self._extract_cause(parts[4:])
        if cause:
            message += f"（{cause}）"
        if status_text:
            message += f"（{status_text}）"
        return message

    def _handle_status(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        status_key = parts[3]
        status = STATUS_TEXT.get(status_key, status_key)
        pokemon = self._format_pokemon_name(ident)
        state = self._state.get_or_create_pokemon(ident, self._translator)
        state.status = status
        if status == "倒下":
            state.fainted = True
        if state.position_index is None:
            state.position_index = position_index_from_ident(ident)
        message = f"{pokemon} 陷入 {status}"
        cause = self._extract_cause(parts[4:])
        if cause:
            message += f"（{cause}）"
        return message

    def _handle_cure_status(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        status_key = parts[3]
        pokemon = self._format_pokemon_name(ident)
        state = self._state.get_or_create_pokemon(ident, self._translator)
        status = STATUS_TEXT.get(status_key, status_key)
        if state.status == status:
            state.status = None
        return f"{pokemon} 摆脱了 {status}"

    def _handle_cure_team(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        side = parts[2]
        side_name = self._format_side_name(side)
        for state in self._state.pokemon.values():
            if state.side == side and state.status != "倒下":
                state.status = None
        return f"{side_name} 的队伍状态被治愈"

    def _handle_boost(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 5:
            return None
        return self._format_boost_change(parts[2], parts[3], parts[4], increase=True)

    def _handle_unboost(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 5:
            return None
        return self._format_boost_change(parts[2], parts[3], parts[4], increase=False)

    def _handle_clear_boost(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        self._state.clear_boosts(parts[2])
        return f"{pokemon} 的能力变化被清除"

    def _handle_clear_all_boost(self, parts: Sequence[str]) -> Optional[str]:
        self._state.clear_boosts()
        return "双方的能力变化被清除"

    def _handle_weather(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        weather = parts[2]
        extras = list(parts[3:])
        return self._format_weather(weather, extras)

    def _handle_field_start(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        return self._format_field_condition(parts[2], start=True)

    def _handle_field_end(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        return self._format_field_condition(parts[2], start=False)

    def _handle_side_start(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        effect = parts[3]
        effect_key = effect.split(": ", 1)[1] if effect.startswith("move: ") else effect
        return self._format_side_condition(parts[2], effect_key, start=True)

    def _handle_side_end(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        effect = parts[3]
        effect_key = effect.split(": ", 1)[1] if effect.startswith("move: ") else effect
        return self._format_side_condition(parts[2], effect_key, start=False)

    def _handle_ability(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        ability = self._translator.translate_ability(parts[3])
        return f"{pokemon} 触发特性 {ability}"

    def _handle_item_gain(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        item = self._translator.translate_item(parts[3])
        extras = [token.strip() for token in parts[4:]]
        cause = self._extract_cause(extras)
        # ``[identify]`` marks reveals (e.g. Frisk) rather than actual gains.
        if "[identify]" in extras:
            message = f"{pokemon} 被发现携带道具 {item}"
        else:
            message = f"{pokemon} 获得了道具 {item}"
        if cause:
            message += f"（{cause}）"
        return message

    def _handle_item_loss(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        item = self._translator.translate_item(parts[3])
        extras = [token.strip() for token in parts[4:]]
        if "[eat]" in extras:
            message = f"{pokemon} 吃掉了 {item}"
        else:
            message = f"{pokemon} 失去了道具 {item}"
        cause = self._extract_cause(extras)
        if cause:
            message += f"（{cause}）"
        return message

    def _handle_terastallize(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        pokemon = self._format_pokemon_name(ident)
        tera_type = self._translator.translate_type(parts[3])
        state = self._state.get_or_create_pokemon(ident, self._translator)
        state.tera_type = tera_type
        return f"{pokemon} 太晶化为 {tera_type}"

    def _handle_mega(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        # Current Showdown protocol: |-mega|ident|base species|mega stone.
        stone_token = parts[4] if len(parts) > 4 and parts[4] else parts[3]
        stone = self._translator.translate_item(stone_token)
        return f"{pokemon} 借助 {stone} 完成了超级进化"

    def _handle_primal(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        return f"{pokemon} 完成了原始回归"

    def _handle_forme_change(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        species = parts[3]
        state = self._state.get_or_create_pokemon(ident, self._translator)
        translated = self._translator.translate_species(species)
        state.name = translated or species
        state.details = species
        if len(parts) > 4 and parts[4] and not parts[4].startswith("["):
            state.apply_condition(parts[4])
        owner = self._format_side_name(state.side)
        display = f"{owner} 的 {state.name}" if owner else state.name
        return f"{display} 改变了形态"

    def _handle_start(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        pokemon = self._format_pokemon_name(ident)
        effect = parts[3]
        state = self._state.get_or_create_pokemon(ident, self._translator)
        if effect == "Dynamax":
            state.volatiles.add("Dynamax")
            return f"{pokemon} 开始极巨化"
        if effect.startswith("typechange") and len(parts) >= 5:
            new_type = self._translator.translate_type(parts[4])
            return f"{pokemon} 的属性变为 {new_type}"
        if effect == "Substitute":
            state.volatiles.add("Substitute")
            return f"{pokemon} 招出替身"
        if effect == "Terastallized":
            return f"{pokemon} 进入太晶化状态"
        return None

    def _handle_end(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        ident = parts[2]
        pokemon = self._format_pokemon_name(ident)
        effect = parts[3]
        state = self._state.get_or_create_pokemon(ident, self._translator)
        if effect == "Dynamax":
            state.volatiles.discard("Dynamax")
            return f"{pokemon} 恢复为普通形态"
        if effect == "Substitute":
            state.volatiles.discard("Substitute")
            return f"{pokemon} 的替身消失了"
        return None

    def _handle_fail(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        move = self._translator.translate_move(parts[3])
        return f"{pokemon} 的 {move} 没能成功"

    def _handle_cant(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        reason = parts[3]
        status_text = STATUS_TEXT.get(reason)
        if status_text:
            return f"{pokemon} 因为{status_text}无法行动"
        if reason.startswith("move: "):
            move_name = self._translator.translate_move(reason.split(": ", 1)[1])
            extra = ""
            if len(parts) >= 5 and parts[4] in STATUS_TEXT:
                extra = STATUS_TEXT[parts[4]]
            elif len(parts) >= 5:
                extra = self._translator.translate(parts[4])
            detail = f"（{extra}）" if extra else ""
            return f"{pokemon} 因为 {move_name} 无法行动{detail}"
        reason_text = self._translator.translate(reason)
        return f"{pokemon} 无法行动（{reason_text}）"

    def _handle_activate(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 4:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        effect = parts[3]
        extras = list(parts[4:])
        suffix = ""
        if effect.startswith("ability: "):
            ability_name = self._translator.translate_ability(effect.split(": ", 1)[1])
            suffix = f"的特性 {ability_name}"
        elif effect.startswith("move: "):
            move_name = self._translator.translate_move(effect.split(": ", 1)[1])
            suffix = f"的招式 {move_name}"
        elif effect.startswith("item: "):
            item_name = self._translator.translate_item(effect.split(": ", 1)[1])
            suffix = f"的道具 {item_name}"
        else:
            suffix = self._translator.translate(effect)
        tags = self._extract_effect_tags(extras)
        tag_text = f"（{'，'.join(tags)}）" if tags else ""
        if suffix.startswith("的"):
            return f"{pokemon} {suffix} 发动{tag_text}"
        return f"{pokemon} 触发了 {suffix}{tag_text}"

    def _handle_message(self, parts: Sequence[str]) -> Optional[str]:
        return parts[2] if len(parts) >= 3 else None

    def _handle_hint(self, parts: Sequence[str]) -> Optional[str]:
        return f"提示：{parts[2]}" if len(parts) >= 3 else None

    def _handle_field_activate(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        effect = parts[2]
        if effect.startswith("move: "):
            effect = effect.split(": ", 1)[1]
        effect_name = FIELD_TEXT.get(effect, self._translator.translate(effect))
        return f"场地效果 {effect_name} 被触发"

    def _handle_crit(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        return f"{pokemon} 命中要害！"

    def _handle_supereffective(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        return f"对 {pokemon} 取得了绝佳效果！"

    def _handle_resisted(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        return f"{pokemon} 抵挡了部分伤害"

    def _handle_immune(self, parts: Sequence[str]) -> Optional[str]:
        if len(parts) < 3:
            return None
        pokemon = self._format_pokemon_name(parts[2])
        return f"{pokemon} 完全无效！"

    def _format_boost_change(
        self, ident: str, stat: str, amount: str, *, increase: bool
    ) -> str:
        pokemon = self._format_pokemon_name(ident)
        stat_name = STAT_TEXT.get(stat, stat)
        symbol = "+" if increase else "-"
        self._state.update_boost(ident, stat, amount, increase=increase)
        return f"{pokemon} 的 {stat_name} {symbol}{amount}"

    def _format_side_condition(self, side: str, effect: str, *, start: bool) -> str:
        side_name = self._format_side_name(side)
        side_token = self._state.side_from_ident(side) or side
        conditions = self._state.field_state.side_conditions.setdefault(side_token, {})
        if start:
            conditions[effect] = max(conditions.get(effect, 0), 1)
        else:
            conditions.pop(effect, None)
        effect_name = SIDE_CONDITION_TEXT.get(
            effect, self._translator.translate(effect)
        )
        if start:
            if side_name:
                return f"{side_name} 场地出现 {effect_name}"
            return f"场地出现 {effect_name}"
        if side_name:
            return f"{side_name} 的 {effect_name} 消失"
        return f"{effect_name} 效果消失"

    def _format_field_condition(self, effect: str, *, start: bool) -> str:
        if effect.startswith("move: "):
            raw_effect = effect.split(": ", 1)[1]
        else:
            raw_effect = effect
        if start:
            if raw_effect in TERRAIN_EFFECTS:
                self._state.field_state.terrain = raw_effect
            else:
                self._state.field_state.field_effects[raw_effect] = 1
        else:
            if raw_effect in TERRAIN_EFFECTS:
                if self._state.field_state.terrain == raw_effect:
                    self._state.field_state.terrain = None
            else:
                self._state.field_state.field_effects.pop(raw_effect, None)
        effect_name = FIELD_TEXT.get(raw_effect, self._translator.translate(raw_effect))
        return f"场地 {('设立' if start else '结束')} {effect_name}"

    def _format_weather(self, weather: str, extras: Sequence[str]) -> Optional[str]:
        if weather == "none":
            self._state.field_state.weather = None
            return "天气效果解除"
        if any(token == "[upkeep]" for token in extras):
            if weather != "none":
                self._state.field_state.weather = weather
            return None
        self._state.field_state.weather = weather
        weather_name = WEATHER_TEXT.get(weather, self._translator.translate(weather))
        return f"天气变为 {weather_name}"

    def _format_side_name(self, side: Optional[str]) -> str:
        if not side:
            return ""
        player = self._players.get(side)
        return player.display_name if player else side

    def _format_pokemon_name(self, ident: str) -> str:
        ident = ident or ""
        state = self._state.pokemon.get(ident)
        base = simplify_ident(ident)
        target = base or ident
        name = state.name if state and state.name else ""
        if not name:
            name = self._translator.translate_species(target) if target else ""
        if not name:
            name = target
        side = self._state.side_from_ident(ident)
        owner = self._format_side_name(side)
        if owner:
            return f"{owner} 的 {name}"
        return name

    def _extract_effect_tags(self, tokens: Sequence[str]) -> List[str]:
        tags: List[str] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if not token:
                idx += 1
                continue
            if token == "[miss]":
                tags.append("未命中")
            elif token == "[crit]":
                tags.append("命中要害")
            elif token == "[ohko]":
                tags.append("一击必杀")
            elif token == "[blocked]":
                tags.append("被挡下")
            elif token == "[fail]":
                tags.append("失败")
            elif token.startswith("[from] ability: "):
                ability = token.split(": ", 1)[1]
                ability_name = self._translator.translate_ability(ability)
                tags.append(f"特性 {ability_name}")
            elif token.startswith("[from] item: "):
                item = token.split(": ", 1)[1]
                item_name = self._translator.translate_item(item)
                tags.append(f"道具 {item_name}")
            elif token.startswith("[from] move: "):
                move = token.split(": ", 1)[1]
                move_name = self._translator.translate_move(move)
                tags.append(f"招式 {move_name}")
            elif (
                token.startswith("[from] ability")
                and idx + 1 < len(tokens)
                and tokens[idx + 1].startswith("[of]")
            ):
                ability = token.split(": ", 1)[1]
                ability_name = self._translator.translate_ability(ability)
                ident = (
                    tokens[idx + 1].split(" ", 1)[1] if " " in tokens[idx + 1] else ""
                )
                source = self._format_pokemon_name(ident)
                tags.append(f"{source} 的特性 {ability_name}")
                idx += 1
            idx += 1
        return tags
