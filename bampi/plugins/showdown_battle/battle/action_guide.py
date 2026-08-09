from __future__ import annotations

from typing import Any


_TARGETS_REQUIRING_FOE = {"adjacentFoe"}
_TARGETS_ALLOWING_ANY_ADJACENT = {"normal", "any"}
_TARGETS_REQUIRING_ALLY = {"adjacentAlly"}
_TARGETS_ALLOWING_ALLY_OR_SELF = {"adjacentAllyOrSelf"}


def build_ai_action_guide(
    request: dict[str, Any],
    *,
    picked_team_size: int | None = None,
) -> dict[str, Any]:
    """Describe the accepted choice syntax from one authoritative request.

    The guide deliberately mirrors :meth:`BattleSession._parse_choice` while
    retaining Showdown's request constraints. It is injected automatically, so
    the model never needs to know this bot's command dialect in advance.
    """

    common: dict[str, Any] = {
        "submission_tool": "choose_battle_action",
        "choice_value": (
            "只填写一条原始对战命令，不要添加斜杠、Markdown、解释、引号或发言。"
        ),
        "indexing": (
            "所有编号均从 1 开始，并且只对应当前 your_private_request 中的数组位置；"
            "命令中使用编号，不使用宝可梦名或招式名。"
        ),
        "not_action_commands": (
            "check_move 是独立查询工具；不要把 check、状态、自然语言或工具名作为 choice 提交。"
        ),
        "retry_rule": "工具拒绝命令时，根据错误修正并在当前回合重新调用。",
        "forfeit": "forfeit 始终表示主动认输；除非确实决定结束对局，否则不要使用。",
    }

    if request.get("teamPreview"):
        return {
            **common,
            "request_type": "team_preview",
            "rules": _build_team_preview_rules(
                request,
                picked_team_size=picked_team_size,
            ),
        }

    if request.get("forceSwitch"):
        return {
            **common,
            "request_type": "forced_switch",
            "rules": _build_forced_switch_rules(request),
        }

    active = request.get("active") or []
    if active:
        return {
            **common,
            "request_type": "move_or_switch",
            "rules": _build_move_rules(request, active),
        }

    return {
        **common,
        "request_type": "unknown",
        "rules": [
            "当前 request 没有可识别的 teamPreview、forceSwitch 或 active 行动。",
            "不要猜测命令；让本轮进入本地安全回退。",
        ],
    }


def _build_team_preview_rules(
    request: dict[str, Any],
    *,
    picked_team_size: int | None,
) -> dict[str, Any]:
    pokemon = (request.get("side") or {}).get("pokemon") or []
    team_size = len(pokemon)
    raw_selection_size = (
        request.get("maxChosenTeamSize")
        or request.get("maxTeamSize")
        or request.get("maxTeam")
        or picked_team_size
    )
    selection_size = (
        int(raw_selection_size)
        if isinstance(raw_selection_size, int) and raw_selection_size > 0
        else team_size
    )
    members = [
        {
            "index": index,
            "ident": mon.get("ident") or mon.get("details") or "未知",
        }
        for index, mon in enumerate(pokemon, start=1)
    ]
    return {
        "members": members,
        "default_command": "team",
        "custom_command_syntax": "team <连续编号>",
        "selection_size": selection_size,
        "custom_command_rules": [
            f"<连续编号> 必须恰好包含 {selection_size} 个互不重复的数字。",
            f"每个数字必须在 1 到 {team_size} 之间。",
            "数字顺序同时决定选择哪些成员以及首发/后续顺序。",
            "示例仅说明语法：team 2134。必须按当前队伍数量和 selection_size 调整。",
        ],
    }


def _build_forced_switch_rules(request: dict[str, Any]) -> dict[str, Any]:
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list):
        required_slots = [bool(value) for value in force_switch]
    else:
        required_slots = [bool(force_switch)]
    slot_count = max(1, len(required_slots))
    switches = _available_switches(request)
    slots: list[dict[str, Any]] = []
    for slot in range(1, slot_count + 1):
        required = required_slots[slot - 1] if slot <= len(required_slots) else False
        prefix = "switch" if slot_count == 1 else f"switch{slot}"
        slot_rules: dict[str, Any] = {
            "actor": slot,
            "must_switch": required,
        }
        if required:
            slot_rules["allowed_commands"] = [
                f"{prefix} {item['pokemon_index']}" for item in switches
            ]
        else:
            slot_rules["allowed_commands"] = [
                "pass" if slot_count == 1 else f"pass{slot}"
            ]
        slots.append(slot_rules)
    return {
        "available_bench_pokemon": switches,
        "active_slots": slots,
        "combined_choice": (
            "单打只提交该位置的一条命令。"
            if slot_count == 1
            else "按位置顺序提交全部位置，用英文逗号分隔，例如：switch1 3, pass2。"
        ),
        "constraints": [
            "must_switch=true 的位置必须选择 allowed_commands 中的一条。",
            "同一只替补宝可梦不能同时分配给两个位置。",
            "must_switch=false 的位置必须 pass，不要为它擅自换人。",
        ],
    }


def _build_move_rules(
    request: dict[str, Any],
    active: list[dict[str, Any]],
) -> dict[str, Any]:
    slot_count = len(active)
    switches = _available_switches(request)
    slots: list[dict[str, Any]] = []
    for slot, actor in enumerate(active, start=1):
        move_prefix = "move" if slot_count == 1 else f"move{slot}"
        switch_prefix = "switch" if slot_count == 1 else f"switch{slot}"
        moves: list[dict[str, Any]] = []
        z_moves = actor.get("canZMove") or []
        for index, move in enumerate(actor.get("moves") or [], start=1):
            disabled = bool(move.get("disabled"))
            pp = _coerce_int(move.get("pp"))
            usable = not disabled and (pp is None or pp > 0)
            modifiers = _move_modifiers(actor, z_moves, move_index=index)
            moves.append(
                {
                    "move_index": index,
                    "move_name": move.get("move") or move.get("id") or "未知",
                    "usable": usable,
                    "unusable_reason": (
                        "disabled=true"
                        if disabled
                        else ("PP=0" if pp is not None and pp <= 0 else None)
                    ),
                    "command": f"{move_prefix} {index}",
                    "target_argument": _target_argument(
                        move.get("target"),
                        actor_slot=slot,
                        slot_count=slot_count,
                    ),
                    "allowed_modifiers": modifiers,
                }
            )

        trapped = bool(actor.get("trapped"))
        slots.append(
            {
                "actor": slot,
                "must_submit_one_action": True,
                "moves": moves,
                "switch_commands": (
                    []
                    if trapped
                    else [
                        f"{switch_prefix} {item['pokemon_index']}" for item in switches
                    ]
                ),
                "switch_blocked": trapped,
                "switch_uncertain": bool(actor.get("maybeTrapped")) and not trapped,
            }
        )

    return {
        "active_slots": slots,
        "available_bench_pokemon": switches,
        "combined_choice": (
            "单打：只提交该位置的一条 move 或 switch 命令。"
            if slot_count == 1
            else (
                "双打：位置 1 和位置 2 各提交恰好一个动作，按位置顺序用英文逗号分隔；"
                "例如 move1 2 1, switch2 4。不要漏掉任一位置。"
            )
        ),
        "move_command_rules": [
            "只能选择 usable=true 的招式。",
            "target_argument.required=true 时，必须在招式编号后添加 allowed_values 中的一个目标位置。",
            "正目标 1/2 表示对手的位置；负目标 -1/-2 表示己方的位置。",
            "需要特殊机制时，把 allowed_modifiers 中的单词放在整条 move 命令末尾；"
            "整回合最多让一只宝可梦使用一次太晶化/Mega/Z/极巨化等一次性机制。",
        ],
        "switch_command_rules": [
            "只能使用 switch_commands 中列出的命令。",
            "双打时，同一只替补不能被两个位置同时选择。",
            "switch_uncertain=true 表示可能受未完全确认的困住效果影响；工具接受后仍以 Showdown 最终校验为准。",
        ],
    }


def _available_switches(request: dict[str, Any]) -> list[dict[str, Any]]:
    pokemon = (request.get("side") or {}).get("pokemon") or []
    result: list[dict[str, Any]] = []
    for index, mon in enumerate(pokemon, start=1):
        condition = str(mon.get("condition") or "")
        if mon.get("active") or "fnt" in condition.casefold():
            continue
        result.append(
            {
                "pokemon_index": index,
                "ident": mon.get("ident") or mon.get("details") or "未知",
                "condition": condition or "未知",
            }
        )
    return result


def _target_argument(
    target: Any,
    *,
    actor_slot: int,
    slot_count: int,
) -> dict[str, Any]:
    target_name = str(target or "normal")
    if slot_count <= 1:
        return {
            "required": False,
            "allowed_values": [],
            "reason": "单打省略目标位置。",
            "showdown_target": target_name,
        }

    foe_values = list(range(1, slot_count + 1))
    ally_values = [-slot for slot in range(1, slot_count + 1) if slot != actor_slot]
    self_value = -actor_slot
    if target_name in _TARGETS_REQUIRING_FOE:
        allowed = foe_values
        reason = "该招式必须指定一个对手位置。"
    elif target_name in _TARGETS_ALLOWING_ANY_ADJACENT:
        allowed = foe_values + ally_values
        reason = "该招式可指定任一对手，或除自己外的己方相邻位置。"
    elif target_name in _TARGETS_REQUIRING_ALLY:
        allowed = ally_values
        reason = "该招式必须指定己方相邻队友。"
    elif target_name in _TARGETS_ALLOWING_ALLY_OR_SELF:
        allowed = ally_values + [self_value]
        reason = "该招式必须指定己方队友或自己。"
    else:
        allowed = []
        reason = "该招式的目标由 Showdown 自动决定，命令中不要添加目标位置。"
    return {
        "required": bool(allowed),
        "allowed_values": allowed,
        "reason": reason,
        "showdown_target": target_name,
    }


def _move_modifiers(
    actor: dict[str, Any],
    z_moves: list[Any],
    *,
    move_index: int,
) -> list[str]:
    modifiers: list[str] = []
    if actor.get("canTerastallize"):
        modifiers.append("tera")
    if actor.get("canMegaEvo"):
        modifiers.append("mega")
    if actor.get("canUltraBurst"):
        modifiers.append("ultra")
    if actor.get("canDynamax") or actor.get("maxMoves"):
        modifiers.append("dynamax")
    if move_index <= len(z_moves) and z_moves[move_index - 1]:
        modifiers.append("zmove")
    return modifiers


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = ["build_ai_action_guide"]
