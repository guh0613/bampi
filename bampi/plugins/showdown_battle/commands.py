from __future__ import annotations

from datetime import datetime

from nonebot import get_driver, logger, on_command, on_message
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    Message,
    MessageEvent,
    MessageSegment,
    PrivateMessageEvent,
)
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.rule import Rule, is_type

from .bridge import ShowdownBridgeError, ShowdownRuntime
from .config import ShowdownBattleConfig
from .formats import BattleFormatConfig, FormatRegistry
from .manager import (
    BattleAIOpponentUnavailable,
    BattleManager,
    BattleRuntimeNotReady,
    BattleSessionConflict,
)
from .move_data import MoveDataRepository, MoveEntry
from .team.editor.service import TeamEditorError
from .team.formatting import (
    format_quick_build_selections,
    format_recommended_set_option,
)
from .team.guide import TeamGuideManager
from .team.repository import TeamRecord, TeamRepository, TeamRepositoryError
from .team.sources import TeamSourceError, TeamSourceService
from .text_assets import (
    MOVE_CATEGORY_TEXT,
    MOVE_TARGET_TEXT,
    PRIVATE_CHECK_HINT,
    PRIVATE_STATUS_HINT,
)
from .translations import TranslationService


_AI_CHALLENGE_ARGUMENTS = frozenset({"bot", "ai", "机器人", "人机", "电脑"})

challenge_commands: list[type[Matcher]] = []
prepare_cmd: type[Matcher] | None = None
private_router: type[Matcher] | None = None
group_battle_router: type[Matcher] | None = None
status_cmd: type[Matcher] | None = None
check_cmd: type[Matcher] | None = None
team_manager_cmd: type[Matcher] | None = None
sample_team_cmd: type[Matcher] | None = None
recommended_set_cmd: type[Matcher] | None = None
team_guide_cmd: type[Matcher] | None = None
team_guide_router: type[Matcher] | None = None
pokemon_help_cmd: type[Matcher] | None = None


def _on_showdown_command(
    command: str,
    *,
    aliases: set[str] | None = None,
    rule: Rule | None = None,
    priority: int,
    block: bool,
) -> type[Matcher]:
    """Register slash and conservative plain-text variants for plugin commands."""
    command_start = get_driver().config.command_start
    added_plain_start = "" not in command_start
    if added_plain_start:
        command_start.add("")
    try:
        return on_command(
            command,
            aliases=aliases,
            rule=rule,
            force_whitespace=True,
            priority=priority,
            block=block,
        )
    finally:
        if added_plain_start:
            command_start.remove("")


def _is_ai_challenge_request(
    *,
    bot_self_id: str,
    bot_name: str,
    target_id: str | None,
    argument: str,
    to_me: bool,
) -> bool:
    """Recognize AI challenges after OneBot strips a leading/trailing Bot at."""

    if target_id is not None:
        return target_id == bot_self_id
    normalized_argument = argument.strip().casefold()
    return (
        to_me
        or normalized_argument == bot_name.strip().casefold()
        or normalized_argument in _AI_CHALLENGE_ARGUMENTS
    )


def _extract_display_name(member: dict) -> str:
    return (
        str(member.get("card") or "").strip()
        or str(member.get("nickname") or "").strip()
        or str(member.get("user_id") or "未知玩家")
    )


def _format_format_display(config: BattleFormatConfig | None) -> str:
    if not config:
        return "未知规则"
    return f"{config.display_name}（{config.format_id}）"


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _build_team_help(registry: FormatRegistry) -> str:
    return (
        "【队伍管理帮助】\n"
        "普通用户推荐直接发送“组队”，按数字菜单创建和管理队伍。\n"
        "发送“我的队伍”可直接打开已保存队伍列表。\n\n"
        "常用快捷命令：\n"
        "• 队伍管理 列表 [规则]\n"
        "• 队伍管理 查看 规则 队伍名\n"
        "• 队伍管理 编辑 规则 队伍名\n"
        "• 队伍管理 添加 规则\n"
        "• 队伍管理 删除 规则 队伍名\n\n"
        "发送“队伍管理 高级帮助”查看完整快捷语法。\n"
        "发送“宝可梦帮助”查看对战和组队总帮助。\n\n" + _build_format_list(registry)
    )


def _build_team_advanced_help(registry: FormatRegistry) -> str:
    return (
        "【队伍管理高级快捷命令】\n"
        "队伍管理 列表 [规则标识]\n"
        "队伍管理 保存 规则标识 队伍名 <Showdown队伍文本或链接>\n"
        "队伍管理 查看 规则标识 队伍名\n"
        "队伍管理 编辑 规则标识 队伍名\n"
        "队伍管理 添加 规则标识\n"
        "队伍管理 删除 规则标识 队伍名\n"
        "队伍管理 重命名 规则标识 旧名 新名\n"
        "队伍管理 抄队 规则标识 样例编号 队伍名\n"
        "队伍管理 一键组队 规则标识 队伍名\n"
        "队伍管理 快速 规则标识 队伍名 | 宝可梦1,宝可梦2,...\n"
        "自选配招编号示例：快龙=2。\n"
        "对战准备阶段还可私聊使用“导入队伍 使用 队伍名”等命令。\n\n"
        + _build_format_list(registry)
    )


def _build_format_list(registry: FormatRegistry) -> str:
    lines = ["可用规则："]
    for config in registry.all():
        suffix = "（随机队伍）" if not config.requires_team else ""
        lines.append(f"- {config.format_id}：{config.display_name}{suffix}")
    return "\n".join(lines)


async def _resolve_format_context(
    *,
    user_id: str,
    text: str,
    manager: BattleManager,
    registry: FormatRegistry,
) -> tuple[BattleFormatConfig | None, str]:
    stripped = text.strip()
    if not stripped:
        session = await manager.get_session_by_user(user_id)
        return (session.format_config, "") if session else (None, "")
    token, *rest = stripped.split(maxsplit=1)
    explicit = registry.resolve_format_token(token)
    if explicit:
        return explicit, rest[0].strip() if rest else ""
    session = await manager.get_session_by_user(user_id)
    if session:
        return session.format_config, text.strip()
    return None, text.strip()


async def _team_command_channel_error(
    *,
    config: ShowdownBattleConfig,
    manager: BattleManager,
    event: MessageEvent,
) -> str | None:
    if isinstance(event, PrivateMessageEvent):
        return None
    if not isinstance(event, GroupMessageEvent):
        return "当前消息类型不支持该队伍功能。"
    if not config.showdown_battle_enabled:
        return "Pokémon Showdown 对战功能当前未启用。"
    if not config.group_is_allowed(event.group_id):
        return "本群未启用 Pokémon Showdown 对战。"
    battle = await manager.get_session_by_user(event.get_user_id())
    if battle is not None and battle.state == "active":
        return "对战进行中，不能再打开队伍工具。"
    if battle is not None and battle.state == "pending":
        group_preparation_allowed = (
            battle.is_ai_battle
            and battle.group_id == event.group_id
            and battle.interaction_channel_for_user(event.get_user_id()) == "group"
        )
        if not group_preparation_allowed:
            if battle.is_ai_battle:
                return (
                    "你当前正在准备对战。若要公开完成与 "
                    f"{manager.bot_name} 的准备流程，请先在对战群发送“对战准备”；"
                    "否则请改为私聊使用。"
                )
            return "玩家对战的准备流程仅支持私聊。"
    return None


def _merge_team_warnings(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(warning for group in groups for warning in group))


def _format_team_warnings(warnings: tuple[str, ...]) -> str:
    if not warnings:
        return ""
    return "\n" + "\n".join(f"⚠️ {warning}" for warning in warnings)


def _parse_positive_index(text: str, *, label: str) -> int:
    normalized = text.strip()
    if not normalized.isdigit() or int(normalized) < 1:
        raise TeamSourceError(f"{label}必须是从 1 开始的数字。")
    return int(normalized)


def _resolve_move_entry_by_name(
    query: str,
    translator: TranslationService,
    move_repository: MoveDataRepository,
) -> MoveEntry | None:
    candidate = translator.resolve_move_name(query)
    if candidate:
        entry = move_repository.search(candidate)
        if entry:
            return entry
    return move_repository.search(query)


def _format_move_details(
    entry: MoveEntry,
    translator: TranslationService,
    *,
    current_pp: int | None = None,
    max_pp: int | None = None,
) -> str:
    data = entry.data
    text = entry.text
    english_name = str(data.get("name") or entry.move_id)
    translated_name = translator.translate_move(english_name)
    header = (
        f"【招式信息】{translated_name}（{english_name}）"
        if translated_name != english_name
        else f"【招式信息】{english_name}"
    )

    move_type = str(data.get("type") or "")
    type_display = translator.translate_type(move_type) or move_type or "-"
    category = str(data.get("category") or "")
    category_display = (
        MOVE_CATEGORY_TEXT.get(category)
        or translator.translate(category)
        or category
        or "-"
    )

    base_power = data.get("basePower")
    power_display = (
        str(int(base_power))
        if isinstance(base_power, (int, float))
        else "--"
        if base_power is None
        else str(base_power)
    )
    accuracy = data.get("accuracy")
    if accuracy is True:
        accuracy_display = "必中"
    elif isinstance(accuracy, (int, float)):
        accuracy_display = f"{int(accuracy)}%"
    else:
        accuracy_display = "--" if accuracy is None else str(accuracy)

    base_pp = data.get("pp")
    if current_pp is not None and max_pp is not None:
        pp_display = f"{current_pp}/{max_pp}"
        if base_pp is not None:
            pp_display += f"（基础 {base_pp}）"
    else:
        pp_display = str(base_pp) if base_pp is not None else "--"

    target_key = str(data.get("target") or "")
    target_display = (
        MOVE_TARGET_TEXT.get(target_key)
        or translator.translate(target_key)
        or target_key
    )
    lines = [
        header,
        f"类型：{type_display} / {category_display}",
        f"威力：{power_display}  命中：{accuracy_display}  "
        f"优先级：{data.get('priority', 0)}",
        f"PP：{pp_display}",
    ]
    if target_display:
        lines.append(f"目标：{target_display}")

    description = text.get("shortDesc") or text.get("desc") or ""
    if not description:
        for generation in range(9, 0, -1):
            historical = text.get(f"gen{generation}")
            if isinstance(historical, dict):
                description = historical.get("shortDesc") or historical.get("desc")
                if description:
                    break
    translated_description = translator.translate_move_description(
        entry.move_id,
        str(description or ""),
    )
    if translated_description:
        lines.append(f"效果：{translated_description}")
    # The in-game flavour text above often just names an effect ("变为戏法
    # 空间状态") without explaining it; Showdown's shortDesc carries the
    # actual mechanics, so surface it (localized when available) whenever it
    # adds information.
    mechanics = translator.translate(str(description)) if description else ""
    if mechanics and mechanics != translated_description:
        lines.append(f"机制：{mechanics}")
    return "\n".join(lines)


def register_commands(
    *,
    config: ShowdownBattleConfig,
    battle_manager: BattleManager,
    format_registry: FormatRegistry,
    translation_service: TranslationService,
    team_repository: TeamRepository,
    runtime: ShowdownRuntime,
    move_repository: MoveDataRepository,
    team_sources: TeamSourceService,
    team_guide: TeamGuideManager,
) -> None:
    global prepare_cmd, private_router, group_battle_router
    global status_cmd, check_cmd, team_manager_cmd
    global sample_team_cmd, recommended_set_cmd
    global team_guide_cmd, team_guide_router, pokemon_help_cmd

    private_router = on_message(
        rule=is_type(PrivateMessageEvent),
        priority=998,
        block=False,
    )

    async def _is_ai_group_choice(event: MessageEvent) -> bool:
        if not isinstance(event, GroupMessageEvent):
            return False
        session = await battle_manager.get_session_by_group(event.group_id)
        if session is None:
            return False
        text = event.get_message().extract_plain_text().strip()
        return session.can_accept_group_choice(event.get_user_id(), text)

    group_battle_router = on_message(
        rule=Rule(_is_ai_group_choice),
        priority=4,
        block=True,
    )
    prepare_cmd = _on_showdown_command(
        "对战准备",
        aliases={
            "导入队伍",
            "导入ps队伍",
            "导入ps",
            "随机ready",
            "随机就绪",
            "双打随机就绪",
            "双打随机ready",
            "双随机就绪",
            "双随机ready",
        },
        priority=5,
        block=True,
    )
    status_cmd = _on_showdown_command(
        "战况",
        aliases={"对战状态", "查看战况"},
        priority=5,
        block=True,
    )
    check_cmd = _on_showdown_command("check", priority=5, block=True)
    team_manager_cmd = _on_showdown_command(
        "队伍管理",
        aliases={"ps队伍", "ps队伍管理", "我的队伍"},
        priority=6,
        block=True,
    )
    sample_team_cmd = _on_showdown_command(
        "队伍样例",
        aliases={"样例队伍", "抄队"},
        priority=6,
        block=True,
    )
    recommended_set_cmd = _on_showdown_command(
        "推荐配招",
        aliases={"配招推荐"},
        priority=6,
        block=True,
    )
    team_guide_cmd = _on_showdown_command(
        "组队",
        aliases={"组队向导", "快速组队"},
        priority=5,
        block=True,
    )
    pokemon_help_cmd = _on_showdown_command(
        "宝可梦帮助",
        aliases={"宝可梦", "ps帮助", "对战帮助"},
        priority=5,
        block=True,
    )

    async def _guide_active(event: MessageEvent) -> bool:
        group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
        return await team_guide.has_state(event.get_user_id(), group_id)

    team_guide_router = on_message(
        rule=Rule(_guide_active),
        priority=4,
        block=True,
    )

    for format_config in format_registry.all():
        _register_challenge_handler(
            config=config,
            manager=battle_manager,
            format_config=format_config,
        )
    _register_prepare_handler(
        manager=battle_manager,
        repository=team_repository,
        runtime=runtime,
        team_sources=team_sources,
    )
    _register_team_handler(
        config=config,
        manager=battle_manager,
        registry=format_registry,
        repository=team_repository,
        runtime=runtime,
        team_sources=team_sources,
        translator=translation_service,
        team_guide=team_guide,
    )
    _register_sample_team_handler(
        config=config,
        manager=battle_manager,
        registry=format_registry,
        team_sources=team_sources,
    )
    _register_recommended_set_handler(
        config=config,
        manager=battle_manager,
        registry=format_registry,
        team_sources=team_sources,
        translator=translation_service,
    )
    _register_status_handler(manager=battle_manager)
    _register_check_handler(
        manager=battle_manager,
        translator=translation_service,
        move_repository=move_repository,
    )
    _register_team_guide(
        config=config,
        manager=battle_manager,
        team_guide=team_guide,
    )
    _register_pokemon_help(
        config=config,
        manager=battle_manager,
        registry=format_registry,
    )
    _register_private_router(manager=battle_manager)
    _register_group_battle_router(manager=battle_manager)


def _register_challenge_handler(
    *,
    config: ShowdownBattleConfig,
    manager: BattleManager,
    format_config: BattleFormatConfig,
) -> None:
    challenge_cmd = _on_showdown_command(
        format_config.challenge_command,
        aliases=set(format_config.challenge_aliases),
        rule=is_type(GroupMessageEvent),
        priority=6,
        block=True,
    )
    challenge_commands.append(challenge_cmd)

    @challenge_cmd.handle()
    async def handle_challenge(
        bot: Bot,
        event: GroupMessageEvent,
        args: Message = CommandArg(),
    ) -> None:
        if not config.showdown_battle_enabled:
            await challenge_cmd.finish("Pokémon Showdown 对战功能当前未启用。")
        if not config.group_is_allowed(event.group_id):
            await challenge_cmd.finish("本群未启用 Pokémon Showdown 对战。")
        if not manager.ready:
            await challenge_cmd.finish(
                f"Pokémon Showdown 运行时不可用：{manager.runtime_error or '尚未初始化'}"
            )

        challenger_id = event.get_user_id()
        target_id = next(
            (
                str(segment.data.get("qq"))
                for segment in event.get_message()
                if segment.type == "at"
                and segment.data.get("qq")
                and segment.data.get("qq") != "all"
            ),
            None,
        )
        argument = args.extract_plain_text().strip().lower()
        ai_requested = _is_ai_challenge_request(
            bot_self_id=str(bot.self_id),
            bot_name=manager.bot_name,
            target_id=target_id,
            argument=argument,
            to_me=event.to_me,
        )
        if not target_id and not ai_requested:
            await challenge_cmd.finish(
                f"请使用“{format_config.challenge_command} @对手”挑战玩家，"
                f"或“{format_config.challenge_command} {manager.bot_name}”"
                f"挑战 {manager.bot_name}。"
            )
        if target_id == challenger_id:
            await challenge_cmd.finish("不能向自己发起挑战。")
        if ai_requested and (
            not config.showdown_battle_ai_enabled or not manager.ai_available
        ):
            await challenge_cmd.finish(
                f"{manager.bot_name} 当前无法参加 Showdown 对战。"
            )

        challenger_info = {
            "user_id": int(challenger_id),
            "card": event.sender.card,
            "nickname": event.sender.nickname,
        }
        challenger = (challenger_id, _extract_display_name(challenger_info))
        if ai_requested:
            try:
                session = await manager.create_ai_session(
                    group_id=event.group_id,
                    challenger=challenger,
                    format_config=format_config,
                )
            except (
                BattleAIOpponentUnavailable,
                BattleSessionConflict,
                BattleRuntimeNotReady,
            ) as exc:
                await challenge_cmd.finish(str(exc))
        else:
            assert target_id is not None
            try:
                opponent_info = await bot.get_group_member_info(
                    group_id=event.group_id,
                    user_id=int(target_id),
                    no_cache=True,
                )
            except Exception:
                await challenge_cmd.finish(
                    "无法确认对手的群成员信息，请重新 @ 群内成员。"
                )
            try:
                session = await manager.create_session(
                    group_id=event.group_id,
                    challenger=challenger,
                    opponent=(target_id, _extract_display_name(opponent_info)),
                    format_config=format_config,
                )
            except (BattleSessionConflict, BattleRuntimeNotReady) as exc:
                await challenge_cmd.finish(str(exc))

        prepare_instruction = "组队" if format_config.requires_team else "对战准备"
        if ai_requested:
            private_tip = (
                f"已创建 {format_config.display_name} 对战，"
                f"你的对手是 {manager.bot_name}。\n"
                "请先选择本局操作频道：\n"
                "• 在发起群发送“对战准备”：使用群聊模式，队伍、招式和操作信息"
                "会对群成员可见；\n"
                "• 在本私聊发送“对战准备”：使用私聊模式，公开对局更新图也会"
                "同步到这里。\n"
            )
            if format_config.requires_team:
                private_tip += (
                    "选定后在同一频道发送“组队”，或使用“导入队伍 <文本或链接>”。"
                )
            else:
                private_tip += "随机规则发送“对战准备”后会同时确认就绪。"
            private_tip += f"\n{manager.bot_name} 已准备。"
        else:
            private_tip = f"已创建 {format_config.display_name} 玩家对战。\n"
            if format_config.requires_team:
                private_tip += (
                    "请发送“组队”，然后按数字菜单选择一键推荐、样例、"
                    "网上链接、自选成员或已保存队伍。\n"
                    "熟悉快捷命令时也可直接发送“导入队伍 <文本或链接>”。"
                )
            else:
                private_tip += "请发送“对战准备”确认就绪，无需自备队伍。"
            private_tip += "\n对战开始后，群内的公开对局更新图会同步到本私聊。"
        private_tip += f"\n请在 {format_config.invite_timeout} 秒内完成准备。"

        if ai_requested:
            try:
                await bot.send_private_msg(
                    user_id=int(challenger_id), message=private_tip
                )
            except Exception:
                logger.warning(
                    "failed to send optional private preparation prompt for "
                    f"showdown Bot battle user_id={challenger_id}"
                )
        else:
            try:
                await bot.send_private_msg(
                    user_id=int(challenger_id), message=private_tip
                )
                assert target_id is not None
                await bot.send_private_msg(user_id=int(target_id), message=private_tip)
            except Exception:
                await session.close()
                await challenge_cmd.finish(
                    "无法向双方发送私聊。请确认机器人具备私聊权限后重试。"
                )

        await session.schedule_invite_timeout(bot)
        if ai_requested:
            if format_config.requires_team:
                group_prepare_tip = (
                    "在本群发送“对战准备”选择群聊模式，再发送“组队”完成准备。"
                )
            else:
                group_prepare_tip = "在本群发送“对战准备”即可选择群聊模式并确认就绪。"
            group_message = MessageSegment.at(int(challenger_id)) + MessageSegment.text(
                f" 向 {manager.bot_name} 发起了 {format_config.display_name} 对战！\n"
                f"{group_prepare_tip}\n"
                "群聊模式会公开队伍、招式和操作信息；也可改在私聊发送"
                "“对战准备”使用私聊模式。"
            )
        else:
            assert target_id is not None
            group_message = (
                MessageSegment.at(int(challenger_id))
                + MessageSegment.text(" 向 ")
                + MessageSegment.at(int(target_id))
                + MessageSegment.text(
                    f" 发起了 {format_config.display_name} 对战！\n"
                    f"双方请私聊机器人发送“{prepare_instruction}”。"
                )
            )
        await bot.send_group_msg(group_id=event.group_id, message=group_message)


def _register_prepare_handler(
    *,
    manager: BattleManager,
    repository: TeamRepository,
    runtime: ShowdownRuntime,
    team_sources: TeamSourceService,
) -> None:
    assert prepare_cmd is not None

    @prepare_cmd.handle()
    async def handle_prepare(
        bot: Bot,
        event: MessageEvent,
        args: Message = CommandArg(),
    ) -> None:
        user_id = event.get_user_id()
        if isinstance(event, GroupMessageEvent):
            session = await manager.get_session_by_group(event.group_id)
            if (
                session is None
                or session.state != "pending"
                or not session.is_ai_battle
                or session.group_id != event.group_id
                or session.get_side_by_user(user_id) is None
            ):
                await prepare_cmd.finish(
                    f"群聊准备仅适用于你在本群发起的 {manager.bot_name} 对战。"
                )
            session.set_interaction_channel(user_id, "group")
            group_mode = True
        elif isinstance(event, PrivateMessageEvent):
            session = await manager.get_session_by_user(user_id)
            if not session or session.state != "pending":
                await prepare_cmd.finish("你当前没有等待准备的对战。")
            group_mode = False
        else:
            await prepare_cmd.finish("请在群聊或私聊中确认准备。")

        format_config = session.format_config
        raw_input = args.extract_plain_text().strip()

        if not format_config.requires_team:
            session.set_interaction_channel(
                user_id, "group" if group_mode else "private"
            )
            await session.set_team(user_id, "", None)
            channel_label = "本群" if group_mode else "私聊"
            await prepare_cmd.send(
                f"✅ 已确认就绪，本局将在{channel_label}接收操作提示。"
            )
        else:
            if not raw_input:
                session.set_interaction_channel(
                    user_id, "group" if group_mode else "private"
                )
                if group_mode:
                    await prepare_cmd.finish(
                        "✅ 已选择群聊模式。本局队伍、招式和操作信息会对群成员可见。\n"
                        "请继续在本群发送“组队”打开队伍中心，或发送“导入队伍 "
                        "<Showdown 队伍文本或链接>”。"
                    )
                await prepare_cmd.finish(
                    "✅ 已选择私聊模式。请发送“组队”打开队伍中心，或发送："
                    "导入队伍 使用 队伍名"
                )
            if raw_input.lower() in {"样例", "sample"}:
                await prepare_cmd.finish(
                    "请先发送“队伍样例”查看列表，再发送“导入队伍 样例 编号”。"
                )
            if raw_input.lower() in {"快速", "quick"}:
                await prepare_cmd.finish(
                    "请提供宝可梦，例如：导入队伍 快速 快龙, 仆刀将军,..."
                )
            source_label: str | None = None
            imported_team = None
            team_warnings: tuple[str, ...] = ()
            try:
                if raw_input.lower() in {
                    "一键组队",
                    "随机推荐",
                    "random",
                }:
                    if not format_config.generated_team_source:
                        await prepare_cmd.finish("该规则不支持一键生成队伍。")
                    imported = await team_sources.generate_team(
                        format_id=format_config.format_id,
                        source_id=format_config.generated_team_source,
                    )
                    imported_team = imported
                    team_text = imported.team_text
                    source_label = imported.label
                elif raw_input.startswith("使用 ") or raw_input.lower().startswith(
                    "use "
                ):
                    team_name = raw_input.split(maxsplit=1)[1].strip()
                    record = await repository.get_team(
                        user_id, session.format_id, team_name
                    )
                    if not record:
                        await prepare_cmd.finish("未找到该规则下保存的同名队伍。")
                    team_text = record.raw
                    source_label = f"保存的队伍「{team_name}」"
                elif raw_input.startswith("样例 ") or raw_input.lower().startswith(
                    "sample "
                ):
                    if not format_config.sample_team_source:
                        await prepare_cmd.finish(
                            "该规则暂无在线样例；请改用 PokePaste/crob.at 链接。"
                        )
                    index = _parse_positive_index(
                        raw_input.split(maxsplit=1)[1], label="样例编号"
                    )
                    prepared = await team_sources.get_compatible_sample(
                        format_id=session.format_id,
                        source_id=format_config.sample_team_source,
                        index=index,
                    )
                    team_text = prepared.team_text
                    source_label = (
                        f"样例「{prepared.sample.name}」by {prepared.sample.author}"
                    )
                elif raw_input.startswith("快速 ") or raw_input.lower().startswith(
                    "quick "
                ):
                    if not format_config.recommended_set_source:
                        await prepare_cmd.finish(
                            "该规则暂无自动配招数据；请改用 PokePaste/crob.at 链接。"
                        )
                    built = await team_sources.build_recommended_team(
                        format_id=session.format_id,
                        source_id=format_config.recommended_set_source,
                        species_input=raw_input.split(maxsplit=1)[1],
                    )
                    team_text = built.team_text
                    selected = format_quick_build_selections(
                        built.selections, session.translator
                    )
                    source_label = f"快速组队：{selected}"
                else:
                    imported = await team_sources.resolve_import(raw_input)
                    imported_team = imported
                    team_text = imported.team_text
                    source_label = imported.label

                if imported_team is not None:
                    team_sources.ensure_format_compatible(
                        imported_team, session.format_id
                    )
                prepared_team = await runtime.prepare_team_for_use(
                    session.format_id, team_text
                )
                team_text = prepared_team.team_text
                packed = prepared_team.packed
                source_warnings = imported_team.warnings if imported_team else ()
                team_warnings = _merge_team_warnings(
                    source_warnings, prepared_team.warnings
                )
            except (TeamRepositoryError, TeamSourceError, ShowdownBridgeError) as exc:
                await prepare_cmd.finish(f"队伍导入失败：{exc}")
            await session.set_team(user_id, packed, team_text)
            suffix = f"（{source_label}）" if source_label else ""
            await prepare_cmd.send(
                f"✅ 队伍导入成功{suffix}，等待对手。"
                f"{_format_team_warnings(team_warnings)}"
            )

        if session.teams_ready():
            await prepare_cmd.send("双方已准备，正在启动对战……")
            try:
                await session.start(bot)
            except ShowdownBridgeError as exc:
                await session.close()
                await prepare_cmd.finish(f"对战启动失败，房间已关闭：{exc}")


def _register_sample_team_handler(
    *,
    config: ShowdownBattleConfig,
    manager: BattleManager,
    registry: FormatRegistry,
    team_sources: TeamSourceService,
) -> None:
    assert sample_team_cmd is not None

    @sample_team_cmd.handle()
    async def handle_sample_team(
        event: MessageEvent,
        args: Message = CommandArg(),
    ) -> None:
        channel_error = await _team_command_channel_error(
            config=config,
            manager=manager,
            event=event,
        )
        if channel_error:
            await sample_team_cmd.finish(channel_error)
        format_config, remainder = await _resolve_format_context(
            user_id=event.get_user_id(),
            text=args.extract_plain_text(),
            manager=manager,
            registry=registry,
        )
        if not format_config:
            await sample_team_cmd.finish(
                "请提供规则，例如：队伍样例 gen9ou\n" + _build_format_list(registry)
            )
        if not format_config.sample_team_source:
            await sample_team_cmd.finish(
                f"{format_config.display_name} 暂无稳定的在线样例源；"
                "可以直接导入 PokePaste 或 crob.at 链接。"
            )
        try:
            if remainder:
                index = _parse_positive_index(remainder, label="样例编号")
                prepared = await team_sources.get_compatible_sample(
                    format_id=format_config.format_id,
                    source_id=format_config.sample_team_source,
                    index=index,
                )
                await sample_team_cmd.finish(
                    f"【{prepared.sample.name}】作者："
                    f"{prepared.sample.author}\n\n{prepared.team_text}\n\n"
                    f"对战准备时发送：导入队伍 样例 {index}\n"
                    f"保存时发送：队伍管理 抄队 {format_config.format_id} "
                    f"{index} 队伍名"
                )
            prepared_samples = await team_sources.list_compatible_samples(
                format_id=format_config.format_id,
                source_id=format_config.sample_team_source,
            )
        except (TeamSourceError, ShowdownBridgeError) as exc:
            await sample_team_cmd.finish(f"读取样例队伍失败：{exc}")
        lines = [f"【{format_config.display_name} 在线样例】"]
        for index, prepared in enumerate(prepared_samples, start=1):
            sample = prepared.sample
            lines.append(f"{index}. {sample.name} — {sample.author}")
        lines.extend(
            [
                "",
                "查看：队伍样例 [规则] 编号",
                "当前对战直接使用：导入队伍 样例 编号",
                "保存：队伍管理 抄队 规则 编号 队伍名",
            ]
        )
        await sample_team_cmd.finish("\n".join(lines))


def _register_recommended_set_handler(
    *,
    config: ShowdownBattleConfig,
    manager: BattleManager,
    registry: FormatRegistry,
    team_sources: TeamSourceService,
    translator: TranslationService,
) -> None:
    assert recommended_set_cmd is not None

    @recommended_set_cmd.handle()
    async def handle_recommended_set(
        event: MessageEvent,
        args: Message = CommandArg(),
    ) -> None:
        channel_error = await _team_command_channel_error(
            config=config,
            manager=manager,
            event=event,
        )
        if channel_error:
            await recommended_set_cmd.finish(channel_error)
        format_config, species_query = await _resolve_format_context(
            user_id=event.get_user_id(),
            text=args.extract_plain_text(),
            manager=manager,
            registry=registry,
        )
        if not format_config or not species_query:
            await recommended_set_cmd.finish(
                "请提供规则和宝可梦，例如：推荐配招 gen9ou 快龙。\n"
                "若正在准备对战，可省略规则。"
            )
        if not format_config.recommended_set_source:
            await recommended_set_cmd.finish(
                f"{format_config.display_name} 暂无稳定的自动配招源。"
            )
        try:
            result = await team_sources.list_recommended_sets(
                format_id=format_config.format_id,
                source_id=format_config.recommended_set_source,
                species_query=species_query,
            )
        except TeamSourceError as exc:
            await recommended_set_cmd.finish(str(exc))
        species_name = result.species
        translated = translator.translate_species(species_name)
        lines = [f"【{translated} 推荐配招】"]
        for index, option in enumerate(result.options, start=1):
            lines.extend(format_recommended_set_option(index, option, translator))
        lines.extend(
            [
                "",
                "快速组队默认采用第 1 套；指定其他配置可写“宝可梦=编号”。",
                "例如：导入队伍 快速 快龙=2, 仆刀将军=1",
            ]
        )
        await recommended_set_cmd.finish("\n".join(lines))


def _register_team_handler(
    *,
    config: ShowdownBattleConfig,
    manager: BattleManager,
    registry: FormatRegistry,
    repository: TeamRepository,
    runtime: ShowdownRuntime,
    team_sources: TeamSourceService,
    translator: TranslationService,
    team_guide: TeamGuideManager,
) -> None:
    assert team_manager_cmd is not None

    @team_manager_cmd.handle()
    async def handle_team_manager(
        event: MessageEvent,
        args: Message = CommandArg(),
    ) -> None:
        user_id = event.get_user_id()
        channel_error = await _team_command_channel_error(
            config=config,
            manager=manager,
            event=event,
        )
        if channel_error:
            await team_manager_cmd.finish(channel_error)
        text = args.extract_plain_text().strip()
        if not text:
            group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
            prompt = await team_guide.start(user_id, group_id, entry="library")
            await team_manager_cmd.finish(prompt)
        keyword, *rest = text.split(maxsplit=1)
        command = keyword.lower()
        rest_text = rest[0].strip() if rest else ""
        aliases = {
            "help": {"帮助", "help", "?"},
            "advanced_help": {"高级帮助", "完整帮助", "advanced"},
            "list": {"列表", "list", "ls"},
            "save": {"保存", "导入", "save"},
            "view": {"查看", "view", "show"},
            "delete": {"删除", "remove", "del", "rm"},
            "rename": {"重命名", "rename"},
            "edit": {"编辑", "edit"},
            "add": {"添加", "新建", "create", "add"},
            "sample": {"抄队", "样例", "sample"},
            "quick": {"快速", "quick"},
            "random": {"一键组队", "随机推荐", "random"},
        }
        action = next(
            (name for name, values in aliases.items() if command in values), None
        )
        try:
            if action == "help":
                await team_manager_cmd.finish(_build_team_help(registry))
            if action == "advanced_help":
                await team_manager_cmd.finish(_build_team_advanced_help(registry))
            if action == "list":
                await _team_list(
                    matcher=team_manager_cmd,
                    registry=registry,
                    repository=repository,
                    user_id=user_id,
                    token=rest_text,
                )
            if action == "save":
                await _team_save(
                    matcher=team_manager_cmd,
                    registry=registry,
                    repository=repository,
                    runtime=runtime,
                    team_sources=team_sources,
                    user_id=user_id,
                    text=rest_text,
                )
            if action == "view":
                await _team_view_or_delete(
                    matcher=team_manager_cmd,
                    registry=registry,
                    repository=repository,
                    user_id=user_id,
                    text=rest_text,
                    delete=False,
                )
            if action == "delete":
                await _team_view_or_delete(
                    matcher=team_manager_cmd,
                    registry=registry,
                    repository=repository,
                    user_id=user_id,
                    text=rest_text,
                    delete=True,
                )
            if action == "rename":
                await _team_rename(
                    matcher=team_manager_cmd,
                    registry=registry,
                    repository=repository,
                    user_id=user_id,
                    text=rest_text,
                )
            if action in {"edit", "add"}:
                parts = rest_text.split(maxsplit=1)
                if not parts or (action == "edit" and len(parts) < 2):
                    usage = (
                        "队伍管理 编辑 规则标识 队伍名"
                        if action == "edit"
                        else "队伍管理 添加 规则标识"
                    )
                    await team_manager_cmd.finish(f"请提供：{usage}")
                format_config = registry.resolve_format_token(parts[0])
                if not format_config:
                    await team_manager_cmd.finish(f"未找到规则：{parts[0]}")
                group_id = (
                    event.group_id if isinstance(event, GroupMessageEvent) else None
                )
                prompt = await team_guide.start_editor(
                    user_id,
                    format_id=format_config.format_id,
                    team_name=parts[1].strip() if action == "edit" else None,
                    group_id=group_id,
                )
                await team_manager_cmd.finish(prompt)
            if action == "sample":
                await _team_save_sample(
                    matcher=team_manager_cmd,
                    registry=registry,
                    repository=repository,
                    runtime=runtime,
                    team_sources=team_sources,
                    user_id=user_id,
                    text=rest_text,
                )
            if action == "quick":
                await _team_save_quick_build(
                    matcher=team_manager_cmd,
                    registry=registry,
                    repository=repository,
                    runtime=runtime,
                    team_sources=team_sources,
                    translator=translator,
                    user_id=user_id,
                    text=rest_text,
                )
            if action == "random":
                await _team_save_generated(
                    matcher=team_manager_cmd,
                    registry=registry,
                    repository=repository,
                    runtime=runtime,
                    team_sources=team_sources,
                    user_id=user_id,
                    text=rest_text,
                )
        except TeamRepositoryError as exc:
            await team_manager_cmd.finish(f"队伍仓库错误：{exc}")
        except TeamEditorError as exc:
            await team_manager_cmd.finish(f"队伍编辑失败：{exc}")
        except TeamSourceError as exc:
            await team_manager_cmd.finish(f"在线队伍错误：{exc}")
        except ShowdownBridgeError as exc:
            await team_manager_cmd.finish(f"队伍处理失败：{exc}")
        await team_manager_cmd.finish("未识别的子命令。\n" + _build_team_help(registry))


async def _team_list(
    *,
    matcher: type[Matcher],
    registry: FormatRegistry,
    repository: TeamRepository,
    user_id: str,
    token: str,
) -> None:
    format_config = registry.resolve_format_token(token) if token else None
    if token and not format_config:
        await matcher.finish(f"未找到规则：{token}\n{_build_format_list(registry)}")
    records = await repository.list_teams(
        user_id, format_config.format_id if format_config else None
    )
    if not records:
        await matcher.finish("你尚未保存符合条件的队伍。")
    grouped: dict[str, list[TeamRecord]] = {}
    for record in records:
        grouped.setdefault(record.format_id, []).append(record)
    lines = ["【队伍列表】"]
    for format_id, subset in grouped.items():
        lines.append(f"{_format_format_display(registry.get(format_id))}：")
        for index, record in enumerate(subset, start=1):
            lines.append(
                f"  {index}. {record.name}（{_format_timestamp(record.updated_at)}）"
            )
    await matcher.finish("\n".join(lines))


async def _team_save(
    *,
    matcher: type[Matcher],
    registry: FormatRegistry,
    repository: TeamRepository,
    runtime: ShowdownRuntime,
    team_sources: TeamSourceService,
    user_id: str,
    text: str,
) -> None:
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        await matcher.finish("请提供：规则标识 队伍名 Showdown队伍文本")
    format_config = registry.resolve_format_token(parts[0])
    if not format_config:
        await matcher.finish(f"未找到规则：{parts[0]}")
    if not format_config.requires_team:
        await matcher.finish(f"{format_config.display_name} 无需保存队伍。")
    team_name = repository.validate_team_name(parts[1])
    imported = await team_sources.resolve_import(parts[2])
    team_sources.ensure_format_compatible(imported, format_config.format_id)
    team_text = imported.team_text
    existing = await repository.get_team(user_id, format_config.format_id, team_name)
    try:
        prepared_team = await runtime.prepare_team_for_use(
            format_config.format_id, team_text
        )
    except ShowdownBridgeError as exc:
        await matcher.finish(f"队伍保存失败：{exc}")
    team_text = prepared_team.team_text
    packed = prepared_team.packed
    await repository.set_team(
        user_id,
        format_config.format_id,
        team_name,
        packed=packed,
        raw=team_text,
    )
    source = f"，来源：{imported.label}" if imported.label else ""
    await matcher.finish(
        f"{'更新' if existing else '保存'}成功："
        f"{_format_format_display(format_config)} 队伍「{team_name}」{source}。"
        f"{_format_team_warnings(_merge_team_warnings(imported.warnings, prepared_team.warnings))}"
    )


async def _team_save_sample(
    *,
    matcher: type[Matcher],
    registry: FormatRegistry,
    repository: TeamRepository,
    runtime: ShowdownRuntime,
    team_sources: TeamSourceService,
    user_id: str,
    text: str,
) -> None:
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        await matcher.finish("请提供：规则标识 样例编号 队伍名")
    format_config = registry.resolve_format_token(parts[0])
    if not format_config:
        await matcher.finish(f"未找到规则：{parts[0]}")
    if not format_config.sample_team_source:
        await matcher.finish("该规则暂无在线样例队伍。")
    index = _parse_positive_index(parts[1], label="样例编号")
    team_name = repository.validate_team_name(parts[2])
    prepared = await team_sources.get_compatible_sample(
        format_id=format_config.format_id,
        source_id=format_config.sample_team_source,
        index=index,
    )
    try:
        prepared_team = await runtime.prepare_team_for_use(
            format_config.format_id, prepared.team_text
        )
    except ShowdownBridgeError as exc:
        await matcher.finish(f"在线样例当前未通过规则校验：{exc}")
    existing = await repository.get_team(user_id, format_config.format_id, team_name)
    await repository.set_team(
        user_id,
        format_config.format_id,
        team_name,
        packed=prepared_team.packed,
        raw=prepared_team.team_text,
    )
    await matcher.finish(
        f"{'更新' if existing else '保存'}成功：队伍「{team_name}」\n"
        f"来源：{prepared.sample.name} — {prepared.sample.author}"
        f"{_format_team_warnings(prepared_team.warnings)}"
    )


async def _team_save_generated(
    *,
    matcher: type[Matcher],
    registry: FormatRegistry,
    repository: TeamRepository,
    runtime: ShowdownRuntime,
    team_sources: TeamSourceService,
    user_id: str,
    text: str,
) -> None:
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await matcher.finish("请提供：规则标识 队伍名")
    format_config = registry.resolve_format_token(parts[0])
    if not format_config:
        await matcher.finish(f"未找到规则：{parts[0]}")
    if not format_config.generated_team_source:
        await matcher.finish("该规则不支持一键生成队伍。")
    team_name = repository.validate_team_name(parts[1])
    imported = await team_sources.generate_team(
        format_id=format_config.format_id,
        source_id=format_config.generated_team_source,
    )
    team_sources.ensure_format_compatible(imported, format_config.format_id)
    try:
        prepared_team = await runtime.prepare_team_for_use(
            format_config.format_id, imported.team_text
        )
    except ShowdownBridgeError as exc:
        await matcher.finish(f"随机推荐队伍未通过当前规则校验：{exc}")
    existing = await repository.get_team(user_id, format_config.format_id, team_name)
    await repository.set_team(
        user_id,
        format_config.format_id,
        team_name,
        packed=prepared_team.packed,
        raw=prepared_team.team_text,
    )
    await matcher.finish(
        f"{'更新' if existing else '保存'}成功：队伍「{team_name}」\n"
        f"来源：{imported.label}。建议查看后按喜好微调。"
        f"{_format_team_warnings(_merge_team_warnings(imported.warnings, prepared_team.warnings))}"
    )


async def _team_save_quick_build(
    *,
    matcher: type[Matcher],
    registry: FormatRegistry,
    repository: TeamRepository,
    runtime: ShowdownRuntime,
    team_sources: TeamSourceService,
    translator: TranslationService,
    user_id: str,
    text: str,
) -> None:
    header, separator, species_input = text.partition("|")
    if not separator:
        await matcher.finish(
            "请提供：规则标识 队伍名 | 宝可梦1, 宝可梦2,...\n指定配招可写成“快龙=2”。"
        )
    header_parts = header.strip().split(maxsplit=1)
    if len(header_parts) < 2:
        await matcher.finish("请在竖线前提供规则标识和队伍名。")
    format_config = registry.resolve_format_token(header_parts[0])
    if not format_config:
        await matcher.finish(f"未找到规则：{header_parts[0]}")
    if not format_config.recommended_set_source:
        await matcher.finish("该规则暂无自动推荐配招数据。")
    team_name = repository.validate_team_name(header_parts[1])
    built = await team_sources.build_recommended_team(
        format_id=format_config.format_id,
        source_id=format_config.recommended_set_source,
        species_input=species_input,
    )
    try:
        prepared_team = await runtime.prepare_team_for_use(
            format_config.format_id, built.team_text
        )
    except ShowdownBridgeError as exc:
        await matcher.finish(f"自动生成的组合未通过规则校验：{exc}")
    existing = await repository.get_team(user_id, format_config.format_id, team_name)
    await repository.set_team(
        user_id,
        format_config.format_id,
        team_name,
        packed=prepared_team.packed,
        raw=prepared_team.team_text,
    )
    selections = format_quick_build_selections(built.selections, translator)
    await matcher.finish(
        f"{'更新' if existing else '保存'}成功：队伍「{team_name}」\n"
        f"采用配招：{selections}\n"
        "可用“队伍管理 查看 规则标识 队伍名”复制并继续微调。"
        f"{_format_team_warnings(prepared_team.warnings)}"
    )


async def _team_view_or_delete(
    *,
    matcher: type[Matcher],
    registry: FormatRegistry,
    repository: TeamRepository,
    user_id: str,
    text: str,
    delete: bool,
) -> None:
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await matcher.finish("请提供规则标识和队伍名称。")
    format_config = registry.resolve_format_token(parts[0])
    if not format_config:
        await matcher.finish(f"未找到规则：{parts[0]}")
    team_name = parts[1].strip()
    if delete:
        removed = await repository.delete_team(
            user_id, format_config.format_id, team_name
        )
        if not removed:
            await matcher.finish("未找到对应队伍。")
        await matcher.finish(f"已删除队伍「{team_name}」。")
    record = await repository.get_team(user_id, format_config.format_id, team_name)
    if not record:
        await matcher.finish("未找到对应队伍。")
    await matcher.finish(
        f"【{format_config.display_name} - {team_name}】\n{record.raw}"
    )


async def _team_rename(
    *,
    matcher: type[Matcher],
    registry: FormatRegistry,
    repository: TeamRepository,
    user_id: str,
    text: str,
) -> None:
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        await matcher.finish("请提供规则标识、旧名称和新名称。")
    format_config = registry.resolve_format_token(parts[0])
    if not format_config:
        await matcher.finish(f"未找到规则：{parts[0]}")
    renamed = await repository.rename_team(
        user_id,
        format_config.format_id,
        parts[1],
        parts[2],
    )
    if not renamed:
        await matcher.finish("重命名失败：旧名称不存在或新名称已被使用。")
    await matcher.finish(f"队伍已重命名为「{parts[2]}」。")


def _register_status_handler(*, manager: BattleManager) -> None:
    assert status_cmd is not None

    @status_cmd.handle()
    async def handle_status(event: MessageEvent) -> None:
        if isinstance(event, GroupMessageEvent):
            session = await manager.get_session_by_group(event.group_id)
            missing = "本群暂无正在进行的对战。"
        elif isinstance(event, PrivateMessageEvent):
            session = await manager.get_session_by_user(event.get_user_id())
            missing = "你当前没有正在进行的对战。\n" + PRIVATE_STATUS_HINT
        else:
            await status_cmd.finish("请在群聊或私聊中使用该指令。")
        if not session or session.state == "finished":
            await status_cmd.finish(missing)
        # Private viewers are battle participants; allow their own held items
        # on the panel. Group panels stay restricted to public knowledge.
        viewer_user_id = (
            event.get_user_id() if isinstance(event, PrivateMessageEvent) else None
        )
        image_data = await session.render_status_image(viewer_user_id=viewer_user_id)
        if image_data:
            await status_cmd.send(MessageSegment.image(image_data))
            if isinstance(event, PrivateMessageEvent):
                await status_cmd.finish(PRIVATE_STATUS_HINT)
            return
        report = session.build_status_report()
        if isinstance(event, PrivateMessageEvent):
            report += "\n" + PRIVATE_STATUS_HINT
        await status_cmd.finish(report)


def _register_check_handler(
    *,
    manager: BattleManager,
    translator: TranslationService,
    move_repository: MoveDataRepository,
) -> None:
    assert check_cmd is not None

    @check_cmd.handle()
    async def handle_check(
        event: MessageEvent,
        args: Message = CommandArg(),
    ) -> None:
        query = args.extract_plain_text().strip()
        if not query:
            await check_cmd.finish("请提供招式编号或名称。\n" + PRIVATE_CHECK_HINT)
        if query.split()[0].isdigit() and isinstance(
            event, (PrivateMessageEvent, GroupMessageEvent)
        ):
            tokens = query.split()
            actor_index = 1
            move_index = int(tokens[0])
            if len(tokens) >= 2 and tokens[1].isdigit():
                actor_index = max(1, int(tokens[0]))
                move_index = int(tokens[1])
            if isinstance(event, GroupMessageEvent):
                session = await manager.get_session_by_group(event.group_id)
            else:
                session = await manager.get_session_by_user(event.get_user_id())
            if not session or session.state != "active":
                await check_cmd.finish("你当前没有正在进行的对战。")
            side = session.get_side_by_user(event.get_user_id())
            if not side:
                await check_cmd.finish("当前操作频道无法按编号查询招式。")
            expected_channel = session.interaction_channel_for_side(side)
            event_channel = (
                "group" if isinstance(event, GroupMessageEvent) else "private"
            )
            if expected_channel != event_channel:
                channel_label = "对战群" if expected_channel == "group" else "私聊"
                await check_cmd.finish(f"本局已选择在{channel_label}操作。")
            request = session.current_requests.get(side, {})
            active = request.get("active") or []
            if actor_index > len(active) or actor_index < 1:
                await check_cmd.finish("当前行动位置编号无效。")
            moves = active[actor_index - 1].get("moves", [])
            if move_index < 1 or move_index > len(moves):
                await check_cmd.finish("招式编号超出范围。")
            move_info = moves[move_index - 1]
            identifier = move_info.get("id") or move_info.get("move")
            entry = move_repository.get(str(identifier)) if identifier else None
            if not entry:
                await check_cmd.finish("暂未找到该招式的详细信息。")
            current_pp = move_info.get("pp")
            max_pp = move_info.get("maxpp")
            message = _format_move_details(
                entry,
                translator,
                current_pp=current_pp if isinstance(current_pp, int) else None,
                max_pp=max_pp if isinstance(max_pp, int) else None,
            )
            if isinstance(event, PrivateMessageEvent):
                message += "\n" + PRIVATE_CHECK_HINT
            await check_cmd.finish(message)

        entry = _resolve_move_entry_by_name(query, translator, move_repository)
        if not entry:
            await check_cmd.finish("未找到该招式，请确认名称是否正确。")
        await check_cmd.finish(_format_move_details(entry, translator))


def _register_pokemon_help(
    *,
    config: ShowdownBattleConfig,
    manager: BattleManager,
    registry: FormatRegistry,
) -> None:
    assert pokemon_help_cmd is not None

    @pokemon_help_cmd.handle()
    async def handle_pokemon_help(
        event: MessageEvent,
        args: Message = CommandArg(),
    ) -> None:
        if isinstance(event, GroupMessageEvent):
            if not config.showdown_battle_enabled:
                await pokemon_help_cmd.finish("Pokémon Showdown 对战功能当前未启用。")
            if not config.group_is_allowed(event.group_id):
                await pokemon_help_cmd.finish("本群未启用 Pokémon Showdown 对战。")
        elif not isinstance(event, PrivateMessageEvent):
            await pokemon_help_cmd.finish("请在群聊或私聊中查看帮助。")

        topic = args.extract_plain_text().strip().lower()
        if topic in {"组队", "队伍", "team"}:
            await pokemon_help_cmd.finish(_pokemon_team_help())
        if topic in {"对战", "挑战", "battle"}:
            await pokemon_help_cmd.finish(
                _pokemon_challenge_help(registry, manager.bot_name)
            )
        if topic in {"对局", "操作", "行动", "play"}:
            await pokemon_help_cmd.finish(_pokemon_battle_action_help(manager.bot_name))
        if topic in {"高级", "命令", "advanced"}:
            await pokemon_help_cmd.finish(
                _build_team_advanced_help(registry)
                + "\n\n"
                + _pokemon_challenge_help(registry, manager.bot_name)
            )
        if topic:
            await pokemon_help_cmd.finish(
                "未找到该帮助主题。可用主题：组队、对战、对局、高级。"
            )

        session = await manager.get_session_by_user(event.get_user_id())
        if session and session.state == "pending":
            channel = session.interaction_channel_for_user(event.get_user_id())
            if session.is_ai_battle and channel == "group":
                channel_tip = "请继续在发起群操作；本局信息会对群成员可见。"
            elif session.is_ai_battle:
                channel_tip = (
                    "可在发起群发送“对战准备”选择群聊模式，或在本私聊发送"
                    "“对战准备”使用私聊模式。"
                )
            else:
                channel_tip = "请继续在私聊操作；公开对局更新图会同步到这里。"
            if session.format_config.requires_team:
                message = (
                    f"【当前状态：{session.format_config.display_name} 准备中】\n"
                    f"{channel_tip}\n发送“组队”可从已保存队伍、一键推荐、链接、"
                    "样例或自选成员中选择；也可使用“导入队伍 <文本或链接>”。"
                )
            else:
                message = (
                    f"【当前状态：{session.format_config.display_name} 准备中】\n"
                    f"{channel_tip}\n发送“对战准备”确认就绪。"
                )
            await pokemon_help_cmd.finish(message)
        if session and session.state == "active":
            await pokemon_help_cmd.finish(
                "【当前状态：对战进行中】\n"
                + _pokemon_battle_action_help(manager.bot_name)
            )
        await pokemon_help_cmd.finish(_pokemon_help_overview(manager.bot_name))


def _pokemon_help_overview(bot_name: str) -> str:
    return (
        "【宝可梦 Showdown 帮助】\n"
        "新手只需记住：\n"
        "• 组队 —— 打开数字菜单创建、查询和管理队伍\n"
        "• 我的队伍 —— 直接打开已保存队伍\n"
        "• 推荐配招 规则 宝可梦 —— 快速查询单体配置\n"
        "• g9ou @对手 —— 发起 Gen9 OU 玩家对战\n"
        f"• g9ou {bot_name} —— 挑战 {bot_name}\n"
        "• 战况 —— 查看当前对局\n\n"
        "详细主题：\n"
        "宝可梦帮助 组队｜对战｜对局｜高级"
    )


def _pokemon_team_help() -> str:
    return (
        "【组队帮助】\n"
        "发送“组队”进入队伍中心，可完成：\n"
        "1. 一键生成、样例抄队、链接导入、手动导入或自选成员\n"
        "2. 逐只添加并编辑道具、特性、招式、性格、EV/IV 等配置\n"
        "3. 分页查看已保存队伍\n"
        "4. 编辑、查看、重命名、删除或复制队伍\n"
        "5. 查询推荐配招和浏览在线样例\n\n"
        "向导内可随时发送：返回、菜单、帮助、0。\n"
        "熟练用户可发送“队伍管理 高级帮助”查看快捷语法。"
    )


def _pokemon_challenge_help(registry: FormatRegistry, bot_name: str) -> str:
    lines = ["【发起对战】在群内发送以下命令："]
    for format_config in registry.all():
        lines.append(
            f"• {format_config.challenge_command} @对手 —— {format_config.display_name}"
        )
        lines.append(
            f"  {format_config.challenge_command} {bot_name} —— 挑战 {bot_name}"
        )
    lines.extend(
        [
            "",
            f"挑战 {bot_name} 后，在群里发送“对战准备”选择群聊模式，",
            "或在私聊发送“对战准备”选择私聊模式；后续提示与操作跟随该频道。",
            "群聊模式会公开队伍、招式和操作信息。自备队伍规则选定频道后发送“组队”。",
            "玩家对战仅在私聊操作，群里的公开对局更新图会同步私发给双方。",
        ]
    )
    return "\n".join(lines)


def _pokemon_battle_action_help(bot_name: str) -> str:
    return (
        "【对局操作】\n"
        f"与 {bot_name} 对战时，准备阶段选择的群聊或私聊就是本局操作频道。\n"
        "群聊模式会公开完整队伍、招式和操作提示；私聊模式会同步公开对局更新图。\n"
        "玩家对战仅在私聊操作，公开对局更新图会同步私发给双方。\n"
        "行动示例：move 1、switch 3；双打：move1 1 1; move2 2 2。\n"
        "特殊机制以当回合提示为准：在 move 指令末尾加 mega（Mega 进化）"
        "或 tera（太晶化）；冠军规则使用 mega，不使用 tera。\n"
        "查看招式：check 招式名，当前操作频道中也可使用 check 1。\n"
        "其他操作：pass、default、forfeit；发送“战况”查看公开战况。"
    )


def _register_team_guide(
    *,
    config: ShowdownBattleConfig,
    manager: BattleManager,
    team_guide: TeamGuideManager,
) -> None:
    assert team_guide_cmd is not None
    assert team_guide_router is not None

    @team_guide_cmd.handle()
    async def handle_team_guide_start(event: MessageEvent) -> None:
        channel_error = await _team_command_channel_error(
            config=config,
            manager=manager,
            event=event,
        )
        if channel_error:
            await team_guide_cmd.finish(channel_error)
        group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
        prompt = await team_guide.start(event.get_user_id(), group_id)
        await team_guide_cmd.finish(prompt)

    @team_guide_router.handle()
    async def handle_team_guide_input(
        bot: Bot,
        event: MessageEvent,
    ) -> None:
        text = event.get_message().extract_plain_text()
        group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
        reply = await team_guide.handle(bot, event.get_user_id(), text, group_id)
        await team_guide_router.finish(reply)


def _register_private_router(*, manager: BattleManager) -> None:
    assert private_router is not None

    @private_router.handle()
    async def handle_private_input(bot: Bot, event: PrivateMessageEvent) -> None:
        session = await manager.get_session_by_user(event.get_user_id())
        if not session or session.state != "active":
            return
        text = event.get_message().extract_plain_text().strip()
        if not text:
            return
        if not session.can_accept_private_choice(event.get_user_id()):
            await bot.send_private_msg(
                user_id=int(event.get_user_id()),
                message="本局已选择群聊模式，请在发起对战的群里提交操作。",
            )
            return
        reply = await session.handle_choice(bot, event.get_user_id(), text)
        if reply:
            await bot.send_private_msg(user_id=int(event.get_user_id()), message=reply)


def _register_group_battle_router(*, manager: BattleManager) -> None:
    assert group_battle_router is not None

    @group_battle_router.handle()
    async def handle_group_battle_input(bot: Bot, event: GroupMessageEvent) -> None:
        session = await manager.get_session_by_group(event.group_id)
        if session is None or not session.is_ai_battle or session.state != "active":
            return
        text = event.get_message().extract_plain_text().strip()
        reply = await session.handle_choice(bot, event.get_user_id(), text)
        if reply:
            message = MessageSegment.at(int(event.get_user_id())) + MessageSegment.text(
                f" {reply}"
            )
            await bot.send_group_msg(group_id=event.group_id, message=message)


__all__ = [
    "challenge_commands",
    "prepare_cmd",
    "private_router",
    "group_battle_router",
    "status_cmd",
    "check_cmd",
    "team_manager_cmd",
    "sample_team_cmd",
    "recommended_set_cmd",
    "team_guide_cmd",
    "team_guide_router",
    "pokemon_help_cmd",
    "register_commands",
]
