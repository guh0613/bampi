from __future__ import annotations

import asyncio
import copy
import re
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, MessageSegment

from .ai_opponent import (
    AIBattleDecisionContext,
    BattleAIAgentSession,
    BattleAIOpponent,
)
from .battle import (
    BattleEventFormatter,
    BattleState,
    PlayerSlot,
    PromptBuilder,
    simplify_ident,
)
from .battle.action_guide import build_ai_action_guide
from .bridge import ShowdownBattleProcess, ShowdownBridgeError, ShowdownRuntime
from .move_data import MoveDataRepository
from .rendering import PokemonBattleRenderer, TeamPreviewPokemon, item_label_key
from .translations import TranslationService

if TYPE_CHECKING:
    from .formats import BattleFormatConfig
    from .manager import BattleManager


_GROUP_CHOICE_PATTERN = re.compile(
    r"^(?:move\d*|switch\d*|pass\d*|team|default|forfeit|undo)(?:\s|$)",
    re.IGNORECASE,
)


@dataclass
class BattleSession:
    session_id: str
    group_id: int
    players: Dict[str, PlayerSlot]
    manager: "BattleManager"
    translator: TranslationService
    format_config: BattleFormatConfig
    runtime: ShowdownRuntime
    move_repository: MoveDataRepository
    renderer: PokemonBattleRenderer
    render_semaphore: asyncio.Semaphore
    ai_opponent: BattleAIOpponent | None = None
    ai_public_history_events: int = 80
    process: Optional[ShowdownBattleProcess] = None
    state: str = "pending"
    current_requests: Dict[str, dict] = field(default_factory=dict)
    _awaiting_resolution: set[str] = field(default_factory=set)
    _event_task: Optional[asyncio.Task[None]] = None
    _group_buffer: List[str] = field(default_factory=list)
    _team_preview_data: Dict[str, List[TeamPreviewPokemon]] = field(
        default_factory=dict
    )
    _team_preview_announced: bool = False
    _team_summary_sent: set[str] = field(default_factory=set)
    _interaction_channels: Dict[str, Literal["private", "group"]] = field(
        default_factory=dict
    )
    _invite_task: Optional[asyncio.Task[None]] = None
    _action_timers: Dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _action_pre_alerts: Dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _request_seq: Dict[str, int] = field(default_factory=dict)
    _timeout_counts: Dict[str, int] = field(default_factory=dict)
    _ai_action_tasks: Dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _ai_choice_error_counts: Dict[str, int] = field(default_factory=dict)
    _ai_fallback_notice_sent: bool = False
    _ai_agent: BattleAIAgentSession | None = field(init=False, default=None)
    _ai_pending_commentary: str = ""
    _public_history: Deque[str] = field(init=False)
    _public_revealed_moves: Dict[str, set[str]] = field(default_factory=dict)
    _public_revealed_items: Dict[str, str] = field(default_factory=dict)
    # Items published by open-team-sheet style rules, keyed by
    # ``side|nickname``. Unlike one-off battle-event reveals (which players
    # are expected to remember themselves), team sheets stay consultable for
    # the whole battle, so drawing them on group-visible images is fair.
    _open_sheet_items: Dict[str, str] = field(default_factory=dict)
    _public_revealed_abilities: Dict[str, str] = field(default_factory=dict)
    _public_revealed_tera_types: Dict[str, str] = field(default_factory=dict)
    battle_state: BattleState = field(default_factory=BattleState)
    formatter: BattleEventFormatter = field(init=False)
    prompts: PromptBuilder = field(init=False)
    _start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _closed: bool = False

    def __post_init__(self) -> None:
        self._public_history = deque(maxlen=max(10, self.ai_public_history_events))
        self._interaction_channels = {
            side: "private" for side, player in self.players.items() if not player.is_ai
        }
        if self.ai_opponent is not None:
            self._ai_agent = self.ai_opponent.create_battle_agent(
                battle_id=self.session_id,
                format_config=self.format_config,
            )
        self.formatter = BattleEventFormatter(
            state=self.battle_state, translator=self.translator, players=self.players
        )
        self.prompts = PromptBuilder(self.translator, self.move_repository)

    @property
    def format_id(self) -> str:
        return self.format_config.format_id

    @property
    def is_ai_battle(self) -> bool:
        return self.ai_opponent is not None and any(
            player.is_ai for player in self.players.values()
        )

    def get_side_by_user(self, user_id: str) -> Optional[str]:
        for side, player in self.players.items():
            if player.user_id == user_id:
                return side
        return None

    def get_player(self, side: str) -> PlayerSlot:
        return self.players[side]

    def interaction_channel_for_side(self, side: str) -> Literal["private", "group"]:
        return self._interaction_channels.get(side, "private")

    def interaction_channel_for_user(self, user_id: str) -> Literal["private", "group"]:
        side = self.get_side_by_user(user_id)
        return self.interaction_channel_for_side(side) if side else "private"

    def set_interaction_channel(
        self, user_id: str, channel: Literal["private", "group"]
    ) -> str:
        side = self.get_side_by_user(user_id)
        if not side or self.players[side].is_ai:
            raise ShowdownBridgeError("未找到对应的对战会话。")
        if channel == "group" and not self.is_ai_battle:
            raise ShowdownBridgeError("玩家对战仅支持在私聊中提交行动。")
        self._interaction_channels[side] = channel
        return side

    def teams_ready(self) -> bool:
        if self.format_config.requires_team:
            return all(p.team_pack for p in self.players.values())
        return all(p.team_pack is not None for p in self.players.values())

    async def set_team(self, user_id: str, pack: str, raw: Optional[str]) -> str:
        side = self.get_side_by_user(user_id)
        if not side:
            raise ShowdownBridgeError("未找到对应的对战会话。")
        player = self.players[side]
        player.team_pack = pack
        player.team_raw = raw
        return side

    def build_specs(self) -> Tuple[dict, dict]:
        p1 = self.players["p1"]
        p2 = self.players["p2"]
        if self.format_config.requires_team:
            if not p1.team_pack or not p2.team_pack:
                raise ShowdownBridgeError("队伍尚未准备完成。")
            p1_team = p1.team_pack
            p2_team = p2.team_pack
        else:
            if p1.team_pack is None or p2.team_pack is None:
                raise ShowdownBridgeError("尚未确认双方已准备。")
            p1_team = p1.team_pack or ""
            p2_team = p2.team_pack or ""
        return (
            {"name": p1.display_name, "team": p1_team},
            {"name": p2.display_name, "team": p2_team},
        )

    async def start(self, bot: Bot) -> None:
        async with self._start_lock:
            if self.process or self.state != "pending":
                return
            p1_spec, p2_spec = self.build_specs()
            process = self.runtime.create_battle_process(
                format_id=self.format_id,
                p1=p1_spec,
                p2=p2_spec,
                logger=logger,
            )
            self.process = process
            try:
                await process.start()
            except Exception:
                self.process = None
                raise
            self.state = "active"
            self._event_task = asyncio.create_task(
                self._consume_events(bot),
                name=f"showdown-session-{self.session_id}",
            )
            if self._invite_task:
                self._invite_task.cancel()
                self._invite_task = None

        notify = "对战已开始，请留意提示并尽快完成指令。"
        for side, player in self.players.items():
            if player.is_ai or self.interaction_channel_for_side(side) == "group":
                continue
            try:
                await self._send_player_message(bot, side, notify)
            except Exception:
                logger.exception(
                    f"failed to send showdown start notice user_id={player.user_id}"
                )
        display = self.format_config.display_name or "Showdown"
        if self.is_ai_battle:
            p1 = self.players["p1"].display_name
            p2 = self.players["p2"].display_name
            start_notice = (
                f"{display} 对战已启动：{p1} vs {p2}。回合日志将同步发送到群内。"
            )
        else:
            start_notice = f"{display} 玩家对战已启动，回合日志将同步发送到群内。"
        try:
            await self._send_group(bot, start_notice)
        except Exception:
            logger.exception("failed to send showdown start notice to group")

    async def _consume_events(self, bot: Bot) -> None:
        assert self.process
        queue = self.process.events
        try:
            while True:
                payload = await queue.get()
                etype = payload.get("type")
                if etype == "update":
                    self._record_public_reveal(payload["line"])
                    formatted = self.formatter.format(payload["line"])
                    if formatted:
                        self._group_buffer.append(formatted)
                        self._public_history.append(formatted)
                elif etype == "separator":
                    await self._flush_group(bot)
                elif etype == "request":
                    side = payload.get("side")
                    request_payload = payload["payload"]
                    if request_payload.get("teamPreview"):
                        await self._handle_team_preview_request(
                            bot, side, request_payload
                        )
                    if side:
                        self._awaiting_resolution.discard(side)
                        # Side requests contain private roster and exact HP data.
                        # Keep them out of the public battle state used by group
                        # logs/status panels; public update lines maintain it.
                        self.current_requests[side] = request_payload
                        player = self.players.get(side)
                        if player is not None and player.is_ai:
                            if self._is_actionable_request(request_payload):
                                # A new actionable request means the previous
                                # choice has resolved and is now public. A
                                # transient `wait` request does not. Flush any
                                # buffered public log before publishing speech.
                                await self._flush_group(bot)
                                await self._flush_ai_commentary(bot)
                            other_side = "p1" if side == "p2" else "p2"
                            preview_waiting = (
                                request_payload.get("teamPreview")
                                and other_side not in self._team_preview_data
                            )
                            if not preview_waiting:
                                self._schedule_ai_action(bot, side, request_payload)
                        else:
                            await self._notify_request(bot, side, request_payload)
                            await self._schedule_action_timeout(
                                bot, side, request_payload
                            )
                elif etype == "error":
                    await self._notify_error(bot, payload)
                elif etype == "win":
                    await self._flush_group(bot)
                    await self._flush_ai_commentary(bot)
                    await self._send_group(bot, f"对战结束，{payload['winner']} 获胜！")
                    break
                elif etype == "tie":
                    await self._flush_group(bot)
                    await self._flush_ai_commentary(bot)
                    await self._send_group(bot, "对战结束，平局！")
                    break
                elif etype == "bridge_error":
                    await self._flush_group(bot)
                    await self._flush_ai_commentary(bot)
                    await self._send_group(bot, "对战出现异常，房间已关闭。")
                    break
                elif etype in {"stream_end", "terminated"}:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"showdown session event loop failed: {self.session_id}")
        finally:
            try:
                await self._flush_group(bot)
            except Exception:
                logger.exception(
                    f"failed to flush showdown group log: {self.session_id}"
                )
            # Ensure timers are stopped even when a QQ send/render operation fails.
            for task in list(self._action_timers.values()):
                task.cancel()
            self._action_timers.clear()
            for task in list(self._action_pre_alerts.values()):
                task.cancel()
            self._action_pre_alerts.clear()
            ai_tasks = list(self._ai_action_tasks.values())
            for task in ai_tasks:
                task.cancel()
            self._ai_action_tasks.clear()
            if ai_tasks:
                await asyncio.gather(*ai_tasks, return_exceptions=True)
            await self._close_ai_agent()
            if self._invite_task:
                self._invite_task.cancel()
                self._invite_task = None
            process = self.process
            self.process = None
            if process is not None:
                try:
                    await process.terminate()
                except Exception:
                    logger.exception(
                        f"failed to terminate showdown process: {self.session_id}"
                    )
            self.state = "finished"
            self._closed = True
            await self.manager.remove(self)

    async def _notify_error(self, bot: Bot, payload: dict) -> None:
        side = payload.get("side")
        message = payload.get("message")
        if not side or not message:
            return
        player = self.players.get(side)
        if not player:
            return
        self._awaiting_resolution.discard(side)
        request = self.current_requests.get(side)
        if player.is_ai:
            count = self._ai_choice_error_counts.get(side, 0) + 1
            self._ai_choice_error_counts[side] = count
            logger.warning(
                "showdown rejected AI choice "
                f"battle_id={self.session_id} side={side} count={count} "
                f"message={message}"
            )
            if request and self.process and self._is_actionable_request(request):
                fallback = self._default_choice_for_request(request)
                # Avoid leaving the battle deadlocked if even the deterministic
                # fallback is rejected repeatedly by the local simulator.
                if count >= 2 or not fallback:
                    fallback = "forfeit"
                self._awaiting_resolution.add(side)
                try:
                    await self.process.send_choice(side, fallback)
                except ShowdownBridgeError:
                    self._awaiting_resolution.discard(side)
            return
        await self._send_player_message(bot, side, f"❌ 指令无效：{message}")
        if request:
            await self._notify_request(bot, side, request)

    def _items_for_viewer(self, viewer_side: Optional[str]) -> Dict[str, str]:
        """Raw held-item ids safe to show a given audience.

        ``viewer_side=None`` builds the public (group) view: only items
        published by open-team-sheet style rules, plus — in battles against
        the bot — the human players' own items, since there is no opponent to
        hide them from. Items merely revealed by battle events are *not*
        drawn: like on cartridge, remembering those is part of the game. A
        player viewer additionally sees their own team's items from their
        private request. The renderer resolves ids into icons and localized
        labels.
        """
        items: Dict[str, str] = dict(self._open_sheet_items)
        own_sides: set[str] = set()
        if self.is_ai_battle:
            own_sides.update(
                side for side, player in self.players.items() if not player.is_ai
            )
        if viewer_side:
            player = self.players.get(viewer_side)
            if player is not None and not player.is_ai:
                own_sides.add(viewer_side)
        for side in own_sides:
            request = self.current_requests.get(side) or {}
            for mon in (request.get("side") or {}).get("pokemon") or []:
                ident = mon.get("ident") or ""
                item = (mon.get("item") or "").strip()
                if not ident or not item:
                    continue
                items[item_label_key(side, simplify_ident(ident))] = item
        return items

    async def _flush_group(self, bot: Bot) -> None:
        if not self._group_buffer:
            return
        lines = list(self._group_buffer)
        self._group_buffer.clear()
        render_kwargs = dict(
            format_name=self.format_config.display_name or "Showdown",
            format_id=self.format_id,
            players=self.players,
            state=self.battle_state,
            translator=self.translator,
            lines=lines,
        )
        public_items = self._items_for_viewer(None)
        public_image = await self._render(
            self.renderer.render_turn_log,
            items=public_items,
            **render_kwargs,
        )
        if not public_image:
            await self._send_battle_update(bot, "\n".join(lines))
            return
        await bot.send_group_msg(
            group_id=self.group_id, message=MessageSegment.image(public_image)
        )
        for side, player in self.players.items():
            if player.is_ai or self.interaction_channel_for_side(side) != "private":
                continue
            image_data = public_image
            viewer_items = self._items_for_viewer(side)
            if viewer_items != public_items:
                # Personalized view: includes the viewer's own held items,
                # which must never reach the group or the opponent.
                variant = await self._render(
                    self.renderer.render_turn_log,
                    items=viewer_items,
                    **render_kwargs,
                )
                if variant:
                    image_data = variant
            try:
                await bot.send_private_msg(
                    user_id=int(player.user_id),
                    message=MessageSegment.image(image_data),
                )
            except Exception:
                logger.exception(
                    "failed to mirror showdown battle update "
                    f"battle_id={self.session_id} user_id={player.user_id}"
                )

    async def _render(
        self,
        renderer: Callable[..., Awaitable[Optional[str]]],
        **kwargs: Any,
    ) -> Optional[str]:
        async with self.render_semaphore:
            return await renderer(**kwargs)

    async def _send_group(self, bot: Bot, text: str) -> None:
        if text.strip():
            await bot.send_group_msg(group_id=self.group_id, message=text)

    async def _send_player_message(self, bot: Bot, side: str, message: Any) -> None:
        player = self.players.get(side)
        if player is None or player.is_ai:
            return
        if self.interaction_channel_for_side(side) == "group":
            outbound = MessageSegment.at(int(player.user_id)) + MessageSegment.text(" ")
            outbound += message
            await bot.send_group_msg(group_id=self.group_id, message=outbound)
            return
        await bot.send_private_msg(user_id=int(player.user_id), message=message)

    async def _send_battle_update(self, bot: Bot, message: Any) -> None:
        """Publish a public battle update and mirror it to private-mode players."""
        await bot.send_group_msg(group_id=self.group_id, message=message)
        for side, player in self.players.items():
            if player.is_ai or self.interaction_channel_for_side(side) != "private":
                continue
            try:
                await bot.send_private_msg(user_id=int(player.user_id), message=message)
            except Exception:
                logger.exception(
                    "failed to mirror showdown battle update "
                    f"battle_id={self.session_id} user_id={player.user_id}"
                )

    async def _flush_ai_commentary(self, bot: Bot) -> None:
        commentary = self._ai_pending_commentary.strip()
        if not commentary:
            return
        self._ai_pending_commentary = ""
        ai_player = next(
            (player for player in self.players.values() if player.is_ai),
            None,
        )
        display_name = ai_player.display_name if ai_player else self.manager.bot_name
        await self._send_group(bot, f"{display_name}：{commentary}")

    async def _handle_team_preview_request(
        self, bot: Bot, side: Optional[str], request: dict
    ) -> None:
        if not side:
            return
        side_info = request.get("side", {})
        pokemon = side_info.get("pokemon", []) or []
        lineup: List[TeamPreviewPokemon] = []
        for mon in pokemon:
            ident = mon.get("ident") or ""
            details = mon.get("details", "") or ""
            base_name = simplify_ident(ident) if ident else ""
            if ident:
                state = self.battle_state.register_pokemon(
                    ident, details, self.translator
                )
                if state and state.name:
                    translated_name = state.name
                else:
                    translated_name = self.translator.translate_species(base_name)
            else:
                translated_name = self.translator.translate_details(details)
            species_name = (
                mon.get("species") or base_name or details.split(",")[0].strip()
            )
            translated_name = translated_name or species_name or "未知宝可梦"

            level: Optional[int] = None
            gender_symbol: Optional[str] = None
            if details:
                for token in [segment.strip() for segment in details.split(",")[1:]]:
                    if not token:
                        continue
                    if token in {"M", "F"}:
                        gender_symbol = "♂" if token == "M" else "♀"
                    elif token.startswith("L"):
                        digits = token[1:]
                        if digits.isdigit():
                            level = int(digits)

            if level is None and self.format_config.picked_team_size:
                level = 50

            # Team-preview requests are side-private and include held items,
            # abilities and Tera types. Only copy publicly visible fields into
            # the group preview model.
            lineup.append(
                TeamPreviewPokemon(
                    ident=ident or species_name or translated_name,
                    species=species_name or base_name or translated_name,
                    display_name=translated_name,
                    level=level,
                    gender=gender_symbol,
                    item=None,
                    ability=None,
                    tera_type=None,
                )
            )
        idents = [mon.get("ident") for mon in pokemon if mon.get("ident")]
        if side and idents:
            self.battle_state.side_rosters[side] = idents
        self._team_preview_data[side] = lineup
        if self._team_preview_announced:
            return
        other_side = "p1" if side == "p2" else "p2"
        if other_side not in self._team_preview_data:
            return
        await self._flush_group(bot)
        image_data = await self._render(
            self.renderer.render_team_preview,
            format_name=self.format_config.display_name or "Showdown",
            format_id=self.format_id,
            players=self.players,
            preview=self._team_preview_data,
        )
        if image_data:
            await self._send_battle_update(bot, MessageSegment.image(image_data))
        else:
            message_lines = ["【队伍预览】"]
            for slot in ("p1", "p2"):
                player = self.players.get(slot)
                entries = self._team_preview_data.get(slot, [])
                display = player.display_name if player else slot
                if entries:
                    formatted = "、".join(
                        f"{idx + 1}. {entry.display_name}"
                        for idx, entry in enumerate(entries)
                    )
                else:
                    formatted = "暂无队伍信息"
                message_lines.append(f"{display}：{formatted}")
            await self._send_battle_update(bot, "\n".join(message_lines))
        self._team_preview_announced = True
        # If the simulator delivered the AI side's preview request first, its
        # decision was intentionally held until both public rosters existed.
        for candidate_side, player in self.players.items():
            if not player.is_ai:
                continue
            candidate_request = self.current_requests.get(candidate_side)
            if candidate_request and candidate_request.get("teamPreview"):
                self._schedule_ai_action(bot, candidate_side, candidate_request)

    async def _notify_request(self, bot: Bot, side: str, request: dict) -> None:
        player = self.players.get(side)
        if not player:
            return
        should_send_summary = False
        if not self.format_config.requires_team:
            should_send_summary = True
        elif request.get("teamPreview"):
            should_send_summary = True
        if should_send_summary and side not in self._team_summary_sent:
            if self.format_config.requires_team:
                summary = self.prompts.build_full_team_summary(request)
            else:
                summary = self.prompts.build_random_team_summary(request)
            if summary:
                await self._send_player_message(bot, side, summary)
                self._team_summary_sent.add(side)
        patched_request = self._request_with_format_constraints(request)
        message = self.prompts.build_request_prompt(patched_request)
        # Append timeout hint per request type
        if message:
            if request.get("teamPreview"):
                secs = int(self.format_config.preview_timeout)
            elif request.get("forceSwitch"):
                secs = int(self.format_config.switch_timeout)
            elif request.get("active"):
                secs = int(self.format_config.move_timeout)
            else:
                secs = 0
            if secs > 0:
                message += f"\n（请在{secs}秒内完成操作，超时将自动执行默认操作）"
        if message:
            await self._send_player_message(bot, side, message)

    def _record_public_reveal(self, line: str) -> None:
        """Retain durable facts from the bridge's already-filtered public stream."""
        if line.startswith("|showteam|"):
            # Open-team-sheet reveal; the packed payload itself contains "|".
            parts = line.split("|", 3)
            if len(parts) >= 4:
                self._record_open_team_sheet(parts[2].strip(), parts[3])
            return
        fields = line.split("|")
        if len(fields) < 4:
            return
        event_type = fields[1]
        ident = fields[2].strip()
        if not ident.startswith(("p1", "p2")):
            return
        value = fields[3].strip()
        if not value:
            return
        if event_type == "move":
            self._public_revealed_moves.setdefault(ident, set()).add(value)
        elif event_type in {"-item", "-enditem"}:
            self._public_revealed_items[ident] = value
        elif event_type == "-ability":
            self._public_revealed_abilities[ident] = value
        elif event_type == "-activate" and value.lower().startswith("ability:"):
            ability = value.split(":", 1)[1].strip()
            if ability:
                self._public_revealed_abilities[ident] = ability
        elif event_type == "-terastallize":
            self._public_revealed_tera_types[ident] = value
        elif event_type == "-mega" and len(fields) >= 5:
            stone = fields[4].strip()
            if stone:
                self._public_revealed_items[ident] = stone

    def _record_open_team_sheet(self, side: str, packed_team: str) -> None:
        """Record held items published by rules such as Force Open Team Sheets."""
        side = (side or "").strip().lower()
        if side not in {"p1", "p2"}:
            return
        for entry in packed_team.split("]"):
            entry_fields = entry.split("|")
            if len(entry_fields) < 3:
                continue
            nickname = entry_fields[0].strip()
            species = entry_fields[1].strip() or nickname
            item = entry_fields[2].strip()
            if not item:
                continue
            for name in {nickname, species}:
                if name:
                    self._open_sheet_items[item_label_key(side, name)] = item
            self._public_revealed_items.setdefault(
                f"{side}: {nickname or species}", item
            )

    def _build_public_knowledge(self) -> tuple[str, ...]:
        idents = sorted(
            set(self._public_revealed_moves)
            | set(self._public_revealed_items)
            | set(self._public_revealed_abilities)
            | set(self._public_revealed_tera_types)
        )
        lines: list[str] = []
        for ident in idents:
            facts: list[str] = []
            moves = sorted(self._public_revealed_moves.get(ident, ()))
            if moves:
                facts.append("moves=" + ", ".join(moves))
            item = self._public_revealed_items.get(ident)
            if item:
                facts.append(f"item={item}")
            ability = self._public_revealed_abilities.get(ident)
            if ability:
                facts.append(f"ability={ability}")
            tera_type = self._public_revealed_tera_types.get(ident)
            if tera_type:
                facts.append(f"tera_type={tera_type}")
            if facts:
                lines.append(f"{ident}: " + "; ".join(facts))
        return tuple(lines)

    def _team_preview_selection_size(self, request: dict) -> int | None:
        for value in (
            request.get("maxChosenTeamSize"),
            request.get("maxTeamSize"),
            request.get("maxTeam"),
            self.format_config.picked_team_size,
        ):
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        pokemon = (request.get("side") or {}).get("pokemon") or []
        return len(pokemon) or None

    def _parse_team_preview_choice(
        self,
        request: dict,
        text: str,
    ) -> tuple[str | None, str | None]:
        parts = text.strip().split(maxsplit=1)
        if not parts or parts[0].lower() != "team":
            return None, "队伍预览请使用 team 指令。"
        if len(parts) == 1 or parts[1].strip().lower() == "default":
            return "team", None

        raw_order = parts[1].strip().replace("，", ",")
        pokemon = (request.get("side") or {}).get("pokemon") or []
        team_size = len(pokemon)
        selection_size = self._team_preview_selection_size(request)
        if not team_size or not selection_size:
            return None, "当前队伍预览数据不完整，请稍后重试。"

        if "," in raw_order:
            raw_indexes = [token.strip() for token in raw_order.split(",")]
            if not raw_indexes or any(not token.isdigit() for token in raw_indexes):
                return None, "队伍编号请使用数字，并用英文逗号分隔。"
            indexes = [int(token) for token in raw_indexes]
        else:
            compact = "".join(raw_order.split())
            if not compact.isdigit():
                return None, "队伍编号只能包含数字。"
            if team_size >= 10:
                return None, "队伍超过 9 只时，请用英文逗号分隔每个编号。"
            indexes = [int(token) for token in compact]

        if len(indexes) != selection_size:
            return (
                None,
                f"本规则需要从 {team_size} 只中选择 {selection_size} 只；"
                f"请恰好填写 {selection_size} 个编号，不要给全部成员排序。",
            )
        if len(set(indexes)) != len(indexes):
            return None, "同一只宝可梦不能在队伍预览中重复选择。"
        invalid = [index for index in indexes if index < 1 or index > team_size]
        if invalid:
            return None, f"队伍编号应在 1 到 {team_size} 之间。"

        encoded = (
            ",".join(str(index) for index in indexes)
            if team_size >= 10
            else "".join(str(index) for index in indexes)
        )
        return f"team {encoded}", None

    def _request_with_format_constraints(self, request: dict) -> dict:
        picked_team_size = self.format_config.picked_team_size
        has_selection_size = any(
            request.get(key) for key in ("maxChosenTeamSize", "maxTeamSize", "maxTeam")
        )
        if request.get("teamPreview") and picked_team_size and not has_selection_size:
            patched_request = dict(request)
            patched_request["maxChosenTeamSize"] = picked_team_size
            return patched_request
        return request

    @staticmethod
    def _is_actionable_request(request: dict) -> bool:
        return not request.get("wait") and any(
            request.get(key) for key in ("teamPreview", "forceSwitch", "active")
        )

    def _schedule_ai_action(self, bot: Bot, side: str, request: dict) -> None:
        ai_agent = self._ai_agent
        player = self.players.get(side)
        if (
            ai_agent is None
            or player is None
            or not player.is_ai
            or self.state != "active"
        ):
            return
        previous = self._ai_action_tasks.pop(side, None)
        if previous is not None and not previous.done():
            previous.cancel()
        seq = self._request_seq.get(side, 0) + 1
        self._request_seq[side] = seq
        if not self._is_actionable_request(request):
            return
        self._ai_choice_error_counts[side] = 0
        request_snapshot = copy.deepcopy(self._request_with_format_constraints(request))
        decision = AIBattleDecisionContext(
            battle_id=self.session_id,
            format_id=self.format_id,
            format_name=self.format_config.display_name or self.format_id,
            game_type=self.format_config.game_type,
            ai_side=side,
            turn_number=self.battle_state.turn_number,
            private_request=request_snapshot,
            action_guide=build_ai_action_guide(
                request_snapshot,
                picked_team_size=self.format_config.picked_team_size,
            ),
            public_status=self.build_status_report(),
            public_knowledge=self._build_public_knowledge(),
            public_events=tuple(self._public_history),
        )

        async def _run() -> None:
            used_fallback = False
            try:
                try:
                    result = await ai_agent.choose_action(
                        decision,
                        normalize_choice=lambda candidate: self._parse_choice(
                            request_snapshot, candidate
                        ),
                    )
                    choice = result.choice
                    if not choice:
                        raise ValueError("AI 未通过工具提交适用于当前请求的行动。")
                    if result.commentary:
                        self._ai_pending_commentary = result.commentary
                except asyncio.CancelledError:
                    raise
                except Exception:
                    used_fallback = True
                    logger.exception(
                        "showdown AI decision failed; using deterministic fallback "
                        f"battle_id={self.session_id} side={side}"
                    )
                    choice = self._default_choice_for_request(request_snapshot)
                    if not choice:
                        choice = "forfeit"

                if self.state != "active" or self._request_seq.get(side) != seq:
                    return
                if self.current_requests.get(side) is not request:
                    return
                process = self.process
                if process is None or side in self._awaiting_resolution:
                    return
                self._awaiting_resolution.add(side)
                try:
                    await process.send_choice(side, choice)
                except ShowdownBridgeError:
                    self._awaiting_resolution.discard(side)
                    self._ai_pending_commentary = ""
                    raise
                logger.info(
                    "showdown AI choice submitted "
                    f"battle_id={self.session_id} side={side} "
                    f"turn={self.battle_state.turn_number} fallback={used_fallback}"
                )
                if used_fallback and not self._ai_fallback_notice_sent:
                    self._ai_fallback_notice_sent = True
                    try:
                        bot_name = self.players[side].display_name
                        await self._send_group(
                            bot,
                            f"{bot_name} 本回合未能正常提交行动，"
                            "已执行安全的默认操作。",
                        )
                    except Exception:
                        logger.exception(
                            "failed to send showdown AI fallback notice "
                            f"battle_id={self.session_id}"
                        )
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception(
                    "showdown AI action task failed "
                    f"battle_id={self.session_id} side={side}"
                )
            finally:
                current = asyncio.current_task()
                if self._ai_action_tasks.get(side) is current:
                    self._ai_action_tasks.pop(side, None)

        self._ai_action_tasks[side] = asyncio.create_task(
            _run(), name=f"showdown-ai-{self.session_id}-{side}-{seq}"
        )

    def can_accept_group_choice(self, user_id: str, text: str) -> bool:
        """Return whether a group message belongs to the human battle channel.

        Routing is intentionally based on command shape rather than current
        request state. This lets duplicate/early/invalid actions receive a
        battle-specific error and prevents them from falling through to the
        ordinary group-chat agent.
        """
        if not self.is_ai_battle or self.state != "active":
            return False
        side = self.get_side_by_user(user_id)
        if not side or self.players[side].is_ai:
            return False
        if self.interaction_channel_for_side(side) != "group":
            return False
        return bool(_GROUP_CHOICE_PATTERN.match(text.strip()))

    def can_accept_private_choice(self, user_id: str) -> bool:
        side = self.get_side_by_user(user_id)
        return bool(
            side
            and not self.players[side].is_ai
            and self.interaction_channel_for_side(side) == "private"
        )

    async def handle_choice(self, bot: Bot, user_id: str, text: str) -> Optional[str]:
        side = self.get_side_by_user(user_id)
        if not side:
            return "你当前没有正在进行的对战。"
        if not self.process:
            return "对战尚未开始，请先导入队伍。"
        request = self.current_requests.get(side)
        if not request:
            return "当前无需操作，请稍候。"
        if request.get("wait"):
            return "当前无需操作，请稍候。"
        if side in self._awaiting_resolution:
            return "上一条指令处理中，请稍候对战回应。"
        choice_parts = text.strip().split(maxsplit=1)
        is_team_command = bool(choice_parts) and choice_parts[0].lower() == "team"
        if request.get("teamPreview") and is_team_command:
            choice, preview_error = self._parse_team_preview_choice(request, text)
            if preview_error:
                return preview_error
        else:
            choice = self._parse_choice(request, text)
        if not choice:
            return "未能识别指令，请按照提示格式输入。"
        if choice == "forfeit":
            await self._flush_group(bot)
            player = self.players.get(side)
            if player:
                await self._send_group(bot, f"{player.display_name} 宣布认输。")
            try:
                await self.process.send_choice(side, choice)
            except ShowdownBridgeError as exc:
                return f"认输指令发送失败：{exc}"
            self.current_requests.pop(side, None)
            return "认输指令已提交。"
        # User responded in time; cancel any pending timer for this side
        timer = self._action_timers.pop(side, None)
        if timer:
            timer.cancel()
        warn_timer = self._action_pre_alerts.pop(side, None)
        if warn_timer:
            warn_timer.cancel()
        self._awaiting_resolution.add(side)
        try:
            await self.process.send_choice(side, choice)
        except ShowdownBridgeError as exc:
            self._awaiting_resolution.discard(side)
            return f"指令发送失败：{exc}"
        return "指令已提交。"

    def build_status_report(self) -> str:
        return self.battle_state.build_status_report(self.players, self.translator)

    async def render_status_image(
        self, viewer_user_id: Optional[str] = None
    ) -> Optional[str]:
        viewer_side = (
            self.get_side_by_user(viewer_user_id) if viewer_user_id else None
        )
        return await self._render(
            self.renderer.render_status_panel,
            format_name=self.format_config.display_name or "Showdown",
            format_id=self.format_id,
            players=self.players,
            state=self.battle_state,
            translator=self.translator,
            items=self._items_for_viewer(viewer_side),
        )

    @staticmethod
    def _default_action_for_actor(actor: dict, *, doubles: bool) -> str:
        moves = actor.get("moves", []) or []
        for idx, move in enumerate(moves, start=1):
            if move.get("disabled"):
                continue
            pp = move.get("pp")
            try:
                if pp is not None and int(pp) <= 0:
                    continue
            except (TypeError, ValueError):
                pass
            action = f"move {idx}"
            if doubles and move.get("target") in {"normal", "any", "adjacentFoe"}:
                action += " 1"
            return action
        return "pass"

    def _parse_choice(self, request: dict, text: str) -> Optional[str]:
        text = text.strip()
        if not text:
            return None
        lower = text.lower()
        if lower == "forfeit":
            return "forfeit"
        if lower == "undo":
            return "undo"
        if lower == "default":
            return self._default_choice_for_request(request)
        # Team preview ordering / selection
        if request.get("teamPreview"):
            choice, _ = self._parse_team_preview_choice(request, text)
            return choice

        # Determine number of active actors (for doubles)
        active_entries = request.get("active") or []
        if not active_entries:
            side_info = request.get("side", {}) or {}
            pokemon = side_info.get("pokemon", []) or []
            active_entries = [mon for mon in pokemon if mon.get("active")]
        num_slots = max(1, len(active_entries))

        # Helper to compute default per actor (used to fill gaps)
        def _default_for_actor(actor: dict) -> str:
            return self._default_action_for_actor(actor, doubles=num_slots > 1)

        # Parse combined actions for doubles
        separators = [";", "；", "|", ","]
        temp = text
        for sep in separators:
            temp = temp.replace(sep, ";")
        segments = [seg.strip() for seg in temp.split(";") if seg.strip()]
        # Short-circuits for simple single commands
        if len(segments) == 1 and num_slots == 1:
            cmd = segments[0].lower()
            if cmd.startswith("move"):
                parts = cmd.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    suffix = ""
                    if len(parts) >= 3:
                        option = parts[2]
                        if option in {
                            "zmove",
                            "mega",
                            "dynamax",
                            "ultra",
                            "terastallize",
                            "tera",
                        }:
                            suffix = (
                                " terastallize" if option == "tera" else f" {option}"
                            )
                    return f"move {parts[1]}{suffix}"
            if cmd.startswith("switch"):
                parts = cmd.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return f"switch {parts[1]}"
            if cmd == "pass":
                return "pass"

        # Multi-slot parsing (move1/move2, switch1/switch2, pass1/pass2)
        actions: List[Optional[str]] = [None] * num_slots
        for seg in segments:
            raw = seg.strip()
            if not raw:
                continue
            tokens = raw.split()
            head = tokens[0].lower()
            actor_idx: Optional[int] = None
            action: Optional[str] = None
            # Resolve actor index in head (e.g. move1/move2)
            if head.startswith("move") and len(head) > 4 and head[4:].isdigit():
                actor_idx = int(head[4:])
                # remaining tokens carry args
                args = tokens[1:]
                if not args:
                    continue
                move_idx = None
                suffix_bits: List[str] = []
                for tok in args:
                    if tok.isdigit() and move_idx is None:
                        move_idx = tok
                        continue
                    low = tok.lower()
                    if low == "tera":
                        suffix_bits.append("terastallize")
                    elif low in {"zmove", "mega", "dynamax", "ultra", "terastallize"}:
                        suffix_bits.append(low)
                    else:
                        suffix_bits.append(tok)
                if move_idx is None:
                    continue
                action = " ".join(["move", move_idx, *suffix_bits]).strip()
            elif head.startswith("switch") and len(head) > 6 and head[6:].isdigit():
                actor_idx = int(head[6:])
                args = tokens[1:]
                if not args or not args[0].isdigit():
                    continue
                action = f"switch {args[0]}"
            elif head.startswith("pass") and len(head) > 4 and head[4:].isdigit():
                actor_idx = int(head[4:])
                action = "pass"
            else:
                # No explicit actor index; assign to the next free slot
                for i in range(num_slots):
                    if actions[i] is None:
                        actor_idx = i + 1
                        break
                if head.startswith("move"):
                    args = tokens[1:]
                    if not args or not args[0].isdigit():
                        continue
                    move_idx = args[0]
                    suffix_bits: List[str] = []
                    for tok in args[1:]:
                        low = tok.lower()
                        if low == "tera":
                            suffix_bits.append("terastallize")
                        elif low in {
                            "zmove",
                            "mega",
                            "dynamax",
                            "ultra",
                            "terastallize",
                        }:
                            suffix_bits.append(low)
                        else:
                            suffix_bits.append(tok)
                    action = " ".join(["move", move_idx, *suffix_bits]).strip()
                elif head.startswith("switch"):
                    args = tokens[1:]
                    if not args or not args[0].isdigit():
                        continue
                    action = f"switch {args[0]}"
                elif head == "pass":
                    action = "pass"
                else:
                    continue

            # Assign parsed action
            if actor_idx is None or actor_idx < 1 or actor_idx > num_slots:
                continue
            actions[actor_idx - 1] = action

        # If we parsed any multi-slot actions, fill gaps and return
        if any(a is not None for a in actions):
            for i in range(num_slots):
                if actions[i] is None:
                    actor = active_entries[i] if i < len(active_entries) else {}
                    actions[i] = _default_for_actor(actor)
            return ", ".join(a or "pass" for a in actions)

        # Simple fallbacks
        if lower.startswith("move"):
            parts = lower.split()
            if len(parts) >= 2 and parts[1].isdigit():
                suffix = ""
                if len(parts) >= 3:
                    option = parts[2]
                    if option in {
                        "zmove",
                        "mega",
                        "dynamax",
                        "ultra",
                        "terastallize",
                        "tera",
                    }:
                        suffix = " terastallize" if option == "tera" else f" {option}"
                return f"move {parts[1]}{suffix}"
        if lower.startswith("switch"):
            parts = lower.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return f"switch {parts[1]}"
        if lower == "pass":
            return "pass"
        return None

    async def schedule_invite_timeout(self, bot: Bot) -> None:
        # Cancel if exists
        if self._invite_task:
            self._invite_task.cancel()
            self._invite_task = None

        seconds = max(30, int(self.format_config.invite_timeout))

        async def _job() -> None:
            try:
                await asyncio.sleep(seconds)
                if self.state != "pending":
                    return
                try:
                    await self._send_group(
                        bot,
                        f"对战已超时取消：{self.format_config.display_name}。"
                        f"未在{seconds}秒内完成准备。",
                    )
                except Exception:
                    logger.exception(
                        f"failed to send invite timeout to group: {self.session_id}"
                    )
                for side, player in self.players.items():
                    if (
                        player.is_ai
                        or self.interaction_channel_for_side(side) == "group"
                    ):
                        continue
                    try:
                        await self._send_player_message(
                            bot,
                            side,
                            f"对战已超时取消：未在{seconds}秒内完成准备。",
                        )
                    except Exception:
                        logger.exception(
                            f"failed to send invite timeout user_id={player.user_id}"
                        )
                self.state = "finished"
                self._closed = True
                await self._close_ai_agent()
                await self.manager.remove(self)
            except asyncio.CancelledError:
                return

        self._invite_task = asyncio.create_task(_job())

    async def _schedule_action_timeout(
        self, bot: Bot, side: str, request: dict
    ) -> None:
        # Cancel previous timer for this side
        prev = self._action_timers.pop(side, None)
        if prev:
            prev.cancel()
        prev_warn = self._action_pre_alerts.pop(side, None)
        if prev_warn:
            prev_warn.cancel()

        # Determine timeout seconds by request type
        if request.get("teamPreview"):
            seconds = int(self.format_config.preview_timeout)
        elif request.get("forceSwitch"):
            seconds = int(self.format_config.switch_timeout)
        elif request.get("active"):
            seconds = int(self.format_config.move_timeout)
        else:
            return
        seconds = max(10, seconds)

        # Bump request sequence for this side
        seq = self._request_seq.get(side, 0) + 1
        self._request_seq[side] = seq

        player = self.players.get(side)
        display_name = player.display_name if player else side

        # Pre-alert 10s before timeout (private message only)
        if seconds > 10 and player:
            warn_delay = seconds - 10

            async def _pre_warn(current_seq: int) -> None:
                try:
                    await asyncio.sleep(warn_delay)
                    # Abort if request has changed or resolved
                    if self._request_seq.get(side) != current_seq:
                        return
                    if self.state != "active":
                        return
                    if side in self._awaiting_resolution:
                        return
                    latest = self.current_requests.get(side)
                    if not latest or latest.get("wait"):
                        return
                    try:
                        await self._send_player_message(
                            bot,
                            side,
                            "即将超时（剩余10秒），请尽快完成操作。",
                        )
                    except Exception:
                        logger.exception(
                            "failed to send action timeout warning "
                            f"user_id={player.user_id}"
                        )
                except asyncio.CancelledError:
                    return

            warn_task = asyncio.create_task(_pre_warn(seq))
            self._action_pre_alerts[side] = warn_task

        async def _on_timeout(current_seq: int) -> None:
            try:
                await asyncio.sleep(seconds)
                # Abort if request has changed or resolved
                if self._request_seq.get(side) != current_seq:
                    return
                if self.state != "active":
                    return
                if side in self._awaiting_resolution:
                    return
                latest = self.current_requests.get(side)
                if not latest or latest.get("wait"):
                    return
                choice = self._default_choice_for_request(latest)
                if not choice:
                    # As a last resort, forfeit on repeated inactivity
                    timeout_count = self._timeout_counts.get(side, 0) + 1
                    self._timeout_counts[side] = timeout_count
                    if timeout_count >= 2:
                        try:
                            await self._send_group(
                                bot, f"{display_name} 多次超时，已判负。"
                            )
                        except Exception:
                            logger.exception(
                                "failed to send repeated timeout notice "
                                f"to group: {self.session_id}"
                            )
                        process = self.process
                        if process is not None:
                            try:
                                await process.send_choice(side, "forfeit")
                            except ShowdownBridgeError:
                                pass
                        return
                    return
                if player and self.interaction_channel_for_side(side) == "private":
                    try:
                        await self._send_player_message(
                            bot,
                            side,
                            f"操作超时（{seconds}秒），已自动代为执行：{choice}",
                        )
                    except Exception:
                        logger.exception(
                            f"failed to send action timeout user_id={player.user_id}"
                        )
                try:
                    await self._send_group(
                        bot, f"{display_name} 操作超时，自动执行：{choice}"
                    )
                except Exception:
                    logger.exception(
                        f"failed to send action timeout to group: {self.session_id}"
                    )
                process = self.process
                if process is None:
                    return
                self._awaiting_resolution.add(side)
                try:
                    await process.send_choice(side, choice)
                except ShowdownBridgeError:
                    self._awaiting_resolution.discard(side)
                    return
            except asyncio.CancelledError:
                return

        task = asyncio.create_task(_on_timeout(seq))
        self._action_timers[side] = task

    def _default_choice_for_request(self, request: dict) -> Optional[str]:
        # Team preview: accept default order
        if request.get("teamPreview"):
            # If a selection size is provided, auto-pick the first N.
            selection_size = self._team_preview_selection_size(request)
            if selection_size:
                indexes = list(range(1, selection_size + 1))
                pokemon = (request.get("side") or {}).get("pokemon") or []
                if len(pokemon) >= 10:
                    encoded = ",".join(str(index) for index in indexes)
                else:
                    encoded = "".join(str(index) for index in indexes)
                return f"team {encoded}"
            return "team"
        if request.get("forceSwitch"):
            side_info = request.get("side", {}) or {}
            pokemon = side_info.get("pokemon", []) or []
            available: list[int] = []
            for idx, mon in enumerate(pokemon, start=1):
                if mon.get("active"):
                    continue
                condition = (mon.get("condition") or "").lower()
                if "fnt" in condition or condition.startswith("0/"):
                    continue
                available.append(idx)
            forced = request.get("forceSwitch") or []
            if isinstance(forced, list) and len(forced) > 1:
                actions: list[str] = []
                for must_switch in forced:
                    if must_switch and available:
                        actions.append(f"switch {available.pop(0)}")
                    else:
                        actions.append("pass")
                return ", ".join(actions)
            if available:
                return f"switch {available[0]}"
            return "pass"
        # Active decision: choose the first non-disabled move with PP
        if request.get("active"):
            active = request.get("active") or []
            if not active:
                return None
            # Doubles: build combined default actions if multiple actives
            if len(active) > 1:
                return ", ".join(
                    self._default_action_for_actor(actor, doubles=True)
                    for actor in active
                )
            return self._default_action_for_actor(active[0], doubles=False)
        return None

    async def _close_ai_agent(self) -> None:
        ai_agent = self._ai_agent
        self._ai_agent = None
        if ai_agent is None:
            return
        try:
            await ai_agent.close()
        except Exception:
            logger.exception(
                f"failed to close showdown AI agent battle_id={self.session_id}"
            )

    async def close(self) -> None:
        if self._closed:
            await self._close_ai_agent()
            await self.manager.remove(self)
            return
        self._closed = True
        self.state = "finished"
        current = asyncio.current_task()

        tasks = [
            task
            for task in (
                self._invite_task,
                *self._action_timers.values(),
                *self._action_pre_alerts.values(),
                *self._ai_action_tasks.values(),
            )
            if task is not None and task is not current
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._invite_task = None
        self._action_timers.clear()
        self._action_pre_alerts.clear()
        self._ai_action_tasks.clear()
        await self._close_ai_agent()

        process = self.process
        self.process = None
        if process is not None:
            try:
                await process.terminate()
            except Exception:
                logger.exception(
                    f"failed to terminate showdown process: {self.session_id}"
                )

        event_task = self._event_task
        if event_task is not None and event_task is not current:
            if not event_task.done():
                event_task.cancel()
            await asyncio.gather(event_task, return_exceptions=True)
        self._event_task = None
        await self.manager.remove(self)
