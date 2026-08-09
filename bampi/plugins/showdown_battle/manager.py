from __future__ import annotations

import asyncio
import uuid

from .ai_opponent import BattleAIError, BattleAIOpponent
from .bridge import ShowdownRuntime, ShowdownRuntimeInfo
from .formats import BattleFormatConfig, FormatRegistry
from .move_data import MoveDataRepository
from .rendering import PokemonBattleRenderer
from .session import BattleSession, PlayerSlot
from .translations import TranslationService


class BattleSessionConflict(RuntimeError):
    pass


class BattleRuntimeNotReady(RuntimeError):
    pass


class BattleAIOpponentUnavailable(RuntimeError):
    pass


class BattleManager:
    def __init__(
        self,
        *,
        translator: TranslationService,
        formats: FormatRegistry,
        runtime: ShowdownRuntime,
        move_repository: MoveDataRepository,
        renderer: PokemonBattleRenderer,
        max_render_concurrency: int,
        ai_opponent: BattleAIOpponent | None = None,
        bot_name: str = "Ophelia",
        ai_public_history_events: int = 80,
    ) -> None:
        self.sessions: dict[str, BattleSession] = {}
        self.user_to_session: dict[str, str] = {}
        self.group_to_session: dict[int, str] = {}
        self._lock = asyncio.Lock()
        self._translator = translator
        self._formats = formats
        self._runtime = runtime
        self._move_repository = move_repository
        self._renderer = renderer
        self._render_semaphore = asyncio.Semaphore(max_render_concurrency)
        self._ai_opponent = ai_opponent
        self._bot_name = bot_name.strip() or "Ophelia"
        self._ai_public_history_events = ai_public_history_events
        self.runtime_info: ShowdownRuntimeInfo | None = None
        self.runtime_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.runtime_info is not None and self.runtime_error is None

    @property
    def ai_available(self) -> bool:
        return self._ai_opponent is not None

    @property
    def bot_name(self) -> str:
        return self._bot_name

    def mark_runtime_ready(self, info: ShowdownRuntimeInfo) -> None:
        self.runtime_info = info
        self.runtime_error = None

    def mark_runtime_unavailable(self, error: str) -> None:
        self.runtime_info = None
        self.runtime_error = error

    async def create_session(
        self,
        *,
        group_id: int,
        challenger: tuple[str, str],
        opponent: tuple[str, str],
        format_config: BattleFormatConfig | None = None,
    ) -> BattleSession:
        async with self._lock:
            return self._create_session_locked(
                group_id=group_id,
                challenger=challenger,
                opponent=opponent,
                format_config=format_config,
                opponent_is_ai=False,
            )

    async def create_ai_session(
        self,
        *,
        group_id: int,
        challenger: tuple[str, str],
        format_config: BattleFormatConfig | None = None,
    ) -> BattleSession:
        ai_opponent = self._ai_opponent
        if ai_opponent is None:
            raise BattleAIOpponentUnavailable(
                f"{self._bot_name} 当前无法参加 Showdown 对战。"
            )
        config = format_config or self._formats.get_default()
        async with self._lock:
            session = self._create_session_locked(
                group_id=group_id,
                challenger=challenger,
                opponent=("__bampi_showdown_ai__", self._bot_name),
                format_config=config,
                opponent_is_ai=True,
            )
        try:
            prepared = await ai_opponent.prepare_team(config)
            ai_player = session.players["p2"]
            ai_player.team_pack = prepared.packed
            ai_player.team_raw = prepared.raw
        except BattleAIError as exc:
            await session.close()
            raise BattleAIOpponentUnavailable(
                f"{self._bot_name} 的队伍准备失败：{exc}"
            ) from exc
        except Exception as exc:
            await session.close()
            raise BattleAIOpponentUnavailable(
                f"{self._bot_name} 的队伍准备失败：{exc}"
            ) from exc
        return session

    def _create_session_locked(
        self,
        *,
        group_id: int,
        challenger: tuple[str, str],
        opponent: tuple[str, str],
        format_config: BattleFormatConfig | None,
        opponent_is_ai: bool,
    ) -> BattleSession:
        if not self.ready:
            raise BattleRuntimeNotReady(
                self.runtime_error or "Pokémon Showdown 运行时尚未就绪。"
            )
        existing_group_id = self.group_to_session.get(group_id)
        if existing_group_id and existing_group_id in self.sessions:
            raise BattleSessionConflict("本群已有正在进行的对战。")
        human_participants = (challenger,) if opponent_is_ai else (challenger, opponent)
        for user_id, _ in human_participants:
            existing_session_id = self.user_to_session.get(user_id)
            if existing_session_id and existing_session_id in self.sessions:
                raise BattleSessionConflict(f"QQ {user_id} 已在另一场对战中。")

        session_id = uuid.uuid4().hex
        players = {
            "p1": PlayerSlot(
                side="p1",
                user_id=challenger[0],
                display_name=challenger[1],
            ),
            "p2": PlayerSlot(
                side="p2",
                user_id=opponent[0],
                display_name=opponent[1],
                is_ai=opponent_is_ai,
            ),
        }
        config = format_config or self._formats.get_default()
        session = BattleSession(
            session_id=session_id,
            group_id=group_id,
            players=players,
            manager=self,
            translator=self._translator,
            format_config=config,
            runtime=self._runtime,
            move_repository=self._move_repository,
            renderer=self._renderer,
            render_semaphore=self._render_semaphore,
            ai_opponent=self._ai_opponent if opponent_is_ai else None,
            ai_public_history_events=self._ai_public_history_events,
        )
        self.sessions[session_id] = session
        self.group_to_session[group_id] = session_id
        for player in players.values():
            if not player.is_ai:
                self.user_to_session[player.user_id] = session_id
        return session

    async def get_session_by_user(self, user_id: str) -> BattleSession | None:
        async with self._lock:
            session_id = self.user_to_session.get(user_id)
            return self.sessions.get(session_id) if session_id else None

    async def get_session_by_group(self, group_id: int) -> BattleSession | None:
        async with self._lock:
            session_id = self.group_to_session.get(group_id)
            return self.sessions.get(session_id) if session_id else None

    async def remove(self, session: BattleSession) -> None:
        async with self._lock:
            if self.sessions.get(session.session_id) is not session:
                return
            self.sessions.pop(session.session_id, None)
            if self.group_to_session.get(session.group_id) == session.session_id:
                self.group_to_session.pop(session.group_id, None)
            for player in session.players.values():
                if self.user_to_session.get(player.user_id) == session.session_id:
                    self.user_to_session.pop(player.user_id, None)

    async def close_all(self) -> None:
        async with self._lock:
            sessions = list(self.sessions.values())
        if sessions:
            await asyncio.gather(
                *(session.close() for session in sessions),
                return_exceptions=True,
            )
        async with self._lock:
            self.sessions.clear()
            self.user_to_session.clear()
            self.group_to_session.clear()
