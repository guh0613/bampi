from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


BattleGameType = Literal["singles", "doubles"]


@dataclass(frozen=True, slots=True)
class BattleFormatConfig:
    format_id: str
    display_name: str
    challenge_command: str
    challenge_aliases: tuple[str, ...]
    description: str
    usage_hint: str
    requires_team: bool = True
    game_type: BattleGameType = "singles"
    picked_team_size: int | None = None
    sample_team_source: str | None = None
    recommended_set_source: str | None = None
    generated_team_source: str | None = None
    invite_timeout: int = 300
    preview_timeout: int = 90
    move_timeout: int = 120
    switch_timeout: int = 90

    @property
    def is_doubles(self) -> bool:
        return self.game_type == "doubles"


class FormatRegistry:
    def __init__(self, formats: Iterable[BattleFormatConfig] = ()) -> None:
        self._by_id: dict[str, BattleFormatConfig] = {}
        self._challenge_triggers: dict[str, BattleFormatConfig] = {}
        self._default_id: str | None = None
        for config in formats:
            self.register(config)

    def register(
        self, config: BattleFormatConfig, *, set_default: bool = False
    ) -> None:
        format_id = config.format_id.strip().lower()
        if not format_id:
            raise ValueError("format_id must not be empty")
        if format_id in self._by_id:
            raise ValueError(f"duplicate format id: {format_id}")

        triggers = (config.challenge_command, *config.challenge_aliases)
        normalized_triggers: list[str] = []
        for trigger in triggers:
            normalized = trigger.strip().lower()
            if not normalized:
                raise ValueError(f"empty challenge trigger for {format_id}")
            owner = self._challenge_triggers.get(normalized)
            if owner is not None:
                raise ValueError(
                    f"duplicate challenge trigger {trigger!r}: "
                    f"{owner.format_id} and {format_id}"
                )
            normalized_triggers.append(normalized)

        self._by_id[format_id] = config
        for trigger in normalized_triggers:
            self._challenge_triggers[trigger] = config
        if set_default or self._default_id is None:
            self._default_id = format_id

    def get_default(self) -> BattleFormatConfig:
        if self._default_id is None:
            raise RuntimeError("未注册默认对战规则。")
        return self._by_id[self._default_id]

    def get(self, format_id: str) -> BattleFormatConfig | None:
        return self._by_id.get(format_id.strip().lower())

    def resolve_challenge_trigger(self, trigger: str) -> BattleFormatConfig | None:
        return self._challenge_triggers.get(trigger.strip().lower())

    def all(self) -> list[BattleFormatConfig]:
        return list(self._by_id.values())

    def resolve_format_token(self, token: str) -> BattleFormatConfig | None:
        normalized = token.strip().lower()
        if not normalized:
            return None
        direct = self._by_id.get(normalized)
        if direct:
            return direct
        for config in self._by_id.values():
            candidates = {
                config.display_name.lower(),
                config.challenge_command.lower(),
                *(alias.lower() for alias in config.challenge_aliases),
            }
            if normalized in candidates:
                return config
        return None


GEN9_OU = BattleFormatConfig(
    format_id="gen9ou",
    display_name="Gen9 单打 OU",
    challenge_command="g9ou",
    challenge_aliases=("g9挑战", "ps挑战", "ps对战", "g9对战"),
    description="第九世代 Smogon OU 单打对战。",
    usage_hint="群内发送“g9ou @对手”，双方私聊机器人导入队伍。",
    sample_team_source="gen9ou",
    recommended_set_source="gen9ou",
    generated_team_source="gen9ou",
    invite_timeout=600,
    preview_timeout=120,
)

GEN9_RANDOM = BattleFormatConfig(
    format_id="gen9randombattle",
    display_name="Gen9 随机单打",
    challenge_command="g9随机",
    challenge_aliases=("随机对战", "g9随机对战", "g9随机挑战"),
    description="第九世代随机单打，无需自备队伍。",
    usage_hint="群内发送“g9随机 @对手”，双方私聊发送“对战准备”。",
    requires_team=False,
    move_timeout=99,
    switch_timeout=60,
)

GEN9_RANDOM_DOUBLES = BattleFormatConfig(
    format_id="gen9randomdoublesbattle",
    display_name="Gen9 随机双打",
    challenge_command="g9双打随机",
    challenge_aliases=("双打随机", "随机双打", "g9随机双打"),
    description="第九世代随机双打，无需自备队伍。",
    usage_hint="群内发送“g9双打随机 @对手”，双方私聊发送“对战准备”。",
    requires_team=False,
    game_type="doubles",
    move_timeout=99,
    switch_timeout=60,
)

GEN9_DOUBLES_OU = BattleFormatConfig(
    format_id="gen9doublesou",
    display_name="Gen9 双打 OU",
    challenge_command="g9双打",
    challenge_aliases=("双打对战", "g9dou", "doubles"),
    description="第九世代 Smogon Doubles OU 双打对战。",
    usage_hint="群内发送“g9双打 @对手”，双方私聊机器人导入队伍。",
    game_type="doubles",
    sample_team_source="gen9doublesou",
    recommended_set_source="gen9doublesou",
    generated_team_source="gen9doublesou",
    invite_timeout=600,
    preview_timeout=120,
)

CHAMPIONS_VGC_2026_REG_M_B = BattleFormatConfig(
    format_id="gen9championsvgc2026regmb",
    display_name="Pokémon Champions VGC 2026 Reg M-B",
    challenge_command="冠军对战",
    challenge_aliases=("champions", "vgc2026", "vgcmb", "冠军赛"),
    description=(
        "Pokémon Champions 2026 Regulation M-B 双打：六选四，"
        "使用 Mega 进化，不使用太晶化。"
    ),
    usage_hint="群内发送“冠军对战 @对手”，双方私聊机器人导入队伍。",
    game_type="doubles",
    picked_team_size=4,
    generated_team_source="gen9championsvgc2026regmb",
    invite_timeout=600,
    preview_timeout=90,
    move_timeout=45,
    switch_timeout=45,
)


def build_default_registry() -> FormatRegistry:
    registry = FormatRegistry()
    registry.register(GEN9_OU, set_default=True)
    registry.register(GEN9_RANDOM)
    registry.register(GEN9_RANDOM_DOUBLES)
    registry.register(GEN9_DOUBLES_OU)
    registry.register(CHAMPIONS_VGC_2026_REG_M_B)
    return registry
