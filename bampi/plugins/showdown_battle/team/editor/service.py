from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from ...bridge import PreparedShowdownTeam, ShowdownRuntime
from ...translations import TranslationService, to_id
from ..repository import TeamRecord
from .models import (
    EditablePokemonSet,
    SpeciesEditorOptions,
    STAT_IDS,
    TeamCatalog,
    TeamDraft,
    TeamFormatRules,
)


_ENTITY_SEPARATORS = re.compile(r"[,，、\n]+")
_STAT_PATTERN = re.compile(
    r"(?P<value>\d+)\s*(?P<stat>HP|Atk|Def|SpA|SpD|Spe|"
    r"生命|体力|攻击|物攻|防御|物防|特攻|特防|速度)",
    re.IGNORECASE,
)
_STAT_ALIASES = {
    "hp": "hp",
    "生命": "hp",
    "体力": "hp",
    "atk": "atk",
    "攻击": "atk",
    "物攻": "atk",
    "def": "def",
    "防御": "def",
    "物防": "def",
    "spa": "spa",
    "特攻": "spa",
    "spd": "spd",
    "特防": "spd",
    "spe": "spe",
    "速度": "spe",
}
_NONE_ALIASES = {
    "无",
    "没有",
    "none",
    "no",
    "clear",
    "清空",
    "默认",
    "重置",
    "reset",
    "default",
    "0",
}


class TeamEditorError(RuntimeError):
    pass


class TeamEditorService:
    """Format-aware operations for safe, in-memory team drafts."""

    def __init__(
        self,
        *,
        runtime: ShowdownRuntime,
        translator: TranslationService,
    ) -> None:
        self._runtime = runtime
        self._translator = translator
        self._catalogs: dict[str, TeamCatalog] = {}
        self._species_options: dict[tuple[str, str], SpeciesEditorOptions] = {}
        self._cache_lock = asyncio.Lock()

    @property
    def translator(self) -> TranslationService:
        return self._translator

    async def new_draft(self, user_id: str, format_id: str) -> TeamDraft:
        catalog = await self.catalog(format_id)
        return TeamDraft(
            user_id=user_id,
            format_id=format_id,
            rules=catalog.rules,
            sets=[],
            dirty=False,
        )

    async def draft_from_record(self, record: TeamRecord) -> TeamDraft:
        catalog, payloads = await asyncio.gather(
            self.catalog(record.format_id),
            self._runtime.import_team_json(record.format_id, record.raw),
        )
        sets = [EditablePokemonSet.from_payload(payload) for payload in payloads]
        return TeamDraft(
            user_id=record.user_id,
            format_id=record.format_id,
            rules=catalog.rules,
            sets=sets,
            team_name=record.name,
            original_updated_at=record.updated_at,
            dirty=False,
        )

    async def catalog(self, format_id: str) -> TeamCatalog:
        cached = self._catalogs.get(format_id)
        if cached is not None:
            return cached
        async with self._cache_lock:
            cached = self._catalogs.get(format_id)
            if cached is not None:
                return cached
            payload = await self._runtime.load_team_builder_catalog(format_id)
            rules_payload = payload.get("rules")
            if not isinstance(rules_payload, dict):
                raise TeamEditorError("Showdown 未返回当前规则的队伍限制。")
            rules = TeamFormatRules(
                format_id=format_id,
                min_team_size=int(rules_payload.get("minTeamSize") or 1),
                max_team_size=int(rules_payload.get("maxTeamSize") or 6),
                picked_team_size=(
                    int(rules_payload["pickedTeamSize"])
                    if isinstance(rules_payload.get("pickedTeamSize"), (int, float))
                    else None
                ),
                max_move_count=int(rules_payload.get("maxMoveCount") or 4),
                min_level=int(rules_payload.get("minLevel") or 1),
                max_level=int(rules_payload.get("maxLevel") or 100),
                default_level=int(rules_payload.get("defaultLevel") or 100),
                stat_value_limit=int(rules_payload.get("statValueLimit") or 255),
                stat_total_limit=(
                    int(rules_payload["statTotalLimit"])
                    if isinstance(rules_payload.get("statTotalLimit"), (int, float))
                    else None
                ),
                uses_stat_points=bool(rules_payload.get("usesStatPoints")),
                supports_tera=bool(rules_payload.get("supportsTera", True)),
            )
            catalog = TeamCatalog(
                rules=rules,
                species=self._string_tuple(payload.get("species")),
                items=self._string_tuple(payload.get("items")),
                natures=self._string_tuple(payload.get("natures")),
                types=self._string_tuple(payload.get("types")),
            )
            if not catalog.species or not catalog.items:
                raise TeamEditorError("Showdown 返回的队伍编辑目录为空。")
            self._catalogs[format_id] = catalog
            return catalog

    async def species_options(
        self, format_id: str, species_input: str
    ) -> SpeciesEditorOptions:
        species = await self.resolve_species(format_id, species_input)
        key = (format_id, to_id(species))
        cached = self._species_options.get(key)
        if cached is not None:
            return cached
        async with self._cache_lock:
            cached = self._species_options.get(key)
            if cached is not None:
                return cached
            payload = await self._runtime.load_species_editor_options(
                format_id, species
            )
            canonical = str(payload.get("species") or "").strip()
            if not canonical:
                raise TeamEditorError("Showdown 未返回宝可梦资料。")
            options = SpeciesEditorOptions(
                species=canonical,
                abilities=self._string_tuple(payload.get("abilities")),
                moves=self._string_tuple(payload.get("moves")),
            )
            self._species_options[key] = options
            return options

    async def create_manual_set(
        self, format_id: str, species_input: str
    ) -> EditablePokemonSet:
        catalog = await self.catalog(format_id)
        options = await self.species_options(format_id, species_input)
        return EditablePokemonSet(
            species=options.species,
            ability=options.abilities[0] if options.abilities else "",
            nature="Serious",
            level=catalog.rules.default_level,
        )

    async def import_single_set(self, format_id: str, text: str) -> EditablePokemonSet:
        payloads = await self._runtime.import_team_json(format_id, text)
        if len(payloads) != 1:
            raise TeamEditorError(
                f"这里需要粘贴单只宝可梦，但检测到 {len(payloads)} 只。"
            )
        pokemon = EditablePokemonSet.from_payload(payloads[0])
        await self.species_options(format_id, pokemon.species)
        return pokemon

    async def import_team_sets(
        self, format_id: str, text: str
    ) -> list[EditablePokemonSet]:
        payloads = await self._runtime.import_team_json(format_id, text)
        return [EditablePokemonSet.from_payload(payload) for payload in payloads]

    async def export_draft(self, draft: TeamDraft) -> str:
        if not draft.sets:
            raise TeamEditorError("队伍中还没有宝可梦。")
        return await self._runtime.export_team_json(
            draft.format_id,
            [pokemon.to_payload() for pokemon in draft.sets],
        )

    async def prepare_draft(self, draft: TeamDraft) -> PreparedShowdownTeam:
        if len(draft.sets) < draft.rules.min_team_size:
            raise TeamEditorError(
                f"当前规则至少需要 {draft.rules.min_team_size} 只宝可梦，"
                f"现在只有 {len(draft.sets)} 只。"
            )
        if len(draft.sets) > draft.rules.max_team_size:
            raise TeamEditorError(
                f"当前规则最多允许 {draft.rules.max_team_size} 只宝可梦。"
            )
        text = await self.export_draft(draft)
        return await self._runtime.prepare_team_for_use(draft.format_id, text)

    async def resolve_species(self, format_id: str, value: str) -> str:
        catalog = await self.catalog(format_id)
        resolved = self._translator.resolve_species_name(value) or value
        return self._resolve_catalog_value(
            resolved,
            catalog.species,
            label="宝可梦",
            display=self._translator.translate_species,
        )

    async def resolve_item(self, format_id: str, value: str) -> str:
        if value.strip().lower() in _NONE_ALIASES:
            return ""
        catalog = await self.catalog(format_id)
        resolved = self._translator.resolve_item_name(value) or value
        return self._resolve_catalog_value(
            resolved,
            catalog.items,
            label="道具",
            display=self._translator.translate_item,
        )

    async def resolve_ability(self, format_id: str, species: str, value: str) -> str:
        options = await self.species_options(format_id, species)
        resolved = self._translator.resolve_ability_name(value) or value
        return self._resolve_catalog_value(
            resolved,
            options.abilities,
            label=f"{self.display_species(species)} 的特性",
            display=self._translator.translate_ability,
        )

    async def resolve_moves(
        self, format_id: str, species: str, value: str
    ) -> list[str]:
        tokens = [
            token.strip()
            for token in _ENTITY_SEPARATORS.split(value.strip())
            if token.strip()
        ]
        if not tokens:
            raise TeamEditorError("请至少提供一个招式，多个招式用逗号分隔。")
        catalog = await self.catalog(format_id)
        if len(tokens) > catalog.rules.max_move_count:
            raise TeamEditorError(
                f"当前规则每只宝可梦最多使用 {catalog.rules.max_move_count} 个招式。"
            )
        options = await self.species_options(format_id, species)
        moves: list[str] = []
        for token in tokens:
            resolved = self._translator.resolve_move_name(token) or token
            move = self._resolve_catalog_value(
                resolved,
                options.moves,
                label=f"{self.display_species(species)} 可学习的招式",
                display=self._translator.translate_move,
            )
            if to_id(move) in {to_id(existing) for existing in moves}:
                raise TeamEditorError(f"招式不能重复：{self.display_move(move)}。")
            moves.append(move)
        return moves

    async def resolve_nature(self, format_id: str, value: str) -> str:
        catalog = await self.catalog(format_id)
        resolved = self._translator.resolve_misc_name(value) or value
        return self._resolve_catalog_value(
            resolved,
            catalog.natures,
            label="性格",
            display=self._translator.translate,
        )

    async def resolve_type(
        self, format_id: str, value: str, *, allow_empty: bool = False
    ) -> str:
        if allow_empty and value.strip().lower() in _NONE_ALIASES:
            return ""
        catalog = await self.catalog(format_id)
        resolved = self._translator.resolve_type_name(value) or value
        return self._resolve_catalog_value(
            resolved,
            catalog.types,
            label="属性",
            display=self._translator.translate_type,
        )

    @staticmethod
    def parse_stats(
        value: str,
        *,
        default: int,
        maximum: int,
        enforce_total: int | None,
        label: str,
    ) -> dict[str, int]:
        normalized = value.strip()
        if normalized.lower() in {"默认", "重置", "reset", "default", "清空", "0"}:
            return {stat: default for stat in STAT_IDS}
        matches = list(_STAT_PATTERN.finditer(normalized))
        if not matches:
            raise TeamEditorError(f"无法识别{label}。示例：252 HP / 252 Atk / 4 SpD。")
        remainder = _STAT_PATTERN.sub("", normalized)
        if remainder.strip(" /,，、;；"):
            raise TeamEditorError(f"{label}中存在无法识别的内容：{remainder.strip()}。")
        result = {stat: default for stat in STAT_IDS}
        seen: set[str] = set()
        for match in matches:
            stat = _STAT_ALIASES[match.group("stat").lower()]
            number = int(match.group("value"))
            if stat in seen:
                raise TeamEditorError(f"{label}中的同一能力不能重复填写。")
            if number < 0 or number > maximum:
                raise TeamEditorError(f"{label}单项必须在 0 至 {maximum} 之间。")
            result[stat] = number
            seen.add(stat)
        if enforce_total is not None and sum(result.values()) > enforce_total:
            raise TeamEditorError(f"{label}总和不能超过 {enforce_total}。")
        return result

    @staticmethod
    def parse_integer(value: str, *, minimum: int, maximum: int, label: str) -> int:
        normalized = value.strip()
        if not normalized.isdigit():
            raise TeamEditorError(f"{label}必须是数字。")
        number = int(normalized)
        if number < minimum or number > maximum:
            raise TeamEditorError(f"{label}必须在 {minimum} 至 {maximum} 之间。")
        return number

    @staticmethod
    def parse_gender(value: str) -> str:
        normalized = value.strip().lower()
        aliases = {
            "m": "M",
            "男": "M",
            "雄": "M",
            "1": "M",
            "f": "F",
            "女": "F",
            "雌": "F",
            "2": "F",
            "无": "",
            "不指定": "",
            "none": "",
            "0": "",
            "3": "",
        }
        if normalized not in aliases:
            raise TeamEditorError("性别请输入 M/雄、F/雌或“无”。")
        return aliases[normalized]

    @staticmethod
    def parse_boolean(value: str, *, label: str) -> bool:
        normalized = value.strip().lower()
        if normalized in {"1", "是", "yes", "y", "true", "开启"}:
            return True
        if normalized in {"2", "0", "否", "no", "n", "false", "关闭"}:
            return False
        raise TeamEditorError(f"{label}请输入 1（是）或 2（否）。")

    def display_species(self, value: str) -> str:
        translated = self._translator.translate_species(value)
        return translated if translated == value else f"{translated}（{value}）"

    def display_item(self, value: str) -> str:
        if not value:
            return "无道具"
        return self._translator.translate_item(value)

    def display_ability(self, value: str) -> str:
        return self._translator.translate_ability(value) if value else "未设置"

    def display_move(self, value: str) -> str:
        return self._translator.translate_move(value)

    def display_type(self, value: str) -> str:
        return self._translator.translate_type(value) if value else "默认"

    def display_nature(self, value: str) -> str:
        return self._translator.translate(value) if value else "未设置"

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(str(item).strip() for item in value if str(item).strip())

    @staticmethod
    def _resolve_catalog_value(
        value: str,
        choices: tuple[str, ...],
        *,
        label: str,
        display: Callable[[str], str],
    ) -> str:
        target = to_id(value)
        by_id = {to_id(choice): choice for choice in choices}
        matched = by_id.get(target)
        if matched:
            return matched
        suggestions = [
            choice
            for choice in choices
            if target and (target in to_id(choice) or to_id(choice) in target)
        ][:5]
        suffix = (
            "\n可能想输入：" + "、".join(display(choice) for choice in suggestions)
            if suggestions
            else ""
        )
        raise TeamEditorError(f"未找到{label}：{value}。{suffix}".rstrip())


__all__ = ["TeamEditorError", "TeamEditorService"]
