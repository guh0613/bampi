from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message, MessageSegment

from bampy.ai import AssistantMessage, TextContent, UserMessage

from bampi.plugins.showdown_battle.ai_opponent import (
    AIBattleDecision,
    AIBattleDecisionContext,
    AIModelSettings,
    AIPreparedTeam,
    BattleAIOpponent,
)
from bampi.plugins.showdown_battle.battle.action_guide import build_ai_action_guide
from bampi.plugins.showdown_battle.battle.event_formatter import BattleEventFormatter
from bampi.plugins.showdown_battle.battle.prompt_builder import PromptBuilder
from bampi.plugins.showdown_battle.battle.state import BattleState, PlayerSlot
from bampi.plugins.showdown_battle.bridge import (
    ShowdownBridgeError,
    ShowdownRuntime,
    ShowdownRuntimeInfo,
    ShowdownTeamValidationError,
)
from bampi.plugins.showdown_battle.commands import (
    _is_ai_challenge_request,
    _pokemon_battle_action_help,
    _pokemon_challenge_help,
    _pokemon_help_overview,
    _team_command_channel_error,
)
from bampi.plugins.showdown_battle.config import PROJECT_ROOT, ShowdownBattleConfig
from bampi.plugins.showdown_battle.formats import (
    CHAMPIONS_VGC_2026_REG_M_B,
    GEN9_OU,
    build_default_registry,
)
from bampi.plugins.showdown_battle.i18n.generator import (
    load_legacy_mapping,
    species_candidates,
)
from bampi.plugins.showdown_battle.manager import (
    BattleManager,
    BattleSessionConflict,
)
from bampi.plugins.showdown_battle.move_data import MoveDataRepository
from bampi.plugins.showdown_battle.team.editor.flow import TeamEditorFlow
from bampi.plugins.showdown_battle.team.editor.models import EditablePokemonSet
from bampi.plugins.showdown_battle.team.editor.service import (
    TeamEditorError,
    TeamEditorService,
)
from bampi.plugins.showdown_battle.team.guide import TeamGuideManager
from bampi.plugins.showdown_battle.team.sources import (
    BuiltTeam,
    PreparedSample,
    RecommendedSetList,
    RecommendedSetOption,
    SampleTeam,
    TeamSourceError,
    TeamSourceService,
)
from bampi.plugins.showdown_battle.team.repository import (
    TeamRepository,
    TeamRepositoryConflict,
    TeamRepositoryError,
)
from bampi.plugins.showdown_battle.translations import TranslationService


class _StubBattleRenderer:
    """In-memory stand-in for PokemonBattleRenderer (no browser/network)."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.turn_log_items: list[dict] = []

    async def render_turn_log(self, **kwargs) -> str:
        self.calls.append("turn_log")
        items = dict(kwargs.get("items") or {})
        self.turn_log_items.append(items)
        return f"base64://stub-turn-log-{len(self.turn_log_items)}"

    async def render_team_preview(self, **kwargs) -> str:
        self.calls.append("team_preview")
        return "base64://stub-team-preview"

    async def render_status_panel(self, **kwargs) -> str:
        self.calls.append("status_panel")
        return "base64://stub-status-panel"

    async def shutdown(self) -> None:
        pass


CATALOG_PATH = PROJECT_ROOT / "bampi/plugins/showdown_battle/assets/i18n/zh_hans.json"
SHOWDOWN_PACKAGE = PROJECT_ROOT / "node_modules/pokemon-showdown"
VALID_GEN9_OU_TEAM = """Pikachu @ Light Ball
Ability: Static
Tera Type: Electric
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch
- Grass Knot
- Encore
"""


def _group_message_event(
    *,
    user_id: int = 100,
    group_id: int = 9,
    text: str = "推荐配招 gen9ou 快龙",
) -> GroupMessageEvent:
    message = Message(text)
    return GroupMessageEvent(
        time=0,
        self_id=1,
        post_type="message",
        sub_type="normal",
        user_id=user_id,
        message_type="group",
        message_id=1,
        message=message,
        original_message=message,
        raw_message=text,
        font=0,
        sender={
            "user_id": user_id,
            "nickname": "test",
            "sex": "unknown",
            "age": 0,
            "card": "",
            "area": "",
            "level": "",
            "role": "member",
            "title": "",
        },
        to_me=False,
        group_id=group_id,
        anonymous=None,
    )


class _GuideSourceStub:
    async def list_compatible_samples(
        self,
        *,
        format_id: str,
        source_id: str,
    ) -> list[PreparedSample]:
        assert format_id == source_id == "gen9ou"
        return [
            PreparedSample(
                sample=SampleTeam(name="向导样例", author="Tester", data=()),
                team_text=VALID_GEN9_OU_TEAM.strip(),
            )
        ]

    async def list_recommended_sets(
        self,
        *,
        format_id: str,
        source_id: str,
        species_query: str,
    ) -> RecommendedSetList:
        assert format_id == source_id == "gen9ou"
        assert species_query == "快龙"
        return RecommendedSetList(
            species="Dragonite",
            options=(
                RecommendedSetOption(
                    name="Dragon Dance",
                    item="Heavy-Duty Boots",
                    ability="Multiscale",
                    nature="Adamant",
                    tera_type="Normal",
                    moves=("Dragon Dance", "Extreme Speed"),
                    evs=(("atk", 252), ("spe", 252)),
                ),
            ),
        )

    async def build_recommended_team(
        self,
        *,
        format_id: str,
        source_id: str,
        species_input: str,
    ) -> BuiltTeam:
        assert format_id == source_id == "gen9ou"
        assert species_input == "Dragonite=1"
        return BuiltTeam(
            team_text="""Dragonite @ Heavy-Duty Boots
Ability: Multiscale
Tera Type: Normal
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Dance
- Extreme Speed
- Earthquake
- Roost""",
            selections=(("Dragonite", "Dragon Dance"),),
        )


CHAMPIONS_OPEN_TEAM_SHEET = """Staraptor @ Staraptite
Ability: Intimidate
- Protect
- Roost
- Close Combat
- Brave Bird

Garchomp @ Life Orb
Ability: Rough Skin
- Protect
- Rock Slide
- Earthquake
- Dragon Claw

Whimsicott @ Occa Berry
Ability: Prankster
- Moonblast
- Tailwind
- Charm
- Light Screen

Delphox @ Delphoxite
Ability: Magician
- Protect
- Psychic
- Heat Wave
- Substitute

Glimmora @ Focus Sash
Ability: Toxic Debris
- Spiky Shield
- Earth Power
- Power Gem
- Sludge Bomb

Kingambit @ Chople Berry
Ability: Defiant
- Kowtow Cleave
- Iron Head
- Sucker Punch
- Low Kick
"""

VALID_CHAMPIONS_TEAM = """Raichu @ Raichunite X
Ability: Static
EVs: 4 HP
Serious Nature
- Thunderbolt

Dragonite @ Dragoninite
Ability: Inner Focus
EVs: 4 HP
Serious Nature
- Dragon Claw

Starmie @ Starminite
Ability: Natural Cure
EVs: 4 HP
Serious Nature
- Psychic

Pikachu @ Light Ball
Ability: Static
EVs: 4 HP
Serious Nature
- Thunderbolt

Charizard @ Leftovers
Ability: Blaze
EVs: 4 HP
Serious Nature
- Flamethrower

Garchomp @ Sitrus Berry
Ability: Rough Skin
EVs: 4 HP
Serious Nature
- Earthquake
"""

PREMEGA_CHAMPIONS_TEAM = VALID_CHAMPIONS_TEAM.replace(
    "Raichu @ Raichunite X\nAbility: Static",
    "Raichu-Mega-X @ Raichunite X\nAbility: Electric Surge",
    1,
)


@pytest.fixture(scope="module")
def translator() -> TranslationService:
    return TranslationService.from_file(CATALOG_PATH)


@pytest.fixture(scope="module")
def runtime() -> ShowdownRuntime:
    if not SHOWDOWN_PACKAGE.is_dir():
        pytest.skip("pokemon-showdown npm package is not installed")
    return ShowdownRuntime(node_bin="node", package_dir=SHOWDOWN_PACKAGE)


def test_config_resolves_paths_and_group_whitelist() -> None:
    config = ShowdownBattleConfig(showdown_battle_group_whitelist="3,1,3")
    assert config.showdown_battle_group_whitelist == [1, 3]
    assert config.group_is_allowed(1)
    assert not config.group_is_allowed(2)
    assert config.package_dir == SHOWDOWN_PACKAGE
    json_config = ShowdownBattleConfig(showdown_battle_group_whitelist='["3", 1]')
    assert json_config.showdown_battle_group_whitelist == [1, 3]
    ai_config = ShowdownBattleConfig(
        showdown_battle_ai_model_api="chat-completions",
        showdown_battle_ai_thinking_level=" HIGH ",
    )
    assert ai_config.showdown_battle_ai_model_api == "openai-completions"
    assert ai_config.showdown_battle_ai_thinking_level == "high"


def test_current_format_registry_is_unique() -> None:
    registry = build_default_registry()
    ids = [config.format_id for config in registry.all()]
    assert len(ids) == len(set(ids)) == 5
    assert "gen9vgc2025regj" not in ids
    assert CHAMPIONS_VGC_2026_REG_M_B.format_id in ids
    assert registry.resolve_challenge_trigger("冠军对战") is CHAMPIONS_VGC_2026_REG_M_B
    assert registry.resolve_challenge_trigger("G9挑战") is GEN9_OU
    assert CHAMPIONS_VGC_2026_REG_M_B.picked_team_size == 4
    assert CHAMPIONS_VGC_2026_REG_M_B.preview_timeout == 90
    assert CHAMPIONS_VGC_2026_REG_M_B.move_timeout == 45
    assert CHAMPIONS_VGC_2026_REG_M_B.switch_timeout == 45
    assert "不使用太晶化" in CHAMPIONS_VGC_2026_REG_M_B.description


def test_namespaced_translation_avoids_context_collisions(
    translator: TranslationService,
) -> None:
    assert translator.translate_move("Psychic") == "精神强念"
    assert translator.translate_type("Psychic") == "超能力"
    assert translator.translate_move("Metronome") == "挥指"
    assert translator.translate_item("Metronome") == "节拍器"
    assert translator.resolve_move_name("精神强念") == "psychic"
    assert translator.translate("Adamant") == "固执"


def test_translation_contains_current_champions_terms(
    translator: TranslationService,
) -> None:
    assert translator.translate_species("Raichu-Mega-X") == "超级雷丘Ｘ"
    assert translator.translate_item("Raichunite X") == "雷丘进化石Ｘ"
    assert translator.translate_ability("Dragonize") == "龙皮肤"
    assert translator.info.pokemon_showdown_version == "0.11.11"


def test_i18n_generator_reads_legacy_catalog_and_mega_candidates(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "translations.js"
    legacy.write_text(
        'var translations = {"Psychic": "精神{强念}",};', encoding="utf-8"
    )
    assert load_legacy_mapping(legacy) == {"Psychic": "精神{强念}"}
    assert species_candidates(
        {"name": "Raichu-Mega-X", "baseSpecies": "Raichu", "forme": "Mega-X"}
    ) == ["Mega Raichu X", "Raichu-Mega-X"]


def test_event_formatter_handles_mega_form_change(
    translator: TranslationService,
) -> None:
    state = BattleState()
    players = {
        "p1": PlayerSlot(side="p1", user_id="1", display_name="甲"),
        "p2": PlayerSlot(side="p2", user_id="2", display_name="乙"),
    }
    formatter = BattleEventFormatter(
        state=state,
        translator=translator,
        players=players,
    )
    state.register_pokemon("p1a: Raichu", "Raichu, L50", translator)

    assert formatter.format("|detailschange|p1a: Raichu|Raichu-Mega-X, L50") is None
    mega = formatter.format("|-mega|p1a: Raichu|Raichu|Raichunite X")
    moved = formatter.format("|move|p1a: Raichu|Thunderbolt|p2a: Pikachu")
    changed = formatter.format("|-formechange|p1a: Raichu|Raichu-Mega-X|100/100")

    assert mega and "雷丘进化石Ｘ" in mega
    assert mega and "超级雷丘Ｘ" in mega
    assert moved and "超级雷丘Ｘ" in moved
    assert changed and "超级雷丘Ｘ" in changed
    assert state.pokemon["p1a: Raichu"].name == "超级雷丘Ｘ"


@pytest.mark.asyncio
async def test_team_repository_round_trip_and_legacy_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "teams.json"
    path.write_text(
        json.dumps(
            {
                "100": {
                    "gen9ou": {
                        "旧队伍": {
                            "packed": "packed-old",
                            "raw": "raw-old",
                            "updated_at": 1,
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repository = TeamRepository(path)
    old = await repository.get_team("100", "gen9ou", "旧队伍")
    assert old and old.packed == "packed-old"

    await repository.set_team(
        "100",
        "gen9ou",
        "新队伍",
        packed="packed-new",
        raw="raw-new",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["users"]["100"]["gen9ou"]["新队伍"]["raw"] == "raw-new"


@pytest.mark.asyncio
async def test_team_repository_rejects_corrupted_storage(tmp_path: Path) -> None:
    path = tmp_path / "teams.json"
    path.write_text("{broken", encoding="utf-8")
    repository = TeamRepository(path)
    with pytest.raises(TeamRepositoryError):
        await repository.list_teams("100")
    assert path.read_text(encoding="utf-8") == "{broken"


@pytest.mark.asyncio
async def test_team_repository_rolls_back_memory_when_persist_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = TeamRepository(tmp_path / "teams.json")
    await repository.set_team("100", "gen9ou", "稳定队伍", packed="old", raw="old")

    def fail_write(_content: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(repository, "_atomic_write", fail_write)
    with pytest.raises(TeamRepositoryError, match="写入队伍仓库失败"):
        await repository.set_team("100", "gen9ou", "临时队伍", packed="new", raw="new")
    records = await repository.list_teams("100", "gen9ou")
    assert [record.name for record in records] == ["稳定队伍"]


@pytest.mark.asyncio
async def test_team_repository_supports_conflict_safe_create_and_update(
    tmp_path: Path,
) -> None:
    repository = TeamRepository(tmp_path / "teams.json")
    created = await repository.create_team(
        "100", "gen9ou", "编辑队", packed="old-pack", raw="old-raw"
    )
    with pytest.raises(TeamRepositoryConflict, match="已经存在"):
        await repository.create_team(
            "100", "gen9ou", "编辑队", packed="other", raw="other"
        )

    updated = await repository.update_team(
        "100",
        "gen9ou",
        "编辑队",
        packed="new-pack",
        raw="new-raw",
        expected_updated_at=created.updated_at,
    )
    assert updated.raw == "new-raw"
    with pytest.raises(TeamRepositoryConflict, match="编辑期间"):
        await repository.update_team(
            "100",
            "gen9ou",
            "编辑队",
            packed="stale-pack",
            raw="stale-raw",
            expected_updated_at=created.updated_at,
        )
    current = await repository.get_team("100", "gen9ou", "编辑队")
    assert current and current.raw == "new-raw"


def test_editable_pokemon_set_preserves_supported_showdown_fields() -> None:
    payload = {
        "name": "小龙",
        "species": "Dragonite",
        "item": "Heavy-Duty Boots",
        "ability": "Multiscale",
        "moves": ["Dragon Dance", "Extreme Speed"],
        "nature": "Adamant",
        "gender": "M",
        "evs": {"hp": 4, "atk": 252, "spe": 252},
        "ivs": {"atk": 0},
        "level": 50,
        "shiny": True,
        "happiness": 128,
        "pokeball": "Luxury Ball",
        "hpType": "Ice",
        "dynamaxLevel": 5,
        "gigantamax": True,
        "teraType": "Normal",
    }
    pokemon = EditablePokemonSet.from_payload(payload)
    exported = pokemon.to_payload()
    assert exported["name"] == "小龙"
    assert exported["gender"] == "M"
    assert exported["evs"] == {
        "hp": 4,
        "atk": 252,
        "def": 0,
        "spa": 0,
        "spd": 0,
        "spe": 252,
    }
    assert exported["ivs"] == {
        "hp": 31,
        "atk": 0,
        "def": 31,
        "spa": 31,
        "spd": 31,
        "spe": 31,
    }
    assert exported["shiny"] is True
    assert exported["happiness"] == 128
    assert exported["pokeball"] == "Luxury Ball"
    assert exported["hpType"] == "Ice"
    assert exported["dynamaxLevel"] == 5
    assert exported["gigantamax"] is True
    assert exported["teraType"] == "Normal"


@pytest.mark.asyncio
async def test_team_editor_service_uses_showdown_catalog_and_chinese_fields(
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    service = TeamEditorService(runtime=runtime, translator=translator)
    catalog = await service.catalog("gen9ou")
    assert catalog.rules.max_team_size == 6
    assert catalog.rules.max_move_count == 4
    assert await service.resolve_species("gen9ou", "快龙") == "Dragonite"
    assert await service.resolve_item("gen9ou", "厚底靴") == "Heavy-Duty Boots"
    assert await service.resolve_ability("gen9ou", "Dragonite", "多重鳞片") == (
        "Multiscale"
    )
    assert await service.resolve_moves(
        "gen9ou", "Dragonite", "龙之舞，神速，地震，羽栖"
    ) == ["Dragon Dance", "Extreme Speed", "Earthquake", "Roost"]
    assert await service.resolve_nature("gen9ou", "固执") == "Adamant"
    assert await service.resolve_type("gen9ou", "一般") == "Normal"

    champions = await service.catalog(CHAMPIONS_VGC_2026_REG_M_B.format_id)
    assert champions.rules.uses_stat_points
    assert not champions.rules.supports_tera
    assert champions.rules.stat_value_limit == 32
    assert champions.rules.stat_total_limit == 66
    assert service.parse_stats(
        "32 HP / 32 Atk / 2 Spe",
        default=0,
        maximum=champions.rules.stat_value_limit,
        enforce_total=champions.rules.stat_total_limit,
        label="Stat Points",
    ) == {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 2}
    with pytest.raises(TeamEditorError, match="总和不能超过 66"):
        service.parse_stats(
            "32 HP / 32 Atk / 3 Spe",
            default=0,
            maximum=32,
            enforce_total=66,
            label="Stat Points",
        )


@pytest.mark.asyncio
async def test_team_editor_creates_manual_team_and_saves_only_after_validation(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    repository = TeamRepository(tmp_path / "teams.json")
    sources = _GuideSourceStub()
    flow = TeamEditorFlow(
        formats=build_default_registry(),
        repository=repository,
        team_sources=sources,  # type: ignore[arg-type]
        service=TeamEditorService(runtime=runtime, translator=translator),
    )
    state = await flow.start_new("100", "gen9ou")
    assert "队伍为空" in state.prompt

    assert "添加宝可梦" in (await flow.handle(state, "1")).message
    assert "宝可梦名称" in (await flow.handle(state, "3")).message
    assert "已添加" in (await flow.handle(state, "快龙")).message

    assert "队伍编辑" in (await flow.handle(state, "返回")).message
    invalid_save = await flow.handle(state, "6")
    assert invalid_save.status == "active"
    assert "操作失败" in invalid_save.message
    assert await repository.list_teams("100") == []
    assert "编辑成员" in (await flow.handle(state, "1")).message

    assert "新道具" in (await flow.handle(state, "4")).message
    assert "已更新道具" in (await flow.handle(state, "厚底靴")).message
    assert "完整招式列表" in (await flow.handle(state, "6")).message
    assert (
        "已更新招式" in (await flow.handle(state, "龙之舞，神速，地震，羽栖")).message
    )
    assert "新性格" in (await flow.handle(state, "7")).message
    assert "已更新性格" in (await flow.handle(state, "固执")).message
    assert "完整 EV" in (await flow.handle(state, "8")).message
    assert "已更新EV" in (await flow.handle(state, "252攻击 / 4特防 / 252速度")).message
    assert "太晶属性" in (await flow.handle(state, "10")).message
    assert "已更新太晶属性" in (await flow.handle(state, "一般")).message

    assert await repository.list_teams("100") == []
    assert "队伍编辑" in (await flow.handle(state, "返回")).message
    assert "保存名称" in (await flow.handle(state, "6")).message
    saved_response = await flow.handle(state, "我的快龙队")
    assert saved_response.status == "saved"
    saved = await repository.get_team("100", "gen9ou", "我的快龙队")
    assert saved
    assert "Dragonite @ Heavy-Duty Boots" in saved.raw
    assert "Tera Type: Normal" in saved.raw
    assert "- Extreme Speed" in saved.raw


@pytest.mark.asyncio
async def test_team_editor_adds_recommended_set_and_supports_discard_confirmation(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    repository = TeamRepository(tmp_path / "teams.json")
    sources = _GuideSourceStub()
    flow = TeamEditorFlow(
        formats=build_default_registry(),
        repository=repository,
        team_sources=sources,  # type: ignore[arg-type]
        service=TeamEditorService(runtime=runtime, translator=translator),
    )
    state = await flow.start_new("200", "gen9ou")
    await flow.handle(state, "添加")
    await flow.handle(state, "1")
    options = await flow.handle(state, "快龙")
    assert "Dragon Dance" in options.message
    added = await flow.handle(state, "1")
    assert "已采用推荐配招" in added.message

    confirmation = await flow.handle(state, "0")
    assert "未保存的修改" in confirmation.message
    continued = await flow.handle(state, "2")
    assert continued.status == "active"
    assert "编辑成员" in continued.message
    discarded = await flow.handle(state, "0")
    assert "未保存的修改" in discarded.message
    discarded = await flow.handle(state, "1")
    assert discarded.status == "cancelled"
    assert await repository.list_teams("200") == []


@pytest.mark.asyncio
async def test_team_editor_existing_team_is_transactional_and_detects_stale_save(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    repository = TeamRepository(tmp_path / "teams.json")
    prepared = await runtime.prepare_team_for_use("gen9ou", VALID_GEN9_OU_TEAM)
    await repository.set_team(
        "300",
        "gen9ou",
        "原队伍",
        packed=prepared.packed,
        raw=prepared.team_text,
    )
    original = await repository.get_team("300", "gen9ou", "原队伍")
    assert original
    flow = TeamEditorFlow(
        formats=build_default_registry(),
        repository=repository,
        team_sources=_GuideSourceStub(),  # type: ignore[arg-type]
        service=TeamEditorService(runtime=runtime, translator=translator),
    )
    state = await flow.start_existing(original)
    await flow.handle(state, "1")
    await flow.handle(state, "4")
    await flow.handle(state, "吃剩的东西")
    unchanged = await repository.get_team("300", "gen9ou", "原队伍")
    assert unchanged and "Light Ball" in unchanged.raw

    await flow.handle(state, "返回")
    saved_response = await flow.handle(state, "6")
    assert saved_response.status == "saved"
    changed = await repository.get_team("300", "gen9ou", "原队伍")
    assert changed and "Leftovers" in changed.raw

    stale_state = await flow.start_existing(changed)
    await flow.handle(stale_state, "1")
    await flow.handle(stale_state, "4")
    await flow.handle(stale_state, "光之黏土")
    await repository.set_team(
        "300",
        "gen9ou",
        "原队伍",
        packed=changed.packed,
        raw=changed.raw,
    )
    await flow.handle(stale_state, "返回")
    conflict = await flow.handle(stale_state, "6")
    assert conflict.status == "active"
    assert "编辑期间已被其他操作更新" in conflict.message


@pytest.mark.asyncio
async def test_battle_stream_forwards_public_half_of_split_updates(
    runtime: ShowdownRuntime,
) -> None:
    process = runtime.create_battle_process(
        format_id="gen9randombattle",
        p1={"name": "A", "team": ""},
        p2={"name": "B", "team": ""},
    )
    stdout = asyncio.StreamReader()
    stdout.feed_data(
        b"update\n"
        b"|split|p1\n"
        b"|-damage|p1a: Secret|73/211\n"
        b"|-damage|p1a: Public|35/100\n\n"
    )
    stdout.feed_eof()
    process._process = SimpleNamespace(stdout=stdout)

    await process._read_stdout()
    events = []
    while not process.events.empty():
        events.append(process.events.get_nowait())
    updates = [event["line"] for event in events if event["type"] == "update"]
    assert updates == ["|-damage|p1a: Public|35/100"]


@pytest.mark.asyncio
async def test_runtime_has_all_enabled_formats(runtime: ShowdownRuntime) -> None:
    registry = build_default_registry()
    ids = [config.format_id for config in registry.all()]
    info = await runtime.inspect(ids)
    assert info.version == "0.11.11"
    assert set(info.formats) == set(ids)


@pytest.mark.asyncio
async def test_runtime_validates_and_packs_team(runtime: ShowdownRuntime) -> None:
    packed = await runtime.validate_and_pack_team("gen9ou", VALID_GEN9_OU_TEAM)
    assert packed.startswith("Pikachu|")
    with pytest.raises(ShowdownTeamValidationError):
        await runtime.validate_team("gen9ou", "MissingNo\n- Tackle")


@pytest.mark.asyncio
async def test_runtime_validates_single_set_without_doubles_team_size_error(
    runtime: ShowdownRuntime,
) -> None:
    with pytest.raises(ShowdownTeamValidationError, match="at least 2"):
        await runtime.validate_team("gen9doublesou", VALID_GEN9_OU_TEAM)
    await runtime.validate_set("gen9doublesou", VALID_GEN9_OU_TEAM)


@pytest.mark.asyncio
async def test_runtime_exports_json_team_and_fills_default_ability(
    runtime: ShowdownRuntime,
) -> None:
    team_text = await runtime.export_team_json(
        "gen9ou",
        [
            {
                "species": "Dragonite",
                "item": "Heavy-Duty Boots",
                "nature": "Adamant",
                "evs": {"atk": 252, "def": 4, "spe": 252},
                "moves": [
                    "Dragon Dance",
                    "Extreme Speed",
                    "Earthquake",
                    "Roost",
                ],
            }
        ],
    )
    assert "Ability: Inner Focus" in team_text
    await runtime.validate_team("gen9ou", team_text)


@pytest.mark.asyncio
async def test_runtime_validates_current_champions_team(
    runtime: ShowdownRuntime,
) -> None:
    packed = await runtime.validate_and_pack_team(
        CHAMPIONS_VGC_2026_REG_M_B.format_id,
        VALID_CHAMPIONS_TEAM,
    )
    assert "RaichuniteX" in packed
    assert "Dragoninite" in packed


@pytest.mark.asyncio
async def test_runtime_canonicalizes_premega_champions_sets(
    runtime: ShowdownRuntime,
) -> None:
    prepared = await runtime.prepare_team_for_use(
        CHAMPIONS_VGC_2026_REG_M_B.format_id,
        PREMEGA_CHAMPIONS_TEAM,
    )

    assert "Raichu-Mega-X" not in prepared.team_text
    assert "Raichu @ Raichunite X" in prepared.team_text
    assert "Ability: Static" in prepared.team_text
    assert prepared.packed.startswith("Raichu||RaichuniteX|Static|")
    assert prepared.warnings and "Mega 形态" in prepared.warnings[0]
    assert "指令末尾添加 mega" in prepared.warnings[0]
    await runtime.validate_team(
        CHAMPIONS_VGC_2026_REG_M_B.format_id,
        prepared.team_text,
    )


@pytest.mark.asyncio
async def test_runtime_completes_champions_open_team_sheet(
    runtime: ShowdownRuntime,
) -> None:
    with pytest.raises(ShowdownTeamValidationError, match="exactly 0 Stat Points"):
        await runtime.validate_team(
            CHAMPIONS_VGC_2026_REG_M_B.format_id,
            CHAMPIONS_OPEN_TEAM_SHEET,
        )

    prepared = await runtime.prepare_team_for_use(
        CHAMPIONS_VGC_2026_REG_M_B.format_id,
        CHAMPIONS_OPEN_TEAM_SHEET,
    )
    assert prepared.team_text.count("Hardy Nature") == 6
    assert prepared.warnings and "公开队伍表" in prepared.warnings[0]
    await runtime.validate_team(
        CHAMPIONS_VGC_2026_REG_M_B.format_id,
        prepared.team_text,
    )


@pytest.mark.asyncio
async def test_online_team_sources_import_samples_and_build_from_chinese_species(
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    dragon_dance = {
        "moves": ["Dragon Dance", "Extreme Speed", "Earthquake", "Roost"],
        "ability": "Multiscale",
        "item": "Heavy-Duty Boots",
        "nature": "Adamant",
        "evs": {"atk": 252, "def": 4, "spe": 252},
        "teratypes": ["Normal", "Ground"],
    }
    choice_band = {
        "moves": ["Outrage", "Extreme Speed", "Earthquake", "Fire Punch"],
        "ability": "Multiscale",
        "item": "Choice Band",
        "nature": "Adamant",
        "evs": {"atk": 252, "def": 4, "spe": 252},
        "teratypes": "Normal",
    }
    kingambit = {
        "moves": ["Swords Dance", "Sucker Punch", "Kowtow Cleave", "Iron Head"],
        "ability": "Supreme Overlord",
        "item": "Leftovers",
        "nature": "Adamant",
        "evs": {"hp": 252, "atk": 252, "spd": 4},
        "teratypes": "Dark",
    }
    sample_data = [
        {"species": "Dragonite", **dragon_dance},
        {"species": "Kingambit", **kingambit},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "pokepast.es" and request.url.path == "/abc123/raw":
            return httpx.Response(200, text=VALID_GEN9_OU_TEAM)
        if request.url.host == "crob.at" and request.url.path == "/api/team/team123":
            return httpx.Response(
                200,
                json={
                    "name": "双队伍",
                    "teams": [
                        {"paste": "first", "format": "gen9ou"},
                        {"paste": VALID_GEN9_OU_TEAM, "format": "gen9ou"},
                    ],
                },
            )
        if request.url.host == "crob.at" and request.url.path == (
            "/api/random-team/gen9ou"
        ):
            return httpx.Response(
                200,
                json={"teamText": VALID_GEN9_OU_TEAM, "statsDate": "2026-07"},
            )
        if request.url.host == "data.pkmn.cc" and request.url.path in {
            "/teams/gen9ou.json",
            "/sets/gen9ou.json",
        }:
            return httpx.Response(
                301,
                headers={
                    "location": (
                        "https://pkmn.github.io/smogon/data" + request.url.path
                    )
                },
            )
        if request.url.path == "/smogon/data/teams/gen9ou.json":
            return httpx.Response(
                200,
                json=[{"name": "入门样例", "author": "Tester", "data": sample_data}],
            )
        if request.url.path == "/smogon/data/sets/gen9ou.json":
            return httpx.Response(
                200,
                json={
                    "Dragonite": {
                        "Outdated Tera Blast": {
                            **dragon_dance,
                            "moves": [
                                "Tera Blast",
                                "Extreme Speed",
                                "Earthquake",
                                "Roost",
                            ],
                        },
                        "Dragon Dance": dragon_dance,
                        "Choice Band": choice_band,
                    },
                    "Kingambit": {"Swords Dance": kingambit},
                },
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = TeamSourceService(
        runtime=runtime,
        translator=translator,
        timeout_seconds=2,
        max_bytes=256 * 1024,
        cache_ttl_seconds=60,
        client=client,
    )
    try:
        imported = await service.resolve_import("https://pokepast.es/abc123")
        assert imported.team_text == VALID_GEN9_OU_TEAM.strip()
        assert imported.label == "PokePaste abc123"

        with pytest.raises(TeamSourceError, match="包含 2 支队伍"):
            await service.resolve_import("https://crob.at/team123")
        crobat = await service.resolve_import("https://crob.at/team123#2")
        assert crobat.team_text == VALID_GEN9_OU_TEAM.strip()
        assert crobat.source_format_id == "gen9ou"
        with pytest.raises(TeamSourceError, match="链接标注的规则"):
            service.ensure_format_compatible(
                crobat, CHAMPIONS_VGC_2026_REG_M_B.format_id
            )
        with pytest.raises(TeamSourceError, match="暂只支持"):
            await service.resolve_import("https://example.com/team")
        generated = await service.generate_team(format_id="gen9ou", source_id="gen9ou")
        assert generated.team_text == VALID_GEN9_OU_TEAM.strip()
        assert "2026-07" in (generated.label or "")

        compatible = await service.list_compatible_samples(
            format_id="gen9ou", source_id="gen9ou"
        )
        assert compatible[0].sample.name == "入门样例"
        await runtime.validate_team("gen9ou", compatible[0].team_text)

        recommendations = await service.list_recommended_sets(
            format_id="gen9ou", source_id="gen9ou", species_query="快龙"
        )
        assert recommendations.set_names == ("Dragon Dance", "Choice Band")
        assert recommendations.options[0].item == "Heavy-Duty Boots"
        assert recommendations.options[0].moves[0] == "Dragon Dance"
        built = await service.build_recommended_team(
            format_id="gen9ou",
            source_id="gen9ou",
            species_input="快龙=2，仆刀将军",
        )
        assert built.selections == (
            ("Dragonite", "Choice Band"),
            ("Kingambit", "Swords Dance"),
        )
        assert "Choice Band" in built.team_text
        await runtime.validate_team("gen9ou", built.team_text)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_team_guide_generates_saves_and_reuses_team_for_pending_battle(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/random-team/gen9ou":
            return httpx.Response(
                200,
                json={"teamText": VALID_GEN9_OU_TEAM, "statsDate": "2026-07"},
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sources = TeamSourceService(
        runtime=runtime,
        translator=translator,
        timeout_seconds=2,
        max_bytes=256 * 1024,
        cache_ttl_seconds=60,
        client=client,
    )
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    group_event = _group_message_event()
    config = ShowdownBattleConfig()
    assert (
        await _team_command_channel_error(
            config=config,
            manager=manager,
            event=group_event,
        )
        is None
    )
    repository = TeamRepository(tmp_path / "teams.json")
    guide = TeamGuideManager(
        manager=manager,
        formats=registry,
        repository=repository,
        runtime=runtime,
        team_sources=sources,
        translator=translator,
        editor_flow=TeamEditorFlow(
            formats=registry,
            repository=repository,
            team_sources=sources,
            service=TeamEditorService(runtime=runtime, translator=translator),
        ),
        idle_ttl_seconds=900,
    )
    try:
        assert "宝可梦队伍中心" in await guide.start("100", group_id=9)
        assert await guide.has_state("100", group_id=9)
        assert not await guide.has_state("100")
        assert "选择规则" in await guide.handle(object(), "100", "1", group_id=9)
        assert "宝可梦队伍中心" in await guide.handle(
            object(), "100", "返回", group_id=9
        )
        assert "选择规则" in await guide.handle(object(), "100", "1", group_id=9)
        assert "Gen9 单打 OU" in await guide.handle(object(), "100", "1", group_id=9)
        assert "保存名称" in await guide.handle(object(), "100", "1", group_id=9)
        assert "已保存队伍" in await guide.handle(
            object(), "100", "向导队伍", group_id=9
        )
        saved = await repository.get_team("100", "gen9ou", "向导队伍")
        assert saved and saved.raw == VALID_GEN9_OU_TEAM.strip()

        assert "向导队伍" in await guide.handle(object(), "100", "1", group_id=9)
        assert VALID_GEN9_OU_TEAM.strip() in await guide.handle(
            object(), "100", "1", group_id=9
        )
        assert "新名称" in await guide.handle(object(), "100", "2", group_id=9)
        assert "已重命名" in await guide.handle(
            object(), "100", "向导队伍改", group_id=9
        )
        assert "向导队伍改" in await guide.handle(object(), "100", "1", group_id=9)
        assert "副本名称" in await guide.handle(object(), "100", "4", group_id=9)
        assert "已复制" in await guide.handle(object(), "100", "向导副本", group_id=9)
        assert len(await repository.list_teams("100", "gen9ou")) == 2
        assert "向导副本" in await guide.handle(object(), "100", "1", group_id=9)
        assert "确定删除" in await guide.handle(object(), "100", "3", group_id=9)
        assert "已删除" in await guide.handle(object(), "100", "1", group_id=9)
        assert len(await repository.list_teams("100", "gen9ou")) == 1

        remaining = (await repository.list_teams("100", "gen9ou"))[0]
        assert "我的队伍" in await guide.start("100", group_id=9, entry="library")
        assert "更新" in await guide.handle(object(), "100", "1", group_id=9)
        assert "队伍编辑" in await guide.handle(object(), "100", "5", group_id=9)
        editor_exit = await guide.handle(object(), "100", "0", group_id=9)
        assert "未修改原队伍" in editor_exit
        assert "我的队伍" in editor_exit

        for index in range(7):
            await repository.set_team(
                "300",
                "gen9ou",
                f"分页队伍{index + 1}",
                packed=remaining.packed,
                raw=remaining.raw,
            )
        assert "1/2" in await guide.start("300", group_id=9, entry="library")
        assert "2/2" in await guide.handle(object(), "300", "下一页", group_id=9)
        assert "1/2" in await guide.handle(object(), "300", "上一页", group_id=9)
        assert "已退出" in await guide.handle(object(), "300", "0", group_id=9)

        assert "宝可梦队伍中心" in await guide.start("100", group_id=9)
        battle = await manager.create_session(
            group_id=1,
            challenger=("100", "甲"),
            opponent=("200", "乙"),
            format_config=GEN9_OU,
        )
        assert "私聊" in (
            await _team_command_channel_error(
                config=config,
                manager=manager,
                event=group_event,
            )
            or ""
        )
        assert "群内组队向导已结束" in await guide.handle(
            object(), "100", "1", group_id=9
        )
        assert not await guide.has_state("100", group_id=9)
        assert "请私聊" in await guide.start("100", group_id=9)
        assert "选择队伍来源" in await guide.start("100")
        assert "向导队伍改" in await guide.handle(object(), "100", "1")
        assert "等待对手" in await guide.handle(object(), "100", "1")
        assert battle.players["p1"].team_pack
        assert not await guide.has_state("100")
    finally:
        await guide.close()
        await manager.close_all()
        await client.aclose()


@pytest.mark.asyncio
async def test_team_guide_browses_recommendations_and_saves_samples(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    repository = TeamRepository(tmp_path / "teams.json")
    guide_sources = _GuideSourceStub()
    guide = TeamGuideManager(
        manager=manager,
        formats=registry,
        repository=repository,
        runtime=runtime,
        team_sources=guide_sources,  # type: ignore[arg-type]
        translator=translator,
        editor_flow=TeamEditorFlow(
            formats=registry,
            repository=repository,
            team_sources=guide_sources,  # type: ignore[arg-type]
            service=TeamEditorService(runtime=runtime, translator=translator),
        ),
        idle_ttl_seconds=900,
    )
    try:
        assert "队伍中心" in await guide.start("400", group_id=9)
        assert "查询推荐配招" in await guide.handle(object(), "400", "3", group_id=9)
        assert "请发送一个宝可梦" in await guide.handle(
            object(), "400", "1", group_id=9
        )
        recommendation = await guide.handle(object(), "400", "快龙", group_id=9)
        assert "快龙 推荐配招" in recommendation
        assert "厚底靴" in recommendation

        assert "队伍中心" in await guide.handle(object(), "400", "菜单", group_id=9)
        assert "浏览样例队伍" in await guide.handle(object(), "400", "4", group_id=9)
        assert "向导样例" in await guide.handle(object(), "400", "1", group_id=9)
        assert VALID_GEN9_OU_TEAM.strip() in await guide.handle(
            object(), "400", "1", group_id=9
        )
        assert "保存名称" in await guide.handle(object(), "400", "1", group_id=9)
        assert "已保存队伍" in await guide.handle(
            object(), "400", "样例收藏", group_id=9
        )
        assert await repository.get_team("400", "gen9ou", "样例收藏")
    finally:
        await guide.close()
        await manager.close_all()


@pytest.mark.asyncio
async def test_move_repository_loads_current_data(
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    repository = MoveDataRepository(runtime)
    await repository.warm_up()
    entry = repository.get("psychic")
    assert entry is not None
    assert entry.data["basePower"] == 90
    assert (
        translator.translate_move_description(
            "psychic", entry.text.get("shortDesc", "")
        )
        == "有10%的机率会让对手的特防降低1阶。"
    )


@pytest.mark.asyncio
async def test_battle_process_starts_and_terminates_cleanly(
    runtime: ShowdownRuntime,
) -> None:
    process = runtime.create_battle_process(
        format_id="gen9randombattle",
        p1={"name": "A", "team": ""},
        p2={"name": "B", "team": ""},
    )
    await process.start()
    requests: set[str] = set()
    try:
        for _ in range(300):
            event = await asyncio.wait_for(process.events.get(), timeout=5)
            if event.get("type") == "request" and event.get("side"):
                requests.add(event["side"])
                if requests == {"p1", "p2"}:
                    break
    finally:
        await process.terminate()
    assert requests == {"p1", "p2"}
    assert process.returncode == 0


@pytest.mark.asyncio
async def test_current_champions_battle_reaches_preview_and_mega_evolves(
    runtime: ShowdownRuntime,
) -> None:
    packed = await runtime.validate_and_pack_team(
        CHAMPIONS_VGC_2026_REG_M_B.format_id,
        VALID_CHAMPIONS_TEAM,
    )
    process = runtime.create_battle_process(
        format_id=CHAMPIONS_VGC_2026_REG_M_B.format_id,
        p1={"name": "A", "team": packed},
        p2={"name": "B", "team": packed},
    )
    await process.start()
    previews: set[str] = set()
    active_requests: set[str] = set()
    mega_sides: set[str] = set()
    teams_selected = False
    actions_sent = False
    try:
        for _ in range(1000):
            event = await asyncio.wait_for(process.events.get(), timeout=5)
            if event.get("type") == "error":
                pytest.fail(f"Showdown rejected a Champions choice: {event}")
            if event.get("type") == "request" and event.get("side"):
                side = event["side"]
                payload = event["payload"]
                if payload.get("teamPreview"):
                    previews.add(side)
                    if previews == {"p1", "p2"} and not teams_selected:
                        await process.send_choice("p1", "team 1234")
                        await process.send_choice("p2", "team 1234")
                        teams_selected = True
                elif payload.get("active"):
                    active_requests.add(side)
                    if active_requests == {"p1", "p2"} and not actions_sent:
                        choice = "move 1 1 mega, move 1 1"
                        await process.send_choice("p1", choice)
                        await process.send_choice("p2", choice)
                        actions_sent = True
            elif event.get("type") == "update" and event["line"].startswith("|-mega|"):
                mega_sides.add(event["line"].split("|", 3)[2][:2])
                if mega_sides == {"p1", "p2"}:
                    break
    finally:
        await process.terminate()
    assert previews == {"p1", "p2"}
    assert mega_sides == {"p1", "p2"}
    assert process.returncode == 0


@pytest.mark.asyncio
async def test_manager_enforces_group_and_user_uniqueness(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    move_repository = MoveDataRepository(runtime)
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=move_repository,
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    session = await manager.create_session(
        group_id=1,
        challenger=("100", "甲"),
        opponent=("200", "乙"),
        format_config=GEN9_OU,
    )
    status_image = await session.render_status_image()
    assert status_image and status_image.startswith("base64://")
    with pytest.raises(BattleSessionConflict):
        await manager.create_session(
            group_id=2,
            challenger=("100", "甲"),
            opponent=("300", "丙"),
            format_config=GEN9_OU,
        )
    await session.close()
    replacement = await manager.create_session(
        group_id=2,
        challenger=("100", "甲"),
        opponent=("300", "丙"),
        format_config=GEN9_OU,
    )
    assert await manager.get_session_by_user("100") is replacement
    await manager.close_all()


@pytest.mark.asyncio
async def test_champions_preview_and_double_forced_switch_defaults(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    session = await manager.create_session(
        group_id=1,
        challenger=("100", "甲"),
        opponent=("200", "乙"),
        format_config=CHAMPIONS_VGC_2026_REG_M_B,
    )
    assert session._default_choice_for_request({"teamPreview": True}) == "team 1234"
    preview_request = {
        "teamPreview": True,
        "maxChosenTeamSize": 4,
        "side": {
            "pokemon": [{"ident": f"p1: Pokemon {index}"} for index in range(1, 7)]
        },
    }
    assert session._parse_choice(preview_request, "team 3416") == "team 3416"
    assert session._parse_choice(preview_request, "team 341625") is None
    assert session._parse_choice(preview_request, "team 3413") is None
    _, preview_error = session._parse_team_preview_choice(
        preview_request,
        "team 341625",
    )
    assert preview_error and "选择 4 只" in preview_error
    await session._handle_team_preview_request(
        object(),
        "p1",
        {
            "side": {
                "pokemon": [
                    {
                        "ident": "p1: Raichu",
                        "details": "Raichu, L50",
                        "item": "Raichunite X",
                        "baseAbility": "Static",
                        "teraType": "Electric",
                    }
                ]
            }
        },
    )
    preview_entry = session._team_preview_data["p1"][0]
    assert preview_entry.item is None
    assert preview_entry.ability is None
    assert preview_entry.tera_type is None

    move_request = {
        "active": [
            {"moves": [{"move": "Thunderbolt", "pp": 15}]},
            {"moves": [{"move": "Dragon Claw", "pp": 15}]},
        ]
    }
    assert (
        session._parse_choice(
            move_request,
            "move1 1 1 mega; move2 1 1",
        )
        == "move 1 1 mega, move 1 1"
    )
    assert (
        session._default_choice_for_request(
            {
                "active": [
                    {"moves": [{"move": "Thunderbolt", "pp": 15, "target": "normal"}]},
                    {"moves": [{"move": "Protect", "pp": 10, "target": "self"}]},
                ]
            }
        )
        == "move 1 1, move 1"
    )

    request = {
        "forceSwitch": [True, True],
        "side": {
            "pokemon": [
                {"active": True, "condition": "0 fnt"},
                {"active": True, "condition": "0 fnt"},
                {"active": False, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }
    assert session._default_choice_for_request(request) == "switch 3, switch 4"
    await manager.close_all()


def test_ai_challenge_recognizes_bot_name_and_at_removed_by_onebot() -> None:
    assert _is_ai_challenge_request(
        bot_self_id="123",
        bot_name="Ophelia",
        target_id=None,
        argument="",
        to_me=True,
    )
    assert _is_ai_challenge_request(
        bot_self_id="123",
        bot_name="Ophelia",
        target_id="123",
        argument="",
        to_me=False,
    )
    assert _is_ai_challenge_request(
        bot_self_id="123",
        bot_name="Ophelia",
        target_id=None,
        argument="opHELia",
        to_me=False,
    )
    # Legacy generic aliases remain accepted but are no longer documented.
    assert _is_ai_challenge_request(
        bot_self_id="123",
        bot_name="Ophelia",
        target_id=None,
        argument="bot",
        to_me=False,
    )
    assert not _is_ai_challenge_request(
        bot_self_id="123",
        bot_name="Ophelia",
        target_id="456",
        argument="",
        to_me=True,
    )
    assert not _is_ai_challenge_request(
        bot_self_id="123",
        bot_name="Ophelia",
        target_id=None,
        argument="",
        to_me=False,
    )


def test_showdown_help_uses_configured_bot_name() -> None:
    registry = build_default_registry()
    messages = (
        _pokemon_help_overview("Ophelia"),
        _pokemon_challenge_help(registry, "Ophelia"),
        _pokemon_battle_action_help("Ophelia"),
    )
    assert all("Ophelia" in message for message in messages)
    assert all("人机" not in message and "AI" not in message for message in messages)
    assert "g9随机 Ophelia" in messages[1]
    assert "g9随机 bot" not in messages[1]


def test_champions_prompts_explain_pick_four_and_mega(
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    builder = PromptBuilder(translator, MoveDataRepository(runtime))
    preview_prompt = builder.build_team_preview_prompt(
        {
            "teamPreview": True,
            "maxChosenTeamSize": 4,
            "side": {
                "pokemon": [
                    {
                        "ident": f"p1: Pokemon {index}",
                        "active": index <= 2,
                    }
                    for index in range(1, 7)
                ]
            },
        }
    )
    assert "6 选 4" in preview_prompt
    assert "不是给全部成员排序" in preview_prompt
    assert "前两个编号是首发" in preview_prompt

    move_prompt = builder.build_move_prompt(
        {
            "active": [
                {
                    "canMegaEvo": True,
                    "moves": [
                        {
                            "move": "Thunderbolt",
                            "pp": 15,
                            "maxpp": 15,
                            "target": "normal",
                        }
                    ],
                },
                {
                    "moves": [
                        {
                            "move": "Protect",
                            "pp": 10,
                            "maxpp": 10,
                            "target": "self",
                        }
                    ]
                },
            ]
        }
    )
    assert "可 Mega 进化" in move_prompt
    assert "move1 1 1 mega" in move_prompt
    assert "[tera]" not in move_prompt
    assert "太晶化" not in move_prompt


def test_random_battle_prompt_lists_available_team_members(
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    builder = PromptBuilder(translator, MoveDataRepository(runtime))
    request = {
        "active": [
            {
                "moves": [
                    {
                        "move": "Thunderbolt",
                        "pp": 15,
                        "maxpp": 15,
                        "disabled": False,
                    }
                ]
            }
        ],
        "side": {
            "pokemon": [
                {
                    "ident": "p1: Pikachu",
                    "details": "Pikachu, L80, M",
                    "condition": "200/200",
                    "active": True,
                },
                {
                    "ident": "p1: Dragonite",
                    "details": "Dragonite, L75, F",
                    "condition": "250/250",
                    "active": False,
                },
                {
                    "ident": "p1: Corviknight",
                    "details": "Corviknight, L78",
                    "condition": "0 fnt",
                    "active": False,
                },
            ]
        },
    }

    prompt = builder.build_move_prompt(request)

    assert "【可换入队友】" in prompt
    assert "2. 快龙" in prompt
    assert "250/250" in prompt
    assert "3. 钢铠鸦" not in prompt


def test_ai_action_guide_describes_exact_single_and_double_commands() -> None:
    singles = build_ai_action_guide(
        {
            "active": [
                {
                    "canTerastallize": "Electric",
                    "moves": [
                        {
                            "move": "Thunderbolt",
                            "pp": 15,
                            "disabled": False,
                            "target": "normal",
                        },
                        {
                            "move": "Encore",
                            "pp": 0,
                            "disabled": True,
                            "target": "normal",
                        },
                    ],
                }
            ],
            "side": {
                "pokemon": [
                    {"ident": "p1: Pikachu", "active": True, "condition": "100/100"},
                    {"ident": "p1: Raichu", "active": False, "condition": "80/100"},
                    {"ident": "p1: Pichu", "active": False, "condition": "0 fnt"},
                ]
            },
        }
    )
    assert singles["request_type"] == "move_or_switch"
    single_rules = singles["rules"]
    first_slot = single_rules["active_slots"][0]
    assert first_slot["moves"][0]["command"] == "move 1"
    assert first_slot["moves"][0]["target_argument"]["required"] is False
    assert first_slot["moves"][0]["allowed_modifiers"] == ["tera"]
    assert first_slot["moves"][1]["usable"] is False
    assert first_slot["switch_commands"] == ["switch 2"]

    doubles = build_ai_action_guide(
        {
            "active": [
                {
                    "moves": [
                        {"move": "Psychic", "pp": 10, "target": "normal"},
                        {"move": "Helping Hand", "pp": 20, "target": "adjacentAlly"},
                    ]
                },
                {
                    "moves": [
                        {"move": "Heat Wave", "pp": 10, "target": "allAdjacentFoes"}
                    ]
                },
            ],
            "side": {
                "pokemon": [
                    {"ident": "p1a: Delphox", "active": True, "condition": "100/100"},
                    {
                        "ident": "p1b: Whimsicott",
                        "active": True,
                        "condition": "100/100",
                    },
                    {"ident": "p1: Garchomp", "active": False, "condition": "100/100"},
                ]
            },
        }
    )
    double_rules = doubles["rules"]
    slot_one = double_rules["active_slots"][0]
    slot_two = double_rules["active_slots"][1]
    assert slot_one["moves"][0]["command"] == "move1 1"
    assert slot_one["moves"][0]["target_argument"]["allowed_values"] == [1, 2, -2]
    assert slot_one["moves"][1]["target_argument"]["allowed_values"] == [-2]
    assert slot_two["moves"][0]["target_argument"]["required"] is False
    assert slot_one["switch_commands"] == ["switch1 3"]
    assert "各提交恰好一个动作" in double_rules["combined_choice"]


def test_ai_action_guide_describes_team_preview_and_partial_forced_switch() -> None:
    preview = build_ai_action_guide(
        {
            "teamPreview": True,
            "maxChosenTeamSize": 4,
            "side": {
                "pokemon": [{"ident": f"p1: Pokemon {index}"} for index in range(1, 7)]
            },
        }
    )
    assert preview["request_type"] == "team_preview"
    assert preview["rules"]["selection_size"] == 4
    assert len(preview["rules"]["members"]) == 6
    assert "恰好包含 4 个互不重复" in preview["rules"]["custom_command_rules"][0]

    forced = build_ai_action_guide(
        {
            "forceSwitch": [True, False],
            "side": {
                "pokemon": [
                    {"ident": "p1a: A", "active": True, "condition": "0 fnt"},
                    {"ident": "p1b: B", "active": True, "condition": "100/100"},
                    {"ident": "p1: C", "active": False, "condition": "100/100"},
                ]
            },
        }
    )
    assert forced["request_type"] == "forced_switch"
    assert forced["rules"]["active_slots"][0]["allowed_commands"] == ["switch1 3"]
    assert forced["rules"]["active_slots"][1]["allowed_commands"] == ["pass2"]


def test_ai_model_settings_inherit_main_chat_model_and_support_override() -> None:
    main = SimpleNamespace(
        bampi_model_provider="anthropic",
        bampi_model_id="claude-sonnet-test",
        bampi_model_api="anthropic-messages",
        bampi_api_key="main-secret",
        bampi_base_url="https://main.example/v1",
        bampi_thinking_level="high",
        bampi_bot_name="群聊小助手",
        bampi_persona="你是群里的冷静吐槽役，说话自然但不刻薄。",
    )
    inherited = AIModelSettings.from_config(ShowdownBattleConfig(), main)
    assert inherited.provider == "anthropic"
    assert inherited.model_id == "claude-sonnet-test"
    assert inherited.model_api == "anthropic-messages"
    assert inherited.api_key == "main-secret"
    assert inherited.base_url == "https://main.example/v1"
    assert inherited.thinking_level == "high"
    assert inherited.bot_name == "群聊小助手"
    assert inherited.persona == "你是群里的冷静吐槽役，说话自然但不刻薄。"

    main.bampi_model_api = "chat-completions"
    aliased = AIModelSettings.from_config(ShowdownBattleConfig(), main)
    assert aliased.model_api == "openai-completions"

    main.bampi_api_key = ""
    main.anthropic_api_key = "nonebot-config-secret"
    main.bampi_model_api = "anthropic-messages"
    from_nonebot_config = AIModelSettings.from_config(ShowdownBattleConfig(), main)
    assert from_nonebot_config.api_key == "nonebot-config-secret"

    overridden = AIModelSettings.from_config(
        ShowdownBattleConfig(
            showdown_battle_ai_model_provider="moonshot",
            showdown_battle_ai_model_id="kimi-test",
            showdown_battle_ai_base_url="https://ai.example/v1",
            showdown_battle_ai_thinking_level="low",
            showdown_battle_ai_persona="对战时保持安静。",
        ),
        main,
    )
    assert overridden.provider == "moonshot"
    assert overridden.model_id == "kimi-test"
    assert overridden.model_api == "auto"
    assert overridden.base_url == "https://ai.example/v1"
    assert overridden.thinking_level == "low"
    assert overridden.persona == "对战时保持安静。"


class _AgentSessionStub:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.messages: list[object] = []
        self.prompts: list[str] = []
        self.message_counts_before_prompt: list[int] = []
        self.move_results: list[str] = []
        self.is_processing = False
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def prompt(self, input, *, source: str = "interactive") -> None:
        assert source == "showdown_battle"
        self.message_counts_before_prompt.append(len(self.messages))
        self.prompts.append(str(input))
        self.messages.append(UserMessage(content=str(input)))
        tools = {tool.name: tool for tool in self.kwargs["tools"]}
        move_result = await tools["check_move"].execute(
            "move-info",
            {"actor": 1, "move_index": 1},
        )
        self.move_results.append(move_result.content[0].text)
        action_result = await tools["choose_battle_action"].execute(
            "choose",
            {"choice": "move 2"},
        )
        assert "行动已提交" in action_result.content[0].text
        self.messages.append(
            AssistantMessage(
                content=[TextContent(text=f"第{len(self.prompts)}回合，来吧！")]
            )
        )

    async def wait_for_idle(self) -> None:
        return None

    def abort(self, reason: str | None = None) -> None:
        self.is_processing = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_ai_opponent_keeps_cumulative_context_and_uses_on_demand_tools(
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    repository = MoveDataRepository(runtime)
    await repository.warm_up()
    created_sessions: list[_AgentSessionStub] = []

    def create_session(**kwargs):
        session = _AgentSessionStub(**kwargs)
        created_sessions.append(session)
        return session

    opponent = BattleAIOpponent(
        settings=AIModelSettings(
            provider="test-provider",
            model_id="test-model",
            model_api="openai-completions",
            api_key="test-secret",
            base_url="https://model.example/v1",
            thinking_level="off",
            decision_timeout_seconds=2,
            max_output_tokens=512,
            max_attempts=2,
            commentary_enabled=True,
            commentary_max_chars=100,
            bot_name="群聊小助手",
            persona="你是群里的冷静吐槽役，说话自然但不刻薄。",
        ),
        runtime=runtime,
        team_sources=object(),  # type: ignore[arg-type]
        move_repository=repository,
        translator=translator,
        agent_session_factory=create_session,
    )
    agent = opponent.create_battle_agent(
        battle_id="battle-1",
        format_config=GEN9_OU,
    )
    decision = AIBattleDecisionContext(
        battle_id="battle-1",
        format_id="gen9ou",
        format_name="Gen9 单打 OU",
        game_type="singles",
        ai_side="p2",
        turn_number=3,
        private_request={
            "active": [
                {
                    "moves": [
                        {"id": "thunderbolt", "move": "Thunderbolt", "pp": 15},
                        {"id": "voltswitch", "move": "Volt Switch", "pp": 20},
                    ]
                }
            ],
            "side": {"pokemon": [{"item": "AI-SECRET-ITEM"}]},
        },
        action_guide={
            "request_type": "move_or_switch",
            "rules": {"allowed": ["move 1", "move 2"]},
        },
        public_status="公开战况",
        public_knowledge=("p1a: Dragonite: moves=Extreme Speed",),
        public_events=("对手使用了公开招式",),
    )

    first = await agent.choose_action(
        decision,
        normalize_choice=lambda choice: (
            choice if choice in {"move 1", "move 2"} else None
        ),
    )
    second = await agent.choose_action(
        decision,
        normalize_choice=lambda choice: (
            choice if choice in {"move 1", "move 2"} else None
        ),
    )

    assert first == AIBattleDecision(choice="move 2", commentary="第1回合，来吧！")
    assert second == AIBattleDecision(choice="move 2", commentary="第2回合，来吧！")
    assert len(created_sessions) == 1
    runtime_session = created_sessions[0]
    assert runtime_session.started
    assert runtime_session.message_counts_before_prompt == [0, 2]
    assert len(agent.messages) == 4
    assert {tool.name for tool in runtime_session.kwargs["tools"]} == {
        "check_move",
        "choose_battle_action",
    }
    assert "十万伏特" in runtime_session.move_results[0]
    assert runtime_session.kwargs["stream_options"].api_key == "test-secret"
    assert (
        "整场对战共享同一个连续会话" in runtime_session.kwargs["custom_system_prompt"]
    )
    assert (
        "所有公开发言必须使用简体中文" in runtime_session.kwargs["custom_system_prompt"]
    )
    assert "不要硬凑口号" in runtime_session.kwargs["custom_system_prompt"]
    assert "你是群里的冷静吐槽役" in runtime_session.kwargs["custom_system_prompt"]
    assert "同一个“群聊小助手”" in runtime_session.kwargs["custom_system_prompt"]

    first_payload = json.loads(runtime_session.prompts[0].split("\n\n", 1)[1])
    assert first_payload["your_private_request"]["side"]["pokemon"][0]["item"] == (
        "AI-SECRET-ITEM"
    )
    assert "opponent_private_request" not in first_payload
    assert first_payload["publicly_revealed_information"] == [
        "p1a: Dragonite: moves=Extreme Speed"
    ]
    assert first_payload["legal_action_guide"]["request_type"] == ("move_or_switch")

    runtime_session.messages.append(
        AssistantMessage(content=[TextContent(text="[[BATTLE_SILENT]]")])
    )
    assert agent._extract_commentary(agent.messages) == ""

    await agent.close()
    assert runtime_session.closed


class _BattleAgentStub:
    def __init__(self) -> None:
        self.decisions: list[AIBattleDecisionContext] = []
        self.closed = False

    async def choose_action(self, decision, *, normalize_choice):
        self.decisions.append(decision)
        request_type = decision.action_guide.get("request_type")
        candidate = {
            "team_preview": "team 21",
            "forced_switch": "switch 2",
        }.get(request_type, "move 1")
        return AIBattleDecision(
            choice=normalize_choice(candidate),
            commentary="这回合结束后再说。",
        )

    async def close(self) -> None:
        self.closed = True


class _AIOpponentStub:
    def __init__(self) -> None:
        self.agent = _BattleAgentStub()

    @property
    def decisions(self) -> list[AIBattleDecisionContext]:
        return self.agent.decisions

    async def prepare_team(self, _format_config) -> AIPreparedTeam:
        return AIPreparedTeam(packed="", raw=None, label="random")

    def create_battle_agent(self, *, battle_id, format_config):
        del battle_id, format_config
        return self.agent


class _ChoiceProcessStub:
    def __init__(self) -> None:
        self.choices: list[tuple[str, str]] = []

    async def send_choice(self, side: str, choice: str) -> None:
        self.choices.append((side, choice))

    async def terminate(self) -> None:
        return None


class _BattleBotStub:
    def __init__(self) -> None:
        self.group_messages: list[object] = []
        self.private_messages: list[object] = []

    async def send_group_msg(self, **kwargs) -> None:
        self.group_messages.append(kwargs)

    async def send_private_msg(self, **kwargs) -> None:
        self.private_messages.append(kwargs)


@pytest.mark.asyncio
async def test_random_team_summary_retries_after_request_without_roster(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    session = await manager.create_session(
        group_id=9,
        challenger=("100", "甲"),
        opponent=("200", "乙"),
        format_config=registry.get("gen9randombattle"),
    )
    bot = _BattleBotStub()
    active = {
        "moves": [
            {
                "move": "Thunderbolt",
                "pp": 15,
                "maxpp": 15,
                "disabled": False,
            }
        ]
    }

    await session._notify_request(bot, "p1", {"active": [active]})
    assert "p1" not in session._team_summary_sent

    await session._notify_request(
        bot,
        "p1",
        {
            "active": [active],
            "side": {
                "pokemon": [
                    {
                        "ident": "p1: Pikachu",
                        "details": "Pikachu, L80",
                        "condition": "200/200",
                        "active": True,
                        "moves": ["thunderbolt"],
                    },
                    {
                        "ident": "p1: Dragonite",
                        "details": "Dragonite, L75",
                        "condition": "250/250",
                        "active": False,
                        "moves": ["extremespeed"],
                    },
                ]
            },
        },
    )

    assert "p1" in session._team_summary_sent
    assert any(
        "【随机队伍详情】" in str(item["message"]) and "快龙" in str(item["message"])
        for item in bot.private_messages
    )
    await manager.close_all()


@pytest.mark.asyncio
async def test_battle_updates_mirror_to_pvp_private_chats(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    session = await manager.create_session(
        group_id=9,
        challenger=("100", "甲"),
        opponent=("200", "乙"),
        format_config=GEN9_OU,
    )
    bot = _BattleBotStub()

    await session._send_battle_update(bot, MessageSegment.image("base64://update"))

    assert len(bot.group_messages) == 1
    assert {item["user_id"] for item in bot.private_messages} == {100, 200}
    with pytest.raises(ShowdownBridgeError, match="玩家对战仅支持"):
        session.set_interaction_channel("100", "group")
    await manager.close_all()


@pytest.mark.asyncio
async def test_ai_battle_routes_private_state_or_group_state_consistently(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    ai = _AIOpponentStub()
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
        ai_opponent=ai,  # type: ignore[arg-type]
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    session = await manager.create_ai_session(
        group_id=9,
        challenger=("100", "人类"),
        format_config=registry.get("gen9randombattle"),
    )
    bot = _BattleBotStub()

    assert session.can_accept_private_choice("100")
    assert not session.can_accept_group_choice("100", "move 1")
    await session._send_player_message(bot, "p1", "私聊操作提示")
    await session._send_battle_update(bot, MessageSegment.image("base64://private"))
    assert len(bot.private_messages) == 2
    assert len(bot.group_messages) == 1

    bot.private_messages.clear()
    bot.group_messages.clear()
    session.set_interaction_channel("100", "group")
    assert not session.can_accept_private_choice("100")
    session.state = "active"
    assert session.can_accept_group_choice("100", "move 1")
    await session._send_player_message(bot, "p1", "群聊操作提示")
    await session._send_battle_update(bot, MessageSegment.image("base64://group"))
    assert not bot.private_messages
    assert len(bot.group_messages) == 2
    assert "群聊操作提示" in str(bot.group_messages[0]["message"])
    await manager.close_all()


@pytest.mark.asyncio
async def test_ai_group_mode_allows_battle_team_guide_in_origin_group(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    ai = _AIOpponentStub()
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
        ai_opponent=ai,  # type: ignore[arg-type]
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    session = await manager.create_ai_session(
        group_id=9,
        challenger=("100", "人类"),
        format_config=GEN9_OU,
    )
    event = _group_message_event(user_id=100, group_id=9, text="组队")
    assert "对战准备" in (
        await _team_command_channel_error(
            config=ShowdownBattleConfig(), manager=manager, event=event
        )
        or ""
    )

    session.set_interaction_channel("100", "group")
    assert (
        await _team_command_channel_error(
            config=ShowdownBattleConfig(), manager=manager, event=event
        )
        is None
    )
    repository = TeamRepository(tmp_path / "teams.json")
    guide_sources = _GuideSourceStub()
    guide = TeamGuideManager(
        manager=manager,
        formats=registry,
        repository=repository,
        runtime=runtime,
        team_sources=guide_sources,  # type: ignore[arg-type]
        translator=translator,
        editor_flow=TeamEditorFlow(
            formats=registry,
            repository=repository,
            team_sources=guide_sources,  # type: ignore[arg-type]
            service=TeamEditorService(runtime=runtime, translator=translator),
        ),
        idle_ttl_seconds=900,
    )
    try:
        assert "选择队伍来源" in await guide.start("100", group_id=9)
        assert await guide.has_state("100", group_id=9)
    finally:
        await guide.close()
        await manager.close_all()


@pytest.mark.asyncio
async def test_ai_battle_maps_only_human_and_keeps_group_choices_out_of_ai_context(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    ai = _AIOpponentStub()
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
        ai_opponent=ai,  # type: ignore[arg-type]
        bot_name="测试Bot",
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    session = await manager.create_ai_session(
        group_id=9,
        challenger=("100", "人类"),
        format_config=registry.get("gen9randombattle"),
    )
    assert session.is_ai_battle
    assert session.players["p2"].is_ai
    assert session.players["p2"].display_name == "测试Bot"
    assert await manager.get_session_by_user("100") is session
    assert await manager.get_session_by_user("__bampi_showdown_ai__") is None

    process = _ChoiceProcessStub()
    bot = _BattleBotStub()
    session.process = process  # type: ignore[assignment]
    session.state = "active"
    human_request = {
        "active": [{"moves": [{"move": "Human Secret Move", "pp": 1}]}],
        "human_private_marker": "DO-NOT-SHARE",
    }
    ai_request = {
        "active": [{"moves": [{"move": "Thunderbolt", "pp": 15}]}],
        "side": {"pokemon": [{"ident": "p2: Pikachu", "active": True}]},
    }
    session.current_requests["p1"] = human_request
    session.current_requests["p2"] = ai_request
    session._record_public_reveal("|move|p1a: Dragonite|Protect|p1a: Dragonite")
    session._record_public_reveal("|-ability|p1a: Dragonite|Multiscale")
    session.set_interaction_channel("100", "group")

    assert session.can_accept_group_choice("100", "move 1")
    assert not session.can_accept_group_choice("999", "move 1")
    assert await session.handle_choice(bot, "100", "move 1") == "指令已提交。"
    assert process.choices == [("p1", "move 1")]

    # Defense in depth: even an accidental call with the human side must not
    # create an AI request containing that side-private payload.
    session._schedule_ai_action(bot, "p1", human_request)
    assert not session._ai_action_tasks

    session._schedule_ai_action(bot, "p2", ai_request)
    tasks = list(session._ai_action_tasks.values())
    await asyncio.gather(*tasks)
    assert process.choices[-1] == ("p2", "move 1")
    assert len(ai.decisions) == 1
    serialized_ai_context = ai.decisions[0].to_prompt_payload()
    assert "Thunderbolt" in serialized_ai_context
    assert "Protect" in serialized_ai_context
    assert "Multiscale" in serialized_ai_context
    assert "DO-NOT-SHARE" not in serialized_ai_context
    assert "Human Secret Move" not in serialized_ai_context
    assert not bot.group_messages
    await session._flush_ai_commentary(bot)
    assert "这回合结束后再说" in str(bot.group_messages[-1]["message"])

    session._awaiting_resolution.discard("p2")
    preview_request = {
        "teamPreview": True,
        "maxTeamSize": 2,
        "side": {
            "pokemon": [
                {"ident": "p2: Pikachu"},
                {"ident": "p2: Dragonite"},
                {"ident": "p2: Corviknight"},
            ]
        },
    }
    session.current_requests["p2"] = preview_request
    session._schedule_ai_action(bot, "p2", preview_request)
    await asyncio.gather(*session._ai_action_tasks.values())
    assert process.choices[-1] == ("p2", "team 21")
    assert ai.decisions[-1].action_guide["request_type"] == "team_preview"
    assert ai.decisions[-1].action_guide["rules"]["selection_size"] == 2

    session._schedule_ai_action(bot, "p2", {"wait": True})
    assert not session._ai_action_tasks
    assert len(ai.decisions) == 2

    await manager.close_all()
    assert ai.agent.closed


def test_render_context_helpers() -> None:
    from bampi.plugins.showdown_battle.rendering.context import (
        calculate_hp_ratio,
        hp_bar_class,
        sanitize_log_lines,
        status_class,
    )
    from bampi.plugins.showdown_battle.rendering.sprites import (
        slug_candidates,
        to_slug,
    )

    assert calculate_hp_ratio("142/211") == pytest.approx(142 / 211)
    assert calculate_hp_ratio("45%") == pytest.approx(0.45)
    assert calculate_hp_ratio("0/100") == 0.0
    assert calculate_hp_ratio("") == 1.0
    assert calculate_hp_ratio("garbage") == 0.0

    assert hp_bar_class(0.8) == "hp-high"
    assert hp_bar_class(0.3) == "hp-mid"
    assert hp_bar_class(0.1) == "hp-low"

    assert status_class("灼伤") == "brn"
    assert status_class("倒下") == "fnt"
    assert status_class("未知状态") == "generic"

    banner, lines = sanitize_log_lines(
        ["—— 第 3 回合 ——", "—— 决胜时刻 ——", "皮卡丘使用了电光一闪。", ""]
    )
    assert banner == "决胜时刻"
    assert lines == ["皮卡丘使用了电光一闪。"]
    banner, lines = sanitize_log_lines([])
    assert banner == ""
    assert lines == ["暂无新的战况日志。"]

    assert to_slug("Landorus-Therian") == "landorustherian"
    assert slug_candidates("Landorus-Therian") == [
        "landorus-therian",
        "landorustherian",
    ]
    assert slug_candidates("Ho-Oh") == ["ho-oh", "hooh"]
    assert slug_candidates("Pikachu") == ["pikachu"]


@pytest.mark.asyncio
async def test_item_data_repository_indexes_ids_and_names(
    runtime: ShowdownRuntime,
) -> None:
    from bampi.plugins.showdown_battle.item_data import ItemDataRepository

    repository = ItemDataRepository(runtime)
    await repository.warm_up()

    entry = repository.get("choiceband")
    assert entry is not None
    assert entry.name == "Choice Band"
    assert isinstance(entry.spritenum, int) and entry.spritenum >= 0
    assert repository.get("Choice Band") is entry
    assert repository.get("Assault Vest") is repository.get("assaultvest")
    assert repository.get("not-an-item") is None


def test_crop_item_icon_cuts_24px_tiles() -> None:
    import io as _io

    from PIL import Image

    from bampi.plugins.showdown_battle.rendering.sprites import _crop_item_icon

    sheet_image = Image.new("RGBA", (384, 48), (0, 0, 0, 0))
    # Paint tile #17 (row 1, column 1) solid red.
    for x in range(24, 48):
        for y in range(24, 48):
            sheet_image.putpixel((x, y), (255, 0, 0, 255))
    buffer = _io.BytesIO()
    sheet_image.save(buffer, format="PNG")
    sheet = buffer.getvalue()

    tile = _crop_item_icon(sheet, 17)
    assert tile is not None
    with Image.open(_io.BytesIO(tile)) as decoded:
        assert decoded.size == (24, 24)
        assert decoded.convert("RGBA").getpixel((12, 12)) == (255, 0, 0, 255)

    # Sprite numbers beyond the sheet must not produce a bogus icon.
    assert _crop_item_icon(sheet, 999) is None


@pytest.mark.asyncio
async def test_sprite_store_serves_cached_item_icons(tmp_path: Path) -> None:
    import base64 as _b64

    from bampi.plugins.showdown_battle.rendering.sprites import SpriteStore

    tiny_png = _b64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    cache = tmp_path / "sprites"
    icon_path = cache / "itemicons" / "68.png"
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    icon_path.write_bytes(tiny_png)

    store = SpriteStore(cache, download_timeout=0.1)
    try:
        uri = await store.get_item_icon_data_uri(68)
        assert uri == f"data:image/png;base64,{_b64.b64encode(tiny_png).decode()}"
        assert await store.get_item_icon_data_uri(None) is None
        assert await store.get_item_icon_data_uri(-1) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_battle_renderer_produces_images(
    tmp_path: Path, translator: TranslationService
) -> None:
    from bampi.browser import find_chromium
    from bampi.plugins.showdown_battle.battle.state import PokemonState
    from bampi.plugins.showdown_battle.rendering import (
        PokemonBattleRenderer,
        TeamPreviewPokemon,
    )

    if find_chromium() is None:
        pytest.skip("no chromium available for render smoke test")

    # Pre-populate the sprite cache with a tiny PNG so the smoke test does
    # not depend on network access; the second species exercises the
    # missing-sprite placeholder path.
    import base64 as _b64

    tiny_png = _b64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    sprite_dir = tmp_path / "sprites"
    for view in ("front", "back"):
        target = sprite_dir / f"gen5-{view}" / "pikachu.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tiny_png)
    icon_path = sprite_dir / "itemicons" / "242.png"
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    icon_path.write_bytes(tiny_png)

    from bampi.plugins.showdown_battle.item_data import ItemDataRepository

    item_repository = ItemDataRepository(runtime=None)  # type: ignore[arg-type]
    item_repository._build_index(
        {"items": {"leftovers": {"name": "Leftovers", "spritenum": 242}}}
    )

    renderer = PokemonBattleRenderer(
        sprite_cache_dir=sprite_dir,
        browser_work_dir=tmp_path / "browser",
        sprite_download_timeout=0.5,
        render_idle_ttl_seconds=0,
        item_repository=item_repository,
    )
    players = {
        "p1": PlayerSlot(side="p1", user_id="100", display_name="甲"),
        "p2": PlayerSlot(side="p2", user_id="200", display_name="乙"),
    }
    state = BattleState(turn_number=2)
    state.pokemon["p1a: Pikachu"] = PokemonState(
        ident="p1a: Pikachu",
        side="p1",
        name="皮卡丘",
        details="Pikachu",
        hp="88/211",
        active=True,
        status="麻痹",
    )
    state.pokemon["p2a: Missing"] = PokemonState(
        ident="p2a: Missing",
        side="p2",
        name="占位兽",
        details="Missingno-Test",
        hp="55/100",
        active=True,
    )
    try:
        turn_image = await renderer.render_turn_log(
            format_name="Gen 9 OU",
            format_id="gen9ou",
            players=players,
            state=state,
            translator=translator,
            lines=["—— 测试横幅 ——", "皮卡丘使用了十万伏特！"],
            # Cached icon path for p1, text-label fallback path for p2.
            items={"p1|pikachu": "leftovers", "p2|missing": "unknownitem"},
        )
        assert turn_image and turn_image.startswith("base64://")

        preview_image = await renderer.render_team_preview(
            format_name="Gen 9 OU",
            format_id="gen9ou",
            players=players,
            preview={
                "p1": [
                    TeamPreviewPokemon(
                        "p1: Pikachu", "Pikachu", "皮卡丘", level=50, gender="♂"
                    )
                ],
                "p2": [
                    TeamPreviewPokemon("p2: Missing", "Missingno-Test", "占位兽")
                ],
            },
        )
        assert preview_image and preview_image.startswith("base64://")

        status_image = await renderer.render_status_panel(
            format_name="Gen 9 OU",
            format_id="gen9ou",
            players=players,
            state=state,
            translator=translator,
        )
        assert status_image and status_image.startswith("base64://")
    finally:
        await renderer.shutdown()


def test_battle_state_adopts_preview_idents(translator: TranslationService) -> None:
    state = BattleState()
    state.register_pokemon("p1: Pikachu", "Pikachu, L50, M", translator)
    state.register_pokemon("p1: Dragonite", "Dragonite, L50", translator)
    assert state.side_rosters["p1"] == ["p1: Pikachu", "p1: Dragonite"]

    adopted = state.register_pokemon("p1a: Dragonite", "Dragonite, L50", translator)
    assert adopted.ident == "p1a: Dragonite"
    assert "p1: Dragonite" not in state.pokemon
    assert state.side_rosters["p1"] == ["p1: Pikachu", "p1a: Dragonite"]


def test_teamsize_tracking_and_ball_rows(translator: TranslationService) -> None:
    from bampi.plugins.showdown_battle.rendering.context import build_ball_rows

    players = {
        "p1": PlayerSlot(side="p1", user_id="100", display_name="甲"),
        "p2": PlayerSlot(side="p2", user_id="200", display_name="乙"),
    }
    state = BattleState()
    formatter = BattleEventFormatter(
        state=state, translator=translator, players=players
    )

    assert formatter.format("|teamsize|p1|3") is None
    assert formatter.format("|teamsize|p2|3") is None
    assert state.team_sizes == {"p1": 3, "p2": 3}

    formatter.format("|switch|p1a: Pikachu|Pikachu, L80|200/200")
    formatter.format("|switch|p2a: Garchomp|Garchomp, L76|250/250")
    formatter.format("|faint|p1a: Pikachu")

    rows = build_ball_rows(state)
    assert rows["p1"] == ["fainted", "unknown", "unknown"]
    assert rows["p2"] == ["alive", "unknown", "unknown"]


def test_ball_rows_ignore_preview_only_entries(
    translator: TranslationService,
) -> None:
    """VGC-style six-pick-four rosters must not inflate the ball row."""
    from bampi.plugins.showdown_battle.rendering.context import build_ball_rows

    state = BattleState()
    for species in ("Pikachu", "Dragonite", "Garchomp", "Kingambit", "Rotom", "Ceruledge"):
        state.register_pokemon(f"p1: {species}", f"{species}, L50", translator)
    state.team_sizes["p1"] = 4
    state.register_pokemon("p1a: Pikachu", "Pikachu, L50", translator)
    state.register_pokemon("p1b: Garchomp", "Garchomp, L50", translator)

    rows = build_ball_rows(state)
    assert rows["p1"] == ["alive", "alive", "unknown", "unknown"]
    assert rows["p2"] == []


def test_event_formatter_explains_damage_and_heal_causes(
    translator: TranslationService,
) -> None:
    players = {
        "p1": PlayerSlot(side="p1", user_id="100", display_name="甲"),
        "p2": PlayerSlot(side="p2", user_id="200", display_name="乙"),
    }
    state = BattleState()
    formatter = BattleEventFormatter(
        state=state, translator=translator, players=players
    )
    formatter.format("|switch|p1a: Pikachu|Pikachu, L80|200/200")
    formatter.format("|switch|p2a: Garchomp|Garchomp, L76|250/250")

    message = formatter.format("|-damage|p1a: Pikachu|180/200|[from] Stealth Rock")
    assert message is not None and "受到隐形岩的伤害" in message

    message = formatter.format("|-damage|p1a: Pikachu|170/200|[from] Sandstorm")
    assert message is not None and "沙暴" in message

    message = formatter.format("|-damage|p1a: Pikachu|160/200 psn|[from] psn")
    assert message is not None and "受到中毒的伤害" in message

    message = formatter.format("|-damage|p1a: Pikachu|150/200|[from] recoil")
    assert message is not None and "反作用力" in message

    message = formatter.format(
        "|-damage|p1a: Pikachu|140/200|[from] ability: Rough Skin|[of] p2a: Garchomp"
    )
    assert message is not None
    assert "烈咬陆鲨的特性" in message and "受到" in message

    message = formatter.format("|-heal|p1a: Pikachu|155/200|[from] item: Leftovers")
    assert message is not None
    assert "通过道具 吃剩的东西回复了体力" in message

    message = formatter.format("|-heal|p1a: Pikachu|165/200|[from] Grassy Terrain")
    assert message is not None and "青草场地" in message

    # Plain lines keep the original concise form.
    message = formatter.format("|-damage|p1a: Pikachu|100/200")
    assert message is not None and message.endswith("HP 100/200")

    message = formatter.format("|-status|p1a: Pikachu|brn|[from] item: Flame Orb")
    assert message is not None and "陷入 灼伤（道具 火焰宝珠）" in message

    message = formatter.format("|-enditem|p1a: Pikachu|Sitrus Berry|[eat]")
    assert message is not None and "吃掉了 文柚果" in message

    message = formatter.format(
        "|-enditem|p1a: Pikachu|Leftovers|[from] move: Knock Off|[of] p2a: Garchomp"
    )
    assert message is not None
    assert "失去了道具 吃剩的东西" in message and "拍落" in message

    message = formatter.format(
        "|-item|p2a: Garchomp|Leftovers|[from] ability: Frisk|[of] p1a: Pikachu|[identify]"
    )
    assert message is not None and "被发现携带道具" in message


@pytest.mark.asyncio
async def test_check_move_details_include_mechanics_line(
    runtime: ShowdownRuntime, translator: TranslationService
) -> None:
    from bampi.plugins.showdown_battle.commands import (
        _format_move_details,
        _resolve_move_entry_by_name,
    )

    repo = MoveDataRepository(runtime)
    await repo.warm_up()

    entry = _resolve_move_entry_by_name("戏法空间", translator, repo)
    assert entry is not None
    details = _format_move_details(entry, translator)
    assert "效果：" in details
    assert "机制：" in details
    assert "速度慢的宝可梦先行动" in details


async def _make_pvp_session(runtime, translator):
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    session = await manager.create_session(
        group_id=9,
        challenger=("100", "甲"),
        opponent=("200", "乙"),
        format_config=registry.get("gen9randombattle"),
    )
    return manager, session


@pytest.mark.asyncio
async def test_item_labels_stay_private_in_pvp_battles(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    manager, session = await _make_pvp_session(runtime, translator)
    session.current_requests["p1"] = {
        "side": {
            "pokemon": [
                {"ident": "p1: Pikachu", "item": "lightball"},
                {"ident": "p1: Dragonite", "item": ""},
            ]
        }
    }

    assert session._items_for_viewer(None) == {}
    assert session._items_for_viewer("p2") == {}
    viewer_items = session._items_for_viewer("p1")
    assert viewer_items == {"p1|pikachu": "lightball"}

    # Items revealed by one-off battle events are remembered by the AI
    # knowledge base but never drawn on images: like on cartridge, players
    # are expected to track those themselves.
    session._record_public_reveal("|-item|p2a: Garchomp|Leftovers")
    session._record_public_reveal("|-enditem|p2a: Garchomp|Leftovers")
    assert session._public_revealed_items["p2a: Garchomp"] == "Leftovers"
    assert session._items_for_viewer(None) == {}
    assert "p2|garchomp" not in session._items_for_viewer("p1")

    await manager.close_all()


@pytest.mark.asyncio
async def test_flush_group_renders_personal_item_variant_for_pvp(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    manager, session = await _make_pvp_session(runtime, translator)
    renderer = session.renderer
    session.current_requests["p1"] = {
        "side": {"pokemon": [{"ident": "p1: Pikachu", "item": "lightball"}]}
    }
    session._group_buffer.append("皮卡丘使用了十万伏特！")
    bot = _BattleBotStub()

    await session._flush_group(bot)

    # One public render plus one personalized render for p1; p2 reuses the
    # public image because their view adds no extra information.
    assert renderer.calls.count("turn_log") == 2
    public_items, p1_items = renderer.turn_log_items
    assert "p1|pikachu" not in public_items
    assert p1_items["p1|pikachu"] == "lightball"

    assert len(bot.group_messages) == 1
    assert "stub-turn-log-1" in str(bot.group_messages[0]["message"])
    private_by_user = {
        item["user_id"]: str(item["message"]) for item in bot.private_messages
    }
    assert "stub-turn-log-2" in private_by_user[100]
    assert "stub-turn-log-1" in private_by_user[200]

    await manager.close_all()


@pytest.mark.asyncio
async def test_ai_battle_shows_human_items_on_public_image(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    ai = _AIOpponentStub()
    registry = build_default_registry()
    manager = BattleManager(
        translator=translator,
        formats=registry,
        runtime=runtime,
        move_repository=MoveDataRepository(runtime),
        renderer=_StubBattleRenderer(),
        max_render_concurrency=1,
        ai_opponent=ai,  # type: ignore[arg-type]
    )
    manager.mark_runtime_ready(
        ShowdownRuntimeInfo(
            version="0.11.11",
            node_version="test",
            formats={item.format_id: item.display_name for item in registry.all()},
        )
    )
    session = await manager.create_ai_session(
        group_id=9,
        challenger=("100", "人类"),
        format_config=registry.get("gen9randombattle"),
    )
    human_side = session.get_side_by_user("100")
    assert human_side is not None
    ai_side = "p2" if human_side == "p1" else "p1"
    session.current_requests[human_side] = {
        "side": {"pokemon": [{"ident": f"{human_side}: Pikachu", "item": "lightball"}]}
    }
    session.current_requests[ai_side] = {
        "side": {"pokemon": [{"ident": f"{ai_side}: Garchomp", "item": "leftovers"}]}
    }

    public_items = session._items_for_viewer(None)
    assert f"{human_side}|pikachu" in public_items
    # The bot's own unrevealed item must stay hidden from the human player.
    assert f"{ai_side}|garchomp" not in public_items
    assert f"{ai_side}|garchomp" not in session._items_for_viewer(human_side)

    await manager.close_all()


@pytest.mark.asyncio
async def test_open_team_sheet_reveals_items_publicly(
    tmp_path: Path,
    runtime: ShowdownRuntime,
    translator: TranslationService,
) -> None:
    manager, session = await _make_pvp_session(runtime, translator)
    session._record_public_reveal(
        "|showteam|p1|Pikachu||lightball|Static|Thunderbolt,Volt Switch|Timid||"
        "|M||50|]Dragonite||leftovers|Multiscale|Extreme Speed|Adamant|||M||50|"
    )

    public_items = session._items_for_viewer(None)
    assert public_items["p1|pikachu"] == "lightball"
    assert public_items["p1|dragonite"] == "leftovers"
    assert not any(key.startswith("p2|") for key in public_items)

    await manager.close_all()
