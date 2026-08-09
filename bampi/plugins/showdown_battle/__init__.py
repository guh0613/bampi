from __future__ import annotations

from nonebot import get_driver, logger
from nonebot.plugin import PluginMetadata, get_plugin_config

from .ai_opponent import AIModelSettings, BattleAIOpponent
from .bridge import ShowdownRuntime, ShowdownRuntimeUnavailable
from .commands import register_commands
from .config import ShowdownBattleConfig
from .formats import FormatRegistry, build_default_registry
from .item_data import ItemDataRepository
from .manager import BattleManager
from .move_data import MoveDataRepository
from .rendering import PokemonBattleRenderer
from .team.editor.flow import TeamEditorFlow
from .team.editor.service import TeamEditorService
from .team.guide import TeamGuideManager
from .team.repository import TeamRepository
from .team.sources import TeamSourceService
from .translations import TranslationService


__plugin_meta__ = PluginMetadata(
    name="Pokémon Showdown 对战",
    description=(
        "在 QQ 群中运行本地 Pokémon Showdown 对战，支持 Gen9 OU、"
        "随机单/双打、Doubles OU 与 Pokémon Champions VGC。"
    ),
    usage=(
        "发送“组队”打开队伍中心，“宝可梦帮助”查看上下文帮助；"
        "群内可发送“g9ou @对手”挑战玩家，或用“g9ou <Bot名称>”"
        "挑战聊天 Bot；其他规则用法相同。"
    ),
    type="application",
    config=ShowdownBattleConfig,
)

plugin_config: ShowdownBattleConfig | None = None
format_registry: FormatRegistry | None = None
translation_service: TranslationService | None = None
showdown_runtime: ShowdownRuntime | None = None
move_repository: MoveDataRepository | None = None
item_repository: ItemDataRepository | None = None
team_repository: TeamRepository | None = None
team_source_service: TeamSourceService | None = None
team_editor_service: TeamEditorService | None = None
team_editor_flow: TeamEditorFlow | None = None
team_guide_manager: TeamGuideManager | None = None
ai_opponent: BattleAIOpponent | None = None
battle_renderer: PokemonBattleRenderer | None = None
battle_manager: BattleManager | None = None

try:
    driver = get_driver()
except ValueError:
    driver = None
else:
    plugin_config = get_plugin_config(ShowdownBattleConfig)
    if plugin_config.showdown_battle_enabled:
        bot_name = (
            str(getattr(driver.config, "bampi_bot_name", "Ophelia") or "").strip()
            or "Ophelia"
        )
        format_registry = build_default_registry()
        translation_service = TranslationService.from_file(plugin_config.i18n_file)
        showdown_runtime = ShowdownRuntime(
            node_bin=plugin_config.resolve_node_binary(),
            package_dir=plugin_config.package_dir,
        )
        move_repository = MoveDataRepository(showdown_runtime)
        item_repository = ItemDataRepository(showdown_runtime)
        team_repository = TeamRepository(plugin_config.team_storage_path)
        team_source_service = TeamSourceService(
            runtime=showdown_runtime,
            translator=translation_service,
            timeout_seconds=(plugin_config.showdown_battle_team_source_timeout_seconds),
            max_bytes=plugin_config.showdown_battle_team_source_max_bytes,
            cache_ttl_seconds=(
                plugin_config.showdown_battle_team_source_cache_ttl_seconds
            ),
        )
        if plugin_config.showdown_battle_ai_enabled:
            try:
                ai_settings = AIModelSettings.from_config(plugin_config, driver.config)
                ai_opponent = BattleAIOpponent(
                    settings=ai_settings,
                    runtime=showdown_runtime,
                    team_sources=team_source_service,
                    move_repository=move_repository,
                    translator=translation_service,
                )
                logger.info(
                    "showdown AI opponent configured "
                    f"provider={ai_opponent.model.provider} "
                    f"api={ai_opponent.model.api} model={ai_opponent.model.id}"
                )
            except Exception:
                ai_opponent = None
                logger.exception("showdown AI opponent configuration failed")
        battle_renderer = PokemonBattleRenderer(
            sprite_cache_dir=plugin_config.sprite_cache_dir,
            browser_work_dir=plugin_config.render_browser_dir,
            sprite_download_timeout=(
                plugin_config.showdown_battle_sprite_download_timeout_seconds
            ),
            browser_executable=plugin_config.showdown_battle_browser_executable,
            render_scale=plugin_config.showdown_battle_render_scale,
            render_idle_ttl_seconds=(
                plugin_config.showdown_battle_render_idle_ttl_seconds
            ),
            item_repository=item_repository,
        )
        battle_manager = BattleManager(
            translator=translation_service,
            formats=format_registry,
            runtime=showdown_runtime,
            move_repository=move_repository,
            renderer=battle_renderer,
            max_render_concurrency=(
                plugin_config.showdown_battle_max_render_concurrency
            ),
            ai_opponent=ai_opponent,
            bot_name=bot_name,
            ai_public_history_events=(
                plugin_config.showdown_battle_ai_public_history_events
            ),
        )
        battle_manager.mark_runtime_unavailable("尚未完成启动检查。")
        team_editor_service = TeamEditorService(
            runtime=showdown_runtime,
            translator=translation_service,
        )
        team_editor_flow = TeamEditorFlow(
            formats=format_registry,
            repository=team_repository,
            team_sources=team_source_service,
            service=team_editor_service,
        )
        team_guide_manager = TeamGuideManager(
            manager=battle_manager,
            formats=format_registry,
            repository=team_repository,
            runtime=showdown_runtime,
            team_sources=team_source_service,
            translator=translation_service,
            editor_flow=team_editor_flow,
            idle_ttl_seconds=plugin_config.showdown_battle_team_guide_ttl_seconds,
        )
        register_commands(
            config=plugin_config,
            battle_manager=battle_manager,
            format_registry=format_registry,
            translation_service=translation_service,
            team_repository=team_repository,
            runtime=showdown_runtime,
            move_repository=move_repository,
            team_sources=team_source_service,
            team_guide=team_guide_manager,
        )
        logger.info(
            "showdown battle plugin configured "
            f"package_dir={plugin_config.package_dir} "
            f"data_dir={plugin_config.data_dir} "
            f"ai_enabled={ai_opponent is not None} "
            f"formats={[item.format_id for item in format_registry.all()]}"
        )
    else:
        logger.info("showdown battle plugin disabled by configuration")

    @driver.on_startup
    async def _start_showdown_battle() -> None:
        if (
            battle_manager is None
            or showdown_runtime is None
            or move_repository is None
            or format_registry is None
            or translation_service is None
        ):
            return
        format_ids = [item.format_id for item in format_registry.all()]
        try:
            info = await showdown_runtime.inspect(format_ids)
            missing = sorted(set(format_ids) - set(info.formats))
            if missing:
                raise ShowdownRuntimeUnavailable(
                    "当前 Showdown 不支持规则：" + ", ".join(missing)
                )
            if translation_service.info.pokemon_showdown_version != info.version:
                raise ShowdownRuntimeUnavailable(
                    "翻译目录与 Showdown 版本不一致："
                    f"{translation_service.info.pokemon_showdown_version} != "
                    f"{info.version}"
                )
            await move_repository.warm_up()
        except Exception as exc:
            battle_manager.mark_runtime_unavailable(str(exc))
            logger.exception("showdown battle startup check failed")
            return
        if item_repository is not None:
            try:
                await item_repository.warm_up()
            except Exception:
                # Item icons are cosmetic; a failed warm-up only means the
                # renderer falls back to text labels.
                logger.exception("showdown item data warm-up failed")
        battle_manager.mark_runtime_ready(info)
        logger.info(
            "showdown battle runtime ready "
            f"showdown={info.version} node={info.node_version} "
            f"formats={info.formats}"
        )

    @driver.on_shutdown
    async def _stop_showdown_battle() -> None:
        if battle_manager is not None:
            await battle_manager.close_all()
            logger.info("showdown battle sessions closed")
        if battle_renderer is not None:
            await battle_renderer.shutdown()
        if team_guide_manager is not None:
            await team_guide_manager.close()
        if team_source_service is not None:
            await team_source_service.close()


__all__ = [
    "ShowdownBattleConfig",
    "ai_opponent",
    "battle_manager",
    "battle_renderer",
    "format_registry",
    "item_repository",
    "move_repository",
    "plugin_config",
    "showdown_runtime",
    "team_editor_flow",
    "team_editor_service",
    "team_guide_manager",
    "team_repository",
    "team_source_service",
    "translation_service",
]
