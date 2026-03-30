from __future__ import annotations

from nonebot import get_driver, logger
from nonebot.plugin import PluginMetadata, get_plugin_config

from .config import BampiChatConfig
from .handler import register_handlers
from .session_manager import GroupSessionManager

__plugin_meta__ = PluginMetadata(
    name="bampi_chat",
    description="基于 bampy 的 NoneBot2 群聊 Agent 插件",
    usage="配置模型与 workspace 后即可在群聊中触发 agent 对话",
    type="application",
    config=BampiChatConfig,
)

plugin_config: BampiChatConfig | None = None
group_session_manager: GroupSessionManager | None = None

try:
    driver = get_driver()
except ValueError:
    driver = None
else:
    plugin_config = get_plugin_config(BampiChatConfig)
    group_session_manager = GroupSessionManager(plugin_config)
    register_handlers(plugin_config, group_session_manager)
    logger.info(
        f"bampi_chat plugin ready enabled={plugin_config.bampi_enabled} "
        f"provider={plugin_config.bampi_model_provider} "
        f"model={plugin_config.bampi_model_id} "
        f"prefixes={plugin_config.bampi_trigger_prefix} "
        f"keywords={plugin_config.bampi_trigger_keywords} "
        f"workspace_dir={plugin_config.bampi_workspace_dir} "
        f"session_dir={plugin_config.bampi_session_dir} "
        f"idle_session_ttl={plugin_config.bampi_session_idle_ttl_seconds}s "
        f"bash_mode={plugin_config.bampi_bash_mode} "
        f"bash_container={plugin_config.bampi_bash_container_name} "
        f"bash_workdir={plugin_config.bampi_bash_container_workdir} "
        f"reply_quote={plugin_config.bampi_reply_with_quote} "
        f"at_sender={plugin_config.bampi_at_sender} "
        f"live_progress={plugin_config.bampi_live_progress_enabled} "
        f"live_text_stream={plugin_config.bampi_live_text_stream_enabled} "
        f"compaction_notice={plugin_config.bampi_threshold_compaction_notice_enabled}"
    )

    @driver.on_shutdown
    async def _close_bampi_sessions() -> None:
        if group_session_manager is not None:
            logger.info("bampi_chat shutting down, closing active sessions")
            await group_session_manager.close_all()
