# 配置参考

Bampi 的聊天能力由 NoneBot 插件 `bampi_chat` 提供。配置项对应 `BampiChatConfig` 字段；写入 `.env` 时一般为**字段名大写**（如 `bampi_api_key` → `BAMPI_API_KEY`）。

**密钥不要提交到 Git。** 项目已将 `.env` 加入 gitignore；可用 `.env.example` 作模板。

列表类字段：

- `bampi_trigger_prefix` / `bampi_trigger_keywords` / `bampi_group_whitelist`：优先用 **JSON 数组**（如 `["@bot","小帮"]`）；若写成单个普通字符串，会当作只含一项的列表。
- `bampi_model_input_types`：支持 **JSON 数组**或**逗号分隔**（如 `text,image`），且必须包含 `text`。

部署步骤见 [快速开始](getting-started.md)；模型协议细节见 [bampy providers](https://github.com/guh0613/bampy/blob/main/docs/providers.md)；整体数据流见 [架构](architecture.md)。

## 最小推荐配置

```env
# 模型（按你的供应商改）
BAMPI_MODEL_PROVIDER=openai
BAMPI_MODEL_ID=gpt-5-mini
BAMPI_MODEL_API=auto
BAMPI_API_KEY=sk-your-key
# BAMPI_BASE_URL=https://api.example.com/v1   # 兼容网关时再填

# 建议先限制群，避免误进无关群
BAMPI_GROUP_WHITELIST=["123456789"]

# 触发（默认前缀已是 @bot，可按需改）
# BAMPI_TRIGGER_PREFIX=["@bot","小帮"]
```

## 模型

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_model_provider` (`BAMPI_MODEL_PROVIDER`) | `openai` | 模型供应商标识 |
| `bampi_model_id` (`BAMPI_MODEL_ID`) | `gpt-5-mini` | 模型名 |
| `bampi_model_api` (`BAMPI_MODEL_API`) | `auto` | 协议：`auto` / `anthropic-messages` / `openai-responses` / `openai-completions` / `google-genai`（及常见别名） |
| `bampi_api_key` (`BAMPI_API_KEY`) | 空 | API Key |
| `bampi_base_url` (`BAMPI_BASE_URL`) | 空 | 自定义 API Base URL |
| `bampi_model_input_types` (`BAMPI_MODEL_INPUT_TYPES`) | 未设则按模型能力 | 覆盖输入类型：`text` / `image`（须含 `text`） |
| `bampi_thinking_level` (`BAMPI_THINKING_LEVEL`) | `off` | 思考强度：`off` / `minimal` / `low` / `medium` / `high` / `xhigh` |

更多 provider 行为见 [bampy providers](https://github.com/guh0613/bampy/blob/main/docs/providers.md)。

## 触发与限流

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_enabled` (`BAMPI_ENABLED`) | `true` | 总开关 |
| `bampi_trigger_prefix` (`BAMPI_TRIGGER_PREFIX`) | `["@bot"]` | 触发前缀列表 |
| `bampi_trigger_keywords` (`BAMPI_TRIGGER_KEYWORDS`) | `[]` | 触发关键词 |
| `bampi_group_whitelist` (`BAMPI_GROUP_WHITELIST`) | `[]` | 群白名单；空=不限制 |
| `bampi_random_reply_prob` (`BAMPI_RANDOM_REPLY_PROB`) | `0.0` | 无前缀/@ 时的随机接话概率（0–1） |
| `bampi_poke_reply_enabled` (`BAMPI_POKE_REPLY_ENABLED`) | `true` | 戳一戳是否触发回复 |
| `bampi_thinking_reaction_enabled` (`BAMPI_THINKING_REACTION_ENABLED`) | `true` | 新消息轮次处理期间是否给触发消息贴临时思考表情 |
| `bampi_thinking_reaction_emoji` (`BAMPI_THINKING_REACTION_EMOJI`) | `仔细分析` | 临时思考表情；默认使用 QQ 原生表情 ID 314，也可配置其他表情名或 QQ 表情 ID |
| `bampi_rate_limit` (`BAMPI_RATE_LIMIT`) | `30` | 限流次数 |
| `bampi_rate_limit_window_seconds` (`BAMPI_RATE_LIMIT_WINDOW_SECONDS`) | `60` | 限流窗口（秒） |
| `bampi_qq_react_tool_enabled` (`BAMPI_QQ_REACT_TOOL_ENABLED`) | `true` | 是否启用贴表情/戳一戳工具 |
| `bampi_reaction_context_enabled` (`BAMPI_REACTION_CONTEXT_ENABLED`) | `true` | 是否把表情回应纳入上下文缓冲 |

## Pokémon Showdown 对战

该插件使用独立的 `SHOWDOWN_BATTLE_*` 配置；完整说明见 [showdown-battle.md](showdown-battle.md)。

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `showdown_battle_enabled` (`SHOWDOWN_BATTLE_ENABLED`) | `true` | 对战插件总开关 |
| `showdown_battle_node_bin` (`SHOWDOWN_BATTLE_NODE_BIN`) | 自动查找 `node` | Node.js 可执行文件 |
| `showdown_battle_package_dir` (`SHOWDOWN_BATTLE_PACKAGE_DIR`) | `node_modules/pokemon-showdown` | 锁定版本的 Showdown npm 包目录 |
| `showdown_battle_data_dir` (`SHOWDOWN_BATTLE_DATA_DIR`) | `data/showdown_battle` | 队伍仓库和立绘缓存目录 |
| `showdown_battle_i18n_file` (`SHOWDOWN_BATTLE_I18N_FILE`) | 插件内 `zh_hans.json` | 生成后的分类翻译目录 |
| `showdown_battle_group_whitelist` (`SHOWDOWN_BATTLE_GROUP_WHITELIST`) | `[]` | 对战群白名单；空=不限制 |
| `showdown_battle_sprite_download_timeout_seconds` | `6` | 下载立绘的超时 |
| `showdown_battle_max_render_concurrency` | `2` | 图片渲染线程并发上限 |
| `showdown_battle_team_source_timeout_seconds` | `10` | 在线队伍与推荐配招请求超时 |
| `showdown_battle_team_source_max_bytes` | `1048576` | 在线队伍数据的响应大小上限 |
| `showdown_battle_team_source_cache_ttl_seconds` | `900` | 样例队伍和配招数据缓存时间 |
| `showdown_battle_team_guide_ttl_seconds` | `900` | 私聊组队向导的空闲超时 |
| `showdown_battle_ai_enabled` (`SHOWDOWN_BATTLE_AI_ENABLED`) | `true` | 是否允许通过 `<规则命令> <BAMPI_BOT_NAME>` 挑战聊天 Bot；旧 `bot` 参数仅作兼容 |
| `showdown_battle_ai_model_provider` / `model_id` | 空 | 空时继承 `bampi_chat` 主模型；非空时独立覆盖 |
| `showdown_battle_ai_model_api` | `auto` | 对战模型 API；未覆盖模型身份时继承主模型的显式 API |
| `showdown_battle_ai_api_key` / `base_url` | 空 | 空时继承主模型配置，之后再按 provider/API 查找环境变量 |
| `showdown_battle_ai_thinking_level` | 空 | 空时继承 `BAMPI_THINKING_LEVEL` |
| `showdown_battle_ai_decision_timeout_seconds` | `60` | 单次 Bot 决策超时，超时后执行本地默认行动 |
| `showdown_battle_ai_max_output_tokens` | `2048` | 对战模型单次决策最大输出 token |
| `showdown_battle_ai_max_attempts` | `2` | 模型未调用行动工具时的最大决策轮数（含首次） |
| `showdown_battle_ai_commentary_enabled` | `true` | 是否把 AI 的自然语言对战发言延迟发送到群里 |
| `showdown_battle_ai_commentary_max_chars` | `500` | 单次 AI 对战发言最大字符数 |
| `showdown_battle_ai_persona` | 空 | 空时继承 `BAMPI_PERSONA` 和 `BAMPI_BOT_NAME`；设置后仅覆盖对战人格 |
| `showdown_battle_ai_public_history_events` | `80` | 每轮自动上下文附带的近期公开事件上限 |

## 合并转发

Bot 只在消息已经触发回复后按需调用 OneBot `get_forward_msg`，无需开启 NapCat 的全局 `parseMultMsg`。当前消息和回复引用中的合并转发都会被展开；嵌套内容保留发送者、时间、消息顺序及可读取的图片/文件。超长转录会截断预览并把完整文本保存到群 workspace 的 `inbox/`。

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_forward_enabled` (`BAMPI_FORWARD_ENABLED`) | `true` | 是否展开 QQ 合并转发消息 |
| `bampi_forward_max_depth` (`BAMPI_FORWARD_MAX_DEPTH`) | `4` | 最大嵌套深度 |
| `bampi_forward_max_nodes` (`BAMPI_FORWARD_MAX_NODES`) | `100` | 单轮最多解析的转发节点总数 |
| `bampi_forward_max_roots` (`BAMPI_FORWARD_MAX_ROOTS`) | `4` | 当前消息和回复引用合计最多处理多少个顶层转发 |
| `bampi_forward_max_api_calls` (`BAMPI_FORWARD_MAX_API_CALLS`) | `8` | 单轮最多调用多少次 `get_forward_msg` |
| `bampi_forward_max_text_chars` (`BAMPI_FORWARD_MAX_TEXT_CHARS`) | `30000` | 当前消息与回复引用合计注入模型的转录预览字符上限；完整内容会另存附件 |
| `bampi_forward_resolve_timeout_seconds` (`BAMPI_FORWARD_RESOLVE_TIMEOUT_SECONDS`) | `10` | 单次读取转发消息的超时 |
| `bampi_forward_max_media_items` (`BAMPI_FORWARD_MAX_MEDIA_ITEMS`) | `8` | 单轮最多下载的转发内图片/文件数量；`0` 表示不下载 |
| `bampi_forward_max_total_media_bytes` (`BAMPI_FORWARD_MAX_TOTAL_MEDIA_BYTES`) | `31457280` | 转发内媒体累计下载上限（默认 30MB） |

NapCat 某些转发解析路径无法恢复真实来源群，因此实现不会使用子消息返回的 `group_id` 查询群成员或文件；转发内文件只有在消息段自带可下载 URL 时才会下载。

## 会话与回复

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_max_turns` (`BAMPI_MAX_TURNS`) | `50` | 单轮 Agent 最大 tool 循环次数量级 |
| `bampi_session_idle_ttl_seconds` (`BAMPI_SESSION_IDLE_TTL_SECONDS`) | `1800` | 会话空闲 TTL（秒） |
| `bampi_workspace_dir` (`BAMPI_WORKSPACE_DIR`) | `data/bampi/workspace` | workspace 根目录 |
| `bampi_session_dir` (`BAMPI_SESSION_DIR`) | `data/bampi/sessions` | 会话持久化目录 |
| `bampi_bot_name` (`BAMPI_BOT_NAME`) | `Ophelia` | Bot 名字；用于默认人设，只想改名时用它（设置了 `bampi_persona` 时以 persona 文本为准） |
| `bampi_persona` (`BAMPI_PERSONA`) | 空 | 额外人设/系统提示片段 |
| `bampi_reply_with_quote` (`BAMPI_REPLY_WITH_QUOTE`) | `true` | 回复是否引用原消息 |
| `bampi_outbound_markup_enabled` (`BAMPI_OUTBOUND_MARKUP_ENABLED`) | `true` | 是否解析 Assistant 回复中的 `@QQ号` / `@昵称(QQ号)` / `[表情:名称]` / `[名称]` 为真实消息段 |
| `bampi_outbound_at_all_enabled` (`BAMPI_OUTBOUND_AT_ALL_ENABLED`) | `false` | 是否允许 `@全体成员` 转为 at all |
| `bampi_outbound_at_limit` (`BAMPI_OUTBOUND_AT_LIMIT`) | `5` | 单条回复最多解析多少个 at；超出保留为文本，`0` 表示不限制 |
| `bampi_live_progress_enabled` (`BAMPI_LIVE_PROGRESS_ENABLED`) | `true` | 工具进度提示 |
| `bampi_live_progress_max_tool_updates` (`BAMPI_LIVE_PROGRESS_MAX_TOOL_UPDATES`) | `0` | 进度更新条数上限；`0` 表示不限制 |
| `bampi_live_text_stream_enabled` (`BAMPI_LIVE_TEXT_STREAM_ENABLED`) | `true` | 是否在 tool call 前发送并渲染该轮完整的中间文字；最终回复始终等待完整生成并统一渲染 |
| `bampi_live_text_stream_min_chars` (`BAMPI_LIVE_TEXT_STREAM_MIN_CHARS`) | `80` | 兼容保留的增量流式阈值；当前完整中间消息发送不使用 |
| `bampi_live_text_stream_force_chars` (`BAMPI_LIVE_TEXT_STREAM_FORCE_CHARS`) | `220` | 兼容保留的增量流式阈值；当前完整中间消息发送不使用 |
| `bampi_live_text_stream_min_interval_seconds` (`BAMPI_LIVE_TEXT_STREAM_MIN_INTERVAL_SECONDS`) | `1.2` | 兼容保留的增量流式间隔；当前完整中间消息发送不使用 |
| `bampi_threshold_compaction_notice_enabled` (`BAMPI_THRESHOLD_COMPACTION_NOTICE_ENABLED`) | `true` | 自动压缩开始时是否提示 |
| `bampi_workspace_cleanup_enabled` (`BAMPI_WORKSPACE_CLEANUP_ENABLED`) | `true` | 是否定期清理过期群 workspace |
| `bampi_workspace_cleanup_ttl_seconds` (`BAMPI_WORKSPACE_CLEANUP_TTL_SECONDS`) | `259200` | 清理 TTL（默认 3 天） |

出站标记只解析 Assistant 正文，工具进度和系统提示始终按纯文本发送。Markdown 行内代码、围栏代码块、链接和图片中的标记也按原文发送。需要原样显示其他标记时，可写成 `\@123456` 或 `\[doge]`。

更细的 live progress 召回时间等，见源码 `config.py`。

## Bash / 沙箱

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_bash_mode` (`BAMPI_BASH_MODE`) | `docker` | `docker` / `local` / `auto` |
| `bampi_bash_container_name` (`BAMPI_BASH_CONTAINER_NAME`) | `bampi-sandbox` | 沙箱容器名 |
| `bampi_bash_container_workdir` (`BAMPI_BASH_CONTAINER_WORKDIR`) | `/workspace` | 容器内工作目录（绝对路径） |
| `bampi_bash_container_shell` (`BAMPI_BASH_CONTAINER_SHELL`) | `/bin/bash` | 容器内 shell |
| `bampi_bash_timeout` (`BAMPI_BASH_TIMEOUT`) | `30` | 默认命令超时（秒） |

## Web

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_web_search_timeout` (`BAMPI_WEB_SEARCH_TIMEOUT`) | `15` | web ask 超时（秒） |
| `bampi_web_search_model` (`BAMPI_WEB_SEARCH_MODEL`) | `grok-4.20-fast` | web ask 所用模型 |
| `bampi_web_search_base_url` (`BAMPI_WEB_SEARCH_BASE_URL`) | 空 | web ask API Base |
| `bampi_web_search_api_key` (`BAMPI_WEB_SEARCH_API_KEY`) | 空 | web ask API Key |
| `bampi_web_search_exa_api_key` (`BAMPI_WEB_SEARCH_EXA_API_KEY`) | 空 | Exa 检索 Key（web search） |
| `bampi_web_search_exa_timeout` (`BAMPI_WEB_SEARCH_EXA_TIMEOUT`) | `20` | Exa 超时（秒） |
| `bampi_web_search_exa_num_results` (`BAMPI_WEB_SEARCH_EXA_NUM_RESULTS`) | `5` | Exa 默认结果数 |

## Browser

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_browser_enabled` (`BAMPI_BROWSER_ENABLED`) | `true` | 是否启用浏览器工具 |
| `bampi_browser_executable_path` (`BAMPI_BROWSER_EXECUTABLE_PATH`) | 空 | Chromium 可执行文件路径；空则自动探测/安装 |
| `bampi_browser_auto_install` (`BAMPI_BROWSER_AUTO_INSTALL`) | `true` | 缺失时是否自动安装 |
| `bampi_browser_headless` (`BAMPI_BROWSER_HEADLESS`) | `true` | 无头模式 |
| `bampi_browser_action_timeout` (`BAMPI_BROWSER_ACTION_TIMEOUT`) | `20` | 单步操作超时（秒） |
| `bampi_browser_idle_ttl_seconds` (`BAMPI_BROWSER_IDLE_TTL_SECONDS`) | `300` | 空闲浏览器回收 TTL |
| `bampi_browser_max_pages` (`BAMPI_BROWSER_MAX_PAGES`) | `6` | 同时打开页面上限 |
| `bampi_browser_allow_private_network` (`BAMPI_BROWSER_ALLOW_PRIVATE_NETWORK`) | `false` | 是否允许访问私网地址 |

视口、录制、batch 上限等偏运维微调项见 `config.py`。

## Service

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_service_enabled` (`BAMPI_SERVICE_ENABLED`) | `true` | 是否启用 service 工具（需 `bash_mode=docker`） |
| `bampi_service_port_range` (`BAMPI_SERVICE_PORT_RANGE`) | `46000-46031` | 端口池 |
| `bampi_service_public_host` (`BAMPI_SERVICE_PUBLIC_HOST`) | `127.0.0.1` | 对外可达主机名/IP |
| `bampi_service_startup_timeout` (`BAMPI_SERVICE_STARTUP_TIMEOUT`) | `20` | 启动就绪等待（秒） |
| `bampi_service_max_active_services_per_group` (`BAMPI_SERVICE_MAX_ACTIVE_SERVICES_PER_GROUP`) | `4` | 每群同时活跃服务上限 |

## Schedule

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_schedule_enabled` (`BAMPI_SCHEDULE_ENABLED`) | `true` | 是否启用定时任务 |
| `bampi_schedule_dir` (`BAMPI_SCHEDULE_DIR`) | `data/bampi/schedules` | 任务持久化目录 |
| `bampi_schedule_timezone` (`BAMPI_SCHEDULE_TIMEZONE`) | `Asia/Shanghai` | 时区 |
| `bampi_schedule_max_active_tasks_per_group` (`BAMPI_SCHEDULE_MAX_ACTIVE_TASKS_PER_GROUP`) | `20` | 每群活跃任务上限 |
| `bampi_schedule_tick_seconds` (`BAMPI_SCHEDULE_TICK_SECONDS`) | `15` | 调度轮询间隔（秒） |

## Memory

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_memory_enabled` (`BAMPI_MEMORY_ENABLED`) | `true` | 是否启用记忆 |
| `bampi_memory_db_path` (`BAMPI_MEMORY_DB_PATH`) | `data/bampi/memory.db` | SQLite 路径 |
| `bampi_memory_archive_retention_days` (`BAMPI_MEMORY_ARCHIVE_RETENTION_DAYS`) | `365` | 归档保留天数 |
| `bampi_memory_archive_llm_summary` (`BAMPI_MEMORY_ARCHIVE_LLM_SUMMARY`) | `true` | 归档是否用 LLM 摘要 |
| `bampi_memory_search_max_results` (`BAMPI_MEMORY_SEARCH_MAX_RESULTS`) | `10` | 检索条数上限 |
| `bampi_memory_embedding_enabled` (`BAMPI_MEMORY_EMBEDDING_ENABLED`) | `false` | 是否启用向量检索 |
| `bampi_memory_model_provider` / `bampi_memory_model_id` / `bampi_memory_api_key` 等 | 空 / `auto` | 记忆专用模型；空则回落到主模型配置 |
| `bampi_memory_profile_cron` (`BAMPI_MEMORY_PROFILE_CRON`) | `0 4 * * *` | 用户画像刷新 cron |

embedding、摘要 token、profile 阈值等细节见 `config.py`。

## 文件上传

| 字段（env） | 默认 | 说明 |
|-------------|------|------|
| `bampi_max_inline_image_size` (`BAMPI_MAX_INLINE_IMAGE_SIZE`) | `5242880` | 内联图片大小上限（字节，默认 5MB） |
| `bampi_max_download_size` (`BAMPI_MAX_DOWNLOAD_SIZE`) | `52428800` | 下载/入站文件大小上限（字节，默认 50MB） |
| `bampi_group_file_upload_host_dir` (`BAMPI_GROUP_FILE_UPLOAD_HOST_DIR`) | `app/.config/QQ/temp` | 群文件上传暂存（宿主机侧路径，对接 NapCat） |
| `bampi_group_file_upload_container_dir` (`BAMPI_GROUP_FILE_UPLOAD_CONTAINER_DIR`) | `/app/.config/QQ/temp` | 对应容器内路径 |

日常用法（触发、命令、Skills、workspace）见 [使用说明](usage.md)。
