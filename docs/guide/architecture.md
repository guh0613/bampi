# 架构概览

Bampi 把 [bampy](https://github.com/guh0613/bampy) Agent 接到 QQ 群聊：协议走 OneBot v11（NapCat），Bot 进程在宿主机，工具默认进入 `bampi-sandbox` 容器。本文只讲 **Bot 侧** 如何接线；Agent loop、工具协议、会话持久化等框架能力见 [bampy 架构](https://github.com/guh0613/bampy/blob/main/docs/architecture.md)。

## 总览

```text
QQ 群聊
  ↕
NapCat（Docker Compose，OneBot v11）
  ↕
NoneBot2（bot.py + bampi_chat 插件）
  ↕
bampy AgentSession（每群一个 session）
  ├── LLM Provider
  ├── Docker sandbox（bash / 文件工具）
  └── SQLite memory（可选）
```

| 层 | 职责 |
|----|------|
| NapCat | 登录 QQ、转发 OneBot 事件与 API |
| NoneBot2 | 接收事件、调用 QQ API、发进度/回复消息 |
| `bampi_chat` | 触发判定、群会话池、工具装配、记忆与定时、Skills 同步 |
| bampy | Agent loop、流式事件、工具执行、会话 NDJSON（详见 [bampy 架构](https://github.com/guh0613/bampy/blob/main/docs/architecture.md)） |
| sandbox | `bampi-sandbox` 容器；宿主机 `data/bampi/workspace` 挂到容器 `/workspace` |
| memory | 可选长期记忆；默认 `data/bampi/memory.db` |

## 组件职责

### NoneBot

- 入口：`bot.py` 注册 OneBot v11 适配器，并从 `pyproject.toml` 加载 `bampi/plugins`。
- 负责事件分发与出站消息（文本、图片、文件、表情回应等）。
- 进度提示、流式片段、失败回退等「对用户可见的反馈」由插件通过 Bot API 发送。

### `bampi_chat`

插件在启动时读取 `BampiChatConfig`，创建 `GroupSessionManager` / `ScheduleManager`，并注册消息与通知处理器。

主要职责：

- **触发**：前缀、关键词、@、戳一戳等 → `should_respond`
- **会话池**：每群一个 `AgentSession`；空闲 TTL、并发预约、`/stop` 等
- **工具装配**：workspace 读写、沙箱 bash、检索/浏览器、记忆、定时、服务等
- **记忆 / 定时**：后台归档与 schedule 调度（可按配置关闭）
- **Skills**：启动或加载时把内置 skills 同步到 workspace，并向 Agent 暴露可按任务选用的 skill 描述

配置项说明见 [配置参考](configuration.md)。群侧用法见 [使用说明](usage.md)。

### bampy

Bot 侧只做「创建 session、注入工具与 system prompt、订阅事件、调用 `prompt`」。agent loop、compaction、扩展与 provider 细节不在本文展开，请看：

- [bampy 架构](https://github.com/guh0613/bampy/blob/main/docs/architecture.md)
- [AgentSession](https://github.com/guh0613/bampy/blob/main/docs/agent-session.md)
- [会话持久化](https://github.com/guh0613/bampy/blob/main/docs/session.md)

### Sandbox

默认 `bampi_bash_mode=docker`，命令在名为 `bampi-sandbox` 的容器内执行，工作目录对应群 workspace（容器内仍在 `/workspace` 树下）。Compose 定义见仓库根目录 `docker-compose.yml`。

### Memory

可选。开启后由插件侧 `MemoryManager` 管理 SQLite 存储，并向 agent 暴露记忆相关工具。路径默认 `data/bampi/memory.db`（见配置）。

## 数据目录约定

默认根目录在仓库下的 `data/bampi/`（均可由 `BAMPI_*` 配置覆盖）：

| 路径 | 用途 |
|------|------|
| `data/bampi/workspace/` | workspace 根；每群一个子目录，内含 `inbox` / `outbox` / `persistent` |
| 容器内 `/workspace` | 上述目录的挂载点 |
| `data/bampi/sessions/` | 群会话 NDJSON（bampy 会话持久化） |
| `data/bampi/memory.db` | 记忆库（启用时） |
| `data/bampi/schedules/` | 定时任务运行时数据（启用时） |

约定用语：

- **workspace**：群工作区；用户文件进 **inbox**，待发文件放 **outbox**
- **session**：每群一个 `AgentSession`
- **sandbox**：`bampi-sandbox` 容器

## 请求生命周期（简短）

1. 群消息进入 NoneBot → `bampi_chat` handler。
2. `should_respond` 判断是否应答（白名单、前缀、关键词、概率等）。
3. 成功预占新轮次后给触发消息贴临时思考回应；解析媒体并把可下载文件写入该群 **inbox**，再构造用户消息内容。
4. `GroupSessionManager` 取/建该群 session，装配工具与 prompt，调用 bampy。
5. Agent 跑工具（常进 sandbox），流式事件驱动进度/文本片段。
6. 最终回复发回群聊；若 **outbox** 根目录有新文件，一并上传后清理，随后撤销临时思考回应。

失败评估、撤回进度、后台 bash 退出通知等属于插件层细节，见源码 `handler.py` / `feedback.py`。

## Skills（Bot 侧位置）

- 仓库可带 **内置 skills**：启动或加载时同步到 workspace 的内置镜像目录（概念上：builtin 源 → workspace 内镜像）。
- 也会加载运维侧预置在群 workspace 下的 skills（当前约定 `.agents/skills`；亦兼容旧路径）。
- QQ 群聊不提供 skill 列表、显式调用或安装命令；Agent 根据用户的自然语言任务与 skill 描述自行选择。
- Skill **内容与机制**（`SKILL.md`、发现规则等）见 [bampy Skills](https://github.com/guh0613/bampy/blob/main/docs/skills.md)。

公开文档**不**列举内置 skill 的具体步骤或目录清单。

## 与 bampy 的边界

| 本仓库（Bampi）文档 | bampy 文档 |
|---------------------|------------|
| 如何接 QQ、NapCat、插件配置、workspace/inbox/outbox | Agent loop、工具协议、provider、扩展、compaction |
| Bot 侧装配与运维路径 | 框架可复用能力 |

继续阅读：[开发指南](development.md) · [配置参考](configuration.md) · [快速开始](getting-started.md)
