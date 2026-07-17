# 开发指南

面向在本仓库改 Bot / 插件逻辑的开发者。架构与数据流见 [架构概览](architecture.md)；配置项见 [配置参考](configuration.md)；Agent 框架本身见 [bampy 文档](../../bampy/docs/architecture.md)。

## 开发环境

- **Python ≥ 3.12**（见 `pyproject.toml` 的 `requires-python`）
- 用 **uv** 管理解释器与依赖
- 运行脚本、测试、Bot 一律使用虚拟环境或 `uv run`，**禁止裸 `python` / `pytest`**（否则容易缺依赖）

```bash
uv sync
uv run pytest
uv run python bot.py
```

本地从零部署（`.env`、Docker、NapCat）见 [快速开始](getting-started.md)。

## 仓库结构（顶层）

公开相关路径（其余内部目录不必当依赖）：

| 路径 | 说明 |
|------|------|
| `bot.py` | NoneBot 入口 |
| `bampi/plugins/` | 插件目录（`pyproject.toml` → `plugin_dirs`） |
| `bampi/plugins/bampi_chat/` | 群聊 Agent 插件 |
| `bampy/` | Agent 框架（uv workspace member） |
| `docker-compose.yml` / `docker/` | NapCat 与 sandbox 镜像 |
| `data/bampi/` | 运行时数据（workspace、sessions、memory、schedules） |
| `docs/guide/` | 本套公开文档 |
| `tests/` | Bot / 插件测试 |
| `pyproject.toml` / `uv.lock` | 依赖与 pytest / nonebot 配置 |

## 插件代码地图

`bampi/plugins/bampi_chat/` 主要模块（一句话）：

| 模块 | 职责 |
|------|------|
| `__init__.py` | 插件元数据、读取配置、启动/关闭 schedule 与 memory |
| `config.py` | `BampiChatConfig`（全部 `BAMPI_*` 配置） |
| `handler.py` | 消息/通知触发、`should_respond`、回复与 outbox 出站 |
| `session_manager.py` | 群会话池、创建 `AgentSession`、工具注入、空闲清理 |
| `skills.py` | 内置 skill 同步、安装、列表/显式调用解析 |
| `schedule_manager.py` | 定时任务持久化与调度 |
| `service_manager.py` | 沙箱内临时 HTTP 服务端口管理 |
| `memory/` | 记忆存储、检索、后台归档 |
| `tools/` | Bot 侧工具实现（workspace、bash、浏览器、记忆、定时等） |
| `prompt.py` / `message_render.py` / `feedback.py` | 系统提示、消息渲染、失败与进度文案 |

改行为时优先落在对应模块，配置统一走 `BampiChatConfig`，避免散落魔法常量。

## 本地跑起来

1. 按 [快速开始](getting-started.md) 配好 `.env`、启动 Compose（NapCat + sandbox）、安装依赖。
2. `uv run python bot.py`。
3. **热重载**：改插件代码或 `BAMPI_*` 配置后，NoneBot 进程通常需要**重启**才会生效；不要假设文件保存即生效。

## 测试

```bash
uv run pytest
```

- `pyproject.toml` 里 `addopts = "-m 'not live'"`：**默认跳过 live 测试**（不打真实 provider API）。
- 需要联真实模型时再显式加 live，例如：`uv run pytest -m live`（会消耗配额、依赖密钥）。
- 贡献前请跑测试。

测试路径包含 `tests/` 与 `bampy/tests/`。

## 依赖

```bash
uv add <package>
```

- Workspace member：**`bampy`**（与本仓库一并 `uv sync`）。

开发组依赖见 `[dependency-groups] dev`（含 `pytest`、`bampy[all-providers]` 等）。

## 文档维护

行为或配置面变更时，同步更新 `docs/guide/`（至少检查 [架构](architecture.md)、[配置](configuration.md)、[使用](usage.md)、本文）。Skills 机制细节仍以 [bampy Skills](../../bampy/docs/skills.md) 为准。

相关链接：[架构概览](architecture.md) · [配置参考](configuration.md) · [bampy README](../../bampy/README.md) · [bampy 架构](../../bampy/docs/architecture.md)
