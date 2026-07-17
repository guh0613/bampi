# 快速开始

把 Bampi 跑到「能在群里回一句」为止。配置细节见 [configuration.md](configuration.md)，日常用法见 [usage.md](usage.md)。

## 1. 环境要求

| 项 | 要求 |
|----|------|
| Python | ≥ 3.12 |
| 包管理 | [uv](https://docs.astral.sh/uv/) |
| Docker | 用于 NapCat 与 `bampi-sandbox` |
| QQ 协议端 | NapCat（本仓库 `docker compose` 提供） |

## 2. 克隆与依赖

```bash
git clone <本仓库地址>
cd bampi
uv sync
```

`bampy` 是本仓库的 **uv workspace member**（见根目录 `pyproject.toml`），`uv sync` 会一并以 editable 方式装好，无需单独安装。

## 3. 配置 `.env`

从示例复制：

```bash
cp .env.example .env
```

下面是一组**最小必填**示例（占位符，勿提交真实密钥）。NoneBot / OneBot 侧变量与 `BAMPI_*` 均可写在同一 `.env`：

```env
DRIVER=~fastapi
HOST=0.0.0.0
PORT=8080

# 若 NapCat 侧启用了 access token，必须与这里一致；未启用可留空或省略
ACCESS_TOKEN=your_shared_token

BAMPI_MODEL_PROVIDER=openai
BAMPI_MODEL_ID=gpt-5-mini
BAMPI_API_KEY=sk-your-key
BAMPI_BASE_URL=https://api.openai.com/v1
```

说明：

- `DRIVER` / `HOST` / `PORT`：NoneBot 监听地址；NapCat 的反向 WS / HTTP 应对准该 `HOST:PORT`。
- `ACCESS_TOKEN`：对接鉴权用；**仅在协议端要求时必填**，两端取值必须一致。
- `BAMPI_MODEL_*` / `BAMPI_API_KEY` / `BAMPI_BASE_URL`：主对话模型；`BASE_URL` 按你实际使用的 OpenAI 兼容或其它端点填写。
- 白名单、触发前缀、沙箱、记忆等均可后续再调，完整项见 [configuration.md](configuration.md)。

## 4. Docker：NapCat 与沙箱

在项目根目录：

```bash
docker compose up -d
```

会拉起两个主要服务：

| 服务 | 作用 |
|------|------|
| `napcat2` | QQ 协议端（OneBot v11），把 QQ 事件转给 NoneBot |
| `bampi-sandbox` | 工具默认执行环境；Bot 在宿主机，bash 等进容器 |

workspace 挂载（宿主机 → 容器）：

```text
./data/bampi/workspace  →  /workspace
```

群会话读写的 inbox / outbox、Skills 同步目录等都落在该 workspace 下。首次启动后若目录不存在，compose / 沙箱会按需创建相关路径。

## 5. NapCat ↔ NoneBot 对接要点

只记原则，不依赖具体 NapCat 配置文件名：

1. 协议使用 **OneBot v11**（WebSocket 和/或 HTTP，按你在 NapCat 里启用的连接方式）。
2. 若启用鉴权，**ACCESS_TOKEN 与 `.env` 中一致**。
3. Bot 在宿主机监听 `HOST:PORT`（示例为 `0.0.0.0:8080`）。
4. 容器内 NapCat 访问宿主机 Bot 时，常用 `host.docker.internal`（本仓库 compose 已配置 `extra_hosts`）；若无效，改用宿主机局域网 IP。
5. 方向要对齐：常见是 NapCat **反向**连到 NoneBot，或按你选用的正向/反向组合配置 URL。

改完后重启对应容器或重载 NapCat 连接，再启动 Bot。

## 6. 启动 Bot

```bash
uv run python bot.py
```

日志里应能看到 OneBot V11 适配器与监听地址。确认 NapCat 已登录 QQ，且与 Bot 的 WS/HTTP 连接状态正常。

## 7. 群内冒烟

1. 把 Bot 拉进测试群（若配置了 `BAMPI_GROUP_WHITELIST`，需包含该群号）。
2. 用触发前缀试一句。默认配置里有 `@bot` 一类前缀，例如：

   ```text
   @bot 你好
   ```

3. 能收到回复即冒烟通过。触发方式、命令与 Skills 用法见 [usage.md](usage.md)。

Agent 框架本身的设计与 API 见 [bampy README](https://github.com/guh0613/bampy) 及其 [docs](https://github.com/guh0613/bampy/blob/main/docs/getting-started.md)。

## 8. 常见问题

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 工具/bash 失败、提示容器不可用 | 沙箱未起或容器名不一致 | `docker compose ps`，确认 `bampi-sandbox` 健康；默认容器名与 `BAMPI_BASH_CONTAINER_NAME` 一致 |
| 模型报错 / 401 | API Key 或 Base URL 错误 | 检查 `BAMPI_API_KEY`、`BAMPI_BASE_URL`、`BAMPI_MODEL_PROVIDER` / `ID` |
| 群里完全不回复 | 未触发，或群不在白名单 | 使用配置的前缀（默认 `@bot`）；检查 `BAMPI_GROUP_WHITELIST`（空列表通常表示不限制，非空则必须命中） |
| NapCat 连不上 Bot | 地址/Token/端口不对 | 核对 `HOST`/`PORT`、`ACCESS_TOKEN`，以及容器侧是否指向 `host.docker.internal` 或宿主机 IP |
| 权限 / 无法发消息 | Bot 非群成员或 QQ 侧权限不足 | 确认已入群、可发言；管理员相关能力视群设置而定 |

更多配置项说明见 [configuration.md](configuration.md)。
