# 使用说明

面向群用户与运营：如何触发对话、常用命令、Skills、进度/流式体验与 workspace。

部署与环境见 [快速开始](getting-started.md)；配置项见 [配置参考](configuration.md)。

## 如何触发对话

每群一个会话（session）。只有白名单内的群会响应（`bampi_group_whitelist` 为空表示不限制）。

满足任一条件即可触发：

| 方式 | 说明 |
|------|------|
| 前缀 | 消息以 `bampi_trigger_prefix` 中任一前缀开头（默认 `@bot`）；前缀会被剥掉再交给 Agent |
| 关键词 | 消息包含 `bampi_trigger_keywords` 中任一关键词 |
| @机器人 | `@` 机器人，或回复机器人的消息（`to_me` / 回复） |
| 戳一戳 | 默认开启（`bampi_poke_reply_enabled`）；戳机器人会触发一轮回复 |
| 随机回复 | `bampi_random_reply_prob` > 0 时，普通消息有概率被接住 |
| 显式 Skill | 消息最开头写 `/skill-name`（见下）也会触发 |

触发后同一群内通常串行处理；进行中时其他人需等待，或由发起者 `/stop`。

## 会话命令

在群里发送（整句匹配，忽略多余空白）：

| 命令 | 作用 | 权限 |
|------|------|------|
| `/stop` | 中止当前进行中的会话，并终止该群正在跑的后台任务 | **发起者**可停自己的会话/后台任务；NoneBot **超管**（`SUPERUSERS`）可强制停止他人会话 |
| `/clear` 或 `/new` | 清空本群对话上下文 | 无进行中会话、且无运行中后台任务时可用 |
| `/compact` | 手动压缩本群上下文（减少 token） | 仅 NoneBot **超管** |

说明：

- 没有进行中的会话/后台任务时发 `/stop`，会提示当前没有可停内容。
- 非发起者、又非超管发 `/stop`，会提示请让发起者操作。
- 压缩失败或无可压缩上下文时，会收到对应提示。

## Skills

Skill 是带 `SKILL.md` 的能力包。仓库可带**内置 skills**，启动时会同步到群 workspace；用户也可自行安装。机制细节见 [bampy Skills](https://github.com/guh0613/bampy/blob/main/docs/skills.md)。

### QQ 侧命令

| 命令 | 作用 |
|------|------|
| `/skills` 或 `/skill list` | 列出已安装 skill |
| `/skill` / `/skills`（无参数） | 同 list |
| `/skill help` | 显示帮助 |
| `/skill show <name>` | 查看某个 skill 简介 |
| `/skill install [--force]` | 发送或**引用** skill 文件（压缩包 / Markdown）后执行，安装到本群 workspace |
| `/skill install https://... [--force]` | 从 URL 安装；`-f` / `--force` 可覆盖已有同名 skill |

### 显式调用

在普通消息**最开头**写 `/skill-name`，例如：

```text
/code-review 看看这个文件
```

会优先按该 skill 处理本轮（具体行为由 skill 自身定义）。用 `/skills` 查看当前可用名称。

## 进度与流式

开启后（默认开），对话过程中你会看到：

- **Live progress**：工具调用、加载 skill、定时/服务等进度提示（短消息滚动更新），便于知道机器人在干活而不是卡住。
- **Live text stream**：较长回复会分片更新同一条消息，边生成边显示，而不是等全部写完才发。

阈值、字数阈值等细节见 [配置参考](configuration.md) 中「会话与回复」一组。上下文自动压缩开始时，也可能收到一句压缩提示（可关）。

## Workspace

默认根目录：`data/bampi/workspace`（容器内常见挂载为 `/workspace`）。

每个群有独立子目录，其下：

| 目录 | 含义 |
|------|------|
| `inbox/` | 用户发来的图片、文件等会落入此处，供 Agent 读取 |
| `outbox/` | Agent 产出、待发回群的文件放这里；回合结束后机器人会尝试发送 |
| `persistent/` | 长期保留目录；不会被定期清理，仅在需要长期复用时使用 |

注意：只有 **`outbox/` 根目录**下的新文件会自动发回群；子目录内的文件不会自动发送。

路径、清理周期等见 [配置参考](configuration.md)。

## 能力面（简述）

Agent 可通过工具完成（不写具体 API）：

| 能力 | 一句话 |
|------|--------|
| 读写文件 | 在群 workspace 内查找、读取、编辑、写入文件 |
| bash | 在沙箱（默认 `bampi-sandbox` 容器）里执行命令 |
| web | 联网检索 / 问答类查询 |
| browser | 无头浏览器打开页面、操作与截图等 |
| memory | 跨会话检索与管理群/用户相关记忆 |
| schedule | 创建与管理本群定时任务 |
| service | 在沙箱内拉起可对外访问的长驻服务（需 docker 模式） |
| qq_react | 对本轮消息贴表情，或戳一戳群成员 |

## 相关文档

- [快速开始](getting-started.md)
- [配置参考](configuration.md)
- [bampy Skills](https://github.com/guh0613/bampy/blob/main/docs/skills.md)
