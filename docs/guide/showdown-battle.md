# Pokémon Showdown 对战插件

插件目录：`bampi/plugins/showdown_battle/`。

## 运行依赖

插件通过项目根目录锁定的 `pokemon-showdown` npm 包运行本地模拟器：

```bash
npm ci --omit=optional --ignore-scripts
```

启动时会检查：

- Node.js 可执行文件；
- `pokemon-showdown` 包及 `dist` 构建产物；
- 翻译目录与 Showdown 版本一致；
- 所有已启用规则 ID 在当前引擎中存在。

检查失败时不会创建新对局，日志会记录具体原因。

## 当前规则与命令

| 规则 | 群内命令 |
|---|---|
| Gen9 单打 OU | `g9ou @对手` / `g9挑战 @对手` |
| Gen9 随机单打 | `g9随机 @对手` |
| Gen9 随机双打 | `g9双打随机 @对手` |
| Gen9 双打 OU | `g9双打 @对手` |
| Pokémon Champions VGC 2026 Reg M-B | `冠军对战 @对手` / `vgc2026 @对手` |

所有规则都支持把 `@对手` 换成当前 `BAMPI_BOT_NAME`，也可以使用 QQ 的真实 @。假设 `BAMPI_BOT_NAME=Ophelia`：

```text
g9ou Ophelia
g9随机 Ophelia
g9双打 Ophelia
冠军对战 Ophelia
```

旧参数 `bot`、`ai`、`机器人`、`人机`、`电脑` 仍作为兼容别名接受，但不会显示在帮助中。

自备队伍规则可在本局选定的操作频道发送以下命令；玩家对战固定为私聊，与 Bot 的群聊模式可直接在原群使用：

```text
导入队伍 <Showdown 队伍文本>
导入队伍 <PokePaste 或 crob.at 链接>
导入队伍 使用 <已保存队伍名>
```

### 队伍中心与统一帮助（推荐）

普通玩家只需记住两个入口：

```text
组队
宝可梦帮助
```

未参加对战时，`组队` 会打开数字菜单式队伍中心：

1. 创建新队伍：一键推荐、样例抄队、链接导入、自选成员、手动文本或逐只编辑；
2. 我的已保存队伍：分页查看，并可逐只编辑、查看完整文本、重命名、删除或复制；
3. 查询推荐配招；
4. 浏览在线样例；
5. 当前页面帮助。

发送 `我的队伍` 或不带参数的 `队伍管理` 可直接打开已保存队伍列表。选择一支队伍后的“编辑队伍成员与配置”会进入结构化编辑器；也可使用 `队伍管理 编辑 <规则> <队伍名>` 直接打开，或使用 `队伍管理 添加 <规则>` 从空队伍开始。编辑器支持推荐配招、粘贴单只 Showdown Export、手动添加、更换种类、道具、特性、招式、性格、EV/IV、太晶属性、昵称、等级等配置，以及复制、删除和调整成员顺序。Champions 规则会按 Stat Points 显示限制，并隐藏太晶选项。

编辑期间的所有改动只存在于内存草稿中；只有完整队伍通过本地 Showdown 规则校验后才会原子保存。退出含有未保存修改的草稿需要二次确认，保存时也会检测队伍是否在其他操作中被更新，避免静默覆盖较新的内容。向导内可随时发送 `返回`、`上一步`、`菜单`、`帮助`；发送 `0`、`取消` 或 `退出` 结束。删除队伍需要二次确认，列表每页最多显示六支队伍，空闲 15 分钟后向导自动失效。

若正在准备自备队伍对战，在本局操作频道发送 `组队` 会自动识别当前规则并直接显示“已保存队伍、一键推荐、链接、样例、自选成员”准备菜单。与 Bot 的群聊模式允许在原群完成准备，其他对战准备仍使用私聊；正式开战后不会再启动队伍工具。群内向导只接管参战玩家在当前群的后续回复，结束后恢复正常聊天。

`宝可梦帮助` 会根据空闲、准备中或对战中状态显示相应说明；也可指定主题：

```text
宝可梦帮助 组队
宝可梦帮助 对战
宝可梦帮助 对局
宝可梦帮助 高级
```

下面的完整命令仍保留给熟悉用法的玩家作为快捷入口，不需要普通用户记忆。

### 快速抄队与组队（快捷命令）

插件只读取明确允许的队伍源，不会抓取任意 URL：

- `pokepast.es`：读取 Showdown Export；
- `crob.at`：通过公开 API 读取队伍，多队伍链接使用 `#1`、`#2` 选择；
- `data.pkmn.cc`：读取 Smogon 样例队伍和推荐配招。

查看与直接使用当前规则下、且已通过本地 Showdown 校验的样例：

```text
队伍样例
队伍样例 gen9ou
队伍样例 gen9ou 1
导入队伍 样例 1
队伍管理 抄队 gen9ou 1 我的队伍
```

完全不想选配置时，可以直接获取当前统计数据生成的完整队伍；Champions Reg M-B 也支持：

```text
导入队伍 一键组队
队伍管理 一键组队 gen9championsvgc2026regmb 我的冠军队
```

按自己选择的宝可梦快速生成队伍：

```text
推荐配招 gen9ou 快龙
导入队伍 快速 快龙=2, 仆刀将军, 赛富豪, 铁武者, 藏玛然特, 古鼎鹿
队伍管理 快速 gen9ou 我的队伍 | 快龙=2, 仆刀将军, 赛富豪
```

`=2` 表示采用“推荐配招”列表中的第 2 套；不指定时采用第 1 套。自动组队只负责选择单体推荐配置并通过规则校验，不保证六只宝可梦之间具有完整的战术协同，保存后可用“队伍管理 查看”复制并微调。

目前可浏览的稳定样例和按宝可梦选择的推荐配招覆盖 Gen9 OU 与 Doubles OU；一键完整组队同时覆盖 Champions Reg M-B。Champions 更新更快，也可直接使用 PokePaste/crob.at 上的当前队伍。所有在线队伍都会由本地 Showdown 再次校验，不兼容当前规则的数据不会进入可选列表。

Champions Reg M-B 为双打六选四：队伍预览只选择四只，前两个编号是首发，后两个是替补顺序；这不是给六只全部排序。该规则使用 Mega 进化且每场最多一次，不使用太晶化。若导入源把成员写成 `Metagross-Mega` 之类的战斗中形态，插件会采用 Showdown 校验后的规范结果，将其还原为基础形态和合法基础特性；实际 Mega 进化需要在行动指令末尾添加 `mega`。这样可避免宝可梦在开局时错误地直接处于 Mega 形态。

部分 Champions 赛事链接公开的是 Open Team Sheet，只包含成员、道具、特性和招式，不包含隐藏的 Stat Points 与性格。导入这类队伍时，插件会为缺失配置补充 `Hardy Nature`，按合法的 0 Stat Points 保存，并明确发出警告；这只能恢复公开信息，无法还原选手的原始实数配置，正式对战前应自行调整。crob.at 链接若标注了规则，插件也会阻止将其误导入其他规则。

随机规则在希望用于本局操作的频道发送：

```text
对战准备
```

随机对战会在首次行动时显示完整己方队伍详情；每次普通行动提示也会列出当前可换入队友、队伍编号和 HP。行动命令包括：

```text
move 1
switch 3
forfeit
team 1234
```

双打可一次提交两个位置的行动：

```text
move1 1 1; move2 2 2
switch1 3; move2 1 1
```

特殊机制只在当回合提示可用时添加到对应的 `move` 指令末尾。例如 Champions 中使用 Mega 进化：

```text
move1 1 1 mega; move2 2 2
```

普通 Gen9 规则若提示可太晶化，则使用 `tera`；不要在 Champions 对局中使用 `tera`。

辅助命令：

```text
战况
check 精神强念
check 1
```

`check` 的招式信息除官方风格的"效果"说明外，还会附带一行"机制"，来自 Showdown 的竞技向描述（已尽量本地化），例如戏法空间会明确写出"接下来5回合，速度慢的宝可梦先行动"。

战报日志会像官方游戏一样标注血量变化的来源：非招式直接造成的伤害与回复（隐形岩、中毒、沙暴、反作用力、吃剩的东西、青草场地、特性接触伤害等）会写明原因，例如"皮卡丘 受到隐形岩的伤害，HP 132/200"、"皮卡丘 通过道具 吃剩的东西回复了体力"；异常状态与道具得失也会注明来源（火焰宝珠致灼伤、拍落造成的道具丢失、吃掉树果等）。

玩家对战只允许私聊准备和提交行动；群里的每次公开对局更新图（包括队伍预览）会同时私发给双方，因此玩家无需在私聊操作与群内战况之间反复切换。

战况图与状态面板在双方名字下方绘制精灵球指示条，标记本方队伍每个位置的存活、濒死与尚未登场状态（队伍总数来自公开的 `|teamsize|` 协议行，VGC 六选四时显示为选择后的 4 格）。在场宝可梦的名字旁还会以像素图标标注携带道具（图标裁自 Showdown 官方 item-icon 雪碧图并落盘缓存，索引取自本地模拟器的道具 `spritenum` 数据；图标不可用时回退为文字标注）：群内公开图只显示 Force Open Team Sheets 类规则通过 `|showteam|` 公开的配置——这类规则下队伍表全程可查，而对战事件中一次性亮出的道具与实机一样需要玩家自己记忆，不会画在图上；玩家对战中私发给每位玩家的镜像图会额外画出自己一方的道具，不会泄露给对手或群聊。与 Bot 对战时对手是机器人，公开图会直接显示玩家自己的道具，但 Bot 未公开的道具仍保持隐藏。私聊发送 `战况` 得到的状态面板同样包含自己一方的道具。

与 Bot 对战时，玩家在准备阶段选择整局使用的操作频道：在发起群发送 `对战准备` 会选择群聊模式，在私聊发送则选择私聊模式。群聊模式会把组队向导、完整己方队伍、招式、本回合操作提示、指令确认和超时提醒都留在原群，群成员可以看到这些信息；私聊模式会把上述私有提示留在私聊，并将群里的公开对局更新图同步到私聊。只有参战账号的消息会被路由，旁观者不能替玩家提交行动。

每场 Bot 对战拥有一个独立的、持续到对局结束的内存 AgentSession。每一轮会在原有对局历史后自动追加 Bot 一侧当前的 Showdown 私有 request、公开战况、已公开且持续保留的招式/道具/特性/太晶信息，以及有上限的近期公开事件。这样模型能像玩家一样维持整局战略，也能让服务商复用不断增长的稳定消息前缀；上下文接近窗口上限时由 AgentSession 自动压缩。会话不会写入聊天 session，也不会跨对局复用，因此不会混入 `bampi_chat` 群聊或另一场对战。

普通战况和当前可选行动已经自动进入每轮上下文，不需要模型为了读取已有信息而额外调用工具。系统还会根据当前 request 自动生成结构化 `legal_action_guide`，明确队伍预览、强制换人、单/双打动作、招式和换人编号、目标位置、特殊机制修饰符及组合命令，因此模型不需要预先了解 Bot 的命令方言。工具只用于按需查询完整 Showdown 招式机制，以及通过 `choose_battle_action` 提交 guide 允许的一条原始行动命令；后续可以按同一边界扩展宝可梦、道具或特性资料工具。人类一侧的私有 request 和原始群聊消息始终不会进入 Bot 对局会话。

对战人格默认继承 `BAMPI_PERSONA` 与 `BAMPI_BOT_NAME`，使公开发言保持和普通群聊一致；它不会读取或共享普通群聊 session 的历史。模型仅在有自然回应时才发言，不要求每回合硬凑台词；不发言时最终输出固定标记 `[[BATTLE_SILENT]]`，该标记会被代码丢弃。Bot 会等到该回合行动已经公开后再把普通发言发送到群里，避免提前暴露选招或隐藏信息。模型超时、没有调用行动工具或提交非法行动时，会回退到本地确定性默认行动，避免卡住 Showdown 进程。

非对战状态下，`组队`、`队伍管理`、`队伍样例` 和 `推荐配招` 均可在群内或私聊使用；群内操作内容对其他成员可见。玩家对战准备和进行期间，队伍相关命令仍只允许私聊。与 Bot 对战且已选择群聊模式时，准备阶段可以直接在原群使用 `组队`、组队向导或 `导入队伍`；选择群聊模式即表示接受队伍内容公开。对战正式开始后不再开放队伍管理命令：

```text
队伍管理 帮助
队伍管理 编辑 gen9ou 我的队伍
队伍管理 添加 gen9ou
队伍管理 高级帮助
```

## 配置

```dotenv
SHOWDOWN_BATTLE_ENABLED=true
SHOWDOWN_BATTLE_NODE_BIN=node
SHOWDOWN_BATTLE_PACKAGE_DIR=node_modules/pokemon-showdown
SHOWDOWN_BATTLE_DATA_DIR=data/showdown_battle
SHOWDOWN_BATTLE_GROUP_WHITELIST=[]
SHOWDOWN_BATTLE_TEAM_SOURCE_TIMEOUT_SECONDS=10
SHOWDOWN_BATTLE_TEAM_SOURCE_CACHE_TTL_SECONDS=900
SHOWDOWN_BATTLE_TEAM_GUIDE_TTL_SECONDS=900

# 对手名称使用 BAMPI_BOT_NAME；对战模型默认继承主模型配置
SHOWDOWN_BATTLE_AI_ENABLED=true
# SHOWDOWN_BATTLE_AI_MODEL_PROVIDER=openai
# SHOWDOWN_BATTLE_AI_MODEL_ID=gpt-5-mini
# SHOWDOWN_BATTLE_AI_MODEL_API=auto
# SHOWDOWN_BATTLE_AI_API_KEY=
# SHOWDOWN_BATTLE_AI_BASE_URL=
# SHOWDOWN_BATTLE_AI_THINKING_LEVEL=medium
SHOWDOWN_BATTLE_AI_DECISION_TIMEOUT_SECONDS=60
SHOWDOWN_BATTLE_AI_COMMENTARY_ENABLED=true
SHOWDOWN_BATTLE_AI_COMMENTARY_MAX_CHARS=500
# 留空时继承 BAMPI_PERSONA；仅在确实需要独立对战人格时覆盖
# SHOWDOWN_BATTLE_AI_PERSONA=
```

只要不设置 `SHOWDOWN_BATTLE_AI_MODEL_PROVIDER`、`SHOWDOWN_BATTLE_AI_MODEL_ID` 等覆盖项，对战模型会使用和 `bampi_chat` 主模型相同的 provider、model、API、base URL、API key 与 thinking level。覆盖了对战模型的 provider/model 身份但将 API 留为 `auto` 时，会根据其 provider 推断 API，不会错误继承主模型的 API 类型。

自备队伍规则下，Bot 会从该规则的一键组队来源取得队伍，并再次通过本地 Showdown 校验；随机规则由 Showdown 本身生成双方队伍。

队伍仓库和立绘缓存位于 `data/showdown_battle/`，该目录默认不进入 Git。

## 中文数据

运行时只读取已经生成的分类目录：

```text
bampi/plugins/showdown_battle/assets/i18n/zh_hans.json
```

翻译按以下优先级生成：

1. 本地人工覆盖 `overrides.zh_hans.json`；
2. `projectpokemon/champout` 的 Pokémon Champions 英文/简中游戏文本；
3. PokeAPI 多语言 CSV；
4. 可选的旧 PSChina 映射；
5. 上一次生成结果。

实体按类别隔离，因此 `Psychic` 可以同时表示招式“精神强念”和属性“超能力”，`Metronome` 也可以同时表示招式“挥指”和道具“节拍器”。

重新生成：

```bash
uv run python bampi/plugins/showdown_battle/i18n/generator.py
```

首次从旧插件迁移时可附加：

```bash
uv run python bampi/plugins/showdown_battle/i18n/generator.py \
  --legacy ~/projects/sanuki-qq/data/pokemon_cache/translations.js
```

生成器固定上游 commit 和 Showdown 版本，避免构建结果随网络数据静默变化。升级 Showdown 时应同时：

1. 更新 `package.json` 与 `package-lock.json`；
2. 更新生成器中的版本和上游 commit；
3. 重新生成翻译目录；
4. 检查缺失翻译统计；
5. 运行 `tests/test_showdown_battle.py`。

## 测试

```bash
uv run pytest -q tests/test_showdown_battle.py
```

测试覆盖规则注册、分类翻译、当前 Champions 术语、在线队伍安全导入、样例兼容性过滤、中文名快速组队、结构化队伍编辑与事务保存、并发更新冲突、队伍仓库迁移、队伍校验、招式数据、模拟器启动/退出、会话唯一性和双打超时默认行动。
