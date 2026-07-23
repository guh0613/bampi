"""将 LLM 回复中的内联标记解析为 OneBot 消息段。

与 ``message_render`` 对称：入站把 at / face 渲染成文本标记，
出站再把相同语法解析回消息段。

支持的标记：

- ``@昵称(123456)`` / ``@123456`` → at 段
- ``@全体成员`` → at all（需配置开启）
- ``[表情:doge]`` / ``[表情#182]`` → face 段
- 裸 ``[doge]`` / ``[笑哭]``（命中表情名白名单时）→ face 段
- ``\\@123456`` / ``\\[doge]`` → 转义后按普通文本发送

未识别或超出护栏的标记原样保留为文本，绝不吞字。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nonebot.adapters.onebot.v11 import Message, MessageSegment

from .message_render import FACE_ID_BY_NAME, QQ_FACE_NAMES

# 精选高频表情（写入 prompt 白名单，同时用于裸 [名称] 容错命中）。
# 顺序大致按常用程度；名称必须存在于 QQ_FACE_NAMES。
CURATED_FACE_NAMES: tuple[str, ...] = (
    "doge",
    "笑哭",
    "捂脸",
    "赞",
    "贴贴",
    "裂开",
    "吃瓜",
    "666",
    "疑问",
    "大哭",
    "敬礼",
    "摸鱼",
    "比心",
    "微笑",
    "流泪",
    "偷笑",
    "得意",
    "害羞",
    "尴尬",
    "发怒",
    "调皮",
    "呲牙",
    "难过",
    "惊恐",
    "流汗",
    "憨笑",
    "拥抱",
    "爱心",
    "心碎",
    "玫瑰",
    "OK",
    "庆祝",
    "打call",
    "emm",
    "呵呵哒",
    "无奈",
    "斜眼笑",
    "大笑",
    "暗中观察",
    "牛啊",
    "喵喵",
    "菜狗",
    "崇拜",
    "耶",
    "尊嘟假嘟",
)

_CURATED_FACE_IDS: dict[str, int] = {}
for _name in CURATED_FACE_NAMES:
    _face_id = FACE_ID_BY_NAME.get(_name.lower())
    if _face_id is not None:
        _CURATED_FACE_IDS[_name.lower()] = _face_id

# @昵称(123456) 或 @123456；反斜杠转义的 @ 不参与匹配。
_AT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_\\])@(?:"
    r"(?P<named>[^\s@()（）]+)\((?P<named_qq>\d+)\)"
    r"|"
    r"(?P<bare_qq>\d+)(?![A-Za-z0-9_])"
    r"|"
    r"(?P<at_all>全体成员)(?!\w)"
    r")"
)

# [表情:名称] / [表情#id] / 裸 [名称]；反斜杠转义的 [ 不参与匹配。
_BRACKET_PATTERN = re.compile(r"(?<!\\)\[(?P<body>[^\]]+)\]")
_ESCAPED_MARKUP_PREFIX = re.compile(r"\\(?P<prefix>@|\[)")


@dataclass(frozen=True, slots=True)
class ComposeOptions:
    """出站标记解析选项。"""

    enabled: bool = True
    at_all_enabled: bool = False
    at_limit: int = 5


def format_curated_face_names_for_prompt() -> str:
    """把精选表情名拼成 prompt 用的逗号分隔列表。"""
    return "、".join(CURATED_FACE_NAMES)


def compose_outbound_message(
    text: str,
    *,
    options: ComposeOptions | None = None,
) -> Message:
    """把回复文本解析为 Message（text / at / face 段）。

    ``options.enabled=False`` 时整段作为纯文本返回。
    """
    opts = options or ComposeOptions()
    if not text:
        return Message()
    if not opts.enabled:
        return Message([MessageSegment.text(text)])

    message = Message()
    at_count = 0
    pos = 0
    while pos < len(text):
        at_match = _AT_PATTERN.search(text, pos)
        bracket_match = _BRACKET_PATTERN.search(text, pos)

        next_match: re.Match[str] | None = None
        kind: str | None = None
        if at_match and bracket_match:
            if at_match.start() <= bracket_match.start():
                next_match, kind = at_match, "at"
            else:
                next_match, kind = bracket_match, "bracket"
        elif at_match:
            next_match, kind = at_match, "at"
        elif bracket_match:
            next_match, kind = bracket_match, "bracket"

        if next_match is None or kind is None:
            _append_text(message, text[pos:])
            break

        if next_match.start() > pos:
            _append_text(message, text[pos : next_match.start()])

        if kind == "at":
            segment, consumed, used_at = _try_compose_at(next_match, opts, at_count)
            if used_at:
                at_count += 1
            if segment is not None:
                message.append(segment)
            else:
                _append_text(message, next_match.group(0))
            pos = next_match.end() if consumed else next_match.start() + 1
            continue

        segment = _try_compose_face(next_match.group("body"))
        if segment is not None:
            message.append(segment)
        else:
            _append_text(message, next_match.group(0))
        pos = next_match.end()

    return message


def append_composed_text(
    message: Message,
    text: str,
    *,
    options: ComposeOptions | None = None,
) -> Message:
    """把 ``text`` 解析后追加到已有 Message（就地修改并返回）。"""
    composed = compose_outbound_message(text, options=options)
    for segment in composed:
        message.append(segment)
    return message


def _append_text(message: Message, text: str) -> None:
    if not text:
        return
    text = _ESCAPED_MARKUP_PREFIX.sub(r"\g<prefix>", text)
    if message and message[-1].type == "text":
        message[-1].data["text"] = str(message[-1].data.get("text", "")) + text
        return
    message.append(MessageSegment.text(text))


def _try_compose_at(
    match: re.Match[str],
    options: ComposeOptions,
    at_count: int,
) -> tuple[MessageSegment | None, bool, bool]:
    """尝试把 at 匹配转为段。

    返回 ``(segment_or_None, consumed, used_at_slot)``：
    - consumed=False 表示应从下一字符重新扫描（当前未用到）
    - used_at_slot 表示是否占用了 at 上限名额
    """
    if match.group("at_all"):
        if not options.at_all_enabled:
            return None, True, False
        if at_count >= options.at_limit > 0:
            return None, True, False
        return MessageSegment.at("all"), True, True

    qq = match.group("named_qq") or match.group("bare_qq")
    if not qq:
        return None, True, False
    if options.at_limit > 0 and at_count >= options.at_limit:
        return None, True, False
    return MessageSegment.at(int(qq)), True, True


def _try_compose_face(body: str) -> MessageSegment | None:
    text = body.strip()
    if not text:
        return None

    for prefix in ("表情:", "表情："):
        if text.startswith(prefix):
            name = text[len(prefix) :].strip()
            face_id = _resolve_face_id_by_name(name)
            if face_id is not None:
                return MessageSegment.face(face_id)
            return None

    for prefix in ("表情#", "表情＃"):
        if text.startswith(prefix):
            raw_id = text[len(prefix) :].strip()
            if raw_id.isdigit():
                face_id = int(raw_id)
                if face_id in QQ_FACE_NAMES:
                    return MessageSegment.face(face_id)
            return None

    # 裸 [doge] / [笑哭]：仅命中精选白名单，避免把任意方括号文案当表情
    face_id = _CURATED_FACE_IDS.get(text.lower())
    if face_id is not None:
        return MessageSegment.face(face_id)
    return None


def _resolve_face_id_by_name(name: str) -> int | None:
    if not name:
        return None
    # 正式语法 [表情:名称] 允许全表；裸括号只走白名单
    return FACE_ID_BY_NAME.get(name.lower())


def compose_options_from_config(config: object) -> ComposeOptions:
    """从 BampiChatConfig（或兼容对象）提取 ComposeOptions。"""
    return ComposeOptions(
        enabled=bool(getattr(config, "bampi_outbound_markup_enabled", True)),
        at_all_enabled=bool(getattr(config, "bampi_outbound_at_all_enabled", False)),
        at_limit=int(getattr(config, "bampi_outbound_at_limit", 5) or 0),
    )
