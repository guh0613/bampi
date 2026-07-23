"""将 LLM 回复中的内联标记解析为 OneBot 消息段。

与 ``message_render`` 对称：入站把 at / face 渲染成文本标记，
出站再把相同语法解析回消息段。

支持的标记：

- ``@昵称(123456)`` / ``@123456`` → at 段
- ``@全体成员`` → at all（需配置开启）
- ``[表情:doge]`` / ``[表情#182]`` → face 段
- 裸 ``[doge]`` / ``[笑哭]``（命中表情名白名单时）→ face 段
- ``\\@123456`` / ``\\[doge]`` → 转义后按普通文本发送

Markdown 行内代码、围栏代码块、行内链接和图片会作为受保护区间原样发送，
其中形似 at / face 的内容不参与解析。

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
    for protected_start, protected_end in _find_markdown_protected_ranges(text):
        if pos < protected_start:
            at_count = _compose_plain_range(
                message,
                text,
                start=pos,
                end=protected_start,
                options=opts,
                at_count=at_count,
            )
        _append_literal_text(message, text[protected_start:protected_end])
        pos = protected_end

    if pos < len(text):
        _compose_plain_range(
            message,
            text,
            start=pos,
            end=len(text),
            options=opts,
            at_count=at_count,
        )

    return message


def _compose_plain_range(
    message: Message,
    text: str,
    *,
    start: int,
    end: int,
    options: ComposeOptions,
    at_count: int,
) -> int:
    """解析 ``text[start:end]``，并返回累计使用的 at 数量。"""
    pos = start
    while pos < end:
        at_match = _AT_PATTERN.search(text, pos, end)
        bracket_match = _BRACKET_PATTERN.search(text, pos, end)

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
            _append_text(message, text[pos:end])
            break

        if next_match.start() > pos:
            _append_text(message, text[pos : next_match.start()])

        if kind == "at":
            segment, consumed, used_at = _try_compose_at(
                next_match,
                options,
                at_count,
            )
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

    return at_count


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
    _append_literal_text(message, text)


def _append_literal_text(message: Message, text: str) -> None:
    """追加不做转义处理的文本，并与相邻 text 段合并。"""
    if not text:
        return
    if message and message[-1].type == "text":
        message[-1].data["text"] = str(message[-1].data.get("text", "")) + text
        return
    message.append(MessageSegment.text(text))


def _find_markdown_protected_ranges(text: str) -> list[tuple[int, int]]:
    """线性扫描需要绕过 QQ 标记解析的 Markdown 区间。

    这里只识别会与本模块语法发生实际冲突的结构，不承担完整 Markdown
    渲染职责：行内代码、反引号/波浪号围栏代码块，以及 ``[label](target)``
    / ``![alt](target)``。未闭合的代码或链接目标保守地保护到文本末尾。
    """
    protected: list[tuple[int, int]] = []
    brackets: list[tuple[int, int]] = []
    length = len(text)
    line_start = 0
    pos = 0

    while pos < length:
        char = text[pos]

        if char == "\n":
            line_start = pos + 1
            pos += 1
            continue

        # Markdown 反斜杠转义：跳过被转义字符，避免将其识别为定界符。
        if char == "\\" and pos + 1 < length:
            if text[pos + 1] == "\n":
                line_start = pos + 2
            pos += 2
            continue

        if char in {"`", "~"}:
            fence_length = _markdown_fence_length(text, pos, line_start)
            if fence_length is not None:
                protected_end = _find_fenced_code_end(
                    text,
                    opening_start=pos,
                    marker=char,
                    opening_length=fence_length,
                )
                _add_protected_range(protected, pos, protected_end)
                if protected_end >= length:
                    break
                line_start = protected_end
                pos = protected_end
                continue

        if char == "`":
            run_length = _delimiter_run_length(text, pos, "`")
            protected_end = _find_inline_code_end(text, pos, run_length)
            if protected_end is None:
                _add_protected_range(protected, pos, length)
                break
            _add_protected_range(protected, pos, protected_end)
            last_newline = text.rfind("\n", pos, protected_end)
            if last_newline >= 0:
                line_start = last_newline + 1
            pos = protected_end
            continue

        if char == "[":
            protected_start = pos
            if (
                pos > 0
                and text[pos - 1] == "!"
                and not _is_backslash_escaped(text, pos - 1)
            ):
                protected_start -= 1
            brackets.append((pos, protected_start))
            pos += 1
            continue

        if char == "]" and brackets:
            _, protected_start = brackets.pop()
            if pos + 1 < length and text[pos + 1] == "(":
                protected_end = _find_parenthesized_end(text, pos + 1)
                if protected_end is None:
                    _add_protected_range(protected, protected_start, length)
                    break
                _add_protected_range(protected, protected_start, protected_end)
                last_newline = text.rfind("\n", pos, protected_end)
                if last_newline >= 0:
                    line_start = last_newline + 1
                pos = protected_end
                continue

        pos += 1

    return protected


def _markdown_fence_length(text: str, pos: int, line_start: int) -> int | None:
    """若 ``pos`` 是 CommonMark 风格围栏开头，返回定界符长度。"""
    indent_width = pos - line_start
    if indent_width > 3 or any(text[index] != " " for index in range(line_start, pos)):
        return None

    marker = text[pos]
    run_length = _delimiter_run_length(text, pos, marker)
    if run_length < 3:
        return None

    if marker == "`":
        line_end = text.find("\n", pos + run_length)
        if line_end < 0:
            line_end = len(text)
        if "`" in text[pos + run_length : line_end]:
            return None
    return run_length


def _find_fenced_code_end(
    text: str,
    *,
    opening_start: int,
    marker: str,
    opening_length: int,
) -> int:
    """返回围栏代码块结束位置；未闭合时返回文本末尾。"""
    opening_line_end = text.find("\n", opening_start + opening_length)
    if opening_line_end < 0:
        return len(text)

    line_start = opening_line_end + 1
    while line_start < len(text):
        line_end = text.find("\n", line_start)
        content_end = len(text) if line_end < 0 else line_end

        marker_start = line_start
        while marker_start < content_end and text[marker_start] == " ":
            marker_start += 1
        indent = marker_start - line_start

        if indent <= 3 and marker_start < content_end:
            run_length = _delimiter_run_length(text, marker_start, marker)
            trailing = text[marker_start + run_length : content_end]
            if run_length >= opening_length and not trailing.strip(" \t"):
                return len(text) if line_end < 0 else line_end + 1

        if line_end < 0:
            break
        line_start = line_end + 1

    return len(text)


def _find_inline_code_end(
    text: str, opening_start: int, opening_length: int
) -> int | None:
    """查找长度完全相同的行内反引号闭合定界符。"""
    pos = opening_start + opening_length
    while pos < len(text):
        candidate = text.find("`", pos)
        if candidate < 0:
            return None
        run_length = _delimiter_run_length(text, candidate, "`")
        if run_length == opening_length:
            return candidate + run_length
        pos = candidate + run_length
    return None


def _find_parenthesized_end(text: str, opening_start: int) -> int | None:
    """查找 Markdown 行内链接目标的配对右括号，支持嵌套和转义。"""
    depth = 1
    pos = opening_start + 1
    while pos < len(text):
        char = text[pos]
        if char == "\\" and pos + 1 < len(text):
            pos += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return pos + 1
        pos += 1
    return None


def _delimiter_run_length(text: str, start: int, marker: str) -> int:
    end = start
    while end < len(text) and text[end] == marker:
        end += 1
    return end - start


def _is_backslash_escaped(text: str, pos: int) -> bool:
    backslashes = 0
    cursor = pos - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _add_protected_range(
    ranges: list[tuple[int, int]],
    start: int,
    end: int,
) -> None:
    """按扫描顺序加入区间，并摊销 O(n) 合并嵌套/相邻区间。"""
    while ranges and ranges[-1][0] >= start:
        _, previous_end = ranges.pop()
        end = max(end, previous_end)
    if ranges and ranges[-1][1] >= start:
        previous_start, previous_end = ranges.pop()
        start = previous_start
        end = max(end, previous_end)
    ranges.append((start, end))


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
