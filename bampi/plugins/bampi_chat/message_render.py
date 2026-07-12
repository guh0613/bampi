"""将 OneBot v11 消息段渲染为 LLM 可读的纯文本。

`Message.extract_plain_text()` 只保留 `text` 段，QQ 特有的消息段
（@ 提及、QQ 表情、商城动画表情等）会被直接丢弃，导致模型看不到
这部分内容。本模块把它们渲染为文本标记：

- ``@昵称(123456)`` / ``@123456`` / ``@全体成员``
- ``[表情:doge]`` / ``[表情#123]``（未知表情 ID 回退为编号）
- ``[动画表情:名称]``、``[骰子:3]``、``[猜拳:2]``

图片、文件、回复等消息段由媒体收集逻辑单独处理，这里跳过。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Protocol

from nonebot import logger

NameResolver = Callable[[str], "str | None"]

# QQ 系统表情 ID -> 名称。NapCat 下发的 face 段通常带 raw.faceText，
# 优先使用；该表仅作为 faceText 缺失时的回退（经典表情表，与
# go-cqhttp / NapCat 前端使用的对照表一致）。
QQ_FACE_NAMES: dict[int, str] = {
    0: "惊讶", 1: "撇嘴", 2: "色", 3: "发呆", 4: "得意", 5: "流泪",
    6: "害羞", 7: "闭嘴", 8: "睡", 9: "大哭", 10: "尴尬", 11: "发怒",
    12: "调皮", 13: "呲牙", 14: "微笑", 15: "难过", 16: "酷", 18: "抓狂",
    19: "吐", 20: "偷笑", 21: "可爱", 22: "白眼", 23: "傲慢", 24: "饥饿",
    25: "困", 26: "惊恐", 27: "流汗", 28: "憨笑", 29: "悠闲", 30: "奋斗",
    31: "咒骂", 32: "疑问", 33: "嘘", 34: "晕", 35: "折磨", 36: "衰",
    37: "骷髅", 38: "敲打", 39: "再见", 41: "发抖", 42: "爱情", 43: "跳跳",
    46: "猪头", 49: "拥抱", 53: "蛋糕", 54: "闪电", 55: "炸弹", 56: "刀",
    57: "足球", 59: "便便", 60: "咖啡", 61: "饭", 63: "玫瑰", 64: "凋谢",
    66: "爱心", 67: "心碎", 69: "礼物", 74: "太阳", 75: "月亮", 76: "赞",
    77: "踩", 78: "握手", 79: "胜利", 85: "飞吻", 86: "怄火", 89: "西瓜",
    96: "冷汗", 97: "擦汗", 98: "抠鼻", 99: "鼓掌", 100: "糗大了",
    101: "坏笑", 102: "左哼哼", 103: "右哼哼", 104: "哈欠", 105: "鄙视",
    106: "委屈", 107: "快哭了", 108: "阴险", 109: "亲亲", 110: "吓",
    111: "可怜", 112: "菜刀", 113: "啤酒", 114: "篮球", 115: "乒乓",
    116: "示爱", 117: "瓢虫", 118: "抱拳", 119: "勾引", 120: "拳头",
    121: "差劲", 122: "爱你", 123: "NO", 124: "OK", 125: "转圈",
    126: "磕头", 127: "回头", 128: "跳绳", 129: "挥手", 130: "激动",
    131: "街舞", 132: "献吻", 133: "左太极", 134: "右太极", 136: "双喜",
    137: "鞭炮", 138: "灯笼", 140: "K歌", 144: "喝彩", 145: "祈祷",
    146: "爆筋", 147: "棒棒糖", 148: "喝奶", 151: "飞机", 158: "钞票",
    168: "药", 169: "手枪", 171: "茶", 172: "眨眼睛", 173: "泪奔",
    174: "无奈", 175: "卖萌", 176: "小纠结", 177: "喷血", 178: "斜眼笑",
    179: "doge", 180: "惊喜", 181: "骚扰", 182: "笑哭", 183: "我最美",
    185: "羊驼", 187: "幽灵", 188: "蛋", 190: "菊花", 192: "红包",
    193: "大笑", 194: "不开心", 197: "冷漠", 198: "呃", 199: "好棒",
    200: "拜托", 201: "点赞", 202: "无聊", 203: "托脸", 204: "吃",
    205: "送花", 206: "害怕", 207: "花痴", 208: "小样儿", 210: "飙泪",
    211: "我不看", 212: "托腮", 214: "啵啵", 215: "糊脸", 216: "拍头",
    217: "扯一扯", 218: "舔一舔", 219: "蹭一蹭", 220: "拽炸天",
    221: "顶呱呱", 222: "抱抱", 223: "暴击", 224: "开枪", 225: "撩一撩",
    226: "拍桌", 227: "拍手", 229: "干杯", 230: "嘲讽", 231: "哼",
    232: "佛系", 233: "掐一掐", 235: "颤抖", 237: "偷看", 238: "扇脸",
    239: "原谅", 240: "喷脸", 241: "生日快乐", 243: "甩头", 244: "扔狗",
    245: "加油必胜", 246: "加油抱抱", 247: "口罩护体", 260: "搬砖中",
    261: "忙到飞起", 262: "脑阔疼", 263: "沧桑", 264: "捂脸",
    265: "辣眼睛", 266: "哦哟", 267: "头秃", 268: "问号脸",
    269: "暗中观察", 270: "emm", 271: "吃瓜", 272: "呵呵哒",
    273: "我酸了", 277: "汪汪", 278: "汗", 281: "无眼笑", 282: "敬礼",
    283: "狂笑", 284: "面无表情", 285: "摸鱼", 286: "魔鬼笑", 287: "哦",
    288: "请", 289: "睁眼", 290: "敬茶", 293: "摸锦鲤", 294: "期待",
    297: "拜谢", 298: "元宝", 299: "牛啊", 300: "胖三斤", 301: "好闪",
    303: "右亲亲", 305: "右拍手", 306: "牛气冲天", 307: "喵喵",
    311: "打call", 312: "变形", 314: "仔细分析", 317: "菜狗",
    318: "崇拜", 319: "比心", 320: "庆祝", 322: "拒绝", 324: "吃糖",
    326: "生气", 332: "举牌牌", 333: "烟花", 334: "虎虎生威",
    336: "豹富", 338: "我想开了", 339: "舒适", 341: "打招呼",
    342: "酸Q", 343: "我方了", 344: "大怨种", 345: "红包多多",
    346: "你真棒棒", 347: "大展宏图", 349: "坚强", 350: "贴贴",
    351: "敲敲", 352: "咦", 353: "拜托", 354: "尊嘟假嘟", 355: "耶",
    356: "666", 357: "裂开",
}


# 反查表：表情名 -> ID（重名时保留 ID 较大的新表情，例如"拜托"）
_FACE_ID_BY_NAME: dict[str, int] = {name.lower(): face_id for face_id, name in QQ_FACE_NAMES.items()}

# Unicode 变体选择符/皮肤色修饰符，解析 emoji 字符时跳过
_EMOJI_MODIFIER_CODEPOINTS = frozenset(
    {0xFE0E, 0xFE0F, 0x200D, 0x1F3FB, 0x1F3FC, 0x1F3FD, 0x1F3FE, 0x1F3FF}
)


def _is_emoji_codepoint(codepoint: int) -> bool:
    """判断码点是否落在常见 emoji 区段（排除 CJK 等普通文字）。"""
    return 0x1F000 <= codepoint <= 0x1FAFF or 0x2190 <= codepoint <= 0x2BFF


def resolve_reaction_emoji_id(value: Any) -> int | None:
    """把模型给出的表情描述解析为贴表情用的 emoji_id。

    支持：表情名（如 `赞`、`doge`、`666`）、emoji 字符（如 `👍`）、
    数字 ID，以及 `[表情:赞]` 形式的渲染标记。
    """
    text = str(value or "").strip().strip("[]").strip()
    for prefix in ("表情:", "表情#"):
        text = text.removeprefix(prefix)
    text = text.strip()
    if not text:
        return None
    named = _FACE_ID_BY_NAME.get(text.lower())
    if named is not None:
        return named
    if text.isdigit():
        return int(text)
    for codepoint in (ord(ch) for ch in text):
        if _is_emoji_codepoint(codepoint) and codepoint not in _EMOJI_MODIFIER_CODEPOINTS:
            return codepoint
    return None


class SupportsGroupMemberInfo(Protocol):
    async def get_group_member_info(self, *, group_id: int, user_id: int) -> Any: ...


def iter_segments(message: Any) -> Iterable[Any]:
    """遍历消息中的消息段；兼容 Message、单个消息段、dict 段和非消息对象。"""
    if message is None:
        return
    if _is_segment(message):
        yield message
        return
    if isinstance(message, str):
        return
    try:
        candidates = list(message)
    except TypeError:
        return
    for item in candidates:
        if _is_segment(item):
            yield item


def _is_segment(value: Any) -> bool:
    if isinstance(value, dict):
        return isinstance(value.get("type"), str)
    return isinstance(getattr(value, "type", None), str) and isinstance(
        getattr(value, "data", None), dict
    )


def segment_type(segment: Any) -> str:
    if isinstance(segment, dict):
        return str(segment.get("type", ""))
    return str(segment.type)


def segment_data(segment: Any) -> dict[str, Any]:
    data = segment.get("data") if isinstance(segment, dict) else segment.data
    return data if isinstance(data, dict) else {}


def render_message_text(message: Any, *, resolve_name: NameResolver | None = None) -> str:
    """渲染整条消息为文本，非 text 段以标记形式内联在原位置。"""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    parts = [
        rendered
        for segment in iter_segments(message)
        if (rendered := render_segment_text(segment, resolve_name=resolve_name))
    ]
    if parts:
        return "".join(parts)
    return "" if _has_segments(message) else str(message)


def _has_segments(message: Any) -> bool:
    for _ in iter_segments(message):
        return True
    return False


def render_segment_text(segment: Any, *, resolve_name: NameResolver | None = None) -> str:
    seg_type = segment_type(segment)
    data = segment_data(segment)
    if seg_type == "text":
        return str(data.get("text", ""))
    if seg_type == "at":
        return _render_at(data, resolve_name)
    if seg_type == "face":
        return _render_face(data)
    if seg_type in {"mface", "marketface"}:
        return _render_mface(data)
    if seg_type == "dice":
        return _render_with_result("骰子", data)
    if seg_type == "rps":
        return _render_with_result("猜拳", data)
    return ""


def _render_at(data: dict[str, Any], resolve_name: NameResolver | None) -> str:
    qq = str(data.get("qq", "") or "").strip()
    if not qq:
        return ""
    if qq.lower() == "all":
        return "@全体成员"
    name = str(data.get("name") or "").strip().lstrip("@").strip()
    if not name and resolve_name is not None:
        name = (resolve_name(qq) or "").strip()
    return f"@{name}({qq})" if name else f"@{qq}"


def _render_face(data: dict[str, Any]) -> str:
    raw = data.get("raw")
    if isinstance(raw, dict):
        face_text = str(raw.get("faceText") or "").strip().lstrip("/").strip("[]").strip()
        if face_text:
            return f"[表情:{face_text}]"
    face_id = _parse_int(data.get("id"))
    if face_id is None:
        return "[表情]"
    name = QQ_FACE_NAMES.get(face_id)
    return f"[表情:{name}]" if name else f"[表情#{face_id}]"


def _render_mface(data: dict[str, Any]) -> str:
    summary = str(data.get("summary") or "").strip().strip("[]").strip()
    return f"[动画表情:{summary}]" if summary else "[动画表情]"


def _render_with_result(label: str, data: dict[str, Any]) -> str:
    result = str(data.get("result", "") or "").strip()
    return f"[{label}:{result}]" if result else f"[{label}]"


def _parse_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def render_event_text(event: Any, *, resolve_name: NameResolver | None = None) -> str:
    """渲染事件的消息文本；事件不携带消息段时回退到纯文本。"""
    message = getattr(event, "message", None)
    if message is not None:
        return render_message_text(message, resolve_name=resolve_name)
    plaintext = getattr(event, "get_plaintext", None)
    if callable(plaintext):
        return plaintext() or ""
    return ""


def message_mentions_user(message: Any, user_id: str) -> bool:
    """判断消息中是否有 @ 指定用户的消息段（不含 @全体成员）。"""
    target = str(user_id)
    for segment in iter_segments(message):
        if segment_type(segment) == "at" and str(segment_data(segment).get("qq", "")).strip() == target:
            return True
    return False


def describe_reaction_emoji(emoji_id: Any) -> str:
    """描述贴表情（表情回应）使用的表情。

    emoji_id 在系统表情范围内是 QQ 表情 ID，否则是 Unicode 码点
    （如 128077 = 👍）。
    """
    parsed = _parse_int(emoji_id)
    if parsed is None:
        return "[表情]"
    name = QQ_FACE_NAMES.get(parsed)
    if name:
        return f"[表情:{name}]"
    if _is_emoji_codepoint(parsed):
        try:
            return chr(parsed)
        except (ValueError, OverflowError):
            pass
    return f"[表情#{parsed}]"


def describe_reaction_emojis(likes: Any) -> str:
    """描述一次贴表情事件里的所有表情（count > 1 时附带次数）。"""
    parts: list[str] = []
    if isinstance(likes, list):
        for like in likes:
            if not isinstance(like, dict):
                continue
            emoji_id = like.get("emoji_id")
            if emoji_id is None:
                continue
            rendered = describe_reaction_emoji(emoji_id)
            count = _parse_int(like.get("count"))
            if count is not None and count > 1:
                rendered = f"{rendered}×{count}"
            parts.append(rendered)
    return " ".join(parts)


def extract_poke_action_texts(raw_info: Any) -> tuple[str, str]:
    """从戳一戳事件的 raw_info 中提取动作文案（如 "拍了拍", "的头"）。"""
    texts: list[str] = []
    if isinstance(raw_info, list):
        for item in raw_info:
            if not isinstance(item, dict) or str(item.get("type", "")) != "nor":
                continue
            txt = str(item.get("txt", "") or "").strip()
            if txt:
                texts.append(txt)
    action = texts[0] if texts else "戳了戳"
    suffix = texts[1] if len(texts) > 1 else ""
    return action, suffix


class MentionNameCache:
    """按 (group_id, user_id) 缓存群成员展示名，避免重复调用 OneBot API。

    空字符串表示查询过但没有结果（负缓存，较短 TTL），避免对失败的
    查询反复请求协议端。
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 600.0,
        negative_ttl_seconds: float = 60.0,
        max_entries: int = 4096,
    ) -> None:
        self._ttl = ttl_seconds
        self._negative_ttl = negative_ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[tuple[str, str], tuple[float, str]] = {}

    def get(self, group_id: str, user_id: str) -> str | None:
        key = (group_id, user_id)
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, name = entry
        if time.monotonic() >= expires_at:
            self._entries.pop(key, None)
            return None
        return name

    def set(self, group_id: str, user_id: str, name: str) -> None:
        if len(self._entries) >= self._max_entries:
            self._evict()
        ttl = self._ttl if name else self._negative_ttl
        self._entries[(group_id, user_id)] = (time.monotonic() + ttl, name)

    def _evict(self) -> None:
        now = time.monotonic()
        for key in [key for key, (expires_at, _) in self._entries.items() if expires_at <= now]:
            self._entries.pop(key, None)
        while len(self._entries) >= self._max_entries:
            oldest_key = min(self._entries, key=lambda key: self._entries[key][0])
            self._entries.pop(oldest_key, None)


async def collect_mention_names(
    bot: SupportsGroupMemberInfo,
    *,
    group_id: str,
    messages: Iterable[Any],
    cache: MentionNameCache,
    max_lookups: int = 8,
) -> dict[str, str]:
    """收集消息（含回复引用）中被 @ 成员的展示名，返回 qq -> 名称。

    优先使用消息段自带的 name；缺失时查询群成员信息并写入缓存。
    查询失败只降级为纯 QQ 号渲染，不影响消息处理。
    """
    names: dict[str, str] = {}
    pending: list[str] = []
    for message in messages:
        for segment in iter_segments(message):
            if segment_type(segment) != "at":
                continue
            data = segment_data(segment)
            qq = str(data.get("qq", "") or "").strip()
            if not qq or qq.lower() == "all" or qq in names or qq in pending:
                continue
            inline_name = str(data.get("name") or "").strip().lstrip("@").strip()
            if inline_name:
                names[qq] = inline_name
                continue
            cached = cache.get(group_id, qq)
            if cached is not None:
                if cached:
                    names[qq] = cached
                continue
            pending.append(qq)

    for qq in pending[:max_lookups]:
        name = await resolve_member_display_name(bot, group_id=group_id, user_id=qq, cache=cache)
        if name:
            names[qq] = name
    return names


async def resolve_member_display_name(
    bot: SupportsGroupMemberInfo,
    *,
    group_id: str,
    user_id: str,
    cache: MentionNameCache,
) -> str | None:
    """查询群成员展示名（群名片优先），结果写入缓存；失败返回 None。"""
    cached = cache.get(group_id, user_id)
    if cached is not None:
        return cached or None
    name = await _fetch_member_display_name(bot, group_id=group_id, user_id=user_id)
    cache.set(group_id, user_id, name or "")
    return name


async def _fetch_member_display_name(
    bot: SupportsGroupMemberInfo,
    *,
    group_id: str,
    user_id: str,
) -> str | None:
    try:
        info = await bot.get_group_member_info(group_id=int(group_id), user_id=int(user_id))
    except Exception as exc:
        logger.debug(
            f"bampi_chat mention name lookup failed group_id={group_id} "
            f"user_id={user_id} error={exc}"
        )
        return None
    if not isinstance(info, dict):
        return None
    card = str(info.get("card") or "").strip()
    nickname = str(info.get("nickname") or "").strip()
    return card or nickname or None
