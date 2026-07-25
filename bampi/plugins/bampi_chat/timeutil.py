"""本地时区与时间展示的统一入口。

存储层（记忆归档、画像、日程）一律保存 UTC ISO 字符串，这样跨时区和夏令时都不会歧义；
但写给模型或用户看的时间必须换算到本地时区，否则模型会把 UTC 的 13:00 读成“下午一点”，
而群里实际已经是晚上九点。所有面向展示的时间格式化都应经过本模块。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from nonebot import logger

DEFAULT_TIMEZONE_NAME = "Asia/Shanghai"

_DATETIME_FORMAT = "%Y-%m-%d %H:%M"
_DATE_FORMAT = "%Y-%m-%d"


@lru_cache(maxsize=1)
def default_timezone() -> tzinfo:
    """默认本地时区；系统缺少 tz 数据库时退化为固定 UTC+8 偏移。"""
    try:
        return ZoneInfo(DEFAULT_TIMEZONE_NAME)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            f"bampi_chat tz database unavailable; using fixed UTC+8 offset for {DEFAULT_TIMEZONE_NAME}"
        )
        return timezone(timedelta(hours=8), name=DEFAULT_TIMEZONE_NAME)


def resolve_timezone(name: str | None) -> tzinfo:
    """把配置中的时区名解析成 tzinfo；空值或无法识别时回退到默认时区。"""
    text = (name or "").strip()
    if not text:
        return default_timezone()
    try:
        return ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            f"bampi_chat unknown timezone={text!r}; falling back to {DEFAULT_TIMEZONE_NAME}"
        )
        return default_timezone()


def timezone_label(tz: tzinfo, *, at: datetime | None = None) -> str:
    """人类可读的偏移标签，例如 `UTC+8`、`UTC+5:30`、`UTC`。"""
    offset = (at or datetime.now(tz)).astimezone(tz).utcoffset() or timedelta(0)
    total_minutes = int(offset.total_seconds() // 60)
    if total_minutes == 0:
        return "UTC"
    sign = "+" if total_minutes > 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"UTC{sign}{hours}" if minutes == 0 else f"UTC{sign}{hours}:{minutes:02d}"


def parse_datetime(value: object, *, assume: tzinfo | None = None) -> datetime | None:
    """解析 ISO 时间串，返回 UTC 的 aware datetime；无法解析时返回 None。

    不带时区的输入按 `assume` 解释（默认 UTC）。存储层读回的值应使用默认值，
    模型或用户提供的时间应传入本地时区。
    """
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=assume or timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_datetime(value: object, *, tz: tzinfo) -> str:
    """把存储的 UTC 时间渲染成本地时区的 `YYYY-MM-DD HH:MM`。"""
    return _format(value, tz=tz, fmt=_DATETIME_FORMAT)


def format_date(value: object, *, tz: tzinfo) -> str:
    """把存储的 UTC 时间渲染成本地时区的 `YYYY-MM-DD`。"""
    return _format(value, tz=tz, fmt=_DATE_FORMAT)


def _format(value: object, *, tz: tzinfo, fmt: str) -> str:
    parsed = parse_datetime(value)
    if parsed is None:
        # 解析不了就原样透出，宁可显示原始值也不要凭空造一个时间。
        return str(value or "").strip()
    return parsed.astimezone(tz).strftime(fmt)
