from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from bampi.plugins.bampi_chat.timeutil import (
    DEFAULT_TIMEZONE_NAME,
    format_date,
    format_datetime,
    parse_datetime,
    resolve_timezone,
    timezone_label,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def test_resolve_timezone_falls_back_on_unknown_name():
    assert resolve_timezone("Asia/Tokyo") == ZoneInfo("Asia/Tokyo")
    assert resolve_timezone("") == ZoneInfo(DEFAULT_TIMEZONE_NAME)
    assert resolve_timezone("Not/AZone") == ZoneInfo(DEFAULT_TIMEZONE_NAME)


def test_timezone_label_renders_offsets():
    assert timezone_label(SHANGHAI) == "UTC+8"
    assert timezone_label(timezone.utc) == "UTC"
    assert timezone_label(ZoneInfo("Asia/Kolkata")) == "UTC+5:30"
    assert timezone_label(timezone(timedelta(hours=-3))) == "UTC-3"


def test_parse_datetime_assumes_utc_by_default_and_honours_assume():
    assert parse_datetime("2026-04-28T13:00:00Z") == datetime(2026, 4, 28, 13, 0, tzinfo=timezone.utc)
    assert parse_datetime("2026-04-28T13:00:00") == datetime(2026, 4, 28, 13, 0, tzinfo=timezone.utc)
    assert parse_datetime("2026-04-28T21:00:00", assume=SHANGHAI) == datetime(
        2026, 4, 28, 13, 0, tzinfo=timezone.utc
    )
    assert parse_datetime("2026-04-28T21:00:00+08:00", assume=timezone.utc) == datetime(
        2026, 4, 28, 13, 0, tzinfo=timezone.utc
    )
    assert parse_datetime("") is None
    assert parse_datetime("上周三") is None


def test_format_converts_stored_utc_to_local():
    assert format_datetime("2026-04-28T13:00:00+00:00", tz=SHANGHAI) == "2026-04-28 21:00"
    # 跨零点的会话在本地时区属于第二天，日期也要跟着走。
    assert format_date("2026-04-28T17:30:00+00:00", tz=SHANGHAI) == "2026-04-29"


def test_format_keeps_unparsable_input_verbatim():
    assert format_datetime("不是时间", tz=SHANGHAI) == "不是时间"
    assert format_datetime(None, tz=SHANGHAI) == ""
