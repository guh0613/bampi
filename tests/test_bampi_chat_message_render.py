from __future__ import annotations

import asyncio

import pytest
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from bampi.plugins.bampi_chat.message_render import (
    MentionNameCache,
    collect_mention_names,
    describe_reaction_emoji,
    describe_reaction_emojis,
    extract_poke_action_texts,
    message_mentions_user,
    render_event_text,
    render_message_text,
)


class FakeMemberInfoBot:
    def __init__(self, members: dict[int, dict[str, str]] | None = None, *, fail: bool = False) -> None:
        self.members = members or {}
        self.fail = fail
        self.calls: list[tuple[int, int]] = []

    async def get_group_member_info(self, *, group_id: int, user_id: int) -> dict[str, str]:
        self.calls.append((group_id, user_id))
        if self.fail or user_id not in self.members:
            raise RuntimeError("member not found")
        return self.members[user_id]


def test_render_keeps_text_segments_verbatim():
    message = Message("帮我看看这个")
    assert render_message_text(message) == "帮我看看这个"


def test_render_at_with_inline_name():
    message = Message([MessageSegment.text("请 ")]) + MessageSegment("at", {"qq": "10001", "name": "张三"})
    message += MessageSegment.text(" 看一下")
    assert render_message_text(message) == "请 @张三(10001) 看一下"


def test_render_at_without_name_falls_back_to_qq():
    message = Message([MessageSegment.at(10001), MessageSegment.text(" 在吗")])
    assert render_message_text(message) == "@10001 在吗"


def test_render_at_uses_resolver_when_name_missing():
    message = Message([MessageSegment.at(10001), MessageSegment.text(" 在吗")])
    rendered = render_message_text(message, resolve_name={"10001": "李四"}.get)
    assert rendered == "@李四(10001) 在吗"


def test_render_at_all():
    message = Message([MessageSegment("at", {"qq": "all"}), MessageSegment.text(" 开会了")])
    assert render_message_text(message) == "@全体成员 开会了"


def test_render_face_prefers_napcat_raw_face_text():
    segment = MessageSegment("face", {"id": "999", "raw": {"faceText": "/贴贴"}})
    assert render_message_text(Message([segment])) == "[表情:贴贴]"


def test_render_face_uses_builtin_table():
    message = Message([MessageSegment.text("好活"), MessageSegment.face(179)])
    assert render_message_text(message) == "好活[表情:doge]"


def test_render_face_unknown_id_falls_back_to_number():
    assert render_message_text(Message([MessageSegment.face(9999)])) == "[表情#9999]"


def test_render_mface_summary():
    segment = MessageSegment("mface", {"summary": "[贴贴]", "emoji_id": "abc"})
    assert render_message_text(Message([segment])) == "[动画表情:贴贴]"


def test_render_skips_media_segments():
    message = Message(
        [
            MessageSegment.text("看图 "),
            MessageSegment.image("https://example.com/a.png"),
            MessageSegment.face(14),
        ]
    )
    assert render_message_text(message) == "看图 [表情:微笑]"


def test_render_dice_and_rps():
    message = Message([MessageSegment("dice", {"result": "3"}), MessageSegment("rps", {})])
    assert render_message_text(message) == "[骰子:3][猜拳]"


def test_render_plain_string_message():
    assert render_message_text("原始说明") == "原始说明"


def test_render_event_text_falls_back_to_plaintext():
    class PlainEvent:
        def get_plaintext(self) -> str:
            return "hello"

    assert render_event_text(PlainEvent()) == "hello"


def test_message_mentions_user():
    message = Message([MessageSegment.text("交给 "), MessageSegment.at(42)])
    assert message_mentions_user(message, "42") is True
    assert message_mentions_user(message, "43") is False
    at_all = Message([MessageSegment("at", {"qq": "all"})])
    assert message_mentions_user(at_all, "42") is False


def test_mention_cache_expires_entries(monkeypatch: pytest.MonkeyPatch):
    now = 1000.0
    monkeypatch.setattr("bampi.plugins.bampi_chat.message_render.time.monotonic", lambda: now)
    cache = MentionNameCache(ttl_seconds=10.0)
    cache.set("g", "1", "张三")
    assert cache.get("g", "1") == "张三"
    now += 11.0
    assert cache.get("g", "1") is None


def test_collect_mention_names_prefers_inline_name():
    bot = FakeMemberInfoBot()
    message = Message([MessageSegment("at", {"qq": "10001", "name": "张三"})])
    names = asyncio.run(
        collect_mention_names(bot, group_id="1", messages=(message,), cache=MentionNameCache())
    )
    assert names == {"10001": "张三"}
    assert bot.calls == []


def test_collect_mention_names_fetches_and_caches():
    bot = FakeMemberInfoBot({10001: {"card": "", "nickname": "李四"}})
    cache = MentionNameCache()
    message = Message([MessageSegment.at(10001)])

    names = asyncio.run(collect_mention_names(bot, group_id="1", messages=(message,), cache=cache))
    assert names == {"10001": "李四"}

    again = asyncio.run(collect_mention_names(bot, group_id="1", messages=(message,), cache=cache))
    assert again == {"10001": "李四"}
    assert bot.calls == [(1, 10001)]


def test_collect_mention_names_caches_failures():
    bot = FakeMemberInfoBot(fail=True)
    cache = MentionNameCache()
    message = Message([MessageSegment.at(10001)])

    names = asyncio.run(collect_mention_names(bot, group_id="1", messages=(message,), cache=cache))
    assert names == {}

    asyncio.run(collect_mention_names(bot, group_id="1", messages=(message,), cache=cache))
    assert bot.calls == [(1, 10001)]


def test_render_supports_dict_segments():
    message = [
        {"type": "text", "data": {"text": "看这个 "}},
        {"type": "at", "data": {"qq": "10001", "name": "张三"}},
        {"type": "face", "data": {"id": "14"}},
    ]
    assert render_message_text(message) == "看这个 @张三(10001)[表情:微笑]"


def test_describe_reaction_emoji_variants():
    assert describe_reaction_emoji("76") == "[表情:赞]"
    assert describe_reaction_emoji(128077) == "👍"
    assert describe_reaction_emoji("not-a-number") == "[表情]"
    assert describe_reaction_emoji("4500") == "[表情#4500]"


def test_describe_reaction_emojis_with_counts():
    likes = [
        {"emoji_id": "76", "count": 3},
        {"emoji_id": "128077", "count": 1},
    ]
    assert describe_reaction_emojis(likes) == "[表情:赞]×3 👍"
    assert describe_reaction_emojis(None) == ""
    assert describe_reaction_emojis([{"count": 2}]) == ""


def test_extract_poke_action_texts():
    raw_info = [
        {"type": "qq", "uid": "u1"},
        {"type": "img", "src": "x"},
        {"type": "nor", "txt": "拍了拍"},
        {"type": "qq", "uid": "u2"},
        {"type": "nor", "txt": "的头"},
    ]
    assert extract_poke_action_texts(raw_info) == ("拍了拍", "的头")
    assert extract_poke_action_texts(None) == ("戳了戳", "")
    assert extract_poke_action_texts([]) == ("戳了戳", "")


def test_collect_mention_names_covers_reply_message_and_skips_all():
    bot = FakeMemberInfoBot({10002: {"card": "群名片", "nickname": "王五"}})
    main_message = Message([MessageSegment("at", {"qq": "all"})])
    reply_message = Message([MessageSegment.at(10002)])
    names = asyncio.run(
        collect_mention_names(
            bot,
            group_id="1",
            messages=(main_message, reply_message, None),
            cache=MentionNameCache(),
        )
    )
    assert names == {"10002": "群名片"}
