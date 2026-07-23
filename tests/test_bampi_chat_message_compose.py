from __future__ import annotations

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.message_compose import (
    ComposeOptions,
    compose_options_from_config,
    compose_outbound_message,
    format_curated_face_names_for_prompt,
)
from bampi.plugins.bampi_chat.message_render import FACE_ID_BY_NAME
from bampi.plugins.bampi_chat.prompt import build_system_prompt


def _segment_snapshot(message) -> list[tuple[str, dict]]:
    return [(seg.type, dict(seg.data)) for seg in message]


def test_compose_plain_text_unchanged():
    message = compose_outbound_message("帮我看看这个")
    assert _segment_snapshot(message) == [("text", {"text": "帮我看看这个"})]


def test_compose_named_at_and_bare_qq():
    message = compose_outbound_message("请 @张三(10001) 和 @10002 看一下")
    assert _segment_snapshot(message) == [
        ("text", {"text": "请 "}),
        ("at", {"qq": "10001"}),
        ("text", {"text": " 和 "}),
        ("at", {"qq": "10002"}),
        ("text", {"text": " 看一下"}),
    ]


def test_compose_at_all_disabled_by_default():
    message = compose_outbound_message("@全体成员 开会了")
    assert _segment_snapshot(message) == [("text", {"text": "@全体成员 开会了"})]


def test_compose_at_all_when_enabled():
    message = compose_outbound_message(
        "@全体成员 开会了",
        options=ComposeOptions(at_all_enabled=True),
    )
    assert _segment_snapshot(message) == [
        ("at", {"qq": "all"}),
        ("text", {"text": " 开会了"}),
    ]


def test_compose_at_limit_excess_kept_as_text():
    text = "@10001 @10002 @10003"
    message = compose_outbound_message(text, options=ComposeOptions(at_limit=2))
    assert _segment_snapshot(message) == [
        ("at", {"qq": "10001"}),
        ("text", {"text": " "}),
        ("at", {"qq": "10002"}),
        ("text", {"text": " @10003"}),
    ]


def test_compose_face_formal_and_numeric():
    doge_id = FACE_ID_BY_NAME["doge"]
    message = compose_outbound_message(f"好活[表情:doge]再来[表情#{doge_id}]")
    assert _segment_snapshot(message) == [
        ("text", {"text": "好活"}),
        ("face", {"id": str(doge_id)}),
        ("text", {"text": "再来"}),
        ("face", {"id": str(doge_id)}),
    ]


def test_compose_unknown_numeric_face_kept_as_text():
    message = compose_outbound_message("未知[表情#999999]")
    assert _segment_snapshot(message) == [
        ("text", {"text": "未知[表情#999999]"}),
    ]


def test_compose_bare_bracket_face_from_whitelist():
    message = compose_outbound_message("无语[doge][笑哭]")
    assert _segment_snapshot(message) == [
        ("text", {"text": "无语"}),
        ("face", {"id": str(FACE_ID_BY_NAME["doge"])}),
        ("face", {"id": str(FACE_ID_BY_NAME["笑哭"])}),
    ]


def test_compose_unknown_face_kept_as_text():
    message = compose_outbound_message("看看[表情:不存在的表情]和[随便]")
    assert _segment_snapshot(message) == [
        ("text", {"text": "看看[表情:不存在的表情]和[随便]"}),
    ]


def test_compose_bare_nickname_at_not_converted():
    message = compose_outbound_message("喂 @张三 在吗")
    assert _segment_snapshot(message) == [("text", {"text": "喂 @张三 在吗"})]


def test_compose_bare_at_does_not_partially_match_ascii_identifier():
    message = compose_outbound_message(
        "账号 @123abc、邮箱 user@123.com 和 @全体成员们 都按文字处理"
    )
    assert _segment_snapshot(message) == [
        (
            "text",
            {"text": "账号 @123abc、邮箱 user@123.com 和 @全体成员们 都按文字处理"},
        ),
    ]


def test_compose_escaped_markup_as_plain_text():
    message = compose_outbound_message(r"原样显示 \@10001 和 \[doge]")
    assert _segment_snapshot(message) == [
        ("text", {"text": "原样显示 @10001 和 [doge]"}),
    ]


def test_compose_protects_inline_code_but_parses_outside_markup():
    message = compose_outbound_message("代码 `@10001 [doge]`，外面 @10002 [笑哭]")
    assert _segment_snapshot(message) == [
        ("text", {"text": "代码 `@10001 [doge]`，外面 "}),
        ("at", {"qq": "10002"}),
        ("text", {"text": " "}),
        ("face", {"id": str(FACE_ID_BY_NAME["笑哭"])}),
    ]


def test_compose_at_limit_is_shared_across_protected_ranges():
    message = compose_outbound_message(
        "@10001 `@10002` @10003",
        options=ComposeOptions(at_limit=1),
    )
    assert _segment_snapshot(message) == [
        ("at", {"qq": "10001"}),
        ("text", {"text": " `@10002` @10003"}),
    ]


def test_compose_protects_fenced_code_but_parses_after_closing_fence():
    message = compose_outbound_message(
        "示例：\n```text\n@10001 [doge]\n```\n外面 @10002"
    )
    assert _segment_snapshot(message) == [
        ("text", {"text": "示例：\n```text\n@10001 [doge]\n```\n外面 "}),
        ("at", {"qq": "10002"}),
    ]


def test_compose_protects_tilde_fenced_code():
    text = "~~~\n@10001 [doge]\n~~~\n"
    message = compose_outbound_message(text)
    assert _segment_snapshot(message) == [("text", {"text": text})]


def test_compose_protects_markdown_links_and_images():
    message = compose_outbound_message(
        "看 [doge](https://example.com/@10001) 和 "
        "![笑哭](https://example.com/a_(1).png)，再 @10002 [doge]"
    )
    assert _segment_snapshot(message) == [
        (
            "text",
            {
                "text": (
                    "看 [doge](https://example.com/@10001) 和 "
                    "![笑哭](https://example.com/a_(1).png)，再 "
                )
            },
        ),
        ("at", {"qq": "10002"}),
        ("text", {"text": " "}),
        ("face", {"id": str(FACE_ID_BY_NAME["doge"])}),
    ]


def test_compose_keeps_bare_face_before_separate_parentheses():
    message = compose_outbound_message("[doge] (补充说明)")
    assert _segment_snapshot(message) == [
        ("face", {"id": str(FACE_ID_BY_NAME["doge"])}),
        ("text", {"text": " (补充说明)"}),
    ]


def test_compose_conservatively_protects_unclosed_inline_code():
    text = "未闭合 `@10001 [doge] 后面 @10002"
    message = compose_outbound_message(text)
    assert _segment_snapshot(message) == [("text", {"text": text})]


def test_compose_conservatively_protects_unclosed_link_target():
    text = "未闭合 [doge](https://example.com/@10001 后面 @10002"
    message = compose_outbound_message(text)
    assert _segment_snapshot(message) == [("text", {"text": text})]


def test_compose_disabled_returns_plain_text():
    message = compose_outbound_message(
        "@张三(10001) [doge]",
        options=ComposeOptions(enabled=False),
    )
    assert _segment_snapshot(message) == [("text", {"text": "@张三(10001) [doge]"})]


def test_compose_mixed_roundtrip_style():
    """入站渲染格式应可被出站完整解析。"""
    text = "回复 @李四(20002)：收到[表情:赞]"
    message = compose_outbound_message(text)
    assert _segment_snapshot(message) == [
        ("text", {"text": "回复 "}),
        ("at", {"qq": "20002"}),
        ("text", {"text": "：收到"}),
        ("face", {"id": str(FACE_ID_BY_NAME["赞"])}),
    ]


def test_compose_options_from_config():
    config = BampiChatConfig(
        bampi_outbound_markup_enabled=False,
        bampi_outbound_at_all_enabled=True,
        bampi_outbound_at_limit=3,
    )
    opts = compose_options_from_config(config)
    assert opts == ComposeOptions(enabled=False, at_all_enabled=True, at_limit=3)


def test_prompt_documents_outbound_markup_when_enabled():
    prompt = build_system_prompt(BampiChatConfig(), ["qq_react"])
    assert "[表情:名称]" in prompt
    assert "优先写 `@QQ号`" in prompt
    assert format_curated_face_names_for_prompt().split("、")[0] in prompt
    assert "内联标记" in prompt
    assert r"\@123456" in prompt
    assert "不要在回复中书写这类标记语法" not in prompt


def test_prompt_forbids_markup_when_disabled():
    prompt = build_system_prompt(
        BampiChatConfig(bampi_outbound_markup_enabled=False),
        [],
    )
    assert "不要在回复中书写这类标记语法" in prompt
    assert "优先写 `@QQ号`" not in prompt
