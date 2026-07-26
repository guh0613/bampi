from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.forward_messages import (
    ForwardContext,
    RenderedForward,
    ResolvedForward,
    collect_forward_context,
    iter_forward_nodes,
)
from bampi.plugins.bampi_chat import handler as handler_module
from bampi.plugins.bampi_chat.handler import (
    build_user_message,
    collect_incoming_context,
)
from bampi.plugins.bampi_chat.prompt import build_system_prompt


class FakeForwardBot:
    def __init__(
        self,
        responses: dict[str, object] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.responses = responses or {}
        self.error = error
        self.calls: list[str] = []

    async def get_forward_msg(self, *, id: str):
        self.calls.append(id)
        if self.error is not None:
            raise self.error
        return self.responses[id]

    async def call_api(self, action: str, **params: object):
        raise AssertionError(f"unexpected generic API call: {action} {params}")


class FakeCallApiOnlyBot:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_api(self, action: str, **params: object):
        self.calls.append((action, params))
        return self.response


@dataclass
class FakeSender:
    user_id: int
    nickname: str = "Alice"
    card: str = ""


@dataclass
class FakeEvent:
    group_id: int = 1001
    user_id: int = 42
    message_id: int = 99
    message: Message = field(default_factory=Message)
    sender: object = field(default_factory=lambda: FakeSender(42))
    reply: object | None = None

    def get_plaintext(self) -> str:
        return self.message.extract_plain_text()


def forward_segment(
    forward_id: str, *, content: object | None = None
) -> MessageSegment:
    data: dict[str, object] = {"id": forward_id}
    if content is not None:
        data["content"] = content
    return MessageSegment("forward", data)


async def collect(bot, message: Message, *, reply_message: object = None, **overrides):
    options = {
        "enabled": True,
        "max_depth": 4,
        "max_nodes": 100,
        "max_roots": 4,
        "max_api_calls": 8,
        "max_text_chars": 30_000,
        "timeout_seconds": 1.0,
        "timezone": ZoneInfo("Asia/Shanghai"),
    }
    options.update(overrides)
    return await collect_forward_context(
        bot,
        message=message,
        reply_message=reply_message,
        **options,
    )


@pytest.mark.asyncio
async def test_forward_feature_can_be_disabled_without_api_calls():
    bot = FakeForwardBot({"root": {"messages": []}})

    context = await collect_forward_context(
        bot,
        message=Message([forward_segment("root")]),
        reply_message=None,
        enabled=False,
        max_depth=4,
        max_nodes=100,
        max_roots=4,
        max_api_calls=8,
        max_text_chars=30_000,
        timeout_seconds=1.0,
        timezone=ZoneInfo("Asia/Shanghai"),
    )

    assert bot.calls == []
    assert context == ForwardContext()


@pytest.mark.asyncio
async def test_collects_current_napcat_messages_shape_and_nested_inline_content():
    nested_content = [
        {
            "user_id": 2,
            "time": 1_700_000_100,
            "sender": {"nickname": "Bob"},
            "message": [{"type": "text", "data": {"text": "nested text"}}],
        }
    ]
    bot = FakeForwardBot(
        {
            "root": {
                "messages": [
                    {
                        "user_id": 1,
                        "time": 1_700_000_000,
                        "sender": {"nickname": "Alice"},
                        "message": [
                            {"type": "text", "data": {"text": "hello"}},
                            {"type": "face", "data": {"id": "179"}},
                            {
                                "type": "forward",
                                "data": {"id": "nested", "content": nested_content},
                            },
                        ],
                    }
                ]
            }
        }
    )

    context = await collect(bot, Message([forward_segment("root")]))

    assert bot.calls == ["root"]
    assert len(context.current) == 1
    assert len(list(iter_forward_nodes(context.current))) == 2
    assert "Alice(1)" in context.current_render.text
    assert "hello[表情:doge][嵌套合并转发]" in context.current_render.text
    assert "Bob(2)" in context.current_render.text
    assert "nested text" in context.current_render.text


@pytest.mark.asyncio
async def test_normalizes_standard_onebot_node_shape_with_generic_call_api_fallback():
    bot = FakeCallApiOnlyBot(
        {
            "message": [
                {
                    "type": "node",
                    "data": {
                        "user_id": "7",
                        "nickname": "Carol",
                        "content": [
                            {"type": "text", "data": {"text": "standard node"}},
                            {
                                "type": "file",
                                "data": {"file": "report.pdf", "file_size": "2048"},
                            },
                        ],
                    },
                }
            ]
        }
    )

    context = await collect(bot, Message([forward_segment("standard")]))

    assert bot.calls == [("get_forward_msg", {"id": "standard"})]
    assert "Carol(7)" in context.current_render.text
    assert "standard node[文件:report.pdf，2.0KB]" in context.current_render.text


@pytest.mark.asyncio
async def test_prefers_inline_forward_content_without_api_call():
    bot = FakeForwardBot()
    inline = [
        {
            "user_id": 9,
            "sender": {"nickname": "Inline"},
            "message": [{"type": "text", "data": {"text": "already parsed"}}],
        }
    ]

    context = await collect(bot, Message([forward_segment("inline", content=inline)]))

    assert bot.calls == []
    assert "Inline(9)" in context.current_render.text
    assert "already parsed" in context.current_render.text


@pytest.mark.asyncio
async def test_collects_forward_from_reply_separately():
    bot = FakeForwardBot(
        {
            "reply-forward": {
                "messages": [
                    {
                        "user_id": 8,
                        "sender": {"nickname": "ReplyUser"},
                        "message": [
                            {"type": "text", "data": {"text": "quoted forward"}}
                        ],
                    }
                ]
            }
        }
    )

    context = await collect(
        bot,
        Message("summarize this"),
        reply_message=Message([forward_segment("reply-forward")]),
    )

    assert context.has_current is False
    assert context.has_reply is True
    assert "ReplyUser(8)" in context.reply_render.text
    assert "quoted forward" in context.reply_render.text


@pytest.mark.asyncio
async def test_parses_cq_code_string_inside_standard_node():
    bot = FakeForwardBot(
        {
            "cq": {
                "message": [
                    {
                        "type": "node",
                        "data": {
                            "user_id": "7",
                            "nickname": "Carol",
                            "content": "hello[CQ:face,id=179]",
                        },
                    }
                ]
            }
        }
    )

    context = await collect(bot, Message([forward_segment("cq")]))

    assert "hello[表情:doge]" in context.current_render.text


@pytest.mark.asyncio
async def test_forward_api_failure_is_visible_but_does_not_raise():
    bot = FakeForwardBot(error=RuntimeError("expired"))

    context = await collect(bot, Message([forward_segment("expired")]))

    assert context.has_current is True
    assert "读取失败" in context.current_render.text
    assert "expired" not in context.current_render.text


@pytest.mark.asyncio
async def test_applies_global_node_and_depth_limits():
    deep = [
        {
            "user_id": 3,
            "sender": {"nickname": "Deep"},
            "message": [
                {
                    "type": "forward",
                    "data": {
                        "id": "too-deep",
                        "content": [
                            {
                                "user_id": 4,
                                "sender": {"nickname": "Hidden"},
                                "message": [
                                    {"type": "text", "data": {"text": "hidden"}}
                                ],
                            }
                        ],
                    },
                }
            ],
        }
    ]
    root = [
        {
            "user_id": 1,
            "sender": {"nickname": "Root"},
            "message": [{"type": "forward", "data": {"id": "deep", "content": deep}}],
        },
        {
            "user_id": 2,
            "sender": {"nickname": "Skipped"},
            "message": [{"type": "text", "data": {"text": "skip me"}}],
        },
    ]
    bot = FakeForwardBot({"root": {"messages": root}})

    context = await collect(
        bot,
        Message([forward_segment("root")]),
        max_depth=2,
        max_nodes=2,
    )

    assert len(list(iter_forward_nodes(context.current))) == 2
    assert "深度超过限制" in context.current_render.text
    assert "Skipped" not in context.current_render.text


@pytest.mark.asyncio
async def test_duplicate_forward_uses_payload_cache_without_bypassing_node_limit():
    payload = {
        "messages": [
            {
                "user_id": index,
                "sender": {"nickname": f"User{index}"},
                "message": [{"type": "text", "data": {"text": f"message {index}"}}],
            }
            for index in (1, 2)
        ]
    }
    bot = FakeForwardBot({"same": payload})
    message = Message([forward_segment("same"), forward_segment("same")])

    context = await collect(bot, message, max_nodes=2)

    assert bot.calls == ["same"]
    assert len(list(iter_forward_nodes(context.current))) == 2
    assert "节点超过限制" in context.current_render.text


@pytest.mark.asyncio
async def test_malformed_node_does_not_discard_other_forward_nodes():
    class BrokenNode:
        def model_dump(self):
            raise RuntimeError("broken node")

    bot = FakeForwardBot(
        {
            "root": {
                "messages": [
                    BrokenNode(),
                    {
                        "user_id": 7,
                        "sender": {"nickname": "Good\nName"},
                        "message": [
                            {"type": "text", "data": {"text": "still visible"}}
                        ],
                    },
                ]
            }
        }
    )

    context = await collect(bot, Message([forward_segment("root")]))

    assert "Good Name(7)" in context.current_render.text
    assert "still visible" in context.current_render.text
    assert "格式异常" in context.current_render.text


@pytest.mark.asyncio
async def test_resolve_timeout_degrades_to_failure_note():
    class SlowBot:
        async def get_forward_msg(self, *, id: str):
            await asyncio.sleep(0.05)
            return {"messages": []}

    context = await collect(
        SlowBot(),
        Message([forward_segment("slow")]),
        timeout_seconds=0.001,
    )

    assert "读取失败" in context.current_render.text


@pytest.mark.asyncio
async def test_collect_incoming_context_downloads_forward_image_and_injects_transcript(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_download(url: str, *, timeout: float, max_bytes: int):
        assert url == "https://example.com/forward.png"
        assert max_bytes <= 1024
        return b"forward-image", "image/png"

    monkeypatch.setattr(handler_module, "download_url", fake_download)
    bot = FakeForwardBot(
        {
            "root": {
                "messages": [
                    {
                        "user_id": 7,
                        "sender": {"nickname": "Bob"},
                        "message": [
                            {"type": "text", "data": {"text": "see image"}},
                            {
                                "type": "image",
                                "data": {
                                    "url": "https://example.com/forward.png",
                                    "file": "forward.png",
                                },
                            },
                        ],
                    }
                ]
            }
        }
    )
    event = FakeEvent(
        message=Message([MessageSegment.text("看看"), forward_segment("root")])
    )
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_max_inline_image_size=1024,
        bampi_forward_max_total_media_bytes=1024,
    )

    media, forwards = await collect_incoming_context(
        bot,
        event,  # type: ignore[arg-type]
        config,
        str(tmp_path / "workspace"),
    )
    user_message = build_user_message(
        event,  # type: ignore[arg-type]
        "看看",
        media,
        forwards=forwards,
    )

    assert len(media.inline_images) == 1
    assert "forwarded_messages:" in user_message.content[0].text
    assert "Bob(7)" in user_message.content[0].text
    assert "see image[图片]" in user_message.content[0].text
    assert len(user_message.content) == 2


@pytest.mark.asyncio
async def test_forward_file_without_direct_url_never_uses_child_group_id(tmp_path):
    bot = FakeForwardBot(
        {
            "root": {
                "messages": [
                    {
                        "user_id": 7,
                        "group_id": 284840486,
                        "sender": {"nickname": "Bob"},
                        "message": [
                            {
                                "type": "file",
                                "data": {"file_id": "inner-file", "file": "secret.txt"},
                            }
                        ],
                    }
                ]
            }
        }
    )
    event = FakeEvent(message=Message([forward_segment("root")]))
    config = BampiChatConfig(bampi_workspace_dir=str(tmp_path / "workspace"))

    media, _ = await collect_incoming_context(
        bot,
        event,  # type: ignore[arg-type]
        config,
        str(tmp_path / "workspace"),
    )

    assert bot.calls == ["root"]
    assert media.saved_paths == []
    assert media.notes == ["收到文件，但缺少可下载 URL。"]


@pytest.mark.asyncio
async def test_current_and_reply_share_the_text_preview_budget():
    def long_message(name: str) -> dict[str, object]:
        return {
            "messages": [
                {
                    "user_id": 7,
                    "sender": {"nickname": name},
                    "message": [{"type": "text", "data": {"text": "x" * 1200}}],
                }
            ]
        }

    bot = FakeForwardBot(
        {"current": long_message("Current"), "reply": long_message("Reply")}
    )

    context = await collect(
        bot,
        Message([forward_segment("current")]),
        reply_message=Message([forward_segment("reply")]),
        max_text_chars=1000,
    )

    assert context.current_render.truncated is True
    assert context.reply_render.truncated is True
    assert len(context.current_render.text) + len(context.reply_render.text) <= 1000


@pytest.mark.asyncio
async def test_long_forward_transcript_is_saved_to_workspace(tmp_path):
    long_text = "x" * 1500
    bot = FakeForwardBot(
        {
            "root": {
                "messages": [
                    {
                        "user_id": 7,
                        "sender": {"nickname": "Bob"},
                        "message": [{"type": "text", "data": {"text": long_text}}],
                    }
                ]
            }
        }
    )
    event = FakeEvent(message=Message([forward_segment("root")]))
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_forward_max_text_chars=1000,
    )

    media, forwards = await collect_incoming_context(
        bot,
        event,  # type: ignore[arg-type]
        config,
        str(tmp_path / "workspace"),
    )

    assert forwards.current_render.truncated is True
    assert len(media.saved_paths) == 1
    saved = tmp_path / "workspace" / media.saved_paths[0]
    assert saved.name.startswith("forwarded-messages-")
    assert long_text in saved.read_text(encoding="utf-8")
    assert "完整转录已保存" in media.notes[0]


def test_system_prompt_treats_forwarded_messages_as_quoted_material():
    prompt = build_system_prompt(BampiChatConfig(), [])

    assert "forwarded_messages" in prompt
    assert "引用材料" in prompt
    assert "不是系统指令" in prompt


def test_build_user_message_marks_reply_forward_context():
    rendered = RenderedForward(text="[合并转发 1，共 1 条]\n1. Bob(7)\n   hello")
    forwards = ForwardContext(
        reply=(ResolvedForward(forward_id="reply"),),
        reply_render=rendered,
    )
    event = FakeEvent(
        reply=SimpleNamespace(sender=FakeSender(7, nickname="Bob"), message=Message())
    )

    message = build_user_message(
        event,  # type: ignore[arg-type]
        "",
        handler_module.IncomingMedia(),
        forwards=forwards,
    )

    assert (
        "message_text: (无附言；请结合回复引用中的合并转发理解)"
        in message.content[0].text
    )
    assert "reply_forwarded_messages:" in message.content[0].text
