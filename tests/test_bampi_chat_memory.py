from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path

import pytest

from bampy.ai.types import AssistantMessage, StopReason, TextContent, ToolCall, ToolResultMessage, UserMessage

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.memory import (
    MemoryManager,
    MemoryMessage,
    MemoryParticipant,
    MemoryToolEvent,
    MemoryUserTurn,
    render_search_results,
)
from bampi.plugins.bampi_chat.memory.embeddings import (
    DEFAULT_EMBEDDING_USER_AGENT,
    OpenAICompatibleEmbeddingProvider,
    build_embedding_provider,
    normalize_openai_embedding_base_url,
    normalize_vector,
)
from bampi.plugins.bampi_chat.memory.schema import CURRENT_SCHEMA_VERSION
from bampi.plugins.bampi_chat.memory.search_text import build_fts_query, extract_search_terms
from bampi.plugins.bampi_chat.memory.vector_index import load_sqlite_vec
from bampi.plugins.bampi_chat.prompt import build_system_prompt
from bampi.plugins.bampi_chat.tools import create_agent_tools


def _seed_memory(manager: MemoryManager) -> dict[str, int]:
    nginx_id = manager.archive_conversation(
        group_id="1001",
        started_at="2026-04-28T21:10:00+08:00",
        ended_at="2026-04-28T22:05:00+08:00",
        participants=[
            MemoryParticipant(user_id="42", nickname="张三"),
            MemoryParticipant(user_id="9000", nickname="Ophelia"),
        ],
        title="配置 nginx 反向代理和 TLS 证书",
        summary=(
            "这次会话讨论并调整了某服务的 nginx 反向代理配置，涉及 443 端口、"
            "app.conf 路径和 reload 校验。最后保留 HTTP 转发，TLS 证书待后续继续处理。"
        ),
        keywords=["nginx", "反向代理", "服务器配置", "TLS 证书"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="上周那个 nginx 反代现在能不能继续改一下，证书还没配完。",
                timestamp="2026-04-28T21:10:00+08:00",
            ),
            MemoryMessage(
                role="assistant",
                content="我先看 /etc/nginx/sites-enabled/app.conf 和当前 reload 状态。",
                timestamp="2026-04-28T21:12:00+08:00",
            ),
            MemoryMessage(
                role="assistant",
                content="配置文件检查通过，后续可以把证书路径接到 server 443 块里。",
                timestamp="2026-04-28T22:04:00+08:00",
            ),
        ],
        tool_events=[
            MemoryToolEvent(
                timestamp="2026-04-28T21:13:00+08:00",
                tool_call_id="tool-1",
                tool_name="read",
                arguments_text='{"path":"/etc/nginx/sites-enabled/app.conf"}',
                result_preview="server { listen 80; proxy_pass http://127.0.0.1:46000; }",
                result_full="server { listen 80; proxy_pass http://127.0.0.1:46000; }",
            ),
            MemoryToolEvent(
                timestamp="2026-04-28T21:20:00+08:00",
                tool_call_id="tool-2",
                tool_name="bash",
                arguments_text='{"command":"nginx -t && systemctl reload nginx"}',
                result_preview="nginx: configuration file /etc/nginx/nginx.conf test is successful; reload ok",
                result_full="nginx: configuration file /etc/nginx/nginx.conf test is successful; reload ok",
            ),
        ],
    )

    crawler_id = manager.archive_conversation(
        group_id="1001",
        started_at="2026-04-30T09:00:00+08:00",
        ended_at="2026-04-30T09:40:00+08:00",
        participants=[MemoryParticipant(user_id="7", nickname="李四")],
        title="排查 Scrapy 和 httpx 爬虫超时",
        summary="讨论毕设数据采集脚本，定位 httpx timeout 和 Scrapy 下载中间件配置。",
        keywords=["Scrapy", "httpx", "爬虫", "毕设"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="7",
                nickname="李四",
                content="我的毕设爬虫 httpx 总是 timeout，Scrapy 中间件要怎么配？",
                timestamp="2026-04-30T09:00:00+08:00",
            ),
            MemoryMessage(
                role="assistant",
                content="先把 timeout、重试和并发降下来，再看代理池是否稳定。",
                timestamp="2026-04-30T09:10:00+08:00",
            ),
        ],
    )

    minecraft_id = manager.archive_conversation(
        group_id="1001",
        started_at="2026-05-01T20:00:00+08:00",
        ended_at="2026-05-01T20:20:00+08:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="Minecraft 服务器白名单",
        summary="讨论 Minecraft 服务器 whitelist.json、端口和玩家 ID。",
        keywords=["Minecraft", "白名单", "服务器"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="MC 服务器白名单加一下 Steve。",
                timestamp="2026-05-01T20:00:00+08:00",
            )
        ],
    )

    other_group_nginx_id = manager.archive_conversation(
        group_id="2002",
        started_at="2026-04-29T10:00:00+08:00",
        ended_at="2026-04-29T10:30:00+08:00",
        participants=[MemoryParticipant(user_id="42", nickname="另一个张三")],
        title="另一个群的 nginx 证书配置",
        summary="这个 archive 不能泄漏到 1001 群。",
        keywords=["nginx", "证书"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="另一个张三",
                content="另一个群里的 nginx 证书。",
                timestamp="2026-04-29T10:00:00+08:00",
            )
        ],
    )
    return {
        "nginx": nginx_id,
        "crawler": crawler_id,
        "minecraft": minecraft_id,
        "other_group_nginx": other_group_nginx_id,
    }


class _SemanticEmbeddingProvider:
    provider = "test-semantic"
    model = "test-vectors"
    dimensions = 4

    def embed_text(self, text: str) -> list[float]:
        folded = text.casefold()
        vector = [0.0, 0.0, 0.0, 0.0]
        if any(
            term in folded
            for term in ("nginx", "tls", "证书", "https", "certbot", "443", "反向代理")
        ):
            vector[0] += 1.0
        if any(
            term in folded
            for term in ("毕业论文", "采集", "爬虫", "crawler", "timeout", "httpx")
        ):
            vector[1] += 1.0
        if any(
            term in folded
            for term in ("minecraft", "mc", "whitelist", "白名单", "steve")
        ):
            vector[2] += 1.0
        if any(term in folded for term in ("rust", "clap", "cli", "命令行")):
            vector[3] += 1.0
        return normalize_vector(vector)


class _FailingEmbeddingProvider:
    provider = "openai-compatible"
    model = "failing"
    dimensions = 0

    def embed_text(self, text: str) -> list[float]:
        raise RuntimeError("embedding provider unavailable")


class _MismatchedEmbeddingProvider:
    provider = "test-mismatch"
    model = "test-vectors"
    dimensions = 2

    def embed_text(self, text: str) -> list[float]:
        folded = text.casefold().strip()
        if folded == "raretoken" or "semantic neighbor" in folded:
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_search_text_generates_cjk_ngrams_and_entities():
    terms = extract_search_terms("上周那个 nginx 反向代理 /etc/nginx/sites-enabled/app.conf 443", for_query=True)

    assert "nginx" in terms
    assert "反向" in terms
    assert "代理" in terms
    assert "app.conf" in terms
    assert "443" in terms
    assert "上周" in terms
    assert build_fts_query("nginx 反向代理") != ""


def test_memory_store_schema_version_and_logical_partitions(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    manager = MemoryManager(db_path)

    assert manager.store.schema_version == CURRENT_SCHEMA_VERSION
    with sqlite3.connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == CURRENT_SCHEMA_VERSION

    archive_id = manager.store.archives.add(
        group_id="1001",
        started_at="2026-05-04T20:00:00+08:00",
        ended_at="2026-05-04T20:10:00+08:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="逻辑分区测试",
        summary="测试 archive/profile/maintenance facade 仍共享同一个 SQLite 事务边界。",
        keywords=["SQLite", "facade"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="这个 archive 通过 archives 分区写入。",
                timestamp="2026-05-04T20:00:00+08:00",
            )
        ],
    )
    edit = manager.store.profiles.add_edit(
        group_id="1001",
        user_id="42",
        edit_type="add",
        content="喜欢清晰的数据分区",
        nickname="张三",
    )

    assert manager.store.archives.open(group_id="1001", archive_id=archive_id) is not None
    assert edit.id is not None
    assert manager.store.profiles.get(group_id="1001", user_id="42") is not None
    assert manager.store.maintenance.cleanup_old_data(archive_retention_days=0) == 0


def test_memory_search_finds_expected_archive_and_keeps_group_isolation(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db")
    ids = _seed_memory(manager)

    hits = manager.search(group_id="1001", query="上周那个服务器证书配置", max_results=3)

    assert [hit.archive.id for hit in hits][:2] == [ids["nginx"], ids["minecraft"]]
    assert "archive" in hits[0].matched_sources
    assert "messages" in hits[0].matched_sources
    assert "张三" in render_search_results(hits)
    assert "archive_id=" in render_search_results(hits)

    other_group_hits = manager.search(group_id="2002", query="nginx 证书", max_results=3)
    assert [hit.archive.id for hit in other_group_hits] == [ids["other_group_nginx"]]


def test_memory_time_search_finds_archives_by_range(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db")
    ids = _seed_memory(manager)

    hits = manager.time_search(
        group_id="1001",
        start_time="2026-04-27T00:00:00+08:00",
        end_time="2026-04-30T23:59:59+08:00",
        max_results=5,
    )

    assert [hit.archive.id for hit in hits] == [ids["crawler"], ids["nginx"]]
    assert all("time_range" in hit.matched_sources for hit in hits)
    assert all(hit.snippets for hit in hits)


def test_memory_search_uses_tool_events_and_user_filter(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db")
    ids = _seed_memory(manager)

    tool_hits = manager.search(group_id="1001", query="app.conf reload 443", max_results=5)
    assert tool_hits[0].archive.id == ids["nginx"]
    assert "tool_events" in tool_hits[0].matched_sources
    assert any("app.conf" in snippet.text or "reload" in snippet.text for snippet in tool_hits[0].snippets)

    zhangsan_hits = manager.search(group_id="1001", query="nginx 配置", user_id="42")
    assert [hit.archive.id for hit in zhangsan_hits] == [ids["nginx"]]

    lisi_hits = manager.search(group_id="1001", query="nginx 配置", user_id="7")
    assert lisi_hits == []


def test_memory_time_search_uses_user_filter(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db")
    ids = _seed_memory(manager)

    hits = manager.time_search(
        group_id="1001",
        start_time="2026-04-27T00:00:00+08:00",
        end_time="2026-04-30T23:59:59+08:00",
        user_id="7",
    )

    assert [hit.archive.id for hit in hits] == [ids["crawler"]]


def test_memory_search_documents_lexical_gap_without_embedding(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db")
    ids = _seed_memory(manager)

    hits = manager.search(group_id="1001", query="HTTPS certbot 部署入口", max_results=3)
    assert hits == []

    exact_hits = manager.search(group_id="1001", query="TLS 证书", max_results=3)
    assert exact_hits[0].archive.id == ids["nginx"]


def test_memory_search_embedding_recovers_semantic_archive(tmp_path: Path):
    manager = MemoryManager(
        tmp_path / "memory.db",
        embedding_provider=_SemanticEmbeddingProvider(),
    )
    ids = _seed_memory(manager)

    hits = manager.search(group_id="1001", query="HTTPS certbot 部署入口", max_results=3)

    assert hits
    assert hits[0].archive.id == ids["nginx"]
    assert "embedding" in hits[0].matched_sources


def test_memory_search_keeps_exact_lexical_hit_ahead_of_embedding_only_candidate(
    tmp_path: Path,
):
    manager = MemoryManager(
        tmp_path / "memory.db",
        embedding_provider=_MismatchedEmbeddingProvider(),
    )
    exact_id = manager.archive_conversation(
        group_id="1001",
        started_at="2026-05-17T10:00:00+08:00",
        ended_at="2026-05-17T10:01:00+08:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="Exact RARETOKEN answer",
        summary="This archive contains RARETOKEN exactly.",
        keywords=["RARETOKEN"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="RARETOKEN exact answer lives here.",
                timestamp="2026-05-17T10:00:00+08:00",
            )
        ],
    )
    semantic_neighbor_id = manager.archive_conversation(
        group_id="1001",
        started_at="2026-05-17T10:02:00+08:00",
        ended_at="2026-05-17T10:03:00+08:00",
        participants=[MemoryParticipant(user_id="7", nickname="李四")],
        title="Semantic neighbor",
        summary="This archive is intentionally unrelated to the exact token.",
        keywords=["neighbor"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="7",
                nickname="李四",
                content="unrelated content",
                timestamp="2026-05-17T10:02:00+08:00",
            )
        ],
    )

    hits = manager.search(group_id="1001", query="RARETOKEN", max_results=5)

    assert [hit.archive.id for hit in hits[:2]] == [exact_id, semantic_neighbor_id]
    assert "embedding" not in hits[0].matched_sources
    assert hits[0].score > hits[1].score


def test_memory_embedding_index_migrates_existing_json_vectors(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    old_manager = MemoryManager(db_path)
    ids = _seed_memory(old_manager)
    provider = _SemanticEmbeddingProvider()

    with sqlite3.connect(db_path) as conn:
        for archive_id in ids.values():
            row = conn.execute(
                """
                SELECT group_id, title, summary, keywords, created_at
                FROM conversation_archives
                WHERE id = ?
                """,
                (archive_id,),
            ).fetchone()
            text = "\n".join([row[1], row[2], " ".join(json.loads(row[3]))])
            vector = provider.embed_text(text)
            conn.execute(
                """
                INSERT OR REPLACE INTO archive_embeddings (
                    archive_id,
                    group_id,
                    provider,
                    model,
                    dimension,
                    vector,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    archive_id,
                    row[0],
                    provider.provider,
                    provider.model,
                    len(vector),
                    json.dumps(vector),
                    row[4],
                ),
            )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    manager = MemoryManager(db_path, embedding_provider=provider)
    hits = manager.search(group_id="1001", query="HTTPS certbot 部署入口", max_results=3)

    assert hits
    assert hits[0].archive.id == ids["nginx"]
    assert "embedding" in hits[0].matched_sources

    with sqlite3.connect(db_path) as conn:
        load_sqlite_vec(conn)
        count = conn.execute("SELECT count(*) FROM archive_embedding_vec").fetchone()[0]

    assert count == len(ids)


def test_embedding_provider_factory_supports_openai_compatible():
    provider = build_embedding_provider(
        provider="openai-compatible",
        model="fake-embedding-model",
        api_key="sk-test",
        base_url="https://example.test",
    )

    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.provider == "openai-compatible"
    assert provider.model == "fake-embedding-model"
    assert provider.base_url == "https://example.test/v1"
    assert provider.user_agent == DEFAULT_EMBEDDING_USER_AGENT
    assert (
        normalize_openai_embedding_base_url("https://example.test/v1/")
        == "https://example.test/v1"
    )


def test_memory_manager_uses_openai_compatible_when_embedding_model_is_configured(
    tmp_path: Path,
):
    config = BampiChatConfig(
        bampi_memory_db_path=str(tmp_path / "memory.db"),
        bampi_memory_embedding_enabled=True,
        bampi_memory_embedding_model="fake-embedding-model",
        bampi_api_key="sk-test",
        bampi_base_url="https://example.test",
    )

    manager = MemoryManager.from_config(config)

    assert isinstance(manager._embedding_provider, OpenAICompatibleEmbeddingProvider)
    assert manager._embedding_provider.base_url == "https://example.test/v1"


def test_openai_compatible_embedding_requires_model():
    with pytest.raises(ValueError, match="bampi_memory_embedding_model"):
        build_embedding_provider(provider="openai-compatible", model="")


def test_memory_embedding_failure_falls_back_to_text_search(tmp_path: Path):
    manager = MemoryManager(
        tmp_path / "memory.db",
        embedding_provider=_FailingEmbeddingProvider(),
    )

    archive_id = manager.archive_conversation(
        group_id="1001",
        started_at="2026-05-04T20:00:00+08:00",
        ended_at="2026-05-04T20:10:00+08:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="nginx TLS 配置",
        summary="讨论 nginx 证书配置。",
        keywords=["nginx", "TLS", "证书"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="nginx 证书怎么配？",
                timestamp="2026-05-04T20:00:00+08:00",
            )
        ],
    )

    hits = manager.search(group_id="1001", query="nginx 证书", max_results=3)

    assert hits
    assert hits[0].archive.id == archive_id
    assert "embedding" not in hits[0].matched_sources


def test_memory_open_returns_compact_transcript_and_tools(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db")
    ids = _seed_memory(manager)
    opened = manager.open_archive(group_id="1001", archive_id=ids["nginx"], mode="compact")

    assert opened is not None
    assert "配置 nginx 反向代理" in opened.text
    assert "工具事件预览" in opened.text
    assert "/etc/nginx/sites-enabled/app.conf" in opened.text

    second_message_id = opened.messages[1].id
    transcript = manager.open_archive(
        group_id="1001",
        archive_id=ids["nginx"],
        mode="transcript",
        around_message_id=second_message_id,
        before=0,
        after=0,
    )

    assert transcript is not None
    assert "我先看 /etc/nginx/sites-enabled/app.conf" in transcript.text
    assert "上周那个 nginx" not in transcript.text

    tools = manager.open_archive(
        group_id="1001",
        archive_id=ids["nginx"],
        mode="tools",
        include_tool_results=True,
    )
    assert tools is not None
    assert "nginx -t" in tools.text
    assert "reload ok" in tools.text


@pytest.mark.asyncio
async def test_memory_tools_render_search_and_open_results(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db")
    ids = _seed_memory(manager)
    tools = create_agent_tools(
        BampiChatConfig(bampi_memory_db_path=str(tmp_path / "memory.db")),
        str(tmp_path / "workspace"),
        group_id="1001",
        memory_manager=manager,
    )
    search_tool = next(tool for tool in tools if tool.name == "memory_search")
    time_search_tool = next(tool for tool in tools if tool.name == "memory_time_search")
    open_tool = next(tool for tool in tools if tool.name == "memory_open")

    search_result = await search_tool.execute("call-1", {"query": "nginx 证书", "max_results": 2})
    assert f"archive_id={ids['nginx']}" in search_result.content[0].text
    assert search_result.details["archives"][0]["archive"]["id"] == ids["nginx"]

    time_search_result = await time_search_tool.execute(
        "call-2",
        {
            "start_time": "2026-04-27T00:00:00+08:00",
            "end_time": "2026-04-30T23:59:59+08:00",
            "max_results": 2,
        },
    )
    assert f"archive_id={ids['crawler']}" in time_search_result.content[0].text
    assert time_search_result.details["archives"][0]["archive"]["id"] == ids["crawler"]

    open_result = await open_tool.execute("call-3", {"archive_id": ids["nginx"], "mode": "compact"})
    assert "TLS 证书" in open_result.content[0].text
    assert open_result.details["archive"]["archive"]["id"] == ids["nginx"]


def test_system_prompt_mentions_memory_tools():
    prompt = build_system_prompt(BampiChatConfig(), ["memory_search", "memory_time_search", "memory_open", "memory_manage"])

    assert "memory_search" in prompt
    assert "memory_time_search" in prompt
    assert "memory_open" in prompt
    assert "memory_manage" in prompt
    assert "nginx 配置 证书" in prompt
    assert BampiChatConfig(bampi_memory_storage_mode=" single ").bampi_memory_storage_mode == "single"


def test_archive_session_extracts_user_metadata_and_tool_events(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db", archive_min_messages=2)
    user = UserMessage(content=[TextContent(text="sender_name: 王五\nmessage_text: 帮我检查 nginx reload")])
    assistant_tool_call = AssistantMessage(
        content=[ToolCall(id="call-1", name="bash", arguments={"command": "nginx -t"})],
        stop_reason=StopReason.TOOL_USE,
    )
    tool_result = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="bash",
        content=[TextContent(text="nginx: configuration file test is successful")],
    )
    assistant = AssistantMessage(content=[TextContent(text="检查通过，可以 reload。")])

    archive_id = manager.archive_session(
        group_id="1001",
        messages=[user, assistant_tool_call, tool_result, assistant],
        user_turns=[MemoryUserTurn(user_id="55", nickname="王五", timestamp=user.timestamp)],
    )

    assert archive_id is not None
    opened = manager.open_archive(group_id="1001", archive_id=archive_id, mode="full")
    assert opened is not None
    assert opened.messages[0].user_id == "55"
    assert opened.messages[0].nickname == "王五"
    assert "nginx -t" in opened.text

    hits = manager.search(group_id="1001", query="nginx reload 王五")
    assert hits[0].archive.id == archive_id


@pytest.mark.asyncio
async def test_memory_manage_defaults_to_current_speaker_and_context_injects(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db")
    tools = create_agent_tools(
        BampiChatConfig(bampi_memory_db_path=str(tmp_path / "memory.db")),
        str(tmp_path / "workspace"),
        group_id="1001",
        memory_manager=manager,
        memory_current_user_provider=lambda: ("42", "张三"),
    )
    manage_tool = next(tool for tool in tools if tool.name == "memory_manage")

    result = await manage_tool.execute(
        "call-1",
        {"action": "add", "content": "喜欢用 Rust 写命令行工具"},
    )

    assert "user_id=42" in result.content[0].text
    context = manager.get_memory_context_for_turn(
        group_id="1001",
        current_user_id="42",
        current_nickname="张三",
    )
    assert "喜欢用 Rust 写命令行工具" in context
    assert "近期补充" in context


@pytest.mark.asyncio
async def test_profile_generation_scan_uses_llm_and_consolidates_pending_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = MemoryManager(
        tmp_path / "memory.db",
        profile_session_threshold=1,
        profile_max_tokens=300,
    )
    manager.archive_conversation(
        group_id="1001",
        started_at="2026-05-02T20:00:00+08:00",
        ended_at="2026-05-02T20:20:00+08:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="讨论 Rust CLI 工具",
        summary="张三讨论用 Rust 写命令行工具，并关注 clap 参数解析。",
        keywords=["Rust", "CLI", "clap"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="我最近喜欢用 Rust 写小工具。",
                timestamp="2026-05-02T20:00:00+08:00",
            ),
            MemoryMessage(
                role="assistant",
                content="可以用 clap 管理参数。",
                timestamp="2026-05-02T20:10:00+08:00",
            ),
        ],
    )
    manager.add_profile_edit(
        group_id="1001",
        user_id="42",
        edit_type="add",
        content="偏好简洁的命令行界面",
        nickname="张三",
    )

    calls: list[tuple[str, str]] = []

    async def fake_complete_simple(_model, ctx, _options):
        calls.append((ctx.system_prompt, ctx.messages[0].content[0].text))
        return AssistantMessage(
            api="test",
            provider="test",
            model="model-a",
            content=[
                TextContent(
                    text=(
                        "**Work context**\n"
                        "{nickname} 常讨论 Rust CLI 工具和参数解析。\n\n"
                        "**Personal context**\n"
                        "{nickname} 偏好简洁的命令行界面。\n\n"
                        "**Top of mind**\n"
                        "2026-05-02 参与了「讨论 Rust CLI 工具」：张三讨论用 Rust 写命令行工具，并关注 clap 参数解析。\n\n"
                        "**Brief history**\n"
                        "*Recent months*\n"
                        "近期经常参与的话题包括：Rust、CLI、clap。\n\n"
                        "*Earlier context*\n"
                        "暂无更早期的可确认上下文。\n\n"
                        "*Long-term background*\n"
                        "暂无长期背景。"
                    )
                )
            ],
        )

    stream_module = importlib.import_module("bampy.ai.stream")
    monkeypatch.setattr(stream_module, "complete_simple", fake_complete_simple)

    assert await manager.run_profile_generation_scan_async(model=object(), api_key="sk-test") == 1
    assert calls
    assert "群聊长期记忆画像生成器" in calls[0][0]
    assert "<pending_edits>" in calls[0][1]
    assert "偏好简洁的命令行界面" in calls[0][1]

    context = manager.get_memory_context_for_turn(
        group_id="1001",
        current_user_id="42",
        current_nickname="张三",
    )

    assert "**Work context**" in context
    assert "Rust、CLI、clap" in context
    assert "偏好简洁的命令行界面" in context
    assert "近期补充" not in context


@pytest.mark.asyncio
async def test_profile_generation_scan_rejects_legacy_llm_profile_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = MemoryManager(
        tmp_path / "memory.db",
        profile_session_threshold=1,
        profile_max_tokens=300,
    )
    manager.archive_conversation(
        group_id="1001",
        started_at="2026-05-02T20:00:00+08:00",
        ended_at="2026-05-02T20:20:00+08:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="讨论 Rust CLI 工具",
        summary="张三讨论用 Rust 写命令行工具。",
        keywords=["Rust", "CLI"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="我最近喜欢用 Rust 写小工具。",
                timestamp="2026-05-02T20:00:00+08:00",
            )
        ],
    )

    async def fake_complete_simple(_model, _ctx, _options):
        return AssistantMessage(
            api="test",
            provider="test",
            model="model-a",
            content=[
                TextContent(
                    text=(
                        "基本信息\n"
                        "{nickname} 是本群成员。\n\n"
                        "兴趣与话题\n"
                        "近期经常参与的话题包括：Rust、CLI。"
                    )
                )
            ],
        )

    stream_module = importlib.import_module("bampy.ai.stream")
    monkeypatch.setattr(stream_module, "complete_simple", fake_complete_simple)

    assert await manager.run_profile_generation_scan_async(model=object(), api_key="sk-test") == 0
    context = manager.get_memory_context_for_turn(
        group_id="1001",
        current_user_id="42",
        current_nickname="张三",
    )

    assert context == ""


@pytest.mark.asyncio
async def test_profile_generation_scan_keeps_pending_data_when_llm_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = MemoryManager(
        tmp_path / "memory.db",
        profile_session_threshold=1,
        profile_max_tokens=300,
    )
    manager.archive_conversation(
        group_id="1001",
        started_at="2026-05-02T20:00:00+08:00",
        ended_at="2026-05-02T20:20:00+08:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="讨论 Rust CLI 工具",
        summary="张三讨论用 Rust 写命令行工具。",
        keywords=["Rust", "CLI"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="我最近喜欢用 Rust 写小工具。",
                timestamp="2026-05-02T20:00:00+08:00",
            )
        ],
    )
    manager.add_profile_edit(
        group_id="1001",
        user_id="42",
        edit_type="add",
        content="偏好简洁的命令行界面",
        nickname="张三",
    )

    async def fake_complete_simple(_model, _ctx, _options):
        return AssistantMessage(
            api="test",
            provider="test",
            model="model-a",
            stop_reason=StopReason.ERROR,
            error_message="provider failed",
        )

    stream_module = importlib.import_module("bampy.ai.stream")
    monkeypatch.setattr(stream_module, "complete_simple", fake_complete_simple)

    assert await manager.run_profile_generation_scan_async(model=object(), api_key="sk-test") == 0
    context = manager.get_memory_context_for_turn(
        group_id="1001",
        current_user_id="42",
        current_nickname="张三",
    )

    assert "偏好简洁的命令行界面" in context
    assert "近期补充" in context
    assert "**Work context**" not in context
    assert manager.store.profiles.pending_edits(group_id="1001", user_id="42")


def test_profile_delete_edit_hides_matching_profile_lines(tmp_path: Path):
    manager = MemoryManager(tmp_path / "memory.db", profile_session_threshold=1)
    manager.archive_conversation(
        group_id="1001",
        started_at="2026-05-02T20:00:00+08:00",
        ended_at="2026-05-02T20:20:00+08:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="讨论原神和 Rust",
        summary="张三聊到原神和 Rust。",
        keywords=["原神", "Rust"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="我以前玩原神，现在写 Rust。",
                timestamp="2026-05-02T20:00:00+08:00",
            )
        ],
    )
    manager.run_profile_generation_scan()
    manager.add_profile_edit(
        group_id="1001",
        user_id="42",
        edit_type="delete",
        content="原神",
        nickname="张三",
    )

    context = manager.get_memory_context_for_turn(
        group_id="1001",
        current_user_id="42",
        current_nickname="张三",
    )

    assert "不要再使用或提及：原神" in context
    profile_part = context.split("[已删除或失效的记忆]", 1)[0]
    assert "原神" not in profile_part


def test_embedding_cleanup_and_user_delete_paths(tmp_path: Path):
    config = BampiChatConfig(
        bampi_memory_db_path=str(tmp_path / "memory.db"),
        bampi_memory_embedding_enabled=True,
        bampi_memory_archive_retention_days=1000,
    )
    manager = MemoryManager.from_config(config)
    old_archive_id = manager.archive_conversation(
        group_id="1001",
        started_at="2000-01-01T00:00:00+00:00",
        ended_at="2000-01-01T00:10:00+00:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="很旧的 nginx 配置",
        summary="旧会话应该被 retention 清理。",
        keywords=["nginx"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="旧 nginx 配置",
                timestamp="2000-01-01T00:00:00+00:00",
            )
        ],
    )
    fresh_archive_id = manager.archive_conversation(
        group_id="1001",
        started_at="2026-05-03T00:00:00+00:00",
        ended_at="2026-05-03T00:10:00+00:00",
        participants=[MemoryParticipant(user_id="42", nickname="张三")],
        title="Rust CLI 参数解析",
        summary="讨论 clap 和命令行参数解析。",
        keywords=["Rust", "CLI", "clap"],
        messages=[
            MemoryMessage(
                role="user",
                user_id="42",
                nickname="张三",
                content="Rust clap 参数怎么设计？",
                timestamp="2026-05-03T00:00:00+00:00",
            )
        ],
    )

    hits = manager.search(group_id="1001", query="Rust clap 参数", max_results=5)
    assert hits[0].archive.id == fresh_archive_id
    assert "embedding" in hits[0].matched_sources

    assert manager.cleanup_old_data() == 1
    assert manager.open_archive(group_id="1001", archive_id=old_archive_id) is None
    assert manager.open_archive(group_id="1001", archive_id=fresh_archive_id) is not None

    manager.add_profile_edit(
        group_id="1001",
        user_id="42",
        edit_type="add",
        content="喜欢 Rust CLI",
        nickname="张三",
    )
    delete_result = manager.delete_user_memory(group_id="1001", user_id="42")
    assert delete_result["messages_deleted"] == 1
    assert manager.get_memory_context_for_turn(
        group_id="1001",
        current_user_id="42",
        current_nickname="张三",
    ) == ""
