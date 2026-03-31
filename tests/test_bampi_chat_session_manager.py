from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.session_manager import GroupSessionManager


@pytest.mark.asyncio
async def test_group_session_manager_reuses_session(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
    )
    manager = GroupSessionManager(config)

    first = await manager.get_or_create("1001")
    second = await manager.get_or_create("1001")
    third = await manager.get_or_create("1002")

    try:
        assert first.session is second.session
        assert first.session is not third.session
        first_workspace = Path(manager.workspace_dir_for_group("1001"))
        third_workspace = Path(manager.workspace_dir_for_group("1002"))
        assert first_workspace != third_workspace
        assert (first_workspace / "inbox").exists()
        assert (first_workspace / "outbox").exists()
        assert (third_workspace / "inbox").exists()
        assert (third_workspace / "outbox").exists()
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_group_session_manager_reserve_interaction_enforces_owner(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
    )
    manager = GroupSessionManager(config)

    try:
        first = await manager.reserve_interaction("1001", "42")
        assert first.action == "start"

        status = await manager.inspect_interaction("1001")
        assert status.is_active is True
        assert status.active_user_id == "42"
        assert status.is_streaming is False

        first.managed.session.agent.state.is_streaming = True
        same_owner = await manager.reserve_interaction("1001", "42")
        other_user = await manager.reserve_interaction("1001", "7")

        assert same_owner.action == "steer"
        assert other_user.action == "busy"

        await manager.complete_interaction("1001")
        cleared = await manager.inspect_interaction("1001")
        assert cleared.is_active is False
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_group_session_manager_releases_idle_sessions(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_session_idle_ttl_seconds=0,
    )
    manager = GroupSessionManager(config)
    first = await manager.get_or_create("1001")
    await manager.close_all()
    second = await manager.get_or_create("1001")
    try:
        assert first.session is not second.session
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_group_session_manager_clears_idle_session_history(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_session_idle_ttl_seconds=1,
    )
    manager = GroupSessionManager(config)
    managed = await manager.get_or_create("1001")
    workspace = Path(manager.workspace_dir_for_group("1001"))
    runtime_file = workspace / "scratch.txt"
    inbox_file = workspace / "inbox" / "upload.txt"
    outbox_file = workspace / "outbox" / "result.txt"
    installed_skill = workspace / ".agents" / "skills" / "docs-search" / "SKILL.md"
    runtime_file.write_text("temporary", encoding="utf-8")
    inbox_file.write_text("incoming", encoding="utf-8")
    outbox_file.write_text("generated", encoding="utf-8")
    installed_skill.parent.mkdir(parents=True, exist_ok=True)
    installed_skill.write_text("# Docs Search\n", encoding="utf-8")
    managed.session.session_manager.append_message({"role": "assistant", "content": "hello"})
    session_file = Path(managed.session.session_manager.session_file or "")

    try:
        assert session_file.exists()
        await manager.complete_interaction("1001")
        await asyncio.sleep(1.1)

        assert not session_file.exists()
        assert not runtime_file.exists()
        assert not inbox_file.exists()
        assert not outbox_file.exists()
        assert installed_skill.exists()
        assert (workspace / ".agents" / "builtin-skills" / "docx" / "SKILL.md").exists()
        assert (workspace / "inbox").is_dir()
        assert (workspace / "outbox").is_dir()
        recreated = await manager.get_or_create("1001")
        assert recreated.session is not managed.session
        assert recreated.session.messages == []
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_group_session_manager_close_idle_clears_stale_session_history(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_session_idle_ttl_seconds=60,
    )
    manager = GroupSessionManager(config)
    managed = await manager.get_or_create("1001")
    workspace = Path(manager.workspace_dir_for_group("1001"))
    runtime_dir = workspace / "artifacts"
    runtime_file = runtime_dir / "notes.txt"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text("temporary", encoding="utf-8")
    managed.session.session_manager.append_message({"role": "assistant", "content": "hello"})
    session_file = Path(managed.session.session_manager.session_file or "")

    try:
        assert session_file.exists()
        managed.last_used_at -= 120
        await manager.close_idle()

        assert not session_file.exists()
        assert not runtime_file.exists()
        assert (workspace / "inbox").is_dir()
        assert (workspace / "outbox").is_dir()
        recreated = await manager.get_or_create("1001")
        assert recreated.session is not managed.session
        assert recreated.session.messages == []
    finally:
        await manager.close_all()


@pytest.mark.asyncio
async def test_group_session_manager_release_preserves_workspace_files(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
    )
    manager = GroupSessionManager(config)
    managed = await manager.get_or_create("1001")
    workspace_file = Path(manager.workspace_dir_for_group("1001")) / ".agents" / "skills" / "docs-search" / "SKILL.md"
    workspace_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_file.write_text("# Docs Search\n", encoding="utf-8")
    managed.session.session_manager.append_message({"role": "assistant", "content": "hello"})
    session_file = Path(managed.session.session_manager.session_file or "")

    try:
        assert session_file.exists()
        await manager.release("1001")

        assert workspace_file.exists()
        assert session_file.exists()
    finally:
        await manager.close_all()
