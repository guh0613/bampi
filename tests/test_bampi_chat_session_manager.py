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
        assert (tmp_path / "workspace" / "inbox").exists()
        assert (tmp_path / "workspace" / "outbox").exists()
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
    managed.session.session_manager.append_message({"role": "assistant", "content": "hello"})
    session_file = Path(managed.session.session_manager.session_file or "")

    try:
        assert session_file.exists()
        await manager.complete_interaction("1001")
        await asyncio.sleep(1.1)

        assert not session_file.exists()
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
    managed.session.session_manager.append_message({"role": "assistant", "content": "hello"})
    session_file = Path(managed.session.session_manager.session_file or "")

    try:
        assert session_file.exists()
        managed.last_used_at -= 120
        await manager.close_idle()

        assert not session_file.exists()
        recreated = await manager.get_or_create("1001")
        assert recreated.session is not managed.session
        assert recreated.session.messages == []
    finally:
        await manager.close_all()
