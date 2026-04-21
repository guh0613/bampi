from __future__ import annotations

from pathlib import Path

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.session_manager import GroupSessionManager


def test_group_session_manager_accepts_custom_ollama_compatible_model(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="ollama",
        bampi_model_id="kimi-k2.5",
        bampi_base_url="https://api.sanuki.cn",
    )

    manager = GroupSessionManager(config)

    model = manager._build_model()

    assert model.provider == "ollama"
    assert model.api == "ollama-responses"
    assert model.id == "kimi-k2.5"
    assert model.base_url == "https://api.sanuki.cn"


def test_group_session_manager_defaults_unknown_custom_provider_to_chat_completions(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
        bampi_base_url="https://api.moonshot.cn/v1",
    )

    manager = GroupSessionManager(config)

    model = manager._build_model()

    assert model.provider == "moonshot"
    assert model.api == "openai-completions"
    assert model.id == "kimi-k2.6"
    assert model.base_url == "https://api.moonshot.cn/v1"


def test_group_session_manager_accepts_explicit_chat_completions_api_alias(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
        bampi_model_api="chat-completions",
        bampi_base_url="https://api.moonshot.cn/v1",
    )

    manager = GroupSessionManager(config)

    model = manager._build_model()

    assert model.provider == "moonshot"
    assert model.api == "openai-completions"
    assert model.id == "kimi-k2.6"
    assert model.base_url == "https://api.moonshot.cn/v1"


def test_group_session_manager_uses_builtin_ollama_model_metadata(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="ollama",
        bampi_model_id="gemini-3-flash",
        bampi_base_url="https://ollama.example.com",
    )

    manager = GroupSessionManager(config)

    model = manager._build_model()

    assert model.provider == "ollama"
    assert model.api == "ollama-responses"
    assert model.id == "gemini-3-flash"
    assert model.reasoning is True
    assert model.context_window == 1_048_576
    assert model.max_tokens == 65_536
    assert model.base_url == "https://ollama.example.com"


def test_group_session_manager_allows_overriding_builtin_model_api(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="openai",
        bampi_model_id="gpt-5-mini",
        bampi_model_api="chat-completions",
        bampi_base_url="https://gateway.example.com/v1",
    )

    manager = GroupSessionManager(config)

    model = manager._build_model()

    assert model.provider == "openai"
    assert model.api == "openai-completions"
    assert model.id == "gpt-5-mini"
    assert model.reasoning is True
    assert model.context_window == 400_000
    assert model.max_tokens == 128_000
    assert model.base_url == "https://gateway.example.com/v1"


@pytest.mark.asyncio
async def test_group_session_manager_resolves_custom_provider_api_key_from_provider_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-secret")
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
    )

    manager = GroupSessionManager(config)

    api_key = await manager._resolve_api_key("moonshot")

    assert api_key == "moonshot-secret"


@pytest.mark.asyncio
async def test_group_session_manager_resolves_custom_openai_compatible_api_key_from_openai_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-compatible-secret")
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
    )

    manager = GroupSessionManager(config)

    api_key = await manager._resolve_api_key("moonshot")

    assert api_key == "openai-compatible-secret"
