from __future__ import annotations

from pathlib import Path

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


def test_group_session_manager_uses_builtin_ollama_model_metadata(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="ollama",
        bampi_model_id="gemini-3-flash-ol",
        bampi_base_url="https://ollama.example.com",
    )

    manager = GroupSessionManager(config)

    model = manager._build_model()

    assert model.provider == "ollama"
    assert model.api == "ollama-responses"
    assert model.id == "gemini-3-flash-ol"
    assert model.reasoning is True
    assert model.context_window == 1_048_576
    assert model.max_tokens == 65_536
    assert model.base_url == "https://ollama.example.com"
