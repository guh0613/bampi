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
