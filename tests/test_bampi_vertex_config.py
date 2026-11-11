from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.session_manager import GroupSessionManager
from bampi.plugins.showdown_battle.ai_opponent import AIModelSettings, BattleAIOpponent
from bampi.plugins.showdown_battle.config import ShowdownBattleConfig


@pytest.mark.parametrize("api", ["google-vertex", "vertex", "vertex-ai", "vertexai", "google_vertex"])
def test_vertex_api_configuration_aliases(api: str):
    config = BampiChatConfig(bampi_model_api=api, bampi_memory_model_api=api)
    assert config.bampi_model_api == config.bampi_memory_model_api == "google-vertex"
    assert ShowdownBattleConfig(showdown_battle_ai_model_api=api).showdown_battle_ai_model_api == "google-vertex"


def _manager(tmp_path: Path, *, model_id: str = "gemini-4-argon", **values) -> GroupSessionManager:
    return GroupSessionManager(BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="google-vertex",
        bampi_model_id=model_id,
        **values,
    ))


@pytest.mark.parametrize("api", ["auto", "google-vertex"])
@pytest.mark.parametrize("model_id", [
    "gemini-4-carbon", "gemini-4-argon", "gemini-3.8-flash", "gemini-3.7-flash",
    "gemini-3.6-flash", "gemini-3.5-flash-lite",
])
def test_vertex_main_and_memory_models_use_vertex_channel(tmp_path: Path, api: str, model_id: str):
    manager = _manager(tmp_path, model_id=model_id, bampi_model_api=api)
    for model in (manager._build_model(), manager._build_memory_model(), manager._build_memory_profile_model()):
        assert model.id == model_id
        assert model.api == "google-vertex"
        assert model.base_url == ""
        assert model.reasoning is True
        assert "image" in model.input_types


def test_unregistered_vertex_model_is_not_misclassified_as_developer_api(tmp_path: Path):
    manager = _manager(tmp_path)
    model = manager._build_custom_model(model_id="my-vertex-model")
    assert model.api == "google-vertex"


async def test_vertex_adc_does_not_inherit_developer_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GOOGLE_VERTEX_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "developer-only")
    monkeypatch.setenv("GEMINI_API_KEY", "developer-only")
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_resolve_nonebot_config_value", lambda _: None)
    assert await manager._resolve_api_key("google-vertex") is None


async def test_vertex_express_uses_its_own_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "vertex-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "developer-only")
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_resolve_nonebot_config_value", lambda _: None)
    assert await manager._resolve_api_key("google-vertex") == "vertex-key"


@pytest.mark.parametrize("model_id", [
    "gemini-4-carbon", "gemini-4-argon", "gemini-3.8-flash", "gemini-3.7-flash",
    "gemini-3.6-flash", "gemini-3.5-flash-lite",
])
def test_showdown_inherits_vertex_model_without_developer_key(monkeypatch, model_id):
    monkeypatch.delenv("GOOGLE_VERTEX_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "developer-only")
    main = SimpleNamespace(
        bampi_model_provider="google-vertex", bampi_model_id=model_id,
        bampi_model_api="auto", bampi_api_key="", google_api_key="developer-only",
    )
    settings = AIModelSettings.from_config(ShowdownBattleConfig(), main)
    assert settings.api_key == ""
    assert BattleAIOpponent._build_model(settings).api == "google-vertex"
    assert BattleAIOpponent._resolve_api_key(settings) is None
