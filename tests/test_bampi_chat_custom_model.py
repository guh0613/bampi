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
    assert model.input_types == ["text"]


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
    assert model.input_types == ["text"]


def test_group_session_manager_accepts_multimodal_custom_model(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6-vision",
        bampi_model_input_types=["text", "image"],
    )

    model = GroupSessionManager(config)._build_model()

    assert model.input_types == ["text", "image"]


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
    assert model.input_types == ["text", "image"]


def test_group_session_manager_allows_overriding_builtin_model_input_types(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="openai",
        bampi_model_id="gpt-5-mini",
        bampi_model_input_types=["text"],
    )

    model = GroupSessionManager(config)._build_model()

    assert model.input_types == ["text"]


def test_model_input_types_normalize_comma_separated_config():
    config = BampiChatConfig(bampi_model_input_types=" Text, IMAGE, text ")

    assert config.bampi_model_input_types == ["text", "image"]


def test_model_input_types_normalize_json_config():
    config = BampiChatConfig(bampi_model_input_types='["text", "image"]')

    assert config.bampi_model_input_types == ["text", "image"]


@pytest.mark.parametrize(
    "input_types",
    [[], ["image"], ["text", "audio"]],
)
def test_model_input_types_reject_invalid_config(input_types: list[str]):
    with pytest.raises(ValueError, match="bampi_model_input_types"):
        BampiChatConfig(bampi_model_input_types=input_types)


def test_text_only_model_session_does_not_advertise_read_image_support(
    tmp_path: Path,
):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
        bampi_browser_enabled=False,
        bampi_memory_enabled=False,
        bampi_service_enabled=False,
        bampi_schedule_enabled=False,
    )
    manager = GroupSessionManager(config)

    session = manager._create_agent_session(
        "1001",
        persist=False,
        session_file=None,
        include_schedule=False,
    )
    read_tool = next(tool for tool in session.get_all_tools() if tool.name == "read")

    assert "image" not in read_tool.description.lower()


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


def test_group_session_manager_memory_model_falls_back_to_main_model(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
        bampi_model_api="chat-completions",
        bampi_base_url="https://api.moonshot.cn/v1",
    )

    manager = GroupSessionManager(config)

    main_model = manager._build_model()
    memory_model = manager._build_memory_model()

    assert memory_model.provider == main_model.provider
    assert memory_model.id == main_model.id
    assert memory_model.api == main_model.api
    assert memory_model.base_url == main_model.base_url


def test_group_session_manager_memory_model_overrides_fields_independently(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
        bampi_model_api="chat-completions",
        bampi_base_url="https://api.moonshot.cn/v1",
        bampi_memory_model_provider="ollama",
        bampi_memory_model_id="qwen3",
        bampi_memory_base_url="https://ollama.example.com",
    )

    manager = GroupSessionManager(config)

    memory_model = manager._build_memory_model()

    assert memory_model.provider == "ollama"
    assert memory_model.id == "qwen3"
    assert memory_model.api == "ollama-responses"
    assert memory_model.base_url == "https://ollama.example.com"


def test_group_session_manager_memory_model_auto_api_does_not_inherit_main_api(
    tmp_path: Path,
):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="anthropic",
        bampi_model_id="claude-sonnet-4",
        bampi_model_api="anthropic-messages",
        bampi_memory_model_provider="moonshot",
        bampi_memory_model_id="kimi-k2.6",
        bampi_memory_base_url="https://api.moonshot.cn/v1",
    )

    manager = GroupSessionManager(config)

    memory_model = manager._build_memory_model()

    assert memory_model.provider == "moonshot"
    assert memory_model.id == "kimi-k2.6"
    assert memory_model.api == "openai-completions"
    assert memory_model.api != "anthropic-messages"


def test_group_session_manager_memory_base_url_only_keeps_main_api(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="openai",
        bampi_model_id="gpt-5-mini",
        bampi_model_api="chat-completions",
        bampi_base_url="https://api.openai.com/v1",
        bampi_memory_base_url="https://gateway.example.com/v1",
    )

    manager = GroupSessionManager(config)

    memory_model = manager._build_memory_model()

    assert memory_model.provider == "openai"
    assert memory_model.id == "gpt-5-mini"
    assert memory_model.api == "openai-completions"
    assert memory_model.base_url == "https://gateway.example.com/v1"


def test_group_session_manager_memory_model_api_override(tmp_path: Path):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
        bampi_model_api="chat-completions",
        bampi_base_url="https://api.moonshot.cn/v1",
        bampi_memory_model_api="openai-responses",
    )

    manager = GroupSessionManager(config)

    memory_model = manager._build_memory_model()

    assert memory_model.provider == "moonshot"
    assert memory_model.id == "kimi-k2.6"
    assert memory_model.api == "openai-responses"
    assert memory_model.base_url == "https://api.moonshot.cn/v1"


@pytest.mark.asyncio
async def test_group_session_manager_memory_api_key_prefers_memory_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-secret")
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
        bampi_api_key="main-secret",
        bampi_memory_api_key="memory-secret",
    )

    manager = GroupSessionManager(config)

    api_key = await manager._resolve_memory_api_key("moonshot")

    assert api_key == "memory-secret"


@pytest.mark.asyncio
async def test_group_session_manager_memory_api_key_falls_back_to_main(
    tmp_path: Path,
):
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="moonshot",
        bampi_model_id="kimi-k2.6",
        bampi_api_key="main-secret",
    )

    manager = GroupSessionManager(config)

    api_key = await manager._resolve_memory_api_key("moonshot")

    assert api_key == "main-secret"


@pytest.mark.asyncio
async def test_group_session_manager_memory_api_key_uses_memory_model_api_for_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
        bampi_model_provider="anthropic",
        bampi_model_id="claude-sonnet-4",
        bampi_model_api="anthropic-messages",
        bampi_memory_model_provider="moonshot",
        bampi_memory_model_id="kimi-k2.6",
        bampi_memory_base_url="https://api.moonshot.cn/v1",
    )

    manager = GroupSessionManager(config)
    memory_model = manager._build_memory_model()

    api_key = await manager._resolve_memory_api_key(
        memory_model.provider,
        configured_api=memory_model.api,
    )

    assert memory_model.api == "openai-completions"
    assert api_key == "openai-secret"
