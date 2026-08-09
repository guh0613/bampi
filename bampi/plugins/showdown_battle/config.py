from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AIThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

_SUPPORTED_MODEL_APIS = {
    "auto",
    "anthropic-messages",
    "openai-responses",
    "openai-completions",
    "google-genai",
}
_MODEL_API_ALIASES = {
    "auto": "auto",
    "builtin": "auto",
    "default": "auto",
    "anthropic": "anthropic-messages",
    "anthropic-messages": "anthropic-messages",
    "google": "google-genai",
    "gemini": "google-genai",
    "google-genai": "google-genai",
    "openai": "openai-responses",
    "responses": "openai-responses",
    "openai-responses": "openai-responses",
    "chat-completions": "openai-completions",
    "chat-completion": "openai-completions",
    "completions": "openai-completions",
    "openai-completions": "openai-completions",
    "openai-chat-completions": "openai-completions",
}


class ShowdownBattleConfig(BaseModel):
    """Runtime configuration for the local Pokémon Showdown battle plugin."""

    showdown_battle_enabled: bool = True
    showdown_battle_node_bin: str = ""
    showdown_battle_package_dir: str = "node_modules/pokemon-showdown"
    showdown_battle_data_dir: str = "data/showdown_battle"
    showdown_battle_i18n_file: str = (
        "bampi/plugins/showdown_battle/assets/i18n/zh_hans.json"
    )
    showdown_battle_group_whitelist: list[int] = Field(default_factory=list)
    showdown_battle_sprite_download_timeout_seconds: float = 6.0
    showdown_battle_max_render_concurrency: int = 2
    showdown_battle_browser_executable: str = ""
    showdown_battle_render_scale: int = 2
    showdown_battle_render_idle_ttl_seconds: int = 180
    showdown_battle_team_source_timeout_seconds: float = 10.0
    showdown_battle_team_source_max_bytes: int = 1_048_576
    showdown_battle_team_source_cache_ttl_seconds: int = 900
    showdown_battle_team_guide_ttl_seconds: int = 900

    showdown_battle_ai_enabled: bool = True
    showdown_battle_ai_model_provider: str = ""
    showdown_battle_ai_model_id: str = ""
    showdown_battle_ai_model_api: str = "auto"
    showdown_battle_ai_api_key: str = ""
    showdown_battle_ai_base_url: str = ""
    showdown_battle_ai_thinking_level: AIThinkingLevel | None = None
    showdown_battle_ai_decision_timeout_seconds: float = 60.0
    showdown_battle_ai_max_output_tokens: int = 2048
    showdown_battle_ai_max_attempts: int = 2
    showdown_battle_ai_commentary_enabled: bool = True
    showdown_battle_ai_commentary_max_chars: int = 500
    showdown_battle_ai_persona: str = ""
    showdown_battle_ai_public_history_events: int = 80

    @field_validator("showdown_battle_group_whitelist", mode="before")
    @classmethod
    def _normalize_group_whitelist(cls, value: object) -> list[int]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "showdown_battle_group_whitelist must be a valid JSON "
                        "array or comma-separated list"
                    ) from exc
            else:
                value = [part.strip() for part in stripped.split(",") if part.strip()]
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("showdown_battle_group_whitelist must be a list")
        return sorted({int(item) for item in value})

    @field_validator("showdown_battle_ai_thinking_level", mode="before")
    @classmethod
    def _normalize_ai_thinking_level(cls, value: object) -> object:
        if value is None or not str(value).strip():
            return None
        return str(value).strip().lower()

    @field_validator("showdown_battle_ai_model_api", mode="before")
    @classmethod
    def _normalize_ai_model_api(cls, value: object) -> str:
        text = str(value or "auto").strip().lower().replace("_", "-") or "auto"
        normalized = _MODEL_API_ALIASES.get(text, text)
        if normalized not in _SUPPORTED_MODEL_APIS:
            raise ValueError(
                "showdown_battle_ai_model_api must be one of: "
                + ", ".join(sorted(_SUPPORTED_MODEL_APIS))
            )
        return normalized

    @field_validator(
        "showdown_battle_ai_model_provider",
        "showdown_battle_ai_model_id",
        "showdown_battle_ai_api_key",
        "showdown_battle_ai_base_url",
        "showdown_battle_ai_persona",
    )
    @classmethod
    def _strip_ai_text(cls, value: str) -> str:
        return value.strip()

    @field_validator(
        "showdown_battle_sprite_download_timeout_seconds",
        "showdown_battle_team_source_timeout_seconds",
        "showdown_battle_ai_decision_timeout_seconds",
    )
    @classmethod
    def _validate_positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout must be positive")
        return value

    @field_validator("showdown_battle_team_source_max_bytes")
    @classmethod
    def _validate_team_source_size(cls, value: int) -> int:
        if value < 4096 or value > 10 * 1024 * 1024:
            raise ValueError("team source size must be between 4 KiB and 10 MiB")
        return value

    @field_validator("showdown_battle_team_source_cache_ttl_seconds")
    @classmethod
    def _validate_team_source_cache_ttl(cls, value: int) -> int:
        if value < 0:
            raise ValueError("team source cache TTL must not be negative")
        return value

    @field_validator("showdown_battle_team_guide_ttl_seconds")
    @classmethod
    def _validate_team_guide_ttl(cls, value: int) -> int:
        if value < 60 or value > 3600:
            raise ValueError("team guide TTL must be between 60 and 3600 seconds")
        return value

    @field_validator("showdown_battle_max_render_concurrency")
    @classmethod
    def _validate_render_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("render concurrency must be at least 1")
        return value

    @field_validator("showdown_battle_render_scale")
    @classmethod
    def _validate_render_scale(cls, value: int) -> int:
        if value < 1 or value > 4:
            raise ValueError("render scale must be between 1 and 4")
        return value

    @field_validator("showdown_battle_render_idle_ttl_seconds")
    @classmethod
    def _validate_render_idle_ttl(cls, value: int) -> int:
        if value < 0:
            raise ValueError("render idle TTL must not be negative")
        return value

    @field_validator("showdown_battle_ai_max_output_tokens")
    @classmethod
    def _validate_ai_max_output_tokens(cls, value: int) -> int:
        if value < 128 or value > 16_384:
            raise ValueError("AI max output tokens must be between 128 and 16384")
        return value

    @field_validator("showdown_battle_ai_max_attempts")
    @classmethod
    def _validate_ai_max_attempts(cls, value: int) -> int:
        if value < 1 or value > 3:
            raise ValueError("AI max attempts must be between 1 and 3")
        return value

    @field_validator("showdown_battle_ai_commentary_max_chars")
    @classmethod
    def _validate_ai_commentary_max_chars(cls, value: int) -> int:
        if value < 0 or value > 2000:
            raise ValueError("AI commentary max chars must be between 0 and 2000")
        return value

    @field_validator("showdown_battle_ai_public_history_events")
    @classmethod
    def _validate_ai_public_history_events(cls, value: int) -> int:
        if value < 10 or value > 500:
            raise ValueError("AI public history events must be between 10 and 500")
        return value

    def resolve_project_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def package_dir(self) -> Path:
        return self.resolve_project_path(self.showdown_battle_package_dir)

    @property
    def data_dir(self) -> Path:
        return self.resolve_project_path(self.showdown_battle_data_dir)

    @property
    def i18n_file(self) -> Path:
        return self.resolve_project_path(self.showdown_battle_i18n_file)

    @property
    def team_storage_path(self) -> Path:
        return self.data_dir / "teams.json"

    @property
    def sprite_cache_dir(self) -> Path:
        return self.data_dir / "sprites"

    @property
    def render_browser_dir(self) -> Path:
        return self.data_dir / "render-browser"

    def resolve_node_binary(self) -> str:
        configured = self.showdown_battle_node_bin.strip()
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.is_absolute():
                return str(candidate)
            resolved = shutil.which(configured)
            return resolved or configured
        return shutil.which("node") or "node"

    def group_is_allowed(self, group_id: int) -> bool:
        whitelist = self.showdown_battle_group_whitelist
        return not whitelist or group_id in whitelist
