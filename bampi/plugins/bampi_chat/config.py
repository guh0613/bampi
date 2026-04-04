from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator


ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
BashMode = Literal["auto", "docker", "local"]

DEFAULT_WORKSPACE_DIR = "data/bampi/workspace"
DEFAULT_SESSION_DIR = "data/bampi/sessions"
DEFAULT_BASH_CONTAINER_NAME = "bampi-sandbox"
DEFAULT_BASH_CONTAINER_WORKDIR = "/workspace"
DEFAULT_BASH_CONTAINER_SHELL = "/bin/bash"


class BampiChatConfig(BaseModel):
    bampi_enabled: bool = True

    bampi_model_provider: str = "openai"
    bampi_model_id: str = "gpt-5-mini"
    bampi_api_key: str = ""
    bampi_base_url: str = ""
    bampi_thinking_level: ThinkingLevel = "off"

    bampi_trigger_prefix: list[str] = Field(default_factory=lambda: ["@bot"])
    bampi_trigger_keywords: list[str] = Field(default_factory=list)
    bampi_group_whitelist: list[str] = Field(default_factory=list)
    bampi_random_reply_prob: float = 0.0
    bampi_rate_limit: int = 30
    bampi_rate_limit_window_seconds: int = 60

    bampi_max_turns: int = 40
    bampi_session_idle_ttl_seconds: int = 30 * 60

    bampi_workspace_dir: str = DEFAULT_WORKSPACE_DIR
    bampi_session_dir: str = DEFAULT_SESSION_DIR

    bampi_persona: str = ""
    bampi_reply_with_quote: bool = True
    bampi_at_sender: bool = False
    bampi_live_progress_enabled: bool = True
    bampi_live_progress_max_tool_updates: int = 0
    bampi_live_progress_error_recall_min_visible_seconds: float = 1.0
    bampi_live_text_stream_enabled: bool = True
    bampi_live_text_stream_min_chars: int = 80
    bampi_live_text_stream_force_chars: int = 220
    bampi_live_text_stream_min_interval_seconds: float = 1.2
    bampi_threshold_compaction_notice_enabled: bool = True

    bampi_bash_mode: BashMode = "docker"
    bampi_bash_container_name: str = DEFAULT_BASH_CONTAINER_NAME
    bampi_bash_container_workdir: str = DEFAULT_BASH_CONTAINER_WORKDIR
    bampi_bash_container_shell: str = DEFAULT_BASH_CONTAINER_SHELL
    bampi_bash_timeout: float = 30.0

    bampi_web_search_timeout: float = 15.0
    bampi_web_search_base_url: str = ""
    bampi_web_search_api_key: str = ""
    bampi_browser_enabled: bool = True
    bampi_browser_headless: bool = True
    bampi_browser_block_images: bool = False
    bampi_browser_launch_timeout: float = 45.0
    bampi_browser_action_timeout: float = 20.0
    bampi_browser_idle_ttl_seconds: int = 5 * 60
    bampi_browser_max_pages: int = 6
    bampi_browser_inline_image_max_bytes: int = 1_000_000

    bampi_max_inline_image_size: int = 5 * 1024 * 1024
    bampi_max_download_size: int = 50 * 1024 * 1024
    bampi_group_file_upload_host_dir: str = "app/.config/QQ/temp"
    bampi_group_file_upload_container_dir: str = "/app/.config/QQ/temp"

    @field_validator(
        "bampi_trigger_prefix",
        "bampi_trigger_keywords",
        "bampi_group_whitelist",
        mode="before",
    )
    @classmethod
    def _normalize_string_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("expected a list of strings")
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result

    @field_validator(
        "bampi_model_provider",
        "bampi_model_id",
        "bampi_base_url",
        "bampi_web_search_base_url",
        "bampi_web_search_api_key",
        "bampi_persona",
        "bampi_workspace_dir",
        "bampi_session_dir",
        "bampi_bash_container_name",
        "bampi_bash_container_workdir",
        "bampi_bash_container_shell",
        "bampi_group_file_upload_host_dir",
        "bampi_group_file_upload_container_dir",
    )
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("bampi_bash_container_name")
    @classmethod
    def _validate_container_name(cls, value: str) -> str:
        if not value:
            raise ValueError("bampi_bash_container_name must not be empty")
        return value

    @field_validator("bampi_bash_container_workdir", "bampi_bash_container_shell")
    @classmethod
    def _validate_container_path(cls, value: str, info) -> str:
        if not value.startswith("/"):
            raise ValueError(f"{info.field_name} must be an absolute POSIX path")
        return PurePosixPath(value).as_posix()
