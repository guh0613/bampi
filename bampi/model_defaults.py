"""Shared defaults for components that inherit the bampi_chat main model."""

from typing import Literal


DEFAULT_MODEL_PROVIDER = "openai"
DEFAULT_MODEL_ID = "gpt-5-mini"
DEFAULT_MODEL_API = "auto"
DEFAULT_MODEL_API_KEY = ""
DEFAULT_MODEL_BASE_URL = ""
DEFAULT_MODEL_THINKING_LEVEL: Literal["off"] = "off"


__all__ = [
    "DEFAULT_MODEL_API",
    "DEFAULT_MODEL_API_KEY",
    "DEFAULT_MODEL_BASE_URL",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_PROVIDER",
    "DEFAULT_MODEL_THINKING_LEVEL",
]
