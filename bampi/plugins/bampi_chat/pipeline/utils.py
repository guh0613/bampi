"""跨模块共享的小型工具函数与轮次上下文辅助。"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from nonebot import logger
from nonebot.adapters.onebot.v11 import Message

if TYPE_CHECKING:
    from ..session_manager import GroupSessionManager


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def log_preview(text: str | None, *, limit: int = 160) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def summarize_segments(message: Message) -> str:
    counts = Counter(segment.type for segment in message)
    if not counts:
        return "empty"
    return ",".join(f"{segment_type}:{counts[segment_type]}" for segment_type in sorted(counts))


def extract_api_message_id(response: Any) -> int | None:
    candidate: Any = None
    if isinstance(response, dict):
        candidate = response.get("message_id")
        if candidate is None:
            nested = response.get("data")
            if isinstance(nested, dict):
                candidate = nested.get("message_id")
    elif response is not None:
        candidate = getattr(response, "message_id", response)

    if candidate is None:
        return None
    try:
        return int(candidate)
    except (TypeError, ValueError):
        logger.warning(f"bampi_chat got non-numeric message_id from api: {candidate!r}")
        return None


def update_qq_turn_context(
    session_manager: GroupSessionManager,
    group_id: str,
    *,
    bot_self_id: str,
    user_id: str,
    message_id: int | None,
) -> None:
    """记录本轮触发消息，供 qq_react 工具定位贴表情/戳一戳目标。"""
    setter = getattr(session_manager, "set_qq_turn_context", None)
    if callable(setter):
        setter(group_id, bot_self_id=bot_self_id, user_id=user_id, message_id=message_id)


def clear_qq_turn_context(session_manager: GroupSessionManager, group_id: str) -> None:
    """清除轮次上下文；定时/后台恢复轮次没有可互动的触发消息。"""
    clearer = getattr(session_manager, "clear_qq_turn_context", None)
    if callable(clearer):
        clearer(group_id)
