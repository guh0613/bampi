"""群消息触发判定、命令识别、限流与表情回应缓冲。"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Protocol

from nonebot import get_driver

from ..config import BampiChatConfig
from ..message_render import NameResolver, message_mentions_user, render_event_text
from .utils import normalize_text


@dataclass(slots=True)
class TriggerDecision:
    should_respond: bool
    reason: str = ""
    direct: bool = False
    cleaned_text: str = ""


class PlaintextEvent(Protocol):
    to_me: bool
    reply: Any

    def get_plaintext(self) -> str: ...


STOP_COMMAND = "/stop"


CLEAR_COMMANDS = {"/clear", "/new"}


COMPACT_COMMAND = "/compact"


ACTIVE_SESSION_BUSY_MESSAGE = "当前群有进行中的会话。如需中止，请让发起者发送 /stop。"


ACTIVE_SESSION_WINDING_DOWN_MESSAGE = "当前会话正在收尾，请稍候。"


STOP_NO_ACTIVE_MESSAGE = "当前没有进行中的会话或后台任务。"


STOP_NOT_OWNER_MESSAGE = "当前会话非你发起，无法停止。如需中止，请让发起者发送 /stop。"


STOPPED_SESSION_MESSAGE = "已停止当前会话。"


STOPPED_BACKGROUND_SESSION_MESSAGE = "已终止后台任务。"


STOPPED_SESSION_AND_BACKGROUND_MESSAGE = "已停止当前会话并终止后台任务。"


FORCE_STOP_PREFIX = "已强制"


BACKGROUND_TASKS_RUNNING_MESSAGE = "仍有后台任务在运行，发送 /stop 终止后再试。"


CLEARED_SESSION_MESSAGE = "已清空对话上下文。"


CLEAR_NO_CONTEXT_MESSAGE = "当前没有可清空的上下文。"


COMPACT_NO_CONTEXT_MESSAGE = "当前没有可压缩的上下文。"


COMPACT_FORBIDDEN_MESSAGE = "权限不足，仅管理员可使用 /compact。"


class GroupRateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self._limit = limit
        self._window = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def allow(self, group_id: str) -> bool:
        if self._limit <= 0:
            return True
        now = time.monotonic()
        bucket = self._buckets.setdefault(group_id, deque())
        while bucket and now - bucket[0] >= self._window:
            bucket.popleft()
        if len(bucket) >= self._limit:
            return False
        bucket.append(now)
        return True


class GroupReactionBuffer:
    """暂存贴表情（表情回应）事件，随下一轮交互作为上下文交给模型。"""

    def __init__(self, *, ttl_seconds: float = 30 * 60, max_per_group: int = 10) -> None:
        self._ttl = ttl_seconds
        self._max_per_group = max_per_group
        self._buckets: dict[str, OrderedDict[str, tuple[float, str]]] = {}

    def add(self, group_id: str, *, dedupe_key: str, note: str) -> None:
        bucket = self._buckets.setdefault(group_id, OrderedDict())
        bucket.pop(dedupe_key, None)
        bucket[dedupe_key] = (time.monotonic(), note)
        self._prune(bucket)

    def drain(self, group_id: str) -> list[str]:
        bucket = self._buckets.pop(group_id, None)
        if not bucket:
            return []
        now = time.monotonic()
        return [note for created_at, note in bucket.values() if now - created_at < self._ttl]

    def _prune(self, bucket: "OrderedDict[str, tuple[float, str]]") -> None:
        now = time.monotonic()
        for key in [key for key, (created_at, _) in bucket.items() if now - created_at >= self._ttl]:
            del bucket[key]
        while len(bucket) > self._max_per_group:
            bucket.popitem(last=False)


def is_group_allowed(group_id: str, config: BampiChatConfig) -> bool:
    whitelist = config.bampi_group_whitelist
    return not whitelist or group_id in whitelist


def is_stop_command(text: str) -> bool:
    return normalize_text(text).lower() == STOP_COMMAND


def is_clear_command(text: str) -> bool:
    return normalize_text(text).lower() in CLEAR_COMMANDS


def is_compact_command(text: str) -> bool:
    return normalize_text(text).lower() == COMPACT_COMMAND


def is_nonebot_superuser(user_id: str | int) -> bool:
    try:
        driver = get_driver()
    except ValueError:
        return False

    configured = getattr(driver.config, "superusers", None) or set()
    return str(user_id) in {str(item) for item in configured}


def interaction_busy_message(status: Any, *, requester_user_id: str | None = None) -> str:
    active_user_id = getattr(status, "active_user_id", None)
    if requester_user_id is not None and active_user_id == requester_user_id:
        return ACTIVE_SESSION_WINDING_DOWN_MESSAGE
    if not bool(getattr(status, "is_streaming", False)):
        return ACTIVE_SESSION_WINDING_DOWN_MESSAGE
    return ACTIVE_SESSION_BUSY_MESSAGE


def build_stop_success_message(
    *,
    force: bool,
    aborted_streaming: bool,
    stopped_background_sessions: bool,
) -> str:
    if aborted_streaming and stopped_background_sessions:
        message = STOPPED_SESSION_AND_BACKGROUND_MESSAGE
    elif stopped_background_sessions:
        message = STOPPED_BACKGROUND_SESSION_MESSAGE
    else:
        message = STOPPED_SESSION_MESSAGE
    if force:
        message = FORCE_STOP_PREFIX + message.removeprefix("已")
    return message


def should_respond(
    event: PlaintextEvent,
    *,
    bot_self_id: str,
    config: BampiChatConfig,
    random_value: float,
    resolve_name: NameResolver | None = None,
) -> TriggerDecision:
    if not config.bampi_enabled:
        return TriggerDecision(False)

    text = normalize_text(render_event_text(event, resolve_name=resolve_name))
    reply_to_bot = is_reply_to_bot(event.reply, bot_self_id)

    if bool(getattr(event, "to_me", False)) or reply_to_bot:
        return TriggerDecision(True, reason="to_me", direct=True, cleaned_text=text)

    if message_mentions_user(getattr(event, "message", None), bot_self_id):
        return TriggerDecision(True, reason="mention", direct=True, cleaned_text=text)

    prefix = matched_prefix(text, config.bampi_trigger_prefix)
    if prefix is not None:
        return TriggerDecision(
            True,
            reason="prefix",
            direct=True,
            cleaned_text=normalize_text(text[len(prefix) :]),
        )

    if text and any(keyword in text for keyword in config.bampi_trigger_keywords):
        return TriggerDecision(True, reason="keyword", direct=True, cleaned_text=text)

    if text and config.bampi_random_reply_prob > 0 and random_value < config.bampi_random_reply_prob:
        return TriggerDecision(True, reason="random", direct=False, cleaned_text=text)

    return TriggerDecision(False)


def matched_prefix(text: str, prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        if text.startswith(prefix):
            return prefix
    return None


def is_reply_to_bot(reply: Any, bot_self_id: str) -> bool:
    if reply is None or getattr(reply, "sender", None) is None:
        return False
    sender_id = getattr(reply.sender, "user_id", None)
    return sender_id is not None and str(sender_id) == str(bot_self_id)
