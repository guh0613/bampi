from __future__ import annotations

import json
from nonebot import logger
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from bampy.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from .search_text import extract_search_terms, normalize_for_search
from .types import MemoryMessage, MemoryParticipant, MemoryToolEvent, MemoryUserTurn



@dataclass(slots=True)
class ArchiveBuildResult:
    started_at: str
    ended_at: str
    participants: list[MemoryParticipant]
    title: str
    summary: str
    keywords: list[str]
    messages: list[MemoryMessage]
    tool_events: list[MemoryToolEvent]


def build_archive_from_agent_messages(
    *,
    group_id: str,
    messages: list[Any],
    user_turns: list[MemoryUserTurn],
    min_messages: int,
    tool_result_preview_chars: int,
    tool_result_full_max_chars: int,
) -> ArchiveBuildResult | None:
    archive_messages: list[MemoryMessage] = []
    tool_events: list[MemoryToolEvent] = []
    participants_by_user: dict[str, MemoryParticipant] = {}
    tool_calls: dict[str, tuple[str, str]] = {}
    user_turn_index = 0

    for raw_message in messages:
        role = _message_role(raw_message)
        if role == "assistant":
            for call in _assistant_tool_calls(raw_message):
                tool_calls[call.id] = (call.name, _json_dumps(call.arguments))
            assistant_text = _assistant_text(raw_message)
            if assistant_text:
                archive_messages.append(
                    MemoryMessage(
                        role="assistant",
                        content=assistant_text,
                        timestamp=_message_timestamp_iso(raw_message),
                    )
                )
            continue

        if role == "user":
            content = _user_text(raw_message)
            if not content:
                continue
            turn = _match_user_turn(raw_message, user_turns, user_turn_index)
            if turn is not None:
                user_turn_index = max(user_turn_index, user_turns.index(turn) + 1)
            nickname = turn.nickname if turn is not None else _extract_sender_name(content)
            user_id = turn.user_id if turn is not None else ""
            if user_id:
                participants_by_user[user_id] = MemoryParticipant(user_id=user_id, nickname=nickname)
            archive_messages.append(
                MemoryMessage(
                    role="user",
                    user_id=user_id,
                    nickname=nickname,
                    content=content,
                    timestamp=_message_timestamp_iso(raw_message),
                )
            )
            continue

        if role == "tool_result":
            tool_event = _tool_event_from_message(
                raw_message,
                tool_calls=tool_calls,
                preview_chars=tool_result_preview_chars,
                full_chars=tool_result_full_max_chars,
            )
            if tool_event is not None:
                tool_events.append(tool_event)

    if len(archive_messages) < max(1, min_messages):
        return None

    timestamps = [message.timestamp for message in archive_messages]
    timestamps.extend(event.timestamp for event in tool_events)
    started_at = min(timestamps) if timestamps else _now_iso()
    ended_at = max(timestamps) if timestamps else started_at
    title, summary, keywords = summarize_archive(archive_messages, tool_events)
    return ArchiveBuildResult(
        started_at=started_at,
        ended_at=ended_at,
        participants=list(participants_by_user.values()),
        title=title,
        summary=summary,
        keywords=keywords,
        messages=archive_messages,
        tool_events=tool_events,
    )


def summarize_archive(
    messages: list[MemoryMessage],
    tool_events: list[MemoryToolEvent],
) -> tuple[str, str, list[str]]:
    user_messages = [message for message in messages if message.role == "user"]
    assistant_messages = [message for message in messages if message.role == "assistant"]
    first_user = _message_body(user_messages[0].content) if user_messages else ""
    last_user = _message_body(user_messages[-1].content) if user_messages else ""
    last_assistant = assistant_messages[-1].content if assistant_messages else ""
    all_text = "\n".join(
        [message.content for message in messages]
        + [event.tool_name for event in tool_events]
        + [event.arguments_text for event in tool_events]
        + [event.result_preview for event in tool_events]
    )
    keywords = _keyword_list(all_text)

    if keywords:
        title = "讨论 " + "、".join(keywords[:4])
    elif first_user:
        title = _truncate(normalize_for_search(first_user), 36)
    else:
        title = "未命名群聊会话"

    pieces: list[str] = []
    if first_user:
        pieces.append(f"本次会话从“{_truncate(first_user, 80)}”开始。")
    if keywords:
        pieces.append("主要涉及：" + "、".join(keywords[:8]) + "。")
    if tool_events:
        tool_names = ", ".join(dict.fromkeys(event.tool_name for event in tool_events if event.tool_name))
        pieces.append(f"期间使用过工具：{tool_names or '若干工具'}。")
    if last_user and last_user != first_user:
        pieces.append(f"最后的用户问题大致是“{_truncate(last_user, 80)}”。")
    if last_assistant:
        pieces.append(f"最后回复停在：{_truncate(last_assistant, 120)}")
    summary = "".join(pieces) or "这次会话没有足够可摘要的文本内容。"
    return title, summary, keywords


_LLM_SUMMARY_SYSTEM_PROMPT = """\
你是一个会话归档助手。给定一段群聊对话记录，请输出结构化的摘要。

要求：
1. title: 简短标题（10-25字），概括这次对话的核心主题
2. summary: 摘要（50-200字），描述对话的起因、过程和结论/结果。重点记录做了什么、解决了什么、结论是什么
3. keywords: 3-8个关键词，用于后续检索

输出格式（严格遵守，不要输出其他内容）：
title: <标题>
summary: <摘要>
keywords: <关键词1>, <关键词2>, ...\
"""

_LLM_SUMMARY_MAX_INPUT_CHARS = 12000


async def summarize_archive_with_llm(
    messages: list[MemoryMessage],
    tool_events: list[MemoryToolEvent],
    *,
    model: Any,
    api_key: str | None = None,
) -> tuple[str, str, list[str]] | None:
    from bampy.ai import Context, SimpleStreamOptions, complete_simple
    from bampy.ai.types import UserMessage as AIUserMessage

    transcript_lines: list[str] = []
    for msg in messages:
        prefix = msg.nickname or msg.role
        transcript_lines.append(f"[{prefix}]: {msg.content}")
    if tool_events:
        transcript_lines.append("\n--- 使用过的工具 ---")
        for event in tool_events[:10]:
            line = f"工具: {event.tool_name}"
            if event.result_preview:
                line += f" → {event.result_preview[:200]}"
            transcript_lines.append(line)

    transcript = "\n".join(transcript_lines)
    if len(transcript) > _LLM_SUMMARY_MAX_INPUT_CHARS:
        transcript = transcript[:_LLM_SUMMARY_MAX_INPUT_CHARS] + "\n...(已截断)"

    context = Context(
        system_prompt=_LLM_SUMMARY_SYSTEM_PROMPT,
        messages=[AIUserMessage(content=transcript)],
    )
    options = SimpleStreamOptions(api_key=api_key)
    try:
        result = await complete_simple(model, context, options)
    except Exception:
        logger.opt(exception=True).warning(
            f"bampi_chat memory LLM summary failed model={getattr(model, 'id', model)}"
        )
        return None

    text = ""
    for block in result.content:
        if getattr(block, "type", "") == "text":
            text += getattr(block, "text", "")
    if not text.strip():
        return None

    return _parse_llm_summary_response(text)


def _parse_llm_summary_response(text: str) -> tuple[str, str, list[str]] | None:
    title = ""
    summary = ""
    keywords: list[str] = []
    for line in text.strip().splitlines():
        lower = line.lower().strip()
        if lower.startswith("title:") or lower.startswith("标题:"):
            title = line.split(":", 1)[1].strip()
        elif lower.startswith("summary:") or lower.startswith("摘要:"):
            summary = line.split(":", 1)[1].strip()
        elif lower.startswith("keywords:") or lower.startswith("关键词:"):
            raw = line.split(":", 1)[1].strip()
            keywords = [k.strip() for k in raw.split(",") if k.strip()]
    if not title and not summary:
        return None
    return title, summary, keywords


def _match_user_turn(
    message: Any,
    user_turns: list[MemoryUserTurn],
    fallback_index: int,
) -> MemoryUserTurn | None:
    timestamp = _message_timestamp_float(message)
    if timestamp is not None:
        for turn in user_turns:
            if turn.timestamp is None:
                continue
            if abs(turn.timestamp - timestamp) < 0.001:
                return turn
    if 0 <= fallback_index < len(user_turns):
        return user_turns[fallback_index]
    return None


def _message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def _user_text(message: Any) -> str:
    if isinstance(message, UserMessage):
        return _content_text(message.content)
    if isinstance(message, Mapping):
        return _content_text(message.get("content", ""))
    return _content_text(getattr(message, "content", ""))


def _assistant_text(message: Any) -> str:
    content = message.get("content", []) if isinstance(message, Mapping) else getattr(message, "content", [])
    parts: list[str] = []
    if isinstance(content, str):
        return normalize_for_search(content)
    for block in content or []:
        block_type = block.get("type") if isinstance(block, Mapping) else getattr(block, "type", "")
        if block_type != "text":
            continue
        text = block.get("text", "") if isinstance(block, Mapping) else getattr(block, "text", "")
        if text:
            parts.append(str(text))
    return normalize_for_search("\n".join(parts))


def _assistant_tool_calls(message: Any) -> list[ToolCall]:
    content = message.get("content", []) if isinstance(message, Mapping) else getattr(message, "content", [])
    calls: list[ToolCall] = []
    for block in content or []:
        block_type = block.get("type") if isinstance(block, Mapping) else getattr(block, "type", "")
        if block_type != "tool_call":
            continue
        if isinstance(block, ToolCall):
            calls.append(block)
            continue
        if isinstance(block, Mapping):
            try:
                calls.append(ToolCall.model_validate(block))
            except Exception:
                continue
    return calls


def _tool_event_from_message(
    message: Any,
    *,
    tool_calls: dict[str, tuple[str, str]],
    preview_chars: int,
    full_chars: int,
) -> MemoryToolEvent | None:
    tool_call_id = str(_mapping_or_attr(message, "tool_call_id", "")).strip()
    tool_name = str(_mapping_or_attr(message, "tool_name", "")).strip()
    if not tool_call_id and not tool_name:
        return None
    mapped_name, mapped_args = tool_calls.get(tool_call_id, ("", ""))
    result_text = _content_text(_mapping_or_attr(message, "content", []))
    details = _mapping_or_attr(message, "details", None)
    if details is not None:
        result_text = "\n".join(part for part in [result_text, _json_dumps(details)] if part)
    return MemoryToolEvent(
        timestamp=_message_timestamp_iso(message),
        tool_call_id=tool_call_id,
        tool_name=tool_name or mapped_name,
        arguments_text=mapped_args,
        result_preview=_truncate(result_text, max(0, preview_chars)),
        result_full=_truncate(result_text, max(0, full_chars)),
        is_error=bool(_mapping_or_attr(message, "is_error", False)),
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return normalize_for_search(content)
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, TextContent):
            parts.append(block.text)
            continue
        if isinstance(block, Mapping):
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("text", "")))
            elif block_type == "image":
                parts.append("[image]")
            continue
        block_type = getattr(block, "type", "")
        if block_type == "text":
            parts.append(str(getattr(block, "text", "")))
        elif block_type == "image":
            parts.append("[image]")
    return normalize_for_search("\n".join(part for part in parts if part))


def _extract_sender_name(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("sender_name:"):
            return line.split(":", 1)[1].strip()
    return ""


def _message_body(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("message_text:"):
            return line.split(":", 1)[1].strip()
    return content.strip()


def _keyword_list(text: str) -> list[str]:
    terms = extract_search_terms(text, for_query=False)
    counts: dict[str, int] = {}
    for term in terms:
        if len(term) <= 1:
            continue
        if term.isdigit() and len(term) < 3:
            continue
        counts[term] = counts.get(term, 0) + 1
    return [
        term
        for term, _count in sorted(counts.items(), key=lambda item: (item[1], len(item[0]), item[0]), reverse=True)
    ][:8]


def _mapping_or_attr(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_timestamp_float(message: Any) -> float | None:
    value = _mapping_or_attr(message, "timestamp", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _message_timestamp_iso(message: Any) -> str:
    value = _mapping_or_attr(message, "timestamp", None)
    if isinstance(value, str):
        return value
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return _now_iso()
    if timestamp > 10_000_000_000:
        timestamp = timestamp / 1000.0
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int) -> str:
    normalized = normalize_for_search(text)
    if limit <= 0:
        return ""
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)
