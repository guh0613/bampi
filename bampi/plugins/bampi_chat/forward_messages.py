"""Resolve and normalize OneBot v11 merged-forward messages.

OneBot implementations disagree on ``get_forward_msg`` response shape.  The
standard describes ``message: [node, ...]`` while current NapCat releases
return ``messages: [OB11Message, ...]``; NapCat may also inline parsed children
in ``forward.data.content``.  This module hides those differences from the
handler and produces normalized immutable models for rendering and media
collection.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol
from zoneinfo import ZoneInfo

from nonebot import logger
from nonebot.adapters.onebot.v11 import Message

from .message_render import (
    iter_segments,
    render_segment_text,
    segment_data,
    segment_type,
)


class SupportsGetForwardMessageApi(Protocol):
    async def get_forward_msg(self, *, id: str) -> Any: ...


class SupportsOneBotCallApi(Protocol):
    async def call_api(self, api: str, **data: Any) -> Any: ...


ForwardApiBot = SupportsGetForwardMessageApi | SupportsOneBotCallApi


@dataclass(slots=True, frozen=True)
class ForwardNode:
    sender_id: str | None
    sender_name: str
    timestamp: int | None
    segments: tuple[dict[str, Any], ...]
    nested: tuple["ResolvedForward", ...] = ()


@dataclass(slots=True, frozen=True)
class ResolvedForward:
    forward_id: str
    nodes: tuple[ForwardNode, ...] = ()
    note: str | None = None
    truncated: bool = False


@dataclass(slots=True, frozen=True)
class RenderedForward:
    text: str = ""
    full_text: str = ""
    truncated: bool = False


@dataclass(slots=True, frozen=True)
class ForwardContext:
    current: tuple[ResolvedForward, ...] = ()
    reply: tuple[ResolvedForward, ...] = ()
    current_render: RenderedForward = field(default_factory=RenderedForward)
    reply_render: RenderedForward = field(default_factory=RenderedForward)

    @property
    def has_current(self) -> bool:
        return bool(self.current)

    @property
    def has_reply(self) -> bool:
        return bool(self.reply)


@dataclass(slots=True)
class _ResolveState:
    nodes_used: int = 0
    api_calls: int = 0
    roots_used: int = 0
    resolving_ids: set[str] = field(default_factory=set)
    payload_cache: dict[str, Any] = field(default_factory=dict)
    failed_ids: set[str] = field(default_factory=set)


def _forward_read_failure(forward_id: str) -> ResolvedForward:
    return ResolvedForward(
        forward_id=forward_id,
        note="合并转发读取失败：消息可能已过期、无权限或协议端无法解析。",
    )


class ForwardResolver:
    """Request-scoped merged-forward resolver with recursion and size limits."""

    def __init__(
        self,
        bot: ForwardApiBot,
        *,
        max_depth: int,
        max_nodes: int,
        max_roots: int,
        max_api_calls: int,
        timeout_seconds: float,
    ) -> None:
        self._bot = bot
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._max_roots = max_roots
        self._max_api_calls = max_api_calls
        self._timeout_seconds = timeout_seconds
        self._state = _ResolveState()

    async def collect(
        self, message: Any, reply_message: Any
    ) -> tuple[tuple[ResolvedForward, ...], tuple[ResolvedForward, ...]]:
        current = await self._collect_from_message(message)
        reply = await self._collect_from_message(reply_message)
        return tuple(current), tuple(reply)

    async def _collect_from_message(self, message: Any) -> list[ResolvedForward]:
        resolved: list[ResolvedForward] = []
        for segment in iter_segments(message):
            if segment_type(segment) != "forward":
                continue
            if self._state.roots_used >= self._max_roots:
                resolved.append(
                    ResolvedForward(
                        forward_id="",
                        note=f"合并转发数量超过限制，仅处理前 {self._max_roots} 个。",
                        truncated=True,
                    )
                )
                break
            self._state.roots_used += 1
            try:
                resolved.append(
                    await self._resolve_forward_data(segment_data(segment), depth=1)
                )
            except Exception as exc:  # one malformed forward must not abort the turn
                logger.warning(
                    f"bampi_chat unexpected forward resolver failure error={exc!r}"
                )
                resolved.append(
                    _forward_read_failure(str(segment_data(segment).get("id") or ""))
                )
        return resolved

    async def _resolve_forward_data(
        self, data: dict[str, Any], *, depth: int
    ) -> ResolvedForward:
        forward_id = str(data.get("id") or data.get("message_id") or "").strip()
        if depth > self._max_depth:
            return ResolvedForward(
                forward_id=forward_id,
                note=f"嵌套合并转发深度超过限制（{self._max_depth} 层）。",
                truncated=True,
            )

        inline_content = _as_sequence(data.get("content"))
        if inline_content:
            return await self._parse_entries(forward_id, inline_content, depth=depth)

        if not forward_id:
            return ResolvedForward(forward_id="", note="合并转发缺少可读取的消息 ID。")

        if forward_id in self._state.resolving_ids:
            return ResolvedForward(
                forward_id=forward_id, note="检测到循环嵌套的合并转发。", truncated=True
            )
        if forward_id in self._state.failed_ids:
            return _forward_read_failure(forward_id)

        self._state.resolving_ids.add(forward_id)
        try:
            if forward_id in self._state.payload_cache:
                payload = self._state.payload_cache[forward_id]
            else:
                if self._state.api_calls >= self._max_api_calls:
                    return ResolvedForward(
                        forward_id=forward_id,
                        note=f"合并转发 API 调用超过限制（{self._max_api_calls} 次）。",
                        truncated=True,
                    )
                self._state.api_calls += 1
                try:
                    payload = await self._fetch(forward_id)
                except Exception as exc:
                    self._state.failed_ids.add(forward_id)
                    logger.info(
                        f"bampi_chat get_forward_msg failed forward_id={forward_id!r} "
                        f"error={exc!r}"
                    )
                    return _forward_read_failure(forward_id)
                self._state.payload_cache[forward_id] = payload

            entries = _extract_forward_entries(payload)
            if not entries:
                return ResolvedForward(
                    forward_id=forward_id, note="合并转发中没有可读取的消息。"
                )
            return await self._parse_entries(forward_id, entries, depth=depth)
        finally:
            self._state.resolving_ids.discard(forward_id)

    async def _fetch(self, forward_id: str) -> Any:
        get_forward_msg = getattr(self._bot, "get_forward_msg", None)
        async with asyncio.timeout(self._timeout_seconds):
            if callable(get_forward_msg):
                return await get_forward_msg(id=forward_id)
            # Keep compatibility with light-weight test doubles and adapters that
            # only expose the generic OneBot action dispatcher.
            call_api = getattr(self._bot, "call_api", None)
            if callable(call_api):
                return await call_api("get_forward_msg", id=forward_id)
            raise RuntimeError("bot does not expose get_forward_msg")

    async def _parse_entries(
        self, forward_id: str, entries: list[Any], *, depth: int
    ) -> ResolvedForward:
        nodes: list[ForwardNode] = []
        truncated = False
        malformed_count = 0
        for entry in entries:
            if self._state.nodes_used >= self._max_nodes:
                truncated = True
                break
            # Reserve the parent before resolving nested forwards so descendants
            # cannot consume the final slot and make the total exceed max_nodes.
            self._state.nodes_used += 1
            try:
                node = await self._parse_entry(entry, depth=depth)
            except Exception as exc:
                self._state.nodes_used -= 1
                malformed_count += 1
                logger.warning(
                    f"bampi_chat skipped malformed forward node "
                    f"forward_id={forward_id!r} error={exc!r}"
                )
                continue
            if node is None:
                self._state.nodes_used -= 1
                malformed_count += 1
                continue
            nodes.append(node)
        note = (
            f"合并转发节点超过限制，仅处理前 {self._max_nodes} 条。"
            if truncated
            else None
        )
        if malformed_count:
            malformed_note = f"另有 {malformed_count} 条格式异常的转发节点已跳过。"
            note = f"{note} {malformed_note}" if note else malformed_note
        if not nodes and note is None:
            note = "合并转发中没有可读取的消息。"
        return ResolvedForward(
            forward_id=forward_id,
            nodes=tuple(nodes),
            note=note,
            truncated=truncated,
        )

    async def _parse_entry(self, entry: Any, *, depth: int) -> ForwardNode | None:
        mapping = _as_mapping(entry)
        if mapping is None:
            if isinstance(entry, str):
                return ForwardNode(
                    None, "unknown-user", None, tuple(_coerce_segments(entry))
                )
            return None

        if str(mapping.get("type") or "") == "node":
            data = _as_mapping(mapping.get("data")) or {}
            if not any(key in data for key in ("content", "message")):
                return None
            sender_id = _clean_optional(data.get("user_id") or data.get("uin"))
            sender_name = _display_name(data)
            timestamp = _parse_timestamp(data.get("time"))
            raw_message = data.get("content")
            if _message_value_is_empty(raw_message):
                raw_message = data.get("message")
        elif _is_segment_like(mapping):
            sender_id = None
            sender_name = "unknown-user"
            timestamp = None
            raw_message = [mapping]
        else:
            if not any(key in mapping for key in ("message", "content", "raw_message")):
                return None
            sender = _as_mapping(mapping.get("sender")) or {}
            sender_id = _clean_optional(mapping.get("user_id") or sender.get("user_id"))
            sender_name = _display_name(sender)
            timestamp = _parse_timestamp(mapping.get("time"))
            raw_message = mapping.get("message")
            if _message_value_is_empty(raw_message):
                raw_message = mapping.get("content")
            if _message_value_is_empty(raw_message):
                raw_message = mapping.get("raw_message")

        segments = tuple(_coerce_segments(raw_message))
        nested: list[ResolvedForward] = []
        for segment in segments:
            if segment_type(segment) != "forward":
                continue
            nested.append(
                await self._resolve_forward_data(segment_data(segment), depth=depth + 1)
            )

        return ForwardNode(
            sender_id=sender_id,
            sender_name=sender_name,
            timestamp=timestamp,
            segments=segments,
            nested=tuple(nested),
        )


async def collect_forward_context(
    bot: ForwardApiBot,
    *,
    message: Any,
    reply_message: Any,
    enabled: bool,
    max_depth: int,
    max_nodes: int,
    max_roots: int,
    max_api_calls: int,
    max_text_chars: int,
    timeout_seconds: float,
    timezone: ZoneInfo,
) -> ForwardContext:
    if not enabled:
        return ForwardContext()

    resolver = ForwardResolver(
        bot,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_roots=max_roots,
        max_api_calls=max_api_calls,
        timeout_seconds=timeout_seconds,
    )
    current, reply = await resolver.collect(message, reply_message)
    current_render, reply_render = render_forward_context(
        current,
        reply,
        max_chars=max_text_chars,
        timezone=timezone,
    )
    return ForwardContext(
        current=current,
        reply=reply,
        current_render=current_render,
        reply_render=reply_render,
    )


def iter_forward_nodes(forwards: Iterable[ResolvedForward]) -> Iterable[ForwardNode]:
    for resolved in forwards:
        for node in resolved.nodes:
            yield node
            for nested in node.nested:
                yield from iter_forward_nodes((nested,))


def render_forward_context(
    current: tuple[ResolvedForward, ...],
    reply: tuple[ResolvedForward, ...],
    *,
    max_chars: int,
    timezone: ZoneInfo,
) -> tuple[RenderedForward, RenderedForward]:
    current_full = _render_full_forward_text(current, timezone=timezone)
    reply_full = _render_full_forward_text(reply, timezone=timezone)
    current_budget, reply_budget = _allocate_text_budgets(
        len(current_full),
        len(reply_full),
        total_budget=max_chars,
    )
    return (
        _truncate_rendered_forward(current_full, max_chars=current_budget),
        _truncate_rendered_forward(reply_full, max_chars=reply_budget),
    )


def _render_full_forward_text(
    forwards: Iterable[ResolvedForward],
    *,
    timezone: ZoneInfo,
) -> str:
    lines: list[str] = []
    for index, resolved in enumerate(forwards, start=1):
        _render_resolved_forward(
            resolved, lines, timezone=timezone, indent=0, label=str(index)
        )
    return "\n".join(lines).strip()


def _allocate_text_budgets(
    current_length: int,
    reply_length: int,
    *,
    total_budget: int,
) -> tuple[int, int]:
    total_budget = max(0, total_budget)
    if current_length == 0:
        return 0, total_budget
    if reply_length == 0:
        return total_budget, 0
    if current_length + reply_length <= total_budget:
        return current_length, reply_length

    current_budget = min(current_length, total_budget // 2)
    reply_budget = min(reply_length, total_budget // 2)
    remaining = total_budget - current_budget - reply_budget
    if remaining <= 0:
        return current_budget, reply_budget

    current_deficit = current_length - current_budget
    reply_deficit = reply_length - reply_budget
    if current_deficit >= reply_deficit:
        extra = min(remaining, current_deficit)
        current_budget += extra
        remaining -= extra
        reply_budget += min(remaining, reply_deficit)
    else:
        extra = min(remaining, reply_deficit)
        reply_budget += extra
        remaining -= extra
        current_budget += min(remaining, current_deficit)
    return current_budget, reply_budget


def _truncate_rendered_forward(full_text: str, *, max_chars: int) -> RenderedForward:
    if not full_text:
        return RenderedForward()
    if len(full_text) <= max_chars:
        return RenderedForward(text=full_text, full_text=full_text, truncated=False)

    marker = "\n…（内容过长，预览已截断）"
    if max_chars <= len(marker):
        preview = marker[-max_chars:] if max_chars > 0 else ""
    else:
        keep = max_chars - len(marker)
        preview = f"{full_text[:keep].rstrip()}{marker}"
    return RenderedForward(text=preview, full_text=full_text, truncated=True)


def _render_resolved_forward(
    resolved: ResolvedForward,
    lines: list[str],
    *,
    timezone: ZoneInfo,
    indent: int,
    label: str,
) -> None:
    prefix = "  " * indent
    lines.append(f"{prefix}[合并转发 {label}，共 {len(resolved.nodes)} 条]")
    if resolved.note:
        lines.append(f"{prefix}  {resolved.note}")
    for node_index, node in enumerate(resolved.nodes, start=1):
        sender = node.sender_name
        if node.sender_id:
            sender = f"{sender}({node.sender_id})"
        timestamp = _format_timestamp(node.timestamp, timezone)
        header = f"{prefix}{node_index}. {sender}"
        if timestamp:
            header += f"  {timestamp}"
        lines.append(header)

        body = _render_forward_segments(node.segments)
        body_lines = body.splitlines() if body else ["(无可渲染内容)"]
        lines.extend(f"{prefix}   {line}" for line in body_lines)
        for nested_index, nested in enumerate(node.nested, start=1):
            _render_resolved_forward(
                nested,
                lines,
                timezone=timezone,
                indent=indent + 1,
                label=f"{label}.{node_index}.{nested_index}",
            )


def _render_forward_segments(segments: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for segment in segments:
        seg_type = segment_type(segment)
        data = segment_data(segment)
        if seg_type == "forward":
            parts.append("[嵌套合并转发]")
            continue
        rendered = render_segment_text(segment)
        if rendered:
            parts.append(rendered)
            continue
        marker = _render_non_text_segment(seg_type, data)
        if marker:
            parts.append(marker)
    return "".join(parts).strip()


def _render_non_text_segment(seg_type: str, data: dict[str, Any]) -> str:
    if seg_type == "image":
        summary = _clean_metadata_text(data.get("summary"), max_chars=200).strip("[]")
        if data.get("emoji_id"):
            return f"[动画表情:{summary}]" if summary else "[动画表情]"
        return f"[图片:{summary}]" if summary else "[图片]"
    if seg_type == "file":
        name = _first_text(data, "name", "file", "file_name", "filename")
        size = _format_file_size(data.get("file_size"))
        detail = "，".join(value for value in (name, size) if value)
        return f"[文件:{detail}]" if detail else "[文件]"
    if seg_type == "video":
        name = _first_text(data, "name", "file")
        return f"[视频:{name}]" if name else "[视频]"
    if seg_type == "record":
        return "[语音]"
    if seg_type == "reply":
        return "[回复消息]"
    if seg_type == "share":
        title = _first_text(data, "title", "content", "url")
        return f"[链接分享:{title}]" if title else "[链接分享]"
    if seg_type == "music":
        title = _first_text(data, "title", "content", "id")
        return f"[音乐:{title}]" if title else "[音乐]"
    if seg_type == "location":
        title = _first_text(data, "title", "content")
        return f"[位置:{title}]" if title else "[位置]"
    if seg_type == "contact":
        contact_id = _first_text(data, "id")
        return f"[联系人:{contact_id}]" if contact_id else "[联系人]"
    if seg_type == "markdown":
        return str(data.get("content") or "")
    if seg_type == "json":
        return "[JSON 卡片]"
    if seg_type == "xml":
        return "[XML 卡片]"
    if seg_type == "miniapp":
        return "[小程序]"
    if seg_type:
        return f"[{seg_type}]"
    return ""


def _format_timestamp(timestamp: int | None, timezone: ZoneInfo) -> str:
    if timestamp is None:
        return ""
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone).strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return ""


def _first_text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = _clean_metadata_text(data.get(key), max_chars=200)
        if text:
            return text
    return ""


def _format_file_size(value: Any) -> str:
    try:
        size = int(str(value))
    except (TypeError, ValueError):
        return ""
    if size < 0:
        return ""
    units = ("B", "KB", "MB", "GB")
    amount = float(size)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.0f}{unit}" if unit == "B" else f"{amount:.1f}{unit}"
        amount /= 1024
    return ""


def _extract_forward_entries(payload: Any) -> list[Any]:
    direct = _as_sequence(payload)
    if direct is not None:
        return direct
    mapping = _as_mapping(payload)
    if mapping is None:
        return []
    if "sender" in mapping and any(
        key in mapping for key in ("message", "content", "raw_message")
    ):
        return [mapping]
    for key in ("messages", "message", "content"):
        entries = _as_sequence(mapping.get(key))
        if entries is not None:
            return entries
    if str(mapping.get("type") or "") == "node":
        return [mapping]
    return []


def _coerce_segments(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        # Standard OneBot custom nodes may carry CQ-code strings rather than
        # array-format segments.  Let the adapter parse those strings so media,
        # mentions and faces are not silently flattened into plain text.
        return [_canonical_segment(segment) for segment in Message(value)]
    if _is_segment_like(value):
        return [_canonical_segment(value)]

    try:
        candidates = list(value)
    except TypeError:
        return []

    segments: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            segments.append({"type": "text", "data": {"text": candidate}})
        elif _is_segment_like(candidate):
            segments.append(_canonical_segment(candidate))
    return segments


def _canonical_segment(value: Any) -> dict[str, Any]:
    return {"type": segment_type(value), "data": dict(segment_data(value))}


def _is_segment_like(value: Any) -> bool:
    if isinstance(value, dict):
        return isinstance(value.get("type"), str) and isinstance(
            value.get("data"), dict
        )
    return isinstance(getattr(value, "type", None), str) and isinstance(
        getattr(value, "data", None), dict
    )


def _message_value_is_empty(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else None
    return None


def _as_sequence(value: Any) -> list[Any] | None:
    if value is None or isinstance(value, (str, bytes, dict)):
        return None
    try:
        return list(value)
    except TypeError:
        return None


def _display_name(mapping: dict[str, Any]) -> str:
    for key in ("card", "nickname", "name"):
        text = _clean_metadata_text(mapping.get(key), max_chars=128)
        if text:
            return text
    return "unknown-user"


def _clean_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_timestamp(value: Any) -> int | None:
    try:
        timestamp = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    # Reject clearly invalid/synthetic values while accepting ordinary epoch seconds.
    return timestamp if timestamp > 0 else None


def _clean_metadata_text(value: Any, *, max_chars: int) -> str:
    return " ".join(str(value or "").split())[:max_chars]
