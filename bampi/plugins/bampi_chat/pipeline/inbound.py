"""入站处理：把 QQ 群消息（文本/图片/文件/合并转发）整理成 agent 输入。"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent

from bampy.ai import ImageContent, TextContent, UserMessage

from ..config import BampiChatConfig
from ..forward_messages import ForwardContext, collect_forward_context, iter_forward_nodes
from ..message_render import (
    NameResolver,
    iter_segments,
    render_message_text,
    segment_data,
    segment_type,
)
from ..timeutil import resolve_timezone
from ..tools.workspace import ensure_workspace_dirs
from .utils import normalize_text


@dataclass(slots=True)
class IncomingMedia:
    inline_images: list[ImageContent] = field(default_factory=list)
    saved_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reply_inline_images: list[ImageContent] = field(default_factory=list)
    reply_saved_paths: list[str] = field(default_factory=list)
    reply_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ForwardMediaBudget:
    remaining_items: int
    remaining_bytes: int
    seen_sources: set[str] = field(default_factory=set)
    limit_noted: bool = False

    def claim(self, source_key: str) -> bool:
        if source_key and source_key in self.seen_sources:
            return False
        if self.remaining_items <= 0 or self.remaining_bytes <= 0:
            return False
        if source_key:
            self.seen_sources.add(source_key)
        self.remaining_items -= 1
        return True

    def consume(self, byte_count: int) -> None:
        self.remaining_bytes = max(0, self.remaining_bytes - max(0, byte_count))


def extract_message_text(message: Any, *, resolve_name: NameResolver | None = None) -> str:
    if message is None:
        return ""
    try:
        return normalize_text(render_message_text(message, resolve_name=resolve_name))
    except Exception:
        logger.warning("bampi_chat failed to render message text")
        return normalize_text(str(message))


def extract_segment_filename(segment: Any) -> str | None:
    data = segment_data(segment)
    for key in ("name", "file", "file_name", "filename"):
        raw_value = data.get(key)
        if raw_value is None:
            continue
        filename = sanitize_filename(str(raw_value))
        if filename:
            return filename
    return None


def sanitize_filename(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text:
        return None

    text = text.replace("\\", "/")
    candidate = Path(text).name.strip()
    if not candidate or candidate in {".", ".."}:
        return None

    candidate = re.sub(r"[\x00-\x1f]", "", candidate)
    candidate = candidate.strip().strip(".")
    return candidate or None


def infer_filename_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    return sanitize_filename(path)


def infer_extension_from_content(data: bytes) -> str:
    if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06") or data.startswith(b"PK\x07\x08"):
        return ".zip"
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"\x1f\x8b\x08"):
        return ".gz"
    return ""


def resolve_inbox_preferred_name(
    *,
    preferred_name: str | None,
    download_url: str | None = None,
    mime_type: str | None = None,
    content: bytes | None = None,
) -> str | None:
    filename = sanitize_filename(preferred_name) or infer_filename_from_url(download_url)
    if filename:
        return filename

    extension = mimetypes.guess_extension(mime_type or "") or ""
    if not extension and content:
        extension = infer_extension_from_content(content)
    if not extension:
        return None
    return f"attachment{extension}"


def display_name(sender: Any) -> str:
    card = getattr(sender, "card", "") or ""
    nickname = getattr(sender, "nickname", "") or ""
    return card.strip() or nickname.strip() or "unknown-user"


def build_user_message(
    event: GroupMessageEvent,
    cleaned_text: str,
    media: IncomingMedia,
    *,
    forwards: ForwardContext | None = None,
    resolve_name: NameResolver | None = None,
    reaction_notes: list[str] | None = None,
) -> UserMessage:
    forwards = forwards or ForwardContext()
    sender_name = display_name(event.sender)
    if cleaned_text:
        body = cleaned_text
    elif forwards.has_current:
        body = "(无附言；本条消息包含合并转发，请结合 forwarded_messages 理解)"
    elif media.inline_images or media.saved_paths:
        body = "(无纯文本内容；本条消息仅包含媒体/文件)"
    elif forwards.has_reply:
        body = "(无附言；请结合回复引用中的合并转发理解)"
    elif media.reply_inline_images or media.reply_saved_paths:
        body = "(无纯文本内容；请结合回复引用内容理解)"
    else:
        body = "(无纯文本内容)"

    lines = [
        f"sender_name: {sender_name}({event.user_id})",
        f"message_text: {body}",
    ]

    if event.reply is not None and getattr(event.reply, "sender", None) is not None:
        reply_name = display_name(event.reply.sender)
        reply_text = extract_message_text(getattr(event.reply, "message", None), resolve_name=resolve_name)
        lines.append(f"reply_to_name: {reply_name}")
        if reply_text:
            lines.append(f"reply_message: {reply_text}")

    if forwards.current_render.text:
        lines.append("forwarded_messages:")
        lines.extend(f"  {line}" for line in forwards.current_render.text.splitlines())
    if forwards.reply_render.text:
        lines.append("reply_forwarded_messages:")
        lines.extend(f"  {line}" for line in forwards.reply_render.text.splitlines())

    if media.inline_images:
        lines.append(f"inline_image_count: {len(media.inline_images)}")
    if media.saved_paths:
        lines.append("workspace_attachments:")
        lines.extend(f"- {path}" for path in media.saved_paths)
    if media.notes:
        lines.append("media_notes:")
        lines.extend(f"- {note}" for note in media.notes)
    if media.reply_inline_images:
        lines.append(f"reply_inline_image_count: {len(media.reply_inline_images)}")
    if media.reply_saved_paths:
        lines.append("reply_workspace_attachments:")
        lines.extend(f"- {path}" for path in media.reply_saved_paths)
    if media.reply_notes:
        lines.append("reply_media_notes:")
        lines.extend(f"- {note}" for note in media.reply_notes)
    if reaction_notes:
        lines.append("recent_reactions:")
        lines.extend(f"- {note}" for note in reaction_notes)

    content: list[TextContent | ImageContent] = [TextContent(text="\n".join(lines))]
    content.extend(media.inline_images)
    content.extend(media.reply_inline_images)
    return UserMessage(content=content)


async def collect_incoming_context(
    bot: Bot,
    event: GroupMessageEvent,
    config: BampiChatConfig,
    workspace_dir: str,
) -> tuple[IncomingMedia, ForwardContext]:
    """Collect ordinary media and lazily expand merged-forward messages."""
    reply_message = getattr(event.reply, "message", None)
    forwards = await collect_forward_context(
        bot,
        message=event.message,
        reply_message=reply_message,
        enabled=config.bampi_forward_enabled,
        max_depth=config.bampi_forward_max_depth,
        max_nodes=config.bampi_forward_max_nodes,
        max_roots=config.bampi_forward_max_roots,
        max_api_calls=config.bampi_forward_max_api_calls,
        max_text_chars=config.bampi_forward_max_text_chars,
        timeout_seconds=config.bampi_forward_resolve_timeout_seconds,
        timezone=resolve_timezone(config.bampi_schedule_timezone),
    )
    media = await collect_incoming_media(
        bot,
        event,
        config,
        workspace_dir,
        forwards=forwards,
    )
    await _persist_truncated_forward_transcripts(
        media=media,
        forwards=forwards,
        workspace_dir=workspace_dir,
    )
    return media, forwards


async def _persist_truncated_forward_transcripts(
    *,
    media: IncomingMedia,
    forwards: ForwardContext,
    workspace_dir: str,
) -> None:
    targets = (
        (
            forwards.current_render,
            media.saved_paths,
            media.notes,
            "forwarded-messages.txt",
            "合并转发",
        ),
        (
            forwards.reply_render,
            media.reply_saved_paths,
            media.reply_notes,
            "reply-forwarded-messages.txt",
            "回复引用中的合并转发",
        ),
    )
    for rendered, saved_paths, notes, preferred_name, label in targets:
        if not rendered.truncated or not rendered.full_text:
            continue
        try:
            saved = await save_bytes_to_inbox(
                workspace_dir,
                rendered.full_text.encode("utf-8"),
                preferred_name=preferred_name,
                mime_type="text/plain",
            )
        except Exception as exc:
            logger.warning(
                f"bampi_chat failed to save full forward transcript error={exc!r}"
            )
            notes.append(f"{label}内容较长，预览已截断，完整转录保存失败。")
            continue
        saved_paths.append(saved)
        notes.append(f"{label}内容较长，完整转录已保存到 {saved}")


async def collect_incoming_media(
    bot: Bot,
    event: GroupMessageEvent,
    config: BampiChatConfig,
    workspace_dir: str,
    *,
    forwards: ForwardContext | None = None,
) -> IncomingMedia:
    ensure_workspace_dirs(workspace_dir)
    media = IncomingMedia()

    await _collect_media_from_message(
        bot=bot,
        event=event,
        message=event.message,
        media=media,
        config=config,
        workspace_dir=workspace_dir,
        from_reply=False,
    )

    reply_message = getattr(event.reply, "message", None)
    if reply_message is not None:
        await _collect_media_from_message(
            bot=bot,
            event=event,
            message=reply_message,
            media=media,
            config=config,
            workspace_dir=workspace_dir,
            from_reply=True,
        )

    if forwards is not None and (forwards.has_current or forwards.has_reply):
        budget = _ForwardMediaBudget(
            remaining_items=config.bampi_forward_max_media_items,
            remaining_bytes=config.bampi_forward_max_total_media_bytes,
        )
        for from_reply, resolved in (
            (False, forwards.current),
            (True, forwards.reply),
        ):
            for node in iter_forward_nodes(resolved):
                await _collect_media_from_message(
                    bot=bot,
                    event=event,
                    message=node.segments,
                    media=media,
                    config=config,
                    workspace_dir=workspace_dir,
                    from_reply=from_reply,
                    allow_group_file_lookup=False,
                    forward_budget=budget,
                )

    return media


def _media_targets(
    media: IncomingMedia,
    *,
    from_reply: bool,
) -> tuple[list[ImageContent], list[str], list[str]]:
    if from_reply:
        return media.reply_inline_images, media.reply_saved_paths, media.reply_notes
    return media.inline_images, media.saved_paths, media.notes


async def _collect_media_from_message(
    *,
    bot: Bot,
    event: GroupMessageEvent,
    message: Any,
    media: IncomingMedia,
    config: BampiChatConfig,
    workspace_dir: str,
    from_reply: bool,
    allow_group_file_lookup: bool = True,
    forward_budget: _ForwardMediaBudget | None = None,
) -> None:
    source = "reply" if from_reply else "message"
    if forward_budget is not None:
        source = f"forward_{source}"
    if message is None:
        return

    for segment in iter_segments(message):
        seg_type = segment_type(segment)
        if seg_type not in {"image", "file"}:
            continue
        data = segment_data(segment)
        source_key = _media_source_key(seg_type, data)
        if forward_budget is not None:
            direct_url = str(data.get("url") or "").strip()
            if direct_url and urlparse(direct_url).scheme.lower() not in {
                "http",
                "https",
            }:
                if not source_key or source_key not in forward_budget.seen_sources:
                    if source_key:
                        forward_budget.seen_sources.add(source_key)
                    _, _, notes = _media_targets(media, from_reply=from_reply)
                    notes.append(
                        "合并转发中的媒体 URL 不是可下载的 HTTP(S) 地址，已跳过。"
                    )
                continue
            if source_key and source_key in forward_budget.seen_sources:
                continue
            if not forward_budget.claim(source_key):
                _note_forward_media_limit(media, from_reply=from_reply, budget=forward_budget)
                continue

        logger.info(
            f"bampi_chat processing {seg_type} segment "
            f"group_id={event.group_id} "
            f"message_id={event.message_id} "
            f"source={source}"
        )
        max_download_bytes = (
            forward_budget.remaining_bytes if forward_budget is not None else None
        )
        if seg_type == "image":
            consumed = await _handle_image_segment(
                segment,
                media,
                config,
                workspace_dir,
                from_reply=from_reply,
                max_download_bytes=max_download_bytes,
            )
        else:
            consumed = await _handle_file_segment(
                bot,
                event,
                segment,
                media,
                config,
                workspace_dir,
                from_reply=from_reply,
                allow_group_file_lookup=allow_group_file_lookup,
                max_download_bytes=max_download_bytes,
            )
        if forward_budget is not None:
            forward_budget.consume(consumed)


def _media_source_key(seg_type: str, data: dict[str, Any]) -> str:
    source = (
        data.get("url")
        or data.get("file_id")
        or data.get("id")
        or data.get("file")
    )
    return f"{seg_type}:{source}" if source else ""


def _note_forward_media_limit(
    media: IncomingMedia,
    *,
    from_reply: bool,
    budget: _ForwardMediaBudget,
) -> None:
    if budget.limit_noted:
        return
    budget.limit_noted = True
    _, _, notes = _media_targets(media, from_reply=from_reply)
    notes.append("合并转发中的媒体较多或总大小超过限制，仅处理了前一部分。")


async def _handle_image_segment(
    segment: Any,
    media: IncomingMedia,
    config: BampiChatConfig,
    workspace_dir: str,
    *,
    from_reply: bool,
    max_download_bytes: int | None = None,
) -> int:
    inline_images, saved_paths, notes = _media_targets(media, from_reply=from_reply)
    data = segment_data(segment)
    url = str(data.get("url") or "").strip()
    if not url:
        logger.warning("bampi_chat image segment missing download url")
        if from_reply:
            notes.append("回复引用里有图片，但适配器未提供可下载 URL。")
        else:
            notes.append("收到图片，但适配器未提供可下载 URL。")
        return 0

    download_limit = config.bampi_max_download_size
    if max_download_bytes is not None:
        download_limit = min(download_limit, max_download_bytes)
    if download_limit <= 0:
        notes.append("图片未下载：已达到本轮合并转发媒体大小上限。")
        return 0

    try:
        content, mime_type = await download_url(
            url,
            timeout=config.bampi_web_search_timeout,
            max_bytes=download_limit,
        )
    except Exception as exc:
        logger.warning(f"bampi_chat failed to download image: {exc}")
        if from_reply:
            notes.append(f"下载回复引用图片失败: {exc}")
        else:
            notes.append(f"下载图片失败: {exc}")
        return 0

    mime_type = mime_type or guess_mime_type(url, default="image/png")
    if len(content) <= config.bampi_max_inline_image_size:
        logger.info(f"bampi_chat inlined image mime_type={mime_type} size={len(content)}")
        inline_images.append(
            ImageContent(
                data=base64.b64encode(content).decode("ascii"),
                mime_type=mime_type,
            )
        )
        return len(content)

    saved = await save_bytes_to_inbox(
        workspace_dir,
        content,
        preferred_name=data.get("file"),
        mime_type=mime_type,
    )
    logger.info(f"bampi_chat saved oversized image path={saved} size={len(content)}")
    saved_paths.append(saved)
    if from_reply:
        notes.append(f"回复引用中的图片过大，已保存到 {saved}")
    else:
        notes.append(f"图片过大，已保存到 {saved}")
    return len(content)


async def _handle_file_segment(
    bot: Bot,
    event: GroupMessageEvent,
    segment: Any,
    media: IncomingMedia,
    config: BampiChatConfig,
    workspace_dir: str,
    *,
    from_reply: bool,
    allow_group_file_lookup: bool = True,
    max_download_bytes: int | None = None,
) -> int:
    data = segment_data(segment)
    file_id = data.get("id") or data.get("file_id")
    direct_url = str(data.get("url", "")).strip()
    if not direct_url and (not file_id or not allow_group_file_lookup):
        logger.warning("bampi_chat file segment missing usable download url")
        _, _, notes = _media_targets(media, from_reply=from_reply)
        if from_reply:
            notes.append("回复引用里有文件，但缺少可下载 URL。")
        else:
            notes.append("收到文件，但缺少可下载 URL。")
        return 0

    _, saved_paths, notes = _media_targets(media, from_reply=from_reply)
    download_limit = config.bampi_max_download_size
    if max_download_bytes is not None:
        download_limit = min(download_limit, max_download_bytes)
    if download_limit <= 0:
        notes.append("文件未下载：已达到本轮合并转发媒体大小上限。")
        return 0

    try:
        if direct_url:
            url = direct_url
        else:
            info = await bot.call_api(
                "get_group_file_url",
                group_id=event.group_id,
                file_id=file_id,
            )
            url = str(info.get("url", ""))
        if not url:
            raise RuntimeError("empty file url")
        content, mime_type = await download_url(
            url,
            timeout=config.bampi_web_search_timeout,
            max_bytes=download_limit,
        )
        preferred_name = resolve_inbox_preferred_name(
            preferred_name=extract_segment_filename(segment) or file_id,
            download_url=url,
            mime_type=mime_type,
            content=content,
        )
        saved = await save_bytes_to_inbox(
            workspace_dir,
            content,
            preferred_name=preferred_name,
            mime_type=mime_type,
        )
        logger.info(
            f"bampi_chat saved group file file_id={file_id} "
            f"preferred_name={preferred_name!r} path={saved} size={len(content)}"
        )
        saved_paths.append(saved)
        return len(content)
    except Exception as exc:
        logger.warning(f"bampi_chat failed to download group file file_id={file_id}: {exc}")
        if from_reply:
            notes.append(f"下载回复引用文件失败: {exc}")
        else:
            notes.append(f"下载群文件失败: {exc}")
        return 0


async def download_url(url: str, *, timeout: float, max_bytes: int) -> tuple[bytes, str]:
    def _download() -> tuple[bytes, str]:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; BampiBot/0.1)"})
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"download exceeds limit: {max_bytes} bytes")
        return data, content_type

    return await asyncio.to_thread(_download)


def guess_mime_type(filename: str | None, *, default: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename or "")
    return mime_type or default


async def save_bytes_to_inbox(
    workspace_dir: str,
    data: bytes,
    *,
    preferred_name: str | None,
    mime_type: str | None,
) -> str:
    inbox = ensure_workspace_dirs(workspace_dir) / "inbox"
    clean_name = sanitize_filename(preferred_name)
    suffix = "".join(Path(clean_name or "").suffixes)
    if not suffix:
        suffix = mimetypes.guess_extension(mime_type or "") or ""
    if not suffix:
        suffix = infer_extension_from_content(data)

    stem = ""
    if clean_name:
        stem = Path(clean_name).name
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        stem = stem.strip().strip(".")

    unique = uuid.uuid4().hex[:12]
    if stem:
        filename = f"{stem}-{unique}{suffix}"
    else:
        filename = f"{unique}{suffix}"
    path = inbox / filename
    await asyncio.to_thread(path.write_bytes, data)
    return f"inbox/{filename}"
