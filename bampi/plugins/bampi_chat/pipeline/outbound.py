"""出站处理：把 agent 回复（文本/渲染图片/outbox 文件）投递到 QQ 群。"""

from __future__ import annotations

import asyncio
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import quote

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.matcher import Matcher

from bampy.ai.types import AssistantMessage, StopReason

from ..config import BampiChatConfig
from ..feedback import (
    FailureAssessment,
    assess_failure,
    build_background_failure_message,
    build_reply_failure_message,
)
from ..message_compose import append_composed_text, compose_options_from_config
from ..rich_render import (
    ImagePart,
    TextPart,
    build_delivery_plan,
    rich_render_options_from_config,
)
from ..rich_render.service import get_renderer as get_rich_renderer
from ..tools.workspace import ensure_workspace_dirs, is_image_file
from .utils import log_preview, normalize_text


@dataclass(slots=True)
class ResponseDispatchResult:
    delivered: bool
    rollback_context: bool = False


@dataclass(slots=True)
class PreparedGroupFileUpload:
    file_uri: str
    cleanup_paths: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class PreparedOutboundImage:
    source: str | bytes
    cleanup_paths: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class GroupReplyTarget:
    group_id: int
    user_id: int | None = None
    reply_message_id: int | None = None


def reply_target_for_event(event: GroupMessageEvent) -> GroupReplyTarget:
    return GroupReplyTarget(
        group_id=int(event.group_id),
        user_id=int(event.user_id),
        reply_message_id=int(event.message_id),
    )


def posix_path_to_file_uri(path: PurePosixPath | str) -> str:
    normalized = PurePosixPath(path).as_posix()
    return f"file://{quote(normalized, safe='/')}"


async def prepare_group_file_upload(
    path: Path,
    config: BampiChatConfig,
) -> PreparedGroupFileUpload:
    host_dir = config.bampi_group_file_upload_host_dir.strip()
    container_dir = config.bampi_group_file_upload_container_dir.strip()

    if host_dir and container_dir:
        staged_path: Path | None = None
        try:
            if not container_dir.startswith("/"):
                raise ValueError("container upload dir must be an absolute POSIX path")

            staging_dir = Path(host_dir).expanduser()
            if not staging_dir.is_absolute():
                staging_dir = staging_dir.resolve()
            await asyncio.to_thread(staging_dir.mkdir, parents=True, exist_ok=True)

            staged_name = f"{uuid.uuid4().hex[:12]}-{path.name}"
            staged_path = staging_dir / staged_name
            await asyncio.to_thread(shutil.copy2, path, staged_path)

            container_path = PurePosixPath(container_dir) / staged_name
            file_uri = posix_path_to_file_uri(container_path)
            logger.info(
                f"bampi_chat staged group upload source={path} "
                f"staged={staged_path} file_uri={file_uri}"
            )
            return PreparedGroupFileUpload(file_uri=file_uri, cleanup_paths=[staged_path])
        except Exception as exc:
            if staged_path is not None:
                try:
                    staged_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(f"bampi_chat failed to cleanup staged upload file: {staged_path}")
            logger.warning(
                f"bampi_chat failed to stage group upload path={path} "
                f"host_dir={host_dir!r} container_dir={container_dir!r}: {exc}. "
                f"Falling back to local file URI."
            )

    file_uri = path.resolve().as_uri()
    logger.info(f"bampi_chat using local group upload file_uri={file_uri}")
    return PreparedGroupFileUpload(file_uri=file_uri)


async def prepare_outbound_image(
    path: Path,
    config: BampiChatConfig,
) -> PreparedOutboundImage:
    file_size = await asyncio.to_thread(lambda: path.stat().st_size)
    if file_size <= config.bampi_max_inline_image_size:
        data = await asyncio.to_thread(path.read_bytes)
        logger.info(
            f"bampi_chat prepared inline image source={path} size={len(data)}"
        )
        return PreparedOutboundImage(source=data)

    prepared = await prepare_group_file_upload(path, config)
    if prepared.cleanup_paths:
        logger.info(
            f"bampi_chat prepared staged image source={path} file_uri={prepared.file_uri}"
        )
        return PreparedOutboundImage(
            source=prepared.file_uri,
            cleanup_paths=prepared.cleanup_paths,
        )

    data = await asyncio.to_thread(path.read_bytes)
    logger.warning(
        f"bampi_chat image staging unavailable, falling back to inline base64 "
        f"source={path} size={len(data)}"
    )
    return PreparedOutboundImage(source=data)


def snapshot_outbox(workspace_dir: str) -> dict[str, float]:
    outbox = ensure_workspace_dirs(workspace_dir) / "outbox"
    snapshot: dict[str, float] = {}
    for path in outbox.iterdir():
        if path.is_file():
            snapshot[path.name] = path.stat().st_mtime
    return snapshot


def find_last_assistant_message(messages: list[Any]) -> AssistantMessage | None:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            return message
    return None


def extract_text_blocks(message: AssistantMessage | None) -> str:
    if message is None:
        return ""
    if isinstance(message.content, str):
        return message.content.strip()

    parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def assistant_message_has_tool_call(message: AssistantMessage | None) -> bool:
    """Whether a completed assistant message continues into tool execution."""
    if message is None or isinstance(message.content, str):
        return False
    return any(
        getattr(block, "type", None) == "tool_call" for block in message.content
    )


def collect_outbox_files(
    workspace_dir: str,
    *,
    before: dict[str, float],
    text: str,
) -> list[Path]:
    outbox = ensure_workspace_dirs(workspace_dir) / "outbox"
    candidates: dict[str, Path] = {}

    for path in outbox.iterdir():
        if not path.is_file():
            continue
        previous = before.get(path.name)
        if previous is None or path.stat().st_mtime > previous:
            candidates[path.name] = path

    pattern = re.compile(r"(?P<path>(?:outbox/|/workspace(?:/[^/\s`'\"()]+)*/outbox/)[^\s`'\"()]+)")
    for match in pattern.finditer(text):
        raw = match.group("path")
        normalized = raw
        if normalized.startswith("/workspace/"):
            _, _, suffix = normalized.partition("/outbox/")
            normalized = f"outbox/{suffix}" if suffix else normalized
        if normalized.startswith("outbox/"):
            path = outbox / normalized.removeprefix("outbox/")
            if path.is_file():
                candidates[path.name] = path

    return sorted(candidates.values(), key=lambda item: item.name.lower())


def build_group_reply_message(
    *,
    config: BampiChatConfig,
    target: GroupReplyTarget,
    text: str,
    parse_outbound_markup: bool = False,
    quote_reply: bool = True,
) -> Message:
    message = Message()
    if (
        config.bampi_reply_with_quote
        and quote_reply
        and target.reply_message_id is not None
    ):
        message += MessageSegment.reply(target.reply_message_id)
    if text:
        if parse_outbound_markup:
            append_composed_text(
                message,
                text,
                options=compose_options_from_config(config),
            )
        else:
            message += MessageSegment.text(text)
    return message


async def append_reply_body(
    message: Message,
    text: str,
    config: BampiChatConfig,
) -> int:
    """Append *text* to *message*, rendering layout-bearing blocks as images.

    Code, formulas and tables become image segments placed where they appeared
    in the reply; everything else goes through the normal outbound markup
    parser. Returns the number of rendered blocks.

    If rendering is disabled or fails, the reply is appended as plain Markdown
    text, so no path here can lose content.
    """
    if not text:
        return 0

    options = rich_render_options_from_config(config)
    renderer = get_rich_renderer(config) if options.enabled else None
    plan = await build_delivery_plan(text, renderer=renderer, options=options)

    compose_options = compose_options_from_config(config)
    rendered = 0
    for part in plan:
        if isinstance(part, TextPart):
            append_composed_text(message, part.text, options=compose_options)
        elif isinstance(part, ImagePart):
            message += MessageSegment.image(part.png)
            rendered += 1
    return rendered


async def _send_group_message_via_bot(
    *,
    bot: Bot,
    target: GroupReplyTarget,
    message: Message,
) -> None:
    await bot.call_api(
        "send_group_msg",
        group_id=target.group_id,
        message=message,
    )


async def send_agent_response_to_target(
    *,
    bot: Bot,
    target: GroupReplyTarget,
    config: BampiChatConfig,
    workspace_dir: str,
    assistant_message: AssistantMessage | None,
    outbox_before: dict[str, float],
    intermediate_text_sent: bool = False,
    quote_reply: bool = True,
    text_prefix: str = "",
    log_label: str = "auto-resume",
    failure_message_builder: Callable[[FailureAssessment], str] = build_background_failure_message,
    empty_reply_text: str | None = "后续处理完成，无新内容可回复。可以发送新消息继续。",
) -> ResponseDispatchResult:
    full_text = extract_text_blocks(assistant_message)
    intermediate_already_delivered = (
        intermediate_text_sent and assistant_message_has_tool_call(assistant_message)
    )
    text = "" if intermediate_already_delivered else full_text
    text = text.lstrip()
    if text_prefix and text:
        text = f"{text_prefix}{text}"
    files = collect_outbox_files(workspace_dir, before=outbox_before, text=full_text)
    stop_reason = getattr(assistant_message, "stop_reason", None)
    error_message = normalize_text(getattr(assistant_message, "error_message", None))
    logger.info(
        f"bampi_chat preparing {log_label} reply group_id={target.group_id} "
        f"reply_message_id={target.reply_message_id} "
        f"text={log_preview(text)!r} "
        f"files={[path.name for path in files]} "
        f"stop_reason={stop_reason} "
        f"error={log_preview(error_message)!r}"
    )

    if stop_reason in {StopReason.ABORTED, "aborted"}:
        logger.info(
            f"bampi_chat skipped aborted {log_label} reply group_id={target.group_id} "
            f"reply_message_id={target.reply_message_id}"
        )
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    if not text and not files:
        if error_message:
            logger.warning(
                f"bampi_chat {log_label} returned no deliverable content "
                f"group_id={target.group_id} "
                f"reply_message_id={target.reply_message_id} "
                f"stop_reason={stop_reason} "
                f"error={log_preview(error_message)!r}"
            )
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=config,
                    target=target,
                    text=failure_message_builder(assess_failure(error_message)),
                ),
            )
            return ResponseDispatchResult(delivered=False, rollback_context=True)
        if intermediate_already_delivered:
            logger.info(
                f"bampi_chat {log_label} intermediate tool turn already delivered "
                f"group_id={target.group_id} "
                f"reply_message_id={target.reply_message_id}"
            )
            return ResponseDispatchResult(delivered=True, rollback_context=False)

        logger.warning(
            f"bampi_chat {log_label} returned empty content "
            f"group_id={target.group_id} "
            f"reply_message_id={target.reply_message_id}"
        )
        if empty_reply_text:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=config,
                    target=target,
                    text=empty_reply_text,
                ),
            )
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    message = build_group_reply_message(
        config=config,
        target=target,
        text="",
        quote_reply=quote_reply,
    )
    rendered_block_count = await append_reply_body(message, text, config)

    uploaded_files: list[Path] = []
    staged_upload_files: list[Path] = []
    failed_artifacts: list[str] = []
    sent_image_count = 0
    try:
        for path in files:
            if not is_image_file(path):
                continue
            try:
                prepared_image = await prepare_outbound_image(path, config)
                staged_upload_files.extend(prepared_image.cleanup_paths)
                message += MessageSegment.image(prepared_image.source)
                sent_image_count += 1
            except Exception:
                logger.exception(f"failed to prepare {log_label} outbox image: {path}")
                failed_artifacts.append(path.name)

        if message:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=message,
            )
            logger.info(
                f"bampi_chat {log_label} reply sent group_id={target.group_id} "
                f"reply_message_id={target.reply_message_id} "
                f"has_text={bool(text)} "
                f"rich_block_count={rendered_block_count} "
                f"image_count={sent_image_count}"
            )

        for path in files:
            if is_image_file(path):
                if path.name in failed_artifacts:
                    continue
                uploaded_files.append(path)
                continue
            try:
                upload = await prepare_group_file_upload(path, config)
                staged_upload_files.extend(upload.cleanup_paths)
                await bot.call_api(
                    "upload_group_file",
                    group_id=target.group_id,
                    file=upload.file_uri,
                    name=path.name,
                )
                logger.info(f"bampi_chat {log_label} uploaded outbox file group_id={target.group_id} path={path}")
                uploaded_files.append(path)
            except Exception:
                logger.exception(f"failed to upload {log_label} outbox file: {path}")
                failed_artifacts.append(path.name)
        if failed_artifacts:
            await _send_group_message_via_bot(
                bot=bot,
                target=target,
                message=build_group_reply_message(
                    config=config,
                    target=target,
                    text=f"有 {len(failed_artifacts)} 个文件已生成但发送失败。",
                ),
            )
    finally:
        for path in uploaded_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup {log_label} outbox file: {path}")
        for staged_path in staged_upload_files:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup {log_label} staged upload file: {staged_path}")
    return ResponseDispatchResult(delivered=True, rollback_context=False)


async def send_agent_response(
    *,
    bot: Bot,
    event: GroupMessageEvent,
    matcher: Matcher,
    config: BampiChatConfig,
    workspace_dir: str,
    assistant_message: AssistantMessage | None,
    outbox_before: dict[str, float],
    intermediate_text_sent: bool = False,
    quote_reply: bool = True,
) -> ResponseDispatchResult:
    full_text = extract_text_blocks(assistant_message)
    intermediate_already_delivered = (
        intermediate_text_sent and assistant_message_has_tool_call(assistant_message)
    )
    text = "" if intermediate_already_delivered else full_text
    text = text.lstrip()
    files = collect_outbox_files(workspace_dir, before=outbox_before, text=full_text)
    stop_reason = getattr(assistant_message, "stop_reason", None)
    error_message = normalize_text(getattr(assistant_message, "error_message", None))
    logger.info(
        f"bampi_chat preparing reply group_id={event.group_id} "
        f"message_id={event.message_id} "
        f"text={log_preview(text)!r} "
        f"files={[path.name for path in files]} "
        f"stop_reason={stop_reason} "
        f"error={log_preview(error_message)!r}"
    )

    if stop_reason in {StopReason.ABORTED, "aborted"}:
        logger.info(
            f"bampi_chat skipped aborted reply group_id={event.group_id} "
            f"message_id={event.message_id}"
        )
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    if not text and not files:
        if error_message:
            logger.warning(
                f"bampi_chat assistant returned no deliverable content "
                f"group_id={event.group_id} "
                f"message_id={event.message_id} "
                f"stop_reason={stop_reason} "
                f"error={log_preview(error_message)!r}"
            )
            await matcher.send(build_reply_failure_message(assess_failure(error_message)))
            return ResponseDispatchResult(delivered=False, rollback_context=True)
        if intermediate_already_delivered:
            logger.info(
                f"bampi_chat final assistant tool turn already delivered "
                f"group_id={event.group_id} "
                f"message_id={event.message_id}"
            )
            return ResponseDispatchResult(delivered=True, rollback_context=False)

        logger.warning(
            f"bampi_chat assistant returned empty content "
            f"group_id={event.group_id} "
            f"message_id={event.message_id}"
        )
        await matcher.send("⚠️ 这次没有生成可发送的内容，请换个说法再试一次。")
        return ResponseDispatchResult(delivered=False, rollback_context=True)

    message = Message()
    if config.bampi_reply_with_quote and quote_reply:
        message += MessageSegment.reply(event.message_id)
    rendered_block_count = await append_reply_body(message, text, config)

    uploaded_files: list[Path] = []
    staged_upload_files: list[Path] = []
    failed_artifacts: list[str] = []
    sent_image_count = 0
    try:
        for path in files:
            if not is_image_file(path):
                continue
            try:
                prepared_image = await prepare_outbound_image(path, config)
                staged_upload_files.extend(prepared_image.cleanup_paths)
                message += MessageSegment.image(prepared_image.source)
                sent_image_count += 1
            except Exception:
                logger.exception(f"failed to prepare outbox image: {path}")
                failed_artifacts.append(path.name)

        if message:
            await matcher.send(message)
            logger.info(
                f"bampi_chat reply sent group_id={event.group_id} "
                f"message_id={event.message_id} "
                f"has_text={bool(text)} "
                f"rich_block_count={rendered_block_count} "
                f"image_count={sent_image_count}"
            )
        else:
            logger.warning(
                f"bampi_chat reply skipped because message was empty "
                f"group_id={event.group_id} "
                f"message_id={event.message_id}"
            )

        for path in files:
            if is_image_file(path):
                if path.name in failed_artifacts:
                    continue
                uploaded_files.append(path)
                continue
            try:
                upload = await prepare_group_file_upload(path, config)
                staged_upload_files.extend(upload.cleanup_paths)
                await bot.call_api(
                    "upload_group_file",
                    group_id=event.group_id,
                    file=upload.file_uri,
                    name=path.name,
                )
                logger.info(f"bampi_chat uploaded outbox file group_id={event.group_id} path={path}")
                uploaded_files.append(path)
            except Exception:
                logger.exception(f"failed to upload outbox file: {path}")
                failed_artifacts.append(path.name)
        if failed_artifacts:
            await matcher.send(
                f"有 {len(failed_artifacts)} 个文件已生成但发送失败。"
            )
    finally:
        for path in uploaded_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup outbox file: {path}")
        for staged_path in staged_upload_files:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"failed to cleanup staged upload file: {staged_path}")
    return ResponseDispatchResult(delivered=True, rollback_context=False)
