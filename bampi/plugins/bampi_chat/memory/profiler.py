from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from nonebot import logger

from .types import MemoryArchive, MemoryParticipant, MemoryProfile, MemoryProfileEdit

_PROFILE_SECTION_HEADINGS = ("Work context", "Personal context", "Top of mind", "Brief history")
_PROFILE_HISTORY_HEADINGS = ("Recent months", "Earlier context", "Long-term background")

_LLM_PROFILE_MAX_INPUT_CHARS = 24_000
_LLM_PROFILE_SYSTEM_PROMPT = """\
你是一个群聊长期记忆画像生成器。你会根据旧画像、新归档摘要和用户明确编辑，生成一个人的个人画像。

只输出画像正文，不要输出 Markdown 代码块、JSON、解释、前后缀。

用户画像的内容格式是：

**Work context**
...

**Personal context**
...

**Top of mind**
...

**Brief history**
*Recent months*
...
*Earlier context*
...
*Long-term background*
...
...

每节内容量根据实际掌握的信息决定——
已知少则写一句，已知多则写密集段落。
不要编造信息来填充篇幅，也不要省略已知信息来缩减篇幅。

要求：
- 只记录有持续价值、可由输入支持的事实，不要臆测。
- 内容标题必须严格按照上述格式输出，包含上述格式要求的标题。
- **重要——参与者归因**：归档摘要来自群聊，同一归档中可能包含多位用户各自独立的对话。\
只提取画像本人（即 `{nickname}` 指代的用户）实际发起、表达或明确认同的观点、偏好和经历。\
不要将同一归档中其他参与者独立讨论的话题归因到画像本人身上。\
判断依据是摘要中的发言者标注——如果摘要提到"[某用户]讨论了X"而该用户不是画像本人，则X不应出现在本画像中。
- 近期事件可以更具体，早期内容要逐步概括。
- 正文中指代画像本人时请使用 `{nickname}`这个占位符，不要使用具体的名字(当前群名片只用于理解输入中的说话人)！正确示例：`{nickname} 经常讨论 Rust 和命令行工具。` 错误示例：`张三 经常讨论 Rust 和命令行工具。`
- 不要输出 QQ 号、group_id、user_id 等代码侧元数据。
- delete/忘记/失效类编辑必须从主画像中移除，不要再提及相关内容。
- 不记录密码、令牌、身份证号、银行卡号、住址等高敏感信息。
"""


@dataclass(slots=True)
class MemoryContextRequest:
    group_id: str
    current_user_id: str
    current_nickname: str = ""
    session_participants: list[MemoryParticipant] | None = None


def render_memory_context(
    *,
    current_user_id: str,
    current_nickname: str,
    profiles: list[tuple[MemoryProfile | None, list[MemoryProfileEdit], str]],
    max_profile_chars_for_others: int = 600,
) -> str:
    sections: list[str] = []
    for profile, edits, display_name in profiles:
        user_id = profile.user_id if profile is not None else ""
        if not user_id and edits:
            user_id = edits[0].user_id
        if not user_id:
            continue

        is_current = user_id == current_user_id
        rendered_name = display_name or current_nickname or (profile.nickname if profile is not None else user_id)
        body = profile.profile if profile is not None else ""
        body = _render_profile_body(
            body,
            nickname=rendered_name,
            edits=edits,
            truncate_chars=0 if is_current else max_profile_chars_for_others,
        )
        if not body.strip():
            continue
        sections.append(f"### {rendered_name or user_id}\n{body.strip()}")

    if not sections:
        return ""
    return (
        "以下是本群长期记忆，只用于理解当前对话；它可能不完整。"
        "用户明确要求记住/更新/忘记时，使用 `memory_manage`。\n\n"
        + "\n\n".join(sections)
    )


def build_profile_from_archives(
    *,
    profile: MemoryProfile,
    archives: list[MemoryArchive],
    pending_edits: list[MemoryProfileEdit],
) -> str:
    keywords = _top_keywords(archives)
    recent_archives = sorted(archives, key=lambda archive: archive.ended_at, reverse=True)[:8]
    additions = [edit for edit in pending_edits if edit.edit_type in {"add", "update"}]
    deletions = [edit for edit in pending_edits if edit.edit_type == "delete"]

    lines: list[str] = [
        "**Work context**",
        "暂时没有明确的工作或学习上下文。",
        "",
        "**Personal context**",
        "{nickname} 是本群成员。",
        "",
        "**Top of mind**",
    ]
    if keywords:
        lines.append("近期经常参与的话题包括：" + "、".join(keywords[:8]) + "。")
    else:
        lines.append("暂时没有足够稳定的近期话题信息。")

    if recent_archives:
        for archive in recent_archives:
            date = _short_date(archive.ended_at)
            summary = archive.summary or archive.title
            if not summary:
                continue
            lines.append(f"{date} 参与了「{archive.title or '未命名会话'}」：{summary}")
    else:
        lines.append("暂无新的可归档会话。")
    for edit in additions[-5:]:
        lines.append(f"{_short_date(edit.created_at)} 补充信息：{edit.content}")

    prior = _filter_deleted_lines(profile.profile, deletions).strip()
    lines.extend(["", "**Brief history**", "*Recent months*"])
    prior_body = _strip_profile_headings(prior)
    if prior_body:
        lines.append(prior_body)
    else:
        lines.append("暂无更早期的稳定背景。")
    lines.extend(["", "*Earlier context*", "暂无更早期的可确认上下文。"])
    lines.extend(["", "*Long-term background*", "暂无长期背景。"])

    body = "\n".join(lines).strip()
    return _replace_nickname_with_placeholder(body, profile.nickname)


async def generate_profile_with_llm(
    *,
    profile: MemoryProfile,
    archives: list[MemoryArchive],
    pending_edits: list[MemoryProfileEdit],
    model: Any,
    api_key: str | None = None,
) -> str | None:
    from bampy.ai.stream import complete_simple
    from bampy.ai.types import Context, SimpleStreamOptions, StopReason, TextContent, UserMessage

    prompt = _build_llm_profile_prompt(
        profile=profile,
        archives=archives,
        pending_edits=pending_edits,
    )
    if len(prompt) > _LLM_PROFILE_MAX_INPUT_CHARS:
        prompt = prompt[:_LLM_PROFILE_MAX_INPUT_CHARS].rstrip() + "\n...(已截断)"

    context = Context(
        system_prompt=_LLM_PROFILE_SYSTEM_PROMPT,
        messages=[UserMessage(content=[TextContent(text=prompt)])],
    )
    options = SimpleStreamOptions(
        api_key=api_key,
        temperature=0.2,
    )
    try:
        result = await complete_simple(model, context, options)
    except Exception:
        logger.opt(exception=True).warning(
            f"bampi_chat memory LLM profile generation failed model={getattr(model, 'id', model)}"
        )
        return None

    if getattr(result, "stop_reason", None) == StopReason.ERROR:
        logger.warning(
            "bampi_chat memory LLM profile generation returned error "
            f"model={getattr(model, 'id', model)} error={getattr(result, 'error_message', '')}"
        )
        return None

    text = _assistant_text(result)
    if not text.strip():
        return None
    cleaned = _clean_llm_profile(
        text,
        nickname=profile.nickname,
        delete_edits=[edit for edit in pending_edits if edit.edit_type == "delete"],
    )
    if cleaned is None:
        logger.warning(
            "bampi_chat memory LLM profile generation returned invalid structure "
            f"model={getattr(model, 'id', model)} group_id={profile.group_id} user_id={profile.user_id}"
        )
    return cleaned


def _render_profile_body(
    profile_text: str,
    *,
    nickname: str,
    edits: list[MemoryProfileEdit],
    truncate_chars: int,
) -> str:
    delete_edits = [edit for edit in edits if edit.edit_type == "delete"]
    body = _filter_deleted_lines(profile_text, delete_edits)
    body = body.replace("{nickname}", nickname or "{nickname}").strip()
    if truncate_chars > 0 and len(body) > truncate_chars:
        body = body[: max(0, truncate_chars - 3)].rstrip() + "..."

    delete_terms = [edit.content.strip() for edit in delete_edits if edit.content.strip()]
    additions = [
        edit
        for edit in edits
        if edit.edit_type in {"add", "update"}
        and not any(term in edit.content for term in delete_terms)
    ]
    if additions:
        lines = [body, "", "[近期补充]"] if body else ["[近期补充]"]
        for edit in additions:
            lines.append(f"- {edit.content}。记录于 {_short_date(edit.created_at)}")
        body = "\n".join(lines)

    if delete_edits:
        lines = [body, "", "[已删除或失效的记忆]"] if body else ["[已删除或失效的记忆]"]
        for edit in delete_edits:
            lines.append(f"- 不要再使用或提及：{edit.content}")
        body = "\n".join(lines)

    return body


def _build_llm_profile_prompt(
    *,
    profile: MemoryProfile,
    archives: list[MemoryArchive],
    pending_edits: list[MemoryProfileEdit],
) -> str:
    deletions = [edit for edit in pending_edits if edit.edit_type == "delete"]
    old_profile = _filter_deleted_lines(profile.profile, deletions).strip() or "无"
    archive_text = _render_archives_for_prompt(archives)
    edit_text = _render_edits_for_prompt(pending_edits)
    nickname = profile.nickname or "未知群名片"
    return f"""\
请为这个群成员生成新的主画像。

<current_display_name_for_reference_do_not_use>
{nickname}
</current_display_name_for_reference_do_not_use>

<metadata>
profile_version: {profile.version}
pending_sessions: {profile.pending_sessions}
</metadata>

<old_profile>
{old_profile}
</old_profile>

<recent_archives>
{archive_text}
</recent_archives>

<pending_edits>
{edit_text}
</pending_edits>

请综合以上信息输出新的画像正文。输出中指代本人时，必须写占位符 `{{nickname}}`，不要把上面的当前群名片写进正文。

注意：以上归档来自群聊，每条归档可能包含多人参与。只使用「{nickname}」本人发起或认同的内容来构建画像，\
其他参与者各自独立讨论的话题不要写入本画像。
"""


def _render_archives_for_prompt(archives: list[MemoryArchive]) -> str:
    if not archives:
        return "无"
    lines: list[str] = []
    for archive in sorted(archives, key=lambda item: item.ended_at, reverse=True)[:20]:
        participants = "、".join(
            participant.nickname
            for participant in archive.participants[:8]
            if participant.nickname
        )
        keywords = "、".join(keyword for keyword in archive.keywords if keyword.strip())
        lines.extend(
            [
                f"- 时间: {_short_date(archive.ended_at)}",
                f"  标题: {archive.title or '未命名会话'}",
                f"  摘要: {archive.summary or '无摘要'}",
                f"  关键词: {keywords or '无'}",
                f"  参与成员: {participants or '未知'}",
            ]
        )
    return "\n".join(lines)


def _render_edits_for_prompt(edits: list[MemoryProfileEdit]) -> str:
    if not edits:
        return "无"
    lines: list[str] = []
    for edit in edits:
        if edit.edit_type == "delete":
            action = "delete/忘记/失效，必须从主画像移除"
        elif edit.edit_type == "update":
            action = "update/更新"
        else:
            action = "add/补充"
        lines.append(f"- [{action}] {_short_date(edit.created_at)}: {edit.content}")
    return "\n".join(lines)


def _assistant_text(result: Any) -> str:
    text = ""
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", "") == "text":
            text += getattr(block, "text", "")
    return text


def _clean_llm_profile(
    text: str,
    *,
    nickname: str,
    delete_edits: list[MemoryProfileEdit],
) -> str | None:
    body = _strip_markdown_fence(text).strip()
    body = _strip_profile_label(body).replace("\r\n", "\n").replace("\r", "\n")
    body = _filter_deleted_lines(body, delete_edits)
    body = _replace_nickname_with_placeholder(body, nickname)
    body = _collapse_blank_lines(body)
    if not body.strip():
        return None
    if not _has_profile_section_structure(body):
        return None
    return body


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _strip_profile_label(text: str) -> str:
    body = text.lstrip()
    labels = ("profile:", "用户画像:", "用户画像：", "画像:", "画像：", "主画像:", "主画像：")
    lower = body.lower()
    for label in labels:
        if lower.startswith(label.lower()):
            return body[len(label) :].lstrip(" \n:-：")
    return body


def _replace_nickname_with_placeholder(text: str, nickname: str) -> str:
    normalized = nickname.strip()
    if len(normalized) < 2:
        return text
    return text.replace(normalized, "{nickname}")


def _collapse_blank_lines(text: str) -> str:
    lines: list[str] = []
    blank = False
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped:
            if not blank:
                lines.append("")
            blank = True
            continue
        lines.append(stripped)
        blank = False
    return "\n".join(lines).strip()


def _has_profile_section_structure(text: str) -> bool:
    folded = text.lower()
    return all(heading.lower() in folded for heading in _PROFILE_SECTION_HEADINGS)


def _strip_profile_headings(text: str) -> str:
    headings = {
        "基本信息",
        "兴趣与话题",
        "近期动态",
        "早期背景",
        *{heading.lower() for heading in _PROFILE_SECTION_HEADINGS},
        *{heading.lower() for heading in _PROFILE_HISTORY_HEADINGS},
    }
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().strip("*")
        if stripped.lower() in headings:
            continue
        lines.append(line)
    return _collapse_blank_lines("\n".join(lines))


def _filter_deleted_lines(text: str, delete_edits: list[MemoryProfileEdit]) -> str:
    if not text or not delete_edits:
        return text or ""
    delete_terms = [edit.content.strip() for edit in delete_edits if edit.content.strip()]
    if not delete_terms:
        return text
    lines: list[str] = []
    for line in text.splitlines():
        if any(term in line for term in delete_terms):
            continue
        lines.append(line)
    return "\n".join(lines)


def _top_keywords(archives: list[MemoryArchive]) -> list[str]:
    counts: dict[str, int] = {}
    for archive in archives:
        for keyword in archive.keywords:
            normalized = keyword.strip()
            if not normalized:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
    return [
        keyword
        for keyword, _count in sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    ]


def _short_date(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "未知时间"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return value[:10] if len(value) >= 10 else value
