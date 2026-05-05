from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .types import MemoryArchive, MemoryParticipant, MemoryProfile, MemoryProfileEdit


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
    max_chars: int,
) -> str:
    nickname = profile.nickname or "{nickname}"
    keywords = _top_keywords(archives)
    recent_archives = sorted(archives, key=lambda archive: archive.ended_at, reverse=True)[:8]
    additions = [edit for edit in pending_edits if edit.edit_type in {"add", "update"}]
    deletions = [edit for edit in pending_edits if edit.edit_type == "delete"]

    lines: list[str] = [
        "基本信息",
        f"{{nickname}} 是本群成员。最近看到的群名片是 {nickname}。",
        "",
        "兴趣与话题",
    ]
    if keywords:
        lines.append("近期经常参与的话题包括：" + "、".join(keywords[:8]) + "。")
    else:
        lines.append("暂时没有足够稳定的话题信息。")

    lines.extend(["", "近期动态"])
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
    lines.extend(["", "早期背景"])
    if prior:
        lines.append(prior)
    else:
        lines.append("暂无更早期的稳定背景。")

    return _truncate_profile("\n".join(lines), max_chars=max_chars)


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


def _truncate_profile(text: str, *, max_chars: int) -> str:
    limit = max(200, max_chars)
    if len(text) <= limit:
        return text.strip()
    return text[: max(0, limit - 3)].rstrip() + "..."
