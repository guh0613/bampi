from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ArchiveMessageRole = Literal["user", "assistant"]
OpenArchiveMode = Literal["compact", "transcript", "tools", "full"]
ProfileEditType = Literal["add", "update", "delete"]


@dataclass(slots=True)
class MemoryParticipant:
    user_id: str
    nickname: str = ""

    @classmethod
    def from_raw(cls, value: Any) -> "MemoryParticipant":
        if isinstance(value, MemoryParticipant):
            return value
        if isinstance(value, dict):
            return cls(
                user_id=str(value.get("user_id", "")).strip(),
                nickname=str(value.get("nickname", "")).strip(),
            )
        return cls(user_id=str(value).strip())

    def to_json(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "nickname": self.nickname,
        }


@dataclass(slots=True)
class MemoryMessage:
    role: ArchiveMessageRole
    content: str
    timestamp: str
    user_id: str = ""
    nickname: str = ""
    id: int | None = None
    archive_id: int | None = None
    group_id: str = ""


@dataclass(slots=True)
class MemoryToolEvent:
    timestamp: str
    tool_call_id: str = ""
    tool_name: str = ""
    arguments_text: str = ""
    result_preview: str = ""
    result_full: str = ""
    is_error: bool = False
    id: int | None = None
    archive_id: int | None = None
    group_id: str = ""


@dataclass(slots=True)
class MemoryArchive:
    id: int
    group_id: str
    started_at: str
    ended_at: str
    participants: list[MemoryParticipant] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    message_count: int = 0
    created_at: str = ""


@dataclass(slots=True)
class MemorySnippet:
    source: Literal["messages", "tool_events", "archive"]
    text: str
    message_id: int | None = None
    tool_event_id: int | None = None
    role: str = ""
    nickname: str = ""
    tool_name: str = ""


@dataclass(slots=True)
class MemorySearchHit:
    archive: MemoryArchive
    score: float
    matched_sources: list[str] = field(default_factory=list)
    snippets: list[MemorySnippet] = field(default_factory=list)


@dataclass(slots=True)
class MemoryOpenedArchive:
    archive: MemoryArchive
    messages: list[MemoryMessage] = field(default_factory=list)
    tool_events: list[MemoryToolEvent] = field(default_factory=list)
    text: str = ""


@dataclass(slots=True)
class MemoryUserTurn:
    user_id: str
    nickname: str = ""
    timestamp: float | None = None


@dataclass(slots=True)
class MemoryProfile:
    user_id: str
    group_id: str
    nickname: str = ""
    profile: str = ""
    version: int = 1
    pending_sessions: int = 0
    updated_at: str = ""
    last_active_at: str = ""


@dataclass(slots=True)
class MemoryProfileEdit:
    user_id: str
    group_id: str
    edit_type: ProfileEditType
    content: str
    created_at: str
    consolidated: bool = False
    id: int | None = None
