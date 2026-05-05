from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import (
    MemoryArchive,
    MemoryMessage,
    MemoryOpenedArchive,
    MemoryParticipant,
    MemoryProfile,
    MemoryProfileEdit,
    MemorySearchHit,
    MemoryToolEvent,
    OpenArchiveMode,
)

if TYPE_CHECKING:
    from .store import MemoryStore


class MemoryArchiveStore:
    """Logical archive partition over the shared memory SQLite database."""

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    def add(
        self,
        *,
        group_id: str,
        started_at: str,
        ended_at: str,
        participants: list[MemoryParticipant | dict[str, Any] | str] | None = None,
        title: str = "",
        summary: str = "",
        keywords: list[str] | None = None,
        messages: list[MemoryMessage | dict[str, Any]] | None = None,
        tool_events: list[MemoryToolEvent | dict[str, Any]] | None = None,
        created_at: str | None = None,
    ) -> int:
        return self._store.add_archive(
            group_id=group_id,
            started_at=started_at,
            ended_at=ended_at,
            participants=participants,
            title=title,
            summary=summary,
            keywords=keywords,
            messages=messages,
            tool_events=tool_events,
            created_at=created_at,
        )

    def search(
        self,
        *,
        group_id: str,
        query: str,
        user_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        max_results: int = 5,
    ) -> list[MemorySearchHit]:
        return self._store.search(
            group_id=group_id,
            query=query,
            user_id=user_id,
            after=after,
            before=before,
            max_results=max_results,
        )

    def time_search(
        self,
        *,
        group_id: str,
        start_time: str | None = None,
        end_time: str | None = None,
        user_id: str | None = None,
        max_results: int = 5,
    ) -> list[MemorySearchHit]:
        return self._store.time_search(
            group_id=group_id,
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            max_results=max_results,
        )

    def open(
        self,
        *,
        archive_id: int,
        group_id: str | None = None,
        mode: OpenArchiveMode = "compact",
        around_message_id: int | None = None,
        before: int = 8,
        after: int = 8,
        include_tool_results: bool = False,
    ) -> MemoryOpenedArchive | None:
        return self._store.open_archive(
            archive_id=archive_id,
            group_id=group_id,
            mode=mode,
            around_message_id=around_message_id,
            before=before,
            after=after,
            include_tool_results=include_tool_results,
        )

    def delete(self, *, archive_id: int, group_id: str | None = None) -> bool:
        return self._store.delete_archive(archive_id=archive_id, group_id=group_id)


class MemoryProfileStore:
    """Logical profile partition over the shared memory SQLite database."""

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    def touch(
        self,
        *,
        group_id: str,
        user_id: str,
        nickname: str = "",
        last_active_at: str | None = None,
    ) -> None:
        self._store.touch_profile(
            group_id=group_id,
            user_id=user_id,
            nickname=nickname,
            last_active_at=last_active_at,
        )

    def add_edit(
        self,
        *,
        group_id: str,
        user_id: str,
        edit_type: str,
        content: str,
        nickname: str = "",
        created_at: str | None = None,
    ) -> MemoryProfileEdit:
        return self._store.add_profile_edit(
            group_id=group_id,
            user_id=user_id,
            edit_type=edit_type,
            content=content,
            nickname=nickname,
            created_at=created_at,
        )

    def get(self, *, group_id: str, user_id: str) -> MemoryProfile | None:
        return self._store.get_profile(group_id=group_id, user_id=user_id)

    def pending_edits(
        self,
        *,
        group_id: str,
        user_id: str,
        limit: int | None = None,
    ) -> list[MemoryProfileEdit]:
        return self._store.get_pending_profile_edits(
            group_id=group_id,
            user_id=user_id,
            limit=limit,
        )

    def due_for_generation(
        self,
        *,
        session_threshold: int,
        max_staleness_days: int,
    ) -> list[MemoryProfile]:
        return self._store.get_profiles_due_for_generation(
            session_threshold=session_threshold,
            max_staleness_days=max_staleness_days,
        )

    def archives_for_generation(
        self,
        *,
        group_id: str,
        user_id: str,
        since: str | None = None,
        limit: int = 20,
    ) -> list[MemoryArchive]:
        return self._store.get_archives_for_profile_generation(
            group_id=group_id,
            user_id=user_id,
            since=since,
            limit=limit,
        )

    def consolidate(
        self,
        *,
        group_id: str,
        user_id: str,
        profile: str,
        edit_ids: list[int] | None = None,
        updated_at: str | None = None,
    ) -> MemoryProfile:
        return self._store.consolidate_profile(
            group_id=group_id,
            user_id=user_id,
            profile=profile,
            edit_ids=edit_ids,
            updated_at=updated_at,
        )

    def delete_user_memory(
        self,
        *,
        group_id: str,
        user_id: str,
        delete_messages: bool = True,
    ) -> dict[str, int]:
        return self._store.delete_user_memory(
            group_id=group_id,
            user_id=user_id,
            delete_messages=delete_messages,
        )


class MemoryMaintenanceStore:
    """Logical maintenance partition over the shared memory SQLite database."""

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    def cleanup_old_data(self, *, archive_retention_days: int) -> int:
        return self._store.cleanup_old_data(archive_retention_days=archive_retention_days)
