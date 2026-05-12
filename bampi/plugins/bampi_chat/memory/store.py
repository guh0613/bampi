from __future__ import annotations

import json
from nonebot import logger
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .embeddings import MemoryEmbeddingProvider
from .schema import CURRENT_SCHEMA_VERSION, initialize_memory_schema
from .search_text import (
    build_fts_query,
    build_search_text,
    like_terms,
    normalize_for_search,
    required_entity_groups,
)
from .store_parts import MemoryArchiveStore, MemoryMaintenanceStore, MemoryProfileStore
from .types import (
    MemoryArchive,
    MemoryMessage,
    MemoryOpenedArchive,
    MemoryParticipant,
    MemoryProfile,
    MemoryProfileEdit,
    MemorySearchHit,
    MemorySnippet,
    MemoryToolEvent,
    OpenArchiveMode,
)
from .vector_index import SqliteVecArchiveIndex



@dataclass(slots=True)
class _Candidate:
    archive_id: int
    score: float = 0.0
    matched_sources: set[str] = field(default_factory=set)
    message_ids: set[int] = field(default_factory=set)
    tool_event_ids: set[int] = field(default_factory=set)

    def add(self, source: str, weight: float, *, rank: float | None = None) -> None:
        rank_boost = 0.0 if rank is None else 1.0 / (1.0 + abs(rank))
        self.score += weight + rank_boost
        self.matched_sources.add(source)


class MemoryStore:
    def __init__(
        self,
        db_path: str | Path,
        *,
        search_snippet_messages: int = 2,
        like_fallback: bool = True,
        embedding_provider: MemoryEmbeddingProvider | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._search_snippet_messages = max(0, search_snippet_messages)
        self._like_fallback = like_fallback
        self._embedding_provider = embedding_provider
        self._vector_index = (
            SqliteVecArchiveIndex(
                provider=embedding_provider.provider,
                model=embedding_provider.model,
            )
            if embedding_provider is not None
            else None
        )
        self.archives = MemoryArchiveStore(self)
        self.profiles = MemoryProfileStore(self)
        self.maintenance = MemoryMaintenanceStore(self)
        self.initialize()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def schema_version(self) -> int:
        return CURRENT_SCHEMA_VERSION

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            initialize_memory_schema(conn)
            if self._vector_index is not None:
                self._vector_index.initialize_connection(conn)
                if (
                    self._embedding_provider is not None
                    and self._embedding_provider.dimensions > 0
                ):
                    self._vector_index.ensure_ready(
                        conn,
                        dimension=self._embedding_provider.dimensions,
                    )
            conn.commit()

    def add_archive(
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
        normalized_group_id = str(group_id).strip()
        normalized_messages = [self._message_from_raw(item) for item in messages or []]
        normalized_tool_events = [self._tool_event_from_raw(item) for item in tool_events or []]
        normalized_participants = self._normalize_participants(participants, normalized_messages)
        normalized_keywords = [str(keyword).strip() for keyword in keywords or [] if str(keyword).strip()]
        now = created_at or _now_iso()
        embedding_text = "\n".join([title, summary, " ".join(normalized_keywords)])

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversation_archives (
                    group_id,
                    started_at,
                    ended_at,
                    participants,
                    title,
                    summary,
                    keywords,
                    message_count,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_group_id,
                    started_at,
                    ended_at,
                    _json_dumps([participant.to_json() for participant in normalized_participants]),
                    title,
                    summary,
                    _json_dumps(normalized_keywords),
                    len(normalized_messages),
                    now,
                ),
            )
            archive_id = int(cursor.lastrowid)
            archive_search_text = build_search_text(
                title,
                summary,
                " ".join(normalized_keywords),
                " ".join(participant.nickname for participant in normalized_participants),
                " ".join(participant.user_id for participant in normalized_participants),
            )
            conn.execute(
                "INSERT INTO archive_fts(archive_id, group_id, search_text) VALUES (?, ?, ?)",
                (archive_id, normalized_group_id, archive_search_text),
            )

            for message in normalized_messages:
                message_id = self._insert_message(
                    conn,
                    archive_id=archive_id,
                    group_id=normalized_group_id,
                    message=message,
                )
                conn.execute(
                    """
                    INSERT INTO message_fts(message_id, archive_id, group_id, search_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        archive_id,
                        normalized_group_id,
                        build_search_text(
                            message.role,
                            message.user_id,
                            message.nickname,
                            message.content,
                        ),
                    ),
                )

            for event in normalized_tool_events:
                tool_event_id = self._insert_tool_event(
                    conn,
                    archive_id=archive_id,
                    group_id=normalized_group_id,
                    event=event,
                )
                conn.execute(
                    """
                    INSERT INTO tool_event_fts(tool_event_id, archive_id, group_id, search_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        tool_event_id,
                        archive_id,
                        normalized_group_id,
                        build_search_text(
                            event.tool_name,
                            event.arguments_text,
                            event.result_preview,
                            event.result_full,
                        ),
                    ),
                )

            self._mark_participants_active(
                conn,
                group_id=normalized_group_id,
                participants=normalized_participants,
                last_active_at=ended_at,
            )
            conn.commit()
        self._insert_archive_embedding(
            archive_id=archive_id,
            group_id=normalized_group_id,
            text=embedding_text,
            created_at=now,
        )
        return archive_id

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
        normalized_group_id = str(group_id).strip()
        normalized_user_id = str(user_id).strip() if user_id is not None else None
        normalized_query = normalize_for_search(query)
        limit = min(max(1, int(max_results or 5)), 10)

        with self._connect() as conn:
            candidates: dict[int, _Candidate] = {}
            if normalized_query:
                self._collect_fts_candidates(
                    conn,
                    group_id=normalized_group_id,
                    query=normalized_query,
                    candidates=candidates,
                )
                if self._like_fallback:
                    self._collect_like_candidates(
                        conn,
                        group_id=normalized_group_id,
                        query=normalized_query,
                        candidates=candidates,
                    )
                self._collect_embedding_candidates(
                    conn,
                    group_id=normalized_group_id,
                    query=normalized_query,
                    candidates=candidates,
                )
            else:
                for row in conn.execute(
                    """
                    SELECT id
                    FROM conversation_archives
                    WHERE group_id = ?
                    ORDER BY ended_at DESC
                    LIMIT ?
                    """,
                    (normalized_group_id, limit),
                ):
                    candidates[int(row["id"])] = _Candidate(archive_id=int(row["id"]))

            hits: list[MemorySearchHit] = []
            entity_groups = required_entity_groups(normalized_query)
            embedding_can_bridge_entities = (
                self._embedding_provider is not None
                and self._embedding_provider.provider != "local-hash"
            )
            for candidate in candidates.values():
                row = conn.execute(
                    "SELECT * FROM conversation_archives WHERE id = ? AND group_id = ?",
                    (candidate.archive_id, normalized_group_id),
                ).fetchone()
                if row is None:
                    continue
                if (
                    entity_groups
                    and not (
                        embedding_can_bridge_entities
                        and "embedding" in candidate.matched_sources
                    )
                    and not self._archive_contains_entity_groups(
                        conn,
                        archive_id=candidate.archive_id,
                        row=row,
                        entity_groups=entity_groups,
                    )
                ):
                    continue
                if not self._archive_passes_filters(
                    conn,
                    archive_id=candidate.archive_id,
                    row=row,
                    user_id=normalized_user_id,
                    after=after,
                    before=before,
                ):
                    continue
                snippets = self._build_snippets(
                    conn,
                    archive_id=candidate.archive_id,
                    query=normalized_query,
                    message_ids=candidate.message_ids,
                    tool_event_ids=candidate.tool_event_ids,
                )
                if not snippets and (
                    "archive" in candidate.matched_sources
                    or "embedding" in candidate.matched_sources
                ):
                    snippets.append(
                        MemorySnippet(
                            source="archive",
                            text=_make_snippet(
                                " ".join([row["title"], row["summary"], row["keywords"]]),
                                like_terms(normalized_query),
                            ),
                        )
                    )
                hits.append(
                    MemorySearchHit(
                        archive=self._archive_from_row(row),
                        score=candidate.score,
                        matched_sources=sorted(candidate.matched_sources),
                        snippets=snippets,
                    )
                )

            hits.sort(key=lambda hit: (hit.score, hit.archive.ended_at, hit.archive.id), reverse=True)
            return hits[:limit]

    def time_search(
        self,
        *,
        group_id: str,
        start_time: str | None = None,
        end_time: str | None = None,
        user_id: str | None = None,
        max_results: int = 5,
    ) -> list[MemorySearchHit]:
        normalized_group_id = str(group_id).strip()
        normalized_user_id = str(user_id).strip() if user_id is not None else None
        normalized_start_time = normalize_for_search(start_time)
        normalized_end_time = normalize_for_search(end_time)
        start_dt = _parse_required_time_bound(normalized_start_time, name="start_time")
        end_dt = _parse_required_time_bound(normalized_end_time, name="end_time")
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError("start_time must be before or equal to end_time")
        limit = min(max(1, int(max_results or 5)), 10)

        with self._connect() as conn:
            hits: list[MemorySearchHit] = []
            for row in conn.execute(
                """
                SELECT *
                FROM conversation_archives
                WHERE group_id = ?
                ORDER BY ended_at DESC, id DESC
                """,
                (normalized_group_id,),
            ):
                archive_id = int(row["id"])
                if not self._archive_overlaps_time_range(
                    row=row,
                    start_time=normalized_start_time,
                    end_time=normalized_end_time,
                    start_dt=start_dt,
                    end_dt=end_dt,
                ):
                    continue
                if not self._archive_passes_filters(
                    conn,
                    archive_id=archive_id,
                    row=row,
                    user_id=normalized_user_id,
                    after=None,
                    before=None,
                ):
                    continue
                hits.append(
                    MemorySearchHit(
                        archive=self._archive_from_row(row),
                        score=0.0,
                        matched_sources=["time_range"],
                        snippets=self._build_time_snippets(conn, archive_id=archive_id),
                    )
                )
                if len(hits) >= limit:
                    break
            return hits

    def open_archive(
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
        if mode not in {"compact", "transcript", "tools", "full"}:
            raise ValueError("mode must be one of: compact, transcript, tools, full")

        with self._connect() as conn:
            if group_id is None:
                row = conn.execute(
                    "SELECT * FROM conversation_archives WHERE id = ?",
                    (archive_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM conversation_archives WHERE id = ? AND group_id = ?",
                    (archive_id, str(group_id).strip()),
                ).fetchone()
            if row is None:
                return None
            archive = self._archive_from_row(row)
            messages = self._messages_for_archive(conn, archive_id)
            tool_events = self._tool_events_for_archive(conn, archive_id)
            text = self._render_open_archive(
                archive=archive,
                messages=messages,
                tool_events=tool_events,
                mode=mode,
                around_message_id=around_message_id,
                before=before,
                after=after,
                include_tool_results=include_tool_results,
            )
            return MemoryOpenedArchive(
                archive=archive,
                messages=messages,
                tool_events=tool_events,
                text=text,
            )

    def touch_profile(
        self,
        *,
        group_id: str,
        user_id: str,
        nickname: str = "",
        last_active_at: str | None = None,
    ) -> None:
        normalized_user_id = str(user_id).strip()
        normalized_group_id = str(group_id).strip()
        if not normalized_user_id or not normalized_group_id:
            return
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id,
                    group_id,
                    nickname,
                    profile,
                    version,
                    pending_sessions,
                    updated_at,
                    last_active_at
                )
                VALUES (?, ?, ?, '', 1, 0, ?, ?)
                ON CONFLICT(user_id, group_id) DO UPDATE SET
                    nickname = CASE
                        WHEN excluded.nickname != '' THEN excluded.nickname
                        ELSE user_profiles.nickname
                    END,
                    last_active_at = CASE
                        WHEN excluded.last_active_at != '' THEN excluded.last_active_at
                        ELSE user_profiles.last_active_at
                    END
                """,
                (
                    normalized_user_id,
                    normalized_group_id,
                    str(nickname).strip(),
                    now,
                    str(last_active_at or ""),
                ),
            )
            conn.commit()

    def add_profile_edit(
        self,
        *,
        group_id: str,
        user_id: str,
        edit_type: str,
        content: str,
        nickname: str = "",
        created_at: str | None = None,
    ) -> MemoryProfileEdit:
        normalized_user_id = str(user_id).strip()
        normalized_group_id = str(group_id).strip()
        normalized_type = str(edit_type).strip().lower()
        normalized_content = str(content).strip()
        if normalized_type not in {"add", "update", "delete"}:
            raise ValueError("edit_type must be one of: add, update, delete")
        if not normalized_user_id:
            raise ValueError("user_id must not be empty")
        if not normalized_group_id:
            raise ValueError("group_id must not be empty")
        if not normalized_content:
            raise ValueError("content must not be empty")

        now = created_at or _now_iso()
        with self._connect() as conn:
            self._touch_profile_in_conn(
                conn,
                group_id=normalized_group_id,
                user_id=normalized_user_id,
                nickname=nickname,
                last_active_at=now,
            )
            cursor = conn.execute(
                """
                INSERT INTO profile_edits (
                    user_id,
                    group_id,
                    edit_type,
                    content,
                    created_at,
                    consolidated
                )
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (
                    normalized_user_id,
                    normalized_group_id,
                    normalized_type,
                    normalized_content,
                    now,
                ),
            )
            edit_id = int(cursor.lastrowid)
            conn.commit()
        return MemoryProfileEdit(
            id=edit_id,
            user_id=normalized_user_id,
            group_id=normalized_group_id,
            edit_type=normalized_type,  # type: ignore[arg-type]
            content=normalized_content,
            created_at=now,
            consolidated=False,
        )

    def get_profile(self, *, group_id: str, user_id: str) -> MemoryProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM user_profiles
                WHERE group_id = ? AND user_id = ?
                """,
                (str(group_id).strip(), str(user_id).strip()),
            ).fetchone()
            return self._profile_from_row(row) if row is not None else None

    def get_pending_profile_edits(
        self,
        *,
        group_id: str,
        user_id: str,
        limit: int | None = None,
    ) -> list[MemoryProfileEdit]:
        query_limit = "" if limit is None else " LIMIT ?"
        params: tuple[Any, ...] = (str(group_id).strip(), str(user_id).strip())
        if limit is not None:
            params = (*params, max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM profile_edits
                WHERE group_id = ? AND user_id = ? AND consolidated = 0
                ORDER BY created_at ASC, id ASC
                {query_limit}
                """,
                params,
            ).fetchall()
            return [self._profile_edit_from_row(row) for row in rows]

    def get_profiles_due_for_generation(
        self,
        *,
        session_threshold: int,
        max_staleness_days: int,
    ) -> list[MemoryProfile]:
        threshold = max(1, int(session_threshold))
        stale_days = max(0, int(max_staleness_days))
        cutoff = datetime.now(timezone.utc) - timedelta(days=stale_days)
        profiles: list[MemoryProfile] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM user_profiles
                WHERE pending_sessions > 0
                ORDER BY pending_sessions DESC, last_active_at DESC
                """
            ).fetchall()
            for row in rows:
                profile = self._profile_from_row(row)
                if profile.pending_sessions >= threshold:
                    profiles.append(profile)
                    continue
                updated_at = _parse_iso_datetime(profile.updated_at)
                if updated_at is None or updated_at <= cutoff:
                    profiles.append(profile)
        return profiles

    def get_archives_for_profile_generation(
        self,
        *,
        group_id: str,
        user_id: str,
        since: str | None = None,
        limit: int = 20,
    ) -> list[MemoryArchive]:
        normalized_group_id = str(group_id).strip()
        normalized_user_id = str(user_id).strip()
        since_dt = _parse_iso_datetime(since or "")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM conversation_archives
                WHERE group_id = ?
                ORDER BY ended_at DESC
                LIMIT ?
                """,
                (normalized_group_id, max(1, int(limit)) * 4),
            ).fetchall()
            archives: list[MemoryArchive] = []
            for row in rows:
                ended_at = _parse_iso_datetime(row["ended_at"])
                if since_dt is not None and ended_at is not None and ended_at <= since_dt:
                    continue
                participants = self._participants_from_json(row["participants"])
                participant_match = any(participant.user_id == normalized_user_id for participant in participants)
                message_match = conn.execute(
                    """
                    SELECT 1
                    FROM messages
                    WHERE archive_id = ? AND user_id = ?
                    LIMIT 1
                    """,
                    (int(row["id"]), normalized_user_id),
                ).fetchone()
                if not participant_match and message_match is None:
                    continue
                archives.append(self._archive_from_row(row))
                if len(archives) >= max(1, int(limit)):
                    break
            return archives

    def consolidate_profile(
        self,
        *,
        group_id: str,
        user_id: str,
        profile: str,
        edit_ids: list[int] | None = None,
        updated_at: str | None = None,
    ) -> MemoryProfile:
        normalized_group_id = str(group_id).strip()
        normalized_user_id = str(user_id).strip()
        now = updated_at or _now_iso()
        with self._connect() as conn:
            self._touch_profile_in_conn(
                conn,
                group_id=normalized_group_id,
                user_id=normalized_user_id,
                nickname="",
                last_active_at="",
            )
            conn.execute(
                """
                UPDATE user_profiles
                SET profile = ?,
                    version = version + 1,
                    updated_at = ?,
                    pending_sessions = 0
                WHERE group_id = ? AND user_id = ?
                """,
                (str(profile).strip(), now, normalized_group_id, normalized_user_id),
            )
            if edit_ids:
                placeholders = ",".join("?" for _ in edit_ids)
                conn.execute(
                    f"""
                    UPDATE profile_edits
                    SET consolidated = 1
                    WHERE group_id = ?
                      AND user_id = ?
                      AND id IN ({placeholders})
                    """,
                    (normalized_group_id, normalized_user_id, *edit_ids),
                )
            else:
                conn.execute(
                    """
                    UPDATE profile_edits
                    SET consolidated = 1
                    WHERE group_id = ? AND user_id = ? AND consolidated = 0
                    """,
                    (normalized_group_id, normalized_user_id),
                )
            conn.commit()
            row = conn.execute(
                """
                SELECT *
                FROM user_profiles
                WHERE group_id = ? AND user_id = ?
                """,
                (normalized_group_id, normalized_user_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("profile consolidation failed")
            return self._profile_from_row(row)

    def cleanup_old_data(self, *, archive_retention_days: int) -> int:
        days = int(archive_retention_days)
        if days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, ended_at
                FROM conversation_archives
                """,
            ).fetchall()
            archive_ids = [
                int(row["id"])
                for row in rows
                if (parsed := _parse_iso_datetime(row["ended_at"])) is not None and parsed < cutoff
            ]
            for archive_id in archive_ids:
                self._delete_archive_in_conn(conn, archive_id=archive_id)
            conn.commit()
            return len(archive_ids)

    def delete_archive(self, *, archive_id: int, group_id: str | None = None) -> bool:
        with self._connect() as conn:
            if group_id is not None:
                row = conn.execute(
                    "SELECT id FROM conversation_archives WHERE id = ? AND group_id = ?",
                    (archive_id, str(group_id).strip()),
                ).fetchone()
                if row is None:
                    return False
            deleted = self._delete_archive_in_conn(conn, archive_id=int(archive_id))
            conn.commit()
            return deleted

    def delete_user_memory(
        self,
        *,
        group_id: str,
        user_id: str,
        delete_messages: bool = True,
    ) -> dict[str, int]:
        normalized_group_id = str(group_id).strip()
        normalized_user_id = str(user_id).strip()
        with self._connect() as conn:
            profile_deleted = conn.execute(
                """
                DELETE FROM user_profiles
                WHERE group_id = ? AND user_id = ?
                """,
                (normalized_group_id, normalized_user_id),
            ).rowcount
            edits_deleted = conn.execute(
                """
                DELETE FROM profile_edits
                WHERE group_id = ? AND user_id = ?
                """,
                (normalized_group_id, normalized_user_id),
            ).rowcount
            messages_deleted = 0
            archives_touched: set[int] = set()
            if delete_messages:
                rows = conn.execute(
                    """
                    SELECT id, archive_id
                    FROM messages
                    WHERE group_id = ? AND user_id = ?
                    """,
                    (normalized_group_id, normalized_user_id),
                ).fetchall()
                message_ids = [int(row["id"]) for row in rows]
                archives_touched = {int(row["archive_id"]) for row in rows}
                if message_ids:
                    placeholders = ",".join("?" for _ in message_ids)
                    conn.execute(
                        f"DELETE FROM message_fts WHERE message_id IN ({placeholders})",
                        tuple(message_ids),
                    )
                    messages_deleted = conn.execute(
                        f"DELETE FROM messages WHERE id IN ({placeholders})",
                        tuple(message_ids),
                    ).rowcount
                self._remove_user_from_archive_participants(
                    conn,
                    group_id=normalized_group_id,
                    user_id=normalized_user_id,
                    archive_ids=archives_touched,
                )
            conn.commit()
        return {
            "profiles_deleted": int(profile_deleted),
            "edits_deleted": int(edits_deleted),
            "messages_deleted": int(messages_deleted),
            "archives_touched": len(archives_touched),
        }

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        if self._vector_index is not None:
            self._vector_index.load_connection(conn)
        try:
            yield conn
        finally:
            conn.close()

    def _collect_fts_candidates(
        self,
        conn: sqlite3.Connection,
        *,
        group_id: str,
        query: str,
        candidates: dict[int, _Candidate],
    ) -> None:
        fts_query = build_fts_query(query)
        if not fts_query:
            return
        self._collect_fts_table(
            conn,
            table="archive_fts",
            select_id="archive_id",
            source="archive",
            weight=7.0,
            group_id=group_id,
            fts_query=fts_query,
            candidates=candidates,
        )
        self._collect_fts_table(
            conn,
            table="message_fts",
            select_id="message_id",
            source="messages",
            weight=4.0,
            group_id=group_id,
            fts_query=fts_query,
            candidates=candidates,
        )
        self._collect_fts_table(
            conn,
            table="tool_event_fts",
            select_id="tool_event_id",
            source="tool_events",
            weight=5.0,
            group_id=group_id,
            fts_query=fts_query,
            candidates=candidates,
        )

    def _collect_fts_table(
        self,
        conn: sqlite3.Connection,
        *,
        table: str,
        select_id: str,
        source: str,
        weight: float,
        group_id: str,
        fts_query: str,
        candidates: dict[int, _Candidate],
    ) -> None:
        try:
            rows = conn.execute(
                f"""
                SELECT {select_id}, archive_id, bm25({table}) AS rank
                FROM {table}
                WHERE {table} MATCH ? AND group_id = ?
                """,
                (fts_query, group_id),
            ).fetchall()
        except sqlite3.OperationalError:
            return
        for row in rows:
            archive_id = int(row["archive_id"])
            candidate = candidates.setdefault(archive_id, _Candidate(archive_id=archive_id))
            candidate.add(source, weight, rank=float(row["rank"]))
            if source == "messages":
                candidate.message_ids.add(int(row[select_id]))
            elif source == "tool_events":
                candidate.tool_event_ids.add(int(row[select_id]))

    def _collect_like_candidates(
        self,
        conn: sqlite3.Connection,
        *,
        group_id: str,
        query: str,
        candidates: dict[int, _Candidate],
    ) -> None:
        for term in like_terms(query):
            pattern = f"%{_escape_like(term)}%"
            for row in conn.execute(
                """
                SELECT id AS archive_id
                FROM conversation_archives
                WHERE group_id = ?
                  AND (
                    title LIKE ? ESCAPE '\\'
                    OR summary LIKE ? ESCAPE '\\'
                    OR keywords LIKE ? ESCAPE '\\'
                    OR participants LIKE ? ESCAPE '\\'
                  )
                """,
                (group_id, pattern, pattern, pattern, pattern),
            ):
                archive_id = int(row["archive_id"])
                candidates.setdefault(archive_id, _Candidate(archive_id=archive_id)).add("archive", 3.0)

            for row in conn.execute(
                """
                SELECT id, archive_id
                FROM messages
                WHERE group_id = ?
                  AND (
                    content LIKE ? ESCAPE '\\'
                    OR nickname LIKE ? ESCAPE '\\'
                    OR user_id LIKE ? ESCAPE '\\'
                  )
                """,
                (group_id, pattern, pattern, pattern),
            ):
                archive_id = int(row["archive_id"])
                candidate = candidates.setdefault(archive_id, _Candidate(archive_id=archive_id))
                candidate.add("messages", 2.0)
                candidate.message_ids.add(int(row["id"]))

            for row in conn.execute(
                """
                SELECT id, archive_id
                FROM tool_events
                WHERE group_id = ?
                  AND (
                    tool_name LIKE ? ESCAPE '\\'
                    OR arguments_text LIKE ? ESCAPE '\\'
                    OR result_preview LIKE ? ESCAPE '\\'
                    OR result_full LIKE ? ESCAPE '\\'
                  )
                """,
                (group_id, pattern, pattern, pattern, pattern),
            ):
                archive_id = int(row["archive_id"])
                candidate = candidates.setdefault(archive_id, _Candidate(archive_id=archive_id))
                candidate.add("tool_events", 2.5)
                candidate.tool_event_ids.add(int(row["id"]))

    def _collect_embedding_candidates(
        self,
        conn: sqlite3.Connection,
        *,
        group_id: str,
        query: str,
        candidates: dict[int, _Candidate],
    ) -> None:
        provider = self._embedding_provider
        if provider is None:
            return
        try:
            query_vector = provider.embed_text(query)
        except Exception:
            logger.opt(exception=True).warning(
                f"bampi_chat memory embedding query failed provider={provider.provider} model={provider.model}"
            )
            return
        if not any(query_vector):
            return

        if self._vector_index is None:
            return

        scored: list[tuple[int, float]] = []
        for archive_id, score in self._vector_index.search(
            conn,
            group_id=group_id,
            query_vector=query_vector,
            limit=20,
        ):
            if score <= 0.05:
                continue
            scored.append((archive_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        if not scored:
            return

        best_score = scored[0][1]
        relative_floor = max(0.05, best_score - 0.08, best_score * 0.9)
        for archive_id, score in scored[:20]:
            if score < relative_floor:
                continue
            candidates.setdefault(archive_id, _Candidate(archive_id=archive_id)).add(
                "embedding",
                2.0 + score * 4.0,
            )

    def _archive_passes_filters(
        self,
        conn: sqlite3.Connection,
        *,
        archive_id: int,
        row: sqlite3.Row,
        user_id: str | None,
        after: str | None,
        before: str | None,
    ) -> bool:
        if after and str(row["ended_at"]) < after:
            return False
        if before and str(row["started_at"]) > before:
            return False
        if not user_id:
            return True
        matched = conn.execute(
            """
            SELECT 1
            FROM messages
            WHERE archive_id = ? AND user_id = ?
            LIMIT 1
            """,
            (archive_id, user_id),
        ).fetchone()
        if matched is not None:
            return True
        for participant in self._participants_from_json(row["participants"]):
            if participant.user_id == user_id:
                return True
        return False

    def _archive_contains_entity_groups(
        self,
        conn: sqlite3.Connection,
        *,
        archive_id: int,
        row: sqlite3.Row,
        entity_groups: list[list[str]],
    ) -> bool:
        if not entity_groups:
            return True
        text_parts = [
            row["title"],
            row["summary"],
            row["keywords"],
            row["participants"],
        ]
        text_parts.extend(
            message_row["content"]
            for message_row in conn.execute(
                "SELECT content FROM messages WHERE archive_id = ?",
                (archive_id,),
            )
        )
        for tool_row in conn.execute(
            """
            SELECT tool_name, arguments_text, result_preview, result_full
            FROM tool_events
            WHERE archive_id = ?
            """,
            (archive_id,),
        ):
            text_parts.extend(
                [
                    tool_row["tool_name"],
                    tool_row["arguments_text"],
                    tool_row["result_preview"],
                    tool_row["result_full"],
                ]
            )

        folded = " ".join(str(part or "") for part in text_parts).casefold()
        return all(any(term.casefold() in folded for term in group) for group in entity_groups)

    def _build_snippets(
        self,
        conn: sqlite3.Connection,
        *,
        archive_id: int,
        query: str,
        message_ids: set[int],
        tool_event_ids: set[int],
    ) -> list[MemorySnippet]:
        terms = like_terms(query)
        snippets: list[MemorySnippet] = []
        rows: list[sqlite3.Row] = []
        if message_ids:
            placeholders = ",".join("?" for _ in message_ids)
            rows = conn.execute(
                f"""
                SELECT *
                FROM messages
                WHERE archive_id = ? AND id IN ({placeholders})
                ORDER BY id
                """,
                (archive_id, *sorted(message_ids)),
            ).fetchall()
        elif terms:
            rows = self._matching_message_rows(conn, archive_id=archive_id, terms=terms)

        for row in rows[: self._search_snippet_messages]:
            snippets.append(
                MemorySnippet(
                    source="messages",
                    message_id=int(row["id"]),
                    role=row["role"],
                    nickname=row["nickname"] or ("assistant" if row["role"] == "assistant" else ""),
                    text=_make_snippet(row["content"], terms),
                )
            )

        tool_rows: list[sqlite3.Row] = []
        if tool_event_ids:
            placeholders = ",".join("?" for _ in tool_event_ids)
            tool_rows = conn.execute(
                f"""
                SELECT *
                FROM tool_events
                WHERE archive_id = ? AND id IN ({placeholders})
                ORDER BY id
                """,
                (archive_id, *sorted(tool_event_ids)),
            ).fetchall()
        elif terms and len(snippets) < self._search_snippet_messages:
            tool_rows = self._matching_tool_rows(conn, archive_id=archive_id, terms=terms)

        remaining = max(0, self._search_snippet_messages - len(snippets))
        for row in tool_rows[:remaining]:
            snippets.append(
                MemorySnippet(
                    source="tool_events",
                    tool_event_id=int(row["id"]),
                    tool_name=row["tool_name"],
                    text=_make_snippet(
                        " ".join([row["arguments_text"], row["result_preview"], row["result_full"]]),
                        terms,
                    ),
                )
            )
        return snippets

    def _matching_message_rows(
        self,
        conn: sqlite3.Connection,
        *,
        archive_id: int,
        terms: list[str],
    ) -> list[sqlite3.Row]:
        rows = conn.execute(
            "SELECT * FROM messages WHERE archive_id = ? ORDER BY id",
            (archive_id,),
        ).fetchall()
        return [
            row
            for row in rows
            if _contains_any(" ".join([row["nickname"], row["content"]]), terms)
        ]

    def _matching_tool_rows(
        self,
        conn: sqlite3.Connection,
        *,
        archive_id: int,
        terms: list[str],
    ) -> list[sqlite3.Row]:
        rows = conn.execute(
            "SELECT * FROM tool_events WHERE archive_id = ? ORDER BY id",
            (archive_id,),
        ).fetchall()
        return [
            row
            for row in rows
            if _contains_any(
                " ".join([row["tool_name"], row["arguments_text"], row["result_preview"], row["result_full"]]),
                terms,
            )
        ]

    def _archive_overlaps_time_range(
        self,
        *,
        row: sqlite3.Row,
        start_time: str,
        end_time: str,
        start_dt: datetime | None,
        end_dt: datetime | None,
    ) -> bool:
        archive_started = _parse_iso_datetime(row["started_at"])
        archive_ended = _parse_iso_datetime(row["ended_at"])
        if archive_started is None or archive_ended is None:
            if start_time and str(row["ended_at"]) < start_time:
                return False
            if end_time and str(row["started_at"]) > end_time:
                return False
            return True
        if start_dt is not None and archive_ended < start_dt:
            return False
        if end_dt is not None and archive_started > end_dt:
            return False
        return True

    def _build_time_snippets(
        self,
        conn: sqlite3.Connection,
        *,
        archive_id: int,
    ) -> list[MemorySnippet]:
        if self._search_snippet_messages <= 0:
            return []
        rows = conn.execute(
            """
            SELECT *
            FROM messages
            WHERE archive_id = ?
            ORDER BY id
            """,
            (archive_id,),
        ).fetchall()
        if not rows:
            return []
        selected = rows[: self._search_snippet_messages]
        if len(rows) > self._search_snippet_messages:
            tail_room = max(1, self._search_snippet_messages // 2)
            head_room = max(1, self._search_snippet_messages - tail_room)
            selected = rows[:head_room] + rows[-tail_room:]
        snippets: list[MemorySnippet] = []
        seen: set[int] = set()
        for row in selected:
            message_id = int(row["id"])
            if message_id in seen:
                continue
            seen.add(message_id)
            snippets.append(
                MemorySnippet(
                    source="messages",
                    message_id=message_id,
                    role=row["role"],
                    nickname=row["nickname"] or ("assistant" if row["role"] == "assistant" else ""),
                    text=_truncate(row["content"], 180),
                )
            )
        return snippets

    def _insert_message(
        self,
        conn: sqlite3.Connection,
        *,
        archive_id: int,
        group_id: str,
        message: MemoryMessage,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO messages (
                archive_id,
                group_id,
                user_id,
                nickname,
                role,
                content,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                archive_id,
                group_id,
                message.user_id,
                message.nickname,
                message.role,
                message.content,
                message.timestamp,
            ),
        )
        return int(cursor.lastrowid)

    def _insert_archive_embedding(
        self,
        *,
        archive_id: int,
        group_id: str,
        text: str,
        created_at: str,
    ) -> None:
        provider = self._embedding_provider
        if provider is None:
            return
        try:
            vector = provider.embed_text(text)
        except Exception:
            logger.opt(exception=True).warning(
                f"bampi_chat memory archive embedding failed archive_id={archive_id} provider={provider.provider} model={provider.model}"
            )
            return
        if not any(vector):
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO archive_embeddings (
                    archive_id,
                    group_id,
                    provider,
                    model,
                    dimension,
                    vector,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    archive_id,
                    group_id,
                    provider.provider,
                    provider.model,
                    len(vector),
                    _json_dumps(vector),
                    created_at,
                ),
            )
            if self._vector_index is not None:
                self._vector_index.upsert(
                    conn,
                    archive_id=archive_id,
                    group_id=group_id,
                    vector=vector,
                )
            conn.commit()

    def _insert_tool_event(
        self,
        conn: sqlite3.Connection,
        *,
        archive_id: int,
        group_id: str,
        event: MemoryToolEvent,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO tool_events (
                archive_id,
                group_id,
                tool_call_id,
                tool_name,
                arguments_text,
                result_preview,
                result_full,
                is_error,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                archive_id,
                group_id,
                event.tool_call_id,
                event.tool_name,
                event.arguments_text,
                event.result_preview,
                event.result_full,
                1 if event.is_error else 0,
                event.timestamp,
            ),
        )
        return int(cursor.lastrowid)

    def _mark_participants_active(
        self,
        conn: sqlite3.Connection,
        *,
        group_id: str,
        participants: list[MemoryParticipant],
        last_active_at: str,
    ) -> None:
        now = _now_iso()
        for participant in participants:
            if not participant.user_id:
                continue
            conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id,
                    group_id,
                    nickname,
                    profile,
                    version,
                    pending_sessions,
                    updated_at,
                    last_active_at
                )
                VALUES (?, ?, ?, '', 1, 1, ?, ?)
                ON CONFLICT(user_id, group_id) DO UPDATE SET
                    nickname = excluded.nickname,
                    pending_sessions = user_profiles.pending_sessions + 1,
                    last_active_at = excluded.last_active_at
                """,
                (
                    participant.user_id,
                    group_id,
                    participant.nickname,
                    now,
                    last_active_at,
                ),
            )

    def _touch_profile_in_conn(
        self,
        conn: sqlite3.Connection,
        *,
        group_id: str,
        user_id: str,
        nickname: str = "",
        last_active_at: str = "",
    ) -> None:
        now = _now_iso()
        conn.execute(
            """
            INSERT INTO user_profiles (
                user_id,
                group_id,
                nickname,
                profile,
                version,
                pending_sessions,
                updated_at,
                last_active_at
            )
            VALUES (?, ?, ?, '', 1, 0, ?, ?)
            ON CONFLICT(user_id, group_id) DO UPDATE SET
                nickname = CASE
                    WHEN excluded.nickname != '' THEN excluded.nickname
                    ELSE user_profiles.nickname
                END,
                last_active_at = CASE
                    WHEN excluded.last_active_at != '' THEN excluded.last_active_at
                    ELSE user_profiles.last_active_at
                END
            """,
            (
                user_id,
                group_id,
                str(nickname).strip(),
                now,
                str(last_active_at or ""),
            ),
        )

    def _delete_archive_in_conn(self, conn: sqlite3.Connection, *, archive_id: int) -> bool:
        row = conn.execute(
            "SELECT id FROM conversation_archives WHERE id = ?",
            (archive_id,),
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM archive_fts WHERE archive_id = ?", (archive_id,))
        conn.execute("DELETE FROM message_fts WHERE archive_id = ?", (archive_id,))
        conn.execute("DELETE FROM tool_event_fts WHERE archive_id = ?", (archive_id,))
        if self._vector_index is not None:
            self._vector_index.delete(conn, archive_id=archive_id)
        conn.execute("DELETE FROM archive_embeddings WHERE archive_id = ?", (archive_id,))
        conn.execute("DELETE FROM conversation_archives WHERE id = ?", (archive_id,))
        return True

    def _remove_user_from_archive_participants(
        self,
        conn: sqlite3.Connection,
        *,
        group_id: str,
        user_id: str,
        archive_ids: set[int],
    ) -> None:
        if not archive_ids:
            return
        for archive_id in archive_ids:
            row = conn.execute(
                """
                SELECT *
                FROM conversation_archives
                WHERE id = ? AND group_id = ?
                """,
                (archive_id, group_id),
            ).fetchone()
            if row is None:
                continue
            participants = [
                participant
                for participant in self._participants_from_json(row["participants"])
                if participant.user_id != user_id
            ]
            message_count = conn.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE archive_id = ?",
                (archive_id,),
            ).fetchone()["count"]
            participants_json = _json_dumps([participant.to_json() for participant in participants])
            conn.execute(
                """
                UPDATE conversation_archives
                SET participants = ?,
                    message_count = ?
                WHERE id = ?
                """,
                (participants_json, int(message_count), archive_id),
            )
            conn.execute("DELETE FROM archive_fts WHERE archive_id = ?", (archive_id,))
            conn.execute(
                """
                INSERT INTO archive_fts(archive_id, group_id, search_text)
                VALUES (?, ?, ?)
                """,
                (
                    archive_id,
                    group_id,
                    build_search_text(
                        row["title"],
                        row["summary"],
                        row["keywords"],
                        " ".join(participant.nickname for participant in participants),
                        " ".join(participant.user_id for participant in participants),
                    ),
                ),
            )

    @staticmethod
    def _normalize_participants(
        participants: list[MemoryParticipant | dict[str, Any] | str] | None,
        messages: list[MemoryMessage],
    ) -> list[MemoryParticipant]:
        by_user: dict[str, MemoryParticipant] = {}
        for raw in participants or []:
            participant = MemoryParticipant.from_raw(raw)
            if not participant.user_id:
                continue
            by_user[participant.user_id] = participant
        for message in messages:
            if message.role != "user" or not message.user_id:
                continue
            existing = by_user.get(message.user_id)
            if existing is None or (message.nickname and not existing.nickname):
                by_user[message.user_id] = MemoryParticipant(
                    user_id=message.user_id,
                    nickname=message.nickname,
                )
        return list(by_user.values())

    @staticmethod
    def _message_from_raw(value: MemoryMessage | dict[str, Any]) -> MemoryMessage:
        if isinstance(value, MemoryMessage):
            return value
        return MemoryMessage(
            role=value.get("role", "user"),
            content=str(value.get("content", "")),
            timestamp=str(value.get("timestamp", "")),
            user_id=str(value.get("user_id", "")).strip(),
            nickname=str(value.get("nickname", "")).strip(),
        )

    @staticmethod
    def _tool_event_from_raw(value: MemoryToolEvent | dict[str, Any]) -> MemoryToolEvent:
        if isinstance(value, MemoryToolEvent):
            return value
        return MemoryToolEvent(
            timestamp=str(value.get("timestamp", "")),
            tool_call_id=str(value.get("tool_call_id", "")).strip(),
            tool_name=str(value.get("tool_name", "")).strip(),
            arguments_text=str(value.get("arguments_text", "")),
            result_preview=str(value.get("result_preview", "")),
            result_full=str(value.get("result_full", "")),
            is_error=bool(value.get("is_error", False)),
        )

    def _archive_from_row(self, row: sqlite3.Row) -> MemoryArchive:
        return MemoryArchive(
            id=int(row["id"]),
            group_id=row["group_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            participants=self._participants_from_json(row["participants"]),
            title=row["title"],
            summary=row["summary"],
            keywords=self._keywords_from_json(row["keywords"]),
            message_count=int(row["message_count"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _profile_from_row(row: sqlite3.Row) -> MemoryProfile:
        return MemoryProfile(
            user_id=row["user_id"],
            group_id=row["group_id"],
            nickname=row["nickname"],
            profile=row["profile"],
            version=int(row["version"]),
            pending_sessions=int(row["pending_sessions"]),
            updated_at=row["updated_at"],
            last_active_at=row["last_active_at"],
        )

    @staticmethod
    def _profile_edit_from_row(row: sqlite3.Row) -> MemoryProfileEdit:
        return MemoryProfileEdit(
            id=int(row["id"]),
            user_id=row["user_id"],
            group_id=row["group_id"],
            edit_type=row["edit_type"],
            content=row["content"],
            created_at=row["created_at"],
            consolidated=bool(row["consolidated"]),
        )

    @staticmethod
    def _participants_from_json(value: str) -> list[MemoryParticipant]:
        data = _json_loads(value, default=[])
        if not isinstance(data, list):
            return []
        return [MemoryParticipant.from_raw(item) for item in data]

    @staticmethod
    def _keywords_from_json(value: str) -> list[str]:
        data = _json_loads(value, default=[])
        if not isinstance(data, list):
            return []
        return [str(item) for item in data if str(item).strip()]

    def _messages_for_archive(self, conn: sqlite3.Connection, archive_id: int) -> list[MemoryMessage]:
        rows = conn.execute(
            "SELECT * FROM messages WHERE archive_id = ? ORDER BY id",
            (archive_id,),
        ).fetchall()
        return [
            MemoryMessage(
                id=int(row["id"]),
                archive_id=int(row["archive_id"]),
                group_id=row["group_id"],
                role=row["role"],
                user_id=row["user_id"],
                nickname=row["nickname"],
                content=row["content"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def _tool_events_for_archive(self, conn: sqlite3.Connection, archive_id: int) -> list[MemoryToolEvent]:
        rows = conn.execute(
            "SELECT * FROM tool_events WHERE archive_id = ? ORDER BY id",
            (archive_id,),
        ).fetchall()
        return [
            MemoryToolEvent(
                id=int(row["id"]),
                archive_id=int(row["archive_id"]),
                group_id=row["group_id"],
                tool_call_id=row["tool_call_id"],
                tool_name=row["tool_name"],
                arguments_text=row["arguments_text"],
                result_preview=row["result_preview"],
                result_full=row["result_full"],
                is_error=bool(row["is_error"]),
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def _render_open_archive(
        self,
        *,
        archive: MemoryArchive,
        messages: list[MemoryMessage],
        tool_events: list[MemoryToolEvent],
        mode: OpenArchiveMode,
        around_message_id: int | None,
        before: int,
        after: int,
        include_tool_results: bool,
    ) -> str:
        lines = _archive_header_lines(archive)
        if mode == "tools":
            lines.extend(["", "工具事件:"])
            lines.extend(_tool_lines(tool_events, include_full=include_tool_results))
            return "\n".join(lines)

        selected_messages = messages
        if around_message_id is not None:
            selected_messages = _window_messages(
                messages,
                around_message_id=around_message_id,
                before=max(0, before),
                after=max(0, after),
            )

        if mode == "compact":
            selected_messages = _compact_messages(messages, around_message_id=around_message_id)
            lines.extend(["", "会话片段:"])
            lines.extend(_message_lines(selected_messages))
            if tool_events:
                lines.extend(["", "工具事件预览:"])
                lines.extend(_tool_lines(tool_events[:8], include_full=include_tool_results))
            return "\n".join(lines)

        if mode in {"transcript", "full"}:
            lines.extend(["", "消息:"])
            lines.extend(_message_lines(selected_messages))
        if mode == "full" and tool_events:
            lines.extend(["", "工具事件:"])
            lines.extend(_tool_lines(tool_events, include_full=include_tool_results))
        return "\n".join(lines)


def _archive_header_lines(archive: MemoryArchive) -> list[str]:
    participants = ", ".join(
        participant.nickname or participant.user_id
        for participant in archive.participants
        if participant.nickname or participant.user_id
    ) or "-"
    keywords = ", ".join(archive.keywords) or "-"
    return [
        f"archive_id={archive.id} | {archive.started_at} ~ {archive.ended_at}",
        f"参与者: {participants}",
        f"标题: {archive.title or '-'}",
        f"摘要: {archive.summary or '-'}",
        f"关键词: {keywords}",
        f"消息数: {archive.message_count}",
    ]


def _message_lines(messages: list[MemoryMessage]) -> list[str]:
    if not messages:
        return ["(无消息)"]
    lines: list[str] = []
    for message in messages:
        speaker = message.nickname if message.role == "user" and message.nickname else message.role
        lines.append(
            f"[message_id={message.id}] {message.timestamp} {speaker}: {message.content}"
        )
    return lines


def _tool_lines(tool_events: list[MemoryToolEvent], *, include_full: bool) -> list[str]:
    if not tool_events:
        return ["(无工具事件)"]
    lines: list[str] = []
    for event in tool_events:
        status = "error" if event.is_error else "ok"
        result = event.result_full if include_full and event.result_full else event.result_preview
        result = _truncate(result, 1200 if include_full else 360)
        lines.append(
            f"[tool_event_id={event.id}] {event.timestamp} {event.tool_name or '-'} "
            f"({status}) args={_truncate(event.arguments_text, 220)} result={result or '-'}"
        )
    return lines


def _compact_messages(
    messages: list[MemoryMessage],
    *,
    around_message_id: int | None,
) -> list[MemoryMessage]:
    if len(messages) <= 8:
        return messages
    selected: list[MemoryMessage] = []
    selected.extend(messages[:2])
    if around_message_id is not None:
        selected.extend(_window_messages(messages, around_message_id=around_message_id, before=3, after=3))
    selected.extend(messages[-3:])
    return _dedupe_messages(selected)


def _window_messages(
    messages: list[MemoryMessage],
    *,
    around_message_id: int,
    before: int,
    after: int,
) -> list[MemoryMessage]:
    index = next((idx for idx, message in enumerate(messages) if message.id == around_message_id), None)
    if index is None:
        return []
    start = max(0, index - before)
    end = min(len(messages), index + after + 1)
    return messages[start:end]


def _dedupe_messages(messages: list[MemoryMessage]) -> list[MemoryMessage]:
    seen: set[int] = set()
    result: list[MemoryMessage] = []
    for message in messages:
        key = message.id
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        result.append(message)
    return result


def _contains_any(text: str, terms: list[str]) -> bool:
    folded = text.casefold()
    return any(term.casefold() in folded for term in terms)


def _make_snippet(text: str, terms: list[str], *, limit: int = 180) -> str:
    normalized = normalize_for_search(text)
    if len(normalized) <= limit:
        return normalized

    folded = normalized.casefold()
    first_index: int | None = None
    for term in terms:
        index = folded.find(term.casefold())
        if index < 0:
            continue
        first_index = index if first_index is None else min(first_index, index)
    if first_index is None:
        return _truncate(normalized, limit)

    start = max(0, first_index - limit // 3)
    end = min(len(normalized), start + limit)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(normalized) else ""
    return f"{prefix}{normalized[start:end]}{suffix}"


def _truncate(text: str, limit: int) -> str:
    normalized = normalize_for_search(text)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 3)]}..."


def _escape_like(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _escape_json_like(value: str) -> str:
    return _escape_like(value.replace('"', '\\"'))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str, *, default: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_required_time_bound(value: str, *, name: str) -> datetime | None:
    if not value:
        return None
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        raise ValueError(f"{name} must be an ISO datetime")
    return parsed
