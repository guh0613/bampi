from __future__ import annotations

from nonebot import logger
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .archiver import build_archive_from_agent_messages, summarize_archive_with_llm
from .embeddings import MemoryEmbeddingProvider, build_embedding_provider
from .profiler import build_profile_from_archives, generate_profile_with_llm, render_memory_context
from .store import MemoryStore
from .types import (
    MemoryMessage,
    MemoryOpenedArchive,
    MemoryParticipant,
    MemoryProfileEdit,
    MemorySearchHit,
    MemorySnippet,
    MemoryToolEvent,
    MemoryUserTurn,
    OpenArchiveMode,
)

if TYPE_CHECKING:
    from ..config import BampiChatConfig




class MemoryManager:
    def __init__(
        self,
        db_path: str | Path,
        *,
        search_max_results: int = 10,
        search_snippet_messages: int = 2,
        like_fallback: bool = True,
        archive_min_messages: int = 3,
        archive_summary_max_tokens: int = 500,
        tool_result_preview_chars: int = 1000,
        tool_result_full_max_chars: int = 20_000,
        archive_retention_days: int = 365,
        embedding_provider: MemoryEmbeddingProvider | None = None,
        storage_mode: str = "single",
        profile_session_threshold: int = 5,
        profile_max_staleness_days: int = 7,
        profile_cron: str = "0 4 * * *",
        profile_max_tokens: int = 1500,
        pending_edits_max_inject: int = 10,
        scheduler_timezone: str = "Asia/Shanghai",
        llm_summary_enabled: bool = True,
    ) -> None:
        self._db_path = Path(db_path)
        self._search_max_results = min(max(1, search_max_results), 10)
        self._search_snippet_messages = max(0, search_snippet_messages)
        self._like_fallback = like_fallback
        self._archive_min_messages = max(1, archive_min_messages)
        self._archive_summary_max_chars = max(200, archive_summary_max_tokens * 4)
        self._tool_result_preview_chars = max(0, tool_result_preview_chars)
        self._tool_result_full_max_chars = max(0, tool_result_full_max_chars)
        self._archive_retention_days = archive_retention_days
        self._embedding_provider = embedding_provider
        self._storage_mode = storage_mode.strip() or "single"
        if self._storage_mode != "single":
            raise ValueError("only single memory storage mode is supported right now")
        self._profile_session_threshold = max(1, profile_session_threshold)
        self._profile_max_staleness_days = max(0, profile_max_staleness_days)
        self._profile_cron = profile_cron.strip() or "0 4 * * *"
        self._profile_max_tokens = max(200, profile_max_tokens)
        self._pending_edits_max_inject = max(0, pending_edits_max_inject)
        self._scheduler_timezone = scheduler_timezone
        self._llm_summary_enabled = llm_summary_enabled
        self._store: MemoryStore | None = None
        self._scheduler: AsyncIOScheduler | None = None

    @classmethod
    def from_config(cls, config: "BampiChatConfig") -> "MemoryManager":
        embedding_provider: MemoryEmbeddingProvider | None = None
        if config.bampi_memory_embedding_enabled:
            provider_name = config.bampi_memory_embedding_provider or (
                "openai-compatible" if config.bampi_memory_embedding_model else "local-hash"
            )
            embedding_provider = build_embedding_provider(
                provider=provider_name,
                model=config.bampi_memory_embedding_model,
                api_key=config.bampi_api_key,
                base_url=config.bampi_base_url,
            )
        return cls(
            config.bampi_memory_db_path,
            search_max_results=config.bampi_memory_search_max_results,
            search_snippet_messages=config.bampi_memory_search_snippet_messages,
            like_fallback=config.bampi_memory_search_like_fallback,
            archive_min_messages=config.bampi_memory_archive_min_messages,
            archive_summary_max_tokens=config.bampi_memory_archive_summary_max_tokens,
            tool_result_preview_chars=config.bampi_memory_tool_result_preview_chars,
            tool_result_full_max_chars=config.bampi_memory_tool_result_full_max_chars,
            archive_retention_days=config.bampi_memory_archive_retention_days,
            embedding_provider=embedding_provider,
            storage_mode=config.bampi_memory_storage_mode,
            profile_session_threshold=config.bampi_memory_profile_session_threshold,
            profile_max_staleness_days=config.bampi_memory_profile_max_staleness_days,
            profile_cron=config.bampi_memory_profile_cron,
            profile_max_tokens=config.bampi_memory_profile_max_tokens,
            pending_edits_max_inject=config.bampi_memory_pending_edits_max_inject,
            scheduler_timezone=config.bampi_schedule_timezone,
            llm_summary_enabled=config.bampi_memory_archive_llm_summary,
        )

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            self._store = MemoryStore(
                self._db_path,
                search_snippet_messages=self._search_snippet_messages,
                like_fallback=self._like_fallback,
                embedding_provider=self._embedding_provider,
            )
        return self._store

    def start_background_tasks(
        self,
        *,
        model: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        if self._scheduler is not None:
            return
        try:
            timezone = ZoneInfo(self._scheduler_timezone)
            trigger = CronTrigger.from_crontab(self._profile_cron, timezone=timezone)
        except Exception:
            logger.opt(exception=True).error(
                f"bampi_chat memory background tasks not started due to invalid schedule "
                f"cron={self._profile_cron} timezone={self._scheduler_timezone}"
            )
            return
        scheduler = AsyncIOScheduler(timezone=timezone)
        async def _maintenance_job() -> None:
            await self.run_memory_maintenance_async(model=model, api_key=api_key)

        scheduler.add_job(
            _maintenance_job,
            trigger,
            id="bampi-memory-maintenance",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
        self._scheduler = scheduler
        logger.info(f"bampi_chat memory background tasks started cron={self._profile_cron}")

    def close_background_tasks(self) -> None:
        scheduler = self._scheduler
        self._scheduler = None
        if scheduler is None:
            return
        scheduler.shutdown(wait=False)
        logger.info("bampi_chat memory background tasks stopped")

    def archive_conversation(
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
        return self.store.archives.add(
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

    def archive_session(
        self,
        *,
        group_id: str,
        messages: list[Any],
        user_turns: list[MemoryUserTurn],
    ) -> int | None:
        built = build_archive_from_agent_messages(
            group_id=group_id,
            messages=messages,
            user_turns=user_turns,
            min_messages=self._archive_min_messages,
            tool_result_preview_chars=self._tool_result_preview_chars,
            tool_result_full_max_chars=self._tool_result_full_max_chars,
        )
        if built is None:
            return None
        return self.archive_conversation(
            group_id=group_id,
            started_at=built.started_at,
            ended_at=built.ended_at,
            participants=built.participants,
            title=built.title,
            summary=_truncate_text(built.summary, self._archive_summary_max_chars),
            keywords=built.keywords,
            messages=built.messages,
            tool_events=built.tool_events,
        )

    async def archive_session_async(
        self,
        *,
        group_id: str,
        messages: list[Any],
        user_turns: list[MemoryUserTurn],
        model: Any = None,
        api_key: str | None = None,
    ) -> int | None:
        built = build_archive_from_agent_messages(
            group_id=group_id,
            messages=messages,
            user_turns=user_turns,
            min_messages=self._archive_min_messages,
            tool_result_preview_chars=self._tool_result_preview_chars,
            tool_result_full_max_chars=self._tool_result_full_max_chars,
        )
        if built is None:
            return None

        title = built.title
        summary = built.summary
        keywords = built.keywords

        if self._llm_summary_enabled and model is not None:
            llm_result = await summarize_archive_with_llm(
                built.messages,
                built.tool_events,
                model=model,
                api_key=api_key,
            )
            if llm_result is not None:
                title, summary, keywords = llm_result
                if not keywords:
                    keywords = built.keywords

        return self.archive_conversation(
            group_id=group_id,
            started_at=built.started_at,
            ended_at=built.ended_at,
            participants=built.participants,
            title=title,
            summary=_truncate_text(summary, self._archive_summary_max_chars),
            keywords=keywords,
            messages=built.messages,
            tool_events=built.tool_events,
        )

    def search(
        self,
        *,
        group_id: str,
        query: str,
        user_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        max_results: int | None = None,
    ) -> list[MemorySearchHit]:
        limit = self._search_max_results if max_results is None else min(max(1, max_results), 10)
        return self.store.archives.search(
            group_id=group_id,
            query=query,
            user_id=user_id,
            after=after,
            before=before,
            max_results=limit,
        )

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
        return self.store.archives.open(
            archive_id=archive_id,
            group_id=group_id,
            mode=mode,
            around_message_id=around_message_id,
            before=before,
            after=after,
            include_tool_results=include_tool_results,
        )

    def add_profile_edit(
        self,
        *,
        group_id: str,
        user_id: str,
        edit_type: str,
        content: str,
        nickname: str = "",
    ) -> MemoryProfileEdit:
        return self.store.profiles.add_edit(
            group_id=group_id,
            user_id=user_id,
            edit_type=edit_type,
            content=content,
            nickname=nickname,
        )

    def get_memory_context_for_turn(
        self,
        *,
        group_id: str,
        current_user_id: str,
        current_nickname: str = "",
        session_participants: list[MemoryParticipant] | None = None,
    ) -> str:
        normalized_user_id = str(current_user_id).strip()
        if not normalized_user_id:
            return ""
        normalized_group_id = str(group_id).strip()
        self.store.profiles.touch(
            group_id=normalized_group_id,
            user_id=normalized_user_id,
            nickname=current_nickname,
        )

        by_user: dict[str, str] = {}
        by_user[normalized_user_id] = current_nickname
        for participant in session_participants or []:
            if not participant.user_id:
                continue
            by_user.setdefault(participant.user_id, participant.nickname)

        profiles: list[tuple[Any, list[MemoryProfileEdit], str]] = []
        for user_id, display_name in by_user.items():
            profile = self.store.profiles.get(group_id=normalized_group_id, user_id=user_id)
            edits = (
                self.store.profiles.pending_edits(
                    group_id=normalized_group_id,
                    user_id=user_id,
                    limit=self._pending_edits_max_inject,
                )
                if self._pending_edits_max_inject > 0
                else []
            )
            if profile is None and not edits:
                continue
            profiles.append((profile, edits, display_name))

        return render_memory_context(
            current_user_id=normalized_user_id,
            current_nickname=current_nickname,
            profiles=profiles,
        )

    def run_memory_maintenance(self) -> dict[str, int]:
        generated = self.run_profile_generation_scan()
        deleted = self.cleanup_old_data()
        return {"profiles_generated": generated, "archives_deleted": deleted}

    async def run_memory_maintenance_async(
        self,
        *,
        model: Any | None = None,
        api_key: str | None = None,
    ) -> dict[str, int]:
        generated = await self.run_profile_generation_scan_async(model=model, api_key=api_key)
        deleted = self.cleanup_old_data()
        return {"profiles_generated": generated, "archives_deleted": deleted}

    def run_profile_generation_scan(self) -> int:
        due_profiles = self.store.profiles.due_for_generation(
            session_threshold=self._profile_session_threshold,
            max_staleness_days=self._profile_max_staleness_days,
        )
        generated = 0
        for profile in due_profiles:
            edits = self.store.profiles.pending_edits(
                group_id=profile.group_id,
                user_id=profile.user_id,
                limit=None,
            )
            archives = self.store.profiles.archives_for_generation(
                group_id=profile.group_id,
                user_id=profile.user_id,
                since=profile.updated_at if profile.profile.strip() else None,
            )
            if not archives and not edits:
                continue
            new_profile = build_profile_from_archives(
                profile=profile,
                archives=archives,
                pending_edits=edits,
            )
            self.store.profiles.consolidate(
                group_id=profile.group_id,
                user_id=profile.user_id,
                profile=new_profile,
                edit_ids=[edit.id for edit in edits if edit.id is not None],
            )
            generated += 1
        if generated:
            logger.info(f"bampi_chat memory generated profiles count={generated}")
        return generated

    async def run_profile_generation_scan_async(
        self,
        *,
        model: Any | None = None,
        api_key: str | None = None,
    ) -> int:
        if model is None:
            return self.run_profile_generation_scan()

        due_profiles = self.store.profiles.due_for_generation(
            session_threshold=self._profile_session_threshold,
            max_staleness_days=self._profile_max_staleness_days,
        )
        generated = 0
        for profile in due_profiles:
            edits = self.store.profiles.pending_edits(
                group_id=profile.group_id,
                user_id=profile.user_id,
                limit=None,
            )
            archives = self.store.profiles.archives_for_generation(
                group_id=profile.group_id,
                user_id=profile.user_id,
                since=profile.updated_at if profile.profile.strip() else None,
            )
            if not archives and not edits:
                continue
            new_profile = await generate_profile_with_llm(
                profile=profile,
                archives=archives,
                pending_edits=edits,
                model=model,
                api_key=api_key,
            )
            if new_profile is None:
                logger.warning(
                    f"bampi_chat memory llm profile generation returned empty output "
                    f"group_id={profile.group_id} user_id={profile.user_id}; "
                    "falling back to rule-based profile generation"
                )
                new_profile = build_profile_from_archives(
                    profile=profile,
                    archives=archives,
                    pending_edits=edits,
                )
            self.store.profiles.consolidate(
                group_id=profile.group_id,
                user_id=profile.user_id,
                profile=new_profile,
                edit_ids=[edit.id for edit in edits if edit.id is not None],
            )
            generated += 1
        if generated:
            logger.info(f"bampi_chat memory generated profiles count={generated}")
        return generated

    def cleanup_old_data(self) -> int:
        return self.store.maintenance.cleanup_old_data(
            archive_retention_days=self._archive_retention_days
        )

    def delete_archive(self, *, archive_id: int, group_id: str | None = None) -> bool:
        return self.store.archives.delete(archive_id=archive_id, group_id=group_id)

    def delete_user_memory(
        self,
        *,
        group_id: str,
        user_id: str,
        delete_messages: bool = True,
    ) -> dict[str, int]:
        return self.store.profiles.delete_user_memory(
            group_id=group_id,
            user_id=user_id,
            delete_messages=delete_messages,
        )


def render_search_results(hits: list[MemorySearchHit]) -> str:
    if not hits:
        return "没有找到相关的历史会话。"

    lines = [f"找到 {len(hits)} 次可能相关的历史会话："]
    for index, hit in enumerate(hits, start=1):
        archive = hit.archive
        participants = ", ".join(
            participant.nickname or participant.user_id
            for participant in archive.participants
            if participant.nickname or participant.user_id
        ) or "-"
        keywords = ", ".join(archive.keywords) or "-"
        sources = ", ".join(hit.matched_sources) or "-"
        lines.extend(
            [
                "",
                f"[{index}] archive_id={archive.id} | {archive.started_at} ~ {archive.ended_at} | 参与者: {participants}",
                f"标题: {archive.title or '-'}",
                f"摘要: {archive.summary or '-'}",
                f"关键词: {keywords}",
                f"命中: {sources}",
            ]
        )
        if hit.snippets:
            lines.append("片段:")
            for snippet in hit.snippets:
                lines.append(f"  {_render_snippet_label(snippet)}: {snippet.text}")
    return "\n".join(lines)


def search_hit_to_dict(hit: MemorySearchHit) -> dict[str, Any]:
    return asdict(hit)


def opened_archive_to_dict(opened: MemoryOpenedArchive) -> dict[str, Any]:
    return asdict(opened)


def _render_snippet_label(snippet: MemorySnippet) -> str:
    if snippet.source == "messages":
        return snippet.nickname or snippet.role or f"message {snippet.message_id}"
    if snippet.source == "tool_events":
        return f"tool {snippet.tool_name or snippet.tool_event_id}"
    return "archive"


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
