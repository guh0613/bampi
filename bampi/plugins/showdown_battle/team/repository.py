from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class TeamRepositoryError(RuntimeError):
    pass


class TeamRepositoryConflict(TeamRepositoryError):
    pass


@dataclass(frozen=True, slots=True)
class TeamRecord:
    user_id: str
    format_id: str
    name: str
    packed: str
    raw: str
    updated_at: float


TeamData = dict[str, dict[str, dict[str, TeamRecord]]]


class TeamRepository:
    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path
        self._lock = asyncio.Lock()
        self._data: TeamData | None = None

    async def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        async with self._lock:
            if self._data is None:
                self._data = await asyncio.to_thread(self._read_from_disk)

    def _read_from_disk(self) -> TeamData:
        if not self._storage_path.exists():
            return {}
        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise TeamRepositoryError(f"读取队伍仓库失败：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise TeamRepositoryError(
                f"队伍仓库 JSON 已损坏：{self._storage_path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise TeamRepositoryError("队伍仓库根节点必须是 JSON 对象。")
        if "schema_version" in payload:
            if payload.get("schema_version") != 1:
                raise TeamRepositoryError(
                    f"不支持的队伍仓库版本：{payload.get('schema_version')}"
                )
            payload = payload.get("users", {})
        if not isinstance(payload, dict):
            raise TeamRepositoryError("队伍仓库 users 节点无效。")

        result: TeamData = {}
        for user_id, formats in payload.items():
            if not isinstance(formats, dict):
                continue
            user_bucket: dict[str, dict[str, TeamRecord]] = {}
            for format_id, teams in formats.items():
                if not isinstance(teams, dict):
                    continue
                format_bucket: dict[str, TeamRecord] = {}
                for name, record in teams.items():
                    parsed = self._parse_record(
                        user_id=str(user_id),
                        format_id=str(format_id),
                        name=str(name),
                        payload=record,
                    )
                    if parsed:
                        format_bucket[parsed.name] = parsed
                if format_bucket:
                    user_bucket[str(format_id)] = format_bucket
            if user_bucket:
                result[str(user_id)] = user_bucket
        return result

    @staticmethod
    def _parse_record(
        *, user_id: str, format_id: str, name: str, payload: Any
    ) -> TeamRecord | None:
        if not isinstance(payload, dict):
            return None
        packed = payload.get("packed")
        raw = payload.get("raw")
        if not isinstance(packed, str) or not isinstance(raw, str):
            return None
        try:
            updated_at = float(payload.get("updated_at", time.time()))
        except (TypeError, ValueError):
            updated_at = time.time()
        return TeamRecord(
            user_id=user_id,
            format_id=format_id,
            name=name,
            packed=packed,
            raw=raw,
            updated_at=updated_at,
        )

    async def _persist_data_locked(self, data: TeamData) -> None:
        payload = {
            "schema_version": 1,
            "users": self._serialize(data),
        }
        content = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        try:
            await asyncio.to_thread(self._atomic_write, content)
        except OSError as exc:
            raise TeamRepositoryError(f"写入队伍仓库失败：{exc}") from exc

    @staticmethod
    def _serialize(data: TeamData) -> dict[str, Any]:
        return {
            user_id: {
                format_id: {
                    name: {
                        "packed": record.packed,
                        "raw": record.raw,
                        "updated_at": record.updated_at,
                    }
                    for name, record in teams.items()
                }
                for format_id, teams in formats.items()
                if teams
            }
            for user_id, formats in data.items()
            if formats
        }

    def _atomic_write(self, content: str) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self._storage_path.name}.",
            dir=self._storage_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self._storage_path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def validate_team_name(team_name: str) -> str:
        normalized = team_name.strip()
        if not normalized:
            raise TeamRepositoryError("队伍名称不能为空。")
        if len(normalized) > 40:
            raise TeamRepositoryError("队伍名称不能超过 40 个字符。")
        if any(char in normalized for char in "\r\n\t"):
            raise TeamRepositoryError("队伍名称不能包含换行或制表符。")
        return normalized

    async def list_teams(
        self, user_id: str, format_id: str | None = None
    ) -> list[TeamRecord]:
        await self._ensure_loaded()
        assert self._data is not None
        async with self._lock:
            formats = self._data.get(user_id, {})
            records = (
                list(formats.get(format_id, {}).values())
                if format_id
                else [record for teams in formats.values() for record in teams.values()]
            )
            records.sort(key=lambda record: record.updated_at, reverse=True)
            return [replace(record) for record in records]

    async def get_team(
        self, user_id: str, format_id: str, team_name: str
    ) -> TeamRecord | None:
        await self._ensure_loaded()
        assert self._data is not None
        async with self._lock:
            record = self._data.get(user_id, {}).get(format_id, {}).get(team_name)
            return replace(record) if record else None

    async def set_team(
        self,
        user_id: str,
        format_id: str,
        team_name: str,
        *,
        packed: str,
        raw: str,
    ) -> TeamRecord:
        name = self.validate_team_name(team_name)
        await self._ensure_loaded()
        assert self._data is not None
        async with self._lock:
            record = TeamRecord(
                user_id=user_id,
                format_id=format_id,
                name=name,
                packed=packed,
                raw=raw,
                updated_at=time.time(),
            )
            candidate = copy.deepcopy(self._data)
            candidate.setdefault(user_id, {}).setdefault(format_id, {})[name] = record
            await self._persist_data_locked(candidate)
            self._data = candidate
            return replace(record)

    async def create_team(
        self,
        user_id: str,
        format_id: str,
        team_name: str,
        *,
        packed: str,
        raw: str,
    ) -> TeamRecord:
        name = self.validate_team_name(team_name)
        await self._ensure_loaded()
        assert self._data is not None
        async with self._lock:
            teams = self._data.get(user_id, {}).get(format_id, {})
            if name in teams:
                raise TeamRepositoryConflict(f"队伍「{name}」已经存在。")
            record = TeamRecord(
                user_id=user_id,
                format_id=format_id,
                name=name,
                packed=packed,
                raw=raw,
                updated_at=time.time(),
            )
            candidate = copy.deepcopy(self._data)
            candidate.setdefault(user_id, {}).setdefault(format_id, {})[name] = record
            await self._persist_data_locked(candidate)
            self._data = candidate
            return replace(record)

    async def update_team(
        self,
        user_id: str,
        format_id: str,
        team_name: str,
        *,
        packed: str,
        raw: str,
        expected_updated_at: float,
    ) -> TeamRecord:
        name = self.validate_team_name(team_name)
        await self._ensure_loaded()
        assert self._data is not None
        async with self._lock:
            current = self._data.get(user_id, {}).get(format_id, {}).get(name)
            if current is None:
                raise TeamRepositoryConflict("原队伍已不存在，无法保存编辑结果。")
            if current.updated_at != expected_updated_at:
                raise TeamRepositoryConflict(
                    "队伍在编辑期间已被其他操作更新。请退出后重新打开，"
                    "以免覆盖较新的内容。"
                )
            record = TeamRecord(
                user_id=user_id,
                format_id=format_id,
                name=name,
                packed=packed,
                raw=raw,
                updated_at=time.time(),
            )
            candidate = copy.deepcopy(self._data)
            candidate[user_id][format_id][name] = record
            await self._persist_data_locked(candidate)
            self._data = candidate
            return replace(record)

    async def delete_team(self, user_id: str, format_id: str, team_name: str) -> bool:
        await self._ensure_loaded()
        assert self._data is not None
        async with self._lock:
            formats = self._data.get(user_id)
            teams = formats.get(format_id) if formats else None
            if not teams or team_name not in teams:
                return False
            candidate = copy.deepcopy(self._data)
            candidate_formats = candidate[user_id]
            candidate_teams = candidate_formats[format_id]
            del candidate_teams[team_name]
            if not candidate_teams:
                candidate_formats.pop(format_id, None)
            if not candidate_formats:
                candidate.pop(user_id, None)
            await self._persist_data_locked(candidate)
            self._data = candidate
            return True

    async def rename_team(
        self,
        user_id: str,
        format_id: str,
        old_name: str,
        new_name: str,
    ) -> bool:
        normalized_new_name = self.validate_team_name(new_name)
        if old_name == normalized_new_name:
            return True
        await self._ensure_loaded()
        assert self._data is not None
        async with self._lock:
            teams = self._data.get(user_id, {}).get(format_id)
            if not teams or old_name not in teams or normalized_new_name in teams:
                return False
            candidate = copy.deepcopy(self._data)
            candidate_teams = candidate[user_id][format_id]
            record = candidate_teams.pop(old_name)
            candidate_teams[normalized_new_name] = TeamRecord(
                user_id=record.user_id,
                format_id=record.format_id,
                name=normalized_new_name,
                packed=record.packed,
                raw=record.raw,
                updated_at=time.time(),
            )
            await self._persist_data_locked(candidate)
            self._data = candidate
            return True
