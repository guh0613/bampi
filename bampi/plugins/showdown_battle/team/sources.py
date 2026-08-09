from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx

from ..bridge import ShowdownBridgeError, ShowdownRuntime
from ..translations import TranslationService, to_id


_POKEPASTE_HOSTS = {"pokepast.es", "www.pokepast.es"}
_CROBAT_HOSTS = {"crob.at", "www.crob.at"}
_SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9]+$")
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,80}$")
_SPECIES_SEPARATOR = re.compile(r"[,，\n]+")


class TeamSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ImportedTeam:
    team_text: str
    label: str | None = None
    source_format_id: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SampleTeam:
    name: str
    author: str
    data: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PreparedSample:
    sample: SampleTeam
    team_text: str


@dataclass(frozen=True, slots=True)
class RecommendedSetOption:
    name: str
    item: str
    ability: str
    nature: str
    tera_type: str
    moves: tuple[str, ...]
    evs: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class RecommendedSetList:
    species: str
    options: tuple[RecommendedSetOption, ...]

    @property
    def set_names(self) -> tuple[str, ...]:
        return tuple(option.name for option in self.options)


@dataclass(frozen=True, slots=True)
class BuiltTeam:
    team_text: str
    selections: tuple[tuple[str, str], ...]


class TeamSourceService:
    """Fetch allowlisted team pastes and stable competitive team datasets."""

    PKMN_BASE_URL = "https://data.pkmn.cc"

    def __init__(
        self,
        *,
        runtime: ShowdownRuntime,
        translator: TranslationService,
        timeout_seconds: float,
        max_bytes: int,
        cache_ttl_seconds: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._runtime = runtime
        self._translator = translator
        self._max_bytes = max_bytes
        self._cache_ttl_seconds = cache_ttl_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            headers={"User-Agent": "bampi-showdown-team-import/1.0"},
        )
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._cache_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def resolve_import(self, raw_input: str) -> ImportedTeam:
        text = raw_input.strip()
        if not text:
            raise TeamSourceError("队伍内容为空。")
        candidate = text.removeprefix("<").removesuffix(">").strip()
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ImportedTeam(team_text=text)
        if any(char.isspace() for char in candidate):
            raise TeamSourceError("队伍链接后不能混入其他文本。")

        host = (parsed.hostname or "").lower()
        if host in _POKEPASTE_HOSTS:
            return await self._import_pokepaste(parsed)
        if host in _CROBAT_HOSTS:
            return await self._import_crobat(parsed)
        raise TeamSourceError(
            "暂只支持 pokepast.es 和 crob.at 队伍链接；"
            "其他网站请复制其中的 Showdown Export 文本。"
        )

    @staticmethod
    def ensure_format_compatible(imported: ImportedTeam, format_id: str) -> None:
        if (
            imported.source_format_id
            and imported.source_format_id.lower() != format_id.lower()
        ):
            raise TeamSourceError(
                f"该链接标注的规则是 {imported.source_format_id}，"
                f"但当前选择的是 {format_id}。请返回并选择对应规则。"
            )

    async def generate_team(
        self,
        *,
        format_id: str,
        source_id: str,
    ) -> ImportedTeam:
        source = self._validate_source_id(source_id)
        last_error: ShowdownBridgeError | None = None
        for _ in range(3):
            payload = await self._fetch_json(
                f"https://crob.at/api/random-team/{source}"
            )
            if not isinstance(payload, dict) or not isinstance(
                payload.get("teamText"), str
            ):
                raise TeamSourceError("随机组队服务返回了无效数据。")
            team_text = self._validate_team_text(payload["teamText"])
            try:
                prepared = await self._runtime.prepare_team_for_use(
                    format_id, team_text
                )
            except ShowdownBridgeError as exc:
                last_error = exc
                continue
            stats_date = str(payload.get("statsDate") or "当前数据").strip()
            return ImportedTeam(
                team_text=prepared.team_text,
                label=f"crob.at 随机推荐（{stats_date}）",
                source_format_id=format_id,
                warnings=prepared.warnings,
            )
        raise TeamSourceError(
            f"连续生成的队伍均不兼容当前本地规则：{last_error or '未知校验错误'}"
        )

    async def list_samples(self, source_id: str) -> list[SampleTeam]:
        source = self._validate_source_id(source_id)
        payload = await self._get_cached_json(
            "teams",
            source,
            f"{self.PKMN_BASE_URL}/teams/{source}.json",
        )
        if not isinstance(payload, list):
            raise TeamSourceError("在线样例队伍数据格式无效。")
        result: list[SampleTeam] = []
        for index, record in enumerate(payload, start=1):
            if not isinstance(record, dict):
                continue
            data = record.get("data")
            if not isinstance(data, list) or not data:
                continue
            sets = tuple(copy.deepcopy(item) for item in data if isinstance(item, dict))
            if not sets:
                continue
            result.append(
                SampleTeam(
                    name=str(record.get("name") or f"样例队伍 {index}").strip(),
                    author=str(record.get("author") or "未知作者").strip(),
                    data=sets,
                )
            )
        if not result:
            raise TeamSourceError("该规则目前没有可用的在线样例队伍。")
        return result

    async def get_sample(self, source_id: str, index: int) -> SampleTeam:
        samples = await self.list_samples(source_id)
        if index < 1 or index > len(samples):
            raise TeamSourceError(f"样例编号应在 1 至 {len(samples)} 之间。")
        return samples[index - 1]

    async def export_sample(
        self,
        *,
        format_id: str,
        source_id: str,
        index: int,
    ) -> tuple[SampleTeam, str]:
        sample = await self.get_sample(source_id, index)
        text = await self._runtime.export_team_json(
            format_id,
            [copy.deepcopy(item) for item in sample.data],
        )
        return sample, text

    async def list_compatible_samples(
        self,
        *,
        format_id: str,
        source_id: str,
    ) -> list[PreparedSample]:
        key = ("compatible", f"{format_id}:{source_id}")
        now = time.monotonic()
        async with self._cache_lock:
            cached = self._cache.get(key)
            if cached and cached[0] > now:
                return copy.deepcopy(cached[1])

        samples = await self.list_samples(source_id)
        compatible: list[PreparedSample] = []
        for sample in samples:
            try:
                team_text = await self._runtime.export_team_json(
                    format_id,
                    [copy.deepcopy(item) for item in sample.data],
                )
                await self._runtime.validate_team(format_id, team_text)
            except ShowdownBridgeError:
                continue
            compatible.append(PreparedSample(sample=sample, team_text=team_text))
        if not compatible:
            raise TeamSourceError(
                "在线样例与当前本地 Showdown 规则均不兼容，请改用队伍链接。"
            )
        async with self._cache_lock:
            self._cache[key] = (
                now + self._cache_ttl_seconds,
                copy.deepcopy(compatible),
            )
        return compatible

    async def get_compatible_sample(
        self,
        *,
        format_id: str,
        source_id: str,
        index: int,
    ) -> PreparedSample:
        samples = await self.list_compatible_samples(
            format_id=format_id,
            source_id=source_id,
        )
        if index < 1 or index > len(samples):
            raise TeamSourceError(f"样例编号应在 1 至 {len(samples)} 之间。")
        return samples[index - 1]

    async def list_recommended_sets(
        self,
        *,
        format_id: str,
        source_id: str,
        species_query: str,
    ) -> RecommendedSetList:
        species, records = await self._find_compatible_species_sets(
            format_id, source_id, species_query
        )
        options: list[RecommendedSetOption] = []
        for name, record in records.items():
            materialized = self._materialize_set(species, record)
            raw_evs = materialized.get("evs")
            evs = (
                tuple(
                    (str(stat), int(value))
                    for stat, value in raw_evs.items()
                    if isinstance(value, (int, float))
                )
                if isinstance(raw_evs, dict)
                else ()
            )
            options.append(
                RecommendedSetOption(
                    name=name,
                    item=str(materialized.get("item") or ""),
                    ability=str(materialized.get("ability") or ""),
                    nature=str(materialized.get("nature") or ""),
                    tera_type=str(materialized.get("teraType") or ""),
                    moves=tuple(str(move) for move in materialized.get("moves", [])),
                    evs=evs,
                )
            )
        return RecommendedSetList(species=species, options=tuple(options))

    async def build_recommended_team(
        self,
        *,
        format_id: str,
        source_id: str,
        species_input: str,
    ) -> BuiltTeam:
        tokens = [
            token.strip()
            for token in _SPECIES_SEPARATOR.split(species_input.strip())
            if token.strip()
        ]
        if not tokens:
            raise TeamSourceError("请至少提供一只宝可梦，多个名称使用逗号分隔。")
        if len(tokens) > 6:
            raise TeamSourceError("快速组队最多接受 6 只宝可梦。")

        team: list[dict[str, Any]] = []
        selections: list[tuple[str, str]] = []
        seen_species: set[str] = set()
        for token in tokens:
            species_query, selector = self._split_set_selector(token)
            species, records = await self._find_compatible_species_sets(
                format_id, source_id, species_query
            )
            species_id = to_id(species)
            if species_id in seen_species:
                raise TeamSourceError(f"不能重复添加 {species}。")
            seen_species.add(species_id)
            set_name, record = self._select_set(records, selector)
            team.append(self._materialize_set(species, record))
            selections.append((species, set_name))

        team_text = await self._runtime.export_team_json(format_id, team)
        return BuiltTeam(team_text=team_text, selections=tuple(selections))

    async def _find_compatible_species_sets(
        self,
        format_id: str,
        source_id: str,
        species_query: str,
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        species, records = await self._find_species_sets(source_id, species_query)
        cache_key = (
            "compatible-sets",
            f"{format_id}:{source_id}:{to_id(species)}",
        )
        now = time.monotonic()
        async with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and cached[0] > now:
                return species, copy.deepcopy(cached[1])

        compatible: dict[str, dict[str, Any]] = {}
        for set_name, record in records.items():
            try:
                materialized = self._materialize_set(species, record)
                team_text = await self._runtime.export_team_json(
                    format_id, [materialized]
                )
                await self._runtime.validate_set(format_id, team_text)
            except (ShowdownBridgeError, TeamSourceError):
                continue
            compatible[set_name] = record
        if not compatible:
            raise TeamSourceError(f"{species_query} 的在线配招均不兼容当前本地规则。")
        async with self._cache_lock:
            self._cache[cache_key] = (
                now + self._cache_ttl_seconds,
                copy.deepcopy(compatible),
            )
        return species, compatible

    async def _find_species_sets(
        self,
        source_id: str,
        species_query: str,
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        source = self._validate_source_id(source_id)
        query = species_query.strip()
        if not query:
            raise TeamSourceError("宝可梦名称不能为空。")
        resolved = self._translator.resolve_species_name(query) or query
        target_id = to_id(resolved)
        payload = await self._get_cached_json(
            "sets",
            source,
            f"{self.PKMN_BASE_URL}/sets/{source}.json",
        )
        if not isinstance(payload, dict):
            raise TeamSourceError("在线推荐配招数据格式无效。")
        for species, records in payload.items():
            if to_id(str(species)) != target_id or not isinstance(records, dict):
                continue
            cleaned = {
                str(name): copy.deepcopy(record)
                for name, record in records.items()
                if isinstance(record, dict)
            }
            if cleaned:
                return str(species), cleaned
        raise TeamSourceError(
            f"{query} 在该规则下暂无推荐配招；可改用其他宝可梦或直接导入链接。"
        )

    async def _import_pokepaste(self, parsed: Any) -> ImportedTeam:
        segments = [unquote(part) for part in parsed.path.split("/") if part]
        if segments and segments[-1].lower() == "raw":
            segments.pop()
        if len(segments) != 1 or not _SLUG_PATTERN.fullmatch(segments[0]):
            raise TeamSourceError("无法识别该 PokePaste 链接。")
        slug = segments[0]
        content = await self._fetch_bytes(
            f"https://pokepast.es/{slug}/raw",
            max_bytes=min(self._max_bytes, 128 * 1024),
        )
        team_text = self._decode_team_text(content)
        return ImportedTeam(team_text=team_text, label=f"PokePaste {slug}")

    async def _import_crobat(self, parsed: Any) -> ImportedTeam:
        segments = [unquote(part) for part in parsed.path.split("/") if part]
        if segments and segments[0].lower() in {"team", "teams"}:
            segments.pop(0)
        if len(segments) != 1 or not _SLUG_PATTERN.fullmatch(segments[0]):
            raise TeamSourceError("无法识别该 crob.at 队伍链接。")
        slug = segments[0]
        selector = 1
        if parsed.fragment:
            if not parsed.fragment.isdigit():
                raise TeamSourceError("多队伍链接请使用 #1、#2 等数字选择队伍。")
            selector = int(parsed.fragment)
        payload = await self._fetch_json(f"https://crob.at/api/team/{slug}")
        if not isinstance(payload, dict):
            raise TeamSourceError("crob.at 返回了无效队伍数据。")
        teams = payload.get("teams")
        if not isinstance(teams, list) or not teams:
            raise TeamSourceError("该 crob.at 链接中没有队伍。")
        if selector < 1 or selector > len(teams):
            raise TeamSourceError(f"该链接只包含 {len(teams)} 支队伍。")
        if len(teams) > 1 and not parsed.fragment:
            raise TeamSourceError(
                f"该链接包含 {len(teams)} 支队伍，请在链接末尾添加 #1 至 #{len(teams)}。"
            )
        selected = teams[selector - 1]
        paste = selected.get("paste") if isinstance(selected, dict) else None
        if not isinstance(paste, str):
            raise TeamSourceError("crob.at 队伍文本缺失。")
        team_text = self._validate_team_text(paste)
        name = str(payload.get("name") or slug).strip()
        source_format = (
            str(selected.get("format") or "").strip()
            if isinstance(selected, dict)
            else ""
        )
        return ImportedTeam(
            team_text=team_text,
            label=f"crob.at「{name}」",
            source_format_id=source_format or None,
        )

    async def _get_cached_json(
        self,
        category: str,
        source_id: str,
        url: str,
    ) -> Any:
        key = (category, source_id)
        now = time.monotonic()
        async with self._cache_lock:
            cached = self._cache.get(key)
            if cached and cached[0] > now:
                return copy.deepcopy(cached[1])
        payload = await self._fetch_json(url)
        expires_at = now + self._cache_ttl_seconds
        async with self._cache_lock:
            self._cache[key] = (expires_at, copy.deepcopy(payload))
        return payload

    async def _fetch_json(self, url: str) -> Any:
        content = await self._fetch_bytes(url, max_bytes=self._max_bytes)
        try:
            return json.loads(content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TeamSourceError("在线队伍数据不是有效的 UTF-8 JSON。") from exc

    async def _fetch_bytes(self, url: str, *, max_bytes: int) -> bytes:
        current_url = url
        try:
            for _ in range(3):
                async with self._client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 307, 308}:
                        location = response.headers.get("location", "")
                        if not self._allowed_redirect(current_url, location):
                            raise TeamSourceError("在线队伍源返回了不受信任的跳转。")
                        current_url = location
                        continue
                    if response.status_code != 200:
                        raise TeamSourceError(
                            f"读取在线队伍失败（HTTP {response.status_code}）。"
                        )
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            announced_size = int(content_length)
                        except ValueError:
                            announced_size = 0
                        if announced_size > max_bytes:
                            raise TeamSourceError("在线队伍数据过大。")
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes(chunk_size=16 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise TeamSourceError("在线队伍数据过大。")
                        chunks.append(chunk)
                    return b"".join(chunks)
            raise TeamSourceError("在线队伍源跳转次数过多。")
        except TeamSourceError:
            raise
        except httpx.HTTPError as exc:
            raise TeamSourceError(f"读取在线队伍失败：{exc}") from exc

    @staticmethod
    def _allowed_redirect(source_url: str, target_url: str) -> bool:
        source = urlsplit(source_url)
        target = urlsplit(target_url)
        return (
            source.scheme == "https"
            and (source.hostname or "").lower() == "data.pkmn.cc"
            and target.scheme == "https"
            and (target.hostname or "").lower() == "pkmn.github.io"
            and target.path.startswith("/smogon/data/")
        )

    @staticmethod
    def _decode_team_text(content: bytes) -> str:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise TeamSourceError("在线队伍文本不是有效的 UTF-8。") from exc
        return TeamSourceService._validate_team_text(text)

    @staticmethod
    def _validate_team_text(text: str) -> str:
        normalized = text.strip()
        if not normalized:
            raise TeamSourceError("在线链接没有返回队伍文本。")
        lowered = normalized[:512].lower()
        if "<html" in lowered or "<!doctype" in lowered:
            raise TeamSourceError("在线链接返回了网页，而不是 Showdown 队伍文本。")
        if len(normalized.encode("utf-8")) > 128 * 1024:
            raise TeamSourceError("在线队伍文本过大。")
        return normalized

    @staticmethod
    def _validate_source_id(source_id: str) -> str:
        normalized = source_id.strip().lower()
        if not _SOURCE_ID_PATTERN.fullmatch(normalized):
            raise TeamSourceError("在线队伍源标识无效。")
        return normalized

    @staticmethod
    def _split_set_selector(token: str) -> tuple[str, str | None]:
        normalized = token.replace("＝", "=", 1)
        if "=" not in normalized:
            return normalized.strip(), None
        species, selector = normalized.split("=", 1)
        if not species.strip() or not selector.strip():
            raise TeamSourceError(f"无法识别配招选择：{token}")
        return species.strip(), selector.strip()

    @staticmethod
    def _select_set(
        records: dict[str, dict[str, Any]], selector: str | None
    ) -> tuple[str, dict[str, Any]]:
        entries = list(records.items())
        if selector is None:
            return entries[0]
        if selector.isdigit():
            index = int(selector)
            if 1 <= index <= len(entries):
                return entries[index - 1]
            raise TeamSourceError(f"配招编号应在 1 至 {len(entries)} 之间。")
        selector_id = to_id(selector)
        for name, record in entries:
            if to_id(name) == selector_id:
                return name, record
        raise TeamSourceError(f"未找到配招：{selector}")

    @classmethod
    def _materialize_set(
        cls,
        species: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"species": species}
        for key in (
            "item",
            "ability",
            "nature",
            "evs",
            "ivs",
            "level",
            "gender",
            "happiness",
            "shiny",
        ):
            selected = cls._first_option(record.get(key))
            if selected is not None:
                result[key] = selected

        moves = record.get("moves")
        if not isinstance(moves, list) or not moves:
            raise TeamSourceError(f"{species} 的推荐配招缺少招式。")
        selected_moves = [cls._first_option(slot) for slot in moves]
        result["moves"] = [move for move in selected_moves if isinstance(move, str)]

        tera_value = record.get("teraType", record.get("teratypes"))
        tera_type = cls._first_option(tera_value)
        if isinstance(tera_type, str):
            result["teraType"] = tera_type
        return result

    @classmethod
    def _first_option(cls, value: Any) -> Any:
        if isinstance(value, list):
            if not value:
                return None
            return cls._first_option(value[0])
        return copy.deepcopy(value)


__all__ = [
    "BuiltTeam",
    "ImportedTeam",
    "PreparedSample",
    "RecommendedSetList",
    "RecommendedSetOption",
    "SampleTeam",
    "TeamSourceError",
    "TeamSourceService",
]
