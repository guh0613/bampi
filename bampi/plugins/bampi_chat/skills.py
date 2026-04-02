from __future__ import annotations

import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from bampy.app import LoadSkillsResult, Skill, SkillDiagnostic, load_skills

DEFAULT_SKILL_INSTALL_DIR = ".agents/skills"
DEFAULT_BUILTIN_SKILL_MIRROR_DIR = ".agents/builtin-skills"
LEGACY_SKILL_INSTALL_DIR = ".bampy/skills"
_EXPLICIT_SKILL_PREFIX_RE = re.compile(
    r"^/(?P<name>[A-Za-z][A-Za-z0-9_-]{0,63})(?P<delimiter>$|[\s,.;:!?，。；：！？、])"
)
_SKILL_ROOT_MARKERS = (
    (".agents", "skills"),
    (".agents", "builtin-skills"),
    (".bampy", "skills"),
)


@dataclass(slots=True)
class ParsedSkillCommand:
    action: Literal["help", "list", "show", "install"]
    argument: str = ""
    force: bool = False


@dataclass(slots=True)
class ExplicitSkillResolution:
    requested_names: list[str]
    skills: list[Skill]
    missing_names: list[str]
    cleaned_text: str
    diagnostics: list[SkillDiagnostic]


@dataclass(slots=True)
class SkillInstallResult:
    installed_names: list[str]
    replaced_names: list[str]
    diagnostics: list[SkillDiagnostic]
    target_root: str


def skill_install_root(workspace_dir: str) -> Path:
    return (Path(workspace_dir).resolve() / DEFAULT_SKILL_INSTALL_DIR).resolve()


def builtin_skill_source_root() -> Path:
    return (Path(__file__).resolve().parent / "builtin_skills").resolve()


def builtin_skill_mirror_root(workspace_dir: str) -> Path:
    return (Path(workspace_dir).resolve() / DEFAULT_BUILTIN_SKILL_MIRROR_DIR).resolve()


def skill_search_roots(workspace_dir: str) -> list[Path]:
    workspace_root = Path(workspace_dir).resolve()
    roots = [
        (workspace_root / DEFAULT_SKILL_INSTALL_DIR).resolve(),
        (workspace_root / DEFAULT_BUILTIN_SKILL_MIRROR_DIR).resolve(),
        (workspace_root / LEGACY_SKILL_INSTALL_DIR).resolve(),
    ]
    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        result.append(root)
    return result


def load_chat_skills(workspace_dir: str) -> LoadSkillsResult:
    diagnostics: list[SkillDiagnostic] = []
    try:
        _sync_builtin_skills_into_workspace(workspace_dir)
    except Exception as exc:
        diagnostics.append(
            SkillDiagnostic(
                type="warning",
                message=f"failed to sync builtin skills: {exc}",
                path=str(builtin_skill_source_root()),
            )
        )

    roots = [str(root) for root in skill_search_roots(workspace_dir) if root.exists()]
    if not roots:
        return LoadSkillsResult(skills=[], diagnostics=diagnostics)

    result = load_skills(
        cwd=workspace_dir,
        skill_paths=roots,
        include_defaults=False,
    )
    result.diagnostics = diagnostics + result.diagnostics
    for skill in result.skills:
        if _allow_implicit_invocation(skill.base_dir) is False:
            skill.disable_model_invocation = True
    return result


def build_prompt_skills(skills: list[Skill], *, workspace_dir: str) -> list[Skill]:
    prompt_skills: list[Skill] = []
    for skill in skills:
        display_path = display_skill_path(skill.file_path, workspace_dir=workspace_dir)
        display_base_dir = display_skill_path(skill.base_dir, workspace_dir=workspace_dir)
        prompt_skills.append(
            Skill(
                name=skill.name,
                description=skill.description,
                file_path=display_path,
                base_dir=display_base_dir,
                source=skill.source,
                disable_model_invocation=skill.disable_model_invocation,
            )
        )
    return prompt_skills


def describe_skill_resource_path(path: str | None) -> tuple[str, str] | None:
    text = (path or "").strip()
    if not text:
        return None

    normalized_parts = PurePosixPath(text.replace("\\", "/")).parts
    if len(normalized_parts) < 3:
        return None

    for marker in _SKILL_ROOT_MARKERS:
        marker_length = len(marker)
        for index in range(len(normalized_parts) - marker_length):
            if normalized_parts[index : index + marker_length] != marker:
                continue
            skill_index = index + marker_length
            if skill_index >= len(normalized_parts):
                return None
            skill_name = normalized_parts[skill_index]
            relative_parts = normalized_parts[skill_index + 1 :]
            if not relative_parts:
                return skill_name, "SKILL.md"
            return skill_name, PurePosixPath(*relative_parts).as_posix()
    return None


def parse_skill_command(text: str | None) -> ParsedSkillCommand | None:
    raw = (text or "").strip()
    if not raw:
        return None

    parts = raw.split(maxsplit=2)
    head = parts[0].lower()
    if head not in {"/skill", "/skills"}:
        return None

    if len(parts) == 1:
        return ParsedSkillCommand(action="list")

    action = parts[1].lower()
    remainder = parts[2].strip() if len(parts) > 2 else ""

    if action == "list":
        return ParsedSkillCommand(action="list")
    if action == "help":
        return ParsedSkillCommand(action="help")
    if action == "show":
        return ParsedSkillCommand(action="show", argument=remainder)
    if action == "install":
        force = False
        source_parts: list[str] = []
        for token in remainder.split():
            if token in {"--force", "-f"}:
                force = True
                continue
            source_parts.append(token)
        return ParsedSkillCommand(
            action="install",
            argument=" ".join(source_parts).strip(),
            force=force,
        )

    return ParsedSkillCommand(action="help", argument=action)


def extract_explicit_skill_names(text: str | None) -> list[str]:
    raw = (text or "").strip()
    if not raw or parse_skill_command(raw) is not None:
        return []

    match = _EXPLICIT_SKILL_PREFIX_RE.match(raw)
    if match is None:
        return []
    return [match.group("name")]


def strip_explicit_skill_mentions(text: str | None) -> str:
    raw = (text or "").strip()
    if not raw or parse_skill_command(raw) is not None:
        return _normalize_whitespace(raw)

    match = _EXPLICIT_SKILL_PREFIX_RE.match(raw)
    if match is None:
        return _normalize_whitespace(raw)
    return _normalize_whitespace(raw[match.end() :])


def resolve_explicit_skills(
    text: str | None,
    *,
    workspace_dir: str,
) -> ExplicitSkillResolution:
    requested_names = extract_explicit_skill_names(text)
    if not requested_names:
        return ExplicitSkillResolution(
            requested_names=[],
            skills=[],
            missing_names=[],
            cleaned_text=_normalize_whitespace(text or ""),
            diagnostics=[],
        )

    loaded = load_chat_skills(workspace_dir)
    by_name = {skill.name.lower(): skill for skill in loaded.skills}
    skills: list[Skill] = []
    missing: list[str] = []

    for requested in requested_names:
        skill = by_name.get(requested.lower())
        if skill is None:
            missing.append(requested)
            continue
        skills.append(skill)

    return ExplicitSkillResolution(
        requested_names=requested_names,
        skills=skills,
        missing_names=missing,
        cleaned_text=strip_explicit_skill_mentions(text),
        diagnostics=loaded.diagnostics,
    )


def build_explicit_skill_payload_text(
    skills: list[Skill],
    *,
    workspace_dir: str,
) -> str:
    if not skills:
        return ""

    sections: list[str] = [
        "explicit_skill_payloads:",
        "以下 skill 由用户显式指定，本轮必须优先遵循；如果 skill 里引用相对路径，请相对 skill 根目录解析。",
    ]

    for skill in skills:
        skill_path = Path(skill.file_path)
        content = skill_path.read_text(encoding="utf-8")
        sections.extend(
            [
                "",
                f"## skill: {skill.name}",
                f"path: {display_skill_path(skill.file_path, workspace_dir=workspace_dir)}",
                f"description: {skill.description}",
                "",
                content.strip(),
            ]
        )

    return "\n".join(section for section in sections if section is not None).strip()


def display_skill_path(path: str, *, workspace_dir: str) -> str:
    resolved = Path(path).resolve()
    workspace_root = Path(workspace_dir).resolve()
    try:
        return PurePosixPath(resolved.relative_to(workspace_root)).as_posix()
    except ValueError:
        return resolved.as_posix()


def format_skill_list(skills: list[Skill], *, workspace_dir: str) -> str:
    if not skills:
        return (
            "当前还没有已安装的 skill。\n"
            "发送或引用一个 skill 压缩包/Markdown 文件后执行 `/skill install`，"
            "或直接使用 `/skill install https://...` 安装。\n"
            "显式调用时，在消息最开头写 `/skill-name`。"
        )

    lines = [f"已安装 {len(skills)} 个 skill："]
    for skill in sorted(skills, key=lambda item: item.name.lower()):
        mode = "仅显式触发" if skill.disable_model_invocation else "可自行调用"
        origin = _skill_origin_label(skill, workspace_dir=workspace_dir)
        lines.append(f"- {skill.name} [{origin}，{mode}]：{skill.description}")
        lines.append(f"  路径：{display_skill_path(skill.file_path, workspace_dir=workspace_dir)}")
    lines.append("显式调用：在消息最开头写 `/skill-name`。")
    return "\n".join(lines)


def format_skill_details(skill: Skill, *, workspace_dir: str) -> str:
    mode = "仅显式触发" if skill.disable_model_invocation else "可自行调用"
    origin = _skill_origin_label(skill, workspace_dir=workspace_dir)
    return "\n".join(
        [
            skill.name,
            f"描述：{skill.description}",
            f"来源：{origin}",
            f"模式：{mode}",
            f"路径：{display_skill_path(skill.file_path, workspace_dir=workspace_dir)}",
            "显式调用：在消息最开头写 `/"
            f"{skill.name}`，或把它和普通问题写在同一条消息里。",
        ]
    )


def format_skill_help() -> str:
    return "\n".join(
        [
            "Skill 命令：",
            "- `/skills` 或 `/skill list`：查看已安装 skill",
            "- `/skill show <name>`：查看 skill 简介",
            "- 发送或引用 skill 文件后执行 `/skill install [--force]`：安装 skill 包",
            "- `/skill install https://... [--force]`：通过 URL 安装 skill 包",
            "显式调用：在普通消息最开头写 `/skill-name`，比如 `/code-review 看看这个文件`。",
        ]
    )


def install_skills_from_source(
    source: str,
    *,
    workspace_dir: str,
    force: bool = False,
    max_bytes: int = 20 * 1024 * 1024,
    timeout: float = 20.0,
) -> SkillInstallResult:
    if not source.strip():
        raise ValueError("缺少安装来源，请提供 inbox 路径、本地路径或 http(s) 链接。")

    target_root = skill_install_root(workspace_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="bampi-skill-install-") as temp_dir:
        temp_root = Path(temp_dir)
        source_path = _materialize_source(
            source,
            workspace_dir=workspace_dir,
            temp_root=temp_root,
            max_bytes=max_bytes,
            timeout=timeout,
        )
        staged_root = _normalize_source_tree(source_path, temp_root=temp_root)
        discovered = load_skills(
            cwd=str(staged_root),
            skill_paths=[str(staged_root)],
            include_defaults=False,
        )
        for skill in discovered.skills:
            if _allow_implicit_invocation(skill.base_dir) is False:
                skill.disable_model_invocation = True

        if not discovered.skills:
            detail = _format_diagnostics(discovered.diagnostics)
            raise ValueError(
                "没有在来源中发现可安装的 skill。"
                + (f"\n{detail}" if detail else "")
            )

        if any(diagnostic.type == "collision" for diagnostic in discovered.diagnostics):
            raise ValueError(
                "来源中存在同名 skill 冲突，请整理后再安装。\n"
                f"{_format_diagnostics(discovered.diagnostics)}"
            )

        installed_names = [skill.name for skill in discovered.skills]
        replaced_names: list[str] = []
        destinations = {skill.name: target_root / skill.name for skill in discovered.skills}

        conflicts = [name for name, destination in destinations.items() if destination.exists()]
        if conflicts and not force:
            joined = ", ".join(sorted(conflicts))
            raise FileExistsError(
                f"以下 skill 已存在：{joined}。如需覆盖，请追加 `--force`。"
            )

        for name, destination in destinations.items():
            if destination.exists():
                replaced_names.append(name)

        for skill in discovered.skills:
            source_root = Path(skill.base_dir).resolve()
            destination = destinations[skill.name]
            if source_root == destination.resolve():
                raise ValueError(f"skill `{skill.name}` 已经位于安装目录中，无需重复安装。")
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source_root, destination)

        return SkillInstallResult(
            installed_names=installed_names,
            replaced_names=replaced_names,
            diagnostics=discovered.diagnostics,
            target_root=str(target_root),
        )


def _materialize_source(
    source: str,
    *,
    workspace_dir: str,
    temp_root: Path,
    max_bytes: int,
    timeout: float,
) -> Path:
    if _looks_like_url(source):
        return _download_source(source, temp_root=temp_root, max_bytes=max_bytes, timeout=timeout)

    candidate = Path(os.path.expanduser(source))
    if not candidate.is_absolute():
        candidate = (Path(workspace_dir).resolve() / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"找不到安装来源：{candidate}")
    return candidate


def _normalize_source_tree(source_path: Path, *, temp_root: Path) -> Path:
    if source_path.is_dir():
        return source_path

    normalized_root = temp_root / "normalized"
    normalized_root.mkdir(parents=True, exist_ok=True)

    if _is_archive(source_path):
        extracted = normalized_root / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        _extract_archive(source_path, extracted)
        return extracted

    if _looks_like_markdown(source_path):
        staged_skill = normalized_root / source_path.stem
        staged_skill.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, staged_skill / "SKILL.md")
        return normalized_root

    raise ValueError(
        "暂不支持该安装来源。请提供 skill 目录、Markdown skill 文件、zip 或 tar 归档。"
    )


def _download_source(
    url: str,
    *,
    temp_root: Path,
    max_bytes: int,
    timeout: float,
) -> Path:
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; BampiBot/0.1)"},
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"下载内容超过限制：{max_bytes} bytes")

    filename = _download_filename(url, content_type=content_type)
    target = temp_root / filename
    target.write_bytes(data)
    return target


def _download_filename(url: str, *, content_type: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path or "")).name
    if not name:
        name = "downloaded-skill"

    lower_name = name.lower()
    if _looks_like_markdown(Path(lower_name)):
        return name
    if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz") or lower_name.endswith(".zip"):
        return name
    if content_type in {"application/zip", "application/x-zip-compressed"}:
        return f"{name}.zip"
    if content_type in {
        "application/gzip",
        "application/x-gzip",
        "application/x-tar",
    }:
        return f"{name}.tar.gz"
    return f"{name}.md"


def _sync_builtin_skills_into_workspace(workspace_dir: str) -> None:
    source_root = builtin_skill_source_root()
    if not source_root.is_dir():
        return

    destination_root = builtin_skill_mirror_root(workspace_dir)
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    for entry in sorted(source_root.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue
        shutil.copytree(entry, destination_root / entry.name)


def _is_archive(path: Path) -> bool:
    if zipfile.is_zipfile(path):
        return True
    try:
        return tarfile.is_tarfile(path)
    except OSError:
        return False


def _looks_like_markdown(path: Path) -> bool:
    return path.suffix.lower() == ".md"


def _extract_archive(source_path: Path, destination: Path) -> None:
    if zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as archive:
            for member in archive.infolist():
                target = _safe_archive_target(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        return

    with tarfile.open(source_path) as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError("skill 归档中不允许符号链接")
            target = _safe_archive_target(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with extracted, open(target, "wb") as dst:
                shutil.copyfileobj(extracted, dst)


def _safe_archive_target(destination: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/").strip("/")
    if not normalized:
        return destination
    target = (destination / normalized).resolve()
    try:
        target.relative_to(destination.resolve())
    except ValueError as exc:
        raise ValueError(f"非法归档路径：{member_name}") from exc
    return target


def _allow_implicit_invocation(base_dir: str) -> bool | None:
    metadata_path = Path(base_dir) / "agents" / "openai.yaml"
    if not metadata_path.is_file():
        return None

    try:
        raw_text = metadata_path.read_text(encoding="utf-8")
    except OSError:
        return None

    parsed = _parse_openai_metadata(raw_text)
    policy = parsed.get("policy")
    if isinstance(policy, dict):
        value = policy.get("allow_implicit_invocation")
        if isinstance(value, bool):
            return value
    return None


def _parse_openai_metadata(raw_text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        yaml = None

    if yaml is not None:
        try:
            loaded = yaml.safe_load(raw_text)
        except Exception:
            loaded = None
        if isinstance(loaded, dict):
            return loaded

    policy: dict[str, Any] = {}
    in_policy = False
    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            in_policy = stripped.startswith("policy:")
            if in_policy and stripped != "policy:":
                value = stripped.partition(":")[2].strip()
                if value.startswith("{") and value.endswith("}"):
                    inner = value.strip("{} ").split(",")
                    for item in inner:
                        key, _, raw_value = item.partition(":")
                        if key.strip() == "allow_implicit_invocation":
                            parsed = _parse_yaml_bool(raw_value.strip())
                            if parsed is not None:
                                policy["allow_implicit_invocation"] = parsed
            continue

        if not in_policy:
            continue
        key, _, raw_value = stripped.partition(":")
        if key.strip() != "allow_implicit_invocation":
            continue
        parsed = _parse_yaml_bool(raw_value.strip())
        if parsed is not None:
            policy["allow_implicit_invocation"] = parsed

    return {"policy": policy} if policy else {}


def _parse_yaml_bool(value: str) -> bool | None:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _skill_origin_label(skill: Skill, *, workspace_dir: str) -> str:
    resolved_path = Path(skill.file_path).resolve()
    builtin_root = builtin_skill_mirror_root(workspace_dir)
    install_root = skill_install_root(workspace_dir)

    try:
        resolved_path.relative_to(builtin_root)
    except ValueError:
        pass
    else:
        return "内置"

    try:
        resolved_path.relative_to(install_root)
    except ValueError:
        return "项目"
    return "已安装"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _format_diagnostics(diagnostics: list[SkillDiagnostic]) -> str:
    lines: list[str] = []
    for diagnostic in diagnostics:
        lines.append(f"- {diagnostic.message} ({diagnostic.path})")
    return "\n".join(lines)
