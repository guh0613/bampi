from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from bampy.app import LoadSkillsResult, Skill, SkillDiagnostic, load_skills

DEFAULT_WORKSPACE_SKILL_DIR = ".agents/skills"
DEFAULT_BUILTIN_SKILL_MIRROR_DIR = ".agents/builtin-skills"
LEGACY_WORKSPACE_SKILL_DIR = ".bampy/skills"
_SKILL_ROOT_MARKERS = (
    (".agents", "skills"),
    (".agents", "builtin-skills"),
    (".bampy", "skills"),
)


@dataclass(slots=True)
class SkillResourceContext:
    skill_name: str
    resource_path: str
    skill_root: str


def builtin_skill_source_root() -> Path:
    return (Path(__file__).resolve().parent / "builtin_skills").resolve()


def builtin_skill_mirror_root(workspace_dir: str) -> Path:
    return (Path(workspace_dir).resolve() / DEFAULT_BUILTIN_SKILL_MIRROR_DIR).resolve()


def skill_search_roots(workspace_dir: str) -> list[Path]:
    workspace_root = Path(workspace_dir).resolve()
    roots = [
        (workspace_root / DEFAULT_WORKSPACE_SKILL_DIR).resolve(),
        (workspace_root / DEFAULT_BUILTIN_SKILL_MIRROR_DIR).resolve(),
        (workspace_root / LEGACY_WORKSPACE_SKILL_DIR).resolve(),
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
    context = describe_skill_resource_context(path)
    if context is None:
        return None
    return context.skill_name, context.resource_path


def describe_skill_resource_context(path: str | None) -> SkillResourceContext | None:
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
            skill_root = PurePosixPath(*normalized_parts[: skill_index + 1]).as_posix()
            if not relative_parts:
                return SkillResourceContext(
                    skill_name=skill_name,
                    resource_path="SKILL.md",
                    skill_root=skill_root,
                )
            return SkillResourceContext(
                skill_name=skill_name,
                resource_path=PurePosixPath(*relative_parts).as_posix(),
                skill_root=skill_root,
            )
    return None


def display_skill_path(path: str, *, workspace_dir: str) -> str:
    resolved = Path(path).resolve()
    workspace_root = Path(workspace_dir).resolve()
    try:
        return PurePosixPath(resolved.relative_to(workspace_root)).as_posix()
    except ValueError:
        return resolved.as_posix()


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
