from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

_GROUP_ALIAS_STORE_VERSION = 1
_GROUP_ALIAS_STORE_LOCK = threading.Lock()
_GROUP_ALIAS_PATTERN = re.compile(r"^chat-[0-9a-f]{8}(?:[0-9a-f]{8})?$")
DEFAULT_WORKSPACE_CLEANUP_TTL_SECONDS = 3 * 24 * 60 * 60
WORKSPACE_CLEANUP_KEEP_ROOT_DIRS = frozenset({"inbox", "outbox"})
WORKSPACE_CLEANUP_PROTECTED_ROOT_DIR_NAMES = frozenset({"persistent"})
WORKSPACE_CLEANUP_PROTECTED_DIR_NAMES = frozenset(
    {
        ".agents",
        ".bampi-services",
        ".bampy",
        ".browser",
        ".git",
        ".hg",
        ".npm",
        ".pnpm-store",
        ".svn",
        ".uv",
        ".venv",
        ".yarn",
        "node_modules",
        "venv",
    }
)
WORKSPACE_CLEANUP_PROTECTED_FILE_PATTERNS = frozenset(
    {
        ".env",
        ".env.*",
        ".npmrc",
        ".yarnrc",
        ".yarnrc.yml",
        "cookies*.json",
        "storage-state*.json",
        "storage_state*.json",
    }
)


@dataclass(slots=True)
class WorkspaceCleanupResult:
    workspace_dir: str
    scanned_files: int = 0
    scanned_dirs: int = 0
    skipped_paths: int = 0
    deleted_files: int = 0
    deleted_dirs: int = 0
    deleted_samples: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def deleted_total(self) -> int:
        return self.deleted_files + self.deleted_dirs

    def record_deleted(self, path: Path, *, root: Path, is_dir: bool) -> None:
        if is_dir:
            self.deleted_dirs += 1
        else:
            self.deleted_files += 1
        if len(self.deleted_samples) < 20:
            self.deleted_samples.append(to_workspace_relative(str(root), path))


def ensure_workspace_dirs(workspace_dir: str) -> Path:
    root = Path(workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "inbox").mkdir(parents=True, exist_ok=True)
    (root / "outbox").mkdir(parents=True, exist_ok=True)
    (root / "persistent").mkdir(parents=True, exist_ok=True)
    return root


def resolve_group_workspace_dir(workspace_root_dir: str, group_id: str) -> Path:
    root = Path(workspace_root_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    alias = group_workspace_name(group_id, workspace_root_dir=str(root))
    target = root / alias
    legacy = root / _legacy_group_workspace_name(group_id)
    if legacy.exists() and not target.exists():
        legacy.rename(target)
    return ensure_workspace_dirs(str(target))


def group_workspace_name(group_id: str, *, workspace_root_dir: str | None = None) -> str:
    if workspace_root_dir is not None:
        return _group_workspace_alias(Path(workspace_root_dir).resolve(), group_id)
    return _legacy_group_workspace_name(group_id)


def _legacy_group_workspace_name(group_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", (group_id or "").strip()).strip("._-")
    return f"group-{sanitized or 'default'}"


def resolve_group_container_workspace(
    container_root: str,
    group_id: str,
    *,
    workspace_root_dir: str | None = None,
) -> str:
    return (
        PurePosixPath(container_root)
        / group_workspace_name(group_id, workspace_root_dir=workspace_root_dir)
    ).as_posix()


def _group_workspace_alias(workspace_root: Path, group_id: str) -> str:
    normalized_group_id = str(group_id or "").strip() or "default"
    store_path = _group_alias_store_path(workspace_root)
    with _GROUP_ALIAS_STORE_LOCK:
        store = _read_group_alias_store(store_path)
        groups = store.setdefault("groups", {})
        alias = groups.get(normalized_group_id)
        if isinstance(alias, str) and _GROUP_ALIAS_PATTERN.fullmatch(alias):
            return alias

        existing_aliases = {
            value
            for value in groups.values()
            if isinstance(value, str) and _GROUP_ALIAS_PATTERN.fullmatch(value)
        }
        alias = _new_group_alias(existing_aliases)
        groups[normalized_group_id] = alias
        _write_group_alias_store(store_path, store)
        return alias


def _group_alias_store_path(workspace_root: Path) -> Path:
    return workspace_root.parent / f".{workspace_root.name}-group-aliases.json"


def _read_group_alias_store(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"version": _GROUP_ALIAS_STORE_VERSION, "groups": {}}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid group alias store: {path}")
    groups = data.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"Invalid group alias store groups: {path}")
    data["version"] = _GROUP_ALIAS_STORE_VERSION
    return data


def _write_group_alias_store(path: Path, store: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp-{secrets.token_hex(4)}")
    payload = json.dumps(store, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path.write_text(payload + "\n", encoding="utf-8")
    temp_path.replace(path)


def _new_group_alias(existing_aliases: set[str]) -> str:
    while True:
        alias = f"chat-{secrets.token_hex(4)}"
        if alias not in existing_aliases:
            return alias


def reset_workspace_files(workspace_dir: str) -> Path:
    root = ensure_workspace_dirs(workspace_dir)
    for entry in root.iterdir():
        if entry.name in {"inbox", "outbox"}:
            _clear_directory_contents(entry)
            continue
        if _is_cleanup_protected_path(root, entry, is_dir=entry.is_dir()):
            continue
        _remove_path(entry)

    return ensure_workspace_dirs(str(root))


def cleanup_stale_workspace_files(
    workspace_dir: str,
    *,
    ttl_seconds: float = DEFAULT_WORKSPACE_CLEANUP_TTL_SECONDS,
    now: float | None = None,
) -> WorkspaceCleanupResult:
    root = ensure_workspace_dirs(workspace_dir)
    current_time = time.time() if now is None else now
    cutoff = current_time - max(0.0, ttl_seconds)
    result = WorkspaceCleanupResult(workspace_dir=str(root))

    dirs_for_empty_prune: list[Path] = []
    dir_last_used: dict[Path, float] = {}
    for current_dir, dir_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory = Path(current_dir)
        result.scanned_dirs += 1
        dirs_for_empty_prune.append(directory)
        with contextlib.suppress(OSError):
            dir_last_used[directory] = _path_last_used(directory)

        kept_dirs: list[str] = []
        for name in dir_names:
            child = directory / name
            if _is_cleanup_protected_path(root, child, is_dir=True):
                result.skipped_paths += 1
                continue
            kept_dirs.append(name)
        dir_names[:] = kept_dirs

        for name in file_names:
            path = directory / name
            result.scanned_files += 1
            if _is_cleanup_protected_path(root, path, is_dir=False):
                result.skipped_paths += 1
                continue
            try:
                if _path_last_used(path) > cutoff:
                    continue
                _remove_path(path)
                result.record_deleted(path, root=root, is_dir=False)
            except OSError as exc:
                result.errors.append(f"{to_workspace_relative(str(root), path)}: {exc}")

    for directory in reversed(dirs_for_empty_prune):
        if directory == root:
            continue
        if directory.parent == root and directory.name in WORKSPACE_CLEANUP_KEEP_ROOT_DIRS:
            continue
        if _is_cleanup_protected_path(root, directory, is_dir=True):
            continue
        try:
            last_used = dir_last_used.get(directory)
            if last_used is None:
                last_used = _path_last_used(directory)
            if last_used > cutoff:
                continue
            if any(directory.iterdir()):
                continue
            directory.rmdir()
            result.record_deleted(directory, root=root, is_dir=True)
        except OSError as exc:
            result.errors.append(f"{to_workspace_relative(str(root), directory)}: {exc}")

    ensure_workspace_dirs(str(root))
    return result


def cleanup_stale_group_workspaces(
    workspace_root_dir: str,
    *,
    ttl_seconds: float = DEFAULT_WORKSPACE_CLEANUP_TTL_SECONDS,
    now: float | None = None,
    skip_workspace_dirs: set[str] | None = None,
) -> list[WorkspaceCleanupResult]:
    root = Path(workspace_root_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    skipped = {Path(item).resolve() for item in (skip_workspace_dirs or set())}
    results: list[WorkspaceCleanupResult] = []
    for workspace_dir in iter_group_workspace_dirs(str(root)):
        if workspace_dir.resolve() in skipped:
            continue
        results.append(
            cleanup_stale_workspace_files(
                str(workspace_dir),
                ttl_seconds=ttl_seconds,
                now=now,
            )
        )
    return results


def iter_group_workspace_dirs(workspace_root_dir: str) -> list[Path]:
    root = Path(workspace_root_dir).resolve()
    if not root.exists():
        return []

    workspaces: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        if _is_group_workspace_dir(entry):
            workspaces.append(ensure_workspace_dirs(str(entry)))
    return workspaces


def mark_workspace_path_used(
    workspace_dir: str,
    user_path: str | None,
    *,
    container_root: str | None = None,
) -> None:
    try:
        path = resolve_workspace_path(
            workspace_dir,
            user_path or ".",
            container_root=container_root,
        )
    except ValueError:
        return
    _touch_access_time(path)


def resolve_workspace_path(
    workspace_dir: str,
    user_path: str | None,
    *,
    container_root: str | None = None,
) -> Path:
    text = (user_path or ".").strip()
    if not text:
        raise ValueError("path must not be empty")

    if container_root:
        try:
            relative = PurePosixPath(text).relative_to(PurePosixPath(container_root))
        except ValueError:
            pass
        else:
            text = relative.as_posix()

    root = Path(workspace_dir).resolve()
    candidate = Path(text)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {user_path}") from exc
    return resolved


def to_workspace_relative(workspace_dir: str, path: Path) -> str:
    return path.resolve().relative_to(Path(workspace_dir).resolve()).as_posix()


def host_to_container_path(workspace_dir: str, path: Path, container_root: str) -> str:
    relative = to_workspace_relative(workspace_dir, path)
    base = container_root.rstrip("/")
    return f"{base}/{relative}" if relative else base


def container_to_host_path(workspace_dir: str, container_path: str, container_root: str) -> Path | None:
    try:
        relative = PurePosixPath(container_path).relative_to(PurePosixPath(container_root))
    except ValueError:
        return None
    return (Path(workspace_dir).resolve() / Path(relative.as_posix())).resolve()


def is_image_file(path: Path) -> bool:
    mime_type, _ = mimetypes.guess_type(path.name)
    return bool(mime_type and mime_type.startswith("image/"))


def _is_group_workspace_dir(path: Path) -> bool:
    if _GROUP_ALIAS_PATTERN.fullmatch(path.name):
        return True
    if path.name.startswith("group-"):
        return True
    return (path / "inbox").is_dir() and (path / "outbox").is_dir()


def _is_cleanup_protected_path(root: Path, path: Path, *, is_dir: bool) -> bool:
    try:
        relative_parts = path.resolve().relative_to(root).parts
    except ValueError:
        return True

    if relative_parts and relative_parts[0] in WORKSPACE_CLEANUP_PROTECTED_ROOT_DIR_NAMES:
        return True
    if any(part in WORKSPACE_CLEANUP_PROTECTED_DIR_NAMES for part in relative_parts):
        return True
    if is_dir:
        return False
    return any(fnmatchcase(path.name, pattern) for pattern in WORKSPACE_CLEANUP_PROTECTED_FILE_PATTERNS)


def _path_last_used(path: Path) -> float:
    stat = path.stat(follow_symlinks=False)
    return max(stat.st_atime, stat.st_mtime)


def _touch_access_time(path: Path) -> None:
    try:
        stat = path.stat(follow_symlinks=False)
        now_ns = time.time_ns()
        os.utime(path, ns=(now_ns, stat.st_mtime_ns), follow_symlinks=False)
    except OSError:
        return


def _clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        _remove_path(child)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)
