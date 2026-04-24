from __future__ import annotations

import json
import mimetypes
import re
import secrets
import shutil
import threading
from pathlib import Path, PurePosixPath

_GROUP_ALIAS_STORE_VERSION = 1
_GROUP_ALIAS_STORE_LOCK = threading.Lock()
_GROUP_ALIAS_PATTERN = re.compile(r"^chat-[0-9a-f]{8}(?:[0-9a-f]{8})?$")


def ensure_workspace_dirs(workspace_dir: str) -> Path:
    root = Path(workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "inbox").mkdir(parents=True, exist_ok=True)
    (root / "outbox").mkdir(parents=True, exist_ok=True)
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
        if entry.name == ".agents":
            _reset_agents_dir(entry)
            continue
        if entry.name == ".bampy":
            _reset_legacy_skill_dir(entry)
            continue
        _remove_path(entry)

    return ensure_workspace_dirs(str(root))


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


def _reset_agents_dir(agents_dir: Path) -> None:
    _prune_directory_except(
        agents_dir,
        keep_names={"skills", "builtin-skills"},
    )


def _reset_legacy_skill_dir(legacy_dir: Path) -> None:
    _prune_directory_except(
        legacy_dir,
        keep_names={"skills"},
    )


def _prune_directory_except(directory: Path, *, keep_names: set[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.name in keep_names:
            continue
        _remove_path(child)


def _clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        _remove_path(child)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)
