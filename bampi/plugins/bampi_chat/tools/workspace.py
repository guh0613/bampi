from __future__ import annotations

import mimetypes
import re
import shutil
from pathlib import Path, PurePosixPath


def ensure_workspace_dirs(workspace_dir: str) -> Path:
    root = Path(workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "inbox").mkdir(parents=True, exist_ok=True)
    (root / "outbox").mkdir(parents=True, exist_ok=True)
    return root


def resolve_group_workspace_dir(workspace_root_dir: str, group_id: str) -> Path:
    root = Path(workspace_root_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return ensure_workspace_dirs(str(root / group_workspace_name(group_id)))


def group_workspace_name(group_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", (group_id or "").strip()).strip("._-")
    return f"group-{sanitized or 'default'}"


def resolve_group_container_workspace(container_root: str, group_id: str) -> str:
    return (PurePosixPath(container_root) / group_workspace_name(group_id)).as_posix()


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
