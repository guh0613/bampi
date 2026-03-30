from __future__ import annotations

import mimetypes
from pathlib import Path, PurePosixPath


def ensure_workspace_dirs(workspace_dir: str) -> Path:
    root = Path(workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "inbox").mkdir(parents=True, exist_ok=True)
    (root / "outbox").mkdir(parents=True, exist_ok=True)
    return root


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


def is_image_file(path: Path) -> bool:
    mime_type, _ = mimetypes.guess_type(path.name)
    return bool(mime_type and mime_type.startswith("image/"))
