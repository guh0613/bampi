from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any


@dataclass(slots=True)
class RefEntry:
    ref: str
    page_id: str
    session_id: str
    backend_node_id: int
    document_generation: int
    session_generation: int
    role: str
    name: str
    frame_id: str | None = None
    nth: int = 0


@dataclass(slots=True)
class PageState:
    page_id: str
    target_id: str
    session_id: str
    url: str = "about:blank"
    title: str = ""
    document_generation: int = 0
    main_frame_id: str | None = None
    refs: dict[str, RefEntry] = field(default_factory=dict)
    snapshot_sequence: int = 0
    session_generations: dict[str, int] = field(default_factory=dict)
    dialog: dict[str, Any] | None = None
    console: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    errors: deque[str] = field(default_factory=lambda: deque(maxlen=100))
    network: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=300))
    network_inflight: set[str] = field(default_factory=set)
    last_network_activity: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class SnapshotResult:
    text: str
    refs: dict[str, RefEntry]
    node_count: int


@dataclass(slots=True)
class CommandOutput:
    text: str
    image_data: bytes | None = None
    image_mime_type: str | None = None


@dataclass(slots=True)
class RecordingState:
    page_id: str
    path: str
    process: Any
    task: Any
    started_at: float
