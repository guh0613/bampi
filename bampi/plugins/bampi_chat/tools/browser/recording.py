from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
from pathlib import Path
import shutil
import time

from .artifacts import ArtifactManager
from .errors import CommandError
from .models import PageState, RecordingState
from .runtime import BrowserRuntime


class RecordingManager:
    def __init__(self, runtime: BrowserRuntime, artifacts: ArtifactManager) -> None:
        self.runtime = runtime
        self.artifacts = artifacts

    async def start(self, page: PageState, path: str | None) -> str:
        if self.runtime.recording is not None:
            raise CommandError(
                f"A recording is already active for {self.runtime.recording.page_id}: "
                f"{self.artifacts.display_path(Path(self.runtime.recording.path))}"
            )
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise CommandError("Recording requires ffmpeg to be installed and available on PATH.")
        requested = Path(path).suffix.lower() if path else ".mp4"
        suffix = requested if requested in {".mp4", ".webm"} else ".mp4"
        output_path = self.artifacts.resolve_path(path, suffix=suffix, stem=f"{page.page_id}-recording")
        codec_args = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32"] if suffix == ".webm" else ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        process = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "image2pipe", "-framerate", str(self.runtime.config.recording_fps),
            "-vcodec", "mjpeg", "-i", "-", *codec_args, str(output_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        task = asyncio.create_task(
            self._capture_loop(page, process),
            name=f"bampi-browser-recording-{page.page_id}",
        )
        self.runtime.recording = RecordingState(page.page_id, str(output_path), process, task, time.monotonic())
        return f"Recording started for {page.page_id}: {self.artifacts.display_path(output_path)}"

    async def _capture_loop(self, page: PageState, process: asyncio.subprocess.Process) -> None:
        interval = 1 / max(1, self.runtime.config.recording_fps)
        deadline = time.monotonic() + self.runtime.config.recording_max_seconds
        try:
            while time.monotonic() < deadline and process.returncode is None:
                started = time.monotonic()
                result = await self.runtime.client.call(
                    "Page.captureScreenshot",
                    {"format": "jpeg", "quality": 80, "fromSurface": True},
                    session_id=page.session_id,
                    timeout=max(5.0, self.runtime.config.action_timeout),
                )
                frame = base64.b64decode(result.get("data", ""))
                if process.stdin is None:
                    break
                process.stdin.write(frame)
                await process.stdin.drain()
                await asyncio.sleep(max(0, interval - (time.monotonic() - started)))
        except (asyncio.CancelledError, BrokenPipeError):
            pass
        finally:
            if process.stdin is not None and not process.stdin.is_closing():
                process.stdin.close()
                with suppress(Exception):
                    await process.stdin.wait_closed()
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=20.0)
                except TimeoutError:
                    process.terminate()
                    await process.wait()

    async def stop(self) -> str:
        state, self.runtime.recording = self.runtime.recording, None
        if state is None:
            raise CommandError("No browser recording is active.")
        state.task.cancel()
        with suppress(asyncio.CancelledError):
            await state.task
        stderr = b""
        if state.process.stderr is not None:
            with suppress(Exception):
                stderr = await state.process.stderr.read()
        path = Path(state.path)
        if state.process.returncode not in {0, None}:
            raise CommandError(f"ffmpeg recording failed: {stderr.decode(errors='replace')[-1000:]}")
        duration = time.monotonic() - state.started_at
        size = path.stat().st_size if path.exists() else 0
        return f"Recording saved: {self.artifacts.display_path(path)} ({duration:.1f}s, {size} bytes)"
