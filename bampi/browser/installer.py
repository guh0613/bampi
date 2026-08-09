from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import sys
import tempfile
import time
from typing import Any
import zipfile

import aiohttp

from .errors import BrowserLaunchError


LAST_KNOWN_GOOD_URL = (
    "https://googlechromelabs.github.io/chrome-for-testing/"
    "last-known-good-versions-with-downloads.json"
)
_MAX_ARCHIVE_BYTES = 600 * 1024 * 1024
_install_lock = asyncio.Lock()


def default_cache_dir() -> Path:
    configured = os.environ.get("BAMPI_BROWSER_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / ".bampi" / "browser" / "chrome-for-testing"


def platform_key() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "mac-arm64"
    if sys.platform == "darwin" and machine in {"x86_64", "amd64"}:
        return "mac-x64"
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}:
        return "linux64"
    if sys.platform.startswith("win") and machine in {"x86_64", "amd64"}:
        return "win64"
    raise BrowserLaunchError(
        f"Chrome for Testing does not publish a build for {sys.platform}/{machine}. "
        "Install Chromium with the system package manager or configure bampi_browser_executable_path."
    )


def _version_key(path: Path) -> tuple[int, ...]:
    raw = path.name.removeprefix("chrome-")
    try:
        return tuple(int(part) for part in raw.split("."))
    except ValueError:
        return ()


def chrome_binary_in(directory: Path, key: str | None = None) -> Path | None:
    key = key or platform_key()
    candidates: dict[str, tuple[str, ...]] = {
        "mac-arm64": (
            "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
            "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        ),
        "mac-x64": (
            "chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
            "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        ),
        "linux64": ("chrome-linux64/chrome", "chrome"),
        "win64": ("chrome-win64/chrome.exe", "chrome.exe"),
    }
    for relative in candidates.get(key, ()):
        candidate = directory / relative
        if candidate.is_file():
            return candidate
    return None


def find_cached_chrome(cache_dir: Path) -> Path | None:
    if not cache_dir.is_dir():
        return None
    versions = sorted(
        (path for path in cache_dir.iterdir() if path.is_dir() and path.name.startswith("chrome-")),
        key=_version_key,
        reverse=True,
    )
    for directory in versions:
        with suppress(BrowserLaunchError):
            binary = chrome_binary_in(directory)
            if binary is not None and os.access(binary, os.X_OK):
                return binary
    return None


def _select_download(metadata: dict[str, Any], key: str) -> tuple[str, str]:
    stable = metadata.get("channels", {}).get("Stable", {})
    version = stable.get("version")
    downloads = stable.get("downloads", {}).get("chrome", [])
    if not isinstance(version, str):
        raise BrowserLaunchError("Chrome for Testing metadata has no Stable version.")
    for entry in downloads:
        if isinstance(entry, dict) and entry.get("platform") == key and isinstance(entry.get("url"), str):
            return version, entry["url"]
    raise BrowserLaunchError(f"Chrome for Testing metadata has no Chrome download for {key}.")


async def ensure_chrome_for_testing(cache_dir: Path, *, timeout: float) -> Path:
    cache_dir = cache_dir.expanduser().resolve()
    cached = find_cached_chrome(cache_dir)
    if cached is not None:
        return cached
    async with _install_lock:
        cached = find_cached_chrome(cache_dir)
        if cached is not None:
            return cached
        key = platform_key()
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = cache_dir / ".install.lock"
        await _acquire_file_lock(lock_path, timeout=timeout)
        try:
            cached = find_cached_chrome(cache_dir)
            if cached is not None:
                return cached
            return await _install_download(cache_dir, key=key, timeout=timeout)
        finally:
            with suppress(OSError):
                lock_path.unlink()


async def _install_download(cache_dir: Path, *, key: str, timeout: float) -> Path:
    client_timeout = aiohttp.ClientTimeout(total=timeout, connect=min(30.0, timeout))
    try:
        async with aiohttp.ClientSession(timeout=client_timeout, trust_env=True, headers={"User-Agent": "bampi-browser/0.1"}) as http:
            async with http.get(LAST_KNOWN_GOOD_URL) as response:
                response.raise_for_status()
                metadata = await response.json(content_type=None)
            version, url = _select_download(metadata, key)
            destination = cache_dir / f"chrome-{version}"
            existing = chrome_binary_in(destination, key)
            if existing is not None:
                return existing
            temporary = Path(tempfile.mkdtemp(prefix=".chrome-install-", dir=cache_dir))
            archive = temporary / "chrome.zip"
            extracted = temporary / "extracted"
            try:
                await _download(http, url, archive)
                await asyncio.to_thread(_extract_zip, archive, extracted)
                binary = chrome_binary_in(extracted, key)
                if binary is None:
                    raise BrowserLaunchError("Downloaded Chrome archive did not contain the expected executable.")
                if os.name == "posix":
                    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                if destination.exists():
                    shutil.rmtree(destination)
                extracted.rename(destination)
                installed = chrome_binary_in(destination, key)
                if installed is None:
                    raise BrowserLaunchError("Chrome installation completed without an executable.")
                (destination / "bampi-install.json").write_text(
                    json.dumps({"version": version, "platform": key, "source": url}, indent=2),
                    encoding="utf-8",
                )
                return installed
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
    except BrowserLaunchError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise BrowserLaunchError(
            f"Could not install Chrome for Testing in {cache_dir}: {exc}. "
            "Install Chrome/Chromium manually or configure bampi_browser_executable_path."
        ) from exc


async def _acquire_file_lock(lock_path: Path, *, timeout: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    stale_after = max(timeout * 2, 900.0)
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            with suppress(OSError):
                if asyncio.get_running_loop().time() < deadline and time.time() - lock_path.stat().st_mtime > stale_after:
                    lock_path.unlink()
                    continue
            if asyncio.get_running_loop().time() >= deadline:
                raise BrowserLaunchError(f"Timed out waiting for Chrome installation lock: {lock_path}")
            await asyncio.sleep(0.5)
            continue
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
        finally:
            os.close(descriptor)
        return


async def _download(http: aiohttp.ClientSession, url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(3):
        if attempt:
            await asyncio.sleep(2 ** attempt)
        try:
            async with http.get(url) as response:
                response.raise_for_status()
                content_length = response.content_length
                if content_length is not None and content_length > _MAX_ARCHIVE_BYTES:
                    raise BrowserLaunchError("Chrome for Testing archive exceeds the 600 MiB safety limit.")
                downloaded = 0
                with destination.open("wb") as output:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > _MAX_ARCHIVE_BYTES:
                            raise BrowserLaunchError("Chrome for Testing archive exceeds the 600 MiB safety limit.")
                        output.write(chunk)
                return
        except BrowserLaunchError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_error = exc
            with suppress(OSError):
                destination.unlink()
    assert last_error is not None
    raise last_error


def _extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise BrowserLaunchError(f"Unsafe path in Chrome archive: {info.filename}")
            output = destination / relative
            if not output.resolve(strict=False).is_relative_to(root):
                raise BrowserLaunchError(f"Unsafe path in Chrome archive: {info.filename}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(info)
            if stat.S_ISLNK(mode):
                link_target = payload.decode("utf-8")
                resolved_target = (output.parent / link_target).resolve(strict=False)
                if not resolved_target.is_relative_to(root):
                    raise BrowserLaunchError(f"Unsafe symlink in Chrome archive: {info.filename}")
                output.symlink_to(link_target)
                continue
            output.write_bytes(payload)
            if os.name == "posix" and mode:
                output.chmod(mode & 0o777)
