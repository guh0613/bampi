from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from .cdp import CdpClient
from .config import BrowserConfig
from .errors import BrowserLaunchError
from .installer import default_cache_dir, ensure_chrome_for_testing, find_cached_chrome


@dataclass(slots=True)
class LaunchedChromium:
    process: asyncio.subprocess.Process
    client: CdpClient
    profile_dir: Path
    executable: str

    async def close(self) -> None:
        if not self.client.closed:
            with suppress(Exception):
                await self.client.call("Browser.close", timeout=3.0)
        await self.client.close()
        if self.process.returncode is None:
            try:
                await asyncio.wait_for(self.process.wait(), timeout=4.0)
            except TimeoutError:
                _terminate_process_group(self.process, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=3.0)
                except TimeoutError:
                    _terminate_process_group(self.process, signal.SIGKILL)
                    await self.process.wait()


def _terminate_process_group(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
    elif sig == signal.SIGKILL:
        process.kill()
    else:
        process.terminate()


def find_chromium(explicit: str | None = None) -> str | None:
    candidates: list[str] = []
    if explicit:
        expanded = str(Path(explicit).expanduser())
        if Path(expanded).is_file() and os.access(expanded, os.X_OK):
            return expanded
        raise BrowserLaunchError(f"Configured Chromium executable is not usable: {expanded}")
    env_path = os.environ.get("BAMPI_BROWSER_EXECUTABLE")
    if env_path:
        expanded = str(Path(env_path).expanduser())
        if Path(expanded).is_file() and os.access(expanded, os.X_OK):
            return expanded
        raise BrowserLaunchError(f"BAMPI_BROWSER_EXECUTABLE is not usable: {expanded}")
    if sys.platform == "darwin":
        candidates.extend(
            [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
                str(Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            ]
        )
    elif sys.platform.startswith("win"):
        for root in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")):
            if root:
                candidates.extend(
                    [
                        str(Path(root) / "Google/Chrome/Application/chrome.exe"),
                        str(Path(root) / "Chromium/Application/chrome.exe"),
                        str(Path(root) / "Microsoft/Edge/Application/msedge.exe"),
                    ]
                )
    else:
        candidates.extend(["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge"])

    for candidate in candidates:
        expanded = str(Path(candidate).expanduser())
        if Path(expanded).is_file() and os.access(expanded, os.X_OK):
            return expanded
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


async def resolve_chromium(config: BrowserConfig) -> str:
    system = find_chromium(config.executable_path)
    if system is not None:
        return system
    cache_dir = Path(config.cache_dir).expanduser() if config.cache_dir else default_cache_dir()
    cached = find_cached_chrome(cache_dir)
    if cached is not None:
        return str(cached)
    if config.auto_install:
        return str(await ensure_chrome_for_testing(cache_dir, timeout=config.install_timeout))
    raise BrowserLaunchError(
        "No Chromium browser was found and automatic Chrome for Testing installation is disabled. "
        "Install Chrome/Chromium or configure bampi_browser_executable_path."
    )


async def launch_chromium(workspace_dir: Path, config: BrowserConfig) -> LaunchedChromium:
    executable = await resolve_chromium(config)
    profile_dir = workspace_dir / ".browser" / "chromium-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    port_file = profile_dir / "DevToolsActivePort"
    with suppress(OSError):
        port_file.unlink()

    args = [
        executable,
        f"--user-data-dir={profile_dir}",
        "--remote-debugging-port=0",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-features=Translate,MediaRouter,OptimizationHints,AutofillServerCommunication",
        "--disable-popup-blocking",
        "--disable-prompt-on-repost",
        "--disable-sync",
        "--metrics-recording-only",
        "--password-store=basic",
        "--use-mock-keychain",
        f"--window-size={config.viewport_width},{config.viewport_height}",
        "about:blank",
    ]
    if config.headless:
        args.insert(1, "--headless=new")

    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise BrowserLaunchError(f"Failed to start Chromium at {executable}: {exc}") from exc

    deadline = time.monotonic() + config.launch_timeout
    websocket_url: str | None = None
    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise BrowserLaunchError(f"Chromium exited during startup with code {process.returncode}.")
        if port_file.is_file():
            try:
                lines = port_file.read_text(encoding="utf-8").splitlines()
                if len(lines) >= 2:
                    websocket_url = f"ws://127.0.0.1:{int(lines[0])}{lines[1]}"
                    break
            except (OSError, ValueError):
                pass
        await asyncio.sleep(0.05)
    if websocket_url is None:
        _terminate_process_group(process, signal.SIGTERM)
        with suppress(Exception):
            await process.wait()
        raise BrowserLaunchError(f"Chromium did not expose DevTools within {config.launch_timeout:g}s.")

    try:
        client = await CdpClient.connect(websocket_url, timeout=config.launch_timeout)
    except Exception as exc:
        _terminate_process_group(process, signal.SIGTERM)
        with suppress(Exception):
            await process.wait()
        raise BrowserLaunchError(f"Could not connect to Chromium DevTools: {exc}") from exc
    return LaunchedChromium(process=process, client=client, profile_dir=profile_dir, executable=executable)
