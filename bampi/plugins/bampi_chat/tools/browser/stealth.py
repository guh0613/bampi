from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import platform as host_platform
import re
import sys
from typing import Any

from .config import BrowserConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Geolocation:
    latitude: float
    longitude: float
    accuracy: float = 35.0


@dataclass(frozen=True, slots=True)
class BrowserIdentity:
    user_agent: str
    brands: tuple[dict[str, str], ...]
    full_version_list: tuple[dict[str, str], ...]
    full_version: str
    platform: str
    ua_platform: str
    ua_platform_version: str
    architecture: str
    bitness: str
    accept_language: str
    languages: tuple[str, ...]
    locale: str
    timezone_id: str
    geolocation: Geolocation | None
    viewport_width: int
    viewport_height: int
    screen_width: int
    screen_height: int
    window_left: int
    window_top: int
    window_width: int
    window_height: int
    device_scale_factor: float

    @property
    def user_agent_metadata(self) -> dict[str, Any]:
        return {
            "brands": list(self.brands),
            "fullVersionList": list(self.full_version_list),
            "fullVersion": self.full_version,
            "platform": self.ua_platform,
            "platformVersion": self.ua_platform_version,
            "architecture": self.architecture,
            "model": "",
            "mobile": False,
            "bitness": self.bitness,
            "wow64": False,
        }

    @property
    def user_agent_override_params(self) -> dict[str, Any]:
        return {
            "userAgent": self.user_agent,
            "acceptLanguage": self.accept_language,
            "platform": self.platform,
            "userAgentMetadata": self.user_agent_metadata,
        }

    @property
    def device_metrics_params(self) -> dict[str, Any]:
        return {
            "width": self.viewport_width,
            "height": self.viewport_height,
            "deviceScaleFactor": self.device_scale_factor,
            "mobile": False,
            "screenWidth": self.screen_width,
            "screenHeight": self.screen_height,
            "positionX": self.window_left,
            "positionY": self.window_top,
            "screenOrientation": {"type": "landscapePrimary", "angle": 0},
        }

    @property
    def window_bounds_params(self) -> dict[str, Any]:
        return {
            "left": self.window_left,
            "top": self.window_top,
            "width": self.window_width,
            "height": self.window_height,
            "windowState": "normal",
        }


@dataclass(frozen=True, slots=True)
class _PlatformProfile:
    ua_comment: str
    platform: str
    ua_platform: str
    ua_platform_version: str
    architecture: str
    bitness: str


_TRACKER_BLOCK_URLS = (
    "*://*.fingerprint.com/*",
    "*://*.fpjs.io/*",
    "*://*.openfpcdn.io/*",
    "*://*.hotjar.com/*",
    "*://*.clarity.ms/*",
    "*://*.fullstory.com/*",
    "*://*.doubleclick.net/*",
    "*://www.google-analytics.com/*",
    "*://ssl.google-analytics.com/*",
    "*://analytics.google.com/*",
    "*://*.googletagmanager.com/gtag/js*",
    "*://connect.facebook.net/*/fbevents.js*",
    "*://*.facebook.com/tr/*",
)

_IMAGE_BLOCK_URLS = (
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.svg",
    "*.avif",
)

_TIMEZONE_GEOLOCATIONS: dict[str, Geolocation] = {
    "Asia/Shanghai": Geolocation(31.2304, 121.4737),
    "Asia/Hong_Kong": Geolocation(22.3193, 114.1694),
    "Asia/Tokyo": Geolocation(35.6762, 139.6503),
    "Asia/Singapore": Geolocation(1.3521, 103.8198),
    "Europe/Berlin": Geolocation(52.52, 13.405),
    "Europe/London": Geolocation(51.5072, -0.1276),
    "America/New_York": Geolocation(40.7128, -74.006),
    "America/Los_Angeles": Geolocation(34.0522, -118.2437),
}

# CSS-pixel desktop resolutions. A screen is only selected when it can contain
# both the configured viewport and a normal desktop browser frame.
_SCREEN_SIZES = (
    (1600, 1200),
    (1920, 1080),
    (1920, 1200),
    (2560, 1440),
    (2880, 1800),
    (3840, 2160),
)
_BROWSER_FRAME_HEIGHT = 80


def build_stealth_identity(
    workspace_dir: Path,
    version_info: dict[str, Any] | None = None,
    *,
    platform_name: str | None = None,
    env: dict[str, str] | None = None,
    viewport_width: int | None = None,
    viewport_height: int | None = None,
) -> BrowserIdentity:
    env_map = env if env is not None else os.environ
    platform = _platform_profile(platform_name or sys.platform)
    full_version = _chrome_version(version_info) or "120.0.0.0"
    major = int(full_version.split(".", 1)[0])
    timezone_id = _timezone_id(env_map)
    locale, languages, accept_language = _locale(env_map, timezone_id)
    viewport_width = viewport_width or 1440
    viewport_height = viewport_height or 1000
    screen_width, screen_height = _screen_size(
        workspace_dir,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    window_width = viewport_width
    window_height = viewport_height + _BROWSER_FRAME_HEIGHT
    window_left = max(0, (screen_width - window_width) // 2)
    window_top = max(0, (screen_height - window_height) // 3)

    return BrowserIdentity(
        # Chrome's low-entropy UA freezes minor/build/patch at 0.0.0. The
        # actual build remains available through high-entropy UA client hints.
        user_agent=_user_agent(platform.ua_comment, major),
        brands=tuple(_brands(major)),
        full_version_list=tuple(_full_version_list(major, full_version)),
        full_version=full_version,
        platform=platform.platform,
        ua_platform=platform.ua_platform,
        ua_platform_version=platform.ua_platform_version,
        architecture=platform.architecture,
        bitness=platform.bitness,
        accept_language=accept_language,
        languages=languages,
        locale=locale,
        timezone_id=timezone_id,
        geolocation=_TIMEZONE_GEOLOCATIONS.get(timezone_id),
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        screen_width=screen_width,
        screen_height=screen_height,
        window_left=window_left,
        window_top=window_top,
        window_width=window_width,
        window_height=window_height,
        device_scale_factor=1.0,
    )


async def apply_window_geometry(client: Any, target_id: str, identity: BrowserIdentity) -> bool:
    if not target_id:
        return False
    try:
        result = await client.call("Browser.getWindowForTarget", {"targetId": target_id})
        window_id = result.get("windowId")
        if not isinstance(window_id, int):
            return False
        await client.call(
            "Browser.setWindowBounds",
            {"windowId": window_id, "bounds": identity.window_bounds_params},
        )
        return True
    except Exception:
        logger.debug("Could not apply browser window geometry for target %s", target_id, exc_info=True)
        return False


async def apply_stealth_to_session(
    client: Any,
    session_id: str,
    identity: BrowserIdentity,
    config: BrowserConfig,
) -> None:
    if config.stealth:
        await _best_effort_call(client, "Emulation.setAutomationOverride", {"enabled": False}, session_id=session_id)
        if not await _best_effort_call(
            client,
            "Emulation.setUserAgentOverride",
            identity.user_agent_override_params,
            session_id=session_id,
        ):
            await _best_effort_call(
                client,
                "Network.setUserAgentOverride",
                identity.user_agent_override_params,
                session_id=session_id,
            )
        await _best_effort_call(client, "Emulation.setLocaleOverride", {"locale": identity.locale}, session_id=session_id)
        await _best_effort_call(
            client,
            "Emulation.setTimezoneOverride",
            {"timezoneId": identity.timezone_id},
            session_id=session_id,
        )
        if identity.geolocation is not None:
            await _best_effort_call(
                client,
                "Emulation.setGeolocationOverride",
                {
                    "latitude": identity.geolocation.latitude,
                    "longitude": identity.geolocation.longitude,
                    "accuracy": identity.geolocation.accuracy,
                },
                session_id=session_id,
            )

    patterns = blocked_url_patterns(config)
    if patterns:
        await _best_effort_call(client, "Network.setBlockedURLs", {"urls": patterns}, session_id=session_id)


def blocked_url_patterns(config: BrowserConfig) -> list[str]:
    patterns: list[str] = []
    if config.stealth:
        patterns.extend(_TRACKER_BLOCK_URLS)
    if config.block_images:
        patterns.extend(_IMAGE_BLOCK_URLS)
    return list(dict.fromkeys(patterns))


def stealth_launch_args(identity: BrowserIdentity) -> list[str]:
    primary_language = identity.languages[0] if identity.languages else "en-US"
    return [
        "--disable-blink-features=AutomationControlled",
        f"--lang={primary_language}",
    ]


async def _best_effort_call(client: Any, method: str, params: dict[str, Any], *, session_id: str) -> bool:
    try:
        await client.call(method, params, session_id=session_id)
        return True
    except Exception:
        logger.debug("CDP stealth method %s failed for session %s", method, session_id, exc_info=True)
        return False


def _chrome_version(version_info: dict[str, Any] | None) -> str | None:
    if not version_info:
        return None
    candidates = [
        str(version_info.get("product") or ""),
        str(version_info.get("userAgent") or ""),
        str(version_info.get("jsVersion") or ""),
    ]
    for candidate in candidates:
        match = re.search(r"(?:HeadlessChrome|Chrome|Chromium)/([0-9]+(?:\.[0-9]+){0,3})", candidate)
        if match:
            return _normalize_version(match.group(1))
    return None


def _normalize_version(raw: str) -> str:
    parts = [part for part in raw.split(".") if part.isdigit()]
    while len(parts) < 4:
        parts.append("0")
    return ".".join(parts[:4])


def _user_agent(comment: str, major: int) -> str:
    return f"Mozilla/5.0 ({comment}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"


def _platform_profile(platform_name: str) -> _PlatformProfile:
    architecture = _machine_architecture()
    if platform_name == "darwin":
        return _PlatformProfile(
            ua_comment="Macintosh; Intel Mac OS X 10_15_7",
            platform="MacIntel",
            ua_platform="macOS",
            ua_platform_version=_host_platform_version(platform_name, fallback="15.0.0"),
            architecture=architecture,
            bitness="64",
        )
    if platform_name.startswith("win"):
        return _PlatformProfile(
            ua_comment="Windows NT 10.0; Win64; x64",
            platform="Win32",
            ua_platform="Windows",
            ua_platform_version="15.0.0",
            architecture=architecture,
            bitness="64",
        )
    return _PlatformProfile(
        ua_comment="X11; Linux x86_64",
        platform="Linux x86_64",
        ua_platform="Linux",
        ua_platform_version=_host_platform_version(platform_name, fallback="6.6.0"),
        architecture=architecture,
        bitness="64",
    )


def _machine_architecture() -> str:
    machine = host_platform.machine().lower()
    return "arm" if "arm" in machine or "aarch64" in machine else "x86"


def _host_platform_version(platform_name: str, *, fallback: str) -> str:
    raw = ""
    if platform_name == sys.platform:
        if platform_name == "darwin":
            raw = host_platform.mac_ver()[0]
        elif platform_name.startswith("linux"):
            raw = host_platform.release()
    parts = re.findall(r"\d+", raw)[:3]
    if not parts:
        return fallback
    while len(parts) < 3:
        parts.append("0")
    return ".".join(parts)


def _brands(major: int) -> list[dict[str, str]]:
    grease_chars = [" ", "(", ":", "-", ".", "/", ")", ";", "=", "?", "_"]
    grease_versions = ["8", "99", "24"]
    permutations = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    ordered = [
        {
            "brand": "Not" + grease_chars[major % len(grease_chars)] + "A" + grease_chars[(major + 1) % len(grease_chars)] + "Brand",
            "version": grease_versions[major % len(grease_versions)],
        },
        {"brand": "Chromium", "version": str(major)},
        {"brand": "Google Chrome", "version": str(major)},
    ]
    return [ordered[index] for index in permutations[major % len(permutations)]]


def _full_version_list(major: int, full_version: str) -> list[dict[str, str]]:
    return [
        {**brand, "version": full_version if brand["brand"] in {"Chromium", "Google Chrome"} else f"{brand['version']}.0.0.0"}
        for brand in _brands(major)
    ]


def _timezone_id(env: dict[str, str]) -> str:
    raw = (env.get("BAMPI_BROWSER_TIMEZONE") or env.get("TZ") or "").strip()
    if _looks_like_iana_timezone(raw):
        return raw
    for path in (Path("/etc/localtime"), Path("/var/db/timezone/localtime")):
        try:
            target = path.resolve(strict=True)
        except OSError:
            continue
        text = target.as_posix()
        marker = "/zoneinfo/"
        if marker in text:
            candidate = text.split(marker, 1)[1]
            if _looks_like_iana_timezone(candidate):
                return candidate
    return "Asia/Shanghai"


def _looks_like_iana_timezone(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_]+/[A-Za-z0-9_+\-/]+", value))


def _locale(env: dict[str, str], timezone_id: str) -> tuple[str, tuple[str, ...], str]:
    raw = (
        env.get("BAMPI_BROWSER_LOCALE")
        or env.get("LC_ALL")
        or env.get("LC_MESSAGES")
        or env.get("LANG")
        or ""
    ).strip()
    normalized = raw.split(".", 1)[0].split("@", 1)[0]
    if not re.fullmatch(r"[A-Za-z]{2,3}[_-][A-Za-z]{2}", normalized):
        normalized = "zh_CN" if timezone_id.startswith("Asia/Shanghai") else "en_US"
    locale = normalized.replace("-", "_")
    primary = locale.replace("_", "-")
    language = primary.split("-", 1)[0]
    languages = (primary, language) if language and language != primary else (primary,)
    accept_language = ",".join([languages[0], *[f"{item};q={0.9 - index * 0.1:.1f}" for index, item in enumerate(languages[1:])]])
    return locale, languages, accept_language


def _screen_size(workspace_dir: Path, *, viewport_width: int, viewport_height: int) -> tuple[int, int]:
    minimum_height = viewport_height + _BROWSER_FRAME_HEIGHT
    eligible = [
        size
        for size in _SCREEN_SIZES
        if size[0] >= viewport_width and size[1] >= minimum_height
    ]
    if not eligible:
        return _round_up(viewport_width + 160, 160), _round_up(minimum_height + 80, 90)
    return eligible[_stable_int(workspace_dir, "screen", len(eligible))]


def _round_up(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def _stable_int(workspace_dir: Path, label: str, modulo: int) -> int:
    digest = hashlib.sha256(f"{workspace_dir.resolve()}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulo
