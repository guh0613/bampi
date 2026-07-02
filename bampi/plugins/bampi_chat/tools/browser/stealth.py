from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from .config import BrowserConfig


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
    hardware_concurrency: int
    device_memory: int
    webgl_vendor: str
    webgl_renderer: str
    viewport_width: int
    viewport_height: int
    screen_width: int
    screen_height: int
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
            "screenOrientation": {"type": "landscapePrimary", "angle": 0},
        }


@dataclass(frozen=True, slots=True)
class _PlatformProfile:
    ua_comment: str
    platform: str
    ua_platform: str
    ua_platform_version: str
    architecture: str
    bitness: str
    webgl_vendor: str
    webgl_renderer: str


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
    user_agent = _user_agent(platform.ua_comment, full_version)
    timezone_id = _timezone_id(env_map)
    locale, languages, accept_language = _locale(env_map, timezone_id)
    screen_width, screen_height = _screen_size(workspace_dir)
    viewport_width = viewport_width or 1440
    viewport_height = viewport_height or 1000
    screen_width = max(screen_width, viewport_width)
    screen_height = max(screen_height, viewport_height)
    hardware_concurrency, device_memory = _machine_shape(workspace_dir)

    return BrowserIdentity(
        user_agent=user_agent,
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
        hardware_concurrency=hardware_concurrency,
        device_memory=device_memory,
        webgl_vendor=platform.webgl_vendor,
        webgl_renderer=platform.webgl_renderer,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        screen_width=screen_width,
        screen_height=screen_height,
        device_scale_factor=2.0 if screen_width >= 2560 and platform.ua_platform == "macOS" else 1.0,
    )


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
        await _install_preload(client, session_id, identity)

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


async def _install_preload(client: Any, session_id: str, identity: BrowserIdentity) -> None:
    source = preload_script(identity)
    params = {"source": source, "runImmediately": True}
    if await _best_effort_call(client, "Page.addScriptToEvaluateOnNewDocument", params, session_id=session_id):
        await _best_effort_call(
            client,
            "Runtime.evaluate",
            {"expression": source, "awaitPromise": False, "returnByValue": True},
            session_id=session_id,
        )
        return
    await _best_effort_call(
        client,
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": source},
        session_id=session_id,
    )
    await _best_effort_call(
        client,
        "Runtime.evaluate",
        {"expression": source, "awaitPromise": False, "returnByValue": True},
        session_id=session_id,
    )


async def _best_effort_call(client: Any, method: str, params: dict[str, Any], *, session_id: str) -> bool:
    try:
        await client.call(method, params, session_id=session_id)
        return True
    except Exception:
        return False


def preload_script(identity: BrowserIdentity) -> str:
    payload = json.dumps(
        {
            "userAgent": identity.user_agent,
            "platform": identity.platform,
            "brands": list(identity.brands),
            "fullVersionList": list(identity.full_version_list),
            "fullVersion": identity.full_version,
            "uaPlatform": identity.ua_platform,
            "uaPlatformVersion": identity.ua_platform_version,
            "architecture": identity.architecture,
            "bitness": identity.bitness,
            "languages": list(identity.languages),
            "hardwareConcurrency": identity.hardware_concurrency,
            "deviceMemory": identity.device_memory,
            "webglVendor": identity.webgl_vendor,
            "webglRenderer": identity.webgl_renderer,
            "screenWidth": identity.screen_width,
            "screenHeight": identity.screen_height,
            "deviceScaleFactor": identity.device_scale_factor,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"""
(() => {{
  const identity = {payload};
  if (globalThis.__bampi_stealth_applied__) return;
  Object.defineProperty(globalThis, "__bampi_stealth_applied__", {{value: true, configurable: false}});

  const nativeToString = Function.prototype.toString;
  const nativeStrings = new WeakMap();
  const markNative = (fn, name) => {{
    try {{ nativeStrings.set(fn, `function ${{name || fn.name || ""}}() {{ [native code] }}`); }} catch (_) {{}}
    return fn;
  }};
  const stealthToString = new Proxy(nativeToString, {{
    apply(target, thisArg, args) {{
      if (nativeStrings.has(thisArg)) return nativeStrings.get(thisArg);
      return Reflect.apply(target, thisArg, args);
    }}
  }});
  nativeStrings.set(stealthToString, "function toString() {{ [native code] }}");
  try {{ Object.defineProperty(Function.prototype, "toString", {{value: stealthToString, configurable: true, writable: true}}); }} catch (_) {{}}

  const defineGetter = (target, prop, value, name) => {{
    if (!target) return;
    const getter = markNative(function() {{ return typeof value === "function" ? value.call(this) : value; }}, name || `get ${{prop}}`);
    try {{ Object.defineProperty(target, prop, {{get: getter, configurable: true}}); }} catch (_) {{}}
  }};

  defineGetter(Navigator.prototype, "webdriver", false, "get webdriver");
  defineGetter(Navigator.prototype, "userAgent", identity.userAgent, "get userAgent");
  defineGetter(
    Navigator.prototype,
    "appVersion",
    identity.userAgent.replace(/^Mozilla\\//, ""),
    "get appVersion"
  );
  defineGetter(Navigator.prototype, "platform", identity.platform, "get platform");
  defineGetter(Navigator.prototype, "language", identity.languages[0] || "en-US", "get language");
  defineGetter(Navigator.prototype, "languages", () => Object.freeze([...identity.languages]), "get languages");
  defineGetter(Navigator.prototype, "hardwareConcurrency", identity.hardwareConcurrency, "get hardwareConcurrency");
  defineGetter(Navigator.prototype, "deviceMemory", identity.deviceMemory, "get deviceMemory");
  defineGetter(Navigator.prototype, "maxTouchPoints", 0, "get maxTouchPoints");

  if (!navigator.userAgentData) {{
    const uaData = {{
      brands: Object.freeze(identity.brands.map((brand) => Object.freeze({{...brand}}))),
      mobile: false,
      platform: identity.uaPlatform,
      getHighEntropyValues: markNative(function(hints) {{
        const values = {{
          architecture: identity.architecture,
          bitness: identity.bitness,
          brands: identity.brands.map((brand) => ({{...brand}})),
          fullVersionList: identity.fullVersionList.map((brand) => ({{...brand}})),
          mobile: false,
          model: "",
          platform: identity.uaPlatform,
          platformVersion: identity.uaPlatformVersion,
          uaFullVersion: identity.fullVersion,
          wow64: false
        }};
        const requested = Array.isArray(hints) ? hints : [];
        const result = {{}};
        for (const key of requested) if (key in values) result[key] = values[key];
        result.brands = values.brands;
        result.mobile = false;
        result.platform = identity.uaPlatform;
        return Promise.resolve(result);
      }}, "getHighEntropyValues"),
      toJSON: markNative(function() {{ return {{brands: this.brands, mobile: this.mobile, platform: this.platform}}; }}, "toJSON")
    }};
    defineGetter(Navigator.prototype, "userAgentData", () => uaData, "get userAgentData");
  }}

  if (!globalThis.chrome) {{
    try {{ Object.defineProperty(globalThis, "chrome", {{value: {{}}, configurable: true}}); }} catch (_) {{}}
  }}
  if (globalThis.chrome && !globalThis.chrome.runtime) {{
    try {{
      Object.defineProperty(globalThis.chrome, "runtime", {{
        value: {{
          PlatformOs: {{MAC: "mac", WIN: "win", ANDROID: "android", CROS: "cros", LINUX: "linux", OPENBSD: "openbsd"}},
          PlatformArch: {{ARM: "arm", ARM64: "arm64", X86_32: "x86-32", X86_64: "x86-64"}},
          PlatformNaclArch: {{ARM: "arm", X86_32: "x86-32", X86_64: "x86-64"}},
          RequestUpdateCheckStatus: {{THROTTLED: "throttled", NO_UPDATE: "no_update", UPDATE_AVAILABLE: "update_available"}},
          OnInstalledReason: {{INSTALL: "install", UPDATE: "update", CHROME_UPDATE: "chrome_update", SHARED_MODULE_UPDATE: "shared_module_update"}},
          OnRestartRequiredReason: {{APP_UPDATE: "app_update", OS_UPDATE: "os_update", PERIODIC: "periodic"}},
          connect: markNative(function() {{ throw new TypeError("Error in invocation of runtime.connect"); }}, "connect"),
          sendMessage: markNative(function() {{ throw new TypeError("Error in invocation of runtime.sendMessage"); }}, "sendMessage")
        }},
        configurable: true
      }});
    }} catch (_) {{}}
  }}

  if (navigator.permissions && navigator.permissions.query) {{
    const originalQuery = navigator.permissions.query.bind(navigator.permissions);
    const patchedQuery = markNative(function(parameters) {{
      const name = parameters && parameters.name;
      if (name === "notifications") {{
        return Promise.resolve({{state: Notification.permission === "denied" ? "denied" : "prompt", onchange: null}});
      }}
      if (name === "geolocation" || name === "camera" || name === "microphone" || name === "midi") {{
        return Promise.resolve({{state: "prompt", onchange: null}});
      }}
      return originalQuery(parameters);
    }}, "query");
    try {{ Object.defineProperty(navigator.permissions, "query", {{value: patchedQuery, configurable: true, writable: true}}); }} catch (_) {{}}
  }}

  const patchWebGL = (proto) => {{
    if (!proto || !proto.getParameter) return;
    const originalGetParameter = proto.getParameter;
    const patchedGetParameter = markNative(function(parameter) {{
      if (parameter === 37445) return identity.webglVendor;
      if (parameter === 37446) return identity.webglRenderer;
      if (parameter === 7936) return "WebKit";
      if (parameter === 7937) return "WebKit WebGL";
      return originalGetParameter.apply(this, arguments);
    }}, "getParameter");
    try {{ Object.defineProperty(proto, "getParameter", {{value: patchedGetParameter, configurable: true, writable: true}}); }} catch (_) {{}}
  }};
  patchWebGL(globalThis.WebGLRenderingContext && WebGLRenderingContext.prototype);
  patchWebGL(globalThis.WebGL2RenderingContext && WebGL2RenderingContext.prototype);

  const outerWidth = Math.max(identity.screenWidth, innerWidth || identity.screenWidth);
  const outerHeight = Math.max(identity.screenHeight, (innerHeight || identity.screenHeight) + 80);
  if (!globalThis.outerWidth || globalThis.outerWidth <= 0) defineGetter(globalThis, "outerWidth", outerWidth, "get outerWidth");
  if (!globalThis.outerHeight || globalThis.outerHeight <= 0) defineGetter(globalThis, "outerHeight", outerHeight, "get outerHeight");
  if (globalThis.devicePixelRatio <= 0) defineGetter(globalThis, "devicePixelRatio", identity.deviceScaleFactor, "get devicePixelRatio");
}})();
""".strip()


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


def _user_agent(comment: str, full_version: str) -> str:
    return f"Mozilla/5.0 ({comment}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{full_version} Safari/537.36"


def _platform_profile(platform_name: str) -> _PlatformProfile:
    if platform_name == "darwin":
        return _PlatformProfile(
            ua_comment="Macintosh; Intel Mac OS X 10_15_7",
            platform="MacIntel",
            ua_platform="macOS",
            ua_platform_version="15.0.0",
            architecture="arm" if _machine_is_arm() else "x86",
            bitness="64",
            webgl_vendor="Google Inc. (Apple)",
            webgl_renderer="ANGLE (Apple, ANGLE Metal Renderer: Apple M2, Unspecified Version)",
        )
    if platform_name.startswith("win"):
        return _PlatformProfile(
            ua_comment="Windows NT 10.0; Win64; x64",
            platform="Win32",
            ua_platform="Windows",
            ua_platform_version="15.0.0",
            architecture="x86",
            bitness="64",
            webgl_vendor="Google Inc. (NVIDIA)",
            webgl_renderer="ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0, D3D11)",
        )
    return _PlatformProfile(
        ua_comment="X11; Linux x86_64",
        platform="Linux x86_64",
        ua_platform="Linux",
        ua_platform_version="6.6.0",
        architecture="x86",
        bitness="64",
        webgl_vendor="Google Inc. (Intel)",
        webgl_renderer="ANGLE (Intel, Mesa Intel(R) UHD Graphics 620, OpenGL 4.6)",
    )


def _machine_is_arm() -> bool:
    machine = os.uname().machine.lower() if hasattr(os, "uname") else ""
    return "arm" in machine or "aarch64" in machine


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


def _screen_size(workspace_dir: Path) -> tuple[int, int]:
    choices = ((1366, 768), (1440, 900), (1536, 864), (1600, 900), (1920, 1080), (2560, 1440))
    return choices[_stable_int(workspace_dir, "screen", len(choices))]


def _machine_shape(workspace_dir: Path) -> tuple[int, int]:
    cpu_choices = (4, 6, 8, 12, 16)
    memory_choices = (4, 8)
    cpu = cpu_choices[_stable_int(workspace_dir, "cpu", len(cpu_choices))]
    memory = memory_choices[_stable_int(workspace_dir, "memory", len(memory_choices))]
    return cpu, memory


def _stable_int(workspace_dir: Path, label: str, modulo: int) -> int:
    digest = hashlib.sha256(f"{workspace_dir.resolve()}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulo
