from __future__ import annotations

import asyncio
import ipaddress
from pathlib import Path, PurePosixPath
import socket
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from .bridge import LocalhostBridgeManager
from .config import BrowserConfig
from .errors import CommandError


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_PROXY_FAKE_IP_NETWORKS = (ipaddress.ip_network("198.18.0.0/15"),)
_METADATA_HOSTS = {"metadata.google.internal", "metadata", "instance-data"}


class NavigationPolicy:
    def __init__(
        self,
        workspace_dir: Path,
        *,
        container_root: str | None,
        bridge: LocalhostBridgeManager,
        bridge_localhost: bool,
        config: BrowserConfig,
    ) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.container_root = PurePosixPath(container_root) if container_root else None
        self.bridge = bridge
        self.bridge_localhost = bridge_localhost
        self.config = config

    async def resolve(self, raw_url: str) -> tuple[str, list[str]]:
        raw_url = raw_url.strip()
        if not raw_url:
            raise CommandError("A URL is required.")
        if "://" not in raw_url and not raw_url.startswith(("about:", "file:")):
            raw_url = "https://" + raw_url
        parsed = urlsplit(raw_url)
        if parsed.scheme == "file":
            return self._resolve_file(parsed)
        if parsed.scheme == "about" and raw_url == "about:blank":
            return raw_url, []
        if parsed.scheme not in {"http", "https"}:
            raise CommandError(f"Navigation scheme '{parsed.scheme or '<missing>'}' is not allowed.")
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise CommandError("The URL has no hostname.")
        if hostname in _LOCAL_HOSTS:
            if self.bridge_localhost:
                target_port = parsed.port or (443 if parsed.scheme == "https" else 80)
                bridge = await self.bridge.get(target_port)
                resolved = urlunsplit(
                    (parsed.scheme, f"{hostname}:{bridge.listen_port}", parsed.path, parsed.query, parsed.fragment)
                )
                return resolved, [f"container localhost:{target_port} bridged to host port {bridge.listen_port}"]
            return raw_url, []
        if not self.config.allow_private_network:
            await self._reject_private_host(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        return raw_url, []

    def display_url(self, raw_url: str) -> str:
        parsed = urlsplit(raw_url)
        if parsed.scheme != "file":
            return raw_url
        try:
            path = Path(unquote(parsed.path)).resolve()
            relative = path.relative_to(self.workspace_dir)
        except (ValueError, OSError):
            return raw_url
        if self.container_root:
            visible = self.container_root / PurePosixPath(relative.as_posix())
            return f"file://{quote(visible.as_posix(), safe='/')}"
        return raw_url

    def _resolve_file(self, parsed) -> tuple[str, list[str]]:
        if parsed.netloc not in {"", "localhost"}:
            relative = PurePosixPath(parsed.netloc, unquote(parsed.path).lstrip("/"))
            candidate = self.workspace_dir / relative.as_posix()
        else:
            raw_path = unquote(parsed.path or "")
            if self.container_root:
                visible = PurePosixPath(raw_path)
                try:
                    relative = visible.relative_to(self.container_root)
                except ValueError:
                    candidate = Path(raw_path)
                else:
                    candidate = self.workspace_dir / relative.as_posix()
            else:
                candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = self.workspace_dir / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.workspace_dir)
        except FileNotFoundError as exc:
            raise CommandError(f"Workspace file does not exist: {candidate}") from exc
        except ValueError as exc:
            raise CommandError("file:// navigation is restricted to the group workspace.") from exc
        return resolved.as_uri(), [f"workspace file: {resolved.relative_to(self.workspace_dir).as_posix()}"]

    async def _reject_private_host(self, hostname: str, port: int) -> None:
        if hostname.rstrip(".") in _METADATA_HOSTS:
            raise CommandError(f"Navigation to metadata host {hostname} is blocked.")
        try:
            direct = ipaddress.ip_address(hostname)
        except ValueError:
            direct_address = False
            loop = asyncio.get_running_loop()
            try:
                infos = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            except socket.gaierror as exc:
                raise CommandError(f"Could not resolve hostname {hostname}: {exc}") from exc
            addresses = {item[4][0] for item in infos}
        else:
            direct_address = True
            addresses = {str(direct)}
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not direct_address and any(ip in network for network in _PROXY_FAKE_IP_NETWORKS):
                # Clash and similar local proxies intentionally return RFC 2544
                # benchmark addresses for public hostnames in fake-IP mode.
                continue
            if ip.is_private or ip.is_link_local or ip.is_loopback or ip.is_reserved or ip.is_unspecified:
                raise CommandError(
                    f"Navigation to private or special-use address {address} is blocked. "
                    "Use localhost for the project sandbox or enable private-network access explicitly."
                )
