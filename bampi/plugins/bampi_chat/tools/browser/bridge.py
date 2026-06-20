from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import os
import signal


_DOCKER_PORT_BRIDGE_SCRIPT = r"""
import os, select, socket, sys
port = int(sys.argv[1])
sock = socket.create_connection(("127.0.0.1", port))
sock.setblocking(False)
stdin_fd, stdout_fd, sock_fd = sys.stdin.fileno(), sys.stdout.fileno(), sock.fileno()
stdin_open = True
while True:
    ready, _, _ = select.select([sock_fd] + ([stdin_fd] if stdin_open else []), [], [])
    if stdin_open and stdin_fd in ready:
        data = os.read(stdin_fd, 65536)
        if not data:
            stdin_open = False
            try: sock.shutdown(socket.SHUT_WR)
            except OSError: pass
        else: sock.sendall(data)
    if sock_fd in ready:
        data = sock.recv(65536)
        if not data: break
        os.write(stdout_fd, data)
sock.close()
""".strip()


@dataclass(slots=True)
class PortBridge:
    target_port: int
    listen_port: int
    server: asyncio.AbstractServer


class LocalhostBridgeManager:
    def __init__(self, container_name: str | None) -> None:
        self._container_name = container_name
        self._bridges: dict[int, PortBridge] = {}
        self._lock = asyncio.Lock()

    async def get(self, target_port: int) -> PortBridge:
        async with self._lock:
            existing = self._bridges.get(target_port)
            if existing is not None:
                return existing
            if not self._container_name:
                raise RuntimeError("Docker localhost bridging is not configured.")
            server = await asyncio.start_server(
                lambda reader, writer: self._handle(target_port, reader, writer),
                "127.0.0.1",
                0,
            )
            sockets = list(server.sockets or [])
            if not sockets:
                server.close()
                await server.wait_closed()
                raise RuntimeError(f"Could not bridge container port {target_port}.")
            bridge = PortBridge(target_port, int(sockets[0].getsockname()[1]), server)
            self._bridges[target_port] = bridge
            return bridge

    async def _handle(
        self,
        target_port: int,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if not self._container_name:
            writer.close()
            return
        process = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", self._container_name, "python3", "-c",
            _DOCKER_PORT_BRIDGE_SCRIPT, str(target_port),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )

        async def upstream() -> None:
            while data := await reader.read(65536):
                if process.stdin is None:
                    return
                process.stdin.write(data)
                await process.stdin.drain()
            if process.stdin is not None:
                process.stdin.close()

        async def downstream() -> None:
            if process.stdout is None:
                return
            while data := await process.stdout.read(65536):
                writer.write(data)
                await writer.drain()

        up = asyncio.create_task(upstream())
        down = asyncio.create_task(downstream())
        try:
            await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (up, down):
                task.cancel()
            if process.returncode is None:
                if os.name == "posix":
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            with suppress(Exception):
                await process.wait()
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def close(self) -> None:
        async with self._lock:
            bridges = list(self._bridges.values())
            self._bridges.clear()
        for bridge in bridges:
            bridge.server.close()
            await bridge.server.wait_closed()
