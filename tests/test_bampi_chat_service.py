from __future__ import annotations

from pathlib import Path

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.prompt import build_system_prompt
from bampi.plugins.bampi_chat.service_manager import ServiceManager, parse_service_port_range
from bampi.plugins.bampi_chat.tools import create_agent_tools
from bampi.plugins.bampi_chat.tools.workspace import resolve_group_workspace_dir


def test_parse_service_port_range_parses_ranges_and_deduplicates():
    assert parse_service_port_range("46000-46002,46001,47000") == [46000, 46001, 46002, 47000]


def test_system_prompt_mentions_service_tool():
    prompt = build_system_prompt(BampiChatConfig(), ["service"])

    assert "service" in prompt
    assert "PORT" in prompt
    assert "46000-46031" in prompt


def test_create_agent_tools_includes_service_when_manager_is_provided(tmp_path: Path):
    manager = ServiceManager(
        workspace_root=str(tmp_path / "workspace-root"),
        visible_container_root="/workspace",
        container_name="bampi-sandbox",
        container_shell="/bin/bash",
        port_range="46000-46003",
        public_host="127.0.0.1",
        startup_timeout=20.0,
        stop_timeout=10.0,
        max_active_services_per_group=4,
    )

    tools = create_agent_tools(
        BampiChatConfig(),
        str(tmp_path / "group-workspace"),
        container_root="/workspace",
        bash_workdir="/workspace/group-1001",
        group_id="1001",
        service_manager=manager,
    )

    assert "service" in [tool.name for tool in tools]


@pytest.mark.asyncio
async def test_service_manager_start_stop_and_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace_root = tmp_path / "workspace-root"
    group_workspace = resolve_group_workspace_dir(str(workspace_root), "1001")
    (group_workspace / "app").mkdir(parents=True, exist_ok=True)

    manager = ServiceManager(
        workspace_root=str(workspace_root),
        visible_container_root="/workspace",
        container_name="bampi-sandbox",
        container_shell="/bin/bash",
        port_range="46000-46001",
        public_host="play.example.com",
        startup_timeout=20.0,
        stop_timeout=10.0,
        max_active_services_per_group=4,
    )

    running_pids: set[int] = set()

    async def fake_launch_service_process(*, actual_cwd: str, runtime_dir: Path, command: str, port: int, env: dict[str, str]) -> None:
        assert actual_cwd == "/workspace/group-1001/app"
        assert command == "python -m http.server \"$PORT\" --bind 0.0.0.0"
        assert port == 46000
        assert env == {"APP_ENV": "test"}
        (runtime_dir / "service.pid").write_text("321", encoding="utf-8")
        (runtime_dir / "service.log").write_text("booted\n", encoding="utf-8")
        running_pids.add(321)

    async def fake_is_pid_running(pid: int | None) -> bool:
        return pid in running_pids

    async def fake_is_host_port_ready(port: int) -> bool:
        return port == 46000

    async def fake_terminate_service_process(pid: int | None) -> None:
        if pid is None:
            return
        running_pids.discard(pid)
        for service in manager._services.values():
            if service.pid == pid:
                Path(service.exit_code_file).write_text("143", encoding="utf-8")

    async def fake_wait_for_service_exit(pid: int | None, *, timeout: float) -> None:
        del timeout
        if pid is None:
            return
        running_pids.discard(pid)

    monkeypatch.setattr(manager, "_launch_service_process", fake_launch_service_process)
    monkeypatch.setattr(manager, "_is_pid_running", fake_is_pid_running)
    monkeypatch.setattr(manager, "_is_host_port_ready", fake_is_host_port_ready)
    monkeypatch.setattr(manager, "_terminate_service_process", fake_terminate_service_process)
    monkeypatch.setattr(manager, "_wait_for_service_exit", fake_wait_for_service_exit)

    started = await manager.start_service(
        group_id="1001",
        workspace_dir=str(group_workspace),
        visible_workspace_root="/workspace",
        actual_container_workdir="/workspace/group-1001",
        command="python -m http.server \"$PORT\" --bind 0.0.0.0",
        name="docs",
        cwd="/workspace/app",
        preferred_port=None,
        replace_existing=False,
        env={"APP_ENV": "test"},
        startup_timeout=5.0,
    )

    assert started.record.service_id == "svc-1"
    assert started.record.name == "docs"
    assert started.record.status == "running"
    assert started.record.port == 46000
    assert started.record.address == "play.example.com:46000"
    assert "booted" in started.startup_log_excerpt

    listed_running = await manager.list_services(group_id="1001")
    assert [service.service_id for service in listed_running] == ["svc-1"]

    reloaded = ServiceManager(
        workspace_root=str(workspace_root),
        visible_container_root="/workspace",
        container_name="bampi-sandbox",
        container_shell="/bin/bash",
        port_range="46000-46001",
        public_host="play.example.com",
        startup_timeout=20.0,
        stop_timeout=10.0,
        max_active_services_per_group=4,
    )
    assert "svc-1" in reloaded._services
    assert reloaded._services["svc-1"].name == "docs"

    stopped = await manager.stop_service(group_id="1001", service_ref="docs")
    assert stopped.status == "stopped"
    assert stopped.exit_code == 143

    listed_all = await manager.list_services(group_id="1001", include_stopped=True)
    assert [service.status for service in listed_all] == ["stopped"]
