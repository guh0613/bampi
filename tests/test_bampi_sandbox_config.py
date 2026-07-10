from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sandbox_runtime_paths_are_executable_and_not_hidden_by_tmpfs() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker/bampi-sandbox/Dockerfile").read_text(encoding="utf-8")

    assert "HOME: /home/bampi" in compose
    assert "./data/bampi/agently-cli:/home/bampi/.agently-cli" in compose
    assert "- /tmp:exec,mode=1777" in compose
    assert "NPM_CONFIG_PREFIX=/usr/local" in dockerfile

    assert "HOME: /tmp" not in compose
    assert "NPM_CONFIG_PREFIX=/tmp" not in dockerfile


def test_sandbox_healthcheck_covers_tmp_execution_and_preinstalled_cli() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker/bampi-sandbox/Dockerfile").read_text(encoding="utf-8")
    healthcheck = (ROOT / "docker/bampi-sandbox/healthcheck.sh").read_text(encoding="utf-8")

    assert 'test: ["CMD", "bampi-sandbox-healthcheck"]' in compose
    assert "healthcheck.sh /usr/local/bin/bampi-sandbox-healthcheck" in dockerfile
    assert "command -v agently-cli" in healthcheck
    assert "mktemp /tmp/" in healthcheck
    assert '"$probe"' in healthcheck
