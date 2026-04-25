from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.handler import IncomingMedia, build_user_message, should_respond
from bampi.plugins.bampi_chat.skills import (
    install_skills_from_source,
    load_chat_skills,
    resolve_explicit_skills,
)
from bampi.plugins.bampi_chat.session_manager import GroupSessionManager


@dataclass
class FakeSender:
    user_id: int
    nickname: str = ""
    card: str = ""


@dataclass
class FakeEvent:
    group_id: int
    user_id: int
    sender: object
    reply: object | None = None


@dataclass
class PlaintextEvent:
    text: str
    to_me: bool = False
    reply: object | None = None

    def get_plaintext(self) -> str:
        return self.text


def _write_skill(skill_dir: Path, content: str) -> Path:
    skill_file = skill_dir / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


def test_should_respond_when_explicit_skill_is_requested():
    decision = should_respond(
        PlaintextEvent("/code-review 帮我看这个文件"),
        bot_self_id="42",
        config=BampiChatConfig(),
        random_value=1.0,
    )

    assert decision.should_respond is True
    assert decision.reason == "skill"
    assert decision.direct is True


def test_should_not_treat_non_prefix_or_formula_text_as_explicit_skill():
    config = BampiChatConfig()

    formula_decision = should_respond(
        PlaintextEvent("这是 $G(s)=\\dfrac{s^2+4s+8}{s^2+5s+3}$"),
        bot_self_id="42",
        config=config,
        random_value=1.0,
    )
    mid_text_decision = should_respond(
        PlaintextEvent("请用 /code-review 帮我看这个文件"),
        bot_self_id="42",
        config=config,
        random_value=1.0,
    )

    assert formula_decision.should_respond is False
    assert mid_text_decision.should_respond is False


def test_load_chat_skills_respects_openai_manual_only_policy(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "manual-only"
    _write_skill(
        skill_dir,
        "---\n"
        "name: manual-only\n"
        "description: Explicit only.\n"
        "---\n\n"
        "# Manual Only\n",
    )
    metadata_dir = skill_dir / "agents"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "openai.yaml").write_text(
        "policy:\n"
        "  allow_implicit_invocation: false\n",
        encoding="utf-8",
    )

    loaded = load_chat_skills(str(tmp_path))
    by_name = {skill.name: skill for skill in loaded.skills}

    assert "manual-only" in by_name
    assert by_name["manual-only"].disable_model_invocation is True


def test_load_chat_skills_includes_builtin_skills(tmp_path: Path):
    loaded = load_chat_skills(str(tmp_path))
    names = {skill.name for skill in loaded.skills}

    assert {"docx", "skill-creator"} <= names
    assert (tmp_path / ".agents" / "builtin-skills" / "docx" / "SKILL.md").exists()
    assert (tmp_path / ".agents" / "builtin-skills" / "skill-creator" / "SKILL.md").exists()


def test_install_skills_from_markdown_file(tmp_path: Path):
    source = tmp_path / "downloads" / "docs-search.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "---\n"
        "name: docs-search\n"
        "description: Search API docs.\n"
        "---\n\n"
        "# Docs Search\n",
        encoding="utf-8",
    )

    result = install_skills_from_source(str(source), workspace_dir=str(tmp_path))
    installed = tmp_path / ".agents" / "skills" / "docs-search" / "SKILL.md"

    assert result.installed_names == ["docs-search"]
    assert installed.exists()
    assert "Docs Search" in installed.read_text(encoding="utf-8")


def test_install_skills_from_zip_archive_preserves_skill_files(tmp_path: Path):
    source_root = tmp_path / "source-skill" / "security-review"
    _write_skill(
        source_root,
        "---\n"
        "name: security-review\n"
        "description: Review for security issues.\n"
        "---\n\n"
        "# Security Review\n",
    )
    reference = source_root / "references" / "checklist.md"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("# Checklist\n", encoding="utf-8")

    archive = tmp_path / "security-review.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(source_root / "SKILL.md", arcname="security-review/SKILL.md")
        bundle.write(reference, arcname="security-review/references/checklist.md")

    result = install_skills_from_source(str(archive), workspace_dir=str(tmp_path))
    installed_ref = tmp_path / ".agents" / "skills" / "security-review" / "references" / "checklist.md"

    assert result.installed_names == ["security-review"]
    assert installed_ref.exists()
    assert installed_ref.read_text(encoding="utf-8") == "# Checklist\n"


def test_build_user_message_includes_explicit_skill_payload(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "code-review"
    _write_skill(
        skill_dir,
        "---\n"
        "name: code-review\n"
        "description: Review code carefully.\n"
        "---\n\n"
        "# Code Review\n\nFollow the checklist.\n",
    )

    resolution = resolve_explicit_skills(
        "/code-review 帮我看看这个补丁",
        workspace_dir=str(tmp_path),
    )
    event = FakeEvent(
        group_id=1001,
        user_id=42,
        sender=FakeSender(user_id=42, nickname="Alice"),
    )

    message = build_user_message(
        event,
        "/code-review 帮我看看这个补丁",
        media=IncomingMedia(),
        workspace_dir=str(tmp_path),
        explicit_skills=resolution,
    )

    assert "requested_skills:\n- code-review" in message.content[0].text
    assert "message_text: 帮我看看这个补丁" in message.content[0].text
    assert len(message.content) == 2
    assert "explicit_skill_payloads:" in message.content[1].text
    assert "base_dir: .agents/skills/code-review" in message.content[1].text
    assert "# Code Review" in message.content[1].text


def test_resolve_explicit_skills_ignores_markdown_math(tmp_path: Path):
    resolution = resolve_explicit_skills(
        "这是**现代控制理论**：$G(s)=\\dfrac{s^2+4s+8}{s^2+5s+3}$，以及 $e^{At}$。",
        workspace_dir=str(tmp_path),
    )

    assert resolution.requested_names == []
    assert resolution.skills == []
    assert resolution.missing_names == []
    assert resolution.cleaned_text == "这是**现代控制理论**：$G(s)=\\dfrac{s^2+4s+8}{s^2+5s+3}$，以及 $e^{At}$。"


@pytest.mark.asyncio
async def test_group_session_manager_session_prompt_lists_installed_skills(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    config = BampiChatConfig(
        bampi_workspace_dir=str(tmp_path / "workspace"),
        bampi_session_dir=str(tmp_path / "sessions"),
    )
    manager = GroupSessionManager(config)
    group_workspace = Path(manager.workspace_dir_for_group("1001"))
    _write_skill(
        group_workspace / ".agents" / "skills" / "docs-search",
        "---\n"
        "name: docs-search\n"
        "description: Search docs.\n"
        "---\n\n"
        "# Docs Search\n",
    )

    managed = await manager.get_or_create("1001")
    try:
        assert "<available_skills>" in managed.session.system_prompt
        assert "docs-search" in managed.session.system_prompt
        assert f"Current working directory: /workspace/{group_workspace.name}" in managed.session.system_prompt
        assert str(group_workspace.resolve()) not in managed.session.system_prompt
        assert ".agents/builtin-skills/docx/SKILL.md" in managed.session.system_prompt
        assert str((group_workspace / ".agents" / "builtin-skills" / "docx" / "SKILL.md").resolve()) not in managed.session.system_prompt
    finally:
        await manager.close_all()
