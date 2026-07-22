from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from bampi.plugins.bampi_chat.config import BampiChatConfig
from bampi.plugins.bampi_chat.handler import should_respond
from bampi.plugins.bampi_chat.skills import (
    builtin_skill_source_root,
    load_chat_skills,
)
from bampi.plugins.bampi_chat.session_manager import GroupSessionManager


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


@pytest.mark.parametrize(
    "text",
    [
        "/code-review 帮我看这个文件",
        "/skill install https://example.com/demo.zip",
        "/skills",
        "/foo/bar",
    ],
)
def test_slash_text_does_not_trigger_a_skill(text: str):
    decision = should_respond(
        PlaintextEvent(text),
        bot_self_id="42",
        config=BampiChatConfig(),
        random_value=1.0,
    )

    assert decision.should_respond is False


def test_slash_text_is_preserved_when_a_normal_trigger_applies():
    text = "/code-review 帮我看这个文件"
    decision = should_respond(
        PlaintextEvent(text, to_me=True),
        bot_self_id="42",
        config=BampiChatConfig(),
        random_value=1.0,
    )

    assert decision.should_respond is True
    assert decision.reason == "to_me"
    assert decision.cleaned_text == text


def test_should_not_trigger_for_formula_or_mid_message_slash_text():
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


def test_load_chat_skills_mirrors_available_builtin_skills(tmp_path: Path):
    builtin_names = {
        skill_file.parent.name for skill_file in builtin_skill_source_root().glob("*/SKILL.md")
    }
    loaded = load_chat_skills(str(tmp_path))
    names = {skill.name for skill in loaded.skills}

    assert names == builtin_names
    for name in builtin_names:
        assert (tmp_path / ".agents" / "builtin-skills" / name / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_group_session_manager_session_prompt_lists_workspace_skills(tmp_path: Path):
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
        builtin_names = {
            skill_file.parent.name for skill_file in builtin_skill_source_root().glob("*/SKILL.md")
        }
        assert "<available_skills>" in managed.session.system_prompt
        assert "docs-search" in managed.session.system_prompt
        assert f"工作目录: /workspace/{group_workspace.name}" in managed.session.system_prompt
        assert str(group_workspace.resolve()) not in managed.session.system_prompt
        for name in builtin_names:
            relative_path = f".agents/builtin-skills/{name}/SKILL.md"
            absolute_path = (group_workspace / relative_path).resolve()
            assert relative_path in managed.session.system_prompt
            assert str(absolute_path) not in managed.session.system_prompt
    finally:
        await manager.close_all()
