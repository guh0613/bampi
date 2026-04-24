from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bampy.app import Skill, format_skills_for_prompt

from .config import BampiChatConfig

PROMPT_TIMEZONE = timezone(timedelta(hours=8), name="UTC+8")


def build_system_prompt(
    config: BampiChatConfig,
    tool_names: list[str],
    *,
    skills: list[Skill] | None = None,
    prompt_cwd: str | None = None,
    append_system_prompt: str | None = None,
) -> str:
    persona = config.bampi_persona.strip() or (
        "你是Ophelia，一个在 QQ 群里协作的中文 AI 助手。"
        "你需要在多人聊天环境中保持自然、可靠、简洁，必要时再展开。"
    )
    default_prompt_cwd = config.bampi_bash_container_workdir if config.bampi_bash_mode != "local" else "."
    effective_prompt_cwd = (prompt_cwd or default_prompt_cwd).replace("\\", "/")
    current_time = datetime.now(PROMPT_TIMEZONE).strftime("%Y-%m-%d %H:%M")

    # ── 运行环境 ──
    env_lines: list[str] = []
    if config.bampi_bash_mode != "local":
        env_lines.extend(
            [
                f"- 工作目录 `{effective_prompt_cwd}`，文件操作优先使用相对路径。",
                "- 常用开发环境已就绪（bash, git, python, node, npm, ripgrep 等）。未预装 gcc/g++、Go 等，需 apt 安装（耗时较长，提前告知用户）。",
                "- `inbox/` 存放群里发来的文件和图片；写到 `outbox/` 的文件会自动发回群里。",
            ]
        )
    else:
        env_lines.extend(
            [
                "- 命令在本地 workspace 中执行，文件操作优先使用相对路径。",
                "- `inbox/` 存放群里发来的文件和图片；写到 `outbox/` 的文件会自动发回群里。",
            ]
        )

    # ── 工具 ──
    tool_lines: list[str] = []
    if any(name in tool_names for name in ("read", "grep", "find", "ls", "edit", "write")):
        tool_lines.append(
            "- 文件探索和修改优先使用专用文件工具；只在它们不适合时再退回 `bash`。"
        )
    if "bash" in tool_names:
        if config.bampi_bash_mode == "docker":
            tool_lines.append(
                f"- `bash` 在 `{effective_prompt_cwd}` 中执行命令。"
            )
        elif config.bampi_bash_mode == "auto":
            tool_lines.append(
                f"- `bash` 默认在容器 `{effective_prompt_cwd}` 中执行，失败时回退本地。"
            )
        else:
            tool_lines.append("- `bash` 在本地 workspace 中执行命令。")
        tool_lines.append(
            "- 长驻命令（dev server、watcher 等）用 `bash` 的后台会话动作（`start`/`status`/`logs`/`input`/`stop`/`list`），不要用 `&` 或 `nohup`。"
        )
        tool_lines.append(
            "- 可用 `bash` 的 `action=start` 并传 `notify_on_exit=true` 让长时间命令在结束后自动回调，你不用一直等待。"
        )
    if "web_search" in tool_names:
        tool_lines.append("- 查最新事实、新闻、模型信息或外部资料时用 `web_search`，不要凭空猜测。批量查询时考虑聚合到一个调用中。")
    if "browser" in tool_names:
        tool_lines.append(
            "- 需要真实打开网页、等待 JS 渲染、点击/输入、截图时用 `browser`；快速查资料用 `web_search`。截图默认写入 `outbox/browser/`。"
        )
    if "service" in tool_names:
        tool_lines.append(
            "- 长期运行且需对外访问的 TCP 服务用 `service`，不要只用普通 `bash` 后台会话。"
        )
        tool_lines.append(
            "- `service` 会自动分配端口并注入 `PORT`/`HOST=0.0.0.0` 环境变量；命令保持前台运行，不要加 `&` 或 `nohup`。"
        )
        advertised_host = config.bampi_service_public_host or "<未配置>"
        tool_lines.append(
            f"- 对外访问主机为 `{advertised_host}`，端口池范围 `{config.bampi_service_port_range}`。"
        )
    if "schedule" in tool_names:
        tool_lines.append(
            "- 用 `schedule` 设置定时或周期任务：一次性用 `date` + `run_at`，周期性用 `cron`。"
        )

    env_section = "\n".join(env_lines)
    tool_section = "\n".join(tool_lines) if tool_lines else "- 当前没有可用工具。"

    prompt = (
        f"{persona}\n\n"
        "## 环境\n"
        f"{env_section}\n\n"
        "## 群聊\n"
        "你在一个多人群聊中。每条消息以 `sender_name:` 开头，附带消息正文和可选的上下文字段（如 `reply_to_name`、`reply_message`、`workspace_attachments` 等）。\n"
        "区分谁在说话很重要——必要时点名回应。包含附件时会出现文件路径（指向 workspace 中的文件）或 image block。\n"
        "如果消息附带 `explicit_skill_payloads` 块，那是用户本轮强制要求使用的 skill 正文，优先遵循。\n\n"
        "## 行为\n"
        "- 以自然、简洁的方式回应，像群友对话。\n"
        "- 只引用通过工具实际获取到的内容，不要编造图片、文件或搜索结果。\n"
        "- 需要代码、排障、搜索或文件操作时，用工具获取事实。\n\n"
    )

    if tool_section != "- 当前没有可用工具。":
        prompt += f"## 工具\n{tool_section}\n\n"

    if append_system_prompt:
        prompt += f"## 额外指令\n{append_system_prompt.strip()}\n\n"

    prompt += format_skills_for_prompt(skills or [])
    prompt += f"\n当前时间(UTC+8): {current_time}\n"
    prompt += f"Current working directory: {effective_prompt_cwd}\n"

    return prompt
