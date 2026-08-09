from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime

from nonebot.adapters.onebot.v11 import Bot

from ..bridge import ShowdownBridgeError, ShowdownRuntime
from ..formats import BattleFormatConfig, FormatRegistry
from ..manager import BattleManager
from ..translations import TranslationService
from .editor.flow import TeamEditorFlow
from .editor.models import TeamEditorState
from .editor.service import TeamEditorError
from .formatting import (
    format_quick_build_selections,
    format_recommended_set_option,
)
from .repository import TeamRecord, TeamRepository, TeamRepositoryError
from .sources import PreparedSample, TeamSourceError, TeamSourceService


_PAGE_SIZE = 6


@dataclass(slots=True)
class TeamGuideView:
    step: str
    format_id: str | None
    choices: tuple[str, ...]
    prompt: str
    page: int
    selected_format_id: str | None
    selected_name: str | None


@dataclass(slots=True)
class TeamGuideState:
    user_id: str
    group_id: int | None
    step: str
    purpose: str
    format_id: str | None = None
    battle_session_id: str | None = None
    candidate_text: str | None = None
    candidate_packed: str | None = None
    candidate_label: str | None = None
    candidate_warnings: tuple[str, ...] = ()
    pending_name: str | None = None
    choices: tuple[str, ...] = ()
    prompt: str = ""
    page: int = 0
    selected_format_id: str | None = None
    selected_name: str | None = None
    editor: TeamEditorState | None = None
    history: list[TeamGuideView] = field(default_factory=list)
    expires_at: float = 0.0


class TeamGuideManager:
    """Conversation-scoped team center and battle preparation wizard."""

    def __init__(
        self,
        *,
        manager: BattleManager,
        formats: FormatRegistry,
        repository: TeamRepository,
        runtime: ShowdownRuntime,
        team_sources: TeamSourceService,
        translator: TranslationService,
        editor_flow: TeamEditorFlow,
        idle_ttl_seconds: int = 900,
    ) -> None:
        self._manager = manager
        self._formats = formats
        self._repository = repository
        self._runtime = runtime
        self._team_sources = team_sources
        self._translator = translator
        self._editor_flow = editor_flow
        self._idle_ttl_seconds = idle_ttl_seconds
        self._states: dict[str, TeamGuideState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._index_lock = asyncio.Lock()

    async def close(self) -> None:
        async with self._index_lock:
            self._states.clear()
            self._locks.clear()

    async def has_state(self, user_id: str, group_id: int | None = None) -> bool:
        async with self._index_lock:
            state = self._states.get(user_id)
            return state is not None and state.group_id == group_id

    async def start(
        self,
        user_id: str,
        group_id: int | None = None,
        *,
        entry: str = "home",
    ) -> str:
        lock = await self._get_lock(user_id)
        async with lock:
            if not self._manager.ready:
                return (
                    "Showdown 运行时当前不可用："
                    f"{self._manager.runtime_error or '尚未初始化'}"
                )
            battle = await self._manager.get_session_by_user(user_id)
            if battle and battle.state == "active":
                return "你正在对战中；发送“宝可梦帮助”可查看当前对局操作。"
            if battle and battle.state == "pending":
                if group_id is not None and not (
                    battle.is_ai_battle
                    and battle.group_id == group_id
                    and battle.interaction_channel_for_user(user_id) == "group"
                ):
                    return (
                        "如需在群里公开完成准备，请先在对战群发送“对战准备”；"
                        "否则请私聊机器人发送“组队”。"
                    )
                if not battle.format_config.requires_team:
                    return "当前是随机队伍规则，直接发送“对战准备”即可。"
                state = TeamGuideState(
                    user_id=user_id,
                    group_id=group_id,
                    step="method",
                    purpose="battle",
                    format_id=battle.format_id,
                    battle_session_id=battle.session_id,
                )
                prompt = self._method_prompt(battle.format_config, for_battle=True)
                if entry == "library":
                    try:
                        records = await self._repository.list_teams(
                            user_id, battle.format_id
                        )
                    except TeamRepositoryError as exc:
                        return f"队伍仓库错误：{exc}\n\n{prompt}"
                    if records:
                        state.prompt = prompt
                        self._push(state)
                        state.step = "saved"
                        state.choices = tuple(record.name for record in records)
                        prompt = self._saved_prompt(state, records)
                    else:
                        prompt = "当前规则下还没有已保存队伍。\n\n" + prompt
            else:
                state = TeamGuideState(
                    user_id=user_id,
                    group_id=group_id,
                    step="home",
                    purpose="save",
                )
                prompt = self._home_prompt()
                state.prompt = prompt
                if entry == "library":
                    try:
                        prompt = await self._open_library(state, push=True)
                    except TeamRepositoryError as exc:
                        prompt = f"队伍仓库错误：{exc}\n\n{state.prompt}"
            self._touch(state)
            state.prompt = prompt
            async with self._index_lock:
                self._states[user_id] = state
            return prompt

    async def start_editor(
        self,
        user_id: str,
        *,
        format_id: str,
        team_name: str | None = None,
        group_id: int | None = None,
    ) -> str:
        lock = await self._get_lock(user_id)
        async with lock:
            if not self._manager.ready:
                return (
                    "Showdown 运行时当前不可用："
                    f"{self._manager.runtime_error or '尚未初始化'}"
                )
            battle = await self._manager.get_session_by_user(user_id)
            if battle is not None and battle.state in {"pending", "active"}:
                return "对战准备或进行期间不能编辑持久化队伍。"
            format_config = self._formats.get(format_id)
            if format_config is None or not format_config.requires_team:
                return "该规则不存在或不需要自备队伍。"
            state = TeamGuideState(
                user_id=user_id,
                group_id=group_id,
                step="home",
                purpose="save",
                prompt=self._home_prompt(),
            )
            if team_name is None:
                state.editor = await self._editor_flow.start_new(
                    user_id, format_config.format_id
                )
            else:
                record = await self._repository.get_team(
                    user_id, format_config.format_id, team_name
                )
                if record is None:
                    return f"未找到队伍「{team_name}」。"
                state.editor = await self._editor_flow.start_existing(record)
            self._touch(state)
            async with self._index_lock:
                self._states[user_id] = state
            return state.editor.prompt

    async def handle(
        self,
        bot: Bot,
        user_id: str,
        raw_text: str,
        group_id: int | None = None,
    ) -> str:
        lock = await self._get_lock(user_id)
        async with lock:
            async with self._index_lock:
                state = self._states.get(user_id)
            if state is None or state.group_id != group_id:
                return "这个会话中没有进行中的组队向导；需要时请重新发送“组队”。"
            if state.expires_at <= time.monotonic():
                await self._remove_state(user_id)
                return "组队向导已超时，请重新发送“组队”。"
            battle = await self._manager.get_session_by_user(user_id)
            if (
                state.purpose == "save"
                and battle is not None
                and battle.state in {"pending", "active"}
            ):
                await self._remove_state(user_id)
                if group_id is not None:
                    return (
                        "检测到你已进入对战，群内组队向导已结束。"
                        "请私聊机器人重新发送“组队”。"
                    )
                return "检测到你的对战状态已变化，请重新发送“组队”。"

            text = raw_text.strip()
            if text.startswith("/"):
                text = text[1:].strip()
            if not text:
                return state.editor.prompt if state.editor is not None else state.prompt
            if state.editor is not None:
                self._touch(state)
                result = await self._editor_flow.handle(state.editor, text)
                if result.status == "active":
                    return result.message
                state.editor = None
                return await self._refresh_library(state, result.message)

            normalized = text.lower()
            if normalized in {"0", "取消", "退出", "cancel", "quit"}:
                await self._remove_state(user_id)
                return "已退出队伍中心。"
            if normalized in {"返回", "上一步", "back", "b"}:
                self._touch(state)
                return self._go_back(state)
            if normalized in {"菜单", "首页", "home"}:
                self._touch(state)
                return self._go_home(state)
            if normalized in {
                "帮助",
                "help",
                "?",
                "宝可梦帮助",
                "ps帮助",
                "对战帮助",
            } or normalized.startswith(("宝可梦帮助 ", "ps帮助 ", "对战帮助 ")):
                self._touch(state)
                return self._context_help(state)
            if normalized == "组队":
                self._touch(state)
                return state.prompt
            if normalized in {"我的队伍", "队伍管理"}:
                self._touch(state)
                if state.purpose == "battle":
                    self._go_home(state)
                    return await self._handle_method(bot, state, "1")
                self._go_home(state)
                return await self._open_library(state, push=True)

            self._touch(state)
            try:
                if state.step == "home":
                    return await self._handle_home(state, text)
                if state.step.startswith("format_"):
                    return await self._handle_format(state, text)
                if state.step == "method":
                    return await self._handle_method(bot, state, text)
                if state.step in {"sample_create", "sample_browse"}:
                    return await self._handle_sample(bot, state, text)
                if state.step == "sample_browse_action":
                    return await self._handle_sample_browse_action(state, text)
                if state.step == "link":
                    return await self._handle_candidate_input(bot, state, text)
                if state.step == "custom":
                    return await self._handle_custom(bot, state, text)
                if state.step == "saved":
                    return await self._handle_saved(bot, state, text)
                if state.step == "recommend_query":
                    return await self._handle_recommend_query(state, text)
                if state.step == "library":
                    return await self._handle_library(state, text)
                if state.step == "library_action":
                    return await self._handle_library_action(state, text)
                if state.step == "library_rename":
                    return await self._handle_library_rename(state, text)
                if state.step == "library_delete":
                    return await self._handle_library_delete(state, text)
                if state.step == "library_duplicate":
                    return await self._handle_library_duplicate(state, text)
                if state.step == "library_duplicate_overwrite":
                    return await self._handle_library_duplicate_overwrite(state, text)
                if state.step == "name":
                    return await self._handle_name(state, text)
                if state.step == "overwrite":
                    return await self._handle_overwrite(state, text)
            except (
                TeamEditorError,
                TeamSourceError,
                TeamRepositoryError,
                ShowdownBridgeError,
            ) as exc:
                return f"操作失败：{exc}\n\n{state.prompt}"
            await self._remove_state(user_id)
            return "队伍中心状态异常，已退出；请重新发送“组队”。"

    async def _handle_home(self, state: TeamGuideState, text: str) -> str:
        aliases = {
            "1": "create",
            "创建": "create",
            "新建": "create",
            "2": "library",
            "我的队伍": "library",
            "队伍": "library",
            "3": "recommend",
            "推荐配招": "recommend",
            "配招": "recommend",
            "4": "samples",
            "样例": "samples",
            "抄队": "samples",
            "5": "help",
            "帮助": "help",
        }
        action = aliases.get(text.strip().lower())
        if action == "create":
            format_ids = tuple(
                config.format_id
                for config in self._formats.all()
                if config.requires_team
            )
            prompt = self._format_prompt("创建新队伍", format_ids)
            self._transition(
                state,
                step="format_create",
                prompt=prompt,
                choices=format_ids,
                format_id=None,
            )
            return prompt
        if action == "library":
            return await self._open_library(state, push=True)
        if action == "recommend":
            format_ids = tuple(
                config.format_id
                for config in self._formats.all()
                if config.recommended_set_source
            )
            prompt = self._format_prompt("查询推荐配招", format_ids)
            self._transition(
                state,
                step="format_recommend",
                prompt=prompt,
                choices=format_ids,
                format_id=None,
            )
            return prompt
        if action == "samples":
            format_ids = tuple(
                config.format_id
                for config in self._formats.all()
                if config.sample_team_source
            )
            prompt = self._format_prompt("浏览样例队伍", format_ids)
            self._transition(
                state,
                step="format_samples",
                prompt=prompt,
                choices=format_ids,
                format_id=None,
            )
            return prompt
        if action == "help":
            return self._home_help() + "\n\n" + state.prompt
        return "请输入首页菜单中的编号。\n\n" + state.prompt

    async def _handle_format(self, state: TeamGuideState, text: str) -> str:
        format_config: BattleFormatConfig | None = None
        if text.isdigit():
            index = int(text)
            if 1 <= index <= len(state.choices):
                format_config = self._formats.get(state.choices[index - 1])
        else:
            resolved = self._formats.resolve_format_token(text)
            if resolved and resolved.format_id in state.choices:
                format_config = resolved
        if format_config is None:
            return "规则编号无效。\n\n" + state.prompt

        if state.step == "format_create":
            prompt = self._method_prompt(format_config, for_battle=False)
            self._transition(
                state,
                step="method",
                prompt=prompt,
                choices=(),
                format_id=format_config.format_id,
            )
            return prompt
        if state.step == "format_recommend":
            prompt = (
                f"【{format_config.display_name} 推荐配招】\n"
                "请发送一个宝可梦名称，例如“快龙”。\n"
                "查询后可继续发送其他名称；回复“返回”重新选择规则。"
            )
            self._transition(
                state,
                step="recommend_query",
                prompt=prompt,
                choices=(),
                format_id=format_config.format_id,
            )
            return prompt
        if state.step == "format_samples":
            return await self._open_sample_list(
                state,
                format_config=format_config,
                step="sample_browse",
                push=True,
            )
        return "当前规则选择状态无效。\n\n" + state.prompt

    async def _handle_method(
        self,
        bot: Bot,
        state: TeamGuideState,
        text: str,
    ) -> str:
        format_config = self._require_format(state)
        normalized = text.strip().lower()
        if state.purpose == "battle":
            aliases = {
                "1": "saved",
                "已保存": "saved",
                "我的队伍": "saved",
                "2": "generate",
                "一键组队": "generate",
                "一键": "generate",
                "3": "link",
                "链接": "link",
                "导入": "link",
                "4": "sample",
                "样例": "sample",
                "抄队": "sample",
                "5": "custom",
                "自选": "custom",
                "自定义": "custom",
            }
        else:
            aliases = {
                "1": "generate",
                "一键组队": "generate",
                "一键": "generate",
                "2": "sample",
                "样例": "sample",
                "抄队": "sample",
                "3": "link",
                "链接": "link",
                "4": "custom",
                "自选": "custom",
                "自定义": "custom",
                "5": "manual",
                "手动": "manual",
                "文本": "manual",
                "6": "editor",
                "逐只编辑": "editor",
                "编辑器": "editor",
            }
        action = aliases.get(normalized)
        if action == "generate":
            if not format_config.generated_team_source:
                return "该规则暂不支持一键组队。\n\n" + state.prompt
            imported = await self._team_sources.generate_team(
                format_id=format_config.format_id,
                source_id=format_config.generated_team_source,
            )
            return await self._accept_candidate(
                bot,
                state,
                imported.team_text,
                imported.label or "一键推荐",
                source_warnings=imported.warnings,
            )
        if action == "sample":
            if not format_config.sample_team_source:
                return "该规则暂无稳定样例，请选择其他可用方式。\n\n" + state.prompt
            return await self._open_sample_list(
                state,
                format_config=format_config,
                step="sample_create",
                push=True,
            )
        if action in {"link", "manual"}:
            if action == "link":
                prompt = (
                    "请粘贴 PokePaste 或 crob.at 链接。\n"
                    "也可以直接粘贴完整 Showdown Export 文本。"
                )
            else:
                prompt = "请直接粘贴完整的 Showdown Export 队伍文本。"
            prompt += "\n回复“返回”重新选择方式；回复 0 退出。"
            self._transition(
                state,
                step="link",
                prompt=prompt,
                choices=(),
                format_id=format_config.format_id,
            )
            return prompt
        if action == "editor":
            state.editor = await self._editor_flow.start_new(
                state.user_id, format_config.format_id
            )
            return state.editor.prompt
        if action == "custom":
            if not format_config.recommended_set_source:
                return "该规则暂无稳定的按成员配招，请选择其他方式。\n\n" + state.prompt
            prompt = (
                "请发送想使用的宝可梦，多个名称用逗号分隔。\n"
                "例如：快龙, 仆刀将军, 赛富豪, 铁武者, 藏玛然特, 古鼎鹿\n"
                "默认采用第 1 套推荐配招；也可写成“快龙=2”。\n"
                "回复“返回”重新选择方式。"
            )
            self._transition(
                state,
                step="custom",
                prompt=prompt,
                choices=(),
                format_id=format_config.format_id,
            )
            return prompt
        if action == "saved":
            records = await self._repository.list_teams(
                state.user_id, format_config.format_id
            )
            if not records:
                return "你在该规则下还没有保存队伍。\n\n" + state.prompt
            self._push(state)
            state.step = "saved"
            state.choices = tuple(record.name for record in records)
            state.page = 0
            state.prompt = self._saved_prompt(state, records)
            return state.prompt
        return "请输入菜单中的编号。\n\n" + state.prompt

    async def _open_sample_list(
        self,
        state: TeamGuideState,
        *,
        format_config: BattleFormatConfig,
        step: str,
        push: bool,
    ) -> str:
        if not format_config.sample_team_source:
            raise TeamSourceError("该规则样例源已不可用。")
        prepared = await self._team_sources.list_compatible_samples(
            format_id=format_config.format_id,
            source_id=format_config.sample_team_source,
        )
        if push:
            self._push(state)
        state.step = step
        state.format_id = format_config.format_id
        state.choices = tuple(item.sample.name for item in prepared)
        state.page = 0
        state.prompt = self._sample_prompt(format_config, prepared, state.page)
        return state.prompt

    async def _handle_sample(
        self,
        bot: Bot,
        state: TeamGuideState,
        text: str,
    ) -> str:
        format_config = self._require_format(state)
        if not format_config.sample_team_source:
            raise TeamSourceError("该规则样例源已不可用。")
        prepared = await self._team_sources.list_compatible_samples(
            format_id=format_config.format_id,
            source_id=format_config.sample_team_source,
        )
        page_action = self._page_action(text)
        if page_action:
            changed = self._change_page(state, len(prepared), page_action)
            state.prompt = self._sample_prompt(format_config, prepared, state.page)
            if not changed:
                return "已经没有更多内容。\n\n" + state.prompt
            return state.prompt
        selected = self._page_selection(text, state.page, len(prepared))
        if selected is None:
            return "请输入本页编号，或发送“下一页”“上一页”。\n\n" + state.prompt
        item = prepared[selected]
        if state.step == "sample_create":
            return await self._accept_candidate(
                bot,
                state,
                item.team_text,
                f"样例「{item.sample.name}」by {item.sample.author}",
            )

        prepared_team = await self._runtime.prepare_team_for_use(
            format_config.format_id, item.team_text
        )
        state.candidate_text = prepared_team.team_text
        state.candidate_packed = prepared_team.packed
        state.candidate_label = f"样例「{item.sample.name}」by {item.sample.author}"
        state.candidate_warnings = prepared_team.warnings
        prompt = (
            f"【样例：{item.sample.name}】作者：{item.sample.author}\n"
            "1. 保存到我的队伍\n"
            "2. 返回样例列表\n"
            "回复 0 退出。"
        )
        self._transition(
            state,
            step="sample_browse_action",
            prompt=prompt,
            choices=(),
            format_id=format_config.format_id,
        )
        warning = self._warning_text(prepared_team.warnings)
        return f"{prepared_team.team_text}\n\n{warning}{prompt}"

    async def _handle_sample_browse_action(
        self,
        state: TeamGuideState,
        text: str,
    ) -> str:
        if text in {"1", "保存"}:
            prompt = "请输入保存名称（最多 40 个字符）；回复“返回”取消保存。"
            self._transition(
                state,
                step="name",
                prompt=prompt,
                choices=(),
                format_id=state.format_id,
            )
            return prompt
        if text in {"2", "列表"}:
            return self._go_back(state)
        return "请输入 1 或 2。\n\n" + state.prompt

    async def _handle_candidate_input(
        self,
        bot: Bot,
        state: TeamGuideState,
        text: str,
    ) -> str:
        imported = await self._team_sources.resolve_import(text)
        format_config = self._require_format(state)
        self._team_sources.ensure_format_compatible(imported, format_config.format_id)
        return await self._accept_candidate(
            bot,
            state,
            imported.team_text,
            imported.label or "Showdown 文本",
            source_warnings=imported.warnings,
        )

    async def _handle_custom(
        self,
        bot: Bot,
        state: TeamGuideState,
        text: str,
    ) -> str:
        format_config = self._require_format(state)
        if not format_config.recommended_set_source:
            raise TeamSourceError("该规则推荐配招源已不可用。")
        built = await self._team_sources.build_recommended_team(
            format_id=format_config.format_id,
            source_id=format_config.recommended_set_source,
            species_input=text,
        )
        selected = format_quick_build_selections(
            built.selections,
            self._translator,
        )
        return await self._accept_candidate(
            bot,
            state,
            built.team_text,
            f"自选成员：{selected}",
        )

    async def _handle_saved(
        self,
        bot: Bot,
        state: TeamGuideState,
        text: str,
    ) -> str:
        format_config = self._require_format(state)
        records = await self._repository.list_teams(
            state.user_id, format_config.format_id
        )
        if not records:
            return "已保存队伍列表为空。\n\n" + self._go_back(state)
        page_action = self._page_action(text)
        if page_action:
            changed = self._change_page(state, len(records), page_action)
            state.choices = tuple(record.name for record in records)
            state.prompt = self._saved_prompt(state, records)
            if not changed:
                return "已经没有更多内容。\n\n" + state.prompt
            return state.prompt
        selected = self._page_selection(text, state.page, len(records))
        if selected is None:
            return "请输入本页队伍编号。\n\n" + state.prompt
        record = records[selected]
        return await self._accept_candidate(
            bot,
            state,
            record.raw,
            f"已保存队伍「{record.name}」",
        )

    async def _handle_recommend_query(
        self,
        state: TeamGuideState,
        text: str,
    ) -> str:
        format_config = self._require_format(state)
        if not format_config.recommended_set_source:
            raise TeamSourceError("该规则推荐配招源已不可用。")
        result = await self._team_sources.list_recommended_sets(
            format_id=format_config.format_id,
            source_id=format_config.recommended_set_source,
            species_query=text,
        )
        translated = self._translator.translate_species(result.species)
        lines = [f"【{translated} 推荐配招】"]
        for index, option in enumerate(result.options, start=1):
            lines.extend(format_recommended_set_option(index, option, self._translator))
        lines.extend(
            [
                "",
                "可继续发送其他宝可梦名称查询。",
                "组队时写“快龙=2”即可采用第 2 套配置。",
                "回复“返回”重新选择规则；回复 0 退出。",
            ]
        )
        return "\n".join(lines)

    async def _open_library(self, state: TeamGuideState, *, push: bool) -> str:
        records = await self._repository.list_teams(state.user_id)
        if not records:
            return "你还没有保存队伍。可以选择“创建新队伍”开始。\n\n" + state.prompt
        if push:
            self._push(state)
        state.step = "library"
        state.format_id = None
        state.choices = tuple(record.name for record in records)
        state.page = 0
        state.selected_format_id = None
        state.selected_name = None
        state.prompt = self._library_prompt(state, records)
        return state.prompt

    async def _handle_library(self, state: TeamGuideState, text: str) -> str:
        records = await self._repository.list_teams(state.user_id)
        if not records:
            return "你的队伍列表为空。\n\n" + self._go_home(state)
        page_action = self._page_action(text)
        if page_action:
            changed = self._change_page(state, len(records), page_action)
            state.choices = tuple(record.name for record in records)
            state.prompt = self._library_prompt(state, records)
            if not changed:
                return "已经没有更多内容。\n\n" + state.prompt
            return state.prompt
        selected = self._page_selection(text, state.page, len(records))
        if selected is None:
            return "请输入本页队伍编号。\n\n" + state.prompt
        record = records[selected]
        state.selected_format_id = record.format_id
        state.selected_name = record.name
        prompt = self._library_action_prompt(record)
        self._transition(
            state,
            step="library_action",
            prompt=prompt,
            choices=(),
            format_id=None,
            selected_format_id=record.format_id,
            selected_name=record.name,
        )
        return prompt

    async def _handle_library_action(self, state: TeamGuideState, text: str) -> str:
        record = await self._selected_record(state)
        if text in {"1", "查看", "show"}:
            return f"【{record.name}｜{self._format_name(record.format_id)}】\n{record.raw}\n\n{state.prompt}"
        if text in {"2", "重命名", "rename"}:
            prompt = f"请发送队伍「{record.name}」的新名称；回复“返回”取消。"
            self._transition(
                state,
                step="library_rename",
                prompt=prompt,
                choices=(),
                format_id=None,
                selected_format_id=record.format_id,
                selected_name=record.name,
            )
            return prompt
        if text in {"3", "删除", "delete"}:
            prompt = (
                f"确定删除队伍「{record.name}」吗？\n"
                "1. 确认删除\n2. 取消\n此操作无法撤销。"
            )
            self._transition(
                state,
                step="library_delete",
                prompt=prompt,
                choices=(),
                format_id=None,
                selected_format_id=record.format_id,
                selected_name=record.name,
            )
            return prompt
        if text in {"4", "复制", "copy"}:
            prompt = f"请发送队伍「{record.name}」的副本名称；回复“返回”取消。"
            self._transition(
                state,
                step="library_duplicate",
                prompt=prompt,
                choices=(),
                format_id=None,
                selected_format_id=record.format_id,
                selected_name=record.name,
            )
            return prompt
        if text in {"5", "编辑", "edit"}:
            state.editor = await self._editor_flow.start_existing(record)
            return state.editor.prompt
        if text in {"6", "列表"}:
            return self._go_back(state)
        return "请输入 1 至 6。\n\n" + state.prompt

    async def _handle_library_rename(
        self,
        state: TeamGuideState,
        text: str,
    ) -> str:
        record = await self._selected_record(state)
        new_name = self._repository.validate_team_name(text)
        if new_name == record.name:
            return "新名称与原名称相同。\n\n" + state.prompt
        existing = await self._repository.get_team(
            state.user_id, record.format_id, new_name
        )
        if existing:
            return f"队伍名称「{new_name}」已被使用，请换一个名称。\n\n{state.prompt}"
        renamed = await self._repository.rename_team(
            state.user_id,
            record.format_id,
            record.name,
            new_name,
        )
        if not renamed:
            raise TeamRepositoryError("原队伍已不存在，可能已在其他会话中修改。")
        return await self._refresh_library(state, f"✅ 已重命名为「{new_name}」。")

    async def _handle_library_delete(
        self,
        state: TeamGuideState,
        text: str,
    ) -> str:
        if text in {"2", "取消", "否", "no"}:
            return self._go_back(state)
        if text not in {"1", "确认", "是", "yes"}:
            return "请输入 1 确认删除，或输入 2 取消。\n\n" + state.prompt
        record = await self._selected_record(state)
        removed = await self._repository.delete_team(
            state.user_id, record.format_id, record.name
        )
        if not removed:
            raise TeamRepositoryError("队伍已不存在，可能已在其他会话中删除。")
        return await self._refresh_library(state, f"✅ 已删除队伍「{record.name}」。")

    async def _handle_library_duplicate(
        self,
        state: TeamGuideState,
        text: str,
    ) -> str:
        record = await self._selected_record(state)
        new_name = self._repository.validate_team_name(text)
        existing = await self._repository.get_team(
            state.user_id, record.format_id, new_name
        )
        if existing:
            state.pending_name = new_name
            prompt = (
                f"队伍「{new_name}」已存在。\n"
                "1. 覆盖现有队伍\n2. 重新输入名称\n回复“返回”取消复制。"
            )
            self._transition(
                state,
                step="library_duplicate_overwrite",
                prompt=prompt,
                choices=(),
                format_id=None,
                selected_format_id=record.format_id,
                selected_name=record.name,
            )
            return prompt
        return await self._copy_selected(state, new_name)

    async def _handle_library_duplicate_overwrite(
        self,
        state: TeamGuideState,
        text: str,
    ) -> str:
        if text == "1":
            if not state.pending_name:
                raise TeamRepositoryError("缺少副本名称。")
            return await self._copy_selected(state, state.pending_name)
        if text == "2":
            state.pending_name = None
            return self._go_back(state)
        return "请输入 1 或 2。\n\n" + state.prompt

    async def _copy_selected(self, state: TeamGuideState, new_name: str) -> str:
        record = await self._selected_record(state)
        await self._repository.set_team(
            state.user_id,
            record.format_id,
            new_name,
            packed=record.packed,
            raw=record.raw,
        )
        state.pending_name = None
        return await self._refresh_library(state, f"✅ 已复制为队伍「{new_name}」。")

    async def _accept_candidate(
        self,
        bot: Bot,
        state: TeamGuideState,
        team_text: str,
        label: str,
        *,
        source_warnings: tuple[str, ...] = (),
    ) -> str:
        format_config = self._require_format(state)
        prepared_team = await self._runtime.prepare_team_for_use(
            format_config.format_id, team_text
        )
        normalized_text = prepared_team.team_text
        packed = prepared_team.packed
        warnings = tuple(dict.fromkeys((*source_warnings, *prepared_team.warnings)))
        warning = self._warning_text(warnings)
        if state.purpose == "battle":
            battle = await self._manager.get_session_by_user(state.user_id)
            if (
                battle is None
                or battle.session_id != state.battle_session_id
                or battle.state != "pending"
            ):
                await self._remove_state(state.user_id)
                return "原对战准备会话已失效，请重新发起对战。"
            await battle.set_team(state.user_id, packed, normalized_text)
            ready = battle.teams_ready()
            await self._remove_state(state.user_id)
            if ready:
                try:
                    await battle.start(bot)
                except ShowdownBridgeError as exc:
                    await battle.close()
                    return f"队伍已导入，但对战启动失败，房间已关闭：{exc}"
                return (
                    f"✅ 已导入{label}。双方准备完成，对战正在启动。\n{warning}"
                ).rstrip()
            return f"✅ 已导入{label}，正在等待对手完成准备。\n{warning}".rstrip()

        state.candidate_text = normalized_text
        state.candidate_packed = packed
        state.candidate_label = label
        state.candidate_warnings = warnings
        prompt = "队伍已通过校验。请回复一个保存名称（最多 40 个字符）。"
        self._transition(
            state,
            step="name",
            prompt=prompt,
            choices=(),
            format_id=format_config.format_id,
        )
        return (
            f"✅ 已生成{label}。\n{warning}{prompt}\n回复“返回”取消保存；回复 0 退出。"
        )

    async def _handle_name(self, state: TeamGuideState, text: str) -> str:
        name = self._repository.validate_team_name(text)
        format_config = self._require_format(state)
        existing = await self._repository.get_team(
            state.user_id, format_config.format_id, name
        )
        if existing:
            state.pending_name = name
            prompt = (
                f"队伍「{name}」已存在。\n"
                "1. 覆盖现有队伍\n2. 重新输入名称\n回复“返回”取消保存。"
            )
            self._transition(
                state,
                step="overwrite",
                prompt=prompt,
                choices=(),
                format_id=format_config.format_id,
            )
            return prompt
        return await self._save_candidate(state, name)

    async def _handle_overwrite(self, state: TeamGuideState, text: str) -> str:
        if text == "1":
            if not state.pending_name:
                raise TeamRepositoryError("缺少待覆盖队伍名称。")
            return await self._save_candidate(state, state.pending_name)
        if text == "2":
            state.pending_name = None
            return self._go_back(state)
        return "请输入 1 或 2。\n\n" + state.prompt

    async def _save_candidate(self, state: TeamGuideState, name: str) -> str:
        format_config = self._require_format(state)
        if state.candidate_text is None or state.candidate_packed is None:
            raise TeamRepositoryError("待保存队伍数据已经失效。")
        await self._repository.set_team(
            state.user_id,
            format_config.format_id,
            name,
            packed=state.candidate_packed,
            raw=state.candidate_text,
        )
        label = state.candidate_label or "新队伍"
        state.pending_name = None
        warning = self._warning_text(state.candidate_warnings)
        return await self._refresh_library(
            state,
            f"✅ 已保存队伍「{name}」（{format_config.display_name}）。\n"
            f"来源：{label}\n{warning}".rstrip(),
        )

    async def _refresh_library(self, state: TeamGuideState, notice: str) -> str:
        state.step = "home"
        state.format_id = None
        state.choices = ()
        state.page = 0
        state.selected_format_id = None
        state.selected_name = None
        state.history.clear()
        state.prompt = self._home_prompt()
        library = await self._open_library(state, push=True)
        return f"{notice}\n\n{library}"

    async def _selected_record(self, state: TeamGuideState) -> TeamRecord:
        if not state.selected_format_id or not state.selected_name:
            raise TeamRepositoryError("未选择队伍，请返回列表重新选择。")
        record = await self._repository.get_team(
            state.user_id,
            state.selected_format_id,
            state.selected_name,
        )
        if record is None:
            raise TeamRepositoryError("所选队伍已不存在，请返回列表刷新。")
        return record

    def _home_prompt(self) -> str:
        return (
            "【宝可梦队伍中心】\n"
            "1. 创建新队伍\n"
            "2. 我的已保存队伍\n"
            "3. 查询推荐配招\n"
            "4. 浏览在线样例\n"
            "5. 使用帮助\n\n"
            "回复编号即可；回复 0 退出。"
        )

    def _home_help(self) -> str:
        return (
            "【队伍中心帮助】\n"
            "• 新手推荐“一键推荐”或“从样例抄队”；\n"
            "• 网上队伍可直接粘贴 PokePaste/crob.at 链接；\n"
            "• 自选成员时可写“快龙=2”指定第 2 套配招；\n"
            "• “我的队伍”支持逐只编辑、查看、重命名、删除和复制；\n"
            "• 任何页面可发送“返回”“菜单”“帮助”或“0”。"
        )

    def _format_prompt(self, title: str, format_ids: tuple[str, ...]) -> str:
        lines = [f"【{title}｜选择规则】"]
        for index, format_id in enumerate(format_ids, start=1):
            config = self._formats.get(format_id)
            if config:
                lines.append(f"{index}. {config.display_name}")
        lines.append("\n回复编号；回复“返回”回到上一页。")
        return "\n".join(lines)

    @staticmethod
    def _method_prompt(
        format_config: BattleFormatConfig,
        *,
        for_battle: bool,
    ) -> str:
        lines = [f"【{format_config.display_name}｜选择队伍来源】"]
        if for_battle:
            lines.extend(
                [
                    "1. 使用我的已保存队伍",
                    "2. 一键推荐完整队伍"
                    + ("" if format_config.generated_team_source else "（暂不可用）"),
                    "3. 粘贴队伍链接或 Showdown 文本",
                    "4. 从当前样例中抄队"
                    + ("" if format_config.sample_team_source else "（暂不可用）"),
                    "5. 自选宝可梦并套用推荐配招"
                    + ("" if format_config.recommended_set_source else "（暂不可用）"),
                ]
            )
        else:
            lines.extend(
                [
                    "1. 一键推荐完整队伍"
                    + ("" if format_config.generated_team_source else "（暂不可用）"),
                    "2. 从当前样例中抄队"
                    + ("" if format_config.sample_team_source else "（暂不可用）"),
                    "3. 粘贴 PokePaste/crob.at 链接",
                    "4. 自选宝可梦并套用推荐配招"
                    + ("" if format_config.recommended_set_source else "（暂不可用）"),
                    "5. 手动粘贴 Showdown 文本",
                    "6. 打开逐只宝可梦编辑器",
                ]
            )
        lines.append("\n回复编号；回复“返回”重新选择规则；回复 0 退出。")
        return "\n".join(lines)

    def _sample_prompt(
        self,
        format_config: BattleFormatConfig,
        prepared: list[PreparedSample],
        page: int,
    ) -> str:
        start, end, total_pages = self._page_bounds(len(prepared), page)
        lines = [f"【{format_config.display_name} 样例｜{page + 1}/{total_pages}】"]
        for local_index, item in enumerate(prepared[start:end], start=1):
            lines.append(f"{local_index}. {item.sample.name} — {item.sample.author}")
        lines.append(self._pagination_hint(page, total_pages))
        return "\n".join(lines)

    def _saved_prompt(self, state: TeamGuideState, records: list[TeamRecord]) -> str:
        start, end, total_pages = self._page_bounds(len(records), state.page)
        lines = [f"【当前规则下的队伍｜{state.page + 1}/{total_pages}】"]
        for local_index, record in enumerate(records[start:end], start=1):
            lines.append(f"{local_index}. {record.name}")
        lines.append(self._pagination_hint(state.page, total_pages))
        return "\n".join(lines)

    def _library_prompt(self, state: TeamGuideState, records: list[TeamRecord]) -> str:
        start, end, total_pages = self._page_bounds(len(records), state.page)
        lines = [f"【我的队伍｜共 {len(records)} 支｜{state.page + 1}/{total_pages}】"]
        for local_index, record in enumerate(records[start:end], start=1):
            lines.append(
                f"{local_index}. {record.name}｜{self._format_name(record.format_id)}"
            )
        lines.append(self._pagination_hint(state.page, total_pages))
        return "\n".join(lines)

    def _library_action_prompt(self, record: TeamRecord) -> str:
        updated = datetime.fromtimestamp(record.updated_at).strftime("%Y-%m-%d %H:%M")
        return (
            f"【{record.name}】\n"
            f"规则：{self._format_name(record.format_id)}\n"
            f"更新：{updated}\n\n"
            "1. 查看完整队伍\n"
            "2. 重命名\n"
            "3. 删除\n"
            "4. 复制为新队伍\n"
            "5. 编辑队伍成员与配置\n"
            "6. 返回队伍列表\n"
            "回复编号即可。"
        )

    def _context_help(self, state: TeamGuideState) -> str:
        if state.step == "home":
            return self._home_help() + "\n\n" + state.prompt
        return (
            "【当前页面操作】\n"
            "按页面提示回复编号或内容；发送“返回”回到上一页，"
            "发送“菜单”回到首页，发送 0 退出。\n\n" + state.prompt
        )

    def _go_home(self, state: TeamGuideState) -> str:
        if state.purpose == "battle":
            format_config = self._require_format(state)
            state.history.clear()
            state.step = "method"
            state.prompt = self._method_prompt(format_config, for_battle=True)
            return state.prompt
        state.history.clear()
        state.step = "home"
        state.format_id = None
        state.choices = ()
        state.page = 0
        state.selected_format_id = None
        state.selected_name = None
        state.prompt = self._home_prompt()
        return state.prompt

    def _go_back(self, state: TeamGuideState) -> str:
        if not state.history:
            if state.purpose == "battle":
                return "当前已在对战准备首页。\n\n" + state.prompt
            return "当前已在队伍中心首页。\n\n" + state.prompt
        view = state.history.pop()
        state.step = view.step
        state.format_id = view.format_id
        state.choices = view.choices
        state.prompt = view.prompt
        state.page = view.page
        state.selected_format_id = view.selected_format_id
        state.selected_name = view.selected_name
        return state.prompt

    def _push(self, state: TeamGuideState) -> None:
        state.history.append(
            TeamGuideView(
                step=state.step,
                format_id=state.format_id,
                choices=state.choices,
                prompt=state.prompt,
                page=state.page,
                selected_format_id=state.selected_format_id,
                selected_name=state.selected_name,
            )
        )

    def _transition(
        self,
        state: TeamGuideState,
        *,
        step: str,
        prompt: str,
        choices: tuple[str, ...],
        format_id: str | None,
        selected_format_id: str | None = None,
        selected_name: str | None = None,
    ) -> None:
        self._push(state)
        state.step = step
        state.prompt = prompt
        state.choices = choices
        state.format_id = format_id
        state.page = 0
        state.selected_format_id = selected_format_id
        state.selected_name = selected_name

    @staticmethod
    def _page_action(text: str) -> str | None:
        normalized = text.strip().lower()
        if normalized in {"下一页", "下页", "next", "n", ">"}:
            return "next"
        if normalized in {"上一页", "上页", "prev", "p", "<"}:
            return "prev"
        return None

    @staticmethod
    def _page_bounds(total: int, page: int) -> tuple[int, int, int]:
        total_pages = max(1, math.ceil(total / _PAGE_SIZE))
        normalized_page = min(max(0, page), total_pages - 1)
        start = normalized_page * _PAGE_SIZE
        return start, min(total, start + _PAGE_SIZE), total_pages

    @staticmethod
    def _change_page(state: TeamGuideState, total: int, action: str) -> bool:
        total_pages = max(1, math.ceil(total / _PAGE_SIZE))
        target = state.page + (1 if action == "next" else -1)
        if target < 0 or target >= total_pages:
            return False
        state.page = target
        return True

    @staticmethod
    def _page_selection(text: str, page: int, total: int) -> int | None:
        if not text.isdigit():
            return None
        local_index = int(text)
        if local_index < 1 or local_index > _PAGE_SIZE:
            return None
        absolute = page * _PAGE_SIZE + local_index - 1
        return absolute if absolute < total else None

    @staticmethod
    def _pagination_hint(page: int, total_pages: int) -> str:
        actions: list[str] = []
        if page > 0:
            actions.append("上一页")
        if page + 1 < total_pages:
            actions.append("下一页")
        page_help = "；".join(actions)
        suffix = f"；{page_help}" if page_help else ""
        return f"\n回复本页编号{suffix}；回复“返回”回到上一页。"

    @staticmethod
    def _warning_text(warnings: tuple[str, ...]) -> str:
        if not warnings:
            return ""
        return "".join(f"⚠️ {warning}\n" for warning in warnings)

    def _format_name(self, format_id: str) -> str:
        config = self._formats.get(format_id)
        return config.display_name if config else format_id

    def _require_format(self, state: TeamGuideState) -> BattleFormatConfig:
        config = self._formats.get(state.format_id or "")
        if config is None:
            raise TeamSourceError("向导中的规则已经失效。")
        return config

    def _touch(self, state: TeamGuideState) -> None:
        state.expires_at = time.monotonic() + self._idle_ttl_seconds

    async def _get_lock(self, user_id: str) -> asyncio.Lock:
        async with self._index_lock:
            return self._locks.setdefault(user_id, asyncio.Lock())

    async def _remove_state(self, user_id: str) -> None:
        async with self._index_lock:
            self._states.pop(user_id, None)


__all__ = ["TeamGuideManager", "TeamGuideState", "TeamGuideView"]
