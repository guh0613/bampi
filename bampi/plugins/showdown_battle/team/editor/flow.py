from __future__ import annotations

import re

from ...bridge import ShowdownBridgeError
from ...formats import FormatRegistry
from ..repository import (
    TeamRecord,
    TeamRepository,
    TeamRepositoryError,
)
from ..sources import TeamSourceError, TeamSourceService
from .member import TeamMemberEditor
from .models import EditorResponse, TeamEditorState
from .presenter import TeamEditorPresenter
from .service import TeamEditorError, TeamEditorService


class TeamEditorFlow:
    """Conversation flow for creating and editing structured team drafts."""

    def __init__(
        self,
        *,
        formats: FormatRegistry,
        repository: TeamRepository,
        team_sources: TeamSourceService,
        service: TeamEditorService,
    ) -> None:
        self._repository = repository
        self._service = service
        self._presenter = TeamEditorPresenter(formats=formats, service=service)
        self._member_editor = TeamMemberEditor(
            formats=formats,
            team_sources=team_sources,
            service=service,
            presenter=self._presenter,
            show_dashboard=self._dashboard,
        )

    async def start_new(self, user_id: str, format_id: str) -> TeamEditorState:
        state = TeamEditorState(draft=await self._service.new_draft(user_id, format_id))
        state.prompt = self._presenter.dashboard(state)
        return state

    async def start_existing(self, record: TeamRecord) -> TeamEditorState:
        state = TeamEditorState(draft=await self._service.draft_from_record(record))
        state.prompt = self._presenter.dashboard(state)
        return state

    async def handle(self, state: TeamEditorState, raw_text: str) -> EditorResponse:
        text = raw_text.strip()
        if text.startswith("/"):
            text = text[1:].strip()
        if not text:
            return EditorResponse(state.prompt)
        normalized = text.lower()

        try:
            if state.step == "discard_confirm":
                return self._handle_discard_confirm(state, normalized)
            if normalized in {"帮助", "help", "?"}:
                return EditorResponse(self._help(state))
            if normalized in {
                "0",
                "取消",
                "退出",
                "quit",
                "cancel",
                "菜单",
                "首页",
                "home",
            }:
                return self._request_discard(state)
            if normalized in {"返回", "上一步", "back", "b"}:
                return self._back(state)

            if state.step == "dashboard":
                return await self._handle_dashboard(state, text)
            if state.step in {
                "member",
                "add_method",
                "add_recommend_species",
                "replace_recommend_species",
                "add_recommend_set",
                "replace_recommend_set",
                "add_paste",
                "replace_paste",
                "add_manual_species",
                "field_advanced",
                "delete_confirm",
            } or state.step.startswith("field_"):
                return await self._member_editor.handle(state, text)
            if state.step == "reorder":
                return self._handle_reorder(state, text)
            if state.step == "save_name":
                return await self._handle_save_name(state, text)
            if state.step == "save_overwrite":
                return await self._handle_save_overwrite(state, text)
        except (TeamEditorError, TeamSourceError, ShowdownBridgeError) as exc:
            return EditorResponse(f"操作失败：{exc}\n\n{state.prompt}")
        except TeamRepositoryError as exc:
            return EditorResponse(f"队伍仓库错误：{exc}\n\n{state.prompt}")
        return EditorResponse("队伍编辑状态异常，请返回队伍列表后重试。")

    async def _handle_dashboard(
        self, state: TeamEditorState, text: str
    ) -> EditorResponse:
        draft = state.draft
        count = len(draft.sets)
        normalized = text.strip().lower()
        aliases = {
            "添加": count + 1,
            "新增": count + 1,
            "add": count + 1,
            "排序": count + 2,
            "调整顺序": count + 2,
            "查看": count + 3,
            "导出": count + 3,
            "校验": count + 4,
            "检查": count + 4,
            "保存": count + 5,
            "完成": count + 5,
            "放弃": count + 6,
        }
        choice = int(text) if text.isdigit() else aliases.get(normalized)
        if choice is None:
            return EditorResponse("请输入成员或操作编号。\n\n" + state.prompt)
        if 1 <= choice <= count:
            state.selected_index = choice - 1
            state.step = "member"
            state.prompt = self._presenter.member(state)
            return EditorResponse(state.prompt)
        if choice == count + 1:
            if count >= draft.rules.max_team_size:
                return EditorResponse(
                    f"当前规则最多允许 {draft.rules.max_team_size} 只宝可梦。\n\n"
                    + state.prompt
                )
            return self._member_editor.open_add_method(state)
        if choice == count + 2:
            if count < 2:
                return EditorResponse(
                    "至少有两只宝可梦才能调整顺序。\n\n" + state.prompt
                )
            state.step = "reorder"
            state.prompt = (
                f"请发送新的完整顺序，必须包含 1 至 {count} 且不能重复。\n"
                f"例如：{' '.join(str(index) for index in range(count, 0, -1))}\n"
                "回复“返回”取消排序。"
            )
            return EditorResponse(state.prompt)
        if choice == count + 3:
            exported = await self._service.export_draft(draft)
            state.prompt = self._presenter.dashboard(state)
            return EditorResponse(f"【当前队伍 Export】\n{exported}\n\n{state.prompt}")
        if choice == count + 4:
            prepared = await self._service.prepare_draft(draft)
            warning = self._warnings(prepared.warnings)
            state.prompt = self._presenter.dashboard(state)
            return EditorResponse(
                f"✅ 队伍已通过当前规则校验。{warning}\n\n{state.prompt}"
            )
        if choice == count + 5:
            return await self._begin_save(state)
        if choice == count + 6:
            return self._request_discard(state)
        return EditorResponse("操作编号无效。\n\n" + state.prompt)

    def _handle_reorder(self, state: TeamEditorState, text: str) -> EditorResponse:
        count = len(state.draft.sets)
        compact = re.sub(r"[\s,，、]+", "", text.strip())
        if compact.isdigit() and count <= 9 and len(compact) == count:
            indexes = [int(char) for char in compact]
        else:
            indexes = [int(item) for item in re.findall(r"\d+", text)]
        expected = list(range(1, count + 1))
        if len(indexes) != count or sorted(indexes) != expected:
            return EditorResponse(
                f"顺序必须完整包含 1 至 {count}，且每个编号只出现一次。\n\n"
                + state.prompt
            )
        state.draft.sets = [state.draft.sets[index - 1] for index in indexes]
        state.draft.dirty = True
        return self._dashboard(state, "✅ 已调整成员顺序。")

    async def _begin_save(self, state: TeamEditorState) -> EditorResponse:
        draft = state.draft
        if draft.team_name and not draft.dirty:
            return EditorResponse("队伍没有未保存的修改。", status="cancelled")
        prepared = await self._service.prepare_draft(draft)
        state.pending_team_text = prepared.team_text
        state.pending_packed = prepared.packed
        state.pending_warnings = prepared.warnings
        if draft.team_name:
            if draft.original_updated_at is None:
                raise TeamEditorError("缺少原队伍版本，无法安全覆盖。")
            await self._repository.update_team(
                draft.user_id,
                draft.format_id,
                draft.team_name,
                packed=prepared.packed,
                raw=prepared.team_text,
                expected_updated_at=draft.original_updated_at,
            )
            return EditorResponse(
                f"✅ 已保存队伍「{draft.team_name}」。"
                f"{self._warnings(prepared.warnings)}",
                status="saved",
            )
        state.step = "save_name"
        state.prompt = (
            "队伍已通过校验。请输入保存名称（最多 40 个字符）；回复“返回”继续编辑。"
        )
        return EditorResponse(
            f"✅ 队伍已通过当前规则校验。{self._warnings(prepared.warnings)}\n\n"
            + state.prompt
        )

    async def _handle_save_name(
        self, state: TeamEditorState, text: str
    ) -> EditorResponse:
        name = self._repository.validate_team_name(text)
        existing = await self._repository.get_team(
            state.draft.user_id, state.draft.format_id, name
        )
        if existing:
            state.pending_name = name
            state.pending_updated_at = existing.updated_at
            state.step = "save_overwrite"
            state.prompt = (
                f"队伍「{name}」已经存在。\n"
                "1. 覆盖现有队伍\n2. 重新输入名称\n回复“返回”继续编辑。"
            )
            return EditorResponse(state.prompt)
        return await self._persist_new(state, name)

    async def _handle_save_overwrite(
        self, state: TeamEditorState, text: str
    ) -> EditorResponse:
        if text == "2":
            state.pending_name = None
            state.pending_updated_at = None
            state.step = "save_name"
            state.prompt = "请重新输入保存名称（最多 40 个字符）。"
            return EditorResponse(state.prompt)
        if text != "1":
            return EditorResponse("请输入 1 或 2。\n\n" + state.prompt)
        if (
            not state.pending_name
            or state.pending_updated_at is None
            or state.pending_team_text is None
            or state.pending_packed is None
        ):
            raise TeamEditorError("待覆盖队伍数据已失效。")
        await self._repository.update_team(
            state.draft.user_id,
            state.draft.format_id,
            state.pending_name,
            packed=state.pending_packed,
            raw=state.pending_team_text,
            expected_updated_at=state.pending_updated_at,
        )
        return EditorResponse(
            f"✅ 已覆盖队伍「{state.pending_name}」。"
            f"{self._warnings(state.pending_warnings)}",
            status="saved",
        )

    async def _persist_new(self, state: TeamEditorState, name: str) -> EditorResponse:
        if state.pending_team_text is None or state.pending_packed is None:
            raise TeamEditorError("待保存队伍数据已失效，请返回后重新保存。")
        await self._repository.create_team(
            state.draft.user_id,
            state.draft.format_id,
            name,
            packed=state.pending_packed,
            raw=state.pending_team_text,
        )
        return EditorResponse(
            f"✅ 已保存新队伍「{name}」。{self._warnings(state.pending_warnings)}",
            status="saved",
        )

    def _request_discard(self, state: TeamEditorState) -> EditorResponse:
        if not state.draft.dirty:
            return EditorResponse(
                "已退出队伍编辑器，未修改原队伍。", status="cancelled"
            )
        state.discard_from_step = state.step
        state.discard_from_prompt = state.prompt
        state.step = "discard_confirm"
        state.prompt = (
            "当前有未保存的修改，确定放弃吗？\n1. 放弃修改并退出\n2. 继续编辑"
        )
        return EditorResponse(state.prompt)

    def _handle_discard_confirm(
        self, state: TeamEditorState, text: str
    ) -> EditorResponse:
        if text in {"1", "确认", "是", "yes"}:
            return EditorResponse("已放弃修改，原队伍没有变化。", status="cancelled")
        if text in {"2", "继续", "否", "no"}:
            state.step = state.discard_from_step or "dashboard"
            state.prompt = state.discard_from_prompt or self._presenter.dashboard(state)
            state.discard_from_step = None
            state.discard_from_prompt = None
            return EditorResponse(state.prompt)
        return EditorResponse("请输入 1 或 2。\n\n" + state.prompt)

    def _back(self, state: TeamEditorState) -> EditorResponse:
        step = state.step
        if step == "dashboard":
            return self._request_discard(state)
        if step == "member":
            return self._dashboard(state)
        if step in {
            "add_method",
            "add_recommend_species",
            "add_paste",
            "add_manual_species",
        }:
            return self._dashboard(state)
        if step == "add_recommend_set":
            state.step = "add_recommend_species"
            state.prompt = "请重新发送宝可梦名称；回复“返回”回到队伍。"
            return EditorResponse(state.prompt)
        if step in {"replace_recommend_species", "replace_paste"}:
            state.step = "member"
            state.prompt = self._presenter.member(state)
            return EditorResponse(state.prompt)
        if step == "replace_recommend_set":
            state.step = "replace_recommend_species"
            state.prompt = "请重新发送宝可梦名称；回复“返回”回到成员设置。"
            return EditorResponse(state.prompt)
        if step == "field_advanced":
            state.step = "member"
            state.prompt = self._presenter.member(state)
            return EditorResponse(state.prompt)
        if step.startswith("field_"):
            if step in {
                "field_nickname",
                "field_level",
                "field_gender",
                "field_shiny",
                "field_happiness",
                "field_pokeball",
                "field_hp_type",
                "field_dynamax_level",
                "field_gigantamax",
            }:
                state.step = "field_advanced"
                state.prompt = self._presenter.advanced(state)
            else:
                state.step = "member"
                state.prompt = self._presenter.member(state)
            return EditorResponse(state.prompt)
        if step in {"delete_confirm"}:
            state.step = "member"
            state.prompt = self._presenter.member(state)
            return EditorResponse(state.prompt)
        if step in {"reorder", "save_name", "save_overwrite"}:
            return self._dashboard(state)
        return self._dashboard(state)

    def _dashboard(
        self, state: TeamEditorState, notice: str | None = None
    ) -> EditorResponse:
        state.step = "dashboard"
        state.selected_index = None
        state.pending_species = None
        state.pending_set_names = ()
        state.prompt = self._presenter.dashboard(state)
        return EditorResponse(f"{notice}\n\n{state.prompt}" if notice else state.prompt)

    @staticmethod
    def _warnings(warnings: tuple[str, ...]) -> str:
        if not warnings:
            return ""
        return "\n" + "\n".join(f"⚠️ {warning}" for warning in warnings)

    @staticmethod
    def _help(state: TeamEditorState) -> str:
        return (
            "【队伍编辑帮助】\n"
            "• 在队伍页选择成员即可逐项修改；\n"
            "• 添加成员可使用推荐配招、单只 Export 或手动创建；\n"
            "• EV、IV 和招式支持中英文及中文逗号；\n"
            "• 所有修改只保存在临时草稿，最终通过 Showdown 校验后才会写入；\n"
            "• 随时发送“返回”，退出未保存草稿时会要求确认。\n\n" + state.prompt
        )


__all__ = ["TeamEditorFlow"]
