from __future__ import annotations

from collections.abc import Callable

from ...formats import FormatRegistry
from ..formatting import format_recommended_set_option
from ..sources import TeamSourceService
from .models import EditablePokemonSet, EditorResponse, TeamEditorState
from .presenter import TeamEditorPresenter
from .service import TeamEditorError, TeamEditorService


class TeamMemberEditor:
    """Member-level add, replace, and field-editing subflow."""

    def __init__(
        self,
        *,
        formats: FormatRegistry,
        team_sources: TeamSourceService,
        service: TeamEditorService,
        presenter: TeamEditorPresenter,
        show_dashboard: Callable[[TeamEditorState, str | None], EditorResponse],
    ) -> None:
        self._formats = formats
        self._team_sources = team_sources
        self._service = service
        self._presenter = presenter
        self._show_dashboard = show_dashboard

    async def handle(self, state: TeamEditorState, text: str) -> EditorResponse:
        if state.step == "member":
            return await self._handle_member(state, text)
        if state.step == "add_method":
            return self._handle_add_method(state, text)
        if state.step in {"add_recommend_species", "replace_recommend_species"}:
            return await self._handle_recommend_species(state, text)
        if state.step in {"add_recommend_set", "replace_recommend_set"}:
            return await self._handle_recommend_set(state, text)
        if state.step in {"add_paste", "replace_paste"}:
            return await self._handle_paste(state, text)
        if state.step == "add_manual_species":
            return await self._handle_manual_species(state, text)
        if state.step == "field_advanced":
            return self._handle_advanced(state, text)
        if state.step.startswith("field_"):
            return await self._handle_field(state, text)
        if state.step == "delete_confirm":
            return self._handle_delete(state, text)
        raise TeamEditorError("未知的成员编辑状态。")

    def open_add_method(self, state: TeamEditorState) -> EditorResponse:
        return self._open_add_method(state)

    async def _handle_member(self, state: TeamEditorState, text: str) -> EditorResponse:
        actions = self._presenter.member_actions(state)
        action = self._resolve_menu_action(text, actions)
        if action is None:
            return EditorResponse(f"请输入 1 至 {len(actions)}。\n\n{state.prompt}")
        if action == "back":
            return self._show_dashboard(state, None)
        if action == "recommend":
            if not self._recommendation_source(state):
                return EditorResponse(
                    "当前规则没有可用的推荐配招源。\n\n" + state.prompt
                )
            state.step = "replace_recommend_species"
            state.prompt = (
                "请发送要替换成的宝可梦名称，例如“快龙”。\n"
                "下一步可以选择具体推荐配招；回复“返回”取消。"
            )
            return EditorResponse(state.prompt)
        if action == "paste":
            state.step = "replace_paste"
            state.prompt = (
                "请粘贴单只宝可梦的 Showdown Export 文本。\n"
                "只接受一个成员；回复“返回”取消。"
            )
            return EditorResponse(state.prompt)
        if action == "species":
            state.step = "field_species"
            state.prompt = (
                "请发送新的宝可梦名称。支持中文或英文；更换后会重置特性并清空招式，"
                "其他设置保留。\n回复“返回”取消。"
            )
            return EditorResponse(state.prompt)
        if action in {
            "item",
            "ability",
            "moves",
            "nature",
            "evs",
            "ivs",
            "tera",
        }:
            return await self._open_field(state, action)
        if action == "advanced":
            state.step = "field_advanced"
            state.prompt = self._presenter.advanced(state)
            return EditorResponse(state.prompt)
        if action == "clone":
            if len(state.draft.sets) >= state.draft.rules.max_team_size:
                return EditorResponse("队伍人数已达到当前规则上限。\n\n" + state.prompt)
            pokemon = self._presenter.selected(state).clone()
            state.draft.sets.append(pokemon)
            state.draft.dirty = True
            return self._show_dashboard(
                state,
                f"✅ 已复制 {self._service.display_species(pokemon.species)}，"
                "请继续修改副本以避免规则冲突。",
            )
        if action == "delete":
            pokemon = self._presenter.selected(state)
            state.step = "delete_confirm"
            state.prompt = (
                f"确定删除 {self._service.display_species(pokemon.species)} 吗？\n"
                "1. 确认删除\n2. 取消"
            )
            return EditorResponse(state.prompt)
        return EditorResponse("该成员操作暂不可用。\n\n" + state.prompt)

    def _open_add_method(self, state: TeamEditorState) -> EditorResponse:
        state.selected_index = None
        state.step = "add_method"
        state.prompt = self._presenter.add_method(
            self._recommendation_source(state) is not None
        )
        return EditorResponse(state.prompt)

    def _handle_add_method(self, state: TeamEditorState, text: str) -> EditorResponse:
        aliases = {
            "1": "recommend",
            "推荐": "recommend",
            "2": "paste",
            "粘贴": "paste",
            "导入": "paste",
            "3": "manual",
            "手动": "manual",
            "4": "back",
        }
        action = aliases.get(text.strip().lower())
        if action == "recommend":
            if not self._recommendation_source(state):
                return EditorResponse(
                    "当前规则暂无推荐配招，请使用粘贴或手动添加。\n\n" + state.prompt
                )
            state.step = "add_recommend_species"
            state.prompt = (
                "请发送要添加的宝可梦名称，例如“快龙”。\n"
                "下一步可以选择具体推荐配招；回复“返回”取消。"
            )
            return EditorResponse(state.prompt)
        if action == "paste":
            state.step = "add_paste"
            state.prompt = (
                "请粘贴单只宝可梦的 Showdown Export 文本。\n"
                "只接受一个成员；回复“返回”取消。"
            )
            return EditorResponse(state.prompt)
        if action == "manual":
            state.step = "add_manual_species"
            state.prompt = (
                "请发送宝可梦名称，支持中文或英文。\n"
                "会创建基础配置，然后回到成员页面继续填写招式等内容。"
            )
            return EditorResponse(state.prompt)
        if action == "back":
            return self._show_dashboard(state, None)
        return EditorResponse("请输入 1 至 4。\n\n" + state.prompt)

    async def _handle_recommend_species(
        self, state: TeamEditorState, text: str
    ) -> EditorResponse:
        source_id = self._recommendation_source(state)
        if not source_id:
            raise TeamEditorError("当前规则没有推荐配招源。")
        result = await self._team_sources.list_recommended_sets(
            format_id=state.draft.format_id,
            source_id=source_id,
            species_query=text,
        )
        state.pending_species = result.species
        state.pending_set_names = result.set_names
        state.step = (
            "add_recommend_set"
            if state.step == "add_recommend_species"
            else "replace_recommend_set"
        )
        lines = [f"【{self._service.display_species(result.species)}｜选择推荐配招】"]
        for index, option in enumerate(result.options, start=1):
            lines.extend(
                format_recommended_set_option(index, option, self._service.translator)
            )
        lines.append("\n回复配招编号；回复“返回”重新输入宝可梦。")
        state.prompt = "\n".join(lines)
        return EditorResponse(state.prompt)

    async def _handle_recommend_set(
        self, state: TeamEditorState, text: str
    ) -> EditorResponse:
        if not text.isdigit():
            return EditorResponse("请输入配招编号。\n\n" + state.prompt)
        index = int(text)
        if index < 1 or index > len(state.pending_set_names):
            return EditorResponse(
                f"配招编号应在 1 至 {len(state.pending_set_names)} 之间。\n\n"
                + state.prompt
            )
        if not state.pending_species:
            raise TeamEditorError("待选择的宝可梦资料已失效。")
        source_id = self._recommendation_source(state)
        if not source_id:
            raise TeamEditorError("当前规则没有推荐配招源。")
        built = await self._team_sources.build_recommended_team(
            format_id=state.draft.format_id,
            source_id=source_id,
            species_input=f"{state.pending_species}={index}",
        )
        pokemon = await self._service.import_single_set(
            state.draft.format_id, built.team_text
        )
        replacing = state.step == "replace_recommend_set"
        return self._apply_imported_member(
            state,
            pokemon,
            replacing=replacing,
            notice=f"✅ 已采用推荐配招「{state.pending_set_names[index - 1]}」。",
        )

    async def _handle_paste(self, state: TeamEditorState, text: str) -> EditorResponse:
        pokemon = await self._service.import_single_set(state.draft.format_id, text)
        return self._apply_imported_member(
            state,
            pokemon,
            replacing=state.step == "replace_paste",
            notice="✅ 已导入单只宝可梦配置。",
        )

    async def _handle_manual_species(
        self, state: TeamEditorState, text: str
    ) -> EditorResponse:
        pokemon = await self._service.create_manual_set(state.draft.format_id, text)
        if len(state.draft.sets) >= state.draft.rules.max_team_size:
            raise TeamEditorError("队伍人数已达到当前规则上限。")
        state.draft.sets.append(pokemon)
        state.draft.dirty = True
        state.selected_index = len(state.draft.sets) - 1
        state.step = "member"
        state.prompt = self._presenter.member(state)
        return EditorResponse(
            f"✅ 已添加 {self._service.display_species(pokemon.species)}。"
            "请继续设置招式和其他配置。\n\n" + state.prompt
        )

    def _apply_imported_member(
        self,
        state: TeamEditorState,
        pokemon: object,
        *,
        replacing: bool,
        notice: str,
    ) -> EditorResponse:
        if not isinstance(pokemon, EditablePokemonSet):
            raise TeamEditorError("导入的成员数据无效。")
        if replacing:
            index = state.selected_index
            if index is None or index >= len(state.draft.sets):
                raise TeamEditorError("原成员已不存在。")
            state.draft.sets[index] = pokemon
            state.step = "member"
            state.draft.dirty = True
            state.prompt = self._presenter.member(state)
            return EditorResponse(f"{notice}\n\n{state.prompt}")
        if len(state.draft.sets) >= state.draft.rules.max_team_size:
            raise TeamEditorError("队伍人数已达到当前规则上限。")
        state.draft.sets.append(pokemon)
        state.draft.dirty = True
        state.selected_index = len(state.draft.sets) - 1
        state.step = "member"
        state.prompt = self._presenter.member(state)
        return EditorResponse(f"{notice}\n\n{state.prompt}")

    async def _open_field(self, state: TeamEditorState, action: str) -> EditorResponse:
        pokemon = self._presenter.selected(state)
        prompts = {
            "item": (
                "field_item",
                f"当前道具：{self._service.display_item(pokemon.item)}\n"
                "请发送新道具的中文或英文名；发送“无”清空道具。",
            ),
            "ability": (
                "field_ability",
                "可选特性加载中……",
            ),
            "moves": (
                "field_moves",
                "请一次发送完整招式列表，多个招式用逗号分隔。\n"
                "例如：龙之舞, 神速, 地震, 羽栖",
            ),
            "nature": (
                "field_nature",
                f"当前性格：{self._service.display_nature(pokemon.nature)}\n"
                "请发送新性格的中文或英文名。",
            ),
            "evs": (
                "field_evs",
                (
                    "请发送完整 Stat Points，例如：32 HP / 32 Atk / 2 Spe。"
                    if state.draft.rules.uses_stat_points
                    else "请发送完整 EV，例如：252 HP / 252 Atk / 4 SpD。"
                )
                + "\n发送“重置”可全部归零。",
            ),
            "ivs": (
                "field_ivs",
                "请发送非默认或完整 IV，例如：0 Atk / 31 Spe。\n"
                "未填写项按 31；发送“重置”可全部恢复 31。",
            ),
            "tera": (
                "field_tera",
                f"当前太晶属性：{self._service.display_type(pokemon.tera_type)}\n"
                "请发送属性的中文或英文名；发送“默认”清空。",
            ),
        }
        step, prompt = prompts[action]
        if action == "ability":
            options = await self._service.species_options(
                state.draft.format_id, pokemon.species
            )
            listed = "、".join(
                self._service.display_ability(ability) for ability in options.abilities
            )
            prompt = (
                f"当前特性：{self._service.display_ability(pokemon.ability)}\n"
                f"可选：{listed or '无'}\n请发送新特性的中文或英文名。"
            )
        state.step = step  # type: ignore[assignment]
        state.prompt = prompt + "\n回复“返回”取消。"
        return EditorResponse(state.prompt)

    async def _handle_field(self, state: TeamEditorState, text: str) -> EditorResponse:
        pokemon = self._presenter.selected(state)
        step = state.step
        if step == "field_species":
            options = await self._service.species_options(state.draft.format_id, text)
            pokemon.species = options.species
            pokemon.ability = options.abilities[0] if options.abilities else ""
            pokemon.moves = []
            notice = "✅ 已更换宝可梦，并重置特性及招式。"
        elif step == "field_item":
            pokemon.item = await self._service.resolve_item(state.draft.format_id, text)
            notice = "✅ 已更新道具。"
        elif step == "field_ability":
            pokemon.ability = await self._service.resolve_ability(
                state.draft.format_id, pokemon.species, text
            )
            notice = "✅ 已更新特性。"
        elif step == "field_moves":
            pokemon.moves = await self._service.resolve_moves(
                state.draft.format_id, pokemon.species, text
            )
            notice = "✅ 已更新招式。"
        elif step == "field_nature":
            pokemon.nature = await self._service.resolve_nature(
                state.draft.format_id, text
            )
            notice = "✅ 已更新性格。"
        elif step == "field_evs":
            label = "Stat Points" if state.draft.rules.uses_stat_points else "EV"
            pokemon.evs = self._service.parse_stats(
                text,
                default=0,
                maximum=state.draft.rules.stat_value_limit,
                enforce_total=state.draft.rules.stat_total_limit,
                label=label,
            )
            notice = f"✅ 已更新{label}。"
        elif step == "field_ivs":
            pokemon.ivs = self._service.parse_stats(
                text,
                default=31,
                maximum=31,
                enforce_total=None,
                label="IV",
            )
            notice = "✅ 已更新 IV。"
        elif step == "field_tera":
            pokemon.tera_type = await self._service.resolve_type(
                state.draft.format_id, text, allow_empty=True
            )
            notice = "✅ 已更新太晶属性。"
        else:
            return await self._handle_advanced_field(state, text)
        state.draft.dirty = True
        state.step = "member"
        state.prompt = self._presenter.member(state)
        return EditorResponse(f"{notice}\n\n{state.prompt}")

    def _handle_advanced(self, state: TeamEditorState, text: str) -> EditorResponse:
        actions = self._presenter.advanced_actions()
        action = self._resolve_menu_action(text, actions)
        if action is None:
            return EditorResponse(f"请输入 1 至 {len(actions)}。\n\n{state.prompt}")
        if action == "back":
            state.step = "member"
            state.prompt = self._presenter.member(state)
            return EditorResponse(state.prompt)
        prompts = {
            "nickname": "请发送昵称（最多 18 个字符）；发送“无”清空。",
            "level": (
                f"请输入等级（{state.draft.rules.min_level} 至 "
                f"{state.draft.rules.max_level}）。"
            ),
            "gender": "请输入 M/雄、F/雌或“无”。",
            "shiny": "是否为闪光？\n1. 是\n2. 否",
            "happiness": "请输入亲密度（0 至 255）。",
            "pokeball": "请输入精灵球中文或英文名；发送“默认”清空。",
            "hp_type": "请输入觉醒力量属性；发送“默认”清空。",
            "dynamax_level": "请输入极巨等级（0 至 10）。",
            "gigantamax": "是否启用超极巨化？\n1. 是\n2. 否",
        }
        state.step = f"field_{action}"  # type: ignore[assignment]
        state.prompt = prompts[action] + "\n回复“返回”取消。"
        return EditorResponse(state.prompt)

    async def _handle_advanced_field(
        self, state: TeamEditorState, text: str
    ) -> EditorResponse:
        pokemon = self._presenter.selected(state)
        step = state.step
        if step == "field_nickname":
            nickname = (
                ""
                if text.strip().lower() in {"无", "清空", "none", "0"}
                else text.strip()
            )
            if len(nickname) > 18 or any(char in nickname for char in "\r\n\t"):
                raise TeamEditorError(
                    "昵称不能超过 18 个字符，且不能包含换行或制表符。"
                )
            pokemon.name = nickname
        elif step == "field_level":
            pokemon.level = self._service.parse_integer(
                text,
                minimum=state.draft.rules.min_level,
                maximum=state.draft.rules.max_level,
                label="等级",
            )
        elif step == "field_gender":
            pokemon.gender = self._service.parse_gender(text)
        elif step == "field_shiny":
            pokemon.shiny = self._service.parse_boolean(text, label="闪光状态")
        elif step == "field_happiness":
            pokemon.happiness = self._service.parse_integer(
                text, minimum=0, maximum=255, label="亲密度"
            )
        elif step == "field_pokeball":
            pokemon.pokeball = await self._service.resolve_item(
                state.draft.format_id, text
            )
        elif step == "field_hp_type":
            pokemon.hp_type = await self._service.resolve_type(
                state.draft.format_id, text, allow_empty=True
            )
        elif step == "field_dynamax_level":
            pokemon.dynamax_level = self._service.parse_integer(
                text, minimum=0, maximum=10, label="极巨等级"
            )
        elif step == "field_gigantamax":
            pokemon.gigantamax = self._service.parse_boolean(text, label="超极巨化状态")
        else:
            raise TeamEditorError("未知的高级设置。")
        state.draft.dirty = True
        state.step = "field_advanced"
        state.prompt = self._presenter.advanced(state)
        return EditorResponse(f"✅ 已更新设置。\n\n{state.prompt}")

    def _handle_delete(self, state: TeamEditorState, text: str) -> EditorResponse:
        if text.strip().lower() in {"2", "取消", "否", "no"}:
            state.step = "member"
            state.prompt = self._presenter.member(state)
            return EditorResponse(state.prompt)
        if text.strip().lower() not in {"1", "确认", "是", "yes"}:
            return EditorResponse("请输入 1 确认，或 2 取消。\n\n" + state.prompt)
        index = state.selected_index
        if index is None or index >= len(state.draft.sets):
            raise TeamEditorError("待删除成员已不存在。")
        removed = state.draft.sets.pop(index)
        state.draft.dirty = True
        state.selected_index = None
        return self._show_dashboard(
            state,
            f"✅ 已删除 {self._service.display_species(removed.species)}。",
        )

    def _recommendation_source(self, state: TeamEditorState) -> str | None:
        config = self._formats.get(state.draft.format_id)
        return config.recommended_set_source if config else None

    @staticmethod
    def _resolve_menu_action(
        text: str, actions: tuple[tuple[str, str], ...]
    ) -> str | None:
        normalized = text.strip().lower()
        if normalized.isdigit():
            index = int(normalized)
            if 1 <= index <= len(actions):
                return actions[index - 1][0]
            return None
        aliases = {
            "推荐": "recommend",
            "粘贴": "paste",
            "导入": "paste",
            "种类": "species",
            "宝可梦": "species",
            "道具": "item",
            "特性": "ability",
            "招式": "moves",
            "性格": "nature",
            "ev": "evs",
            "ivs": "ivs",
            "iv": "ivs",
            "太晶": "tera",
            "更多": "advanced",
            "复制": "clone",
            "删除": "delete",
            "返回": "back",
        }
        action = aliases.get(normalized)
        return action if action in {item[0] for item in actions} else None


__all__ = ["TeamMemberEditor"]
