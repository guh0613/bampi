from __future__ import annotations

from ...formats import FormatRegistry
from .models import EditablePokemonSet, TeamEditorState
from .service import TeamEditorService


class TeamEditorPresenter:
    def __init__(
        self,
        *,
        formats: FormatRegistry,
        service: TeamEditorService,
    ) -> None:
        self._formats = formats
        self._service = service

    def dashboard(self, state: TeamEditorState) -> str:
        draft = state.draft
        config = self._formats.get(draft.format_id)
        format_name = config.display_name if config else draft.format_id
        name = draft.team_name or "未命名新队伍"
        dirty = "｜有未保存修改" if draft.dirty else ""
        lines = [
            f"【队伍编辑｜{name}】",
            f"规则：{format_name}｜成员：{len(draft.sets)}/{draft.rules.max_team_size}{dirty}",
            "",
        ]
        if draft.sets:
            for index, pokemon in enumerate(draft.sets, start=1):
                lines.append(f"{index}. {self.member_summary(pokemon)}")
        else:
            lines.append("（队伍为空，请先添加宝可梦）")
        start = len(draft.sets) + 1
        lines.extend(
            [
                "",
                f"{start}. 添加宝可梦",
                f"{start + 1}. 调整成员顺序",
                f"{start + 2}. 查看完整 Showdown 文本",
                f"{start + 3}. 校验队伍",
                f"{start + 4}. 校验并保存",
                f"{start + 5}. 放弃并返回",
                "",
                "回复成员或操作编号；也可发送“添加”“校验”“保存”“返回”。",
            ]
        )
        return "\n".join(lines)

    def member(self, state: TeamEditorState) -> str:
        pokemon = self.selected(state)
        lines = [
            f"【编辑成员 {state.selected_index + 1}｜{self._service.display_species(pokemon.species)}】",
            f"道具：{self._service.display_item(pokemon.item)}",
            f"特性：{self._service.display_ability(pokemon.ability)}",
            f"性格：{self._service.display_nature(pokemon.nature)}",
            "招式："
            + (
                " / ".join(self._service.display_move(move) for move in pokemon.moves)
                if pokemon.moves
                else "未设置"
            ),
            "",
        ]
        for index, (_, label) in enumerate(self.member_actions(state), start=1):
            lines.append(f"{index}. {label}")
        lines.append("\n回复编号；回复“返回”回到队伍。")
        return "\n".join(lines)

    @staticmethod
    def member_actions(state: TeamEditorState) -> tuple[tuple[str, str], ...]:
        actions: list[tuple[str, str]] = [
            ("recommend", "套用推荐配招（替换整只配置）"),
            ("paste", "粘贴单只 Showdown Export（替换整只配置）"),
            ("species", "更换宝可梦种类"),
            ("item", "修改道具"),
            ("ability", "修改特性"),
            ("moves", "修改招式"),
            ("nature", "修改性格"),
            (
                "evs",
                "修改 Stat Points" if state.draft.rules.uses_stat_points else "修改 EV",
            ),
            ("ivs", "修改 IV"),
        ]
        if state.draft.rules.supports_tera:
            actions.append(("tera", "修改太晶属性"))
        actions.extend(
            [
                ("advanced", "更多设置"),
                ("clone", "复制该成员"),
                ("delete", "删除该成员"),
                ("back", "返回队伍"),
            ]
        )
        return tuple(actions)

    def advanced(self, state: TeamEditorState) -> str:
        pokemon = self.selected(state)
        lines = [
            f"【更多设置｜{self._service.display_species(pokemon.species)}】",
            f"昵称：{pokemon.name or '无'}",
            f"等级：{pokemon.level}",
            f"性别：{pokemon.gender or '不指定'}",
            f"闪光：{'是' if pokemon.shiny else '否'}",
            f"亲密度：{pokemon.happiness}",
            f"精灵球：{pokemon.pokeball or '默认'}",
            f"觉醒力量属性：{pokemon.hp_type or '默认'}",
            f"极巨等级：{pokemon.dynamax_level}",
            f"超极巨化：{'是' if pokemon.gigantamax else '否'}",
            "",
        ]
        for index, (_, label) in enumerate(self.advanced_actions(), start=1):
            lines.append(f"{index}. {label}")
        lines.append("\n回复编号；回复“返回”回到成员设置。")
        return "\n".join(lines)

    @staticmethod
    def advanced_actions() -> tuple[tuple[str, str], ...]:
        return (
            ("nickname", "修改昵称"),
            ("level", "修改等级"),
            ("gender", "修改性别"),
            ("shiny", "修改闪光状态"),
            ("happiness", "修改亲密度"),
            ("pokeball", "修改精灵球"),
            ("hp_type", "修改觉醒力量属性"),
            ("dynamax_level", "修改极巨等级"),
            ("gigantamax", "修改超极巨化状态"),
            ("back", "返回成员设置"),
        )

    @staticmethod
    def add_method(recommendation_available: bool) -> str:
        suffix = "" if recommendation_available else "（当前规则暂不可用）"
        return (
            "【添加宝可梦】\n"
            f"1. 按中文名选择推荐配招{suffix}\n"
            "2. 粘贴单只 Showdown Export\n"
            "3. 输入宝可梦名称后手动编辑\n"
            "4. 返回队伍\n\n"
            "回复编号即可。"
        )

    def member_summary(self, pokemon: EditablePokemonSet) -> str:
        species = self._service.display_species(pokemon.species)
        item = self._service.display_item(pokemon.item)
        ability = self._service.display_ability(pokemon.ability)
        completeness = "" if pokemon.moves else "｜⚠️未设置招式"
        return f"{species}｜{item}｜{ability}{completeness}"

    @staticmethod
    def selected(state: TeamEditorState) -> EditablePokemonSet:
        index = state.selected_index
        if index is None or index < 0 or index >= len(state.draft.sets):
            raise RuntimeError("队伍编辑器未选择有效成员。")
        return state.draft.sets[index]


__all__ = ["TeamEditorPresenter"]
