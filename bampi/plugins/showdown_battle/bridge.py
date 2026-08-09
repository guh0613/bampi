from __future__ import annotations

import asyncio
import json
import logging
import re
from asyncio.subprocess import Process
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ShowdownBridgeError(RuntimeError):
    pass


class ShowdownRuntimeUnavailable(ShowdownBridgeError):
    pass


class ShowdownTeamPackError(ShowdownBridgeError):
    pass


class ShowdownTeamValidationError(ShowdownTeamPackError):
    pass


@dataclass(frozen=True, slots=True)
class ShowdownRuntimeInfo:
    version: str
    node_version: str
    formats: dict[str, str]


@dataclass(frozen=True, slots=True)
class PreparedShowdownTeam:
    team_text: str
    packed: str
    warnings: tuple[str, ...] = ()


class ShowdownRuntime:
    """Reproducible interface to the pinned pokemon-showdown npm package."""

    def __init__(self, *, node_bin: str, package_dir: Path) -> None:
        self.node_bin = node_bin
        self.package_dir = package_dir.resolve()
        self.cli_script = self.package_dir / "pokemon-showdown"
        self._skip_build = True

    @property
    def common_flags(self) -> list[str]:
        return ["--skip-build"] if self._skip_build else []

    async def inspect(self, format_ids: list[str]) -> ShowdownRuntimeInfo:
        self._ensure_layout()
        script = """
const path = require('path');
const root = process.argv[1];
const ids = JSON.parse(process.argv[2]);
const pkg = require(path.join(root, 'package.json'));
const {Dex} = require(path.join(root, 'dist', 'sim'));
Dex.includeFormats();
const formats = {};
for (const id of ids) {
  const format = Dex.formats.get(id);
  if (format.exists) formats[id] = format.name;
}
process.stdout.write(JSON.stringify({version: pkg.version, formats}));
"""
        stdout, _ = await self._run_node(
            ["-e", script, str(self.package_dir), json.dumps(format_ids)],
            timeout=20,
        )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ShowdownRuntimeUnavailable(
                "Pokémon Showdown 运行时返回了无效的检查结果。"
            ) from exc

        node_stdout, _ = await self._run_node(["--version"], timeout=5)
        return ShowdownRuntimeInfo(
            version=str(payload.get("version") or "unknown"),
            node_version=node_stdout.strip(),
            formats={str(k): str(v) for k, v in payload.get("formats", {}).items()},
        )

    async def pack_team(self, team_text: str, *, timeout: float = 15.0) -> str:
        text = team_text.strip()
        if not text:
            raise ShowdownTeamPackError(
                "队伍内容为空，请粘贴 Showdown 队伍文本或导出字符串。"
            )
        stdout, stderr, returncode = await self._run_cli(
            ["pack-team"], input_data=f"{text}\n", timeout=timeout
        )
        if returncode != 0:
            raise ShowdownTeamPackError(
                stderr.strip() or stdout.strip() or "队伍格式无法识别。"
            )
        packed = stdout.strip()
        if not packed:
            raise ShowdownTeamPackError("Showdown 未返回打包后的队伍。")
        return packed

    async def validate_team(
        self, format_id: str, team_text: str, *, timeout: float = 15.0
    ) -> None:
        text = team_text.strip()
        if not text:
            raise ShowdownTeamValidationError(
                "队伍内容为空，请粘贴 Showdown 队伍文本或导出字符串。"
            )
        if not format_id.strip():
            raise ShowdownTeamValidationError("未指定对战规则，无法校验队伍。")
        stdout, stderr, returncode = await self._run_cli(
            ["validate-team", format_id],
            input_data=f"{text}\n",
            timeout=timeout,
        )
        if returncode != 0:
            raise ShowdownTeamValidationError(
                stderr.strip() or stdout.strip() or "队伍未通过对应规则的合法性校验。"
            )

    async def validate_set(
        self, format_id: str, set_text: str, *, timeout: float = 15.0
    ) -> None:
        """Validate one Pokémon set without applying whole-team size rules."""
        self._ensure_layout()
        text = set_text.strip()
        if not text:
            raise ShowdownTeamValidationError("宝可梦配置为空。")
        script = r"""
const path = require('path');
const fs = require('fs');
const root = process.argv[1];
const formatId = process.argv[2];
const input = fs.readFileSync(0, 'utf8');
try {
  const {Dex, Teams} = require(path.join(root, 'dist', 'sim'));
  const {TeamValidator} = require(path.join(root, 'dist', 'sim', 'team-validator.js'));
  Dex.includeFormats();
  const validator = TeamValidator.get(formatId);
  const team = Teams.import(input);
  if (!team || team.length !== 1) throw new Error('Expected exactly one Pokémon set.');
  const validateSet = validator.format.validateSet || validator.validateSet;
  const problems = validateSet.call(validator, team[0], {});
  process.stdout.write(JSON.stringify({problems: problems || null}));
} catch (error) {
  process.stdout.write(JSON.stringify({
    error: error && error.message ? error.message : String(error),
  }));
}
"""
        stdout, _ = await self._run_node(
            ["-e", script, str(self.package_dir), format_id],
            timeout=timeout,
            input_data=text,
        )
        payload = self._decode_json_result(stdout, operation="单体配置校验")
        problems = payload.get("problems")
        if isinstance(problems, list) and problems:
            raise ShowdownTeamValidationError(
                "\n".join(str(problem) for problem in problems)
            )

    async def validate_and_pack_team(
        self, format_id: str, team_text: str, *, timeout: float = 15.0
    ) -> str:
        prepared = await self._validate_normalize_and_pack_team(
            format_id,
            team_text,
            timeout=timeout,
        )
        return prepared.packed

    async def prepare_team_for_use(
        self,
        format_id: str,
        team_text: str,
        *,
        timeout: float = 15.0,
    ) -> PreparedShowdownTeam:
        """Validate and canonicalize user-facing team input for battle use."""
        normalized = team_text.strip()
        completion_warnings: tuple[str, ...] = ()
        try:
            prepared = await self._validate_normalize_and_pack_team(
                format_id,
                normalized,
                timeout=timeout,
            )
        except ShowdownTeamValidationError:
            if "champions" not in format_id.lower():
                raise
            completed, count = self._complete_champions_open_team_sheet(normalized)
            if not count:
                raise
            prepared = await self._validate_normalize_and_pack_team(
                format_id,
                completed,
                timeout=timeout,
            )
            completion_warnings = (
                f"检测到 {count} 只宝可梦未提供 Stat Points 和性格。"
                "这通常是公开队伍表隐藏了实数配置；已补充 Hardy（勤奋）"
                "中性性格并按 0 Stat Points 导入。原始隐藏配置无法从链接恢复，"
                "建议正式对战前自行调整。",
            )
        return PreparedShowdownTeam(
            team_text=prepared.team_text,
            packed=prepared.packed,
            warnings=tuple(dict.fromkeys((*completion_warnings, *prepared.warnings))),
        )

    async def _validate_normalize_and_pack_team(
        self,
        format_id: str,
        team_text: str,
        *,
        timeout: float,
    ) -> PreparedShowdownTeam:
        """Run validation and packing on the same mutable Showdown team.

        Pokémon Showdown's validator canonicalizes battle-only formes (notably
        Mega formes) back to their legal starting species and abilities. Calling
        the standalone ``validate-team`` and ``pack-team`` commands separately
        loses those mutations, which can otherwise make a Mega forme start the
        battle already transformed.
        """
        text = team_text.strip()
        if not text:
            raise ShowdownTeamValidationError(
                "队伍内容为空，请粘贴 Showdown 队伍文本或导出字符串。"
            )
        if not format_id.strip():
            raise ShowdownTeamValidationError("未指定对战规则，无法校验队伍。")

        script = r"""
const path = require('path');
const root = process.argv[1];
const formatId = process.argv[2];
const fs = require('fs');
const teamText = fs.readFileSync(0, 'utf8');
try {
  const {Dex, Teams} = require(path.join(root, 'dist', 'sim'));
  const {TeamValidator} = require(path.join(root, 'dist', 'sim', 'team-validator.js'));
  Dex.includeFormats();
  const format = Dex.formats.get(formatId);
  if (!format.exists) throw new Error(`Unknown format: ${formatId}`);
  const dex = Dex.forFormat(formatId);
  const team = Teams.import(teamText);
  if (!team || !team.length) throw new Error('The team is empty or malformed.');
  const before = team.map(set => ({species: set.species, ability: set.ability}));
  const problems = TeamValidator.get(formatId).validateTeam(team);
  if (problems && problems.length) {
    process.stdout.write(JSON.stringify({problems}));
  } else {
    let megaFormeCount = 0;
    let megaAbilityResetCount = 0;
    for (let index = 0; index < team.length; index++) {
      const original = before[index];
      const originalSpecies = dex.species.get(original.species);
      if (!originalSpecies.isMega) continue;
      megaFormeCount++;
      if (original.ability !== team[index].ability) megaAbilityResetCount++;
    }
    process.stdout.write(JSON.stringify({
      teamText: Teams.export(team, {
        useStatPoints: format.mod.startsWith('champions'),
      }).trim(),
      packed: Teams.pack(team),
      megaFormeCount,
      megaAbilityResetCount,
    }));
  }
} catch (error) {
  process.stdout.write(JSON.stringify({
    error: error && error.message ? error.message : String(error),
  }));
}
"""
        stdout, _ = await self._run_node(
            ["-e", script, str(self.package_dir), format_id],
            timeout=timeout,
            input_data=text,
        )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ShowdownRuntimeUnavailable(
                "Pokémon Showdown 返回了无效的队伍校验结果。"
            ) from exc

        problems = payload.get("problems")
        if isinstance(problems, list) and problems:
            raise ShowdownTeamValidationError(
                "\n".join(str(problem) for problem in problems)
            )
        error = payload.get("error")
        if error:
            raise ShowdownTeamValidationError(str(error))
        normalized_text = payload.get("teamText")
        packed = payload.get("packed")
        if not isinstance(normalized_text, str) or not normalized_text.strip():
            raise ShowdownRuntimeUnavailable("Showdown 未返回规范化后的队伍文本。")
        normalized_text = "\n".join(
            line.rstrip() for line in normalized_text.strip().splitlines()
        )
        if not isinstance(packed, str) or not packed.strip():
            raise ShowdownRuntimeUnavailable("Showdown 未返回打包后的队伍。")

        warnings: tuple[str, ...] = ()
        mega_forme_count = payload.get("megaFormeCount")
        if isinstance(mega_forme_count, int) and mega_forme_count > 0:
            ability_note = ""
            reset_count = payload.get("megaAbilityResetCount")
            if isinstance(reset_count, int) and reset_count > 0:
                ability_note = f"其中 {reset_count} 只的基础特性也已按规则还原。"
            warnings = (
                f"检测到 {mega_forme_count} 只宝可梦以 Mega 形态导入；"
                "已还原为合法的基础形态。"
                f"{ability_note}Mega 进化需在对局行动指令末尾添加 mega。",
            )
        return PreparedShowdownTeam(
            team_text=normalized_text,
            packed=packed.strip(),
            warnings=warnings,
        )

    @staticmethod
    def _complete_champions_open_team_sheet(team_text: str) -> tuple[str, int]:
        nature_pattern = re.compile(
            r"^(?:Hardy|Lonely|Brave|Adamant|Naughty|Bold|Docile|Relaxed|"
            r"Impish|Lax|Timid|Hasty|Serious|Jolly|Naive|Modest|Mild|Quiet|"
            r"Bashful|Rash|Calm|Gentle|Sassy|Careful|Quirky) Nature$",
            re.IGNORECASE,
        )
        blocks = re.split(r"\n\s*\n", team_text.strip())
        completed: list[str] = []
        changed = 0
        for block in blocks:
            lines = [line.rstrip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            has_nature = any(nature_pattern.fullmatch(line.strip()) for line in lines)
            ev_total = 0
            for line in lines:
                if not line.lstrip().lower().startswith("evs:"):
                    continue
                ev_total += sum(int(value) for value in re.findall(r"\b(\d+)\b", line))
            has_move = any(line.lstrip().startswith("-") for line in lines)
            if not has_nature and ev_total == 0 and has_move:
                move_index = next(
                    (
                        index
                        for index, line in enumerate(lines)
                        if line.lstrip().startswith("-")
                    ),
                    len(lines),
                )
                lines.insert(move_index, "Hardy Nature")
                changed += 1
            completed.append("\n".join(lines))
        return "\n\n".join(completed), changed

    async def import_team_json(
        self,
        format_id: str,
        team_text: str,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, Any]]:
        """Parse any Showdown team format into canonical editable JSON sets."""
        self._ensure_layout()
        text = team_text.strip()
        if not text:
            raise ShowdownTeamPackError("队伍内容为空，无法打开编辑器。")
        if not format_id.strip():
            raise ShowdownTeamPackError("未指定队伍规则。")
        script = r"""
const path = require('path');
const fs = require('fs');
const root = process.argv[1];
const formatId = process.argv[2];
const input = fs.readFileSync(0, 'utf8');
try {
  const {Dex, Teams} = require(path.join(root, 'dist', 'sim'));
  Dex.includeFormats();
  const format = Dex.formats.get(formatId);
  if (!format.exists) throw new Error(`Unknown format: ${formatId}`);
  const team = Teams.import(input);
  if (!team || !team.length) throw new Error('The team is empty or malformed.');
  if (team.length > 24) throw new Error('The team has too many Pokémon.');
  process.stdout.write(JSON.stringify({team}));
} catch (error) {
  process.stdout.write(JSON.stringify({
    error: error && error.message ? error.message : String(error),
  }));
}
"""
        stdout, _ = await self._run_node(
            ["-e", script, str(self.package_dir), format_id],
            timeout=timeout,
            input_data=text,
        )
        payload = self._decode_json_result(stdout, operation="队伍解析")
        team = payload.get("team")
        if (
            not isinstance(team, list)
            or not team
            or not all(isinstance(item, dict) for item in team)
        ):
            raise ShowdownTeamPackError("Showdown 未返回有效的结构化队伍。")
        return [dict(item) for item in team]

    async def load_team_builder_catalog(
        self,
        format_id: str,
        *,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        """Load format-aware canonical names and editor limits from Showdown."""
        self._ensure_layout()
        script = r"""
const path = require('path');
const root = process.argv[1];
const formatId = process.argv[2];
try {
  const {Dex} = require(path.join(root, 'dist', 'sim'));
  const {TeamValidator} = require(path.join(root, 'dist', 'sim', 'team-validator.js'));
  Dex.includeFormats();
  const validator = TeamValidator.get(formatId);
  const dex = validator.dex;
  const standard = entry => entry.exists && !['Future', 'Custom'].includes(entry.isNonstandard);
  process.stdout.write(JSON.stringify({
    rules: {
      minTeamSize: validator.ruleTable.minTeamSize,
      maxTeamSize: validator.ruleTable.maxTeamSize,
      pickedTeamSize: validator.ruleTable.pickedTeamSize,
      maxMoveCount: validator.ruleTable.maxMoveCount,
      minLevel: validator.ruleTable.minLevel,
      maxLevel: validator.ruleTable.maxLevel,
      defaultLevel: validator.ruleTable.defaultLevel,
      statValueLimit: validator.format.mod.startsWith('champions') ? 32 : 255,
      statTotalLimit: validator.ruleTable.evLimit,
      usesStatPoints: validator.format.mod.startsWith('champions'),
      supportsTera: !validator.format.mod.startsWith('champions'),
    },
    species: dex.species.all().filter(standard).map(entry => entry.name),
    items: dex.items.all().filter(standard).map(entry => entry.name),
    natures: dex.natures.all().map(entry => entry.name),
    types: dex.types.names(),
  }));
} catch (error) {
  process.stdout.write(JSON.stringify({
    error: error && error.message ? error.message : String(error),
  }));
}
"""
        stdout, _ = await self._run_node(
            ["-e", script, str(self.package_dir), format_id],
            timeout=timeout,
        )
        return self._decode_json_result(stdout, operation="编辑器数据加载")

    async def load_species_editor_options(
        self,
        format_id: str,
        species_name: str,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Return canonical species name, regular abilities and move pool."""
        self._ensure_layout()
        script = r"""
const path = require('path');
const root = process.argv[1];
const formatId = process.argv[2];
const speciesInput = process.argv[3];
try {
  const {Dex} = require(path.join(root, 'dist', 'sim'));
  const {TeamValidator} = require(path.join(root, 'dist', 'sim', 'team-validator.js'));
  Dex.includeFormats();
  const validator = TeamValidator.get(formatId);
  const dex = validator.dex;
  const species = dex.species.get(speciesInput);
  if (!species.exists) throw new Error(`Unknown species: ${speciesInput}`);
  const moves = [...dex.species.getMovePool(species.id)]
    .map(id => dex.moves.get(id))
    .filter(move => move.exists)
    .map(move => move.name)
    .sort((a, b) => a.localeCompare(b));
  const abilities = [...new Set(Object.values(species.abilities).filter(Boolean))];
  process.stdout.write(JSON.stringify({species: species.name, abilities, moves}));
} catch (error) {
  process.stdout.write(JSON.stringify({
    error: error && error.message ? error.message : String(error),
  }));
}
"""
        stdout, _ = await self._run_node(
            [
                "-e",
                script,
                str(self.package_dir),
                format_id,
                species_name,
            ],
            timeout=timeout,
        )
        return self._decode_json_result(stdout, operation="宝可梦编辑数据加载")

    @staticmethod
    def _decode_json_result(stdout: str, *, operation: str) -> dict[str, Any]:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ShowdownRuntimeUnavailable(
                f"Pokémon Showdown 返回了无效的{operation}结果。"
            ) from exc
        if not isinstance(payload, dict):
            raise ShowdownRuntimeUnavailable(
                f"Pokémon Showdown 返回了无效的{operation}结果。"
            )
        error = payload.get("error")
        if error:
            raise ShowdownTeamPackError(str(error))
        return payload

    async def export_team_json(
        self,
        format_id: str,
        team: list[dict[str, Any]],
        *,
        timeout: float = 15.0,
    ) -> str:
        """Convert trusted Showdown JSON sets to human-readable export format."""
        self._ensure_layout()
        if not format_id.strip():
            raise ShowdownTeamPackError("未指定队伍规则。")
        if not team or len(team) > 24:
            raise ShowdownTeamPackError("队伍必须包含 1 至 24 只宝可梦。")
        encoded = json.dumps(team, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 128 * 1024:
            raise ShowdownTeamPackError("JSON 队伍数据过大。")
        script = """
const path = require('path');
const root = process.argv[1];
const formatId = process.argv[2];
const team = JSON.parse(process.argv[3]);
const {Dex, Teams} = require(path.join(root, 'dist', 'sim'));
Dex.includeFormats();
const dex = Dex.forFormat(formatId);
for (const set of team) {
  const species = dex.species.get(set.species || set.name || '');
  if (!species.exists) throw new Error(`Unknown species: ${set.species || set.name}`);
  set.species = species.name;
  if (!set.ability) set.ability = species.abilities['0'] || '';
}
process.stdout.write(Teams.export(team));
"""
        try:
            stdout, _ = await self._run_node(
                ["-e", script, str(self.package_dir), format_id, encoded],
                timeout=timeout,
            )
        except ShowdownRuntimeUnavailable as exc:
            raise ShowdownTeamPackError(f"转换在线队伍失败：{exc}") from exc
        exported = stdout.strip()
        if not exported:
            raise ShowdownTeamPackError("Showdown 未能导出在线队伍。")
        return exported

    def create_battle_process(
        self,
        *,
        format_id: str,
        p1: dict[str, Any],
        p2: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> ShowdownBattleProcess:
        return ShowdownBattleProcess(
            runtime=self,
            format_id=format_id,
            p1=p1,
            p2=p2,
            logger=logger,
        )

    def load_move_payload(self) -> dict[str, dict[str, Any]]:
        """Load canonical move data in a worker thread during plugin startup."""
        self._ensure_layout()
        import subprocess

        script = """
const path = require('path');
const root = process.argv[1];
const moves = require(path.join(root, 'dist', 'data', 'moves.js')).Moves;
const movesText = require(path.join(root, 'dist', 'data', 'text', 'moves.js')).MovesText;
process.stdout.write(JSON.stringify({moves, movesText}));
"""
        try:
            completed = subprocess.run(
                [self.node_bin, "-e", script, str(self.package_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
                cwd=str(self.package_dir),
            )
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise ShowdownRuntimeUnavailable(
                f"加载 Showdown 招式数据失败：{exc}"
            ) from exc
        return {
            "moves": payload.get("moves") or {},
            "movesText": payload.get("movesText") or {},
        }

    def load_item_payload(self) -> dict[str, dict[str, Any]]:
        """Load canonical item names and sprite indices in a worker thread.

        Only ``name`` and ``spritenum`` are extracted: the sprite index is
        what locates an item's icon on Showdown's item-icon sprite sheet.
        """
        self._ensure_layout()
        import subprocess

        script = """
const path = require('path');
const root = process.argv[1];
const items = require(path.join(root, 'dist', 'data', 'items.js')).Items;
const out = {};
for (const id in items) {
  const item = items[id];
  out[id] = {name: item.name, spritenum: item.spritenum};
}
process.stdout.write(JSON.stringify({items: out}));
"""
        try:
            completed = subprocess.run(
                [self.node_bin, "-e", script, str(self.package_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
                cwd=str(self.package_dir),
            )
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise ShowdownRuntimeUnavailable(
                f"加载 Showdown 道具数据失败：{exc}"
            ) from exc
        return {"items": payload.get("items") or {}}

    def _ensure_layout(self) -> None:
        if not self.package_dir.is_dir():
            raise ShowdownRuntimeUnavailable(
                f"未找到 pokemon-showdown npm 包：{self.package_dir}。"
                "请先运行 npm ci --omit=optional --ignore-scripts。"
            )
        if not self.cli_script.is_file():
            raise ShowdownRuntimeUnavailable(
                f"未找到 pokemon-showdown CLI：{self.cli_script}"
            )
        if not (self.package_dir / "dist" / "sim" / "index.js").is_file():
            raise ShowdownRuntimeUnavailable(
                "pokemon-showdown 缺少 dist 构建产物，请重新运行 npm ci。"
            )

    async def _run_node(
        self,
        args: list[str],
        *,
        timeout: float,
        input_data: str | None = None,
    ) -> tuple[str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                self.node_bin,
                *args,
                stdin=(asyncio.subprocess.PIPE if input_data is not None else None),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.package_dir),
            )
        except OSError as exc:
            raise ShowdownRuntimeUnavailable(
                f"无法启动 Node.js（{self.node_bin}）：{exc}"
            ) from exc
        try:
            stdin_data = input_data.encode("utf-8") if input_data is not None else None
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_data),
                timeout,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise ShowdownRuntimeUnavailable("Node.js 运行时检查超时。") from exc
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise ShowdownRuntimeUnavailable(
                stderr_text.strip() or stdout_text.strip() or "Node.js 命令执行失败。"
            )
        return stdout_text, stderr_text

    async def _run_cli(
        self,
        cli_args: list[str],
        *,
        input_data: str,
        timeout: float,
    ) -> tuple[str, str, int]:
        self._ensure_layout()
        args = [
            self.node_bin,
            str(self.cli_script),
            *self.common_flags,
            *cli_args,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.package_dir),
                limit=1 << 18,
            )
        except OSError as exc:
            raise ShowdownRuntimeUnavailable(
                f"无法启动 Pokémon Showdown：{exc}"
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_data.encode("utf-8")), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise ShowdownBridgeError("Pokémon Showdown 命令执行超时。") from exc
        return (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            process.returncode or 0,
        )


class ShowdownBattleProcess:
    def __init__(
        self,
        *,
        runtime: ShowdownRuntime,
        format_id: str,
        p1: dict[str, Any],
        p2: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> None:
        self._runtime = runtime
        self._format_id = format_id
        self._p1 = p1
        self._p2 = p2
        self._logger = logger or logging.getLogger(__name__)
        self._process: Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._running = asyncio.Event()
        self._write_lock = asyncio.Lock()
        self._terminate_lock = asyncio.Lock()
        self._terminated = False

    @property
    def events(self) -> asyncio.Queue[dict[str, Any]]:
        return self._queue

    @property
    def returncode(self) -> int | None:
        return self._process.returncode if self._process else None

    async def start(self) -> None:
        self._runtime._ensure_layout()
        if self._process and self._process.returncode is None:
            raise ShowdownBridgeError("Battle process already running")
        args = [
            self._runtime.node_bin,
            str(self._runtime.cli_script),
            *self._runtime.common_flags,
            "simulate-battle",
        ]
        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._runtime.package_dir),
                limit=1 << 18,
            )
        except OSError as exc:
            raise ShowdownBridgeError(f"无法启动对战进程：{exc}") from exc

        self._reader_task = asyncio.create_task(
            self._read_stdout(), name=f"showdown-stdout-{self._format_id}"
        )
        self._stderr_task = asyncio.create_task(
            self._read_stderr(), name=f"showdown-stderr-{self._format_id}"
        )
        try:
            await self._write(f">start {json.dumps({'formatid': self._format_id})}")
            await self._write(f">player p1 {json.dumps(self._p1, ensure_ascii=False)}")
            await self._write(f">player p2 {json.dumps(self._p2, ensure_ascii=False)}")
        except Exception:
            await self.terminate()
            raise
        self._running.set()

    async def send_choice(self, side: str, choice: str) -> None:
        await self._running.wait()
        if choice == "forfeit":
            await self._write(f">forcelose {side}")
        else:
            await self._write(f">{side} {choice}")

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self._terminated:
                return
            self._terminated = True
            process = self._process
            current = asyncio.current_task()

            if process and process.stdin:
                try:
                    process.stdin.close()
                    await process.stdin.wait_closed()
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    AttributeError,
                    OSError,
                ):
                    pass

            if process and process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

            tasks = [
                task
                for task in (self._reader_task, self._stderr_task)
                if task is not None and task is not current
            ]
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self._queue.put({"type": "terminated"})

    async def _write(self, line: str) -> None:
        process = self._process
        if not process or not process.stdin or process.returncode is not None:
            raise ShowdownBridgeError("对战进程输入流不可用。")
        async with self._write_lock:
            try:
                process.stdin.write((line + "\n").encode("utf-8"))
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise ShowdownBridgeError("对战进程已退出。") from exc

    async def _read_stdout(self) -> None:
        assert self._process and self._process.stdout
        channel: str | None = None
        current_side: str | None = None
        split_state = 0
        try:
            while line_bytes := await self._process.stdout.readline():
                text = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                if not text:
                    channel = None
                    current_side = None
                    split_state = 0
                    await self._queue.put({"type": "separator"})
                    continue
                if text in {"update", "sideupdate", "requesteddata"}:
                    channel = text
                    split_state = 0
                    continue
                if channel == "sideupdate":
                    if current_side is None:
                        current_side = text.strip()
                        continue
                else:
                    current_side = None

                if channel == "update":
                    if text.startswith("|split|"):
                        # A split block is marker, player-only line, public line.
                        # Group narration must never receive the private first line.
                        split_state = 1
                        continue
                    if split_state == 1:
                        split_state = 2
                        continue
                    if split_state == 2:
                        split_state = 0

                if text.startswith("|request|"):
                    try:
                        payload = json.loads(text[len("|request|") :])
                    except json.JSONDecodeError as exc:
                        await self._queue.put(
                            {"type": "bridge_error", "message": str(exc)}
                        )
                    else:
                        await self._queue.put(
                            {
                                "type": "request",
                                "side": current_side,
                                "payload": payload,
                            }
                        )
                    continue
                if text.startswith("|error|"):
                    await self._queue.put(
                        {
                            "type": "error",
                            "side": current_side,
                            "message": text[len("|error|") :],
                        }
                    )
                    continue
                if channel != "update":
                    continue
                if text.startswith("|win|"):
                    await self._queue.put(
                        {"type": "win", "winner": text[len("|win|") :]}
                    )
                elif text == "|tie" or text.startswith("|tie|"):
                    await self._queue.put({"type": "tie"})
                elif text.startswith("|end"):
                    await self._queue.put({"type": "end"})
                elif text.startswith("|"):
                    await self._queue.put({"type": "update", "line": text})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.exception(f"读取 Showdown 输出失败: {exc}")
            await self._queue.put({"type": "bridge_error", "message": str(exc)})
        finally:
            await self._queue.put({"type": "stream_end"})

    async def _read_stderr(self) -> None:
        assert self._process and self._process.stderr
        try:
            while line_bytes := await self._process.stderr.readline():
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                self._logger.debug(f"[showdown stderr] {line}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.exception(f"读取 Showdown 错误流失败: {exc}")
