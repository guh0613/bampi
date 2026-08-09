from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = PROJECT_ROOT / "bampi/plugins/showdown_battle/assets/i18n/zh_hans.json"
DEFAULT_OVERRIDES = (
    PROJECT_ROOT / "bampi/plugins/showdown_battle/assets/i18n/overrides.zh_hans.json"
)
DEFAULT_SHOWDOWN_PACKAGE = PROJECT_ROOT / "node_modules/pokemon-showdown"

POKEAPI_COMMIT = "fb1605aac09064bb34a12a8b790c2b800b4d0550"
CHAMPOUT_COMMIT = "0c1141656e1a66ae304ac3ee1e7126a00914d1f2"
SHOWDOWN_VERSION = "0.11.11"

POKEAPI_BASE = (
    f"https://raw.githubusercontent.com/PokeAPI/pokeapi/{POKEAPI_COMMIT}/data/v2/csv"
)
CHAMPOUT_BASE = (
    "https://raw.githubusercontent.com/projectpokemon/champout/"
    f"{CHAMPOUT_COMMIT}/rom-txt"
)

CATEGORY_FILES = {
    "species": "monsname_syn.json",
    "forms": "zkn_form_syn.json",
    "moves": "wazaname.json",
    "abilities": "tokusei.json",
    "items": "itemname.json",
    "types": "typename.json",
}
DESCRIPTION_FILES = {
    "moves": "wazainfo_syn.json",
    "abilities": "tokuseiinfo_syn.json",
    "items": "iteminfo_syn.json",
}
POKEAPI_FILES = {
    "species": ("pokemon_species_names.csv", "pokemon_species_id"),
    "moves": ("move_names.csv", "move_id"),
    "abilities": ("ability_names.csv", "ability_id"),
    "items": ("item_names.csv", "item_id"),
    "types": ("type_names.csv", "type_id"),
}


def to_id(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("’", "'"))
    return "".join(ch for ch in normalized.lower() if ch.isascii() and ch.isalnum())


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "bampi-showdown-i18n-generator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def load_json_url(url: str) -> dict[str, Any]:
    return json.loads(fetch_bytes(url))


def load_showdown_manifest(package_dir: Path) -> dict[str, list[dict[str, Any]]]:
    script = r"""
const path = require('path');
const root = process.argv[1];
const {Dex} = require(path.join(root, 'dist', 'sim'));
const out = {};
for (const kind of ['species', 'moves', 'abilities', 'items', 'types']) {
  out[kind] = Dex[kind].all().map(entry => ({
    id: entry.id,
    name: entry.name,
    num: entry.num,
    baseSpecies: entry.baseSpecies || '',
    forme: entry.forme || '',
    isNonstandard: entry.isNonstandard || null,
    shortDesc: entry.shortDesc || entry.desc || '',
  }));
}
const pkg = require(path.join(root, 'package.json'));
out.version = pkg.version;
process.stdout.write(JSON.stringify(out));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(package_dir.resolve())],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    version = str(payload.pop("version", ""))
    if version != SHOWDOWN_VERSION:
        raise RuntimeError(
            f"Expected pokemon-showdown {SHOWDOWN_VERSION}, found {version}"
        )
    return payload


def load_pokeapi_maps() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for category, (filename, id_column) in POKEAPI_FILES.items():
        text = fetch_bytes(f"{POKEAPI_BASE}/{filename}").decode("utf-8")
        grouped: dict[str, dict[str, str]] = {}
        for row in csv.DictReader(io.StringIO(text)):
            grouped.setdefault(row[id_column], {})[row["local_language_id"]] = row[
                "name"
            ]
        mapping: dict[str, str] = {}
        for names in grouped.values():
            english = names.get("9", "").strip()
            chinese = names.get("12", "").strip()
            if english and chinese:
                mapping.setdefault(to_id(english), chinese)
        result[category] = mapping
    return result


def _champout_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["LabelName"]): row
        for row in payload.get("mSDataSet", [])
        if row.get("LabelName")
    }


def load_champout_maps() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    names: dict[str, dict[str, str]] = {}
    descriptions: dict[str, dict[str, str]] = {}

    for category, filename in CATEGORY_FILES.items():
        english_rows = _champout_rows(load_json_url(f"{CHAMPOUT_BASE}/usa/{filename}"))
        chinese_rows = _champout_rows(load_json_url(f"{CHAMPOUT_BASE}/sch/{filename}"))
        mapping: dict[str, str] = {}
        for label, english_row in english_rows.items():
            chinese_row = chinese_rows.get(label)
            if not chinese_row:
                continue
            english = str(english_row.get("OriginalText") or "").strip()
            chinese = str(chinese_row.get("OriginalText") or "").strip()
            if english and chinese:
                mapping.setdefault(to_id(english), chinese)
        names[category] = mapping

    for category, filename in DESCRIPTION_FILES.items():
        english_rows = _champout_rows(load_json_url(f"{CHAMPOUT_BASE}/usa/{filename}"))
        chinese_rows = _champout_rows(load_json_url(f"{CHAMPOUT_BASE}/sch/{filename}"))
        name_filename = CATEGORY_FILES[category]
        name_rows = _champout_rows(
            load_json_url(f"{CHAMPOUT_BASE}/usa/{name_filename}")
        )
        mapping: dict[str, str] = {}
        for name_label, name_row in name_rows.items():
            suffix = name_label.rsplit("_", 1)[-1]
            info_label = next(
                (label for label in english_rows if label.rsplit("_", 1)[-1] == suffix),
                None,
            )
            if not info_label:
                continue
            chinese_row = chinese_rows.get(info_label)
            if not chinese_row:
                continue
            english_name = str(name_row.get("OriginalText") or "").strip()
            chinese_description = str(chinese_row.get("OriginalText") or "").strip()
            if english_name and chinese_description:
                mapping.setdefault(
                    to_id(english_name),
                    re.sub(r"\s+", " ", chinese_description),
                )
        descriptions[category] = mapping

    return names, descriptions


def load_legacy_mapping(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    match = re.search(r"var\s+translations\s*=", text)
    if not match:
        raise RuntimeError(f"No translations object found in {path}")
    start = text.find("{", match.end())
    if start < 0:
        raise RuntimeError(f"No translations object found in {path}")
    depth = 0
    end = None
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise RuntimeError(f"Unclosed translations object in {path}")
    block = re.sub(r",\s*([}\]])", r"\1", text[start:end])
    parsed = json.loads(block)
    return {
        str(key): str(value)
        for key, value in parsed.items()
        if str(key).strip() and str(value).strip()
    }


def load_existing_catalog(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def species_candidates(entry: dict[str, Any]) -> list[str]:
    candidates = [str(entry["name"])]
    base = str(entry.get("baseSpecies") or "")
    forme = str(entry.get("forme") or "")
    if base and forme.startswith("Mega"):
        suffix = forme.removeprefix("Mega").lstrip("-")
        candidates.insert(0, f"Mega {base}{f' {suffix}' if suffix else ''}")
    return candidates


def generate_catalog(
    *,
    package_dir: Path,
    output_path: Path,
    overrides_path: Path,
    legacy_path: Path | None,
) -> dict[str, Any]:
    manifest = load_showdown_manifest(package_dir)
    pokeapi = load_pokeapi_maps()
    champout, champout_descriptions = load_champout_maps()
    legacy = load_legacy_mapping(legacy_path)
    existing = load_existing_catalog(output_path)
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))

    catalog: dict[str, Any] = {
        "meta": {
            "schema_version": 1,
            "pokemon_showdown_version": SHOWDOWN_VERSION,
            "pokeapi_commit": POKEAPI_COMMIT,
            "champout_commit": CHAMPOUT_COMMIT,
            "priority": ["overrides", "champout", "pokeapi", "legacy", "existing"],
        }
    }
    source_counts: dict[str, Counter[str]] = {}

    for category in ("species", "moves", "abilities", "items", "types"):
        category_catalog: dict[str, str] = {}
        counts: Counter[str] = Counter()
        previous = existing.get(category, {})
        category_overrides = overrides.get(category, {})
        champ_category = champout.get(category, {})
        if category == "species":
            champ_category = {**champ_category, **champout.get("forms", {})}
        poke_category = pokeapi.get(category, {})

        for entry in manifest[category]:
            entity_id = str(entry["id"])
            candidates = (
                species_candidates(entry)
                if category == "species"
                else [str(entry["name"])]
            )
            translated = category_overrides.get(entity_id)
            source = "overrides"
            if not translated:
                translated = next(
                    (
                        champ_category.get(to_id(candidate))
                        for candidate in candidates
                        if champ_category.get(to_id(candidate))
                    ),
                    None,
                )
                source = "champout"
            if not translated:
                translated = next(
                    (
                        poke_category.get(to_id(candidate))
                        for candidate in candidates
                        if poke_category.get(to_id(candidate))
                    ),
                    None,
                )
                source = "pokeapi"
            if not translated:
                translated = next(
                    (
                        legacy.get(candidate)
                        for candidate in candidates
                        if legacy.get(candidate)
                    ),
                    None,
                )
                source = "legacy"
            if not translated:
                translated = previous.get(entity_id)
                source = "existing"
            if translated and translated not in candidates:
                category_catalog[entity_id] = str(translated)
                counts[source] += 1
            else:
                counts["missing"] += 1

        catalog[category] = dict(sorted(category_catalog.items()))
        source_counts[category] = counts

    move_descriptions: dict[str, str] = {}
    previous_descriptions = existing.get("move_descriptions", {})
    description_overrides = overrides.get("move_descriptions", {})
    description_counts: Counter[str] = Counter()
    for entry in manifest["moves"]:
        move_id = str(entry["id"])
        name = str(entry["name"])
        translated = description_overrides.get(move_id)
        source = "overrides"
        if not translated:
            translated = champout_descriptions.get("moves", {}).get(to_id(name))
            source = "champout"
        if not translated:
            english_description = str(entry.get("shortDesc") or "").strip()
            translated = legacy.get(english_description)
            source = "legacy"
        if not translated:
            translated = previous_descriptions.get(move_id)
            source = "existing"
        if translated:
            move_descriptions[move_id] = re.sub(r"\s+", " ", str(translated)).strip()
            description_counts[source] += 1
        else:
            description_counts["missing"] += 1
    catalog["move_descriptions"] = dict(sorted(move_descriptions.items()))

    misc = dict(existing.get("misc", {}))
    misc.update(legacy)
    misc.update(overrides.get("misc", {}))
    catalog["misc"] = dict(sorted(misc.items()))
    catalog["meta"]["source_counts"] = {
        **{key: dict(value) for key, value in source_counts.items()},
        "move_descriptions": dict(description_counts),
    }
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Showdown zh-Hans catalog")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_SHOWDOWN_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--legacy", type=Path)
    args = parser.parse_args()

    catalog = generate_catalog(
        package_dir=args.package_dir,
        output_path=args.output,
        overrides_path=args.overrides,
        legacy_path=args.legacy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = catalog["meta"]["source_counts"]
    print(f"wrote {args.output}")
    for category, values in counts.items():
        print(category, values)


if __name__ == "__main__":
    main()
