from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


logger = logging.getLogger(__name__)


def to_id(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("’", "'"))
    return "".join(ch for ch in normalized.lower() if ch.isascii() and ch.isalnum())


@dataclass(frozen=True, slots=True)
class TranslationCatalogInfo:
    schema_version: int
    pokemon_showdown_version: str
    pokeapi_commit: str
    champout_commit: str


class TranslationService:
    """Namespaced Simplified Chinese catalog for Pokémon Showdown entities."""

    def __init__(self, catalog: Mapping[str, Any]) -> None:
        meta = catalog.get("meta", {})
        self.info = TranslationCatalogInfo(
            schema_version=int(meta.get("schema_version", 0)),
            pokemon_showdown_version=str(
                meta.get("pokemon_showdown_version", "unknown")
            ),
            pokeapi_commit=str(meta.get("pokeapi_commit", "unknown")),
            champout_commit=str(meta.get("champout_commit", "unknown")),
        )
        if self.info.schema_version != 1:
            raise ValueError(
                f"unsupported translation catalog schema: {self.info.schema_version}"
            )

        self._species = self._load_namespace(catalog, "species")
        self._moves = self._load_namespace(catalog, "moves")
        self._items = self._load_namespace(catalog, "items")
        self._abilities = self._load_namespace(catalog, "abilities")
        self._types = self._load_namespace(catalog, "types")
        self._move_descriptions = self._load_namespace(catalog, "move_descriptions")
        self._misc = {
            str(key): str(value)
            for key, value in dict(catalog.get("misc", {})).items()
            if str(key).strip() and str(value).strip()
        }
        self._species_reverse = self._build_reverse(self._species)
        self._move_reverse = self._build_reverse(self._moves)
        self._item_reverse = self._build_reverse(self._items)
        self._ability_reverse = self._build_reverse(self._abilities)
        self._type_reverse = self._build_reverse(self._types)
        self._misc_reverse = self._build_reverse(self._misc)

    @staticmethod
    def _load_namespace(catalog: Mapping[str, Any], namespace: str) -> dict[str, str]:
        payload = catalog.get(namespace, {})
        if not isinstance(payload, Mapping):
            raise ValueError(f"translation namespace {namespace!r} must be an object")
        return {
            to_id(str(key)): str(value)
            for key, value in payload.items()
            if to_id(str(key)) and str(value).strip()
        }

    @staticmethod
    def _build_reverse(mapping: Mapping[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for entity_id, translated in mapping.items():
            result.setdefault(translated.strip(), entity_id)
        return result

    @classmethod
    def from_file(cls, path: Path) -> TranslationService:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"无法读取翻译目录：{path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"翻译目录 JSON 无效：{path}: {exc}") from exc
        service = cls(payload)
        logger.info(
            "loaded showdown translation catalog path=%s showdown=%s "
            "species=%d moves=%d abilities=%d items=%d types=%d",
            path,
            service.info.pokemon_showdown_version,
            len(service._species),
            len(service._moves),
            len(service._abilities),
            len(service._items),
            len(service._types),
        )
        return service

    def translate(self, text: str | None) -> str:
        if text is None:
            return ""
        stripped = text.strip()
        if not stripped:
            return text
        return self._misc.get(stripped, text)

    def translate_species(self, name: str | None) -> str:
        return self._translate_entity(self._species, name)

    def translate_move(self, move: str | None) -> str:
        return self._translate_entity(self._moves, move)

    def translate_item(self, item: str | None) -> str:
        return self._translate_entity(self._items, item)

    def translate_ability(self, ability: str | None) -> str:
        return self._translate_entity(self._abilities, ability)

    def translate_type(self, type_name: str | None) -> str:
        return self._translate_entity(self._types, type_name)

    def translate_move_description(
        self, move_id: str | None, fallback: str = ""
    ) -> str:
        if move_id:
            translated = self._move_descriptions.get(to_id(move_id))
            if translated:
                return translated
        if fallback:
            return self._misc.get(fallback.strip(), fallback)
        return ""

    def resolve_species_name(self, species: str | None) -> str | None:
        if species is None:
            return None
        stripped = species.strip()
        if not stripped:
            return None
        return self._species_reverse.get(stripped, stripped)

    def resolve_move_name(self, move: str | None) -> str | None:
        return self._resolve_entity(self._move_reverse, move)

    def resolve_item_name(self, item: str | None) -> str | None:
        return self._resolve_entity(self._item_reverse, item)

    def resolve_ability_name(self, ability: str | None) -> str | None:
        return self._resolve_entity(self._ability_reverse, ability)

    def resolve_type_name(self, type_name: str | None) -> str | None:
        return self._resolve_entity(self._type_reverse, type_name)

    def resolve_misc_name(self, value: str | None) -> str | None:
        return self._resolve_entity(self._misc_reverse, value)

    def translate_details(self, details: str | None) -> str:
        if not details:
            return ""
        parts = [segment.strip() for segment in details.split(",")]
        if not parts:
            return self.translate(details)
        parts[0] = self.translate_species(parts[0])
        return ", ".join(filter(None, parts))

    @staticmethod
    def _resolve_entity(
        reverse_mapping: Mapping[str, str], value: str | None
    ) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        return reverse_mapping.get(stripped, stripped)

    @staticmethod
    def _translate_entity(mapping: Mapping[str, str], value: str | None) -> str:
        if value is None:
            return ""
        stripped = value.strip()
        if not stripped:
            return value
        return mapping.get(to_id(stripped), value)
