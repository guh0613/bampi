from __future__ import annotations

import re
from collections.abc import Iterable

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_ENTITY_RE = re.compile(
    r"https?://[^\s`'\"<>]+"
    r"|[A-Za-z0-9][A-Za-z0-9_./:@%+#=-]{1,}"
    r"|(?<!\d)\d{2,5}(?!\d)"
)
_TRIM_CHARS = " \t\r\n,.;:!?()[]{}<>\"'`，。！？、；：（）【】《》"


def normalize_for_search(text: object) -> str:
    return " ".join(str(text or "").split())


def cjk_ngrams(text: str, *, sizes: tuple[int, ...] = (2, 3)) -> list[str]:
    result: list[str] = []
    for size in sizes:
        if len(text) < size:
            continue
        result.extend(text[index : index + size] for index in range(len(text) - size + 1))
    return result


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = item.strip(_TRIM_CHARS)
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _entity_variants(token: str) -> list[str]:
    normalized = token.strip(_TRIM_CHARS).casefold()
    if not normalized:
        return []

    variants = [normalized]
    variants.extend(part for part in normalized.split("/") if "." in part and len(part) >= 2)
    split_parts = [
        part
        for part in re.split(r"[^a-z0-9_]+", normalized)
        if len(part) >= 2
    ]
    variants.extend(split_parts)
    return variants


def extract_search_terms(
    text: object,
    *,
    for_query: bool = False,
    max_terms: int = 80,
) -> list[str]:
    normalized = normalize_for_search(text)
    if not normalized:
        return []

    terms: list[str] = []
    for match in _ENTITY_RE.finditer(normalized):
        terms.extend(_entity_variants(match.group(0)))

    for match in _CJK_RE.finditer(normalized):
        sequence = match.group(0)
        if 1 < len(sequence) <= 8:
            terms.append(sequence)
        elif len(sequence) == 1 and not for_query:
            terms.append(sequence)
        terms.extend(cjk_ngrams(sequence))

    return _dedupe(terms)[:max_terms]


def build_search_text(*parts: object) -> str:
    raw_parts = [normalize_for_search(part) for part in parts if normalize_for_search(part)]
    terms: list[str] = []
    for part in raw_parts:
        terms.extend(extract_search_terms(part, for_query=False))
    token_block = " ".join(_dedupe(terms))
    if token_block:
        raw_parts.append(token_block)
    return "\n".join(raw_parts)


def build_fts_query(query: str, *, max_terms: int = 48) -> str:
    terms = extract_search_terms(query, for_query=True, max_terms=max_terms)
    if not terms:
        return ""
    return " OR ".join(_quote_fts_term(term) for term in terms)


def like_terms(query: str, *, max_terms: int = 16) -> list[str]:
    normalized = normalize_for_search(query)
    terms = extract_search_terms(normalized, for_query=True, max_terms=max_terms)
    if len(normalized) >= 2:
        terms.insert(0, normalized)
    return _dedupe(terms)[:max_terms]


def required_entity_groups(query: str, *, max_groups: int = 8) -> list[list[str]]:
    groups: list[list[str]] = []
    for match in _ENTITY_RE.finditer(normalize_for_search(query)):
        variants = _entity_variants(match.group(0))
        if variants:
            groups.append(_dedupe(variants))
        if len(groups) >= max_groups:
            break
    return groups


def _quote_fts_term(term: str) -> str:
    escaped = term.replace('"', '""')
    return f'"{escaped}"'
