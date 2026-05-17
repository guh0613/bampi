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

_CJK_STOPWORDS: set[str] = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "什么", "为什么", "怎么", "那个", "这个", "那些", "这些",
    "可以", "不是", "就是", "还是", "但是", "因为", "所以",
    "如果", "虽然", "而且", "或者", "已经", "可能", "应该",
    "比较", "一下", "一些", "这样", "那样", "怎样",
    "吗", "吧", "呢", "啊", "哦", "嗯", "呀", "哈",
}

_WEAK_NUMERIC_RE = re.compile(r"^\d{2}$")


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
        token = match.group(0)
        if for_query and _WEAK_NUMERIC_RE.match(token):
            continue
        terms.extend(_entity_variants(token))

    for match in _CJK_RE.finditer(normalized):
        sequence = match.group(0)
        if 1 < len(sequence) <= 8:
            terms.append(sequence)
        elif len(sequence) == 1 and not for_query:
            terms.append(sequence)
        ngrams = cjk_ngrams(sequence)
        if for_query:
            ngrams = [ng for ng in ngrams if ng not in _CJK_STOPWORDS]
        terms.extend(ngrams)

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
