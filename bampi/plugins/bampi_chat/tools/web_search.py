from __future__ import annotations

import asyncio
import html
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from bampy.app import tool


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, description="The search query to look up on the web")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum number of results to return")


def _normalize_result_url(raw_url: str) -> str | None:
    value = html.unescape(raw_url.strip())
    if not value:
        return None
    if value.startswith("//"):
        value = f"https:{value}"
    if value.startswith("/l/?") or "duckduckgo.com/l/?" in value:
        parsed = urlparse(value)
        encoded = parse_qs(parsed.query).get("uddg", [])
        if encoded:
            return unquote(encoded[0])
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return None


def parse_duckduckgo_results(markup: str, max_results: int) -> list[SearchResult]:
    matches = re.finditer(
        r"<a[^>]+href=[\"'](?P<url>[^\"']+)[\"'][^>]*>(?P<title>.*?)</a>",
        markup,
        flags=re.IGNORECASE | re.DOTALL,
    )

    results: list[SearchResult] = []
    seen: set[str] = set()
    for match in matches:
        url = _normalize_result_url(match.group("url"))
        if not url or url in seen:
            continue
        if "duckduckgo.com" in urlparse(url).netloc:
            continue
        title = re.sub(r"<[^>]+>", "", match.group("title"))
        title = " ".join(html.unescape(title).split())
        if not title:
            continue
        seen.add(url)
        results.append(SearchResult(title=title, url=url))
        if len(results) >= max_results:
            break
    return results


def _fetch_search_results(query: str, timeout: float) -> str:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; BampiBot/0.1; +https://example.invalid)",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def create_web_search_tool(timeout: float):
    @tool(
        name="web_search",
        description="Search the web for current external information and return a compact result list.",
        parameters=WebSearchInput,
    )
    async def web_search(query: str, max_results: int = 5) -> str:
        markup = await asyncio.to_thread(_fetch_search_results, query, timeout)
        results = parse_duckduckgo_results(markup, max_results)
        if not results:
            return f"No web search results found for: {query}"

        lines = [f"Web search results for: {query}"]
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. {result.title}")
            lines.append(f"   URL: {result.url}")
        return "\n".join(lines)

    return web_search
