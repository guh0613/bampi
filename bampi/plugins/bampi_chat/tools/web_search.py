from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field

from bampy.app import tool

DEFAULT_WEB_SEARCH_MODEL = "grok-4.20-auto"
DEFAULT_WEB_SEARCH_USER_AGENT = "Mozilla/5.0 (compatible; BampiBot/0.1; +https://example.invalid)"


class WebAskInput(BaseModel):
    query: str = Field(min_length=1, description="The question to ask the web-enabled AI agent")


class WebSearchInput(BaseModel):
    query: str = Field(
        min_length=1,
        description="A natural-language description of what you are looking for",
    )
    category: str | None = Field(
        default=None,
        description=(
            "Optional category filter to narrow results. "
            "One of: company, research paper, news, personal site, financial report, people"
        ),
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Only return results from these domains, e.g. ['github.com', 'arxiv.org']",
    )
    num_results: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of results to return (default 5). Fewer results + more text per result for deep reading; more results + less text for broad discovery.",
    )
    max_text_chars: int | None = Field(
        default=None,
        ge=500,
        le=10000,
        description="Max characters of page text per result (default 2000). Increase for detailed docs, decrease for quick lookups.",
    )


def _normalize_base_url(base_url: str) -> str:
    value = base_url.strip()
    if not value:
        raise ValueError("web_search is not configured: bampi_web_search_base_url is empty")
    if not value.rstrip("/").endswith("/v1"):
        return value.rstrip("/") + "/v1"
    return value.rstrip("/")


def _extract_text_parts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        if "text" in value:
            return _extract_text_parts(value["text"])
        if "content" in value:
            return _extract_text_parts(value["content"])
        return []
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_extract_text_parts(item))
        return parts

    text = getattr(value, "text", None)
    if text is not None:
        return _extract_text_parts(text)

    content = getattr(value, "content", None)
    if content is not None:
        return _extract_text_parts(content)

    if hasattr(value, "model_dump"):
        try:
            return _extract_text_parts(value.model_dump(exclude_none=True))
        except TypeError:
            return _extract_text_parts(value.model_dump())
    return []


def _extract_message_text(message: Any) -> str:
    parts = _extract_text_parts(getattr(message, "content", None))
    if not parts and hasattr(message, "model_dump"):
        try:
            dump = message.model_dump(exclude_none=True)
        except TypeError:
            dump = message.model_dump()
        parts = _extract_text_parts(dump.get("content"))
    return "\n".join(part for part in parts if part).strip()


def _compact_response_text(text: str) -> str:
    answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return answer or text.strip()


async def _read_sse_text(response: httpx.Response) -> str:
    chunks: list[str] = []
    async for line in response.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            chunks.extend(_extract_text_parts(delta.get("content")))
            message = choice.get("message")
            if message is not None:
                chunks.extend(_extract_text_parts(message.get("content")))
    return "".join(chunks).strip()


async def _query_search_agent(
    query: str,
    *,
    timeout: float,
    base_url: str,
    api_key: str,
    model: str = DEFAULT_WEB_SEARCH_MODEL,
    user_agent: str = DEFAULT_WEB_SEARCH_USER_AGENT,
) -> str:
    if not api_key.strip():
        raise ValueError("web_search is not configured: bampi_web_search_api_key is empty")

    url = _normalize_base_url(base_url) + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": user_agent,
    }
    payload = {
        "model": model.strip() or DEFAULT_WEB_SEARCH_MODEL,
        "messages": [
            {
                "role": "user",
                "content": query.strip(),
            },
        ],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text[:400].strip()
                if detail:
                    raise RuntimeError(
                        f"search agent request failed with status {response.status_code}: {detail}"
                    ) from exc
                raise RuntimeError(
                    f"search agent request failed with status {response.status_code}"
                ) from exc

            content_type = (response.headers.get("content-type") or "").lower()
            if "text/event-stream" in content_type:
                answer = await _read_sse_text(response)
            else:
                body = await response.aread()
                if "application/json" in content_type:
                    data = json.loads(body)
                    choices = data.get("choices") or []
                    if not choices:
                        raise RuntimeError("search agent returned no choices")
                    answer = _extract_message_text(choices[0].get("message"))
                else:
                    answer = body.decode("utf-8", errors="replace").strip()

    answer = _compact_response_text(answer)
    if not answer:
        raise RuntimeError("search agent returned no answer text")
    return answer


def create_web_ask_tool(
    timeout: float,
    *,
    base_url: str,
    api_key: str,
    model: str = DEFAULT_WEB_SEARCH_MODEL,
    user_agent: str = DEFAULT_WEB_SEARCH_USER_AGENT,
):
    @tool(
        name="web_ask",
        description=(
            "Search the web via an AI agent with broad, real-time retrieval capabilities. "
            "Reaches more sources and fresher information than web_search, "
            "but returns a summarized answer instead of raw page content. "
            "Best when you need the widest net or the most up-to-date results. "
            "Use web_search instead when you need original page text, exact quotes, or source URLs."
        ),
        parameters=WebAskInput,
    )
    async def web_ask(query: str) -> str:
        try:
            answer = await _query_search_agent(
                query,
                timeout=timeout,
                base_url=base_url,
                api_key=api_key,
                model=model,
                user_agent=user_agent,
            )
        except Exception as exc:
            return f"Web ask failed for: {query}\nError: {exc}"
        return answer

    return web_ask


EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_VALID_CATEGORIES = frozenset(
    {"company", "research paper", "news", "personal site", "financial report", "people"}
)


def _format_exa_results(results: list[dict], query: str) -> str:
    if not results:
        return f"No results found for: {query}"

    parts: list[str] = []
    for i, item in enumerate(results, 1):
        title = item.get("title", "")
        url = item.get("url", "")
        header = f"[{i}] {title}\n    {url}"

        extras: list[str] = []
        if author := item.get("author"):
            extras.append(f"Author: {author}")
        if date := item.get("publishedDate"):
            extras.append(f"Published: {date[:10]}")

        block = header
        if extras:
            block += "\n    " + " | ".join(extras)
        if highlights := item.get("highlights"):
            block += "\n" + "\n".join(f"  > {h.strip()}" for h in highlights if h.strip())
        if text := item.get("text"):
            block += "\n" + text.strip()
        parts.append(block)

    return "\n\n".join(parts)


def create_web_search_tool(
    *,
    api_key: str,
    timeout: float = 20.0,
    default_num_results: int = 5,
):
    @tool(
        name="web_search",
        description=(
            "Search the web and retrieve actual page content with source URLs. "
            "Returns original text and highlights from real web pages. "
            "Best for finding documentation, official information, articles, product specs, "
            "or any query where you need verifiable sources and raw content. "
            "Use natural-language queries describing what you want to find — NOT keyword strings or search-engine syntax."
        ),
        parameters=WebSearchInput,
    )
    async def web_search(
        query: str,
        category: str | None = None,
        include_domains: list[str] | None = None,
        num_results: int | None = None,
        max_text_chars: int | None = None,
    ) -> str:
        if not api_key.strip():
            return "web_search is not configured: exa api key is empty"

        n = min(max(num_results or default_num_results, 1), 20)
        text_chars = min(max(max_text_chars or 2000, 500), 10000)

        body: dict = {
            "query": query.strip(),
            "type": "auto",
            "numResults": n,
            "contents": {
                "text": {"maxCharacters": text_chars, "verbosity": "compact"},
                "highlights": True,
            },
        }
        if category and category in EXA_VALID_CATEGORIES:
            body["category"] = category
        if include_domains:
            body["includeDomains"] = include_domains[:50]

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": api_key.strip(),
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(EXA_SEARCH_URL, headers=headers, json=body)
                resp.raise_for_status()
            data = resp.json()
            return _format_exa_results(data.get("results", []), query)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400].strip()
            return f"Web search failed ({exc.response.status_code}): {detail}"
        except Exception as exc:
            return f"Web search failed: {exc}"

    return web_search
