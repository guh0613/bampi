from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field

from bampy.app import tool

DEFAULT_WEB_SEARCH_MODEL = "grok-4.20-beta"
DEFAULT_WEB_SEARCH_USER_AGENT = "Mozilla/5.0 (compatible; BampiBot/0.1; +https://example.invalid)"


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, description="The search query to look up on the web")


def _normalize_base_url(base_url: str) -> str:
    value = base_url.strip()
    if not value:
        raise ValueError("web_search is not configured: bampi_base_url is empty")
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


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _format_browse_page_line(line: str) -> str:
    payload_text = line[len("browse_page "):].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return line
    url = payload.get("url")
    if isinstance(url, str) and url.strip():
        return f"browse_page: {url.strip()}"
    return line


def _extract_useful_thinking_lines(thinking: str) -> list[str]:
    useful: list[str] = []
    for raw_line in thinking.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue

        normalized = line.lstrip("-* ").strip()
        low = normalized.lower()

        if normalized.startswith("[WebSearch]"):
            useful.append(normalized)
            continue
        if normalized.startswith("browse_page "):
            useful.append(_format_browse_page_line(normalized))
            continue
        if "web_search tool" in low:
            useful.append("tool_call: web_search")
            continue
        if low.startswith("planning to "):
            continue
        if low.startswith(("searching ", "checking ", "browsing ", "opening ")):
            useful.append(normalized)
            continue
        if re.search(r"https?://\S+", normalized):
            useful.append(normalized)

    return _dedupe_preserve_order(useful)


def _compact_response_text(text: str) -> str:
    trace_items: list[str] = []

    def _replace_thinking(match: re.Match[str]) -> str:
        trace_items.extend(_extract_useful_thinking_lines(match.group(1)))
        return ""

    answer = re.sub(r"<think>(.*?)</think>", _replace_thinking, text, flags=re.DOTALL).strip()
    trace_items = _dedupe_preserve_order(trace_items)
    if not trace_items:
        return answer or text.strip()

    trace_block = "\n".join(["Search trace:"] + [f"- {item}" for item in trace_items])
    if answer:
        return f"{trace_block}\n\n{answer}"
    return trace_block


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
    user_agent: str = DEFAULT_WEB_SEARCH_USER_AGENT,
) -> str:
    if not api_key.strip():
        raise ValueError("web_search is not configured: bampi_api_key is empty")

    url = _normalize_base_url(base_url) + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": user_agent,
    }
    payload = {
        "model": DEFAULT_WEB_SEARCH_MODEL,
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


def create_web_search_tool(
    timeout: float,
    *,
    base_url: str,
    api_key: str,
    user_agent: str = DEFAULT_WEB_SEARCH_USER_AGENT,
):
    @tool(
        name="web_search",
        description="Search the web for current external information and return a concise answer with sources.",
        parameters=WebSearchInput,
    )
    async def web_search(query: str) -> str:
        try:
            answer = await _query_search_agent(
                query,
                timeout=timeout,
                base_url=base_url,
                api_key=api_key,
                user_agent=user_agent,
            )
        except Exception as exc:
            return f"Web search failed for: {query}\nError: {exc}"
        return f"Web search results for: {query}\n{answer}"

    return web_search
