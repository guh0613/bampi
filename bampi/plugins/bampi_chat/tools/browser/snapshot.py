from __future__ import annotations

from collections import Counter
from contextlib import suppress
import re
from typing import Any

from .errors import CommandError
from .interaction import InteractionEngine
from .models import PageState, RefEntry, SnapshotResult
from .runtime import BrowserRuntime


INTERACTIVE_ROLES = {
    "button", "checkbox", "combobox", "gridcell", "link", "listbox", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "radio", "scrollbar", "searchbox",
    "slider", "spinbutton", "switch", "tab", "textbox", "treeitem",
}
CONTENT_ROLES = {"article", "cell", "heading", "img", "listitem", "region", "row"}
SKIP_ROLES = {"Iframe", "InlineTextBox", "none", "presentation"}

_CURSOR_SCRIPT = r"""
(() => {
  const limit = %d;
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const s = getComputedStyle(el);
    return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity || 1) > 0.01;
  };
  const nativeInteractive = new Set(['A','BUTTON','INPUT','SELECT','TEXTAREA','SUMMARY','OPTION']);
  const result = [];
  const seen = new Set();
  const add = (el) => { if (el && !seen.has(el) && result.length < limit) { seen.add(el); result.push(el); } };
  for (const label of document.querySelectorAll('label')) {
    const control = label.control || label.querySelector('input[type=checkbox],input[type=radio]');
    if (!control || !['checkbox','radio'].includes(control.type) || !visible(label)) continue;
    const r = control.getBoundingClientRect();
    const s = getComputedStyle(control);
    if (r.width < 2 || r.height < 2 || s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity || 1) < 0.01) add(label);
  }
  let i = 0;
  for (const el of document.querySelectorAll('*')) {
    if (result.length >= limit) break;
    if (!visible(el) || nativeInteractive.has(el.tagName)) continue;
    const cursor = getComputedStyle(el).cursor;
    const parentCursor = el.parentElement && getComputedStyle(el.parentElement).cursor;
    const ownPointer = !['auto','default','text','inherit','initial','unset'].includes(cursor);
    const parentPointer = parentCursor === cursor;
    const interactive = el.hasAttribute('onclick') || el.isContentEditable ||
      (el.hasAttribute('tabindex') && el.tabIndex >= 0) || (ownPointer && !parentPointer);
    if (!interactive) continue;
    add(el); i++;
  }
  return result;
})()
"""


def _ax_value(node: dict[str, Any], key: str) -> Any:
    value = node.get(key)
    if isinstance(value, dict):
        return value.get("value")
    return value


def _clean(value: Any, limit: int = 180) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", str(value or ""))
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


class SnapshotEngine:
    def __init__(self, runtime: BrowserRuntime) -> None:
        self.runtime = runtime

    async def capture(
        self,
        page: PageState,
        *,
        max_nodes: int = 180,
        interactive: bool = False,
        max_depth: int | None = None,
        scope: str | None = None,
        show_urls: bool = False,
    ) -> SnapshotResult:
        page.snapshot_sequence += 1
        refs: dict[str, RefEntry] = {}
        lines: list[str] = []
        next_ref = 1
        node_count = 0
        scope_session: str | None = None
        allowed_backend_ids: set[int] | None = None
        if scope:
            scope_session, allowed_backend_ids = await self._scope_backend_ids(page, scope)

        if scope_session in {None, page.session_id}:
            main = await self._capture_tree(
                page, page.session_id, frame_id=None, params={}, refs=refs,
                next_ref=next_ref, max_nodes=max_nodes, include_cursor=True,
                interactive=interactive, max_depth=max_depth, show_urls=show_urls,
                allowed_backend_ids=allowed_backend_ids,
            )
            lines.extend(main[0])
            next_ref, node_count = main[1], main[2]

        with suppress(Exception):
            frame_tree = (await self.runtime.client.call("Page.getFrameTree", session_id=page.session_id)).get("frameTree", {})
            child_frames = self._flatten_child_frames(frame_tree)
            for frame_id, frame_url in child_frames:
                if node_count >= max_nodes:
                    break
                effective_session = self.runtime.frame_sessions.get(frame_id, page.session_id)
                if scope_session is not None and effective_session != scope_session:
                    continue
                params = {} if effective_session != page.session_id else {"frameId": frame_id}
                try:
                    child = await self._capture_tree(
                        page, effective_session, frame_id=frame_id, params=params, refs=refs,
                        next_ref=next_ref, max_nodes=max_nodes - node_count, include_cursor=True,
                        interactive=interactive, max_depth=max_depth, show_urls=show_urls,
                        allowed_backend_ids=allowed_backend_ids,
                    )
                except Exception:
                    continue
                if child[0]:
                    lines.append(f'- iframe "{_clean(frame_url, 120)}"')
                    lines.extend("  " + line for line in child[0])
                    next_ref, added = child[1], child[2]
                    node_count += added

        page.refs = refs
        header = f'page {page.page_id} "{_clean(page.title, 100)}" {self.runtime.policy.display_url(page.url)}'
        if node_count >= max_nodes:
            lines.append(f"- … snapshot truncated at {max_nodes} nodes")
        return SnapshotResult("\n".join([header, *lines]), refs, node_count)

    async def _capture_tree(
        self,
        page: PageState,
        session_id: str,
        *,
        frame_id: str | None,
        params: dict[str, Any],
        refs: dict[str, RefEntry],
        next_ref: int,
        max_nodes: int,
        include_cursor: bool,
        interactive: bool,
        max_depth: int | None,
        show_urls: bool,
        allowed_backend_ids: set[int] | None,
    ) -> tuple[list[str], int, int]:
        cursor = await self._cursor_elements(session_id, page.snapshot_sequence, frame_id) if include_cursor else {}
        result = await self.runtime.client.call(
            "Accessibility.getFullAXTree", params, session_id=session_id,
            timeout=self.runtime.config.action_timeout,
        )
        raw_nodes = [node for node in result.get("nodes", []) if isinstance(node, dict)]
        by_id = {str(node.get("nodeId")): node for node in raw_nodes}
        child_ids = {str(child) for node in raw_nodes for child in node.get("childIds", [])}
        roots = [str(node.get("nodeId")) for node in raw_nodes if str(node.get("nodeId")) not in child_ids]
        duplicate_counts = Counter(
            (_clean(_ax_value(node, "role"), 60), _clean(_ax_value(node, "name"), 180))
            for node in raw_nodes
        )
        role_seen: Counter[tuple[str, str]] = Counter()
        rendered_backend: set[int] = set()
        lines: list[str] = []
        rendered = 0

        def visit(node_id: str, depth: int, parent_name: str = "") -> None:
            nonlocal next_ref, rendered
            if rendered >= max_nodes or (max_depth is not None and depth > max_depth):
                return
            node = by_id.get(node_id)
            if node is None:
                return
            role = _clean(_ax_value(node, "role"), 60)
            name = _clean(_ax_value(node, "name"), 180)
            ignored = bool(node.get("ignored")) or role in SKIP_ROLES
            children = [str(item) for item in node.get("childIds", [])]
            if ignored:
                for child in children:
                    visit(child, depth, parent_name)
                return

            backend = node.get("backendDOMNodeId")
            backend_id = int(backend) if isinstance(backend, (int, float)) else None
            cursor_info = cursor.get(backend_id) if backend_id is not None else None
            if cursor_info is not None and role in {"", "generic"}:
                role = _clean(cursor_info.get("role"), 60) or "clickable"
                name = name or _clean(cursor_info.get("label"), 180)
            should_ref = backend_id is not None and (
                role in INTERACTIVE_ROLES or (role in CONTENT_ROLES and bool(name)) or cursor_info is not None
            )
            in_scope = allowed_backend_ids is None or backend_id is None or backend_id in allowed_backend_ids
            should_render = in_scope and (not interactive or should_ref)
            ref_text = ""
            if should_ref and backend_id is not None and in_scope:
                ref = f"@e{next_ref}"
                next_ref += 1
                key = (role, name)
                nth = role_seen[key]
                role_seen[key] += 1
                refs[ref] = RefEntry(
                    ref=ref,
                    page_id=page.page_id,
                    session_id=session_id,
                    backend_node_id=backend_id,
                    document_generation=page.document_generation,
                    session_generation=page.session_generations.get(session_id, 0),
                    role=role or "clickable",
                    name=name or _clean((cursor_info or {}).get("label"), 180),
                    frame_id=frame_id,
                    nth=nth,
                )
                ref_text = f" [{ref}]"
                rendered_backend.add(backend_id)

            properties: list[str] = []
            value = _clean(_ax_value(node, "value"), 100)
            if value and value != name:
                properties.append(f'value="{value}"')
            for prop in node.get("properties", []):
                if not isinstance(prop, dict):
                    continue
                prop_name = str(prop.get("name") or "")
                if prop_name == "url" and show_urls:
                    prop_value = _clean(_ax_value(prop, "value"), 240)
                    if prop_value:
                        properties.append(f'url="{prop_value}"')
                    continue
                if prop_name not in {"checked", "disabled", "expanded", "focused", "level", "pressed", "required", "selected"}:
                    continue
                prop_value = _ax_value(prop, "value")
                if prop_value not in {False, None, "false", "undefined"}:
                    properties.append(f"{prop_name}={str(prop_value).lower()}")
            suffix = (" " + " ".join(properties)) if properties else ""
            indent = "  " * depth
            if should_render and role == "StaticText":
                if name and name != parent_name:
                    lines.append(f'{indent}- text "{name}"')
                    rendered += 1
            elif should_render and ((role and role != "generic") or name or ref_text):
                display_role = role or "generic"
                label = f' "{name}"' if name else ""
                duplicate = duplicate_counts[(role, name)]
                nth_text = f" nth={role_seen[(role, name)] - 1}" if should_ref and duplicate > 1 else ""
                lines.append(f"{indent}- {display_role}{label}{ref_text}{nth_text}{suffix}")
                rendered += 1
            for child in children:
                visit(
                    child,
                    depth + (1 if should_render and role not in {"generic", "RootWebArea", "WebArea"} else 0),
                    name or parent_name,
                )

        for root in roots:
            visit(root, 0)

        for backend_id, info in cursor.items():
            if rendered >= max_nodes:
                break
            if backend_id in rendered_backend:
                continue
            if allowed_backend_ids is not None and backend_id not in allowed_backend_ids:
                continue
            ref = f"@e{next_ref}"
            next_ref += 1
            role = _clean(info.get("role"), 60) or "clickable"
            name = _clean(info.get("label"), 180)
            refs[ref] = RefEntry(
                ref=ref, page_id=page.page_id, session_id=session_id,
                backend_node_id=backend_id, document_generation=page.document_generation,
                session_generation=page.session_generations.get(session_id, 0),
                role=role, name=name, frame_id=frame_id,
            )
            lines.append(f'- {role}{f" \"{name}\"" if name else ""} [{ref}]')
            rendered += 1
        return lines, next_ref, rendered

    async def _cursor_elements(
        self,
        session_id: str,
        sequence: int,
        frame_id: str | None,
    ) -> dict[int, dict[str, str]]:
        object_group = f"bampi-snapshot-{sequence}-{frame_id or 'main'}"
        params: dict[str, Any] = {
            "expression": _CURSOR_SCRIPT % 120,
            "returnByValue": False,
            "objectGroup": object_group,
        }
        if frame_id:
            context = self.runtime.execution_contexts.get(frame_id)
            if context is not None and context[0] == session_id:
                params["contextId"] = context[1]
        output: dict[int, dict[str, str]] = {}
        try:
            result = await self.runtime.client.call("Runtime.evaluate", params, session_id=session_id)
            array_id = result.get("result", {}).get("objectId")
            if not array_id:
                return {}
            properties = await self.runtime.client.call(
                "Runtime.getProperties", {"objectId": array_id, "ownProperties": True}, session_id=session_id
            )
            element_ids = []
            for prop in properties.get("result", []):
                if not isinstance(prop, dict) or not str(prop.get("name", "")).isdigit():
                    continue
                object_id = prop.get("value", {}).get("objectId")
                if object_id:
                    element_ids.append(object_id)
            for object_id in element_ids:
                metadata = await self.runtime.client.call(
                    "Runtime.callFunctionOn",
                    {
                        "objectId": object_id,
                        "functionDeclaration": "function(){const c=this.matches('label')?(this.control||this.querySelector('input[type=checkbox],input[type=radio]')):null;const label=(this.getAttribute('aria-label')||this.getAttribute('title')||this.innerText||this.textContent||'').replace(/\\s+/g,' ').trim().slice(0,120);return {label,role:c?.type||this.getAttribute('role')||'clickable'}}",
                        "returnByValue": True,
                    },
                    session_id=session_id,
                )
                described = await self.runtime.client.call(
                    "DOM.describeNode", {"objectId": object_id}, session_id=session_id
                )
                node = described.get("node", {})
                backend = node.get("backendNodeId")
                info = metadata.get("result", {}).get("value")
                if isinstance(backend, int) and isinstance(info, dict):
                    output[backend] = info
        finally:
            with suppress(Exception):
                await self.runtime.client.call(
                    "Runtime.releaseObjectGroup", {"objectGroup": object_group},
                    session_id=session_id,
                )
        return output

    async def _scope_backend_ids(self, page: PageState, scope: str) -> tuple[str, set[int]]:
        element = await InteractionEngine(self.runtime).resolve(page, scope)
        described = await self.runtime.client.call(
            "DOM.describeNode",
            {"backendNodeId": element.backend_node_id, "depth": -1, "pierce": True},
            session_id=element.session_id,
        )
        ids: set[int] = set()

        def collect(node: Any) -> None:
            if isinstance(node, list):
                for item in node:
                    collect(item)
                return
            if not isinstance(node, dict):
                return
            backend = node.get("backendNodeId")
            if isinstance(backend, int):
                ids.add(backend)
            for key in ("children", "shadowRoots", "pseudoElements", "distributedNodes"):
                collect(node.get(key))
            collect(node.get("contentDocument"))
            collect(node.get("templateContent"))

        collect(described.get("node"))
        if not ids:
            raise CommandError(f"Could not build a snapshot scope for {scope!r}.")
        return element.session_id, ids

    @staticmethod
    def _flatten_child_frames(tree: dict[str, Any]) -> list[tuple[str, str]]:
        output: list[tuple[str, str]] = []

        def walk(node: dict[str, Any]) -> None:
            for child in node.get("childFrames", []) or []:
                if not isinstance(child, dict):
                    continue
                frame = child.get("frame") if isinstance(child.get("frame"), dict) else {}
                frame_id = frame.get("id")
                if isinstance(frame_id, str):
                    output.append((frame_id, str(frame.get("url") or "iframe")))
                walk(child)

        walk(tree)
        return output
