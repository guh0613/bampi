from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import re
import time
from typing import Any

from .errors import CommandError, StaleRefError
from .models import PageState, RefEntry
from .runtime import BrowserRuntime


@dataclass(slots=True)
class ResolvedElement:
    session_id: str
    backend_node_id: int
    object_id: str
    label: str
    ref: RefEntry | None = None


class InteractionEngine:
    def __init__(self, runtime: BrowserRuntime) -> None:
        self.runtime = runtime

    async def evaluate(
        self,
        page: PageState,
        expression: str,
        *,
        await_promise: bool = True,
        return_by_value: bool = True,
    ) -> Any:
        result = await self.runtime.client.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": return_by_value,
                "userGesture": True,
            },
            session_id=page.session_id,
            timeout=self.runtime.config.action_timeout,
        )
        details = result.get("exceptionDetails")
        if isinstance(details, dict):
            exception = details.get("exception", {})
            raise CommandError(str(exception.get("description") or details.get("text") or "JavaScript evaluation failed"))
        remote = result.get("result", {})
        if return_by_value:
            return remote.get("value", remote.get("description"))
        return remote

    async def resolve(self, page: PageState, target: str) -> ResolvedElement:
        target = target.strip()
        if target.startswith("@"): 
            entry = page.refs.get(target)
            if entry is None:
                raise CommandError(f"Unknown ref {target}. Run `snapshot` to obtain current element refs.")
            if entry.document_generation != page.document_generation:
                raise StaleRefError(f"Ref {target} is stale because the page navigated. Run `snapshot` again.")
            if entry.session_generation != page.session_generations.get(entry.session_id, 0):
                raise StaleRefError(f"Ref {target} is stale because its frame navigated. Run `snapshot` again.")
            return await self._resolve_ref(page, entry)
        if target.startswith("role="):
            return await self._resolve_role(page, target)
        expression = self._selector_expression(target)
        remote = await self.runtime.client.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": False, "awaitPromise": False},
            session_id=page.session_id,
        )
        details = remote.get("exceptionDetails")
        if isinstance(details, dict):
            description = str(details.get("exception", {}).get("description") or details.get("text") or "")
            match = re.search(r"__BAMPI_AMBIGUOUS__:(\d+)", description)
            if match:
                raise CommandError(
                    f"Target {target!r} matched {match.group(1)} elements. Use `snapshot` and an @ref, "
                    "or make the selector/semantic target unique."
                )
            raise CommandError(description or f"Could not resolve {target!r}.")
        result = remote.get("result", {})
        object_id = result.get("objectId")
        if not object_id or result.get("subtype") == "null":
            raise CommandError(f"No element matched {target!r}.")
        await self.runtime.client.call(
            "DOM.getDocument", {"depth": 0}, session_id=page.session_id,
        )
        requested = await self.runtime.client.call(
            "DOM.requestNode", {"objectId": object_id}, session_id=page.session_id
        )
        described = await self.runtime.client.call(
            "DOM.describeNode", {"nodeId": requested.get("nodeId")}, session_id=page.session_id
        )
        backend = described.get("node", {}).get("backendNodeId")
        if not isinstance(backend, int):
            raise CommandError(f"Could not resolve DOM identity for {target!r}.")
        return ResolvedElement(page.session_id, backend, object_id, target)

    async def _resolve_role(self, page: PageState, target: str) -> ResolvedElement:
        match = re.fullmatch(r"role=([^\[]+?)(?:\[name=(.*)\])?", target, flags=re.IGNORECASE)
        if match is None:
            raise CommandError("Role targets use role=button or role=button[name=Submit].")
        role = match.group(1).strip().lower()
        raw_name = match.group(2)
        name = raw_name.strip().strip("\"'") if raw_name is not None else None
        tree = await self.runtime.client.call("Accessibility.getFullAXTree", session_id=page.session_id)
        role_matches: list[tuple[int, str]] = []
        for node in tree.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_role = str(node.get("role", {}).get("value") or "").lower()
            node_name = " ".join(str(node.get("name", {}).get("value") or "").split())
            backend = node.get("backendDOMNodeId")
            if node_role == role and isinstance(backend, int):
                role_matches.append((backend, node_name))
        matches = role_matches
        if name is not None:
            normalized = " ".join(name.split())
            exact = [item for item in role_matches if item[1] == normalized]
            matches = exact or [item for item in role_matches if normalized.casefold() in item[1].casefold()]
        if not matches:
            suffix = f" with name {name!r}" if name is not None else ""
            raise CommandError(f"No element matched role={role}{suffix}.")
        if len(matches) != 1:
            suffix = f"[name={name}]" if name is not None else ""
            raise CommandError(
                f"Target role={role}{suffix} matched {len(matches)} elements. "
                "Use `snapshot` and an @ref or provide a unique accessible name."
            )
        backend, accessible_name = matches[0]
        resolved = await self.runtime.client.call(
            "DOM.resolveNode", {"backendNodeId": backend}, session_id=page.session_id
        )
        object_id = resolved.get("object", {}).get("objectId")
        if not object_id:
            raise CommandError(f"Could not resolve role target {target!r} to a DOM element.")
        return ResolvedElement(page.session_id, backend, object_id, f"{role} {accessible_name!r}")

    async def _resolve_ref(self, page: PageState, entry: RefEntry) -> ResolvedElement:
        try:
            resolved = await self.runtime.client.call(
                "DOM.resolveNode", {"backendNodeId": entry.backend_node_id}, session_id=entry.session_id
            )
            object_id = resolved.get("object", {}).get("objectId")
            if not object_id:
                raise RuntimeError("missing object id")
            return ResolvedElement(entry.session_id, entry.backend_node_id, object_id, entry.ref, entry)
        except Exception:
            tree = await self.runtime.client.call(
                "Accessibility.getFullAXTree", session_id=entry.session_id
            )
            candidates = []
            for node in tree.get("nodes", []):
                if not isinstance(node, dict):
                    continue
                role = node.get("role", {}).get("value")
                name = node.get("name", {}).get("value")
                backend = node.get("backendDOMNodeId")
                if role == entry.role and name == entry.name and isinstance(backend, int):
                    candidates.append(backend)
            if len(candidates) != 1:
                raise StaleRefError(
                    f"Ref {entry.ref} no longer identifies one unique element. Run `snapshot` again."
                )
            entry.backend_node_id = candidates[0]
            resolved = await self.runtime.client.call(
                "DOM.resolveNode", {"backendNodeId": entry.backend_node_id}, session_id=entry.session_id
            )
            object_id = resolved.get("object", {}).get("objectId")
            if not object_id:
                raise StaleRefError(f"Ref {entry.ref} can no longer be resolved. Run `snapshot` again.")
            return ResolvedElement(entry.session_id, entry.backend_node_id, object_id, entry.ref, entry)

    @staticmethod
    def _selector_expression(target: str) -> str:
        if target.startswith("css="):
            target = target[4:]
        if target.startswith("label="):
            query = json.dumps(target[6:])
            return r"""(()=>{const q=%s,n=s=>' '.concat(s||'').replace(/\s+/g,' ').trim();const labels=[...document.querySelectorAll('label')];const exact=labels.filter(e=>n(e.innerText||e.textContent)===n(q));const m=exact.length?exact:labels.filter(e=>n(e.innerText||e.textContent).toLowerCase().includes(n(q).toLowerCase()));const controls=m.map(e=>e.control||e.querySelector('input,textarea,select,button')).filter(Boolean);if(controls.length>1)throw new Error('__BAMPI_AMBIGUOUS__:'+controls.length);return controls[0]||null})()""" % query
        if target.startswith("placeholder="):
            query = json.dumps(target[12:])
            return r"""(()=>{const q=%s,n=s=>' '.concat(s||'').replace(/\s+/g,' ').trim();const all=[...document.querySelectorAll('[placeholder]')];const exact=all.filter(e=>n(e.getAttribute('placeholder'))===n(q));const m=exact.length?exact:all.filter(e=>n(e.getAttribute('placeholder')).toLowerCase().includes(n(q).toLowerCase()));if(m.length>1)throw new Error('__BAMPI_AMBIGUOUS__:'+m.length);return m[0]||null})()""" % query
        if target.startswith("testid="):
            query = json.dumps(target[7:])
            return """(()=>{const q=%s;const m=[...document.querySelectorAll('[data-testid]')].filter(e=>e.getAttribute('data-testid')===q);if(m.length>1)throw new Error('__BAMPI_AMBIGUOUS__:'+m.length);return m[0]||null})()""" % query
        if target.startswith("text="):
            text = json.dumps(target[5:])
            return (
                "(()=>{const q=" + text + ",n=s=>' '.concat(s||'').replace(/\\s+/g,' ').trim();"
                "const all=[...document.querySelectorAll('button,a,input,label,[role],summary,option,h1,h2,h3,h4,h5,h6,p,li,td,th')];"
                "const exact=all.filter(e=>n(e.innerText||e.textContent||e.value)===n(q));"
                "const m=exact.length?exact:all.filter(e=>n(e.innerText||e.textContent||e.value).toLowerCase().includes(n(q).toLowerCase()));"
                "if(m.length>1)throw new Error('__BAMPI_AMBIGUOUS__:'+m.length);return m[0]||null})()"
            )
        selector = json.dumps(target)
        return (
            f"(()=>{{const m=[...document.querySelectorAll({selector})];"
            "if(m.length>1)throw new Error('__BAMPI_AMBIGUOUS__:'+m.length);return m[0]||null})()"
        )

    async def get_attribute(self, page: PageState, target: str, name: str) -> str | None:
        element = await self.resolve(page, target)
        result = await self.runtime.client.call(
            "Runtime.callFunctionOn",
            {
                "objectId": element.object_id,
                "functionDeclaration": "function(name){return this.getAttribute(name)}",
                "arguments": [{"value": name}],
                "returnByValue": True,
            },
            session_id=element.session_id,
        )
        return result.get("result", {}).get("value")

    async def get_value(self, page: PageState, target: str) -> Any:
        element = await self.resolve(page, target)
        result = await self.runtime.client.call(
            "Runtime.callFunctionOn",
            {
                "objectId": element.object_id,
                "functionDeclaration": "function(){return 'value' in this?this.value:(this.textContent||'')}",
                "returnByValue": True,
            },
            session_id=element.session_id,
        )
        return result.get("result", {}).get("value")

    async def count(self, page: PageState, target: str) -> int:
        if target.startswith("role="):
            match = re.fullmatch(r"role=([^\[]+?)(?:\[name=(.*)\])?", target, flags=re.IGNORECASE)
            if match is None:
                raise CommandError("Role targets use role=button or role=button[name=Submit].")
            role = match.group(1).strip().lower()
            name = match.group(2)
            if name is not None:
                name = " ".join(name.strip().strip("\"'").split())
            tree = await self.runtime.client.call("Accessibility.getFullAXTree", session_id=page.session_id)
            count = 0
            for node in tree.get("nodes", []):
                if not isinstance(node, dict) or str(node.get("role", {}).get("value") or "").lower() != role:
                    continue
                node_name = " ".join(str(node.get("name", {}).get("value") or "").split())
                if name is None or node_name == name:
                    count += 1
            return count
        expression = self._count_expression(target)
        value = await self.evaluate(page, expression)
        return int(value or 0)

    @staticmethod
    def _count_expression(target: str) -> str:
        if target.startswith("css="):
            target = target[4:]
        if target.startswith("label="):
            query = json.dumps(target[6:])
            return f"[...document.querySelectorAll('label')].filter(e=>(e.innerText||e.textContent||'').trim()==={query}).length"
        if target.startswith("placeholder="):
            query = json.dumps(target[12:])
            return f"[...document.querySelectorAll('[placeholder]')].filter(e=>e.getAttribute('placeholder')==={query}).length"
        if target.startswith("testid="):
            query = json.dumps(target[7:])
            return f"[...document.querySelectorAll('[data-testid]')].filter(e=>e.getAttribute('data-testid')==={query}).length"
        if target.startswith("text="):
            query = json.dumps(target[5:])
            return f"[...document.querySelectorAll('button,a,input,label,[role],summary,option,h1,h2,h3,h4,h5,h6,p,li,td,th')].filter(e=>(e.innerText||e.textContent||e.value||'').trim()==={query}).length"
        return f"document.querySelectorAll({json.dumps(target)}).length"

    async def box(self, element: ResolvedElement) -> tuple[float, float, float, float]:
        await self.runtime.client.call(
            "DOM.scrollIntoViewIfNeeded",
            {"backendNodeId": element.backend_node_id},
            session_id=element.session_id,
        )
        model = await self.runtime.client.call(
            "DOM.getBoxModel", {"backendNodeId": element.backend_node_id}, session_id=element.session_id
        )
        quad = model.get("model", {}).get("content") or model.get("model", {}).get("border")
        if not isinstance(quad, list) or len(quad) < 8:
            raise CommandError(f"Element {element.label} has no visible box.")
        xs, ys = quad[0::2], quad[1::2]
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)

    async def _center(self, element: ResolvedElement) -> tuple[float, float]:
        x, y, width, height = await self.box(element)
        return x + width / 2, y + height / 2

    async def _check_hit_target(self, element: ResolvedElement, x: float, y: float) -> None:
        result = await self.runtime.client.call(
            "Runtime.callFunctionOn",
            {
                "objectId": element.object_id,
                "functionDeclaration": "function(x,y){let d=this.ownerDocument||document;while(d.defaultView&&d.defaultView.frameElement)d=d.defaultView.frameElement.ownerDocument;let lx=x,ly=y,h=d.elementFromPoint(lx,ly);while(h&&(h.tagName==='IFRAME'||h.tagName==='FRAME')&&h.contentDocument&&h!==this){const r=h.getBoundingClientRect();lx-=r.x+h.clientLeft;ly-=r.y+h.clientTop;d=h.contentDocument;h=d.elementFromPoint(lx,ly)}const up=n=>n&&(n.parentNode||n.host||(n.getRootNode&&n.getRootNode().host));let ok=h===this;for(let n=h;!ok&&n;n=up(n))ok=n===this;for(let n=this;!ok&&n;n=up(n))ok=n===h;return {ok,hit:h?`${h.tagName.toLowerCase()}${h.id?'#'+h.id:''}`:'none'}}",
                "arguments": [{"value": x}, {"value": y}],
                "returnByValue": True,
            },
            session_id=element.session_id,
        )
        value = result.get("result", {}).get("value", {})
        if isinstance(value, dict) and value.get("ok") is False:
            raise CommandError(
                f"Element {element.label} is covered at its center by {value.get('hit', 'another element')}. "
                "Scroll, dismiss the overlay, or take a new snapshot."
            )

    async def click(
        self,
        page: PageState,
        target: str,
        *,
        count: int = 1,
        button: str = "left",
    ) -> None:
        element = await self.resolve(page, target)
        x, y = await self._center(element)
        await self._check_hit_target(element, x, y)
        await self.runtime.client.call(
            "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y}, session_id=element.session_id
        )
        for click_count in range(1, count + 1):
            pressed = False
            try:
                await self.runtime.client.call(
                    "Input.dispatchMouseEvent",
                    {"type": "mousePressed", "x": x, "y": y, "button": button, "clickCount": click_count},
                    session_id=element.session_id,
                )
                pressed = True
            finally:
                if pressed and not self.runtime.client.closed:
                    release = asyncio.create_task(
                        self.runtime.client.call(
                            "Input.dispatchMouseEvent",
                            {"type": "mouseReleased", "x": x, "y": y, "button": button, "clickCount": click_count},
                            session_id=element.session_id,
                        )
                    )
                    try:
                        await asyncio.shield(release)
                    except asyncio.CancelledError:
                        with suppress(Exception):
                            await release
                        raise
                    except Exception:
                        # A click may synchronously navigate and detach the old target.
                        if self.runtime.client.closed:
                            raise

    async def hover(self, page: PageState, target: str) -> None:
        element = await self.resolve(page, target)
        x, y = await self._center(element)
        await self.runtime.client.call(
            "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y}, session_id=element.session_id
        )

    async def focus(self, page: PageState, target: str) -> None:
        element = await self.resolve(page, target)
        await self.runtime.client.call(
            "Runtime.callFunctionOn",
            {"objectId": element.object_id, "functionDeclaration": "function(){this.focus()}", "userGesture": True},
            session_id=element.session_id,
        )

    async def fill(self, page: PageState, target: str, text: str, *, append: bool = False) -> None:
        element = await self.resolve(page, target)
        preparation = "function(){this.focus();if(this.isContentEditable){this.textContent='';this.dispatchEvent(new InputEvent('input',{bubbles:true,inputType:'deleteContentBackward'}));return}const p=this instanceof HTMLTextAreaElement?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;const s=Object.getOwnPropertyDescriptor(p,'value')?.set;if(s)s.call(this,'');else this.value='';this.dispatchEvent(new InputEvent('input',{bubbles:true,inputType:'deleteContentBackward'}))}"
        await self.runtime.client.call(
            "Runtime.callFunctionOn",
            {"objectId": element.object_id, "functionDeclaration": "function(){this.focus()}" if append else preparation},
            session_id=element.session_id,
        )
        if text:
            await self.runtime.client.call(
                "Input.insertText", {"text": text}, session_id=page.session_id
            )

    async def press(self, page: PageState, keys: str) -> None:
        parts = [part for part in re.split(r"\+", keys) if part]
        key = parts[-1] if parts else keys
        modifiers = 0
        for modifier in parts[:-1]:
            normalized = modifier.lower()
            modifiers |= {"alt": 1, "control": 2, "ctrl": 2, "meta": 4, "command": 4, "shift": 8}.get(normalized, 0)
        key_names = {
            "esc": "Escape", "return": "Enter", "space": " ", "del": "Delete",
            "arrowup": "ArrowUp", "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
        }
        key = key_names.get(key.lower(), key)
        virtual_codes = {
            "Backspace": 8, "Tab": 9, "Enter": 13, "Shift": 16, "Control": 17, "Alt": 18,
            "Escape": 27, " ": 32, "PageUp": 33, "PageDown": 34, "End": 35, "Home": 36,
            "ArrowLeft": 37, "ArrowUp": 38, "ArrowRight": 39, "ArrowDown": 40, "Delete": 46,
        }
        vk = virtual_codes.get(key, ord(key.upper()) if len(key) == 1 else 0)
        code = key if len(key) > 1 else ("Key" + key.upper() if key.isalpha() else key)
        base = {"key": key, "code": code, "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk, "modifiers": modifiers}
        await self.runtime.client.call("Input.dispatchKeyEvent", {"type": "rawKeyDown", **base}, session_id=page.session_id)
        if len(key) == 1 and not modifiers:
            await self.runtime.client.call("Input.dispatchKeyEvent", {"type": "char", "text": key, **base}, session_id=page.session_id)
        await self.runtime.client.call("Input.dispatchKeyEvent", {"type": "keyUp", **base}, session_id=page.session_id)

    async def select(self, page: PageState, target: str, values: list[str]) -> None:
        element = await self.resolve(page, target)
        result = await self.runtime.client.call(
            "Runtime.callFunctionOn",
            {
                "objectId": element.object_id,
                "functionDeclaration": "function(values){if(!(this instanceof HTMLSelectElement))throw new Error('target is not a select');for(const o of this.options)o.selected=values.includes(o.value)||values.includes(o.text);this.dispatchEvent(new Event('input',{bubbles:true}));this.dispatchEvent(new Event('change',{bubbles:true}));return [...this.selectedOptions].map(o=>o.value)}",
                "arguments": [{"value": values}], "returnByValue": True,
            },
            session_id=element.session_id,
        )
        if result.get("exceptionDetails"):
            raise CommandError("The target is not a selectable <select> element.")

    async def set_checked(self, page: PageState, target: str, desired: bool) -> None:
        element = await self.resolve(page, target)
        state = await self.runtime.client.call(
            "Runtime.callFunctionOn",
            {"objectId": element.object_id, "functionDeclaration": "function(){return !!this.checked}", "returnByValue": True},
            session_id=element.session_id,
        )
        checked = bool(state.get("result", {}).get("value"))
        if checked != desired:
            await self.click(page, target)

    async def drag(self, page: PageState, source: str, target: str, *, html5: bool = False) -> None:
        source_el = await self.resolve(page, source)
        target_el = await self.resolve(page, target)
        if source_el.session_id != target_el.session_id:
            raise CommandError("Drag source and target must be in the same page or frame.")
        if html5:
            result = await self.runtime.client.call(
                "Runtime.callFunctionOn",
                {
                    "objectId": source_el.object_id,
                    "functionDeclaration": "function(target){const d=new DataTransfer();this.dispatchEvent(new DragEvent('dragstart',{bubbles:true,dataTransfer:d}));target.dispatchEvent(new DragEvent('dragenter',{bubbles:true,dataTransfer:d}));target.dispatchEvent(new DragEvent('dragover',{bubbles:true,cancelable:true,dataTransfer:d}));target.dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:d}));this.dispatchEvent(new DragEvent('dragend',{bubbles:true,dataTransfer:d}));}",
                    "arguments": [{"objectId": target_el.object_id}],
                },
                session_id=source_el.session_id,
            )
            if result.get("exceptionDetails"):
                raise CommandError("HTML5 drag failed; try pointer drag without --html5.")
            return
        sx, sy = await self._center(source_el)
        tx, ty = await self._center(target_el)
        await self.runtime.client.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": sx, "y": sy}, session_id=source_el.session_id)
        await self.runtime.client.call("Input.dispatchMouseEvent", {"type": "mousePressed", "x": sx, "y": sy, "button": "left", "buttons": 1, "clickCount": 1}, session_id=source_el.session_id)
        try:
            for step in range(1, 11):
                ratio = step / 10
                await self.runtime.client.call(
                    "Input.dispatchMouseEvent",
                    {"type": "mouseMoved", "x": sx + (tx - sx) * ratio, "y": sy + (ty - sy) * ratio, "button": "left", "buttons": 1},
                    session_id=source_el.session_id,
                )
                await asyncio.sleep(0.02)
        finally:
            await self.runtime.client.call("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": tx, "y": ty, "button": "left", "buttons": 0, "clickCount": 1}, session_id=source_el.session_id)

    async def upload(self, page: PageState, target: str, paths: list[Path]) -> None:
        element = await self.resolve(page, target)
        await self.runtime.client.call(
            "DOM.setFileInputFiles",
            {"files": [str(path) for path in paths], "backendNodeId": element.backend_node_id},
            session_id=element.session_id,
        )

    async def scroll(self, page: PageState, direction: str, amount: int = 800) -> None:
        if direction.startswith("@") or direction.startswith(("css=", "text=", "#", ".", "[")):
            element = await self.resolve(page, direction)
            await self.runtime.client.call(
                "DOM.scrollIntoViewIfNeeded", {"backendNodeId": element.backend_node_id}, session_id=element.session_id
            )
            return
        expressions = {
            "top": "scrollTo({top:0,behavior:'instant'})",
            "bottom": "scrollTo({top:document.documentElement.scrollHeight,behavior:'instant'})",
            "up": f"scrollBy(0,{-abs(amount)})",
            "down": f"scrollBy(0,{abs(amount)})",
            "left": f"scrollBy({-abs(amount)},0)",
            "right": f"scrollBy({abs(amount)},0)",
        }
        expression = expressions.get(direction)
        if expression is None:
            raise CommandError("scroll expects up, down, left, right, top, bottom, or an element target.")
        await self.evaluate(page, expression)

    async def extract(self, page: PageState, target: str | None, *, html: bool, max_chars: int) -> str:
        if target:
            element = await self.resolve(page, target)
            function = "function(){return this.outerHTML}" if html else "function(){return this.innerText||this.textContent||this.value||''}"
            result = await self.runtime.client.call(
                "Runtime.callFunctionOn",
                {"objectId": element.object_id, "functionDeclaration": function, "returnByValue": True},
                session_id=element.session_id,
            )
            value = result.get("result", {}).get("value", "")
        else:
            value = await self.evaluate(page, "document.documentElement.outerHTML" if html else "document.body?.innerText||''")
        text = str(value or "")
        return text if len(text) <= max_chars else text[:max_chars] + "\n… [truncated]"

    async def wait(
        self,
        page: PageState,
        *,
        seconds: float | None = None,
        url: str | None = None,
        text: str | None = None,
        target: str | None = None,
        state: str = "visible",
        load: str | None = None,
        condition: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if seconds is not None:
            await asyncio.sleep(seconds)
            return
        deadline = time.monotonic() + (timeout or self.runtime.config.action_timeout)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                await self.runtime.refresh_page_info(page)
                if url is not None and fnmatch.fnmatch(page.url, url):
                    return
                if text is not None:
                    body = str(await self.evaluate(page, "document.body?.innerText||''") or "")
                    if text in body:
                        return
                if target is not None:
                    element = await self.resolve(page, target)
                    visible = await self.runtime.client.call(
                        "Runtime.callFunctionOn",
                        {"objectId": element.object_id, "functionDeclaration": "function(){const r=this.getBoundingClientRect(),s=getComputedStyle(this);return r.width>1&&r.height>1&&s.display!=='none'&&s.visibility!=='hidden'}", "returnByValue": True},
                        session_id=element.session_id,
                    )
                    is_visible = bool(visible.get("result", {}).get("value"))
                    if (state == "visible" and is_visible) or (state == "hidden" and not is_visible):
                        return
                if load is not None:
                    ready_state = str(await self.evaluate(page, "document.readyState") or "")
                    if load == "domcontentloaded" and ready_state in {"interactive", "complete"}:
                        return
                    if load == "load" and ready_state == "complete":
                        return
                    if (
                        load == "networkidle"
                        and not page.network_inflight
                        and time.monotonic() - page.last_network_activity >= 0.5
                    ):
                        return
                if condition is not None and bool(await self.evaluate(page, condition)):
                    return
            except Exception as exc:
                last_error = exc
                if target is not None and state in {"hidden", "detached"}:
                    return
            await asyncio.sleep(0.1)
        expectation = (
            f"url={url!r}" if url else f"text={text!r}" if text else
            f"{target!r} state={state}" if target else f"load={load}" if load else f"condition={condition!r}"
        )
        suffix = f" Last error: {last_error}" if last_error else ""
        raise CommandError(f"Timed out waiting for {expectation}.{suffix}")
