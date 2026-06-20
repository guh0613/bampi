from __future__ import annotations

import base64
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from .errors import CommandError
from .interaction import InteractionEngine
from .models import CommandOutput, PageState
from .runtime import BrowserRuntime


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")[:-3]


class ArtifactManager:
    def __init__(self, runtime: BrowserRuntime, interaction: InteractionEngine) -> None:
        self.runtime = runtime
        self.interaction = interaction

    def resolve_path(self, raw_path: str | None, *, suffix: str, stem: str) -> Path:
        relative = Path(raw_path) if raw_path else Path("outbox") / "browser" / f"{_timestamp()}-{stem}{suffix}"
        if relative.is_absolute():
            if self.runtime.container_root:
                container_root = Path(self.runtime.container_root)
                try:
                    relative = relative.relative_to(container_root)
                except ValueError as exc:
                    raise CommandError("Artifact paths must be inside the group workspace.") from exc
            else:
                try:
                    relative = relative.relative_to(self.runtime.workspace_dir)
                except ValueError as exc:
                    raise CommandError("Artifact paths must be inside the group workspace.") from exc
        candidate = (self.runtime.workspace_dir / relative).resolve()
        try:
            candidate.relative_to(self.runtime.workspace_dir.resolve())
        except ValueError as exc:
            raise CommandError("Artifact path escapes the group workspace.") from exc
        if candidate.suffix.lower() != suffix.lower():
            candidate = candidate.with_suffix(suffix)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def display_path(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.runtime.workspace_dir.resolve()).as_posix()
        if self.runtime.container_root:
            return (Path(self.runtime.container_root) / relative).as_posix()
        return relative

    async def screenshot(
        self,
        page: PageState,
        *,
        path: str | None,
        target: str | None,
        full_page: bool,
        jpeg: bool,
        quality: int,
        inline: bool,
        annotate: bool,
    ) -> CommandOutput:
        params: dict[str, Any] = {
            "format": "jpeg" if jpeg else "png",
            "fromSurface": True,
            "captureBeyondViewport": bool(full_page or target),
        }
        if jpeg:
            params["quality"] = quality
        if target:
            element = await self.interaction.resolve(page, target)
            x, y, width, height = await self.interaction.box(element)
            params["clip"] = {"x": x, "y": y, "width": width, "height": height, "scale": 1}
        elif full_page:
            metrics = await self.runtime.client.call("Page.getLayoutMetrics", session_id=page.session_id)
            size = metrics.get("cssContentSize") or metrics.get("contentSize") or {}
            params["clip"] = {
                "x": 0,
                "y": 0,
                "width": max(1, float(size.get("width", self.runtime.config.viewport_width))),
                "height": max(1, float(size.get("height", self.runtime.config.viewport_height))),
                "scale": 1,
            }
        if annotate:
            await self._add_ref_overlay(page)
        try:
            result = await self.runtime.client.call(
                "Page.captureScreenshot", params, session_id=page.session_id,
                timeout=max(self.runtime.config.action_timeout, 30.0),
            )
        finally:
            if annotate:
                await self._remove_ref_overlay(page)
        data = base64.b64decode(result.get("data", ""))
        suffix = ".jpg" if jpeg else ".png"
        output_path = self.resolve_path(path, suffix=suffix, stem=page.page_id)
        output_path.write_bytes(data)
        image = data if inline and len(data) <= self.runtime.config.inline_image_max_bytes else None
        return CommandOutput(
            text=f"Screenshot saved: {self.display_path(output_path)} ({len(data)} bytes)",
            image_data=image,
            image_mime_type="image/jpeg" if jpeg else "image/png",
        )

    async def _add_ref_overlay(self, page: PageState) -> None:
        boxes: list[dict[str, Any]] = []
        for ref, entry in list(page.refs.items())[:100]:
            if entry.session_id != page.session_id:
                continue
            try:
                model = await self.runtime.client.call(
                    "DOM.getBoxModel", {"backendNodeId": entry.backend_node_id}, session_id=entry.session_id
                )
                quad = model.get("model", {}).get("border") or model.get("model", {}).get("content")
                if not isinstance(quad, list) or len(quad) < 8:
                    continue
                xs, ys = quad[0::2], quad[1::2]
                boxes.append({"ref": ref, "x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)})
            except Exception:
                continue
        script = """
        (boxes => {
          document.getElementById('__bampi_ref_overlay__')?.remove();
          const root = document.createElement('div');
          root.id = '__bampi_ref_overlay__';
          root.style.cssText = 'position:fixed;inset:0;z-index:2147483647;pointer-events:none';
          for (const b of boxes) {
            const box = document.createElement('div');
            box.style.cssText = `position:fixed;left:${b.x}px;top:${b.y}px;width:${b.w}px;height:${b.h}px;border:2px solid #ff2d55;box-sizing:border-box`;
            const label = document.createElement('span');
            label.textContent = b.ref;
            label.style.cssText = 'position:absolute;left:-2px;top:-20px;background:#ff2d55;color:white;padding:1px 4px;font:12px/16px monospace;border-radius:3px';
            box.appendChild(label); root.appendChild(box);
          }
          document.documentElement.appendChild(root);
        })(%s)
        """ % json.dumps(boxes, ensure_ascii=False)
        await self.interaction.evaluate(page, script)

    async def _remove_ref_overlay(self, page: PageState) -> None:
        try:
            await self.interaction.evaluate(page, "document.getElementById('__bampi_ref_overlay__')?.remove()")
        except Exception:
            pass

    async def pdf(self, page: PageState, path: str | None) -> CommandOutput:
        result = await self.runtime.client.call(
            "Page.printToPDF",
            {"printBackground": True, "preferCSSPageSize": True},
            session_id=page.session_id,
            timeout=max(self.runtime.config.action_timeout, 30.0),
        )
        data = base64.b64decode(result.get("data", ""))
        output_path = self.resolve_path(path, suffix=".pdf", stem=page.page_id)
        output_path.write_bytes(data)
        return CommandOutput(f"PDF saved: {self.display_path(output_path)} ({len(data)} bytes)")

    def list_downloads(self) -> str:
        directory = self.runtime.workspace_dir / "outbox" / "browser" / "downloads"
        files = sorted((path for path in directory.glob("*") if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)
        if not files:
            return "No browser downloads have been saved."
        lines = ["Browser downloads:"]
        for path in files[:30]:
            lines.append(f"- {self.display_path(path)} ({path.stat().st_size} bytes)")
        return "\n".join(lines)
