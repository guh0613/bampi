from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import json
from pathlib import Path
import shlex

from bampy.agent.cancellation import CancellationToken

from .artifacts import ArtifactManager
from .errors import CommandError
from .interaction import InteractionEngine
from .models import CommandOutput, PageState
from .recording import RecordingManager
from .runtime import BrowserRuntime
from .snapshot import SnapshotEngine


UpdateCallback = Callable[[str], Awaitable[None]]


HELP_TEXT = """Browser commands (targets: @e3, CSS/css=..., text=..., label=..., placeholder=..., testid=..., role=button[name=Submit]):
  open URL | goto URL | snapshot [--interactive] [--depth N] [--scope TARGET] [--urls] [--max N]
  click TARGET [--right] | dblclick TARGET | hover TARGET | focus TARGET
  fill TARGET "text" | type TARGET "text" | press KEY | select TARGET VALUE...
  check TARGET | uncheck TARGET | drag SOURCE TARGET [--html5] | upload TARGET PATH...
  wait SECONDS | wait --url GLOB | wait --text TEXT | wait TARGET [--state visible|hidden|detached] | wait --load domcontentloaded|load|networkidle | wait --fn JS
  extract [TARGET] [--html] [--max N] | get attr TARGET NAME | get value TARGET | get count TARGET | eval "JavaScript"
  scroll up|down|left|right|top|bottom|TARGET [AMOUNT]
  tabs | tab PAGE_ID | close [PAGE_ID...] [--all] [--others] | reload | back | forward
  screenshot [PATH] [--target TARGET] [--full] [--annotate] [--jpeg] [--quality N] [--no-inline]
  pdf [PATH] | record start [PATH.mp4|PATH.webm] | record stop | downloads
  cookies [get|clear|set NAME VALUE] | storage local|session [get|clear|set KEY VALUE]
  dialog accept [TEXT]|dismiss | state save|load [PATH]
  console [--clear] | errors [--clear] | network [--clear]
  viewport WIDTH HEIGHT | offline on|off | headers JSON_OBJECT | reset [--profile]
  batch [--continue] followed by one command per line (max 32; nested batch is rejected)
Use shell-style quoting only for grouping text; commands are parsed internally and never executed by a shell."""


def _split(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError as exc:
        raise CommandError(f"Invalid browser command quoting: {exc}") from exc


def _take_option(tokens: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for index, token in enumerate(tokens):
        if token.startswith(prefix):
            tokens.pop(index)
            return token[len(prefix):]
        if token == name:
            if index + 1 >= len(tokens):
                raise CommandError(f"{name} requires a value.")
            tokens.pop(index)
            return tokens.pop(index)
    return default


def _flag(tokens: list[str], name: str) -> bool:
    if name not in tokens:
        return False
    tokens.remove(name)
    return True


def _positive_int(value: str | None, *, name: str, default: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CommandError(f"{name} must be an integer.") from exc
    if not 1 <= parsed <= maximum:
        raise CommandError(f"{name} must be between 1 and {maximum}.")
    return parsed


class BrowserCommandDispatcher:
    def __init__(
        self,
        runtime: BrowserRuntime,
        *,
        active_page_id: str | None,
        cancellation: CancellationToken | None,
    ) -> None:
        self.runtime = runtime
        self.active_page_id = active_page_id
        self.cancellation = cancellation
        self.interaction = InteractionEngine(runtime)
        self.snapshot = SnapshotEngine(runtime)
        self.artifacts = ArtifactManager(runtime, self.interaction)
        self.recording = RecordingManager(runtime, self.artifacts)

    def _cancel_check(self) -> None:
        if self.cancellation:
            self.cancellation.raise_if_cancelled()

    def page(self, page_id: str | None = None) -> PageState:
        requested = page_id or self.active_page_id
        try:
            page = self.runtime.get_page(requested)
        except CommandError:
            if page_id is not None:
                raise
            page = self.runtime.get_page()
        self.active_page_id = page.page_id
        return page

    async def execute(self, command: str, *, on_update: UpdateCallback | None = None) -> CommandOutput:
        stripped = command.strip()
        if not stripped:
            raise CommandError("Browser command cannot be empty. Use `help` for syntax.")
        first_line = stripped.splitlines()[0]
        first_tokens = _split(first_line)
        if first_tokens and first_tokens[0].lower() == "batch":
            return await self._batch(stripped, on_update=on_update)
        return await self._single(stripped)

    async def _single(self, command: str) -> CommandOutput:
        self._cancel_check()
        tokens = _split(command)
        if not tokens:
            raise CommandError("Browser command cannot be empty.")
        name = tokens.pop(0).lower()

        if name in {"help", "?"}:
            return CommandOutput(HELP_TEXT)
        if name == "open":
            self._expect_count(tokens, exact=1, usage="open URL")
            resolved, notes = await self.runtime.policy.resolve(tokens[0])
            blank_pages = [page for page in self.runtime.pages.values() if page.url == "about:blank"]
            page = blank_pages[0] if self.active_page_id is None and len(self.runtime.pages) == 1 and blank_pages else await self.runtime.create_page()
            self.active_page_id = page.page_id
            await self._navigate(page, resolved)
            return CommandOutput(self._page_summary(page, notes))
        if name == "goto":
            self._expect_count(tokens, exact=1, usage="goto URL")
            page = self.page()
            resolved, notes = await self.runtime.policy.resolve(tokens[0])
            await self._navigate(page, resolved)
            return CommandOutput(self._page_summary(page, notes))
        if name == "snapshot":
            interactive = _flag(tokens, "--interactive")
            show_urls = _flag(tokens, "--urls")
            scope = _take_option(tokens, "--scope")
            depth_raw = _take_option(tokens, "--depth")
            if depth_raw is None:
                max_depth = None
            else:
                try:
                    max_depth = int(depth_raw)
                except ValueError as exc:
                    raise CommandError("--depth must be an integer.") from exc
                if not 0 <= max_depth <= 30:
                    raise CommandError("--depth must be between 0 and 30.")
            max_nodes = _positive_int(_take_option(tokens, "--max"), name="--max", default=180, maximum=600)
            self._expect_count(tokens, exact=0, usage="snapshot [--interactive] [--depth N] [--scope TARGET] [--urls] [--max N]")
            page = self.page()
            await self.runtime.refresh_page_info(page)
            result = await self.snapshot.capture(
                page,
                max_nodes=max_nodes,
                interactive=interactive,
                max_depth=max_depth,
                scope=scope,
                show_urls=show_urls,
            )
            return CommandOutput(result.text)
        if name in {"click", "dblclick"}:
            right = _flag(tokens, "--right")
            self._expect_count(tokens, exact=1, usage=f"{name} TARGET [--right]")
            page = self.page()
            generation = page.document_generation
            page_count = len(self.runtime.pages)
            await self.interaction.click(page, tokens[0], count=2 if name == "dblclick" else 1, button="right" if right else "left")
            page = await self._settle_after_action(page, generation=generation, page_count=page_count)
            await self.runtime.refresh_page_info(page)
            return CommandOutput(f"{name}ed {tokens[0]} on {page.page_id}. URL: {self.runtime.policy.display_url(page.url)}")
        if name in {"hover", "focus"}:
            self._expect_count(tokens, exact=1, usage=f"{name} TARGET")
            page = self.page()
            await getattr(self.interaction, name)(page, tokens[0])
            return CommandOutput(f"{name.title()}ed {tokens[0]} on {page.page_id}.")
        if name in {"fill", "type"}:
            self._expect_count(tokens, minimum=2, usage=f'{name} TARGET "text"')
            target, text = tokens[0], " ".join(tokens[1:])
            page = self.page()
            await self.interaction.fill(page, target, text, append=name == "type")
            return CommandOutput(f"{'Appended to' if name == 'type' else 'Filled'} {target} on {page.page_id}.")
        if name == "press":
            self._expect_count(tokens, exact=1, usage="press KEY")
            page = self.page()
            await self.interaction.press(page, tokens[0])
            return CommandOutput(f"Pressed {tokens[0]} on {page.page_id}.")
        if name == "select":
            self._expect_count(tokens, minimum=2, usage="select TARGET VALUE...")
            page = self.page()
            await self.interaction.select(page, tokens[0], tokens[1:])
            return CommandOutput(f"Selected {tokens[1:]} in {tokens[0]}.")
        if name in {"check", "uncheck"}:
            self._expect_count(tokens, exact=1, usage=f"{name} TARGET")
            page = self.page()
            await self.interaction.set_checked(page, tokens[0], name == "check")
            return CommandOutput(f"{name.title()}ed {tokens[0]}.")
        if name == "drag":
            html5 = _flag(tokens, "--html5")
            self._expect_count(tokens, exact=2, usage="drag SOURCE TARGET [--html5]")
            page = self.page()
            await self.interaction.drag(page, tokens[0], tokens[1], html5=html5)
            return CommandOutput(f"Dragged {tokens[0]} to {tokens[1]} ({'HTML5' if html5 else 'pointer'} mode).")
        if name == "upload":
            self._expect_count(tokens, minimum=2, usage="upload TARGET PATH...")
            page = self.page()
            paths = [self._workspace_path(value, must_exist=True) for value in tokens[1:]]
            await self.interaction.upload(page, tokens[0], paths)
            return CommandOutput(f"Uploaded {len(paths)} file(s) through {tokens[0]}.")
        if name == "wait":
            return await self._wait(tokens)
        if name == "extract":
            html = _flag(tokens, "--html")
            max_chars = _positive_int(_take_option(tokens, "--max"), name="--max", default=8_000, maximum=50_000)
            self._expect_count(tokens, minimum=0, maximum=1, usage="extract [TARGET] [--html] [--max N]")
            page = self.page()
            text = await self.interaction.extract(page, tokens[0] if tokens else None, html=html, max_chars=max_chars)
            return CommandOutput(text)
        if name == "get":
            return await self._get(tokens)
        if name in {"eval", "evaluate"}:
            expression = self._raw_remainder(command)
            if not expression:
                raise CommandError('Usage: eval "JavaScript"')
            page = self.page()
            value = await self.interaction.evaluate(page, expression)
            rendered = json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value
            return CommandOutput(rendered[:50_000] + ("\n… [truncated]" if len(rendered) > 50_000 else ""))
        if name == "scroll":
            self._expect_count(tokens, minimum=1, maximum=2, usage="scroll DIRECTION|TARGET [AMOUNT]")
            amount = int(tokens[1]) if len(tokens) == 2 else 800
            page = self.page()
            await self.interaction.scroll(page, tokens[0], amount)
            return CommandOutput(f"Scrolled {tokens[0]} on {page.page_id}.")
        if name in {"tabs", "pages"}:
            self._expect_count(tokens, exact=0, usage="tabs")
            return CommandOutput(await self._tabs())
        if name in {"tab", "switch"}:
            self._expect_count(tokens, exact=1, usage="tab PAGE_ID")
            page = self.runtime.get_page(tokens[0])
            self.active_page_id = page.page_id
            await self.runtime.client.call("Page.bringToFront", session_id=page.session_id)
            return CommandOutput(f"Active tab: {self._page_summary(page)}")
        if name in {"close", "close-tab"}:
            close_all = _flag(tokens, "--all")
            close_others = _flag(tokens, "--others")
            if close_all and close_others:
                raise CommandError("Cannot combine --all and --others.")
            if close_all:
                self._expect_count(tokens, exact=0, usage="close --all")
                closed = list(self.runtime.pages.values())
                for page in closed:
                    await self.runtime.close_page(page)
                self.active_page_id = None
                return CommandOutput(f"Closed {len(closed)} tab(s). 0 tab(s) remain.")
            if close_others:
                self._expect_count(tokens, minimum=0, maximum=1, usage="close --others [PAGE_ID]")
                keep = self.page(tokens[0] if tokens else None)
                to_close = [p for p in self.runtime.pages.values() if p.page_id != keep.page_id]
                for page in to_close:
                    await self.runtime.close_page(page)
                self.active_page_id = keep.page_id
                return CommandOutput(f"Closed {len(to_close)} tab(s). Kept {keep.page_id}. {len(self.runtime.pages)} tab(s) remain.")
            if len(tokens) > 1:
                ids = list(tokens)
                closed_ids: list[str] = []
                for pid in ids:
                    page = self.runtime.pages.get(pid)
                    if page is None:
                        raise CommandError(f"Unknown tab {pid}. Use `tabs` to list open tabs.")
                    await self.runtime.close_page(page)
                    closed_ids.append(pid)
                if self.active_page_id not in self.runtime.pages:
                    self.active_page_id = next(reversed(self.runtime.pages), None)
                return CommandOutput(f"Closed {', '.join(closed_ids)}. {len(self.runtime.pages)} tab(s) remain.")
            page = self.page(tokens[0] if tokens else None)
            await self.runtime.close_page(page)
            self.active_page_id = next(reversed(self.runtime.pages), None)
            return CommandOutput(f"Closed {page.page_id}. {len(self.runtime.pages)} tab(s) remain.")
        if name in {"reload", "back", "forward"}:
            self._expect_count(tokens, exact=0, usage=name)
            return await self._history(name)
        if name == "screenshot":
            return await self._screenshot(tokens)
        if name == "pdf":
            self._expect_count(tokens, minimum=0, maximum=1, usage="pdf [PATH]")
            return await self.artifacts.pdf(self.page(), tokens[0] if tokens else None)
        if name == "record":
            self._expect_count(tokens, minimum=1, maximum=2, usage="record start [PATH] | record stop")
            action = tokens[0].lower()
            if action == "start":
                return CommandOutput(await self.recording.start(self.page(), tokens[1] if len(tokens) > 1 else None))
            if action == "stop" and len(tokens) == 1:
                return CommandOutput(await self.recording.stop())
            raise CommandError("Usage: record start [PATH.mp4|PATH.webm] | record stop")
        if name == "downloads":
            self._expect_count(tokens, exact=0, usage="downloads")
            return CommandOutput(self.artifacts.list_downloads())
        if name == "dialog":
            return await self._dialog(tokens)
        if name == "state":
            return await self._state(tokens)
        if name == "cookies":
            return await self._cookies(tokens)
        if name == "storage":
            return await self._storage(tokens)
        if name in {"console", "errors", "network"}:
            return self._diagnostics(name, tokens)
        if name == "viewport":
            self._expect_count(tokens, exact=2, usage="viewport WIDTH HEIGHT")
            try:
                width, height = int(tokens[0]), int(tokens[1])
            except ValueError as exc:
                raise CommandError("Viewport width and height must be integers.") from exc
            if not (200 <= width <= 7680 and 200 <= height <= 4320):
                raise CommandError("Viewport dimensions are outside the supported range.")
            page = self.page()
            await self.runtime.client.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False}, session_id=page.session_id)
            return CommandOutput(f"Viewport set to {width}x{height} on {page.page_id}.")
        if name == "offline":
            self._expect_count(tokens, exact=1, usage="offline on|off")
            enabled = self._on_off(tokens[0])
            page = self.page()
            await self.runtime.client.call("Network.emulateNetworkConditions", {"offline": enabled, "latency": 0, "downloadThroughput": -1, "uploadThroughput": -1}, session_id=page.session_id)
            return CommandOutput(f"Offline mode {'enabled' if enabled else 'disabled'} on {page.page_id}.")
        if name == "headers":
            raw_headers = self._raw_remainder(command)
            if not raw_headers:
                raise CommandError('Usage: headers {"Name":"value"}')
            try:
                headers = json.loads(raw_headers)
            except json.JSONDecodeError as exc:
                raise CommandError(f"Headers must be a JSON object: {exc}") from exc
            if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
                raise CommandError("Headers must be a JSON object containing only string values.")
            page = self.page()
            await self.runtime.client.call("Network.setExtraHTTPHeaders", {"headers": headers}, session_id=page.session_id)
            return CommandOutput(f"Set {len(headers)} extra header(s) on {page.page_id}.")
        if name == "reset":
            clear_profile = _flag(tokens, "--profile")
            self._expect_count(tokens, exact=0, usage="reset [--profile]")
            await self.runtime.close(clear_profile=clear_profile)
            self.active_page_id = None
            return CommandOutput("Browser stopped." + (" Persistent profile removed." if clear_profile else ""))
        raise CommandError(f"Unknown browser command {name!r}. Use `help` for the command list.")

    async def _batch(self, command: str, *, on_update: UpdateCallback | None) -> CommandOutput:
        lines = [line.strip() for line in command.splitlines()]
        header = _split(lines[0])
        continue_on_error = "--continue" in header[1:]
        unknown = [token for token in header[1:] if token != "--continue"]
        if unknown:
            raise CommandError(f"Unknown batch option(s): {' '.join(unknown)}")
        commands = [line for line in lines[1:] if line and not line.startswith("#")]
        if not commands:
            raise CommandError("batch requires one or more commands on following lines.")
        if len(commands) > self.runtime.config.batch_max_commands:
            raise CommandError(f"batch accepts at most {self.runtime.config.batch_max_commands} commands.")
        if any((_split(line) or [""])[0].lower() == "batch" for line in commands):
            raise CommandError("Nested batch commands are not allowed.")

        async def run() -> CommandOutput:
            summaries: list[str] = []
            last_image: CommandOutput | None = None
            failures = 0
            for index, line in enumerate(commands, 1):
                self._cancel_check()
                try:
                    output = await self._single(line)
                    summary = output.text.replace("\n", " ")[:300]
                    summaries.append(f"{index}. ✓ {line} — {summary}")
                    if output.image_data is not None:
                        last_image = output
                    if on_update:
                        await on_update(f"batch {index}/{len(commands)} ✓ {line}: {summary[:160]}")
                except Exception as exc:
                    failures += 1
                    summaries.append(f"{index}. ✗ {line} — {exc}")
                    if on_update:
                        await on_update(f"batch {index}/{len(commands)} ✗ {line}: {exc}")
                    if not continue_on_error:
                        skipped = len(commands) - index
                        raise CommandError(
                            f"Batch failed at step {index}/{len(commands)}; {skipped} command(s) skipped.\n"
                            + "\n".join(summaries)
                        ) from exc
            heading = "Batch completed" + (f" with {failures} failure(s)" if failures else " successfully")
            text = heading + ":\n" + "\n".join(summaries)
            return CommandOutput(
                text=text,
                image_data=last_image.image_data if last_image else None,
                image_mime_type=last_image.image_mime_type if last_image else None,
            )

        try:
            return await asyncio.wait_for(run(), timeout=self.runtime.config.batch_timeout)
        except TimeoutError as exc:
            raise CommandError(f"Batch exceeded the {self.runtime.config.batch_timeout:g}s total timeout.") from exc

    async def _navigate(self, page: PageState, url: str) -> None:
        result = await self.runtime.client.call(
            "Page.navigate", {"url": url}, session_id=page.session_id,
            timeout=self.runtime.config.action_timeout,
        )
        if result.get("errorText"):
            raise CommandError(f"Navigation failed: {result['errorText']}")
        await self._wait_ready(page)
        await self.runtime.refresh_page_info(page)

    async def _wait_ready(self, page: PageState) -> None:
        deadline = asyncio.get_running_loop().time() + self.runtime.config.action_timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                state = await self.interaction.evaluate(page, "document.readyState")
                if state in {"interactive", "complete"}:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.05)
        raise CommandError("Navigation timed out waiting for DOMContentLoaded.")

    async def _wait(self, tokens: list[str]) -> CommandOutput:
        url = _take_option(tokens, "--url")
        text = _take_option(tokens, "--text")
        load = _take_option(tokens, "--load")
        condition = _take_option(tokens, "--fn")
        state = _take_option(tokens, "--state", "visible") or "visible"
        timeout_raw = _take_option(tokens, "--timeout")
        timeout = float(timeout_raw) if timeout_raw is not None else None
        page = self.page()
        selected = sum(value is not None for value in (url, text, load, condition))
        if selected > 1:
            raise CommandError("wait accepts only one of --url, --text, --load, or --fn.")
        if url is not None:
            self._expect_count(tokens, exact=0, usage="wait --url GLOB")
            await self.interaction.wait(page, url=url, timeout=timeout)
            return CommandOutput(f"URL matched {url!r}: {page.url}")
        if text is not None:
            self._expect_count(tokens, exact=0, usage="wait --text TEXT")
            await self.interaction.wait(page, text=text, timeout=timeout)
            return CommandOutput(f"Text appeared: {text!r}")
        if load is not None:
            self._expect_count(tokens, exact=0, usage="wait --load domcontentloaded|load|networkidle")
            if load not in {"domcontentloaded", "load", "networkidle"}:
                raise CommandError("wait --load must be domcontentloaded, load, or networkidle.")
            await self.interaction.wait(page, load=load, timeout=timeout)
            return CommandOutput(f"Page reached {load}.")
        if condition is not None:
            self._expect_count(tokens, exact=0, usage='wait --fn "JavaScript condition"')
            await self.interaction.wait(page, condition=condition, timeout=timeout)
            return CommandOutput("JavaScript wait condition became truthy.")
        self._expect_count(tokens, exact=1, usage="wait SECONDS | wait TARGET [--state STATE]")
        try:
            seconds = float(tokens[0])
        except ValueError:
            if state not in {"visible", "hidden", "detached"}:
                raise CommandError("wait --state must be visible, hidden, or detached.")
            await self.interaction.wait(page, target=tokens[0], state=state, timeout=timeout)
            return CommandOutput(f"{tokens[0]} reached state {state}.")
        if not 0 <= seconds <= 120:
            raise CommandError("Wait duration must be between 0 and 120 seconds.")
        await self.interaction.wait(page, seconds=seconds)
        return CommandOutput(f"Waited {seconds:g}s.")

    async def _settle_after_action(self, page: PageState, *, generation: int, page_count: int) -> PageState:
        await asyncio.sleep(0.05)
        if len(self.runtime.pages) > page_count:
            popup = next(reversed(self.runtime.pages.values()))
            self.active_page_id = popup.page_id
            with suppress(CommandError):
                await self._wait_ready(popup)
            return popup
        if page.document_generation != generation:
            with suppress(CommandError):
                await self._wait_ready(page)
            return page
        if page.navigating:
            deadline = asyncio.get_running_loop().time() + self.runtime.config.action_timeout
            while page.navigating and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
                if self.cancellation and self.cancellation.cancelled:
                    break
            if page.document_generation != generation:
                with suppress(CommandError):
                    await self._wait_ready(page)
            return page
        return page

    async def _get(self, tokens: list[str]) -> CommandOutput:
        self._expect_count(tokens, minimum=2, usage="get attr TARGET NAME | get value TARGET | get count TARGET")
        action = tokens.pop(0).lower()
        page = self.page()
        if action == "attr":
            self._expect_count(tokens, exact=2, usage="get attr TARGET NAME")
            value = await self.interaction.get_attribute(page, tokens[0], tokens[1])
            return CommandOutput("null" if value is None else str(value))
        if action == "value":
            self._expect_count(tokens, exact=1, usage="get value TARGET")
            value = await self.interaction.get_value(page, tokens[0])
            return CommandOutput(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value)
        if action == "count":
            self._expect_count(tokens, exact=1, usage="get count TARGET")
            return CommandOutput(str(await self.interaction.count(page, tokens[0])))
        raise CommandError("get action must be attr, value, or count.")

    async def _tabs(self) -> str:
        lines = ["Open tabs:"]
        for page in self.runtime.pages.values():
            await self.runtime.refresh_page_info(page)
            marker = "*" if page.page_id == self.active_page_id else " "
            lines.append(f'{marker} {page.page_id} "{page.title}" {self.runtime.policy.display_url(page.url)}')
        return "\n".join(lines)

    async def _history(self, action: str) -> CommandOutput:
        page = self.page()
        if action == "reload":
            await self.runtime.client.call("Page.reload", {"ignoreCache": False}, session_id=page.session_id)
        else:
            history = await self.runtime.client.call("Page.getNavigationHistory", session_id=page.session_id)
            entries = history.get("entries", [])
            current = int(history.get("currentIndex", 0))
            target = current - 1 if action == "back" else current + 1
            if target < 0 or target >= len(entries):
                raise CommandError(f"No {action} history entry is available.")
            await self.runtime.client.call("Page.navigateToHistoryEntry", {"entryId": entries[target]["id"]}, session_id=page.session_id)
        await self._wait_ready(page)
        await self.runtime.refresh_page_info(page)
        return CommandOutput(self._page_summary(page))

    async def _screenshot(self, tokens: list[str]) -> CommandOutput:
        target = _take_option(tokens, "--target")
        full = _flag(tokens, "--full")
        jpeg = _flag(tokens, "--jpeg")
        no_inline = _flag(tokens, "--no-inline")
        annotate = _flag(tokens, "--annotate")
        quality = _positive_int(_take_option(tokens, "--quality"), name="--quality", default=85, maximum=100)
        self._expect_count(tokens, minimum=0, maximum=1, usage="screenshot [PATH] [--target TARGET] [--full] [--jpeg]")
        return await self.artifacts.screenshot(
            self.page(), path=tokens[0] if tokens else None, target=target,
            full_page=full, jpeg=jpeg, quality=quality, inline=not no_inline, annotate=annotate,
        )

    async def _dialog(self, tokens: list[str]) -> CommandOutput:
        self._expect_count(tokens, minimum=1, usage="dialog accept [TEXT] | dialog dismiss")
        action = tokens.pop(0).lower()
        page = self.page()
        if page.dialog is None:
            raise CommandError("No JavaScript dialog is currently open.")
        if action == "accept":
            await self.runtime.client.call(
                "Page.handleJavaScriptDialog",
                {"accept": True, "promptText": " ".join(tokens)},
                session_id=page.session_id,
            )
        elif action == "dismiss" and not tokens:
            await self.runtime.client.call(
                "Page.handleJavaScriptDialog", {"accept": False}, session_id=page.session_id
            )
        else:
            raise CommandError("Usage: dialog accept [TEXT] | dialog dismiss")
        page.dialog = None
        return CommandOutput(f"Dialog {action}ed on {page.page_id}.")

    async def _state(self, tokens: list[str]) -> CommandOutput:
        self._expect_count(tokens, minimum=1, maximum=2, usage="state save|load [PATH]")
        action = tokens[0].lower()
        path = self._workspace_path(tokens[1] if len(tokens) > 1 else ".browser/state.json", must_exist=action == "load")
        page = self.page()
        if action == "save":
            cookies = (await self.runtime.client.call("Network.getAllCookies", session_id=page.session_id)).get("cookies", [])
            local = await self.interaction.evaluate(page, "Object.fromEntries(Object.entries(localStorage))")
            session = await self.interaction.evaluate(page, "Object.fromEntries(Object.entries(sessionStorage))")
            payload = {"version": 1, "url": page.url, "cookies": cookies, "localStorage": local, "sessionStorage": session}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return CommandOutput(f"Browser state saved: {path.relative_to(self.runtime.workspace_dir).as_posix()}")
        if action == "load":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CommandError(f"Could not load browser state: {exc}") from exc
            allowed = {"name", "value", "url", "domain", "path", "secure", "httpOnly", "sameSite", "expires", "priority", "sameParty", "sourceScheme", "sourcePort", "partitionKey"}
            cookies = [{key: value for key, value in cookie.items() if key in allowed} for cookie in payload.get("cookies", []) if isinstance(cookie, dict)]
            if cookies:
                await self.runtime.client.call("Network.setCookies", {"cookies": cookies}, session_id=page.session_id)
            for object_name, values in (("localStorage", payload.get("localStorage", {})), ("sessionStorage", payload.get("sessionStorage", {}))):
                if isinstance(values, dict):
                    await self.interaction.evaluate(page, f"{object_name}.clear();Object.entries({json.dumps(values, ensure_ascii=False)}).forEach(([k,v])=>{object_name}.setItem(k,v))")
            return CommandOutput(f"Browser state loaded: {path.relative_to(self.runtime.workspace_dir).as_posix()}")
        raise CommandError("state action must be save or load.")

    async def _cookies(self, tokens: list[str]) -> CommandOutput:
        action = tokens.pop(0).lower() if tokens else "get"
        page = self.page()
        if action == "get":
            self._expect_count(tokens, exact=0, usage="cookies get")
            result = await self.runtime.client.call("Network.getAllCookies", session_id=page.session_id)
            cookies = result.get("cookies", [])
            lines = [f"Cookies ({len(cookies)}; values redacted):"]
            for cookie in cookies[:100]:
                lines.append(f"- {cookie.get('name')} domain={cookie.get('domain')} path={cookie.get('path')} secure={cookie.get('secure')}")
            return CommandOutput("\n".join(lines))
        if action == "clear":
            self._expect_count(tokens, exact=0, usage="cookies clear")
            await self.runtime.client.call("Network.clearBrowserCookies", session_id=page.session_id)
            return CommandOutput("Browser cookies cleared.")
        if action == "set":
            self._expect_count(tokens, exact=2, usage="cookies set NAME VALUE")
            result = await self.runtime.client.call("Network.setCookie", {"name": tokens[0], "value": tokens[1], "url": page.url}, session_id=page.session_id)
            if result.get("success") is False:
                raise CommandError("Chromium rejected the cookie.")
            return CommandOutput(f"Cookie {tokens[0]!r} set for the current URL.")
        raise CommandError("cookies action must be get, set, or clear.")

    async def _storage(self, tokens: list[str]) -> CommandOutput:
        self._expect_count(tokens, minimum=1, usage="storage local|session [get|clear|set KEY VALUE]")
        kind = tokens.pop(0).lower()
        if kind not in {"local", "session"}:
            raise CommandError("storage kind must be local or session.")
        object_name = "localStorage" if kind == "local" else "sessionStorage"
        action = tokens.pop(0).lower() if tokens else "get"
        page = self.page()
        if action == "get":
            self._expect_count(tokens, exact=0, usage=f"storage {kind} get")
            value = await self.interaction.evaluate(page, f"Object.fromEntries(Object.entries({object_name}))")
            return CommandOutput(json.dumps(value, ensure_ascii=False, indent=2))
        if action == "clear":
            self._expect_count(tokens, exact=0, usage=f"storage {kind} clear")
            await self.interaction.evaluate(page, f"{object_name}.clear()")
            return CommandOutput(f"{kind} storage cleared.")
        if action == "set":
            self._expect_count(tokens, exact=2, usage=f"storage {kind} set KEY VALUE")
            await self.interaction.evaluate(page, f"{object_name}.setItem({json.dumps(tokens[0])},{json.dumps(tokens[1])})")
            return CommandOutput(f"Set {tokens[0]!r} in {kind} storage.")
        raise CommandError("storage action must be get, set, or clear.")

    def _diagnostics(self, name: str, tokens: list[str]) -> CommandOutput:
        clear = _flag(tokens, "--clear")
        self._expect_count(tokens, exact=0, usage=f"{name} [--clear]")
        page = self.page()
        collection = getattr(page, name)
        if clear:
            count = len(collection)
            collection.clear()
            return CommandOutput(f"Cleared {count} {name} entr{'y' if count == 1 else 'ies'}.")
        if name == "network":
            lines = [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in collection]
        else:
            lines = list(collection)
        return CommandOutput("\n".join(lines[-100:]) or f"No {name} entries captured.")

    def _workspace_path(self, raw: str, *, must_exist: bool) -> Path:
        path = Path(raw)
        if path.is_absolute() and self.runtime.container_root:
            try:
                path = path.relative_to(Path(self.runtime.container_root))
            except ValueError as exc:
                raise CommandError("Upload paths must be inside the group workspace.") from exc
        candidate = (self.runtime.workspace_dir / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            candidate.relative_to(self.runtime.workspace_dir.resolve())
        except ValueError as exc:
            raise CommandError("Upload path escapes the group workspace.") from exc
        if must_exist and not candidate.is_file():
            raise CommandError(f"Upload file does not exist: {raw}")
        return candidate

    def _page_summary(self, page: PageState, notes: list[str] | None = None) -> str:
        base = f'{page.page_id} "{page.title}" {self.runtime.policy.display_url(page.url)}'
        return base if not notes else base + "\n" + "\n".join(f"Note: {note}" for note in notes)

    @staticmethod
    def _on_off(value: str) -> bool:
        normalized = value.lower()
        if normalized in {"on", "true", "1"}:
            return True
        if normalized in {"off", "false", "0"}:
            return False
        raise CommandError("Expected on or off.")

    @staticmethod
    def _raw_remainder(command: str) -> str:
        parts = command.strip().split(None, 1)
        remainder = parts[1].strip() if len(parts) == 2 else ""
        if len(remainder) >= 2 and remainder[0] == remainder[-1] and remainder[0] in {'"', "'"}:
            parsed = _split(remainder)
            if len(parsed) == 1:
                return parsed[0]
        return remainder

    @staticmethod
    def _expect_count(
        tokens: list[str], *, exact: int | None = None, minimum: int | None = None,
        maximum: int | None = None, usage: str,
    ) -> None:
        valid = True
        if exact is not None:
            valid = len(tokens) == exact
        if minimum is not None:
            valid = valid and len(tokens) >= minimum
        if maximum is not None:
            valid = valid and len(tokens) <= maximum
        if not valid:
            raise CommandError(f"Usage: {usage}")
