from __future__ import annotations

import asyncio
from contextlib import suppress
import re

import pytest

from bampi.plugins.bampi_chat.tools.browser.config import BrowserConfig
from bampi.plugins.bampi_chat.tools.browser.launcher import find_chromium
from bampi.plugins.bampi_chat.tools.browser.runtime import BrowserRuntime


_FINGERPRINT_EXPRESSION = r"""
(async () => {
  const webglIdentity = (canvas) => {
    try {
      const context = canvas.getContext("webgl");
      const debug = context && context.getExtension("WEBGL_debug_renderer_info");
      if (!context || !debug) return null;
      return {
        vendor: context.getParameter(debug.UNMASKED_VENDOR_WEBGL),
        renderer: context.getParameter(debug.UNMASKED_RENDERER_WEBGL),
      };
    } catch (_) {
      return null;
    }
  };
  const worker = await new Promise((resolve, reject) => {
    const source = `
      const webglIdentity = () => {
        try {
          if (typeof OffscreenCanvas !== "function") return null;
          const context = new OffscreenCanvas(16, 16).getContext("webgl");
          const debug = context && context.getExtension("WEBGL_debug_renderer_info");
          if (!context || !debug) return null;
          return {
            vendor: context.getParameter(debug.UNMASKED_VENDOR_WEBGL),
            renderer: context.getParameter(debug.UNMASKED_RENDERER_WEBGL),
          };
        } catch (_) {
          return null;
        }
      };
      postMessage({
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        hardwareConcurrency: navigator.hardwareConcurrency,
        webgl: webglIdentity(),
      });
    `;
    const instance = new Worker(URL.createObjectURL(new Blob([source], {type: "text/javascript"})));
    instance.onmessage = (event) => {
      resolve(event.data);
      instance.terminate();
    };
    instance.onerror = (event) => reject(new Error(event.message || "worker failed"));
  });
  const uaData = navigator.userAgentData;
  return {
    markerInstalled: Object.prototype.hasOwnProperty.call(globalThis, "__bampi_stealth_applied__"),
    window: {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      languages: navigator.languages,
      hardwareConcurrency: navigator.hardwareConcurrency,
      webdriver: navigator.webdriver,
      webgl: webglIdentity(document.createElement("canvas")),
      uaData: uaData ? {
        low: uaData.toJSON(),
        high: await uaData.getHighEntropyValues([
          "architecture",
          "bitness",
          "fullVersionList",
          "platformVersion",
          "uaFullVersion",
          "wow64",
        ]),
      } : null,
    },
    worker,
    geometry: {
      screenWidth: screen.width,
      screenHeight: screen.height,
      innerWidth,
      innerHeight,
      outerWidth,
      outerHeight,
    },
    nativeFunctions: {
      permissionsQuery: navigator.permissions.query.toString(),
      hardwareConcurrencyGetter:
        Object.getOwnPropertyDescriptor(Navigator.prototype, "hardwareConcurrency").get.toString(),
      webglGetParameter: WebGLRenderingContext.prototype.getParameter.toString(),
    },
  };
})()
"""


@pytest.mark.asyncio
async def test_minimal_stealth_is_consistent_across_window_and_worker(tmp_path) -> None:
    executable = find_chromium()
    if executable is None:
        pytest.skip("A local Chromium executable is required for the browser consistency test")

    request_received: asyncio.Future[dict[str, str]] = asyncio.get_running_loop().create_future()

    async def serve_page(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            lines = request.decode("iso-8859-1").split("\r\n")
            headers = {
                name.strip().lower(): value.strip()
                for line in lines[1:]
                if ":" in line
                for name, value in [line.split(":", 1)]
            }
            if not request_received.done():
                request_received.set_result(headers)
            body = b"<!doctype html><title>fingerprint test</title>"
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    server = await asyncio.start_server(serve_page, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    runtime = BrowserRuntime(
        tmp_path,
        BrowserConfig(executable_path=executable, auto_install=False, headless=True),
        container_root=None,
        container_name=None,
        bridge_localhost=False,
    )
    try:
        await runtime.start()
        page = runtime.get_page()
        await runtime.client.call(
            "Page.navigate",
            {"url": f"http://127.0.0.1:{port}/"},
            session_id=page.session_id,
        )
        request_headers = await asyncio.wait_for(request_received, timeout=5)
        for _ in range(100):
            ready = await runtime.client.call(
                "Runtime.evaluate",
                {"expression": "document.readyState", "returnByValue": True},
                session_id=page.session_id,
            )
            if ready.get("result", {}).get("value") == "complete":
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("Local fingerprint page did not finish loading")
        evaluated = await runtime.client.call(
            "Runtime.evaluate",
            {
                "expression": _FINGERPRINT_EXPRESSION,
                "awaitPromise": True,
                "returnByValue": True,
            },
            session_id=page.session_id,
        )
        assert "exceptionDetails" not in evaluated
        fingerprint = evaluated["result"]["value"]
        version = await runtime.client.call("Browser.getVersion")
    finally:
        await runtime.close()
        server.close()
        await server.wait_closed()

    full_version = re.search(r"(?:Chrome|Chromium)/(\d+(?:\.\d+){3})", version["product"]).group(1)
    major = full_version.split(".", 1)[0]
    window = fingerprint["window"]
    worker = fingerprint["worker"]

    assert window["userAgent"] == worker["userAgent"]
    assert f"Chrome/{major}.0.0.0" in window["userAgent"]
    assert "HeadlessChrome" not in window["userAgent"]
    assert window["platform"] == worker["platform"]
    assert window["hardwareConcurrency"] == worker["hardwareConcurrency"]
    if window["webgl"] is not None and worker["webgl"] is not None:
        assert window["webgl"] == worker["webgl"]

    assert window["webdriver"] is False
    assert fingerprint["markerInstalled"] is False
    assert all("[native code]" in source for source in fingerprint["nativeFunctions"].values())
    assert window["uaData"] is not None
    assert any(
        entry["brand"] == "Google Chrome" and entry["version"] == full_version
        for entry in window["uaData"]["high"]["fullVersionList"]
    )
    assert request_headers["user-agent"] == window["userAgent"]
    assert f'v="{major}"' in request_headers["sec-ch-ua"]
    assert request_headers["accept-language"].startswith(window["languages"][0])

    geometry = fingerprint["geometry"]
    assert geometry["screenWidth"] >= geometry["outerWidth"] >= geometry["innerWidth"]
    assert geometry["screenHeight"] >= geometry["outerHeight"] > geometry["innerHeight"]
