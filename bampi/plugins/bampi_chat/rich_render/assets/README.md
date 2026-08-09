# Vendored render assets

Everything the rich-block renderer needs is committed here rather than fetched
from a CDN. The renderer runs on every reply that contains a code block, table,
or display formula, so a network round-trip on the hot path would be both a
latency cost and a failure mode.

| Path | Source | Notes |
| --- | --- | --- |
| `shiki.min.js` | [Shiki](https://shiki.style) 4.4.2 | Fine-grained bundle, built locally (see below) |
| `katex/` | [KaTeX](https://katex.org) 0.16.x npm `dist/` | `katex.min.js`, `katex.min.css`, `fonts/*.woff2` |
| `fonts/` | [Maple Mono](https://github.com/subframe7536/maple-font) via `@fontsource/maple-mono` 5.3.0 | Latin subset, weights 400/700 + 400 italic |

Only the `woff2` KaTeX fonts are shipped. `katex.min.css` lists `woff2` first
and Chromium always supports it, so the `woff`/`ttf` variants are never
requested.

## Rebuilding `shiki.min.js`

The bundle uses Shiki's **JavaScript** regex engine rather than Oniguruma, which
is what keeps it to a single file with no `.wasm` sidecar — important because the
render page is loaded from `file://` with no network.

```bash
mkdir shiki-build && cd shiki-build
npm init -y && npm i shiki esbuild
# write entry.js (see below), then:
./node_modules/.bin/esbuild entry.js --bundle --format=iife --minify \
    --outfile=shiki.min.js
```

`entry.js` imports `createHighlighterCore` from `shiki/core` and
`createJavaScriptRegexEngine` from `shiki/engine/javascript`, registers the
languages listed in `LANGS` plus the `one-dark-pro` theme, and exposes
`globalThis.__shikiTokenize(code, lang, theme)` returning `codeToTokens` output.

Tokens are returned rather than HTML on purpose: the renderer assembles its own
per-line markup so it can apply indentation-aware soft wrapping, which Shiki's
flat `codeToHtml` output cannot express.

## Licences

Shiki (MIT), KaTeX (MIT), Maple Mono (OFL-1.1). Upstream licence texts ship
inside the respective npm packages.
