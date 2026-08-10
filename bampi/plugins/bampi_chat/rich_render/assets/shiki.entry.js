// Fine-grained Shiki bundle for offline, in-page highlighting.
//
// The JavaScript regex engine is chosen deliberately over Oniguruma: it keeps
// the bundle to a single JS file with no .wasm sidecar, which matters because
// the renderer loads this from a local file:// page with no network.

import { createHighlighterCore } from 'shiki/core'
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript'

const LANGS = [
  import('@shikijs/langs/jsx'),
  import('@shikijs/langs/tsx'),
  import('@shikijs/langs/javascript'),
  import('@shikijs/langs/typescript'),
  import('@shikijs/langs/python'),
  import('@shikijs/langs/bash'),
  import('@shikijs/langs/json'),
  import('@shikijs/langs/yaml'),
  import('@shikijs/langs/html'),
  import('@shikijs/langs/css'),
  import('@shikijs/langs/sql'),
  import('@shikijs/langs/go'),
  import('@shikijs/langs/rust'),
  import('@shikijs/langs/haskell'),
  import('@shikijs/langs/lean'),
  import('@shikijs/langs/java'),
  import('@shikijs/langs/c'),
  import('@shikijs/langs/cpp'),
  import('@shikijs/langs/markdown'),
  import('@shikijs/langs/diff'),
]

const THEMES = [
  import('@shikijs/themes/one-dark-pro'),
]

const ready = createHighlighterCore({
  themes: THEMES,
  langs: LANGS,
  engine: createJavaScriptRegexEngine({ forgiving: true }),
})

// Returns tokens rather than HTML: the renderer builds its own per-line markup
// so it can apply indentation-aware soft wrapping, which codeToHtml's flat
// <pre> output cannot express.
globalThis.__shikiTokenize = async function (code, lang, theme) {
  const highlighter = await ready
  const loaded = highlighter.getLoadedLanguages()
  const resolved = loaded.includes(lang) ? lang : 'txt'
  const result = highlighter.codeToTokens(code, { lang: resolved, theme })
  return {
    tokens: result.tokens,
    fg: result.fg,
    bg: result.bg,
    lang: resolved,
  }
}
