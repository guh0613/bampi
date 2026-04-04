from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = ROOT / "bampi" / "plugins" / "bampi_chat" / "builtin_skills" / "md2img" / "scripts" / "render_md.py"


def test_md2img_render_script_includes_cjk_font_fallback_and_font_wait(tmp_path: Path):
    source = tmp_path / "render_src.md"
    output = tmp_path / "render.html"
    source.write_text("# 标题\n\n中文正文\n\n`代码里的中文`\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), str(source), str(output)],
        capture_output=True,
        text=True,
        check=True,
    )

    html = output.read_text(encoding="utf-8")

    assert "Bampi Sans CJK" in html
    assert "noto-sans-sc-chinese-simplified-400-normal.woff2" in html
    assert "noto-sans-sc-chinese-simplified-700-normal.woff2" in html
    assert "document.fonts.ready" in html
    assert "finishRendering(800);" in html
    assert "font-family: var(--font-mono), var(--font-sans)" in html
    assert result.stderr.strip() == f"OK {output}"
