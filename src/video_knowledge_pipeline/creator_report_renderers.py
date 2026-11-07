"""Bounded export adapters; reuse MarkdownIt, Pillow, Pandoc and markdown-pdf.

No downloader, browser, model client or new document layout engine lives here.
See docs/creator-report-adapter-20261030.md for provenance and decisions.
"""
from __future__ import annotations

import base64
import html
import io
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from .file_hash import sha256_file


CSS = """
body{max-width:860px;margin:36px auto;padding:0 24px;color:#222;background:#fff;
font:17px/1.75 system-ui,'Microsoft YaHei',sans-serif} h1{font-size:30px}
h2{margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.35em}
h3{margin-top:1.6em} img{display:block;max-width:100%;max-height:400px;margin:16px auto}
blockquote{margin:20px 0;padding:12px 18px;background:#f6f2e7;border-left:4px solid #8a6b20}
table{border-collapse:collapse;width:100%} th,td{border:1px solid #ddd;padding:8px;text-align:left}
pre{white-space:pre-wrap;overflow-wrap:anywhere} a{color:#175d8d}
@media print{body{margin:0;font-size:12pt}h2,h3{break-after:avoid}img{break-inside:avoid}}
"""
PDF_CSS = CSS + "body{font-size:11pt;line-height:1.45;margin:0;padding:0}h1{font-size:22pt}h2{font-size:16pt;margin-top:1em}h3{font-size:13pt;margin-top:1em}"


def bundle_file(root: Path, value: str | Path) -> Path:
    """Public assets must belong to the Bundle, including after symlink resolution."""
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("asset must be an existing file inside this Bundle")
    return path


def frame_preview(root: Path, value: str) -> dict[str, Any]:
    """Decode/re-encode a small preview; never embed original metadata or external URLs."""
    from PIL import Image
    path = bundle_file(root, value)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("only raster JPG/PNG/WebP previews are supported")
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("preview source exceeds 20 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                if source.width * source.height > 24_000_000:
                    raise ValueError("preview source exceeds 24 megapixels")
                source.thumbnail((960, 540))
                image = source.convert("RGB")
                target = io.BytesIO()
                image.save(target, format="JPEG", quality=78, optimize=True)
    except Image.DecompressionBombError as exc:
        raise ValueError("preview decompression limit exceeded") from exc
    return {"source": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
            "data_uri": "data:image/jpeg;base64," + base64.b64encode(target.getvalue()).decode("ascii")}


def html_document(markdown: str, title: str) -> str:
    # All content strings are escaped by the projection, and raw HTML stays disabled.
    body = MarkdownIt("commonmark", {"html": False}).enable("table").render(markdown)
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; "
            "img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style><body>{body}</body></html>")


def export_format(markdown: str, target: Path, *, title: str, pandoc: str = "") -> None:
    suffix = target.suffix
    if suffix == ".md":
        target.write_text(markdown, encoding="utf-8")
    elif suffix == ".html":
        target.write_text(html_document(markdown, title), encoding="utf-8")
    elif suffix == ".pdf":
        # Unlike BiliNote's whole export module, this public API imports no app/provider config.
        from markdown_pdf import MarkdownPdf, Section
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(markdown, toc=True), user_css=PDF_CSS)
        pdf.meta["title"] = title
        pdf.save(str(target))
    elif suffix == ".docx":
        binary = pandoc or shutil.which("pandoc")
        if not binary:
            raise RuntimeError("Pandoc unavailable; install/configure it separately or request HTML/Markdown")
        # No shell or filters; commonmark input contains only escaped text and embedded previews.
        result = subprocess.run([binary, "--from=commonmark", "--to=docx", "--standalone", "-o", str(target)],
                                input=markdown, encoding="utf-8", capture_output=True, timeout=90, check=False)
        if result.returncode:
            raise RuntimeError(f"Pandoc exited {result.returncode}: {result.stderr[:500]}")
    else:
        raise ValueError(f"unsupported export format: {suffix}")
    if not target.is_file() or not target.stat().st_size:
        raise RuntimeError("renderer did not produce a nonempty artifact")
