from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.webui_bridge import _render_review_fill_guide


def test_review_fill_guide_resolves_relative_review_html_uri(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    text = _render_review_fill_guide(
        title="Relative bundle",
        bundle_dir=Path("bundle"),
        review_html_path=Path("bundle/review.html"),
        review_notes_path=Path("bundle/review-notes.json"),
        review_notes_template_path=Path("bundle/review-notes.template.json"),
        apply_review_notes_args_path=Path("bundle/apply.args.json"),
        review_count=0,
    )

    assert (tmp_path / "bundle" / "review.html").resolve().as_uri() in text
