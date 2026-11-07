from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline import lecture_workflow


def test_refresh_lecture_review_rebinds_canonical_review_html(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A full review refresh must finish with the canonical-only HTML refresh.

    Intent: prevent a rebuilt bundle from temporarily presenting zero transcript
    rows while its canonical transcript artifact still exists.
    Decision: reuse ``refresh_bundle_review_html`` after the established full
    bundle exporter instead of adding another renderer.
    Reason: the canonical transcript binding lives in the lightweight refresh
    front door and must be the final projection installed for human review.
    Evidence: the production regression was reproduced with an existing bundle
    whose source-arbitrated transcript survived a full refresh.
    Effective scope: review HTML projection only; Timeline and canonical
    transcript artifacts remain owned by their existing pipelines.
    """

    review_path = tmp_path / "review-notes.json"
    review_path.write_text(
        json.dumps({"schema": "lecture_review_notes.v1", "reviews": []}),
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "webui-bundle"
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        lecture_workflow,
        "import_lecture_review",
        lambda root, review_json: {"root": str(root), "review_json": str(review_json)},
    )

    def fake_export(root, *, output_dir, target):
        calls.append(("export", Path(output_dir)))
        return {"manifest_path": str(Path(output_dir) / "manifest.json")}

    def fake_refresh(output_dir):
        calls.append(("canonical_refresh", Path(output_dir)))
        return {
            "schema": "lecture_review_html_refresh.v1",
            "review_html_path": str(Path(output_dir) / "review.html"),
        }

    monkeypatch.setattr(lecture_workflow, "export_webui_bundle", fake_export)
    monkeypatch.setattr(
        lecture_workflow,
        "refresh_bundle_review_html",
        fake_refresh,
    )

    result = lecture_workflow.refresh_lecture_review_outputs(
        tmp_path,
        review_path,
        webui_output_dir=bundle_dir,
    )

    assert calls == [
        ("export", bundle_dir),
        ("canonical_refresh", bundle_dir),
    ]
    assert result["review_html_refresh"]["schema"] == "lecture_review_html_refresh.v1"
