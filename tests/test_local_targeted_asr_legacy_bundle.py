from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline import local_targeted_asr_execution as targeted_execution
from video_knowledge_pipeline.storage import write_json


def test_local_targeted_asr_uses_legacy_bundle_sources_path(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    media = tmp_path / "legacy-source.mp4"
    media.write_bytes(b"media fixture")
    write_json(bundle / "manifest.json", {"sources": [{"path": str(media)}]})
    plan = bundle / "local-targeted-asr-plan.json"
    write_json(
        plan,
        {
            "retry_plan": {
                "window_count": 1,
                "windows": [{"retry_id": "semantic-evidence-0001", "start": 0.0, "end": 3.0}],
            }
        },
    )
    captured: list[Path] = []

    def fake_prepare(media_path, quality_report, output_dir, *, execute):
        captured.append(Path(media_path))
        return {"status": "planned", "artifacts": []}

    monkeypatch.setattr(targeted_execution, "prepare_asr_retry_snippets", fake_prepare)
    result = targeted_execution.run_local_targeted_asr_evidence(
        bundle,
        input_plan=plan,
        execute=False,
    )

    assert result["status"] == "planned"
    assert captured == [media.resolve()]
