from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.local_targeted_asr_plan import build_local_targeted_asr_plan
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.transcript_quality_gate import run_transcript_quality_gate


def test_generic_high_risk_heuristic_does_not_create_asr_debt(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(
        bundle / "manifest.json",
        {
            "duration_seconds": 20,
            "corrected_transcript_json": "corrected-transcript.json",
            "transcript_semantic_correction_pack_json": "transcript-semantic-correction-pack.json",
        },
    )
    write_json(
        bundle / "corrected-transcript.json",
        {"segments": [{"start": 0, "end": 20, "text": "这是一段完整的课程转写内容。"}]},
    )
    write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {
            "candidates": [
                {
                    "candidate_id": "generic-number",
                    "correction_type": "number",
                    "risk_level": "high",
                    "llm_review_defer_reason": "needs_conflicting_external_evidence",
                    "start": 10,
                    "end": 11,
                    "original_text": "一次",
                    "candidate_text": "",
                    "suggested_text": "",
                    "evidence_source_types": ["asr"],
                }
            ]
        },
    )

    plan = build_local_targeted_asr_plan(bundle, write=True)
    quality = run_transcript_quality_gate(bundle, write=False)

    assert plan["status"] == "no_targeted_evidence_needed"
    assert quality["ok"] is True
    assert quality["semantic_evidence"]["required"] is False


def test_local_targeted_asr_plan_prefers_explicit_preview_pack_over_stale_disk(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle-preview"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {})
    write_json(bundle / "transcript-semantic-correction-pack.json", {"candidates": []})
    preview_pack = {
        "candidates": [
            {
                "candidate_id": "preview-name",
                "correction_type": "proper_noun",
                "risk_level": "high",
                "llm_review_defer_reason": "needs_conflicting_external_evidence",
                "start": 7,
                "end": 10,
                "original_text": "中医人寿",
                "candidate_text": "中意人寿",
                "evidence_source_types": ["asr"],
            }
        ]
    }

    result = build_local_targeted_asr_plan(
        bundle,
        pack_data=preview_pack,
        max_windows=4,
        write=False,
    )

    assert result["candidate_count"] == 1
    assert result["selected_candidate_count"] == 1
    assert result["retry_plan"]["window_count"] == 1
    assert not (bundle / "local-targeted-asr-plan.json").exists()
