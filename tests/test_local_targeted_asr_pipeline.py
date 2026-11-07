from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline import transcript_evidence_correction_pipeline as correction_pipeline
from video_knowledge_pipeline.storage import write_json


def test_pipeline_rebuilds_semantic_evidence_after_local_targeted_asr(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(
        bundle / "manifest.json",
        {"normalized_transcript_json": "normalized-transcript.json"},
    )
    write_json(
        bundle / "normalized-transcript.json",
        {"segments": [{"start": 0.0, "end": 20.0, "text": "中威讲短视频营销。"}]},
    )
    pack_path = bundle / "transcript-semantic-correction-pack.json"
    pack_calls: list[int] = []
    plan_calls: list[int] = []
    execution_calls: list[dict] = []

    def fake_pack(root, **kwargs):
        pack_calls.append(1)
        if len(pack_calls) == 1:
            payload = {
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": "risk-person",
                        "correction_type": "proper_noun",
                        "risk_level": "high",
                        "llm_review_defer_reason": "needs_conflicting_external_evidence",
                        "start": 7.0,
                        "end": 10.0,
                        "evidence_source_types": ["asr"],
                    }
                ],
            }
        else:
            payload = {"candidate_count": 0, "candidates": []}
        write_json(pack_path, payload)
        return {"status": "pack_ready", "ok": True, "pack_json": str(pack_path), "candidate_count": payload["candidate_count"]}

    def fake_plan(root, **kwargs):
        plan_calls.append(1)
        windows = [{"retry_id": "semantic-evidence-0001"}] if len(plan_calls) == 1 else []
        return {"status": "planned" if windows else "no_targeted_evidence_needed", "ok": True, "retry_plan": {"window_count": len(windows), "windows": windows}}

    def fake_execution(root, **kwargs):
        execution_calls.append(kwargs)
        return {"status": "completed", "ok": True, "segment_count": 1}

    monkeypatch.setattr(correction_pipeline, "build_transcript_semantic_correction_pack", fake_pack)
    monkeypatch.setattr(correction_pipeline, "build_local_targeted_asr_plan", fake_plan)
    monkeypatch.setattr(correction_pipeline, "run_local_targeted_asr_evidence", fake_execution)

    result = correction_pipeline.run_transcript_evidence_correction_pipeline(
        bundle,
        quality_profile="",
        execute_local_targeted_asr=True,
        execute_llm=False,
        use_agent_substitute=False,
        run_postprocess=False,
        run_source_arbitration=False,
        run_readable_llm=False,
        run_agent_readable_rewrite=False,
        materialise_corrected_alias=False,
        refresh_exports=False,
        write=True,
    )

    assert len(execution_calls) == 1
    assert execution_calls[0]["execute"] is True
    assert len(pack_calls) == 2
    assert len(plan_calls) == 2
    assert result["local_targeted_asr_execution"]["status"] == "completed"
    assert result["local_targeted_asr_plan"]["status"] == "no_targeted_evidence_needed"
    names = [row["name"] for row in result["steps"]]
    assert "semantic_correction_pack_after_local_targeted_asr" in names
    assert "local_targeted_asr_plan_after_execution" in names
