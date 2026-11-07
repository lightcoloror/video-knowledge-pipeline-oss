from __future__ import annotations

from pathlib import Path

import pytest

from video_knowledge_pipeline.cli import build_parser, run_mcp_call
from video_knowledge_pipeline.content_clip_alignment import check_content_clip_alignment
from video_knowledge_pipeline.content_clip_candidate_pack import adapt_script_clip_request_to_content, build_content_clip_candidate_pack
from video_knowledge_pipeline.content_clip_query_profiles import list_content_clip_query_profiles
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.video_workbench import export_video_workbench


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    (bundle / "exports").mkdir(parents=True)
    transcript = bundle / "corrected-transcript.json"
    write_json(
        transcript,
        {
            "segments": [
                {"segment_id": "seg-001", "source_segment_ids": ["seg-001"], "start": 0.0, "end": 4.0, "text": "医生解释质子治疗费用需要三十万元。", "speaker": "speaker-global-001", "speaker_role": "customer", "words": [{"word": "医生", "start": 0.0, "end": 0.5}, {"word": "质子治疗", "start": 1.0, "end": 2.0}]},
                {"segment_id": "seg-002", "source_segment_ids": ["seg-002"], "start": 4.0, "end": 8.0, "text": "页面显示保费测算表。", "speaker": "speaker-global-002", "speaker_role": "presenter"},
            ]
        },
    )
    write_json(
        bundle / "timeline.json",
        [
            {"index": 1, "start": 0.0, "end": 4.0, "transcript": "医生解释质子治疗费用需要三十万元。"},
            {"index": 2, "start": 4.0, "end": 8.0, "transcript": "页面显示保费测算表。", "ocr_text": "保费测算 30万元", "visual_understanding": {"summary": "讲师指向表格"}, "frame_paths": ["frames/0002.jpg"]},
        ],
    )
    write_json(bundle / "manifest.json", {"schema": "video_knowledge_pipeline.bundle.v1", "title": "合成通用片段", "corrected_transcript": "corrected-transcript.json"})
    write_json(
        bundle / "exports" / "technical-shot-boundaries.json",
        {
            "schema": "video_knowledge_pipeline.technical_shot_boundaries.v1",
            "ok": True,
            "status": "completed",
            "boundary_kind": "technical_shot",
            "backend": "saved",
            "shots": [
                {"shot_id": "shot-001", "start": 0.0, "end": 4.0},
                {"shot_id": "shot-002", "start": 4.0, "end": 8.0},
            ],
        },
    )
    return bundle, transcript


def _request(path: Path, transcript: Path, *, purpose: str = "quote") -> Path:
    clip = {
        "clip_id": f"clip-{purpose}",
        "purpose": purpose,
        "query": "医生解释质子治疗费用" if purpose == "quote" else "保费测算表",
        "match_modes": ["quote", "semantic"] if purpose == "quote" else ["ocr", "visual"],
        "required": True,
        "state": "planned",
        "required_modalities": ["asr"] if purpose == "quote" else ["visual", "shot"],
        "optional_modalities": ["shot"] if purpose == "quote" else ["ocr", "asr"],
        "must_include": ["质子治疗"] if purpose == "quote" else ["保费测算"],
        "must_exclude": ["主持人串场"],
        "speaker_constraints": {"allowed_roles": ["customer"] if purpose == "quote" else [], "excluded_roles": ["interviewer"]},
        "duration": {"minimum_seconds": 2.0, "preferred_seconds": 4.0, "maximum_seconds": 8.0},
        "boundary_policy": "complete_sentence" if purpose == "quote" else "whole_technical_shot",
        "source_scope": {"video_ids": [], "time_ranges": [{"start": 0.0, "end": 4.0}] if purpose == "quote" else [{"start": 4.0, "end": 8.0}]},
    }
    write_json(path, {"schema": "video_knowledge_pipeline.content_clip_request.v1", "request_id": f"request-{purpose}", "source_transcript": {"path": str(transcript), "sha256": sha256_file(transcript)}, "clips": [clip], "operator_boundary": {"local_only": True}})
    return path


def _confirm(bundle: Path, *, evidence: dict[str, Path], approved_window: dict | None = None) -> Path:
    pack = read_json(bundle / "exports" / "content-clip-candidate-pack.json")
    review = read_json(bundle / "content-clip-review-notes.todo.json")
    review["review_status"] = "human_confirmed"
    review["review_id"] = "review-001"
    review["candidate_pack"]["artifact_sha256"] = sha256_file(bundle / "exports" / "content-clip-candidate-pack.json")
    clip = review["clips"][0]
    candidate = pack["clips"][0]["candidates"][0]
    clip.update(
        {
            "selected_candidate_id": candidate["candidate_id"],
            "fine_cut_order": 1,
            "fine_cut_output": "clips/clip-001.mp4",
            "approved_window": approved_window,
            "approved_source_segment_ids": candidate["source_segment_ids"],
            "expected_speaker_role": "customer" if pack["clips"][0]["purpose"] == "quote" else "",
            "expected_speaker_ids": ["speaker-global-001"] if pack["clips"][0]["purpose"] == "quote" else [],
            "approved_clip_text": pack["clips"][0]["query"],
            "subtitle_text": pack["clips"][0]["query"],
            "human_confirmed": True,
        }
    )
    for modality, path in evidence.items():
        clip["clip_evidence"][modality] = {"path": str(path), "sha256": sha256_file(path)}
        clip["modality_reviews"][modality] = "human_confirmed"
    output = bundle / "content-clip-review-notes.confirmed.json"
    write_json(output, review)
    return output


def _fine_cut(path: Path, start: float, end: float) -> Path:
    write_json(path, {"clips": [{"order": 1, "output": "clips/clip-001.mp4", "keep_ranges": [[start, end]]}]})
    return path


def test_profiles_candidate_pack_boundary_and_idempotence(tmp_path: Path) -> None:
    bundle, transcript = _bundle(tmp_path)
    request = _request(tmp_path / "screen-request.json", transcript, purpose="visual_event")
    first = build_content_clip_candidate_pack(bundle, request, top_k=3, write=True)
    second = build_content_clip_candidate_pack(bundle, request, top_k=3, write=False)

    assert len(list_content_clip_query_profiles()) == 9
    assert first["pack_sha256"] == second["pack_sha256"]
    candidate = first["clips"][0]["candidates"][0]
    assert candidate["boundary"]["source_shot_ids"] == ["technical-shot-0002"]
    assert candidate["boundary"]["recommended_cut_range"]["start"] == 4.0
    assert candidate["modality_evidence"]["visual"]["status"] == "confirmed"
    assert candidate["missing_required_modalities"] == []
    assert candidate["ranking"]["raw_scores_compared_across_retrievers"] is False
    assert first["publication_allowed"] is False
    assert first["operator_boundary"]["automatic_model_escalation"] is False


def test_quote_alignment_ready_and_mcp_workbench_paths(tmp_path: Path) -> None:
    bundle, transcript = _bundle(tmp_path)
    request = _request(tmp_path / "quote-request.json", transcript)
    build_content_clip_candidate_pack(bundle, request, top_k=2, write=True)
    clip_asr = tmp_path / "clip-asr.json"
    write_json(clip_asr, {"segments": [{"segment_id": "clip-seg", "start": 0.0, "end": 4.0, "text": "医生解释质子治疗费用需要三十万元。", "speaker": "speaker-global-001", "speaker_role": "customer"}]})
    review = _confirm(bundle, evidence={"asr": clip_asr})
    fine_cut = _fine_cut(tmp_path / "fine-cut.json", 0.0, 4.0)

    result = check_content_clip_alignment(bundle, review, fine_cut, write=True)

    assert result["status"] == "ready_for_human_final_review"
    assert result["summary"]["issue_count"] == 0
    assert result["publication_allowed"] is False
    pack = read_json(bundle / "exports" / "content-clip-candidate-pack.json")
    assert pack["clips"][0]["candidates"][0]["boundary"]["word_timestamp_used"] is True
    mcp_result = run_mcp_call("content_clip_alignment_check", bundle / "mcp-content-clip-alignment-check.args.json")
    assert mcp_result["status"] == "ready_for_human_final_review"
    workbench = export_video_workbench(bundle, write=False)
    artifact_keys = {row["key"] for row in workbench["artifacts"]}
    assert "content_clip_candidate_pack_markdown" in artifact_keys
    assert "content_clip_alignment_check_markdown" in artifact_keys
    parser = build_parser()
    assert parser.parse_args(["content-clip-candidate-pack", str(bundle), str(request), "--no-write"]).command == "content-clip-candidate-pack"


def test_visual_event_requires_clip_only_visual_review(tmp_path: Path) -> None:
    bundle, transcript = _bundle(tmp_path)
    request = _request(tmp_path / "visual-request.json", transcript, purpose="visual_event")
    build_content_clip_candidate_pack(bundle, request, top_k=2, write=True)
    review = _confirm(bundle, evidence={}, approved_window={"start": 4.0, "end": 8.0})
    fine_cut = _fine_cut(tmp_path / "fine-cut.json", 4.0, 8.0)

    result = check_content_clip_alignment(bundle, review, fine_cut, write=False)

    assert result["status"] == "needs_visual_review"
    codes = {issue["code"] for issue in result["clips"][0]["issues"]}
    assert "required_multimodal_evidence_missing" in codes


def test_pack_hash_drift_and_missing_shot_fail_closed(tmp_path: Path) -> None:
    bundle, transcript = _bundle(tmp_path)
    request = _request(tmp_path / "visual-request.json", transcript, purpose="visual_event")
    build_content_clip_candidate_pack(bundle, request, top_k=2, write=True)
    visual = tmp_path / "visual.json"
    write_json(visual, {"frames": [{"frame_id": "f1", "observation": "讲师指向表格"}]})
    review = _confirm(bundle, evidence={"visual": visual}, approved_window={"start": 4.0, "end": 8.0})
    fine_cut = _fine_cut(tmp_path / "fine-cut.json", 4.0, 8.0)
    pack_path = bundle / "exports" / "content-clip-candidate-pack.json"
    pack = read_json(pack_path)
    pack["summary"]["candidate_count"] = 999
    write_json(pack_path, pack)

    with pytest.raises(ValueError, match="hash changed"):
        check_content_clip_alignment(bundle, review, fine_cut, write=False)

    (bundle / "exports" / "technical-shot-boundaries.json").unlink()
    rebuilt = build_content_clip_candidate_pack(bundle, request, top_k=2, write=False)
    candidate = rebuilt["clips"][0]["candidates"][0]
    assert candidate["eligibility_status"] == "missing_required_evidence"
    assert "shot" in candidate["missing_required_modalities"]


def test_legacy_script_request_maps_to_generic_contract() -> None:
    legacy = {
        "schema": "video_knowledge_pipeline.script_clip_request.v1",
        "request_id": "legacy-001",
        "source_transcript": {"path": "transcript.json", "sha256": "a" * 64},
        "slots": [{"slot_id": "slot-001", "state": "recorded", "required": True, "search_queries": ["复查"], "expected_quote": "三个月后复查", "preferred_window": {"start": 1.0, "end": 3.0}, "required_speaker_roles": ["customer"], "excluded_speaker_roles": ["interviewer"]}],
    }
    content = adapt_script_clip_request_to_content(legacy)
    assert content["schema"] == "video_knowledge_pipeline.content_clip_request.v1"
    assert content["clips"][0]["profile_id"] == "spoken-quote-v1"
    assert content["clips"][0]["source_scope"]["time_ranges"] == [{"start": 1.0, "end": 3.0}]


def test_explicit_video_scope_excludes_other_bundle(tmp_path: Path) -> None:
    bundle, transcript = _bundle(tmp_path)
    request = _request(tmp_path / "request.json", transcript)
    payload = read_json(request)
    payload["clips"][0]["source_scope"]["video_ids"] = ["another-video"]
    write_json(request, payload)

    with pytest.raises(ValueError, match="source_scope excludes"):
        build_content_clip_candidate_pack(bundle, request, write=False)


def test_must_include_exclude_and_explicit_speaker_constraints_are_hard_filters(tmp_path: Path) -> None:
    bundle, transcript = _bundle(tmp_path)
    request = _request(tmp_path / "request.json", transcript)
    payload = read_json(request)
    payload["clips"][0]["must_exclude"] = ["质子治疗"]
    write_json(request, payload)
    excluded = build_content_clip_candidate_pack(bundle, request, write=False)
    assert excluded["clips"][0]["candidates"] == []

    payload["clips"][0]["must_exclude"] = []
    payload["clips"][0]["speaker_constraints"]["allowed_roles"] = ["presenter"]
    write_json(request, payload)
    wrong_speaker = build_content_clip_candidate_pack(bundle, request, write=False)
    assert wrong_speaker["clips"][0]["candidates"] == []
