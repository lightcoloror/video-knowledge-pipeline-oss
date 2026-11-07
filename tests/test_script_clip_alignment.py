from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest

from video_knowledge_pipeline.cli import build_parser, main, run_mcp_call
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.script_clip_alignment import check_script_clip_alignment
from video_knowledge_pipeline.script_clip_candidate_pack import build_script_clip_candidate_pack
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.video_workbench import export_video_workbench
from video_knowledge_pipeline import mcp_server


def _bundle(tmp_path: Path, *, roles: bool = True) -> tuple[Path, Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    script = tmp_path / "script.md"
    script.write_text("# 合成采访脚本\n\n仅用于离线测试。", encoding="utf-8")
    transcript = bundle / "corrected-transcript.json"
    segments = [
        {
            "segment_id": "seg-customer-01",
            "source_segment_ids": ["seg-customer-01"],
            "start": 0.0,
            "end": 4.0,
            "text": "医生说三个月以后复查，我后来按时去了。",
            "speaker": "speaker-global-001",
            "speaker_role": "customer" if roles else "",
        },
        {
            "segment_id": "seg-interviewer-01",
            "source_segment_ids": ["seg-interviewer-01"],
            "start": 4.0,
            "end": 7.0,
            "text": "你当时是不是很担心费用？",
            "speaker": "speaker-global-002",
            "speaker_role": "interviewer" if roles else "",
        },
        {
            "segment_id": "seg-customer-02",
            "source_segment_ids": ["seg-customer-02"],
            "start": 7.0,
            "end": 11.0,
            "text": "听到接近三十万元，钱还是有点吃力。",
            "speaker": "speaker-global-001",
            "speaker_role": "customer" if roles else "",
        },
        {
            "segment_id": "seg-family-01",
            "source_segment_ids": ["seg-family-01"],
            "start": 11.0,
            "end": 15.0,
            "text": "后来我们把付款节点梳理清楚了。",
            "speaker": "speaker-global-003",
            "speaker_role": "family" if roles else "",
        },
    ]
    write_json(transcript, {"schema": "video_knowledge_pipeline.postprocessed_transcript.v1", "segments": segments})
    write_json(
        bundle / "timeline.json",
        [
            {"index": index, "start": row["start"], "end": row["end"], "transcript": row["text"]}
            for index, row in enumerate(segments, start=1)
        ],
    )
    write_json(
        bundle / "manifest.json",
        {
            "schema": "video_knowledge_pipeline.bundle.v1",
            "title": "合成采访",
            "corrected_transcript": "corrected-transcript.json",
        },
    )
    return bundle, script, transcript


def _request(
    path: Path,
    script: Path,
    transcript: Path,
    *,
    slots: list[dict] | None = None,
) -> Path:
    write_json(
        path,
        {
            "schema": "video_knowledge_pipeline.script_clip_request.v1",
            "request_id": "synthetic-script-clips-001",
            "script": {"path": str(script), "sha256": sha256_file(script)},
            "source_transcript": {"path": str(transcript), "sha256": sha256_file(transcript)},
            "slots": slots
            or [
                {
                    "slot_id": "episode-01-customer-quote",
                    "state": "recorded",
                    "story_segment_ref": "episode-01",
                    "episode_binding": "episode-01",
                    "required": True,
                    "search_queries": ["三个月以后复查"],
                    "expected_quote": "医生说三个月以后复查，我后来按时去了。",
                    "subtitle_candidate": "医生说三个月以后复查，我后来按时去了。",
                    "preferred_window": {"start": 0.0, "end": 4.0},
                    "required_speaker_roles": ["customer"],
                    "excluded_speaker_roles": ["interviewer"],
                }
            ],
            "operator_boundary": {"local_only": True, "review_only": True, "publication_allowed": False},
        },
    )
    return path


def _clip_transcript(path: Path, text: str, *, speaker: str = "speaker-global-001", role: str = "customer") -> Path:
    write_json(
        path,
        {
            "segments": [
                {
                    "segment_id": "clip-seg-001",
                    "source_segment_ids": ["seg-customer-01"],
                    "start": 0.0,
                    "end": 4.0,
                    "text": text,
                    "speaker": speaker,
                    "speaker_role": role,
                }
            ]
        },
    )
    return path


def _confirmed_review(bundle: Path, clip_transcript: Path) -> Path:
    pack = read_json(bundle / "exports" / "script-clip-candidate-pack.json")
    todo = read_json(bundle / "script-clip-review-notes.todo.json")
    candidate = pack["slots"][0]["candidates"][0]
    todo["review_status"] = "human_confirmed"
    todo["review_id"] = "review-synthetic-001"
    todo["candidate_pack"]["artifact_sha256"] = sha256_file(bundle / "exports" / "script-clip-candidate-pack.json")
    row = todo["slots"][0]
    row.update(
        {
            "selected_candidate_id": candidate["candidate_id"],
            "fine_cut_order": 1,
            "fine_cut_output": "clips/episode-01.mp4",
            "clip_transcript_path": str(clip_transcript),
            "clip_transcript_sha256": sha256_file(clip_transcript),
            "approved_quote": "医生说三个月以后复查，我后来按时去了。",
            "approved_clip_text": "医生说三个月以后复查，我后来按时去了。",
            "approved_window": {"start": 0.0, "end": 4.0},
            "approved_source_segment_ids": ["seg-customer-01"],
            "expected_speaker_role": "customer",
            "expected_speaker_ids": ["speaker-global-001"],
            "excluded_speaker_roles": ["interviewer"],
            "excluded_speaker_ids": ["speaker-global-002"],
            "label": "customer_quote",
            "subtitle_text": "医生说三个月以后复查，我后来按时去了。",
            "human_confirmed": True,
        }
    )
    review_path = bundle / "script-clip-review-notes.confirmed.json"
    write_json(review_path, todo)
    return review_path


def test_candidate_pack_reuses_local_retrieval_and_preserves_sources(tmp_path: Path) -> None:
    bundle, script, transcript = _bundle(tmp_path, roles=False)
    request = _request(tmp_path / "request.json", script, transcript)
    timeline_hash = sha256_file(bundle / "timeline.json")
    transcript_hash = sha256_file(transcript)

    result = build_script_clip_candidate_pack(bundle, request, top_k=3, write=True)

    assert result["schema"] == "video_knowledge_pipeline.script_clip_candidate_pack.v1"
    assert result["publication_allowed"] is False
    assert result["slots"][0]["candidates"]
    candidate = result["slots"][0]["candidates"][0]
    assert candidate["source_time_range"]["start"] == 0.0
    assert candidate["source_time_range"]["end"] == 4.0
    assert candidate["retrieval"]["origin"] == "script_preferred_window"
    assert candidate["snippet"] == "医生说三个月以后复查，我后来按时去了。 你当时是不是很担心费用？"
    assert candidate["speaker_evidence"]["role_status"] == "unresolved"
    assert candidate["speaker_evidence"]["role_inference_performed"] is False
    assert sha256_file(bundle / "timeline.json") == timeline_hash
    assert sha256_file(transcript) == transcript_hash
    assert (bundle / "exports" / "script-clip-candidate-pack.md").exists()
    assert (bundle / "script-clip-review-notes.todo.json").exists()
    assert read_json(bundle / "script-clip-review-notes.todo.json")["candidate_pack"]["artifact_sha256"]

    mcp = run_mcp_call("script_clip_candidate_pack", bundle / "mcp-script-clip-candidate-pack.args.json")
    assert mcp["schema"] == result["schema"]
    assert mcp["operator_boundary"]["external_provider_called"] is False


def test_alignment_ready_requires_clip_transcript_and_remains_review_only(tmp_path: Path) -> None:
    bundle, script, transcript = _bundle(tmp_path)
    request = _request(tmp_path / "request.json", script, transcript)
    build_script_clip_candidate_pack(bundle, request, top_k=2, write=True)
    clip_transcript = _clip_transcript(tmp_path / "clip.json", "医生说三个月以后复查，我后来按时去了。")
    review = _confirmed_review(bundle, clip_transcript)
    fine_cut = tmp_path / "fine-cut.json"
    write_json(
        fine_cut,
        {
            "schema": "synthetic-fine-cut.v1",
            "clips": [
                {"order": 1, "topic": "episode-01", "keep_ranges": [["00:00:00.000", "00:00:04.000"]], "output": "clips/episode-01.mp4"}
            ],
        },
    )

    result = check_script_clip_alignment(bundle, review, fine_cut, write=True)

    assert result["status"] == "ready_for_human_final_review"
    assert result["summary"]["issue_count"] == 0
    assert result["publication_allowed"] is False
    assert result["operator_boundary"]["human_final_review_required"] is True
    assert (bundle / "exports" / "script-clip-alignment-check.json").exists()
    assert (bundle / "script-clip-repair.todo.json").exists()
    mcp = run_mcp_call("script_clip_alignment_check", bundle / "mcp-script-clip-alignment-check.args.json")
    assert mcp["status"] == "ready_for_human_final_review"
    workbench = export_video_workbench(bundle, write=False)
    artifact_keys = {row["key"] for row in workbench["artifacts"]}
    assert "script_clip_candidate_pack_markdown" in artifact_keys
    assert "script_clip_alignment_check_markdown" in artifact_keys
    reuse = {row["key"]: row for row in workbench["external_reuse_status"]["capabilities"]}
    assert reuse["script_clip_alignment"]["run_count"] == 2


def test_alignment_flags_cross_source_clip_speaker_cut_subtitle_and_binding_failures(tmp_path: Path) -> None:
    bundle, script, transcript = _bundle(tmp_path, roles=False)
    slots = [
        {
            "slot_id": "slot-missing",
            "state": "planned",
            "story_segment_ref": "episode-missing",
            "episode_binding": "episode-missing",
            "required": True,
            "search_queries": [],
            "expected_quote": "不存在的候选",
            "subtitle_candidate": "",
            "required_speaker_roles": ["customer"],
            "excluded_speaker_roles": [],
        },
        {
            "slot_id": "slot-mixed",
            "state": "recorded",
            "story_segment_ref": "episode-shared",
            "episode_binding": "episode-shared",
            "required": True,
            "search_queries": ["三十万元"],
            "expected_quote": "听到接近三十万元，钱还是有点吃力。",
            "subtitle_candidate": "因为保险公司一定赔付三十万元，所以客户及时治疗。",
            "preferred_window": {"start": 4.0, "end": 11.0},
            "required_speaker_roles": ["customer"],
            "excluded_speaker_roles": ["interviewer"],
        },
        {
            "slot_id": "slot-source-only",
            "state": "recorded",
            "story_segment_ref": "episode-shared",
            "episode_binding": "episode-shared",
            "required": True,
            "search_queries": ["后来按时去了"],
            "expected_quote": "医生说三个月以后复查，我后来按时去了。",
            "subtitle_candidate": "医生说三个月以后复查，我后来按时去了。",
            "preferred_window": {"start": 0.0, "end": 4.0},
            "required_speaker_roles": ["customer"],
            "excluded_speaker_roles": [],
        },
    ]
    request = _request(tmp_path / "request.json", script, transcript, slots=slots)
    pack = build_script_clip_candidate_pack(bundle, request, top_k=2, write=True)
    todo = read_json(bundle / "script-clip-review-notes.todo.json")
    todo["review_status"] = "human_confirmed"
    todo["review_id"] = "review-failures"
    todo["candidate_pack"]["artifact_sha256"] = sha256_file(bundle / "exports" / "script-clip-candidate-pack.json")
    by_id = {row["slot_id"]: row for row in todo["slots"]}
    pack_by_id = {row["slot_id"]: row for row in pack["slots"]}

    mixed_clip = _clip_transcript(tmp_path / "mixed-clip.json", "听到接近三十万元，钱还是有点吃力，所以保险公司一定赔付30万。")
    mixed = by_id["slot-mixed"]
    mixed.update(
        {
            "selected_candidate_id": pack_by_id["slot-mixed"]["candidates"][0]["candidate_id"],
            "fine_cut_order": 1,
            "fine_cut_output": "clips/mixed.mp4",
            "clip_transcript_path": str(mixed_clip),
            "clip_transcript_sha256": sha256_file(mixed_clip),
            "approved_quote": "听到接近三十万元，钱还是有点吃力。",
            "approved_clip_text": "听到接近三十万元，钱还是有点吃力。",
            "approved_window": {"start": 7.0, "end": 11.0},
            "approved_source_segment_ids": ["seg-customer-02"],
            "expected_speaker_role": "",
            "expected_speaker_ids": ["speaker-global-001"],
            "excluded_speaker_roles": ["interviewer"],
            "excluded_speaker_ids": ["speaker-global-002"],
            "label": "customer_quote",
            "subtitle_text": "因为保险公司一定赔付三十万元，所以客户及时治疗。",
            "human_confirmed": True,
        }
    )
    source_only_clip = _clip_transcript(tmp_path / "source-only-clip.json", "这是另一段完全不同的内容。")
    source_only = by_id["slot-source-only"]
    source_only.update(
        {
            "episode_binding": "episode-shared",
            "selected_candidate_id": pack_by_id["slot-source-only"]["candidates"][0]["candidate_id"],
            "fine_cut_order": 2,
            "fine_cut_output": "clips/source-only.mp4",
            "clip_transcript_path": str(source_only_clip),
            "clip_transcript_sha256": sha256_file(source_only_clip),
            "approved_quote": "医生说三个月以后复查，我后来按时去了。",
            "approved_clip_text": "医生说三个月以后复查，我后来按时去了。",
            "approved_window": {"start": 0.0, "end": 4.0},
            "approved_source_segment_ids": ["seg-customer-01"],
            "expected_speaker_role": "",
            "expected_speaker_ids": [],
            "excluded_speaker_roles": [],
            "excluded_speaker_ids": [],
            "label": "customer_quote",
            "subtitle_text": "医生说三个月以后复查，我后来按时去了。",
            "human_confirmed": True,
        }
    )
    review = bundle / "review-failures.json"
    write_json(review, todo)
    fine_cut = tmp_path / "fine-cut-failures.json"
    write_json(
        fine_cut,
        {
            "clips": [
                {"order": 1, "keep_ranges": [["00:00:04.500", "00:00:10.500"]], "output": "clips/mixed.mp4"},
                {"order": 2, "keep_ranges": [["00:00:00.000", "00:00:04.000"]], "output": "clips/source-only.mp4"},
            ]
        },
    )

    result = check_script_clip_alignment(bundle, review, fine_cut, write=False)
    codes = {issue["code"] for row in result["slots"] for issue in row["issues"]}

    assert {
        "missing_required_slot",
        "candidate_not_searched",
        "speaker_role_unresolved",
        "excluded_speaker_present",
        "cut_outside_approved_window",
        "approved_quote_missing_after_cut",
        "clip_contains_unreviewed_claim",
        "subtitle_semantic_expansion",
        "sentence_fragment_at_boundary",
        "multiple_speakers_mislabeled_as_customer_quote",
        "source_only_not_clip_present",
        "duplicate_or_conflicting_episode_binding",
    } <= codes
    assert result["status"] == "needs_candidate_selection"
    assert result["publication_allowed"] is False


def test_candidate_pack_and_alignment_fail_closed_on_hash_drift_and_no_write(tmp_path: Path) -> None:
    bundle, script, transcript = _bundle(tmp_path)
    request = _request(tmp_path / "request.json", script, transcript)
    preview = build_script_clip_candidate_pack(bundle, request, top_k=2, write=False)
    assert preview["slots"][0]["candidates"]
    assert not (bundle / "exports" / "script-clip-candidate-pack.json").exists()
    with pytest.raises(ValueError, match="sqlite retrieval writes"):
        build_script_clip_candidate_pack(bundle, request, retrieval_backend="sqlite", write=False)

    build_script_clip_candidate_pack(bundle, request, top_k=2, write=True)
    clip = _clip_transcript(tmp_path / "clip.json", "医生说三个月以后复查，我后来按时去了。")
    review = _confirmed_review(bundle, clip)
    pack_path = bundle / "exports" / "script-clip-candidate-pack.json"
    pack = read_json(pack_path)
    pack["status"] = "needs_search"
    write_json(pack_path, pack)
    fine_cut = tmp_path / "fine-cut.json"
    write_json(fine_cut, {"clips": [{"order": 1, "keep_ranges": [[0.0, 4.0]], "output": "clips/episode-01.mp4"}]})
    with pytest.raises(ValueError, match="semantic hash changed"):
        check_script_clip_alignment(bundle, review, fine_cut, write=False)


def test_request_and_review_slot_identity_fail_closed(tmp_path: Path) -> None:
    bundle, script, transcript = _bundle(tmp_path)
    duplicate_slot = {
        "slot_id": "duplicate-slot",
        "state": "recorded",
        "story_segment_ref": "episode-01",
        "required": True,
        "search_queries": ["三个月以后复查"],
        "preferred_window": {"start": 0.0, "end": 4.0},
    }
    duplicate_request = _request(
        tmp_path / "duplicate-request.json",
        script,
        transcript,
        slots=[duplicate_slot, dict(duplicate_slot)],
    )
    with pytest.raises(ValueError, match="duplicate slot_id"):
        build_script_clip_candidate_pack(bundle, duplicate_request, write=False)

    request = _request(tmp_path / "request.json", script, transcript)
    build_script_clip_candidate_pack(bundle, request, top_k=2, write=True)
    clip = _clip_transcript(tmp_path / "clip.json", "医生说三个月以后复查，我后来按时去了。")
    review = _confirmed_review(bundle, clip)
    review_payload = read_json(review)
    unknown = dict(review_payload["slots"][0])
    unknown["slot_id"] = "unknown-slot"
    review_payload["slots"].append(unknown)
    write_json(review, review_payload)
    fine_cut = tmp_path / "fine-cut.json"
    write_json(fine_cut, {"clips": [{"order": 1, "keep_ranges": [[0.0, 4.0]], "output": "clips/episode-01.mp4"}]})
    with pytest.raises(ValueError, match="outside candidate pack"):
        check_script_clip_alignment(bundle, review, fine_cut, write=False)


def test_candidate_window_allows_unrelated_legacy_chunk_overlap(tmp_path: Path) -> None:
    bundle, script, transcript = _bundle(tmp_path)
    payload = read_json(transcript)
    payload["segments"].extend(
        [
            {"segment_id": "late-001", "start": 100.01, "end": 101.0, "text": "远端分块一"},
            {"segment_id": "late-002", "start": 100.0, "end": 102.0, "text": "远端分块二"},
        ]
    )
    write_json(transcript, payload)
    request = _request(tmp_path / "request.json", script, transcript)

    result = build_script_clip_candidate_pack(bundle, request, top_k=2, write=False)

    receipt = result["slots"][0]["candidates"][0]["reference_window_receipt"]
    assert receipt["window"]["validation_scope"] == "window"
    assert receipt["segment_count"] >= 1


def test_cli_contracts_are_registered(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle, script, transcript = _bundle(tmp_path)
    request = _request(tmp_path / "request.json", script, transcript)
    parsed = build_parser().parse_args(["script-clip-candidate-pack", str(bundle), str(request), "--no-write"])
    assert parsed.command == "script-clip-candidate-pack"
    assert main(["script-clip-candidate-pack", str(bundle), str(request), "--no-write"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == "video_knowledge_pipeline.script_clip_candidate_pack.v1"
    source = inspect.getsource(mcp_server.main)
    assert "def script_clip_candidate_pack_tool(" in source
    assert "def script_clip_alignment_check_tool(" in source
