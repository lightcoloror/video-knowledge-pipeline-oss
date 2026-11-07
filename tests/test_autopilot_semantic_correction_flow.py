from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import autopilot, bundle_next
from video_knowledge_pipeline.artifact_validation import sha256_file
from video_knowledge_pipeline.transcript_semantic_correction import (
    build_transcript_semantic_candidate_discovery_pack,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _semantic_status(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "status": "needs_llm_or_codex_review",
        "candidate_discovery_status": "imported",
        "candidate_count": 2,
        "review_required_count": 0,
        "candidate_discovery_complete": True,
        "closure_status": "missing",
        "artifact_identity": {"current": True},
        "candidate_discovery_artifacts": {},
        "artifacts": {},
    }
    value.update(overrides)
    return value


def _paged_discovery_bundle(tmp_path: Path, *, cue_count: int = 17) -> Path:
    root = tmp_path / "paged-discovery-bundle"
    root.mkdir()
    segments = [
        {
            "index": index,
            "segment_id": f"cue-{index:04d}",
            "start": index * 3.0,
            "end": index * 3.0 + 2.0,
            "text": f"合成课程正文第{index}段，需要逐段完成候选发现。",
        }
        for index in range(cue_count)
    ]
    _write_json(
        root / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "title": "Paged discovery synthetic",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    _write_json(root / "normalized-transcript.json", {"segments": segments})
    _write_json(root / "timeline.json", segments)
    _write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "status": "no_candidates",
            "candidate_count": 0,
            "candidates": [],
        },
    )
    return root


@pytest.mark.parametrize(
    ("status", "expected_key", "human_required"),
    [
        (
            _semantic_status(
                candidate_discovery_status="not_planned",
                candidate_discovery_complete=False,
            ),
            "semantic_candidate_discovery",
            False,
        ),
        (
            _semantic_status(
                candidate_discovery_status="suggestions_ready",
                candidate_discovery_complete=False,
            ),
            "semantic_candidate_import",
            False,
        ),
        (_semantic_status(), "semantic_correction_review", False),
        (
            _semantic_status(
                status="needs_human_review_or_new_result",
                review_required_count=2,
            ),
            "semantic_correction_review_gap",
            True,
        ),
    ],
)
def test_bundle_next_semantic_gate_exposes_exact_unfinished_action(
    monkeypatch: pytest.MonkeyPatch,
    status: dict[str, object],
    expected_key: str,
    human_required: bool,
) -> None:
    monkeypatch.setattr(
        bundle_next,
        "transcript_semantic_correction_status",
        lambda *_args, **_kwargs: status,
    )
    monkeypatch.setattr(
        bundle_next,
        "_semantic_discovery_scan_state",
        lambda _root: {
            "complete": bool(status["candidate_discovery_complete"]),
            "next_cursor": 0,
        },
    )
    writes: list[tuple[Path, object]] = []
    monkeypatch.setattr(bundle_next, "write_json", lambda path, value: writes.append((Path(path), value)))

    action = bundle_next._next_semantic_correction_action(Path("synthetic-bundle"), {})

    assert action is not None
    assert action["status"] == (
        "human_review_required" if human_required else "semantic_correction_pending"
    )
    assert action["key"] == expected_key
    assert action["human_required"] is human_required
    assert action["semantic_status"]["candidate_discovery_status"] == status["candidate_discovery_status"]
    if not human_required:
        assert writes


def test_bundle_next_semantic_gate_requires_real_closure_but_allows_closed_review_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = iter(
        [
            _semantic_status(
                status="arbitrated_no_change",
                candidate_count=0,
                closure_status="missing",
            ),
            _semantic_status(
                status="needs_impact_report",
                closure_status="completed_no_text_changes",
                review_required_count=3,
            ),
        ]
    )
    monkeypatch.setattr(
        bundle_next,
        "transcript_semantic_correction_status",
        lambda *_args, **_kwargs: next(statuses),
    )
    monkeypatch.setattr(
        bundle_next,
        "_semantic_discovery_scan_state",
        lambda _root: {"complete": True, "next_cursor": None},
    )
    monkeypatch.setattr(bundle_next, "write_json", lambda *_args, **_kwargs: None)

    closure = bundle_next._next_semantic_correction_action(Path("synthetic-bundle"), {})
    closed = bundle_next._next_semantic_correction_action(Path("synthetic-bundle"), {})

    assert closure is not None
    assert closure["key"] == "semantic_correction_review"
    assert closed is None


def test_semantic_discovery_work_item_is_single_bounded_hash_bound_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).resolve()
    monkeypatch.setattr(
        autopilot,
        "build_transcript_semantic_candidate_discovery_pack",
        lambda *_args, **_kwargs: {
            "status": "discovery_prompt_ready",
            "segment_count": 8,
            "artifacts": {
                "candidate_discovery_pack_json": str(source),
                "candidate_discovery_prompt_markdown": str(source),
            },
        },
    )
    monkeypatch.setattr(autopilot, "_identity_refs", lambda _root: [])
    monkeypatch.setattr(autopilot, "_bundle_identity", lambda _root: "bundle-sha")
    monkeypatch.setattr(
        autopilot,
        "_refresh_route",
        lambda _job, _root: {"resolved_route": "local-only"},
    )
    job = {
        "job_id": "job-1",
        "state_revision": 3,
        "runner": "agent-session",
        "job_path": str(source),
        "execution_route": {"requested_route": "local-only"},
        "policy_snapshot": {
            "online_derived_artifacts_allowed": False,
            "local_vision_provider": {},
        },
    }
    action = {
        "status": "semantic_correction_pending",
        "kind": "semantic_correction",
        "key": "semantic_candidate_discovery",
        "mcp_tool": "transcript_semantic_candidate_discovery_pack",
        "reason": "scan bounded transcript windows",
    }

    work_item = autopilot._prepare_work_item(
        source.parent, job, action, fingerprint="fp"
    )

    assert work_item["task_type"] == "transcript_semantic_candidate_discovery"
    assert work_item["model_policy"]["executor"] == "agent_session"
    assert work_item["model_policy"]["provider_health_check"] == "not_applicable"
    assert work_item["output_schema"]["properties"]["suggestions"]["maxItems"] == 8
    assert {row["role"] for row in work_item["input_refs"]} >= {
        "semantic_candidate_discovery_pack",
        "semantic_candidate_discovery_prompt",
    }
    assert len(
        json.dumps(work_item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ) <= autopilot.MAX_WORK_ITEM_BYTES


def test_semantic_review_submission_uses_strong_validator_then_existing_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pack = {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
        "candidates": [{"candidate_id": "semcorr-0001"}],
    }
    model_result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
        "source": "autopilot_agent_session",
        "model_actual": "gpt-5.6-luna",
        "decisions": [],
        "candidate_selection": {"selected_candidate_count": 0},
    }
    work_item = {
        "task_type": "transcript_semantic_correction_review",
        "task_id": "task-1",
        "input_refs": [
            {"role": "semantic_correction_source_pack", "path": "source-pack.json"}
        ],
    }
    writes: list[tuple[Path, object]] = []
    closures: list[dict[str, object]] = []
    monkeypatch.setattr(autopilot, "read_json", lambda _path: source_pack)
    monkeypatch.setattr(
        autopilot,
        "validate_transcript_semantic_model_output",
        lambda *_args, **_kwargs: {
            "quality_gate_passed": True,
            "status": "qualified",
            "contract_issues": [],
            "quality_issues": [],
        },
    )
    monkeypatch.setattr(
        autopilot,
        "write_json",
        lambda path, value: writes.append((Path(path), value)),
    )

    def _closure(*_args: object, **kwargs: object) -> dict[str, object]:
        closures.append(dict(kwargs))
        return {"status": "completed_no_text_changes", "ok": True}

    monkeypatch.setattr(autopilot, "transcript_semantic_correction_closure", _closure)

    imported, issues, effective_status = autopilot._import_semantic_submission(
        Path("synthetic-bundle"),
        work_item,
        model_result,
        Path("synthetic-run"),
    )

    assert issues == []
    assert effective_status == "succeeded"
    assert imported["status"] == "completed_no_text_changes"
    assert closures == [
        {
            "input_json": Path("synthetic-run") / "semantic-review-task-1.json",
            "min_confidence": 0.88,
            "auto_apply": True,
            "refresh_exports": False,
            "write": True,
        }
    ]
    assert writes[0][1] == model_result


def test_semantic_review_submission_fails_closed_before_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_item = {
        "task_type": "transcript_semantic_correction_review",
        "task_id": "task-1",
        "input_refs": [
            {"role": "semantic_correction_source_pack", "path": "source-pack.json"}
        ],
    }
    monkeypatch.setattr(autopilot, "read_json", lambda _path: {"schema": "bad"})
    monkeypatch.setattr(
        autopilot,
        "validate_transcript_semantic_model_output",
        lambda *_args, **_kwargs: {
            "quality_gate_passed": False,
            "status": "contract_failed",
            "contract_issues": [{"key": "semantic_result_schema_mismatch", "detail": "bad"}],
            "quality_issues": [],
        },
    )
    closure_calls: list[bool] = []
    monkeypatch.setattr(
        autopilot,
        "transcript_semantic_correction_closure",
        lambda *_args, **_kwargs: closure_calls.append(True),
    )

    imported, issues, effective_status = autopilot._import_semantic_submission(
        Path("synthetic-bundle"),
        work_item,
        {"schema": "bad"},
        Path("synthetic-run"),
    )

    assert imported["status"] == "strong_validation_rejected"
    assert {row["key"] for row in issues} == {"semantic_result_schema_mismatch"}
    assert effective_status == "rejected"
    assert closure_calls == []


def test_autopilot_discovery_requires_three_imported_pages_before_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _paged_discovery_bundle(tmp_path, cue_count=17)
    run_dir = root / "runs" / "autopilot-page-test"
    job_path = run_dir / "autopilot-job.json"
    job: dict[str, object] = {
        "job_path": str(job_path),
        "semantic_candidate_discovery": {},
    }
    monkeypatch.setattr(
        bundle_next,
        "transcript_semantic_correction_status",
        lambda *_args, **_kwargs: _semantic_status(
            candidate_discovery_status="imported",
            candidate_count=0,
        ),
    )

    cursor = 0
    selected: list[int] = []
    for page_number in range(1, 4):
        prepared = build_transcript_semantic_candidate_discovery_pack(
            root,
            limit=8,
            cursor=cursor,
            write=True,
        )
        selected.extend(prepared["selected_source_indexes"])
        work_item = {
            "task_type": "transcript_semantic_candidate_discovery",
            "semantic_task": {
                "cursor": prepared["cursor"],
                "next_cursor": prepared["next_cursor"],
                "complete": prepared["complete"],
                "coverage": prepared["coverage"],
                "transcript_sha256": prepared["transcript_sha256"],
                "discovery_pack_sha256": sha256_file(
                    root / "transcript-semantic-candidate-discovery-pack.json"
                ),
            },
        }
        autopilot._record_semantic_discovery_progress(job, work_item)
        _write_json(job_path, job)

        action = bundle_next._next_semantic_correction_action(root, {})
        state = job["semantic_candidate_discovery"]
        assert isinstance(state, dict)
        if page_number < 3:
            assert prepared["complete"] is False
            assert action is not None
            assert action["key"] == "semantic_candidate_discovery"
            assert action["discovery_cursor"] == prepared["next_cursor"]
            cursor = int(prepared["next_cursor"])
        else:
            assert prepared["complete"] is True
            assert prepared["next_cursor"] is None
            assert state["complete"] is True
            assert state["coverage"]["covered_retained_cue_count"] == 17
            assert action is not None
            assert action["key"] == "semantic_correction_review"

    assert selected == list(range(17))
    assert len(job["semantic_candidate_discovery"]["batches"]) == 3


def test_semantic_discovery_transcript_hash_drift_resets_cursor(
    tmp_path: Path,
) -> None:
    root = _paged_discovery_bundle(tmp_path, cue_count=17)
    first = build_transcript_semantic_candidate_discovery_pack(
        root,
        limit=8,
        cursor=0,
        write=True,
    )
    job: dict[str, object] = {"semantic_candidate_discovery": {}}
    autopilot._record_semantic_discovery_progress(
        job,
        {
            "task_type": "transcript_semantic_candidate_discovery",
            "semantic_task": {
                "cursor": first["cursor"],
                "next_cursor": first["next_cursor"],
                "complete": first["complete"],
                "coverage": first["coverage"],
                "transcript_sha256": first["transcript_sha256"],
                "discovery_pack_sha256": sha256_file(
                    root / "transcript-semantic-candidate-discovery-pack.json"
                ),
            },
        },
    )
    transcript_path = root / "normalized-transcript.json"
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    payload["segments"][0]["text"] += "（输入发生变化）"
    _write_json(transcript_path, payload)

    preview = build_transcript_semantic_candidate_discovery_pack(
        root,
        limit=8,
        cursor=8,
        write=False,
    )
    cursor = autopilot._semantic_discovery_cursor(job, preview)
    scan = bundle_next._semantic_discovery_scan_state(root)

    assert cursor == 0
    assert job["semantic_candidate_discovery"]["batches"] == []
    assert job["semantic_candidate_discovery"]["transcript_sha256"] == preview["transcript_sha256"]
    assert scan["hash_drift"] is True
    assert scan["complete"] is False
    assert scan["next_cursor"] == 0
