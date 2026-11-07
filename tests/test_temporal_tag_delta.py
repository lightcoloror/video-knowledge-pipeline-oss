from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.general_tagger_adapter import run_general_tagger
from video_knowledge_pipeline.temporal_tag_delta import (
    apply_temporal_tag_delta_restore,
    analyze_temporal_tag_observations,
    run_temporal_tag_delta,
)


def _observation(position: int, tags: list[str], *, ocr_text: str = "") -> dict:
    return {
        "frame_position": position,
        "frame_path": f"frame-{position}.jpg",
        "tags": tags,
        "ocr_text": ocr_text,
    }


def _bundle_with_sequences(tmp_path: Path, *, count: int = 1, frames: int = 5) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"title": "synthetic"}), encoding="utf-8"
    )
    timeline = []
    for index in range(1, count + 1):
        frame_paths = []
        for position in range(1, frames + 1):
            path = bundle / f"segment-{index}-frame-{position}.jpg"
            path.write_bytes(f"synthetic-{index}-{position}".encode())
            frame_paths.append(
                {
                    "path": path.name,
                    "timestamp": float((index - 1) * 10 + position - 1),
                    "frame_id": f"segment-{index}-source-{position}",
                }
            )
        timeline.append(
            {
                "index": index,
                "start": float((index - 1) * 10),
                "end": float(index * 10),
                "transcript": "讲师讲解静态课件。",
                "temporal_frame_paths": frame_paths,
            }
        )
    (bundle / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False), encoding="utf-8"
    )
    return bundle


def test_tag_delta_smooths_single_frame_gap_and_flicker() -> None:
    result = analyze_temporal_tag_observations(
        [
            _observation(1, ["讲师", "白板"]),
            _observation(2, ["讲师", "临时噪声"]),
            _observation(3, ["讲师", "白板"]),
        ]
    )

    assert result["status"] == "completed"
    assert result["decision"] == "coarse_weak_evidence"
    assert result["smoothed_observations"][1]["tags"] == ["白板", "讲师"]
    assert result["stable_tags"] == ["白板", "讲师"]
    assert result["transitions"] == []
    assert "可能" in result["coarse_summary"]
    assert "不据此推断动作" in result["coarse_summary"]
    assert result["limitations"]


def test_tag_delta_escalates_dynamic_or_cross_channel_change() -> None:
    result = analyze_temporal_tag_observations(
        [
            _observation(1, ["桌面", "窗口"], ocr_text="首页"),
            _observation(2, ["菜单", "鼠标"], ocr_text="设置"),
            _observation(3, ["对话框", "按钮"], ocr_text="确认"),
        ],
        transcript="接下来点击按钮，切换到设置页。",
        supporting_signals={
            "ocr_changed": True,
            "scene_boundary": False,
            "frame_change_status": "dynamic",
        },
    )

    assert result["decision"] == "temporal_multimodal_required"
    assert "high_tag_change" in result["escalation_reasons"]
    assert "ocr_changed" in result["escalation_reasons"]
    assert "asr_dynamic_term" in result["escalation_reasons"]
    assert "需升级连续帧多模态理解" in result["coarse_summary"]


def test_temporal_tag_delta_legacy_import_remains_readable_but_not_v2_sufficient(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"title": "synthetic"}), encoding="utf-8")
    frame_paths = []
    for position in range(1, 6):
        frame = bundle / f"f{position}.jpg"
        frame.write_bytes(f"synthetic-{position}".encode())
        frame_paths.append(
            {
                "path": frame.name,
                "timestamp": float(position - 1) * 0.75,
                "frame_id": f"source-{position}",
            }
        )
    timeline = [
        {
            "index": 1,
            "start": 0,
            "end": 4,
            "transcript": "讲师站在白板前。",
            "temporal_frame_paths": frame_paths,
        }
    ]
    (bundle / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
    input_path = bundle / "tag-input.json"
    input_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.temporal_tag_delta_input.v1",
                "items": [
                    {
                        "index": 1,
                        "frames": [
                            _observation(1, ["讲师", "白板"]),
                            _observation(2, ["讲师", "白板"]),
                            _observation(3, ["讲师", "白板"]),
                            _observation(4, ["讲师", "白板"]),
                            _observation(5, ["讲师", "白板"]),
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_temporal_tag_delta(bundle, input_json=input_path)
    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0]

    assert result["status"] == "needs_frame_tags"
    assert result["coarse_static_count"] == 0
    assert updated["temporal_tag_delta"]["decision"] == "review_gap"
    assert updated["temporal_tag_delta"]["evidence_level"] == "unavailable"
    assert "legacy_v1_identity_unverified" in updated["temporal_tag_delta"]["escalation_reasons"]
    assert updated["temporal_multimodal_escalation"]["required"] is False
    assert updated["temporal_multimodal_escalation"]["fallback_route"] == "local_retag_or_review_gap"
    assert "temporal_visual_understanding" not in updated
    assert (bundle / "temporal-tag-delta.json").is_file()
    assert (bundle / "temporal-tag-delta.md").is_file()


def test_general_tagger_continuous_mode_tags_every_ordered_frame(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    frame_paths = []
    for position in range(1, 4):
        path = bundle / f"frame-{position}.jpg"
        path.write_bytes(f"synthetic-{position}".encode())
        frame_paths.append(path.name)
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 7,
                    "start": 0,
                    "end": 3,
                    "temporal_frame_paths": frame_paths,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = run_general_tagger(
        bundle,
        frame_mode="continuous",
        execute=True,
        import_annotations=False,
        write=False,
        _inference_backend=lambda path: {"tags_zh": [path.stem], "tags_en": []},
    )

    assert result["status"] == "completed"
    assert result["planned_frame_count"] == 3
    assert [row["frame_position"] for row in result["annotations"]] == [1, 2, 3]
    assert [row["frame_id"] for row in result["annotations"]] == [
        "timeline-000007-frame-001",
        "timeline-000007-frame-002",
        "timeline-000007-frame-003",
    ]


def test_general_tagger_checkpoint_revision_invalidates_frame_cache(tmp_path: Path) -> None:
    bundle = _bundle_with_sequences(tmp_path, frames=5)
    checkpoint = tmp_path / "ram-plus.pth"
    checkpoint.write_bytes(b"revision-one")
    cache_dir = tmp_path / "cache"
    first_calls: list[str] = []
    run_general_tagger(
        bundle,
        checkpoint_path=checkpoint,
        frame_mode="continuous",
        execute=True,
        import_annotations=False,
        write=False,
        timeline_indexes=[1],
        per_timeline_limit=5,
        cache_dir=cache_dir,
        run_id="cache-revision-test",
        _inference_backend=lambda path: (
            first_calls.append(path.name) or {"tags_zh": ["讲师"], "tags_en": []}
        ),
    )
    checkpoint.write_bytes(b"revision-two")
    second_calls: list[str] = []
    second = run_general_tagger(
        bundle,
        checkpoint_path=checkpoint,
        frame_mode="continuous",
        execute=True,
        import_annotations=False,
        write=False,
        timeline_indexes=[1],
        per_timeline_limit=5,
        cache_dir=cache_dir,
        run_id="cache-revision-test",
        _inference_backend=lambda path: (
            second_calls.append(path.name) or {"tags_zh": ["课件"], "tags_en": []}
        ),
    )

    assert len(first_calls) == 5
    assert len(second_calls) == 5
    assert second["cache_hit_count"] == 0


def test_temporal_tag_delta_limits_sequences_and_backend_calls(tmp_path: Path) -> None:
    bundle = _bundle_with_sequences(tmp_path, count=3, frames=8)
    calls: list[str] = []

    result = run_temporal_tag_delta(
        bundle,
        execute_tagger=True,
        indexes="1,3",
        frame_count=5,
        limit=1,
        write=False,
        _inference_backend=lambda path: (
            calls.append(path.name) or {"tags_zh": ["讲师", "课件"], "tags_en": []}
        ),
    )

    assert result["selected_indexes"] == [1]
    assert result["remaining_indexes"] == [3]
    assert any("--indexes 3" in command for command in result["next_actions"])
    assert result["processed_frame_count"] == 5
    assert len(calls) == 5
    assert all(name.startswith("segment-1-") for name in calls)
    pack = result["sequence_pack"]
    assert pack["schema"] == "video_knowledge_pipeline.temporal_tag_sequence.v2"
    assert pack["timeline_sha256"]
    assert pack["manifest_sha256"]
    assert pack["sequences"][0]["sequence_sha256"]
    assert len(pack["sequences"][0]["frames"]) == 5
    assert all(frame["sha256"] for frame in pack["sequences"][0]["frames"])

    with pytest.raises(ValueError, match="between 1 and 16"):
        run_temporal_tag_delta(bundle, limit=17, write=False)


def test_temporal_tag_delta_rejects_frame_outside_bundle(tmp_path: Path) -> None:
    bundle = _bundle_with_sequences(tmp_path)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    timeline[0]["temporal_frame_paths"][0]["path"] = str(outside)
    (bundle / "timeline.json").write_text(json.dumps(timeline), encoding="utf-8")

    with pytest.raises(ValueError, match="outside_bundle"):
        run_temporal_tag_delta(bundle, write=False)


def test_temporal_tag_quality_gate_preserves_supported_transient_and_boundaries() -> None:
    result = analyze_temporal_tag_observations(
        [
            _observation(1, ["讲师", "课件"], ocr_text="标题"),
            {**_observation(2, ["讲师", "弹窗"], ocr_text="确认"), "boundary_before": True},
            _observation(3, ["讲师", "课件"], ocr_text="标题"),
            _observation(4, ["讲师", "课件"], ocr_text="标题"),
            _observation(5, ["讲师", "课件"], ocr_text="标题"),
        ],
        expected_frame_count=5,
        supporting_signals={"scene_boundary": True},
    )

    assert "弹窗" in result["smoothed_observations"][1]["tags"]
    assert result["decision"] == "temporal_multimodal_required"
    assert "scene_boundary" in result["escalation_reasons"]


def test_temporal_tag_quality_gate_requires_complete_observation() -> None:
    result = analyze_temporal_tag_observations(
        [_observation(position, ["讲师", "课件"]) for position in range(1, 5)],
        expected_frame_count=5,
        min_frames=5,
    )

    assert result["status"] == "review_gap"
    assert result["decision"] == "review_gap"
    assert result["evidence_level"] == "unavailable"
    assert "incomplete_frame_observation" in result["escalation_reasons"]


def test_temporal_tag_delta_invalid_import_is_atomic_and_restore_is_complete(tmp_path: Path) -> None:
    bundle = _bundle_with_sequences(tmp_path, frames=5)
    before = (bundle / "timeline.json").read_bytes()
    prepared = run_temporal_tag_delta(bundle, write=False)
    invalid = dict(prepared["sequence_pack"])
    invalid["sequences"] = [dict(prepared["sequence_pack"]["sequences"][0])]
    invalid["sequences"][0]["frames"] = [
        dict(frame) for frame in prepared["sequence_pack"]["sequences"][0]["frames"]
    ]
    invalid["sequences"][0]["frames"][0]["sha256"] = "0" * 64
    invalid_path = bundle / "invalid-tag-input.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")

    rejected = run_temporal_tag_delta(bundle, input_json=invalid_path)

    assert rejected["status"] == "rejected"
    assert (bundle / "timeline.json").read_bytes() == before
    assert Path(rejected["validation_report_path"]).is_file()

    completed = run_temporal_tag_delta(
        bundle,
        execute_tagger=True,
        _inference_backend=lambda _: {"tags_zh": ["讲师", "课件"], "tags_en": []},
    )
    assert completed["status"] == "completed"
    restore = apply_temporal_tag_delta_restore(
        bundle,
        completed["restore_plan"],
        execute=True,
        confirm_run_id=completed["run_id"],
    )
    assert restore["status"] == "restored"
    assert (bundle / "timeline.json").read_bytes() == before


def test_temporal_tag_delta_prepare_then_strict_import_keeps_bundle_identity(tmp_path: Path) -> None:
    bundle = _bundle_with_sequences(tmp_path, frames=5)
    timeline_before = (bundle / "timeline.json").read_bytes()
    manifest_before = (bundle / "manifest.json").read_bytes()

    prepared = run_temporal_tag_delta(bundle)

    assert prepared["status"] == "prepared"
    assert (bundle / "timeline.json").read_bytes() == timeline_before
    assert (bundle / "manifest.json").read_bytes() == manifest_before
    template_path = Path(prepared["input_template_path"])
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    payload["result_metadata"] = {
        "executor": "synthetic_external_tagger",
        "model_actual": "synthetic-tags-v1",
        "model_revision": "fixture",
    }
    for sequence in payload["sequences"]:
        for frame in sequence["frames"]:
            frame["tags"] = ["讲师", "课件"]
            frame["confidence"] = 0.9
    invalid_payload = json.loads(json.dumps(payload))
    invalid_payload["result_metadata"]["model_actual"] = ""
    invalid_path = bundle / "strict-invalid-before-success.json"
    invalid_path.write_text(json.dumps(invalid_payload, ensure_ascii=False), encoding="utf-8")
    rejected = run_temporal_tag_delta(
        bundle,
        input_json=invalid_path,
        resume=prepared["run_id"],
    )
    assert rejected["status"] == "rejected"
    assert (bundle / "timeline.json").read_bytes() == timeline_before
    assert (bundle / "manifest.json").read_bytes() == manifest_before
    template_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    imported = run_temporal_tag_delta(
        bundle,
        input_json=template_path,
        resume=prepared["run_id"],
    )

    assert imported["status"] == "completed"
    assert imported["coarse_static_count"] == 1
    assert imported["processed_frame_count"] == 5
    assert imported["items"][0]["evidence_level"] == "candidate_weak"
    assert imported["items"][0]["raw_observations"][0]["model_actual"] == "synthetic-tags-v1"


def test_temporal_tag_delta_strict_import_rejects_task_set_metadata_and_long_fields(tmp_path: Path) -> None:
    bundle = _bundle_with_sequences(tmp_path, frames=5)
    prepared = run_temporal_tag_delta(bundle, write=False)

    def valid_payload() -> dict:
        payload = json.loads(json.dumps(prepared["sequence_pack"]))
        payload["result_metadata"] = {
            "executor": "synthetic_external_tagger",
            "model_actual": "synthetic-tags-v1",
            "model_revision": "fixture",
        }
        for sequence in payload["sequences"]:
            for frame in sequence["frames"]:
                frame["tags"] = ["讲师", "课件"]
        return payload

    payloads = []
    missing_task = valid_payload()
    missing_task["sequences"] = []
    payloads.append(missing_task)
    extra_task = valid_payload()
    extra_task["sequences"].append(json.loads(json.dumps(extra_task["sequences"][0])))
    payloads.append(extra_task)
    missing_model = valid_payload()
    missing_model["result_metadata"]["model_actual"] = ""
    payloads.append(missing_model)
    long_tag = valid_payload()
    long_tag["sequences"][0]["frames"][0]["tags"] = ["x" * 201]
    payloads.append(long_tag)

    for position, payload in enumerate(payloads, start=1):
        path = bundle / f"invalid-strict-{position}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        rejected = run_temporal_tag_delta(
            bundle,
            input_json=path,
            resume=prepared["run_id"],
            write=False,
        )
        assert rejected["status"] == "rejected"


def test_temporal_tag_delta_resume_reuses_successful_frame_checkpoints(tmp_path: Path) -> None:
    bundle = _bundle_with_sequences(tmp_path, frames=5)
    first_calls: list[str] = []

    def interrupted(path: Path) -> dict:
        first_calls.append(path.name)
        if len(first_calls) == 3:
            raise RuntimeError("synthetic interruption")
        return {"tags_zh": ["讲师", "课件"], "tags_en": []}

    first = run_temporal_tag_delta(
        bundle,
        execute_tagger=True,
        resume="checkpoint-test",
        _inference_backend=interrupted,
    )
    assert first["status"] == "needs_frame_tags"
    assert len(first_calls) == 3

    resumed_calls: list[str] = []
    resumed = run_temporal_tag_delta(
        bundle,
        execute_tagger=True,
        resume="checkpoint-test",
        _inference_backend=lambda path: (
            resumed_calls.append(path.name) or {"tags_zh": ["讲师", "课件"], "tags_en": []}
        ),
    )

    assert resumed["status"] == "completed"
    assert len(resumed_calls) == 3
    assert resumed["tagger"]["cache_hit_count"] == 2


def test_temporal_tag_delta_runtime_frame_drift_fails_before_timeline_commit(tmp_path: Path) -> None:
    bundle = _bundle_with_sequences(tmp_path, frames=5)
    timeline_before = (bundle / "timeline.json").read_bytes()
    calls = 0

    def drifting_backend(path: Path) -> dict:
        nonlocal calls
        calls += 1
        if calls == 5:
            (bundle / "segment-1-frame-1.jpg").write_bytes(b"drifted-after-observation")
        return {"tags_zh": ["讲师", "课件"], "tags_en": []}

    with pytest.raises(RuntimeError, match="frame_hash_drift_before_commit"):
        run_temporal_tag_delta(bundle, execute_tagger=True, _inference_backend=drifting_backend)

    assert (bundle / "timeline.json").read_bytes() == timeline_before
