from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.media_equivalence_audit as audit_module
from video_knowledge_pipeline.media_equivalence_audit import (
    DEFAULT_QUALITY_POLICY,
    QUALITY_POLICY_ARCHIVAL,
    QUALITY_POLICY_PRACTICAL,
    _parser,
    align_audio_fingerprints,
    audit_provenance_references,
    compare_asr_plans,
    compare_retained_quality,
    compare_visual_evidence,
    render_media_equivalence_markdown,
)


# Intent: prove the deletion gate fails closed without invoking FFmpeg or any
# model. Decision: exercise pure contract seams with synthetic fingerprints and
# local JSON fixtures. Reason: automatic tests must not upload or delete media.
# Evidence: the production module keeps extraction separate from aggregation.
# Effective scope: offline regression only.


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_aligned_audio_reports_full_voiced_coverage() -> None:
    fingerprint = [0xAAAAAAAA ^ index for index in range(400)]

    result = align_audio_fingerprints(
        fingerprint,
        [0x12345678] * 8 + fingerprint + [0x87654321] * 4,
        candidate_duration_seconds=50.0,
        retained_duration_seconds=51.5,
        voiced_intervals=[(0.0, 50.0)],
        max_alignment_seconds=5.0,
    )

    assert result["candidate_voiced_coverage"] == 1.0
    assert result["mean_similarity"] == 1.0
    assert result["longest_unmatched_voiced_run_seconds"] == 0.0


def test_asr_plan_requires_same_stable_parameters(tmp_path: Path) -> None:
    left = _write_json(
        tmp_path / "left.json",
        {
            "provider": "sensevoice",
            "model": "iic/SenseVoiceSmall",
            "batch_size_s": 120,
            "input_path": "left.wav",
        },
    )
    right = _write_json(
        tmp_path / "right.json",
        {
            "provider": "sensevoice",
            "model": "iic/SenseVoiceSmall",
            "batch_size_s": 60,
            "input_path": "right.wav",
        },
    )

    result = compare_asr_plans(left, right)

    assert result["status"] == "failed"
    assert result["reasons"] == ["asr_execution_signature_mismatch"]
    assert "input_path" not in json.dumps(result)


def test_asr_plan_ignores_readiness_metadata_drift(tmp_path: Path) -> None:
    execution = {
        "provider": "sensevoice",
        "model": "iic/SenseVoiceSmall",
        "batch_size_s": 60,
    }
    candidate = _write_json(
        tmp_path / "candidate-plan.json",
        {**execution, "model_readiness": {"speaker": {"model": ""}}},
    )
    retained = _write_json(tmp_path / "retained-plan.json", execution)

    result = compare_asr_plans(candidate, retained)

    assert result["status"] == "passed"


def test_unique_visual_evidence_blocks_deletion(tmp_path: Path) -> None:
    candidate = _write_json(
        tmp_path / "candidate.json",
        [
            {
                "id": "slide-1",
                "start": 10,
                "end": 20,
                "visual_text": "第一章 客户需求",
            },
            {
                "id": "slide-2",
                "start": 20,
                "end": 30,
                "visual_text": "独有费率表",
            },
        ],
    )
    retained = _write_json(
        tmp_path / "retained.json",
        [
            {
                "id": "slide-a",
                "start": 10,
                "end": 20,
                "visual_text": "第一章 客户需求",
            }
        ],
    )

    result = compare_visual_evidence([candidate], [retained], coverage_threshold=1.0)

    assert result["status"] == "failed"
    assert result["candidate_visual_coverage"] == 0.5
    assert result["unmatched_candidate_records"][0]["candidate_id"] == "slide-2"
    assert "独有费率表" not in json.dumps(result, ensure_ascii=False)


def test_scene_index_match_requires_close_time_range(tmp_path: Path) -> None:
    candidate = _write_json(
        tmp_path / "candidate-scenes.json",
        {
            "scenes": [
                {"index": 1, "start": 0.0, "end": 10.0, "visual_text": "场景一"},
                {"index": 2, "start": 10.0, "end": 20.0, "visual_text": "场景二"},
            ]
        },
    )
    retained = _write_json(
        tmp_path / "retained-scenes.json",
        {
            "scenes": [
                {"index": 1, "start": 0.0, "end": 10.0, "visual_text": "场景一"},
                {"index": 2, "start": 40.0, "end": 50.0, "visual_text": "场景二"},
            ]
        },
    )

    result = compare_visual_evidence([candidate], [retained], coverage_threshold=1.0)

    assert result["status"] == "failed"
    assert result["candidate_visual_coverage"] == 0.5
    assert result["unmatched_candidate_records"][0]["candidate_id"].endswith(":000002")


def test_scene_only_evidence_is_not_a_complete_visual_gate(tmp_path: Path) -> None:
    candidate = _write_json(
        tmp_path / "candidate-scene-only.json",
        {"scenes": [{"index": 1, "start": 0.0, "end": 10.0}]},
    )
    retained = _write_json(
        tmp_path / "retained-scene-only.json",
        {"scenes": [{"index": 1, "start": 0.0, "end": 10.0}]},
    )

    result = compare_visual_evidence([candidate], [retained], coverage_threshold=1.0)

    assert result["status"] == "unavailable"
    assert result["reasons"] == [
        "candidate_visual_content_evidence_missing",
        "retained_visual_content_evidence_missing",
    ]


def test_existing_provenance_reference_requires_rebind(tmp_path: Path) -> None:
    candidate = tmp_path / "low.mp4"
    retained = tmp_path / "high.mp4"
    candidate.write_bytes(b"candidate")
    retained.write_bytes(b"retained")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"source": str(candidate)}, ensure_ascii=False),
        encoding="utf-8",
    )

    blocked = audit_provenance_references(
        candidate,
        retained,
        reference_roots=[bundle],
        rebind_report=None,
    )

    assert blocked["status"] == "failed"
    assert blocked["candidate_reference_count"] == 1

    rebind = _write_json(
        tmp_path / "rebind.json",
        {
            "status": "completed",
            "candidate_sha256": __import__("hashlib").sha256(b"candidate").hexdigest(),
            "retained_sha256": __import__("hashlib").sha256(b"retained").hexdigest(),
            "all_references_updated": True,
        },
    )
    passed = audit_provenance_references(
        candidate,
        retained,
        reference_roots=[tmp_path / "empty"],
        rebind_report=rebind,
    )
    assert passed["status"] == "passed"


def test_retained_quality_rejects_lower_resolution() -> None:
    candidate = {
        "duration_seconds": 60.0,
        "video": {"width": 1920, "height": 1080},
        "audio": {"sample_rate": 48000, "channels": 2},
    }
    retained = {
        "duration_seconds": 60.0,
        "video": {"width": 1280, "height": 720},
        "audio": {"sample_rate": 48000, "channels": 2},
    }

    result = compare_retained_quality(candidate, retained)

    assert result["status"] == "failed"
    assert "pixel_count_not_lower" in result["reasons"]


def test_archival_quality_rejects_frame_rate_bitrate_tradeoff() -> None:
    candidate = {
        "duration_seconds": 60.0,
        "container_bit_rate": 4_600_000,
        "video": {"width": 2736, "height": 1824, "fps": 5.0},
        "audio": {"sample_rate": 48000, "channels": 2},
    }
    retained = {
        "duration_seconds": 60.0,
        "container_bit_rate": 1_700_000,
        "video": {"width": 2736, "height": 1824, "fps": 10.0},
        "audio": {"sample_rate": 48000, "channels": 2},
    }

    result = compare_retained_quality(
        candidate,
        retained,
        policy=QUALITY_POLICY_ARCHIVAL,
    )

    assert result["status"] == "failed"
    assert "container_bit_rate_not_lower" in result["reasons"]


def test_practical_course_quality_keeps_higher_resolution_over_60fps() -> None:
    candidate = {
        "duration_seconds": 60.0,
        "container_bit_rate": 556_000,
        "video": {"width": 1280, "height": 720, "fps": 60.0},
        "audio": {"sample_rate": 44100, "bit_rate": 129_000, "channels": 2},
    }
    retained = {
        "duration_seconds": 60.0,
        "container_bit_rate": 928_000,
        "video": {"width": 1920, "height": 1080, "fps": 23.3},
        "audio": {"sample_rate": 48000, "bit_rate": 127_000, "channels": 2},
    }

    result = compare_retained_quality(
        candidate,
        retained,
        policy=QUALITY_POLICY_PRACTICAL,
    )

    assert result["status"] == "passed"
    assert "frame_rate_lower" in result["non_blocking_tradeoffs"]
    assert "audio_bit_rate_lower" in result["non_blocking_tradeoffs"]


def test_practical_course_quality_allows_44100_hz_higher_bitrate_speech() -> None:
    candidate = {
        "duration_seconds": 60.0,
        "video": {"width": 640, "height": 360, "fps": 14.79},
        "audio": {"sample_rate": 48000, "bit_rate": 45_375, "channels": 2},
    }
    retained = {
        "duration_seconds": 60.0,
        "video": {"width": 1280, "height": 720, "fps": 14.79},
        "audio": {"sample_rate": 44100, "bit_rate": 134_367, "channels": 2},
    }

    result = compare_retained_quality(candidate, retained)

    assert result["status"] == "passed"
    assert result["policy"] == DEFAULT_QUALITY_POLICY
    assert result["non_blocking_tradeoffs"] == ["audio_sample_rate_lower"]


def test_practical_course_quality_requires_review_for_combined_material_loss() -> None:
    candidate = {
        "duration_seconds": 60.0,
        "video": {"width": 1280, "height": 720, "fps": 10.0},
        "audio": {"sample_rate": 48000, "bit_rate": 317_375, "channels": 2},
    }
    retained = {
        "duration_seconds": 60.0,
        "video": {"width": 1280, "height": 720, "fps": 5.0},
        "audio": {"sample_rate": 44100, "bit_rate": 130_184, "channels": 2},
    }

    result = compare_retained_quality(candidate, retained)

    assert result["status"] == "review_required"
    assert result["reasons"] == ["combined_motion_and_audio_quality_tradeoff"]
    assert set(result["human_review_tradeoffs"]) == {
        "frame_rate_lower",
        "audio_sample_rate_lower",
        "audio_bit_rate_lower",
    }


def test_archival_quality_blocks_any_technical_downgrade() -> None:
    candidate = {
        "duration_seconds": 60.0,
        "container_bit_rate": 2_000_000,
        "video": {"width": 1280, "height": 720, "fps": 60.0},
        "audio": {"sample_rate": 48000, "bit_rate": 192_000, "channels": 2},
    }
    retained = {
        "duration_seconds": 60.0,
        "container_bit_rate": 3_000_000,
        "video": {"width": 1920, "height": 1080, "fps": 30.0},
        "audio": {"sample_rate": 44100, "bit_rate": 256_000, "channels": 2},
    }

    result = compare_retained_quality(
        candidate,
        retained,
        policy=QUALITY_POLICY_ARCHIVAL,
    )

    assert result["status"] == "failed"
    assert result["strict_technical_quality_gate"] is True
    assert "frame_rate_not_lower" in result["reasons"]
    assert "audio_sample_rate_not_lower" in result["reasons"]


def test_archival_quality_blocks_codec_and_stream_bitrate_changes() -> None:
    candidate = {
        "duration_seconds": 60.0,
        "container_bit_rate": 1_500_000,
        "video": {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 25.0,
            "bit_rate": 1_200_000,
        },
        "audio": {
            "codec": "aac",
            "sample_rate": 48000,
            "bit_rate": 192_000,
            "channels": 2,
        },
    }
    retained = {
        "duration_seconds": 60.0,
        "container_bit_rate": 1_600_000,
        "video": {
            "codec": "hevc",
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
            "bit_rate": 900_000,
        },
        "audio": {
            "codec": "opus",
            "sample_rate": 48000,
            "bit_rate": 128_000,
            "channels": 2,
        },
    }

    result = compare_retained_quality(
        candidate,
        retained,
        policy=QUALITY_POLICY_ARCHIVAL,
    )

    assert result["status"] == "failed"
    assert set(result["reasons"]) >= {
        "video_codec_unchanged",
        "video_bit_rate_not_lower",
        "audio_codec_unchanged",
        "audio_bit_rate_not_lower",
    }


def test_audit_applies_default_and_archival_policy_without_weakening_content_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_path = tmp_path / "candidate.mp4"
    retained_path = tmp_path / "retained.mp4"
    candidate_path.write_bytes(b"candidate")
    retained_path.write_bytes(b"retained")
    candidate_probe = {
        "duration_seconds": 60.0,
        "container_bit_rate": 556_000,
        "video": {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 60.0,
            "bit_rate": 413_000,
        },
        "audio": {
            "codec": "aac",
            "sample_rate": 44100,
            "bit_rate": 129_000,
            "channels": 2,
        },
    }
    retained_probe = {
        "duration_seconds": 60.0,
        "container_bit_rate": 928_000,
        "video": {
            "codec": "h264",
            "width": 1920,
            "height": 1080,
            "fps": 23.3,
            "bit_rate": 794_000,
        },
        "audio": {
            "codec": "aac",
            "sample_rate": 48000,
            "bit_rate": 127_000,
            "channels": 2,
        },
    }

    monkeypatch.setattr(
        audit_module,
        "probe_media",
        lambda path: (
            candidate_probe
            if Path(path).name == candidate_path.name
            else retained_probe
        ),
    )
    passed_gate = {"status": "passed", "reasons": []}
    monkeypatch.setattr(
        audit_module,
        "compare_audio_content",
        lambda *args, **kwargs: passed_gate,
    )
    monkeypatch.setattr(
        audit_module,
        "compare_transcript_content",
        lambda *args, **kwargs: passed_gate,
    )
    monkeypatch.setattr(
        audit_module,
        "compare_visual_evidence",
        lambda *args, **kwargs: passed_gate,
    )
    monkeypatch.setattr(
        audit_module,
        "audit_provenance_references",
        lambda *args, **kwargs: passed_gate,
    )

    practical = audit_module.audit_media_equivalence(
        candidate_path,
        retained_path,
    )
    archival = audit_module.audit_media_equivalence(
        candidate_path,
        retained_path,
        quality_policy=QUALITY_POLICY_ARCHIVAL,
    )

    assert practical["status"] == "safe_to_delete_candidate"
    assert practical["safe_to_delete"] is True
    assert practical["quality_policy"] == QUALITY_POLICY_PRACTICAL
    assert archival["status"] == "quality_tradeoff_review"
    assert archival["safe_to_delete"] is False
    assert archival["decision_category"] == "content_equivalent_quality_tradeoff"
    assert archival["automatic_delete"] is False


def test_cli_defaults_to_practical_course_policy() -> None:
    args = _parser().parse_args(
        [
            "candidate.mp4",
            "retained.mp4",
            "--output-json",
            "report.json",
        ]
    )

    assert args.policy == QUALITY_POLICY_PRACTICAL


def test_markdown_makes_non_destructive_boundary_explicit() -> None:
    report = {
        "status": "blocked",
        "quality_policy": QUALITY_POLICY_PRACTICAL,
        "decision_category": "evidence_incomplete",
        "safe_to_delete": False,
        "candidate": {"path": "candidate.mp4"},
        "retained": {"path": "retained.mp4"},
        "gates": {
            "audio_content_containment": {
                "status": "unavailable",
                "reasons": ["missing"],
            }
        },
        "required_actions": ["补齐证据。"],
    }

    rendered = render_media_equivalence_markdown(report)

    assert "自动删除：`false`" in rendered
    assert "判定策略：`practical_course`" in rendered
    assert "本报告不执行删除" in rendered
