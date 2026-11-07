from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.shot_language_analysis import (
    SCHEMA,
    run_shot_language_analysis,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    frame_a = root / "frames" / "frame_01_0000000000ms.jpg"
    frame_b = root / "frames" / "frame_02_0000001000ms.jpg"
    frame_a.parent.mkdir(parents=True)
    frame_a.write_bytes(b"a")
    frame_b.write_bytes(b"b")
    _write_json(
        root / "manifest.json",
        {"schema": "lecture_webui_bundle.v1", "title": "fixture"},
    )
    _write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 2.0,
                "corrected_transcript": "spoken evidence",
                "visual_text": "screen evidence",
                "frame_paths": [str(frame_a), str(frame_b)],
                "temporal_visual_understanding": {
                    "event_sequence": ["subject raises a hand"]
                },
                "visual_understanding": {
                    "composition": {"rule_of_thirds": True},
                    "lighting": "soft",
                },
            }
        ],
    )
    _write_json(
        root / "exports" / "technical-shot-boundaries.json",
        {
            "schema": "video_knowledge_pipeline.technical_shot_boundaries.v1",
            "status": "completed",
            "ok": True,
            "backend": "autoshot",
            "boundary_kind": "technical_shot",
            "shots": [
                {
                    "shot_id": "technical-shot-0001",
                    "index": 1,
                    "start": 0.0,
                    "end": 2.0,
                }
            ],
        },
    )
    return root


def test_shot_facts_bind_each_non_unknown_field_to_evidence(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)

    result = run_shot_language_analysis(
        root,
        execute=True,
        write=False,
        route_id="local-qwen-vl",
        _shot_type_analyzer=lambda path: {
            "\u666f\u522b": "\u4e2d\u666f",
            "confidence": 0.91,
        },
        _movement_analyzer=lambda paths: {
            "\u955c\u5934\u8fd0\u52a8": "\u56fa\u5b9a\u955c\u5934",
            "confidence": 0.88,
        },
    )

    assert result["schema"] == SCHEMA
    assert result["status"] == "completed"
    fields = result["shots"][0]["fields"]
    assert fields["shot_type"]["value"] == "medium"
    assert fields["shot_type"]["status"] == "inferred"
    assert fields["camera_movement"]["value"] == "static"
    assert fields["dialogue_or_narration"]["status"] == "confirmed"
    assert fields["screen_text"]["status"] == "confirmed"
    for field in fields.values():
        if field["status"] != "unavailable":
            assert field["evidence_ids"]
    assert result["operator_boundary"]["no_remote_fallback"] is True
    assert result["operator_boundary"]["no_model_download"] is True


def test_low_confidence_stays_inferred_and_queues_explicit_local_vlm(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)

    result = run_shot_language_analysis(
        root,
        execute=True,
        write=False,
        route_id="local-qwen-vl",
        _shot_type_analyzer=lambda path: {
            "\u666f\u522b": "\u7279\u5199",
            "confidence": 0.2,
        },
        _movement_analyzer=lambda paths: {
            "\u955c\u5934\u8fd0\u52a8": "\u6447\u955c\u5934",
            "confidence": 0.3,
        },
    )

    fields = result["shots"][0]["fields"]
    assert fields["shot_type"]["status"] == "inferred"
    assert fields["shot_type"]["missing_evidence"]
    assert fields["camera_movement"]["status"] == "inferred"
    assert result["local_vlm_escalations"][0]["automatic_execution"] is False
    assert result["local_vlm_escalations"][0]["route_id"] == "local-qwen-vl"


def test_no_execute_never_claims_model_fields(tmp_path: Path) -> None:
    root = _bundle(tmp_path)

    result = run_shot_language_analysis(root, execute=False, write=False)

    fields = result["shots"][0]["fields"]
    assert fields["shot_type"]["status"] == "unavailable"
    assert fields["shot_type"]["value"] is None
    assert fields["camera_movement"]["status"] == "unavailable"
    assert result["runtime_errors"] == {}


def test_webui_asset_paths_are_reused_as_shot_frame_evidence(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    copied = root / "assets" / "0001-frame.jpg"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(b"copied-frame")
    timeline = json.loads((root / "timeline.json").read_text(encoding="utf-8"))
    timeline[0].pop("frame_paths", None)
    timeline[0]["assets"] = [
        {
            "path": "assets/0001-frame.jpg",
            "source": r"C:\outside-bundle\original-frame.jpg",
            "copied": "true",
        }
    ]
    _write_json(root / "timeline.json", timeline)

    result = run_shot_language_analysis(
        root,
        execute=True,
        write=False,
        _shot_type_analyzer=lambda path: {
            "shot_type": "medium",
            "confidence": 0.9,
        },
        _movement_analyzer=lambda paths: {
            "camera_movement": "static",
            "confidence": 0.9,
        },
    )

    shot = result["shots"][0]
    assert [row["source_path"] for row in shot["frame_manifest"]] == [
        str(copied.resolve())
    ]
    assert shot["fields"]["reference_frames"]["status"] == "confirmed"
    assert shot["fields"]["shot_type"]["evidence_ids"]
    assert all(
        "outside-bundle" not in row["source_path"]
        for row in shot["frame_manifest"]
    )


def test_execute_extracts_three_frames_from_hash_bound_technical_media(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _bundle(tmp_path)
    media = root / "media.mp4"
    media.write_bytes(b"hash-bound-media")
    boundary_path = root / "exports" / "technical-shot-boundaries.json"
    boundaries = json.loads(boundary_path.read_text(encoding="utf-8"))
    boundaries["media"] = {
        "status": "bound",
        "path": str(media),
        "sha256": sha256_file(media),
    }
    _write_json(boundary_path, boundaries)
    timeline = json.loads((root / "timeline.json").read_text(encoding="utf-8"))
    timeline[0].pop("frame_paths", None)
    _write_json(root / "timeline.json", timeline)

    def fake_extract(media_path, output_dir, segments) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for index, segment in enumerate(segments, start=1):
            frame = output / f"{index:03d}.jpg"
            frame.write_bytes(f"frame-{index}".encode())
            segment.frame_paths.append(str(frame))

    def fake_prepare(frames, **kwargs):
        manifest = [
            {
                "frame_id": f"F{index:02d}",
                "source_index": index,
                "filename": Path(path).name,
                "source_path": str(path),
                "timestamp_ms": None,
                "timestamp": "",
                "bytes": Path(path).stat().st_size,
            }
            for index, path in enumerate(frames, start=1)
        ]
        return {
            "frame_manifest": manifest,
            "frame_mapping": [
                {
                    "frame_id": row["frame_id"],
                    "source_index": row["source_index"],
                    "representative_frame_id": row["frame_id"],
                }
                for row in manifest
            ],
            "prepared_image_paths": list(frames),
            "contact_sheet_path": "",
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.temporal_frame_preprocess.prepare_temporal_image_probe",
        fake_prepare,
    )

    result = run_shot_language_analysis(
        root,
        execute=True,
        write=True,
        _frame_extractor=fake_extract,
        _shot_type_analyzer=lambda path: {
            "shot_type": "medium",
            "confidence": 0.9,
        },
        _movement_analyzer=lambda paths: {
            "camera_movement": "static",
            "confidence": 0.9,
        },
    )

    assert result["media_provenance"]["status"] == "verified"
    assert result["operator_boundary"]["technical_media_hash_verified"] is True
    assert len(result["shots"][0]["frame_manifest"]) == 3
    assert result["shots"][0]["fields"]["reference_frames"]["status"] == "confirmed"


def test_shot_facts_block_without_verified_technical_shots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1"})
    _write_json(root / "timeline.json", [{"index": 1, "start": 0, "end": 5}])

    result = run_shot_language_analysis(root, execute=False, write=False)

    assert result["status"] == "blocked_missing_technical_shots"
    assert result["shot_count"] == 0
    assert result["operator_boundary"]["no_chapter_or_timeline_range_fallback"]
