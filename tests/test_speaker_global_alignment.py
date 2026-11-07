from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from video_knowledge_pipeline import funasr_python_runner
from video_knowledge_pipeline.campplus_speaker_center_sidecar import (
    CANDIDATE_SCHEMA,
    build_campplus_speaker_center_sidecar,
)
from video_knowledge_pipeline.speaker_global_alignment import (
    PRIVATE_ALIGNMENT_SCHEMA,
    align_chunk_speaker_records,
    bind_local_voiceprint_role,
    delete_local_voiceprint,
    enroll_local_voiceprints,
    match_local_voiceprints,
    write_alignment_artifacts,
)
from video_knowledge_pipeline.transcript_speakers import cue_speaker
from video_knowledge_pipeline.speaker_shared_session_alignment import (
    build_shared_session_speaker_alignment,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "chunk_index": 0,
            "record_index": 0,
            "_speaker_embedding_centers": [
                {"local_speaker_id": "0", "center": [1.0, 0.0]},
                {"local_speaker_id": "1", "center": [0.0, 1.0]},
            ],
            "sentence_info": [
                {"start": 0, "end": 1000, "text": "甲", "spk": 0},
                {"start": 2000, "end": 3000, "text": "乙", "spk": 1},
            ],
        },
        {
            "chunk_index": 1,
            "record_index": 0,
            "_speaker_embedding_centers": [
                {"local_speaker_id": "0", "center": [0.01, 0.99]},
                {"local_speaker_id": "1", "center": [0.99, 0.01]},
            ],
            "sentence_info": [
                {"start": 4000, "end": 5000, "text": "乙", "spk": 0},
                {"start": 6000, "end": 7000, "text": "甲", "spk": 1},
            ],
        },
    ]


def test_chunk_local_labels_are_mapped_to_stable_anonymous_ids() -> None:
    mapped, public, private = align_chunk_speaker_records(_records())

    assert public["status"] == "candidate"
    assert public["chunk_local_speaker_count"] == 4
    assert public["global_speaker_count"] == 2
    assert public["mapped_sentence_count"] == 4
    assert mapped[0]["sentence_info"][0]["speaker_global_id"] == "speaker-global-001"
    assert mapped[1]["sentence_info"][1]["speaker_global_id"] == "speaker-global-001"
    assert mapped[0]["sentence_info"][1]["speaker_global_id"] == "speaker-global-002"
    assert mapped[1]["sentence_info"][0]["speaker_global_id"] == "speaker-global-002"
    assert mapped[0]["sentence_info"][0]["spk"] == 0
    assert private["biometric_data"] is True
    assert len(private["global_centers"]) == 2
    assert "global_centers" not in public


def test_two_active_clusters_cannot_collapse_to_one_global_id() -> None:
    records = _records()[:1]
    records.append(
        {
            "chunk_index": 1,
            "record_index": 0,
            "_speaker_embedding_centers": [
                {"local_speaker_id": "0", "center": [1.0, 0.01]},
                {"local_speaker_id": "1", "center": [0.99, 0.02]},
            ],
            "sentence_info": [
                {"start": 4000, "end": 5000, "text": "甲", "spk": 0},
                {"start": 6000, "end": 7000, "text": "丙", "spk": 1},
            ],
        }
    )

    mapped, public, _private = align_chunk_speaker_records(records)
    global_ids = {
        row["speaker_global_id"] for row in mapped[1]["sentence_info"]
    }

    assert len(global_ids) == 2
    assert public["global_speaker_count"] >= 2


def test_operator_known_count_reuses_funasr_oracle_and_repairs_within_chunk_split() -> None:
    records = [
        {
            "chunk_index": 0,
            "record_index": 0,
            "_speaker_embedding_centers": [
                {"local_speaker_id": str(index), "center": center}
                for index, center in enumerate(
                    ([1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [-1.0, 0.0], [0.01, 0.99])
                )
            ],
            "sentence_info": [
                {"start": index * 1000, "end": (index + 1) * 1000, "text": str(index), "spk": index}
                for index in range(5)
            ],
        }
    ]

    mapped, public, private = align_chunk_speaker_records(
        records,
        expected_speaker_count=3,
        oracle_clusterer=lambda _centers, expected: [7, 4, 7, 9, 4]
        if expected == 3
        else [],
    )

    assert public["expected_speaker_count"] == 3
    assert public["global_speaker_count"] == 3
    assert [row["speaker_global_id"] for row in mapped[0]["sentence_info"]] == [
        "speaker-global-001",
        "speaker-global-002",
        "speaker-global-001",
        "speaker-global-003",
        "speaker-global-002",
    ]
    assert len(private["global_centers"]) == 3
    assert {row["method"] for row in public["mappings"]} == {
        "funasr_spectral_oracle_count"
    }


def test_missing_centers_fail_closed_without_relabeling() -> None:
    records = [
        {
            "chunk_index": 0,
            "record_index": 0,
            "sentence_info": [
                {"start": 0, "end": 1000, "text": "甲", "spk": 0}
            ],
        }
    ]

    mapped, public, private = align_chunk_speaker_records(records)

    assert public["status"] == "unavailable"
    assert public["missing_center_chunk_indexes"] == [0]
    assert "speaker_global_id" not in mapped[0]["sentence_info"][0]
    assert private["global_centers"] == []


def test_public_artifact_excludes_embeddings_and_double_run_is_stable(
    tmp_path: Path,
) -> None:
    mapped_a, public_a, private_a = align_chunk_speaker_records(_records())
    mapped_b, public_b, private_b = align_chunk_speaker_records(_records())
    public_a.pop("updated_at")
    public_b.pop("updated_at")
    private_a.pop("updated_at")
    private_b.pop("updated_at")

    assert mapped_a == mapped_b
    assert public_a == public_b
    assert private_a == private_b

    result = write_alignment_artifacts(
        tmp_path / "raw-asr-output.json", public_a, private_a, write=True
    )
    public_path = Path(result["artifacts"]["public_path"])
    private_path = Path(result["artifacts"]["private_path"])
    public_text = public_path.read_text(encoding="utf-8")
    private_payload = json.loads(private_path.read_text(encoding="utf-8"))

    assert '"center"' not in public_text
    assert private_payload["schema"] == PRIVATE_ALIGNMENT_SCHEMA
    assert private_payload["must_remain_local"] is True


def test_voiceprint_registry_is_explicit_local_candidate_only_and_deletable(
    tmp_path: Path,
) -> None:
    _mapped, public, private = align_chunk_speaker_records(_records())
    artifacts = write_alignment_artifacts(
        tmp_path / "first.json", public, private, write=True
    )
    private_path = artifacts["artifacts"]["private_path"]
    registry = tmp_path / "voiceprints.private.json"

    blocked = enroll_local_voiceprints(
        private_path,
        registry,
        source_id="video-a",
        confirm_local_biometric_storage=False,
    )
    assert blocked["status"] == "blocked"
    assert not registry.exists()

    first = enroll_local_voiceprints(
        private_path,
        registry,
        source_id="video-a",
        confirm_local_biometric_storage=True,
    )
    repeated = enroll_local_voiceprints(
        private_path,
        registry,
        source_id="video-a",
        confirm_local_biometric_storage=True,
    )
    self_match = match_local_voiceprints(private_path, registry)
    second_private = dict(private)
    second_private["source_revision"] = "second-video-revision"
    second_private_path = tmp_path / "second-speaker-global-alignment.private.json"
    second_private_path.write_text(
        json.dumps(second_private, ensure_ascii=False), encoding="utf-8"
    )
    matched = match_local_voiceprints(second_private_path, registry)

    assert first["added_count"] == 2
    assert repeated["added_count"] == 0
    assert all(row["status"] == "no_match" for row in self_match["matches"])
    assert all(row["status"] == "suspected_same_speaker" for row in matched["matches"])
    assert all(row["identity_confirmed"] is False for row in matched["matches"])
    assert "center" not in json.dumps(matched)

    voiceprint_id = json.loads(registry.read_text(encoding="utf-8"))["entries"][0][
        "voiceprint_id"
    ]
    blocked_role = bind_local_voiceprint_role(
        registry,
        voiceprint_id,
        "采访者",
        confirm_role_binding=False,
    )
    bound_role = bind_local_voiceprint_role(
        registry,
        voiceprint_id,
        "采访者",
        confirm_role_binding=True,
    )
    assert blocked_role["status"] == "blocked"
    assert bound_role["status"] == "role_bound"
    assert json.loads(registry.read_text(encoding="utf-8"))["entries"][0][
        "role_status"
    ] == "human_confirmed"
    blocked_delete = delete_local_voiceprint(
        registry, voiceprint_id, confirm_delete=False
    )
    deleted = delete_local_voiceprint(registry, voiceprint_id, confirm_delete=True)
    assert blocked_delete["status"] == "blocked"
    assert deleted["status"] == "deleted"
    assert deleted["deleted_count"] == 1


def test_downstream_reader_prefers_global_anonymous_id() -> None:
    cue = {
        "spk": 4,
        "speaker_local_cluster": "4",
        "speaker_global_id": "speaker-global-002",
    }

    assert cue_speaker(cue) == "speaker-global-002"


def test_funasr_runner_requests_and_serializes_only_speaker_centers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "dialogue.wav"
    output = tmp_path / "raw.json"
    media.write_bytes(b"audio")
    captured: dict[str, object] = {}

    class ArrayLike:
        def tolist(self) -> list[list[float]]:
            return [[1.0, 0.0], [0.0, 1.0]]

    class FakeAutoModel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            captured.update(kwargs)
            return [
                {
                    "text": "甲乙",
                    "spk_embedding_center": ArrayLike(),
                    "sentence_info": [
                        {"start": 0, "end": 500, "text": "甲", "spk": 0},
                        {"start": 500, "end": 1000, "text": "乙", "spk": 1},
                    ],
                }
            ]

    monkeypatch.setitem(
        sys.modules,
        "funasr",
        type("FakeFunASR", (), {"AutoModel": FakeAutoModel}),
    )
    monkeypatch.setattr(funasr_python_runner, "_select_device", lambda _value: "cpu")

    result = funasr_python_runner.run_funasr(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="iic/SenseVoiceSmall",
        spk_model="cam++",
        device="cpu",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert captured["return_spk_center"] is True
    assert "spk_embedding_center" not in json.dumps(payload)
    assert payload["result"][0]["_speaker_embedding_centers"] == [
        {"local_speaker_id": "0", "center": [1.0, 0.0]},
        {"local_speaker_id": "1", "center": [0.0, 1.0]},
    ]
    assert payload["speaker_embedding_evidence"]["biometric_data"] is True


def test_campplus_center_sidecar_is_candidate_only_and_does_not_rerun_asr(
    tmp_path: Path,
) -> None:
    media = tmp_path / "interview.mp4"
    source = tmp_path / "raw-asr-output.json"
    output = tmp_path / "speaker-sidecar"
    media.write_bytes(b"synthetic-media")
    original = {
        "schema": "video_knowledge_pipeline.funasr_chunked_raw_output.v1",
        "input": str(media),
        "duration_seconds": 20.0,
        "provider": "sensevoice",
        "chunk_results": [
            {
                "chunk_index": 0,
                "record_index": 0,
                "sentence_info": [
                    {
                        "start": index * 3000,
                        "end": index * 3000 + 2000,
                        "text": f"segment-{index}",
                        "spk": index,
                    }
                    for index in range(5)
                ],
            }
        ],
    }
    source.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")

    def fake_clip(
        _media: Path, destination: Path, windows: list[dict[str, object]]
    ) -> dict[str, object]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"clip-{windows[0]['start']}".encode())
        return {"tool": "fake-ffmpeg", "window_count": len(windows)}

    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.99, 0.01],
        [-1.0, 0.0],
        [0.01, 0.99],
    ]
    result = build_campplus_speaker_center_sidecar(
        source,
        media,
        output,
        expected_speaker_count=3,
        execute=True,
        clip_builder=fake_clip,
        embedding_extractor=lambda clips, device: (
            vectors,
            {"provider": "fake-campplus", "device": device, "clip_count": len(clips)},
        ),
        oracle_clusterer=lambda _centers, _expected: [0, 1, 0, 2, 1],
    )
    candidate = json.loads(
        Path(result["artifacts"]["candidate_transcript"]).read_text(encoding="utf-8")
    )
    public = Path(result["artifacts"]["public_sidecar"]).read_text(encoding="utf-8")

    assert result["status"] == "needs_human_review"
    assert result["operator_boundary"]["asr_reexecuted"] is False
    assert result["alignment"]["global_speaker_count"] == 3
    assert candidate["schema"] == CANDIDATE_SCHEMA
    assert candidate["candidate_only"] is True
    assert len(
        {
            row["speaker_global_id"]
            for row in candidate["chunk_results"][0]["sentence_info"]
        }
    ) == 3
    assert "\"center\"" not in public
    assert json.loads(source.read_text(encoding="utf-8")) == original


def test_shared_session_alignment_requires_confirmation_and_keeps_roles_unconfirmed(
    tmp_path: Path,
) -> None:
    centers: list[Path] = []
    candidates: list[Path] = []
    for recording_index in range(2):
        center_path = tmp_path / f"centers-{recording_index}.private.json"
        candidate_path = tmp_path / f"candidate-{recording_index}.json"
        center_path.write_text(
            json.dumps(
                {
                    "biometric_data": True,
                    "centers": [
                        {
                            "chunk_index": 0,
                            "local_speaker_id": str(index),
                            "center": [1.0, float(index + recording_index + 1)],
                        }
                        for index in range(3)
                    ],
                }
            ),
            encoding="utf-8",
        )
        candidate_path.write_text(
            json.dumps(
                {
                    "schema": "candidate",
                    "chunk_results": [
                        {
                            "chunk_index": 0,
                            "sentence_info": [
                                {
                                    "start": index * 1000,
                                    "end": index * 1000 + 900,
                                    "text": f"recording-{recording_index}-speaker-{index}",
                                    "spk": index,
                                    "speaker_local_cluster": str(index),
                                    "speaker_global_id": f"recording-local-{index}",
                                }
                                for index in range(3)
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        centers.append(center_path)
        candidates.append(candidate_path)

    blocked = build_shared_session_speaker_alignment(
        centers,
        candidates,
        tmp_path / "blocked",
        expected_speaker_count=3,
        confirm_shared_participant_set=False,
    )
    result = build_shared_session_speaker_alignment(
        centers,
        candidates,
        tmp_path / "shared",
        expected_speaker_count=3,
        confirm_shared_participant_set=True,
        oracle_clusterer=lambda _centers, _expected: [9, 5, 7, 9, 5, 7],
    )
    first = json.loads(
        Path(result["candidate_transcripts"][0]["path"]).read_text(encoding="utf-8")
    )
    second = json.loads(
        Path(result["candidate_transcripts"][1]["path"]).read_text(encoding="utf-8")
    )
    review = json.loads(Path(result["role_review_path"]).read_text(encoding="utf-8"))

    assert blocked["status"] == "blocked"
    assert result["global_speaker_count"] == 3
    assert [
        row["speaker_global_id"] for row in first["chunk_results"][0]["sentence_info"]
    ] == [
        row["speaker_global_id"] for row in second["chunk_results"][0]["sentence_info"]
    ]
    assert all(
        row["recording_local_speaker_global_id"].startswith("recording-local-")
        for row in first["chunk_results"][0]["sentence_info"]
    )
    assert {row["role_status"] for row in review["assignments"]} == {"unconfirmed"}
    assert '"center"' not in json.dumps(result)


def test_shared_session_alignment_uses_per_segment_samples_and_marks_impure_local_cluster(
    tmp_path: Path,
) -> None:
    centers = tmp_path / "centers.private.json"
    candidate = tmp_path / "candidate.json"
    centers.write_text(
        json.dumps(
            {
                "biometric_data": True,
                "centers": [
                    {"chunk_index": 0, "local_speaker_id": "0", "center": [1.0, 0.0]}
                ],
                "samples": [
                    {
                        "chunk_index": 0,
                        "local_speaker_id": "0",
                        "source_segment_id": "chunk-0000-sentence-00000",
                        "duration_seconds": 4.0,
                        "center": [1.0, 0.0],
                    },
                    {
                        "chunk_index": 0,
                        "local_speaker_id": "0",
                        "source_segment_id": "chunk-0000-sentence-00001",
                        "duration_seconds": 2.0,
                        "center": [0.0, 1.0],
                    },
                    {
                        "chunk_index": 0,
                        "local_speaker_id": "1",
                        "source_segment_id": "chunk-0000-sentence-00002",
                        "duration_seconds": 5.0,
                        "center": [-1.0, 0.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "chunk_results": [
                    {
                        "chunk_index": 0,
                        "sentence_info": [
                            {"start": 0, "end": 4000, "text": "甲", "spk": 0},
                            {"start": 4000, "end": 6000, "text": "乙", "spk": 0},
                            {"start": 6000, "end": 11000, "text": "丙", "spk": 1},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_shared_session_speaker_alignment(
        [centers],
        [candidate],
        tmp_path / "shared-samples",
        expected_speaker_count=3,
        confirm_shared_participant_set=True,
        oracle_clusterer=lambda _vectors, _expected: [3, 7, 9],
    )
    derived = json.loads(
        Path(result["candidate_transcripts"][0]["path"]).read_text(encoding="utf-8")
    )
    sentences = derived["chunk_results"][0]["sentence_info"]

    assert result["sample_embedding_count"] == 3
    assert [row["speaker_global_id"] for row in sentences] == [
        "speaker-global-001",
        "speaker-global-002",
        "speaker-global-003",
    ]
    assert all(
        row["speaker_global_assignment"]["method"]
        == "exact_source_segment_campplus_sample"
        for row in sentences
    )
    assert result["mappings"][0]["status"] == "needs_human_review"
