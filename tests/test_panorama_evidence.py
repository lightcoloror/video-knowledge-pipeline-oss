from __future__ import annotations

from video_knowledge_pipeline.panorama_evidence import (
    aggregate_panorama_evidence,
    panorama_claim_scope,
)


def _session(*, directions: list[str], level: str = "confirmed") -> dict[str, object]:
    geometry = {
        "front": (0.0, 0.0),
        "right": (90.0, 0.0),
        "back": (180.0, 0.0),
        "left": (-90.0, 0.0),
        "up": (0.0, 90.0),
        "down": (0.0, -90.0),
    }
    return {
        "sample_index": 1,
        "timestamp_seconds": 5.0,
        "evidence_level": level,
        "viewport_observations": [
            {
                "viewport_id": f"vp-{direction}",
                "direction": direction,
                "yaw": geometry[direction][0],
                "pitch": geometry[direction][1],
                "observed": True,
                "observation": f"visible in {direction}",
            }
            for direction in directions
        ],
    }


def test_full_six_face_session_has_geometric_and_semantic_coverage_but_not_segment_global_negation() -> None:
    item = {"panorama_visual_understanding": {"sessions": [_session(directions=["front", "right", "back", "left", "up", "down"])]}}

    evidence = aggregate_panorama_evidence(item)

    assert evidence["geometric"]["status"] == "complete"
    assert evidence["geometric"]["covered_directions"] == ["front", "right", "back", "left", "up", "down"]
    assert evidence["semantic"]["confirmed_sessions"] == 1
    assert evidence["temporal"]["status"] == "not_required"
    assert panorama_claim_scope(evidence, claim_type="negative", requested_scope="segment_global")["allowed"] is False
    assert panorama_claim_scope(evidence, claim_type="negative", requested_scope="observed_timestamp")["allowed"] is True


def test_partial_faces_and_missing_poles_are_review_gaps() -> None:
    item = {"panorama_visual_understanding": {"sessions": [_session(directions=["front"])]}}

    evidence = aggregate_panorama_evidence(item)

    assert evidence["geometric"]["status"] == "incomplete"
    assert set(evidence["geometric"]["missing_directions"]) == {"right", "back", "left", "up", "down"}
    assert {row["key"] for row in evidence["review_gaps"]} >= {"panorama_faces_missing", "panorama_poles_missing"}
    assert panorama_claim_scope(evidence, claim_type="negative", requested_scope="observed_timestamp")["allowed"] is False


def test_seam_adjacent_entity_stays_possible_duplicate_without_verified_tracker() -> None:
    session = _session(directions=["front", "right", "back", "left", "up", "down"])
    session["viewport_observations"][0]["entity_candidates"] = ["person-red"]  # type: ignore[index]
    session["viewport_observations"][3]["entity_candidates"] = ["person-red"]  # type: ignore[index]
    item = {"panorama_visual_understanding": {"sessions": [session]}}

    unverified = aggregate_panorama_evidence(item)
    verified = aggregate_panorama_evidence(item, verified_entity_pairs={("vp-front", "vp-left", "person-red")})

    assert unverified["cross_view_entities"][0]["status"] == "possible_duplicate"
    assert verified["cross_view_entities"][0]["status"] == "verified_same_entity"


def test_missing_temporal_direction_tracks_are_separate_from_six_face_coverage() -> None:
    item = {
        "panorama_visual_understanding": {"sessions": [_session(directions=["front", "right", "back", "left", "up", "down"])]},
        "panorama_temporal_understanding": {
            "required_directions": ["front", "right", "back"],
            "tracks": [{"direction": "front", "evidence_level": "confirmed", "observed_frame_count": 5}],
        },
    }

    evidence = aggregate_panorama_evidence(item)

    assert evidence["geometric"]["status"] == "complete"
    assert evidence["temporal"]["status"] == "incomplete"
    assert evidence["temporal"]["missing_directions"] == ["right", "back"]
    assert "panorama_temporal_directions_missing" in {row["key"] for row in evidence["review_gaps"]}
