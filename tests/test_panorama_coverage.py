from __future__ import annotations

from video_knowledge_pipeline.knowledge_coverage import build_knowledge_coverage
from video_knowledge_pipeline.visual_evidence import evidence_level_counts, panorama_evidence_state


def _complete_panorama() -> dict[str, object]:
    directions = ["front", "right", "back", "left", "up", "down"]
    return {
        "schema": "video_knowledge_pipeline.panorama_visual_understanding.v1",
        "sessions": [
            {
                "sample_index": 1,
                "timestamp_seconds": 5.0,
                "evidence_level": "confirmed",
                "viewport_observations": [
                    {
                        "viewport_id": f"vp-{direction}",
                        "direction": direction,
                        "observed": True,
                        "observation": direction,
                    }
                    for direction in directions
                ],
            }
        ],
    }


def test_panorama_coverage_is_orthogonal_to_existing_visual_counts() -> None:
    manifest = {
        "panorama_profile": {"schema": "video_knowledge_pipeline.panorama_profile.v1", "ready": True},
        "panorama_analysis_plan": {"selected_indexes": [1, 2], "temporal_required_indexes": []},
    }
    timeline = [
        {"index": 1, "panorama_visual_understanding": _complete_panorama()},
        {"index": 2},
    ]

    report = build_knowledge_coverage(manifest, timeline)

    panorama = report["panorama_coverage"]
    channel = next(row for row in report["channels"] if row["key"] == "panorama_spherical_evidence")
    assert panorama["required"] is True
    assert panorama["geometric_complete_segments"] == 1
    assert panorama["semantic_covered_segments"] == 1
    assert panorama["missing_segment_indexes"] == [2]
    assert channel["status"] == "weak"
    assert report["visual_evidence_levels"] == evidence_level_counts(timeline)["visual"]
    assert panorama_evidence_state(timeline[0])["consumable"] is True
    assert panorama_evidence_state(timeline[1])["evidence_level"] == "not_seen"


def test_historical_non_panorama_bundle_remains_compatible_and_not_required() -> None:
    timeline = [{"index": 1, "transcript": "legacy", "visual_route": "unknown"}]

    report = build_knowledge_coverage({}, timeline)

    panorama = report["panorama_coverage"]
    channel = next(row for row in report["channels"] if row["key"] == "panorama_spherical_evidence")
    assert panorama["required"] is False
    assert channel["expected_count"] == 0
    assert channel["status"] == "not_required"
    assert report["status"] != "blocked" or all(row["key"] != "panorama_spherical_evidence" for row in report["blockers"])
