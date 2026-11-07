from __future__ import annotations

from video_knowledge_pipeline.cli import build_parser


def test_pullfilm_v2_cli_defaults_are_strict_local_and_explicit() -> None:
    parser = build_parser()

    detection = parser.parse_args(["technical-shot-detection", "bundle"])
    explicit_strict = parser.parse_args(
        ["technical-shot-detection", "bundle", "--strict"]
    )
    language = parser.parse_args(["shot-language-analysis", "bundle"])
    structure = parser.parse_args(
        ["video-structure", "bundle", "--content-profile", "filmed-v1"]
    )

    assert detection.backend == "autoshot"
    assert detection.allow_fallback is False
    assert explicit_strict.strict is True
    assert language.execution_location == "local"
    assert language.execute is False
    assert structure.content_profile == "filmed-v1"


def test_pullfilm_v2_review_and_fusion_cli_are_separate_actions() -> None:
    parser = build_parser()

    fusion = parser.parse_args(
        [
            "technical-shot-fusion",
            "bundle",
            "autoshot.json",
            "omnishotcut.json",
            "--frame-rate",
            "25",
        ]
    )
    review = parser.parse_args(
        ["shot-review-apply", "bundle", "shot-review-notes.json"]
    )

    assert fusion.tolerance_frames == 2
    assert fusion.candidate_paths == ["autoshot.json", "omnishotcut.json"]
    assert review.command == "shot-review-apply"
