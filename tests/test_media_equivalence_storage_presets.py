from __future__ import annotations

import pytest

from video_knowledge_pipeline.media_equivalence_audit import (
    QUALITY_POLICY_ARCHIVAL,
    QUALITY_POLICY_PRACTICAL,
    _parser,
    normalize_quality_policy,
)


@pytest.mark.parametrize(
    ("requested", "canonical"),
    [
        ("space_saving", QUALITY_POLICY_PRACTICAL),
        ("archive_lossless", QUALITY_POLICY_ARCHIVAL),
        ("practical_course", QUALITY_POLICY_PRACTICAL),
        ("archival_lossless", QUALITY_POLICY_ARCHIVAL),
    ],
)
def test_storage_preset_aliases_keep_legacy_policy_compatibility(
    requested: str, canonical: str
) -> None:
    assert normalize_quality_policy(requested) == canonical


def test_cli_default_keeps_stable_internal_contract() -> None:
    parsed = _parser().parse_args(
        ["candidate.mp4", "retained.mp4", "--output-json", "audit.json"]
    )

    assert parsed.policy == QUALITY_POLICY_PRACTICAL


def test_cli_accepts_archive_lossless_without_renaming_internal_contract() -> None:
    parsed = _parser().parse_args(
        [
            "candidate.mp4",
            "retained.mp4",
            "--output-json",
            "audit.json",
            "--policy",
            "archive_lossless",
        ]
    )

    assert normalize_quality_policy(parsed.policy) == QUALITY_POLICY_ARCHIVAL


def test_unknown_storage_preset_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported quality policy"):
        normalize_quality_policy("delete_everything")
