from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import video_knowledge_pipeline.consented_model_batch as consented_model_batch
import video_knowledge_pipeline.model_business_authorization as business_authorization
import video_knowledge_pipeline.model_connector_consent as model_consent
import video_knowledge_pipeline.vision_export_consent as vision_consent
from video_knowledge_pipeline.models import now_iso
from video_knowledge_pipeline.time_utils import (
    parse_utc_datetime_or_none,
    utc_now_iso_seconds,
)


def test_utc_now_iso_seconds_preserves_aware_second_precision() -> None:
    parsed = datetime.fromisoformat(utc_now_iso_seconds())

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    assert parsed.microsecond == 0


def test_local_and_utc_timestamp_contracts_remain_distinct() -> None:
    assert datetime.fromisoformat(now_iso()).tzinfo is None
    assert datetime.fromisoformat(utc_now_iso_seconds()).tzinfo is not None


def test_batch_private_entrypoint_delegates_to_utc_owner() -> None:
    assert consented_model_batch._now_iso is utc_now_iso_seconds


def test_parse_utc_datetime_or_none_preserves_existing_consent_contract() -> None:
    zulu = parse_utc_datetime_or_none("2026-07-23T01:02:03Z")
    offset = parse_utc_datetime_or_none("2026-07-23T09:02:03+08:00")
    naive = parse_utc_datetime_or_none("2026-07-23T01:02:03")

    assert zulu == datetime(2026, 7, 23, 1, 2, 3, tzinfo=timezone.utc)
    assert offset == zulu
    assert naive == zulu
    assert parse_utc_datetime_or_none("not-a-timestamp") is None
    assert parse_utc_datetime_or_none(None) is None


def test_consent_parsers_share_utc_owner() -> None:
    assert business_authorization._parse_datetime is parse_utc_datetime_or_none
    assert model_consent._parse_datetime is parse_utc_datetime_or_none
    assert vision_consent._parse_datetime is parse_utc_datetime_or_none


def test_utc_iso_generation_has_one_owner() -> None:
    source_root = Path(__file__).parents[1] / "src" / "video_knowledge_pipeline"
    implementation = (
        "datetime.now(timezone.utc).replace(microsecond=0).isoformat()"
    )
    owners = {
        path.name
        for path in source_root.glob("*.py")
        if implementation in path.read_text(encoding="utf-8-sig")
    }

    assert owners == {"time_utils.py"}


def test_utc_datetime_parsing_has_one_owner() -> None:
    source_root = Path(__file__).parents[1] / "src" / "video_knowledge_pipeline"
    implementation = 'datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))'
    owners = {
        path.name
        for path in source_root.glob("*.py")
        if implementation in path.read_text(encoding="utf-8-sig")
    }

    assert owners == {"time_utils.py"}
