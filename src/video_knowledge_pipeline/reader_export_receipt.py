from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any

from .file_hash import sha256_file
from .models import now_iso


READER_EXPORT_RECEIPT_SCHEMA = "video_knowledge_pipeline.reader_export_receipt.v2"


def build_reader_export_receipt(
    *,
    canonical_transcript: Path | None,
    full_transcript: Path,
    full_body: Path,
    reading_note: Path,
    reader_payloads: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    def digest(path):
        if reader_payloads is not None:
            return hashlib.sha256(reader_payloads[str(path)]).hexdigest()
        return sha256_file(path) if path.is_file() else ''
    return {
        "schema": READER_EXPORT_RECEIPT_SCHEMA,
        "created_at": now_iso(),
        "canonical_transcript_path": str(canonical_transcript) if canonical_transcript else "",
        "canonical_transcript_sha256": sha256_file(canonical_transcript) if canonical_transcript and canonical_transcript.is_file() else "",
        "full_transcript_path": str(full_transcript),
        "full_transcript_sha256": digest(full_transcript),
        "full_body_path": str(full_body),
        "full_body_sha256": digest(full_body),
        "reading_note_path": str(reading_note),
        "reading_note_sha256": digest(reading_note),
    }


def receipt_matches_reader_files(
    receipt: dict[str, Any],
    *,
    canonical_transcript: Path,
    full_transcript: Path,
    full_body: Path,
    reading_note: Path,
    reader_payloads: dict[str, bytes] | None = None,
) -> bool:
    if not isinstance(receipt, dict):
        return False
    if receipt.get("schema") != READER_EXPORT_RECEIPT_SCHEMA:
        return False
    if str(receipt.get("canonical_transcript_sha256") or "") != sha256_file(canonical_transcript):
        return False
    def digest(path):
        return hashlib.sha256(reader_payloads[str(path)]).hexdigest() if reader_payloads is not None else sha256_file(path)
    if str(receipt.get("full_transcript_sha256") or "") != digest(full_transcript):
        return False
    if str(receipt.get("full_body_sha256") or "") != digest(full_body):
        return False
    return str(receipt.get("reading_note_sha256") or "") == digest(reading_note)
