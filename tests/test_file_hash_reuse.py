from __future__ import annotations

import hashlib
from pathlib import Path

import video_knowledge_pipeline.consented_model_batch as consented_model_batch
import video_knowledge_pipeline.file_hash as file_hash
import video_knowledge_pipeline.model_connector_consent as model_connector_consent
import video_knowledge_pipeline.video as video


def test_file_hash_owner_streams_and_video_keeps_compatibility(tmp_path: Path) -> None:
    payload = (b"vkp-file-hash" * 100_000) + b"tail"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(payload)

    assert file_hash.sha256_file(artifact) == hashlib.sha256(payload).hexdigest()
    assert video.sha256_file is file_hash.sha256_file
    assert consented_model_batch._file_sha256 is file_hash.sha256_file
    assert model_connector_consent._sha256 is file_hash.sha256_file


def test_source_has_one_streaming_file_hash_owner() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "video_knowledge_pipeline"
    sources = {
        path.name: path.read_text(encoding="utf-8-sig")
        for path in source_root.glob("*.py")
    }

    full_file_hash = "hashlib.sha256(path.read_bytes()).hexdigest()"
    assert not [name for name, source in sources.items() if full_file_hash in source]

    streaming_loop = "for chunk in iter(lambda: handle.read(1024 * 1024), b\"\")"
    assert [name for name, source in sources.items() if streaming_loop in source] == [
        "file_hash.py"
    ]
