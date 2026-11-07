from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.storage import write_text_atomic
from video_knowledge_pipeline.transcript_semantic_transaction import (
    _ALLOWED_TARGETS,
    apply_transcript_semantic_restore,
    commit_transcript_semantic_artifacts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_semantic_artifact_commit_is_hash_bound_and_restorable(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    canonical = bundle / "source-arbitrated-transcript.json"
    write_text_atomic(manifest, '{"revision": 1}')
    write_text_atomic(canonical, '{"segments": ["old"]}')

    result = commit_transcript_semantic_artifacts(
        bundle,
        artifacts={
            "source-arbitrated-transcript.json": '{"segments": ["new"]}',
            "transcript-semantic-correction-closure.json": '{"status": "completed"}',
        },
        expected_inputs={"manifest.json": _sha(manifest)},
        operation_identity={"pack_sha256": "a" * 64, "validation_sha256": "b" * 64},
    )

    assert result["status"] == "committed"
    assert result["transaction_id"].startswith("semcorr-")
    assert canonical.read_text(encoding="utf-8") == '{"segments": ["new"]}'
    plan = Path(result["restore_plan"])
    assert plan.is_file()

    preview = apply_transcript_semantic_restore(bundle, plan, execute=False)
    assert preview["status"] == "ready"
    restored = apply_transcript_semantic_restore(
        bundle,
        plan,
        execute=True,
        confirm_transaction_id=result["transaction_id"],
    )
    assert restored["status"] == "restored"
    assert canonical.read_text(encoding="utf-8") == '{"segments": ["old"]}'
    assert not (bundle / "transcript-semantic-correction-closure.json").exists()


def test_semantic_artifact_commit_rejects_precondition_drift_without_changes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    target = bundle / "source-arbitrated-transcript.json"
    write_text_atomic(manifest, '{"revision": 2}')
    write_text_atomic(target, "old")

    before = target.read_bytes()
    with pytest.raises(RuntimeError, match="semantic_transaction_input_drift:manifest.json"):
        commit_transcript_semantic_artifacts(
            bundle,
            artifacts={"source-arbitrated-transcript.json": "new"},
            expected_inputs={"manifest.json": "0" * 64},
            operation_identity={"pack_sha256": "a" * 64},
        )
    assert target.read_bytes() == before


def test_semantic_restore_rejects_changed_output(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    target = bundle / "source-arbitrated-transcript.md"
    write_text_atomic(target, "before")
    result = commit_transcript_semantic_artifacts(
        bundle,
        artifacts={"source-arbitrated-transcript.md": "after"},
        expected_inputs={},
        operation_identity={"pack_sha256": "a" * 64},
    )
    write_text_atomic(target, "operator edit")

    receipt = apply_transcript_semantic_restore(
        bundle,
        result["restore_plan"],
        execute=True,
        confirm_transaction_id=result["transaction_id"],
    )
    assert receipt["status"] == "blocked"
    assert "output_drift:source-arbitrated-transcript.md" in receipt["blockers"]
    assert target.read_text(encoding="utf-8") == "operator edit"


def test_semantic_transaction_rejects_targets_outside_allowlist(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(ValueError, match="semantic_transaction_target_not_allowed"):
        commit_transcript_semantic_artifacts(
            bundle,
            artifacts={"timeline.json": "[]"},
            expected_inputs={},
            operation_identity={"pack_sha256": "a" * 64},
        )


def test_semantic_transaction_rejects_bundle_local_target_symlink(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    target = bundle / "source-arbitrated-transcript.json"
    try:
        target.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="semantic_transaction_target_is_link"):
        commit_transcript_semantic_artifacts(
            bundle,
            artifacts={"source-arbitrated-transcript.json": "new"},
            expected_inputs={},
            operation_identity={"pack_sha256": "a" * 64},
        )
    assert outside.read_text(encoding="utf-8") == "outside"


@pytest.mark.parametrize("crash_target", sorted(_ALLOWED_TARGETS))
def test_semantic_transaction_crash_after_each_target_is_restorable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_target: str,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifacts = {name: f"after:{name}" for name in _ALLOWED_TARGETS}
    before: dict[str, bytes | None] = {}
    for index, name in enumerate(sorted(_ALLOWED_TARGETS)):
        target = bundle / name
        if index % 2 == 0:
            target.write_text(f"before:{name}", encoding="utf-8")
            before[name] = target.read_bytes()
        else:
            before[name] = None

    import video_knowledge_pipeline.transcript_semantic_transaction as transaction_module

    real_write = transaction_module.write_text_atomic
    crashed = False

    def crash_after_target_write(path: str | Path, content: str) -> None:
        nonlocal crashed
        real_write(path, content)
        candidate = Path(path)
        if not crashed and candidate.parent == bundle and candidate.name == crash_target:
            crashed = True
            raise SystemExit(f"injected crash after {crash_target}")

    monkeypatch.setattr(transaction_module, "write_text_atomic", crash_after_target_write)
    with pytest.raises(SystemExit, match="injected crash"):
        commit_transcript_semantic_artifacts(
            bundle,
            artifacts=artifacts,
            expected_inputs={},
            operation_identity={"pack_sha256": "a" * 64},
        )
    monkeypatch.setattr(transaction_module, "write_text_atomic", real_write)

    plans = list((bundle / "runs" / "transcript-semantic-correction").glob("*/restore-plan.json"))
    journals = list((bundle / "runs" / "transcript-semantic-correction").glob("*/transaction-journal.json"))
    assert len(plans) == 1
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert len(journal["restore_entries"]) == len(_ALLOWED_TARGETS)
    assert journal["restore_plan"].endswith("restore-plan.json")

    restored = apply_transcript_semantic_restore(
        bundle,
        journals[0],
        execute=True,
        confirm_transaction_id=journal["transaction_id"],
    )
    assert restored["status"] == "restored"
    for name, prior in before.items():
        target = bundle / name
        if prior is None:
            assert not target.exists()
        else:
            assert target.read_bytes() == prior
