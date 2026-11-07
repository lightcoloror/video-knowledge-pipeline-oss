from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic


JOURNAL_SCHEMA = "video_knowledge_pipeline.transcript_semantic_transaction_journal.v1"
RESTORE_SCHEMA = "video_knowledge_pipeline.transcript_semantic_restore_plan.v1"
RESTORE_RECEIPT_SCHEMA = "video_knowledge_pipeline.transcript_semantic_restore_receipt.v1"

_ALLOWED_TARGETS = frozenset(
    {
        "manifest.json",
        "source-arbitrated-transcript.json",
        "source-arbitrated-transcript.srt",
        "source-arbitrated-transcript.md",
        "transcript-semantic-correction-decision-ledger.json",
        "transcript-semantic-correction-closure.json",
        "transcript-semantic-correction-closure.md",
        "mcp-transcript-semantic-correction-closure.args.json",
    }
)

_PROFILE_TARGETS = {
    "knowledge_export": frozenset({
        "manifest.json", "knowledge-export-freshness.json",
        "exports/knowledge-note.md", "exports/full-transcript.md", "exports/full-body.md",
        "exports/transcript-projection-audit.json", "exports/smart-summary.md",
        "exports/smart-summary-codex-prompt.md", "exports/extraction-audit.md",
        "exports/key-segments.md", "exports/short-video-script-drafts.md", "exports/highlight-post-drafts.md",
        "exports/content-candidate-pack.json", "exports/content-candidate-pack.md",
        "exports/content-material-card.json", "exports/content-material-card.md",
        "exports/benchmark-script.json", "exports/benchmark-script.md",
        "exports/reader-export-receipt.json", "exports/knowledge-export-dependency-snapshot.json",
        "exports/export-summary.json", "exports/smart-summary-quality.json", "exports/smart-summary-quality.md",
    }),
    "transcript_semantic": _ALLOWED_TARGETS,
    "summary_claim_review": frozenset({"summary-claim-review.json"}),
    "summary_reduce_snapshot": frozenset({
        "manifest.json",
        "exports/smart-summary-global-reduce.json", "exports/smart-summary-global-reduce.md",
        "exports/smart-summary-global-reduce-prompt.md", "exports/smart-summary-global-reduce-candidate.md",
        "exports/smart-summary-global-reduce-raw-response.txt", "exports/smart-summary-reader-plan.json",
        "exports/smart-summary-chapter-fact-pack.json",
    }),
    "summary_install": frozenset({
        "exports/smart-summary.codex.md", "exports/smart-summary-dependency-snapshot.json",
        "exports/smart-summary-quality.json", "exports/smart-summary-quality.md",
        "exports/smart-summary-codex-status.json", "exports/smart-summary-codex-status.md",
    }),
}

_PROFILE_RUNS = {"knowledge_export": "knowledge-export", "transcript_semantic": "transcript-semantic-correction",
                 "summary_claim_review": "summary-claim-review",
                 "summary_reduce_snapshot": "summary-reduce-snapshot", "summary_install": "summary-install"}
_INDEXED_PROFILES = frozenset({"summary_reduce_snapshot", "summary_install", "knowledge_export"})


def summary_transaction_status(bundle_dir: str | Path) -> dict[str, Any]:
    statuses = [artifact_transaction_status(bundle_dir, profile=p) for p in sorted(_INDEXED_PROFILES)]
    return next((row for row in statuses if not row['passed']),
                {'passed': True, 'status': 'ready', 'profiles': statuses})


def artifact_transaction_status(bundle_dir: str | Path, *, profile: str) -> dict[str, Any]:
    """Read one profile pointer and its existing journal, never scan run history.

    This index stores identity only. The existing journal/restore receipts and
    current target Hashes remain the recovery authority; no second state machine.
    """
    root = _bundle_root(bundle_dir)
    _profile_targets(profile)
    result = {"status": "legacy_not_assessed", "passed": True, "profile": profile,
              "required": False, "recovery_required": False}
    def document(path):
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("transaction_metadata_budget")
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("transaction_metadata_object_required")
        return value
    try:
        directory = Path("runs") / _PROFILE_RUNS[profile]
        pointer = _bundle_path(root, directory / "latest-transaction.json", label="transaction_pointer")
        if not pointer.exists():
            return result
        result["required"] = True
        value = document(pointer)
        if (set(value) != {"schema", "transaction_id", "profile"} or value["profile"] != profile
                or value["schema"] != "video_knowledge_pipeline.artifact_transaction_pointer.v1"
                or not re.fullmatch(r"semcorr-[0-9a-f]{20}", str(value["transaction_id"]))):
            raise ValueError("transaction_pointer_identity_drift")
        run = directory / value["transaction_id"]
        journal_path = _bundle_path(root, run / "transaction-journal.json", label="transaction_journal")
        journal = document(journal_path)
        if (journal.get("schema") != JOURNAL_SCHEMA or journal.get("profile") != profile
                or journal.get("transaction_id") != value["transaction_id"]
                or Path(str(journal.get("bundle_dir", ""))).resolve() != root):
            raise ValueError("transaction_journal_identity_drift")
        result.update(transaction_id=value["transaction_id"], journal=journal_path.relative_to(root).as_posix(),
                      restore_plan=(run / "restore-plan.json").as_posix())
        if journal.get("status") in {"committed", "rolled_back"}:
            return {**result, "status": journal["status"]}
        # Explicit restore, not a stale cursor or file existence, may clear a
        # process-loss barrier. Verify both receipt and every current before Hash.
        receipt_path = _bundle_path(root, run / "restore-receipt.json", label="restore_receipt")
        if receipt_path.is_file():
            receipt = document(receipt_path)
            plan = _load_restore_document(root, _bundle_path(root, run / "restore-plan.json", label="restore_plan"))
            if (receipt.get("schema") == RESTORE_RECEIPT_SCHEMA and receipt.get("execute") is True
                    and receipt.get("transaction_id") == value["transaction_id"]
                    and receipt.get("status") in {"restored", "already_restored"}
                    and all(_restore_entry_state(_target_path(root, row['target'], profile=profile), row) == "before"
                            for row in plan["entries"])):
                return {**result, "status": "restored"}
        raise ValueError("pending_artifact_transaction: explicit restore required")
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        return {**result, "status": "blocked", "passed": False, "required": True,
                "recovery_required": True, "errors": [str(exc)]}


def _profile_targets(profile: str) -> frozenset[str]:
    if profile not in _PROFILE_TARGETS:
        raise ValueError("unknown_artifact_transaction_profile")
    return _PROFILE_TARGETS[profile]


def commit_transcript_semantic_artifacts(
    bundle_dir: str | Path,
    *,
    artifacts: Mapping[str, str | bytes],
    expected_inputs: Mapping[str, str],
    operation_identity: Mapping[str, Any],
    profile: str = "transcript_semantic",
) -> dict[str, Any]:
    """Reuse one journal engine, with staging/replay serialized by the Bundle lock."""
    root = _bundle_root(bundle_dir)
    with bundle_write_lock(root, operation="artifact_transaction", timeout_seconds=2.0):
        return _commit_artifacts(root, artifacts=artifacts, expected_inputs=expected_inputs,
                                 operation_identity=operation_identity, profile=profile)


def _commit_artifacts(
    bundle_dir: str | Path, *, artifacts: Mapping[str, str | bytes], expected_inputs: Mapping[str, str],
    operation_identity: Mapping[str, Any], profile: str,
) -> dict[str, Any]:
    """Commit the semantic-closure write set with a recoverable journal.

    Every target is staged and hashed within the caller's Bundle lock. Input
    preconditions are then rechecked, existing targets are backed up, and each
    target is atomically replaced. Ordinary write failures are rolled back in
    the same lock; process loss remains recoverable through the persisted plan.
    """

    root = _bundle_root(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    payloads = _normalise_artifacts(artifacts, profile=profile)
    if not payloads:
        raise ValueError("semantic_transaction_requires_artifacts")
    if profile in _INDEXED_PROFILES:
        health = summary_transaction_status(root)
        if not health['passed']:
            raise RuntimeError(f"pending_artifact_transaction:{health}")
    identity = _normalise_identity(operation_identity)
    if profile != "transcript_semantic":
        identity["artifact_profile"] = profile
    transaction_id = _transaction_id(payloads, expected_inputs, identity)
    run_dir = _bundle_path(
        root,
        Path("runs") / _PROFILE_RUNS[profile] / transaction_id,
        label="transaction_run",
    )
    stage_dir = run_dir / "stage"
    before_dir = run_dir / "before"
    journal_path = run_dir / "transaction-journal.json"
    restore_path = run_dir / "restore-plan.json"
    if run_dir.exists():
        # Never replace an interrupted run's original before-images on retry.
        # Completed identical submissions are no-ops, not a second transaction.
        if not journal_path.is_file():
            raise RuntimeError(f"existing_transaction_requires_restore:{run_dir}")
        journal = read_json(journal_path)
        if (not isinstance(journal, dict) or journal.get("status") != "committed"
                or journal.get("transaction_id") != transaction_id
                or journal.get("operation_identity") != identity):
            raise RuntimeError(f"existing_transaction_requires_restore:{journal_path}")
        plan = _load_restore_document(root, restore_path)
        entries = plan["entries"]
        if {row["target"] for row in entries} != set(payloads):
            raise RuntimeError("existing_transaction_artifact_drift")
        for row in entries:
            target = _target_path(root, row["target"], profile=profile)
            if not target.is_file() or _sha256_file(target) != row["after_sha256"]:
                raise RuntimeError(f"existing_transaction_output_drift:{row['target']}")
        _verify_expected_inputs(root, {k: v for k, v in expected_inputs.items() if k not in payloads})
        return {"schema": "video_knowledge_pipeline.transcript_semantic_transaction_receipt.v1",
                "status": "already_committed", "ok": True, "changed": False, "profile": profile,
                "transaction_id": transaction_id, "journal": str(journal_path),
                "restore_plan": str(restore_path), "artifact_count": len(entries), "artifacts": entries}
    stage_dir.mkdir(parents=True, exist_ok=True)
    before_dir.mkdir(parents=True, exist_ok=True)

    staged: list[dict[str, Any]] = []
    for name, payload in payloads.items():
        stage_path = stage_dir / name
        if isinstance(payload, bytes):
            from .codex_vision_session import _write_bytes_atomic
            _write_bytes_atomic(stage_path, payload)
        else:
            write_text_atomic(stage_path, payload)
        staged.append(
            {
                "target": name,
                "stage_path": str(stage_path.relative_to(root)),
                "after_sha256": _sha256_file(stage_path),
                "size": stage_path.stat().st_size,
                "payload_kind": "bytes" if isinstance(payload, bytes) else "text",
            }
        )

    journal: dict[str, Any] = {
        "schema": JOURNAL_SCHEMA,
        "profile": profile,
        "transaction_id": transaction_id,
        "bundle_dir": str(root),
        "status": "staged",
        "operation_identity": identity,
        "expected_inputs": dict(sorted((str(k), str(v)) for k, v in expected_inputs.items())),
        "artifacts": staged,
        "committed_targets": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    write_json(journal_path, journal)

    restore_entries: list[dict[str, Any]] = []
    with bundle_write_lock(root, operation="transcript_semantic_transaction", timeout_seconds=2.0):
        _verify_expected_inputs(root, expected_inputs)
        for row in staged:
            name = str(row["target"])
            target = _target_path(root, name, profile=profile)
            existed = target.is_file()
            before_sha = _sha256_file(target) if existed else ""
            backup_path = before_dir / name
            if existed:
                from .codex_vision_session import _write_bytes_atomic
                _write_bytes_atomic(backup_path, target.read_bytes())
            restore_entries.append(
                {
                    "target": name,
                    "existed_before": existed,
                    "before_sha256": before_sha,
                    "backup_path": str(backup_path.relative_to(root)) if existed else "",
                    "after_sha256": str(row["after_sha256"]),
                }
            )

        restore = {
            "schema": RESTORE_SCHEMA,
            "profile": profile,
            "transaction_id": transaction_id,
            "bundle_dir": str(root),
            "operation_identity": identity,
            "entries": restore_entries,
            "journal_path": str(journal_path.relative_to(root)),
            "status": "prepared",
            "created_at": now_iso(),
        }
        # The complete restore plan must exist before the first business target
        # is replaced. A hard process loss can therefore be recovered without
        # trusting the possibly stale committed_targets cursor in the journal.
        write_json(restore_path, restore)
        journal["status"] = "prepared"
        journal["restore_entries"] = restore_entries
        journal["restore_plan"] = str(restore_path.relative_to(root))
        journal["updated_at"] = now_iso()
        write_json(journal_path, journal)
        if profile in _INDEXED_PROFILES:
            # Index only a recovery-ready transaction, before its first target
            # write. Failure during staging never invalidates the prior snapshot.
            write_json(run_dir.parent / "latest-transaction.json", {
                "schema": "video_knowledge_pipeline.artifact_transaction_pointer.v1",
                "profile": profile, "transaction_id": transaction_id})
        try:
            for row in staged:
                name = str(row["target"])
                stage_path = _bundle_path(root, str(row["stage_path"]), label="transaction_stage")
                target = _target_path(root, name, profile=profile)
                if _sha256_file(stage_path) != row["after_sha256"]:
                    raise RuntimeError(f"semantic_transaction_stage_drift:{name}")
                if row["payload_kind"] == "bytes":
                    from .codex_vision_session import _write_bytes_atomic
                    _write_bytes_atomic(target, stage_path.read_bytes())
                else:
                    write_text_atomic(target, stage_path.read_text(encoding="utf-8"))
                if _sha256_file(target) != row["after_sha256"]:
                    raise RuntimeError(f"semantic_transaction_write_verification_failed:{name}")
                journal["committed_targets"].append(name)
                journal["status"] = "committing"
                journal["updated_at"] = now_iso()
                write_json(journal_path, journal)
        except Exception:
            rollback_errors = _restore_entries(root, restore_entries, require_after_hash=False, profile=profile)
            journal["status"] = "rolled_back" if not rollback_errors else "rollback_failed"
            journal["rollback_errors"] = rollback_errors
            journal["updated_at"] = now_iso()
            write_json(journal_path, journal)
            raise

        journal["status"] = "committed"
        journal["updated_at"] = now_iso()
        write_json(journal_path, journal)

    return {
        "schema": "video_knowledge_pipeline.transcript_semantic_transaction_receipt.v1",
        "status": "committed",
        "profile": profile,
        "ok": True,
        "transaction_id": transaction_id,
        "journal": str(journal_path),
        "restore_plan": str(restore_path),
        "artifact_count": len(staged),
        "artifacts": restore_entries,
        "operator_boundary": {
            "raw_asr_untouched": True,
            "timeline_untouched": True,
            "allowlisted_targets_only": True,
            "restore_requires_exact_after_hash": True,
        },
        "updated_at": now_iso(),
    }


def apply_transcript_semantic_restore(
    bundle_dir: str | Path,
    plan_json: str | Path,
    *,
    execute: bool = False,
    confirm_transaction_id: str = "",
) -> dict[str, Any]:
    root = _bundle_root(bundle_dir)
    plan_path = _bundle_path(root, Path(plan_json).expanduser(), label="restore_plan")
    plan = _load_restore_document(root, plan_path)
    profile = plan.get("profile", "transcript_semantic")
    transaction_id = str(plan.get("transaction_id") or "")
    blockers: list[str] = []
    if execute and confirm_transaction_id != transaction_id:
        blockers.append("confirm_transaction_id_mismatch")
    entries = [row for row in plan.get("entries") or [] if isinstance(row, dict)]
    restore_states: dict[str, str] = {}
    for row in entries:
        name = str(row.get("target") or "")
        target = _target_path(root, name, profile=profile)
        state = _restore_entry_state(target, row)
        restore_states[name] = state
        if state == "drifted":
            blockers.append(f"output_drift:{name}")
        if state == "after" and row.get("existed_before"):
            backup = _bundle_path(root, str(row.get("backup_path") or ""), label="restore_backup")
            try:
                _require_within(root, backup, label="restore_backup")
            except ValueError:
                blockers.append(f"backup_outside_bundle:{name}")
                continue
            if not backup.is_file() or _sha256_file(backup) != str(row.get("before_sha256") or ""):
                blockers.append(f"backup_drift:{name}")
    blockers = sorted(set(blockers))
    all_already_restored = bool(entries) and all(
        restore_states.get(str(row.get("target") or "")) == "before"
        for row in entries
    )
    status = (
        "blocked"
        if blockers
        else "already_restored"
        if all_already_restored
        else "ready"
    )
    if execute and not blockers and not all_already_restored:
        with bundle_write_lock(root, operation="transcript_semantic_restore", timeout_seconds=2.0):
            second_pass = _restore_preflight_blockers(root, entries, profile=profile)
            if second_pass:
                blockers.extend(second_pass)
                status = "blocked"
            else:
                errors = _restore_entries(root, entries, require_after_hash=True, profile=profile)
                if errors:
                    blockers.extend(errors)
                    status = "restore_failed"
                else:
                    status = "restored"
    receipt = {
        "schema": RESTORE_RECEIPT_SCHEMA,
        "status": status,
        "ok": status in {"ready", "restored", "already_restored"},
        "execute": bool(execute),
        "transaction_id": transaction_id,
        "plan_json": str(plan_path),
        "blockers": sorted(set(blockers)),
        "restored_targets": [
            str(row.get("target") or "")
            for row in entries
            if status == "restored" and restore_states.get(str(row.get("target") or "")) == "after"
        ],
        "already_restored_targets": [
            str(row.get("target") or "")
            for row in entries
            if restore_states.get(str(row.get("target") or "")) == "before"
        ],
        "operator_boundary": {
            "exact_hash_guard": True,
            "allowlisted_targets_only": True,
            "raw_asr_untouched": True,
            "timeline_untouched": True,
        },
        "updated_at": now_iso(),
    }
    receipt_path = plan_path.with_name("restore-receipt.json")
    write_json(receipt_path, receipt)
    return receipt


def _normalise_artifacts(artifacts: Mapping[str, str | bytes], *, profile: str = "transcript_semantic") -> dict[str, str | bytes]:
    result: dict[str, str | bytes] = {}
    for raw_name, raw_payload in artifacts.items():
        name = str(raw_name or "").strip().replace("\\", "/")
        if name not in _profile_targets(profile):
            raise ValueError(f"semantic_transaction_target_not_allowed:{name}")
        result[name] = raw_payload if isinstance(raw_payload, bytes) else str(raw_payload)
    return dict(sorted(result.items()))


def _normalise_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    result = {str(key): str(value) for key, value in identity.items() if str(key) and str(value)}
    if not result.get("pack_sha256"):
        raise ValueError("semantic_transaction_requires_pack_sha256")
    return dict(sorted(result.items()))


def _transaction_id(
    payloads: Mapping[str, str | bytes],
    expected_inputs: Mapping[str, str],
    identity: Mapping[str, str],
) -> str:
    envelope = {
        "artifacts": {name: _sha256_bytes(payload if isinstance(payload, bytes) else payload.encode("utf-8")) for name, payload in payloads.items()},
        "expected_inputs": dict(sorted((str(k), str(v)) for k, v in expected_inputs.items())),
        "operation_identity": dict(identity),
    }
    binary_targets = sorted(name for name, payload in payloads.items() if isinstance(payload, bytes))
    if binary_targets:
        # Preserve historical all-text identities, but do not alias two writers
        # whose newline/encoding serialization can differ on Windows.
        envelope["binary_targets"] = binary_targets
    digest = _sha256_bytes(json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return f"semcorr-{digest[:20]}"


def _verify_expected_inputs(root: Path, expected_inputs: Mapping[str, str]) -> None:
    for raw_name, expected_sha in sorted(expected_inputs.items()):
        name = str(raw_name or "").strip().replace("\\", "/")
        path = _bundle_path(root, name, label="expected_input")
        actual = _sha256_file(path) if path.is_file() else ""
        if actual != str(expected_sha or ""):
            raise RuntimeError(f"semantic_transaction_input_drift:{name}")


def _target_path(root: Path, name: str, *, profile: str = "transcript_semantic") -> Path:
    if name not in _profile_targets(profile):
        raise ValueError(f"semantic_transaction_target_not_allowed:{name}")
    return _bundle_path(root, name, label="target")


def _restore_preflight_blockers(root: Path, entries: list[dict[str, Any]], *, profile: str = "transcript_semantic") -> list[str]:
    blockers: list[str] = []
    for row in entries:
        name = str(row.get("target") or "")
        try:
            target = _target_path(root, name, profile=profile)
            state = _restore_entry_state(target, row)
        except (OSError, RuntimeError, ValueError) as exc:
            blockers.append(f"output_path_invalid:{name}:{exc}")
            continue
        if state == "drifted":
            blockers.append(f"output_drift:{name}")
        if state == "after" and row.get("existed_before"):
            try:
                backup = _bundle_path(root, str(row.get("backup_path") or ""), label="restore_backup")
            except (OSError, RuntimeError, ValueError) as exc:
                blockers.append(f"backup_path_invalid:{name}:{exc}")
                continue
            if not backup.is_file() or _sha256_file(backup) != str(row.get("before_sha256") or ""):
                blockers.append(f"backup_drift:{name}")
    return blockers


def _restore_entries(
    root: Path,
    entries: list[dict[str, Any]],
    *,
    require_after_hash: bool,
    profile: str = "transcript_semantic",
) -> list[str]:
    errors: list[str] = []
    for row in reversed(entries):
        name = str(row.get("target") or "")
        try:
            target = _target_path(root, name, profile=profile)
            state = _restore_entry_state(target, row)
            if state == "before":
                continue
            if state != "after":
                raise RuntimeError("output_drift")
            if row.get("existed_before"):
                backup = _bundle_path(root, str(row.get("backup_path") or ""), label="restore_backup")
                if not backup.is_file() or _sha256_file(backup) != str(row.get("before_sha256") or ""):
                    raise RuntimeError("backup_hash_mismatch")
                from .codex_vision_session import _write_bytes_atomic
                _write_bytes_atomic(target, backup.read_bytes())
            elif target.exists():
                target.unlink()
        except Exception as exc:  # noqa: BLE001 - every exact target error is retained in the receipt.
            errors.append(f"restore_failed:{name}:{exc}")
    return errors


def _load_restore_document(root: Path, path: Path) -> dict[str, Any]:
    document = read_json(path)
    if not isinstance(document, dict):
        raise ValueError("invalid transcript semantic restore document")
    schema = str(document.get("schema") or "")
    if schema == RESTORE_SCHEMA:
        plan = dict(document)
    elif schema == JOURNAL_SCHEMA:
        restore_entries = document.get("restore_entries")
        if not isinstance(restore_entries, list) or not restore_entries:
            raise ValueError("transcript semantic journal is not recovery-ready")
        plan = {
            "schema": RESTORE_SCHEMA,
            "profile": document.get("profile", "transcript_semantic"),
            "transaction_id": document.get("transaction_id"),
            "bundle_dir": document.get("bundle_dir"),
            "operation_identity": document.get("operation_identity") or {},
            "entries": restore_entries,
            "journal_path": str(path.relative_to(root)),
            "created_at": document.get("created_at"),
            "recovered_from": "transaction_journal",
        }
    else:
        raise ValueError("invalid transcript semantic restore plan")
    if Path(str(plan.get("bundle_dir") or "")).resolve() != root.resolve():
        raise ValueError("transcript semantic restore bundle identity mismatch")
    transaction_id = str(plan.get("transaction_id") or "")
    if not transaction_id.startswith("semcorr-"):
        raise ValueError("invalid transcript semantic restore transaction id")
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("transcript semantic restore plan has no entries")
    seen: set[str] = set()
    profile = plan.get("profile", "transcript_semantic")
    _profile_targets(profile)
    if profile != "transcript_semantic" and (plan.get("operation_identity") or {}).get("artifact_profile") != profile:
        raise ValueError("artifact_transaction_profile_identity_mismatch")
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("invalid transcript semantic restore entry")
        name = str(row.get("target") or "")
        if name in seen:
            raise ValueError(f"duplicate transcript semantic restore target:{name}")
        _target_path(root, name, profile=profile)
        seen.add(name)
    return plan


def _restore_entry_state(target: Path, row: Mapping[str, Any]) -> str:
    """Classify a target without trusting a crash-stale journal cursor."""

    existed_before = bool(row.get("existed_before"))
    before_sha = str(row.get("before_sha256") or "")
    after_sha = str(row.get("after_sha256") or "")
    if target.exists():
        if not target.is_file() or _is_linklike(target):
            return "drifted"
        actual = _sha256_file(target)
        if existed_before and before_sha and actual == before_sha:
            return "before"
        if after_sha and actual == after_sha:
            return "after"
        return "drifted"
    return "before" if not existed_before else "drifted"


def _bundle_root(bundle_dir: str | Path) -> Path:
    lexical = Path(os.path.abspath(str(Path(bundle_dir).expanduser())))
    if _is_linklike(lexical):
        raise RuntimeError(f"semantic_transaction_bundle_is_link:{lexical}")
    return lexical.resolve()


def _bundle_path(root: Path, raw_path: str | Path, *, label: str) -> Path:
    candidate = Path(raw_path)
    lexical = candidate if candidate.is_absolute() else root / candidate
    lexical = Path(os.path.abspath(str(lexical)))
    _require_lexical_within(root, lexical, label=label)
    _reject_link_components(root, lexical, label=label)
    _require_within(root, lexical.resolve(), label=label)
    # Return the checked lexical path. Returning resolve() here would erase the
    # identity of a Bundle-local symlink between validation and use.
    return lexical


def _require_lexical_within(root: Path, path: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"semantic_transaction_{label}_outside_bundle:{path}") from exc


def _reject_link_components(root: Path, path: Path, *, label: str) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if _is_linklike(current):
            raise RuntimeError(f"semantic_transaction_{label}_is_link:{relative.as_posix()}")


def _is_linklike(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0) or 0)
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0) or 0)
        return bool(reparse_flag and attributes & reparse_flag)
    except FileNotFoundError:
        return False


def _require_within(root: Path, path: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"semantic_transaction_{label}_outside_bundle:{path}") from exc


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
