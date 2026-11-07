"""Durable Codex handoff; evidence and candidates, never model execution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_freshness import build_dependency_snapshot, validate_dependency_snapshot
from .file_hash import sha256_file
from .models import now_iso
from .powershell import quote_powershell_literal as quote
from .storage import read_json, write_json

SCHEMA = "video_knowledge_pipeline.smart_summary_handoff.v1"
STATE = "exports/smart-summary-handoff.json"
WATERMARK = "> VKP review-required：机器候选，尚未通过质量与来源审核，不是正式智能总结。"


def mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def summary_input_path(root: Path) -> Path | None:
    # Reuse the same transcript selection used by the summary evidence pack.
    from .smart_summary_input_pack import _transcript_path
    return _transcript_path(root, mapping(root / "manifest.json"), mapping(root / "sidecar.json"))


def handoff_snapshot(root: Path) -> dict[str, Any]:
    from .smart_summary_codex import _smart_summary_dependency_inputs
    inputs = _smart_summary_dependency_inputs(root)
    source = summary_input_path(root)
    if source is None or not source.is_file():
        raise ValueError("summary handoff requires a readable transcript")
    # Shared snapshots reject paths outside the Bundle. Never silently omit a
    # selected external input and certify the remaining dependencies as complete.
    inputs.append({"role": "summary_transcript", "path": source})
    for name in ("smart-summary-input-pack.json", "smart-summary-chapter-fact-pack.json",
                 "smart-summary-chapters.json", "smart-summary-llm-rewrite-pack.md"):
        path = root / "exports" / name
        if path.is_file():
            inputs.append({"role": name, "path": path})
    unique = {str(Path(row["path"]).resolve()): row for row in inputs}
    return build_dependency_snapshot(root, subject="smart-summary-handoff", inputs=unique.values(),
        source_run_id="smart-summary-codex", producer_schema=SCHEMA)


def handoff_status(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    value = mapping(root / STATE)
    if not value:
        return {"status": "missing", "ok": False}
    try:
        snapshot = value["dependency_snapshot"]
        validation = validate_dependency_snapshot(root, snapshot)
        current = handoff_snapshot(root)
        if not validation["passed"] or current["snapshot_sha256"] != snapshot["snapshot_sha256"]:
            raise ValueError("summary evidence or selected input set changed")
        for key, digest in (("raw_candidate_path", "candidate_sha256"),
                            ("review_candidate_path", "review_candidate_sha256")):
            if value.get(key):
                path = _inside(root, value[key])
                if not path.is_file() or sha256_file(path) != value.get(digest):
                    raise ValueError("registered candidate content changed")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {**value, "status": "blocked", "freshness": "stale", "ok": False, "reason": "stale_summary_handoff",
                "detail": str(exc), "next_actions": [
                    "Prepare a new Codex handoff from current inputs and rewrite/review the candidate; do not promote the old candidate."]}
    return {**value, "freshness": "fresh"}


def prepare_summary_handoff(root: Path, *, agent_available: bool = False,
                            candidate_path: Path | None = None,
                            write: bool = True, reason: str = "needs_llm_rewrite") -> dict[str, Any]:
    previous = handoff_status(root)
    if previous.get("freshness") == "fresh":
        return {**previous, "agent_available": bool(agent_available), "reused": True}
    try:
        snapshot = handoff_snapshot(root)
    except (OSError, ValueError) as exc:
        return {"status": "blocked", "ok": False, "reason": "summary_inputs_unavailable",
                "detail": str(exc), "next_actions": ["Repair the summary transcript/evidence binding before preparing the handoff."]}
    identity = snapshot["snapshot_sha256"]
    directory = root / "runs" / "smart-summary-handoff" / identity
    prompt = root / "exports" / "smart-summary-llm-rewrite-pack.md"
    candidate = _inside(root, str(candidate_path)) if candidate_path else directory / "candidate.md"
    command = (f".\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex {quote(str(root))} "
               f"--input-md {quote(str(candidate))} --handoff-id {identity}")
    result = {
        "schema": SCHEMA, "handoff_id": identity, "status": "awaiting-agent", "ok": False,
        "legacy_status": "needs_llm_rewrite", "reason": reason,
        "agent_available": bool(agent_available), "dependency_snapshot": snapshot,
        "prompt_path": str(prompt), "candidate_output_path": str(candidate),
        "submit_command": command, "updated_at": now_iso(), "reused": False,
        "next_actions": [f"Read {prompt} and its chapter/evidence references in the current authorized Agent session.",
                         f"Write the evidence-grounded Markdown candidate to {candidate}; retain all unresolved review items.",
                         command],
        "operator_boundary": {"model_called": False, "new_thread_started": False,
                              "provider_fallback": False, "formal_output_written": False},
    }
    if write:
        directory.mkdir(parents=True, exist_ok=True)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        write_json(directory / "handoff.json", result)
        write_json(root / STATE, result)
        _register(root, result)
    return result


def validate_summary_submission(root: Path, source: Path, *, handoff_id: str = "") -> dict[str, Any]:
    """Validate the prepared dependency set before any admission/preparation write."""
    current = handoff_status(root)
    if current.get("status") == "missing" and not handoff_id:
        return {"passed": True, "status": "legacy_explicit_input"}
    if current.get("freshness") != "fresh" or (handoff_id and handoff_id != current.get("handoff_id")):
        return {"passed": False, "status": "blocked_stale_candidate", "handoff": current}
    # A displayed watermarked copy is not an installable final candidate.
    if WATERMARK in source.read_text(encoding="utf-8-sig"):
        return {"passed": False, "status": "blocked_review_candidate", "handoff": current}
    return {"passed": True, "status": "fresh", "handoff": current,
            "dependency_snapshot": current["dependency_snapshot"]}


def register_review_candidate(root: Path, source: Path, *, gate: dict[str, Any],
                              validation: dict[str, Any] | None = None,
                              write: bool = True) -> dict[str, Any]:
    from .codex_vision_session import _write_bytes_atomic
    if source.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("summary_candidate_budget")
    raw = source.read_bytes()
    text = raw.decode("utf-8-sig")
    if not text.strip():
        raise ValueError("empty summary candidate")
    import hashlib
    digest = hashlib.sha256(raw).hexdigest()
    current = handoff_status(root)
    if current.get("status") == "missing":
        try:
            snapshot = handoff_snapshot(root)
        except (OSError, ValueError):
            snapshot = {}  # Retain the candidate, explicitly non-promotable.
        current = {"schema": SCHEMA, "handoff_id": snapshot.get("snapshot_sha256", "unbound"),
                   "dependency_snapshot": snapshot}
    directory = root / "runs" / "smart-summary-handoff" / current["handoff_id"] / digest
    raw_path = directory / "raw-candidate.md"
    review_path = directory / "review-required.md"
    review_bytes = (WATERMARK + "\n\n" + text).encode("utf-8")
    unresolved = list(gate.get("blocking_reasons") or [])
    validation = validation or {}
    if validation and validation.get("passed") is not True:
        unresolved.append({"key": "candidate_validation", "status": validation.get("status"),
                           "detail": validation.get("detail", ""),
                           "checks": validation.get("checks", []), "issues": validation.get("issues", [])})
    report = mapping(root / "transcript-quality-gate.json")
    unresolved.extend(row for row in report.get("issues", []) if isinstance(row, dict) and row.get("passed") is not True)
    result = {**current, "status": "review-required", "ok": False,
              "candidate_sha256": digest, "source_path": str(source),
              "raw_candidate_path": raw_path.relative_to(root).as_posix(),
              "review_candidate_path": review_path.relative_to(root).as_posix(),
              "review_candidate_sha256": hashlib.sha256(review_bytes).hexdigest(),
              "production_gate": gate, "candidate_validation": validation,
              "remaining_review_items": unresolved,
              "review_evidence": {"transcript_quality": report},
              "updated_at": now_iso(), "formal_output_written": False,
              "next_actions": ["Resolve the recorded quality/review gaps; revalidate dependencies before formal installation.",
                  f".\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex {quote(str(root))} "
                  f"--input-md {quote(str(raw_path))} --handoff-id {current['handoff_id']}"]}
    if write:
        for path, content in ((raw_path, raw), (review_path, review_bytes)):
            if path.exists() and path.read_bytes() != content:
                raise ValueError("immutable summary candidate was modified")
            if not path.exists():
                _write_bytes_atomic(path, content)
        write_json(directory / "review.json", result)
        write_json(root / STATE, result)
        _register(root, result)
    return result


def record_summary_install(root: Path, result: dict[str, Any], *, write: bool) -> None:
    if not write or result.get("ok") is not True:
        return
    current = mapping(root / STATE)
    if current:
        current.update(status="completed", ok=True, installed_candidate_sha256=result.get("candidate_sha256"),
                       updated_at=now_iso(), next_actions=result.get("next_actions", []))
        write_json(root / STATE, current)
        _register(root, current)


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("candidate path outside bundle")
    return path


def _register(root: Path, result: dict[str, Any]) -> None:
    from .run_artifact_registry import register_bundle_run
    review = result.get("review_candidate_path")
    artifacts = [{"key": "handoff", "path": str(root / STATE)}]
    if review:
        artifacts.append({"key": "review_required_candidate", "path": str(root / review),
                          "description": "Watermarked machine candidate; not a formal summary."})
    register_bundle_run(root, run_type="smart_summary_handoff",
        run_id="summary-handoff-" + str(result["handoff_id"])[:20], status=result["status"],
        title="Codex summary handoff", summary="Candidate/Agent stage only; formal gates remain required.",
        artifacts=artifacts, inputs={"candidate_sha256": result.get("candidate_sha256", "")},
        dependency_snapshot=result.get("dependency_snapshot", {}), next_actions=result.get("next_actions", []),
        operator_boundary={"no_model_call": True, "not_formal_output": True})


def summary_handoff_action(root: Path) -> dict[str, Any] | None:
    from .smart_summary_codex import _existing_codex_summary
    installed = _existing_codex_summary(root)
    if installed:
        return None  # Existing summary quality is checked by the completion gate.
    if summary_input_path(root) is None:
        return None
    current = handoff_status(root)
    review = current.get("status") in {"review-required", "blocked"}
    return {"status": "review-required" if review else "awaiting-agent", "kind": "smart_summary_handoff",
            "key": "smart_summary_handoff", "label": "复核智能总结候选" if review else "由当前 Agent 接力撰写智能总结",
            "reason": "质量门禁保留；候选登记、正式安装与可读导出分别验收。",
            "mcp_tool": "generate_smart_summary_with_codex", "human_required": review,
            "command": (current.get("next_actions") or [
                f".\\scripts\\video-knowledge.ps1 generate-smart-summary-with-codex {quote(str(root))}"])[-1],
            "handoff": current}
