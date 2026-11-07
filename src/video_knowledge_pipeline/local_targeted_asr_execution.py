from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any

from .asr_execution import run_asr_plan
from .asr_local_targeted_evidence import build_local_targeted_asr_evidence
from .asr_retry_snippets import prepare_asr_retry_snippets
from .asr_runner import plan_asr_run
from .local_targeted_asr_plan import build_local_targeted_asr_plan
from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json


SCHEMA = "video_knowledge_pipeline.local_targeted_asr_execution.v1"
_MEDIA_KEYS = (
    "media_path",
    "local_media_path",
    "source_media_path",
    "source_video_path",
    "video_path",
    "local_video_path",
    "source_path",
    "path",
)


def run_local_targeted_asr_evidence(
    bundle_dir: str | Path,
    *,
    media_path: str | Path | None = None,
    input_plan: str | Path | None = None,
    preset: str = "qwen3-asr-0.6b",
    language: str = "zh",
    model: str | None = None,
    timeout_seconds: int = 900,
    execute: bool = False,
    allow_cpu: bool = False,
    max_windows: int = 24,
    padding_seconds: float = 3.0,
    qwen_timestamps: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Close the local high-risk-ASR evidence loop for one bundle.

    The operation is deliberately evidence-only: it extracts the bounded clips
    selected by the semantic pack, runs a distinct local ASR preset, shifts the
    verified clip timings back to global time, and registers the result as a
    secondary ASR sidecar. It never directly replaces the canonical transcript.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    if execute and not write:
        raise ValueError("execute=True requires write=True so verified evidence can be registered")

    plan, plan_path = _resolve_plan(
        root,
        input_plan=input_plan,
        max_windows=max_windows,
        padding_seconds=padding_seconds,
        write=write,
    )
    retry_plan = plan.get("retry_plan") if isinstance(plan.get("retry_plan"), dict) else {}
    windows = [row for row in retry_plan.get("windows") or [] if isinstance(row, dict)]
    report_path = root / "local-targeted-asr-execution.json"
    markdown_path = root / "local-targeted-asr-execution.md"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": "preview",
        "ok": False,
        "execute": bool(execute),
        "media_path": "",
        "input_plan": str(plan_path),
        "preset": str(preset),
        "model": str(model or ""),
        "language": str(language),
        "timeout_seconds": int(timeout_seconds or 0),
        "allow_cpu": bool(allow_cpu),
        "window_count": len(windows),
        "snippet_result": {},
        "asr_runs": [],
        "failed_runs": [],
        "evidence": {},
        "registered_secondary_asr_sources": [],
        "artifacts": {
            "execution_json": str(report_path),
            "execution_markdown": str(markdown_path),
            "plan_json": str(plan_path),
            "snippet_manifest": str(root / "local-targeted-asr-snippets" / "asr-retry-snippets.json"),
            "evidence_json": str(root / "local-targeted-asr-evidence.json"),
        },
        "operator_boundary": {
            "local_only": True,
            "audio_stays_local": True,
            "independent_second_asr": False,
            "model_independence": "requires_model_lineage_check",
            "candidate_only_evidence": True,
            "canonical_transcript_modified": False,
            "automatic_text_promotion_allowed": False,
            "media_path_resolution": "explicit_path_then_bundle_manifest_only",
        },
        "updated_at": now_iso(),
    }
    if not windows:
        result.update(
            {
                "status": "no_targeted_evidence_needed",
                "ok": True,
                "next_actions": ["No unresolved high-risk semantic candidates need a second local ASR pass."],
            }
        )
        return _write_result(root, result, write=write)

    manifest = _read_manifest(root)
    media = _resolve_media_path(root, manifest, media_path)
    if media is None:
        result.update(
            {
                "status": "blocked_media_path_missing",
                "next_actions": ["Add an existing local media path to manifest.json or provide --media-path, then rerun this one command."],
            }
        )
        return _write_result(root, result, write=write)
    result["media_path"] = str(media)

    snippets = prepare_asr_retry_snippets(
        media,
        plan,
        root / "local-targeted-asr-snippets",
        execute=execute,
    )
    result["snippet_result"] = _summary(snippets)
    snippet_manifest = Path(str(root / "local-targeted-asr-snippets" / "asr-retry-snippets.json")).resolve()
    result["artifacts"]["snippet_manifest"] = str(snippet_manifest)
    if not execute:
        result.update(
            {
                "status": "planned",
                "ok": True,
                "next_actions": ["Rerun with --execute to extract the planned local clips, run the independent ASR preset, and register candidate-only evidence."],
            }
        )
        return _write_result(root, result, write=write)

    completed_artifacts = [
        row for row in snippets.get("artifacts") or []
        if isinstance(row, dict) and str(row.get("status") or "") == "completed"
    ]
    if not completed_artifacts:
        result.update(
            {
                "status": "snippet_extraction_failed",
                "failed_runs": list(snippets.get("failed_chunks") or []),
                "next_actions": ["Inspect local-targeted-asr-snippets/asr-retry-snippets.json and repair only the failed local FFmpeg clips."],
            }
        )
        return _write_result(root, result, write=True)
    if len(completed_artifacts) != len(windows):
        result.update(
            {
                "status": "snippet_extraction_incomplete",
                "failed_runs": list(snippets.get("failed_chunks") or []),
                "next_actions": ["Every planned factual-risk clip must be extracted before local evidence can be registered. Repair the failed FFmpeg clips and rerun."],
            }
        )
        return _write_result(root, result, write=True)

    raw_outputs: list[Path] = []
    for position, artifact in enumerate(completed_artifacts):
        retry_id = str(artifact.get("retry_id") or "clip")
        clip = Path(str(artifact.get("path") or "")).expanduser().resolve()
        run_workspace = root / "local-targeted-asr-runs" / retry_id
        run_plan = plan_asr_run(
            run_workspace,
            clip,
            preset=preset,
            language=language,
            model=model or None,
            qwen_timestamps=qwen_timestamps,
        )
        run = run_asr_plan(
            run_plan["plan_path"],
            execute=True,
            normalize=False,
            timeout_seconds=timeout_seconds,
        )
        run_row = {
            "retry_id": retry_id,
            "clip_path": str(clip),
            "plan_path": str(run_plan["plan_path"]),
            "status": str(run.get("status") or ""),
            "raw_output_json": str(run.get("raw_output_json") or ""),
            "preset": str(run_plan.get("preset") or preset),
            "provider": str(run_plan.get("provider") or ""),
        }
        raw_output = Path(run_row["raw_output_json"]).expanduser() if run_row["raw_output_json"] else None
        if run_row["status"] == "ok" and raw_output and raw_output.exists():
            raw_outputs.append(raw_output.resolve())
            result["asr_runs"].append(run_row)
        else:
            run_row["error"] = str(run.get("stderr") or run.get("status") or "local_asr_failed")
            diagnostic = _runner_failure(run)
            run_row.update(diagnostic)
            result["failed_runs"].append(run_row)
            if diagnostic.get("error_code") in {"qwen3_asr_model_not_ready", "qwen3_aligner_not_ready"}:
                result["not_run"] = [
                    {"retry_id": str(row.get("retry_id") or ""),
                     "reason": "shared_runtime_preflight_failed"}
                    for row in completed_artifacts[position + 1:]
                ]
                break

    if not raw_outputs:
        result.update(
            {
                "status": "local_asr_failed",
                "next_actions": ["Inspect local-targeted-asr-runs for the failed preset and rerun the same bundle command after the local model is ready."],
            }
        )
        return _write_result(root, result, write=True)

    evidence_path = root / "local-targeted-asr-evidence.json"
    evidence = build_local_targeted_asr_evidence(
        [snippet_manifest],
        raw_outputs,
        output_json=evidence_path,
        require_gpu=not allow_cpu,
        write=True,
    )
    result["evidence"] = _summary(evidence)
    result["artifacts"]["evidence_json"] = str(evidence_path)
    if evidence.get("status") == "completed":
        result["registered_secondary_asr_sources"] = _register_secondary_evidence(root, evidence_path, evidence)
        result.update(
            {
                "status": "completed",
                "ok": True,
                "next_actions": ["The verified second-ASR evidence is registered. Rebuild the semantic pack and quality gate; this command does that automatically when used through transcript-evidence-correction-pipeline --execute-local-targeted-asr."],
            }
        )
    else:
        result.update(
            {
                "status": "evidence_incomplete",
                "next_actions": ["Only completed, GPU-verified local clips can be registered as semantic evidence. Repair the listed failed clip ASR runs and rerun."],
            }
        )
    return _write_result(root, result, write=True)


def _runner_failure(run: dict[str, Any]) -> dict[str, Any]:
    """Retain the runner's precise failure instead of only its progress stderr."""
    value = run.get("raw_output_json") or run.get("expected_output_json")
    if not value:
        return {}
    path = Path(str(value)).expanduser()
    if not path.is_file():
        return {}
    if not run.get("raw_output_json"):
        try:
            started = datetime.fromisoformat(str(run["started_at"])).timestamp()
        except (KeyError, TypeError, ValueError):
            return {}
        if path.stat().st_mtime < started:
            return {}
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("status") != "failed":
        return {}
    if run.get("status") != "failed":
        return {}
    return {"raw_output_json": str(path.resolve()), **{
        key: payload[key] for key in ("error", "error_code", "model_readiness", "aligner_readiness")
        if key in payload
    }}


def _resolve_plan(
    root: Path,
    *,
    input_plan: str | Path | None,
    max_windows: int,
    padding_seconds: float,
    write: bool,
) -> tuple[dict[str, Any], Path]:
    if input_plan:
        path = Path(input_plan).expanduser()
        path = path if path.is_absolute() else root / path
        path = path.resolve()
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"local targeted ASR plan must be a JSON object: {path}")
        return payload, path
    plan = build_local_targeted_asr_plan(
        root,
        max_windows=max_windows,
        padding_seconds=padding_seconds,
        write=write,
    )
    return plan, (root / "local-targeted-asr-plan.json").resolve()


def _resolve_media_path(root: Path, manifest: dict[str, Any], explicit: str | Path | None) -> Path | None:
    candidates: list[Any] = [explicit]
    candidates.extend(manifest.get(key) for key in _MEDIA_KEYS)
    for container_key in ("source_media", "source", "video", "source_package"):
        container = manifest.get(container_key)
        if isinstance(container, dict):
            candidates.extend(container.get(key) for key in _MEDIA_KEYS)
    for source in manifest.get("sources") or []:
        if isinstance(source, dict):
            candidates.extend(source.get(key) for key in _MEDIA_KEYS)
    for value in candidates:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            return path.resolve()
    return None


def _register_secondary_evidence(root: Path, evidence_path: Path, evidence: dict[str, Any]) -> list[str]:
    with bundle_write_lock(root, operation="register_local_targeted_asr_evidence", timeout_seconds=1.0):
        manifest = _read_manifest(root)
        existing = manifest.get("asr_secondary_transcripts")
        values = existing if isinstance(existing, list) else ([existing] if existing else [])
        legacy = manifest.get("asr_secondary_transcript")
        if legacy and legacy not in values:
            values = [legacy, *values]
        registered: list[str] = []
        for value in [*values, str(evidence_path.resolve())]:
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = root / path
            resolved = str(path.resolve())
            if resolved not in registered:
                registered.append(resolved)
        manifest["asr_secondary_transcripts"] = registered
        manifest.setdefault("asr_secondary_transcript", registered[0])
        manifest["local_targeted_asr_evidence_json"] = "local-targeted-asr-evidence.json"
        manifest["local_targeted_asr_evidence_summary"] = {
            "status": evidence.get("status"),
            "segment_count": evidence.get("segment_count"),
            "completed_window_count": evidence.get("completed_window_count"),
            "candidate_only": True,
            "updated_at": now_iso(),
        }
        write_json(root / "manifest.json", manifest)
    return registered


def _read_manifest(root: Path) -> dict[str, Any]:
    value = read_json(root / "manifest.json")
    return value if isinstance(value, dict) else {}


def _summary(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "ok",
        "window_count",
        "completed_count",
        "failed_count",
        "completed_window_count",
        "missing_window_count",
        "failed_output_count",
        "segment_count",
        "output_json",
    )
    return {key: value.get(key) for key in keys if key in value}


def _write_result(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    if not write:
        return result
    report_path = root / "local-targeted-asr-execution.json"
    markdown_path = root / "local-targeted-asr-execution.md"
    write_json(report_path, result)
    lines = [
        "# Local Targeted ASR Execution",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Media: `{result.get('media_path', '')}`",
        f"- Windows: `{result.get('window_count', 0)}`",
        f"- Preset: `{result.get('preset', '')}`",
        "",
        "## Next Actions",
        "",
    ]
    lines.extend(f"- {item}" for item in result.get("next_actions") or [])
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    with bundle_write_lock(root, operation="write_local_targeted_asr_execution_report", timeout_seconds=1.0):
        manifest = _read_manifest(root)
        manifest["local_targeted_asr_execution_json"] = "local-targeted-asr-execution.json"
        manifest["local_targeted_asr_execution_markdown"] = "local-targeted-asr-execution.md"
        manifest["local_targeted_asr_execution_summary"] = {
            "status": result.get("status"),
            "window_count": result.get("window_count"),
            "updated_at": result.get("updated_at"),
        }
        write_json(root / "manifest.json", manifest)
    return result
