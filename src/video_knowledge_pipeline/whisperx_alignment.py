from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_execution import run_asr_plan
from .asr_runner import plan_whisperx_alignment
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import ensure_project_dirs, read_json, write_json

SCHEMA = "video_knowledge_pipeline.whisperx_alignment_run.v1"


def run_whisperx_alignment(
    root: str | Path,
    media_path: str | Path,
    *,
    language: str = "zh",
    model: str = "large-v3",
    execute: bool = False,
    timeout_seconds: int = 1800,
    write: bool = True,
) -> dict[str, Any]:
    """Plan or run WhisperX as an alignment evidence branch.

    WhisperX output is intentionally kept as an auxiliary alignment transcript.
    It does not replace SenseVoice/FunASR, corrected-transcript.json, or the
    final full-transcript unless a later arbitration step explicitly promotes it.
    """

    root_path = Path(root).expanduser().resolve()
    media = Path(media_path).expanduser().resolve()
    plan = plan_whisperx_alignment(root_path, media, language=language, model=model or None)
    run = run_asr_plan(plan["plan_path"], execute=execute, timeout_seconds=timeout_seconds) if execute else run_asr_plan(plan["plan_path"], execute=False, timeout_seconds=timeout_seconds)
    normalized = run.get("normalized") if isinstance(run.get("normalized"), dict) else {}
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "root": str(root_path),
        "media_path": str(media),
        "execute": bool(execute),
        "language": language,
        "model": model or "large-v3",
        "status": _status(run, execute=execute),
        "plan_path": plan.get("plan_path", ""),
        "asr_run_status": run.get("status", ""),
        "raw_output_json": run.get("raw_output_json", ""),
        "alignment_transcript_json": normalized.get("json_path", ""),
        "alignment_transcript_srt": normalized.get("srt_path", ""),
        "word_level_alignment": _has_word_alignment(normalized.get("json_path", "")),
        "operator_boundary": {
            "purpose": "timestamp_and_speaker_alignment_only",
            "does_not_replace_primary_asr": True,
            "does_not_promote_corrected_transcript": True,
            "execute_required_for_local_heavy_run": True,
        },
        "next_actions": _next_actions(run, execute=execute),
        "updated_at": now_iso(),
    }
    if write:
        _write_outputs(root_path, result)
    return result


def _status(run: dict[str, Any], *, execute: bool) -> str:
    if not execute:
        return "preview"
    if run.get("status") == "ok":
        return "alignment_ready"
    return str(run.get("status") or "alignment_failed")


def _has_word_alignment(path: str) -> bool:
    if not path:
        return False
    source = Path(path)
    if not source.exists():
        return False
    try:
        data = read_json(source)
    except Exception:
        return False
    segments = data.get("segments") if isinstance(data, dict) else []
    if not isinstance(segments, list):
        return False
    return any(isinstance(row, dict) and isinstance((row.get("metadata") or {}).get("words"), list) and (row.get("metadata") or {}).get("words") for row in segments)


def _next_actions(run: dict[str, Any], *, execute: bool) -> list[str]:
    if not execute:
        return [
            "Run again with --execute only when word-level timestamps or speaker labels are needed.",
            "Keep SenseVoice/FunASR as primary ASR; use this output as alignment evidence.",
        ]
    if run.get("status") == "ok":
        return [
            "Use timeline-alignment-audit to compare review_start, ASR starts, frame times, and WhisperX alignment.",
            "Feed alignment evidence into review UI; do not auto-replace corrected transcript.",
        ]
    return [
        "Inspect the ASR run report and install/configure WhisperX if command_not_found or module missing.",
        "Keep using SenseVoice/FunASR transcript while alignment is unavailable.",
    ]


def _write_outputs(root: Path, result: dict[str, Any]) -> None:
    paths = ensure_project_dirs(root)
    json_path = paths["transcripts"] / "whisperx-alignment-run.json"
    md_path = paths["notes"] / "whisperx-alignment-run.md"
    write_json(json_path, result)
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    args_path = root / "mcp-run-whisperx-alignment.args.json"
    write_json(
        args_path,
        {
            "root": str(root),
            "media_path": result.get("media_path", ""),
            "language": result.get("language", "zh"),
            "model": result.get("model", "large-v3"),
            "execute": False,
            "timeout_seconds": 1800,
            "write": True,
        },
    )
    result["json_path"] = str(json_path)
    result["markdown_path"] = str(md_path)
    result["mcp_args_path"] = str(args_path)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict):
            manifest["whisperx_alignment_run_json"] = str(json_path.relative_to(root)) if _is_relative_to(json_path, root) else str(json_path)
            manifest["whisperx_alignment_report"] = str(md_path.relative_to(root)) if _is_relative_to(md_path, root) else str(md_path)
            if result.get("alignment_transcript_json"):
                alignment_path = Path(str(result["alignment_transcript_json"]))
                manifest["whisperx_alignment_transcript_json"] = str(alignment_path.relative_to(root)) if _is_relative_to(alignment_path, root) else str(alignment_path)
            write_json(manifest_path, manifest)
        result["run_registry"] = register_bundle_run(
            root,
            run_type="whisperx_alignment",
            run_id="whisperx-alignment",
            status="completed" if result.get("status") == "alignment_ready" else "needs_execution" if result.get("status") == "preview" else "needs_retry",
            title="WhisperX alignment evidence",
            summary=f"status={result.get('status', '')}; execute={bool(result.get('execute'))}; does not replace primary ASR.",
            inputs={"media_path": result.get("media_path", ""), "plan_path": result.get("plan_path", "")},
            parameters={"language": result.get("language", "zh"), "model": result.get("model", "large-v3")},
            artifacts=[
                {"key": "alignment_run_json", "path": str(json_path)},
                {"key": "alignment_report", "path": str(md_path)},
                {"key": "alignment_transcript_json", "path": result.get("alignment_transcript_json", "")},
                {"key": "alignment_transcript_srt", "path": result.get("alignment_transcript_srt", "")},
            ],
            failed_items=[] if result.get("status") in {"alignment_ready", "preview"} else [{"item": "whisperx_alignment", "reason": result.get("status", ""), "detail": result.get("asr_run_status", "")}],
            retry_command=f".\\scripts\\video-knowledge.ps1 run-whisperx-alignment '{root}' '{result.get('media_path', '')}' --execute",
            next_actions=result.get("next_actions", []),
            operator_boundary=result.get("operator_boundary", {}),
            write=True,
        )


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# WhisperX Alignment Run",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Execute: `{bool(result.get('execute'))}`",
        f"- Media: `{result.get('media_path', '')}`",
        f"- Plan: `{result.get('plan_path', '')}`",
        f"- Raw output: `{result.get('raw_output_json', '')}`",
        f"- Alignment transcript: `{result.get('alignment_transcript_json', '')}`",
        f"- Word-level alignment: `{bool(result.get('word_level_alignment'))}`",
        "",
        "## Boundary",
        "",
        "WhisperX is used only for timestamp/speaker alignment evidence. It does not replace SenseVoice/FunASR or corrected-transcript.json.",
        "",
        "## Next Actions",
        "",
    ]
    for action in result.get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
