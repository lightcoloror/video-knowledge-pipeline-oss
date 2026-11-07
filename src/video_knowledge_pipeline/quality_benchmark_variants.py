from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .asr_execution import run_asr_plan
from .asr_runner import plan_asr_run
from .entity_lexicon import build_entity_lexicon
from .models import now_iso
from .storage import read_json, write_json
from .transcript import parse_transcript


SCHEMA = "video_knowledge_pipeline.quality_benchmark_variant_run.v1"
VARIANT_PRESETS: dict[str, dict[str, Any]] = {
    "sensevoice_raw": {"preset": "sensevoice", "punc_model": ""},
    "sensevoice_full_punc": {"preset": "sensevoice", "punc_model": "ct-punc"},
    "contextual_paraformer_no_hotword": {"preset": "contextual-paraformer", "punc_model": "ct-punc", "hotword_mode": "none"},
    "contextual_paraformer_hotword": {"preset": "contextual-paraformer", "punc_model": "ct-punc", "hotword_mode": "entity_evidence"},
    "qwen3_asr_1_7b": {"preset": "qwen3-asr-1.7b"},
    "qwen3_asr_0_6b": {"preset": "qwen3-asr-0.6b"},
    "fun_asr_nano": {"preset": "fun-asr-nano"},
}
DEFAULT_VARIANTS = ("sensevoice_raw", "sensevoice_full_punc", "qwen3_asr_1_7b")


def execute_quality_benchmark_variants(
    manifest_json: str | Path,
    *,
    variants: list[str] | None = None,
    execute: bool = False,
    resume: bool = True,
    retry_failed: bool = False,
    limit: int = 0,
    timeout_seconds: int = 1800,
    write: bool = True,
    plan_builder: Callable[..., dict[str, Any]] | None = None,
    plan_executor: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path = Path(manifest_json).expanduser().resolve()
    if execute and not write:
        raise ValueError("execute-variants requires write=true so plans, logs, and transcript paths remain auditable")
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("quality benchmark manifest must be a JSON object")
    selected = _selected_variants(variants)
    build_plan = plan_builder or plan_asr_run
    execute_plan = plan_executor or run_asr_plan
    run_root = path.parent / "variant-runs"
    rows: list[dict[str, Any]] = []
    attempted = 0
    samples = [sample for sample in manifest.get("samples") or [] if isinstance(sample, dict)]
    for sample in samples:
        sample_id = str(sample.get("sample_id") or f"sample-{len(rows) + 1}")
        sample_runs = sample.setdefault("variant_runs", {})
        sample_variants = sample.setdefault("variants", {})
        for variant in selected:
            if limit > 0 and attempted >= limit:
                break
            existing = _transcript_ready(sample_variants.get(variant))
            previous = sample_runs.get(variant) if isinstance(sample_runs.get(variant), dict) else {}
            if resume and existing:
                rows.append(_row(sample_id, variant, "skipped_existing", transcript_path=str(sample_variants.get(variant) or "")))
                continue
            if resume and previous.get("status") in {"failed", "blocked", "timeout"} and not retry_failed:
                rows.append(_row(sample_id, variant, "skipped_previous_failure", error=str(previous.get("error") or "")))
                continue
            clip = Path(str(sample.get("audio_clip_path") or "")).expanduser()
            if not clip.exists():
                row = _row(sample_id, variant, "blocked", error="audio_clip_missing")
                sample_runs[variant] = row
                rows.append(row)
                attempted += 1
                continue
            attempted += 1
            if not write and not execute:
                rows.append(_row(sample_id, variant, "planned_no_write"))
                continue
            workspace = run_root / _safe_name(sample_id) / variant
            profile = VARIANT_PRESETS[variant]
            kwargs: dict[str, Any] = {
                "preset": profile["preset"],
                # Intent: compare dialect models on the actual source language.
                # Decision: consume the source-bound sample language written by
                # quality-benchmark build, retaining zh for legacy manifests.
                # Reason: forcing Chinese on a pure Cantonese sample invalidates
                # the Qwen/SenseVoice comparison before model quality is tested.
                # Evidence: both upstream adapters expose explicit yue/Cantonese
                # selection and the interview run was already pinned to yue.
                # Effective scope: local benchmark variants only.
                "language": str(sample.get("asr_language") or "zh").strip(),
            }
            if "punc_model" in profile:
                kwargs["punc_model"] = profile["punc_model"]
            if profile.get("hotword_mode") == "entity_evidence":
                kwargs["hotword"] = _entity_hotword_for_sample(sample)
            elif profile.get("hotword_mode") == "none":
                kwargs["hotword"] = ""
            if variant in {"qwen3_asr_1_7b", "qwen3_asr_0_6b"}:
                kwargs["qwen_timestamps"] = False
                kwargs["hotword"] = _entity_hotword_for_sample(sample)
            try:
                plan = build_plan(workspace, clip, **kwargs)
                plan_path = str(plan.get("plan_path") or "")
                if not execute:
                    row = _row(
                        sample_id,
                        variant,
                        "planned",
                        plan_path=plan_path,
                        device=str(plan.get("local_asr_device") or ""),
                        model_ready=plan.get("model_ready") or {},
                    )
                else:
                    run = execute_plan(plan_path, execute=True, timeout_seconds=timeout_seconds)
                    transcript_path = _normalized_path(run)
                    status = "completed" if str(run.get("status") or "") == "ok" and _transcript_ready(transcript_path) else str(run.get("status") or "failed")
                    if status == "completed":
                        transcript_path = _offset_clip_transcript(
                            transcript_path,
                            workspace / "benchmark-normalized-transcript.json",
                            offset_seconds=float(sample.get("start_seconds") or 0.0),
                        )
                    row = _row(
                        sample_id,
                        variant,
                        status,
                        plan_path=plan_path,
                        transcript_path=transcript_path,
                        device=str(plan.get("local_asr_device") or ""),
                        model_ready=plan.get("model_ready") or {},
                        error=_run_error(run) if status != "completed" else "",
                    )
                    if status == "completed":
                        sample_variants[variant] = transcript_path
                        if variant == "sensevoice_full_punc" and str(sample.get("human_review_status") or "") != "completed":
                            draft_text = _transcript_text(transcript_path)
                            if draft_text:
                                sample["asr_draft_text"] = draft_text
                                sample["asr_draft_source"] = transcript_path
                                sample["draft_source"] = "sample_clip_sensevoice_full_punc"
                sample_runs[variant] = row
                rows.append(row)
            except Exception as exc:
                row = _row(sample_id, variant, "failed", error=f"{type(exc).__name__}: {exc}")
                sample_runs[variant] = row
                rows.append(row)
            finally:
                if write:
                    manifest["variant_execution_checkpoint"] = {
                        "sample_id": sample_id,
                        "variant": variant,
                        "attempted_count": attempted,
                        "updated_at": now_iso(),
                    }
                    write_json(path, manifest)
        if limit > 0 and attempted >= limit:
            break
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    result = {
        "schema": SCHEMA,
        "manifest_json": str(path),
        "execute": bool(execute),
        "resume": bool(resume),
        "retry_failed": bool(retry_failed),
        "variants": selected,
        "sample_count": len(samples),
        "attempted_count": attempted,
        "status_counts": counts,
        "status": _overall_status(rows, execute=execute),
        "runs": rows,
        "operator_boundary": {
            "local_only": True,
            "no_cloud_asr": True,
            "does_not_switch_default_model": True,
            "qwen_0_6b_fallback_is_explicit": True,
            "preview_does_not_run_models": True,
            "no_write_preview_does_not_create_plans": True,
        },
        "next_actions": _next_actions(rows, execute=execute, manifest_path=path),
        "updated_at": now_iso(),
    }
    if write:
        manifest["variant_execution"] = {
            "status": result["status"],
            "variants": selected,
            "status_counts": counts,
            "updated_at": result["updated_at"],
            "report_json": "quality-benchmark-variant-run.json",
            "report_markdown": "quality-benchmark-variant-run.md",
        }
        write_json(path, manifest)
        write_json(path.parent / "quality-benchmark-variant-run.json", result)
        (path.parent / "quality-benchmark-variant-run.md").write_text(_render_markdown(result), encoding="utf-8")
        from .quality_benchmark import _render_review_html
        (path.parent / "quality-benchmark-review.html").write_text(_render_review_html(manifest), encoding="utf-8")
    return result


def _selected_variants(values: list[str] | None) -> list[str]:
    requested = [str(value).strip() for value in (values or DEFAULT_VARIANTS) if str(value).strip()]
    unknown = [value for value in requested if value not in VARIANT_PRESETS]
    if unknown:
        raise ValueError(f"unsupported quality benchmark variants: {', '.join(unknown)}")
    return list(dict.fromkeys(requested))


def _normalized_path(run: dict[str, Any]) -> str:
    normalized = run.get("normalized") if isinstance(run.get("normalized"), dict) else {}
    for key in ("json_path", "normalized_transcript_json", "transcript_json"):
        value = str(normalized.get(key) or "").strip()
        if value:
            return str(Path(value).expanduser().resolve())
    return ""


def _transcript_text(value: Any) -> str:
    path = Path(str(value or "")).expanduser()
    if not path.exists():
        return ""
    try:
        return "\n".join(
            str(cue.text or "").strip()
            for cue in parse_transcript(path)
            if str(cue.text or "").strip()
        ).strip()
    except Exception:
        return ""

def _entity_hotword_for_sample(sample: dict[str, Any]) -> str:
    explicit = sample.get("context_hotwords")
    if isinstance(explicit, list):
        return " ".join(dict.fromkeys(str(value).strip() for value in explicit if str(value).strip()))
    if str(explicit or "").strip():
        return str(explicit).strip()
    bundle = Path(str(sample.get("bundle_dir") or "")).expanduser()
    if not bundle.exists():
        return ""
    lexicon = build_entity_lexicon(bundle, write=False)
    return str(lexicon.get("hotword_text") or "").strip()

def _transcript_ready(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    path = Path(text).expanduser()
    if not path.exists():
        return False
    try:
        return any(str(cue.text or "").strip() for cue in parse_transcript(path))
    except Exception:
        return False


def _run_error(run: dict[str, Any]) -> str:
    raw_path = str(run.get("raw_output_json") or "").strip()
    if raw_path:
        payload = read_json(Path(raw_path))
        if isinstance(payload, dict):
            code = str(payload.get("error_code") or "").strip()
            message = str(payload.get("error") or "").strip()
            if code or message:
                return ": ".join(value for value in (code, message) if value)[:2000]
    stderr = str(run.get("stderr") or "").strip()
    stdout = str(run.get("stdout") or "").strip()
    return (stderr or stdout or str(run.get("status") or "ASR execution failed"))[-2000:]


def _offset_clip_transcript(
    transcript_path: str,
    output_path: Path,
    *,
    offset_seconds: float,
) -> str:
    """Convert clip-relative ASR timestamps back to the source-video timeline."""

    path = Path(transcript_path).expanduser().resolve()
    payload = read_json(path)
    if not isinstance(payload, dict) or offset_seconds <= 0:
        return str(path)
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return str(path)
    shifted = dict(payload)
    shifted_segments: list[Any] = []
    for row in segments:
        if not isinstance(row, dict):
            shifted_segments.append(row)
            continue
        item = dict(row)
        original_start = float(item.get("start") or 0.0)
        original_end = float(item.get("end") or original_start)
        item["start"] = round(original_start + offset_seconds, 6)
        item["end"] = round(original_end + offset_seconds, 6)
        if isinstance(item.get("words"), list):
            item["words"] = [
                {
                    **word,
                    "start": round(float(word.get("start") or 0.0) + offset_seconds, 6),
                    "end": round(float(word.get("end") or word.get("start") or 0.0) + offset_seconds, 6),
                }
                if isinstance(word, dict)
                else word
                for word in item["words"]
            ]
        shifted_segments.append(item)
    shifted["segments"] = shifted_segments
    shifted["benchmark_time_offset_seconds"] = offset_seconds
    shifted["source_transcript_path"] = str(path)
    write_json(output_path, shifted)
    return str(output_path)


def _row(
    sample_id: str,
    variant: str,
    status: str,
    *,
    plan_path: str = "",
    transcript_path: str = "",
    device: str = "",
    model_ready: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "variant": variant,
        "preset": VARIANT_PRESETS[variant]["preset"],
        "status": status,
        "plan_path": plan_path,
        "transcript_path": transcript_path,
        "device": device,
        "model_ready": model_ready or {},
        "error": error,
        "updated_at": now_iso(),
    }


def _overall_status(rows: list[dict[str, Any]], *, execute: bool) -> str:
    if not rows:
        return "no_runs"
    statuses = {str(row.get("status") or "") for row in rows}
    if not execute:
        return "planned"
    if statuses <= {"completed", "skipped_existing"}:
        return "completed"
    if "completed" in statuses or "skipped_existing" in statuses:
        return "partially_completed"
    return "blocked_or_failed"


def _next_actions(rows: list[dict[str, Any]], *, execute: bool, manifest_path: Path) -> list[str]:
    if not execute:
        return [
            f".\\scripts\\video-knowledge.ps1 quality-benchmark execute-variants '{manifest_path}' --execute --resume",
        ]
    failed = [row for row in rows if row.get("status") in {"failed", "blocked", "timeout", "asr_model_not_ready"}]
    actions = []
    if failed:
        actions.append(
            f".\\scripts\\video-knowledge.ps1 quality-benchmark execute-variants '{manifest_path}' --execute --resume --retry-failed"
        )
    actions.append(f".\\scripts\\video-knowledge.ps1 quality-benchmark run '{manifest_path}'")
    return actions


def _safe_name(value: str) -> str:
    text = "".join(char if char.isalnum() or char in "-_" else "-" for char in value).strip("-")
    return text[:100] or "sample"


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Quality Benchmark Variant Run",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Execute: `{result.get('execute')}`",
        f"- Attempts: `{result.get('attempted_count')}`",
        f"- Variants: `{', '.join(result.get('variants') or [])}`",
        "",
        "| Sample | Variant | Status | Device | Transcript | Error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("runs") or []:
        lines.append(
            f"| {row.get('sample_id', '')} | `{row.get('variant', '')}` | `{row.get('status', '')}` | "
            f"`{row.get('device', '')}` | `{row.get('transcript_path', '')}` | {str(row.get('error') or '').replace('|', '/')} |"
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- `{action}`" for action in result.get("next_actions") or [])
    return "\n".join(lines).rstrip() + "\n"
