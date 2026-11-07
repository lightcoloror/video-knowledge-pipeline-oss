"""Resumable local ASR orchestration. Recognition outputs remain independent."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .asr_consensus import build_asr_consensus
from .asr_execution import run_asr_plan
from .asr_runner import plan_asr_run, _local_qwen_aligner_path, _model_ready
from .asr_review_audit import build_review_audit
from .asr_comparison_alignment import critical_differences
from .audio_silence_probe import probe_audio_silence
from .file_hash import sha256_file
from .interval_coverage import merge_intervals
from .local_targeted_asr_execution import run_local_targeted_asr_evidence
from .storage import bundle_write_lock, read_json, write_json
from .transcript import parse_transcript
from .video import probe_video
from .qwen_failed_window_recovery import plan_qwen_failed_window_recovery, run_qwen_failed_window_recovery


def _path(root: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    p = Path(value).expanduser()
    p = p if p.is_absolute() else root / p
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.resolve()


def daily_windows(cues: list[Any], risks: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    """Include factual risks, independently uncovered audio and uniform samples."""
    ranges = [(r["start"], r["end"]) for r in risks]
    ranges += [(c.start, c.end) for c in cues if set(critical_differences(c.text, "")) - {"lexical_change"}]
    if cues:
        ranges += [(cues[i].start, cues[i].end) for i in sorted({0, len(cues) // 2, len(cues) - 1})]
    windows = []
    duration = max([c.end for c in cues] + [b for _, b in ranges] + [0])
    for start, end in merge_intervals((max(0, a - 2), min(duration, b + 2)) for a, b in ranges):
        while start < end:
            finish = min(end, start + 30)
            windows.append({"retry_id": f"dual-risk-{len(windows):04d}", "start": start, "end": finish,
                            "reasons": ["factual_or_independent_audio_risk_or_uniform_sample"]})
            start = finish
    return {"windows": windows[:limit], "remaining_windows": windows[limit:],
            "total_window_count": len(windows), "selection_is_exhaustive": False}


def run_dual_asr_workflow(bundle_dir: str | Path, *, media_path: str | Path | None = None,
                          primary_transcript: str | Path | None = None,
                          secondary_transcript: str | Path | None = None,
                          mode: str = "important", primary_preset: str = "sensevoice",
                          secondary_preset: str = "qwen3-asr-1.7b",
                          secondary_timestamps: str = "auto",
                          speech_activity_json: str | Path | None = None,
                          diarization_json: str | Path | None = None,
                          execute: bool = False, timeout_seconds: int = 1800,
                          max_windows: int = 24, allow_cpu: bool = False,
                          recover_failed_windows: bool = True,
                          recovery_window_seconds: int = 60,
                          recovery_max_windows: int = 24,
                          write: bool = True) -> dict[str, Any]:
    if mode not in {"daily", "important"}:
        raise ValueError("mode must be daily or important")
    if secondary_timestamps not in {"auto", "required", "window"}:
        raise ValueError("secondary_timestamps must be auto, required or window")
    if execute and not write:
        raise ValueError("execute requires durable write=True")
    if max_windows < 1 or timeout_seconds <= 0:
        raise ValueError("max_windows and timeout_seconds must be positive")
    if (type(recover_failed_windows) is not bool
            or type(recovery_window_seconds) is not int or recovery_window_seconds < 1
            or type(recovery_max_windows) is not int or recovery_max_windows < 1):
        raise ValueError('invalid failed-window recovery policy')
    # This entry point never selects a cloud model or silently downgrades size.
    if primary_preset != "sensevoice" or secondary_preset not in {"qwen3-asr-1.7b", "qwen3-asr-0.6b"}:
        raise ValueError("local workflow supports SenseVoice + an explicitly selected Qwen3-ASR preset")
    root = Path(bundle_dir).expanduser().resolve()
    manifest = read_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    prior_review = read_json(root / 'dual-asr-review.json') if (root / 'dual-asr-review.json').is_file() else {}
    media = _path(root, media_path or manifest.get("media_path") or manifest.get("source_path"))
    primary = _path(root, primary_transcript)
    secondary = _path(root, secondary_transcript)
    activity = _path(root, speech_activity_json)
    diarization = _path(root, diarization_json)
    if primary and secondary and primary == secondary:
        raise ValueError("two independent transcripts required; the same file is not a second ASR")
    aligner = _local_qwen_aligner_path() or "Qwen/Qwen3-ForcedAligner-0.6B"
    aligner_ready = bool(_model_ready(preset="qwen3-forced-aligner", model=aligner).get("ready")) if secondary_timestamps == "auto" else secondary_timestamps == "required"
    timestamps = {"requested": secondary_timestamps, "use_forced_aligner": aligner_ready,
                  "precision": "forced_alignment" if aligner_ready else "audio_window_only"}
    inputs = {"mode": mode, "primary_preset": primary_preset, "secondary_preset": secondary_preset,
              "secondary_timestamps": timestamps,
              "max_windows": max_windows, "allow_cpu": allow_cpu, "workflow_version": 2}
    inputs['recovery_policy'] = {'enabled': recover_failed_windows,
        'window_seconds': recovery_window_seconds, 'max_windows': recovery_max_windows}
    for name, path in (("media", media), ("primary", primary), ("secondary", secondary),
                       ("activity", activity), ("diarization", diarization)):
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)} if path else None
    # A higher daily review budget reuses the same underlying ASR jobs.
    fingerprint = hashlib.sha256(json.dumps({k: v for k, v in inputs.items() if k not in {"max_windows", 'recovery_policy'}}, sort_keys=True).encode()).hexdigest()
    workspace = root / "dual-asr-runs" / fingerprint[:20]
    result: dict[str, Any] = {"schema": "video_knowledge_pipeline.dual_asr_workflow.v1",
        "mode": mode, "status": "planned", "ok": False, "execute": execute, "inputs": inputs,
        "execution_id": fingerprint, "primary_transcript": str(primary or ""),
        "secondary_transcript": str(secondary or ""), "stages": {},
        "dual_full_transcription_complete": False, "quality_verified": False,
        "model_switch_allowed": False, "automatic_text_promotion_allowed": False,
        "local_only": True, "provenance": "supplied_transcripts_not_execution_attested" if primary or secondary else "local_runner",
        "resume_argv": ["dual-asr-review", str(root), "--mode", mode, "--execute",
                        "--secondary-preset", secondary_preset, "--secondary-timestamps", secondary_timestamps,
                        "--max-windows", str(max_windows)]}
    for flag, path in (("--media-path", media), ("--primary-transcript", primary),
                       ("--secondary-transcript", secondary), ("--speech-activity-json", activity),
                       ("--diarization-json", diarization)):
        if path:
            result["resume_argv"].extend([flag, str(path)])
    if allow_cpu:
        result["resume_argv"].append("--allow-cpu")
    if not recover_failed_windows:
        result['resume_argv'].append('--no-recover-failed-windows')
    result['resume_argv'].extend(['--recovery-window-seconds', str(recovery_window_seconds),
                                  '--recovery-max-windows', str(recovery_max_windows)])

    def save() -> dict[str, Any]:
        if write:
            write_json(workspace / "state.json", result)
            write_json(root / "dual-asr-review.json", result)
        return result

    def full_run(role: str, supplied: Path | None, preset: str) -> Path | None:
        if supplied:
            if not parse_transcript(supplied):
                raise ValueError(f"empty {role} transcript")
            result["stages"][role] = {"status": "supplied", "path": str(supplied), "sha256": sha256_file(supplied)}
            return supplied
        cached = previous.get("stages", {}).get(role, {})
        path = Path(cached["path"]) if cached.get("path") else None
        if path and path.is_file() and cached.get("sha256") == sha256_file(path) and cached.get("status") in {"completed", "composed_reference"}:
            result["stages"][role] = {**cached, "reused": True}
            return path
        if not execute:
            result["stages"][role] = {"status": "pending", "preset": preset}
            return None
        if not media:
            raise ValueError("local media is required for execution")
        result["stages"][role] = {"status": "running", "preset": preset}
        save()
        started = time.perf_counter()
        plan = plan_asr_run(workspace / role, media, preset=preset,
                            **({"qwen_timestamps": aligner_ready} if role == "secondary" else {}))
        run = run_asr_plan(plan["plan_path"], execute=True, normalize=True, resume=True, timeout_seconds=timeout_seconds)
        if role == 'secondary' and recover_failed_windows and run.get('status') in {'degraded', 'failed'}:
            raw_path = Path(run.get('raw_output_json') or run.get('expected_output_json') or '')
            raw = read_json(raw_path) if raw_path.is_file() else {}
            failures = raw.get('failed_chunks') or []
            if failures and any(row.get('retry_exhausted') is not True for row in failures):
                # Reuse the original checkpoint once; the child enforces its
                # persisted per-window attempt ceiling, including healthy reuse.
                run = run_asr_plan(plan['plan_path'], execute=True, normalize=True,
                    resume=True, timeout_seconds=timeout_seconds)
                raw_path = Path(run.get('raw_output_json') or run.get('expected_output_json') or '')
                raw = read_json(raw_path) if raw_path.is_file() else {}
            failures = raw.get('failed_chunks') or []
            if failures and all(row.get('retry_exhausted') is True for row in failures):
                recovery_plan = plan_qwen_failed_window_recovery(raw_path,
                    output_dir=workspace / role / 'failed-window-recovery' /
                        f"{recovery_window_seconds}s-{recovery_max_windows}-{sha256_file(raw_path)[:12]}",
                    asr_plan_json=plan['plan_path'], media_path=media,
                    speech_activity_json=activity,
                    window_seconds=recovery_window_seconds, max_windows=recovery_max_windows)
                recovered = run_qwen_failed_window_recovery(recovery_plan, execute=True,
                    timeout_seconds=timeout_seconds)
                result['stages']['secondary_recovery'] = recovered
                if recovered.get('status') == 'completed_composed_reference':
                    path = Path(recovered['path']).resolve()
                    result['stages'][role] = {'status': 'composed_reference', 'path': str(path),
                        'sha256': sha256_file(path), 'preset': preset,
                        'plan_path': plan['plan_path'], 'run_status': recovered['status'],
                        'single_run_execution_attested': False, 'timestamp_capability': timestamps,
                        'elapsed_seconds': round(time.perf_counter()-started, 3)}
                    save()
                    return path
        output = (run.get("normalized") or {}).get("json_path")
        if run.get("status") != "ok" or not output or not Path(output).is_file() or not parse_transcript(output):
            result["stages"][role] = {"status": "failed", "run_status": run.get("status"), "plan_path": plan["plan_path"]}
            return None
        path = Path(output).resolve()
        result["stages"][role] = {"status": "completed", "path": str(path), "sha256": sha256_file(path), "preset": preset,
                                 "plan_path": plan["plan_path"], "run_status": run["status"]}
        result["stages"][role]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if role == "secondary":
            result["stages"][role]["timestamp_capability"] = timestamps
        save()
        return path

    # Keep simultaneous invocations from running the same GPU jobs twice.
    if write:
        workspace.mkdir(parents=True, exist_ok=True)
    from contextlib import nullcontext
    with bundle_write_lock(workspace, operation="dual_asr_review", timeout_seconds=1) if write else nullcontext():
        previous = read_json(workspace / "state.json") if (workspace / "state.json").exists() else {}
        stage = "primary"
        try:
            primary = full_run("primary", primary, primary_preset)
            result["primary_transcript"] = str(primary or "")
            if not primary:
                result["status"] = "primary_failed" if execute else "primary_pending"
                return save()
            result["status"] = "single_model_completed"
            save()
            stage = "audit"
            if not activity and media:
                # Recover existing audit sidecars independently of ASR execution
                # identity. A non-silent scan remains a non-silent scan.
                candidates = [previous.get('speech_activity_json')]
                candidates.append(prior_review.get('speech_activity_json'))
                if secondary and secondary.is_file():
                    supplied_payload = read_json(secondary) if secondary.suffix.lower() == '.json' else {}
                    if isinstance(supplied_payload, dict):
                        candidates.append(supplied_payload.get('speech_activity_json'))
                for candidate in candidates:
                    if not candidate or not Path(candidate).is_file():
                        continue
                    try:
                        checked = build_review_audit([], [], [], media=media, speech_activity_json=candidate)
                        if checked['coverage'].get('binding', {}).get('media_binding') == 'verified':
                            activity = Path(candidate).resolve()
                            result['activity_reused'] = True
                            break
                    except (OSError, ValueError, KeyError) as exc:
                        result.setdefault('activity_reuse_rejections', []).append(str(exc))
            if execute and media and not activity:
                try:
                    duration = probe_video(media).duration_seconds
                    probe = probe_audio_silence(media, start=0, end=duration, timeout_seconds=timeout_seconds)
                    if not probe.get("ok"):
                        raise ValueError(probe.get("error", "audio activity probe failed"))
                    activity = workspace / "audio-activity.json"
                    write_json(activity, {"source_media_sha256": inputs["media"]["sha256"],
                        "speech_intervals": probe["activity_intervals"], "detector": "ffmpeg_silencedetect",
                        "evidence_kind": "non_silent_audio_not_proven_speech"})
                except (OSError, ValueError, RuntimeError) as exc:
                    result["activity_probe_error"] = str(exc)
            cues = parse_transcript(primary)
            audit = build_review_audit(cues, [], [], media=media, speech_activity_json=activity, diarization_json=diarization)
            result["speech_activity_json"] = str(activity or "")
            result['audit_inputs'] = {'activity': {'path': str(activity), 'sha256': sha256_file(activity)} if activity else None}
            result["primary_review_audit"] = audit
            result["review_audit"] = audit
            stage = "secondary"
            if mode == "important" or secondary:
                secondary = full_run("secondary", secondary, secondary_preset)
            else:
                selection = daily_windows(cues, audit["independent_risk_windows"], limit=max_windows)
                result["targeted_selection"] = selection
                if selection["remaining_windows"]:
                    position = result["resume_argv"].index("--max-windows") + 1
                    result["resume_argv"][position] = str(selection["total_window_count"])
                    result["next_action"] = "Resume with the expanded saved budget to include the remaining risk windows; existing local ASR jobs are reusable."
                if execute and selection["windows"]:
                    if not media:
                        raise ValueError("local media is required for targeted ASR")
                    target_root = workspace / "targeted"
                    write_json(target_root / "manifest.json", {"media_path": str(media)})
                    plan_path = target_root / "selection.json"
                    write_json(plan_path, {"retry_plan": selection, "source_media": inputs["media"]})
                    run = run_local_targeted_asr_evidence(target_root, media_path=media, input_plan=plan_path,
                        preset=secondary_preset, execute=True, timeout_seconds=timeout_seconds, allow_cpu=allow_cpu,
                        qwen_timestamps=aligner_ready)
                    result["stages"]["secondary"] = {"status": run["status"], "artifacts": run.get("artifacts", {})}
                    if run.get("status") == "completed":
                        secondary = Path(run["artifacts"]["evidence_json"])
            result["secondary_transcript"] = str(secondary or "")
            if not secondary:
                result["status"] = "secondary_failed" if execute else "secondary_pending"
                return save()
            stage = "comparison"
            consensus = build_asr_consensus(root, primary_transcript=primary, secondary_transcript=secondary,
                media_path=media, speech_activity_json=activity, diarization_json=diarization, write=write,
                secondary_scope="targeted" if mode == "daily" else "full")
            result["consensus"] = {k: consensus[k] for k in ("status", "counts", "review_audit", "review_required_count")}
            # Keep daily selection's single-model risks above, but publish the
            # audit that actually includes the completed secondary comparison.
            result["review_audit"] = consensus["review_audit"]
            attested = all(result["stages"].get(r, {}).get("status") == "completed" for r in ("primary", "secondary"))
            result["dual_full_transcription_complete"] = mode == "important" and attested
            result["status"] = ("dual_model_completed" if attested else "dual_inputs_compared") if mode == "important" else "targeted_review_completed"
            if result.get("targeted_selection", {}).get("remaining_windows"):
                result["status"] = "targeted_review_partial"
            result["ok"] = result["status"] != "targeted_review_partial"
            return save()
        except (OSError, ValueError, RuntimeError) as exc:
            result.update(status=f"{stage}_failed", error=str(exc), ok=False)
            return save()
