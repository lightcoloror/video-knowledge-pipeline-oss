from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .asr_runner import _resolve_python_executable, default_local_asr_device
from .media_tools import resolve_media_tool
from .models import now_iso
from .storage import ensure_project_dirs, write_json

SCHEMA = "video_knowledge_pipeline.asr_ab_sample_plan.v1"


def plan_asr_ab_sample(
    workspace_dir: str | Path,
    media_path: str | Path,
    *,
    sample_start_seconds: float = 0.0,
    duration_seconds: float = 300.0,
    language: str = "zh",
    cloud_provider_config: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Plan a 5-minute ASR A/B sample without uploading or running heavy jobs."""

    root = Path(workspace_dir).expanduser().resolve()
    paths = ensure_project_dirs(root)
    media = Path(media_path).expanduser().resolve()
    if not media.exists():
        raise FileNotFoundError(f"media not found: {media}")
    sample_dir = paths["transcripts"] / "asr-ab-sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    media_is_existing_sample = media.parent == sample_dir and ".sample-" in media.stem
    sample_media = media if media_is_existing_sample else sample_dir / f"{media.stem}.sample-{int(sample_start_seconds)}-{int(duration_seconds)}{media.suffix or '.mp4'}"
    ffmpeg = resolve_media_tool("ffmpeg") or "ffmpeg"
    extract_command = [] if media_is_existing_sample else [
        ffmpeg,
        "-y",
        "-ss",
        str(float(sample_start_seconds or 0.0)),
        "-t",
        str(float(duration_seconds or 300.0)),
        "-i",
        str(media),
        "-c",
        "copy",
        str(sample_media),
    ]
    variants: list[dict[str, Any]] = []
    variants.append(_local_sensevoice_variant("sensevoice_basic", sample_media, sample_dir, language=language, punc_model="", use_itn=False, merge_vad=True))
    variants.append(_local_sensevoice_variant("sensevoice_full_punc", sample_media, sample_dir, language=language, punc_model="ct-punc", use_itn=True, merge_vad=True))
    variants.append(
        _local_sensevoice_variant(
            "sensevoice_full_punc_campp",
            sample_media,
            sample_dir,
            language=language,
            punc_model="ct-punc",
            spk_model="cam++",
            device="cuda",
            use_itn=True,
            merge_vad=True,
        )
    )
    variants.append(
        _local_sensevoice_variant(
            "sensevoice_full_punc_campp_oracle_2",
            sample_media,
            sample_dir,
            language=language,
            punc_model="ct-punc",
            spk_model="cam++",
            device="cuda",
            preset_speaker_count=2,
            evaluation_only_known_speaker_count=True,
            use_itn=True,
            merge_vad=True,
        )
    )
    variants.append(
        _moss_transcribe_diarize_variant(
            "moss_transcribe_diarize",
            sample_media,
            sample_dir,
            language=language,
        )
    )
    variants.append(_dolphin_variant("dolphin", sample_media, sample_dir, language=language))
    variants.append(_whisperx_alignment_variant("whisperx_alignment", sample_media, sample_dir, language=language))
    variants.append(_cloud_asr_variant("openai_cloud_asr", sample_media, sample_dir, language=language, provider_config=cloud_provider_config or {}))
    result = {
        "schema": SCHEMA,
        "workspace_dir": str(root),
        "media_path": str(media),
        "sample_media_path": str(sample_media),
        "sample_start_seconds": float(sample_start_seconds or 0.0),
        "duration_seconds": float(duration_seconds or 300.0),
        "sample_extract_command": extract_command,
        "sample_extract_powershell": _powershell_join(extract_command) if extract_command else "",
        "sample_reused_existing": media_is_existing_sample,
        "variants": variants,
        "status": "planned",
        "ok": True,
        "operator_boundary": {
            "preview_only": True,
            "does_not_upload_audio": True,
            "cloud_asr_requires_run_cloud_asr_plan_execute": True,
            "dolphin_requires_local_install": True,
            "whisperx_alignment_is_timestamp_evidence_only": True,
            "campp_requires_prepared_local_model": True,
            "campp_variant_is_gpu_only": True,
            "campp_oracle_count_is_evaluation_only": True,
            "moss_requires_explicit_local_runtime_and_model": True,
            "moss_never_falls_back_to_another_asr": True,
            "same_sample_required_for_fair_comparison": True,
        },
        "next_actions": [
            "Extract the sample media with sample_extract_command if it is not already an existing sample.",
            "Run sensevoice_basic and sensevoice_full_punc locally first.",
            "After CAM++ is explicitly prepared, run sensevoice_full_punc_campp on the same sample with CUDA; speaker labels remain anonymous candidate evidence.",
            "For a reference window already known to contain two speakers, run sensevoice_full_punc_campp_oracle_2 only as a diagnostic ceiling; never promote it as an automatic production route.",
            "After the pinned MOSS runtime and local model are explicitly prepared, run moss_transcribe_diarize on the exact same sample; evaluate its anonymous speaker output with the existing DER/cpCER/tcpCER gates.",
            "Only run cloud ASR after explicit approval for this sample.",
            "Run Dolphin only after the local package/model is available; use it as second evidence, not as default promotion.",
            "Run WhisperX only when word-level timestamps/speaker alignment are needed; it is alignment evidence, not a default ASR replacement.",
            "Compare WER-like term errors, punctuation/readability, timestamps, and speaker labels before adding a second ASR as default.",
        ],
        "artifacts": {"json": str(sample_dir / "asr-ab-sample-plan.json"), "markdown": str(sample_dir / "asr-ab-sample-plan.md")},
        "updated_at": now_iso(),
    }
    if write:
        write_json(sample_dir / "asr-ab-sample-plan.json", result)
        (sample_dir / "asr-ab-sample-plan.md").write_text(_render_markdown(result), encoding="utf-8")
    return result


def _local_sensevoice_variant(
    key: str,
    sample_media: Path,
    output_dir: Path,
    *,
    language: str,
    punc_model: str,
    spk_model: str = "",
    device: str = "",
    speaker_merge_threshold: float | None = None,
    preset_speaker_count: int | None = None,
    evaluation_only_known_speaker_count: bool = False,
    use_itn: bool,
    merge_vad: bool,
) -> dict[str, Any]:
    raw_output = output_dir / f"{key}.raw-asr-output.json"
    local_device = str(device or "").strip() or default_local_asr_device(
        _resolve_python_executable()
    )
    command = [
        _resolve_python_executable(),
        "-m",
        "video_knowledge_pipeline.funasr_python_runner",
        "--input",
        str(sample_media),
        "--output",
        str(raw_output),
        "--provider",
        "sensevoice",
        "--model",
        "iic/SenseVoiceSmall",
        "--vad-model",
        "fsmn-vad",
        "--language",
        language,
        "--device",
        local_device,
        "--merge-length-s",
        "15",
        "--vad-max-single-segment-time-ms",
        "30000",
    ]
    if punc_model:
        command.extend(["--punc-model", punc_model])
    if spk_model:
        command.extend(["--spk-model", spk_model])
    if speaker_merge_threshold is not None:
        command.extend(
            ["--speaker-merge-threshold", str(float(speaker_merge_threshold))]
        )
    if preset_speaker_count is not None:
        command.extend(["--preset-speaker-count", str(int(preset_speaker_count))])
    command.append("--use-itn" if use_itn else "--no-use-itn")
    command.append("--merge-vad" if merge_vad else "--no-merge-vad")
    return {
        "key": key,
        "status": "planned_sample_first",
        "role": (
            "local_speaker_diagnostic_upper_bound"
            if evaluation_only_known_speaker_count
            else "local_primary_asr_with_candidate_speaker_labels"
            if spk_model
            else "local_primary_asr_ab"
        ),
        "execute": False,
        "sample_media_must_exist": True,
        "expected_output_json": str(raw_output),
        "command": command,
        "powershell": _powershell_join(command),
        "runner": "funasr_python",
        "full_mode": {
            "vad_model": "fsmn-vad",
            "punc_model": punc_model,
            "spk_model": spk_model,
            "speaker_merge_threshold": speaker_merge_threshold,
            "preset_speaker_count": preset_speaker_count,
            "use_itn": bool(use_itn),
            "merge_vad": bool(merge_vad),
            "merge_length_s": 15,
            "vad_max_single_segment_time_ms": 30000,
        },
        "operator_boundary": {
            "does_not_promote_any_transcript": True,
            "speaker_labels_are_anonymous_candidates": bool(spk_model),
            "speaker_roles_are_not_inferred": True,
            "requires_prepared_local_speaker_model": bool(spk_model),
            "gpu_only": bool(spk_model),
            **(
                {
                    "evaluation_only_known_speaker_count": True,
                    "known_speaker_count": int(preset_speaker_count or 0),
                    "must_not_become_automatic_production_route": True,
                }
                if evaluation_only_known_speaker_count
                else {}
            ),
        },
    }


def _moss_transcribe_diarize_variant(
    key: str,
    sample_media: Path,
    output_dir: Path,
    *,
    language: str,
) -> dict[str, Any]:
    """Describe the existing MOSS preset without duplicating its CLI command.

    Intent: compare a model-native speaker-attributed transcript with CAM++
    on the exact same bounded local sample.
    Decision: defer command construction and readiness checks to the existing
    ``plan_asr_run(..., preset="moss-transcribe-diarize")`` front door.
    Reason: the pinned upstream ``mtd-subtitle`` CLI owns inference and writes
    raw ``segments.json`` with ``postprocess=False``; duplicating that command
    or parser here would create a second ASR integration path.
    Evidence: OpenMOSS commit ``eda4b9f`` parser/subtitle tests pass 18/18 and
    VKP already normalizes its start/end/text/speaker contract.
    Effective scope: preview/run metadata for one local A/B variant only; no
    model download, provider fallback, transcript promotion, or role inference.
    """

    return {
        "key": key,
        "status": "planned_optional_local_adapter",
        "role": "model_native_speaker_attributed_asr_candidate",
        "preset": "moss-transcribe-diarize",
        "provider": "moss-transcribe-diarize",
        "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "language": language,
        "execute": False,
        "sample_media_must_exist": True,
        "sample_media_path": str(sample_media),
        "candidate_workspace": str(output_dir / key / "workspace"),
        "runner": "existing_asr_plan_front_door",
        "upstream": {
            "project": "OpenMOSS/MOSS-Transcribe-Diarize",
            "commit": "eda4b9f13f1574765a80438c9797780a9bd48112",
            "entrypoint": "mtd-subtitle",
            "output_contract": "segments.json:start,end,text,speaker",
            "postprocess": False,
        },
        "operator_boundary": {
            "does_not_promote_any_transcript": True,
            "local_only": True,
            "requires_explicit_local_runtime_and_model": True,
            "does_not_download_model": True,
            "does_not_fallback_to_another_asr": True,
            "speaker_labels_are_anonymous_candidates": True,
            "speaker_roles_are_not_inferred": True,
        },
    }


def _dolphin_variant(key: str, sample_media: Path, output_dir: Path, *, language: str) -> dict[str, Any]:
    raw_output = output_dir / f"{key}.raw-asr-output.json"
    local_device = default_local_asr_device(_resolve_python_executable())
    model = os.environ.get("LECTURE_ASR_DOLPHIN_MODEL", "").strip() or "small"
    command = [
        _resolve_python_executable(),
        "-m",
        "video_knowledge_pipeline.dolphin_python_runner",
        "--input",
        str(sample_media),
        "--output",
        str(raw_output),
        "--model",
        model,
        "--language",
        language,
        "--region",
        "CN",
        "--device",
        local_device,
    ]
    return {
        "key": key,
        "status": "planned_optional_local_adapter",
        "role": "candidate_second_local_asr",
        "execute": False,
        "sample_media_must_exist": True,
        "expected_output_json": str(raw_output),
        "command": command,
        "powershell": _powershell_join(command),
        "runner": "dolphin_python",
        "provider": "dolphin",
        "model": model,
        "language": language,
        "operator_boundary": {
            "does_not_replace_primary_asr": True,
            "second_evidence_source_only": True,
            "requires_local_dolphin_package": True,
        },
        "notes": "Dolphin is a second local ASR evidence source for A/B. It is not the VKP default ASR and does not promote transcripts automatically. Default model is 'small' for a public-download smoke; word timestamps are intentionally disabled by default on Windows because the Dolphin/TorchCodec path is fragile. Use WhisperX for alignment.",
    }


def _whisperx_alignment_variant(key: str, sample_media: Path, output_dir: Path, *, language: str) -> dict[str, Any]:
    workspace = output_dir.parent.parent
    model = os.environ.get("LECTURE_ASR_WHISPERX_MODEL", "").strip() or "large-v3"
    command = [
        _resolve_python_executable(),
        "-m",
        "video_knowledge_pipeline.cli",
        "run-whisperx-alignment",
        str(workspace),
        str(sample_media),
        "--language",
        language,
        "--model",
        model,
        "--execute",
    ]
    return {
        "key": key,
        "status": "planned_optional_alignment",
        "role": "timestamp_speaker_alignment_evidence",
        "execute": False,
        "sample_media_must_exist": True,
        "expected_output_json": str(output_dir / key / "whisperx-alignment-run.json"),
        "command": command,
        "powershell": _powershell_join(command),
        "runner": "whisperx_alignment",
        "provider": "whisperx",
        "model": model,
        "language": language,
        "operator_boundary": {
            "does_not_replace_primary_asr": True,
            "alignment_evidence_only": True,
            "requires_local_whisperx_runtime": True,
            "use_for_timestamps_and_speaker_labels": True,
        },
        "notes": "WhisperX is included in the A/B matrix only as timestamp/speaker alignment evidence. It should not be promoted as the main transcript source without a separate arbitration step.",
    }


def _cloud_asr_variant(key: str, sample_media: Path, output_dir: Path, *, language: str, provider_config: dict[str, Any]) -> dict[str, Any]:
    safe_cfg = {k: v for k, v in provider_config.items() if k != "api_key"}
    model = str(safe_cfg.get("model") or "gpt-4o-transcribe")
    base_url = str(safe_cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    return {
        "key": key,
        "status": "planned_upload_disabled",
        "role": "cloud_quality_reference_optional",
        "execute": False,
        "expected_output_json": str(output_dir / f"{key}.raw-asr-output.json"),
        "request_plan": {
            "interface": "openai_audio_transcriptions",
            "url": f"{base_url}/audio/transcriptions",
            "model": model,
            "language": language,
            "audio_path": str(sample_media),
            "audio_exists": sample_media.exists(),
        },
        "provider_config": safe_cfg,
        "operator_boundary": {
            "preview_first": True,
            "execute_required_for_network_call": True,
            "will_upload_sample_audio_only_after_execute": True,
            "provider_config_runtime_only": True,
            "secrets_redacted": True,
        },
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ASR A/B Sample Plan",
        "",
        f"- Media: `{result.get('media_path', '')}`",
        f"- Sample: `{result.get('sample_start_seconds', 0)}` + `{result.get('duration_seconds', 0)}` seconds",
        f"- Sample file: `{result.get('sample_media_path', '')}`",
        "",
        "## Sample Extract",
        "",
        "```powershell",
        str(result.get("sample_extract_powershell") or ""),
        "```",
        "",
        "## Variants",
        "",
        "| Key | Status | Role | Execute by default |",
        "| --- | --- | --- | --- |",
    ]
    for row in result.get("variants") or []:
        lines.append(f"| `{row.get('key', '')}` | `{row.get('status', '')}` | `{row.get('role', '')}` | `{row.get('execute', False)}` |")
    lines.extend(["", "## Next Actions", ""])
    for action in result.get("next_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"


def _powershell_join(parts: list[str]) -> str:
    def quote(value: str) -> str:
        if not value:
            return "''"
        if any(ch.isspace() for ch in value) or any(ch in value for ch in "'`()[]{};&"):
            return "'" + value.replace("'", "''") + "'"
        return value
    return " ".join(quote(str(part)) for part in parts)
