from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import asr_runtime_profile
from .file_hash import sha256_file
from .funasr_python_runner import _resolve_local_model
from .media_tools import resolve_media_tool
from .models import new_id
from .path_defaults import local_model_root
from .powershell import quote_powershell_argument as _quote_powershell_arg
from .storage import ensure_project_dirs, write_json

ASR_PRESETS: dict[str, dict[str, Any]] = {
    "funasr": {
        "label": "FunASR Paraformer Chinese",
        "provider": "funasr",
        "command": "funasr",
        "env_command": "LECTURE_FUNASR_COMMAND",
        "module": "funasr",
        "output_file": "funasr-output.json",
        "notes": "Best first choice for Chinese lecture speech when installed. Uses FunASR CLI-style arguments.",
    },
    "sensevoice": {
        "label": "SenseVoice via FunASR",
        "provider": "sensevoice",
        "command": "funasr",
        "env_command": "LECTURE_FUNASR_COMMAND",
        "module": "funasr",
        "output_file": "sensevoice-output.json",
        "notes": "SenseVoice usually runs through FunASR with a SenseVoice model id.",
    },
    "fun-asr-nano": {
        "label": "Fun-ASR-Nano", "provider": "funasr", "command": "funasr",
        "env_command": "LECTURE_FUNASR_COMMAND", "module": "funasr",
        "output_file": "fun-asr-nano-output.json", "default_model": "FunAudioLLM/Fun-ASR-Nano-2512",
        "notes": "Quality challenger for Chinese ASR. Benchmark before promotion.",
    },
    "contextual-paraformer": {
        "label": "FunASR Contextual Paraformer", "provider": "funasr", "command": "funasr",
        "env_command": "LECTURE_FUNASR_COMMAND", "module": "funasr",
        "output_file": "contextual-paraformer-output.json",
        "default_model": "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
        "notes": "Contextual Chinese ASR with evidence-derived hotwords. Benchmark before promotion.",
    },
    "qwen3-asr-0.6b": {
        "label": "Qwen3-ASR 0.6B", "provider": "qwen3_asr", "command": "qwen-asr",
        "env_command": "LECTURE_QWEN3_ASR_COMMAND", "module": "qwen_asr",
        "output_file": "qwen3-asr-0.6b-output.json", "default_model": "Qwen/Qwen3-ASR-0.6B",
        "notes": "Official Qwen3-ASR fallback.",
    },
    "qwen3-asr-1.7b": {
        "label": "Qwen3-ASR 1.7B", "provider": "qwen3_asr", "command": "qwen-asr",
        "env_command": "LECTURE_QWEN3_ASR_COMMAND", "module": "qwen_asr",
        "output_file": "qwen3-asr-1.7b-output.json", "default_model": "Qwen/Qwen3-ASR-1.7B",
        "notes": "Quality-profile second ASR.",
    },
    "qwen3-forced-aligner": {
        "label": "Qwen3 Forced Aligner 0.6B", "provider": "qwen3_forced_aligner", "command": "qwen-asr",
        "env_command": "LECTURE_QWEN3_ASR_COMMAND", "module": "qwen_asr",
        "output_file": "qwen3-forced-aligner-output.json", "default_model": "Qwen/Qwen3-ForcedAligner-0.6B",
        "notes": "Alignment-only branch.",
    },    "whisperx": {
        "label": "WhisperX",
        "provider": "whisperx",
        "command": "whisperx",
        "env_command": "LECTURE_WHISPERX_COMMAND",
        "module": "whisperx",
        "output_file": "whisperx-output.json",
        "default_model": "large-v3",
        "notes": "Precision timestamp enhancement branch. Use after SenseVoice when word-level alignment or diarization is needed.",
    },
    "faster-whisper": {
        "label": "faster-whisper",
        "provider": "faster-whisper",
        "command": "faster-whisper",
        "env_command": "LECTURE_FASTER_WHISPER_COMMAND",
        "module": "faster_whisper",
        "output_file": "faster-whisper-output.json",
        "notes": "Lightweight local Whisper backend; CLI availability depends on installation.",
    },
    "moss-transcribe-diarize": {
        "label": "MOSS Transcribe Diarize",
        "provider": "moss-transcribe-diarize",
        "command": "mtd-subtitle",
        "env_command": "LECTURE_MOSS_TRANSCRIBE_COMMAND",
        "module": "moss_transcribe_diarize",
        "output_file": "segments.json",
        "default_model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "notes": "Optional local long-form multi-speaker challenger. Use for explicit A/B trials; it does not replace SenseVoice.",
    },
}


def detect_asr_runners() -> dict[str, Any]:
    tools = []
    for name, preset in ASR_PRESETS.items():
        command_path = _resolve_command_path(preset)
        module_available = importlib.util.find_spec(str(preset["module"])) is not None
        runtime_probe = _command_runtime_probe(preset=name, command_path=command_path)
        runnable = bool(command_path) and bool(runtime_probe.get("ready"))
        tools.append(
            {
                "name": name,
                "label": preset["label"],
                "provider": preset["provider"],
                "command": preset["command"],
                "command_path": command_path or "",
                "entrypoint_available": bool(command_path),
                "env_command": preset.get("env_command", ""),
                "module": preset["module"],
                "module_available": module_available,
                "runtime_probe": runtime_probe,
                "runnable": runnable,
                "available": runnable,
                "notes": preset["notes"],
            }
        )
    return {
        "ffmpeg": resolve_media_tool("ffmpeg"),
        "tools": tools,
        "recommended_order": ["sensevoice", "contextual-paraformer", "qwen3-asr-1.7b", "qwen3-asr-0.6b", "fun-asr-nano", "qwen3-forced-aligner", "whisperx", "moss-transcribe-diarize", "faster-whisper"],
    }


def plan_asr_run(
    root: str | Path,
    media_path: str | Path,
    *,
    preset: str = "funasr",
    language: str = "zh",
    model: str | None = None,
    punc_model: str | None = None,
    spk_model: str | None = None,
    hotword: str | None = None,
    use_itn: bool = True,
    merge_vad: bool = True,
    merge_length_s: int = 15,
    vad_max_single_segment_time_ms: int = 30000,
    chunk_boundary_mode: str = "fixed_duration",
    chunk_overlap_seconds: float = 5.0,
    qwen_timestamps: bool = True,
    qwen_timestamp_basis: str = "chunk_relative",
    vad_json: str | Path | None = None,
    transcript_path: str | Path | None = None,
    alignment_chunk_seconds: int = 300,
) -> dict[str, Any]:
    if preset not in ASR_PRESETS:
        raise ValueError(f"unsupported ASR preset: {preset}")
    if type(alignment_chunk_seconds) is not int or alignment_chunk_seconds < 30:
        raise ValueError("alignment_chunk_seconds must be an integer of at least 30 seconds")
    if preset != "qwen3-forced-aligner" and alignment_chunk_seconds != 300:
        raise ValueError("alignment_chunk_seconds is only supported by qwen3-forced-aligner")
    root_path = Path(root).expanduser().resolve()
    paths = ensure_project_dirs(root_path)
    media = Path(media_path).expanduser().resolve()
    if not media.exists():
        raise FileNotFoundError(f"media not found: {media}")
    vad_evidence = Path(vad_json).expanduser().resolve() if vad_json else None
    if vad_evidence is not None and not vad_evidence.is_file():
        raise FileNotFoundError(f"VAD evidence not found: {vad_evidence}")
    if vad_evidence is not None and preset not in {"qwen3-asr-0.6b", "qwen3-asr-1.7b"}:
        raise ValueError("vad_json priority evidence is supported only by Qwen3-ASR presets")
    if str(qwen_timestamp_basis) not in {"chunk_relative", "media_absolute"}:
        raise ValueError("qwen_timestamp_basis must be chunk_relative or media_absolute")
    alignment_transcript = _resolve_alignment_transcript(root_path, transcript_path) if preset == "qwen3-forced-aligner" else None
    if preset == "qwen3-forced-aligner" and alignment_transcript is None:
        raise FileNotFoundError("qwen3-forced-aligner requires transcript_path or a transcript JSON in the workspace")

    runtime_profile = asr_runtime_profile()
    model = model or _runtime_model_for_preset(preset, runtime_profile)
    if punc_model is None:
        punc_model = str(runtime_profile.get("punc_model") or "").strip() or None
    if spk_model is None and runtime_profile.get("enable_diarization"):
        spk_model = str(runtime_profile.get("spk_model") or "").strip() or None
    if merge_length_s == 15:
        merge_length_s = int(runtime_profile.get("merge_length_s") or merge_length_s)
    if preset == "contextual-paraformer" and vad_max_single_segment_time_ms == 30000:
        vad_max_single_segment_time_ms = 20000

    run_id = new_id("asr_run")
    output_dir = paths["transcripts"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    preset_info = ASR_PRESETS[preset]
    output_json = output_dir / ("segments.json" if preset == "moss-transcribe-diarize" else "raw-asr-output.json")
    python_executable = _resolve_python_executable()
    configured_device = str(runtime_profile.get("device") or "").strip()
    local_device = configured_device if configured_device in {"cuda", "cpu", "auto"} else default_local_asr_device(python_executable)
    command_path = _resolve_command_path(preset_info)
    runtime_probe = _command_runtime_probe(preset=preset, command_path=command_path)
    module_available = _module_available_in_python(str(preset_info["module"]), python_executable)
    use_python_runner = preset in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer", "qwen3-asr-0.6b", "qwen3-asr-1.7b", "qwen3-forced-aligner", "faster-whisper", "whisperx"} and module_available and (preset != "whisperx" or not command_path)
    command = _command_for_preset(
        preset=preset,
        command=_resolve_command(preset_info),
        media=media,
        output_dir=output_dir,
        output_json=output_json,
        language=language,
        model=model,
        punc_model=punc_model,
        spk_model=spk_model,
        hotword=hotword,
        use_itn=use_itn,
        merge_vad=merge_vad,
        merge_length_s=merge_length_s,
        vad_max_single_segment_time_ms=vad_max_single_segment_time_ms,
        chunk_boundary_mode=chunk_boundary_mode,
        chunk_overlap_seconds=chunk_overlap_seconds,
        python_executable=python_executable,
        local_device=local_device,
        use_python_runner=use_python_runner,
        qwen_timestamps=qwen_timestamps,
        qwen_timestamp_basis=qwen_timestamp_basis,
        vad_json=vad_evidence,
        alignment_transcript=alignment_transcript,
        alignment_chunk_seconds=alignment_chunk_seconds,
    )
    normalize_command = [
        "video-knowledge",
        "normalize-asr",
        str(paths["root"]),
        str(output_json),
        "--provider",
        str(preset_info["provider"]),
        "--title",
        media.stem,
    ]
    runtime_available = (bool(command_path) and bool(runtime_probe.get("ready"))) or use_python_runner
    speaker_model_readiness = _speaker_model_readiness(
        preset=preset,
        model=spk_model,
    )
    availability_blockers = []
    if speaker_model_readiness["required"] and not speaker_model_readiness["ready"]:
        availability_blockers.append(
            f"speaker_model_missing_or_not_downloaded:{speaker_model_readiness['model']}"
        )
    plan = {
        "run_id": run_id,
        "project": str(paths["root"]),
        "preset": preset,
        "provider": preset_info["provider"],
        "media_path": str(media),
        "alignment_transcript": str(alignment_transcript) if alignment_transcript else "",
        "alignment_chunk_seconds": alignment_chunk_seconds if preset == "qwen3-forced-aligner" else None,
        "vad_priority_evidence": {
            "path": str(vad_evidence) if vad_evidence else "",
            "sha256": sha256_file(vad_evidence) if vad_evidence else "",
            "candidate_only": True,
            "does_not_prove_speech": True,
        },
        "qwen_timestamp_basis": str(qwen_timestamp_basis),
        "output_dir": str(output_dir),
        "expected_output_json": str(output_json),
        "provider_output_file": str(preset_info["output_file"]),
        "command": command,
        "runner": _runner_name(preset, use_python_runner),
        "asr_mode": "full" if preset in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer", "qwen3-asr-0.6b", "qwen3-asr-1.7b", "moss-transcribe-diarize"} else ("alignment" if preset in {"whisperx", "qwen3-forced-aligner"} else "fallback"),
        "full_mode": {
            "vad_model": "fsmn-vad" if preset in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer"} else "",
            "punc_model": _default_punc_model(preset, punc_model),
            "spk_model": spk_model or "",
            "hotword_count": len([value for value in str(hotword or "").split() if value]),
            "hotword_supplied": bool(str(hotword or "").strip()),
            "use_itn": bool(use_itn),
            "merge_vad": bool(merge_vad),
            "merge_length_s": int(merge_length_s or 15),
            "vad_max_single_segment_time_ms": int(vad_max_single_segment_time_ms or 30000),
            "batch_size_s": 60,
            "chunk_boundary_mode": chunk_boundary_mode,
            "chunk_overlap_seconds": float(chunk_overlap_seconds),
        },
        "oom_recovery": {
            "enabled": bool(use_python_runner and local_device == "cuda" and preset in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer"}),
            "trigger": "cuda_out_of_memory_only",
            "gpu_retry": {"batch_size_s": 10, "vad_max_single_segment_time_ms": 15000},
            "cpu_retry_after": "second_cuda_out_of_memory_only",
        },
        "asr_runtime_profile": _safe_asr_runtime_profile(runtime_profile),
        "python_executable": python_executable,
        "local_asr_device": local_device,
        "pythonpath": str(Path(__file__).resolve().parents[1]),
        "powershell": _powershell_join(command),
        "normalize_command": normalize_command,
        "normalize_powershell": _powershell_join(normalize_command),
        "available": bool(runtime_available and not availability_blockers),
        "availability": {
            "command": str(preset_info["command"]),
            "command_path": command_path or "",
            "entrypoint_available": bool(command_path),
            "module": str(preset_info["module"]),
            "module_available": module_available,
            "python_module_runner_available": bool(use_python_runner),
            "python_executable": python_executable,
            "runtime_probe": runtime_probe,
            "runtime_ready": bool(runtime_available),
            "required_models_ready": not availability_blockers,
            "blockers": availability_blockers,
        },
        "model_ready": _model_ready(preset=preset, model=model or _default_model_for_preset(preset)),
        "model_readiness": {
            "speaker": speaker_model_readiness,
            "required_models_ready": not availability_blockers,
        },
        "notes": preset_info["notes"],
    }
    write_json(output_dir / "asr-run-plan.json", plan)
    return {**plan, "plan_path": str(output_dir / "asr-run-plan.json")}


def _command_for_preset(
    *,
    preset: str,
    command: str,
    media: Path,
    output_dir: Path,
    output_json: Path,
    language: str,
    model: str | None,
    punc_model: str | None,
    spk_model: str | None,
    use_itn: bool,
    merge_vad: bool,
    merge_length_s: int,
    vad_max_single_segment_time_ms: int,
    chunk_boundary_mode: str,
    chunk_overlap_seconds: float,
    python_executable: str,
    local_device: str,
    use_python_runner: bool,
    qwen_timestamps: bool,
    alignment_transcript: Path | None,
    qwen_timestamp_basis: str = "chunk_relative",
    vad_json: Path | None = None,
    hotword: str | None = None,
    alignment_chunk_seconds: int = 300,
) -> list[str]:
    if preset == "moss-transcribe-diarize":
        return [
            command,
            str(media),
            "--backend",
            "hf",
            "--model",
            model or _default_model_for_preset(preset),
            "--out-dir",
            str(output_dir),
            "--device",
            local_device,
            "--max-new-tokens",
            str(_moss_max_new_tokens()),
        ]
    if use_python_runner and preset in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer"}:
        command_parts = [
            python_executable,
            "-m",
            "video_knowledge_pipeline.funasr_chunked_runner",
            "--input",
            str(media),
            "--output",
            str(output_json),
            "--provider",
            preset,
            "--model",
            model or _default_model_for_preset(preset),
            "--vad-model",
            "fsmn-vad",
            "--language",
            language,
            "--device",
            local_device,
            "--batch-size-s",
            "60",
            "--chunk-seconds",
            "300",
            "--chunk-overlap-seconds",
            f"{float(chunk_overlap_seconds):g}",
            "--chunk-boundary-mode",
            chunk_boundary_mode,
            "--merge-length-s",
            str(int(merge_length_s or 15)),
            "--vad-max-single-segment-time-ms",
            str(int(vad_max_single_segment_time_ms or 30000)),
        ]
        selected_punc_model = _default_punc_model(preset, punc_model)
        if selected_punc_model:
            command_parts.extend(["--punc-model", selected_punc_model])
        if spk_model:
            command_parts.extend(["--spk-model", spk_model])
        if str(hotword or "").strip():
            command_parts.extend(["--hotword", str(hotword).strip()])
        command_parts.append("--use-itn" if use_itn else "--no-use-itn")
        command_parts.append("--merge-vad" if merge_vad else "--no-merge-vad")
        return command_parts
    if use_python_runner and preset == "qwen3-forced-aligner":
        return [
            python_executable,
            "-m",
            "video_knowledge_pipeline.qwen3_forced_aligner_runner",
            "--input",
            str(media),
            "--transcript",
            str(alignment_transcript),
            "--output",
            str(output_json),
            "--model",
            model or _local_qwen_aligner_path() or "Qwen/Qwen3-ForcedAligner-0.6B",
            "--language",
            language,
            "--device",
            local_device,
            "--chunk-seconds",
            str(alignment_chunk_seconds),
        ]
    if use_python_runner and preset in {"qwen3-asr-0.6b", "qwen3-asr-1.7b"}:
        selected_model = model or _local_qwen_model_path(preset) or _default_model_for_preset(preset)
        command_parts = [
            python_executable,
            "-m",
            "video_knowledge_pipeline.qwen3_asr_python_runner",
            "--input",
            str(media),
            "--output",
            str(output_json),
            "--model",
            selected_model,
            "--language",
            language,
            "--device",
            local_device,
            "--chunk-seconds",
            "300",
            "--timestamp-basis",
            str(qwen_timestamp_basis),
        ]
        if vad_json is not None:
            command_parts.extend(["--vad-json", str(vad_json)])
        if qwen_timestamps:
            command_parts.extend(["--forced-aligner", _local_qwen_aligner_path() or "Qwen/Qwen3-ForcedAligner-0.6B"])
        else:
            command_parts.append("--no-timestamps")
            # Bound text-only spans and token use when the optional aligner is
            # unavailable. These are audio-window times, never word alignment.
            command_parts[command_parts.index("--chunk-seconds") + 1] = "30"
        # Intent: reuse Qwen3-ASR's official context channel for reviewed
        # domain terms rather than inventing a provider-specific prompt layer.
        # Decision: map VKP's existing evidence-derived ``hotword`` argument to
        # Qwen's ``--context`` only for the Qwen presets.
        # Reason: the same plan contract already prevents evaluation references
        # from becoming correction evidence, while Qwen has no FunASR-style
        # hotword flag.
        # Evidence: qwen-asr 0.0.6 ``Qwen3ASRModel.transcribe`` accepts context
        # and language together.
        # Effective scope: local Qwen candidate inference; no cloud call,
        # fallback, canonical transcript mutation, or reference import.
        if str(hotword or "").strip():
            command_parts.extend(["--context", str(hotword).strip()])
        return command_parts
    if use_python_runner and preset == "faster-whisper":
        return [
            python_executable,
            "-m",
            "video_knowledge_pipeline.faster_whisper_python_runner",
            "--input",
            str(media),
            "--output",
            str(output_json),
            "--model",
            model or "large-v3",
            "--language",
            language,
            "--device",
            local_device,
            "--compute-type",
            _default_faster_whisper_compute_type(local_device),
            "--vad-filter",
        ]
    if preset == "funasr":
        command_parts = [
            command,
            "++model=paraformer-zh",
            "++vad_model=fsmn-vad",
            f"++punc_model={_default_punc_model(preset, punc_model) or 'ct-punc'}",
            f"++use_itn={str(bool(use_itn))}",
            f"++merge_vad={str(bool(merge_vad))}",
            f"++merge_length_s={int(merge_length_s or 15)}",
            f"++input={media}",
            f"++output_dir={output_dir}",
        ]
        if spk_model:
            command_parts.append(f"++spk_model={spk_model}")
        if str(hotword or "").strip():
            command_parts.append(f"++hotword={str(hotword).strip()}")
        return command_parts
    if preset == "sensevoice":
        command_parts = [
            command,
            f"++model={model or 'iic/SenseVoiceSmall'}",
            "++vad_model=fsmn-vad",
            f"++punc_model={_default_punc_model(preset, punc_model) or 'ct-punc'}",
            f"++use_itn={str(bool(use_itn))}",
            f"++merge_vad={str(bool(merge_vad))}",
            f"++merge_length_s={int(merge_length_s or 15)}",
            f"++input={media}",
            f"++output_dir={output_dir}",
        ]
        if spk_model:
            command_parts.append(f"++spk_model={spk_model}")
        if str(hotword or "").strip():
            command_parts.append(f"++hotword={str(hotword).strip()}")
        return command_parts
    if use_python_runner and preset == "whisperx":
        return [
            python_executable,
            "-m",
            "whisperx",
            str(media),
            "--model",
            model or "large-v3",
            "--language",
            language,
            "--output_dir",
            str(output_dir),
            "--output_format",
            "json",
            "--device",
            local_device,
        ]
    if preset == "whisperx":
        return [
            command,
            str(media),
            "--model",
            model or "large-v3",
            "--language",
            language,
            "--output_dir",
            str(output_dir),
            "--output_format",
            "json",
            "--device",
            local_device if local_device != "auto" else "cuda",
        ]
    if preset == "faster-whisper":
        return [
            command,
            str(media),
            "--model",
            model or "large-v3",
            "--language",
            language,
            "--output_dir",
            str(output_dir),
            "--device",
            local_device if local_device != "auto" else "cuda",
            "--compute-type",
            _default_faster_whisper_compute_type(local_device),
        ]
    raise ValueError(f"unsupported ASR preset: {preset}")


def plan_whisperx_alignment(
    root: str | Path,
    media_path: str | Path,
    *,
    language: str = "zh",
    model: str | None = None,
) -> dict[str, Any]:
    """Plan a WhisperX run for word-level alignment without replacing the primary ASR path."""
    return plan_asr_run(root, media_path, preset="whisperx", language=language, model=model or _default_model_for_preset("whisperx"))



def default_local_asr_device(python_executable: str | None = None) -> str:
    """GPU-first local ASR device selection.

    LECTURE_ASR_DEVICE remains the explicit override. Without it, VKP probes
    the local ASR Python environment and prefers CUDA; CPU is the fallback.
    """
    explicit = os.environ.get("LECTURE_ASR_DEVICE", "").strip().lower()
    if explicit in {"cuda", "cpu", "auto"}:
        return explicit
    return "cuda" if _cuda_available_in_python(python_executable or _resolve_python_executable()) else "cpu"


def _cuda_available_in_python(python_executable: str) -> bool:
    try:
        completed = subprocess.run(
            [
                python_executable,
                "-c",
                "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
            check=False,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _default_punc_model(preset: str, punc_model: str | None) -> str:
    if punc_model is not None:
        return str(punc_model).strip()
    if preset in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer"}:
        return "ct-punc"
    return ""



def _default_faster_whisper_compute_type(local_device: str) -> str:
    explicit = os.environ.get("LECTURE_ASR_COMPUTE_TYPE", "").strip()
    if explicit:
        return explicit
    if local_device == "cuda":
        return "float16"
    if local_device == "cpu":
        return "int8"
    return "auto"


def _default_model_for_preset(preset: str) -> str:
    if preset == "sensevoice":
        return "iic/SenseVoiceSmall"
    if preset == "funasr":
        return "paraformer-zh"
    if preset == "fun-asr-nano":
        return "FunAudioLLM/Fun-ASR-Nano-2512"
    if preset == "contextual-paraformer":
        return "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
    if preset == "qwen3-asr-0.6b":
        return "Qwen/Qwen3-ASR-0.6B"
    if preset == "qwen3-asr-1.7b":
        return "Qwen/Qwen3-ASR-1.7B"
    if preset == "qwen3-forced-aligner":
        return "Qwen/Qwen3-ForcedAligner-0.6B"
    if preset in {"whisperx", "faster-whisper"}:
        return "large-v3"
    if preset == "moss-transcribe-diarize":
        return "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    return ""


def _moss_max_new_tokens() -> int:
    """Return the bounded generation budget for the reused upstream MOSS CLI.

    Intent: prevent a successful-looking long-media run from ending at the
    upstream CLI's 2048-token default.
    Decision: reuse ``mtd-subtitle --max-new-tokens`` with a conservative
    default of 8192 and an explicit environment override.
    Reason: the fixed 300-second GPU sample stopped at 178.44 seconds after
    generating exactly 2048 tokens.
    Evidence: the upstream CLI exposes this option and recommends raising it
    for longer audio.
    Scope: MOSS-Transcribe-Diarize command construction only.
    """
    raw = os.environ.get("LECTURE_MOSS_MAX_NEW_TOKENS", "8192").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "LECTURE_MOSS_MAX_NEW_TOKENS must be an integer between 2048 and 65536"
        ) from exc
    if value < 2048 or value > 65536:
        raise ValueError(
            "LECTURE_MOSS_MAX_NEW_TOKENS must be between 2048 and 65536"
        )
    return value


def _runtime_model_for_preset(preset: str, runtime_profile: dict[str, Any]) -> str | None:
    configured_provider = str(runtime_profile.get("provider") or "").strip()
    configured_model = str(runtime_profile.get("model") or "").strip()
    if preset == "sensevoice" and configured_provider == "funasr_sensevoice" and configured_model:
        return configured_model
    return None


def _safe_asr_runtime_profile(profile: dict[str, Any]) -> dict[str, Any]:
    safe = json.loads(json.dumps(profile, ensure_ascii=False))
    service = safe.get("openai_compatible")
    if isinstance(service, dict):
        service.pop("api_key", None)
    return safe


def _local_qwen_model_path(preset: str) -> str | None:
    names = {
        "qwen3-asr-0.6b": "Qwen3-ASR-0.6B",
        "qwen3-asr-1.7b": "Qwen3-ASR-1.7B",
        "qwen3-forced-aligner": "Qwen3-ForcedAligner-0.6B",
    }
    name = names.get(preset)
    if not name:
        return None
    roots = [local_model_root()]
    for root in roots:
        candidate = root.expanduser() / name
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _resolve_alignment_transcript(root: Path, value: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if value:
        path = Path(value).expanduser()
        candidates.append(path if path.is_absolute() else root / path)
    candidates.extend(
        [
            root / "corrected-transcript.json",
            root / "source-arbitrated-transcript.json",
            root / "postprocessed-transcript.json",
            root / "normalized-transcript.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _local_qwen_aligner_path() -> str | None:
    return _local_qwen_model_path("qwen3-forced-aligner")


def _preset_available(preset: dict[str, Any]) -> bool:
    return bool(_resolve_command_path(preset))


def _runner_name(preset: str, use_python_runner: bool) -> str:
    if preset == "moss-transcribe-diarize":
        return "moss_transcribe_diarize_cli"
    if not use_python_runner:
        return "subprocess_command"
    if preset in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer"}:
        return "funasr_python"
    if preset in {"qwen3-asr-0.6b", "qwen3-asr-1.7b", "qwen3-forced-aligner"}:
        return "qwen3_forced_aligner_python" if preset == "qwen3-forced-aligner" else "qwen3_asr_python"
    if preset == "faster-whisper":
        return "faster_whisper_python"
    if preset == "whisperx":
        return "whisperx_python_module"
    return "subprocess_command"


def _resolve_python_executable() -> str:
    env_python = os.environ.get("LECTURE_ASR_PYTHON", "").strip()
    if env_python:
        return env_python
    bin_dir = os.environ.get("LECTURE_ASR_BIN_DIR", "").strip()
    if bin_dir:
        scripts = Path(bin_dir).expanduser()
        for candidate in (scripts / "python.exe", scripts.parent / "python.exe"):
            if candidate.exists():
                return str(candidate.resolve())
    root = Path(__file__).resolve().parents[2]
    for env_dir in (root / ".conda-lecture-asr", root / ".venv-lecture-asr"):
        for candidate in (env_dir / "python.exe", env_dir / "Scripts" / "python.exe"):
            if candidate.exists():
                return str(candidate.resolve())
    return shutil.which("python") or "python"


def _module_available_in_python(module: str, python_executable: str) -> bool:
    if python_executable in {"python", ""}:
        return importlib.util.find_spec(module) is not None
    import subprocess

    try:
        completed = subprocess.run(
            [python_executable, "-c", f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module!r}) else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _command_runtime_probe(*, preset: str, command_path: str) -> dict[str, Any]:
    """Use an upstream CLI self-test when entrypoint presence is not enough."""
    if not command_path:
        return {
            "required": preset == "moss-transcribe-diarize",
            "ready": False,
            "status": "entrypoint_missing",
            "blocker": "command_not_found",
        }
    if preset != "moss-transcribe-diarize":
        return {
            "required": False,
            "ready": True,
            "status": "not_required",
            "blocker": "",
        }
    try:
        completed = subprocess.run(
            [command_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "required": True,
            "ready": False,
            "status": "self_test_timeout",
            "blocker": "cli_self_test_timeout",
        }
    except OSError:
        return {
            "required": True,
            "ready": False,
            "status": "entrypoint_not_executable",
            "blocker": "cli_entrypoint_not_executable",
        }
    if completed.returncode == 0:
        return {
            "required": True,
            "ready": True,
            "status": "ready",
            "blocker": "",
            "exit_code": 0,
        }
    blocker = _classify_command_probe_failure(
        "\n".join([completed.stdout or "", completed.stderr or ""])
    )
    return {
        "required": True,
        "ready": False,
        "status": "dependency_not_ready",
        "blocker": blocker,
        "exit_code": int(completed.returncode),
    }


def _classify_command_probe_failure(output: str) -> str:
    normalized = str(output or "")
    marker = "No module named "
    if marker in normalized:
        value = normalized.split(marker, 1)[1].splitlines()[0].strip().strip("'\"")
        safe_module = "".join(
            character for character in value if character.isalnum() or character in "._-"
        )
        return f"missing_python_dependency:{safe_module or 'unknown'}"
    lowered = normalized.lower()
    if "requires tokenizers" in lowered or (
        "tokenizers" in lowered and "found" in lowered
    ):
        return "python_dependency_conflict:tokenizers"
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return "python_dependency_import_error"
    return "cli_self_test_failed"


_MOSS_REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "configuration_moss_transcribe_diarize.py",
    "modeling_moss_transcribe_diarize.py",
    "processing_moss_transcribe_diarize.py",
)
_MOSS_SINGLE_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_MOSS_WEIGHT_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _model_ready(*, preset: str, model: str) -> dict[str, Any]:
    model_id = model or ("iic/SenseVoiceSmall" if preset == "sensevoice" else "paraformer-zh")
    if preset == "moss-transcribe-diarize":
        return _moss_model_ready(model_id)
    if preset in {"qwen3-asr-0.6b", "qwen3-asr-1.7b", "qwen3-forced-aligner"}:
        return _qwen_model_ready(model_id)
    direct_path = Path(model_id).expanduser()
    if direct_path.exists():
        resolved = str(direct_path.resolve())
        return {
            "model": model_id,
            "ready": True,
            "cache_matches": [resolved],
            "status": "ready",
        }
    cache_roots = [Path(os.environ.get("MODELSCOPE_CACHE", "")).expanduser() if os.environ.get("MODELSCOPE_CACHE") else None, Path.home() / ".cache" / "modelscope", Path.home() / ".cache" / "huggingface" / "hub", local_model_root().expanduser()]
    matches = []
    for root in [path for path in cache_roots if path]:
        if not root.exists():
            continue
        model_parts = model_id.split("/")
        model_name = model_parts[-1]
        candidates = [
            root / model_id,
            root / model_id.replace("/", "\\"),
            root / model_id.replace("/", "--"),
            root / model_name,
            root / "hub" / "models" / model_id,
            root / "hub" / "models" / model_id.replace("/", "\\"),
            root / "hub" / model_id.replace("/", "--"),
            root / "models" / model_id,
            root / "models" / model_id.replace("/", "\\"),
            root / "models" / model_name,
            root / f"models--{model_id.replace('/', '--')}",
        ]
        matches.extend(str(candidate) for candidate in candidates if candidate.exists())
    matches = sorted(set(matches))
    return {
        "model": model_id,
        "ready": bool(matches),
        "cache_matches": matches,
        "status": "ready" if matches else "unknown_or_not_downloaded",
    }


def _qwen_model_ready(model_id: str) -> dict[str, Any]:
    """Validate an offline Qwen ASR snapshot instead of trusting a cache shell.

    Intent: prevent an interrupted Hub directory from being reported as a ready
    local model. Decision: reuse VKP's indexed-weight validation contract used
    by the MOSS adapter and require config plus every referenced shard. Reason:
    Hugging Face creates the repo/snapshot directory before large weights finish.
    Evidence: the 2026-08-11 Cantonese smoke found only config/index and two
    ``.incomplete`` blobs while the old directory-existence probe returned ready.
    Effective scope: Qwen3-ASR/ForcedAligner readiness and offline source choice;
    no network access, download, fallback, or transcript promotion is added.
    """

    candidates: set[Path] = set()
    direct_path = Path(model_id).expanduser()
    if direct_path.exists():
        candidates.update(_expand_huggingface_snapshot_path(direct_path))
    model_name = model_id.split("/")[-1]
    roots = _unique_paths(
        [
            Path(os.environ["HF_HUB_CACHE"]).expanduser()
            if os.environ.get("HF_HUB_CACHE")
            else None,
            Path(os.environ["HF_HOME"]).expanduser() / "hub"
            if os.environ.get("HF_HOME")
            else None,
            Path.home() / ".cache" / "huggingface" / "hub",
            Path(os.environ["MODELSCOPE_CACHE"]).expanduser()
            if os.environ.get("MODELSCOPE_CACHE")
            else None,
            Path.home() / ".cache" / "modelscope",
            local_model_root().expanduser(),
        ]
    )
    for root in roots:
        if not root.exists():
            continue
        for candidate in (
            root / model_id,
            root / model_name,
            root / "hub" / "models" / model_id,
            root / "models" / model_id,
            root / f"models--{model_id.replace('/', '--')}",
        ):
            if candidate.exists():
                candidates.update(_expand_huggingface_snapshot_path(candidate))

    snapshots = [
        _qwen_snapshot_content_status(path)
        for path in sorted(candidates, key=lambda value: str(value).lower())
    ]
    ready_paths = [row["path"] for row in snapshots if row["ready"]]
    incomplete = [row for row in snapshots if not row["ready"]]
    return {
        "model": model_id,
        "ready": bool(ready_paths),
        "cache_matches": ready_paths,
        "incomplete_cache_matches": incomplete,
        "status": (
            "ready"
            if ready_paths
            else "incomplete_cache"
            if incomplete
            else "unknown_or_not_downloaded"
        ),
        "network_access": "disabled",
    }


def _qwen_snapshot_content_status(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    missing_files = [name for name in ("config.json",) if not (resolved / name).is_file()]
    single_weight = next(
        (
            name
            for name in _MOSS_SINGLE_WEIGHT_FILES
            if _usable_model_weight(resolved / name)
        ),
        "",
    )
    indexed_weight = ""
    missing_weight_shards: list[str] = []
    invalid_weight_indexes: list[str] = []
    if not single_weight:
        for name in _MOSS_WEIGHT_INDEX_FILES:
            index_path = resolved / name
            if not index_path.is_file():
                continue
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                invalid_weight_indexes.append(name)
                continue
            weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
            shard_names = sorted(
                {str(value) for value in (weight_map or {}).values() if str(value).strip()}
            )
            if not shard_names:
                invalid_weight_indexes.append(name)
                continue
            missing = [shard for shard in shard_names if not _usable_model_weight(resolved / shard)]
            if missing:
                missing_weight_shards.extend(missing)
                continue
            indexed_weight = name
            break
    ready = not missing_files and bool(single_weight or indexed_weight)
    return {
        "path": str(resolved),
        "ready": ready,
        "status": (
            "ready"
            if ready
            else "missing_required_snapshot_files"
            if missing_files
            else "incomplete_weight_shards"
            if missing_weight_shards
            else "invalid_weight_index"
            if invalid_weight_indexes
            else "missing_model_weights"
        ),
        "missing_files": missing_files,
        "weight_artifact": single_weight or indexed_weight,
        "missing_weight_shards": sorted(set(missing_weight_shards)),
        "invalid_weight_indexes": sorted(set(invalid_weight_indexes)),
    }


def _moss_model_ready(model_id: str) -> dict[str, Any]:
    """Validate a complete local MOSS snapshot without contacting the Hub.

    Intent: distinguish a downloaded model from an empty Hugging Face repo shell.
    Decision: reuse ``huggingface_hub.scan_cache_dir`` when available, then apply
    the exact files required by the pinned MOSS ``AutoModel``/``AutoProcessor``
    source contract. Direct exported model directories use the same content gate.
    Reason: directory existence alone can represent an interrupted download and
    would otherwise allow inference to start or trigger an implicit download.
    Evidence: huggingface_hub 0.30.2 scans only valid revisions and reports broken
    blobs; MOSS commit eda4b9f loads config, processor, tokenizer, remote-code
    modules and safetensor/bin weights from the selected snapshot.
    Effective scope: MOSS readiness metadata only; no download, model load, ASR,
    network request, fallback, or transcript promotion occurs here.
    """
    candidates: set[Path] = set()
    direct_path = Path(model_id).expanduser()
    if direct_path.exists():
        candidates.update(_expand_huggingface_snapshot_path(direct_path))

    scanner: dict[str, Any] = {
        "library": "huggingface_hub.scan_cache_dir",
        "available": False,
        "version": "",
        "warning_count": 0,
        "scanned_roots": [],
    }
    hf_roots = _unique_paths(
        [
            Path(os.environ["HF_HUB_CACHE"]).expanduser()
            if os.environ.get("HF_HUB_CACHE")
            else None,
            Path(os.environ["HF_HOME"]).expanduser() / "hub"
            if os.environ.get("HF_HOME")
            else None,
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )
    try:
        import huggingface_hub
        from huggingface_hub import scan_cache_dir

        scanner["available"] = True
        scanner["version"] = str(huggingface_hub.__version__)
        for root in hf_roots:
            if not root.is_dir():
                continue
            scanner["scanned_roots"].append(str(root.resolve()))
            try:
                cache_info = scan_cache_dir(root)
            except (OSError, ValueError):
                scanner["warning_count"] += 1
                continue
            scanner["warning_count"] += len(cache_info.warnings)
            for repo in cache_info.repos:
                if repo.repo_type != "model" or repo.repo_id != model_id:
                    continue
                candidates.update(revision.snapshot_path for revision in repo.revisions)
    except ImportError:
        pass

    model_name = model_id.split("/")[-1]
    local_roots = _unique_paths(
        [
            Path(os.environ["MODELSCOPE_CACHE"]).expanduser()
            if os.environ.get("MODELSCOPE_CACHE")
            else None,
            Path.home() / ".cache" / "modelscope",
            local_model_root().expanduser(),
        ]
    )
    for root in local_roots:
        if not root.exists():
            continue
        for candidate in (
            root / model_id,
            root / model_name,
            root / "hub" / "models" / model_id,
            root / "models" / model_id,
            root / f"models--{model_id.replace('/', '--')}",
        ):
            if candidate.exists():
                candidates.update(_expand_huggingface_snapshot_path(candidate))

    snapshots = [
        _moss_snapshot_content_status(path)
        for path in sorted(candidates, key=lambda value: str(value).lower())
    ]
    ready_paths = [row["path"] for row in snapshots if row["ready"]]
    incomplete = [row for row in snapshots if not row["ready"]]
    return {
        "model": model_id,
        "ready": bool(ready_paths),
        "cache_matches": ready_paths,
        "incomplete_cache_matches": incomplete,
        "status": (
            "ready"
            if ready_paths
            else "incomplete_cache"
            if incomplete
            else "unknown_or_not_downloaded"
        ),
        "required_snapshot_files": list(_MOSS_REQUIRED_SNAPSHOT_FILES),
        "accepted_weight_layouts": {
            "single": list(_MOSS_SINGLE_WEIGHT_FILES),
            "indexed": list(_MOSS_WEIGHT_INDEX_FILES),
        },
        "cache_scanner": scanner,
        "network_access": "disabled",
    }


def _unique_paths(values: list[Path | None]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        key = str(value).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _expand_huggingface_snapshot_path(path: Path) -> list[Path]:
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        return sorted(
            (row for row in snapshots.iterdir() if row.is_dir()),
            key=lambda value: value.name,
        )
    return [path]


def _moss_snapshot_content_status(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    missing_files = [
        name for name in _MOSS_REQUIRED_SNAPSHOT_FILES if not (resolved / name).is_file()
    ]
    single_weight = next(
        (
            name
            for name in _MOSS_SINGLE_WEIGHT_FILES
            if _usable_model_weight(resolved / name)
        ),
        "",
    )
    indexed_weight = ""
    missing_weight_shards: list[str] = []
    invalid_weight_indexes: list[str] = []
    if not single_weight:
        for name in _MOSS_WEIGHT_INDEX_FILES:
            index_path = resolved / name
            if not index_path.is_file():
                continue
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                invalid_weight_indexes.append(name)
                continue
            weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
            shard_names = sorted(
                {
                    str(value)
                    for value in (weight_map or {}).values()
                    if str(value).strip()
                }
            )
            if not shard_names:
                invalid_weight_indexes.append(name)
                continue
            missing = [
                shard
                for shard in shard_names
                if not _usable_model_weight(resolved / shard)
            ]
            if missing:
                missing_weight_shards.extend(missing)
                continue
            indexed_weight = name
            break
    weight_ready = bool(single_weight or indexed_weight)
    ready = not missing_files and weight_ready
    if ready:
        status = "ready"
    elif missing_files:
        status = "missing_required_snapshot_files"
    elif missing_weight_shards:
        status = "incomplete_weight_shards"
    elif invalid_weight_indexes:
        status = "invalid_weight_index"
    else:
        status = "missing_model_weights"
    return {
        "path": str(resolved),
        "ready": ready,
        "status": status,
        "missing_files": missing_files,
        "weight_artifact": single_weight or indexed_weight,
        "missing_weight_shards": sorted(set(missing_weight_shards)),
        "invalid_weight_indexes": sorted(set(invalid_weight_indexes)),
    }


def _usable_model_weight(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with path.open("rb") as source:
            prefix = source.read(200)
    except OSError:
        return False
    return b"version https://git-lfs.github.com/spec/v1" not in prefix


def _speaker_model_readiness(*, preset: str, model: str | None) -> dict[str, Any]:
    """Require a prepared local speaker model when diarization is explicitly enabled."""
    requested_model = str(model or "").strip()
    required = bool(
        requested_model
        and preset in {"funasr", "sensevoice", "fun-asr-nano", "contextual-paraformer"}
    )
    if not required:
        return {
            "model": requested_model,
            "resolved_model": "",
            "required": False,
            "ready": True,
            "status": "not_requested" if not requested_model else "not_applicable",
        }
    resolved_model = _resolve_local_model(requested_model)
    resolved_path = Path(resolved_model).expanduser()
    ready = resolved_path.exists()
    return {
        "model": requested_model,
        "resolved_model": str(resolved_path.resolve()) if ready else resolved_model,
        "required": True,
        "ready": ready,
        "status": "ready" if ready else "missing_or_not_downloaded",
    }


def _resolve_command(preset: dict[str, Any]) -> str:
    return _resolve_command_path(preset) or str(preset["command"])


def _resolve_command_path(preset: dict[str, Any]) -> str:
    env_name = str(preset.get("env_command") or "")
    if env_name:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            env_path = Path(env_value).expanduser()
            if env_path.exists():
                return str(env_path.resolve())
            return env_value
    bin_dir = os.environ.get("LECTURE_ASR_BIN_DIR", "").strip()
    if bin_dir:
        resolved = _command_from_bin_dir(Path(bin_dir).expanduser(), str(preset["command"]))
        if resolved:
            return resolved
    root = Path(__file__).resolve().parents[2]
    for default_bin in _default_command_bin_dirs(root, preset):
        resolved = _command_from_bin_dir(default_bin, str(preset["command"]))
        if resolved:
            return resolved
    return shutil.which(str(preset["command"])) or ""


def _default_command_bin_dirs(root: Path, preset: dict[str, Any]) -> list[Path]:
    """Return reviewed project-local command locations in precedence order.

    Intent: recognize an already-provisioned dedicated MOSS environment without
    asking callers to repeat an environment variable on every invocation.
    Decision: extend the existing command resolver with the one fixed local
    environment path; keep the normal shared ASR environments ahead of it.
    Reason: launcher discovery and runtime readiness are separate checks. The
    project currently has ``.local/moss-runtime-py312/Scripts/mtd-subtitle.exe``
    but its upstream import self-test still correctly reports missing deps.
    Evidence: the isolated upstream parser/subtitle contract passes 18/18 while
    ``mtd-subtitle --help`` fails closed on missing ``transformers``.
    Effective scope: local command discovery only; no install, download,
    inference, provider fallback, PATH mutation, or readiness override.
    """

    bins = [
        root / ".conda-lecture-asr" / "Scripts",
        root / ".venv-lecture-asr" / "Scripts",
    ]
    if str(preset.get("command") or "") == "mtd-subtitle":
        bins.append(root / ".local" / "moss-runtime-py312" / "Scripts")
    return bins


def _command_from_bin_dir(bin_dir: Path, command: str) -> str:
    candidate = bin_dir / command
    windows_candidate = candidate.with_suffix(".exe")
    if candidate.exists():
        return str(candidate.resolve())
    if windows_candidate.exists():
        return str(windows_candidate.resolve())
    return ""


def _powershell_join(args: list[str]) -> str:
    return " ".join(_quote_powershell_arg(arg) for arg in args)
