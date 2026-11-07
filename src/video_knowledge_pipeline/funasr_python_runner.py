from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from .media_tools import resolve_media_tool
from .storage import write_json

FUNASR_MODELSCOPE_ALIASES: dict[str, tuple[str, ...]] = {
    "fsmn-vad": ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",),
    "ct-punc": (
        "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    ),
    "cam++": (
        "iic/speech_campplus_sv_zh-cn_16k-common",
        "iic/speech_campplus_sv_zh-cn_16k-common-pytorch",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="funasr-python-runner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default="sensevoice")
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--punc-model", default="")
    parser.add_argument("--spk-model", default="")
    parser.add_argument(
        "--speaker-merge-threshold",
        type=float,
        default=None,
        help="Optional upstream CAM++ cosine merge threshold in (0, 1]",
    )
    parser.add_argument(
        "--preset-speaker-count",
        type=int,
        default=None,
        help="Optional known speaker count for evaluation; never infer roles",
    )
    parser.add_argument("--language", default="zh")
    parser.add_argument("--hotword", default="", help="Optional evidence-derived hotwords; never derive from evaluation references")
    parser.add_argument("--batch-size-s", type=int, default=60)
    parser.add_argument("--use-itn", action="store_true", dest="use_itn")
    parser.add_argument("--no-use-itn", action="store_false", dest="use_itn")
    parser.set_defaults(use_itn=True)
    parser.add_argument("--merge-vad", action="store_true", dest="merge_vad")
    parser.add_argument("--no-merge-vad", action="store_false", dest="merge_vad")
    parser.set_defaults(merge_vad=True)
    parser.add_argument("--merge-length-s", type=int, default=15)
    parser.add_argument("--vad-max-single-segment-time-ms", type=int, default=30000)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=os.environ.get("LECTURE_ASR_DEVICE", "auto"))
    args = parser.parse_args(argv)

    result = run_funasr(
        input_path=args.input,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        vad_model=args.vad_model,
        punc_model=args.punc_model,
        spk_model=args.spk_model,
        speaker_merge_threshold=args.speaker_merge_threshold,
        preset_speaker_count=args.preset_speaker_count,
        language=args.language,
        hotword=args.hotword,
        batch_size_s=args.batch_size_s,
        use_itn=args.use_itn,
        merge_vad=args.merge_vad,
        merge_length_s=args.merge_length_s,
        vad_max_single_segment_time_ms=args.vad_max_single_segment_time_ms,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def run_funasr(
    *,
    input_path: str,
    output_path: str,
    provider: str,
    model: str,
    vad_model: str = "fsmn-vad",
    punc_model: str = "",
    spk_model: str = "",
    speaker_merge_threshold: float | None = None,
    preset_speaker_count: int | None = None,
    language: str = "zh",
    hotword: str = "",
    batch_size_s: int = 60,
    use_itn: bool = True,
    merge_vad: bool = True,
    merge_length_s: int = 15,
    vad_max_single_segment_time_ms: int = 30000,
    device: str = "auto",
) -> dict[str, Any]:
    media = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not media.exists():
        return {"ok": False, "error": f"input not found: {media}", "output_path": str(output)}
    if speaker_merge_threshold is not None and not (
        0.0 < float(speaker_merge_threshold) <= 1.0
    ):
        return {
            "ok": False,
            "error": "speaker_merge_threshold must be in (0, 1]",
            "output_path": str(output),
        }
    if preset_speaker_count is not None and int(preset_speaker_count) < 1:
        return {
            "ok": False,
            "error": "preset_speaker_count must be at least 1",
            "output_path": str(output),
        }
    try:
        from funasr import AutoModel  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency.
        return {"ok": False, "error": f"funasr import failed: {exc}", "output_path": str(output)}

    resolved_model = _resolve_local_model(model)
    resolved_vad_model = _resolve_local_model(vad_model) if vad_model else ""
    resolved_punc_model = _resolve_local_model(punc_model) if punc_model else ""
    resolved_spk_model = _resolve_local_model(spk_model) if spk_model else ""
    selected_device = _select_device(device)


    model_kwargs: dict[str, Any] = {"model": resolved_model}
    if selected_device in {"cuda", "cpu"}:
        model_kwargs["device"] = selected_device
    if vad_model:
        model_kwargs["vad_model"] = resolved_vad_model
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": int(vad_max_single_segment_time_ms or 30000)}
    if punc_model:
        model_kwargs["punc_model"] = resolved_punc_model
    if spk_model:
        model_kwargs["spk_model"] = resolved_spk_model
        # Intent: expose the fixed upstream CAM++ clustering controls for a
        # bounded quality experiment instead of implementing another clusterer.
        # Decision: pass merge_thr through AutoModel(spk_kwargs.cb_kwargs) and
        # the optional known count through generate(preset_spk_num).
        # Reason: a two-speaker oracle run separates count-estimation error from
        # attribution error, while remaining directly comparable to the default.
        # Evidence: FunASR 1.3.30 auto_model.py consumes both fields and its
        # ClusterBackend default is merge_thr=0.78.
        # Effective scope: explicit local runs only; no identity inference,
        # download, provider call, fallback, or transcript promotion.
        if speaker_merge_threshold is not None:
            model_kwargs["spk_kwargs"] = {
                "cb_kwargs": {"merge_thr": float(speaker_merge_threshold)}
            }
    runtime_metrics_state = _start_runtime_metrics(selected_device)
    try:
        asr_model = AutoModel(**model_kwargs)
        generate_kwargs: dict[str, Any] = {
            "input": str(media),
            "language": language,
            "batch_size_s": max(int(batch_size_s or 60), 1),
            "use_itn": bool(use_itn),
            "merge_vad": bool(merge_vad),
            "merge_length_s": max(int(merge_length_s or 15), 1),
            # Intent: preserve sentence-level timing instead of emitting one
            # coarse record for every five-minute chunk.
            # Decision: reuse the exact timestamp flags used by FunASR 1.3.30's
            # official CLI for SRT/TSV output.
            # Reason: SenseVoice text is otherwise complete but untimed, which
            # makes overlap ownership and per-window quality checks unreliable.
            # Evidence: pinned upstream funasr/cli.py sets these three flags
            # before AutoModel.generate; AutoModel falls back to VAD segment
            # timing when the ASR model has no token timestamps.
            # Effective scope: local FunASR result structure only; raw text,
            # model selection, routing, downloads, and network behavior do not
            # change.
            "sentence_timestamp": True,
            "output_timestamp": True,
            "return_time_stamps": True,
        }
        if str(hotword or "").strip():
            generate_kwargs["hotword"] = str(hotword).strip()
        if preset_speaker_count is not None:
            generate_kwargs["preset_spk_num"] = int(preset_speaker_count)
        if spk_model:
            # Intent: preserve one bounded centroid per chunk-local CAM++ label.
            # Decision: request FunASR's existing ``return_spk_center`` output;
            # never expose per-window embeddings or run a second voice model.
            # Reason: recording-global IDs cannot be recovered from numeric
            # labels alone because every five-minute child starts again at 0.
            # Evidence: pinned FunASR 1.3.30 ``postprocess`` returns centers in
            # the same corrected-label order as ``sentence_info[].spk``.
            # Effective scope: explicit local CAM++ child artifacts only. The
            # vectors are biometric local evidence and are stripped from public
            # transcript/Bundle exports by the parent runner.
            generate_kwargs["return_spk_center"] = True
        generated = asr_model.generate(**generate_kwargs)
    except Exception as exc:  # pragma: no cover - optional model runtime.
        runtime_metrics = _finish_runtime_metrics(runtime_metrics_state)
        # Intent: make a failed optional model actionable from the parent chunk report.
        # Decision: preserve Python's native traceback in the existing child JSON.
        # Reason: stderr also contains upstream progress bars and can hide the cause.
        # Evidence: the first CAM++ run's None-boundary exception was otherwise masked.
        # Effective scope: failure diagnostics only; no retry or transcript changes.
        return {
            "ok": False,
            "error": f"funasr generate failed: {exc}",
            "error_traceback": traceback.format_exc(),
            "output_path": str(output),
            "runtime_metrics": runtime_metrics,
        }

    runtime_metrics = _finish_runtime_metrics(runtime_metrics_state)
    generated, speaker_embedding_evidence = _prepare_generated_output(
        generated,
        retain_speaker_centers=bool(spk_model),
    )
    payload = {
        "schema": "video_knowledge_funasr_raw_output.v1",
        "provider": provider,
        "model": model,
        "resolved_model": resolved_model,
        "vad_model": vad_model,
        "resolved_vad_model": resolved_vad_model,
        "punc_model": punc_model,
        "resolved_punc_model": resolved_punc_model,
        "spk_model": spk_model,
        "resolved_spk_model": resolved_spk_model,
        "speaker_merge_threshold": speaker_merge_threshold,
        "preset_speaker_count": preset_speaker_count,
        "language": language,
        "hotword_count": len([value for value in str(hotword or "").split() if value]),
        "hotword_supplied": bool(str(hotword or "").strip()),
        "device": selected_device,
        "use_itn": bool(use_itn),
        "merge_vad": bool(merge_vad),
        "merge_length_s": int(merge_length_s or 15),
        "vad_max_single_segment_time_ms": int(vad_max_single_segment_time_ms or 30000),
        "input": str(media),
        "duration_seconds": _media_duration_seconds(media),
        "runtime_metrics": runtime_metrics,
        "speaker_embedding_evidence": speaker_embedding_evidence,
        "result": generated,
    }
    write_json(output, payload)
    return {"ok": True, "error": "", "output_path": str(output), "records": _record_count(generated)}


def _prepare_generated_output(
    generated: Any,
    *,
    retain_speaker_centers: bool,
) -> tuple[Any, dict[str, Any]]:
    """Serialize only per-speaker centers and mark their privacy boundary."""

    value = _json_safe(generated)
    records = value if isinstance(value, list) else [value]
    center_count = 0
    dimensions: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        centers = record.pop("spk_embedding_center", None)
        if centers is None or not retain_speaker_centers:
            continue
        center_rows = centers if isinstance(centers, list) else []
        if center_rows and not isinstance(center_rows[0], list):
            center_rows = [center_rows]
        private_rows: list[dict[str, Any]] = []
        for index, center in enumerate(center_rows):
            if not isinstance(center, list) or not center:
                continue
            vector = [float(item) for item in center]
            private_rows.append(
                {"local_speaker_id": str(index), "center": vector}
            )
            center_count += 1
            dimensions.add(len(vector))
        if private_rows:
            record["_speaker_embedding_centers"] = private_rows
    return value, {
        "status": "available" if center_count else "not_requested" if not retain_speaker_centers else "unavailable",
        "center_count": center_count,
        "dimensions": sorted(dimensions),
        "biometric_data": bool(center_count),
        "must_remain_local": bool(center_count),
        "must_not_be_committed": bool(center_count),
        "per_window_embeddings_retained": False,
        "person_identity_inferred": False,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


def _start_runtime_metrics(device: str) -> dict[str, Any]:
    """Start bounded local inference timing and optional CUDA peak tracking.

    Intent: make ASR stability evidence include elapsed time and GPU pressure.
    Decision: reuse PyTorch's native CUDA peak-memory counters when available.
    Reason: completion alone cannot distinguish stable GPU runs from near-OOM runs.
    Evidence: the existing FunASR runtime already depends on PyTorch for CUDA.
    Effective scope: local child-process telemetry only; inference is unchanged.
    """

    state: dict[str, Any] = {
        "started_perf_counter": time.perf_counter(),
        "device": str(device or ""),
        "cuda_peak_tracking": False,
    }
    if state["device"] != "cuda":
        return state
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()):
            torch.cuda.reset_peak_memory_stats()
            state["cuda_peak_tracking"] = True
    except Exception:
        pass
    return state


def _finish_runtime_metrics(state: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"device": str(state.get("device") or "")}
    if bool(state.get("cuda_peak_tracking")):
        try:
            import torch  # type: ignore

            torch.cuda.synchronize()
            result["cuda_peak_memory_allocated_mib"] = round(
                float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0), 3
            )
            result["cuda_peak_memory_reserved_mib"] = round(
                float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0), 3
            )
        except Exception:
            result["cuda_peak_memory_unavailable"] = True
    started = float(state.get("started_perf_counter") or time.perf_counter())
    result["elapsed_seconds"] = round(max(time.perf_counter() - started, 0.0), 3)
    return result

def _media_duration_seconds(media: Path) -> float:
    """Probe duration through VKP's shared, environment-independent resolver.

    Intent: keep ASR duration evidence valid inside isolated Python runtimes.
    Decision: reuse ``media_tools.resolve_media_tool`` before falling back to
    a bare executable name.
    Reason: the lecture ASR environment can run FFmpeg-generated WAV files
    while lacking ffprobe on its child PATH, which previously wrote 0 seconds.
    Evidence: VKP already centralizes FFMPEG_BINARY, FFPROBE_BINARY,
    LECTURE_FFMPEG_DIR, known tool directories, and PATH lookup.
    Effective scope: local FunASR metadata probing only; model inference and
    transcript text are unchanged.
    """

    ffprobe = (
        os.environ.get("FFPROBE")
        or resolve_media_tool("ffprobe")
        or "ffprobe"
    )
    try:
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(media)],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15,
            check=False,
        )
        if completed.returncode == 0:
            return max(float(completed.stdout.strip() or 0.0), 0.0)
    except Exception:
        pass
    return 0.0


def _select_device(device: str) -> str:
    requested = str(device or "auto").strip().lower()
    if requested in {"cuda", "cpu"}:
        return requested
    try:
        import torch  # type: ignore

        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        return "cpu"


def _record_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        result = value.get("result")
        if isinstance(result, list):
            return len(result)
        return 1
    return 0


def _resolve_local_model(model: str) -> str:
    """Prefer an already downloaded model path to avoid ModelScope lock waits."""
    clean = str(model or "").strip()
    if not clean:
        return clean
    if os.environ.get("LECTURE_ASR_FORCE_REMOTE_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}:
        return clean
    direct = Path(clean).expanduser()
    if direct.exists():
        return str(direct.resolve())
    for candidate in _local_model_candidates(clean):
        if candidate.exists():
            return str(candidate.resolve())
    return clean


def _local_model_candidates(model: str) -> list[Path]:
    model_ids = FUNASR_MODELSCOPE_ALIASES.get(model, (model,))
    model_names = [item.split("/")[-1] for item in model_ids]
    roots: list[Path] = []
    if os.environ.get("MODELSCOPE_CACHE"):
        roots.append(Path(os.environ["MODELSCOPE_CACHE"]).expanduser())
    roots.extend(
        [
            Path.home() / ".cache" / "modelscope",
            Path.home() / ".cache" / "modelscope" / "hub",
            Path.home() / ".cache" / "modelscope" / "hub" / "models",
        ]
    )
    candidates: list[Path] = []
    for root in roots:
        for model_id, model_name in zip(model_ids, model_names):
            candidates.extend(
                [
                    root / model_id,
                    root / model_id.replace("/", "\\"),
                    root / model_id.replace("/", "--"),
                    root / "hub" / "models" / model_id,
                    root / "hub" / "models" / model_id.replace("/", "\\"),
                    root / "models" / model_id,
                    root / "models" / model_id.replace("/", "\\"),
                    root / "models" / model_name,
                    root / model_name,
                ]
            )
    return candidates


if __name__ == "__main__":
    raise SystemExit(main())
