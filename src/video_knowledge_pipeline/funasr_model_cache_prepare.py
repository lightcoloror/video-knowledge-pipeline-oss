from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .funasr_python_runner import _resolve_local_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="funasr-model-cache-prepare")
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--vad-model", default="")
    parser.add_argument("--punc-model", default="")
    parser.add_argument("--spk-model", default="")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=os.environ.get("LECTURE_ASR_DEVICE", "auto"))
    args = parser.parse_args(argv)
    result = prepare_funasr_model_cache(
        model=args.model,
        vad_model=args.vad_model,
        punc_model=args.punc_model,
        spk_model=args.spk_model,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def prepare_funasr_model_cache(*, model: str, vad_model: str = "", punc_model: str = "", spk_model: str = "", device: str = "auto") -> dict[str, Any]:
    try:
        from funasr import AutoModel  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"funasr import failed: {exc}"}
    resolved_model = _resolve_local_model(model)
    resolved_vad = _resolve_local_model(vad_model) if vad_model else ""
    resolved_punc = _resolve_local_model(punc_model) if punc_model else ""
    resolved_spk = _resolve_local_model(spk_model) if spk_model else ""
    kwargs: dict[str, Any] = {"model": resolved_model}
    selected = _select_device(device)
    if selected in {"cuda", "cpu"}:
        kwargs["device"] = selected
    if vad_model:
        kwargs["vad_model"] = resolved_vad
    if punc_model:
        kwargs["punc_model"] = resolved_punc
    if spk_model:
        kwargs["spk_model"] = resolved_spk
    try:
        AutoModel(**kwargs)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"AutoModel prepare failed: {exc}",
            "resolved": {"model": resolved_model, "vad_model": resolved_vad, "punc_model": resolved_punc, "spk_model": resolved_spk},
        }
    return {
        "ok": True,
        "error": "",
        "device": selected,
        "resolved": {"model": resolved_model, "vad_model": resolved_vad, "punc_model": resolved_punc, "spk_model": resolved_spk},
    }


def _select_device(device: str) -> str:
    requested = str(device or "auto").strip().lower()
    if requested in {"cuda", "cpu"}:
        return requested
    try:
        import torch  # type: ignore
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
