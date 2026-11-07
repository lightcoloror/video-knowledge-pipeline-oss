from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .funasr_python_runner import _resolve_local_model
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.campplus_embedding_batch_private.v1"
UPSTREAM_COMMIT = "16cd165ac3946cc8c08bf845331f91fefec8e1a9"


def extract_campplus_embeddings(
    clips: list[Path], device: str
) -> tuple[list[list[float]], dict[str, Any]]:
    """Directly call the pinned FunASR CAM++ model in the ASR environment.

    Intent: keep the main VKP environment free of heavy ASR dependencies.
    Decision: reuse the same isolated Python and local CAM++ snapshot as the
    existing FunASR ASR runner; this module is only a batch process boundary.
    Reason: installing FunASR twice would create version and CUDA drift.
    Evidence: current production ASR plans pin ``.conda-lecture-asr`` and the
    official CAM++ smoke returns ``spk_embedding`` from ``AutoModel.generate``.
    Effective scope: local audio embeddings only; no ASR, download, CPU
    fallback, identity assignment, transcript mutation, or provider call.
    """

    resolved = Path(_resolve_local_model("cam++")).expanduser()
    if not resolved.is_dir():
        raise RuntimeError(
            "local CAM++ model is not ready; implicit download is forbidden"
        )
    weights = resolved / "campplus_cn_common.bin"
    config = resolved / "config.yaml"
    if not weights.is_file() or not config.is_file():
        raise RuntimeError("local CAM++ model cache is incomplete")
    try:
        import torch  # type: ignore
        from funasr import AutoModel  # type: ignore
    except Exception as exc:  # pragma: no cover - isolated optional runtime.
        raise RuntimeError(f"FunASR CAM++ runtime unavailable: {exc}") from exc
    if device != "cuda" or not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is required for CAM++ speaker-center extraction")
    model = AutoModel(model=str(resolved.resolve()), device="cuda", disable_update=True)
    vectors: list[list[float]] = []
    for clip in clips:
        generated = model.generate(input=str(clip))
        record = (
            generated[0] if isinstance(generated, list) and generated else generated
        )
        if not isinstance(record, dict) or record.get("spk_embedding") is None:
            raise RuntimeError(f"CAM++ emitted no spk_embedding for {clip}")
        value = record["spk_embedding"]
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        while (
            isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
        ):
            value = value[0]
        vectors.append([float(item) for item in value])
    return vectors, {
        "provider": "FunASR CAM++",
        "model_path": str(resolved.resolve()),
        "weights_sha256": sha256_file(weights),
        "device": "cuda",
        "automatic_cpu_fallback": False,
        "upstream_project": "modelscope/FunASR",
        "upstream_version": "1.3.30",
        "upstream_commit": UPSTREAM_COMMIT,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="campplus-embedding-runner")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    request = read_json(Path(args.request).expanduser().resolve())
    if not isinstance(request, dict) or request.get("schema") != SCHEMA:
        raise ValueError("unsupported CAM++ embedding request")
    clips = [
        Path(str(value)).expanduser().resolve() for value in request.get("clips") or []
    ]
    if not clips or any(not path.is_file() for path in clips):
        raise ValueError("CAM++ embedding request contains missing clips")
    vectors, model = extract_campplus_embeddings(
        clips, str(request.get("device") or "")
    )
    output = {
        "schema": SCHEMA,
        "status": "completed",
        "biometric_data": True,
        "must_remain_local": True,
        "must_not_be_committed": True,
        "model": model,
        "vectors": vectors,
    }
    write_json(Path(args.output).expanduser().resolve(), output)
    print(json.dumps({"status": "completed", "embedding_count": len(vectors)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
