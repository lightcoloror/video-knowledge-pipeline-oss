from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .funasr_python_runner import _resolve_local_model, _select_device
from .storage import write_json


SCHEMA = "video_knowledge_pipeline.punctuation_model_runner.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="punctuation-model-runner")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", default="ct-punc")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args(argv)
    result = run_punctuation_model_request(
        args.request_json,
        args.output_json,
        model=args.model,
        device=args.device,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "outputs"}, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def run_punctuation_model_request(
    request_json: str | Path,
    output_json: str | Path,
    *,
    model: str = "ct-punc",
    device: str = "auto",
) -> dict[str, Any]:
    request_path = Path(request_json).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    blocks = request.get("blocks") if isinstance(request, dict) else None
    if not isinstance(blocks, list):
        return _write_failure(output_path, "invalid_request", "request.blocks must be a list")
    resolved_model = _resolve_local_model(model)
    selected_device = _select_device(device)
    try:
        from funasr import AutoModel  # type: ignore

        kwargs: dict[str, Any] = {"model": resolved_model}
        if selected_device in {"cuda", "cpu"}:
            kwargs["device"] = selected_device
        runtime = AutoModel(**kwargs)
        outputs = [
            _generated_text(runtime.generate(input=str(row.get("text") or "")))
            for row in blocks
            if isinstance(row, dict)
        ]
        result = {
            "schema": SCHEMA,
            "ok": True,
            "model": model,
            "resolved_model": resolved_model,
            "device": selected_device,
            "block_count": len(blocks),
            "outputs": outputs,
        }
    except Exception as exc:
        result = {
            "schema": SCHEMA,
            "ok": False,
            "model": model,
            "resolved_model": resolved_model,
            "device": selected_device,
            "error_code": "punctuation_model_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "outputs": [],
        }
    write_json(output_path, result)
    return result


def _generated_text(value: Any) -> str:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        for key in ("text", "text_postprocessed", "sentence", "value"):
            if value.get(key):
                return str(value[key]).strip()
        nested = value.get("result")
        if isinstance(nested, list) and nested:
            return _generated_text(nested[0])
    return str(value or "").strip()


def _write_failure(path: Path, code: str, error: str) -> dict[str, Any]:
    result = {"schema": SCHEMA, "ok": False, "error_code": code, "error": error, "outputs": []}
    write_json(path, result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())