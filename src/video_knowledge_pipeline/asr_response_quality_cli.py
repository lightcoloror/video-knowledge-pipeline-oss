from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .asr_response_quality import assess_asr_response
from .storage import read_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess verbose ASR output and prepare exact retry windows"
    )
    parser.add_argument("response_json")
    parser.add_argument("output_json")
    parser.add_argument("--task-instructions", default="")
    parser.add_argument("--asr-prompt", default="")
    parser.add_argument("--vad-json", default="")
    parser.add_argument("--media-duration-seconds", type=float)
    parser.add_argument("--retry-overlap-seconds", type=float, default=1.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _mapping(args.response_json, required=True)
    vad_payload = _mapping(args.vad_json, required=False) if args.vad_json else {}
    vad_intervals = vad_payload.get("intervals") or vad_payload.get("segments") or []
    task_instructions = str(args.task_instructions or "")
    asr_prompt = str(args.asr_prompt or "")
    instruction_source = "explicit_cli" if task_instructions or asr_prompt else "none"
    consent_path = ""
    if instruction_source == "none":
        consent_path = str(payload.get("consent_path") or "")
        consent = _mapping(consent_path, required=False) if consent_path else {}
        task_instructions = str(consent.get("instructions") or "")
        asr_prompt = str(consent.get("asr_prompt") or "")
        if task_instructions or asr_prompt:
            instruction_source = "connector_consent"
    result = assess_asr_response(
        payload,
        task_instructions=task_instructions,
        asr_prompt=asr_prompt,
        vad_intervals=vad_intervals if isinstance(vad_intervals, list) else [],
        media_duration_seconds=args.media_duration_seconds,
        retry_overlap_seconds=args.retry_overlap_seconds,
    )
    result["input_context"] = {
        "instruction_source": instruction_source,
        "task_instructions_sha256": _text_sha256(task_instructions),
        "asr_prompt_sha256": _text_sha256(asr_prompt),
        "consent_path": str(Path(consent_path).expanduser().resolve())
        if consent_path
        else "",
        "plaintext_instructions_persisted": False,
    }
    destination = Path(args.output_json).expanduser().resolve()
    write_json(destination, result)
    print(
        json.dumps(
            {**result, "report_path": str(destination)}, ensure_ascii=False, indent=2
        )
    )
    return 0 if result["status"] in {"passed", "review_required"} else 2


def _mapping(value: str, *, required: bool) -> dict[str, Any]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"JSON artifact not found: {path}")
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _text_sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest() if value else ""


if __name__ == "__main__":
    raise SystemExit(main())
