"""Thin VKP adapter for shared context, ASR reuse and changed-window policy."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from video_knowledge_pipeline.path_defaults import workspace_root


def _shared(module_name="shared_context_efficiency", filename="context_efficiency.py"):
    configured = str(os.environ.get("SELF_MEDIA_SYSTEM_ROOT") or "").strip()
    root = Path(configured).expanduser().resolve() if configured else workspace_root() / "self-media-creation-system"
    spec = importlib.util.spec_from_file_location(module_name, root / "scripts" / filename)
    if not spec or not spec.loader:
        raise RuntimeError("shared context_efficiency.py unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def consume_handoff(handoff: Path):
    agent = _shared("shared_self_media_agent", "self_media_agent.py")
    return agent.validate_step_handoff(handoff, expected_tool_id="video-knowledge-pipeline")


def decide_asr(handoff: Path, audio: Path, previous_receipt: Path | None, output: Path, changed_windows: list[str]):
    envelope = consume_handoff(handoff)
    if envelope["step"]["output_contract"] != "asr-decision-receipt.v1":
        raise RuntimeError("ASR decision handoff contract required")
    shared = _shared()
    manifest_path = (handoff.parent / envelope["context"]["manifest"]["path"]).resolve()
    receipt_path = (handoff.parent / envelope["context"]["receipt"]["path"]).resolve()
    receipt_bytes = receipt_path.read_bytes()
    if shared.sha256_bytes(receipt_bytes) != envelope["context"]["receipt"]["sha256"]:
        raise RuntimeError("context receipt changed after validation")
    receipt = json.loads(receipt_bytes)
    media_rows = [row for row in receipt["artifacts"]
                  if row.get("role") == "source_media" or row.get("source_role") == "source_media"]
    matching = [row for row in media_rows
                if shared.resolve_manifest_path(manifest_path, row["path"]) == audio.expanduser().resolve()]
    if not matching:
        raise RuntimeError("audio is not a bound context media artifact")
    # Use the already validated receipt hash; no second adapter-level media scan.
    if changed_windows and previous_receipt:
        previous = json.loads(previous_receipt.read_text(encoding="utf-8"))
        if previous.get("audio_sha256") == matching[0]["sha256"]:
            raise RuntimeError("unchanged audio cannot claim changed-window rerun")
    result = shared.decide_asr(audio, previous_receipt, output, changed_windows)
    if result["decision"] == "reuse_existing_asr" and changed_windows:
        raise RuntimeError("unchanged audio cannot claim changed-window rerun")
    return result


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    decide = subparsers.add_parser("decide-asr")
    decide.add_argument("--handoff", type=Path, required=True)
    decide.add_argument("--audio", type=Path, required=True)
    decide.add_argument("--previous-receipt", type=Path)
    decide.add_argument("--output", type=Path, required=True)
    decide.add_argument("--changed-window", action="append", default=[])
    args = parser.parse_args()
    result = decide_asr(
        args.handoff,
        args.audio,
        args.previous_receipt,
        args.output,
        args.changed_window,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
