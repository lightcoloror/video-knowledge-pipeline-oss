from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def run_worker(
    *,
    backend: str,
    media_path: Path,
    source_root: Path,
    checkpoint_path: Path,
    frame_rate: float,
    threshold: float,
    max_decoded_frames: int,
) -> dict[str, Any]:
    if backend == "autoshot":
        return _run_autoshot(
            media_path=media_path,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            frame_rate=frame_rate,
            threshold=threshold,
            max_decoded_frames=max_decoded_frames,
        )
    if backend == "omnishotcut":
        return _run_omnishotcut(
            media_path=media_path,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            frame_rate=frame_rate,
            max_decoded_frames=max_decoded_frames,
        )
    raise ValueError("backend must be autoshot or omnishotcut")


def _run_autoshot(
    *,
    media_path: Path,
    source_root: Path,
    checkpoint_path: Path,
    frame_rate: float,
    threshold: float,
    max_decoded_frames: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("AutoShot requires CUDA; CPU fallback is disabled")

    source_text = str(source_root)
    sys.path.insert(0, source_text)
    try:
        from supernet_flattransf_3_8_8_8_13_12_0_16_60 import (
            TransNetV2Supernet,
        )
        from utils import get_batches, get_frames, predictions_to_scenes

        frames = get_frames(str(media_path))
        if len(frames) <= 0:
            raise RuntimeError("AutoShot decoded no frames")
        if len(frames) > int(max_decoded_frames):
            raise RuntimeError(
                f"AutoShot decoded {len(frames)} frames, above "
                f"the safety limit {max_decoded_frames}"
            )

        model = TransNetV2Supernet().eval()
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint.get("net", checkpoint)
        current = model.state_dict()
        compatible = {
            key: value for key, value in state.items() if key in current
        }
        if not compatible:
            raise RuntimeError("AutoShot checkpoint has no compatible parameters")
        current.update(compatible)
        model.load_state_dict(current)
        model = model.cuda(0).eval()

        predictions: list[Any] = []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for batch in get_batches(frames):
                tensor = torch.from_numpy(
                    batch.transpose((3, 0, 1, 2))[np.newaxis, ...]
                ).float().cuda(0)
                logits = model(tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                values = torch.sigmoid(logits[0]).detach().cpu().numpy()
                predictions.append(values[25:75])
        scores = np.concatenate(predictions, axis=0)[: len(frames)]
        binary = (scores > float(threshold)).astype(np.uint8)
        scenes = predictions_to_scenes(binary).tolist()
        return {
            "schema": "video_knowledge_pipeline.saved_shot_predictions.v1",
            "source_format": "autoshot_scenes",
            "fps": frame_rate,
            "threshold": float(threshold),
            "scenes": scenes,
            "decoded_frames": len(frames),
            "batch_window_frames": 100,
            "kept_center_frames": 50,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "device": torch.cuda.get_device_name(0),
        }
    finally:
        if sys.path and sys.path[0] == source_text:
            sys.path.pop(0)


def _run_omnishotcut(
    *,
    media_path: Path,
    source_root: Path,
    checkpoint_path: Path,
    frame_rate: float,
    max_decoded_frames: int,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("OmniShotCut requires CUDA; CPU fallback is disabled")

    source_text = str(source_root)
    sys.path.insert(0, source_text)
    try:
        import omnishotcut

        model = omnishotcut.load(str(checkpoint_path))
        ranges, intra_labels, inter_labels = model.inference(
            str(media_path),
            mode="default",
            overlap=20,
        )
        if ranges:
            last_frame = max(int(row[1]) for row in ranges)
            if last_frame + 1 > int(max_decoded_frames):
                raise RuntimeError(
                    f"OmniShotCut output exceeds frame safety limit "
                    f"{max_decoded_frames}"
                )
        return {
            "schema": "video_knowledge_pipeline.saved_shot_predictions.v1",
            "source_format": "omnishotcut_scenes",
            "fps": frame_rate,
            "scenes": [[int(row[0]), int(row[1])] for row in ranges],
            "intra_labels": list(intra_labels),
            "inter_labels": list(inter_labels),
            "overlap_frames": 20,
            "device": torch.cuda.get_device_name(0),
        }
    finally:
        if sys.path and sys.path[0] == source_text:
            sys.path.pop(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--frame-rate", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=0.296)
    parser.add_argument("--max-decoded-frames", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_worker(
        backend=args.backend,
        media_path=Path(args.media_path).expanduser().resolve(),
        source_root=Path(args.source_root).expanduser().resolve(),
        checkpoint_path=Path(args.checkpoint_path).expanduser().resolve(),
        frame_rate=float(args.frame_rate),
        threshold=float(args.threshold),
        max_decoded_frames=max(1, int(args.max_decoded_frames)),
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
