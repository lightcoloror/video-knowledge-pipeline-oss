from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.campplus_oracle_cluster_private.v1"


def cluster_campplus_centers(
    centers: list[list[float]], expected_speaker_count: int
) -> list[int]:
    """Reuse FunASR's fixed spectral clusterer with an operator-known count."""

    import numpy as np  # type: ignore
    from funasr.models.campplus.cluster_backend import SpectralCluster  # type: ignore

    state = np.random.get_state()
    try:
        np.random.seed(0)
        labels = SpectralCluster()(
            np.asarray(centers, dtype="float32"), int(expected_speaker_count)
        )
    finally:
        np.random.set_state(state)
    return [int(value) for value in labels.tolist()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="campplus-oracle-cluster-runner")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    request = read_json(Path(args.request).expanduser().resolve())
    if not isinstance(request, dict) or request.get("schema") != SCHEMA:
        raise ValueError("unsupported CAM++ oracle cluster request")
    centers = request.get("centers")
    expected = int(request.get("expected_speaker_count") or 0)
    if not isinstance(centers, list) or not centers or not 0 < expected <= len(centers):
        raise ValueError("invalid CAM++ oracle cluster request")
    labels = cluster_campplus_centers(centers, expected)
    write_json(
        Path(args.output).expanduser().resolve(),
        {
            "schema": SCHEMA,
            "status": "completed",
            "biometric_data": True,
            "must_remain_local": True,
            "labels": labels,
        },
    )
    print(json.dumps({"status": "completed", "label_count": len(labels)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
