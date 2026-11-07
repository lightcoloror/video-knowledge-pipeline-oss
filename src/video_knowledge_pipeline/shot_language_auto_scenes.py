from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .path_defaults import source_reviews_root


AUTO_SCENES_COMMIT = "2c34db3520e1319292bb456a0e610a0ef195e78b"
DEFAULT_SOURCE_ROOT = source_reviews_root() / "video-workflow-wave-20260720" / "auto-scenes-extraction"


class AutoScenesShotLanguageRuntime:
    """Thin runtime around the pinned Auto Scenes public analyzers."""

    def __init__(
        self,
        *,
        source_root: str | Path | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.root = (
            Path(source_root).expanduser().resolve()
            if source_root
            else DEFAULT_SOURCE_ROOT.resolve()
        )
        if not self.root.is_dir():
            raise FileNotFoundError(f"Auto Scenes source not found: {self.root}")
        actual = _git_commit(self.root)
        if actual != AUTO_SCENES_COMMIT:
            raise RuntimeError(
                "Auto Scenes commit mismatch: "
                f"expected={AUTO_SCENES_COMMIT} actual={actual}"
            )
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from A_coreUtils.aftertreatment.optical_flow_analyzer import (
            OpticalFlowAnalyzer,
        )

        self._flow = OpticalFlowAnalyzer(sample_interval=1)
        self._classifier: Any = None
        self._model_path = (
            Path(model_path).expanduser().resolve()
            if model_path
            else self.root / "models" / "aslakey_shot_scale"
        )

    def analyze_shot_type(self, image_path: str) -> dict[str, Any]:
        return dict(self._load_classifier().analyze(image_path))

    def analyze_movement(self, image_paths: list[str]) -> dict[str, Any]:
        import cv2

        frames = []
        for value in image_paths:
            frame = cv2.imread(str(value))
            if frame is not None:
                frames.append(frame)
        if len(frames) < 2:
            return {
                "\u955c\u5934\u8fd0\u52a8": "\u672a\u77e5",
                "confidence": 0.0,
                "reason": "fewer than two decodable frames",
            }
        return dict(self._flow.analyze_frames(frames))

    def _load_classifier(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        if not self._model_path.is_dir():
            raise FileNotFoundError(
                "Auto Scenes DINO shot-scale model is not installed; "
                "automatic download is disabled"
            )
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Auto Scenes DINO shot-scale requires CUDA; "
                "CPU fallback is disabled"
            )
        from A_coreUtils.aftertreatment.shot_type_classifier import (
            DinoV2ShotClassifier,
        )

        classifier = DinoV2ShotClassifier(model_path=str(self._model_path))
        classifier.load()
        if "cuda" not in str(getattr(classifier, "_device", "")).lower():
            classifier.unload()
            raise RuntimeError(
                "Auto Scenes DINO did not load on CUDA; "
                "CPU fallback is disabled"
            )
        self._classifier = classifier
        return classifier

    def close(self) -> None:
        if self._classifier is not None:
            self._classifier.unload()
            self._classifier = None


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "cannot verify source commit")
    return result.stdout.strip()
