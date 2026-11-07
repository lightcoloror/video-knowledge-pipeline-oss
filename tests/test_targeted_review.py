from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from video_knowledge_pipeline.targeted_review import (
    DECISIONS_FILENAME,
    HTML_FILENAME,
    PACK_FILENAME,
    TargetedReviewError,
    _playback_duration_seconds,
    compose_review_receipt,
    load_targeted_review_bundle,
    prepare_targeted_review_pack,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_probe(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        sha256=digest(path), duration_seconds=60.0, width=1080, height=1920, fps=30.0
    )


def request_pack(root: Path, *, lane: str = "audio", end: float = 8.0) -> Path:
    media = root / "candidate.mp4"
    media.write_bytes(b"synthetic-video-source")
    payload = {
        "schema_version": "self-media-human-review-request-pack.v1",
        "scope": "synthetic",
        "requests": [
            {
                "review_id": "review-001",
                "source_path": "candidate.mp4",
                "artifact_sha256": digest(media).upper(),
                "review_start_seconds": 2.0,
                "review_end_seconds": end,
                "review_lane": lane,
                "review_purpose": "确认这个时间范围",
                "playback_mode": "audio_only" if lane == "audio" else "normal",
                "review_access_mode": "source_time_range",
                "decision_options": ["pass", "needs_rework", "uncertain"],
                "status": "pending",
            }
        ],
    }
    path = root / "requests.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TargetedReviewTests(unittest.TestCase):
    def managed_temp(self):
        return tempfile.TemporaryDirectory()

    def test_zero_copy_review_surface(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            output = root / "review"
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                receipt = prepare_targeted_review_pack(source, output)
            self.assertEqual(receipt["media_copies_created"], 0)
            self.assertEqual(receipt["lanes"], ["audio"])
            self.assertTrue((output / PACK_FILENAME).is_file())
            self.assertTrue((output / HTML_FILENAME).is_file())
            self.assertTrue((output / DECISIONS_FILENAME).is_file())
            self.assertFalse(list(output.glob("*.mp4")))
            html = (output / HTML_FILENAME).read_text(encoding="utf-8")
            self.assertIn("到点自动暂停", html)
            self.assertIn("audio-only", html)
            self.assertIn("frame_step", html)

    def test_hash_drift_fails_closed(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["requests"][0]["artifact_sha256"] = "0" * 64
            source.write_text(json.dumps(payload), encoding="utf-8")
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                with self.assertRaisesRegex(TargetedReviewError, "artifact_hash_drift"):
                    prepare_targeted_review_pack(source, root / "bad-hash")

    def test_receipt_binding_and_boundary(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root, lane="compliance")
            output = root / "review"
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                prepare_targeted_review_pack(source, output)
            _, pack = load_targeted_review_bundle(output, verify_sources=False)
            receipt = compose_review_receipt(pack, "tester", [{"review_id": "review-001", "decision": "pass", "note": "ok"}])
            self.assertEqual(receipt["status"], "complete")
            self.assertFalse(receipt["authority_boundary"]["automatic_publish"])
            with self.assertRaisesRegex(TargetedReviewError, "unknown_review_id"):
                compose_review_receipt(pack, "tester", [{"review_id": "wrong", "decision": "pass"}])

    def test_all_review_lanes_share_one_zero_copy_surface(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            lanes = ["audio", "subtitle", "picture", "privacy", "compliance", "publication", "mixed"]
            modes = {
                "audio": "audio_only",
                "picture": "muted",
                "privacy": "frame_step",
            }
            base = payload["requests"][0]
            payload["requests"] = [
                {
                    **base,
                    "review_id": f"review-{index:03d}",
                    "review_lane": lane,
                    "review_purpose": f"核对{lane}",
                    "playback_mode": modes.get(lane, "normal"),
                    "review_start_seconds": float(index),
                    "review_end_seconds": float(index + 5),
                }
                for index, lane in enumerate(lanes, start=1)
            ]
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output = root / "review"
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                receipt = prepare_targeted_review_pack(source, output)
            self.assertEqual(receipt["request_count"], len(lanes))
            self.assertEqual(set(receipt["lanes"]), set(lanes))
            self.assertEqual(receipt["media_copies_created"], 0)

    def test_publication_review_includes_a_longer_container_audio_tail(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root, lane="publication", end=60.2)
            output = root / "review"
            with (
                patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe),
                patch(
                    "video_knowledge_pipeline.targeted_review._playback_duration_seconds",
                    return_value=60.25,
                ),
            ):
                prepare_targeted_review_pack(source, output)
            pack = json.loads((output / PACK_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(pack["sources"][0]["duration_seconds"], 60.25)
            self.assertEqual(pack["requests"][0]["review_end_seconds"], 60.2)

    def test_playback_duration_prefers_the_longer_ffprobe_container_duration(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="60.250000\n")
        with (
            patch(
                "video_knowledge_pipeline.targeted_review.resolve_media_tool",
                return_value="ffprobe.exe",
            ),
            patch(
                "video_knowledge_pipeline.targeted_review.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            duration = _playback_duration_seconds(Path("candidate.mp4"), 60.0)
        self.assertEqual(duration, 60.25)
        self.assertEqual(run.call_args.args[0][-1], "candidate.mp4")

    def test_playback_duration_falls_back_when_ffprobe_is_unavailable(self) -> None:
        with patch(
            "video_knowledge_pipeline.targeted_review.resolve_media_tool",
            return_value="",
        ):
            self.assertEqual(
                _playback_duration_seconds(Path("candidate.mp4"), 60.0),
                60.0,
            )


if __name__ == "__main__":
    unittest.main()
