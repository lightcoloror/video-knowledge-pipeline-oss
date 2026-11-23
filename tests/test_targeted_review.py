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
    _video_codec_name,
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
                "decision_labels": {"pass": "孩子", "needs_rework": "爸爸", "uncertain": "同时／不确定"},
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
            self.assertFalse(list(output.glob("*.mp3")))
            self.assertFalse(list(output.glob("*.wav")))
            pack = json.loads((output / PACK_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(pack["requests"][0]["review_access_mode"], "source_time_range")
            self.assertEqual(pack["requests"][0]["review_start_seconds"], 2.0)
            self.assertEqual(pack["requests"][0]["review_end_seconds"], 8.0)
            self.assertEqual(pack["requests"][0]["time_positioning_granularity"], "time_range")
            self.assertEqual(
                pack["requests"][0]["decision_labels"],
                {"pass": "孩子", "needs_rework": "爸爸", "uncertain": "同时／不确定"},
            )
            html = (output / HTML_FILENAME).read_text(encoding="utf-8")
            self.assertIn("到点自动暂停", html)
            self.assertIn("id=\"playbackButton\"", html)
            self.assertIn("▶ 播放审核区间", html)
            self.assertIn('id="fullPlaybackButton"', html)
            self.assertIn("▶ 从头播放全篇", html)
            self.assertIn("function togglePlayback()", html)
            self.assertIn("function toggleFullPlayback()", html)
            self.assertIn("let fullPlaybackMode = false", html)
            self.assertIn("function activeTimelineBounds(row)", html)
            self.assertIn("return {start: 0, end: duration}", html)
            self.assertIn('id="subtitlePreviewOverlay"', html)
            self.assertIn("function updateSubtitlePreview()", html)
            self.assertIn("row.subtitle_cues || []", html)
            self.assertIn("currentTime >= item.start_seconds && currentTime < item.end_seconds", html)
            self.assertIn("overlay.textContent = cues[0]?.text || ''", html)
            self.assertIn("selectRequest(${index}, true)", html)
            self.assertIn("function playFromCurrentPosition(row)", html)
            self.assertIn("若未自动播放，请点“播放审核区间”", html)
            self.assertIn('id="playbackRate" aria-label="播放速度"', html)
            self.assertIn('<option value="0.5">0.5× 慢放</option>', html)
            self.assertIn('<option value="1" selected>1× 正常</option>', html)
            self.assertIn('<option value="2">2×</option>', html)
            self.assertIn("function setPlaybackRate(value)", html)
            self.assertIn("player.playbackRate = playbackRate", html)
            self.assertIn('<video id="player" controls preload="auto"></video>', html)
            self.assertIn("听声音（画面和进度条保留）", html)
            self.assertIn('id="currentTimeReadout"', html)
            timeline_markup = html.split('<div class="review-timeline"', 1)[1].split("</div>", 1)[0]
            self.assertIn('id="currentTimeReadout"', timeline_markup)
            self.assertIn("当前 ${preciseTimecode(player.currentTime)}", html)
            self.assertIn('id="reviewTimeline" type="range"', html)
            self.assertIn("aria-label=\"拖动定位原片审核时间\"", html)
            self.assertIn("function seekReviewRange(value)", html)
            self.assertIn("timeline.max = String(bounds.end)", html)
            self.assertIn("if (fullPlaybackMode) {", html)
            self.assertIn("addEventListener('ended'", html)
            self.assertIn("全篇连续播放中", html)
            self.assertIn("addEventListener('seeked', updateCurrentTimeReadout)", html)
            self.assertIn("addEventListener('timeupdate', event", html)
            self.assertIn("function refreshTimeDisplayWhilePlaying()", html)
            self.assertIn("requestAnimationFrame(tick)", html)
            self.assertIn("addEventListener('ratechange', event", html)
            self.assertNotIn("audio-mask", html)
            self.assertNotIn("video-shell.audio-only", html)
            self.assertIn('preload="auto"', html)
            self.assertIn("function showReviewStart(row, autoplay = false)", html)
            self.assertIn("loadedmetadata', () => showReviewStart(row, autoplay)", html)
            self.assertIn("row.review_end_seconds - 1/(source.fps || 30)", html)
            self.assertIn("保留最后画面", html)
            self.assertIn("if(state.current < 0) selectRequest(0)", html)
            self.assertIn("frame_step", html)
            self.assertIn("同时／不确定", html)
            self.assertIn("重点判断", html)
            self.assertIn("canPlayType", html)
            self.assertIn("HEVC/H.265", html)
            self.assertIn("H.264/AVC 浏览器审核代理", html)
            self.assertIn("审核媒体源", html)
            self.assertIn("句段级定位（词级时间不可用，已回退）", html)
            self.assertIn("video_codec", pack["sources"][0])

    def test_subtitle_preview_cues_follow_candidate_times_without_media_copies(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["requests"][0]["subtitle_cues"] = [
                {
                    "start_seconds": 2.25,
                    "end_seconds": 4.5,
                    "text": "Row, row, row your boat,\n划呀划，划着你的小船，",
                    "speaker": "child",
                    "motion": "bounce",
                    "font_size_1080": 96,
                    "emphasis_reason": "合成测试：强调第一句",
                }
            ]
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output = root / "review"
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                receipt = prepare_targeted_review_pack(source, output)
            pack = json.loads((output / PACK_FILENAME).read_text(encoding="utf-8"))
            cue = pack["requests"][0]["subtitle_cues"][0]
            self.assertEqual(receipt["media_copies_created"], 0)
            self.assertEqual(cue["timecode_start"], "00:02.250")
            self.assertEqual(cue["timecode_end"], "00:04.500")
            self.assertEqual(cue["text"], "Row, row, row your boat,\n划呀划，划着你的小船，")
            for name in ("speaker", "motion", "font_size_1080", "emphasis_reason"):
                self.assertEqual(cue[name], payload["requests"][0]["subtitle_cues"][0][name])
            html = (output / HTML_FILENAME).read_text(encoding="utf-8")
            self.assertIn('id="subtitlePreviewOverlay"', html)
            self.assertIn("function updateSubtitlePreview()", html)
            self.assertFalse(list(output.glob("*.mp4")))

    def test_subtitle_preview_cue_outside_review_range_is_rejected(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["requests"][0]["subtitle_cues"] = [
                {"start_seconds": 1.0, "end_seconds": 3.0, "text": "out of bounds"}
            ]
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                with self.assertRaisesRegex(
                    TargetedReviewError,
                    "subtitle_cue_outside_review_range:review-001:0",
                ):
                    prepare_targeted_review_pack(source, root / "review")

    def test_explicit_full_source_review_allows_single_full_duration_request(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root, lane="subtitle", end=60.0)
            payload = json.loads(source.read_text(encoding="utf-8"))
            request = payload["requests"][0]
            request.update(
                {
                    "review_start_seconds": 0.0,
                    "review_end_seconds": 60.0,
                    "issue_start_seconds": 0.0,
                    "issue_end_seconds": 60.0,
                    "full_source_review": True,
                    "playback_mode": "normal",
                }
            )
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output = root / "review"
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                receipt = prepare_targeted_review_pack(source, output)
            pack = json.loads((output / PACK_FILENAME).read_text(encoding="utf-8"))
            resolved = pack["requests"][0]
            self.assertEqual(receipt["media_copies_created"], 0)
            self.assertEqual(resolved["review_start_seconds"], 0.0)
            self.assertEqual(resolved["review_end_seconds"], 60.0)
            self.assertEqual(resolved["issue_start_seconds"], 0.0)
            self.assertEqual(resolved["issue_end_seconds"], 60.0)
            self.assertEqual(resolved["review_access_mode"], "source_time_range")
            self.assertEqual(resolved["time_positioning_granularity"], "time_range")
            self.assertNotIn("full_source_review", resolved)
            self.assertFalse(list(output.glob("*.mp4")))
            self.assertFalse(list(output.glob("*.mp3")))
            self.assertFalse(list(output.glob("*.wav")))

    def test_full_source_review_requires_single_item_and_entire_source_bounds(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root, lane="subtitle", end=20.0)
            payload = json.loads(source.read_text(encoding="utf-8"))
            request = payload["requests"][0]
            request["full_source_review"] = True
            request["review_start_seconds"] = 0.0
            request["review_end_seconds"] = 20.0
            request["issue_start_seconds"] = 0.0
            request["issue_end_seconds"] = 20.0
            payload["requests"].append({**request, "review_id": "review-002"})
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                with self.assertRaisesRegex(
                    TargetedReviewError,
                    "full_source_review_must_be_single_item",
                ):
                    prepare_targeted_review_pack(source, root / "multiple")

        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root, lane="subtitle", end=20.0)
            payload = json.loads(source.read_text(encoding="utf-8"))
            request = payload["requests"][0]
            request["full_source_review"] = True
            request["review_start_seconds"] = 1.0
            request["review_end_seconds"] = 20.0
            request["issue_start_seconds"] = 1.0
            request["issue_end_seconds"] = 20.0
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                with self.assertRaisesRegex(
                    TargetedReviewError,
                    "full_source_review_must_cover_entire_source:review-001",
                ):
                    prepare_targeted_review_pack(source, root / "partial")

    def test_unmarked_review_range_over_15_seconds_stays_blocked(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root, lane="subtitle", end=20.0)
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                with self.assertRaisesRegex(
                    TargetedReviewError,
                    "targeted_review_range_exceeds_15_seconds:review-001",
                ):
                    prepare_targeted_review_pack(source, root / "review")

    def test_audio_word_time_falls_back_to_sentence_segment(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            request = payload["requests"][0]
            request.pop("review_start_seconds")
            request.pop("review_end_seconds")
            request["audio_time_position"] = {
                "word": {"start_seconds": 4.25, "end_seconds": 4.25},
                "sentence_segment": {"start_seconds": 2.25, "end_seconds": 7.25},
            }
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output = root / "review"
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                receipt = prepare_targeted_review_pack(source, output)
            pack = json.loads((output / PACK_FILENAME).read_text(encoding="utf-8"))
            resolved = pack["requests"][0]
            self.assertEqual(receipt["media_copies_created"], 0)
            self.assertEqual(resolved["review_start_seconds"], 2.25)
            self.assertEqual(resolved["review_end_seconds"], 7.25)
            self.assertEqual(resolved["issue_start_seconds"], 2.25)
            self.assertEqual(resolved["issue_end_seconds"], 7.25)
            self.assertEqual(resolved["time_positioning_granularity"], "sentence_segment")
            self.assertEqual(resolved["time_positioning_fallback_reason"], "word_timestamp_invalid_or_zero_duration")
            self.assertFalse(list(output.glob("*.mp4")))
            self.assertFalse(list(output.glob("*.mp3")))
            html = (output / HTML_FILENAME).read_text(encoding="utf-8")
            self.assertIn("句段级定位（词级时间不可用，已回退）", html)

    def test_valid_word_time_uses_sentence_segment_as_playback_context(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            request = payload["requests"][0]
            request.pop("review_start_seconds")
            request.pop("review_end_seconds")
            request["audio_time_position"] = {
                "word": {"start_seconds": 4.25, "end_seconds": 4.6},
                "sentence_segment": {"start_seconds": 2.25, "end_seconds": 7.25},
            }
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output = root / "review"
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                prepare_targeted_review_pack(source, output)
            pack = json.loads((output / PACK_FILENAME).read_text(encoding="utf-8"))
            resolved = pack["requests"][0]
            self.assertEqual(resolved["review_start_seconds"], 2.25)
            self.assertEqual(resolved["review_end_seconds"], 7.25)
            self.assertEqual(resolved["issue_start_seconds"], 4.25)
            self.assertEqual(resolved["issue_end_seconds"], 4.6)
            self.assertEqual(resolved["time_positioning_granularity"], "word")
            self.assertNotIn("time_positioning_fallback_reason", resolved)

    def test_audio_word_time_without_usable_sentence_fallback_fails_closed(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            request = payload["requests"][0]
            request.pop("review_start_seconds")
            request.pop("review_end_seconds")
            request["audio_time_position"] = {
                "word": {"start_seconds": 4.25, "end_seconds": 4.25},
                "sentence_segment": None,
            }
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                with self.assertRaisesRegex(
                    TargetedReviewError,
                    "audio_position_unavailable_without_sentence_segment",
                ):
                    prepare_targeted_review_pack(source, root / "review")

    def test_independent_review_clip_request_fails_closed(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["requests"][0]["review_access_mode"] = "independent_clip"
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                with self.assertRaisesRegex(
                    TargetedReviewError,
                    "independent_clip_not_supported_by_default:review-001",
                ):
                    prepare_targeted_review_pack(source, root / "review")

    def test_video_codec_probe_records_hevc_for_browser_compatibility(self) -> None:
        with patch(
            "video_knowledge_pipeline.targeted_review.resolve_media_tool",
            return_value="ffprobe.exe",
        ), patch(
            "video_knowledge_pipeline.targeted_review.subprocess.run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout='{"streams":[{"codec_name":"hevc"}]}',
            ),
        ):
            self.assertEqual(_video_codec_name(Path("candidate.mp4")), "hevc")

    def test_custom_decision_labels_require_every_stable_decision_key(self) -> None:
        with self.managed_temp() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["requests"][0]["decision_labels"].pop("uncertain")
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                with self.assertRaisesRegex(TargetedReviewError, "decision_label_invalid:review-001:uncertain"):
                    prepare_targeted_review_pack(source, root / "invalid-labels")

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
