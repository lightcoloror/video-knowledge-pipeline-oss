"""Synthetic-only coverage tests; working lyric literal replaced for public distribution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from video_knowledge_pipeline.subtitle_coverage import audit_subtitle_coverage, observations_from_asr, check_subtitles


class SubtitleCoverageTests(unittest.TestCase):
    def setUp(self):
        self.request = {"source_sha256": "a" * 64, "duration_seconds": 10,
                        "asr": {"execution_status": "completed", "coverage_quality": "adequate"},
                        "observations": [{"start_seconds": 1, "end_seconds": 3, "text": "hello",
                                          "kind": "speech", "evidence_ref": "synthetic-ASR"}]}
        self.pack = {"requests": [{"review_id": "v3", "artifact_sha256": "a" * 64,
                                   "full_source_review": True, "review_start_seconds": 0,
                                   "review_end_seconds": 10, "subtitle_cues": [
                                       {"start_seconds": 1, "end_seconds": 3, "text": "hello"}]}]}

    def audit(self, review=None):
        return audit_subtitle_coverage(self.request, self.pack, source_hash="a" * 64, review=review)

    def receipt(self, report):
        return {"input_fingerprint": report["input_fingerprint"], "source_sha256": "a" * 64,
                "review_id": "v3", "full_timeline_listened": True, "full_timeline_viewed": True,
                "reviewer": "synthetic tester", "reviewed_at": "2026-09-25T20:00:00+08:00", "decision": "pass",
                "resolutions": {x["finding_id"]: {"decision": "accepted_after_listening", "reason": "fixture review"}
                                for x in report["findings"]}}

    def test_mechanical_coverage_never_fabricates_human_pass(self):
        report = self.audit()
        self.assertFalse(report["render_ready"])
        self.assertTrue(self.audit(self.receipt(report))["render_ready"])

    def test_tail_missing_and_many_small_gaps_are_not_hidden(self):
        self.pack["requests"][0]["subtitle_cues"][0]["end_seconds"] = 2.9
        self.assertIn("audio_without_caption_or_reason", [x["code"] for x in self.audit()["findings"]])

    def test_omission_needs_reason_and_evidence(self):
        self.pack["requests"][0]["subtitle_cues"] = []
        self.request["omissions"] = [{"start_seconds": 1, "end_seconds": 3,
                                     "reason": "already shown in title", "evidence_ref": "title-review"}]
        self.assertEqual(self.audit()["findings"], [])
        self.request["omissions"][0]["reason"] = ""
        with self.assertRaises(ValueError):
            self.audit()

    def test_long_wrong_subtitle_does_not_prove_text_correspondence(self):
        self.pack["requests"][0]["subtitle_cues"] = [{"start_seconds": 0, "end_seconds": 10, "text": "wrong"}]
        self.assertIn("text_correspondence_review", [x["code"] for x in self.audit()["findings"]])

    def test_runtime_success_empty_asr_stays_unverified(self):
        self.request["asr"]["coverage_quality"] = "insufficient"
        self.request["observations"] = []
        codes = {x["code"] for x in self.audit()["findings"]}
        self.assertEqual(codes, {"asr_coverage_unverified", "no_audio_observations"})

    def test_stale_draft_or_changed_omission_invalidates_review(self):
        receipt = self.receipt(self.audit())
        receipt["review_id"] = "v2"
        self.assertFalse(self.audit(receipt)["render_ready"])
        receipt = self.receipt(self.audit())
        self.request["omissions"] = [{"start_seconds": 4, "end_seconds": 5,
                                     "reason": "laughter", "evidence_ref": "review"}]
        self.assertFalse(self.audit(receipt)["human_review_valid"])

    def test_wrong_source_and_invalid_intervals_fail(self):
        self.request["source_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.audit()
        self.request["source_sha256"] = "a" * 64
        self.pack["requests"][0]["subtitle_cues"][0]["end_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            self.audit()

    def test_lyrics_preserved_despite_asr_conflict(self):
        self.pack["requests"][0]["subtitle_cues"][0]["text"] = "完全虚构合成歌词甲乙丙"
        before = copy.deepcopy(self.pack)
        report = self.audit()
        self.assertEqual(before, self.pack)
        self.assertFalse(report["render_ready"])
        self.assertEqual(report["subtitle_cues_unchanged"], before["requests"][0]["subtitle_cues"])
        conflict = next(x for x in report["findings"] if x["code"] == "text_correspondence_review")
        receipt = self.receipt(report)
        receipt["resolutions"].pop(conflict["finding_id"])
        self.assertFalse(self.audit(receipt)["render_ready"])
        receipt["resolutions"][conflict["finding_id"]] = {
            "decision": "accepted_after_listening", "reason": "Synthetic lyric was explicitly retained"}
        final = self.audit(receipt)
        self.assertTrue(final["render_ready"])
        self.assertEqual(final["subtitle_cues_unchanged"], before["requests"][0]["subtitle_cues"])

    def test_direct_audit_does_not_claim_caller_labels_or_asr_quality_are_verified(self):
        report = self.audit(self.receipt(self.audit()))
        self.assertTrue(report["render_ready"])
        self.assertFalse(report["boundaries"]["evidence_ref_hashes_verified"])
        self.assertFalse(report["boundaries"]["asr_coverage_quality_verified"])
        self.assertFalse(report["boundaries"]["source_file_hash_checked"])
        self.assertFalse(report["boundaries"]["publication_authorized"])

    def test_expected_song_and_outro_missing_from_asr(self):
        self.request["expected_audio"] = [{"start_seconds": 8, "end_seconds": 10, "kind": "singing"}]
        codes = {x["code"] for x in self.audit()["findings"]}
        self.assertIn("expected_audio_not_recognized", codes)
        self.assertIn("expected_audio_without_caption_or_reason", codes)

    def test_punctuation_does_not_span_entire_song(self):
        raw = {"schema": "video_knowledge_funasr_chunked_raw_output.v1", "result": [{
            "words": ["你", "好", "。", "笑"], "timestamp": [[100, 200], [300, 400], [7000, 7100], [9000, 9500]]}]}
        rows = observations_from_asr(raw, evidence_ref="synthetic", duration=10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["end_seconds"], .4)
        self.assertEqual(rows[1]["start_seconds"], 9)

    def test_mismatched_word_timing_cannot_silently_drop_one_of_multiple_records(self):
        raw = {"schema": "video_knowledge_funasr_chunked_raw_output.v1", "result": [
            {"words": ["synthetic", "speech"], "timestamp": [[0, 100]]},
            {"words": ["tail"], "timestamp": [[1000, 1100]]}]}
        with self.assertRaisesRegex(ValueError, "length mismatch without sentence fallback"):
            observations_from_asr(raw, evidence_ref="synthetic", duration=10)

    def test_mismatched_word_timing_retains_a_valid_sentence_fallback(self):
        raw = {"schema": "video_knowledge_funasr_chunked_raw_output.v1", "result": [
            {"words": ["synthetic", "speech"], "timestamp": [[0, 100]],
             "sentence_info": [{"text": "synthetic speech", "start": 0, "end": 500}]},
            {"words": ["tail"], "timestamp": [[1000, 1100]]}]}
        rows = observations_from_asr(raw, evidence_ref="synthetic", duration=10)
        self.assertEqual([row["text"] for row in rows], ["synthetic speech", "tail"])
        self.assertEqual(rows[0]["end_seconds"], .5)


class SubtitleCoverageFileBindingTests(unittest.TestCase):
    """Real temporary bytes and hashes; duration probe is explicitly mocked."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = self.root / "synthetic-byte-fixture.bin"
        self.media.write_bytes(b"Synthetic bytes only; not a valid media file")
        self.source_hash = hashlib.sha256(self.media.read_bytes()).hexdigest()
        self.request = {"source_path": str(self.media), "source_sha256": self.source_hash,
            "duration_seconds": 10, "asr": {"execution_status": "completed", "coverage_quality": "adequate"},
            "observations": [{"start_seconds": 1, "end_seconds": 3, "text": "synthetic",
                "kind": "speech", "evidence_ref": "synthetic label"}]}
        self.pack = {"requests": [{"review_id": "synthetic-1", "artifact_sha256": self.source_hash,
            "full_source_review": True, "review_start_seconds": 0, "review_end_seconds": 10,
            "subtitle_cues": [{"start_seconds": 1, "end_seconds": 3, "text": "synthetic"}]}]}
        self.request_path = self.root / "request.json"
        self.pack_path = self.root / "pack.json"
        self.pack_path.write_text(json.dumps(self.pack), encoding="utf-8")

    def call(self, duration=10, **kwargs):
        self.request_path.write_text(json.dumps(self.request), encoding="utf-8")
        with patch("video_knowledge_pipeline.video.probe_video",
                   return_value=SimpleNamespace(duration_seconds=duration)) as probe:
            result = check_subtitles(self.request_path, self.pack_path, **kwargs)
        probe.assert_called_once_with(self.media.resolve())
        return result

    def asr(self):
        self.asr_path = self.root / "asr.json"
        raw = {"schema": "video_knowledge_pipeline.subtitle_asr_observations.v1", "status": "completed",
            "input": str(self.media), "segments": [{"start_seconds": 1, "end_seconds": 3, "text": "synthetic"}]}
        self.asr_path.write_text(json.dumps(raw), encoding="utf-8")
        self.request["asr_binding"] = {"source_sha256": self.source_hash,
            "asr_sha256": hashlib.sha256(self.asr_path.read_bytes()).hexdigest()}
        return raw

    def test_file_entry_checks_bytes_without_fabricating_human_review(self):
        report = self.call()
        self.assertFalse(report["render_ready"])
        self.assertTrue(report["boundaries"]["source_file_hash_checked"])
        self.assertTrue(report["boundaries"]["source_duration_probe_checked"])
        self.assertEqual(report["boundaries"]["declared_evidence_file_hash_checks"], 0)
        self.assertFalse(report["boundaries"]["evidence_ref_hashes_verified"])

    def test_changed_source_bytes_are_rejected(self):
        self.media.write_bytes(b"Changed synthetic bytes")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            self.call()

    def test_changed_duration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "edited source duration mismatch"):
            self.call(duration=11)

    def test_declared_evidence_hash_is_rechecked_each_call(self):
        evidence = self.root / "evidence.txt"
        evidence.write_text("Synthetic evidence bytes", encoding="utf-8")
        self.request["evidence_files"] = [{"path": str(evidence),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]
        self.assertEqual(self.call()["boundaries"]["declared_evidence_file_hash_checks"], 1)
        evidence.write_text("Changed evidence bytes", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evidence file changed"):
            self.call()

    def test_asr_binding_is_checked_but_success_is_not_coverage_quality(self):
        self.asr()
        report = self.call(asr_path=self.asr_path)
        self.assertTrue(report["boundaries"]["asr_source_output_hash_binding_checked"])
        self.assertIn("asr_coverage_unverified", {x["code"] for x in report["findings"]})
        self.assertFalse(report["render_ready"])

    def test_wrong_asr_source_binding_is_rejected(self):
        self.asr()
        self.request["asr_binding"]["source_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "ASR must have exact source/output hash binding"):
            self.call(asr_path=self.asr_path)

    def test_asr_byte_drift_is_rejected_even_when_parsed_json_is_unchanged(self):
        self.asr()
        with self.asr_path.open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "ASR must have exact source/output hash binding"):
            self.call(asr_path=self.asr_path)

    def test_asr_declared_other_input_is_rejected_even_with_matching_bytes_hash(self):
        raw = self.asr()
        raw["input"] = str(self.root / "another-synthetic-source.bin")
        self.asr_path.write_text(json.dumps(raw), encoding="utf-8")
        self.request["asr_binding"]["asr_sha256"] = hashlib.sha256(self.asr_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "ASR belongs to another media path"):
            self.call(asr_path=self.asr_path)


if __name__ == "__main__":
    unittest.main()
