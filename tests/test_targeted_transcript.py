from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_knowledge_pipeline.targeted_transcript import (
    SOURCE_SCHEMA, compose_transcript_draft, normalize_transcript_source,
    render_transcript_editor, save_transcript_draft, load_transcript_draft,
    TranscriptDraftConflict,
)
from video_knowledge_pipeline.targeted_review import prepare_targeted_review_pack, load_targeted_review_bundle
from test_targeted_review import request_pack, fake_probe


def fixture():
    text = "正常错词，继续。"
    sha = hashlib.sha256(text.encode()).hexdigest()
    sources = [{"source_id": "source-001", "sha256": "a" * 64, "label": "candidate.mp4", "duration_seconds": 60}]
    raw = {"schema_version": SOURCE_SCHEMA, "script_reference": "synthetic", "chapters": [{
        "number": "01", "file": "candidate.mp4", "source_sha256": "a" * 64, "source_text_hash": sha,
        "segments": [{"id": "s1", "text": text, "start": 2, "end": 8, "timings": [
            {"start_char": 2, "end_char": 4, "text": "错词", "start": 3, "end": 4, "source_timestamp_index": 2}]}]}]}
    return raw, sources


def review(pack):
    ch = pack["transcript_edit"]["chapters"][0]
    text = ch["segments"][0]["text"]
    return {"pack_payload_sha256": pack["pack_payload_sha256"], "review": {"schema": "rwe.transcript-review.v3", "chapters": [{
        "chapter": "01", "file": ch["file"], "source_hash": ch["source_text_hash"], "original_text": text,
        "doc": {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": text,
                    "marks": [{"type": "sourceOrigin", "attrs": {"sourceId": "01:s1"}}]}]}]},
        "revisions": [{"effect": "transcript_only", "intent": "纠错", "decision": "保留原声", "reason": "识别错字", "evidence": "synthetic", "effective_scope": "01:s1"}]}]}}


class TargetedTranscriptTests(unittest.TestCase):
    def test_export_naming_context_is_rendered_only_for_its_bound_review_pack(self):
        pack = self.pack()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_path = root / 'targeted-transcript-export-context.json'
            context_path.write_text(json.dumps({'pack_payload_sha256': pack['pack_payload_sha256'], 'topic_title': 'fixture-topic'}), encoding='utf-8')
            self.assertIn('fixture-topic', render_transcript_editor(pack, root, 'test-token'))
            context_path.write_text(json.dumps({'pack_payload_sha256': 'wrong', 'topic_title': 'fixture-topic'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'export_context_pack_changed'):
                render_transcript_editor(pack, root, 'test-token')

    def pack(self):
        raw, sources = fixture()
        return {"pack_payload_sha256": "p" * 64, "sources": sources, "transcript_edit": normalize_transcript_source(raw, sources)}

    def test_native_times_are_preserved_and_normalization_is_idempotent(self):
        p = self.pack()
        self.assertEqual(normalize_transcript_source(p["transcript_edit"], p["sources"]), p["transcript_edit"])
        self.assertEqual(p["transcript_edit"]["chapters"][0]["segments"][0]["timings"][0]["start"], 3)

    def test_source_hash_file_and_original_text_must_all_match(self):
        for key, value in [("source_sha256", "b" * 64), ("file", "other.mp4"), ("source_text_hash", "b" * 64)]:
            raw, sources = fixture()
            raw["chapters"][0][key] = value
            with self.assertRaises(ValueError):
                normalize_transcript_source(raw, sources)

    def test_character_ranges_and_times_fail_closed(self):
        for key, value in [("start", float("nan")), ("end", 100), ("end_char", 200), ("text", "别字"), ("start_char", True)]:
            raw, sources = fixture()
            raw["chapters"][0]["segments"][0]["timings"][0][key] = value
            with self.assertRaises(ValueError):
                normalize_transcript_source(raw, sources)

    def test_draft_retains_explicit_review_only_boundary(self):
        p = self.pack()
        result = compose_transcript_draft(p, review(p))
        self.assertEqual(result["status"], "draft_only")
        self.assertFalse(result["media_modified"])
        self.assertFalse(result["publication_allowed"])

    def test_pack_drift_and_changed_original_are_rejected(self):
        p = self.pack()
        r = review(p)
        r["pack_payload_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "pack_changed"):
            compose_transcript_draft(p, r)
        r = review(p)
        r["review"]["chapters"][0]["original_text"] = "changed"
        with self.assertRaisesRegex(ValueError, "original_changed"):
            compose_transcript_draft(p, r)

    def test_changed_doc_is_rejected_even_if_claimed_original_hash_matches(self):
        p = self.pack()
        r = review(p)
        r["review"]["chapters"][0]["doc"]["content"][0]["content"][0]["text"] = "changed"
        with self.assertRaisesRegex(ValueError, "doc_original_changed"):
            compose_transcript_draft(p, r)

    def test_insertions_do_not_change_preserved_original(self):
        p = self.pack()
        r = review(p)
        r["review"]["chapters"][0]["doc"]["content"][0]["content"].append({"type": "text", "text": "更正", "marks": [{"type": "insertion"}]})
        compose_transcript_draft(p, r)

    def test_every_revision_needs_all_five_explanation_fields(self):
        p = self.pack()
        for key in ("intent", "decision", "reason", "evidence", "effective_scope"):
            r = review(p)
            r["review"]["chapters"][0]["revisions"][0][key] = " "
            with self.assertRaisesRegex(ValueError, "explanation_required"):
                compose_transcript_draft(p, r)

    def test_save_restore_and_stale_draft(self):
        p = self.pack()
        with tempfile.TemporaryDirectory() as d:
            save_transcript_draft(Path(d), p, review(p))
            self.assertEqual(load_transcript_draft(Path(d), p)["review"], review(p)["review"])
            p["pack_payload_sha256"] = "new"
            self.assertIsNone(load_transcript_draft(Path(d), p))

    def test_stale_browser_cannot_overwrite_a_newer_draft(self):
        p=self.pack()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);first=save_transcript_draft(root,p,review(p))
            stale=review(p)
            with self.assertRaises(TranscriptDraftConflict):save_transcript_draft(root,p,stale)
            update=review(p);update['base_revision_sha256']=first['revision_sha256'];update['review']['chapters'][0]['revisions'][0]['reason']='newer reason'
            second=save_transcript_draft(root,p,update)
            self.assertNotEqual(second['revision_sha256'],first['revision_sha256'])
            before=(root/'targeted-transcript-review.draft.json').read_bytes()
            with self.assertRaises(TranscriptDraftConflict):save_transcript_draft(root,p,update)
            self.assertEqual((root/'targeted-transcript-review.draft.json').read_bytes(),before)

    def test_concurrent_same_base_saves_allow_only_one_writer(self):
        import concurrent.futures
        import threading
        p=self.pack()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);first=save_transcript_draft(root,p,review(p));barrier=threading.Barrier(2)
            def save(reason):
                r=review(p);r['base_revision_sha256']=first['revision_sha256'];r['review']['chapters'][0]['revisions'][0]['reason']=reason;barrier.wait()
                try:save_transcript_draft(root,p,r);return 'saved'
                except TranscriptDraftConflict:return 'conflict'
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sorted(pool.map(save,['one','two'])),['conflict','saved'])

    def test_editor_bootstrap_uses_scoped_local_assets_and_escapes_script(self):
        p = self.pack()
        p["transcript_edit"]["script_reference"] = "</script><img src=x>"
        with tempfile.TemporaryDirectory() as d:
            html = render_transcript_editor(p, Path(d), "test-token")
        self.assertIn("<\\/script>", html)
        self.assertIn('/targeted-transcript-assets/editor.bundle.js', html)
        self.assertIn('/api/targeted-review/transcript/save', html)
        self.assertNotIn('src="transcript-editor/data.js"', html)

    def test_optional_integration_uses_existing_pack_and_original_media(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            req = request_pack(root)
            payload = json.loads(req.read_text(encoding="utf-8"))
            raw, _ = fixture()
            raw["chapters"][0]["source_sha256"] = hashlib.sha256((root / "candidate.mp4").read_bytes()).hexdigest()
            payload["transcript_edit"] = raw
            req.write_text(json.dumps(payload), encoding="utf-8")
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                receipt = prepare_targeted_review_pack(req, root / "review")
            self.assertEqual(receipt["media_copies_created"], 0)
            _, pack = load_targeted_review_bundle(root / "review")
            self.assertIn("transcript_edit", pack)
            html = (root / "review/targeted-review.html").read_text(encoding="utf-8")
            self.assertIn('id="transcriptReviewFrame"', html)
            self.assertIn('id="exactReviewTab"', html)


if __name__ == "__main__":
    unittest.main()
