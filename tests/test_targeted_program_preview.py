from __future__ import annotations
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from video_knowledge_pipeline.targeted_program_preview import FILENAME, load_program_preview, transcript_render_revision
from video_knowledge_pipeline.targeted_review import TargetedReviewHandler


class ContinuousPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'review'
        self.bundle.mkdir()
        self.media = self.root / 'candidate.mp4'
        self.media.write_bytes(b'0123456789')  # Synthetic bytes exercise HTTP ranges, not codec validity.
        self.pack = {'pack_payload_sha256': 'a'*64}
        self.bindings = {}
        for name in ['targeted-review-pack.json', 'targeted-transcript-review.draft.json']:
            (self.bundle / name).write_text('{}', encoding='utf-8')
            self.bindings[name] = hashlib.sha256(b'{}').hexdigest()
        self.config = dict(schema='vkp.targeted-program-preview.v1', pack_payload_sha256='a'*64,
                           artifact_root=str(self.root), video={'path': str(self.media),
                           'sha256': hashlib.sha256(self.media.read_bytes()).hexdigest()},
                           review_bindings=self.bindings, duration_s=10)
        self.write_config()

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self):
        (self.bundle / FILENAME).write_text(json.dumps(self.config), encoding='utf-8')

    def test_absent_optional_binding_keeps_existing_review_available(self):
        (self.bundle / FILENAME).unlink()
        self.assertIsNone(load_program_preview(self.bundle, self.pack))

    def test_changed_saved_revision_and_outside_media_are_rejected(self):
        draft = self.bundle / 'targeted-transcript-review.draft.json'
        draft.write_text('{"changed":true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '修订已变化'):
            load_program_preview(self.bundle, self.pack)
        draft.write_text('{}', encoding='utf-8')
        self.config['artifact_root'] = str(self.bundle)
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'outside'):
            load_program_preview(self.bundle, self.pack)

    def test_changed_media_cannot_be_played_under_old_binding(self):
        self.media.write_bytes(b'replaced')
        with self.assertRaisesRegex(ValueError, '文件已变化'):
            load_program_preview(self.bundle, self.pack)

    def test_export_clock_does_not_expire_render_but_text_edit_does(self):
        path = self.bundle / 'targeted-transcript-review.draft.json'
        review = {'exported_at': 'first', 'chapters': [{'chapter':'01', 'proposed_text':'已核对文字',
                  'preview': {'sections':[{'start':1,'end':2}], 'displaySubtitles':[]}}]}
        self.config['transcript_render_sha256'] = transcript_render_revision(review)
        path.write_text(json.dumps({'review':review}), encoding='utf-8')
        self.write_config()
        review['exported_at'] = 'second'
        path.write_text(json.dumps({'review':review}), encoding='utf-8')
        self.assertIsNotNone(load_program_preview(self.bundle, self.pack))
        review['chapters'][0]['proposed_text'] = '新增了实质修订'
        path.write_text(json.dumps({'review':review}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '修订已变化'):
            load_program_preview(self.bundle, self.pack)

    def test_http_ranges_and_stale_revision_conflict(self):
        class Handler(TargetedReviewHandler):
            def log_message(self, *args): pass
        Handler.bundle_dir = self.bundle
        Handler.pack = self.pack
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:'+str(server.server_address[1])
        try:
            request = Request(base+'/targeted-program-media', headers={'Range':'bytes=2-4'})
            with urlopen(request) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), b'234')
                self.assertEqual(response.headers['Content-Range'], 'bytes 2-4/10')
            with urlopen(base+'/targeted-program-preview') as response:
                self.assertEqual(response.headers['X-Frame-Options'], 'SAMEORIGIN')
                self.assertIn(b'programPlayer', response.read())
            (self.bundle/'targeted-transcript-review.draft.json').write_text('changed', encoding='utf-8')
            with self.assertRaises(HTTPError) as error:
                urlopen(base+'/targeted-program-media')
            self.assertEqual(error.exception.code, 409)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)


if __name__ == '__main__': unittest.main()
