from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from test_targeted_review import fake_probe, request_pack
from video_knowledge_pipeline.targeted_review import (
    PACK_FILENAME,
    TargetedReviewError,
    _schema,
    prepare_targeted_review_pack,
)


class PublicBaseCompatibilityTests(unittest.TestCase):
    """Thin regression tests for the public-era contract kept by the overlay.

    These are deliberately separate from the feature tests: they protect the
    backward base pack, the public schema compatibility and HTML script escaping.
    """

    def run_pack(self, root: Path):
        source = request_pack(root)
        output = root / "review"
        with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
            receipt = prepare_targeted_review_pack(source, output)
        return source, output, receipt

    def test_public_era_pack_without_new_optional_metadata_prepares(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, output, receipt = self.run_pack(root)
            pack = json.loads((output / PACK_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(receipt["media_copies_created"], 0)
            self.assertEqual(receipt["lanes"], ["audio"])
            self.assertNotIn("project_identity", pack)
            self.assertNotIn("transcript_edit", pack)
            self.assertNotIn("subtitle_cues", pack["requests"][0])
            self.assertEqual(pack["requests"][0]["review_access_mode"], "source_time_range")
            self.assertIn("video_codec", pack["sources"][0])

    def test_pack_without_video_codec_remains_schema_valid_for_public_era_packs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, output, _ = self.run_pack(root)
            pack = json.loads((output / PACK_FILENAME).read_text(encoding="utf-8"))
            # Simulate a pack produced by the older public module, whose source
            # rows did not contain video_codec, while keeping payload integrity.
            from video_knowledge_pipeline.targeted_review import PACK_HASH_FIELD, _payload_sha256

            pack["sources"][0].pop("video_codec")
            pack[PACK_HASH_FIELD] = _payload_sha256(pack, PACK_HASH_FIELD)
            errors = list(Draft202012Validator(_schema("targeted-review-pack.v1.schema.json")).iter_errors(pack))
            self.assertEqual(errors, [])

    def test_review_question_script_markup_is_escaped_inside_the_page(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["requests"][0]["review_question"] = "确认</script><img src=x onerror=alert(1)>"
            payload["requests"][0]["review_purpose"] = "确认</script><img src=x onerror=alert(1)>"
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output = root / "review"
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                prepare_targeted_review_pack(source, output)
            html = (output / "targeted-review.html").read_text(encoding="utf-8")
            self.assertIn("<\\/script>", html)
            self.assertNotIn("</script><img", html)

    def test_unsupported_public_request_schema_still_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = request_pack(root)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["schema_version"] = "self-media-unknown-review-pack.v1"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with patch("video_knowledge_pipeline.targeted_review.probe_video", side_effect=fake_probe):
                with self.assertRaisesRegex(TargetedReviewError, "unsupported_review_request_schema"):
                    prepare_targeted_review_pack(source, root / "review")


if __name__ == "__main__":
    unittest.main()
