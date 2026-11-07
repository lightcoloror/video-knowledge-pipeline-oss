from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from video_knowledge_pipeline import lecture_package, multimodal_sample_review, path_utils


def test_file_uri_or_empty_matches_pathlib_for_local_paths(tmp_path: Path) -> None:
    local_path = tmp_path / "含 空格's.txt"
    local_path.write_text("fixture", encoding="utf-8")

    assert path_utils.file_uri_or_empty(str(local_path)) == local_path.resolve().as_uri()
    assert path_utils.file_uri_or_empty("") == ""


def test_file_uri_or_empty_preserves_fail_closed_contract() -> None:
    with patch.object(path_utils, "Path", side_effect=OSError("unresolvable")):
        assert path_utils.file_uri_or_empty("bad-path") == ""


def test_review_entrypoints_share_file_uri_owner() -> None:
    assert lecture_package._file_uri is path_utils.file_uri_or_empty
    assert multimodal_sample_review._file_url is path_utils.file_uri_or_empty


def test_file_uri_conversion_has_one_owner() -> None:
    source_root = Path(path_utils.__file__).parent
    implementations = {
        path.name
        for path in source_root.glob("*.py")
        if ".expanduser().resolve().as_uri()" in path.read_text(encoding="utf-8")
    }

    assert implementations == {"path_utils.py"}
