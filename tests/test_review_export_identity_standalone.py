"""Standalone synthetic checks for the review-export identity module.

Scope and honesty boundaries:
- Every root, identity, transcript edit, decision and media value here is synthetic.
  No real approval, clinical, customer or business outcome is encoded or implied.
- These checks cover the existing standalone ``review_export_context`` module and
  its public ``storage.py`` read/write helpers in a minimal checkout. They do not
  cover browser/media/end-to-end pipelines, a consumer bridge/editor, or a real
  human review.
- The module records real runtime paths when it resolves a root; captured values
  can differ per machine. This is not sanitized public runtime output and gives
  no OS sandbox guarantee.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import review_export_context as rec


IDENTITY_A = {
    "topic_title": "合成选题甲",
    "video_title": "合成视频甲",
    "video_id": "synthetic-video-a",
}
IDENTITY_B = {
    "topic_title": "Synthetic topic B",
    "video_title": "Synthetic video B",
    "video_id": "synthetic-video-b",
}
REQUIRED_FIELDS = ["topic_title", "video_title", "video_id"]
CONTEXT_KEYS = {"schema", "topic_title", "video_title", "video_id", "time_zone"}


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized(identity: dict[str, str]) -> dict[str, str]:
    return {**identity, "time_zone": "Asia/Shanghai"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_operator_root(base: Path, name: str) -> tuple[Path, dict[str, str]]:
    """Create a synthetic operator root with decision and media placeholders."""
    root = base / name
    root.mkdir(parents=True)
    decisions = root / "review-decisions.json"
    decisions.write_bytes(
        json.dumps(
            {
                "schema": "synthetic.review-decisions.v1",
                "decisions": [{"clip": "clip-001", "approved": False}],
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )
    media = root / "media" / "synthetic-source.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"synthetic-media-placeholder:not-a-real-video")
    before = {
        "review-decisions.json": _sha256(decisions),
        "media/synthetic-source.mp4": _sha256(media),
    }
    return root, before


def _assert_placeholders_unchanged(root: Path, before: dict[str, str]) -> None:
    for relative, digest in before.items():
        path = root / relative
        assert path.is_file(), f"placeholder removed: {relative}"
        assert _sha256(path) == digest, f"placeholder bytes changed: {relative}"


def _cli_configure(root: Path, identity: dict[str, str]) -> int:
    return rec.main(
        [
            "configure",
            str(root),
            "--topic-title",
            identity["topic_title"],
            "--video-title",
            identity["video_title"],
            "--video-id",
            identity["video_id"],
        ]
    )


def test_configure_and_status_two_synthetic_roots(tmp_path, capsys):
    root_a, before_a = _make_operator_root(tmp_path, "operator-a")
    root_b, before_b = _make_operator_root(tmp_path, "operator-b")

    assert _cli_configure(root_a, IDENTITY_A) == 0
    configure_a = json.loads(capsys.readouterr().out)
    assert configure_a["status"] == "configured"
    assert configure_a["review_decisions_modified"] is False
    assert configure_a["project_identity"] == _normalized(IDENTITY_A)
    stored_a = _read_json(root_a / rec.CONTEXT_FILENAME)
    assert stored_a["schema"] == "vkp.review-export-context.v1"
    assert set(stored_a) == CONTEXT_KEYS
    assert stored_a["time_zone"] == "Asia/Shanghai"

    assert _cli_configure(root_b, IDENTITY_B) == 0
    capsys.readouterr()
    assert rec.main(["status", str(root_a)]) == 0
    status_a = json.loads(capsys.readouterr().out)
    assert status_a["status"] == "configured"
    assert status_a["project_identity"] == _normalized(IDENTITY_A)
    assert status_a["required_fields"] == REQUIRED_FIELDS

    assert rec.main(["status", str(root_b)]) == 0
    status_b = json.loads(capsys.readouterr().out)
    assert status_b["project_identity"] == _normalized(IDENTITY_B)

    # status is read-only: same files and bytes before and after
    snapshot = {
        path.relative_to(root_a).as_posix(): _sha256(path)
        for path in root_a.rglob("*")
        if path.is_file()
    }
    assert rec.main(["status", str(root_a)]) == 0
    capsys.readouterr()
    after = {
        path.relative_to(root_a).as_posix(): _sha256(path)
        for path in root_a.rglob("*")
        if path.is_file()
    }
    assert after == snapshot

    _assert_placeholders_unchanged(root_a, before_a)
    _assert_placeholders_unchanged(root_b, before_b)


def test_missing_identity_vs_explicit_empty_claim(tmp_path, capsys):
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    assert rec.configured_identity(missing_root) is None
    assert rec.resolve_review_export_context({}, root=missing_root) is None
    assert rec.main(["status", str(missing_root)]) == 0
    missing = json.loads(capsys.readouterr().out)
    assert missing["status"] == "missing_project_identity"
    assert missing["project_identity"] is None
    assert missing["required_fields"] == REQUIRED_FIELDS

    # An empty dict is a claim: it must not fall through to the other manifest key.
    claimed_root = tmp_path / "claimed"
    claimed_root.mkdir()
    _write_json(
        claimed_root / "manifest.json",
        {"project_identity": {}, "review_export_context": IDENTITY_A},
    )
    assert rec.configured_identity(claimed_root) == {}
    resolved = rec.resolve_review_export_context({}, root=claimed_root)
    assert resolved is not None
    assert resolved["schema"] == "vkp.review-export-context.v1"
    assert resolved["topic_title"] == ""
    assert resolved["video_title"] == ""
    assert resolved["video_id"] == ""
    assert resolved["time_zone"] == "Asia/Shanghai"

    # An inline empty claim also stops precedence instead of using a lower layer.
    inline = rec.resolve_review_export_context(
        {"review_export_context": {}, "project_identity": IDENTITY_B}
    )
    assert inline["topic_title"] == ""
    assert inline["video_id"] == ""

    # status reports the incomplete claim as a configuration error, not as missing.
    with pytest.raises(ValueError) as excinfo:
        rec.main(["status", str(claimed_root)])
    assert "缺少选题、视频名称或视频标识" in str(excinfo.value)


def test_required_identity_and_timezone_errors(tmp_path):
    invalid_claims = [
        {},
        {"topic_title": " synthetic topic ", "video_title": "video"},
        {"topic_title": "synthetic topic", "video_id": "video"},
        {"video_title": "video", "video_id": "video"},
        {"topic_title": "   ", "video_title": "video", "video_id": "video"},
        {"topic_title": "topic", "video_title": "video", "video_id": 7},
    ]
    for claim in invalid_claims:
        with pytest.raises(ValueError) as excinfo:
            rec.normalize_project_identity(claim)
        assert "缺少选题、视频名称或视频标识" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        rec.normalize_project_identity({**IDENTITY_A, "time_zone": "UTC"})
    assert "北京时间" in str(excinfo.value)

    normalized = rec.normalize_project_identity(
        {**IDENTITY_A, "topic_title": "  Synthetic topic  "}
    )
    assert normalized["topic_title"] == "Synthetic topic"
    assert normalized["time_zone"] == "Asia/Shanghai"

    root = tmp_path / "operator"
    root.mkdir()
    with pytest.raises(ValueError) as excinfo:
        rec.main(
            [
                "configure",
                str(root),
                "--topic-title",
                " ",
                "--video-title",
                "synthetic video",
                "--video-id",
                "synthetic-video",
            ]
        )
    assert "缺少选题、视频名称或视频标识" in str(excinfo.value)
    assert not (root / rec.CONTEXT_FILENAME).exists()


def test_allow_incomplete_differs_from_full_validation(tmp_path):
    assert rec.normalize_project_identity({}, allow_incomplete=True) == {
        "topic_title": "",
        "video_title": "",
        "video_id": "",
        "time_zone": "Asia/Shanghai",
    }
    with pytest.raises(ValueError):
        rec.normalize_project_identity({})

    partial = rec.normalize_project_identity(
        {"topic_title": "  仅选题  ", "video_title": "", "video_id": ""},
        allow_incomplete=True,
    )
    assert partial["topic_title"] == "仅选题"
    assert partial["video_title"] == ""

    # allow_incomplete never relaxes the Beijing-time contract
    with pytest.raises(ValueError) as excinfo:
        rec.normalize_project_identity({"time_zone": "UTC"}, allow_incomplete=True)
    assert "北京时间" in str(excinfo.value)

    # resolve displays an incomplete claim; preparation still requires full identity
    resolved = rec.resolve_review_export_context({"review_export_context": {"topic_title": "只有选题"}})
    assert resolved["topic_title"] == "只有选题"
    assert resolved["video_id"] == ""
    with pytest.raises(ValueError):
        rec.identity_for_preparation({"project_identity": {"topic_title": "只有选题"}}, tmp_path, tmp_path)


def test_malformed_claimed_shapes_raise(tmp_path):
    manifest_root = tmp_path / "manifest-claim"
    manifest_root.mkdir()
    _write_json(manifest_root / "manifest.json", {"project_identity": ["not", "an", "object"]})
    assert rec.configured_identity(manifest_root) == ["not", "an", "object"]
    with pytest.raises(ValueError) as excinfo:
        rec.resolve_review_export_context({}, root=manifest_root)
    assert "项目导出配置必须是JSON对象" in str(excinfo.value)

    # Inline non-object claims fail instead of being dropped or ignored.
    for claimed in ("text-claim", ["claim"], 7):
        with pytest.raises(ValueError) as excinfo:
            rec.resolve_review_export_context({"project_identity": claimed}, root=tmp_path)
        assert "项目导出配置必须是JSON对象" in str(excinfo.value)

    # A malformed on-disk context file raises instead of being treated as missing.
    context_root = tmp_path / "context-file"
    context_root.mkdir()
    (context_root / rec.CONTEXT_FILENAME).write_text(
        '["synthetic", "array"]', encoding="utf-8"
    )
    with pytest.raises(ValueError) as excinfo:
        rec.configured_identity(context_root)
    assert "项目导出配置必须是JSON对象" in str(excinfo.value)

    broken_root = tmp_path / "broken-json"
    broken_root.mkdir()
    (broken_root / rec.CONTEXT_FILENAME).write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        rec.configured_identity(broken_root)


def test_inline_config_precedence_and_root_discovery(tmp_path):
    root = tmp_path / "operator"
    root.mkdir()
    _write_json(
        root / rec.CONTEXT_FILENAME,
        {"schema": "vkp.review-export-context.v1", **_normalized(IDENTITY_B)},
    )

    layered = rec.resolve_review_export_context(
        {"review_export_context": IDENTITY_A, "project_identity": IDENTITY_B},
        root=root,
    )
    assert layered["video_id"] == IDENTITY_A["video_id"]

    project_only = rec.resolve_review_export_context({"project_identity": IDENTITY_B}, root=root)
    assert project_only["video_id"] == IDENTITY_B["video_id"]

    from_root = rec.resolve_review_export_context({}, root=root)
    assert from_root["video_id"] == IDENTITY_B["video_id"]

    explicit = {**IDENTITY_A, "video_id": "explicit-video"}
    from_explicit = rec.resolve_review_export_context(
        {"review_export_context": IDENTITY_A}, explicit=explicit, root=root
    )
    assert from_explicit["video_id"] == "explicit-video"

    # bundle_dir discovery resolves the operator root from the context itself
    stored = rec.resolve_review_export_context({"bundle_dir": str(root)})
    assert stored["video_id"] == IDENTITY_B["video_id"]
    assert Path(stored["bundle_dir"]) == root.resolve()

    # media fallback becomes a source catalog entry; it asserts no approval
    with_media = rec.resolve_review_export_context(
        {"review_export_context": IDENTITY_A, "media": {"path": "media/synthetic.mp4"}}
    )
    assert with_media["sources"] == [
        {"source_id": "source-001", "path": "media/synthetic.mp4", "file": "synthetic.mp4"}
    ]


def test_identity_for_preparation_precedence(tmp_path):
    input_root = tmp_path / "input-root"
    output_root = tmp_path / "output-root"
    input_root.mkdir()
    output_root.mkdir()
    _write_json(
        input_root / rec.CONTEXT_FILENAME,
        {"schema": "vkp.review-export-context.v1", **_normalized(IDENTITY_A)},
    )
    _write_json(
        output_root / rec.CONTEXT_FILENAME,
        {"schema": "vkp.review-export-context.v1", **_normalized(IDENTITY_B)},
    )

    assert (
        rec.identity_for_preparation({}, input_root, output_root)["video_id"]
        == IDENTITY_B["video_id"]
    )
    explicit = {**IDENTITY_A, "video_id": "explicit-video"}
    assert (
        rec.identity_for_preparation(
            {"review_export_context": IDENTITY_B},
            input_root,
            output_root,
            explicit=explicit,
        )["video_id"]
        == "explicit-video"
    )

    raw_project = {"project_identity": IDENTITY_A, "review_export_context": IDENTITY_B}
    assert (
        rec.identity_for_preparation(raw_project, input_root, output_root)["video_id"]
        == IDENTITY_A["video_id"]
    )
    assert (
        rec.identity_for_preparation(
            {"review_export_context": IDENTITY_A}, input_root, output_root
        )["video_id"]
        == IDENTITY_A["video_id"]
    )

    only_input = tmp_path / "only-input"
    no_config = tmp_path / "no-config"
    only_input.mkdir()
    no_config.mkdir()
    _write_json(
        only_input / rec.CONTEXT_FILENAME,
        {"schema": "vkp.review-export-context.v1", **_normalized(IDENTITY_A)},
    )
    assert (
        rec.identity_for_preparation({}, only_input, no_config)["video_id"]
        == IDENTITY_A["video_id"]
    )
    assert rec.identity_for_preparation({}, no_config, no_config) is None

    # An explicitly empty claim does not fall through to a root configuration.
    with pytest.raises(ValueError):
        rec.identity_for_preparation({"project_identity": {}}, only_input, only_input)


def test_stale_pack_binding_raises_and_matching_pack_is_kept():
    stale = {
        "pack_payload_sha256": "pack-current",
        "review_export_context": {**IDENTITY_A, "pack_payload_sha256": "pack-stale"},
    }
    with pytest.raises(ValueError) as excinfo:
        rec.resolve_review_export_context(stale)
    assert "review_export_context_pack_changed" in str(excinfo.value)

    matched = rec.resolve_review_export_context(
        {
            "pack_payload_sha256": "pack-same",
            "review_export_context": {**IDENTITY_A, "pack_payload_sha256": "pack-same"},
        }
    )
    assert matched["pack_payload_sha256"] == "pack-same"

    # A candidate without a recorded pack binding accepts the current pack hash.
    unbound = rec.resolve_review_export_context(
        {"pack_payload_sha256": "pack-current", "review_export_context": IDENTITY_A}
    )
    assert unbound["pack_payload_sha256"] == "pack-current"


def test_stale_chapter_source_binding_raises_and_matching_binding_is_kept():
    context = {
        "transcript_edit": {
            "chapters": [
                {"number": 1, "source_text_hash": "hash-1", "file": "sources/one.txt"},
            ]
        }
    }
    with pytest.raises(ValueError) as excinfo:
        rec.resolve_review_export_context(
            {
                **context,
                "review_export_context": {
                    **IDENTITY_A,
                    "source_hashes": {1: "hash-stale"},
                },
            }
        )
    assert "导出命名上下文与当前视频来源不匹配" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        rec.resolve_review_export_context(
            {
                **context,
                "review_export_context": {
                    **IDENTITY_A,
                    "source_files": {1: "sources/stale.txt"},
                },
            }
        )
    assert "导出命名上下文与当前视频来源不匹配" in str(excinfo.value)

    matched = rec.resolve_review_export_context(
        {
            **context,
            "review_export_context": {
                **IDENTITY_A,
                "source_hashes": {1: "hash-1"},
                "source_files": {1: "sources/one.txt"},
            },
        }
    )
    assert matched["source_hashes"] == {1: "hash-1"}
    assert matched["source_files"] == {1: "sources/one.txt"}


def test_write_bound_context_persists_only_resolvable_identity(tmp_path):
    root = tmp_path / "operator"
    root.mkdir()
    context = {
        "review_export_context": IDENTITY_A,
        "pack_payload_sha256": "pack-bound",
        "media": {"path": "media/synthetic.mp4"},
    }
    written = rec.write_bound_review_export_context(root, context)
    assert written is not None
    path = root / rec.CONTEXT_FILENAME
    assert path.is_file()
    assert _read_json(path) == written
    assert written["pack_payload_sha256"] == "pack-bound"
    assert written["sources"][0]["path"] == "media/synthetic.mp4"

    reloaded = rec.resolve_review_export_context({}, root=root)
    assert reloaded["video_id"] == IDENTITY_A["video_id"]

    empty_root = tmp_path / "no-identity"
    empty_root.mkdir()
    assert rec.write_bound_review_export_context(empty_root, {}) is None
    assert not (empty_root / rec.CONTEXT_FILENAME).exists()


def test_configure_overwrites_existing_context_without_touching_operator_files(tmp_path, capsys):
    root, before = _make_operator_root(tmp_path, "operator")
    _write_json(
        root / rec.CONTEXT_FILENAME,
        {"schema": "old.synthetic.schema", "topic_title": "旧合成配置"},
    )
    _write_json(root / "unrelated.json", {"schema": "synthetic.unrelated.v1", "keep": True})
    unrelated_before = _sha256(root / "unrelated.json")

    assert _cli_configure(root, IDENTITY_A) == 0
    capsys.readouterr()
    stored = _read_json(root / rec.CONTEXT_FILENAME)
    assert stored["schema"] == "vkp.review-export-context.v1"
    assert stored["video_id"] == IDENTITY_A["video_id"]
    assert set(stored) == CONTEXT_KEYS
    _assert_placeholders_unchanged(root, before)
    assert _sha256(root / "unrelated.json") == unrelated_before

    # Legacy filename is read by status when the current context file is absent.
    legacy_root, _ = _make_operator_root(tmp_path, "legacy")
    _write_json(
        legacy_root / rec.LEGACY_FILENAME,
        {"schema": "vkp.review-export-context.v1", **_normalized(IDENTITY_B)},
    )
    assert rec.main(["status", str(legacy_root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "configured"
    assert status["project_identity"] == _normalized(IDENTITY_B)

    # The current context filename takes precedence over the legacy filename.
    _write_json(
        legacy_root / rec.CONTEXT_FILENAME,
        {"schema": "vkp.review-export-context.v1", **_normalized(IDENTITY_A)},
    )
    assert rec.main(["status", str(legacy_root)]) == 0
    assert (
        json.loads(capsys.readouterr().out)["project_identity"]["video_id"]
        == IDENTITY_A["video_id"]
    )

    # A non-directory operator root is rejected before any write.
    file_root = tmp_path / "not-a-directory.txt"
    file_root.write_text("synthetic", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        rec.main(["status", str(file_root)])
    assert "项目目录不存在" in str(excinfo.value)


def test_project_root_and_source_catalog_edges(tmp_path):
    assert rec.project_root({}) is None
    assert rec.project_root({"bundle_dir": ""}) is None
    assert rec.project_root({"bundle_dir": str(tmp_path)}) == tmp_path.resolve()
    assert rec.project_root({"artifacts": {"bundle_dir": str(tmp_path)}}) == tmp_path.resolve()
    explicit_root = rec.project_root(
        {
            "bundle_dir": str(tmp_path / "a"),
            "artifacts": {"bundle_dir": str(tmp_path / "b")},
        }
    )
    assert explicit_root == (tmp_path / "a").resolve()

    catalog = rec.source_catalog(
        {
            "sources": [
                {"path": "media/one.mp4"},
                {"source_id": "src-2", "source_path": "media/two.mp4", "sha256": "hash-2"},
                {"video_id": "video-3", "label": "three.mp4", "source_sha256": "hash-3"},
                "ignored-non-object",
                {"note": "no path or label or digest"},
            ]
        }
    )
    assert catalog == [
        {"source_id": "source-001", "path": "media/one.mp4", "file": "one.mp4"},
        {"source_id": "src-2", "path": "media/two.mp4", "file": "two.mp4", "sha256": "hash-2"},
        {"source_id": "video-3", "file": "three.mp4", "sha256": "hash-3"},
    ]
    assert rec.source_catalog({"media": {"path": "media/fallback.mp4"}}) == [
        {"source_id": "source-001", "path": "media/fallback.mp4", "file": "fallback.mp4"}
    ]
    assert rec.source_catalog({"sources": [], "media": {"path": "media/x.mp4"}}) == [
        {"source_id": "source-001", "path": "media/x.mp4", "file": "x.mp4"}
    ]
    assert rec.source_catalog({"sources": {"path": "not-a-list"}}) == []
