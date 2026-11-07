from __future__ import annotations

from pathlib import Path

import pytest

from video_knowledge_pipeline.human_keypoint_review import (
    build_human_keypoint_goldset,
)
from video_knowledge_pipeline.review_writeback import (
    apply_review_payload_to_bundle,
)
from video_knowledge_pipeline.smart_summary_codex import (
    _human_key_point_recall,
)
from video_knowledge_pipeline.smart_summary_keypoint_eval import GOLDSET_SCHEMA
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.webui_bridge import refresh_bundle_review_html


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    write_json(
        root / "manifest.json",
        {
            "schema": "lecture_webui_bundle.v1",
            "title": "人工关键点写回测试",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    write_json(
        root / "normalized-transcript.json",
        {
            "segments": [
                {
                    "id": "seg-0001",
                    "start": 10.0,
                    "end": 18.0,
                    "text": "孩子不带身故责任。",
                },
                {
                    "id": "seg-0002",
                    "start": 18.0,
                    "end": 28.0,
                    "text": "顾问次日提供对比方案。",
                },
            ]
        },
    )
    write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 10.0,
                "end": 18.0,
                "transcript": "孩子不带身故责任。",
                "source_segment_ids": ["seg-0001"],
                "needs_human_review": True,
            },
            {
                "index": 2,
                "start": 18.0,
                "end": 28.0,
                "transcript": "顾问次日提供对比方案。",
                "source_segment_ids": ["seg-0002"],
                "needs_human_review": True,
            },
        ],
    )
    return root


def _review(
    timeline_index: int,
    text: str,
    *,
    aliases: list[str] | None = None,
) -> dict[str, object]:
    return {
        "timeline_index": timeline_index,
        "status": "accepted",
        "human_key_point_confirmed": True,
        "human_key_point_text": text,
        "human_key_point_aliases": aliases or [],
    }


def test_confirmed_review_rows_merge_into_source_bound_goldset(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)

    first = build_human_keypoint_goldset(
        root,
        {"reviews": [_review(1, "孩子不带身故责任", aliases=["儿童方案不含身故责任"])]},
        write=True,
    )
    second = build_human_keypoint_goldset(
        root,
        {"reviews": [_review(2, "顾问次日提供对比方案")]},
        write=True,
    )

    assert first["status"] == "written"
    assert second["status"] == "written"
    goldset = read_json(root / "exports" / "human-key-points.json")
    assert goldset["schema"] == GOLDSET_SCHEMA
    assert [row["source_timeline_index"] for row in goldset["key_points"]] == [1, 2]
    assert goldset["key_points"][0]["source_kind"] == "human_confirmed"
    assert goldset["key_points"][0]["evidence_ids"] == [
        "timeline:1",
        "seg-0001",
    ]
    assert goldset["key_points"][0]["time_range"] == (
        "00:00:10.000 - 00:00:18.000"
    )
    assert len(goldset["source_bindings"]) == 2
    assert all(len(row["sha256"]) == 64 for row in goldset["source_bindings"])
    manifest = read_json(root / "manifest.json")
    assert manifest["human_key_points_json"] == "exports/human-key-points.json"


def test_non_keypoint_review_is_a_noop_even_with_unrelated_bad_goldset(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    exports = root / "exports"
    exports.mkdir()
    write_json(exports / "human-key-points.json", {"schema": "legacy.invalid"})

    result = build_human_keypoint_goldset(
        root,
        {"reviews": [{"timeline_index": 1, "status": "accepted"}]},
        write=False,
    )

    assert result["status"] == "not_updated"
    assert result["incoming_confirmed_count"] == 0


def test_claim_scope_survives_existing_review_writeback(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    scope = {'subject': '孩子', 'property': '身故责任', 'negation': '不带'}
    review = dict(_review(1, '孩子不带身故责任'), human_key_point_claim_scope=scope)
    build_human_keypoint_goldset(root, {'reviews': [review]}, write=True)
    build_human_keypoint_goldset(root, {'reviews': [_review(1, '孩子不带身故责任')]}, write=True)
    goldset = read_json(root / 'exports/human-key-points.json')
    assert goldset['key_points'][0]['claim_scope'] == scope
    assert _human_key_point_recall(root, '孩子带身故责任。')['recall'] == 0


def test_invalid_scope_is_rejected_before_goldset_mutation(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    manifest_before = (root / 'manifest.json').read_bytes()
    review = dict(_review(1, '孩子不带身故责任'), human_key_point_claim_scope={'subject': 42})
    with pytest.raises(ValueError, match='claim_scope'):
        build_human_keypoint_goldset(root, {'reviews': [review]}, write=True)
    assert (root / 'manifest.json').read_bytes() == manifest_before
    assert not (root / 'exports/human-key-points.json').exists()


def test_source_bound_reference_drift_is_not_a_completeness_pass(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    build_human_keypoint_goldset(root, {'reviews': [_review(1, '孩子不带身故责任')]}, write=True)
    assert _human_key_point_recall(root, '孩子不带身故责任')['recall'] == 1
    timeline = read_json(root / 'timeline.json')
    timeline[0]['transcript'] = 'synthetic changed evidence'
    write_json(root / 'timeline.json', timeline)
    result = _human_key_point_recall(root, '孩子不带身故责任')
    assert result['evaluated'] is False
    assert result['recall'] is None
    assert result['reference_freshness']['status'] == 'stale'


def test_stale_goldset_rebind_requires_exact_hash_and_all_prior_reviews(tmp_path: Path) -> None:
    from video_knowledge_pipeline.file_hash import sha256_file

    root = _bundle(tmp_path)
    rows = [_review(1, '孩子不带身故责任'), _review(2, '顾问次日提供对比方案')]
    build_human_keypoint_goldset(root, {'reviews': rows}, write=True)
    gold_path = root / 'exports/human-key-points.json'
    prior_hash = sha256_file(gold_path)
    before = gold_path.read_bytes()
    timeline = read_json(root / 'timeline.json')
    timeline[0]['transcript'] = '孩子不带身故责任。补充说明。'
    write_json(root / 'timeline.json', timeline)
    for payload in (
        {'reviews': rows},
        {'reviews': rows, 'human_key_point_rebind_prior_sha256': '0'*64},
        {'reviews': rows[:1], 'human_key_point_rebind_prior_sha256': prior_hash},
    ):
        with pytest.raises(ValueError, match='bindings drifted'):
            build_human_keypoint_goldset(root, payload, write=True)
        assert gold_path.read_bytes() == before
    payload = {'reviews': rows, 'human_key_point_rebind_prior_sha256': prior_hash}
    out = build_human_keypoint_goldset(root, payload, write=True)
    assert out['status'] == 'written'
    rebound = read_json(gold_path)
    snapshot = root / rebound['reference_rebind']['prior_snapshot']
    assert read_json(snapshot)['source_bindings'] != rebound['source_bindings']
    assert _human_key_point_recall(root, '孩子不带身故责任。顾问次日提供对比方案。')['reference_freshness']['status'] == 'fresh'


def test_actual_summary_quality_gate_consumes_scoped_recall_and_drift(tmp_path: Path) -> None:
    from video_knowledge_pipeline.smart_summary_codex import smart_summary_quality_check

    root = _bundle(tmp_path)
    build_human_keypoint_goldset(root, {'reviews': [_review(1, '孩子不带身故责任')]}, write=True)
    summary = root / 'exports/smart-summary.codex.md'
    summary.write_text('## 关键观点\n孩子带身故责任。', encoding='utf-8')
    before = (root / 'timeline.json').read_bytes()
    report = smart_summary_quality_check(root, summary_path=summary, write=False)
    check = next(row for row in report['checks'] if row['key'] == 'human_key_point_recall')
    assert check['passed'] is False
    assert report['quality_metrics']['human_key_point_recall']['recall'] == 0
    assert (root / 'timeline.json').read_bytes() == before
    timeline = read_json(root / 'timeline.json')
    timeline[0]['transcript'] = 'synthetic source drift'
    write_json(root / 'timeline.json', timeline)
    report = smart_summary_quality_check(root, summary_path=summary, write=False)
    assert report['quality_metrics']['human_key_point_recall']['reference_freshness']['status'] == 'stale'
    assert report['quality_metrics']['human_key_point_recall']['evaluated'] is False
    assert report['passed'] is False


def test_confirmed_keypoint_requires_text_and_exact_timeline_binding(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)

    with pytest.raises(ValueError, match="requires human_key_point_text"):
        build_human_keypoint_goldset(
            root,
            {"reviews": [_review(1, "")]},
            write=False,
        )
    with pytest.raises(ValueError, match="no bound timeline item"):
        build_human_keypoint_goldset(
            root,
            {"reviews": [_review(99, "不存在的时间线关键点")]},
            write=False,
        )


def test_existing_review_ui_writes_goldset_and_summary_recall_consumes_it(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    refresh_bundle_review_html(root)
    page = (root / "review.html").read_text(encoding="utf-8")

    assert 'data-review-filter="human-key-point"' in page
    assert "human-key-point-confirmed" in page
    assert "human-key-point-text" in page
    assert "只有人工勾选并正式“保存到 VKP”后" in page

    result = apply_review_payload_to_bundle(
        root,
        {
            "schema": "lecture_review_notes.v1",
            "package_title": "人工关键点写回测试",
            "reviews": [
                _review(
                    1,
                    "孩子不带身故责任",
                    aliases=["儿童方案不含身故责任"],
                )
            ],
        },
        write=True,
        refresh_exports=False,
    )

    assert result["ok"] is True
    assert result["human_key_points"]["status"] == "written"
    timeline = read_json(root / "timeline.json")
    assert timeline[0]["human_review"]["human_key_point_confirmed"] is True
    assert timeline[0]["human_review"]["human_key_point_aliases"] == [
        "儿童方案不含身故责任"
    ]
    recall = _human_key_point_recall(root, "儿童方案不含身故责任。")
    assert recall["evaluated"] is True
    assert recall["recall"] == 1.0
    assert recall["decisions"][0]["method"] == "explicit_alias"
