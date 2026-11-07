from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

from video_knowledge_pipeline.review_http import build_server
from video_knowledge_pipeline.review_writeback import semantic_review_notes_from_payload
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.webui_bridge import refresh_bundle_review_html


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1", "title": "审核写回测试"})
    write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 10.0,
                "end": 18.0,
                "transcript": "请使用MIAAPP中的PDF版本。",
                "quality_issues": ["needs_human_review"],
                "needs_human_review": True,
            }
        ],
    )
    write_json(
        root / "transcript-semantic-correction-pack.json",
        {
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "start": 12.0,
                    "end": 16.0,
                    "time_range": "00:00:12.000 - 00:00:16.000",
                    "original_text": "MIAAPP",
                    "candidate_text": "",
                    "risk_level": "medium",
                    "reason": "proper_noun",
                    "evidence_ids": ["asr_segment_1"],
                }
            ]
        },
    )
    return root


def test_comment_correction_resolves_to_one_exact_semantic_candidate(tmp_path: Path) -> None:
    root = _bundle(tmp_path)

    result = semantic_review_notes_from_payload(
        root,
        {
            "reviews": [
                {
                    "timeline_index": 1,
                    "status": "needs_fix",
                    "tags": ["asr_ocr_error"],
                    "comment": "MIAAPP应为明亚APP",
                }
            ]
        },
    )

    assert result["unresolved"] == []
    assert result["reviews"][0]["candidate_id"] == "semcorr-0001"
    assert result["reviews"][0]["original_text"] == "MIAAPP"
    assert result["reviews"][0]["corrected_text"] == "明亚APP"
    assert result["reviews"][0]["human_confirmed"] is True


def test_review_html_has_autosave_writeback_and_candidate_fields(tmp_path: Path) -> None:
    root = _bundle(tmp_path)

    refresh_bundle_review_html(root)
    page = (root / "review.html").read_text(encoding="utf-8")

    assert "corrected_transcript" in page
    assert "semantic-corrected-text" in page
    assert "semcorr-0001" in page
    assert "草稿已自动保存" in page
    assert "保存到 VKP" in page
    assert "saveReviewToVkp" in page
    assert "vkp_review_draft.v2" in page
    assert "start-review-webui.ps1" in page


def test_loopback_review_server_applies_selected_review_notes(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    server = build_server(root, port=0, csrf_token="test-token", refresh=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urlopen(base + "/api/review/status", timeout=5) as response:
            status = json.loads(response.read().decode("utf-8"))
        body = json.dumps(
            {
                "bundle_revision": status["bundle_revision"],
                "review_notes": {
                    "schema": "lecture_review_notes.v1",
                    "package_title": "审核写回测试",
                    "reviews": [
                        {
                            "timeline_index": 1,
                            "status": "needs_fix",
                            "comment": "保留为待复核备注",
                            "tags": ["missing_info"],
                        }
                    ],
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            base + "/api/review/apply",
            method="POST",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-VKP-Review-Token": "test-token",
                "Origin": base,
            },
        )
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    notes = read_json(root / "review-notes.json")
    timeline = read_json(root / "timeline.json")
    assert notes["reviews"][0]["comment"] == "保留为待复核备注"
    assert timeline[0]["human_review"]["comment"] == "保留为待复核备注"
    assert (root / "review-writeback-report.json").is_file()

def test_loopback_workbench_applies_hash_bound_shot_review(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    write_json(
        root / "exports" / "technical-shot-boundaries.json",
        {
            "schema": "video_knowledge_pipeline.technical_shot_boundaries.v1",
            "status": "completed",
            "ok": True,
            "boundary_kind": "technical_shot",
            "backend": "autoshot",
            "shots": [
                {"shot_id": "technical-shot-0001", "index": 1, "start": 0.0, "end": 10.0},
                {"shot_id": "technical-shot-0002", "index": 2, "start": 10.0, "end": 18.0},
            ],
        },
    )
    server = build_server(root, port=0, csrf_token="shot-token", refresh=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urlopen(base + "/workbench", timeout=15) as response:
            page = response.read().decode("utf-8")
        assert "VKP_SHOT_REVIEW_API" in page
        with urlopen(base + "/api/review/status", timeout=5) as response:
            status = json.loads(response.read().decode("utf-8"))
        workbench = read_json(root / "video-workbench.json")
        notes = workbench["shot_review"]["template"]
        notes["review_status"] = "human_confirmed"
        notes["review_id"] = "loopback-shot-review"
        notes["shots"][0]["end"] = 9.5
        notes["shots"][1]["start"] = 9.5
        body = json.dumps(
            {
                "bundle_revision": status["bundle_revision"],
                "shot_review_notes": notes,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            base + "/api/shot-review/apply",
            method="POST",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-VKP-Review-Token": "shot-token",
                "Origin": base,
            },
        )
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    reviewed = read_json(root / "exports" / "technical-shot-boundaries.reviewed.json")
    assert reviewed["review_id"] == "loopback-shot-review"
    assert reviewed["shots"][0]["end"] == 9.5
    timeline = read_json(root / "timeline.json")
    assert "human_review" not in timeline[0]
