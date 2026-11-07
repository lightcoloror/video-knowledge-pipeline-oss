from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from video_knowledge_pipeline.review_http import build_server


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _browser_path() -> Path:
    configured = str(os.environ.get("VKP_PLAYWRIGHT_BROWSER") or "").strip()
    if not configured:
        pytest.skip("set VKP_PLAYWRIGHT_BROWSER to run the optional local browser regression")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        pytest.skip(f"configured Playwright browser is unavailable: {path}")
    return path


def _bundle(root: Path) -> Path:
    bundle = root / "subtitle-browser-bundle"
    bundle.mkdir(parents=True)
    media = bundle / "synthetic.mp4"
    media.write_bytes(b"synthetic-browser-fixture")
    _write_json(
        bundle / "manifest.json",
        {
            "title": "合成粤语采访",
            "media_path": str(media),
            "media_duration_seconds": 5,
            "normalized_transcript_json": "normalized-transcript.json",
            "mandarin_translated_transcript_json": "mandarin-translated-transcript.json",
        },
    )
    _write_json(
        bundle / "timeline.json",
        [{"index": 0, "start": 0, "end": 2, "transcript": "第一段"}],
    )
    _write_json(
        bundle / "normalized-transcript.json",
        {
            "segments": [
                {
                    "segment_id": "segment-0001",
                    "source_segment_ids": ["raw-1"],
                    "start": 0,
                    "end": 2,
                    "text": "第一段粤语原文",
                    "speaker_global_id": "speaker-global-001",
                },
                {
                    "segment_id": "segment-0002",
                    "source_segment_ids": ["raw-2"],
                    "start": 2,
                    "end": 5,
                    "text": "第二段粤语原文",
                    "speaker_global_id": "speaker-global-002",
                },
            ]
        },
    )
    _write_json(
        bundle / "mandarin-translated-transcript.json",
        {
            "schema": "video_knowledge_pipeline.translated_transcript.v1",
            "segments": [
                {"segment_id": "segment-0001", "text": "第一段普通话"},
                {"segment_id": "segment-0002", "text": "第二段普通话"},
            ],
        },
    )
    return bundle


def test_browser_restores_draft_and_requires_explicit_apply(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    browser_path = _browser_path()
    bundle = _bundle(tmp_path)
    server = build_server(bundle, port=0, csrf_token="browser-token", refresh=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/subtitle-editor"
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=str(browser_path))
            page = browser.new_page()
            page.on("dialog", lambda dialog: dialog.accept())
            page.goto(url, wait_until="domcontentloaded")
            page.locator('.cue[data-idx="0"]').wait_for(state="attached")
            page.locator('.cue[data-idx="0"]').click(force=True)
            page.locator("#cue-panel-text").wait_for(state="visible")
            page.locator("#cue-panel-text").fill("人工修正的粤语原文")
            page.locator("#cue-panel-text").dispatch_event("input")
            page.locator("#vkp-mandarin-text").fill("人工修正的普通话")
            page.locator("#vkp-review-state").wait_for(state="visible")
            sync_api.expect(page.locator("#vkp-review-state")).to_contain_text("草稿")
            assert not (bundle / "subtitle-review-apply-receipt.json").exists()

            page.reload(wait_until="domcontentloaded")
            page.locator('.cue[data-idx="0"]').wait_for(state="attached")
            page.locator('.cue[data-idx="0"]').click(force=True)
            page.locator("#cue-panel-text").wait_for(state="visible")
            assert page.locator("#cue-panel-text").input_value() == "人工修正的粤语原文"
            assert page.locator("#vkp-mandarin-text").input_value() == "人工修正的普通话"

            page.locator("#vkp-apply-review").click()
            sync_api.expect(page.locator("#vkp-review-state")).to_contain_text("已写回 VKP")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert (bundle / "subtitle-review-apply-receipt.json").is_file()


def test_browser_blocks_apply_after_source_revision_changes(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    browser_path = _browser_path()
    bundle = _bundle(tmp_path)
    server = build_server(bundle, port=0, csrf_token="browser-token", refresh=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/subtitle-editor"
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=str(browser_path))
            page = browser.new_page()
            page.on("dialog", lambda dialog: dialog.accept())
            page.goto(url, wait_until="domcontentloaded")
            page.locator('.cue[data-idx="0"]').wait_for(state="attached")
            page.locator('.cue[data-idx="0"]').click(force=True)
            page.locator("#cue-panel-text").wait_for(state="visible")
            transcript = json.loads((bundle / "normalized-transcript.json").read_text(encoding="utf-8"))
            transcript["segments"][0]["text"] = "Bundle 已在页面打开后变化"
            _write_json(bundle / "normalized-transcript.json", transcript)

            page.locator("#vkp-apply-review").click()
            sync_api.expect(page.locator("#vkp-review-state")).to_contain_text("写回失败")
            assert "changed" in page.locator("#vkp-review-state").inner_text().lower()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert not (bundle / "subtitle-review-apply-receipt.json").exists()
