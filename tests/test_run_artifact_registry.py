from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import build_parser, main as cli_main
from video_knowledge_pipeline.run_artifact_registry import build_run_artifact_registry, register_bundle_run


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    (bundle / "artifact.txt").write_text("ok", encoding="utf-8")
    return bundle


def test_register_bundle_run_writes_run_and_registry(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    run = register_bundle_run(
        bundle,
        run_type="ebook_batch",
        run_id="ebook-batch-1",
        status="needs_retry",
        title="ebook batch",
        artifacts=[{"key": "artifact", "path": "artifact.txt"}],
        failed_items=[{"index": 3, "reason": "ocr_text_empty"}],
        retry_command="retry me",
        resource_requirements={"cpu": 2, "gpu": 1, "network": 0},
        write=True,
    )

    assert run["schema"] == "video_knowledge_pipeline.run_artifact.v1"
    assert run["resource_requirements"] == {"cpu": 2, "gpu": 1, "network": 0}
    assert (bundle / "runs" / "ebook-batch-1" / "run.json").exists()
    assert (bundle / "runs" / "ebook-batch-1" / "run.md").exists()
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    assert registry["schema"] == "video_knowledge_pipeline.run_artifact_registry.v1"
    assert registry["run_count"] == 1
    artifact = json.loads((bundle / "runs" / "ebook-batch-1" / "run.json").read_text(encoding="utf-8"))["artifacts"][0]
    assert artifact["bytes"] == 2
    assert len(artifact["sha256"]) == 64
    assert registry["runs"][0]["freshness"]["status"] == "not_recorded"
    assert registry["runs"][0]["failed_count"] == 1
    assert registry["runs"][0]["retry_command"] == "retry me"
    assert registry["runs"][0]["resource_requirements"] == {"cpu": 2, "gpu": 1, "network": 0}
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_artifact_registry_json"] == "run-artifact-registry.json"
    assert manifest["mcp_run_artifact_registry_args"] == "mcp-run-artifact-registry.args.json"


def test_run_artifact_registry_cli_contract(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    register_bundle_run(bundle, run_type="vision_review_queue", run_id="vision-review-queue", status="needs_execution", write=True)

    args = build_parser().parse_args(["run-artifact-registry", str(bundle), "--no-write"])
    assert args.command == "run-artifact-registry"

    result = build_run_artifact_registry(bundle, write=False)
    assert result["run_count"] == 1
    assert result["runs"][0]["run_id"] == "vision-review-queue"

    assert cli_main(["run-artifact-registry", str(bundle), "--no-write"]) == 0


def test_registry_keeps_unchanged_manifest_bytes(tmp_path: Path) -> None:
    from video_knowledge_pipeline.run_artifact_registry import RUN_REGISTRY_MANIFEST_FIELDS
    root = _bundle(tmp_path)
    manifest = root / 'manifest.json'
    raw = json.dumps({'schema': 'lecture_webui_bundle.v1', **RUN_REGISTRY_MANIFEST_FIELDS},
                     ensure_ascii=False, indent=2).encode('utf-8') + b'\n'
    manifest.write_bytes(raw)
    build_run_artifact_registry(root, write=True)
    assert manifest.read_bytes() == raw
