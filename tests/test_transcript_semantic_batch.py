from __future__ import annotations

import hashlib
import json
from pathlib import Path

from video_knowledge_pipeline.cli import main
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.transcript_semantic_batch import transcript_semantic_acceptance, transcript_semantic_batch_acceptance, transcript_semantic_batch_codex_review_draft, transcript_semantic_batch_import_review_notes, transcript_semantic_batch_review_pack, transcript_semantic_repair_queue, transcript_semantic_repair_run
from video_knowledge_pipeline.transcript_semantic_correction import build_transcript_semantic_candidate_discovery_pack, build_transcript_semantic_correction_pack, validate_transcript_semantic_correction


def _llm_summary_text(text: str) -> str:
    return f"生成方式：`codex_llm_rewrite_final`\n\n{text}"

def _write_manifest(bundle: Path, title: str) -> None:
    bundle.mkdir(parents=True)
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1", "title": title, "normalized_transcript_json": "normalized-transcript.json"})
    write_json(bundle / "timeline.json", [])
    write_json(bundle / "normalized-transcript.json", {"segments": []})


def _write_accepted_semantic_bundle(bundle: Path) -> None:
    _write_manifest(bundle, "accepted semantic bundle")
    write_json(bundle / "transcript-semantic-correction-pack.json", {"schema": "pack", "candidate_count": 1, "candidates": [{"candidate_id": "semcorr-0001", "correction_type": "ordinary_word", "risk_level": "medium", "evidence_source_types": ["ocr"]}]})
    accepted_decision = {"candidate_id": "semcorr-0001", "original_text": "play right", "corrected_text": "Playwright", "correction_type": "proper_noun"}
    write_json(bundle / "transcript-semantic-correction-validation.json", {"accepted_decision_count": 1, "review_required_count": 0, "decisions": [accepted_decision], "accepted_decisions": [accepted_decision]})
    write_json(bundle / "transcript-semantic-correction-closure.json", {"status": "completed", "applied_correction_count": 1, "changed_segment_count": 1})
    write_json(bundle / "transcript-semantic-correction-impact-report.json", {"status": "passed", "final_residual_error_total": 0, "accepted_decision_count": 1})
    write_json(bundle / "transcript-semantic-readable-impact-report.json", {"status": "passed", "required_readable_residual_total": 0, "accepted_decision_count": 1})
    write_json(bundle / "transcript-semantic-summary-impact-report.json", {"status": "passed", "summary_residual_original_total": 0, "summary_corrected_hit_total": 1, "summary_absorption_rate": 1.0})
    write_json(bundle / "transcript-quality-gate.json", {"status": "passed", "ok": True, "quality_layers": {"raw_asr_integrity": "passed", "semantic_accuracy": "passed", "readable_projection": "passed"}})
    discovery = build_transcript_semantic_candidate_discovery_pack(bundle, limit=8, cursor=0, write=True)
    discovery_path = bundle / "transcript-semantic-candidate-discovery-pack.json"
    write_json(
        bundle / "runs" / "autopilot-synthetic-semantic-acceptance" / "autopilot-job.json",
        {
            "semantic_candidate_discovery": {
                "transcript_sha256": discovery["transcript_sha256"],
                "last_discovery_pack_sha256": sha256_file(discovery_path),
                "complete": True,
                "next_cursor": None,
                "coverage": discovery["coverage"],
            }
        },
    )
    canonical = bundle / "source-arbitrated-transcript.json"
    write_json(canonical, {"segments": [{"start": 0, "end": 3, "text": "Playwright"}]})
    canonical_hash = hashlib.sha256(canonical.read_bytes()).hexdigest()
    exports = bundle / "exports"
    exports.mkdir(exist_ok=True)
    (exports / "full-transcript.md").write_text(
        f"Canonical source SHA-256: `{canonical_hash}`\n\nPlaywright",
        encoding="utf-8",
    )
    (exports / "knowledge-note.md").write_text(
        f"Canonical transcript SHA-256: `{canonical_hash}`\n\nPlaywright",
        encoding="utf-8",
    )
    (exports / "full-body.md").write_text(
        f"Canonical source SHA-256: `{canonical_hash}`\n\nPlaywright",
        encoding="utf-8",
    )
    write_json(exports / "smart-summary-input-pack.json", {"transcript_source": str(canonical), "transcript_source_sha256": canonical_hash})
    (exports / "smart-summary.md").write_text(_llm_summary_text("Playwright"), encoding="utf-8")


def test_transcript_semantic_acceptance_writes_single_bundle_proof(tmp_path: Path) -> None:
    bundle = tmp_path / "single-proof" / "webui-bundle"
    _write_accepted_semantic_bundle(bundle)

    result = transcript_semantic_acceptance(bundle, write=True)

    assert result["schema"] == "video_knowledge_pipeline.transcript_semantic_acceptance.v1"
    assert result["status"] == "accepted"
    assert result["ok"] is True
    assert result["acceptance_state"] == "accepted"
    assert result["item"]["accepted"] is True
    assert Path(result["json_path"]).exists()
    markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
    assert "Transcript Semantic Correction Acceptance" in markdown
    assert "Read-only report" in markdown

def test_transcript_semantic_acceptance_blocks_canonical_export_hash_mismatch(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "hash-mismatch" / "webui-bundle"
    _write_accepted_semantic_bundle(bundle)
    (bundle / "exports" / "full-transcript.md").write_text(
        "Canonical source SHA-256: `" + ("0" * 64) + "`\n\nPlaywright",
        encoding="utf-8",
    )

    result = transcript_semantic_acceptance(bundle, write=False)

    assert result["ok"] is False
    assert result["acceptance_state"] == "needs_canonical_export_refresh"
    assert result["canonical_transcript_integrity"]["passed"] is False

    queue = transcript_semantic_repair_queue(
        bundle,
        target_bundle_count=1,
        write=False,
    )
    item = queue["items"][0]
    assert item["acceptance_state"] == "needs_canonical_export_refresh"
    assert item["action_key"] == "refresh_exports_or_review"
    assert item["action_kind"] == "local_export_refresh"

    assert result["canonical_transcript_integrity"]["issues"][0]["key"] == (
        "full_transcript_canonical_hash_mismatch"
    )


def test_transcript_semantic_acceptance_cli_writes_report(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "single-cli" / "webui-bundle"
    _write_accepted_semantic_bundle(bundle)
    output_dir = tmp_path / "single-report"

    exit_code = main(["transcript-semantic-acceptance", str(bundle), "--output-dir", str(output_dir)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "accepted"
    assert payload["ok"] is True
    assert read_json(output_dir / "transcript-semantic-acceptance.json")["acceptance_state"] == "accepted"
    assert (output_dir / "transcript-semantic-acceptance.md").exists()

def test_transcript_semantic_batch_acceptance_summarizes_multiple_bundles(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    accepted = root / "one" / "webui-bundle"
    missing = root / "two" / "webui-bundle"
    _write_accepted_semantic_bundle(accepted)
    _write_manifest(missing, "missing semantic pack")

    result = transcript_semantic_batch_acceptance(root, target_bundle_count=2, write=True)

    assert result["status"] == "needs_semantic_correction_action"
    assert result["ok"] is False
    assert result["summary"]["accepted_count"] == 1
    assert result["summary"]["not_accepted_count"] == 1
    assert result["summary"]["by_acceptance_state"]["accepted"] == 1
    assert result["summary"]["by_acceptance_state"]["needs_pack"] == 1
    assert result["items"][0]["accepted"] is True
    assert result["items"][1]["acceptance_state"] == "needs_pack"
    assert Path(result["json_path"]).exists()
    assert Path(result["markdown_path"]).exists()
    assert "transcript-semantic-correction-pack" in "\n".join(result["next_actions"])


def test_transcript_semantic_batch_acceptance_reports_target_count_gap(tmp_path: Path) -> None:
    bundle = tmp_path / "single" / "webui-bundle"
    _write_accepted_semantic_bundle(bundle)

    result = transcript_semantic_batch_acceptance(bundle, target_bundle_count=3, write=False)

    assert result["status"] == "needs_more_bundles_for_batch_acceptance"
    assert result["summary"]["target_bundle_count_met"] is False
    assert result["summary"]["accepted_count"] == 1
    assert "1/3" in result["next_actions"][0]


def test_transcript_semantic_batch_acceptance_cli_writes_report(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "cli" / "webui-bundle"
    _write_accepted_semantic_bundle(bundle)
    output_dir = tmp_path / "reports"

    exit_code = main(["transcript-semantic-batch-acceptance", str(bundle), "--target-bundle-count", "1", "--output-dir", str(output_dir)])

    assert exit_code == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["status"] == "accepted"
    assert payload["ok"] is True
    written = read_json(output_dir / "transcript-semantic-batch-acceptance.json")
    assert written["summary"]["accepted_count"] == 1
    assert (output_dir / "transcript-semantic-batch-acceptance.md").exists()

def test_transcript_semantic_batch_acceptance_limit_bounds_discovered_bundles(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    for idx in range(4):
        _write_manifest(root / f"item-{idx}" / "webui-bundle", f"bundle {idx}")

    result = transcript_semantic_batch_acceptance(root, target_bundle_count=3, limit=2, write=False)

    assert result["bundle_count"] == 2
    assert result["discovered_bundle_count"] == 4
    assert result["limit"] == 2
    assert result["limited"] is True
    assert result["status"] == "needs_more_bundles_for_batch_acceptance"


def test_transcript_semantic_repair_queue_builds_preview_actions(tmp_path: Path) -> None:
    root = tmp_path / "queue"
    accepted = root / "accepted" / "webui-bundle"
    missing = root / "missing" / "webui-bundle"
    prompt_ready = root / "prompt" / "webui-bundle"
    _write_accepted_semantic_bundle(accepted)
    _write_manifest(missing, "missing semantic pack")
    _write_manifest(prompt_ready, "prompt ready bundle")
    write_json(prompt_ready / "transcript-semantic-correction-pack.json", {"schema": "pack", "candidate_count": 1, "candidates": [{"candidate_id": "semcorr-0001", "correction_type": "proper_noun", "risk_level": "medium"}]})
    (prompt_ready / "transcript-semantic-correction-llm-prompt.md").write_text("# prompt", encoding="utf-8")
    manifest = read_json(prompt_ready / "manifest.json")
    manifest["transcript_semantic_correction_llm_prompt_markdown"] = "transcript-semantic-correction-llm-prompt.md"
    manifest["transcript_semantic_correction_llm_draft_summary"] = {"status": "planned", "decision_count": 0}
    write_json(prompt_ready / "manifest.json", manifest)

    result = transcript_semantic_repair_queue(root, target_bundle_count=3, write=True)

    assert result["schema"] == "video_knowledge_pipeline.transcript_semantic_repair_queue.v1"
    assert result["status"] == "needs_human_review"
    assert result["summary"]["action_required_count"] == 2
    assert result["summary"]["machine_action_available_count"] == 1
    assert result["summary"]["human_review_required_count"] == 1
    rows = {Path(row["bundle_dir"]).parent.name: row for row in result["items"]}
    assert rows["accepted"]["action_key"] == "none"
    assert rows["missing"]["action_key"] == "build_pack"
    assert rows["missing"]["machine_action_available"] is True
    assert "transcript-semantic-correction-pack" in rows["missing"]["retry_command"]
    assert rows["prompt"]["action_key"] == "execute_llm_or_use_codex"
    assert rows["prompt"]["human_review_required"] is True
    assert rows["prompt"]["llm_draft_status"] == "prompt_ready"
    assert Path(result["json_path"]).exists()
    assert "Repair Queue" in Path(result["markdown_path"]).read_text(encoding="utf-8")



def test_transcript_semantic_repair_queue_requires_summary_impact_after_readable_passed(tmp_path: Path) -> None:
    bundle = tmp_path / "summary-impact" / "webui-bundle"
    _write_accepted_semantic_bundle(bundle)
    (bundle / "transcript-semantic-summary-impact-report.json").unlink()

    result = transcript_semantic_repair_queue(bundle, target_bundle_count=1, write=False)

    row = result["items"][0]
    assert result["status"] == "machine_actions_available"
    assert row["acceptance_state"] == "needs_summary_impact_report"
    assert row["action_key"] == "run_summary_impact"
    assert row["machine_action_available"] is True
    assert "transcript-semantic-summary-impact-report" in row["retry_command"]


def test_transcript_semantic_batch_accepts_no_evaluable_summary_replacements(tmp_path: Path) -> None:
    bundle = tmp_path / "summary-no-evaluable" / "webui-bundle"
    _write_accepted_semantic_bundle(bundle)
    write_json(
        bundle / "transcript-semantic-summary-impact-report.json",
        {
            "status": "no_evaluable_replacements",
            "summary_residual_original_total": 0,
            "summary_corrected_hit_total": 0,
            "summary_absorption_rate": 0.0,
        },
    )

    queue = transcript_semantic_repair_queue(bundle, target_bundle_count=1, write=False)
    acceptance = transcript_semantic_batch_acceptance(bundle, target_bundle_count=1, write=False)

    assert queue["status"] == "complete"
    assert queue["items"][0]["acceptance_state"] == "accepted"
    assert queue["items"][0]["action_key"] == "none"
    assert acceptance["status"] == "accepted"
    assert acceptance["items"][0]["acceptance_state"] == "accepted"

def test_transcript_semantic_repair_queue_cli_writes_report(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "cli-queue" / "webui-bundle"
    _write_manifest(bundle, "missing semantic pack")
    output_dir = tmp_path / "queue-report"

    exit_code = main(["transcript-semantic-repair-queue", str(bundle), "--target-bundle-count", "1", "--output-dir", str(output_dir)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "machine_actions_available"
    assert payload["summary"]["by_action_key"]["build_pack"] == 1
    written = read_json(output_dir / "transcript-semantic-repair-queue.json")
    assert written["items"][0]["action_key"] == "build_pack"
    assert (output_dir / "transcript-semantic-repair-queue.md").exists()


def test_transcript_semantic_repair_run_previews_without_execution(tmp_path: Path) -> None:
    bundle = tmp_path / "run-preview" / "webui-bundle"
    _write_manifest(bundle, "missing semantic pack")

    result = transcript_semantic_repair_run(bundle, target_bundle_count=1, execute_safe_actions=False, write=True)

    assert result["schema"] == "video_knowledge_pipeline.transcript_semantic_repair_run.v1"
    assert result["status"] == "planned"
    assert result["summary"]["planned_count"] == 1
    assert result["executions"][0]["action_key"] == "build_pack"
    assert not (bundle / "transcript-semantic-correction-pack.json").exists()
    assert Path(result["json_path"]).exists()
    assert Path(result["markdown_path"]).exists()


def test_transcript_semantic_repair_run_executes_safe_pack_action(tmp_path: Path) -> None:
    bundle = tmp_path / "run-execute" / "webui-bundle"
    _write_manifest(bundle, "missing semantic pack")

    result = transcript_semantic_repair_run(bundle, target_bundle_count=1, execute_safe_actions=True, max_actions=1, write=True)

    assert result["status"] == "completed"
    assert result["summary"]["executed_count"] == 1
    assert result["executions"][0]["action_key"] == "build_pack"
    assert result["executions"][0]["run_status"] == "executed"
    assert (bundle / "transcript-semantic-correction-pack.json").exists()
    assert result["after_queue"]["items"][0]["action_key"] in {"none", "run_candidate_discovery", "run_llm_draft_preview", "execute_llm_or_use_codex"}



def test_transcript_semantic_repair_run_executes_summary_impact_report(tmp_path: Path) -> None:
    bundle = tmp_path / "run-summary-impact" / "webui-bundle"
    _write_accepted_semantic_bundle(bundle)
    (bundle / "transcript-semantic-summary-impact-report.json").unlink()

    result = transcript_semantic_repair_run(bundle, target_bundle_count=1, execute_safe_actions=True, max_actions=1, write=True)

    assert result["status"] == "completed"
    assert result["executions"][0]["action_key"] == "run_summary_impact"
    assert result["executions"][0]["result_status"] == "passed"
    assert (bundle / "transcript-semantic-summary-impact-report.json").exists()
    assert result["after_queue"]["items"][0]["action_key"] == "none"


def test_transcript_semantic_repair_run_closure_refreshes_readable_exports(tmp_path: Path) -> None:
    bundle = tmp_path / "run-closure-refresh" / "webui-bundle"
    bundle.mkdir(parents=True)
    write_json(bundle / "manifest.json", {"schema": "lecture_webui_bundle.v1", "title": "closure refresh", "normalized_transcript_json": "normalized-transcript.json"})
    write_json(bundle / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "\u4eca\u5929\u8bb2 browser base"}]})
    write_json(bundle / "timeline.json", [{"index": 0, "start": 0, "end": 4, "transcript": "\u4eca\u5929\u8bb2 browser base", "visual_text": "Browserbase"}])
    (bundle / "exports").mkdir(exist_ok=True)
    (bundle / "exports" / "full-transcript.md").write_text("\u4eca\u5929\u8bb2 browser base", encoding="utf-8")
    (bundle / "exports" / "smart-summary.md").write_text("\u8bfe\u7a0b\u63d0\u5230 browser base", encoding="utf-8")

    pack = build_transcript_semantic_correction_pack(bundle, write=True)
    candidate = next(row for row in pack["candidates"] if row.get("has_conflict"))
    result_path = bundle / "transcript-semantic-correction-result.codex.md"
    result_payload = {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
        "decisions": [
            {
                "candidate_id": candidate["candidate_id"],
                "action": "replace",
                "correction_type": candidate["correction_type"],
                "original_text": candidate["original_text"],
                "corrected_text": "Browserbase",
                "confidence": 0.94,
                "semantic_rationale": "OCR evidence shows Browserbase on screen.",
                "evidence_ids": candidate["evidence_ids"],
                "safe_to_apply": True,
            }
        ],
    }
    result_path.write_text("```json\n" + json.dumps(result_payload, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    validate_transcript_semantic_correction(bundle, input_json=result_path, write=True)

    queued = transcript_semantic_repair_queue(bundle, target_bundle_count=1, write=False)
    assert queued["items"][0]["action_key"] == "run_closure"

    result = transcript_semantic_repair_run(bundle, target_bundle_count=1, execute_safe_actions=True, allow_closure=True, max_actions=1, write=True)

    execution = result["executions"][0]
    assert execution["action_key"] == "run_closure"
    assert execution["result_status"] == "closed_and_refreshed_exports"
    assert execution["result"]["closure"]["status"] == "completed"
    assert execution["result"]["impact"]["status"] in {"passed", "needs_fix"}
    assert execution["result"]["readable_impact"]["status"] == "passed"
    full_transcript = (bundle / "exports" / "full-transcript.md").read_text(encoding="utf-8")
    assert "Browserbase" in full_transcript
    assert "browser base" not in full_transcript
    assert (bundle / "transcript-semantic-summary-impact-report.json").exists()
    assert result["after_queue"]["items"][0]["action_key"] == "run_candidate_discovery"
    assert result["after_queue"]["items"][0]["acceptance_state"] == "needs_candidate_discovery"


def _write_prompt_ready_semantic_bundle(bundle: Path) -> None:
    _write_manifest(bundle, "prompt ready semantic bundle")
    write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {
            "schema": "pack",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "correction_type": "proper_noun",
                    "risk_level": "medium",
                    "original_text": "playright",
                    "context_text": "playright client 是浏览器自动化工具",
                }
            ],
        },
    )
    (bundle / "transcript-semantic-correction-llm-prompt.md").write_text("# prompt", encoding="utf-8")
    manifest = read_json(bundle / "manifest.json")
    manifest["transcript_semantic_correction_llm_prompt_markdown"] = "transcript-semantic-correction-llm-prompt.md"
    manifest["transcript_semantic_correction_llm_draft_summary"] = {"status": "planned", "decision_count": 0}
    write_json(bundle / "manifest.json", manifest)


def test_transcript_semantic_repair_run_closure_accepts_keep_original_no_evaluable_summary(tmp_path: Path) -> None:
    bundle = tmp_path / "run-closure-keep-original" / "webui-bundle"
    _write_manifest(bundle, "closure keep original")
    write_json(bundle / "normalized-transcript.json", {"segments": [{"start": 0, "end": 2, "text": "嗯"}]})
    write_json(bundle / "timeline.json", [{"index": 0, "start": 0, "end": 2, "transcript": "嗯"}])
    write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {
            "schema": "pack",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "correction_type": "ordinary_word",
                    "risk_level": "medium",
                    "original_text": "嗯",
                    "context_text": "嗯",
                    "evidence_ids": ["asr_segment_0"],
                    "evidence": [{"evidence_id": "asr_segment_0", "source_type": "asr_or_subtitle", "text": "嗯"}],
                }
            ],
        },
    )
    review = {
        "schema": "video_knowledge_pipeline.transcript_semantic_batch_review_notes.v1",
        "reviews": [
            {
                "bundle_dir": str(bundle),
                "candidate_id": "semcorr-0001",
                "review_status": "keep_original",
                "confidence": 0.93,
                "review_note": "Low-information filler word; keep original and close the candidate.",
            }
        ],
    }
    review_path = tmp_path / "review-notes.json"
    write_json(review_path, review)
    transcript_semantic_batch_import_review_notes(review_path, output_dir=tmp_path, write=True)

    queued = transcript_semantic_repair_queue(bundle, target_bundle_count=1, write=False)
    assert queued["items"][0]["action_key"] == "run_closure"

    result = transcript_semantic_repair_run(bundle, target_bundle_count=1, execute_safe_actions=True, allow_closure=True, max_actions=1, write=True)

    execution = result["executions"][0]
    assert result["status"] == "completed"
    assert result["ok"] is True
    assert execution["action_key"] == "run_closure"
    assert execution["result_ok"] is True
    assert execution["result"]["closure"]["status"] == "completed_no_text_changes"
    assert execution["result"]["summary_impact"]["status"] == "no_evaluable_replacements"
    assert result["after_queue"]["items"][0]["action_key"] == "run_candidate_discovery"
    assert result["after_queue"]["items"][0]["acceptance_state"] == "needs_candidate_discovery"


def test_transcript_semantic_repair_run_uses_local_codex_draft_without_allow_llm(tmp_path: Path) -> None:
    bundle = tmp_path / "run-codex-local" / "webui-bundle"
    _write_prompt_ready_semantic_bundle(bundle)

    result = transcript_semantic_repair_run(bundle, target_bundle_count=1, execute_safe_actions=True, max_actions=1, write=False)

    assert result["status"] == "completed"
    assert result["executions"][0]["action_key"] == "execute_llm_or_use_codex"
    assert result["executions"][0]["run_status"] == "executed"
    assert result["executions"][0]["executed"] is True
    assert result["executions"][0]["result_status"] == "draft_ready"
    assert result["executions"][0]["result"]["operator_boundary"]["no_cloud_call"] is True
    assert result["after_queue"]["items"][0]["llm_draft_status"] == "codex_draft_ready"
    assert result["after_queue"]["items"][0]["action_key"] == "validate_result"
    assert (bundle / "transcript-semantic-correction-result.codex.json").exists()


def test_transcript_semantic_repair_run_can_execute_allowed_llm_provider(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.transcript_semantic_correction as semcorr

    bundle = tmp_path / "run-llm-allowed" / "webui-bundle"
    _write_prompt_ready_semantic_bundle(bundle)
    calls = []

    def fake_llm_draft(root, *, provider_config=None, execute=False, limit=80, write=True, **kwargs):
        calls.append({"root": str(root), "provider_config": provider_config, "execute": execute, "limit": limit, "write": write})
        write_json(Path(root) / "transcript-semantic-correction-result.llm.json", {"schema": "fake", "decisions": []})
        return {"status": "executed", "ok": True, "decision_count": 0}

    monkeypatch.setattr(semcorr, "build_transcript_semantic_correction_llm_draft", fake_llm_draft)
    provider = {"provider": "custom_openai_compatible", "base_url": "http://example.invalid/v1", "model": "fake", "api_key": "test"}

    result = transcript_semantic_repair_run(bundle, target_bundle_count=1, execute_safe_actions=True, allow_llm=True, provider_config=provider, llm_limit=7, max_actions=1, write=False)

    assert result["status"] == "completed"
    assert result["allow_llm"] is True
    assert result["executions"][0]["action_key"] == "execute_llm_or_use_codex"
    assert result["executions"][0]["run_status"] == "executed"
    assert result["executions"][0]["result_status"] == "executed"
    assert calls == [{"root": str(bundle.resolve()), "provider_config": provider, "execute": True, "limit": 7, "write": True}]

def test_transcript_semantic_repair_run_cli_preview(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "cli-run" / "webui-bundle"
    _write_manifest(bundle, "missing semantic pack")
    output_dir = tmp_path / "run-report"

    exit_code = main(["transcript-semantic-repair-run", str(bundle), "--target-bundle-count", "1", "--output-dir", str(output_dir), "--max-actions", "1"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned"
    assert payload["executions"][0]["action_key"] == "build_pack"
    assert (output_dir / "transcript-semantic-repair-run.json").exists()
    assert (output_dir / "transcript-semantic-repair-run.md").exists()


def test_transcript_semantic_batch_review_pack_imports_back_to_bundle(tmp_path: Path) -> None:
    root = tmp_path / "batch-review"
    bundle = root / "sample" / "webui-bundle"
    _write_manifest(bundle, "sample bundle")
    write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {
            "schema": "pack",
            "candidate_count": 2,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "correction_type": "proper_noun",
                    "risk_level": "medium",
                    "original_text": "playright",
                    "context_text": "playright client",
                    "evidence": [{"evidence_id": "ev-1", "source_type": "page_metadata", "text": "Playwright client"}],
                },
                {
                    "candidate_id": "semcorr-0002",
                    "correction_type": "ordinary_word",
                    "risk_level": "medium",
                    "original_text": "不确定词",
                    "context_text": "这里需要结合课程上下文判断",
                    "evidence": [{"evidence_id": "ev-2", "source_type": "asr_or_subtitle", "text": "不确定词"}],
                },
            ],
        },
    )
    (bundle / "transcript-semantic-correction-llm-prompt.md").write_text("# prompt", encoding="utf-8")
    manifest = read_json(bundle / "manifest.json")
    manifest["transcript_semantic_correction_llm_prompt_markdown"] = "transcript-semantic-correction-llm-prompt.md"
    manifest["transcript_semantic_correction_codex_draft_summary"] = {"status": "no_safe_draft_decisions", "decision_count": 0}
    manifest["transcript_semantic_correction_result_codex_json"] = "transcript-semantic-correction-result.codex.json"
    write_json(bundle / "manifest.json", manifest)
    write_json(bundle / "transcript-semantic-correction-result.codex.json", {"schema": "result", "decisions": []})

    pack = transcript_semantic_batch_review_pack(root, output_dir=root / "review", target_bundle_count=1, write=True)
    assert pack["review_item_count"] == 2
    assert pack["todo"]["reviews"][0]["evidence_ids"] == ["ev-1"]
    assert pack["todo"]["reviews"][1]["evidence_ids"] == ["ev-2"]

    todo = pack["todo"]
    todo["reviews"][0]["review_status"] = "accept_correction"
    todo["reviews"][0]["corrected_text"] = "Playwright"
    todo["reviews"][0]["confidence"] = 0.95
    todo["reviews"][0]["review_note"] = "Metadata supports Playwright."
    todo["reviews"][1]["review_status"] = "needs_more_evidence"
    todo["reviews"][1]["confidence"] = 0.2
    todo["reviews"][1]["review_note"] = "Current ASR evidence is insufficient."
    notes = root / "review" / "filled-review-notes.json"
    write_json(notes, todo)

    imported = transcript_semantic_batch_import_review_notes(notes, output_dir=root / "review", write=True)

    assert imported["status"] == "imported_partial_review_remaining"
    assert imported["imported_decision_count"] == 2
    assert imported["accepted_decision_count"] == 1
    assert imported["review_required_count"] == 1
    assert imported["closure_ready_bundle_count"] == 1
    assert imported["open_review_bundle_count"] == 1
    assert imported["by_validation_status"] == {"accepted_with_rejections": 1}
    assert imported["post_import_next_action_counts"]["run_closure"] == 1
    assert imported["imports"][0]["status"] == "imported"
    assert imported["bundle_summaries"][0]["accepted_decision_count"] == 1
    assert imported["bundle_summaries"][0]["review_required_count"] == 1
    validation = read_json(bundle / "transcript-semantic-correction-validation.json")
    assert validation["status"] == "accepted_with_rejections"
    assert validation["accepted_decision_count"] == 1
    assert validation["review_required_count"] == 1
    queue = transcript_semantic_repair_queue(root, output_dir=root / "review", target_bundle_count=1, write=False)
    assert queue["items"][0]["action_key"] == "run_closure"


def test_transcript_semantic_batch_codex_review_draft_is_conservative(tmp_path: Path) -> None:
    pack = tmp_path / "batch-review-pack.json"
    write_json(
        pack,
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_batch_review_pack.v1",
            "items": [
                {
                    "review_id": "r1",
                    "bundle_dir": str(tmp_path / "b1" / "webui-bundle"),
                    "bundle_title": "title",
                    "candidate_id": "semcorr-0001",
                    "correction_type": "proper_noun",
                    "risk_level": "medium",
                    "original_text": "playright",
                    "suggested_text": "Playwright",
                    "evidence_ids": ["ev-1"],
                    "evidence": [{"evidence_id": "ev-1", "source_type": "page_metadata", "text": "Playwright client"}],
                },
                {
                    "review_id": "r2",
                    "bundle_dir": str(tmp_path / "b1" / "webui-bundle"),
                    "bundle_title": "title",
                    "candidate_id": "semcorr-0002",
                    "correction_type": "ordinary_word",
                    "risk_level": "medium",
                    "original_text": "嗯",
                    "suggested_text": "",
                    "evidence_ids": ["ev-2"],
                    "evidence": [{"evidence_id": "ev-2", "source_type": "asr_or_subtitle", "text": "嗯"}],
                },
                {
                    "review_id": "r3",
                    "bundle_dir": str(tmp_path / "b1" / "webui-bundle"),
                    "bundle_title": "title",
                    "candidate_id": "semcorr-0003",
                    "correction_type": "segment_boundary",
                    "risk_level": "high",
                    "original_text": "long ambiguous segment",
                    "suggested_text": "",
                    "evidence_ids": ["ev-3"],
                    "evidence": [{"evidence_id": "ev-3", "source_type": "asr_or_subtitle", "text": "long ambiguous segment"}],
                },
            ],
        },
    )

    draft = transcript_semantic_batch_codex_review_draft(pack, output_dir=tmp_path, write=True)

    statuses = {row["candidate_id"]: row["review_status"] for row in draft["reviews"]}
    assert statuses == {"semcorr-0001": "accept_correction", "semcorr-0002": "keep_original", "semcorr-0003": "needs_more_evidence"}
    assert draft["by_review_status"] == {"accept_correction": 1, "keep_original": 1, "needs_more_evidence": 1}
    assert (tmp_path / "transcript-semantic-batch-review-notes.codex-draft.json").exists()






def test_transcript_semantic_repair_queue_runs_candidate_discovery_before_accepting_no_candidates(tmp_path: Path) -> None:
    bundle = tmp_path / "candidate-discovery" / "webui-bundle"
    _write_manifest(bundle, "candidate discovery smoke")
    write_json(bundle / "timeline.json", [{"index": 1, "start": 0, "end": 4, "transcript": "今天天气很好，我们继续学习。"}])
    write_json(bundle / "normalized-transcript.json", {"segments": [{"start": 0, "end": 4, "text": "今天天气很好，我们继续学习。"}]})
    build_transcript_semantic_correction_pack(bundle, write=True)

    queue = transcript_semantic_repair_queue(bundle, output_dir=bundle / "exports", target_bundle_count=1, limit=1, write=True)

    assert queue["items"][0]["semantic_status"] == "no_candidates"
    assert queue["items"][0]["acceptance_state"] == "needs_candidate_discovery"
    assert queue["items"][0]["action_key"] == "run_candidate_discovery"

    run = transcript_semantic_repair_run(
        bundle,
        output_dir=bundle / "exports",
        target_bundle_count=1,
        limit=1,
        execute_safe_actions=True,
        max_actions=1,
        allow_closure=False,
        allow_llm=False,
        write=True,
    )

    assert run["executions"][0]["action_key"] == "run_candidate_discovery"
    assert run["executions"][0]["executed"] is True
    assert (bundle / "transcript-semantic-candidate-discovery-pack.json").exists()
    after = transcript_semantic_repair_queue(bundle, output_dir=bundle / "exports", target_bundle_count=1, limit=1, write=False)
    assert after["items"][0]["candidate_discovery_status"] in {"prompt_ready", "no_segments_selected"}
    assert after["items"][0]["action_key"] in {"run_candidate_discovery_llm_preview", "none"}


def test_transcript_semantic_repair_queue_reviews_imported_suggestions_before_validation(tmp_path: Path) -> None:
    bundle = tmp_path / "imported-suggestions" / "webui-bundle"
    _write_manifest(bundle, "imported suggestions semantic bundle")
    write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
            "status": "pack_ready",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "start": 0,
                    "end": 4,
                    "correction_type": "concept",
                    "risk_level": "medium",
                    "original_text": "这个很重要",
                    "candidate_text": "客户信任建立流程",
                    "evidence_ids": ["candidate_discovery_suggestion"],
                    "evidence_source_types": ["candidate_discovery_suggestion"],
                }
            ],
        },
    )
    write_json(
        bundle / "transcript-semantic-candidate-suggestions-import.json",
        {
            "schema": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions_import.v1",
            "status": "imported",
            "suggestion_count": 1,
            "imported_candidate_count": 1,
            "skipped_count": 0,
            "imported_candidate_ids": ["semcorr-0001"],
        },
    )

    queue = transcript_semantic_repair_queue(bundle, target_bundle_count=1, write=False)

    item = queue["items"][0]
    assert item["semantic_status"] == "needs_llm_or_codex_review"
    assert item["next_action_key"] == "run_llm_draft_preview"
    assert item["action_key"] == "run_llm_draft_preview"
    assert item["machine_action_available"] is True
    assert "validate-transcript-semantic-correction" not in item["retry_command"]

def test_transcript_semantic_repair_queue_keeps_summary_impact_machine_action_with_optional_review(tmp_path: Path) -> None:
    bundle = tmp_path / "summary-impact-with-review" / "webui-bundle"
    _write_accepted_semantic_bundle(bundle)
    (bundle / "transcript-semantic-summary-impact-report.json").unlink()
    validation = read_json(bundle / "transcript-semantic-correction-validation.json")
    validation["review_required_count"] = 1
    validation["review_required_decisions"] = [
        {
            "candidate_id": "semcorr-review-0001",
            "original_text": "这个",
            "corrected_text": "",
            "correction_type": "ordinary_word",
            "needs_human_review": True,
            "confidence": 0.5,
        }
    ]
    write_json(bundle / "transcript-semantic-correction-validation.json", validation)

    result = transcript_semantic_repair_queue(bundle, target_bundle_count=1, write=False)

    row = result["items"][0]
    assert result["status"] == "machine_actions_available"
    assert row["review_required_count"] == 1
    assert row["acceptance_state"] == "needs_summary_impact_report"
    assert row["action_key"] == "run_summary_impact"
    assert row["machine_action_available"] is True
    assert row["human_review_required"] is False
    assert "transcript-semantic-summary-impact-report" in row["retry_command"]
