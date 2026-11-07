from __future__ import annotations

from video_knowledge_pipeline import cli


def test_no_agent_substitute_disables_agent_readable_rewrite(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_pipeline(bundle_dir, **kwargs):
        captured["bundle_dir"] = bundle_dir
        captured.update(kwargs)
        return {"status": "previewed", "ok": True}

    monkeypatch.setattr(cli, "run_transcript_evidence_correction_pipeline", fake_pipeline)

    exit_code = cli.main(
        [
            "transcript-evidence-correction-pipeline",
            "fixture-bundle",
            "--no-agent-substitute",
            "--no-readable-llm",
            "--no-write",
            "--additional-secondary-asr-json",
            "secondary-a.json",
            "--additional-secondary-asr-json",
            "secondary-b.json",
        ]
    )

    assert exit_code == 0
    assert captured["use_agent_substitute"] is False
    assert captured["run_agent_readable_rewrite"] is False
    assert captured["run_readable_llm"] is False
    assert '"status": "previewed"' in capsys.readouterr().out
    assert captured["additional_secondary_asr_json"] == ["secondary-a.json", "secondary-b.json"]


def test_transcript_source_arbitration_cli_uses_supported_contract(
    monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_arbitration(bundle_dir, **kwargs):
        captured["bundle_dir"] = bundle_dir
        captured.update(kwargs)
        return {"status": "completed", "ok": True}

    monkeypatch.setattr(cli, "arbitrate_transcript_sources", fake_arbitration)

    exit_code = cli.main(
        [
            "transcript-source-arbitration",
            "fixture-bundle",
            "--asr-json",
            "primary-asr.json",
            "--no-write",
        ]
    )

    assert exit_code == 0
    assert captured["bundle_dir"] == "fixture-bundle"
    assert captured["asr_json"] == "primary-asr.json"
    assert "secondary_asr_json" not in captured
    assert '"status": "completed"' in capsys.readouterr().out
