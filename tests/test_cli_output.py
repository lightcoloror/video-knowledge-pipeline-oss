import json

from video_knowledge_pipeline.cli_output import render_cli_result


def test_small_cli_result_remains_complete() -> None:
    result = {"schema": "test.v1", "status": "ok", "items": [1, 2]}

    rendered = json.loads(render_cli_result(result, inline_limit=1000))

    assert rendered == result


def test_large_cli_result_defaults_to_paths_counts_and_hashes() -> None:
    result = {
        "schema": "test.v1",
        "status": "completed",
        "report_path": "D:/bundle/report.json",
        "artifact_sha256": "a" * 64,
        "item_count": 200,
        "items": [{"text": "x" * 1000} for _ in range(20)],
        "summary": {"total": 200, "indexes": list(range(200))},
    }

    rendered = json.loads(render_cli_result(result, inline_limit=100))

    assert rendered["status"] == "completed"
    assert rendered["paths"]["report_path"].endswith("report.json")
    assert rendered["counts"]["item_count"] == 200
    assert rendered["identities"]["artifact_sha256"] == "a" * 64
    assert rendered["summary"]["indexes_count"] == 200
    assert "items" not in rendered
    assert rendered["stdout_policy"]["full_result_omitted"] is True


def test_verbose_cli_result_preserves_full_payload() -> None:
    result = {"items": [{"text": "x" * 1000}]}

    rendered = json.loads(
        render_cli_result(result, verbose=True, inline_limit=10)
    )

    assert rendered == result
