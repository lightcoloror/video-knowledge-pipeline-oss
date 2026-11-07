from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_candidate_suite import (
    PLAN_SCHEMA,
    prepare_model_candidate_suite,
)


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _profile(profile_id: str, *, capabilities: list[str]) -> dict[str, object]:
    return {
        "id": profile_id,
        "name": profile_id,
        "provider": "openai_compatible",
        "litellm_provider": "openai",
        "adapter_backend": "proxy",
        "base_url": f"https://{profile_id}.example/v1",
        "model": f"model-{profile_id}",
        "location": "remote",
        "capabilities": capabilities,
        "secret_ref": f"dpapi:{profile_id}",
        "timeout_seconds": 30,
        "enabled": True,
    }


def _settings(path: Path, profiles: list[dict[str, object]]) -> Path:
    return _write(
        path,
        {
            "schema": "video_knowledge_pipeline.local_model_api_settings.v2",
            "profiles": profiles,
            "task_routes": {},
            "route_pools": [],
            "route_bindings": {},
            "updated_at": "",
        },
    )


def test_prepare_suite_isolates_each_candidate_and_hashes_artifact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "input.txt"
    artifact.write_text("fixed evidence", encoding="utf-8")
    settings = _settings(
        tmp_path / "settings.json",
        [
            _profile("candidate-a", capabilities=["text"]),
            _profile("candidate-b", capabilities=["text"]),
        ],
    )
    plan = _write(
        tmp_path / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "gateway": {"requested_port": 8777},
            "cases": [
                {
                    "id": "text-fixed",
                    "task": "provider_task_benchmark",
                    "model_type": "text_llm",
                    "artifacts": [str(artifact)],
                    "instructions": "Return fixed JSON.",
                    "expected_format": "json",
                    "expires_hours": 6,
                    "max_estimated_cost_usd": 0.25,
                    "max_cost_per_call_usd": 0.25,
                    "max_retries_per_call": 0,
                    "candidates": [
                        {
                            "profile_id": "candidate-a",
                            "route_task": "text_llm",
                            "contract_status": "equivalent",
                        },
                        {"profile_id": "candidate-b", "route_task": "text_llm"},
                    ],
                }
            ],
        },
    )

    result = prepare_model_candidate_suite(
        plan,
        settings_path=settings,
        output_dir=tmp_path / "prepared",
    )

    assert result["status"] == "ready_for_operator_consent"
    assert result["candidate_count"] == 2
    assert result["candidates"][0]["contract_status"] == "equivalent"
    assert len({row["route_revision"] for row in result["candidates"]}) == 2
    assert all(row["artifacts"][0]["sha256"] for row in result["candidates"])
    assert all(Path(row["settings_path"]).is_file() for row in result["candidates"])
    assert all(row["expires_hours"] == 6 for row in result["candidates"])
    assert all(row["max_calls"] == 1 for row in result["candidates"])
    assert all(row["max_estimated_cost_usd"] == 0.25 for row in result["candidates"])
    assert all(row["max_cost_per_call_usd"] == 0.25 for row in result["candidates"])
    assert all(row["max_retries_per_call"] == 0 for row in result["candidates"])
    assert result["operator_boundary"]["remote_requests_made"] == 0
    saved = (tmp_path / "prepared" / "prepared-suite.json").read_text(encoding="utf-8")
    assert "api_key" not in saved.lower()
    for row in result["candidates"]:
        candidate_settings = json.loads(
            Path(row["settings_path"]).read_text(encoding="utf-8")
        )
        assert len(candidate_settings["profiles"]) == 1
        assert candidate_settings["profiles"][0]["id"] == row["profile_id"]


def test_prepare_suite_rejects_profile_capability_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "frame.jpg"
    artifact.write_bytes(b"frame")
    settings = _settings(
        tmp_path / "settings.json",
        [_profile("text-only", capabilities=["text"])],
    )
    plan = _write(
        tmp_path / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "cases": [
                {
                    "id": "vision-fixed",
                    "task": "multimodal_frame_analysis",
                    "model_type": "semantic_frame",
                    "artifacts": [str(artifact)],
                    "instructions": "Return JSON.",
                    "candidates": [
                        {"profile_id": "text-only", "route_task": "semantic_frame"}
                    ],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="does not support vision"):
        prepare_model_candidate_suite(
            plan,
            settings_path=settings,
            output_dir=tmp_path / "prepared",
        )


def test_prepare_asr_suite_locks_bounded_prompt_and_consent_limits(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "audio.mp3"
    artifact.write_bytes(b"audio")
    settings = _settings(
        tmp_path / "settings.json",
        [_profile("asr-candidate", capabilities=["asr"])],
    )
    plan = _write(
        tmp_path / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "cases": [
                {
                    "id": "asr-second-source",
                    "task": "cloud_asr",
                    "model_type": "asr",
                    "artifacts": [str(artifact)],
                    "instructions": "Audit-only task instructions.",
                    "asr_prompt": "domain term",
                    "max_estimated_cost_usd": 0.2,
                    "max_cost_per_call_usd": 0.2,
                    "max_retries_per_call": 0,
                    "candidates": [
                        {
                            "profile_id": "asr-candidate",
                            "route_task": "asr",
                        }
                    ],
                }
            ],
        },
    )

    result = prepare_model_candidate_suite(
        plan,
        settings_path=settings,
        output_dir=tmp_path / "prepared",
    )
    candidate = result["candidates"][0]

    assert candidate["asr_prompt"] == "domain term"
    assert candidate["asr_prompt_sha256"]
    assert candidate["max_calls"] == 1
    assert candidate["max_estimated_cost_usd"] == 0.2
    assert candidate["max_cost_per_call_usd"] == 0.2
    assert candidate["max_retries_per_call"] == 0


def test_prepare_suite_locks_json_mode_and_m3_no_thinking(tmp_path: Path) -> None:
    artifact = tmp_path / "frame.jpg"
    artifact.write_bytes(b"frame")
    profile = _profile("ark-m3", capabilities=["vision"])
    profile.update(
        {
            "provider": "volcengine_coding_plan",
            "litellm_provider": "openai",
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "model": "minimax-m3",
        }
    )
    settings = _settings(tmp_path / "settings.json", [profile])
    plan = _write(
        tmp_path / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "cases": [
                {
                    "id": "vision-json",
                    "task": "multimodal_frame_analysis",
                    "model_type": "semantic_frame",
                    "artifacts": [str(artifact)],
                    "instructions": "Return strict JSON.",
                    "expected_format": "json",
                    "request_options": {"max_tokens": 512},
                    "output_contract": {
                        "format": "json",
                        "required_keys": {
                            "scene": "string",
                            "state_changes": "array",
                        },
                        "nonempty_keys": ["state_changes"],
                    },
                    "candidates": [
                        {
                            "profile_id": "ark-m3",
                            "route_task": "semantic_frame",
                        }
                    ],
                }
            ],
        },
    )

    result = prepare_model_candidate_suite(
        plan,
        settings_path=settings,
        output_dir=tmp_path / "prepared",
    )

    candidate = result["candidates"][0]
    assert candidate["route_locked_request_options"] == {
        "max_tokens": 512,
        "response_format": "json_object",
        "thinking_mode": "disabled",
    }
    candidate_settings = json.loads(
        Path(candidate["settings_path"]).read_text(encoding="utf-8")
    )
    assert candidate_settings["profiles"][0]["provider_options"] == {
        "max_tokens": 512,
        "response_format": "json_object",
        "thinking_mode": "disabled",
    }
    assert Path(candidate["output_contract_path"]).is_file()
    assert candidate["output_contract"]["required_keys"] == {
        "scene": "string",
        "state_changes": "array",
    }


def test_fixed_suite_runner_is_windows_powershell_code_page_safe() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-model-candidate-fixed-suite.ps1"
    )
    source = runner.read_text(encoding="utf-8")

    assert source.isascii()
    assert '[string]$PortRecordPath = ""' in source
    assert "function Resolve-PortRecordPath" in source
    assert '"VKP LiteLLM Proxy"' in source
    assert "-PortRecordPath is required with -Execute" in source
    assert "AI-obsidian" not in source
    assert "used by syncthing" not in source
    assert '[string]$CandidateIds = ""' in source
    assert '[string]$RunOutputPath = ""' in source
    assert "function Get-LatestExecutionReport" in source
    assert "Get-Content -Raw -Encoding utf8 -LiteralPath $reportFile.FullName" in source
    assert "$executionOutput" not in source
    assert "$executionResultLines" in source
    assert "front-door status=" in source
    assert "[void]$process.WaitForExit(10000)" in source
    assert "foreach ($candidate in $selectedCandidates)" in source
