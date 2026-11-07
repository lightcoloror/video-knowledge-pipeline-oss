from __future__ import annotations


from video_knowledge_pipeline.volcengine_model_task_matrix import (
    TASK_RECOMMENDATIONS,
    run_volcengine_model_task_matrix,
)
from video_knowledge_pipeline.volcengine_model_routing import volcengine_model_routing


def test_tool_terms_route_includes_configured_doubao_without_name_filter() -> None:
    result = volcengine_model_routing(write=False)

    assert result["ok"] is True
    assert result["route"] == "tool_terms"
    models = [row["model"] for row in result["route_steps"]]
    assert models == ["doubao-seed-2.0-pro", "deepseek-v4-flash", "deepseek-v4-pro", "kimi-k2.6", "glm-5.2"]

    by_model = {row["model"]: row for row in result["route_steps"]}
    assert by_model["deepseek-v4-flash"]["stage"] == "batch_triage"
    assert by_model["deepseek-v4-pro"]["stage"] == "final_arbitration"

    provider_configs = {row["model"]: row for row in result["provider_configs"]}
    assert all("thinking" not in row for row in provider_configs.values())
    assert "api_key" not in provider_configs["deepseek-v4-pro"]


def test_tool_terms_no_doubao_excludes_kimi_code_default() -> None:
    result = volcengine_model_routing(write=False)

    models = {row["model"] for row in result["route_steps"]}
    assert "kimi-k2.7-code" not in models
    assert result["mixed_route_notes"]["kimi-k2.7-code"]["status"] == "coding_plan_alias_verified_by_execution"
    assert "doubao-seed-2.0-code" not in result["mixed_route_notes"]


def test_model_task_matrix_includes_and_accepts_doubao_models() -> None:
    result = run_volcengine_model_task_matrix(write=False)

    assert result["ok"] is True
    assert any(row["model"].casefold().startswith("doubao") for row in result["model_profiles"])
    assert "doubao-seed-2.0-pro" in TASK_RECOMMENDATIONS["tool_terms"]
    selected = run_volcengine_model_task_matrix(models=["doubao-seed-2.0-pro"], write=False)
    assert [row["model"] for row in selected["model_profiles"]] == ["doubao-seed-2.0-pro"]


def test_unknown_route_is_safe_and_no_network() -> None:
    result = volcengine_model_routing(route="missing", write=False)

    assert result["ok"] is False
    assert result["status"] == "unknown_route"
    assert result["operator_boundary"]["does_not_call_network"] is True