from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline import model_api_settings as settings_module
from video_knowledge_pipeline.model_api_settings import upsert_model_api_profile
from video_knowledge_pipeline.model_gateway import render_litellm_config


def _profile(*, capacity: bool) -> dict[str, object]:
    profile: dict[str, object] = {
        "id": "remote-capacity-test",
        "name": "Remote capacity test",
        "provider": "openai_compatible",
        "adapter_backend": "proxy",
        "location": "remote",
        "capabilities": ["text"],
        "base_url": "https://models.example.com/v1",
        "model": "remote-text",
        "timeout_seconds": 120,
        "enabled": True,
    }
    if capacity:
        profile.update(
            {
                "rpm": 60,
                "tpm": 120_000,
                "max_parallel_requests": 2,
            }
        )
    return profile


def _render(
    tmp_path: Path,
    monkeypatch,
    *,
    capacity: bool,
) -> dict[str, object]:
    """Build a secretless fake route and return the LiteLLM render receipt."""

    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    monkeypatch.setattr(
        settings_module,
        "_protect_secret",
        lambda value: "cipher:" + value.encode().hex(),
    )
    monkeypatch.setattr(
        settings_module,
        "_unprotect_secret",
        lambda value: bytes.fromhex(value.removeprefix("cipher:")).decode(),
    )
    upsert_model_api_profile(
        _profile(capacity=capacity),
        tasks=["summary_rewrite"],
        api_key="fake-local-test-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    return render_litellm_config(
        settings_path=settings_path,
        secrets_path=secrets_path,
        output_path=tmp_path / "litellm.yaml",
        write=False,
    )


def test_gateway_render_warns_when_capacity_policy_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Unbounded profiles remain usable but must never look production-ready.

    Intent: expose missing provider concurrency and rate budgets before a batch.
    Decision: reuse LiteLLM's mature capacity fields and report missing values;
    VKP does not implement a second rate limiter.
    Reason: the runtime already renders rpm/tpm/max_parallel_requests, but an
    empty profile previously looked identical to a deliberately sized route.
    Evidence: the current local settings contain no capacity value on any
    configured profile.
    Effective scope: local render/doctor receipts only; no provider call,
    credential disclosure, route fallback or settings mutation.
    """

    result = _render(tmp_path, monkeypatch, capacity=False)

    assert result["ready_for_start"] is True
    assert result["capacity_policy_ready"] is False
    assert result["capacity_warnings"] == [
        {
            "profile_id": "remote-capacity-test",
            "route_id": result["capacity_warnings"][0]["route_id"],
            "capability": "text",
            "execution_location": "remote",
            "status": "capacity_policy_incomplete",
            "missing": ["rpm", "tpm", "max_parallel_requests"],
            "next_action": "set_profile_capacity_limits",
        }
    ]


def test_gateway_render_accepts_complete_litellm_capacity_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _render(tmp_path, monkeypatch, capacity=True)

    assert result["capacity_policy_ready"] is True
    assert result["capacity_warnings"] == []

def test_gateway_render_does_not_invent_token_budget_for_asr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Audio providers may publish duration quotas instead of TPM.

    Intent: keep the gateway capacity gate accurate for non-token endpoints.
    Decision: require RPM and per-deployment concurrency for ASR/OCR, while
    text and vision continue to require RPM, TPM and concurrency.
    Reason: a fabricated TPM would make an audio route look safer than it is.
    Evidence: Groq publishes ASH/ASD rather than TPM for Whisper.
    Effective scope: secretless gateway render diagnostics only.
    """

    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    monkeypatch.setattr(
        settings_module,
        "_protect_secret",
        lambda value: "cipher:" + value.encode().hex(),
    )
    monkeypatch.setattr(
        settings_module,
        "_unprotect_secret",
        lambda value: bytes.fromhex(value.removeprefix("cipher:")).decode(),
    )
    upsert_model_api_profile(
        {
            "id": "remote-asr-capacity-test",
            "name": "Remote ASR capacity test",
            "provider": "groq_asr",
            "adapter_backend": "proxy",
            "location": "remote",
            "capabilities": ["asr"],
            "base_url": "https://api.groq.com/openai/v1",
            "model": "whisper-large-v3-turbo",
            "timeout_seconds": 120,
            "rpm": 20,
            "max_parallel_requests": 1,
            "enabled": True,
        },
        tasks=["asr"],
        api_key="fake-local-test-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    result = render_litellm_config(
        settings_path=settings_path,
        secrets_path=secrets_path,
        output_path=tmp_path / "litellm.yaml",
        write=False,
    )

    assert result["capacity_policy_ready"] is True
    assert result["capacity_warnings"] == []
