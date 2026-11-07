from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.content_profile.v1"
DEFAULT_PROFILE = "course-or-general-v1"
SUPPORTED_PROFILES = (
    DEFAULT_PROFILE,
    "interview-v1",
    "medical-insurance-interview-v1",
)


def profile_requirements(profile_id: str) -> dict[str, Any]:
    """Return evidence requirements; it never infers facts or grants approval."""

    profile = _normalise_profile(profile_id)
    interview = profile in {"interview-v1", "medical-insurance-interview-v1"}
    regulated_interview = profile == "medical-insurance-interview-v1"
    return {
        "profile_id": profile,
        "speaker_diarization_required": interview,
        "speaker_role_review_required": regulated_interview,
        "semantic_fact_review_required": regulated_interview,
        "privacy_review_required": regulated_interview,
        "amount_and_number_review_required": regulated_interview,
        "individual_case_boundary_required": regulated_interview,
        "human_publication_approval_required": True,
        "summary_sections": (
            ["基本信息", "事实时间线", "受访者原话与感受", "已确认保险与医疗信息", "待核实事项", "隐私与发布边界"]
            if regulated_interview
            else (["基本信息", "事实与原话", "感受与待核实事项", "发布边界"] if interview else [])
        ),
        "forbidden_default_sections": (
            ["方法论", "可执行动作清单", "高频话术", "可复用表达"]
            if interview
            else []
        ),
    }


def resolve_content_profile(
    bundle_dir: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    value = manifest
    if value is None:
        manifest_path = root / "manifest.json"
        loaded = read_json(manifest_path) if manifest_path.is_file() else {}
        value = loaded if isinstance(loaded, dict) else {}
    explicit = str(value.get("content_profile") or value.get("video_content_profile") or "").strip()
    if explicit:
        profile = _normalise_profile(explicit)
        return {
            "schema": SCHEMA,
            "profile_id": profile,
            "status": "explicit",
            "explicit": True,
            "requirements": profile_requirements(profile),
        }

    title = str(value.get("title") or root.name).strip()
    inferred = "interview-v1" if "采访" in title else DEFAULT_PROFILE
    return {
        "schema": SCHEMA,
        "profile_id": inferred,
        "status": "inferred_from_title" if inferred != DEFAULT_PROFILE else "default",
        "explicit": False,
        "requirements": profile_requirements(inferred),
        "operator_boundary": {
            "title_inference_cannot_select_regulated_profile": True,
            "explicit_profile_required_for_medical_insurance_interview": True,
        },
    }


def apply_content_profile(
    bundle_dir: str | Path,
    *,
    profile_id: str,
    write: bool = True,
) -> dict[str, Any]:
    """Apply requirements only; no review is inferred and no evidence is changed."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    loaded = read_json(manifest_path)
    if not isinstance(loaded, dict):
        raise ValueError("manifest.json must be a JSON object")
    profile = _normalise_profile(profile_id)
    requirements = profile_requirements(profile)
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "profile_id": profile,
        "status": "applied" if write else "validated",
        "requirements": requirements,
        "review_templates": {
            "privacy": "privacy-review.json",
            "semantic_fact": "medical-insurance-fact-review.json",
        },
        "operator_boundary": {
            "does_not_infer_review_pass": True,
            "does_not_modify_transcript_or_timeline": True,
            "publication_still_requires_human_approval": True,
        },
        "updated_at": now_iso(),
    }
    if not write:
        return result

    loaded["content_profile"] = profile
    transcript_requirements = loaded.get("transcript_requirements")
    transcript_requirements = transcript_requirements if isinstance(transcript_requirements, dict) else {}
    if requirements["speaker_diarization_required"]:
        transcript_requirements["speaker_diarization_required"] = True
        transcript_requirements["expected_speaker_count"] = max(
            2, int(transcript_requirements.get("expected_speaker_count") or 0)
        )
    loaded["transcript_requirements"] = transcript_requirements
    loaded["content_profile_requirements"] = requirements
    write_json(manifest_path, loaded)
    if profile == "medical-insurance-interview-v1":
        _write_review_template(
            root / "privacy-review.json",
            schema="video_knowledge_pipeline.privacy_review.v1",
            review_kind="privacy",
        )
        _write_review_template(
            root / "medical-insurance-fact-review.json",
            schema="video_knowledge_pipeline.medical_insurance_fact_review.v1",
            review_kind="medical_insurance_fact",
        )
    write_json(root / "content-profile.json", result)
    return result


def _write_review_template(path: Path, *, schema: str, review_kind: str) -> None:
    if path.exists():
        return
    write_json(
        path,
        {
            "schema": schema,
            "review_kind": review_kind,
            "review_scope": "source_fidelity_only",
            "status": "needs_human_review",
            "reviewed_items": [],
            "unresolved_items": [],
            "human_confirmed": False,
            "operator_boundary": {
                "template_only": True,
                "does_not_assert_facts": True,
                "no_external_medical_or_insurance_fact_check": True,
            },
        },
    )


def _normalise_profile(profile_id: str) -> str:
    value = str(profile_id or DEFAULT_PROFILE).strip().lower()
    aliases = {
        "course": DEFAULT_PROFILE,
        "course_or_general": DEFAULT_PROFILE,
        "general": DEFAULT_PROFILE,
        "interview": "interview-v1",
        "medical-interview": "medical-insurance-interview-v1",
        "medical_insurance_interview": "medical-insurance-interview-v1",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported content profile: {profile_id}; allowed={SUPPORTED_PROFILES}")
    return value
