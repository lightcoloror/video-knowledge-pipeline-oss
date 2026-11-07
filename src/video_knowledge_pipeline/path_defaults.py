from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    configured = str(os.environ.get("VKP_WORKSPACE_ROOT") or "").strip()
    return Path(configured).expanduser().resolve() if configured else project_root().parent


def provider_env_file() -> Path:
    configured = str(os.environ.get("VKP_PROVIDER_ENV_FILE") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return workspace_root() / "provider-config" / "model-provider.env"


def port_record_path() -> Path:
    configured = str(os.environ.get("VKP_PORT_RECORD_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return project_root() / ".local" / "local-port-record.md"


def openclaw_compose_path() -> Path:
    configured = str(os.environ.get("VKP_OPENCLAW_COMPOSE_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".openclaw" / "docker-compose.yml"


def tool_source_review_root() -> Path:
    configured = str(os.environ.get("VKP_TOOL_SOURCE_REVIEW_ROOT") or "").strip()
    return Path(configured).expanduser().resolve() if configured else workspace_root() / "tool-source-review"


def source_reviews_root() -> Path:
    configured = str(os.environ.get("VKP_SOURCE_REVIEWS_ROOT") or "").strip()
    return Path(configured).expanduser().resolve() if configured else workspace_root() / "source-reviews"


def local_model_root() -> Path:
    configured = str(os.environ.get("VKP_LOCAL_MODEL_ROOT") or "").strip()
    return Path(configured).expanduser().resolve() if configured else project_root() / ".local" / "models"


def dataset_root() -> Path:
    configured = str(os.environ.get("VKP_DATASET_ROOT") or "").strip()
    return Path(configured).expanduser().resolve() if configured else project_root() / ".local" / "datasets"


def video_tool_lab_root() -> Path:
    configured = str(os.environ.get("VKP_VIDEO_TOOL_LAB_ROOT") or "").strip()
    return Path(configured).expanduser().resolve() if configured else workspace_root() / "video-tool-lab"