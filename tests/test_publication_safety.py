from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCAL_WINDOWS_USERNAME = "light" + "color"
PERSONAL_PATH_MARKERS = (
    f"c:\\users\\{LOCAL_WINDOWS_USERNAME}",
    f"d:\\users\\{LOCAL_WINDOWS_USERNAME}",
    "d:\\used-by-codex",
    "d:/used-by-codex",
    "d:\\mediagodownloads",
    "d:/mediagodownloads",
    "d:\\localllm",
    "d:/localllm",
    "e:\\vkp-datasets",
    "e:/vkp-datasets",
)
LOCAL_ONLY_UNTRACKED_FILES = {
    "cell",
    "docs/plans/2026-07-23-parallel-validation-and-batch-closure.md",
    "docs/plans/2026-07-24-current-parallel-validation-matrix.md",
    "docs/plans/2026-07-30-vkp-production-closure.md",
}


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def test_tracked_snapshot_does_not_expose_personal_absolute_paths() -> None:
    findings: list[str] = []
    for relative_path in _tracked_files():
        if relative_path == "tests/test_publication_safety.py":
            continue
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").casefold()
        except UnicodeDecodeError:
            continue
        for marker in PERSONAL_PATH_MARKERS:
            if marker in text:
                findings.append(f"{relative_path}: {marker}")

    assert findings == []


def test_known_local_only_files_are_not_tracked() -> None:
    tracked = set(_tracked_files())
    assert tracked.isdisjoint(LOCAL_ONLY_UNTRACKED_FILES)


def test_secrets_credentials_and_runtime_outputs_are_not_tracked() -> None:
    blocked_paths: list[str] = []
    for relative_path in _tracked_files():
        normalized = relative_path.replace("\\", "/").casefold()
        name = Path(normalized).name
        if (
            normalized.startswith((".local/", "openclaw-runs/", "outputs/"))
            or "/secrets/" in f"/{normalized}"
            or "/credentials/" in f"/{normalized}"
            or name == ".env"
            or (name.startswith(".env.") and name != ".env.example")
            or name.endswith((".pem", ".key", ".pfx", ".p12", ".credential.xml"))
        ):
            blocked_paths.append(relative_path)

    assert blocked_paths == []
SECRET_PATTERNS = {
    "openai_style": re.compile(r"(?<![A-Za-z0-9])" + r"sk-" + r"[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(
        r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "google_api_key": re.compile(r"(?<![A-Za-z0-9])AI" + r"za[0-9A-Za-z_-]{30,}"),
    "aws_access_key": re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "huggingface_token": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"),
    "modelscope_token": re.compile(
        r"(?<![A-Za-z0-9])ms-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    ),
    "private_key": re.compile(
        "-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "bearer_token": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~-]{20,}"),
}
PRIVATE_FIXTURE_MARKERS = (
    "一家三口" + "重疾险配置咨询沟通记录",
    "根情况来" + "的嘛",
    "送了一" + "外险",
    "活医" + "保",
    "民亚" + "保险",
)
BLOCKED_RELEASE_SUFFIXES = {
    ".aac",
    ".avi",
    ".ckpt",
    ".flac",
    ".gguf",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".onnx",
    ".opus",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
    ".webm",
}
BLOCKED_RELEASE_NAMES = {
    "full-transcript.md",
    "human-key-points.json",
    "normalized-transcript.json",
    "raw-asr-output.json",
    "review-notes.json",
}
REQUIRED_COMPLIANCE_FILES = {
    "CHANGE-RATIONALE.md",
    "LICENSE",
    "NOTICE",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "src/video_knowledge_pipeline/static/wavesurfer-7.12.11/LICENSE",
}


def _tracked_texts() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for relative_path in _tracked_files():
        if relative_path == "tests/test_publication_safety.py":
            continue
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            continue
        try:
            rows.append((relative_path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return rows


def test_tracked_snapshot_does_not_contain_secret_shapes() -> None:
    findings: list[str] = []
    for relative_path, text in _tracked_texts():
        for rule_name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{relative_path}: {rule_name}")

    assert findings == []


def test_tracked_snapshot_does_not_contain_private_recording_fixture() -> None:
    findings: list[str] = []
    for relative_path, text in _tracked_texts():
        for marker in PRIVATE_FIXTURE_MARKERS:
            if marker in text:
                findings.append(f"{relative_path}: private_fixture_marker")

    assert findings == []


def test_tracked_snapshot_excludes_media_models_and_generated_reader_outputs() -> None:
    findings: list[str] = []
    for relative_path in _tracked_files():
        path = Path(relative_path)
        normalized = relative_path.replace("\\", "/").casefold()
        if path.suffix.casefold() in BLOCKED_RELEASE_SUFFIXES:
            findings.append(relative_path)
        elif path.name.casefold() in BLOCKED_RELEASE_NAMES:
            findings.append(relative_path)
        elif (
            not normalized.startswith("docs/")
            and path.name.casefold().startswith("smart-summary")
            and path.suffix == ".md"
        ):
            findings.append(relative_path)
        elif normalized.startswith(("datasets/", "models/")):
            findings.append(relative_path)

    assert findings == []


def test_public_compliance_files_and_vendored_license_are_present() -> None:
    tracked = set(_tracked_files())
    assert REQUIRED_COMPLIANCE_FILES <= tracked

    wavesurfer_license = (
        REPOSITORY_ROOT
        / "src/video_knowledge_pipeline/static/wavesurfer-7.12.11/LICENSE"
    ).read_text(encoding="utf-8")
    assert wavesurfer_license.startswith("BSD 3-Clause License")
