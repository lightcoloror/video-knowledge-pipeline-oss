from __future__ import annotations

import shutil
import uuid
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT.parent / ".runtime" / "vkp-portable-tests"


@contextmanager
def portable_test_directory(label: str) -> Iterator[Path]:
    """Provide a short-lived project-adjacent test directory on Windows.

    The shared Windows pytest temp root on this workstation can inherit an ACL
    that denies its own cleanup.  A unique project-adjacent directory keeps the
    test hermetic without changing production behavior or relying on that root.
    """

    path = RUNTIME_ROOT / f"{label}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=False)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Opt-in replacement for pytest's host-global ``tmp_path`` fixture.

    Load this module with ``-p portable_test_runtime`` for broader local suites
    on Windows hosts whose pytest temp root has a broken ACL.  Ordinary CI and
    test runs retain pytest's built-in fixture.
    """

    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.node.name)[:80]
    with portable_test_directory(label or "pytest") as path:
        yield path
