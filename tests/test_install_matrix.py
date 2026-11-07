from __future__ import annotations

import tomllib
from pathlib import Path


def _project() -> dict:
    with Path("pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _pyproject() -> dict:
    with Path("pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_core_and_local_install_do_not_bundle_model_runtimes() -> None:
    project = _project()
    extras = project["optional-dependencies"]

    assert project["dependencies"] == [
        "jieba>=0.42.1,<1",
        "jsonschema>=4.25,<5",
        "markdown-it-py>=3.0",
        "psutil==7.2.2",
        "rapidfuzz>=3.14.5,<4",
        "transitions==0.9.3",
    ]
    assert extras["core"] == []
    assert extras["local"] == ["ruptures==1.1.10"]
    assert not any("litellm" in item.lower() for item in project["dependencies"])


def test_online_and_hybrid_share_the_pinned_proxy_dependency() -> None:
    extras = _project()["optional-dependencies"]
    expected = ["litellm[proxy]>=1.81.7,<2"]

    assert extras["online"] == expected
    assert extras["hybrid"] == expected


def test_evaluation_extra_is_optional_and_model_free() -> None:
    """Keep mature speaker metrics optional, pinned, and weight-free.

    Intent: make DER and speaker-attributed text metrics reproducible without
    expanding the core runtime. Decision: pin pyannote.metrics by release range
    and MeetEval by the reviewed Windows C++20 fix commit. Reason: PyPI 0.4.3
    predates that Windows build fix. Evidence: both isolated upstream suites and
    VKP adapters pass locally. Effective scope: the explicit evaluation extra;
    no ASR or model weights are installed by core/local modes.
    """

    extras = _project()["optional-dependencies"]

    assert extras["evaluation"] == [
        "pyannote-metrics>=4.1,<5",
        (
            "meeteval @ git+https://github.com/fgnt/meeteval.git@"
            "184ff17eb77fd6db4aba27a9e303a6a3edb09364"
        ),
    ]
    assert "evaluation" not in _project()["dependencies"]


def test_all_install_modes_use_the_same_business_package() -> None:
    project = _project()

    assert project["name"] == "video-knowledge-pipeline"
    assert set((project["optional-dependencies"])) >= {
        "core",
        "local",
        "online",
        "hybrid",
        "evaluation",
        "test",
        "dev",
    }


def test_public_clone_has_explicit_build_and_test_contracts() -> None:
    document = _pyproject()
    assert document["build-system"] == {
        "requires": ["setuptools>=77,<82", "wheel>=0.45,<1"],
        "build-backend": "setuptools.build_meta",
    }
    extras = document["project"]["optional-dependencies"]
    assert extras["test"] == [
        "pytest>=9,<10",
        "mcp>=1,<2",
        "pillow>=10,<13",
    ]
    assert set(extras["test"]).issubset(set(extras["dev"]))


def test_model_gateway_smoke_readiness_cli_is_packaged() -> None:
    scripts = _project()["scripts"]

    assert scripts["video-knowledge-model-smoke-readiness"] == (
        "video_knowledge_pipeline.model_gateway_smoke_readiness:main"
    )
