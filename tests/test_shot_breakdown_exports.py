from __future__ import annotations

import csv
import io
import tomllib
from pathlib import Path

from video_knowledge_pipeline.shot_breakdown_exports import (
    render_shot_breakdown_csv,
    render_shot_breakdown_logseq,
)


def _result() -> dict:
    return {
        "title": "拍摄素材样片",
        "status": "completed",
        "shot_count": 1,
        "readiness": {"ready_count": 0},
        "shots": [
            {
                "shot_id": "shot-0001",
                "start_time": "00:00:00.000",
                "end_time": "00:00:03.000",
                "duration": 3.0,
                "facts": {
                    "shot_type": "medium",
                    "camera_movement": "static",
                    "subject_action": "人物讲解",
                    "composition": {"value": "居中构图"},
                    "lighting": {},
                    "color_profile": "暖色",
                    "dialogue_or_narration": "示例对白",
                    "screen_text": "示例标题",
                },
                "fact_fields": {
                    "shot_type": {
                        "status": "inferred",
                        "evidence_ids": ["frame:0001"],
                    }
                },
                "unknown_fields": ["lighting"],
                "human_confirmed": False,
            }
        ],
    }


def test_logseq_projection_is_nested_and_not_collapsed_by_default() -> None:
    text = render_shot_breakdown_logseq(_result())

    assert text.startswith("- 逐镜头拉片：拍摄素材样片\n")
    assert "    - shot-0001" in text
    assert "      - 景别：medium" in text
    assert "collapsed::" not in text


def test_csv_projection_is_machine_readable_and_evidence_bound() -> None:
    rows = list(csv.DictReader(io.StringIO(render_shot_breakdown_csv(_result()))))

    assert len(rows) == 1
    assert rows[0]["shot_id"] == "shot-0001"
    assert rows[0]["shot_type"] == "medium"
    assert "frame:0001" in rows[0]["evidence_ids"]


def test_wavesurfer_assets_are_declared_as_package_data() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["video_knowledge_pipeline"]

    assert "static/*.js" in package_data
    assert "static/wavesurfer-7.12.11/*" in package_data
