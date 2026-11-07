from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from video_knowledge_pipeline.panorama_preflight import panorama_doctor


def _resolver(values: dict[str, str]):
    return lambda name: values.get(name, "")


def _successful_runner(command: list[str]) -> CompletedProcess[str]:
    if "-filters" in command:
        return CompletedProcess(command, 0, stdout=" .SC v360 V->V Convert 360 projection", stderr="")
    if "-show_streams" in command:
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 3840,
                    "height": 1920,
                    "side_data_list": [{"side_data_type": "Spherical Mapping", "projection": "equirectangular"}],
                }
            ]
        }
        return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
    return CompletedProcess(command, 0, stdout="", stderr="")


def test_doctor_reports_missing_media_tools_without_writing(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())

    report = panorama_doctor(
        tmp_path / "missing.mp4",
        projection="equirectangular_mono",
        tool_resolver=_resolver({}),
        command_runner=_successful_runner,
    )

    assert report["ok"] is False
    assert report["ready_to_prepare"] is False
    assert {row["key"] for row in report["blockers"]} == {"ffmpeg_missing", "ffprobe_missing", "media_missing"}
    assert list(tmp_path.iterdir()) == before


def test_doctor_uses_ffprobe_metadata_and_v360_capability(tmp_path: Path) -> None:
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"synthetic placeholder")

    report = panorama_doctor(
        media,
        tool_resolver=_resolver({"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}),
        command_runner=_successful_runner,
    )

    assert report["ok"] is True
    assert report["ready_to_prepare"] is True
    assert report["profile"]["projection"] == "equirectangular_mono"
    assert report["capabilities"]["v360"] is True
    assert set(report["capabilities"]["optional_workers"]) == {"py360convert", "equilib"}
    assert all(
        value in {"available", "unavailable_not_installed"}
        for value in report["capabilities"]["optional_workers"].values()
    )
    assert report["provider_health_check"] == "not_applicable"
    assert report["unattended_execution"] is False


def test_doctor_does_not_treat_two_to_one_ratio_as_authoritative(tmp_path: Path) -> None:
    media = tmp_path / "ordinary.mp4"
    media.write_bytes(b"synthetic placeholder")

    def runner(command: list[str]) -> CompletedProcess[str]:
        if "-filters" in command:
            return CompletedProcess(command, 0, stdout="v360", stderr="")
        payload = {"streams": [{"codec_type": "video", "width": 2000, "height": 1000}]}
        return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    report = panorama_doctor(
        media,
        tool_resolver=_resolver({"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}),
        command_runner=runner,
    )

    assert report["ready_to_prepare"] is False
    assert report["profile"]["projection"] == "unknown"
    assert "--projection" in report["recovery_command"]


def test_doctor_fails_when_v360_filter_is_unavailable(tmp_path: Path) -> None:
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"synthetic placeholder")

    def runner(command: list[str]) -> CompletedProcess[str]:
        if "-filters" in command:
            return CompletedProcess(command, 0, stdout="scale crop", stderr="")
        return CompletedProcess(command, 0, stdout=json.dumps({"streams": [{"codec_type": "video"}]}), stderr="")

    report = panorama_doctor(
        media,
        projection="equirectangular_mono",
        tool_resolver=_resolver({"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}),
        command_runner=runner,
    )

    assert report["ready_to_prepare"] is False
    assert "v360_unavailable" in {row["key"] for row in report["blockers"]}


def test_doctor_reports_invalid_ffprobe_json(tmp_path: Path) -> None:
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"synthetic placeholder")

    def runner(command: list[str]) -> CompletedProcess[str]:
        if "-filters" in command:
            return CompletedProcess(command, 0, stdout="v360", stderr="")
        return CompletedProcess(command, 0, stdout="not-json", stderr="")

    report = panorama_doctor(
        media,
        projection="equirectangular_mono",
        tool_resolver=_resolver({"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}),
        command_runner=runner,
    )

    assert report["ready_to_prepare"] is False
    assert "ffprobe_invalid_json" in {row["key"] for row in report["blockers"]}


def test_doctor_can_check_tools_without_media() -> None:
    report = panorama_doctor(
        None,
        tool_resolver=_resolver({"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}),
        command_runner=_successful_runner,
    )

    assert report["ok"] is True
    assert report["ready_to_prepare"] is False
    assert report["capabilities"]["v360"] is True
    assert report["profile"] is None
