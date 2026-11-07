from __future__ import annotations

import json
import subprocess
import sys
from types import ModuleType
from contextlib import nullcontext
from pathlib import Path

import video_knowledge_pipeline.ocr_backfill as ocr_backfill
import video_knowledge_pipeline.captiocr_resolver as captiocr_resolver
from video_knowledge_pipeline.captiocr_resolver import resolve_tesseract_runtime


def test_explicit_rapidocr_reuses_offline_owner_and_keeps_low_confidence_pending(tmp_path,monkeypatch):
    owner=tmp_path/'ebook_markdown_pipeline'
    owner.mkdir()
    monkeypatch.setattr('video_knowledge_pipeline.path_defaults.workspace_root',lambda:tmp_path)
    module=ModuleType('ebook_markdown_pipeline.ocr_providers')
    called=[]
    def create(**kwargs):
        called.append(kwargs)
        return lambda path:[{'text':'可疑识别','score':.4}]
    module.create_rapidocr_engine=create
    module.rapidocr_engine_context=nullcontext
    module.normalize_rapidocr_blocks=lambda raw:raw
    monkeypatch.setitem(sys.modules,'ebook_markdown_pipeline',ModuleType('ebook_markdown_pipeline'))
    monkeypatch.setitem(sys.modules,'ebook_markdown_pipeline.ocr_providers',module)
    runner,rows=ocr_backfill._run_rapidocr_candidates([{'index':1,'image_path':'frame.png'}])
    assert called==[{'device':'cpu','offline':True}]
    assert runner['available'] is True
    assert rows[0]['ok'] is False
    assert rows[0]['authoritative_observation'] is True
    assert rows[0]['recognition_status']=='suspicious'
    assert rows[0]['ocr_blocks'][0]['text']=='可疑识别'


def test_missing_rapidocr_owner_is_preflight_blocked_without_execution(tmp_path,monkeypatch):
    monkeypatch.setattr('video_knowledge_pipeline.path_defaults.workspace_root',lambda:tmp_path)
    runner,rows=ocr_backfill._run_rapidocr_candidates([{'index':1,'image_path':'frame.png'}])
    assert runner['available'] is False
    assert rows[0]['executed'] is False
    assert 'not found' in rows[0]['stderr']


def test_stale_image_bound_ocr_import_is_rejected(tmp_path):
    image=tmp_path/'frame.png'
    image.write_bytes(b'new pixels')
    result=ocr_backfill._results_from_imported_entries([{'index':1,'text':'旧图文字',
        'source_sha256':'0'*64}], [{'index':1,'image_path':str(image)}])[0]
    assert result['ok'] is False
    assert result['authoritative_observation'] is False
    assert result['stderr']=='OCR source image hash mismatch'


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    assets = bundle / "assets"
    assets.mkdir(parents=True)
    frame = assets / "frame.jpg"
    frame.write_bytes(b"synthetic-image")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 1,
                    "visual_route": "semantic_frame",
                    "frame_paths": [str(frame)],
                    "quality_issues": ["missing_visual_text"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return bundle


def test_tesseract_runtime_reports_requested_missing_languages(tmp_path: Path) -> None:
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"")
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"fixture")

    result = resolve_tesseract_runtime(
        explicit_cmd=executable,
        explicit_tessdata=tessdata,
        required_languages="chi_sim+eng",
    )

    assert result["runtime_available"] is True
    assert result["installed_languages"] == ["eng"]
    assert result["requested_languages"] == ["chi_sim", "eng"]
    assert result["missing_languages"] == ["chi_sim"]
    assert result["language_ready"] is False
    assert result["status"] == "missing_language_packs"


def test_tesseract_runtime_auto_selects_language_ready_installation(
    tmp_path: Path, monkeypatch
) -> None:
    incomplete_root = tmp_path / "incomplete"
    complete_root = tmp_path / "complete"
    incomplete_cmd = incomplete_root / "tesseract.exe"
    complete_cmd = complete_root / "tesseract.exe"
    incomplete_cmd.parent.mkdir(parents=True)
    complete_cmd.parent.mkdir(parents=True)
    incomplete_cmd.write_bytes(b"")
    complete_cmd.write_bytes(b"")
    incomplete_data = incomplete_root / "tessdata"
    complete_data = complete_root / "tessdata"
    incomplete_data.mkdir()
    complete_data.mkdir()
    (incomplete_data / "chi_sim.traineddata").write_bytes(b"fixture")
    (complete_data / "chi_sim.traineddata").write_bytes(b"fixture")
    (complete_data / "eng.traineddata").write_bytes(b"fixture")

    monkeypatch.setattr(
        captiocr_resolver,
        "_tesseract_cmd_candidates",
        lambda explicit_cmd=None: [incomplete_cmd, complete_cmd],
    )

    result = resolve_tesseract_runtime(required_languages="chi_sim+eng")

    assert result["status"] == "ready"
    assert result["cmd"] == str(complete_cmd.resolve())
    assert result["tessdata_prefix"] == str(complete_data.resolve())
    assert result["missing_languages"] == []
    assert result["selection_reason"] == "required_languages_ready"


def test_ocr_missing_language_fails_before_subprocess_and_is_visible(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    tesseract = {
        "available": True,
        "runtime_available": True,
        "cmd": str(tmp_path / "tesseract.exe"),
        "tessdata_prefix": str(tmp_path / "tessdata"),
        "requested_languages": ["chi_sim", "eng"],
        "installed_languages": ["eng"],
        "missing_languages": ["chi_sim"],
        "language_ready": False,
        "status": "missing_language_packs",
    }
    monkeypatch.setattr(ocr_backfill, "resolve_tesseract_runtime", lambda **kwargs: tesseract)
    monkeypatch.setattr(
        ocr_backfill,
        "_load_captiocr_runner",
        lambda *args, **kwargs: (
            {
                "available": False,
                "name": "captiocr",
                "root": "",
                "tesseract": tesseract,
                "error": "CaptiOCR unavailable",
            },
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        ocr_backfill.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess must not run")),
    )

    result = ocr_backfill.run_ocr_backfill(bundle, execute=True, language="chi_sim+eng")

    assert result["ok"] is False
    assert result["status"] == "missing_language_packs"
    assert result["capabilities"]["tesseract"]["missing_languages"] == ["chi_sim"]
    assert result["summary"]["failed"] == 0
    assert result["summary"]["blocked"] == 1
    assert result["summary"]["whole_bundle"]["executed"] == 0
    assert "missing_language_packs" in result["items"][0]["stderr"]
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "missing_language_packs" in report
    assert "chi_sim" in report


def test_multilingual_ocr_bypasses_captiocr_and_runs_tesseract_once(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    tesseract = {
        "available": True,
        "runtime_available": True,
        "cmd": str(tmp_path / "tesseract.exe"),
        "tessdata_prefix": str(tmp_path / "tessdata"),
        "requested_languages": ["chi_sim", "eng"],
        "installed_languages": ["chi_sim", "eng"],
        "missing_languages": [],
        "language_ready": True,
        "status": "ready",
    }
    monkeypatch.setattr(ocr_backfill, "resolve_tesseract_runtime", lambda **kwargs: tesseract)
    monkeypatch.setattr(
        ocr_backfill,
        "_load_captiocr_runner",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CaptiOCR must be bypassed for multilingual OCR")),
    )
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, stdout="level\tconf\ttext\n5\t95\t中文\n5\t95\tEnglish\n", stderr="")

    monkeypatch.setattr(ocr_backfill.subprocess, "run", fake_run)

    result = ocr_backfill.run_ocr_backfill(bundle, execute=True, language="chi_sim+eng")

    assert result["ok"] is True
    assert result["runner"]["name"] == "tesseract_cli"
    assert result["runner"]["route_reason"] == "multilingual_request_requires_tesseract_cli"
    assert result["capabilities"]["captiocr"]["available"] is False
    assert commands == [[tesseract["cmd"], str((bundle / "assets" / "frame.jpg").resolve()), "stdout", "-l", "chi_sim+eng", "tsv"]]
    assert environments[0]["TESSDATA_PREFIX"] == tesseract["tessdata_prefix"]
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[0]["visual_text"] == "中文 English"


def test_tesseract_low_confidence_and_malformed_output_are_not_quality_success():
    text, confidence = ocr_backfill._tesseract_tsv_observation('level\tconf\ttext\n5\t10\tR或去者鼻\n')
    assert text == 'R或去者鼻'
    assert confidence['valid_tsv'] and not confidence['quality_threshold_passed']
    text, confidence = ocr_backfill._tesseract_tsv_observation('unstructured engine error')
    assert text == '' and not confidence['valid_tsv']


def test_tesseract_empty_valid_observation_is_distinct_from_parse_failure():
    text, confidence = ocr_backfill._tesseract_tsv_observation('level\tconf\ttext\n1\t-1\t\n')
    assert text == '' and confidence['valid_tsv']
    assert confidence['word_count'] == 0


def test_multilingual_missing_pack_bypasses_captiocr_and_subprocess(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    tesseract = {
        "available": True,
        "runtime_available": True,
        "cmd": str(tmp_path / "tesseract.exe"),
        "tessdata_prefix": str(tmp_path / "tessdata"),
        "requested_languages": ["chi_sim", "eng"],
        "installed_languages": ["eng"],
        "missing_languages": ["chi_sim"],
        "language_ready": False,
        "status": "missing_language_packs",
    }
    monkeypatch.setattr(ocr_backfill, "resolve_tesseract_runtime", lambda **kwargs: tesseract)
    monkeypatch.setattr(
        ocr_backfill,
        "_load_captiocr_runner",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CaptiOCR must be bypassed for multilingual OCR")),
    )
    monkeypatch.setattr(
        ocr_backfill.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess must not run")),
    )

    result = ocr_backfill.run_ocr_backfill(bundle, execute=True, language="chi_sim+eng")

    assert result["ok"] is False
    assert result["status"] == "missing_language_packs"
    assert result["runner"]["name"] == "tesseract_cli"
    assert result["runner"]["route_reason"] == "multilingual_request_requires_tesseract_cli"


def test_ocr_reports_both_backends_unavailable(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    tesseract = {
        "available": False,
        "runtime_available": False,
        "cmd": "",
        "tessdata_prefix": "",
        "requested_languages": ["eng"],
        "installed_languages": [],
        "missing_languages": ["eng"],
        "language_ready": False,
        "status": "runtime_unavailable",
    }
    monkeypatch.setattr(
        ocr_backfill,
        "_load_captiocr_runner",
        lambda *args, **kwargs: (
            {
                "available": False,
                "name": "captiocr",
                "root": "",
                "tesseract": tesseract,
                "error": "CaptiOCR import unavailable",
            },
            None,
            None,
        ),
    )

    result = ocr_backfill.run_ocr_backfill(bundle, execute=True, language="eng")

    assert result["ok"] is False
    assert result["status"] == "ocr_backend_unavailable"
    assert result["capabilities"]["captiocr"]["available"] is False
    assert result["capabilities"]["tesseract"]["runtime_available"] is False
    assert result["recovery_commands"]
