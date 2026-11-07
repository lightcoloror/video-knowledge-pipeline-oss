from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.punctuation_model_stage import run_punctuation_model_stage


def _write_bundle(root: Path) -> Path:
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"normalized_transcript_json": "normalized-transcript.json"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "normalized-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 5, "text": "大家好今天讨论获客"},
                    {"start": 5, "end": 10, "text": "首先建立信任然后确认需求"},
                    {"start": 10, "end": 15, "text": "最后约定下一步"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


class _GoodPunctuationModel:
    def generate(self, *, input: str):
        text = input.replace("今天", "，今天").replace("首先", "。首先").replace("然后", "，然后").replace("最后", "。最后")
        if not text.endswith("。"):
            text += "。"
        return [{"text": text}]


class _MutatingPunctuationModel:
    def generate(self, *, input: str):
        return [{"text": input.replace("获客", "销售") + "。"}]


def test_punctuation_stage_promotes_only_character_locked_output(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    monkeypatch.setattr(
        "video_knowledge_pipeline.punctuation_model_stage._run_model_blocks",
        lambda root, blocks, model, device: (
            [_GoodPunctuationModel().generate(input="".join(cue.text for cue in block))[0]["text"] for block in blocks],
            {"python_executable": "fake-python"},
        ),
    )

    result = run_punctuation_model_stage(bundle, execute=True, promote=True, write=True)

    assert result["status"] == "completed"
    assert result["char_lock_passed"] is True
    assert result["quality_gate_passed"] is True
    assert result["promoted"] is True
    punctuated = json.loads((bundle / "punctuated-transcript.json").read_text(encoding="utf-8"))
    corrected = json.loads((bundle / "corrected-transcript.json").read_text(encoding="utf-8"))
    assert punctuated["character_lock"]["passed"] is True
    assert punctuated["segments"] == corrected["segments"]
    assert "，" in "".join(row["text"] for row in punctuated["segments"])
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["punctuated_transcript_json"] == "punctuated-transcript.json"


def test_punctuation_stage_rejects_any_text_mutation(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path / "mutating")
    monkeypatch.setattr(
        "video_knowledge_pipeline.punctuation_model_stage._run_model_blocks",
        lambda root, blocks, model, device: (
            [_MutatingPunctuationModel().generate(input="".join(cue.text for cue in block))[0]["text"] for block in blocks],
            {"python_executable": "fake-python"},
        ),
    )

    result = run_punctuation_model_stage(bundle, execute=True, promote=True, write=True)

    assert result["status"] == "char_lock_failed"
    assert result["char_lock_passed"] is False
    assert result["promoted"] is False
    assert not (bundle / "punctuated-transcript.json").exists()
    assert not (bundle / "corrected-transcript.json").exists()


def test_punctuation_stage_preview_does_not_load_model(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path / "preview")
    monkeypatch.setattr(
        "video_knowledge_pipeline.punctuation_model_stage._run_model_blocks",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = run_punctuation_model_stage(bundle, execute=False, write=False)

    assert result["status"] == "planned"
    assert result["execute"] is False
    assert not (bundle / "punctuated-transcript.json").exists()