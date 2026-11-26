from __future__ import annotations

import tempfile
import wave
from pathlib import Path
import numpy as np

from video_knowledge_pipeline.pre_publish_health_check import (
    check_audio_loudness_and_peaks,
    check_subtitles_and_compliance,
    check_covers,
)


def test_audio_loudness_check():
    with tempfile.TemporaryDirectory() as td:
        wav_file = Path(td) / "test.wav"
        # Create 1s stereo 16-bit sine wave at -6dB
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sig = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        stereo = np.column_stack([sig, sig]).flatten()

        with wave.open(str(wav_file), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        res = check_audio_loudness_and_peaks(wav_file)
        assert res["valid"] is True
        assert -6.5 <= res["peak_dbfs"] <= -5.5
        assert res["no_clipping"] is True


def test_subtitle_compliance_catches_violations():
    with tempfile.TemporaryDirectory() as td:
        srt_file = Path(td) / "bad.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\n包含示例禁用词的字幕\n\n"
            "2\n00:00:03,500 --> 00:00:05,000\n合成错词甲\n",
            encoding="utf-8",
        )
        corr_file = Path(td) / "corr.json"
        corr_file.write_text('{"合成错词甲": "合成正词甲"}', encoding="utf-8")

        res = check_subtitles_and_compliance(
            srt_file,
            corr_file,
            prohibited_terms=["示例禁用词"],
        )
        assert res["valid"] is False
        assert "示例禁用词" in res["prohibited_term_hits"]
        assert len(res["regression_hits"]) == 1


def test_subtitle_compliance_passes_clean():
    with tempfile.TemporaryDirectory() as td:
        srt_file = Path(td) / "good.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\n合成测试合规字幕\n\n"
            "2\n00:00:03,500 --> 00:00:05,000\n合成正词甲\n",
            encoding="utf-8",
        )
        corr_file = Path(td) / "corr.json"
        corr_file.write_text('{"合成错词甲": "合成正词甲"}', encoding="utf-8")

        res = check_subtitles_and_compliance(
            srt_file,
            corr_file,
            prohibited_terms=["示例禁用词"],
        )
        assert res["valid"] is True
        assert res["prohibited_term_hits"] == []
        assert res["regression_hits"] == []


def test_covers_check():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "封面-示例-9比16-主图.jpg").touch()
        (tdp / "封面-示例-3比4-竖图.jpg").touch()
        (tdp / "封面-示例-16比9-横屏.png").touch()
        (tdp / "封面-示例-4比3-横图.png").touch()

        res = check_covers(tdp)
        assert res["valid"] is True
        assert res["covers_complete"] is True
