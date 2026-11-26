from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import wave
from pathlib import Path
from typing import Any, Sequence

import av
import numpy as np

SCHEMA = "video_knowledge_pipeline.pre_publish_health_check.v1"
SAFE_SUBTITLE_WIDTH_MAX = 900
PROHIBITED_TERMS: tuple[str, ...] = ()


def check_container_and_stream(mp4_path: Path) -> dict[str, Any]:
    """1. 容器与流式播放规范 (FastStart + yuv420p + 连续解包探测)"""
    if not mp4_path.is_file():
        raise FileNotFoundError(f"成片文件不存在: {mp4_path}")

    with open(mp4_path, "rb") as f:
        head_1mb = f.read(1024 * 1024)
        has_faststart = b"moov" in head_1mb

    container = av.open(str(mp4_path))
    v_stream = container.streams.video[0]
    a_stream = container.streams.audio[0]

    v_codec = v_stream.codec_context.name
    v_pix_fmt = v_stream.codec_context.pix_fmt
    v_w = v_stream.width
    v_h = v_stream.height
    v_fps = float(v_stream.average_rate)
    v_dur = float(v_stream.duration * v_stream.time_base) if v_stream.duration else None

    a_codec = a_stream.codec_context.name
    a_rate = a_stream.codec_context.sample_rate
    a_channels = a_stream.codec_context.channels
    a_dur = float(a_stream.duration * a_stream.time_base) if a_stream.duration else None

    v_dec = 0
    for p in container.demux(v_stream):
        for _ in p.decode():
            if _ is not None:
                v_dec += 1
        if v_dec >= 30:
            break

    container.close()

    return {
        "faststart": has_faststart,
        "video_codec": v_codec,
        "pix_fmt": v_pix_fmt,
        "resolution": f"{v_w}x{v_h}",
        "fps": round(v_fps, 2),
        "video_duration_s": round(v_dur, 3) if v_dur else "N/A",
        "audio_codec": a_codec,
        "audio_sample_rate": a_rate,
        "audio_channels": a_channels,
        "audio_duration_s": round(a_dur, 3) if a_dur else "N/A",
        "decodable_probe": v_dec >= 30,
        "valid": has_faststart and v_pix_fmt == "yuv420p" and v_dec >= 30,
    }


def check_audio_loudness_and_peaks(wav_path: Path) -> dict[str, Any]:
    """2. 声学与响度安全规范 (峰值防有损转码爆音)"""
    if not wav_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {wav_path}")

    with wave.open(str(wav_path), "rb") as wf:
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        dur = nframes / framerate

        chunk_size = 48000 * 5
        max_abs = 0.0
        sum_sq = 0.0
        total_samples = 0

        while True:
            raw = wf.readframes(chunk_size)
            if not raw:
                break
            if sampwidth == 2:
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 3:
                b = bytearray(raw)
                s24 = np.zeros(len(b) // 3, dtype=np.float32)
                for i in range(len(s24)):
                    v = int.from_bytes(b[i*3 : (i+1)*3], 'little', signed=True)
                    s24[i] = v / 8388608.0
                data = s24
            else:
                data = np.frombuffer(raw, dtype=np.float32)

            curr_max = float(np.max(np.abs(data)))
            if curr_max > max_abs:
                max_abs = curr_max
            sum_sq += float(np.sum(data ** 2))
            total_samples += len(data)

    rms = math.sqrt(sum_sq / max(1, total_samples))
    peak_dbfs = 20 * math.log10(max(1e-6, max_abs))
    rms_dbfs = 20 * math.log10(max(1e-6, rms))

    return {
        "duration_s": round(dur, 3),
        "peak_dbfs": round(peak_dbfs, 2),
        "rms_dbfs": round(rms_dbfs, 2),
        "no_clipping": peak_dbfs <= -0.5,
        "valid": peak_dbfs <= -0.5,
    }


def check_subtitles_and_compliance(
    srt_path: Path,
    corrections_path: Path | None = None,
    prohibited_terms: Sequence[str] = PROHIBITED_TERMS,
) -> dict[str, Any]:
    """3. 字幕合规、安全区与历史错词回退门禁"""
    if not srt_path.is_file():
        raise FileNotFoundError(f"SRT 文件不存在: {srt_path}")

    content = srt_path.read_text(encoding="utf-8")

    # 3.1 事实红线与禁止用词扫描
    prohibited_hits = []
    for term in prohibited_terms:
        if term in content:
            prohibited_hits.append(term)

    # 3.2 历史错字回退扫描 (SSOT 字典)
    corrections = {}
    if corrections_path and corrections_path.is_file():
        corrections = json.loads(corrections_path.read_text(encoding="utf-8"))

    flattened_cues = re.sub(r"\n(?!\d+\n\d{2}:)", "", content)
    regression_hits = []
    for typo, target in corrections.items():
        if len(typo) >= 2 and (typo in flattened_cues or typo in content):
            regression_hits.append(f"{typo} (应为: {target})")

    # 3.3 字幕安全宽度超标与时序检查
    blocks = re.split(r"\n\s*\n", content.strip())
    max_char_len = 0
    longest_cue = ""
    cue_count = 0
    time_overlap_errors = []
    prev_end = 0.0

    def parse_time(ts: str) -> float:
        h, m, rest = ts.strip().split(":")
        s, ms = rest.split(",")
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) >= 3:
            cue_count += 1
            t_s, t_e = lines[1].split(" --> ")
            s_sec = parse_time(t_s)
            e_sec = parse_time(t_e)
            if s_sec < prev_end - 0.05:
                time_overlap_errors.append(f"Cue {lines[0]}: {t_s} overlap with prev {prev_end:.3f}")
            prev_end = e_sec

            for l in lines[2:]:
                if len(l) > max_char_len:
                    max_char_len = len(l)
                    longest_cue = l

    valid = (
        len(prohibited_hits) == 0 and
        len(regression_hits) == 0 and
        len(time_overlap_errors) == 0 and
        max_char_len <= 16
    )

    return {
        "cue_count": cue_count,
        "prohibited_term_hits": prohibited_hits,
        "regression_hits": regression_hits,
        "max_line_chars": max_char_len,
        "longest_line_sample": longest_cue,
        "time_order_valid": len(time_overlap_errors) == 0,
        "no_overflow_risk": max_char_len <= 16,
        "valid": valid,
    }


def check_covers(cover_dir: Path = Path(".")) -> dict[str, Any]:
    """4. 宣发物料与多比例封面完备性 (9:16/3:4/16:9/4:3)"""
    ratio_aliases = {
        "9:16": ["9_16", "9比16"],
        "3:4": ["3_4", "3比4"],
        "16:9": ["16_9", "16比9"],
        "4:3": ["4_3", "4比3"],
    }
    found_covers = {}
    for r, aliases in ratio_aliases.items():
        found = False
        for a in aliases:
            matches = list(cover_dir.glob(f"*封面*{a}*.png")) + list(cover_dir.glob(f"*封面*{a}*.jpg"))
            if matches:
                found = True
                break
        found_covers[r] = found
    all_ready = all(found_covers.values())
    return {
        "ratios_ready": found_covers,
        "covers_complete": all_ready,
        "valid": all_ready,
    }


def run_full_pre_publish_gate(
    mp4_path: Path,
    wav_path: Path,
    srt_path: Path,
    corrections_path: Path | None = None,
    cover_dir: Path = Path("."),
) -> tuple[bool, dict[str, Any]]:
    c_res = check_container_and_stream(mp4_path)
    a_res = check_audio_loudness_and_peaks(wav_path)
    s_res = check_subtitles_and_compliance(srt_path, corrections_path)
    cov_res = check_covers(cover_dir)

    all_passed = bool(
        c_res.get("valid") and
        a_res.get("valid") and
        s_res.get("valid") and
        cov_res.get("valid")
    )

    summary = {
        "all_passed": all_passed,
        "container": c_res,
        "acoustics": a_res,
        "subtitles": s_res,
        "covers": cov_res,
    }
    return all_passed, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pre-publish quality gate across MP4, audio, subtitles and covers."
    )
    parser.add_argument("--mp4", type=Path, required=True, help="Master MP4 file")
    parser.add_argument("--wav", type=Path, required=True, help="Master audio WAV file")
    parser.add_argument("--srt", type=Path, required=True, help="Master subtitle SRT file")
    parser.add_argument("--corrections", type=Path, default=None, help="Corrections JSON")
    parser.add_argument("--covers", type=Path, default=Path("."), help="Covers directory")
    args = parser.parse_args(argv)

    passed, res = run_full_pre_publish_gate(
        args.mp4, args.wav, args.srt, args.corrections, args.covers
    )

    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
