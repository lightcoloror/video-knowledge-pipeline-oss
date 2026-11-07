from __future__ import annotations

import hashlib
import math
import struct
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageDraw

from video_knowledge_pipeline.media_tools import local_tool_subprocess_env, resolve_media_tool


WIDTH = 768
HEIGHT = 384
FPS = 5
FRAME_COUNT = 10


def generate_panorama_fixture(output_dir: Path) -> dict[str, object]:
    """Generate one tiny, local ERP video with deterministic visual/audio cues."""

    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    frame_dir = root / "frames"
    frame_dir.mkdir()
    for frame_index in range(FRAME_COUNT):
        image = Image.new("RGB", (WIDTH, HEIGHT), "black")
        draw = ImageDraw.Draw(image)
        bands = (
            (0, WIDTH // 4, "BACK-SEAM", (130, 40, 40)),
            (WIDTH // 4, WIDTH // 2, "LEFT", (40, 110, 60)),
            (WIDTH // 2, 3 * WIDTH // 4, "FRONT", (40, 70, 150)),
            (3 * WIDTH // 4, WIDTH, "RIGHT", (120, 90, 30)),
        )
        for left, right, label, colour in bands:
            draw.rectangle((left, 0, right - 1, HEIGHT - 1), fill=colour)
            draw.text((left + 10, HEIGHT // 2), label, fill="white")
        draw.rectangle((0, 0, WIDTH - 1, 54), fill=(85, 35, 120))
        draw.text((WIDTH // 2 - 15, 18), "UP", fill="white")
        draw.rectangle((0, HEIGHT - 55, WIDTH - 1, HEIGHT - 1), fill=(20, 105, 110))
        draw.text((WIDTH // 2 - 20, HEIGHT - 38), "DOWN", fill="white")
        draw.rectangle((0, 120, 24, 155), fill="yellow")
        draw.rectangle((WIDTH - 25, 120, WIDTH - 1, 155), fill="yellow")
        draw.text((4, 132), "S", fill="black")
        draw.text((WIDTH - 20, 132), "S", fill="black")
        moving_x = WIDTH // 2 + frame_index * 12
        draw.ellipse((moving_x, 210, moving_x + 28, 238), fill="orange", outline="white")
        draw.text((WIDTH // 2 + 16, 88), "PANORAMA OCR 360", fill="white")
        draw.text((12, 12), f"SYNTHETIC FRAME {frame_index:02d}", fill="white")
        image.save(frame_dir / f"frame-{frame_index:03d}.png", optimize=False)

    audio_path = root / "synthetic-speech-like.wav"
    _write_speech_like_audio(audio_path, seconds=FRAME_COUNT / FPS)
    media_path = root / "synthetic-panorama-erp.mp4"
    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to generate the panorama acceptance fixture")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(frame_dir / "frame-%03d.png"),
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-metadata:s:v:0",
        "projection=equirectangular",
        str(media_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=local_tool_subprocess_env(),
    )
    if completed.returncode != 0 or not media_path.is_file():
        raise RuntimeError(f"synthetic panorama generation failed: {completed.stderr[:500]}")
    return {
        "media_path": media_path,
        "audio_path": audio_path,
        "frame_dir": frame_dir,
        "media_sha256": _sha256(media_path),
        "audio_sha256": _sha256(audio_path),
        "duration_seconds": FRAME_COUNT / FPS,
        "contains": [
            "six_direction_labels",
            "seam_crossing_marker",
            "moving_marker",
            "ocr_label",
            "synthetic_speech_like_audio",
        ],
    }


def _write_speech_like_audio(path: Path, *, seconds: float) -> None:
    sample_rate = 16_000
    sample_count = int(round(seconds * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_count):
            elapsed = index / sample_rate
            syllable = int(elapsed * 5) % 4
            frequency = (180.0, 230.0, 200.0, 260.0)[syllable]
            envelope = 0.25 + 0.75 * abs(math.sin(math.pi * 5 * elapsed))
            value = int(8_000 * envelope * math.sin(2 * math.pi * frequency * elapsed))
            frames.extend(struct.pack("<h", value))
        handle.writeframes(bytes(frames))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
