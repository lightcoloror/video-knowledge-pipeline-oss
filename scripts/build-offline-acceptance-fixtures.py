"""Build small, non-sensitive VKP offline acceptance fixtures.

Public adaptation of VKP's original task-local fixture builder. Source evidence:
upstream commit ``d8541dc19247e37e4d6bc6a3a1e1689a2a98b6f0``, blob
``d313a4e1509aa9efc88471add24a0d08aa902866``, path
``scripts/build-offline-acceptance-fixtures.py`` (AGPL-3.0-only). The fixture
contract schema, the three fixture IDs/types and the original helper functions
``_render_card``, ``_mux_cards``, ``_duration`` and ``_sha256`` are retained.

This copy removes private task-local source-review discovery, environment
fallback and host font search. Default source media is a first-party generated
PCM tone (stdlib ``wave``/``math``) and default cards are synthetic ASCII drawn
with Pillow. A tone is not a speech recording and does not establish ASR
quality. The original Chinese card text is retained as an explicit optional
``--card-language zh`` mode that requires an operator-supplied local font; the
default never reads host fonts. Operators may explicitly declare local speech
media with an audio specification (local path, expected SHA-256, provenance and
use-rights statement); every declaration is validated before any output is
written or rendered. No audio, fonts, binaries or model weights are bundled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCHEMA = "video_knowledge_pipeline.offline_fixture_set.v1"

DEFAULT_TONE_SECONDS = 2.0
DEFAULT_TONE_FREQUENCY_HZ = 440.0
TONE_SAMPLE_RATE = 16000
TONE_AMPLITUDE = 0.18

# Small documented fixture bounds: validated before any output, allocation or
# audio probe. They keep the default/example media tiny and prevent a declared
# tone setting from allocating an unbounded buffer.
TONE_SECONDS_MIN = 0.25
TONE_SECONDS_MAX = 5.0
TONE_FREQUENCY_MIN_HZ = 20.0
TONE_FREQUENCY_MAX_HZ = 4000.0

CARD_SIZE = (1280, 720)
CARD_LANGUAGES = ("ascii", "zh")
MEDIA_THREADS = 2
FFMPEG_TIMEOUT_SECONDS = 120
FFPROBE_TIMEOUT_SECONDS = 60

SAFE_CARD_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")

EXPECTED_ARTIFACTS = [
    "normalized-transcript.json",
    "frame-manifest.json",
    "ocr_or_visual_evidence",
    "review.html",
    "exports/knowledge-note.md",
    "exports/full-transcript.md",
]

# The original three fixture IDs and types are preserved. ``cards`` is the
# default synthetic ASCII set; ``cards_zh`` restores the original Chinese card
# text as an optional mode that requires an explicitly supplied local font.
FIXTURES: list[dict[str, Any]] = [
    {
        "id": "talking-head-synthetic",
        "type": "talking_video",
        "cards": [
            (
                "Local talking video fixture",
                [
                    "Audio: first-party generated tone",
                    "Goal: ASR and timestamp contract",
                    "Boundary: no semantics claim",
                ],
            )
        ],
        "cards_zh": [
            (
                "本地讲话视频 fixture",
                ["讲话音频：开源测试素材", "目标：ASR 与时间戳", "边界：不要求语义正确"],
            )
        ],
    },
    {
        "id": "slides-screen-synthetic",
        "type": "slides_or_screen_video",
        "cards": [
            ("Lesson page 1: input", ["Local video or audio", "Source hash", "Processing config"]),
            ("Lesson page 2: processing", ["Local ASR contract", "Frame extraction", "Unified timeline"]),
            ("Lesson page 3: output", ["Review bundle", "Transcript targets", "Knowledge note targets"]),
        ],
        "cards_zh": [
            ("课程页一：输入", ["本地视频或音频", "来源哈希", "处理配置"]),
            ("课程页二：处理", ["SenseVoice ASR", "抽帧与 OCR", "统一时间线"]),
            ("课程页三：输出", ["审核包", "逐字稿", "知识笔记"]),
        ],
    },
    {
        "id": "mixed-visual-synthetic",
        "type": "mixed_visual_video",
        "cards": [
            ("Narration", ["Observe the overall flow", "Then locate difficult segments"]),
            ("Operation screen", ["State: local processing", "Progress: sixty percent", "Next: review screen text"]),
            ("Conclusion and risk", ["Do not invent visual findings", "External calls require approval"]),
        ],
        "cards_zh": [
            ("讲师说明", ["先观察整体流程", "再定位疑难片段"]),
            ("操作界面", ["状态：本地处理", "进度：百分之六十", "下一步：复核屏幕文字"]),
            ("结论与风险", ["视觉未执行时不得补故事", "外部调用需要明确批准"]),
        ],
    },
]


class FixtureInputError(ValueError):
    """A declared fixture input failed preflight; no fixture output was produced."""


# ---------------------------------------------------------------------------
# Small local seams. Tests replace these to stay synthetic and file-free.
# ---------------------------------------------------------------------------


def _path_exists(path: Path) -> bool:
    return path.exists()


def _is_directory(path: Path) -> bool:
    return path.is_dir()


def _is_regular_file(path: Path) -> bool:
    return path.is_file()


def _directory_has_entries(path: Path) -> bool:
    return any(path.iterdir())


def _resolve_output_path(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def _resolve_operator_path(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise FixtureInputError(f"expected a JSON object in {path}")
    return data


def _write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_json(path: Path, data: Any) -> None:
    _write_text_lf(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_media(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
    )


def _validate_tone_settings(seconds: float, frequency_hz: float) -> None:
    if not math.isfinite(seconds) or not (TONE_SECONDS_MIN <= seconds <= TONE_SECONDS_MAX):
        raise FixtureInputError(
            f"tone duration must be finite and between {TONE_SECONDS_MIN:g} and {TONE_SECONDS_MAX:g} seconds"
        )
    if not math.isfinite(frequency_hz) or not (TONE_FREQUENCY_MIN_HZ <= frequency_hz <= TONE_FREQUENCY_MAX_HZ):
        raise FixtureInputError(
            "tone frequency must be finite and between "
            f"{TONE_FREQUENCY_MIN_HZ:g} and {TONE_FREQUENCY_MAX_HZ:g} Hz"
        )


def _parse_duration_text(text: str) -> float:
    raw = (text or "").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise FixtureInputError(f"ffprobe returned a non-numeric duration: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise FixtureInputError(f"duration must be finite and positive: {raw!r}")
    rounded = round(value, 3)
    if rounded <= 0:
        raise FixtureInputError(f"duration is too small to represent at 0.001 s precision: {raw!r}")
    return rounded


def _probe_command(ffprobe: str, path: Path) -> list[str]:
    """ffprobe argv supported by the supplied ffprobe build (no ``-nostdin``).

    Input protocol is restricted to local files. Stdin is closed with
    ``subprocess.DEVNULL`` in ``_run_media``; this argv is a local-file request,
    not proof that every network path is blocked.
    """
    return [
        ffprobe,
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]


def _duration(ffprobe: str, path: Path) -> float:
    completed = _run_media(_probe_command(ffprobe, path), FFPROBE_TIMEOUT_SECONDS)
    return _parse_duration_text(completed.stdout)


# ---------------------------------------------------------------------------
# Default first-party media
# ---------------------------------------------------------------------------


def _write_tone(path: Path, seconds: float = DEFAULT_TONE_SECONDS, frequency_hz: float = DEFAULT_TONE_FREQUENCY_HZ) -> None:
    _validate_tone_settings(seconds, frequency_hz)
    frames = int(round(TONE_SAMPLE_RATE * seconds))
    if frames <= 0:  # defensive; bounds already guarantee a positive count
        raise FixtureInputError("tone duration rounds to zero samples")
    amplitude = TONE_AMPLITUDE * 32767
    payload = bytearray()
    for index in range(frames):
        sample = int(round(amplitude * math.sin(2.0 * math.pi * frequency_hz * index / TONE_SAMPLE_RATE)))
        payload += int(sample).to_bytes(2, "little", signed=True)
    _ensure_parent_dir(path.parent)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TONE_SAMPLE_RATE)
        handle.writeframes(bytes(payload))


def _font(font_path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the declared local font, or Pillow's bundled default font.

    The default path never searches host font directories. ``load_default(size)``
    exists since Pillow 10.1; the TypeError fallback keeps Pillow 10.0 usable.
    """
    if font_path is None:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()
    return ImageFont.truetype(str(font_path), size=size)


def _render_card(path: Path, title: str, bullets: list[str], *, accent: tuple[int, int, int], font_path: Path | None = None) -> None:
    width, height = CARD_SIZE
    image = Image.new("RGB", CARD_SIZE, (245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 92), fill=accent)
    draw.text((54, 22), title, font=_font(font_path, 40), fill="white")
    draw.rectangle((54, 142, width - 54, height - 80), outline=(190, 197, 205), width=3)
    for index, bullet in enumerate(bullets, start=1):
        y = 180 + (index - 1) * 130
        draw.ellipse((86, y + 12, 118, y + 44), fill=accent)
        draw.text((145, y), bullet, font=_font(font_path, 34), fill=(28, 35, 43))
    draw.text((54, height - 50), "VKP synthetic offline acceptance fixture", font=_font(font_path, 20), fill=(95, 103, 112))
    image.save(path)


# ---------------------------------------------------------------------------
# FFmpeg concat/mux: generated relative card basenames with safe=1
# ---------------------------------------------------------------------------


def _concat_text(images: list[Path], card_duration: float) -> str:
    if not images:
        raise FixtureInputError("at least one card image is required")
    parents = {image.parent for image in images}
    if len(parents) != 1:
        raise FixtureInputError("all card images must share one directory")
    if not math.isfinite(card_duration) or card_duration <= 0:
        raise FixtureInputError("card duration must be finite and positive")
    lines = ["ffconcat version 1.0"]
    for image in images:
        if not SAFE_CARD_NAME.fullmatch(image.name):
            raise FixtureInputError(f"card file name is not safe for ffconcat: {image.name!r}")
        lines.extend([f"file '{image.name}'", f"duration {card_duration:.6f}"])
    lines.append(f"file '{images[-1].name}'")
    return "\n".join(lines) + "\n"


def _mux_command(ffmpeg: str, concat: Path, audio: Path, output: Path) -> list[str]:
    """Bounded local FFmpeg argv.

    The thread values are argv requests for input decoders, the encoder and the
    filter pools; they are not a measured OS thread/CPU/RAM peak.
    """
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-filter_threads",
        str(MEDIA_THREADS),
        "-filter_complex_threads",
        str(MEDIA_THREADS),
        "-f",
        "concat",
        "-safe",
        "1",
        "-protocol_whitelist",
        "file",
        "-threads",
        str(MEDIA_THREADS),
        "-i",
        str(concat),
        "-protocol_whitelist",
        "file",
        "-threads",
        str(MEDIA_THREADS),
        "-i",
        str(audio),
        "-threads",
        str(MEDIA_THREADS),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "25",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]


def _mux_cards(ffmpeg: str, ffprobe: str, images: list[Path], audio: Path, output: Path) -> None:
    if not images:
        raise FixtureInputError("at least one card image is required")
    if output.parent != images[0].parent:
        raise FixtureInputError("card images and the .mp4 output must share one directory")
    audio_duration = _duration(ffprobe, audio)
    card_duration = max(0.5, audio_duration / max(1, len(images)))
    concat = output.parent / "cards.ffconcat"
    _write_text_lf(concat, _concat_text(images, card_duration))
    _run_media(_mux_command(ffmpeg, concat, audio, output), FFMPEG_TIMEOUT_SECONDS)
    if not _is_regular_file(output) or _file_size(output) <= 0:
        raise FixtureInputError(f"FFmpeg did not create a nonempty media file: {output}")


# ---------------------------------------------------------------------------
# Input preflight
# ---------------------------------------------------------------------------


def _preflight_output_root(output: Path) -> None:
    if _path_exists(output):
        if not _is_directory(output):
            raise FixtureInputError(f"output path exists and is not a directory: {output}")
        if _directory_has_entries(output):
            raise FixtureInputError(
                "refusing to write into a nonempty output directory; pass a fresh explicit output root"
            )
    # A missing output root is created only after every declared input is validated.


def _preflight_card_font(card_font: str | None) -> Path | None:
    if not card_font:
        return None
    path = _resolve_operator_path(card_font)
    if not _is_regular_file(path):
        raise FixtureInputError(f"declared card font is not a regular local file: {path}")
    return path


def _fixture_plan(audio_by_id: dict[str, dict[str, Any]] | None = None, card_language: str = "ascii") -> list[dict[str, Any]]:
    """Describe all fixtures without writing or rendering anything."""
    if card_language not in CARD_LANGUAGES:
        raise FixtureInputError(f"card language must be one of: {', '.join(CARD_LANGUAGES)}")
    declared = audio_by_id or {}
    plan: list[dict[str, Any]] = []
    for spec in FIXTURES:
        audio = declared.get(spec["id"])
        if audio is None:
            entry_audio = {
                "kind": "first_party_generated_pcm_tone",
                "declared": False,
                "path": None,
                "sha256": None,
                "duration_seconds": None,
                "provenance": "first-party generated PCM tone (stdlib wave/math); not a speech recording",
                "use_rights_statement": "first-party generated locally; no third-party rights claimed",
                "source_project": "first-party generated PCM tone",
            }
        else:
            entry_audio = dict(audio)
            entry_audio["declared"] = True
        cards_key = "cards" if card_language == "ascii" else "cards_zh"
        plan.append(
            {
                "fixture_id": spec["id"],
                "fixture_type": spec["type"],
                "cards": [[title, list(bullets)] for title, bullets in spec[cards_key]],
                "audio": entry_audio,
            }
        )
    return plan


def _validate_audio_spec_entry(entry: Any, index: int, known_ids: set[str], seen: set[str]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise FixtureInputError(f"audio[{index}] must be an object")
    fixture_id = entry.get("fixture_id")
    if not isinstance(fixture_id, str) or fixture_id not in known_ids:
        raise FixtureInputError(f"audio[{index}].fixture_id is not one of the three original fixture IDs")
    if fixture_id in seen:
        raise FixtureInputError(f"audio[{index}] repeats fixture_id {fixture_id!r}")
    seen.add(fixture_id)
    path_text = entry.get("path")
    if not isinstance(path_text, str) or not path_text.strip():
        raise FixtureInputError(f"audio[{index}].path must be a nonempty local path")
    expected = entry.get("expected_sha256")
    if not isinstance(expected, str) or not SHA256_HEX.fullmatch(expected):
        raise FixtureInputError(f"audio[{index}].expected_sha256 must be 64 hexadecimal characters")
    provenance = entry.get("operator_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise FixtureInputError(f"audio[{index}].operator_provenance must be a nonempty statement")
    rights = entry.get("use_rights_statement")
    if not isinstance(rights, str) or not rights.strip():
        raise FixtureInputError(f"audio[{index}].use_rights_statement must be a nonempty statement")
    path = _resolve_operator_path(path_text)
    if not _is_regular_file(path):
        raise FixtureInputError(f"audio[{index}] path is not a regular local file: {path}")
    return {
        "fixture_id": fixture_id,
        "path": path,
        "expected_sha256": expected,
        "operator_provenance": provenance.strip(),
        "use_rights_statement": rights.strip(),
    }


def _preflight_operator_audio(spec_path: str | None, ffprobe: str) -> dict[str, dict[str, Any]]:
    """Validate every declared audio entry before any output write or render.

    Three ordered passes: shapes/existence for all entries, then SHA-256 for all,
    then duration probing for all. A defect in a later entry therefore fails
    before the first entry's media is probed or rendered.
    """
    if not spec_path:
        return {}
    data = _read_json_object(_resolve_operator_path(spec_path))
    entries = data.get("audio")
    if not isinstance(entries, list) or not entries:
        raise FixtureInputError("audio specification must contain a nonempty 'audio' list")
    known_ids = {spec["id"] for spec in FIXTURES}
    seen: set[str] = set()
    validated = [
        _validate_audio_spec_entry(entry, index, known_ids, seen) for index, entry in enumerate(entries)
    ]

    declared: dict[str, dict[str, Any]] = {}
    for entry in validated:
        digest = _sha256(entry["path"])
        if digest.lower() != entry["expected_sha256"].lower():
            raise FixtureInputError(
                f"audio SHA-256 mismatch for {entry['path']}: declared {entry['expected_sha256']}, computed {digest}"
            )
        declared[entry["fixture_id"]] = {
            "kind": "operator_supplied_speech_candidate",
            "path": entry["path"],
            "sha256": digest,
            "duration_seconds": None,
            "provenance": entry["operator_provenance"],
            "use_rights_statement": entry["use_rights_statement"],
            "source_project": "operator-declared local speech media",
        }

    for entry in declared.values():
        entry["duration_seconds"] = _duration(ffprobe, entry["path"])
    return declared


# ---------------------------------------------------------------------------
# Manifest semantics
# ---------------------------------------------------------------------------


def _audio_summary(audio: dict[str, Any]) -> dict[str, Any]:
    is_tone = audio["kind"] == "first_party_generated_pcm_tone"
    if is_tone:
        speech_reference = "none"
    else:
        speech_reference = (
            "operator-declared speech media; provenance/rights are declared only and not independently verified; "
            "no speech or ASR quality is established"
        )
    return {
        "source_audio_project": audio["source_project"],
        "source_audio_path": str(audio["path"]),
        "source_audio_sha256": audio["sha256"],
        "source_audio_duration_seconds": audio["duration_seconds"],
        "source_audio_kind": audio["kind"],
        "source_audio_provenance": audio["provenance"],
        "source_audio_use_rights_statement": audio["use_rights_statement"],
        "source_audio_independently_verified": False,
        "speech_reference": speech_reference,
        "asr_reference": "none",
        "quality_reference_required": False,
    }


def _manifest_entry(spec: dict[str, Any], *, video: Path, video_sha256: str, video_duration: float, audio: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "fixture_id": spec["id"],
        "fixture_type": spec["type"],
        "media_path": str(video),
        "sha256": video_sha256,
        "duration_seconds": video_duration,
    }
    entry.update(_audio_summary(audio))
    entry.update(
        {
            "media_scope": "synthetic_cards_plus_declared_local_audio_muxed_to_mp4",
            "timing_scope": "container_duration_and_declared_timeline_only",
            "hash_scope": (
                "sha256_of_declared_bytes_only; not source authenticity, semantic truth or speech quality"
            ),
            "expected_artifacts": list(EXPECTED_ARTIFACTS),
            "expected_artifacts_status": "targets_not_proven_outputs",
            "claims": {
                "normalized_transcript_created": False,
                "ocr_created": False,
                "human_review_created": False,
                "knowledge_notes_created": False,
                "asr_quality_established": False,
            },
            "privacy": (
                "synthetic_cards_plus_first_party_generated_tone_local_only"
                if audio["kind"] == "first_party_generated_pcm_tone"
                else "synthetic_cards_plus_operator_declared_local_media_local_only"
            ),
        }
    )
    return entry


def _aggregate_generated_media(
    fixtures: list[dict[str, Any]],
    *,
    card_language: str,
    font_declared: bool,
    tone_seconds: float,
    tone_frequency_hz: float,
) -> dict[str, Any]:
    """Derive aggregate provenance from the actual built fixtures."""
    tone_count = sum(
        1 for fixture in fixtures if fixture["source_audio_kind"] == "first_party_generated_pcm_tone"
    )
    operator_count = len(fixtures) - tone_count
    audio_kinds = sorted({fixture["source_audio_kind"] for fixture in fixtures})
    if card_language == "zh":
        cards = "synthetic_chinese_drawn_with_pillow_and_operator_declared_local_font"
    elif font_declared:
        cards = "synthetic_ascii_drawn_with_pillow_and_operator_declared_local_font"
    else:
        cards = "synthetic_ascii_drawn_with_pillow_default_font"
    return {
        "card_language": card_language,
        "cards": cards,
        "system_font_search": False,
        "external_asset_lookup": False,
        "operator_supplied_local_inputs": bool(operator_count or font_declared),
        "audio_kinds": audio_kinds,
        "generated_tone_fixtures": tone_count,
        "operator_declared_audio_fixtures": operator_count,
        "audio_durations_seconds": [fixture["source_audio_duration_seconds"] for fixture in fixtures],
        "tone_seconds_requested": tone_seconds if tone_count else None,
        "tone_frequency_hz": tone_frequency_hz if tone_count else None,
        "tone_sample_rate": TONE_SAMPLE_RATE if tone_count else None,
    }


def _speech_reference_summary(tone_count: int, operator_count: int) -> str:
    if not operator_count:
        return "none"
    if not tone_count:
        return (
            "operator-declared speech media for all fixtures; provenance/rights are declared only and not "
            "independently verified; no speech or ASR quality is established"
        )
    return (
        f"none for {tone_count} generated-tone fixtures; operator-declared speech media for {operator_count} "
        "fixtures, declared only and not independently verified; no speech or ASR quality is established"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    output_dir: str | Path,
    *,
    audio_spec_path: str | None = None,
    card_font_path: str | None = None,
    card_language: str = "ascii",
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    tone_seconds: float = DEFAULT_TONE_SECONDS,
    tone_frequency_hz: float = DEFAULT_TONE_FREQUENCY_HZ,
) -> dict[str, Any]:
    """Build all three fixtures into a fresh explicit output root.

    Returns ``{"manifest", "manifest_path", "output_dir"}``. On any failure an
    exception propagates and no success summary is printed; partial outputs are
    left in place as evidence.
    """
    _validate_tone_settings(tone_seconds, tone_frequency_hz)
    if card_language not in CARD_LANGUAGES:
        raise FixtureInputError(f"card language must be one of: {', '.join(CARD_LANGUAGES)}")

    output = _resolve_output_path(str(output_dir))
    _preflight_output_root(output)

    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FixtureInputError("ffmpeg and ffprobe are required on PATH or must be passed explicitly")

    declared_audio = _preflight_operator_audio(audio_spec_path, ffprobe)
    font_path = _preflight_card_font(card_font_path)
    if card_language == "zh" and font_path is None:
        raise FixtureInputError(
            "Chinese card text requires an explicitly supplied licensed local font via --card-font; "
            "the default never searches host fonts"
        )
    plan = _fixture_plan(declared_audio, card_language)

    _prepare_output_dir(output)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "fixture_count": len(plan),
        "asr_quality_evidence": False,
        "claims": {
            "normalized_transcript_created": False,
            "ocr_created": False,
            "human_review_created": False,
            "knowledge_notes_created": False,
            "asr_quality_established": False,
            "full_workflow_completed": False,
        },
        "fixtures": [],
    }

    for planned, spec in zip(plan, FIXTURES):
        fixture_dir = output / planned["fixture_id"]
        _ensure_parent_dir(fixture_dir)
        images: list[Path] = []
        for card_index, (title, bullets) in enumerate(planned["cards"], start=1):
            image_path = fixture_dir / f"card-{card_index:02d}.png"
            _render_card(image_path, title, bullets, accent=(36 + card_index * 30, 120, 170), font_path=font_path)
            images.append(image_path)

        audio_meta = planned["audio"]
        if audio_meta["kind"] == "first_party_generated_pcm_tone":
            tone_path = output / "generated-audio" / f"{planned['fixture_id']}-tone.wav"
            _write_tone(tone_path, tone_seconds, tone_frequency_hz)
            audio = {
                "kind": "first_party_generated_pcm_tone",
                "path": tone_path,
                "sha256": _sha256(tone_path),
                "duration_seconds": _duration(ffprobe, tone_path),
                "provenance": audio_meta["provenance"],
                "use_rights_statement": audio_meta["use_rights_statement"],
                "source_project": audio_meta["source_project"],
            }
        else:
            audio = dict(audio_meta)

        video_path = fixture_dir / f"{planned['fixture_id']}.mp4"
        _mux_cards(ffmpeg, ffprobe, images, audio["path"], video_path)
        manifest["fixtures"].append(
            _manifest_entry(
                spec,
                video=video_path,
                video_sha256=_sha256(video_path),
                video_duration=_duration(ffprobe, video_path),
                audio=audio,
            )
        )

    manifest["generated_media"] = _aggregate_generated_media(
        manifest["fixtures"],
        card_language=card_language,
        font_declared=font_path is not None,
        tone_seconds=tone_seconds,
        tone_frequency_hz=tone_frequency_hz,
    )
    manifest["speech_reference"] = _speech_reference_summary(
        manifest["generated_media"]["generated_tone_fixtures"],
        manifest["generated_media"]["operator_declared_audio_fixtures"],
    )
    manifest_path = output / "fixture-contract.json"
    _write_json(manifest_path, manifest)
    return {"manifest": manifest, "manifest_path": str(manifest_path), "output_dir": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build non-sensitive offline VKP acceptance fixtures")
    parser.add_argument("output_dir", help="fresh explicit output root owned by this run")
    parser.add_argument("--audio-spec", default=None, help="optional operator-declared local speech media JSON")
    parser.add_argument("--card-font", default=None, help="optional explicitly licensed local font file")
    parser.add_argument(
        "--card-language",
        choices=CARD_LANGUAGES,
        default="ascii",
        help="card text language; 'zh' restores the original Chinese cards and requires --card-font",
    )
    parser.add_argument("--tone-seconds", type=float, default=DEFAULT_TONE_SECONDS, help="generated tone duration")
    parser.add_argument("--tone-frequency", type=float, default=DEFAULT_TONE_FREQUENCY_HZ, help="generated tone frequency")
    args = parser.parse_args(argv)
    try:
        result = run(
            args.output_dir,
            audio_spec_path=args.audio_spec,
            card_font_path=args.card_font,
            card_language=args.card_language,
            tone_seconds=args.tone_seconds,
            tone_frequency_hz=args.tone_frequency,
        )
    except FixtureInputError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": result["manifest_path"],
                "fixture_count": result["manifest"]["fixture_count"],
                "speech_reference": result["manifest"]["speech_reference"],
                "generated_media": result["manifest"]["generated_media"],
                "asr_quality_evidence": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
