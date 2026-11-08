"""Pure synthetic unit tests for the public offline fixture tools.

These tests exercise the real validation, containment, source-selection, row,
hash, aggregate and manifest logic in the public scripts. They never import the
whole VKP package, never run FFmpeg/ffprobe/subprocess, never open a socket, and
use no real temporary files or directory fixtures: filesystem probes, resolution,
reads, writes, hashes, durations and Pillow/wave access are replaced with
in-memory fakes at the module seams.

Synthetic roots are native absolute paths so the same expectations hold on POSIX
and Windows without touching the real filesystem. Windows drive/UNC strings are
still exercised as real lexical-validator cases; this suite does not claim a
Linux/POSIX runtime run.

Stdlib ``unittest`` only; Pillow is already a declared dependency of the
candidate scripts (imported when those modules load). Run from the candidate
root:

    python -m unittest discover -s tests -p "test_offline_fixture_tools_synthetic.py" -v
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from unittest import mock


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = CANDIDATE_ROOT / "scripts"
EXAMPLES_DIR = CANDIDATE_ROOT / "examples"


def _load_candidate_module(module_name: str, file_name: str, base_dir: Path = SCRIPTS_DIR):
    # Deliberately not registered in sys.modules: scoped loading avoids
    # polluting the process for other same-process suites and is restored by
    # simply not persisting the module name.
    spec = importlib.util.spec_from_file_location(module_name, base_dir / file_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate script {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_candidate_module("vkp_offline_builder_synthetic_tests", "build-offline-acceptance-fixtures.py")
EXPORTER = _load_candidate_module("vkp_offline_exporter_synthetic_tests", "export-offline-frame-manifest.py")
EXAMPLE = _load_candidate_module("vkp_offline_example_synthetic_tests", "offline_fixture_roundtrip.py", EXAMPLES_DIR)


# Native absolute synthetic roots: these are logical fixtures, never created on
# disk. On Windows the anchor is drive-absolute (C:), which keeps the separate
# "D:"/UNC/drive-relative lexical cases below meaningful as foreign/drive cases.
_NATIVE_ANCHOR = Path("C:/") if os.name == "nt" else Path("/")
FAKE_ROOT = _NATIVE_ANCHOR / "synthetic-bundle"
FAKE_OUTPUT = _NATIVE_ANCHOR / "synthetic-fixture-output"
OUTSIDE_ROOT = _NATIVE_ANCHOR / "outside-synthetic"
OPERATOR_ROOT = _NATIVE_ANCHOR / "operator-synthetic"
DECLARED_ROOT = _NATIVE_ANCHOR / "declared"
WORK_ROOT = _NATIVE_ANCHOR / "synthetic-out"
COMMAND_ROOT = _NATIVE_ANCHOR / "x"
OTHER_ROOT = _NATIVE_ANCHOR / "other"
MEDIA_ROOT = _NATIVE_ANCHOR / "synthetic"
_ABSENT = object()


def _declared(name: str) -> str:
    return str(DECLARED_ROOT / name)


def _work(*parts: str) -> str:
    return str(WORK_ROOT.joinpath(*parts))


def _command(*parts: str) -> str:
    return str(COMMAND_ROOT.joinpath(*parts))


def _key(path: Path) -> str:
    return str(path)


def _lexical_resolve(root: Path, reference: Path) -> Path:
    return reference if reference.is_absolute() else root / reference


def _frame_key(relative: str) -> str:
    return _key(FAKE_ROOT / relative)


class _FakeBinaryHandle:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._offset :]
        else:
            chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> "_FakeBinaryHandle":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class ExporterFakeIO:
    """In-memory bundle filesystem used to exercise the real exporter logic."""

    def __init__(
        self,
        *,
        timeline,
        bundle_manifest=None,
        manifest_present: bool = True,
        regular=(),
        existing=(),
        hashes=None,
        resolver=None,
        symlinks=(),
        destination_resolutions=None,
        existing_frame_manifests=None,
        nonregular_inputs=(),
        extra_reads=None,
    ) -> None:
        self.timeline = timeline
        self.bundle_manifest = {} if bundle_manifest is None else bundle_manifest
        self.manifest_present = manifest_present
        self.regular = set(regular)
        self.existing = set(existing)
        self.hashes = dict(hashes or {})
        self.resolver = resolver or _lexical_resolve
        self.symlinks = set(symlinks)
        self.destination_resolutions = dict(destination_resolutions or {})
        self.existing_frame_manifests = dict(existing_frame_manifests or {})
        self.extra_reads = dict(extra_reads or {})
        self.is_regular_calls: list[str] = []
        self.path_exists_calls: list[str] = []
        self.sha_calls: list[str] = []
        self.resolve_calls: list[tuple[str, str]] = []
        self.resolve_destination_calls: list[str] = []
        self.read_calls: list[str] = []
        self.writes: list[tuple[str, object]] = []
        self.events: list[tuple[str, str]] = []
        # Real bundle members are regular files. Tests that need a malformed
        # directory/FIFO-like topology declare it explicitly instead of making
        # every probe return True.
        self.existing.add(_key(FAKE_ROOT / "timeline.json"))
        self.regular.add(_key(FAKE_ROOT / "timeline.json"))
        if manifest_present:
            self.existing.add(_key(FAKE_ROOT / "manifest.json"))
            self.regular.add(_key(FAKE_ROOT / "manifest.json"))
        self.regular -= set(nonregular_inputs)

    def bundle_dir(self, _text: str) -> Path:
        return FAKE_ROOT

    def operator_path(self, text: str) -> Path:
        return Path(text)

    def is_regular(self, path: Path) -> bool:
        self.is_regular_calls.append(_key(path))
        return _key(path) in self.regular

    def path_exists(self, path: Path) -> bool:
        self.path_exists_calls.append(_key(path))
        return _key(path) in self.existing

    def is_symlink(self, path: Path) -> bool:
        return _key(path) in self.symlinks

    def resolve_destination(self, _root: Path, destination: Path) -> Path:
        self.resolve_destination_calls.append(_key(destination))
        self.events.append(("resolve_destination", _key(destination)))
        return self.destination_resolutions.get(_key(destination), destination)

    def sha(self, path: Path) -> str:
        self.sha_calls.append(_key(path))
        if _key(path) not in self.hashes:
            raise AssertionError(f"unexpected hash read for synthetic path {path}")
        return self.hashes[_key(path)]

    def read_json(self, path: Path):
        key = _key(path)
        self.read_calls.append(key)
        if key in self.extra_reads:
            return self.extra_reads[key]
        if key == _key(FAKE_ROOT / "timeline.json"):
            return self.timeline
        if key == _key(FAKE_ROOT / "manifest.json"):
            return self.bundle_manifest
        if key == _key(FAKE_ROOT / "frame-manifest.json"):
            content = self.existing_frame_manifests.get(_key(path), _ABSENT)
            if content is _ABSENT:
                raise AssertionError(f"unexpected frame-manifest read for synthetic path {path}")
            return content
        raise AssertionError(f"unexpected read for synthetic path {path}")

    def write_json(self, path: Path, data) -> None:
        self.events.append(("write", _key(path)))
        self.writes.append((_key(path), data))

    def resolve(self, root: Path, reference: Path) -> Path:
        self.resolve_calls.append((_key(root), _key(reference)))
        return self.resolver(root, reference)


def _patch_exporter(fake: ExporterFakeIO):
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(EXPORTER, "_resolve_bundle_dir", fake.bundle_dir))
    stack.enter_context(mock.patch.object(EXPORTER, "_resolve_operator_path", fake.operator_path))
    stack.enter_context(mock.patch.object(EXPORTER, "_path_exists", fake.path_exists))
    stack.enter_context(mock.patch.object(EXPORTER, "_is_directory", lambda path: True))
    stack.enter_context(mock.patch.object(EXPORTER, "_is_regular_file", fake.is_regular))
    stack.enter_context(mock.patch.object(EXPORTER, "_is_symlink", fake.is_symlink))
    stack.enter_context(mock.patch.object(EXPORTER, "_resolve_destination", fake.resolve_destination))
    stack.enter_context(mock.patch.object(EXPORTER, "_read", fake.read_json))
    stack.enter_context(mock.patch.object(EXPORTER, "_write_json", fake.write_json))
    stack.enter_context(mock.patch.object(EXPORTER, "_sha256", fake.sha))
    stack.enter_context(mock.patch.object(EXPORTER, "_resolve", fake.resolve))
    return stack


def _valid_audio_entry(fixture_id: str, path_text: str, digest: str) -> dict:
    return {
        "fixture_id": fixture_id,
        "path": path_text,
        "expected_sha256": digest,
        "operator_provenance": "synthetic operator declaration for tests",
        "use_rights_statement": "synthetic declaration; no real media exists",
    }


def _tone_audio() -> dict:
    return {
        "kind": "first_party_generated_pcm_tone",
        "path": FAKE_OUTPUT / "generated-audio/tone.wav",
        "sha256": "a" * 64,
        "duration_seconds": 2.0,
        "provenance": "first-party generated PCM tone (stdlib wave/math); not a speech recording",
        "use_rights_statement": "first-party generated locally; no third-party rights claimed",
        "source_project": "first-party generated PCM tone",
    }


def _operator_audio() -> dict:
    return {
        "kind": "operator_supplied_speech_candidate",
        "path": DECLARED_ROOT / "speech.wav",
        "sha256": "b" * 64,
        "duration_seconds": 2.0,
        "provenance": "operator declaration in tests",
        "use_rights_statement": "operator declaration in tests",
        "source_project": "operator-declared local speech media",
    }


class LoaderScopeTests(unittest.TestCase):
    def test_candidate_loader_does_not_persist_module_names(self) -> None:
        self.assertNotIn("vkp_offline_builder_synthetic_tests", sys.modules)
        self.assertNotIn("vkp_offline_exporter_synthetic_tests", sys.modules)
        self.assertNotIn("vkp_offline_example_synthetic_tests", sys.modules)

    def test_scoped_patches_restore_patched_globals(self) -> None:
        original = BUILDER.ImageFont.truetype
        with mock.patch.object(BUILDER.ImageFont, "truetype", side_effect=AssertionError("patched")):
            self.assertIsNot(BUILDER.ImageFont.truetype, original)
        self.assertIs(BUILDER.ImageFont.truetype, original)


class BuilderReuseTests(unittest.TestCase):
    def test_original_helper_functions_are_retained(self) -> None:
        for name in ("_render_card", "_mux_cards", "_duration", "_sha256"):
            self.assertTrue(callable(getattr(BUILDER, name)), name)

    def test_default_plan_keeps_original_ids_types_and_generated_tone(self) -> None:
        plan = BUILDER._fixture_plan()
        self.assertEqual(
            [entry["fixture_id"] for entry in plan],
            ["talking-head-synthetic", "slides-screen-synthetic", "mixed-visual-synthetic"],
        )
        self.assertEqual(
            [entry["fixture_type"] for entry in plan],
            ["talking_video", "slides_or_screen_video", "mixed_visual_video"],
        )
        for entry in plan:
            audio = entry["audio"]
            self.assertEqual(audio["kind"], "first_party_generated_pcm_tone")
            self.assertFalse(audio["declared"])
            self.assertIsNone(audio["path"])
            self.assertIn("not a speech", audio["provenance"])
            for title, bullets in entry["cards"]:
                self.assertTrue(title.isascii())
                for bullet in bullets:
                    self.assertTrue(bullet.isascii())

    def test_optional_chinese_cards_are_retained_with_supplied_font_only(self) -> None:
        plan = BUILDER._fixture_plan(card_language="zh")
        self.assertEqual(plan[0]["cards"][0][0], "本地讲话视频 fixture")
        self.assertFalse(plan[0]["cards"][0][0].isascii())
        self.assertEqual(plan[1]["cards"][0][0], "课程页一：输入")
        with self.assertRaises(BUILDER.FixtureInputError):
            BUILDER._fixture_plan(card_language="xx")

    def test_default_font_reads_no_host_font_and_declared_font_is_used(self) -> None:
        sentinel = object()
        with mock.patch.object(BUILDER.ImageFont, "load_default", return_value=sentinel) as load_default:
            with mock.patch.object(BUILDER.ImageFont, "truetype", side_effect=AssertionError("host font read")):
                self.assertIs(BUILDER._font(None, 24), sentinel)
        load_default.assert_called_once_with(size=24)
        with mock.patch.object(BUILDER.ImageFont, "truetype", return_value=sentinel) as truetype:
            self.assertIs(BUILDER._font(Path("declared-font.ttf"), 24), sentinel)
        truetype.assert_called_once_with("declared-font.ttf", size=24)

    def test_manifest_entry_is_honest_about_tone_and_targets(self) -> None:
        entry = BUILDER._manifest_entry(
            BUILDER.FIXTURES[0],
            video=FAKE_OUTPUT / "talking-head-synthetic/talking-head-synthetic.mp4",
            video_sha256="c" * 64,
            video_duration=2.0,
            audio=_tone_audio(),
        )
        self.assertEqual(entry["speech_reference"], "none")
        self.assertEqual(entry["asr_reference"], "none")
        self.assertFalse(entry["quality_reference_required"])
        self.assertEqual(entry["expected_artifacts"], BUILDER.EXPECTED_ARTIFACTS)
        self.assertEqual(entry["expected_artifacts_status"], "targets_not_proven_outputs")
        self.assertFalse(entry["source_audio_independently_verified"])
        self.assertEqual(entry["source_audio_sha256"], "a" * 64)
        for claim in entry["claims"].values():
            self.assertFalse(claim)
        self.assertIn("synthetic_cards", entry["media_scope"])
        self.assertIn("sha256_of_declared_bytes_only", entry["hash_scope"])

    def test_operator_speech_manifest_entry_keeps_capability_without_overclaim(self) -> None:
        entry = BUILDER._manifest_entry(
            BUILDER.FIXTURES[0],
            video=FAKE_OUTPUT / "talking-head-synthetic/talking-head-synthetic.mp4",
            video_sha256="c" * 64,
            video_duration=2.0,
            audio=_operator_audio(),
        )
        self.assertEqual(entry["source_audio_kind"], "operator_supplied_speech_candidate")
        self.assertIn("operator-declared", entry["speech_reference"])
        self.assertIn("not independently verified", entry["speech_reference"])
        self.assertEqual(entry["asr_reference"], "none")
        self.assertEqual(entry["source_audio_use_rights_statement"], "operator declaration in tests")
        self.assertFalse(entry["source_audio_independently_verified"])


class BuilderInputPreflightTests(unittest.TestCase):
    def test_duration_parser_rejects_nonfinite_zero_negative_text_and_rounded_zero(self) -> None:
        for raw in ("", " ", "nan", "inf", "-1", "0", "abc", "0.0004"):
            with self.subTest(raw=raw):
                with self.assertRaises(BUILDER.FixtureInputError):
                    BUILDER._parse_duration_text(raw)
        self.assertEqual(BUILDER._parse_duration_text(" 2.5 "), 2.5)
        self.assertEqual(BUILDER._parse_duration_text("0.001"), 0.001)

    def test_tone_bounds_reject_excessive_values_without_allocation(self) -> None:
        BUILDER._validate_tone_settings(2.0, 440.0)
        for seconds, frequency in (
            (BUILDER.TONE_SECONDS_MIN - 0.01, 440.0),
            (0, 440.0),
            (-1, 440.0),
            (BUILDER.TONE_SECONDS_MAX + 0.01, 440.0),
            (float("nan"), 440.0),
            (float("inf"), 440.0),
            (2.0, BUILDER.TONE_FREQUENCY_MIN_HZ - 1),
            (2.0, BUILDER.TONE_FREQUENCY_MAX_HZ + 1),
            (2.0, 0),
        ):
            with self.subTest(seconds=seconds, frequency=frequency):
                with self.assertRaises(BUILDER.FixtureInputError):
                    BUILDER._validate_tone_settings(seconds, frequency)
        with mock.patch.object(BUILDER.wave, "open", side_effect=AssertionError("must not allocate")):
            with self.assertRaises(BUILDER.FixtureInputError):
                BUILDER._write_tone(Path(_work("tone.wav")), 0.01, 440.0)
            with self.assertRaises(BUILDER.FixtureInputError):
                BUILDER._write_tone(Path(_work("tone.wav")), 2.0, 99999.0)

    def test_invalid_tone_settings_rejected_before_output_resolution(self) -> None:
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(BUILDER, "_resolve_output_path", side_effect=AssertionError("must not resolve output"))
            )
            with self.assertRaises(BUILDER.FixtureInputError):
                BUILDER.run(str(FAKE_OUTPUT), tone_seconds=0.01)
            with self.assertRaises(BUILDER.FixtureInputError):
                BUILDER.run(str(FAKE_OUTPUT), tone_frequency_hz=5000.0)

    def test_output_root_refuses_nonempty_and_plain_files(self) -> None:
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(BUILDER, "_path_exists", lambda path: True))
            stack.enter_context(mock.patch.object(BUILDER, "_is_directory", lambda path: True))
            stack.enter_context(mock.patch.object(BUILDER, "_directory_has_entries", lambda path: True))
            with self.assertRaises(BUILDER.FixtureInputError):
                BUILDER._preflight_output_root(FAKE_OUTPUT)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(BUILDER, "_path_exists", lambda path: True))
            stack.enter_context(mock.patch.object(BUILDER, "_is_directory", lambda path: False))
            with self.assertRaises(BUILDER.FixtureInputError):
                BUILDER._preflight_output_root(FAKE_OUTPUT)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(BUILDER, "_path_exists", lambda path: True))
            stack.enter_context(mock.patch.object(BUILDER, "_is_directory", lambda path: True))
            stack.enter_context(mock.patch.object(BUILDER, "_directory_has_entries", lambda path: False))
            BUILDER._preflight_output_root(FAKE_OUTPUT)

    def test_invalid_second_audio_spec_rejected_before_any_write_or_media_process(self) -> None:
        spec = {
            "audio": [
                _valid_audio_entry("talking-head-synthetic", _declared("one.wav"), "a" * 64),
                _valid_audio_entry("slides-screen-synthetic", _declared("two.wav"), "b" * 64),
            ]
        }
        first = Path(_declared("one.wav"))
        second = Path(_declared("two.wav"))
        hashes = {_key(first): "a" * 64, _key(second): "c" * 64}
        duration_calls: list[str] = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(BUILDER, "_resolve_output_path", lambda text: FAKE_OUTPUT))
            stack.enter_context(mock.patch.object(BUILDER, "_preflight_output_root", lambda output: None))
            stack.enter_context(mock.patch.object(BUILDER, "_read_json_object", lambda path: spec))
            stack.enter_context(mock.patch.object(BUILDER, "_resolve_operator_path", lambda text: Path(text)))
            stack.enter_context(mock.patch.object(BUILDER, "_is_regular_file", lambda path: True))
            stack.enter_context(mock.patch.object(BUILDER, "_sha256", lambda path: hashes[_key(path)]))
            stack.enter_context(
                mock.patch.object(
                    BUILDER,
                    "_duration",
                    lambda ffprobe, path: duration_calls.append(_key(path)) or 2.0,
                )
            )
            for never in ("_write_tone", "_render_card", "_mux_cards", "_write_text_lf", "_write_json", "_ensure_parent_dir", "_prepare_output_dir"):
                stack.enter_context(
                    mock.patch.object(BUILDER, never, side_effect=AssertionError(f"{never} must not run"))
                )
            with self.assertRaises(BUILDER.FixtureInputError) as caught:
                BUILDER.run(
                    str(FAKE_OUTPUT),
                    audio_spec_path=_declared("audio-spec.json"),
                    ffmpeg="ffmpeg",
                    ffprobe="ffprobe",
                )
        self.assertIn("SHA-256 mismatch", str(caught.exception))
        self.assertEqual(duration_calls, [])

    def test_audio_spec_shape_errors_are_rejected(self) -> None:
        cases = {
            "missing audio list": {},
            "empty audio list": {"audio": []},
            "audio not a list": {"audio": {}},
            "unknown fixture id": {
                "audio": [_valid_audio_entry("not-a-fixture", _declared("a.wav"), "a" * 64)]
            },
            "duplicate fixture id": {
                "audio": [
                    _valid_audio_entry("talking-head-synthetic", _declared("a.wav"), "a" * 64),
                    _valid_audio_entry("talking-head-synthetic", _declared("b.wav"), "b" * 64),
                ]
            },
            "missing path": {
                "audio": [
                    {
                        "fixture_id": "talking-head-synthetic",
                        "expected_sha256": "a" * 64,
                        "operator_provenance": "p",
                        "use_rights_statement": "r",
                    }
                ]
            },
            "bad sha": {
                "audio": [_valid_audio_entry("talking-head-synthetic", _declared("a.wav"), "xyz")]
            },
            "missing provenance": {
                "audio": [
                    {
                        "fixture_id": "talking-head-synthetic",
                        "path": _declared("a.wav"),
                        "expected_sha256": "a" * 64,
                        "use_rights_statement": "r",
                    }
                ]
            },
            "missing rights": {
                "audio": [
                    {
                        "fixture_id": "talking-head-synthetic",
                        "path": _declared("a.wav"),
                        "expected_sha256": "a" * 64,
                        "operator_provenance": "p",
                    }
                ]
            },
            "entry not an object": {"audio": ["bad"]},
        }
        for label, spec in cases.items():
            with self.subTest(label=label):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(BUILDER, "_read_json_object", lambda path: spec))
                    stack.enter_context(mock.patch.object(BUILDER, "_resolve_operator_path", lambda text: Path(text)))
                    stack.enter_context(mock.patch.object(BUILDER, "_is_regular_file", lambda path: True))
                    stack.enter_context(mock.patch.object(BUILDER, "_sha256", lambda path: "a" * 64))
                    stack.enter_context(mock.patch.object(BUILDER, "_duration", lambda ffprobe, path: 2.0))
                    with self.assertRaises(BUILDER.FixtureInputError):
                        BUILDER._preflight_operator_audio(_declared("spec.json"), "ffprobe")

    def test_valid_operator_audio_is_validated_hashed_and_retained(self) -> None:
        entries = [
            _valid_audio_entry("talking-head-synthetic", _declared("one.wav"), "a" * 64),
            _valid_audio_entry("slides-screen-synthetic", _declared("two.wav"), "b" * 64),
            _valid_audio_entry("mixed-visual-synthetic", _declared("three.wav"), "c" * 64),
        ]
        spec = {"audio": entries}
        hashes = {
            _key(Path(_declared("one.wav"))): "a" * 64,
            _key(Path(_declared("two.wav"))): "b" * 64,
            _key(Path(_declared("three.wav"))): "c" * 64,
        }
        duration_calls: list[str] = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(BUILDER, "_read_json_object", lambda path: spec))
            stack.enter_context(mock.patch.object(BUILDER, "_resolve_operator_path", lambda text: Path(text)))
            stack.enter_context(mock.patch.object(BUILDER, "_is_regular_file", lambda path: True))
            stack.enter_context(mock.patch.object(BUILDER, "_sha256", lambda path: hashes[_key(path)]))
            stack.enter_context(
                mock.patch.object(
                    BUILDER, "_duration", lambda ffprobe, path: duration_calls.append(_key(path)) or 2.0
                )
            )
            declared = BUILDER._preflight_operator_audio(_declared("spec.json"), "ffprobe")
        self.assertEqual(set(declared), {entry["fixture_id"] for entry in entries})
        for audio in declared.values():
            self.assertEqual(audio["kind"], "operator_supplied_speech_candidate")
            self.assertEqual(audio["duration_seconds"], 2.0)
        self.assertEqual(len(duration_calls), 3)


class BuilderInMemoryRunTests(unittest.TestCase):
    """Exercises ``run`` end-to-end with in-memory seams (no real media/paths)."""

    def _run_builder_in_memory(
        self,
        *,
        spec_entries=None,
        card_language: str = "ascii",
        card_font: str | None = None,
        tone_seconds: float = 2.0,
        tone_frequency_hz: float = 440.0,
    ):
        operator_hashes: dict[str, str] = {}
        operator_durations: dict[str, float] = {}
        for entry in spec_entries or []:
            operator_hashes[_key(Path(entry["path"]))] = entry["expected_sha256"]
            operator_durations[_key(Path(entry["path"]))] = entry.get("_duration_seconds", 2.0)

        state = {
            "writes": [],
            "tone_calls": [],
            "render_calls": [],
            "mux_calls": [],
            "prepared": [],
        }

        def fake_sha(path: Path) -> str:
            key = _key(path)
            if key in operator_hashes:
                return operator_hashes[key]
            return hashlib.sha256(key.encode("utf-8")).hexdigest()

        def fake_duration(_ffprobe: str, path: Path) -> float:
            key = _key(path)
            if key in operator_durations:
                return operator_durations[key]
            if key.endswith(".mp4"):
                return 2.0
            if "-tone.wav" in key:
                return tone_seconds
            return 2.0

        def fake_write_tone(path: Path, seconds: float, frequency_hz: float) -> None:
            state["tone_calls"].append((path, seconds, frequency_hz))

        def fake_render_card(path, title, bullets, *, accent, font_path=None) -> None:
            state["render_calls"].append(
                {"path": path, "title": title, "bullets": list(bullets), "font_path": font_path}
            )

        def fake_mux_cards(ffmpeg, ffprobe, images, audio, output) -> None:
            state["mux_calls"].append({"images": list(images), "audio": audio, "output": output})

        def fake_write_json(path: Path, data) -> None:
            state["writes"].append((path, data))

        spec_payload = {"audio": spec_entries} if spec_entries is not None else None
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(BUILDER, "_resolve_output_path", lambda text: FAKE_OUTPUT))
            stack.enter_context(mock.patch.object(BUILDER, "_preflight_output_root", lambda output: None))
            stack.enter_context(mock.patch.object(BUILDER, "_prepare_output_dir", lambda path: state["prepared"].append(path)))
            stack.enter_context(mock.patch.object(BUILDER, "_ensure_parent_dir", lambda path: None))
            stack.enter_context(mock.patch.object(BUILDER, "_read_json_object", lambda path: spec_payload))
            stack.enter_context(mock.patch.object(BUILDER, "_resolve_operator_path", lambda text: Path(text)))
            stack.enter_context(mock.patch.object(BUILDER, "_is_regular_file", lambda path: True))
            stack.enter_context(mock.patch.object(BUILDER, "_sha256", fake_sha))
            stack.enter_context(mock.patch.object(BUILDER, "_duration", fake_duration))
            stack.enter_context(mock.patch.object(BUILDER, "_write_tone", fake_write_tone))
            stack.enter_context(mock.patch.object(BUILDER, "_render_card", fake_render_card))
            stack.enter_context(mock.patch.object(BUILDER, "_mux_cards", fake_mux_cards))
            stack.enter_context(mock.patch.object(BUILDER, "_write_json", fake_write_json))
            result = BUILDER.run(
                str(FAKE_OUTPUT),
                audio_spec_path=_declared("spec.json") if spec_entries is not None else None,
                card_font_path=card_font,
                card_language=card_language,
                ffmpeg="ffmpeg",
                ffprobe="ffprobe",
                tone_seconds=tone_seconds,
                tone_frequency_hz=tone_frequency_hz,
            )
        return result, state

    def test_mixed_tone_and_operator_aggregate_is_derived_from_actual_inputs(self) -> None:
        operator_entry = _valid_audio_entry("slides-screen-synthetic", _declared("slides.wav"), "a" * 64)
        operator_entry["_duration_seconds"] = 3.5
        result, state = self._run_builder_in_memory(spec_entries=[operator_entry])
        manifest = result["manifest"]
        aggregate = manifest["generated_media"]
        self.assertEqual(aggregate["generated_tone_fixtures"], 2)
        self.assertEqual(aggregate["operator_declared_audio_fixtures"], 1)
        self.assertEqual(
            aggregate["audio_kinds"],
            ["first_party_generated_pcm_tone", "operator_supplied_speech_candidate"],
        )
        self.assertEqual(aggregate["audio_durations_seconds"], [2.0, 3.5, 2.0])
        self.assertEqual(aggregate["tone_seconds_requested"], 2.0)
        self.assertEqual(aggregate["tone_frequency_hz"], 440.0)
        self.assertEqual(aggregate["card_language"], "ascii")
        self.assertIn("default_font", aggregate["cards"])
        self.assertTrue(aggregate["operator_supplied_local_inputs"])
        self.assertFalse(aggregate["external_asset_lookup"])
        self.assertFalse(aggregate["system_font_search"])
        self.assertIn("none for 2 generated-tone fixtures", manifest["speech_reference"])
        self.assertIn("not independently verified", manifest["speech_reference"])
        self.assertEqual(len(state["tone_calls"]), 2)
        self.assertEqual(len(state["mux_calls"]), 3)
        self.assertEqual(len(state["writes"]), 1)

    def test_all_operator_aggregate_does_not_claim_tone(self) -> None:
        entries = [
            _valid_audio_entry("talking-head-synthetic", _declared("one.wav"), "a" * 64),
            _valid_audio_entry("slides-screen-synthetic", _declared("two.wav"), "b" * 64),
            _valid_audio_entry("mixed-visual-synthetic", _declared("three.wav"), "c" * 64),
        ]
        for entry, duration in zip(entries, (1.5, 2.5, 3.0)):
            entry["_duration_seconds"] = duration
        result, state = self._run_builder_in_memory(spec_entries=entries)
        aggregate = result["manifest"]["generated_media"]
        self.assertEqual(aggregate["generated_tone_fixtures"], 0)
        self.assertEqual(aggregate["operator_declared_audio_fixtures"], 3)
        self.assertEqual(aggregate["audio_kinds"], ["operator_supplied_speech_candidate"])
        self.assertEqual(aggregate["audio_durations_seconds"], [1.5, 2.5, 3.0])
        self.assertIsNone(aggregate["tone_seconds_requested"])
        self.assertIsNone(aggregate["tone_frequency_hz"])
        self.assertIsNone(aggregate["tone_sample_rate"])
        self.assertTrue(result["manifest"]["speech_reference"].startswith("operator-declared speech media for all fixtures"))
        self.assertEqual(state["tone_calls"], [])
        self.assertEqual(len(state["mux_calls"]), 3)

    def test_chinese_cards_require_supplied_font_and_aggregate_reports_it(self) -> None:
        with self.assertRaises(BUILDER.FixtureInputError):
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(BUILDER, "_resolve_output_path", lambda text: FAKE_OUTPUT))
                stack.enter_context(mock.patch.object(BUILDER, "_preflight_output_root", lambda output: None))
                stack.enter_context(mock.patch.object(BUILDER, "_prepare_output_dir", lambda path: None))
                stack.enter_context(mock.patch.object(BUILDER, "_is_regular_file", lambda path: True))
                stack.enter_context(mock.patch.object(BUILDER, "shutil", mock.MagicMock(which=lambda name: name)))
                BUILDER.run(str(FAKE_OUTPUT), card_language="zh")
        result, state = self._run_builder_in_memory(
            card_language="zh", card_font=_declared("font.ttf")
        )
        aggregate = result["manifest"]["generated_media"]
        self.assertEqual(aggregate["card_language"], "zh")
        self.assertIn("chinese", aggregate["cards"])
        self.assertIn("operator_declared_local_font", aggregate["cards"])
        self.assertTrue(aggregate["operator_supplied_local_inputs"])
        self.assertEqual(state["render_calls"][0]["title"], "本地讲话视频 fixture")
        self.assertEqual(state["render_calls"][0]["font_path"], Path(_declared("font.ttf")))


class BuilderMediaCommandTests(unittest.TestCase):
    def test_probe_command_is_ffprobe_supported_and_local_file_only(self) -> None:
        command = BUILDER._probe_command("ffprobe", Path(_work("tone.wav")))
        self.assertNotIn("-nostdin", command)
        self.assertEqual(command[command.index("-protocol_whitelist") + 1], "file")
        self.assertEqual(command[-1], str(Path(_work("tone.wav"))))
        captured = {}
        with mock.patch.object(
            BUILDER,
            "_run_media",
            lambda cmd, timeout: captured.update(command=cmd, timeout=timeout) or SimpleNamespace(stdout="2.0"),
        ):
            self.assertEqual(BUILDER._duration("ffprobe", Path(_work("tone.wav"))), 2.0)
        self.assertEqual(captured["command"], command)
        self.assertEqual(captured["timeout"], BUILDER.FFPROBE_TIMEOUT_SECONDS)

    def test_run_media_closes_stdin_for_local_tools(self) -> None:
        captured = {}
        with mock.patch.object(
            BUILDER.subprocess,
            "run",
            lambda command, **kwargs: captured.update(command=command, kwargs=kwargs) or SimpleNamespace(stdout=""),
        ):
            BUILDER._run_media(["ffmpeg", "-version"], 5)
        self.assertIs(captured["kwargs"]["stdin"], BUILDER.subprocess.DEVNULL)
        self.assertTrue(captured["kwargs"]["check"])
        self.assertEqual(captured["kwargs"]["timeout"], 5)

    def test_concat_uses_relative_safe_basenames_and_rejects_unsafe_names(self) -> None:
        images = [
            FAKE_OUTPUT / "talking-head-synthetic/card-01.png",
            FAKE_OUTPUT / "talking-head-synthetic/card-02.png",
        ]
        text = BUILDER._concat_text(images, 1.25)
        self.assertIn("ffconcat version 1.0", text)
        self.assertIn("file 'card-01.png'", text)
        self.assertIn("file 'card-02.png'", text)
        self.assertNotIn(str(FAKE_OUTPUT), text)
        self.assertTrue(text.endswith("file 'card-02.png'\n"))
        for bad_name in ("bad name.png", "bad'name.png"):
            with self.subTest(bad_name=bad_name):
                with self.assertRaises(BUILDER.FixtureInputError):
                    BUILDER._concat_text([MEDIA_ROOT / bad_name], 1.0)
        with self.assertRaises(BUILDER.FixtureInputError):
            BUILDER._concat_text([], 1.0)
        with self.assertRaises(BUILDER.FixtureInputError):
            BUILDER._concat_text(images, 0)

    def test_mux_command_bounds_decoder_encoder_and_filter_threads(self) -> None:
        command = BUILDER._mux_command(
            "ffmpeg", Path(_command("cards.ffconcat")), Path(_command("tone.wav")), Path(_command("out.mp4"))
        )
        self.assertIsInstance(command, list)
        self.assertTrue(all(isinstance(part, str) for part in command))
        self.assertEqual(command[0], "ffmpeg")
        self.assertEqual(command[command.index("-safe") + 1], "1")
        self.assertEqual(command.count("-protocol_whitelist"), 2)
        thread_values = [command[index + 1] for index, part in enumerate(command) if part == "-threads"]
        self.assertGreaterEqual(len(thread_values), 3)
        self.assertTrue(all(value == str(BUILDER.MEDIA_THREADS) for value in thread_values))
        self.assertEqual(command[command.index("-filter_threads") + 1], str(BUILDER.MEDIA_THREADS))
        self.assertEqual(command[command.index("-filter_complex_threads") + 1], str(BUILDER.MEDIA_THREADS))
        self.assertNotIn("-filter_complex", command)
        self.assertEqual(command[-1], str(Path(_command("out.mp4"))))

    def test_mux_cards_writes_relative_concat_and_bounds_media_call(self) -> None:
        images = [Path(_work("fixture/card-01.png")), Path(_work("fixture/card-02.png"))]
        audio = Path(_work("generated-audio/tone.wav"))
        output = Path(_work("fixture/talking-head-synthetic.mp4"))
        written: dict = {}
        media_calls: list[tuple] = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(BUILDER, "_duration", lambda ffprobe, path: 2.0))
            stack.enter_context(
                mock.patch.object(BUILDER, "_write_text_lf", lambda path, text: written.update(path=path, text=text))
            )
            stack.enter_context(
                mock.patch.object(
                    BUILDER,
                    "_run_media",
                    lambda command, timeout: media_calls.append((command, timeout))
                    or SimpleNamespace(stdout="2.0"),
                )
            )
            stack.enter_context(mock.patch.object(BUILDER, "_is_regular_file", lambda path: True))
            stack.enter_context(mock.patch.object(BUILDER, "_file_size", lambda path: 1234))
            BUILDER._mux_cards("ffmpeg", "ffprobe", images, audio, output)
        self.assertEqual(written["path"], output.parent / "cards.ffconcat")
        self.assertEqual(written["text"], BUILDER._concat_text(images, 1.0))
        self.assertEqual(len(media_calls), 1)
        command, timeout = media_calls[0]
        self.assertEqual(timeout, BUILDER.FFMPEG_TIMEOUT_SECONDS)
        self.assertEqual(command, BUILDER._mux_command("ffmpeg", output.parent / "cards.ffconcat", audio, output))

    def test_mux_cards_rejects_cross_directory_cards_before_media_process(self) -> None:
        with mock.patch.object(BUILDER, "_duration", side_effect=AssertionError("must not probe")):
            with self.assertRaises(BUILDER.FixtureInputError):
                BUILDER._mux_cards(
                    "ffmpeg",
                    "ffprobe",
                    [OTHER_ROOT / "card-01.png"],
                    Path(_work("tone.wav")),
                    Path(_work("video.mp4")),
                )

    def test_write_tone_uses_stdlib_wave_and_rejects_bad_durations(self) -> None:
        class FakeWaveWriter:
            def __init__(self, path, mode) -> None:
                self.path = path
                self.mode = mode
                self.params: dict = {}
                self.frames = b""

            def setnchannels(self, value) -> None:
                self.params["channels"] = value

            def setsampwidth(self, value) -> None:
                self.params["width"] = value

            def setframerate(self, value) -> None:
                self.params["rate"] = value

            def writeframes(self, data) -> None:
                self.frames = data

            def __enter__(self):
                return self

            def __exit__(self, *args) -> bool:
                return False

        writers: list[FakeWaveWriter] = []

        def fake_open(path, mode):
            writer = FakeWaveWriter(path, mode)
            writers.append(writer)
            return writer

        # The source's documented minimum is TONE_SECONDS_MIN; the frozen test
        # used 0.1 s, below that bound. Exercise the smallest valid duration.
        seconds = BUILDER.TONE_SECONDS_MIN
        with mock.patch.object(BUILDER.wave, "open", fake_open):
            with mock.patch.object(BUILDER, "_ensure_parent_dir", lambda path: None):
                BUILDER._write_tone(Path(_work("tone.wav")), seconds)
        self.assertEqual(len(writers), 1)
        writer = writers[0]
        self.assertEqual(writer.mode, "wb")
        self.assertEqual(writer.params, {"channels": 1, "width": 2, "rate": BUILDER.TONE_SAMPLE_RATE})
        expected_frames = int(round(BUILDER.TONE_SAMPLE_RATE * seconds))
        self.assertEqual(len(writer.frames), expected_frames * 2)
        self.assertNotEqual(writer.frames, b"\x00" * len(writer.frames))
        writers.clear()
        for bad_seconds, bad_frequency in ((0, 440.0), (-1, 440.0), (float("nan"), 440.0), (1.0, 0)):
            with self.subTest(seconds=bad_seconds, frequency=bad_frequency):
                with self.assertRaises(BUILDER.FixtureInputError):
                    BUILDER._write_tone(Path(_work("tone.wav")), bad_seconds, bad_frequency)
        self.assertEqual(writers, [])

    def test_sha256_hashes_only_declared_bytes(self) -> None:
        handle = _FakeBinaryHandle(b"abc")
        with mock.patch.object(BUILDER.Path, "open", return_value=handle):
            digest = BUILDER._sha256(MEDIA_ROOT / "fake.bin")
        self.assertEqual(digest, hashlib.sha256(b"abc").hexdigest())


class ExporterValidationTests(unittest.TestCase):
    def test_malformed_json_fails_before_any_write(self) -> None:
        class MalformedFake(ExporterFakeIO):
            def read_json(self, path):
                self.read_calls.append(_key(path))
                raise json.JSONDecodeError("Expecting value", "timeline", 0)

        fake = MalformedFake(timeline=[])
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.writes, [])

    def test_empty_timeline_and_reference_free_timeline_fail_without_writes(self) -> None:
        for timeline in ([], [{"index": 0, "start": 0, "end": 1}]):
            with self.subTest(timeline=timeline):
                fake = ExporterFakeIO(timeline=timeline)
                with _patch_exporter(fake):
                    with self.assertRaises(EXPORTER.FramePlanError):
                        EXPORTER.run_export(FAKE_ROOT)
                self.assertEqual(fake.writes, [])

    def test_directory_timeline_is_a_malformed_input_error(self) -> None:
        class DirectoryTimelineFake(ExporterFakeIO):
            def read_json(self, path):
                self.read_calls.append(_key(path))
                raise IsADirectoryError(errno.EISDIR, "is a directory", str(path))

        fake = DirectoryTimelineFake(timeline=[])
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.writes, [])

    def test_malformed_nested_shapes_fail_without_writes_or_reads(self) -> None:
        cases = {
            "timeline not a list": {"item": "not-a-list"},
            "item not an object": {"item": ["bad"]},
            "integrated_visual not an object": {"item": [{"index": 0, "integrated_visual": "bad"}]},
            "evidence not a list": {
                "item": [{"index": 0, "integrated_visual": {"evidence_frame_paths": "frames/a.png"}}]
            },
            "temporal not a list": {"item": [{"index": 0, "temporal_frame_paths": "frames/a.png"}]},
            "assets not a list": {"item": [{"index": 0, "assets": {"source": "frames/a.png"}}]},
            "asset not an object": {"item": [{"index": 0, "assets": ["bad"]}]},
            "reference not a string": {"item": [{"index": 0, "temporal_frame_paths": [123]}]},
            "manifest not an object": {"item": [], "manifest": []},
        }
        for label, case in cases.items():
            with self.subTest(label=label):
                fake = ExporterFakeIO(timeline=case["item"], bundle_manifest=case.get("manifest", {}))
                with _patch_exporter(fake):
                    with self.assertRaises(EXPORTER.FramePlanError):
                        EXPORTER.run_export(FAKE_ROOT)
                self.assertEqual(fake.writes, [])
                self.assertEqual(
                    set(fake.is_regular_calls)
                    - {_key(FAKE_ROOT / "timeline.json"), _key(FAKE_ROOT / "manifest.json")},
                    set(),
                )
                self.assertEqual(fake.sha_calls, [])

    def test_missing_directory_and_empty_references_are_visible_and_nonzero(self) -> None:
        timeline = [
            {
                "index": 0,
                "start": 0.0,
                "end": 2.0,
                "temporal_frame_paths": ["frames/missing.png", "", "frames"],
            }
        ]
        fake = ExporterFakeIO(
            timeline=timeline,
            regular=set(),
            existing={_frame_key("frames")},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["exit_code"], 1)
        self.assertEqual(summary["frame_count"], 3)
        self.assertEqual(summary["missing_frame_count"], 1)
        self.assertEqual(summary["blocked_frame_count"], 2)
        self.assertEqual(len(summary["incomplete_locations"]), 3)
        self.assertEqual(fake.sha_calls, [])
        self.assertEqual(len(fake.writes), 1)
        self.assertEqual(fake.writes[0][0], _key(FAKE_ROOT / "frame-manifest.json"))
        manifest = fake.writes[0][1]
        statuses = {row["reference"]: row["evidence_status"] for row in manifest["frames"]}
        self.assertEqual(statuses["frames/missing.png"], "missing")
        self.assertEqual(statuses[""], "blocked")
        self.assertEqual(statuses["frames"], "blocked")
        blocked_reasons = {row["reference"]: row["blocked_reason"] for row in manifest["frames"]}
        self.assertEqual(blocked_reasons[""], EXPORTER.BLOCKED_EMPTY)
        self.assertEqual(blocked_reasons["frames"], EXPORTER.BLOCKED_NOT_REGULAR)

    def test_outside_root_references_are_blocked_before_any_read(self) -> None:
        def resolver(root: Path, reference: Path) -> Path:
            if reference == Path("frames/escape.png"):
                return OUTSIDE_ROOT / "escape.png"
            return reference if reference.is_absolute() else root / reference

        timeline = [
            {
                "index": 0,
                "temporal_frame_paths": ["../outside.png", str(OUTSIDE_ROOT / "plain-outside.png"), "frames/escape.png"],
            }
        ]
        fake = ExporterFakeIO(timeline=timeline, resolver=resolver)
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        manifest = fake.writes[0][1]
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["blocked_frame_count"], 3)
        self.assertEqual(summary["available_frame_count"], 0)
        for row in manifest["frames"]:
            self.assertEqual(row["evidence_status"], "blocked")
            self.assertIn(
                row["blocked_reason"],
                {EXPORTER.BLOCKED_TRAVERSAL, EXPORTER.BLOCKED_OUTSIDE},
            )
        reasons = {row["reference"]: row["blocked_reason"] for row in manifest["frames"]}
        self.assertEqual(reasons["../outside.png"], EXPORTER.BLOCKED_TRAVERSAL)
        self.assertEqual(reasons[str(OUTSIDE_ROOT / "plain-outside.png")], EXPORTER.BLOCKED_OUTSIDE)
        self.assertEqual(reasons["frames/escape.png"], EXPORTER.BLOCKED_OUTSIDE)
        self.assertEqual(
            set(fake.is_regular_calls)
            - {_key(FAKE_ROOT / "timeline.json"), _key(FAKE_ROOT / "manifest.json")},
            set(),
        )
        self.assertEqual(fake.sha_calls, [])
        self.assertEqual(len(fake.writes), 1)

    def test_foreign_drive_and_unc_are_blocked_without_resolution(self) -> None:
        timeline = [
            {
                "index": 0,
                "temporal_frame_paths": [
                    "D:/other-synthetic/frame.png",
                    "\\\\server\\share\\frame.png",
                    "C:drive-relative-frame.png",
                ],
            }
        ]
        fake = ExporterFakeIO(timeline=timeline)
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        manifest = fake.writes[0][1]
        self.assertEqual(summary["blocked_frame_count"], 3)
        reasons = {row["reference"]: row["blocked_reason"] for row in manifest["frames"]}
        self.assertEqual(reasons["D:/other-synthetic/frame.png"], EXPORTER.BLOCKED_FOREIGN_DRIVE)
        self.assertEqual(reasons["\\\\server\\share\\frame.png"], EXPORTER.BLOCKED_UNC)
        # "C:drive-relative..." is drive-relative only when the bundle root itself
        # carries a drive (Windows). On POSIX there is no same-drive root, so the
        # same lexical form is a foreign-drive case. Both are refused before any
        # resolution or read; this is real validator behavior, not a skip/mock.
        self.assertEqual(
            reasons["C:drive-relative-frame.png"],
            EXPORTER.BLOCKED_DRIVE_RELATIVE if FAKE_ROOT.drive else EXPORTER.BLOCKED_FOREIGN_DRIVE,
        )
        self.assertEqual(
            {reference for _root, reference in fake.resolve_calls},
            {_key(FAKE_ROOT / "timeline.json"), _key(FAKE_ROOT / "manifest.json")},
        )
        self.assertEqual(
            set(fake.is_regular_calls)
            - {_key(FAKE_ROOT / "timeline.json"), _key(FAKE_ROOT / "manifest.json")},
            set(),
        )
        self.assertEqual(fake.sha_calls, [])

    def test_relative_references_resolve_against_bundle_root_not_cwd(self) -> None:
        self.assertEqual(
            EXPORTER._join_reference(FAKE_ROOT, Path("frames/a.png")),
            FAKE_ROOT / "frames/a.png",
        )
        absolute_inside = FAKE_ROOT / "frames/inside.png"
        self.assertEqual(EXPORTER._join_reference(FAKE_ROOT, absolute_inside), absolute_inside)
        timeline = [
            {
                "index": 0,
                "temporal_frame_paths": ["frames/inside.png", str(FAKE_ROOT / "frames/absolute-inside.png")],
            }
        ]
        fake = ExporterFakeIO(
            timeline=timeline,
            regular={_frame_key("frames/inside.png"), _key(FAKE_ROOT / "frames/absolute-inside.png")},
            hashes={
                _frame_key("frames/inside.png"): "a" * 64,
                _key(FAKE_ROOT / "frames/absolute-inside.png"): "b" * 64,
            },
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["frame_count"], 2)
        for root_arg, _reference_arg in fake.resolve_calls:
            self.assertEqual(root_arg, _key(FAKE_ROOT))

    def test_case_sensitive_paths_and_repeated_timeline_context(self) -> None:
        timeline = [
            {"index": 0, "start": 0.0, "end": 1.0, "temporal_frame_paths": ["frames/A.PNG"]},
            {"index": 1, "start": 1.0, "end": 2.0, "temporal_frame_paths": ["frames/a.png"]},
            {
                "index": 2,
                "start": 2.0,
                "end": 3.0,
                "assets": [{"source": "frames/A.PNG"}],
                "integrated_visual": {"evidence_frame_paths": ["frames/A.PNG"]},
            },
        ]
        upper_key = _frame_key("frames/A.PNG")
        lower_key = _frame_key("frames/a.png")
        fake = ExporterFakeIO(
            timeline=timeline,
            regular={upper_key, lower_key},
            hashes={upper_key: "a" * 64, lower_key: "b" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        manifest = fake.writes[0][1]
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["frame_count"], 3)
        self.assertEqual(summary["occurrence_count"], 4)
        rows = {(row["role"], row["reference"]): row for row in manifest["frames"]}
        self.assertIn(("temporal_frame", "frames/A.PNG"), rows)
        self.assertIn(("temporal_frame", "frames/a.png"), rows)
        keyframe = rows[("keyframe", "frames/A.PNG")]
        self.assertEqual(keyframe["occurrence_count"], 2)
        self.assertEqual([item["timeline_index"] for item in keyframe["occurrences"]], [2, 2])
        self.assertEqual(keyframe["timeline_index"], 2)
        self.assertEqual(rows[("temporal_frame", "frames/A.PNG")]["sha256"], "a" * 64)
        self.assertEqual(rows[("temporal_frame", "frames/a.png")]["sha256"], "b" * 64)
        self.assertNotEqual(
            rows[("temporal_frame", "frames/A.PNG")]["path"],
            rows[("temporal_frame", "frames/a.png")]["path"],
        )

    def test_multiple_timeline_occurrences_keep_primary_fields(self) -> None:
        timeline = [
            {"index": 7, "start": 1.5, "end": 2.5, "temporal_frame_paths": ["frames/shared.png"]},
            {"index": 8, "start": 3.5, "end": 4.5, "temporal_frame_paths": ["frames/shared.png"]},
        ]
        key = _frame_key("frames/shared.png")
        fake = ExporterFakeIO(timeline=timeline, regular={key}, hashes={key: "a" * 64})
        with _patch_exporter(fake):
            EXPORTER.run_export(FAKE_ROOT)
        manifest = fake.writes[0][1]
        self.assertEqual(manifest["frame_count"], 1)
        row = manifest["frames"][0]
        self.assertEqual(row["timeline_index"], 7)
        self.assertEqual(row["start_seconds"], 1.5)
        self.assertEqual(row["occurrence_count"], 2)
        self.assertEqual(manifest["occurrence_count"], 2)

    def test_asset_without_source_or_path_is_visible_but_not_a_frame(self) -> None:
        timeline = [
            {
                "index": 0,
                "assets": [{"kind": "audio"}],
                "temporal_frame_paths": ["frames/a.png"],
            }
        ]
        key = _frame_key("frames/a.png")
        fake = ExporterFakeIO(timeline=timeline, regular={key}, hashes={key: "a" * 64})
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        manifest = fake.writes[0][1]
        self.assertTrue(summary["ok"])
        self.assertEqual(manifest["input_validation"]["skipped_asset_entries"][0]["reason"], "asset_without_source_or_path")
        self.assertEqual(manifest["frame_count"], 1)


class ExporterPermissionTests(unittest.TestCase):
    def test_permission_error_reading_timeline_propagates_before_writes(self) -> None:
        class DeniedReadFake(ExporterFakeIO):
            def read_json(self, path):
                self.read_calls.append(_key(path))
                raise PermissionError(errno.EACCES, "access denied", str(path))

        fake = DeniedReadFake(timeline=[])
        with _patch_exporter(fake):
            with self.assertRaises(PermissionError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.writes, [])

    def test_permission_error_during_reference_resolution_stops_before_writes(self) -> None:
        def denied_resolver(_root: Path, reference: Path) -> Path:
            raise PermissionError(errno.EACCES, "access denied", _key(reference))

        timeline = [{"index": 0, "temporal_frame_paths": ["frames/a.png"]}]
        fake = ExporterFakeIO(timeline=timeline, resolver=denied_resolver)
        with _patch_exporter(fake):
            with self.assertRaises(PermissionError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.writes, [])
        self.assertEqual(
            set(fake.is_regular_calls)
            - {_key(FAKE_ROOT / "timeline.json"), _key(FAKE_ROOT / "manifest.json")},
            set(),
        )
        self.assertEqual(fake.sha_calls, [])

    def test_eperm_from_regular_file_probe_is_not_masked(self) -> None:
        frame = _frame_key("frames/a.png")

        class EpermFake(ExporterFakeIO):
            def is_regular(self, path: Path) -> bool:
                self.is_regular_calls.append(_key(path))
                if _key(path) == frame:
                    raise OSError(errno.EPERM, "operation not permitted", str(path))
                return _key(path) in self.regular

        timeline = [{"index": 0, "temporal_frame_paths": ["frames/a.png"]}]
        fake = EpermFake(timeline=timeline)
        with _patch_exporter(fake):
            with self.assertRaises(OSError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertIn(frame, fake.is_regular_calls)
        self.assertEqual(fake.writes, [])


class ExporterBundleInputTests(unittest.TestCase):
    """Pre-read resolve/contain/regular checks for the two fixed bundle JSON members."""

    def _resolver_with(self, declared: Path, resolved: Path):
        declared_key = _key(declared)

        def resolver(root: Path, reference: Path) -> Path:
            if _key(reference) == declared_key:
                return resolved
            return reference if reference.is_absolute() else root / reference

        return resolver

    def test_outside_root_timeline_link_fails_before_any_read(self) -> None:
        resolver = self._resolver_with(
            FAKE_ROOT / "timeline.json", OUTSIDE_ROOT / "timeline.json"
        )
        fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}], resolver=resolver
        )
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError) as caught:
                EXPORTER.run_export(FAKE_ROOT)
        self.assertIn("outside bundle root", str(caught.exception))
        self.assertEqual(fake.read_calls, [])
        self.assertEqual(fake.is_regular_calls, [])
        self.assertEqual(fake.writes, [])

    def test_outside_root_manifest_link_fails_before_manifest_read(self) -> None:
        resolver = self._resolver_with(
            FAKE_ROOT / "manifest.json", OUTSIDE_ROOT / "manifest.json"
        )
        fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}], resolver=resolver
        )
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError) as caught:
                EXPORTER.run_export(FAKE_ROOT)
        self.assertIn("outside bundle root", str(caught.exception))
        self.assertEqual(fake.read_calls, [_key(FAKE_ROOT / "timeline.json")])
        self.assertEqual(fake.writes, [])

    def test_nonregular_timeline_is_refused_before_open(self) -> None:
        fake = ExporterFakeIO(
            timeline=[],
            nonregular_inputs={_key(FAKE_ROOT / "timeline.json")},
        )
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError) as caught:
                EXPORTER.run_export(FAKE_ROOT)
        self.assertIn("not a regular file", str(caught.exception))
        self.assertEqual(fake.read_calls, [])
        self.assertEqual(fake.writes, [])

    def test_nonregular_manifest_is_refused_before_open(self) -> None:
        fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            nonregular_inputs={_key(FAKE_ROOT / "manifest.json")},
        )
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError) as caught:
                EXPORTER.run_export(FAKE_ROOT)
        self.assertIn("not a regular file", str(caught.exception))
        self.assertEqual(fake.read_calls, [_key(FAKE_ROOT / "timeline.json")])
        self.assertEqual(fake.writes, [])

    def test_absent_optional_manifest_remains_allowed(self) -> None:
        frame = _frame_key("frames/a.png")
        fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            manifest_present=False,
            regular={frame},
            hashes={frame: "a" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        self.assertTrue(summary["ok"])
        self.assertEqual(fake.read_calls, [_key(FAKE_ROOT / "timeline.json")])
        self.assertNotIn(_key(FAKE_ROOT / "manifest.json"), fake.is_regular_calls)
        manifest = fake.writes[0][1]
        self.assertFalse(manifest["input_validation"]["bundle_manifest_present"])

    def test_contained_resolved_regular_member_is_read(self) -> None:
        declared = FAKE_ROOT / "timeline.json"
        actual = FAKE_ROOT / "real-timeline.json"
        content = [{"index": 0, "temporal_frame_paths": ["frames/a.png"]}]
        frame = _frame_key("frames/a.png")
        fake = ExporterFakeIO(
            timeline=content,
            regular={frame, _key(actual)},
            existing={_key(actual)},
            hashes={frame: "a" * 64},
            resolver=self._resolver_with(declared, actual),
            extra_reads={_key(actual): content},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        self.assertTrue(summary["ok"])
        self.assertEqual(fake.read_calls[0], _key(actual))
        self.assertIn(_key(actual), fake.is_regular_calls)

    def test_permission_error_resolving_bundle_input_propagates(self) -> None:
        def denied(_root: Path, reference: Path) -> Path:
            raise PermissionError(errno.EACCES, "access denied", _key(reference))

        fake = ExporterFakeIO(timeline=[], resolver=denied)
        with _patch_exporter(fake):
            with self.assertRaises(PermissionError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.writes, [])
        self.assertEqual(fake.read_calls, [])

    def test_permission_error_statting_timeline_is_not_missing(self) -> None:
        timeline_key = _key(FAKE_ROOT / "timeline.json")

        class DeniedStatFake(ExporterFakeIO):
            def is_regular(self, path: Path) -> bool:
                if _key(path) == timeline_key:
                    raise PermissionError(errno.EACCES, "access denied", str(path))
                return _key(path) in self.regular

        fake = DeniedStatFake(timeline=[])
        with _patch_exporter(fake):
            with self.assertRaises(PermissionError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.read_calls, [])
        self.assertEqual(fake.writes, [])


class ExporterSourceMediaTests(unittest.TestCase):
    def test_timeline_video_key_same_root_is_discovered_and_hashed(self) -> None:
        timeline = [{"index": 0, "video_key": "media/capture.mp4", "temporal_frame_paths": ["frames/a.png"]}]
        frame = _frame_key("frames/a.png")
        media = _frame_key("media/capture.mp4")
        fake = ExporterFakeIO(
            timeline=timeline, regular={frame, media}, hashes={frame: "a" * 64, media: "d" * 64}
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        manifest = fake.writes[0][1]
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["source_media_status"], "discovered_contained")
        self.assertEqual(summary["source_media_selection"], "timeline.json:video_key")
        self.assertEqual(manifest["source_media_path"], _key(FAKE_ROOT / "media/capture.mp4"))
        self.assertEqual(manifest["source_media_sha256"], "d" * 64)
        self.assertTrue(manifest["source_media"]["contained_by_bundle_root"])
        self.assertIn(media, fake.sha_calls)

    def test_manifest_media_path_outside_root_is_not_read_and_no_gate(self) -> None:
        timeline = [{"index": 0, "temporal_frame_paths": ["frames/a.png"]}]
        frame = _frame_key("frames/a.png")
        outside = str(OUTSIDE_ROOT / "capture.mp4")
        fake = ExporterFakeIO(
            timeline=timeline,
            bundle_manifest={"media_path": outside},
            regular={frame},
            hashes={frame: "a" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        manifest = fake.writes[0][1]
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["source_media_status"], "no_contained_regular_candidate")
        candidates = manifest["source_media"]["candidates"]
        self.assertEqual(candidates[0]["status"], "blocked")
        self.assertEqual(candidates[0]["blocked_reason"], EXPORTER.BLOCKED_OUTSIDE)
        self.assertNotIn(outside, fake.is_regular_calls)
        self.assertNotIn(outside, fake.sha_calls)
        self.assertEqual(manifest["source_media_path"], "")
        self.assertTrue(summary["available_frame_count"], 1)

    def test_relative_source_media_resolves_against_bundle_root(self) -> None:
        timeline = [{"index": 0, "temporal_frame_paths": ["frames/a.png"]}]
        frame = _frame_key("frames/a.png")
        media = _frame_key("media/capture.mp4")
        fake = ExporterFakeIO(
            timeline=timeline,
            bundle_manifest={"source_media_path": "media/capture.mp4"},
            regular={frame, media},
            hashes={frame: "a" * 64, media: "e" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        manifest = fake.writes[0][1]
        self.assertEqual(summary["source_media_status"], "discovered_contained")
        self.assertEqual(summary["source_media_selection"], "manifest.json:source_media_path")
        self.assertEqual(manifest["source_media_path"], _key(FAKE_ROOT / "media/capture.mp4"))
        self.assertIn((_key(FAKE_ROOT), _key(Path("media/capture.mp4"))), fake.resolve_calls)

    def test_explicit_operator_source_media_may_be_outside_and_overrides_bundle_values(self) -> None:
        timeline = [{"index": 0, "temporal_frame_paths": ["frames/a.png"]}]
        frame = _frame_key("frames/a.png")
        source = OPERATOR_ROOT / "source.mp4"
        untrusted = str(OUTSIDE_ROOT / "untrusted.mp4")
        fake = ExporterFakeIO(
            timeline=timeline,
            bundle_manifest={"media_path": untrusted},
            regular={frame, _key(source)},
            hashes={frame: "a" * 64, _key(source): "d" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT, source_media_path=str(source))
        manifest = fake.writes[0][1]
        self.assertEqual(summary["source_media_status"], "explicit_operator_declared")
        self.assertEqual(summary["source_media_selection"], "--source-media")
        self.assertEqual(manifest["source_media"]["path"], _key(source))
        self.assertFalse(manifest["source_media"]["contained_by_bundle_root"])
        self.assertEqual(manifest["source_media_sha256"], "d" * 64)
        self.assertNotIn(untrusted, fake.sha_calls)

    def test_frame_only_bundle_has_no_source_media_gate(self) -> None:
        timeline = [{"index": 0, "temporal_frame_paths": ["frames/a.png"]}]
        frame = _frame_key("frames/a.png")
        fake = ExporterFakeIO(timeline=timeline, regular={frame}, hashes={frame: "a" * 64})
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["source_media_status"], "not_declared")


class ExporterSourceMediaAuthorizationTests(unittest.TestCase):
    """Metadata distinguishes contained autodiscovery from explicit authorization."""

    def test_contained_autodiscovery_is_not_reported_as_explicit_only(self) -> None:
        frame = _frame_key("frames/a.png")
        media = _frame_key("media/capture.mp4")
        timeline = [
            {"index": 0, "video_key": "media/capture.mp4", "temporal_frame_paths": ["frames/a.png"]}
        ]
        fake = ExporterFakeIO(
            timeline=timeline,
            regular={frame, media},
            hashes={frame: "a" * 64, media: "d" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        manifest = fake.writes[0][1]
        boundary = manifest["operator_boundary"]
        self.assertEqual(
            boundary["source_media_authorization"], "bundle_root_contained_autodiscovery"
        )
        self.assertTrue(boundary["bundle_root_source_media_autodiscovery"])
        self.assertTrue(boundary["outside_root_source_media_requires_explicit_operator_path"])
        self.assertNotIn("explicit_source_media_only", boundary)
        self.assertEqual(
            manifest["source_media"]["authorization"], boundary["source_media_authorization"]
        )
        self.assertEqual(
            summary["source_media_authorization"], boundary["source_media_authorization"]
        )
        self.assertTrue(manifest["source_media"]["contained_by_bundle_root"])

    def test_explicit_outside_media_marks_operator_authorization(self) -> None:
        frame = _frame_key("frames/a.png")
        source = OPERATOR_ROOT / "source.mp4"
        fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            regular={frame, _key(source)},
            hashes={frame: "a" * 64, _key(source): "d" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT, source_media_path=str(source))
        manifest = fake.writes[0][1]
        boundary = manifest["operator_boundary"]
        self.assertEqual(
            boundary["source_media_authorization"],
            "explicit_operator_path_outside_bundle_root",
        )
        self.assertFalse(manifest["source_media"]["contained_by_bundle_root"])
        self.assertEqual(
            summary["source_media_authorization"], "explicit_operator_path_outside_bundle_root"
        )
        self.assertTrue(boundary["bundle_root_source_media_autodiscovery"])
        self.assertTrue(boundary["outside_root_source_media_requires_explicit_operator_path"])
        self.assertNotIn("explicit_source_media_only", boundary)

    def test_explicit_inside_media_keeps_inside_authorization(self) -> None:
        frame = _frame_key("frames/a.png")
        source = FAKE_ROOT / "media/source.mp4"
        fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            regular={frame, _key(source)},
            hashes={frame: "a" * 64, _key(source): "e" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT, source_media_path=str(source))
        manifest = fake.writes[0][1]
        self.assertEqual(
            manifest["operator_boundary"]["source_media_authorization"],
            "explicit_operator_path_inside_bundle_root",
        )
        self.assertTrue(manifest["source_media"]["contained_by_bundle_root"])
        self.assertEqual(
            summary["source_media_authorization"], "explicit_operator_path_inside_bundle_root"
        )

    def test_frame_only_bundle_reports_none_authorization(self) -> None:
        frame = _frame_key("frames/a.png")
        fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            regular={frame},
            hashes={frame: "a" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        boundary = fake.writes[0][1]["operator_boundary"]
        self.assertEqual(boundary["source_media_authorization"], "none")
        self.assertEqual(summary["source_media_authorization"], "none")


class ExporterDestinationTests(unittest.TestCase):
    def _valid_frame_fake(self, *, extra_regular=(), **kwargs) -> ExporterFakeIO:
        frame = _frame_key("frames/a.png")
        return ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            regular={frame, *extra_regular},
            hashes={frame: "a" * 64},
            **kwargs,
        )

    def test_symlink_frame_manifest_destination_refused_before_any_write(self) -> None:
        frame_destination = _key(FAKE_ROOT / "frame-manifest.json")
        fake = self._valid_frame_fake(symlinks={frame_destination})
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.writes, [])

    def test_frame_manifest_destination_resolving_outside_root_refused(self) -> None:
        frame_destination = _key(FAKE_ROOT / "frame-manifest.json")
        fake = self._valid_frame_fake(
            destination_resolutions={frame_destination: OUTSIDE_ROOT / "frame-manifest.json"}
        )
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.writes, [])

    def test_unrelated_existing_frame_manifest_refused(self) -> None:
        frame_destination = _key(FAKE_ROOT / "frame-manifest.json")
        fake = self._valid_frame_fake(
            extra_regular={frame_destination},
            existing={frame_destination},
            existing_frame_manifests={frame_destination: {"hello": "unrelated"}},
        )
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.writes, [])

    def test_owned_frame_manifest_refresh_is_allowed(self) -> None:
        frame_destination = _key(FAKE_ROOT / "frame-manifest.json")
        fake = self._valid_frame_fake(
            extra_regular={frame_destination},
            existing={frame_destination},
            existing_frame_manifests={frame_destination: {"schema": EXPORTER.SCHEMA, "frames": []}},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        self.assertTrue(summary["ok"])
        self.assertEqual([path for path, _ in fake.writes], [frame_destination])

    def test_nonregular_frame_manifest_destination_refused(self) -> None:
        frame_destination = _key(FAKE_ROOT / "frame-manifest.json")
        fake = self._valid_frame_fake(existing={frame_destination})
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError):
                EXPORTER.run_export(FAKE_ROOT)
        self.assertEqual(fake.writes, [])

    def test_manifest_json_symlink_refused_when_registering(self) -> None:
        manifest_destination = _key(FAKE_ROOT / "manifest.json")
        fake = self._valid_frame_fake(bundle_manifest={"project": "vkp"}, symlinks={manifest_destination})
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError):
                EXPORTER.run_export(FAKE_ROOT, register_bundle_manifest=True)
        self.assertEqual(fake.writes, [])

    def test_registration_is_opt_in_and_preserves_unrelated_manifest_keys(self) -> None:
        frame = _frame_key("frames/a.png")
        bundle_manifest = {"project": "vkp", "frame_manifest": "old.json"}
        fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            bundle_manifest=bundle_manifest,
            regular={frame},
            hashes={frame: "a" * 64},
        )
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        self.assertTrue(summary["ok"])
        self.assertFalse(summary["registered_bundle_manifest"])
        self.assertEqual([path for path, _ in fake.writes], [_key(FAKE_ROOT / "frame-manifest.json")])
        self.assertEqual(bundle_manifest, {"project": "vkp", "frame_manifest": "old.json"})

        registered_fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            bundle_manifest={"project": "vkp", "frame_manifest": "old.json"},
            regular={frame},
            hashes={frame: "a" * 64},
        )
        with _patch_exporter(registered_fake):
            summary = EXPORTER.run_export(FAKE_ROOT, register_bundle_manifest=True)
        self.assertTrue(summary["ok"])
        self.assertTrue(summary["registered_bundle_manifest"])
        paths = [path for path, _ in registered_fake.writes]
        self.assertEqual(paths, [_key(FAKE_ROOT / "frame-manifest.json"), _key(FAKE_ROOT / "manifest.json")])
        updated = registered_fake.writes[1][1]
        self.assertEqual(updated["project"], "vkp")
        self.assertEqual(updated["frame_manifest"], "frame-manifest.json")
        self.assertEqual(registered_fake.bundle_manifest, {"project": "vkp", "frame_manifest": "old.json"})

    def test_incomplete_bundle_is_not_registered(self) -> None:
        timeline = [{"index": 0, "temporal_frame_paths": ["frames/missing.png"]}]
        fake = ExporterFakeIO(timeline=timeline)
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT, register_bundle_manifest=True)
        self.assertFalse(summary["ok"])
        self.assertFalse(summary["registered_bundle_manifest"])
        self.assertEqual(summary["registration_skipped"], "incomplete_manifest_not_registered")
        self.assertEqual([path for path, _ in fake.writes], [_key(FAKE_ROOT / "frame-manifest.json")])


class ExporterDestinationValidationTests(unittest.TestCase):
    """A parsed destination is never accepted as proof that it is regular."""

    def test_nonregular_existing_destination_refused_even_when_known_valid(self) -> None:
        destination = FAKE_ROOT / "manifest.json"
        for kind in ("frame_manifest", "bundle_manifest"):
            with self.subTest(kind=kind):
                fake = ExporterFakeIO(timeline=[], nonregular_inputs={_key(destination)})
                with _patch_exporter(fake):
                    with self.assertRaises(EXPORTER.FramePlanError) as caught:
                        EXPORTER._validate_output_destination(
                            FAKE_ROOT,
                            destination,
                            kind=kind,
                            known_valid_existing=True,
                        )
                self.assertIn("non-regular", str(caught.exception))

    def test_regular_owned_frame_manifest_still_allowed_with_known_valid_flag(self) -> None:
        destination = FAKE_ROOT / "frame-manifest.json"
        fake = ExporterFakeIO(
            timeline=[],
            regular={_key(destination)},
            existing={_key(destination)},
            existing_frame_manifests={_key(destination): {"schema": EXPORTER.SCHEMA}},
        )
        with _patch_exporter(fake):
            EXPORTER._validate_output_destination(
                FAKE_ROOT,
                destination,
                kind="frame_manifest",
                known_valid_existing=True,
            )
        self.assertEqual(fake.read_calls, [_key(destination)])

    def test_unrelated_artifact_refused_even_when_known_valid(self) -> None:
        destination = FAKE_ROOT / "frame-manifest.json"
        fake = ExporterFakeIO(
            timeline=[],
            regular={_key(destination)},
            existing={_key(destination)},
            existing_frame_manifests={_key(destination): {"hello": "unrelated"}},
        )
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError):
                EXPORTER._validate_output_destination(
                    FAKE_ROOT,
                    destination,
                    kind="frame_manifest",
                    known_valid_existing=True,
                )


class ExporterDestinationStatusTests(unittest.TestCase):
    """Pure computation defers destination status; the writer reports it."""

    def _valid_fake(self, **kwargs) -> ExporterFakeIO:
        frame = _frame_key("frames/a.png")
        return ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            regular={frame},
            hashes={frame: "a" * 64},
            **kwargs,
        )

    def test_pure_build_marks_destinations_not_checked(self) -> None:
        fake = self._valid_fake()
        with _patch_exporter(fake):
            manifest = EXPORTER.build_frame_manifest(FAKE_ROOT, fake.timeline, {})
        boundary = manifest["operator_boundary"]
        self.assertFalse(boundary["output_destinations_validated"])
        self.assertEqual(
            boundary["output_destinations_validation"], "not_checked_pure_computation"
        )
        self.assertEqual(fake.resolve_destination_calls, [])
        self.assertEqual(fake.writes, [])

    def test_writing_path_reports_validated_after_checks_before_write(self) -> None:
        fake = self._valid_fake()
        with _patch_exporter(fake):
            summary = EXPORTER.run_export(FAKE_ROOT)
        self.assertTrue(summary["output_destinations_validated"])
        written = fake.writes[0][1]
        self.assertTrue(written["operator_boundary"]["output_destinations_validated"])
        self.assertEqual(
            written["operator_boundary"]["output_destinations_validation"],
            "validated_before_write",
        )
        destination_key = _key(FAKE_ROOT / "frame-manifest.json")
        self.assertLess(
            fake.events.index(("resolve_destination", destination_key)),
            fake.events.index(("write", destination_key)),
        )

    def test_registration_destination_validated_before_any_write(self) -> None:
        frame = _frame_key("frames/a.png")
        manifest_destination = _key(FAKE_ROOT / "manifest.json")
        fake = ExporterFakeIO(
            timeline=[{"index": 0, "temporal_frame_paths": ["frames/a.png"]}],
            bundle_manifest={"project": "vkp"},
            regular={frame},
            hashes={frame: "a" * 64},
            symlinks={manifest_destination},
        )
        with _patch_exporter(fake):
            with self.assertRaises(EXPORTER.FramePlanError):
                EXPORTER.run_export(FAKE_ROOT, register_bundle_manifest=True)
        self.assertEqual(fake.writes, [])
        self.assertIn(
            _key(FAKE_ROOT / "frame-manifest.json"), fake.resolve_destination_calls
        )


class ExporterManifestSemanticsTests(unittest.TestCase):
    def test_extractor_provenance_is_not_fabricated_and_no_process_imports(self) -> None:
        timeline = [{"index": 0, "temporal_frame_paths": ["frames/a.png"]}]
        frame = _frame_key("frames/a.png")
        fake = ExporterFakeIO(timeline=timeline, regular={frame}, hashes={frame: "a" * 64})
        with _patch_exporter(fake):
            manifest = EXPORTER.build_frame_manifest(FAKE_ROOT, fake.timeline, {})
        self.assertEqual(manifest["extractor"]["name"], "unknown")
        self.assertEqual(manifest["extractor"]["provenance"], "not_probed")
        self.assertIsNone(manifest["extractor"]["version"])
        self.assertFalse(hasattr(EXPORTER, "subprocess"))
        self.assertFalse(hasattr(EXPORTER, "shutil"))

    def test_frame_manifest_v1_fields_and_counts_are_present(self) -> None:
        timeline = [{"index": 0, "temporal_frame_paths": ["frames/a.png"]}]
        frame = _frame_key("frames/a.png")
        fake = ExporterFakeIO(timeline=timeline, regular={frame}, hashes={frame: "a" * 64})
        with _patch_exporter(fake):
            manifest = EXPORTER.build_frame_manifest(FAKE_ROOT, fake.timeline, {})
        for field in (
            "schema",
            "generated_at",
            "bundle_dir",
            "source_media_path",
            "source_media_sha256",
            "source_media",
            "extractor",
            "frame_count",
            "available_frame_count",
            "missing_frame_count",
            "frames",
            "operator_boundary",
        ):
            self.assertIn(field, manifest)
        self.assertEqual(manifest["schema"], "video_knowledge_pipeline.frame_manifest.v1")
        self.assertEqual(manifest["frame_count"], 1)
        self.assertEqual(manifest["available_frame_count"], 1)
        self.assertEqual(manifest["missing_frame_count"], 0)
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["operator_boundary"]["output_destinations_validated"])
        self.assertEqual(
            manifest["operator_boundary"]["output_destinations_validation"],
            "not_checked_pure_computation",
        )
        row = manifest["frames"][0]
        for field in ("timeline_index", "start_seconds", "end_seconds", "role", "path", "exists", "sha256", "evidence_status"):
            self.assertIn(field, row)
        self.assertEqual(row["role"], "temporal_frame")
        self.assertTrue(row["exists"])
        self.assertEqual(row["sha256"], "a" * 64)

    def test_empty_string_reference_never_resolves_to_bundle_root(self) -> None:
        timeline = [{"index": 0, "temporal_frame_paths": [""]}]
        fake = ExporterFakeIO(timeline=timeline)
        with _patch_exporter(fake):
            manifest = EXPORTER.build_frame_manifest(FAKE_ROOT, fake.timeline, {})
        self.assertEqual(manifest["blocked_frame_count"], 1)
        self.assertEqual(manifest["frames"][0]["evidence_status"], "blocked")
        self.assertEqual(manifest["frames"][0]["blocked_reason"], EXPORTER.BLOCKED_EMPTY)
        self.assertEqual(fake.resolve_calls, [])
        self.assertEqual(fake.is_regular_calls, [])


class ExporterLexicalPathTests(unittest.TestCase):
    """Pure POSIX/Windows lexical cases; no claim of a Linux runtime run."""

    def test_pure_posix_and_windows_lexical_helpers(self) -> None:
        posix_abs = str(PurePosixPath("/srv/bundle/frames/a.png"))
        posix_rel = str(PurePosixPath("bundle/frames/a.png"))
        windows_abs = str(PureWindowsPath("C:/bundle/frames/a.png"))
        windows_drive_relative = str(PureWindowsPath("C:bundle/frames/a.png"))
        windows_unc = str(PureWindowsPath("//server/share/frames/a.png"))
        self.assertTrue(EXPORTER._is_absolute_reference(posix_abs))
        self.assertFalse(EXPORTER._is_absolute_reference(posix_rel))
        self.assertIsNone(EXPORTER._reference_drive(posix_abs))
        self.assertFalse(EXPORTER._is_unc_reference(posix_abs))
        self.assertEqual(EXPORTER._reference_drive(windows_abs), "C:")
        self.assertTrue(EXPORTER._is_absolute_reference(windows_abs))
        self.assertFalse(EXPORTER._is_absolute_reference(windows_drive_relative))
        self.assertTrue(EXPORTER._is_unc_reference(windows_unc))

    def test_windows_style_references_remain_real_validator_cases(self) -> None:
        # These are lexical classifications; all three return before resolution,
        # so the cases are filesystem-free on both platforms.
        windows_root = Path("C:/synthetic-bundle")
        self.assertEqual(
            EXPORTER._classify_reference(windows_root, "D:/other-synthetic/frame.png")[
                "blocked_reason"
            ],
            EXPORTER.BLOCKED_FOREIGN_DRIVE,
        )
        self.assertEqual(
            EXPORTER._classify_reference(windows_root, "\\\\server\\share\\frame.png")[
                "blocked_reason"
            ],
            EXPORTER.BLOCKED_UNC,
        )
        self.assertEqual(
            EXPORTER._classify_reference(windows_root, "C:drive-relative-frame.png")[
                "blocked_reason"
            ],
            EXPORTER.BLOCKED_DRIVE_RELATIVE,
        )


class ExampleHelperTests(unittest.TestCase):
    def test_frame_command_uses_local_protocol_and_bounded_threads(self) -> None:
        command = EXAMPLE._frame_command("ffmpeg", MEDIA_ROOT / "video.mp4", MEDIA_ROOT / "frame.png")
        self.assertIsInstance(command, list)
        self.assertEqual(command[0], "ffmpeg")
        self.assertEqual(command[command.index("-protocol_whitelist") + 1], "file")
        thread_values = [command[index + 1] for index, part in enumerate(command) if part == "-threads"]
        self.assertGreaterEqual(len(thread_values), 2)
        self.assertTrue(all(value == str(EXAMPLE.EXTRACT_THREADS) for value in thread_values))
        self.assertEqual(command[command.index("-filter_threads") + 1], str(EXAMPLE.EXTRACT_THREADS))
        self.assertEqual(command[-1], str(MEDIA_ROOT / "frame.png"))

    def test_extract_frame_passes_bounded_timeout_and_checks_output(self) -> None:
        captured = {}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(EXAMPLE, "_ensure_parent_dir", lambda path: None))
            stack.enter_context(
                mock.patch.object(
                    EXAMPLE,
                    "_run_process",
                    lambda command, timeout: captured.update(command=command, timeout=timeout) or SimpleNamespace(),
                )
            )
            stack.enter_context(mock.patch.object(EXAMPLE, "_is_regular_file", lambda path: True))
            stack.enter_context(mock.patch.object(EXAMPLE, "_file_size", lambda path: 10))
            EXAMPLE._extract_frame("ffmpeg", MEDIA_ROOT / "video.mp4", MEDIA_ROOT / "frame.png")
        self.assertEqual(captured["timeout"], EXAMPLE.EXTRACT_TIMEOUT_SECONDS)
        self.assertEqual(
            captured["command"],
            EXAMPLE._frame_command("ffmpeg", MEDIA_ROOT / "video.mp4", MEDIA_ROOT / "frame.png"),
        )

    def test_report_paths_use_documented_bases(self) -> None:
        paths = EXAMPLE._report_paths("talking-head-synthetic", "talking-head-synthetic.mp4", "frame-00.png")
        self.assertEqual(
            paths["media_path_relative_to_output_root"],
            "fixtures/talking-head-synthetic/talking-head-synthetic.mp4",
        )
        self.assertEqual(paths["frame_path_relative_to_output_root"], "bundle/frames/frame-00.png")
        self.assertEqual(
            paths["frame_reference_in_timeline_relative_to_bundle_root"], "frames/frame-00.png"
        )

    def test_run_process_closes_stdin(self) -> None:
        captured = {}
        with mock.patch.object(
            EXAMPLE.subprocess,
            "run",
            lambda command, **kwargs: captured.update(command=command, kwargs=kwargs) or SimpleNamespace(),
        ):
            EXAMPLE._run_process(["ffmpeg", "-version"], 5)
        self.assertIs(captured["kwargs"]["stdin"], EXAMPLE.subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
