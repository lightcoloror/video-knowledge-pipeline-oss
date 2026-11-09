# Targeted Pacing Review Module Reference

Target: `src/video_knowledge_pipeline/targeted_pacing.py`
License: AGPL-3.0-only
Tests: `tests/test_targeted_pacing.py`

## Overview
`targeted_pacing.py` provides a read-only, standard-library-only validation bridge for pacing candidate packets and operator review states. It validates Python data and reads a local candidate-packet file through the supplied root without console entrypoints, FFmpeg execution, external model dependencies, or media modification. Pending pacing suggestions remain reversible and are never promoted to an approved Edit Decision List (EDL).

## Core APIs
1. `load_pacing_candidates(root: Path, pack: dict) -> dict | None`:
   - Checks for `targeted-transcript-pacing.candidates.json` under `root`; returns `None` if absent.
   - Enforces schema `vkp.targeted-pacing-candidates.v1`, canonical JSON SHA-256 self-digest matching, and parent `pack_payload_sha256` binding.
   - Verifies each candidate against chapter recorded identities (`source_hash` and `source_sha256`).
   - Validates ordered, finite timestamp windows (`0 <= protect_start <= pause_start < start < end < pause_end <= protect_end <= duration`).
   - Rejects candidate intervals that overlap transcript word segment timings (`1e-7` tolerance).
2. `validate_pacing_review(state: dict | None, packet: dict) -> None`:
   - Returns safely if `state` is `None`.
   - Requires valid `packet_sha256` reference, boolean `enabled` flag, and recognized decision identifiers.
   - Permits actions `keep_pending_review`, `keep`, and `compress`.
   - Restricts `compress` to items where `audio_checked` and `context_checked` are literally `True`, `semantic_role` equals `'ordinary'`, and non-empty `evidence` text is provided.

## Operational Boundaries
All hash checks and boolean flags evaluate supplied record fields only; they do not establish cryptographic source provenance, audio listening authenticity, or external human consent. The module does not suggest cuts, render video, or provide filesystem sandboxing. One bounded command imported the actual module successfully under Python 3.13.5 and discovered the four unchanged synthetic tests. The first test encountered PermissionError while writing its synthetic TemporaryDirectory file: 0 passed, 1 error, 3 not run under fail-fast. Standard-library cleanup then attempted a permission reset and was also denied. No manual permission repair, alternate host/path/interpreter, cleanup or retry followed. The original suite did not pass; no functional branch, real media/hash authenticity, audio review, UI or full-workflow success is claimed.

The optional transcript editor, cue/player UI, generated frontend bundles and upstream vendor material are not part of this small module increment. Their full existing sources and unresolved conditions are retained separately; this source does not replace them or remove their guards.
