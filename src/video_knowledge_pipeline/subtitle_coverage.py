"""Evidence-bound subtitle coverage, deliberately separate from ASR success."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .interval_coverage import interval_coverage

SCHEMA = "video_knowledge_pipeline.subtitle_coverage.v1"


def read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _interval(row: dict, duration: float) -> tuple[float, float]:
    start, end = float(row["start_seconds"]), float(row["end_seconds"])
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= duration:
        raise ValueError(f"invalid interval: {start}, {end}; duration={duration}")
    return start, end


def _text(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def observations_from_asr(raw: dict, *, evidence_ref: str, duration: float) -> list[dict]:
    """Adapt VKP FunASR raw milliseconds or explicit normalized seconds.

    Punctuation timestamps must not stretch two short phrases across an entire
    song. Prefer timed words, group only adjacent words, retain segment fallback
    as candidate evidence (never a statement that the intervening audio is known).
    """
    observations: list[dict] = []
    if str(raw.get("schema", "")).startswith("video_knowledge_funasr_"):
        for record in raw.get("result", []):
            words, times = record.get("words", []), record.get("timestamp", [])
            if words and len(words) == len(times):
                groups: list[dict] = []
                for word, pair in zip(words, times):
                    if not _text(str(word)):
                        continue
                    start, end = float(pair[0]) / 1000, float(pair[1]) / 1000
                    _interval({"start_seconds": start, "end_seconds": end}, duration)
                    if groups and 0 <= start - groups[-1]["end_seconds"] <= .7:
                        groups[-1]["end_seconds"] = end
                        groups[-1]["text"] += str(word)
                    else:
                        groups.append({"start_seconds": start, "end_seconds": end, "text": str(word)})
                observations.extend(groups)
            else:
                sentences = record.get("sentence_info", [])
                if (words and len(words) != len(times)
                        and any(_text(str(word)) for word in words)
                        and not any(_text(sentence.get("text", "")) for sentence in sentences)):
                    raise ValueError("ASR word/timestamp length mismatch without sentence fallback")
                for sentence in sentences:
                    if _text(sentence.get("text", "")):
                        observations.append({"start_seconds": float(sentence["start"]) / 1000,
                                             "end_seconds": float(sentence["end"]) / 1000,
                                             "text": sentence["text"]})
    elif raw.get("schema") == "video_knowledge_pipeline.subtitle_asr_observations.v1":
        observations = [dict(row) for row in raw["segments"]]
    else:
        raise ValueError("unsupported ASR format; provide explicit seconds adapter schema")
    for row in observations:
        _interval(row, duration)
        row.update({"kind": "speech", "evidence_ref": evidence_ref, "origin": "asr"})
    return observations


def audit_subtitle_coverage(request: dict, review_pack: dict, *, source_hash: str,
                            review: dict | None = None) -> dict:
    """All timing is edited-source seconds; no transcript rewriting or rendering.

    request binds ASR/independent audio observations and omission decisions to the
    exact edited source. A human receipt additionally binds the complete audit
    input fingerprint, so changing a cue/omission/evidence invalidates approval.
    This pure API does not read evidence bytes or assess the truth of caller
    supplied ASR quality/listening assertions. evidence_ref is a label; use
    check_subtitles for declared local-file hash and duration checks.
    """
    duration = float(request["duration_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("positive finite duration required")
    if request.get("source_sha256") != source_hash:
        raise ValueError("source hash mismatch")
    requests = review_pack.get("requests", [])
    if len(requests) != 1:
        raise ValueError("one full-timeline subtitle review request required")
    draft = requests[0]
    if draft.get("artifact_sha256") != source_hash or draft.get("full_source_review") is not True:
        raise ValueError("subtitle pack must bind the same source and full timeline")
    if draft.get("review_start_seconds") != 0 or abs(float(draft["review_end_seconds"]) - duration) > .001:
        raise ValueError("subtitle pack must span the entire edited source")
    cues = draft["subtitle_cues"]
    for cue in cues:
        _interval(cue, duration)
        if not _text(cue.get("text", "")):
            raise ValueError("empty subtitle cue")
    omissions = request.get("omissions", [])
    for row in omissions:
        _interval(row, duration)
        if not row.get("reason", "").strip() or not row.get("evidence_ref", "").strip():
            raise ValueError("omission requires reason and evidence_ref")
    observations = request.get("observations", [])
    for row in observations:
        _interval(row, duration)
        if row.get("kind") not in {"speech", "singing", "vocalization", "audio_activity"}:
            raise ValueError("unknown observation kind")
        if not row.get("evidence_ref", "").strip():
            raise ValueError("observation requires evidence_ref")
    binding = fingerprint({"schema": SCHEMA, "request": request, "review_pack": review_pack})
    findings: list[dict] = []

    def flag(code: str, start: float, end: float, detail: str) -> None:
        body = {"code": code, "start_seconds": start, "end_seconds": end, "detail": detail}
        findings.append({"finding_id": fingerprint(body)[:20], **body})

    asr = request.get("asr", {})
    if asr.get("execution_status") != "completed" or asr.get("coverage_quality") != "adequate":
        flag("asr_coverage_unverified", 0, duration,
             "ASR execution success does not establish singing/speech completeness")
    if not observations:
        flag("no_audio_observations", 0, duration, "No recognized or independently observed audio spans")
    asr_intervals = [_interval(x, duration) for x in observations if x.get("origin") == "asr"]
    for row in request.get("expected_audio", []):
        start, end = _interval(row, duration)
        support = interval_coverage([(start, end)], asr_intervals,
                                    minimum_gap_seconds=.5, minimum_coverage_ratio=.5)
        if support["status"] != "passed":
            flag("expected_audio_not_recognized", start, end,
                 "Expected edited speech/song/outro has weak ASR evidence; check by listening")
        coverage = interval_coverage([(start, end)], [_interval(x, duration) for x in [*cues, *omissions]],
                                    minimum_gap_seconds=.08, maximum_uncovered_seconds=.15,
                                    minimum_coverage_ratio=.98)
        if coverage["status"] != "passed":
            flag("expected_audio_without_caption_or_reason", start, end,
                 json.dumps(coverage["all_gaps"], ensure_ascii=False))
    resolved = [*cues, *omissions]
    for row in observations:
        start, end = _interval(row, duration)
        coverage = interval_coverage([(start, end)], [_interval(x, duration) for x in resolved],
                                    minimum_gap_seconds=.08, maximum_uncovered_seconds=.15,
                                    minimum_coverage_ratio=.98)
        if coverage["status"] != "passed":
            flag("audio_without_caption_or_reason", start, end,
                 json.dumps(coverage["all_gaps"], ensure_ascii=False))
        # Activity/VAD is not speech truth. It needs classification even if a
        # long subtitle happens to cover it; text match is not semantic proof.
        if row["kind"] == "audio_activity":
            flag("unclassified_audio_activity", start, end, "Listen: speech, song, laughter, music or silence?")
        expected = _text(row.get("text", ""))
        overlaps = [x for x in cues if x["start_seconds"] < end and x["end_seconds"] > start]
        omitted = interval_coverage([(start, end)], [_interval(x, duration) for x in omissions],
                                   minimum_gap_seconds=.08, maximum_uncovered_seconds=.15,
                                   minimum_coverage_ratio=.98)["status"] == "passed"
        if expected and not omitted and expected not in _text("".join(x["text"] for x in overlaps)):
            flag("text_correspondence_review", start, end,
                 "Observed text differs; preserve human lyrics and adjudicate, never auto-normalize: " + row["text"])
    review = review or {}
    review_valid = (
        review.get("input_fingerprint") == binding
        and review.get("source_sha256") == source_hash
        and review.get("review_id") == draft.get("review_id")
        and review.get("full_timeline_listened") is True
        and review.get("full_timeline_viewed") is True
        and bool(str(review.get("reviewer", "")).strip())
        and bool(str(review.get("reviewed_at", "")).strip())
        and review.get("decision") == "pass"
    )
    resolutions = review.get("resolutions", {}) if review_valid else {}
    unresolved = []
    for item in findings:
        resolution = resolutions.get(item["finding_id"], {})
        item["resolved"] = (resolution.get("decision") == "accepted_after_listening"
                            and bool(str(resolution.get("reason", "")).strip()))
        if not item["resolved"]:
            unresolved.append(item["finding_id"])
    ready = bool(review_valid and not unresolved)
    review_request = dict(draft)
    review_request.update({
        "status": "pending", "review_purpose": "整片字幕缺漏与识别盲区复核",
        "review_question": "连续看听整片，核对正文、歌词、片头片尾及笑声；按 findings 记录原因。不要把 ASR 漏检当静音，不改写已认可唱词。",
        "coverage_input_fingerprint": binding, "coverage_findings": findings,
    })
    return {
        "schema": SCHEMA, "source_sha256": source_hash, "duration_seconds": duration,
        "input_fingerprint": binding, "status": "passed" if ready else "review_required",
        "render_ready": ready, "human_review_valid": bool(review_valid),
        "findings": findings, "unresolved_finding_ids": unresolved,
        "omissions": omissions, "subtitle_cues_unchanged": cues,
        "subtitle_free_intervals": interval_coverage([(0, duration)], [_interval(x, duration) for x in cues],
                                                       minimum_gap_seconds=0)["all_gaps"],
        "review_request_pack": {"schema_version": "self-media-human-review-request-pack.v1",
                                "scope": "full_video_subtitle_coverage_review", "requests": [review_request]},
        "boundaries": {"semantic_equivalence_proven": False, "publication_authorized": False,
                       "media_modified": False, "asr_success_is_coverage_pass": False,
                       "evidence_ref_hashes_verified": False,
                       "asr_coverage_quality_verified": False,
                       "source_file_hash_checked": False,
                       "source_duration_probe_checked": False,
                       "declared_evidence_file_hash_checks": 0,
                       "asr_source_output_hash_binding_checked": False},
    }


def check_subtitles(request_path: str | Path, pack_path: str | Path, *,
                    review_path: str | Path | None = None,
                    asr_path: str | Path | None = None) -> dict:
    request = read_object(request_path)
    media = Path(request["source_path"]).resolve()
    actual_hash = sha256_file(media)
    # The CLI does not trust a stale declared duration.
    from .video import probe_video
    if abs(probe_video(media).duration_seconds - float(request["duration_seconds"])) > .05:
        raise ValueError("edited source duration mismatch")
    for evidence in request.get("evidence_files", []):
        if sha256_file(Path(evidence["path"])) != evidence["sha256"]:
            raise ValueError("evidence file changed: " + evidence["path"])
    if asr_path:
        path = Path(asr_path).resolve()
        raw = read_object(path)
        # Raw runners generally record the source path, not an immutable hash.
        # A separate binding captured when the ASR ran is required; never infer
        # freshness solely because a current file happens to have that path.
        binding = request.get("asr_binding", {})
        if binding.get("source_sha256") != actual_hash or binding.get("asr_sha256") != sha256_file(path):
            raise ValueError("ASR must have exact source/output hash binding")
        if raw.get("input") and Path(raw["input"]).resolve() != media:
            raise ValueError("ASR belongs to another media path")
        request["observations"] = [*request.get("observations", []),
            *observations_from_asr(raw, evidence_ref=str(path), duration=float(request["duration_seconds"]))]
        request["asr"] = {"execution_status": raw.get("status", "unknown"),
                          "coverage_quality": "unverified", "sha256": sha256_file(path)}
    report = audit_subtitle_coverage(request, read_object(pack_path), source_hash=actual_hash,
                                     review=read_object(review_path) if review_path else None)
    report["boundaries"].update(source_file_hash_checked=True,
        source_duration_probe_checked=True,
        declared_evidence_file_hash_checks=len(request.get("evidence_files", [])),
        asr_source_output_hash_binding_checked=bool(asr_path))
    return report
