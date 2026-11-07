from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(?:\[(?P<bracket>(?:speaker|spk)[\s_-]*\d+|s\d+|说话人\s*\d+)\]"
    r"|(?P<plain>(?:speaker|spk)[\s_-]*\d+|s\d+|说话人\s*\d+))"
    r"\s*[：:]\s*(?P<text>.*)$",
    re.IGNORECASE,
)
_CHINESE_SPEAKER_RE = re.compile(r"^说话人\s*(\d+)$")


def cue_speaker(cue: Any) -> str:
    """Return the immutable diarization cluster identity carried by a cue."""

    # Intent: keep one anonymous speaker label across resumable ASR chunks.
    # Decision: prefer the derived recording-global ID when present, while
    # retaining the original local CAM++ cluster in cue metadata.
    # Reason: chunk-local 0/1 labels restart for every child process and cannot
    # represent a recording-level speaker identity.
    # Evidence: speaker_global_alignment adapts FunASR's tested centroid mapper;
    # no person name or role is inferred here.
    # Effective scope: reader/normalized anonymous labels only. Sources without
    # ``speaker_global_id`` retain the historical resolver order.
    return _first_value(
        cue,
        "speaker_global_id",
        "speaker",
        "speaker_id",
        "spk",
        "spk_id",
    )


def cue_speaker_role(cue: Any) -> str:
    """Return an optional role without conflating it with cluster identity."""

    return _first_value(cue, "speaker_role", "role")


def speaker_label_map(cues: Iterable[Any]) -> dict[str, str]:
    """Create stable anonymous Chinese labels in first-appearance order.

    Intent: retain MOSS/FunASR speaker clusters through VKP's existing transcript
    contract.
    Decision: adapt the upstream ``speaker`` field to ``说话人N`` only for reader
    output while preserving the raw cluster ID in JSON.
    Reason: role or personal-name inference is a separate, uncertain task and
    must never overwrite diarization identity.
    Evidence: MOSS ``segments.json`` and subtitle exporters keep
    ``start/end/text/speaker`` as separate fields.
    Effective scope: transcript serialization and display only; this module does
    not run diarization or infer who a speaker is.
    """

    labels: dict[str, str] = {}
    for cue in cues:
        raw = cue_speaker(cue)
        key = speaker_key(raw)
        if not key or key in labels:
            continue
        match = _CHINESE_SPEAKER_RE.fullmatch(raw)
        labels[key] = f"说话人{match.group(1)}" if match else f"说话人{len(labels) + 1}"
    return labels


def speaker_display_name(cue: Any, labels: Mapping[str, str] | None = None) -> str:
    raw = cue_speaker(cue)
    if not raw:
        return ""
    label = (labels or {}).get(speaker_key(raw), "")
    if not label:
        match = _CHINESE_SPEAKER_RE.fullmatch(raw)
        label = f"说话人{match.group(1)}" if match else raw
    role = cue_speaker_role(cue)
    if role and speaker_key(role) not in {speaker_key(raw), speaker_key(label)}:
        return f"{role}（{label}）"
    return label


def speaker_payload(cue: Any, labels: Mapping[str, str] | None = None) -> dict[str, str]:
    raw = cue_speaker(cue)
    role = cue_speaker_role(cue)
    payload: dict[str, str] = {}
    if raw:
        payload["speaker"] = raw
        payload["speaker_label"] = speaker_display_name(cue, labels)
    if role:
        payload["speaker_role"] = role
    return payload


def split_speaker_prefix(text: str) -> tuple[str, str]:
    """Parse the speaker prefix emitted by MOSS-style SRT exports."""

    value = str(text or "").strip()
    match = _SPEAKER_PREFIX_RE.match(value)
    if not match:
        return "", value
    speaker = normalise_speaker_value(match.group("bracket") or match.group("plain"))
    return speaker, str(match.group("text") or "").strip()


def speaker_key(value: str) -> str:
    return re.sub(r"[\s_-]+", "", normalise_speaker_value(value)).casefold()


def normalise_speaker_value(value: Any) -> str:
    """Normalize a diarization ID without dropping the numeric cluster ``0``.

    Intent: preserve every anonymous speaker cluster emitted by CAM++/FunASR.
    Decision: treat only ``None`` as missing; stringify integers including zero.
    Reason: ``value or ""`` collapsed cluster ``0`` and made a two-speaker
    recording appear to contain just one labeled speaker.
    Evidence: the real CAM++ trial emitted both ``spk: 0`` and ``spk: 1``.
    Effective scope: speaker metadata normalization and anonymous labels only;
    no role, name, or identity is inferred.
    """

    text = "" if value is None else str(value).strip()
    match = _CHINESE_SPEAKER_RE.fullmatch(text)
    return f"说话人{match.group(1)}" if match else text


def _first_value(value: Any, *keys: str) -> str:
    metadata: Mapping[str, Any] = {}
    if isinstance(value, Mapping):
        direct = value
        raw_metadata = value.get("metadata")
        if isinstance(raw_metadata, Mapping):
            metadata = raw_metadata
    else:
        direct = {
            key: getattr(value, key, "")
            for key in keys
        }
        raw_metadata = getattr(value, "metadata", {})
        if isinstance(raw_metadata, Mapping):
            metadata = raw_metadata
    for key in keys:
        candidate = direct.get(key) if isinstance(direct, Mapping) else ""
        if candidate is None or not str(candidate).strip():
            candidate = metadata.get(key)
        if candidate is not None and str(candidate).strip():
            return normalise_speaker_value(candidate)
    return ""
