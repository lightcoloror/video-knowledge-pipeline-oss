"""Bounded JSON source locators, adapting existing Bundle identity/path gates.

Like Docling provenance, these identify a source object, not its truth. Typed
selectors use normal JSON container access; there is no new query language.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from .canonical_json import canonical_json_sha256

SCHEMA = 'video_knowledge_pipeline.evidence_locator.v1'
TEXT_SCHEMA = 'video_knowledge_pipeline.text_evidence_locator.v1'
MAX_BYTES = 64 * 1024 * 1024
MAX_LOCATORS = 4096


def _read_source_bytes(root, relative_path, extensions, max_bytes):
    from .courseware_material import safe_path
    if (not isinstance(relative_path, str) or len(relative_path) > 1024
            or '\\' in relative_path or PureWindowsPath(relative_path).drive
            or Path(relative_path).is_absolute() or Path(relative_path).suffix.lower() not in extensions):
        raise ValueError('source_locator requires Bundle-relative supported source path')
    path = safe_path(Path(root).absolute(), relative_path)
    if not 0 < max_bytes <= MAX_BYTES or path.stat().st_size > max_bytes:
        raise ValueError('source_locator file budget exceeded')
    with path.open('rb') as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError('source_locator file budget exceeded')
    return path.relative_to(Path(root).absolute()).as_posix(), raw


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('source_locator duplicate JSON key')
        result[key] = value
    return result


def _select(data: Any, selector: Any) -> Any:
    if not isinstance(selector, list) or not 1 <= len(selector) <= 16:
        raise ValueError('source_locator selector budget/type')
    value = data
    for key in selector:
        if type(key) is int and key >= 0 and isinstance(value, list) and key < len(value):
            value = value[key]
        elif type(key) is str and 0 < len(key) <= 512 and isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise ValueError('source_locator selector missing or invalid')
    return value


@dataclass(frozen=True)
class JsonEvidenceSnapshot:
    artifact_path: str
    artifact_sha256: str
    data: Any
    size_bytes: int

    @classmethod
    def read(cls, root: Path, relative_path: str, *, max_bytes: int = MAX_BYTES):
        path, raw = _read_source_bytes(root, relative_path, {'.json'}, max_bytes)
        data = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite JSON')))
        return cls(path, hashlib.sha256(raw).hexdigest(), data, len(raw))

    def selected(self, selector):
        return _select(self.data, selector)

    def locator(self, selector: list[str | int]) -> dict:
        selected = self.selected(selector)
        return dict(schema=SCHEMA, artifact_path=self.artifact_path,
                    artifact_sha256=self.artifact_sha256, selector=list(selector),
                    selected_sha256=canonical_json_sha256(selected))


@dataclass(frozen=True)
class TextEvidenceSnapshot:
    """Exact UTF-8-sig decoded character ranges; raw bytes retain file identity.

    Coordinates are Unicode code points, end-exclusive, not byte offsets. No
    newline normalization, HTML cleaning, parser or media reading happens here.
    """
    artifact_path: str
    artifact_sha256: str
    data: str
    size_bytes: int

    @classmethod
    def read(cls, root: Path, relative_path: str, *, max_bytes: int = MAX_BYTES):
        path, raw = _read_source_bytes(root, relative_path, {'.srt', '.vtt', '.txt', '.md'}, max_bytes)
        return cls(path, hashlib.sha256(raw).hexdigest(), raw.decode('utf-8-sig'), len(raw))

    def selected(self, selector):
        if (not isinstance(selector, list) or len(selector) != 2
                or any(type(n) is not int for n in selector)
                or not 0 <= selector[0] < selector[1] <= len(self.data)):
            raise ValueError('source_locator character range invalid')
        return self.data[selector[0]:selector[1]]

    def locator(self, selector):
        return dict(schema=TEXT_SCHEMA, artifact_path=self.artifact_path,
                    artifact_sha256=self.artifact_sha256, selector=list(selector),
                    selected_sha256=canonical_json_sha256(self.selected(selector)))


def read_evidence_snapshot(root, path, schema, *, max_bytes=MAX_BYTES):
    reader = {SCHEMA: JsonEvidenceSnapshot, TEXT_SCHEMA: TextEvidenceSnapshot}.get(schema)
    if reader is None:
        raise ValueError('source_locator schema invalid')
    return reader.read(root, path, max_bytes=max_bytes)


def verify_evidence_locators(root: Path, locators: list[dict]) -> dict:
    """Verify all or raise; hash the exact bytes parsed, never mutate sources."""
    if not isinstance(locators, list) or len(locators) > MAX_LOCATORS:
        raise ValueError('source_locator count budget exceeded')
    cache = {}
    verified = set()
    total_bytes = 0
    for loc in locators:
        if (not isinstance(loc, dict) or set(loc) != {'schema', 'artifact_path', 'artifact_sha256', 'selector', 'selected_sha256'}
                or loc.get('schema') not in (SCHEMA, TEXT_SCHEMA)):
            raise ValueError('source_locator schema invalid')
        for key in ('artifact_sha256', 'selected_sha256'):
            if not isinstance(loc[key], str) or not re.fullmatch('[0-9a-f]{64}', loc[key]):
                raise ValueError('source_locator hash invalid')
        path = loc['artifact_path']
        if not isinstance(path, str):
            raise ValueError('source_locator path invalid')
        if path not in cache:
            cache[path] = read_evidence_snapshot(root, path, loc['schema'], max_bytes=MAX_BYTES - total_bytes)
            total_bytes += cache[path].size_bytes
        snap = cache[path]
        identity = canonical_json_sha256(loc)
        if identity in verified:
            continue
        if snap.artifact_sha256 != loc['artifact_sha256'] or snap.locator(loc['selector']) != loc:
            raise ValueError('source_locator identity drift')
        verified.add(identity)
    return dict(status='verified' if locators else 'not_assessed', verified_count=len(locators),
                unique_locator_count=len(verified), artifact_count=len(cache), bytes_read=total_bytes,
                scope='source_object_identity_only', interpretation_accuracy='not_assessed')


def audit_fact_pack_locators(root: Path, fact_pack: dict) -> dict:
    locators = []
    missing = []
    frame_refs = []
    for section in fact_pack.get('sections') or []:
        for ref in section.get('evidence_refs') or []:
            if ref.get('member_evidence_ids'):
                continue
            frame_refs.append(ref)
            if 'source_locators' in ref:
                values = ref['source_locators']
                if not isinstance(values, list) or len(values) > MAX_LOCATORS - len(locators):
                    raise ValueError('source_locator count budget exceeded')
                locators.extend(values)
                if not values:
                    missing.append(ref.get('evidence_id'))
            else:
                missing.append(ref.get('evidence_id'))
    report = verify_evidence_locators(root, locators)
    from .courseware_structure import audit_frame_regions
    report['frame_regions'] = audit_frame_regions(root, frame_refs)
    report['unlocated_evidence_ids'] = sorted(set(str(v) for v in missing))
    if locators and missing:
        report['status'] = 'partially_verified'
    return report
