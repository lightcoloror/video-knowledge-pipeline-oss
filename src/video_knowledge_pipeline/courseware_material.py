"""Lossless, opt-in video courseware evidence; not a second OCR engine.

The scan journal is independent of the lecture timeline. Derived page material
never rewrites ASR or claims that detector candidates equal all displayed pages.
"""
from __future__ import annotations

import json
import math
import re
import stat
from pathlib import Path
from typing import Any
from uuid import uuid4

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic

SCAN_SCHEMA = "video_knowledge_pipeline.courseware_scan.v1"
MATERIAL_SCHEMA = "video_knowledge_pipeline.courseware_material.v1"
PAGE_SCHEMA = "video_knowledge_pipeline.courseware_page_result.v1"
BLOCK_KINDS = {"title", "body", "table", "chart", "footnote", "annotation"}


def safe_path(root: Path, value: str, *, exists: bool = True) -> Path:
    root = root.absolute()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.absolute()
    if not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError("courseware path outside bundle")
    if path.resolve() != path:
        raise ValueError("courseware symlink/junction path rejected")
    for parent in [path, *path.parents]:
        if parent.exists() or parent.is_symlink():
            attrs = parent.lstat()
            if parent.is_symlink() or getattr(attrs, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError("courseware reparse path rejected")
        if parent == root:
            break
    if exists and not path.is_file():
        raise ValueError(f"courseware artifact missing: {path.name}")
    return path


def seal(value: dict) -> dict:
    value = {key: val for key, val in value.items() if key != "state_sha256"}
    return {**value, "state_sha256": canonical_json_sha256(value)}


def store_scan(root: Path, state: dict) -> None:
    identity = canonical_json_sha256({key: state.get(key) for key in ("schema", "source", "scope", "windows", "detector")})
    state = seal({**state, "identity": identity})
    write_json(safe_path(root, "courseware/scan.json", exists=False), state)


def read_scan(root: Path) -> dict:
    state = read_json(safe_path(root, "courseware/scan.json"))
    if not isinstance(state, dict) or state.get("schema") != SCAN_SCHEMA:
        raise ValueError("invalid courseware scan schema")
    if seal(state)["state_sha256"] != state.get("state_sha256"):
        raise ValueError("courseware scan hash drift")
    identity = canonical_json_sha256({key: state.get(key) for key in ("schema", "source", "scope", "windows", "detector")})
    if state.get("identity") != identity:
        raise ValueError("courseware scan identity drift")
    return state


def _page_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"page-[a-zA-Z0-9-]{1,64}", value):
        raise ValueError("invalid courseware page_id")
    return value


def _check_frame(root: Path, candidate: dict) -> Path:
    path = safe_path(root, str(candidate.get("frame") or ""))
    if sha256_file(path) != candidate.get("frame_sha256"):
        raise ValueError("courseware frame hash drift")
    return path


def _validate_page(root: Path, state: dict, value: dict) -> dict:
    if not isinstance(value, dict) or value.get("schema") != PAGE_SCHEMA:
        raise ValueError("invalid courseware page schema")
    page_id = _page_id(value.get("page_id"))
    matches = [row for row in state["candidates"] if row["page_id"] == page_id]
    if len(matches) != 1 or value.get("scan_identity") != state["identity"]:
        raise ValueError("courseware page identity mismatch")
    candidate = matches[0]
    _check_frame(root, candidate)
    if value.get("frame_sha256") != candidate.get("frame_sha256"):
        raise ValueError("courseware result frame hash mismatch")
    time = value.get("time")
    if not isinstance(time, (float, int)) or not math.isfinite(time) or abs(time - candidate["time"]) > 0.001:
        raise ValueError("courseware page timestamp mismatch")
    frame_time = value.get("frame_time", time)
    if not isinstance(frame_time, (int, float)) or not math.isfinite(frame_time) or abs(frame_time - float(candidate.get("frame_time", candidate["time"]))) > 0.001:
        raise ValueError("courseware screenshot timestamp mismatch")
    for key in ("executor", "model_actual"):
        if not isinstance(value.get(key), str) or not value[key].strip() or len(value[key]) > 200:
            raise ValueError(f"courseware {key} required")
    if value.get("status") not in {"extracted", "unavailable"}:
        raise ValueError("invalid page status")
    blocks = value.get("blocks")
    gaps = value.get("gaps")
    if not isinstance(blocks, list) or not isinstance(gaps, list):
        raise ValueError("courseware blocks and gaps required")
    if len(blocks) > 10000 or len(gaps) > 1000:
        raise ValueError("courseware field budget exceeded")
    for block in blocks:
        if not isinstance(block, dict) or block.get("kind") not in BLOCK_KINDS or not isinstance(block.get("text"), str):
            raise ValueError("invalid raw courseware block")
    if any(not isinstance(g, str) or not g.strip() or len(g) > 4000 for g in gaps):
        raise ValueError("invalid courseware review gap")
    if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("courseware page budget exceeded; split into tiles")
    if value["status"] == "unavailable" and not gaps:
        raise ValueError("unavailable page needs a reason")
    from .courseware_structure import validate_native_structure
    validate_native_structure(value)
    from .courseware_structure import validate_native_frame_binding
    validate_native_frame_binding(root, value, candidate)
    if 'ocr_geometry' in value:
        from .courseware_ocr_geometry import validate_geometry
        validate_geometry(root, value, candidate)
    if "frame_visual_identity" in value:
        from .courseware_dedupe import validate_frame_visual_identity
        validate_frame_visual_identity(root, candidate, value["frame_visual_identity"])
    if "exact_frame_reuse" in value:
        from .courseware_exact_reuse import validate_exact_frame_reuse
        validate_exact_frame_reuse(root, state, value)
    if 'incremental_ocr' in value:
        from .courseware_incremental_ocr import validate_incremental_page
        validate_incremental_page(root, value)
    return candidate


def import_courseware_page(bundle_dir: str | Path, value: dict) -> dict:
    root = Path(bundle_dir).absolute()
    with bundle_write_lock(root, operation="courseware_page_import"):
        raw = safe_path(root, f"courseware/raw/{uuid4().hex}.json", exists=False)
        write_json(raw, value)
        try:
            state = read_scan(root)
            _validate_page(root, state, value)
            path = safe_path(root, f"courseware/pages/{_page_id(value['page_id'])}.json", exists=False)
            if path.exists():
                if read_json(path) == value:
                    return {"ok": True, "status": "no_op", "page_id": value["page_id"]}
                prior_hash = sha256_file(path)
                if value.get("supersedes_sha256") != prior_hash:
                    raise ValueError("existing courseware page result; explicit supersedes_sha256 required")
                history = safe_path(root, f"courseware/history/{prior_hash}.json", exists=False)
                if not history.exists():
                    write_json(history, read_json(path))
            write_json(path, value)
        except (ValueError, OSError) as exc:
            write_json(raw.with_suffix(".validation.json"), {"ok": False, "changed": False, "error": str(exc)})
            raise
        write_json(raw.with_suffix(".validation.json"), {"ok": True, "changed": True, "page_id": value["page_id"]})
    return {"ok": True, "status": "imported", "page_id": value["page_id"]}


def build_courseware_material(bundle_dir: str | Path, *, write: bool = True) -> dict:
    root = Path(bundle_dir).absolute()
    scan_path = safe_path(root, "courseware/scan.json", exists=False)
    if not scan_path.exists():
        if safe_path(root, "courseware/material.json", exists=False).exists():
            raise ValueError("courseware scan receipt missing; cannot silently disable existing channel")
        return {"active": False}
    from .visual_structure import _ebook_markdown_quality

    state = read_scan(root)
    pages, gaps = [], []
    ids = set()
    dependencies = [{"path": "courseware/scan.json", "sha256": sha256_file(scan_path)}]
    material_chars = 0
    for candidate in sorted(state["candidates"], key=lambda row: row["time"]):
        page_id = _page_id(candidate["page_id"])
        if page_id in ids:
            raise ValueError("duplicate courseware candidate")
        ids.add(page_id)
        page = {**candidate, "blocks": [], "status": "not_extracted", "gaps": [], "source_kind": "video_displayed_courseware"}
        if candidate.get("frame"):
            frame = _check_frame(root, candidate)
            dependencies.append({"path": candidate["frame"], "sha256": sha256_file(frame)})
        else:
            page["gaps"].append("native screenshot not captured")
        receipt_path = safe_path(root, f"courseware/pages/{page_id}.json", exists=False)
        if receipt_path.exists():
            from .evidence_locator import JsonEvidenceSnapshot
            source_snapshot = JsonEvidenceSnapshot.read(root, receipt_path.relative_to(root).as_posix())
            value = source_snapshot.data
            _validate_page(root, state, value)
            page.update({key: value[key] for key in ("gaps", "status", "executor", "model_actual")})
            # Never modify the raw object used to compute selected-object identity.
            page["blocks"] = [{**block, "source_locators": [source_snapshot.locator(["blocks", number])]}
                              for number, block in enumerate(value["blocks"])]
            from .courseware_structure import project_native_table
            for number, block in enumerate(page["blocks"]):
                block.update(project_native_table(value, block, number, source_snapshot))
                if 'ocr_geometry' in value:
                    from .courseware_ocr_geometry import project_ocr_block
                    block.update(project_ocr_block(value, block, number, source_snapshot))
            if 'region_evidence' in value and 'incremental_ocr' in value:
                from .courseware_incremental_ocr import project_incremental_blocks
                incremental = project_incremental_blocks(root, value)
                dependencies.append(incremental['dependency'])
                for block, projection in zip(page['blocks'], incremental['blocks']):
                    block['source_locators'].extend(projection['source_locators'])
                    block.update({k:v for k,v in projection.items() if k != 'source_locators'})
            if "exact_frame_reuse" in value:
                page["exact_frame_reuse"] = value["exact_frame_reuse"]
                source_path = safe_path(root, f"courseware/pages/{value['exact_frame_reuse']['source_page_id']}.json")
                dependencies.append({"path": source_path.relative_to(root).as_posix(), "sha256": sha256_file(source_path)})
            text = "\n\n".join(_block_content(block) for block in page["blocks"])
            material_chars += len(text)
            if material_chars > 32 * 1024 * 1024:
                raise ValueError("courseware material budget exceeded; use explicit bounded scopes, never truncate")
            quality = _ebook_markdown_quality(text, image_path=candidate.get("frame"))
            if quality["quality"] != "usable":
                page["gaps"] = [*page["gaps"], f"OCR {quality['quality']}: {quality['reason']}"]
                page["status"] = "unavailable"
            dependencies.append({"path": receipt_path.relative_to(root).as_posix(), "sha256": source_snapshot.artifact_sha256})
        else:
            page["gaps"].append("page extraction not imported")
        for number, block in enumerate(page["blocks"]):
            block["evidence_id"] = f"courseware:{page_id}:{number}"
        pages.append(page)
        gaps.extend({"page_id": page_id, "time": page["time"], "reason": reason} for reason in page["gaps"])
    scanned = set(state["completed_windows"]) == set(range(len(state["windows"]))) and bool(state["windows"]) and not state.get("failure")
    extracted = bool(pages) and all(p["status"] == "extracted" and not p["gaps"] for p in pages)
    if not scanned:
        gaps.append({"reason": "full-scope detector scan not completed"})
    scope_path = safe_path(root, "content-scope.json", exists=False)
    if state.get("content_scope_sha256") != (sha256_file(scope_path) if scope_path.exists() else None):
        raise ValueError("courseware content scope drift; prepare a new scan")
    payload = {"schema": MATERIAL_SCHEMA, "active": True, "scan_identity": state["identity"],
               "source_sha256": state["source"]["sha256"], "scope": state["scope"], "pages": pages,
               "dependencies": dependencies, "scan_complete": scanned, "extraction_complete": extracted,
               "review_gaps": gaps, "candidate_count": len(pages), "total_page_count": None,
               "source_kind": "video_displayed_courseware", "external_courseware_is_separate": True,
               "raw_asr_modified": False, "evidence_level": "source_observation_not_external_fact"}
    payload["revision"] = canonical_json_sha256(payload)
    review_path = safe_path(root, "courseware/independent-review.json", exists=False)
    reviewed = False
    if review_path.exists():
        review = read_json(review_path)
        reviewed = review.get("material_revision") == payload["revision"] and review.get("scope_reviewed") is True and set(review.get("page_ids", [])) == ids and not review.get("gaps")
    payload["independent_review_complete"] = reviewed
    payload["independent_review_sha256"] = sha256_file(review_path) if review_path.exists() else None
    payload["complete"] = scanned and extracted and reviewed
    payload["limitations"] = ["Detector candidate count is not the total number of displayed pages.",
                              "Small changes, occlusions and flashes require independent full-scope review.",
                              "Original slide claims may conflict with later speech; preserve both with their times."]
    if write:
        with bundle_write_lock(root, operation="courseware_material_projection"):
            write_json(safe_path(root, "courseware/material.json", exists=False), payload)
            write_text_atomic(safe_path(root, "courseware/material.md", exists=False), render_courseware_material(payload))
            from .run_artifact_registry import register_bundle_run
            register_bundle_run(root, run_type="courseware_material", run_id="courseware-material",
                                status="completed" if payload["complete"] else "needs_review",
                                title="Full video-displayed courseware material",
                                summary=f"candidates={len(pages)}; scan={scanned}; extracted={extracted}; independently_reviewed={reviewed}",
                                inputs={"scan": str(scan_path)}, parameters={"lossless": True},
                                artifacts=[{"key": "courseware_material", "path": "courseware/material.json"},
                                           {"key": "courseware_material_markdown", "path": "courseware/material.md"}],
                                failed_items=[{"id": str(n), "reason": gap["reason"]} for n, gap in enumerate(gaps)],
                                next_actions=["Review complete scan scope and individual pages; candidate count is not total page count."],
                                operator_boundary={"source_media_read_only": True, "no_provider_call": True, "no_asr_overwrite": True}, write=True)
    return payload


def import_courseware_review(bundle_dir: str | Path, value: dict) -> dict:
    root = Path(bundle_dir).absolute()
    with bundle_write_lock(root, operation="courseware_independent_review"):
        material = build_courseware_material(root, write=False)
        if value.get("material_revision") != material.get("revision"):
            raise ValueError("independent review material revision mismatch")
        if not value.get("reviewer") or not value.get("method") or not isinstance(value.get("gaps"), list):
            raise ValueError("reviewer, independent method and gaps required")
        ids = value.get("page_ids")
        expected = {p["page_id"] for p in material["pages"]}
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError("independent review must cover exact page set")
        if value.get("scope_reviewed") is not True or not material["scan_complete"]:
            raise ValueError("full-scope independent review required")
        path = safe_path(root, "courseware/independent-review.json", exists=False)
        if path.exists() and read_json(path) != value:
            prior_hash = sha256_file(path)
            if value.get("supersedes_sha256") != prior_hash:
                raise ValueError("existing independent review; explicit supersedes_sha256 required")
            history = safe_path(root, f"courseware/history/{prior_hash}.json", exists=False)
            if not history.exists():
                write_json(history, read_json(path))
        write_json(path, value)
    return {"ok": True, "status": "review_recorded"}


def material_for_range(material: dict, start: float, end: float) -> dict:
    if not material.get("active"):
        return {"active": False}
    return {**material, "pages": [p for p in material["pages"] if p["time"] < end and p.get("end", p["time"] + 0.001) > start]}


def render_courseware_material(material: dict) -> str:
    if not material.get("active"):
        return ""
    lines = ["# Full video-displayed courseware material", "",
             f"Revision: {material['revision']}",
             f"Scan={material['scan_complete']}; extraction={material['extraction_complete']}; independent_review={material['independent_review_complete']}",
             "Raw slide text only. Keep speech, later corrections and AI synthesis separate. Do not treat slide claims as external facts."]
    for page in material["pages"]:
        lines.extend(["", f"## {page['page_id']} @ {page['time']:.3f}s", f"Frame: [{page.get('frame', '')}](../{page.get('frame', '')}); requested capture time={page.get('frame_time', page['time'])}; SHA256: {page.get('frame_sha256', '')}"])
        for block in page["blocks"]:
            lines.extend([f"### {block['kind']} [{block['evidence_id']}]", _block_content(block)])
        lines.extend(f"REVIEW GAP: {g}" for g in page["gaps"])
    lines.extend(["", "## Scope/review gaps", json.dumps(material["review_gaps"], ensure_ascii=False)])
    return "\n\n".join(lines) + "\n"


def _block_content(block: dict) -> str:
    structured = {key: value for key, value in block.items() if key not in {"kind", "text", "evidence_id", "source_locators", "structural_evidence", "ocr_evidence_context", "recognition_provenance", "incremental_region_context"}}
    return block["text"] + ("\n" + json.dumps(structured, ensure_ascii=False) if structured else "")


def courseware_prompt(material: dict, *, max_chars: int) -> str:
    text = render_courseware_model_input(material)
    if len(text) > max_chars:
        raise ValueError(f"courseware_input_budget: requires {len(text)} chars; budget {max_chars}; split sections or increase explicit budget")
    return text


def render_courseware_model_input(material: dict, *, deduplicate: bool = True) -> str:
    """Keep all observed content; reference the full, hash-bound gap ledger.

    Missing-page bookkeeping is not slide content. It stays lossless in the
    audit artifact instead of consuming the model budget thousands of times.
    """
    if not material.get("active"):
        return ""
    pages = material["pages"]
    observed = [page for page in pages if page.get("blocks")]
    gap_receipt = {
        "review_gap_count": len(material["review_gaps"]),
        "review_gaps_sha256": canonical_json_sha256(material["review_gaps"]),
        "unobserved_candidate_count": len(pages) - len(observed),
        "candidate_count_in_range": len(pages),
        "full_audit_artifact": "courseware/material.json",
        "material_revision": material["revision"],
        "complete": material["complete"],
        "scan_complete": material["scan_complete"],
        "extraction_complete": material["extraction_complete"],
        "independent_review_complete": material["independent_review_complete"],
    }
    lines = ["# Video-displayed courseware: full observed text, incomplete evidence stays incomplete",
             "Raw slide text only. Keep speech, later corrections and AI synthesis separate. "
             "Do not treat slide claims as external facts. Unobserved candidates are NOT extracted; "
             "the gap receipt is not content and cannot support a comprehensive-completion claim.",
             json.dumps(gap_receipt, ensure_ascii=False)]
    seen_blocks = {}
    for page in observed:
        lines.extend([f"## {page['page_id']} @ {page['time']:.3f}s–{page['end']:.3f}s",
                      f"Frame SHA256: {page.get('frame_sha256', '')}; capture time={page.get('frame_time', page['time'])}"])
        for block in page["blocks"]:
            # Frame-specific structure/provenance is retained independently;
            # it must not prevent reuse of the exact same textual payload.
            content = block['text']
            identity = canonical_json_sha256({"kind": block["kind"], "content": content})
            lines.append(f"### {block['kind']} [{block['evidence_id']}]")
            reference = (f"EXACT TEXT REUSE: [{seen_blocks.get(identity, '')}]; content SHA256={identity}. "
                         "Same observed text at this occurrence, not proof of identical visuals.")
            if deduplicate and identity in seen_blocks and len(reference) < len(content):
                lines.append(reference)
            else:
                lines.append(content)
                seen_blocks.setdefault(identity, block['evidence_id'])
            structured = {k: v for k, v in block.items() if k not in {'kind', 'text', 'evidence_id', 'source_locators', 'structural_evidence', 'ocr_evidence_context'}}
            if structured:
                lines.append(json.dumps(structured, ensure_ascii=False))
        lines.extend(f"REVIEW GAP: {gap}" for gap in page["gaps"])
    return "\n\n".join(lines) + "\n"


def courseware_reading_signature(blocks: list[dict]) -> list[dict]:
    """Whitespace-only review grouping, never an OCR or semantic equality proof."""
    return [{'kind': b['kind'],
             'text': (' '.join(b['text'].split()) if b['kind'] in {'title', 'body', 'footnote'}
                      and not any(marker in b['text'] for marker in ('|', '```', '\t'))
                      else b['text'])} for b in blocks]


def build_courseware_groups(bundle_dir: str | Path, *, write=True) -> dict:
    """Exact observed-text grouping only: no inferred visual/page equivalence."""
    root = Path(bundle_dir).absolute()

    def project() -> dict:
        material = build_courseware_material(root, write=False)
        if not material.get("active"):
            return {"active": False}
        groups = {}
        reading_groups = {}
        unobserved = []
        for index, page in enumerate(material["pages"], 1):
            if not page.get("blocks"):
                unobserved.append({"index": index, "page_id": page["page_id"], "time": page["time"]})
                continue
            blocks = [{k: v for k, v in b.items() if k not in {"evidence_id", "source_locators", "structural_evidence", "ocr_evidence_context", "recognition_provenance", "incremental_region_context"}} for b in page["blocks"]]
            digest = canonical_json_sha256(blocks)
            group = groups.setdefault(digest, {"text_sha256": digest, "blocks": blocks, "members": []})
            group["members"].append({"index": index, "page_id": page["page_id"], "time": page["time"],
                                     "end": page["end"], "frame_sha256": page.get("frame_sha256"),
                                     "evidence_refs": [b["evidence_id"] for b in page["blocks"]],
                                     "source_locators": [loc for b in page["blocks"] for loc in b.get("source_locators", [])],
                                     "gaps": page["gaps"]})
            # Reading projection only; original blocks and per-occurrence evidence
            # remain authoritative. Keep digits, punctuation, order and block kind.
            signature = courseware_reading_signature(blocks)
            reading_key = canonical_json_sha256(signature)
            reading = reading_groups.setdefault(reading_key, {
                "reading_sha256": reading_key, "representative_page_id": page["page_id"],
                "members": [], "proves_visual_equivalence": False,
                "proves_structure_equivalence": False,
                "allows_ocr_skip": False})
            reading["members"].append(group["members"][-1])
        value = {"schema": "video_knowledge_pipeline.courseware_text_groups.v1", "active": True,
                 "material_revision": material["revision"], "candidate_count": material["candidate_count"],
                 "observed_text_groups": list(groups.values()), "observed_text_group_count": len(groups),
                 "reading_groups": list(reading_groups.values()),
                 "reading_group_count": len(reading_groups),
                 "reading_duplicate_occurrence_count": sum(len(g["members"])-1 for g in reading_groups.values()),
                 "unobserved_candidates": unobserved, "unobserved_candidate_count": len(unobserved),
                 "proves_visual_equivalence": False, "extraction_complete": material["extraction_complete"],
                 "complete": material["complete"], "limitations": [
                     "Equal OCR text does not prove equal slides, tables, diagrams or annotations.",
                     "All candidate occurrences remain; unseen candidates must still be extracted/reviewed."]}
        value["sha256"] = canonical_json_sha256(value)
        return value

    if not write:
        return project()
    with bundle_write_lock(root, operation="courseware_text_groups"):
        value = project()
        write_json(safe_path(root, "courseware/groups.json", exists=False), value)
        return value


def assert_material_fresh(root: Path, recorded: dict) -> dict:
    current = build_courseware_material(root, write=False)
    if bool(current.get("active")) != bool(recorded.get("active")):
        raise ValueError("courseware_material_stale: channel added or removed")
    if current.get("active") and any(recorded.get(key) != current[key] for key in ("revision", "independent_review_complete", "independent_review_sha256")):
        raise ValueError("courseware_material_stale: rebuild input pack, chapters and section workflow")
    return current


def courseware_summary_gate(root: Path, summary_path: Path | None = None) -> dict:
    """Formal quality needs a fresh consumption receipt, not merely a material file."""
    try:
        material = build_courseware_material(root, write=False)
        if not material.get("active"):
            return {"passed": True, "detail": "courseware channel not configured"}
        path = root / "exports/smart-summary-section-llm-revisions.json"
        revisions = read_json(path) if path.is_file() else {}
        assert_material_fresh(root, revisions.get("courseware_material", {}))
        rows = revisions.get("rows", [])
        consumed = bool(rows) and all(row.get("courseware_revision") == material["revision"] for row in rows)
        workflow = read_json(safe_path(root, "exports/smart-summary-section-workflow.json"))
        assert_material_fresh(root, workflow.get("courseware_material", {}))
        expected = {row["section_id"] for row in workflow.get("sections", [])}
        consumed = consumed and bool(expected) and len(rows) == len(expected) and {row.get("section_id") for row in rows} == expected
        summary = summary_path or root / "exports/smart-summary.md"
        consumed = consumed and summary.is_file() and summary.stat().st_mtime_ns >= path.stat().st_mtime_ns
        return {"passed": bool(material["complete"] and consumed),
                "detail": f"scan={material['scan_complete']}; extraction={material['extraction_complete']}; independent_review={material['independent_review_complete']}; section_consumption={consumed}"}
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return {"passed": False, "detail": str(exc)}
