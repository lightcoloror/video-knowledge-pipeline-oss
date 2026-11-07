from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .file_hash import sha256_file as _sha256
from .model_task_gateway import model_task_api_call
from .models import now_iso
from .storage import read_json, write_json
from .visual_structure import run_visual_structure_plan


SCHEMA = "video_knowledge_pipeline.ocr_route.v1"
BACKENDS = {"local", "online"}
LOCAL_BACKEND = "ebook_markdown_pipeline"
ONLINE_TASK = "online_ocr"


def run_ocr_route(
    bundle_dir: str | Path,
    *,
    backend: str = "local",
    execute_local: bool = False,
    connector_result_json: str | Path | None = None,
    provider_config: dict[str, Any] | None = None,
    indexes: list[int] | None = None,
    limit: int = 8,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Select local OCR or plan/import consent-gated online OCR.

    Online execution is deliberately absent here. The trusted connector owns
    the network call; this route only prepares exact artifacts and imports its
    audited result.
    """

    root = Path(bundle_dir).expanduser().resolve()
    selected = str(backend or "local").strip().lower()
    if selected not in BACKENDS:
        raise ValueError(f"unsupported OCR backend: {backend}; expected local or online")
    if selected == "local":
        local = run_visual_structure_plan(
            root,
            execute_ebook_pipeline=bool(execute_local),
            indexes=indexes,
            limit=limit or None,
            timeout_seconds=timeout_seconds,
        )
        return _local_result(root, local, execute_local=execute_local)

    preview = run_visual_structure_plan(
        root,
        execute_ebook_pipeline=False,
        indexes=indexes,
        limit=limit or None,
        timeout_seconds=timeout_seconds,
    )
    candidates = [row for row in preview.get("items") or [] if isinstance(row, dict)]
    image_paths = _candidate_image_paths(candidates)
    instructions = _online_instructions(candidates)
    request = model_task_api_call(
        ONLINE_TASK,
        provider_config=provider_config,
        prompt=instructions,
        image_paths=image_paths[:1],
        execute=False,
        write=False,
    )
    consent_request = {
        "task": ONLINE_TASK,
        "root_dir": str(root),
        "artifact_paths": image_paths,
        "instructions": instructions,
        "purpose": "OCR selected lecture frames into Markdown and structured text evidence",
        "max_calls": len(image_paths),
        "confirm_data_export": False,
        "provider_config_runtime_only": True,
    }
    request_path = root / "online-ocr-consent-request.json"
    write_json(request_path, consent_request)

    imported: dict[str, Any] = {}
    import_candidates = candidates
    imported_rows: list[dict[str, Any]] = []
    import_path = root / "online-ocr-import.json"
    status = "planned_requires_consent"
    if connector_result_json:
        import_candidates = _connector_import_candidates(
            connector_result_json,
            candidates=candidates,
            indexes=indexes,
        )
        imported_rows = normalize_online_ocr_result(
            connector_result_json,
            candidates=import_candidates,
        )
        write_json(import_path, {"schema": SCHEMA, "items": imported_rows})
        imported = run_visual_structure_plan(
            root,
            input_json=import_path,
            indexes=indexes,
            limit=limit or None,
            timeout_seconds=timeout_seconds,
        )
        status = "imported" if imported_rows else "empty_connector_result"

    result = {
        "schema": SCHEMA,
        "backend": "online",
        "backend_id": ONLINE_TASK,
        "status": status if image_paths or connector_result_json else "no_candidates",
        "bundle_dir": str(root),
        "candidate_count": len(candidates),
        "import_candidate_count": len(import_candidates),
        "imported_row_count": len(imported_rows),
        "image_count": len(image_paths),
        "image_paths": image_paths,
        "request_plan": request.get("request_plan") or {},
        "consent_request": consent_request,
        "connector_result_json": str(Path(connector_result_json).expanduser().resolve()) if connector_result_json else "",
        "import": imported,
        "artifacts": {
            "consent_request_json": str(request_path),
            "import_json": str(import_path) if connector_result_json else "",
        },
        "downstream": {
            "timeline_fields": ["visual_text", "structured_visual"],
            "entity_lexicon": "build-entity-lexicon",
            "adaptive_asr_context": "adaptive-asr-route",
        },
        "operator_boundary": {
            "online_execution_exposed_here": False,
            "trusted_connector_required": True,
            "exact_artifact_hashes_required": True,
            "consent_and_call_limit_required": True,
            "trusted_destination_required": True,
            "provider_credentials_runtime_only": True,
            "one_artifact_per_counted_call": True,
        },
        "updated_at": now_iso(),
    }
    write_json(root / "ocr-route.json", result)
    (root / "ocr-route.md").write_text(_render(result), encoding="utf-8")
    return result


def normalize_online_ocr_result(
    connector_result_json: str | Path,
    *,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = read_json(Path(connector_result_json).expanduser().resolve())
    if not isinstance(payload, dict):
        raise ValueError("online OCR connector result must be a JSON object")
    task = str(payload.get("task") or "")
    if task and task != ONLINE_TASK:
        raise ValueError(f"connector task mismatch: expected {ONLINE_TASK}, got {task}")
    nested_result = payload.get("model_result")
    model_result: dict[str, Any] = nested_result if isinstance(nested_result, dict) else payload
    model_status = str(model_result.get("status") or payload.get("status") or "")
    if not bool(model_result.get("ok", payload.get("ok", False))) and model_status != "partial_ocr_failure":
        raise ValueError(f"online OCR connector result is not successful: {model_status}")
    parsed = _parse_content(model_result.get("content"))
    pages = _pages(parsed)
    by_index: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_index = _integer(candidate.get("index"))
        if candidate_index is not None:
            by_index[candidate_index] = candidate
    by_name = {
        Path(path).name.casefold(): row
        for row in candidates
        for path in _row_image_paths(row)
    }
    rows: list[dict[str, Any]] = []
    for position, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        candidate = _match_candidate(page, position=position, candidates=candidates, by_index=by_index, by_name=by_name)
        index = _integer(candidate.get("index")) or _integer(page.get("timeline_index")) or _integer(page.get("index"))
        markdown = str(page.get("markdown") or page.get("visual_text") or page.get("text") or "").strip()
        if index is None or not markdown:
            continue
        image_path = _first_path(page.get("image_path"), *_row_image_paths(candidate))
        rows.append(
            {
                "index": index,
                "type": str(page.get("type") or "document_visual"),
                "source": "online_ocr",
                "visual_text": markdown,
                "markdown": markdown,
                "structured_visual": page.get("structured_visual") or [
                    {
                        "type": "ocr_page",
                        "page_index": _integer(page.get("index")),
                        "dimensions": page.get("dimensions") or {},
                    }
                ],
                "confidence": page.get("confidence"),
                "uncertainties": page.get("uncertainties") or [],
                "image_path": image_path,
                "evidence_paths": [image_path] if image_path else [],
                "source_artifact_sha256": str(page.get("source_artifact_sha256") or ""),
                "evidence_status": "candidate",
            }
        )
    return rows


def _connector_import_candidates(
    connector_result_json: str | Path,
    *,
    candidates: list[dict[str, Any]],
    indexes: list[int] | None,
) -> list[dict[str, Any]]:
    """Bind an audited connector result back to exact Timeline indexes.

    Already-reviewed Timeline rows can be absent from the current unresolved OCR
    candidate set. In that case an explicit index list may restore the mapping,
    but only when it has the same cardinality as the connector artifact list and
    every artifact still matches the consent upload manifest.
    """

    payload = read_json(Path(connector_result_json).expanduser().resolve())
    if not isinstance(payload, dict):
        raise ValueError("online OCR connector result must be a JSON object")
    artifact_paths = [
        str(value or "").strip()
        for value in payload.get("artifact_paths") or []
        if str(value or "").strip()
    ]
    if not artifact_paths:
        raise ValueError("online OCR connector result is missing audited artifact_paths")
    manifest = payload.get("upload_manifest") if isinstance(payload.get("upload_manifest"), dict) else {}
    manifest_rows = [row for row in manifest.get("files") or [] if isinstance(row, dict)]
    manifest_by_path = {
        str(Path(str(row.get("path") or "")).expanduser().resolve()).casefold(): row
        for row in manifest_rows
        if str(row.get("path") or "").strip()
    }
    requested_indexes = [int(value) for value in indexes or []]
    if requested_indexes and len(requested_indexes) != len(artifact_paths):
        raise ValueError("explicit OCR import indexes must match the connector artifact count")
    candidate_by_path = {
        str(Path(path).expanduser().resolve()).casefold(): row
        for row in candidates
        for path in _row_image_paths(row)
        if Path(path).expanduser().is_file()
    }
    mapped: list[dict[str, Any]] = []
    for position, value in enumerate(artifact_paths):
        path = Path(value).expanduser().resolve()
        key = str(path).casefold()
        manifest_row = manifest_by_path.get(key)
        if manifest_row is None:
            raise ValueError(f"OCR connector artifact is absent from upload manifest: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"OCR connector artifact no longer exists: {path}")
        if path.stat().st_size != int(manifest_row.get("bytes") or -1):
            raise ValueError(f"OCR connector artifact size changed after execution: {path}")
        if _sha256(path) != str(manifest_row.get("sha256") or "").lower():
            raise ValueError(f"OCR connector artifact hash changed after execution: {path}")
        candidate = candidate_by_path.get(key)
        if candidate is not None:
            mapped.append(dict(candidate))
            continue
        if not requested_indexes:
            raise ValueError(
                "OCR connector artifact is not in the current candidate set; "
                "pass exact --indexes to import an already-reviewed Timeline row"
            )
        mapped.append({"index": requested_indexes[position], "image_path": str(path)})
    return mapped


def _local_result(root: Path, local: dict[str, Any], *, execute_local: bool) -> dict[str, Any]:
    result = {
        "schema": SCHEMA,
        "backend": "local",
        "backend_id": LOCAL_BACKEND,
        "status": "completed" if execute_local else "planned",
        "bundle_dir": str(root),
        "local_result": local,
        "downstream": {
            "timeline_fields": ["visual_text", "structured_visual"],
            "entity_lexicon": "build-entity-lexicon",
            "adaptive_asr_context": "adaptive-asr-route",
        },
        "operator_boundary": {"network_call": False, "media_uploaded": False},
        "updated_at": now_iso(),
    }
    write_json(root / "ocr-route.json", result)
    (root / "ocr-route.md").write_text(_render(result), encoding="utf-8")
    return result


def _candidate_image_paths(candidates: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for row in candidates:
        for value in _row_image_paths(row):
            path = Path(value).expanduser().resolve()
            key = str(path).casefold()
            if not path.is_file() or key in seen:
                continue
            seen.add(key)
            paths.append(str(path))
            break
    return paths


def _row_image_paths(row: dict[str, Any]) -> list[str]:
    values = [
        row.get("image_path"),
        row.get("frame_path"),
        row.get("asset_path"),
        *((row.get("evidence_paths") or []) if isinstance(row.get("evidence_paths"), list) else []),
    ]
    return [str(value) for value in values if str(value or "").strip()]


def _online_instructions(candidates: list[dict[str, Any]]) -> str:
    mapping = [
        f"- index={row.get('index')}; image={Path(paths[0]).name}"
        for row in candidates
        if (paths := _row_image_paths(row))
    ]
    return "\n".join(
        [
            "对每张课件/PPT截图做高保真 OCR，并输出严格 JSON。",
            "返回格式：{\"pages\":[{\"index\":整数,\"image_path\":\"文件名\",\"markdown\":\"...\",\"confidence\":0到1,\"uncertainties\":[]}]}。",
            "保留标题、段落、列表、表格、数字、公式、代码与阅读顺序；不可臆造看不清的字。",
            "这些 OCR 结果只是证据候选，不能直接改写 ASR 原文。",
            "输入映射：",
            *mapping,
        ]
    )


def _parse_content(content: Any) -> Any:
    if isinstance(content, (dict, list)):
        return content
    text = str(content or "").strip()
    fence = chr(96) * 3
    fenced = re.fullmatch(rf"{fence}(?:json)?\s*(.*?)\s*{fence}", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"pages": [{"markdown": text}]} if text else {"pages": []}


def _pages(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        values = data.get("pages") or data.get("items") or data.get("results")
        if isinstance(values, list):
            return values
        return [data]
    return []


def _match_candidate(
    page: dict[str, Any],
    *,
    position: int,
    candidates: list[dict[str, Any]],
    by_index: dict[int, dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    index = _integer(page.get("index"))
    if index is not None and index in by_index:
        return by_index[index]
    name = Path(str(page.get("image_path") or page.get("image") or "")).name.casefold()
    if name and name in by_name:
        return by_name[name]
    return candidates[position] if position < len(candidates) else {}


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_path(*values: Any) -> str:
    texts = [str(value or "").strip() for value in values if str(value or "").strip()]
    for text in texts:
        path = Path(text).expanduser()
        if path.is_file():
            return str(path.resolve())
    return texts[0] if texts else ""


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# OCR Route",
        "",
        f"- backend: {result['backend']}",
        f"- backend id: {result['backend_id']}",
        f"- status: {result['status']}",
        "- downstream: timeline visual text -> entity lexicon -> ASR context/hotwords",
    ]
    if result["backend"] == "online":
        lines.extend(
            [
                f"- images: {result['image_count']}",
                "- execution: Trusted Model Connector only; this route never performs the network call",
                "- import: pass the audited connector execution JSON back to this route",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Choose local ebook Markdown OCR or consent-gated online OCR")
    parser.add_argument("bundle_dir")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="local")
    parser.add_argument("--execute-local", action="store_true")
    parser.add_argument("--connector-result-json", default="")
    parser.add_argument("--provider-config", default="", help="Runtime-only JSON or JSON file; never written to route artifacts")
    parser.add_argument("--indexes", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    provider_config = _provider_config_arg(args.provider_config)
    indexes = [int(value.strip()) for value in args.indexes.split(",") if value.strip()]
    result = run_ocr_route(
        args.bundle_dir,
        backend=args.backend,
        execute_local=args.execute_local,
        connector_result_json=args.connector_result_json or None,
        provider_config=provider_config,
        indexes=indexes or None,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _provider_config_arg(value: str) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    data = read_json(path.resolve()) if path.is_file() else json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("provider config must be a JSON object")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
