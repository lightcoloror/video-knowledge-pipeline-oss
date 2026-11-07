from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

from .models import now_iso
from .path_defaults import local_model_root, source_reviews_root, tool_source_review_root
from .storage import bundle_write_lock, read_json, write_json
from .tagger_import import import_tagger_annotations
from .file_hash import sha256_file
from .stage_cache import StageCache
from .canonical_json import canonical_json_sha256
from .courseware_material import safe_path


SCHEMA = "video_knowledge_pipeline.general_tagger.v1"
STATUS_SCHEMA = "video_knowledge_pipeline.general_tagger_status.v1"
MODEL = "ram_plus_swin_large_14m"
IMPLEMENTATION = "recognize_anything_ram_plus"
RECOGNITION_CONTRACT = 'video_knowledge_pipeline.tagger_recognition_binding.v1'
DEFAULT_SOURCE_ROOT = tool_source_review_root() / "recognize-anything"
DEFAULT_CHECKPOINT = local_model_root() / "recognize-anything" / "ram_plus_swin_large_14m.pth"
DEFAULT_TOKENIZER_ROOT = local_model_root() / "bert-base-uncased-tokenizer"
DEFAULT_THRESHOLD_FLOOR = 0.75
COMPATIBILITY_BASELINES = ["cl_tagger", "wd_eva02_large_tagger_v3"]


def tagger_input_hashes(root):
    manifest = safe_path(root,'manifest.json',exists=False)
    data = read_json(manifest) if manifest.is_file() else None
    # Exclude only the four bookkeeping fields written by this producer.
    # Source identity/policy/title changes remain binding failures.
    if isinstance(data,dict):
        data = {k:v for k,v in data.items() if k not in {
            'general_tagger_json','general_tagger_markdown','mcp_general_tagger_args','general_tagger_summary'}}
    return {'manifest_source_projection_sha256':canonical_json_sha256(data),
            'timeline_sha256':sha256_file(safe_path(root,'timeline.json'))}


def validate_tagger_recognition(root, row, *, expected_index=None, check_bundle_inputs=True):
    """Validate new producer receipts; legacy external annotations stay unbound."""
    binding = row.get('recognition_binding')
    if binding is None:
        return
    if not isinstance(binding,dict) or binding.get('schema') != RECOGNITION_CONTRACT:
        raise ValueError('tagger recognition binding invalid')
    if check_bundle_inputs and binding.get('input_hashes') != tagger_input_hashes(root):
        raise ValueError('tagger input identity drift')
    frame = safe_path(root, binding.get('frame_path',''))
    if (sha256_file(frame) != binding.get('frame_sha256') or row.get('artifact_sha256') != binding.get('frame_sha256')
            or safe_path(root,row.get('artifact_path','')) != frame):
        raise ValueError('tagger frame identity drift')
    index = expected_index if expected_index is not None else row.get('timeline_index',row.get('index'))
    if index != binding.get('timeline_index'):
        raise ValueError('tagger target identity drift')
    for key in ('start','end','timestamp','frame_id'):
        if row.get(key) != binding.get(key):
            raise ValueError('tagger timestamp identity drift')
    if row.get('text') != '、'.join(row.get('tags') or []):
        raise ValueError('tagger selected text identity drift')
    from .evidence_lineage import project_recognition_provenance
    record = row.get('recognition_provenance')
    expected = project_recognition_provenance({'recognition_provenance':{
        'executor':row.get('executor',''),'model_actual':row.get('model_actual',''),
        'frame_sha256s':[binding['frame_sha256']]}}, selected_field=['tags'], text=row.get('text',''))
    if record != expected:
        raise ValueError('tagger recognition provenance drift')


def general_tagger_status(
    *,
    source_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    source_inventory_path: str | Path | None = None,
) -> dict[str, Any]:
    discovery = _inventory_model_discovery(source_inventory_path)
    source = _source_root(source_root, discovery=discovery)
    checkpoint = _checkpoint(checkpoint_path, discovery=discovery)
    tokenizer = _tokenizer_root(discovery=discovery)
    blockers: list[str] = []
    if source is None:
        blockers.append("recognize_anything_source_missing")
    if checkpoint is None:
        blockers.append("ram_plus_checkpoint_missing")
    if tokenizer is None:
        blockers.append("bert_tokenizer_missing")
    return {
        "schema": STATUS_SCHEMA,
        "status": "ready" if not blockers else "needs_setup",
        "selected_model": MODEL,
        "implementation": IMPLEMENTATION,
        "domain": "general_real_world_images",
        "languages": ["zh", "en"],
        "source_root": str(source or ""),
        "checkpoint_path": str(checkpoint or ""),
        "tokenizer_root": str(tokenizer or ""),
        "model_discovery": discovery,
        "threshold_floor": DEFAULT_THRESHOLD_FLOOR,
        "blockers": blockers,
        "compatibility_baselines": list(COMPATIBILITY_BASELINES),
        "automatic_remote_fallback": False,
        "automatic_model_download": False,
        "device_policy": {
            "gpu_required": True,
            "default_device": "cuda",
            "cpu_fallback_allowed": False,
        },
        "updated_at": now_iso(),
    }


def run_general_tagger(
    bundle_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
    prefer_language: str = "zh",
    limit: int = 0,
    frame_mode: str = "representative",
    execute: bool = False,
    import_annotations: bool = True,
    write: bool = True,
    timeline_indexes: list[int] | tuple[int, ...] | set[int] | None = None,
    per_timeline_limit: int = 0,
    selected_artifact_paths: list[str] | tuple[str, ...] | None = None,
    cache_dir: str | Path | None = None,
    run_id: str = "",
    _inference_backend: Callable[[Path], dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Run RAM++ over bundle keyframes and import candidate tag evidence."""

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if prefer_language not in {"zh", "en"}:
        raise ValueError("prefer_language must be zh or en")
    if frame_mode not in {"representative", "continuous"}:
        raise ValueError("frame_mode must be representative or continuous")
    root = Path(bundle_dir).expanduser().resolve()
    timeline_path = root / "timeline.json"
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")
    timeline = read_json(timeline_path)
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must be an array")

    setup = general_tagger_status(source_root=source_root, checkpoint_path=checkpoint_path)
    source = _source_root(source_root)
    checkpoint = _checkpoint(checkpoint_path)
    frames = bundle_frame_records(
        root,
        timeline,
        frame_mode=frame_mode,
        timeline_indexes=timeline_indexes,
        per_timeline_limit=per_timeline_limit,
    )
    if selected_artifact_paths is not None:
        by_path = {str(row["frame_path"]).casefold(): row for row in frames}
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for raw_path in selected_artifact_paths:
            resolved = str(Path(raw_path).expanduser().resolve())
            row = by_path.get(resolved.casefold())
            if row is None:
                missing.append(resolved)
            else:
                selected.append(row)
        if missing:
            raise ValueError(f"selected_frame_not_in_bundle: {missing[0]}")
        frames = selected
    if int(limit) > 0:
        frames = frames[: int(limit)]
    status = "planned"
    error: dict[str, Any] | None = None
    annotations: list[dict[str, Any]] = []
    backend = _inference_backend
    revision = sha256_file(checkpoint) if checkpoint is not None else "test-backend"
    cache_hits = 0
    input_hashes = tagger_input_hashes(root)

    if execute:
        blockers = list(setup.get("blockers") or [])
        if not frames:
            blockers.append("bundle_keyframes_missing")
        if blockers and backend is None:
            status = "blocked_setup"
            error = {"code": "general_tagger_runtime_not_ready", "blockers": sorted(set(blockers))}
        else:
            try:
                for frame in frames:
                    frame_path = safe_path(root,str(frame['frame_path']))
                    frame_sha256 = sha256_file(frame_path)
                    cache = (
                        StageCache(
                            Path(cache_dir) / str(frame["frame_id"]),
                            frame["frame_path"],
                        )
                        if cache_dir is not None
                        else None
                    )
                    cache_identity = canonical_json_sha256({'model':MODEL,'revision':revision,
                        'frame_sha256':frame_sha256,'frame':{k:v for k,v in frame.items() if k != 'frame_path'},
                        'language':prefer_language,'adapter_sha256':sha256_file(Path(__file__))})
                    cached = (
                        cache.load_json("continuous-tag-frame", "annotation.json", identity=cache_identity)
                        if cache is not None
                        else None
                    )
                    if isinstance(cached, dict):
                        if not isinstance(cached.get('recognition_binding'),dict):
                            raise ValueError('tagger cache recognition binding missing')
                        # The tagger sees only the frame. A valid exact-frame,
                        # model/config/time/language cache can be projected into
                        # a new Bundle checkpoint; retain its original binding.
                        validate_tagger_recognition(root,cached,check_bundle_inputs=False)
                        cached = {**cached,'recognition_origin':cached.get('recognition_origin') or cached['recognition_binding'],
                                  'recognition_binding':{**cached['recognition_binding'],'input_hashes':input_hashes}}
                        validate_tagger_recognition(root,cached)
                        annotations.append({**cached, "cache_hit": True})
                        cache_hits += 1
                        continue
                    if backend is None:
                        backend = _ram_plus_backend(source, checkpoint, device=device)
                    output = backend(frame["frame_path"])
                    if sha256_file(frame_path) != frame_sha256 or tagger_input_hashes(root) != input_hashes:
                        raise ValueError('tagger input/frame drift during recognition')
                    labels_en = _labels(output.get("tags_en"))
                    labels_zh = _labels(output.get("tags_zh"))
                    selected = labels_zh if prefer_language == "zh" and labels_zh else labels_en
                    if prefer_language == "en" and labels_en:
                        selected = labels_en
                    annotation = {
                            "schema": "video_knowledge_pipeline.tagger_annotation.v1",
                            "index": frame["timeline_index"],
                            "timeline_index": frame["timeline_index"],
                            "frame_position": frame["frame_position"],
                            "frame_id": frame["frame_id"],
                            "timestamp": frame["timestamp"],
                            "start": frame["start"],
                            "end": frame["end"],
                            "tags": selected,
                            "labels_en": labels_en,
                            "labels_zh": labels_zh,
                            "text": "、".join(selected),
                            "source": "ram_plus_general",
                            "executor": str(output.get('executor') or ('ram_plus_general' if _inference_backend is None else 'injected_backend')),
                            "model": MODEL,
                            "model_actual": str(output.get('model_actual') or (MODEL if _inference_backend is None else '')),
                            "model_revision": revision,
                            "run_id": str(run_id or ""),
                            "artifact_path": str(frame["frame_path"]),
                            "artifact_sha256": frame_sha256,
                            "confidence": output.get("confidence"),
                            "tag_confidences": output.get("tag_confidences") or {},
                            "tag_vocabulary": "ram_plus_open_set",
                            "candidate_only": True,
                            "human_review_required": True,
                            "cache_hit": False,
                        }
                    annotation['recognition_binding'] = {'schema':RECOGNITION_CONTRACT,
                        'input_hashes':input_hashes,'frame_path':frame_path.relative_to(root).as_posix(),
                        'frame_sha256':frame_sha256,
                        **{k:frame[k] for k in ('frame_id','timeline_index','timestamp','start','end')}}
                    from .evidence_lineage import project_recognition_provenance
                    annotation['recognition_provenance'] = project_recognition_provenance({'recognition_provenance':{
                        'executor':annotation['executor'],'model_actual':annotation['model_actual'],
                        'frame_sha256s':[frame_sha256]}},selected_field=['tags'],text=annotation['text'])
                    validate_tagger_recognition(root,annotation)
                    annotations.append(annotation)
                    if cache is not None:
                        cache.store_json(
                            "continuous-tag-frame",
                            "annotation.json",
                            annotation,
                            identity=cache_identity,
                        )
                for annotation in annotations:
                    validate_tagger_recognition(root,annotation)
                status = "completed"
            except Exception as exc:  # optional local model errors never trigger a remote route
                status = "failed"
                error = {"code": "general_tagger_execution_failed", "message": f"{type(exc).__name__}: {exc}"}

    result = {
        "schema": SCHEMA,
        "recognition_contract": RECOGNITION_CONTRACT,
        "status": status,
        "ok": status == "completed",
        "bundle_dir": str(root),
        "implementation": IMPLEMENTATION,
        "model": MODEL,
        "model_revision": revision,
        "source_root": str(source or ""),
        "checkpoint_path": str(checkpoint or ""),
        "device": device,
        "prefer_language": prefer_language,
        "frame_mode": frame_mode,
        "planned_frame_count": len(frames),
        "annotation_count": len(annotations),
        "cache_hit_count": cache_hits,
        "run_id": str(run_id or ""),
        "annotations": annotations,
        "error": error,
        "compatibility_baselines": list(COMPATIBILITY_BASELINES),
        "operator_boundary": {
            "local_only": True,
            "remote_calls_made": 0,
            "automatic_model_download": False,
            "automatic_remote_fallback": False,
            "gpu_required": True,
            "cpu_fallback_allowed": False,
            "candidate_only": True,
            "does_not_overwrite_ocr_asr_or_human_facts": True,
        },
        "artifacts": {
            "json": "exports/general-tagger.json",
            "markdown": "exports/general-tagger.md",
        },
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, result)
        if status == "completed" and import_annotations:
            imported = import_tagger_annotations(
                root,
                root / "exports" / "general-tagger.json",
                source="ram_plus_general",
                write=True,
            )
            result["timeline_import"] = {
                "status": "completed",
                "updated_count": imported["updated_count"],
                "updated_indexes": imported["updated_indexes"],
            }
            _write_result(root, result)
    return result


def _ram_plus_backend(
    source_root: Path | None,
    checkpoint: Path | None,
    *,
    device: str,
) -> Callable[[Path], dict[str, object]]:
    if source_root is None:
        raise FileNotFoundError("Recognize Anything source is required")
    if checkpoint is None:
        raise FileNotFoundError("RAM++ checkpoint is required")
    tokenizer_root = _tokenizer_root()
    if tokenizer_root is None:
        raise FileNotFoundError("bert-base-uncased tokenizer is required")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import torch
    from PIL import Image
    from ram import get_transform, inference_ram
    from ram.models import ram_plus

    if device == "cpu":
        raise RuntimeError("gpu_required_for_local_model")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable_gpu_required")
    resolved_device = "cuda"
    transform = get_transform(image_size=384)
    model = ram_plus(
        pretrained=str(checkpoint),
        image_size=384,
        vit="swin_l",
        text_encoder_type=str(tokenizer_root),
    )
    model.class_threshold = torch.clamp(
        model.class_threshold,
        min=DEFAULT_THRESHOLD_FLOOR,
    )
    model.eval()
    model = model.to(torch.device(resolved_device))

    def infer(frame_path: Path) -> dict[str, object]:
        with Image.open(frame_path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(torch.device(resolved_device))
        with torch.no_grad():
            output = inference_ram(tensor, model)
        return {
            "tags_en": _split_pipe_labels(output[0] if len(output) > 0 else ""),
            "tags_zh": _split_pipe_labels(output[1] if len(output) > 1 else ""),
        }

    return infer


def bundle_frame_records(
    root: Path,
    timeline: list[Any],
    *,
    frame_mode: str = "representative",
    timeline_indexes: list[int] | tuple[int, ...] | set[int] | None = None,
    per_timeline_limit: int = 0,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    allowed_indexes = {int(value) for value in timeline_indexes or [] if int(value) > 0}
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        timeline_index = _int(item.get("index")) or position
        if allowed_indexes and timeline_index not in allowed_indexes:
            continue
        values = (
            item.get("temporal_frame_paths")
            if frame_mode == "continuous" and isinstance(item.get("temporal_frame_paths"), list)
            else item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
        )
        if not values and isinstance(item.get("evidence_frame_paths"), list):
            values = item["evidence_frame_paths"]
        integrated = item.get("integrated_visual") if isinstance(item.get("integrated_visual"), dict) else {}
        if not values and isinstance(integrated.get("evidence_frame_paths"), list):
            values = integrated["evidence_frame_paths"]
        if not values and item.get("frame_path"):
            values = [item["frame_path"]]
        frames = _image_records(
            root,
            values,
            timeline_index=timeline_index,
            start=_float(item.get("start")),
            end=_float(item.get("end")),
            require_bundle_contained=frame_mode == "continuous",
        )
        if frame_mode == "representative":
            frames = frames[:1]
        elif int(per_timeline_limit) > 0:
            frames = _evenly_select(frames, int(per_timeline_limit))
        if not frames:
            continue
        for frame_position, frame in enumerate(frames, start=1):
            result.append(
                {
                    "timeline_index": timeline_index,
                    "start": _float(item.get("start")),
                    "end": _float(item.get("end")),
                    "frame_position": frame_position,
                    "frame_id": f"timeline-{timeline_index:06d}-frame-{frame_position:03d}",
                    "source_frame_id": frame.get("source_frame_id", ""),
                    "timestamp": frame["timestamp"],
                    "frame_path": frame["frame_path"],
                }
            )
    return result


def _image_records(
    root: Path,
    values: list[Any],
    *,
    timeline_index: int,
    start: float,
    end: float,
    require_bundle_contained: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    value_count = len(values)
    for source_position, value in enumerate(values, start=1):
        raw_path = value.get("path") if isinstance(value, dict) else value
        path = Path(str(raw_path or "")).expanduser()
        if not path.is_absolute():
            path = root / path
        # Validate the lexical path before resolve can erase a reparse hop.
        # Both representative and continuous recognition use Bundle copies.
        try:
            resolved = safe_path(root, str(path))
        except ValueError as exc:
            if str(exc) == 'courseware path outside bundle':
                raise ValueError(f'frame_path_outside_bundle: {path}') from exc
            raise
        if require_bundle_contained:
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"frame_path_outside_bundle: {resolved}") from exc
        if not resolved.exists() or not resolved.is_file():
            continue
        if resolved.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        if isinstance(value, dict) and value.get("timestamp") is not None:
            timestamp = _float(value.get("timestamp"))
        elif value_count <= 1:
            timestamp = start
        else:
            timestamp = start + (end - start) * (source_position - 1) / (value_count - 1)
        result.append(
            {
                "timeline_index": timeline_index,
                "source_position": source_position,
                "source_frame_id": str(value.get("frame_id") or "") if isinstance(value, dict) else "",
                "timestamp": round(timestamp, 6),
                "frame_path": resolved,
            }
        )
    return result


def _evenly_select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]
    indexes = [round(position * (len(rows) - 1) / (limit - 1)) for position in range(limit)]
    return [rows[index] for index in indexes]


def _source_root(
    value: str | Path | None,
    *,
    discovery: dict[str, Any] | None = None,
) -> Path | None:
    candidate = value or os.environ.get("VKP_RECOGNIZE_ANYTHING_SOURCE")
    if candidate:
        path = Path(candidate).expanduser()
        if path.exists() and path.is_dir():
            return path.resolve()
    inventory_path = str((discovery or {}).get("source_root") or "")
    candidate_path = Path(inventory_path).expanduser() if inventory_path else None
    if candidate_path and candidate_path.is_dir():
        return candidate_path.resolve()
    return DEFAULT_SOURCE_ROOT.resolve() if DEFAULT_SOURCE_ROOT.exists() else None


def _checkpoint(
    value: str | Path | None,
    *,
    discovery: dict[str, Any] | None = None,
) -> Path | None:
    candidate = value or os.environ.get("VKP_RAM_PLUS_CHECKPOINT")
    if candidate:
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            return path.resolve()
    inventory_path = str((discovery or {}).get("checkpoint_path") or "")
    candidate_path = Path(inventory_path).expanduser() if inventory_path else None
    if candidate_path and candidate_path.is_file():
        return candidate_path.resolve()
    return DEFAULT_CHECKPOINT.resolve() if DEFAULT_CHECKPOINT.exists() else None


def _tokenizer_root(*, discovery: dict[str, Any] | None = None) -> Path | None:
    candidate = os.environ.get("VKP_RAM_PLUS_TOKENIZER")
    if candidate:
        path = Path(candidate).expanduser()
        if path.exists() and path.is_dir() and (path / "vocab.txt").is_file():
            return path.resolve()
    inventory_path = str((discovery or {}).get("tokenizer_root") or "")
    candidate_path = Path(inventory_path).expanduser() if inventory_path else None
    if candidate_path and candidate_path.is_dir() and (candidate_path / "vocab.txt").is_file():
        return candidate_path.resolve()
    if DEFAULT_TOKENIZER_ROOT.exists() and (DEFAULT_TOKENIZER_ROOT / "vocab.txt").is_file():
        return DEFAULT_TOKENIZER_ROOT.resolve()
    return None


def _inventory_model_discovery(value: str | Path | None = None) -> dict[str, Any]:
    inventory_path = (
        Path(value).expanduser().resolve()
        if value is not None
        else (source_reviews_root() / "SOURCE_INVENTORY.json").resolve()
    )
    result = {
        "source": "source_inventory",
        "inventory_path": str(inventory_path),
        "inventory_available": inventory_path.is_file(),
        "entry_found": False,
        "source_root": "",
        "deployment_path": "",
        "checkpoint_path": "",
        "tokenizer_root": "",
    }
    if not inventory_path.is_file():
        return result
    try:
        payload = read_json(inventory_path)
    except (OSError, ValueError):
        return result
    entries = payload.get("entries") if isinstance(payload, dict) else []
    entry = next(
        (
            row
            for row in entries or []
            if isinstance(row, dict) and str(row.get("name") or "").casefold() == "recognize-anything"
        ),
        None,
    )
    if not isinstance(entry, dict):
        return result
    source_text = str(entry.get("local_path") or "").strip()
    deployment_text = str(entry.get("deployment_path") or "").strip()
    source = Path(source_text).expanduser() if source_text else None
    deployment = Path(deployment_text).expanduser() if deployment_text else None
    checkpoint = deployment / "ram_plus_swin_large_14m.pth" if deployment else None
    tokenizer = deployment.parent / "bert-base-uncased-tokenizer" if deployment else None
    return {
        **result,
        "entry_found": True,
        "source_root": str(source.resolve()) if source and source.is_dir() else "",
        "deployment_path": str(deployment.resolve()) if deployment and deployment.is_dir() else "",
        "checkpoint_path": str(checkpoint.resolve()) if checkpoint and checkpoint.is_file() else "",
        "tokenizer_root": str(tokenizer.resolve()) if tokenizer and tokenizer.is_dir() and (tokenizer / "vocab.txt").is_file() else "",
    }


def _write_result(root: Path, result: dict[str, Any]) -> None:
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    with bundle_write_lock(root, operation="general_tagger", timeout_seconds=1.0):
        write_json(exports / "general-tagger.json", result)
        (exports / "general-tagger.md").write_text(_render_markdown(result), encoding="utf-8")
        write_json(
            root / "mcp-general-tagger.args.json",
            {
                "bundle_dir": str(root),
                "source_root": result["source_root"],
                "checkpoint_path": result["checkpoint_path"],
                "device": "cuda",
                "prefer_language": result["prefer_language"],
                "limit": 0,
                "execute": False,
                "import_annotations": True,
                "write": True,
            },
        )
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if isinstance(manifest, dict):
            manifest["general_tagger_json"] = "exports/general-tagger.json"
            manifest["general_tagger_markdown"] = "exports/general-tagger.md"
            manifest["mcp_general_tagger_args"] = "mcp-general-tagger.args.json"
            manifest["general_tagger_summary"] = {
                "status": result["status"],
                "model": MODEL,
                "model_revision": result["model_revision"],
                "annotation_count": result["annotation_count"],
                "updated_at": result["updated_at"],
            }
            write_json(manifest_path, manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# General Image Tagger",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Model: `{result.get('model')}`",
        f"- Planned frames: `{result.get('planned_frame_count')}`",
        f"- Annotations: `{result.get('annotation_count')}`",
        "- Domain: general real-world images; CL/WD remain compatibility baselines only.",
        "- Boundary: local candidate evidence, no automatic download or remote fallback.",
        "- Device policy: CUDA GPU required; no automatic CPU fallback.",
        "",
        "| Timeline | Tags | Frame SHA-256 |",
        "| ---: | --- | --- |",
    ]
    for row in result.get("annotations") or []:
        lines.append(
            f"| {row.get('timeline_index')} | {_md(', '.join(row.get('tags') or []))} | `{row.get('artifact_sha256')}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _labels(value: object) -> list[str]:
    if isinstance(value, list):
        return _unique([str(item).strip() for item in value if str(item).strip()])
    return _split_pipe_labels(str(value or ""))


def _split_pipe_labels(value: str) -> list[str]:
    return _unique([part.strip() for part in str(value or "").split("|") if part.strip()])


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
