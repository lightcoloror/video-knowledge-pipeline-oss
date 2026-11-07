from __future__ import annotations

import ast
from pathlib import Path

import video_knowledge_pipeline.batch_repair as batch_repair
import video_knowledge_pipeline.batch_run as batch_run
import video_knowledge_pipeline.knowledge_note_export as knowledge_note_export
import video_knowledge_pipeline.smart_summary_chapters as smart_summary_chapters
import video_knowledge_pipeline.smart_summary_input_pack as smart_summary_input_pack
import video_knowledge_pipeline.term_impact_gate as term_impact_gate
import video_knowledge_pipeline.term_text as term_text
import video_knowledge_pipeline.transcript_main_route_status as main_route_status
import video_knowledge_pipeline.video_workbench as video_workbench
import video_knowledge_pipeline.vision_review_queue as vision_review_queue
from video_knowledge_pipeline.storage import read_json_object_or_empty


def test_optional_object_reader_preserves_existing_fail_closed_contract(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    array = tmp_path / "array.json"
    object_path = tmp_path / "object.json"
    bom = tmp_path / "bom.json"
    invalid.write_text("{", encoding="utf-8")
    array.write_text("[]", encoding="utf-8")
    object_path.write_text('{"课程":"保险"}', encoding="utf-8")
    bom.write_text('{"value":1}', encoding="utf-8-sig")

    assert read_json_object_or_empty(missing) == {}
    assert read_json_object_or_empty(invalid) == {}
    assert read_json_object_or_empty(array) == {}
    assert read_json_object_or_empty(object_path) == {"课程": "保险"}
    assert read_json_object_or_empty(bom) == {}
    assert read_json_object_or_empty(tmp_path) == {}


def test_existing_private_readers_are_compatibility_aliases() -> None:
    assert batch_repair._read_json is read_json_object_or_empty
    assert batch_run._read_bundle_json is read_json_object_or_empty
    assert knowledge_note_export._read_optional_mapping is read_json_object_or_empty
    assert smart_summary_chapters._read_mapping is read_json_object_or_empty
    assert smart_summary_input_pack._read_optional_mapping is read_json_object_or_empty
    assert term_impact_gate._read_optional_object is read_json_object_or_empty
    assert term_text._read_json_object is read_json_object_or_empty
    assert main_route_status._read_json_object is read_json_object_or_empty
    assert video_workbench._read_optional_object is read_json_object_or_empty
    assert vision_review_queue._read_object is read_json_object_or_empty


def test_migrated_modules_do_not_redefine_compatibility_reader_names() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "video_knowledge_pipeline"
    expected = {
        "batch_repair.py": "_read_json",
        "batch_run.py": "_read_bundle_json",
        "knowledge_note_export.py": "_read_optional_mapping",
        "smart_summary_chapters.py": "_read_mapping",
        "smart_summary_input_pack.py": "_read_optional_mapping",
        "term_impact_gate.py": "_read_optional_object",
        "term_text.py": "_read_json_object",
        "transcript_main_route_status.py": "_read_json_object",
        "video_workbench.py": "_read_optional_object",
        "vision_review_queue.py": "_read_object",
    }
    for filename, function_name in expected.items():
        tree = ast.parse((source_root / filename).read_text(encoding="utf-8-sig"))
        definitions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert function_name not in definitions

    video_rag_source = (source_root / "video_rag_pack.py").read_text(
        encoding="utf-8-sig"
    )
    assert "return read_json_object_or_empty(path)" in video_rag_source
