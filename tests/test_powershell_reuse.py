from __future__ import annotations

import ast
from pathlib import Path

import video_knowledge_pipeline.asr_runner as asr_runner
import video_knowledge_pipeline.external_reuse_run_artifacts as external_runs
import video_knowledge_pipeline.frame_recapture as frame_recapture
import video_knowledge_pipeline.lecture_pipeline as lecture_pipeline
import video_knowledge_pipeline.multimodal_sample_review as sample_review
from video_knowledge_pipeline.powershell import (
    quote_powershell_argument,
    quote_powershell_literal,
)


def test_literal_quoting_uses_powershell_single_quote_contract() -> None:
    assert quote_powershell_literal("") == "''"
    assert quote_powershell_literal("D:/课件/O'Brien.mp4") == (
        "'D:/课件/O''Brien.mp4'"
    )
    assert quote_powershell_literal("$value; Remove-Item") == (
        "'$value; Remove-Item'"
    )


def test_argument_quoting_preserves_the_existing_command_contract() -> None:
    assert quote_powershell_argument("--execute") == "--execute"
    assert quote_powershell_argument("") == "''"
    assert quote_powershell_argument("two words") == "'two words'"
    assert quote_powershell_argument("a'b") == "'a''b'"
    assert quote_powershell_argument("$value") == "'$value'"
    assert quote_powershell_argument('a"b') == "'a\"b'"
    assert quote_powershell_argument("a|b") == "'a|b'"
    assert quote_powershell_argument("<input>") == "'<input>'"


def test_existing_public_and_private_front_doors_delegate_to_shared_owner() -> None:
    assert external_runs.ps_quote is quote_powershell_literal
    assert frame_recapture._quote_command_part is quote_powershell_argument
    assert sample_review._quote_ps_path is quote_powershell_literal
    assert sample_review._ps_quote is quote_powershell_literal
    assert asr_runner._quote_powershell_arg is quote_powershell_argument
    assert lecture_pipeline._quote_powershell_arg is quote_powershell_argument
    assert asr_runner._powershell_join(["tool", "two words"]) == "tool 'two words'"
    assert lecture_pipeline._powershell_join(["tool", "two words"]) == "tool 'two words'"


def test_powershell_quoting_has_one_owner() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "video_knowledge_pipeline"
    definitions: set[tuple[str, str]] = set()
    for path in source_root.glob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if "quote" not in node.name and "ps_quote" not in node.name:
                continue
            block = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            if ".replace(\"'\", \"''\")" in block:
                definitions.add((path.name, node.name))

    assert definitions == {("powershell.py", "quote_powershell_literal")}
