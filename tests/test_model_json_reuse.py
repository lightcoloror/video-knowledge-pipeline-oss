from __future__ import annotations

import video_knowledge_pipeline.coding_tool_provider_parity as parity
import video_knowledge_pipeline.model_json as model_json
import video_knowledge_pipeline.openclaw_integration as openclaw_integration
import video_knowledge_pipeline.text_llm_gateway as text_gateway
import video_knowledge_pipeline.video_source as video_source
import video_knowledge_pipeline.vision_api as vision_api


def test_model_json_consumers_share_the_vsummary_extractor() -> None:
    assert text_gateway.extract_json_document is model_json.extract_json_document
    assert parity.extract_json_document is model_json.extract_json_document
    assert vision_api.extract_json_document is model_json.extract_json_document


def test_shared_extractor_handles_fences_surrounding_text_and_arrays() -> None:
    payload = '{"summary": "brace } inside string", "items": [1, 2]}'

    assert model_json.extract_json_document(
        f"prefix\n```json\n{payload}\n```\nsuffix", require_object=True
    ) == {"summary": "brace } inside string", "items": [1, 2]}
    assert parity._parse_json_content(f"analysis before {payload} after") == {
        "summary": "brace } inside string",
        "items": [1, 2],
    }
    assert vision_api.parse_model_json("prefix [1, 2] suffix") == {
        "result": [1, 2]
    }


def test_last_document_extractor_uses_terminal_subprocess_payload() -> None:
    stdout = (
        'progress {"stage": 1}\ncompleted\n'
        '{"ok": true, "result": {"items": [1, 2]}}\ntrailer'
    )
    expected = {"ok": True, "result": {"items": [1, 2]}}

    assert model_json.extract_last_json_document(
        stdout, require_object=True
    ) == expected
    assert parity._json_from_stdout(stdout) == expected
    assert openclaw_integration._parse_json_stdout(stdout) == expected


def test_stdout_consumer_return_contracts_are_preserved() -> None:
    assert parity._json_from_stdout("not json") is None
    assert openclaw_integration._parse_json_stdout("not json") is None
    assert video_source._parse_json_stdout("") == {}
    assert video_source._parse_json_stdout("not json") == {
        "raw_stdout": "not json"
    }
    assert video_source._parse_json_stdout("progress\n[1, 2]\ntrailer") == {
        "result": [1, 2]
    }


def test_consumer_failure_contracts_are_preserved() -> None:
    assert parity._parse_json_content("not json") is None
    assert vision_api.parse_model_json("not json") == {
        "_parse_failed": True,
        "summary": "not json",
        "raw_content": "not json",
    }
