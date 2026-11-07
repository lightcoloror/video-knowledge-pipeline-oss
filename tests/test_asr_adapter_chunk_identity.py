from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.asr_adapter import normalize_asr_output


def test_sensevoice_chunked_normalization_namespaces_repeated_segment_ids(
    tmp_path: Path,
) -> None:
    """Multi-chunk IDs stay unique without changing text or time boundaries."""
    raw = tmp_path / "sensevoice-chunked.json"
    raw.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_funasr_chunked_raw_output.v1",
                "result": [
                    {
                        "sentence_info": [
                            {
                                "id": "segment-000001",
                                "start": 1000,
                                "end": 2000,
                                "text": "第一块",
                            }
                        ]
                    },
                    {
                        "sentence_info": [
                            {
                                "id": "segment-000001",
                                "start": 301000,
                                "end": 302000,
                                "text": "第二块",
                            }
                        ]
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = normalize_asr_output(
        tmp_path / "workspace", raw, provider="sensevoice", title="分块"
    )
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert [row["segment_id"] for row in payload["segments"]] == [
        "funasr-record-0000:segment-000001",
        "funasr-record-0001:segment-000001",
    ]
    assert [row["source_segment_ids"] for row in payload["segments"]] == [
        ["funasr-record-0000:segment-000001"],
        ["funasr-record-0001:segment-000001"],
    ]
    assert [(row["start"], row["end"], row["text"]) for row in payload["segments"]] == [
        (1.0, 2.0, "第一块"),
        (301.0, 302.0, "第二块"),
    ]
