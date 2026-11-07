from __future__ import annotations

import video_knowledge_pipeline.asr_chunk_batch_merge as chunk_merge
import video_knowledge_pipeline.asr_consensus as asr_consensus
import video_knowledge_pipeline.asr_evidence_autoadjudication as autoadjudication
import video_knowledge_pipeline.asr_response_quality as response_quality
import video_knowledge_pipeline.term_text as term_text
from video_knowledge_pipeline.text_normalization import (
    compact_ascii_cjk,
    compact_ascii_cjk_after_lowering,
)


def test_compact_ascii_cjk_preserves_existing_strict_contract() -> None:
    assert compact_ascii_cjk("KİABC 中文-1") == "abc中文1"
    assert compact_ascii_cjk(0) == ""


def test_compact_ascii_cjk_after_lowering_preserves_unicode_case_contract() -> None:
    assert compact_ascii_cjk_after_lowering("KİABC 中文-1") == "kiabc中文1"
    assert compact_ascii_cjk_after_lowering(None) == ""


def test_strict_asr_callers_share_one_owner() -> None:
    assert chunk_merge._compact_text is compact_ascii_cjk
    assert response_quality._compact_text is compact_ascii_cjk
    assert autoadjudication._compact is compact_ascii_cjk


def test_lower_first_callers_share_one_owner() -> None:
    assert asr_consensus._compact is compact_ascii_cjk_after_lowering
    assert term_text._normalise_term is compact_ascii_cjk_after_lowering