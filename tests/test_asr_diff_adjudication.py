from __future__ import annotations

import json
import pytest
from pathlib import Path

from video_knowledge_pipeline.asr_consensus import build_asr_consensus
from video_knowledge_pipeline.asr_diff_adjudication import (
    _positioned_differences,
    _tokens,
    apply_asr_diff_adjudication,
    build_asr_diff_adjudication,
)


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"title": "ASR diff fixture"}), encoding="utf-8")
    primary = root / "primary.json"
    secondary = root / "secondary.json"
    primary.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 10, "text": "使用 Playwright MCP 操作浏览器"},
                    {"start": 10, "end": 20, "text": "最后检查结果"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    secondary.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 10, "text": "使用 Playwright client 操作浏览器"},
                    {"start": 10, "end": 20, "text": "最后检查结果"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    build_asr_consensus(root, primary_transcript=primary, secondary_transcript=secondary, write=True)
    return root, primary, secondary


def test_build_asr_diff_adjudication_positions_clusters_and_anonymises(tmp_path: Path) -> None:
    root, _, _ = _bundle(tmp_path)

    result = build_asr_diff_adjudication(root, write=True)

    assert result["status"] == "ready_for_adjudication"
    assert result["difference_count"] >= 1
    assert result["lexical_difference_count"] == 1
    assert result["surface_review_count"] == 1  # retained agreement spot-check
    assert result["cluster_count"] == 2
    diff = result["differences"][0]
    assert diff["primary_text"] == "MCP"
    assert diff["secondary_text"] == "client"
    assert diff["primary_char_start"] < diff["primary_char_end"]
    assert diff["estimated_time"]["start"] >= 0
    cluster = result["clusters"][0]
    assert {cluster["candidate_a"], cluster["candidate_b"]} == {"MCP", "client"}
    assert "primary" not in cluster
    assert (root / "asr-consensus-adjudication.todo.json").exists()


def test_apply_asr_diff_adjudication_only_applies_valid_local_patch(tmp_path: Path) -> None:
    root, _, _ = _bundle(tmp_path)
    pack = build_asr_diff_adjudication(root, write=True)
    diff = pack["differences"][0]
    choice = "A" if diff["candidate_a_text"] == "client" else "B"
    decisions = root / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "diff_id": diff["diff_id"],
                        "choice": choice,
                        "confidence": 0.96,
                        "evidence_refs": ["asr-consensus-clips/asr-consensus-0000.wav"],
                        "reason": "local audio supports client",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = apply_asr_diff_adjudication(root, decisions_json=decisions, write=True)

    assert result["summary"]["applied_patches"] == 1
    patched = json.loads((root / "asr-consensus-patched-transcript.json").read_text(encoding="utf-8"))
    assert patched["segments"][0]["text"] == "使用 Playwright client 操作浏览器"
    assert not (root / "source-arbitrated-transcript.json").exists()


def test_apply_asr_diff_adjudication_rejects_change_without_evidence(tmp_path: Path) -> None:
    root, _, _ = _bundle(tmp_path)
    pack = build_asr_diff_adjudication(root, write=True)
    diff = pack["differences"][0]
    choice = "A" if diff["candidate_a_text"] == "client" else "B"
    decisions = root / "decisions.json"
    decisions.write_text(
        json.dumps({"rows": [{"diff_id": diff["diff_id"], "choice": choice, "confidence": 0.99, "evidence_refs": []}]}),
        encoding="utf-8",
    )

    result = apply_asr_diff_adjudication(root, decisions_json=decisions, write=False)

    assert result["summary"]["applied_patches"] == 0
    assert result["rejected"][0]["reason"] == "evidence_required"

def test_diff_tokenizer_keeps_decimal_and_drops_whitespace() -> None:
    rows = _tokens("价格 100.5 元")

    assert [row["text"] for row in rows] == ["价", "格", "100.5", "元"]
    assert rows[2]["start"] == 3
    assert rows[2]["end"] == 8

def test_diff_adjudication_ignores_punctuation_only_changes() -> None:
    window = {
        "window_id": "punctuation-only",
        "primary_index": 0,
        "secondary_index": 0,
        "primary_start": 0,
        "primary_end": 1,
    }

    assert _positioned_differences(window, "你好。", "你好！") == []


@pytest.mark.parametrize('changed_patch_cue',[False,True])
def test_promotion_preserves_canonical_extra_rows_and_refuses_stale_cue(tmp_path,changed_patch_cue):
    root,primary,_=_bundle(tmp_path)
    canonical=root/'canonical.json'
    value={'provider':'existing','segments':[
        {'start':0,'end':10,'text':'使用 Playwright MCP 操作浏览器','id':'keep-id','metadata':{'keep':True}},
        {'start':10,'end':10.1,'text':'','id':'silence-placeholder'},
        {'start':10.1,'end':20,'text':'人工修正后的其他句子','metadata':{'human':True}}]}
    if changed_patch_cue:
        value['segments'][0]['text']='已经人工改过当前句子'
    canonical.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
    (root/'manifest.json').write_text(json.dumps({'corrected_transcript_json':'canonical.json'}),encoding='utf-8')
    pack=build_asr_diff_adjudication(root)
    d=next(row for row in pack['differences'] if row['primary_text']=='MCP')
    decisions=root/'decisions.json'
    decisions.write_text(json.dumps({'rows':[{'diff_id':d['diff_id'],
        'choice':'A' if d['candidate_a_text']=='client' else 'B','confidence':.99,
        'evidence_refs':['synthetic fixture'],'reason':'fixture independent source'}]}),encoding='utf-8')
    before=canonical.read_bytes()
    if changed_patch_cue:
        with pytest.raises(ValueError,match='canonical cue'):
            apply_asr_diff_adjudication(root,decisions_json=decisions,promote=True)
        assert canonical.read_bytes()==before
        assert not (root/'asr-consensus-patched-transcript.json').exists()
    else:
        result=apply_asr_diff_adjudication(root,decisions_json=decisions,promote=True)
        output=json.loads((root/'source-arbitrated-transcript.json').read_text(encoding='utf-8'))
        assert result['canonical_promotion']['preserved_segment_count']==3
        assert output['segments'][0]['text']=='使用 Playwright client 操作浏览器'
        assert output['segments'][0]['metadata']=={'keep':True}
        assert output['segments'][0]['id']=='keep-id'
        assert output['segments'][1:]==value['segments'][1:]
        assert canonical.read_bytes()==before


def test_promotion_refuses_changed_canonical_selection_even_with_old_file_intact(tmp_path):
    root,primary,_=_bundle(tmp_path)
    canonical=root/'canonical.json'
    canonical.write_bytes(primary.read_bytes())
    manifest=root/'manifest.json'
    manifest.write_text(json.dumps({'corrected_transcript_json':'canonical.json'}),encoding='utf-8')
    pack=build_asr_diff_adjudication(root)
    d=pack['differences'][0]
    decisions=root/'decisions.json'
    decisions.write_text(json.dumps({'rows':[{'diff_id':d['diff_id'],
        'choice':'A' if d['candidate_a_text']=='client' else 'B','confidence':.99,
        'evidence_refs':['fixture']}]}),encoding='utf-8')
    replacement=root/'replacement.json'
    replacement.write_bytes(canonical.read_bytes())
    manifest.write_text(json.dumps({'corrected_transcript_json':'replacement.json'}),encoding='utf-8')
    with pytest.raises(ValueError,match='selection changed'):
        apply_asr_diff_adjudication(root,decisions_json=decisions,promote=True)
    assert not (root/'asr-consensus-patched-transcript.json').exists()


def _promotion_decisions(root, pack):
    diff = next(row for row in pack['differences'] if row['primary_text'] == 'MCP')
    path = root / 'decisions.json'
    path.write_text(json.dumps({'rows': [{
        'diff_id': diff['diff_id'],
        'choice': 'A' if diff['candidate_a_text'] == 'client' else 'B',
        'confidence': .99, 'evidence_refs': ['synthetic fixture'],
    }]}), encoding='utf-8')
    return path


def test_legacy_pack_cannot_replace_an_unbound_existing_canonical(tmp_path):
    root, primary, _ = _bundle(tmp_path)
    canonical = root / 'canonical.json'
    canonical.write_bytes(primary.read_bytes())
    (root / 'manifest.json').write_text(json.dumps({
        'corrected_transcript_json': 'canonical.json',
    }), encoding='utf-8')
    pack = build_asr_diff_adjudication(root)
    decisions = _promotion_decisions(root, pack)
    pack['source_bindings'].pop('canonical')
    (root / 'asr-consensus-adjudication-pack.json').write_text(json.dumps(pack), encoding='utf-8')
    before = canonical.read_bytes()
    with pytest.raises(ValueError, match='canonical.*binding'):
        apply_asr_diff_adjudication(root, decisions_json=decisions, promote=True)
    assert canonical.read_bytes() == before
    assert not (root / 'asr-consensus-patched-transcript.json').exists()
    assert not (root / 'source-arbitrated-transcript.json').exists()


def test_legacy_pack_without_existing_canonical_can_still_promote(tmp_path):
    root, _, _ = _bundle(tmp_path)
    pack = build_asr_diff_adjudication(root)
    decisions = _promotion_decisions(root, pack)
    result = apply_asr_diff_adjudication(root, decisions_json=decisions, promote=True)
    assert result['canonical_promotion']['basis'] == 'primary_without_existing_canonical'
    assert (root / 'source-arbitrated-transcript.json').is_file()


@pytest.mark.parametrize('existing_canonical', [False, True])
def test_promotion_rechecks_canonical_selection_inside_write_lock(tmp_path, monkeypatch, existing_canonical):
    from contextlib import contextmanager
    from video_knowledge_pipeline import asr_diff_adjudication as adapter

    root, primary, _ = _bundle(tmp_path)
    canonical = root / 'canonical.json'
    if existing_canonical:
        canonical.write_bytes(primary.read_bytes())
        (root / 'manifest.json').write_text(json.dumps({
            'corrected_transcript_json': 'canonical.json',
        }), encoding='utf-8')
    pack = build_asr_diff_adjudication(root)
    decisions = _promotion_decisions(root, pack)
    replacement = root / 'replacement.json'
    replacement.write_bytes(primary.read_bytes())

    @contextmanager
    def changed_just_before_lock(*args, **kwargs):
        (root / 'manifest.json').write_text(json.dumps({
            'corrected_transcript_json': 'replacement.json',
        }), encoding='utf-8')
        yield

    monkeypatch.setattr(adapter, 'bundle_write_lock', changed_just_before_lock)
    with pytest.raises(ValueError, match='canonical selection changed'):
        apply_asr_diff_adjudication(root, decisions_json=decisions, promote=True)
    assert not (root / 'asr-consensus-patched-transcript.json').exists()
    assert not (root / 'source-arbitrated-transcript.json').exists()
    assert replacement.read_bytes() == primary.read_bytes()
