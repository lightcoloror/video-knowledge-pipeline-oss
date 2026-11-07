"""Bounded reference integrity projection for the existing summary fact pack.

Reuses graphlib and VKP canonical hashing. ALCE's reference admission and
per-reference accounting are adapted only for literal quotations here; there is
no new NLI model, source-of-truth store or claim of factual verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
import re
from typing import Any, Iterable

from .canonical_json import canonical_json_sha256

MAX_REFS = 4096
MAX_GROUP_MEMBERS = 256
MAX_EDGES = 32768
MAX_GROUP_DEPTH = 32


def _review_only(row: dict[str, Any]) -> bool:
    return (row.get('fact_status') == 'review_gap_not_fact'
            or row.get('source_kind') == 'review_gap'
            or row.get('source_excerpt_truncated') is True
            or row.get('evidence_level') in ('unavailable', 'not_seen'))


def _quote_text(value: str) -> str:
    # Layout whitespace may differ. Decimal points, signs, slashes and percent
    # marks are semantic content: do not reuse the lexical _compact matcher.
    text = re.sub(r'\s+', ' ', value).strip().strip('“”"')
    # Chinese layout wrapping is removable, but "now here" != "nowhere"
    # and two separate digit tokens must never become one number.
    return re.sub(r'(?<=[\u4e00-\u9fff，。！？；：、]) | (?=[\u4e00-\u9fff，。！？；：、])', '', text)


@dataclass
class CitationIndex:
    refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    leaves: dict[str, tuple[str, ...]] = field(default_factory=dict)
    usable_leaves: dict[str, tuple[str, ...]] = field(default_factory=dict)
    eligible: set[str] = field(default_factory=set)
    review_only: set[str] = field(default_factory=set)
    snippets: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    snapshot_sha256: str | None = None

    def resolved_ids(self, ids: Iterable[str], *, usable_only: bool = False) -> list[str]:
        if self.errors:
            return []
        projection = self.usable_leaves if usable_only else self.leaves
        return sorted({leaf for name in ids for leaf in projection.get(name, ())})

    def quote_matches(self, text: str, ids: Iterable[str]) -> list[str]:
        quote = _quote_text(text)
        if not quote:
            return []
        # A verbatim quotation must occur in ONE supplied source excerpt.
        # Cross-cue quotations require a future verified contiguous locator;
        # arbitrary concatenation of snippets is not such a locator.
        return [name for name in self.resolved_ids(ids, usable_only=True)
                if quote in _quote_text(self.snippets.get(name, ''))]

    def recognition_lineage(self, ids, *, votes_by_id=None):
        """Reuse correction provenance grouping; weights are counts, not trust."""
        from .evidence_lineage import aggregate_source_votes
        names = self.resolved_ids(ids, usable_only=True)
        if len(names) > 512:
            return {'status': 'not_assessed', 'reason': 'lineage_scope_budget',
                    'independence_is_verified': False, 'independent_source_count': 0}
        rows = []
        for name in names:
            ref = self.refs[name]
            kind = ref.get('observation_kind') or ref.get('source_kind')
            source_type = {'visual': 'visual_text', 'asr': 'asr_or_subtitle',
                           'temporal_visual_understanding': 'temporal_visual',
                           'continuous_tag_weak_evidence': 'tagger'}.get(kind, kind)
            rows.append({**ref, 'source_type': source_type})
        votes = [{'source_weight': 1, 'vote': (votes_by_id or {}).get(name, 'neutral')} for name in names]
        return {**aggregate_source_votes(rows, votes), 'weight_scope': 'unit_observation_counts_not_confidence'}

    def audit(self, items: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        rows = []
        for location, item in items:
            ids = item.get('evidence_ids') or []
            leaf_ids = self.resolved_ids(ids)
            row = dict(location=location, evidence_ids=list(ids), leaf_evidence_ids=leaf_ids,
                       semantic_support='not_assessed', source_freshness='not_assessed',
                       redundant_reference_path_count=sum(len(self.leaves.get(name, ())) for name in ids) - len(leaf_ids))
            row['source_lineage'] = self.recognition_lineage(ids)
            if item.get('kind') == 'verbatim_quote':
                matches = self.quote_matches(str(item.get('text') or ''), ids)
                row.update(quote_matching_evidence_ids=matches,
                           quote_nonmatching_evidence_ids=sorted(set(leaf_ids) - set(matches)),
                           quote_citation_precision=len(matches) / len(leaf_ids) if leaf_ids else 0.0,
                           quote_support='literal_excerpt_match' if matches else 'not_found')
            rows.append(row)
        return dict(schema='video_knowledge_pipeline.summary_citation_integrity.v1',
                    assessment_scope='reference_integrity_not_entailment',
                    semantic_support='not_assessed', factual_truth='not_assessed',
                    reference_snapshot_sha256=self.snapshot_sha256,
                    reference_integrity='invalid' if self.errors else 'valid',
                    unique_leaf_count=len(self.snippets), items=rows)


def build_citation_index(fact_pack: dict[str, Any]) -> CitationIndex:
    """Expand chapter groups without promoting group labels into observations.

    Identical references may be reused across chapters. Conflicting definitions
    of one ID, missing members, cycles and budget overruns reject the entire
    projection. No files are read or mutated and no metadata hash is called a
    verified source hash.
    """
    result = CitationIndex()
    sections = fact_pack.get('sections', [])
    if not isinstance(sections, list) or len(sections) > MAX_REFS:
        result.errors.append('evidence_index:invalid_sections')
        return result
    graph: dict[str, tuple[str, ...]] = {}
    identities: dict[str, str] = {}
    row_count = edge_count = 0
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get('evidence_refs', []), list):
            result.errors.append('evidence_index:invalid_section_refs')
            continue
        for ref in section.get('evidence_refs', []):
            row_count += 1
            if row_count > MAX_REFS:
                result.errors.append('evidence_index:reference_budget_exceeded')
                return result
            if not isinstance(ref, dict):
                result.errors.append('evidence_index:invalid_reference')
                continue
            name = ref.get('evidence_id')
            if not isinstance(name, str) or not name or len(name) > 512:
                result.errors.append('evidence_index:invalid_id')
                continue
            snippet = ref.get('snippet', '')
            if not isinstance(snippet, str) or len(snippet) > 65536:
                result.errors.append(f'evidence_index:invalid_snippet:{name}')
                continue
            if ('snippet_canonical_sha256' in ref
                    and ref['snippet_canonical_sha256'] != canonical_json_sha256(snippet)):
                result.errors.append(f'evidence_index:observation_snippet_drift:{name}')
            if 'recognition_provenance' in ref:
                from .evidence_lineage import project_recognition_provenance
                record = ref['recognition_provenance']
                projected = project_recognition_provenance({'recognition_provenance': record},
                                                            selected_field=[], text='')
                if projected['status'] == 'invalid' or (isinstance(record, dict) and record.get('status') == 'invalid'):
                    result.errors.append(f'evidence_index:invalid_recognition_provenance:{name}')
            try:
                identity = canonical_json_sha256(ref)
            except (TypeError, ValueError, RecursionError):
                result.errors.append(f'evidence_index:invalid_metadata:{name}')
                continue
            if name in identities:
                if identities[name] != identity:
                    result.errors.append(f'evidence_index:conflicting_id:{name}')
                continue
            identities[name] = identity
            result.refs[name] = ref
            members = ref.get('member_evidence_ids', [])
            is_group = ref.get('source_kind') == 'evidence_group'
            if (not isinstance(members, list) or len(members) > MAX_GROUP_MEMBERS
                    or any(not isinstance(x, str) or not x or len(x) > 512 for x in members)
                    or len(set(members)) != len(members)
                    or (is_group and not members)
                    or (not is_group and 'member_evidence_ids' in ref)):
                result.errors.append(f'evidence_index:invalid_members:{name}')
                continue
            # Groups are indexing containers, never additional source snippets.
            if is_group and snippet:
                result.errors.append(f'evidence_index:group_has_source_text:{name}')
                continue
            graph[name] = tuple(members)
            edge_count += len(members)
            if edge_count > MAX_EDGES:
                result.errors.append('evidence_index:edge_budget_exceeded')
                return result
    for name, members in graph.items():
        if set(members) - graph.keys():
            result.errors.append(f'evidence_index:missing_member:{name}')
    if result.errors:
        return result
    try:
        ordered = tuple(TopologicalSorter(graph).static_order())
    except CycleError:
        result.errors.append('evidence_index:cyclic_group')
        return result
    depths: dict[str, int] = {}
    expanded_count = 0
    for name in ordered:
        members = graph[name]
        depths[name] = 1 + max((depths[x] for x in members), default=0)
        if depths[name] > MAX_GROUP_DEPTH:
            result.errors.append('evidence_index:group_depth_exceeded')
            return result
        leaves = tuple(sorted({leaf for x in members for leaf in result.leaves[x]})) if members else (name,)
        expanded_count += len(leaves)
        if expanded_count > MAX_EDGES:
            result.errors.append('evidence_index:expanded_reference_budget_exceeded')
            return result
        result.leaves[name] = leaves
        row = result.refs[name]
        usable = tuple(sorted({leaf for x in members for leaf in result.usable_leaves[x]})) if members else (name,)
        if _review_only(row):
            usable = ()
        result.usable_leaves[name] = usable
        is_review = not usable
        (result.review_only if is_review else result.eligible).add(name)
        if not members:
            result.snippets[name] = str(row.get('snippet', ''))
    result.snapshot_sha256 = canonical_json_sha256(sorted(identities.items()))
    return result
