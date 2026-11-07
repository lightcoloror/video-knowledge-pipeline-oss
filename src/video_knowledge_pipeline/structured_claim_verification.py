"""Thin declared-claim comparison adapter, not a replacement NLI framework.

Adapts RAGChecker's claim/source matrix and ALCE's joint / citation-set tests.
Uses VKP's existing reference index, canonical hashes, numeric normalization and
jsonschema. A consistent annotation is NOT evidence that its interpretation is
correct: admission/extraction accuracy, complete decomposition and external
truth remain separate review requirements.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, localcontext
from itertools import combinations
import re
from typing import Any

from jsonschema import Draft202012Validator

from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .numeric_normalization import UNIT_PATTERN, _number_unit
from .summary_citation_audit import CitationIndex

IDENTITY_FIELDS = ('subject', 'property', 'time_scope', 'version')
ALL_FIELDS = (*IDENTITY_FIELDS, 'value', 'unit', 'negated', 'conditions')
_TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 512, 'pattern': r'\S'}
_ID = {'type': 'string', 'minLength': 1, 'maxLength': 512, 'pattern': r'\S'}
_HASH = {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}
_SLOT_PROPERTIES = {
    'subject': _TEXT, 'property': _TEXT, 'value': _TEXT,
    'unit': {'type': 'string', 'maxLength': 64},
    'time_scope': {'type': 'string', 'maxLength': 512},
    'version': {'type': 'string', 'maxLength': 128},
    'negated': {'type': 'boolean'},
    'conditions': {'type': 'array', 'maxItems': 16, 'uniqueItems': True, 'items': _TEXT},
}
SLOTS_SCHEMA = dict(type='object', additionalProperties=False, required=list(ALL_FIELDS),
                    properties=_SLOT_PROPERTIES)
PARTIAL_SLOTS_SCHEMA = dict(type='object', additionalProperties=False,
                            required=list(IDENTITY_FIELDS), properties=_SLOT_PROPERTIES,
                            dependentRequired={'value': ['unit'], 'unit': ['value']})
CLAIM_SCHEMA = dict(type='object', additionalProperties=False,
                    required=['claim_id', 'output_location', 'output_canonical_sha256', 'evidence_ids', 'slots'],
                    properties=dict(claim_id=_ID, output_location=_ID, output_canonical_sha256=_HASH,
                                    evidence_ids=dict(type='array', minItems=1, maxItems=8,
                                                      uniqueItems=True, items=_ID), slots=SLOTS_SCHEMA))
CLAIMS_SCHEMA = dict(type='array', minItems=1, maxItems=64, items=CLAIM_SCHEMA)
ASSERTION_SCHEMA = dict(type='object', additionalProperties=False,
    required=['assertion_id', 'evidence_id', 'context_id', 'reference_canonical_sha256', 'span', 'slots'],
    properties=dict(assertion_id=_ID, evidence_id=_ID, context_id=_ID,
        reference_canonical_sha256=_HASH, slots=PARTIAL_SLOTS_SCHEMA,
        span=dict(type='object', additionalProperties=False, required=['start', 'end', 'canonical_sha256'],
                  properties=dict(start=dict(type='integer', minimum=0),
                                  end=dict(type='integer', minimum=1), canonical_sha256=_HASH))))
QUALIFICATION_SCHEMA = dict(type='array', maxItems=128, items=dict(
    type='object', additionalProperties=False,
    required=['assertion_id', 'qualifier_id', 'status', 'reason', 'condition_indexes'],
    properties=dict(assertion_id=_ID, qualifier_id=_ID,
        status={'enum': ['applies', 'not_applicable', 'unresolved']}, reason=_TEXT,
        condition_indexes=dict(type='array', maxItems=16, uniqueItems=True,
                               items=dict(type='integer', minimum=0, maximum=15)),
        slot_fields=dict(type='array', maxItems=8, uniqueItems=True, items={'enum': list(ALL_FIELDS)}))))
SCOPE_SCHEMA = dict(type='object', additionalProperties=False,
    required=['schema', 'reference_snapshot_sha256', 'required_locations', 'assertions'],
    properties=dict(schema={'const': 'video_knowledge_pipeline.claim_verification_scope.v1'},
        reference_snapshot_sha256=_HASH,
        required_locations=dict(type='array', minItems=1, maxItems=64, uniqueItems=True, items=_ID),
        assertions=dict(type='array', maxItems=128, items=ASSERTION_SCHEMA),
        qualification_assessments=QUALIFICATION_SCHEMA))


def _qualification_audit(assertions, assessments, index):
    """Audit declared note applicability; do not infer it from a table link."""
    expected, errors, unresolved = {}, [], []
    by_id = {row['assertion_id']: row for row in assertions}
    for assertion in assertions:
        context = (index.refs.get(assertion['evidence_id']) or {}).get('source_context') or {}
        if context.get('scope') != 'declared_native_table_associations_not_verified_semantics':
            continue
        refs = context.get('qualification_refs')
        if refs is None:
            unresolved.append(dict(assertion_id=assertion['assertion_id'],
                                   reason='legacy_qualification_catalog_missing'))
            continue
        if (not isinstance(refs, list) or len(refs) > 16 or
                any(not isinstance(ref, str) or not ref or len(ref) > 512 for ref in refs)
                or len(set(refs)) != len(refs)):
            errors.append('claim_protocol:qualification_catalog_invalid')
            continue
        roles = context.get('qualification_roles')
        if roles is not None and (not isinstance(roles, dict) or set(roles) != set(refs)
                or any(not isinstance(role, str) or role not in {'footnotes', 'captions', 'references', 'row_section'}
                       for role in roles.values())):
            errors.append('claim_protocol:qualification_roles_invalid')
            continue
        for ref in refs:
            expected[(assertion['assertion_id'], ref)] = (roles or {}).get(ref, 'footnotes')
    seen = set()
    for row in assessments:
        key = (row['assertion_id'], row['qualifier_id'])
        if key not in expected or key in seen:
            errors.append('claim_protocol:qualification_extra_or_duplicate')
            continue
        seen.add(key)
        conditions = by_id[row['assertion_id']]['slots'].get('conditions', [])
        indexes = row['condition_indexes']
        fields = row.get('slot_fields', [])
        slots = by_id[row['assertion_id']]['slots']
        footnote = expected[key] == 'footnotes'
        bad_fields = (any(field not in slots or slots[field] == []
                          or (isinstance(slots[field], str) and not slots[field].strip()) for field in fields)
                      or ('conditions' in fields and not indexes)
                      or (indexes and not footnote and 'conditions' not in fields))
        if ((row['status'] == 'applies' and (not indexes if footnote else not fields))
                or (footnote and fields) or bad_fields
                or (row['status'] != 'applies' and (indexes or fields))
                or any(i >= len(conditions) for i in indexes)):
            errors.append('claim_protocol:qualification_condition_mapping_invalid')
        if row['status'] == 'unresolved':
            unresolved.append(dict(assertion_id=key[0], qualifier_id=key[1], reason='applicability_unresolved'))
    for key in sorted(expected.keys() - seen):
        unresolved.append(dict(assertion_id=key[0], qualifier_id=key[1], reason='applicability_not_assessed'))
    return dict(errors=errors, unresolved=unresolved,
                assessment_scope='declared_note_applicability_not_verified_semantics')


def _quantity(slots: dict[str, Any]) -> tuple:
    value, unit = slots['value'], slots['unit']
    # Reuse the existing canonical unit mapping, with stdlib Decimal arithmetic
    # for explicit structured numbers. The legacy text matcher rounds floats
    # to eight places; that cannot certify exact claim values. Unknown units,
    # Chinese/qualified values and ranges stay literal instead of being guessed.
    if (re.fullmatch(UNIT_PATTERN, unit)
            and re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value)):
        scale, dimension, canonical_unit = _number_unit(unit)
        with localcontext() as context:
            context.prec = len(value) + 32
            number = Decimal(value) * Decimal(str(scale))
        return ('normalized_quantity', dimension, canonical_unit, number)
    return ('literal_quantity', value, unit)


def _field_value(slots: dict[str, Any], name: str) -> Any:
    if name == 'quantity':
        return _quantity(slots)
    if name == 'conditions':
        return tuple(sorted(slots[name]))
    return slots[name]


def _compare(observation: dict[str, Any], claim: dict[str, Any]) -> dict[str, Any]:
    context_diff = [key for key in IDENTITY_FIELDS if observation.get(key) != claim[key]]
    if context_diff:
        return dict(status='insufficient_evidence', reasons=['context_mismatch:' + ','.join(context_diff)])
    missing = [key for key in ALL_FIELDS if key not in observation]
    # Incompatible qualification is not proof of logical contradiction. It
    # blocks support; compare opposite values only under matching conditions.
    if 'conditions' in observation and _field_value(observation, 'conditions') != _field_value(claim, 'conditions'):
        return dict(status='insufficient_evidence', reasons=['conditions_not_preserved'])
    if missing:
        return dict(status='insufficient_evidence', reasons=['missing_fields:' + ','.join(missing)])
    if _quantity(observation) != _quantity(claim):
        # Two different excluded values do not contradict each other: "not 30"
        # is neither support nor disproof of "not 90".
        return dict(status='insufficient_evidence' if claim['negated'] or observation['negated'] else 'contradiction',
                    reasons=['different_value_or_unit'])
    if observation['negated'] != claim['negated']:
        return dict(status='contradiction', reasons=['opposite_negation'])
    return dict(status='supported', reasons=[])


def _context_key(assertion: dict[str, Any]) -> tuple:
    return (assertion['context_id'], *(assertion['slots'][key] for key in IDENTITY_FIELDS))


def _merge(assertions: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    merged: dict[str, Any] = {}
    values: dict[str, Any] = {}
    conflicts = set()
    for assertion in assertions:
        slots = assertion['slots']
        for key in (*IDENTITY_FIELDS, 'quantity', 'negated', 'conditions'):
            if (key == 'quantity' and 'value' not in slots) or (key != 'quantity' and key not in slots):
                continue
            value = _field_value(slots, key)
            if key in values and values[key] != value:
                conflicts.add(key)
            values[key] = value
        merged.update(slots)
    return merged, sorted(conflicts)


def _group_verdict(assertions: list[dict[str, Any]], slots: dict[str, Any]) -> dict[str, Any]:
    merged, conflicts = _merge(assertions)
    if conflicts:
        # Internal source disagreement must never be repaired by selecting only
        # the favorable annotation. Context/condition disagreement is a gap.
        return dict(status='insufficient_evidence', reasons=['source_disagreement:' + ','.join(conflicts)])
    return _compare(merged, slots)


def _one_claim(claim: dict[str, Any], assertions: list[dict[str, Any]], index: CitationIndex) -> dict[str, Any]:
    cited = index.resolved_ids(claim['evidence_ids'], usable_only=True)
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for assertion in assertions:
        groups[_context_key(assertion)].append(assertion)
    matrix = [dict(assertion_id=row['assertion_id'], evidence_id=row['evidence_id'],
                   **_compare(row['slots'], claim['slots'])) for row in assertions]
    contradictions = {row['assertion_id'] for row in matrix if row['status'] == 'contradiction'}
    conflicted_groups = []
    group_verdicts = []
    # Evaluate ALL admitted annotations, not only the model's chosen citations.
    for key, rows in groups.items():
        if any(rows[0]['slots'][field] != claim['slots'][field] for field in IDENTITY_FIELDS):
            continue
        verdict = _group_verdict(rows, claim['slots'])
        group_verdicts.append(dict(context_id=key[0], assertion_ids=[r['assertion_id'] for r in rows], **verdict))
        if verdict['status'] == 'contradiction':
            contradictions.update(row['assertion_id'] for row in rows)
        if any(reason.startswith('source_disagreement:') for reason in verdict['reasons']):
            conflicted_groups.append(key[0])
    supporting_sets: list[list[str]] = []
    # Bounded adaptation of ALCE joint / leave-one-out support. A subset must
    # establish a complete claim in ONE explicitly declared structural context.
    for size in range(1, len(cited) + 1):
        for subset in combinations(cited, size):
            if any(set(prior) <= set(subset) for prior in supporting_sets):
                continue
            supported = False
            for rows in groups.values():
                selected = [row for row in rows if row['evidence_id'] in subset]
                if selected and _group_verdict(selected, claim['slots'])['status'] == 'supported':
                    supported = True
                    break
            if supported:
                supporting_sets.append(list(subset))
    relevant = {name for subset in supporting_sets for name in subset}
    status = ('contradiction' if contradictions else 'insufficient_evidence' if conflicted_groups
              else 'supported' if supporting_sets else 'insufficient_evidence')
    # Keep every admitted contrary observation. Shared grouping affects only
    # diagnostic counts; it cannot turn majority agreement into a verdict.
    votes = {}
    for row in matrix:
        name = row['evidence_id']
        if row['status'] == 'contradiction':
            votes[name] = 'supports_original'
        elif votes.get(name) != 'supports_original':
            votes[name] = 'supports_candidate' if name in relevant else 'neutral'
    lineage = index.recognition_lineage(votes, votes_by_id=votes)
    return dict(claim_id=claim['claim_id'], output_location=claim['output_location'], status=status,
                source_lineage=lineage,
                assertion_matrix=matrix, context_matrix=group_verdicts,
                minimal_supporting_sets=supporting_sets,
                contradicting_assertion_ids=sorted(contradictions),
                conflicted_context_ids=sorted(set(conflicted_groups)),
                cited_leaf_ids=cited, irrelevant_citation_ids=sorted(set(cited) - relevant),
                citation_precision=len(relevant) / len(cited) if cited and status == 'supported' else 0.0)


def select_bounded_claim_citations(claim: dict[str, Any], assertions: list[dict[str, Any]],
                                   index: CitationIndex) -> list[str]:
    """Choose one bounded proof from a large declared citation scope.

    Reuse the existing same-context comparison and leave-one-out operation.
    This is a witness selection, not a verdict: callers MUST still supply ALL
    assertions to audit_structured_claims so contrary evidence is never dropped.
    It returns one irreducible set, not every globally minimal support set.
    """
    allowed = index.resolved_ids(claim['evidence_ids'], usable_only=True)
    if len(assertions) > 128:
        # Let the unchanged scope validator reject it without unbounded witness
        # search. This helper never relaxes the existing assertion budget.
        return allowed[:8]
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in assertions:
        if row['evidence_id'] in allowed:
            groups[_context_key(row)].append(row)
    for rows in groups.values():
        if _group_verdict(rows, claim['slots'])['status'] != 'supported':
            continue
        names = sorted({row['evidence_id'] for row in rows})
        for name in list(names):
            remaining = [row for row in rows if row['evidence_id'] in names and row['evidence_id'] != name]
            if remaining and _group_verdict(remaining, claim['slots'])['status'] == 'supported':
                names.remove(name)
        if len(names) <= 8:
            return names
    # Keep the original bounded protocol evaluable when no proof was found.
    # Its verdict is insufficient/contradiction; this does not fabricate support.
    return allowed[:8]


def audit_structured_claims(claims: Any, *, scope: Any, citation_index: CitationIndex,
                            output_texts: dict[str, str], output_citations: dict[str, list[str]]) -> dict[str, Any]:
    """Validate the entire bound batch, then compare declared annotations.

    No filesystem or network IO. An invalid batch returns zero verdicts. Slots
    are interpretations; source/span hashes prevent drift but do not validate
    that interpretation. Only the supplied scoped claims are being compared.
    """
    result: dict[str, Any] = dict(schema='video_knowledge_pipeline.structured_claim_audit.v1',
        valid=False, errors=[], claims=[], assessment_scope='declared_structured_claim_consistency',
        factual_truth='not_assessed', source_interpretation_accuracy='not_assessed',
        output_interpretation_accuracy='not_assessed',
        decomposition_completeness='not_assessed', source_file_freshness='not_assessed')
    errors = result['errors']
    try:
        if len(canonical_json_bytes([scope, claims])) > 256 * 1024:
            errors.append('claim_protocol:batch_budget_exceeded')
            return result
    except (TypeError, ValueError, RecursionError):
        errors.append('claim_protocol:invalid_json_metadata')
        return result
    for label, value, schema in [('scope', scope, SCOPE_SCHEMA), ('claims', claims, CLAIMS_SCHEMA)]:
        for error in Draft202012Validator(schema).iter_errors(value):
            path = '.'.join(str(x) for x in error.absolute_path)
            errors.append(f'claim_protocol:{label}.{path}:{error.validator}')
    if errors:
        return result
    if citation_index.errors or scope['reference_snapshot_sha256'] != citation_index.snapshot_sha256:
        errors.append('claim_protocol:reference_snapshot_drift')
    required = set(scope['required_locations'])
    actual = {row['output_location'] for row in claims}
    if required != actual or not required <= output_texts.keys():
        errors.append('claim_protocol:output_scope_mismatch')
    for key, rows in [('claim_id', claims), ('assertion_id', scope['assertions'])]:
        if len({row[key] for row in rows}) != len(rows):
            errors.append('claim_protocol:duplicate_' + key)
    for position, claim in enumerate(claims):
        if any(prior['output_location'] == claim['output_location']
               and _compare(prior['slots'], claim['slots'])['status'] == 'supported'
               for prior in claims[:position]):
            errors.append('claim_protocol:duplicate_claim_content:' + claim['claim_id'])
    for assertion in scope['assertions']:
        name = assertion['evidence_id']
        ref = citation_index.refs.get(name)
        if not ref or name not in citation_index.eligible or citation_index.leaves.get(name) != (name,):
            errors.append('claim_protocol:assertion_requires_eligible_leaf:' + name)
            continue
        if canonical_json_sha256(ref) != assertion['reference_canonical_sha256']:
            errors.append('claim_protocol:reference_drift:' + name)
        start, end = assertion['span']['start'], assertion['span']['end']
        snippet = str(ref.get('snippet') or '')
        if not (0 <= start < end <= len(snippet)) or canonical_json_sha256(snippet[start:end]) != assertion['span']['canonical_sha256']:
            errors.append('claim_protocol:source_span_drift:' + name)
    for claim in claims:
        location = claim['output_location']
        if canonical_json_sha256(output_texts.get(location)) != claim['output_canonical_sha256']:
            errors.append('claim_protocol:output_drift:' + claim['claim_id'])
        names = claim['evidence_ids']
        leaves = citation_index.resolved_ids(names, usable_only=True)
        allowed = citation_index.resolved_ids(output_citations.get(location, []), usable_only=True)
        if (any(name not in citation_index.eligible for name in names) or not leaves or len(leaves) > 8
                or not set(leaves) <= set(allowed)):
            errors.append('claim_protocol:invalid_citation_scope:' + claim['claim_id'])
    if errors:
        return result
    qualification = _qualification_audit(scope['assertions'], scope.get('qualification_assessments', []), citation_index)
    result['qualification_audit'] = qualification
    errors.extend(qualification['errors'])
    if errors:
        return result
    result.update(valid=True, reference_snapshot_sha256=citation_index.snapshot_sha256,
                  scope_sha256=canonical_json_sha256(scope), claims_sha256=canonical_json_sha256(claims),
                  required_locations=sorted(required))
    unresolved = {row['assertion_id'] for row in qualification['unresolved']}
    usable_assertions = [row for row in scope['assertions'] if row['assertion_id'] not in unresolved]
    result['claims'] = [_one_claim(row, usable_assertions, citation_index) for row in claims]
    for claim in result['claims']:
        claim['assertion_matrix'].extend(dict(assertion_id=row['assertion_id'], evidence_id=row['evidence_id'],
            status='insufficient_evidence', reasons=['qualification_applicability_unresolved'])
            for row in scope['assertions'] if row['assertion_id'] in unresolved)
    result['counts'] = {status: sum(row['status'] == status for row in result['claims'])
                        for status in ('supported', 'contradiction', 'insufficient_evidence')}
    return result
