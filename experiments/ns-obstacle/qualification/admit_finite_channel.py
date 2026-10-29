"""Publish FD1 admission only from complete evidence and an independent review.

This does not perform the raw-array audit or release training. The reviewed check
contract fixes requiredness; candidates cannot promote diagnostics or waive gates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import operator

ROOT = Path(__file__).resolve().parents[1]
GATE = 'G2_FIXED_FINITE_CHANNEL_V1'
POPULATION = {
    'train': [15 + 1.25*j for j in range(9)],
    'development': [16.875, 23.125],
    'validation': [16, 18, 22, 24],
}
SPACES = ('p4_g4_h1', 'p4_g4_h2', 'p4_g4_h3', 'p3_g3_h3', 'p3_g4_h3')
VARIANTS = ('baseline', 'solver_100x', 'quadrature_plus4')
SCOPE_KEYS = ('amendment_id', 'original_G2_status', 'domain_sensitivity_status',
              'domain_independence_admitted', 'required_scope',
              'geometry_population_and_numerical_thresholds_unchanged')
OPS = dict(le=operator.le, lt=operator.lt, ge=operator.ge, gt=operator.gt, eq=operator.eq)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Missing exact SHA256 binding')
    return value


def expected_scope():
    path = ROOT/'reviews/FD1-SCOPE-CONTRACT.json'
    scope = json.loads(path.read_text())
    if (scope['qualification_gate'] != GATE or
            scope['base_design_sha256'] != sha(ROOT/'DESIGN.md') or
            scope['base_campaign_sha256'] != sha(ROOT/'campaign.json')):
        raise ValueError('Frozen scientific scope differs')
    return dict(scope_contract_sha256=sha(path), **{k: scope[k] for k in SCOPE_KEYS})


def exact_scope(actual, expected):
    return (isinstance(actual, dict) and actual == expected and
            all(type(actual[k]) is type(v) for k, v in expected.items()))


def check_rows(rows, requirements, reynolds=None):
    if not requirements or not any(x.get('required') is True for x in requirements):
        raise ValueError('Empty or diagnostic-only gate contract')
    names = [x['name'] for x in rows]
    expected = [x['name'] for x in requirements]
    if len(set(names)) != len(names) or len(set(expected)) != len(expected) or set(names) != set(expected):
        raise ValueError('Incomplete, duplicated or unregistered checks')
    by_name = {x['name']: x for x in rows}
    for requirement in requirements:
        row = by_name[requirement['name']]
        if type(requirement['required']) is not bool or not requirement['criterion_source']:
            raise ValueError('Check requiredness/source must be frozen explicitly')
        if row['role'] != requirement['role'] or row['threshold'] != requirement['threshold']:
            raise ValueError('Candidate changed a check role or threshold')
        applies = requirement.get('applies_to_re', 'all')
        if applies != 'all' and (not isinstance(applies, list) or not applies):
            raise ValueError('Invalid parameter applicability')
        active = applies == 'all' or reynolds in applies
        if not active:
            if row['status'] != 'not_applicable' or row['value'] is not None:
                raise ValueError('Anchor checks may not be transferred to other parameters')
            continue
        if row['status'] in ('missing', 'unresolved'):
            if requirement['required']:
                raise ValueError('Required evidence missing/unresolved: ' + row['name'])
            continue
        digest(row['evidence_sha256'])
        op = requirement['operator']
        if op == 'true':
            if requirement['threshold'] is not True or type(row['value']) is not bool:
                raise ValueError('Invalid boolean certificate check')
            passed = row['value'] is True
        else:
            if op not in OPS or any(type(x) not in (int, float) or not math.isfinite(x)
                                   for x in (row['value'], row['threshold'])):
                raise ValueError('Invalid numerical check')
            passed = OPS[op](row['value'], row['threshold'])
        if row['status'] != ('pass' if passed else 'fail'):
            raise ValueError('Saved check status disagrees with its actual value')
        if requirement['required'] and not passed:
            raise ValueError('Required numerical gate failed: ' + row['name'])


def validate(candidate, contract, review, bindings):
    scope = expected_scope()
    if (candidate.get('schema') != 'ns-obstacle-FD1-reference-qualification-candidate-v1' or
            candidate.get('qualification_gate') != GATE or not exact_scope(candidate.get('qualification_scope'), scope) or
            candidate.get('admission') is not False or candidate.get('training_released') is not False or
            candidate.get('original_domain_gate') != 'failed' or candidate.get('population') != POPULATION):
        raise ValueError('Incomplete or incorrectly scoped FD1 candidate')
    if (contract.get('schema') != 'ns-obstacle-FD1-check-contract-v1' or
            candidate.get('check_contract_sha256') != bindings['check_contract_sha256']):
        raise ValueError('Exact reviewed check contract required')
    scientific_contracts = {
        'design_sha256': sha(ROOT/'DESIGN.md'), 'campaign_sha256': sha(ROOT/'campaign.json'),
        'scope_contract_sha256': scope['scope_contract_sha256'],
        'noise_protocol_sha256': sha(ROOT/'reviews/RESOLVED-NOISE-PROTOCOL.md'),
    }
    if contract.get('scientific_contracts') != scientific_contracts:
        raise ValueError('Check contract differs from frozen science/protocol')
    for key in ('assembler_sha256', 'retention_index_sha256', 'raw_evidence_index_sha256'):
        digest(candidate['provenance'][key])
    expected = [(re, split) for split, values in POPULATION.items() for re in values]
    if [(r['Re'], r['split']) for r in candidate['rows']] != expected:
        raise ValueError('All fifteen registered parameters, in order, are required')
    if (review.get('schema') != 'ns-obstacle-FD1-population-independent-review-v1' or
            review.get('passed') is not True or review.get('gate') != GATE or
            not exact_scope(review.get('qualification_scope'), scope) or
            review.get('reviewed_population') != POPULATION or
            review.get('raw_evidence_reviewed') is not True or
            review.get('retention_reviewed') is not True or
            review.get('unresolved_required_controls') != []):
        raise ValueError('Complete independent numerical and retention review required')
    for key in ('candidate_sha256', 'check_contract_sha256', 'publisher_sha256'):
        if review.get(key) != bindings[key]:
            raise ValueError('Independent review binding differs: ' + key)
    check_rows(candidate['shared_controls']['checks'], contract['shared'])
    adopted = []
    for row in candidate['rows']:
        unresolved = [x['name'] for x in row['checks'] if x['status'] in ('missing', 'unresolved')]
        if sorted(row['missing_or_unresolved']) != sorted(unresolved):
            raise ValueError('Unresolved diagnostic inventory differs from actual checks')
        states = {(s['space'], s['variant']): s for s in row['states']}
        if len(row['states']) != 15 or set(states) != {(s, v) for s in SPACES for v in VARIANTS}:
            raise ValueError('Complete five-space baseline/S/Q coverage required')
        for state in states.values():
            digest(state['export_manifest_sha256'])
        check_rows(row['checks'], contract['per_parameter'], row['Re'])
        reference = row['adopted_reference']
        if reference['export_manifest_sha256'] != states['p4_g4_h3', 'baseline']['export_manifest_sha256']:
            raise ValueError('Adopted truth must be the actual P4/g4/h3 baseline')
        if reference['storage_mode'] != ('expanded_full_fields' if row['Re'] == 20 else 'coefficient_primary'):
            raise ValueError('Unexpected adopted-reference storage mode')
        fields = ('export_manifest_sha256', 'config_sha256', 'mesh_sha256', 'layout_sha256', 'solution_sha256')
        adopted.append(dict(Re=row['Re'], split=row['split'], storage_mode=reference['storage_mode'],
                            **{k: digest(reference[k]) for k in fields}))
    if len({r['mesh_sha256'] for r in adopted}) != 1 or len({r['layout_sha256'] for r in adopted}) != 1:
        raise ValueError('All adopted states must share the exact fine mesh and layout')
    return scope, adopted


def main():
    parser = argparse.ArgumentParser()
    for name in ('candidate', 'check-contract', 'review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    bindings = dict(candidate_sha256=sha(args.candidate), check_contract_sha256=sha(args.check_contract),
                    publisher_sha256=sha(__file__))
    candidate, contract, review = (json.loads(p.read_text()) for p in (args.candidate, args.check_contract, args.review))
    scope, adopted = validate(candidate, contract, review, bindings)
    record = dict(schema='ns-obstacle-FD1-reference-admission-v1', gate=GATE, passed=True,
                  qualification_scope=scope, population=POPULATION, admitted_references=adopted,
                  independent_review_sha256=sha(args.review), **bindings,
                  original_domain_gate='failed', training_released=False,
                  excluded_claims=json.loads((ROOT/'reviews/FD1-SCOPE-CONTRACT.json').read_text())['excluded_claims'])
    # Never overwrite a previous admission or silently create an unreviewed result.
    with args.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
