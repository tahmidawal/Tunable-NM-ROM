"""Seal all three validation panels before owners may draw evaluation cohorts.

This records identities, coverage, and the already-selected settings. It never
selects configurations. Run only after the independent field/selection audits.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from generate_modcp_comparison import provenance_check, relative
from modcp_audit import digest, summarize_rows


CASES = {'burgers2d', 'wave_reflective', 'wave_absorbing'}


def entry(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    case, cfg = data['case_name'], data['config']
    if case not in CASES or data['status'] != 'validation_frozen' or cfg.get('smoke') or cfg.get('smoke_only'):
        raise ValueError('Only a completed scientific validation panel can be sealed')
    provenance_check(data['provenance'])
    if cfg['meshes'] != [256, 512] or cfg['validation_case_ids'] != list(range(16)):
        raise ValueError('Validation panel does not cover the approved meshes/cohort')
    groups = defaultdict(list)
    for row in data['invocations']:
        if row['split'] != 'validation':
            raise ValueError('Evaluation was opened before the global seal')
        groups[row['intervals'], row['method'], row['configuration']].append(row)
    for rows in groups.values():
        summary = summarize_rows(rows, cfg['validation_case_ids'], cfg['validation_repetitions'], .01)
        if not summary['complete_coverage']:
            raise ValueError('A validation configuration has incomplete case/repetition coverage')
    for mesh in cfg['meshes']:
        methods = {method for n, method, _ in groups if n == mesh}
        if not {'cp', 'modcp', 'film'}.issubset(methods) or not methods.difference({'cp', 'modcp', 'film'}):
            raise ValueError('Validation omits a decoder arm or classical comparison')
    for selection in data['selections']:
        key = selection['intervals'], selection['method'], selection.get('configuration')
        if key[-1] is not None and key not in groups:
            raise ValueError('Frozen selection does not identify a completed validation setting')
    names = ['selections.json', 'selection_freeze.json'] if case == 'burgers2d' else [
        f'frozen_selection_{n}.json' for n in cfg['meshes']]
    proofs = {name: digest(path.parent/name) for name in names}
    seed = cfg['seeds']['evaluation'] if case == 'burgers2d' else cfg['evaluation_seed']
    return case, dict(handoff_sha256=digest(path), selection_proof_sha256=proofs,
                      evaluation_seed=seed, source_commit=data['provenance']['commit'],
                      validation_job_id=data['provenance']['job_id'], source=relative(path),
                      selected_configurations=data['selections'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entries = [entry(path) for path in args.input]
    cases = dict(entries)
    if len(entries) != len(CASES) or cases.keys() != CASES:
        raise ValueError('Exactly one completed validation panel per pilot case is required')
    if args.output.exists():
        raise ValueError('Refusing to replace an existing global cohort seal')
    seal = dict(schema='modcp-global-validation-seal-v1', cases=cases,
                created_utc=datetime.now(timezone.utc).isoformat(), evaluation_generated=False)
    args.output.write_text(json.dumps(seal, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'path': str(args.output), 'sha256': digest(args.output)}, indent=2))


if __name__ == '__main__':
    main()
