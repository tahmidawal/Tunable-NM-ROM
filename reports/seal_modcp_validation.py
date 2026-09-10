"""Seal all three validation panels before owners may draw evaluation cohorts.

This records identities, coverage, and the already-selected settings. It never
selects configurations. Run after source/reference/selection audits. Wave
validation stores bounded observations, so independent recomputation of every
full-grid field error applies to final evaluation, not unsaved validation fields.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from math import prod, isclose
from pathlib import Path

from generate_modcp_comparison import provenance_check, relative
from modcp_audit import digest, summarize_rows, row_error


CASES = {'burgers2d', 'wave_reflective', 'wave_absorbing'}


def entry(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    case, cfg = data['case_name'], data['config']
    if case not in CASES or data['status'] != 'validation_frozen' or cfg.get('smoke') or cfg.get('smoke_only'):
        raise ValueError('Only a completed scientific validation panel can be sealed')
    provenance_check(data['provenance'])
    if data.get('evaluation_opened') or any(r.get('split') == 'evaluation' for r in data.get('references', [])):
        raise ValueError('Evaluation was opened before the global seal')
    if data.get('physical_cases', {}).get('evaluation'):
        raise ValueError('Evaluation parameters were generated before the global seal')
    if cfg['meshes'] != [256, 512] or cfg['validation_case_ids'] != list(range(16)):
        raise ValueError('Validation panel does not cover the approved meshes/cohort')
    groups = defaultdict(list)
    for row in data['invocations']:
        if row['split'] != 'validation':
            raise ValueError('Evaluation was opened before the global seal')
        groups[row['intervals'], row['method'], row['configuration']].append(row)
        if 'provenance' in row and any(str(row['provenance'][k]) != str(data['provenance'][k]) for k in ('job_id', 'gpu', 'commit')):
            raise ValueError('Validation timing panel mixes allocations or source versions')
    for rows in groups.values():
        summary = summarize_rows(rows, cfg['validation_case_ids'], cfg['validation_repetitions'], .01)
        if not summary['complete_coverage']:
            raise ValueError('A validation configuration has incomplete case/repetition coverage')
    for mesh in cfg['meshes']:
        methods = {method for n, method, _ in groups if n == mesh}
        if not {'cp', 'modcp', 'film'}.issubset(methods) or not methods.difference({'cp', 'modcp', 'film'}):
            raise ValueError('Validation omits a decoder arm or classical comparison')
        if case == 'burgers2d':
            inventory = dict(cp=24, modcp=24, film=24, newton_bicgstab=9)
        else:
            rom_count = prod(len(cfg[k]) for k in ('eq_multipliers', 'gn_caps', 'gn_tolerances', 'time_steps'))
            inventory = dict(cp=rom_count, modcp=rom_count, film=rom_count,
                             cg=len(cfg['cg_tolerances'])*len(cfg['time_steps']), rk4=len(cfg['rk4_cfls']))
            if case == 'wave_reflective':
                inventory.update(spectral=1, cn_direct=len(cfg['time_steps']))
        actual = {method: sum(n == mesh and m == method for n, m, _ in groups) for method in methods}
        if actual != inventory:
            raise ValueError('Validation omits part of the declared configuration sweep')
        for method in methods:
            declarations = [s['target'] for s in data['selections'] if s['intervals'] == mesh and s['method'] == method and s['target'] is not None]
            if sorted(declarations) != sorted(cfg['targets']):
                raise ValueError('Frozen selection omits or duplicates a method/target declaration')
    proxies = defaultdict(list)
    for row in data['selection_timings']:
        proxies[row['intervals'], row['method'], row['configuration']].append(row)
        if 'provenance' in row and any(str(row['provenance'][k]) != str(data['provenance'][k]) for k in ('job_id', 'gpu', 'commit')):
            raise ValueError('Selection proxy mixes allocations or source versions')
    if proxies.keys() != groups.keys():
        raise ValueError('Selection proxies do not cover the entire validation sweep')
    for key, rows in proxies.items():
        # Older Burgers validation proxies retain finite/error/hash but no
        # completion field; compare them to their actual case-zero invocation.
        base = next(r for r in groups[key] if r['case'] == 0 and r['rep'] == 0)
        if not summarize_rows(rows, [0], 7, .01)['complete_coverage']:
            raise ValueError('Selection proxy repetitions are incomplete')
        for row in rows:
            if row['finite'] != base['finite'] or not isclose(row_error(row), row_error(base), rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError('Selection proxy physical output differs from its validation case')
            if 'completed' in row and row['completed'] != base['completed']:
                raise ValueError('Selection proxy completion differs from its validation case')
            if 'field_sha256' in row and row['field_sha256'] != base.get('field_sha256'):
                raise ValueError('Selection proxy field hash differs from its validation case')
    for selection in data['selections']:
        key = selection['intervals'], selection['method'], selection.get('configuration')
        if key[-1] is not None and key not in groups:
            raise ValueError('Frozen selection does not identify a completed validation setting')
        if key[-1] is not None and selection['target'] is not None:
            if not summarize_rows(groups[key], cfg['validation_case_ids'], 1, selection['target'])['qualified']:
                raise ValueError('Frozen target selection fails its own validation target')
    names = ['selections.json', 'selection_freeze.json'] if case == 'burgers2d' else [
        f'frozen_selection_{n}.json' for n in cfg['meshes']]
    proofs = {name: digest(path.parent/name) for name in names}
    if case == 'burgers2d':
        if json.loads((path.parent/names[0]).read_text()) != data['selections']:
            raise ValueError('Burgers proof selections differ from handoff')
        freeze = json.loads((path.parent/names[1]).read_text())
        if freeze['selection_sha256'] != proofs['selections.json']:
            raise ValueError('Burgers frozen selection hash mismatch')
        for field, name in (('validation_invocations_sha256', 'invocations.jsonl'), ('selection_timings_sha256', 'selection_timings.jsonl')):
            if freeze[field] != digest(path.parent/name):
                raise ValueError('Burgers frozen validation evidence hash mismatch')
    else:
        for name in names:
            freeze = json.loads((path.parent/name).read_text())
            if name != f"frozen_selection_{freeze['intervals']}.json":
                raise ValueError('Wave selection proof mesh differs from its filename')
            if freeze['evaluation_opened_at_selection'] is not False:
                raise ValueError('Wave evaluation opened at selection')
            if freeze['selections'] != [s for s in data['selections'] if s['intervals'] == freeze['intervals']]:
                raise ValueError('Wave proof selections differ from handoff')
            declared = {s['configuration'] for s in freeze['selections'] if s['configuration'] is not None}
            executable = [s['id'] for s in freeze['selected_settings']]
            if len(executable) != len(set(executable)) or set(executable) != declared:
                raise ValueError('Wave executable settings differ from frozen selection IDs')
            for artifact, expected in freeze['quadrature_sha256'].items():
                if digest(path.parent/artifact) != expected:
                    raise ValueError('Wave frozen quadrature hash mismatch')
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
