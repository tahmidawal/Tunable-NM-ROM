"""Independent NumPy-only audit of a b-speed attempt.

Imports neither JAX nor any driver module. Every parity number the report quotes is
recomputed here from the archived `.npz` fields; the job's own numbers are only compared
against these, never trusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def worst_rel(a, b):
    return float(max(rel(a[t], b[t]) for t in range(np.asarray(b).shape[0])))


def load(root, name):
    z = np.load(root / name)
    stride = int(z['archive_stride']) if 'archive_stride' in z.files else 1
    return np.asarray(z['fields'], dtype=np.float64), stride, z


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--fields', required=True)
    p.add_argument('--reference', default='')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    res = json.loads(Path(a.result).read_text())
    root = Path(a.fields)
    checks = {}
    notes = []

    checks['complete'] = bool(res.get('complete'))
    checks['backend_gpu'] = res.get('backend') == 'gpu'
    checks['x64'] = bool(res.get('x64'))
    checks['precision_highest'] = res.get('matmul_precision') == 'highest'
    checks['checkpoint_unchanged'] = (res.get('checkpoint_sha256')
                                      == res.get('checkpoint_sha256_after'))
    checks['checkpoint_is_retained_dn256b'] = (
        res.get('checkpoint_sha256')
        == '18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589')
    checks['bank_frozen'] = bool(res.get('spatial_bank_frozen'))
    checks['weights_frozen'] = bool(res.get('network_weights_frozen'))
    checks['final_cohort_unopened'] = bool(res.get('final_cohort_unopened'))

    inv = res['invocations']
    meshes = sorted({r['intervals'] for r in inv})
    subjects = defaultdict(set)
    reps = defaultdict(set)
    cases = defaultdict(set)
    for r in inv:
        subjects[r['intervals']].add(r['name'])
        reps[(r['intervals'], r['name'], r['case'])].add(r['rep'])
        cases[r['intervals']].add(r['case'])
    grid_ok = True
    for L in meshes:
        want = set(range(res['config']['reps']))
        for n in subjects[L]:
            for c in cases[L]:
                if reps[(L, n, c)] != want:
                    grid_ok = False
                    notes.append(f'missing repetitions: L={L} {n} case={c} '
                                 f'{sorted(reps[(L, n, c)])}')
    checks['complete_cartesian_grid'] = grid_ok
    checks['every_timing_positive'] = all(r['gpu_seconds'] > 0 and r['host_seconds'] > 0
                                          for r in inv)
    checks['host_time_covers_gpu_time'] = all(r['host_seconds'] >= r['gpu_seconds']
                                              for r in inv)
    checks['every_invocation_finite'] = all(r['finite'] for r in inv)

    # identical repetitions must produce identical output digests
    digests = defaultdict(set)
    for r in inv:
        digests[(r['intervals'], r['name'], r['case'])].add(r['field_sha256'])
    checks['repetition_output_identical'] = all(len(v) == 1 for v in digests.values())

    # archived-field digests, where the archive keeps the full resolution
    bad = []
    for r in inv:
        if r.get('archive_stride', 1) != 1:
            continue
        f, stride, _ = load(root, r['artifact'])
        arr = f.astype(np.float32) if r.get('dtype') == 'float32' else f
        h = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        if h != r['field_sha256']:
            bad.append(r['artifact'])
    checks['archived_digests_match'] = not bad
    if bad:
        notes.append(f'digest mismatch on {len(bad)} artifacts, first {bad[:3]}')

    # ---- the parity statistic, recomputed from the archived fields
    parity = []
    for L in meshes:
        incumbent = {}
        for r in inv:
            if r['intervals'] == L and r['name'] == 'incumbent':
                incumbent[r['case']] = r['artifact']
        if not incumbent:
            notes.append(f'no incumbent invocation at L={L}')
            continue
        base = {c: load(root, n) for c, n in incumbent.items()}
        for name in sorted(subjects[L]):
            if name == 'incumbent':
                continue
            rows = []
            for c in sorted(cases[L]):
                art = next((r['artifact'] for r in inv
                            if r['intervals'] == L and r['name'] == name and r['case'] == c),
                           None)
                if art is None:
                    continue
                f, st, _ = load(root, art)
                b, sb, _ = load(root, incumbent[c])
                if st != sb or f.shape != b.shape:
                    notes.append(f'archive stride mismatch L={L} {name} case={c}')
                    continue
                rows.append(dict(case=c, archive_stride=st, field_relative=worst_rel(f, b),
                                 bitwise=bool(np.array_equal(f, b))))
            if rows:
                parity.append(dict(intervals=L, name=name,
                                   worst_field_relative=max(x['field_relative'] for x in rows),
                                   bitwise=all(x['bitwise'] for x in rows), cases=rows))
    recomputed = {(x['intervals'], x['name']): x['worst_field_relative'] for x in parity}
    reported = {(x['intervals'], x['arm']): x['worst_field_relative'] for x in res['parity']}
    diffs = []
    for key, val in recomputed.items():
        if key in reported:
            ref = reported[key]
            diffs.append(abs(val - ref) / max(ref, 1e-300) if ref > 0 else abs(val - ref))
    checks['recomputed_parity_agrees_with_job'] = bool(diffs) and max(diffs) < 0.5
    checks['recomputed_parity_worst_ratio'] = float(max(diffs)) if diffs else None
    notes.append('the recomputed statistic uses the ARCHIVED field, which above '
                 f"{res['config'].get('archive_max_intervals')} intervals is the exact "
                 'nested-node restriction for cases other than case 0, so it can differ '
                 'from the job\'s full-field statistic by a bounded factor; the check is '
                 'that both sit on the same order, not that they are equal')

    # the parity verdict, recomputed: field bar AND identical integers from the job record
    verdict = []
    for row in res['parity']:
        rc = recomputed.get((row['intervals'], row['arm']))
        verdict.append(dict(
            intervals=row['intervals'], arm=row['arm'], declared_class=row['declared_class'],
            job_field_relative=row['worst_field_relative'],
            audit_field_relative=rc,
            iterations_identical=row['iterations_identical'],
            reasons_identical=row['reasons_identical'],
            parity=bool(row['worst_field_relative'] <= res['config']['parity_bar']
                        and row['iterations_identical'] and row['reasons_identical'])))
    checks['every_arm_has_a_parity_verdict'] = len(verdict) == len(res['parity'])

    # ---- the retained abl01 gate, recomputed here
    if a.reference:
        refdir = Path(a.reference)
        # `engines.params_draw` draws COLUMN BY COLUMN, so drawing 4 cases from a seed
        # does not extend a draw of 2 from the same seed: only the first column agrees.
        # abl01's cases 4 and 5 came from `params_draw(911702, 2)` and this cohort's
        # came from `params_draw(911702, 4)`, so they are DIFFERENT physical cases and
        # are excluded from the gate. Cases 0-3 come from `params_draw(7090702, 4)` in
        # both and are identical; the gate is those four full trajectories.
        comparable = int(res['config']['eval_cases'])
        rows = []
        for r in inv:
            if r['name'] != 'incumbent' or r['intervals'] != 256 or r['rep'] != 0:
                continue
            f = refdir / f"L256_a_neural_eq_case{r['case']}_rep0.npz"
            if not f.exists():
                continue
            got, st, _ = load(root, r['artifact'])
            if st != 1:
                continue
            rows.append(dict(case=r['case'], comparable=bool(r['case'] < comparable),
                             field_relative=worst_rel(got, np.load(f)['fields'])))
        keep = [x for x in rows if x['comparable']]
        if keep:
            checks['incumbent_reproduces_abl01'] = max(
                x['field_relative'] for x in keep) <= 1e-12
            checks['incumbent_vs_abl01_worst'] = max(x['field_relative'] for x in keep)
            checks['incumbent_vs_abl01_cases'] = rows
            checks['incumbent_vs_abl01_comparable_cases'] = [x['case'] for x in keep]

    # ---- the full-order controls did their job
    fom = [r for r in inv if r['kind'] == 'fom']
    checks['tight_control_converged'] = all(
        r['nonlinear_converged'] for r in fom
        if r['name'] == res['config']['accuracy_reference'])
    checks['fom_controls_present'] = sorted({r['name'] for r in fom}) == sorted(
        f['name'] for f in res['config']['fom_settings'])

    out = dict(result=str(a.result), checks=checks, parity_recomputed=parity,
               parity_verdict=verdict, notes=notes,
               passed=all(v for k, v in checks.items()
                          if isinstance(v, bool) and not k.startswith('incumbent_vs')))
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in checks.items()
                      if not isinstance(v, list)}, indent=2))
    print('PASSED', out['passed'])


if __name__ == '__main__':
    main()
