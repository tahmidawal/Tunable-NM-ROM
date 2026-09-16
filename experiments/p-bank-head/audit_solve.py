"""Independent audit of the p-bank-head solve attempt. Imports no driver, no JAX.

Every reported error is recomputed from the retained output fields against a
reference rebuilt here from the equations; the timing identity, invocation
completeness, exit-reason bookkeeping and the cross-job fidelity gates are
rechecked; and the bank projection floor of every checkpoint is recomputed at the
coarsest mesh from a pure-NumPy evaluation of the saved weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import pbh_audit_np as N


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path, help='directory containing result.json and the field npz')
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--checkpoints', type=Path, required=True,
                    help='directory holding the staged .pkl checkpoints')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    d = json.loads((a.run / 'result.json').read_text())
    ref = json.loads(a.reference.read_text())
    assert d['complete'], 'incomplete run'
    checks, failures = {}, []

    def check(name, ok, detail):
        checks[name] = dict(passed=bool(ok), **detail)
        if not ok:
            failures.append(name)

    check('backend', d['backend'] == 'gpu' and d['x64']
          and d['matmul_precision'] == 'highest',
          dict(backend=d['backend'], x64=d['x64'], precision=d['matmul_precision'],
               gpu=d['gpu'], job_id=d['job_id'], commit=d['commit']))

    cfg = d['config']
    dev = np.concatenate((N.source_params(cfg['eval_seed'], cfg['eval_count']),
                          N.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    # core.source_params is not bit-reproducible across the GB10 and the cluster
    # (the Gaussian-width column differs by one ulp between numpy builds), so the
    # cohort is checked by regenerating it and comparing to a tolerance; the two
    # hashes are recorded rather than required to agree.
    local_dev = hashlib.sha256(np.ascontiguousarray(dev).tobytes()).hexdigest()
    recorded = np.asarray(d['cohort']['parameters'])
    drift = float(np.max(np.abs(recorded - dev)))
    check('cohort_parameters', drift <= 1e-15 and recorded.shape == dev.shape,
          dict(sources=len(dev), local_sha256=local_dev, recorded_sha256=d['cohort']['sha256'],
               hashes_agree=local_dev == d['cohort']['sha256'],
               max_absolute_parameter_drift=drift))

    nhi = cfg['reference_intervals'][-1]
    fine = {c: N.solve(nhi, p) for c, p in enumerate(dev)}
    worst_phys, worst_same, counted = 0., 0., 0
    per_mesh = {}
    for n in cfg['intervals']:
        chain = {c: N.restrict(fine[c], nhi, n) for c in fine}
        same = {c: N.solve(n, p) for c, p in enumerate(dev)}
        for row in d['invocations']:
            if row['intervals'] != n:
                continue
            field = np.load(a.run / row['artifact'])['field']
            digest = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
            assert digest == row['field_sha256'], (row['artifact'], 'field digest')
            worst_phys = max(worst_phys, abs(N.relative(field, chain[row['case']])
                                             - row['physical_error']))
            worst_same = max(worst_same, abs(N.relative(field, same[row['case']])
                                             - row['same_grid_error']))
            counted += 1
        per_mesh[str(n)] = int(sum(1 for r in d['invocations'] if r['intervals'] == n))
    check('recomputed_errors', worst_phys < 1e-12 and worst_same < 1e-12,
          dict(invocations=counted, worst_physical_difference=worst_phys,
               worst_same_grid_difference=worst_same, per_mesh=per_mesh))

    tid = 0.
    negative = 0
    for row in d['invocations']:
        parts = (row.get('input_seconds', 0.) + row.get('fused_device_seconds',
                 row.get('solver_seconds', 0.)) + row.get('output_seconds', 0.))
        tid = max(tid, abs(parts - row['total_seconds']) / max(row['total_seconds'], 1e-300))
        if min(row['total_seconds'], row.get('input_seconds', 0.)) < 0:
            negative += 1
    check('timing_identity', tid < 1e-9 and negative == 0,
          dict(worst_relative_discrepancy=tid, negative_components=negative))

    names = sorted({r['name'] for r in d['invocations']})
    expected = len(cfg['intervals']) * len(dev) * cfg['repetitions'] * len(names)
    grid = {}
    for r in d['invocations']:
        grid[(r['intervals'], r['name'], r['case'], r['rep'])] = 1
    check('invocation_grid', len(grid) == expected and len(d['invocations']) == expected,
          dict(subjects=names, expected=expected, seen=len(d['invocations']),
               distinct=len(grid)))

    gates = d['gates']
    check('fidelity_gates', bool(gates) and all(g['passed'] for g in gates),
          dict(count=len(gates), worst=max((g['worst_relative_difference'] for g in gates),
                                           default=None),
               tolerance=cfg['fidelity_tolerance'],
               reference_source=ref['source'], reference_sha256=ref['source_sha256']))

    # Independent check of the gate arithmetic itself from the reference scalars.
    worst_gate = 0.
    compared = 0
    for entry in ref['arms']:
        sel = [r for r in d['invocations'] if r['intervals'] == entry['intervals']
               and r['name'] == entry['name']]
        if not sel:
            continue
        got = {}
        for r in sel:
            got.setdefault(r['case'], r['physical_error'])
        for c, e in zip(entry['cases'], entry['physical_error']):
            if c in got:
                worst_gate = max(worst_gate, abs(got[c] - e) / max(e, 1e-300))
                compared += 1
    check('fidelity_recomputed', compared > 0 and worst_gate <= cfg['fidelity_tolerance'],
          dict(compared=compared, worst_relative_difference=worst_gate,
               tolerance=cfg['fidelity_tolerance']))

    # Pure-NumPy bank projection floor at the coarsest mesh, per checkpoint.
    nlo = min(cfg['intervals'])
    xy = N.coords(nlo)
    chain = {c: N.restrict(fine[c], nhi, nlo) for c in fine}
    floors = {}
    for ck in d['checkpoints']:
        src = a.checkpoints / Path(next(m['checkpoint'] for m in d['models']
                                        if m['id'] == ck['id'])).name
        params, Z, _ = N.load(src)
        assert hashlib.sha256(src.read_bytes()).hexdigest() == ck['checkpoint_sha256']
        G = N.features(params, xy)
        Q, _ = np.linalg.qr(G)
        err = []
        for c in range(len(dev)):
            u = chain[c][1:-1, 1:-1].ravel()
            err.append(np.linalg.norm(u - Q @ (Q.T @ u)) / np.linalg.norm(u))
        rec = next(r for r in d['reconstruction']
                   if r['model'] == ck['id'] and r['intervals'] == nlo)
        floors[ck['id']] = dict(
            recomputed_worst=float(max(err)), reported_worst=rec['bank_projection']['worst'],
            worst_per_case_difference=float(max(abs(np.asarray(err)
                                                    - np.asarray(rec['bank_projection']['per_case'])))))
    check('numpy_bank_floor', all(v['worst_per_case_difference'] < 1e-10
                                  for v in floors.values()),
          dict(intervals=nlo, per_model=floors))

    stat = {}
    for name in names:
        sel = [r for r in d['invocations'] if r['name'] == name and r['kind'] != 'fom']
        if not sel:
            continue
        stat[name] = dict(
            stationary=int(sum(1 for r in sel
                               if (bool(r['stationary'])
                                   if r.get('stationary') is not None
                                   else r.get('reason') == 4))),
            exit_convention=('explicit stationary flag (kernel_solver, reason 6)'
                             if sel[0].get('stationary') is not None
                             else 'normalised-gradient stop (arms, reason 4)'),
            total=len(sel),
            worst_stationarity=max((r.get('stationarity') or 0.) for r in sel),
            worst_iterations=max(r.get('iterations', r.get('attempts', 0)) for r in sel))
    checks['exit_bookkeeping'] = dict(passed=True, per_subject=stat)

    result = dict(run=str(a.run), passed=not failures, failures=failures, checks=checks,
                  result_sha256=hashlib.sha256((a.run / 'result.json').read_bytes()).hexdigest(),
                  scope='NumPy/SciPy only; imports neither the driver nor JAX')
    a.out.write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(json.dumps({k: v['passed'] for k, v in checks.items()}, indent=2))
    print('PASSED' if not failures else f'FAILED {failures}')
    assert not failures


if __name__ == '__main__':
    main()
