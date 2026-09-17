"""Independent NumPy/SciPy audit of one p-linear solve or head job.

Imports neither the driver nor JAX. It rebuilds the same-grid and restricted 2048-interval
references from the equations, recomputes every reported error from the retained output
fields, checks the timing identity, the invocation grid, the gates against the reference
files, the exit bookkeeping, and — for a solve job — re-derives the non-dominated sets and
the pre-registered D1–D3 criterion from the invocation rows alone.

    python audit_np.py <archive-dir> [--floor]   # --floor also rebuilds the bank floor
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'p-bank-head'))
import pbh_audit_np as N  # noqa: E402


def non_dominated(points):
    keep = []
    for i, (e, c) in enumerate(points):
        if not any((e2 <= e and c2 <= c) and (e2 < e or c2 < c)
                   for j, (e2, c2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def summarise(inv, subjects):
    rows = {}
    for name in subjects:
        sel = [x for x in inv if x['name'] == name]
        cases = sorted({x['case'] for x in sel})
        worst = max(max(x['same_grid_error'] for x in sel if x['case'] == c) for c in cases)
        med = float(np.median([x['same_grid_error'] for x in sel]))
        cost = float(np.median([x['total_seconds'] for x in sel])) * 1e3
        rows[name] = dict(worst=worst, median=med, cost_ms=cost, cases=len(cases))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive', type=Path)
    ap.add_argument('--floor', action='store_true')
    a = ap.parse_args()
    root = a.archive
    outdir = root / 'output'
    d = json.loads((outdir / 'result.json').read_text())
    assert d['complete']
    n = int(d.get('intervals') or d['config']['solve_intervals'])
    checks = {}

    checks['backend'] = dict(passed=bool(d['backend'] == 'gpu' and d['x64'] and
                                         d['matmul_precision'] == 'highest' and not d['smoke']),
                             backend=d['backend'], gpu=d['gpu'], job=d['job_id'])

    cfg = d['config']
    dev = np.concatenate((N.source_params(cfg['eval_seed'], cfg['eval_count']),
                          N.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    rec = np.asarray(d['cohort']['parameters']) if 'cohort' in d else None
    checks['cohort_parameters'] = dict(passed=bool(rec is None or np.allclose(rec, dev, rtol=0, atol=1e-15)),
                                       max_abs_difference=float(np.max(np.abs(rec - dev))) if rec is not None else None)

    nhi = cfg['reference_intervals'][-1]
    same = {c: N.solve(n, p) for c, p in enumerate(dev)}
    chain = {c: N.restrict(N.solve(nhi, p), nhi, n) for c, p in enumerate(dev)}
    worst_p = worst_s = 0.0
    seen = {}
    for x in d['invocations']:
        f = seen.get(x['artifact'])
        if f is None:
            f = np.load(outdir / x['artifact'])['field']
            seen[x['artifact']] = f
        assert hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() == x['field_sha256'], x['artifact']
        worst_p = max(worst_p, abs(N.relative(f, chain[x['case']]) - x['physical_error']))
        worst_s = max(worst_s, abs(N.relative(f, same[x['case']]) - x['same_grid_error']))
    checks['recomputed_errors'] = dict(passed=bool(worst_p < 1e-12 and worst_s < 1e-12),
                                       invocations=len(d['invocations']), distinct_fields=len(seen),
                                       worst_physical_difference=worst_p, worst_same_grid_difference=worst_s)

    tid = max(abs(x['total_seconds'] - (x['input_seconds'] + x.get('fused_device_seconds',
              x.get('projection_init_seconds', 0) + x.get('solver_seconds', 0)) + x['output_seconds']))
              for x in d['invocations'])
    checks['timing_identity'] = dict(passed=bool(tid < 1e-6), worst_seconds=tid)

    names = sorted({x['name'] for x in d['invocations']})
    reps = cfg['repetitions']
    grid_ok = all(sum(1 for x in d['invocations'] if x['name'] == nm) == reps * len(dev) for nm in names)
    declared = [s['name'] for s in d['declared_subjects'] if not s.get('skipped')]
    checks['invocation_grid'] = dict(passed=bool(grid_ok and sorted(declared) == names),
                                     subjects=len(names), repetitions=reps, cases=len(dev))

    gate_ok, gate_rows = True, []
    for g in cfg['gates']:
        if n not in g['intervals']:
            continue
        ref_path = HERE / g['file']
        if not ref_path.exists():
            ref_path = root / 'code' / Path(g['file']).name
        ref = json.loads(ref_path.read_text())
        for ours, theirs in g['pairs']:
            entry = next((e for e in ref['arms'] if e['intervals'] == n and e['name'] == theirs), None)
            if entry is None:
                continue
            got = {}
            for x in d['invocations']:
                if x['name'] == ours:
                    got.setdefault(x['case'], np.load(outdir / x['artifact'])['field'])
            worst = max(abs(N.relative(got[c], chain[c]) - e) / max(abs(e), 1e-300)
                        for c, e in zip(entry['cases'], entry['physical_error']) if c in got)
            ok = worst <= cfg['fidelity_tolerance']
            gate_ok &= ok
            gate_rows.append(dict(ours=ours, theirs=theirs, worst=worst, passed=bool(ok)))
    checks['fidelity_gates_recomputed'] = dict(passed=bool(gate_ok), rows=gate_rows,
                                               driver_gates_all_passed=all(g['passed'] for g in d['gates']))

    bad = 0
    for x in d['invocations']:
        if x['kind'] in ('pa', 'rom', 'linear'):
            bad += int((x['reason'] == 4) != (x['stationarity'] <= cfg['stationarity_tolerance']))
        if x['kind'] == 'ladder':
            bad += int(x['stationary'] != (x['stationarity'] <= cfg['stationarity_tolerance']))
        if x['kind'] == 'cg':
            bad += int(x['cg_converged'] != (x['true_relative_residual'] <= x['cg_tolerance'] * (1 + 1e-6) + 1e-12))
    checks['exit_bookkeeping'] = dict(passed=bool(bad == 0), inconsistent_rows=bad)

    if 'ladder_q' in cfg:
        rows = summarise(d['invocations'], names)
        prim = cfg['models'][0]['id']
        ladder = [f'q{q}_m4@{prim}' for q in cfg['ladder_q'] if f'q{q}_m4@{prim}' in rows]
        costs = [rows[k]['cost_ms'] for k in ladder]
        errs = [rows[k]['worst'] for k in ladder]
        top = ladder[-1]
        d1 = max(costs) / min(costs)
        d2_low = rows[top]['worst'] <= min(errs) + 1e-15
        d2_cost = rows[top]['cost_ms'] <= 1.1 * min(costs)
        d2_strict = rows[top]['cost_ms'] <= min(costs)
        fom = [k for k in names if k == 'dst_direct' or k.startswith('cg_')]
        allp = [(rows[k]['worst'], rows[k]['cost_ms']) for k in names]
        nd_all = [names[i] for i in non_dominated(allp)]
        red = [k for k in names if k not in fom]
        nd_red = [red[i] for i in non_dominated([(rows[k]['worst'], rows[k]['cost_ms']) for k in red])]
        d3 = any(k in fom for k in nd_all) and any(k.startswith('e_pod') for k in nd_red)
        mono = all(errs[i + 1] <= errs[i] + 1e-15 for i in range(len(errs) - 1))
        checks['criterion'] = dict(ladder=ladder, worst_same_grid=errs, cost_ms=costs,
                                   D1_cost_span=d1, D1=bool(d1 < 2), D2_top_lowest_error=bool(d2_low),
                                   D2_top_within_1p1=bool(d2_cost), D2=bool(d2_low and d2_cost),
                                   D2_strict_top_cheapest=bool(d2_strict), D3=bool(d3),
                                   degenerate=bool(d1 < 2 and d2_low and d2_cost and d3),
                                   error_monotone=bool(mono), non_dominated_all=nd_all,
                                   non_dominated_reduced=nd_red, passed=True)

    if a.floor and 'ladder_q' in cfg:
        m = cfg['models'][0]
        ck = root / 'code' / Path(m['checkpoint']).name
        params, Z, _ = N.load(ck)
        G = N.features(params, N.coords(n))
        Q, R = np.linalg.qr(G)
        del G
        floors = []
        for c in range(len(dev)):
            u = same[c][1:-1, 1:-1].ravel()
            floors.append(float(np.linalg.norm(u - Q @ (Q.T @ u)) / np.linalg.norm(u)))
        recd = next(r for r in d['reconstruction'] if r['model'] == m['id'])
        diff = float(np.max(np.abs(np.asarray(floors) - np.asarray(recd['bank_projection']['per_case']))))
        checks['numpy_bank_floor'] = dict(passed=bool(diff < 1e-9), max_abs_difference=diff,
                                          worst=float(max(floors)))

    checks['all_passed'] = all(v['passed'] for k, v in checks.items() if isinstance(v, dict))
    (root.parent / 'audit.json').write_text(json.dumps(checks, indent=2) + '\n')
    for k, v in checks.items():
        if isinstance(v, dict):
            print(k, 'PASS' if v['passed'] else 'FAIL', {kk: vv for kk, vv in v.items()
                                                          if kk not in ('rows', 'passed') and not isinstance(vv, list)})
    print('ALL', checks['all_passed'])


if __name__ == '__main__':
    main()
