"""Independent NumPy audit of one hires-poisson L-shape attempt. Imports no JAX, no driver and
not `lsh_core`.

References are NOT re-solved (a second sparse factorisation of the 2n mesh costs what the job
costs); they are VERIFIED: the audit regenerates every source from the seed with its own code
and applies its own matrix-free masked five-point stencil to the saved same-grid and 2n
reference fields. A small residual certifies the unique discrete solution up to
cond(A) * residual, which is recorded. Then every saved field's two errors are recomputed and
compared with every recorded row, and the table / selections / verdict are derived.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HEAD_MODEL = 'head_sdf_R512_K16'


def params(seed, count):
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, count); cy = rng.uniform(0.15, 0.85, count)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), count)); a = rng.uniform(0.5, 2.0, count)
    d = np.column_stack((cx, cy, w, a))
    return d[~((d[:, 0] >= 0.5) & (d[:, 1] >= 0.5))]


def mask_of(n):
    m = np.zeros((n + 1, n + 1), bool)
    m[1:n, 1:n] = True
    m[n // 2:, n // 2:] = False
    return m


def source(n, p, m):
    x = np.linspace(0., 1., n + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.where(m, p[3] * np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2) / (2 * p[2] ** 2)), 0.)


def residual(n, u, f, m):
    assert not u[~m].any(), 'reference is non-zero outside the open domain'
    p = np.pad(u, 1)
    au = n * n * (4 * u - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:])
    return float(np.linalg.norm((au - f)[m]) / np.linalg.norm(f[m]))


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--subsample', type=int, default=256)
    ap.add_argument('--delete-fields', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    R = json.loads((out / 'result.json').read_text())
    assert R['complete'] and R['backend'] == 'gpu' and R['x64'] and R['matmul_precision'] == 'highest'
    cfg, n = R['config'], R['intervals']
    dev = params(cfg['cohorts']['development']['seed'], cfg['cohorts']['development']['draw'])[:cfg['case_count']]
    assert float(np.max(np.abs(dev - np.asarray(R['cohort']['parameters'])))) < 1e-12
    m, m2 = mask_of(n), mask_of(2 * n)
    stride = max(1, n // a.subsample)
    (out / 'sub').mkdir(exist_ok=True)
    inv = R['invocations']
    rec, refs, checks, worst_diff = {}, [], 0, 0.0
    for case, p in enumerate(dev):
        same = np.load(out / 'fields' / f'reference_same_case{case}.npy')
        full = np.load(out / 'fields' / f'reference_fine_full_case{case}.npy')
        fine = np.load(out / 'fields' / f'reference_fine_case{case}.npy')
        assert np.array_equal(full[::2, ::2], fine)
        r = dict(case=case, same_residual=residual(n, same, source(n, p, m), m),
                 fine_residual=residual(2 * n, full, source(2 * n, p, m2), m2))
        # round-off floor of the evaluation grows like N^2 (lshape DESIGN A7): eps*8N^2*|u|/|f|
        r['same_floor'] = float(np.finfo(float).eps * 8 * n * n * np.linalg.norm(same) / np.linalg.norm(source(n, p, m)))
        r['fine_floor'] = float(np.finfo(float).eps * 32 * n * n * np.linalg.norm(full) / np.linalg.norm(source(2 * n, p, m2)))
        assert r['same_residual'] <= max(1e-11, 2 * r['same_floor']), r
        assert r['fine_residual'] <= max(1e-11, 2 * r['fine_floor']), r
        refs.append(r)
        del full
        for name in sorted({x['name'] for x in inv if x['case'] == case}):
            rows = [x for x in inv if x['case'] == case and x['name'] == name]
            u = np.load(out / 'fields' / rows[0]['saved_field'])
            h = hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest()
            assert rows[0]['field_sha256'] == h
            assert u.shape == (n + 1, n + 1) and np.isfinite(u).all() and not u[~m].any()
            es, ep = rel(u, same), rel(u, fine)
            for x in rows:
                assert x['matches_saved_field'] == (x['field_sha256'] == h)
                if x['field_sha256'] == h:
                    d = max(abs(es - x['same_grid_error']), abs(ep - x['physical_error']))
                    assert d <= 1e-9 * max(es, ep) + 1e-11, (name, case)
                    worst_diff = max(worst_diff, d); checks += 2
            rec[(name, case)] = (es, ep)
            np.save(out / 'sub' / f"{rows[0]['saved_field'][:-4]}.npy", u[::stride, ::stride])
    table = {}
    for name in sorted({x['name'] for x in inv}):
        rows = [x for x in inv if x['name'] == name]
        cases = sorted({x['case'] for x in rows})
        table[name] = dict(
            family=rows[0]['family'], q=rows[0]['q'], model=rows[0]['model'], tolerance=rows[0]['tolerance'],
            coarse_intervals=rows[0]['coarse_intervals'], where=rows[0].get('where', 'gpu'), cases=len(cases),
            repetitions_per_case=min(sum(1 for x in rows if x['case'] == c) for c in cases),
            worst_same_grid=max(rec[(name, c)][0] for c in cases),
            median_same_grid=float(np.median([rec[(name, c)][0] for c in cases])),
            worst_physical=max(rec[(name, c)][1] for c in cases),
            median_total_ms=1e3 * float(np.median([x['total_seconds'] for x in rows])),
            median_device_ms=1e3 * float(np.median([x['fused_device_seconds'] for x in rows])),
            median_input_ms=1e3 * float(np.median([x['input_seconds'] for x in rows])),
            median_output_ms=1e3 * float(np.median([x['output_seconds'] for x in rows])),
            median_iterations=(float(np.median([x['iterations'] for x in rows])) if 'iterations' in rows[0] and rows[0]['family'] != 'nm-rom' else None),
            median_lm_attempts=(float(np.median([x['iterations'] for x in rows])) if rows[0]['family'] == 'nm-rom' else None),
            valid=all(x.get('stationary', True) and x.get('converged', True) for x in rows))
    want = {(c, r) for c in range(len(dev)) for r in range(cfg['repetitions'])}
    coverage = all({(x['case'], x['rep']) for x in inv if x['name'] == name} == want for name in R['declared_subjects'])

    def pick(fams, bound):
        ok = [k for k, v in table.items() if v['family'] in fams and v['valid'] and v['worst_physical'] <= bound]
        return min(ok, key=lambda k: table[k]['median_total_ms']) if ok else None
    selections = []
    for name, r in table.items():
        if r['family'] != 'nm-rom':
            continue
        row = dict(rom=name, q=r['q'], worst_same_grid=r['worst_same_grid'], worst_physical=r['worst_physical'],
                   median_total_ms=r['median_total_ms'], median_device_ms=r['median_device_ms'])
        for label, k in (('named_cg_1e-2', 'cg_0.01'), ('fastest_cg_matched', pick(('cg',), r['worst_physical'])),
                         ('fastest_coarse_matched', pick(('coarse-direct', 'coarse-cg'), r['worst_physical'])),
                         ('dst_direct', 'fom_splu_cpu')):
            row[label] = None if k is None else dict(
                comparator=k, worst_physical=table[k]['worst_physical'], median_total_ms=table[k]['median_total_ms'],
                speedup_total=table[k]['median_total_ms'] / r['median_total_ms'],
                speedup_device=table[k]['median_device_ms'] / r['median_device_ms'])
        selections.append(row)
    qtop = max(cfg['correction_ladder'])
    head = next(x for x in selections if x['rom'] == f'rom_q{qtop}_lean@{HEAD_MODEL}')
    fast = next(x for x in selections if x['rom'] == f'rom_q0_lean@{HEAD_MODEL}')
    verdict = dict(arm=head['rom'], comparator='cg_0.01', worst_same_grid=head['worst_same_grid'],
                   speedup_total=head['named_cg_1e-2']['speedup_total'],
                   accuracy_bar_1pct=bool(head['worst_same_grid'] <= 0.01),
                   stretch_bar_0p5pct=bool(head['worst_same_grid'] <= 0.005),
                   speed_bar_5x=bool(head['named_cg_1e-2']['speedup_total'] >= 5.0),
                   fast_arm=dict(arm=fast['rom'], worst_same_grid=fast['worst_same_grid'],
                                 speedup_total=fast['named_cg_1e-2']['speedup_total']))
    verdict['bar_met'] = bool(verdict['accuracy_bar_1pct'] and verdict['speed_bar_5x'])
    gates = dict(driver_gates=all(R['driver_gates'].values()), coverage=coverage,
                 final_uuid_matches=R['device_guard_final_uuid'] == R['gpu_uuid'],
                 parity=bool(R['parity']) and all(p['passed'] for p in R['parity']))
    audit = dict(passed=all(gates.values()), gates=gates, intervals=n, job_id=R['job_id'], commit=R['commit'],
                 gpu=R['gpu'], gpu_uuid=R['gpu_uuid'], error_checks=checks, worst_error_difference=worst_diff,
                 reference_checks=refs, parity=R['parity'], arm_setup=R['arm_setup'], fom=R['fom'],
                 fine_reference=R['fine_reference'], device_memory=R.get('device_memory'),
                 headline_model=HEAD_MODEL, verdict=verdict, selections=selections, table=table,
                 result_sha256=hashlib.sha256((out / 'result.json').read_bytes()).hexdigest())
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if a.delete_fields and audit['passed']:
        for p in (out / 'fields').glob('*.npy'):
            p.unlink()
        (out / 'fields').rmdir()
    print('AUDIT', 'PASSED' if audit['passed'] else 'FAILED', gates, checks, worst_diff, flush=True)
    print('VERDICT', verdict, flush=True)


if __name__ == '__main__':
    main()
