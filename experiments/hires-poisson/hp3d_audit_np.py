"""Independent NumPy/SciPy audit of one hires-poisson 3D attempt. Imports no JAX and no driver.

    python hp3d_audit_np.py <output-dir> [--subsample 32] [--delete-fields]

Regenerates the cohort and forcings from the seed with its own code, solves the seven-point
problem by SciPy DST-I on n and 2n, verifies both solves by the stencil residual, recomputes
both errors for every saved field, then derives the per-mesh table, the protocol selections
and the bar verdict (accurate arm q=96, fast arm q=0, variant `leandst64`).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dstn

HEADLINE = 'leandst64'
VARIANTS = ('retained', 'lean64', 'leandst64', 'leandst32', 'onestart64')
PARITY = dict(lean64=1e-12, leandst64=1e-10, leandst32=1e-4)


def family(seed, count):
    a = np.random.default_rng(seed).random((count, 5))
    return a * np.array([.3, .3, .3, .05, .4]) + np.array([.35, .35, .35, .10, .8])


def source(n, p):
    a = np.arange(1, n, dtype=np.float64) / n
    X = np.stack(np.meshgrid(a, a, a, indexing='ij'), -1)
    mask = 64 * np.prod(X * (1 - X), axis=-1)
    return p[4] * mask * np.exp(-np.sum((X - p[:3]) ** 2, axis=-1) / (2 * p[3] ** 2))


def solve(n, f):
    k = np.arange(1, n)
    l = 4. * n * n * np.sin(np.pi * k / (2 * n)) ** 2
    lam = l[:, None, None] + l[None, :, None] + l[None, None, :]
    return dstn(dstn(f, type=1, norm='ortho', workers=8) / lam, type=1, norm='ortho', workers=8)


def stencil_residual(n, u, f):
    p = np.pad(u, 1)
    au = n * n * (6 * u - p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1] - p[1:-1, 2:, 1:-1]
                  - p[1:-1, :-2, 1:-1] - p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2])
    return float(np.linalg.norm(au - f) / np.linalg.norm(f))


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--subsample', type=int, default=32)
    ap.add_argument('--delete-fields', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    R = json.loads((out / 'result.json').read_text())
    assert R['complete'] and R['backend'] == 'gpu' and R['x64'] and R['matmul_precision'] == 'highest'
    cfg = R['config']
    dev = family(cfg['cohort_seed'], cfg['cohort_draw'])[:cfg['case_count']]
    seed_match = float(np.max(np.abs(dev - np.asarray(R['cohort']['parameters']))))
    assert seed_match < 1e-12
    assert R['checkpoint_sha256'] == cfg['accepted_checkpoint_sha256'], 'not the accepted checkpoint'
    (out / 'sub').mkdir(exist_ok=True)
    parity_np = []
    inv = R['invocations']
    checks, worst_diff, refres, meshes = 0, 0.0, [], {}
    for n in cfg['meshes']:
        stride = max(1, n // a.subsample)
        rec = {}
        for case, p in enumerate(dev):
            f = source(n, p)
            same = solve(n, f)
            f2 = source(2 * n, p)
            fine2 = solve(2 * n, f2)
            rr = dict(intervals=n, case=case, same=stencil_residual(n, same, f),
                      fine=stencil_residual(2 * n, fine2, f2))
            assert rr['same'] < 1e-8 and rr['fine'] < 1e-8, rr
            refres.append(rr)
            fine = fine2[1::2, 1::2, 1::2]
            del f2, fine2
            for name in sorted({x['name'] for x in inv if x['intervals'] == n and x['case'] == case}):
                rows = [x for x in inv if x['intervals'] == n and x['case'] == case and x['name'] == name]
                u = np.load(out / 'fields' / rows[0]['saved_field'])
                h = hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest()
                assert rows[0]['field_sha256'] == h
                assert u.shape == (n - 1,) * 3 and u.dtype == np.float64 and np.isfinite(u).all()
                es, ep = rel(u, same), rel(u, fine)
                for x in rows:
                    assert x['matches_saved_field'] == (x['field_sha256'] == h)
                    if x['field_sha256'] == h:
                        d = max(abs(es - x['same_grid_error']), abs(ep - x['physical_error']))
                        assert d <= 1e-9 * max(es, ep) + 1e-11, (name, case, es, ep)
                        worst_diff = max(worst_diff, d)
                        checks += 2
                rec[(name, case)] = (es, ep)
                np.save(out / 'sub' / f'N{n}_{name}_case{case}.npy', u[::stride, ::stride, ::stride])
        # parity recomputed here from the saved fields, not read from the driver
        for q in cfg['q_ladder']:
            for v, limit in PARITY.items():
                worst = 0.0
                for case in range(len(dev)):
                    fa = next(x for x in inv if x['intervals'] == n and x['case'] == case and x['name'] == f'rom_q{q}_{v}')
                    fb = next(x for x in inv if x['intervals'] == n and x['case'] == case and x['name'] == f'rom_q{q}_retained')
                    worst = max(worst, rel(np.load(out / 'fields' / fa['saved_field']),
                                           np.load(out / 'fields' / fb['saved_field'])))
                parity_np.append(dict(intervals=n, q=q, variant=v, worst_field_relative=worst, limit=limit,
                                      passed=bool(worst <= limit)))
        table = {}
        for name in sorted({x['name'] for x in inv if x['intervals'] == n}):
            rows = [x for x in inv if x['intervals'] == n and x['name'] == name]
            cases = sorted({x['case'] for x in rows})
            table[name] = dict(
                family=rows[0]['family'], q=rows[0]['q'], tolerance=rows[0]['tolerance'],
                coarse_intervals=rows[0]['coarse_intervals'], cases=len(cases),
                repetitions_per_case=min(sum(1 for x in rows if x['case'] == c) for c in cases),
                worst_same_grid=max(rec[(name, c)][0] for c in cases),
                median_same_grid=float(np.median([rec[(name, c)][0] for c in cases])),
                worst_physical=max(rec[(name, c)][1] for c in cases),
                median_total_ms=1e3 * float(np.median([x['total_seconds'] for x in rows])),
                median_device_ms=1e3 * float(np.median([x['fused_device_seconds'] for x in rows])),
                median_input_ms=1e3 * float(np.median([x['input_seconds'] for x in rows])),
                median_output_ms=1e3 * float(np.median([x['output_seconds'] for x in rows])),
                median_iterations=(float(np.median([x['iterations'] for x in rows])) if 'iterations' in rows[0] else None),
                median_lm_attempts=(float(np.median([x['attempts'] for x in rows])) if 'attempts' in rows[0] else None))
        required = [f'rom_q{q}_{v}' for q in cfg['q_ladder'] for v in VARIANTS] + \
                   [f"rom_q{R['R']}_linear", 'dst_direct'] + [f'cg_{t:g}' for t in cfg['cg_tolerances']] + \
                   [f'coarse{nc}_dst' for nc in cfg['coarse_intervals'] if nc < n] + \
                   [f'coarse{nc}_cg_{t:g}' for nc in cfg['coarse_intervals'] if nc < n for t in cfg['coarse_cg_tolerances']]
        want = {(c, r) for c in range(len(dev)) for r in range(cfg['repetitions'])}
        coverage = all({(x['case'], x['rep']) for x in inv if x['intervals'] == n and x['name'] == name} == want
                       for name in required)
        valid = {name for name in table
                 if all(x.get('stationary', True) and x.get('cg_converged', True)
                        for x in inv if x['intervals'] == n and x['name'] == name)}

        def pick(fams, bound):
            ok = [k for k, v in table.items() if v['family'] in fams and k in valid and v['worst_physical'] <= bound]
            return min(ok, key=lambda k: table[k]['median_total_ms']) if ok else None
        selections = []
        for name, r in table.items():
            if r['family'] not in ('nm-rom', 'linear-rom'):
                continue
            row = dict(rom=name, q=r['q'], worst_same_grid=r['worst_same_grid'],
                       worst_physical=r['worst_physical'], median_total_ms=r['median_total_ms'],
                       median_device_ms=r['median_device_ms'])
            for label, k in (('named_cg_1e-2', 'cg_0.01'), ('fastest_cg_matched', pick(('cg',), r['worst_physical'])),
                             ('fastest_coarse_matched', pick(('coarse-dst', 'coarse-cg'), r['worst_physical'])),
                             ('dst_direct', 'dst_direct')):
                row[label] = None if k is None or k not in table else dict(
                    comparator=k, worst_physical=table[k]['worst_physical'],
                    median_total_ms=table[k]['median_total_ms'],
                    speedup_total=table[k]['median_total_ms'] / r['median_total_ms'],
                    speedup_device=table[k]['median_device_ms'] / r['median_device_ms'])
            selections.append(row)
        head = next(x for x in selections if x['rom'] == f'rom_q96_{HEADLINE}')
        verdict = dict(arm=head['rom'], comparator='cg_0.01', worst_same_grid=head['worst_same_grid'],
                       speedup_total=head['named_cg_1e-2']['speedup_total'],
                       accuracy_bar_1pct=bool(head['worst_same_grid'] <= 0.01),
                       stretch_bar_0p5pct=bool(head['worst_same_grid'] <= 0.005),
                       speed_bar_5x=bool(head['named_cg_1e-2']['speedup_total'] >= 5.0))
        verdict['bar_met'] = bool(verdict['accuracy_bar_1pct'] and verdict['speed_bar_5x'])
        fast = next(x for x in selections if x['rom'] == f'rom_q0_{HEADLINE}')
        verdict['fast_arm'] = dict(arm=fast['rom'], worst_same_grid=fast['worst_same_grid'],
                                   speedup_total=fast['named_cg_1e-2']['speedup_total'])
        meshes[str(n)] = dict(coverage=coverage, verdict=verdict, selections=selections, table=table,
                              mesh=next(m for m in R['meshes'] if m['intervals'] == n))
        print('VERDICT', n, verdict, flush=True)
    unstable = sorted({(x['intervals'], x['name']) for x in inv if not x['matches_saved_field']})
    gates = dict(driver_gates=bool(R.get('gates')) and all(R['gates'].values()),
                 final_uuid_matches=R['device_guard_final_uuid'] == R['gpu_uuid'],
                 deterministic=not unstable, coverage=all(m['coverage'] for m in meshes.values()),
                 parity_recomputed=bool(parity_np) and all(p['passed'] for p in parity_np))
    audit = dict(passed=all(gates.values()), gates=gates, job_id=R['job_id'], commit=R['commit'],
                 gpu=R['gpu'], gpu_uuid=R['gpu_uuid'], error_checks=checks, worst_error_difference=worst_diff,
                 reference_checks=refres, nondeterministic_subjects=unstable, parity=parity_np,
                 driver_parity=R['parity'],
                 device_memory=R.get('device_memory'), headline_variant=HEADLINE, meshes=meshes,
                 result_sha256=hashlib.sha256((out / 'result.json').read_bytes()).hexdigest())
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if a.delete_fields and audit['passed']:          # failed gates keep the evidence
        for p in (out / 'fields').glob('*.npy'):
            p.unlink()
        (out / 'fields').rmdir()
    print('AUDIT', 'PASSED' if audit['passed'] else 'FAILED', gates, checks, worst_diff, flush=True)


if __name__ == '__main__':
    main()
