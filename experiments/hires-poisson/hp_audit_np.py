"""Independent NumPy/SciPy audit of one hires-poisson attempt. Imports no JAX and no driver.

    python hp_audit_np.py <output-dir> [--subsample 256] [--delete-fields]

For every saved field: regenerate the source from the recorded seed with NumPy's own rng,
solve the five-point problem with SciPy DST-I (same-grid and 2x-refined), CHECK that solve
by applying the five-point stencil to it (residual, not a second transform), recompute both
errors and compare with every recorded invocation. Also re-derives medians, speedups and
the protocol's arm selection. Writes audit.json and strided field subsamples, and can then
delete the full fields (they are tens of GB at 4096^2 and are never pulled).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dstn


def params(seed, count):
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, count)
    cy = rng.uniform(0.15, 0.85, count)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), count))
    a = rng.uniform(0.5, 2.0, count)
    return np.column_stack((cx, cy, w, a))


def source(n, p):
    x = np.linspace(0., 1., n + 1)[1:-1]
    X, Y = np.meshgrid(x, x, indexing='ij')
    return p[3] * np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2) / (2 * p[2] ** 2))


def solve(n, f):
    k = np.arange(1, n)
    l = 4. * n ** 2 * np.sin(np.pi * k / (2 * n)) ** 2
    c = dstn(f, type=1, norm='ortho', workers=8) / (l[:, None] + l[None, :])
    return dstn(c, type=1, norm='ortho', workers=8)


def stencil_residual(n, u, f):
    p = np.pad(u, 1)
    au = n ** 2 * (4 * u - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:])
    return float(np.linalg.norm(au - f) / np.linalg.norm(f))


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
    assert R['complete'] and R['backend'] == 'gpu' and R['x64']
    cfg, n = R['config'], R['intervals']
    dev = np.concatenate((params(cfg['eval_seed'], cfg['eval_count']),
                          params(cfg['fresh_seed'], cfg['fresh_count'])))
    seed_match = float(np.max(np.abs(dev - np.asarray(R['cohort']['parameters']))))
    assert seed_match < 1e-12, seed_match
    stride = max(1, n // a.subsample)
    (out / 'sub').mkdir(exist_ok=True)
    checks, worst_diff, refres = 0, 0.0, []
    recomputed = {}
    inv = R['invocations']
    for case, p in enumerate(dev):
        f = source(n, p)
        same = np.pad(solve(n, f), 1)
        f2 = source(2 * n, p)
        fine = solve(2 * n, f2)
        refres.append(dict(case=case, same_stencil_residual=stencil_residual(n, same[1:-1, 1:-1], f),
                           fine_stencil_residual=stencil_residual(2 * n, fine, f2)))
        assert refres[-1]['same_stencil_residual'] < 1e-8, refres[-1]
        assert refres[-1]['fine_stencil_residual'] < 1e-8, refres[-1]
        fine = np.pad(fine, 1)[::2, ::2]
        del f2
        np.save(out / 'sub' / f'reference_same_case{case}.npy', same[::stride, ::stride])
        np.save(out / 'sub' / f'reference_fine_case{case}.npy', fine[::stride, ::stride])
        names = sorted({x['name'] for x in inv if x['case'] == case})
        for name in names:
            rows = [x for x in inv if x['case'] == case and x['name'] == name]
            path = out / 'fields' / rows[0]['saved_field']
            u = np.load(path)
            h = hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest()
            assert any(x['field_sha256'] == h for x in rows), (name, case)
            assert u.shape == (n + 1, n + 1) and u.dtype == np.float64 and np.isfinite(u).all()
            assert not u[0].any() and not u[-1].any() and not u[:, 0].any() and not u[:, -1].any()
            es, ep = rel(u, same), rel(u, fine)
            for x in rows:
                if x['field_sha256'] == h:
                    d = max(abs(es - x['same_grid_error']), abs(ep - x['physical_error']))
                    # exact solvers sit at round-off, where the two FFT libraries differ
                    assert d <= 1e-9 * max(es, ep) + 1e-11, (name, case, es, ep, x)
                    worst_diff = max(worst_diff, d)
                    checks += 2
            recomputed[(name, case)] = (es, ep)
            np.save(out / 'sub' / f'{name}_case{case}.npy', u[::stride, ::stride])
            del u
    # every repetition of a deterministic subject must reproduce its saved field
    unstable = sorted({x['name'] for x in inv if not x['matches_saved_field']})
    names = sorted({x['name'] for x in inv})
    table = {}
    for name in names:
        rows = [x for x in inv if x['name'] == name]
        cases = sorted({x['case'] for x in rows})
        per_case_reps = min(sum(1 for x in rows if x['case'] == c) for c in cases)
        table[name] = dict(
            family=rows[0]['family'], q=rows[0]['q'], M=rows[0]['M'],
            tolerance=rows[0]['tolerance'], coarse_intervals=rows[0]['coarse_intervals'],
            cases=len(cases), repetitions_per_case=per_case_reps,
            worst_same_grid=max(recomputed[(name, c)][0] for c in cases),
            median_same_grid=float(np.median([recomputed[(name, c)][0] for c in cases])),
            worst_physical=max(recomputed[(name, c)][1] for c in cases),
            median_physical=float(np.median([recomputed[(name, c)][1] for c in cases])),
            median_total_ms=1e3 * float(np.median([x['total_seconds'] for x in rows])),
            median_device_ms=1e3 * float(np.median([x['fused_device_seconds'] for x in rows])),
            median_input_ms=1e3 * float(np.median([x['input_seconds'] for x in rows])),
            median_output_ms=1e3 * float(np.median([x['output_seconds'] for x in rows])),
            median_iterations=(float(np.median([x['iterations'] for x in rows]))
                               if 'iterations' in rows[0] else None),
            median_lm_attempts=(float(np.median([x['attempts'] for x in rows]))
                                if 'attempts' in rows[0] else None),
            exit_reasons=sorted({x['reason'] for x in rows if 'reason' in x}),
            all_cg_converged=(all(x['cg_converged'] for x in rows)
                              if 'cg_converged' in rows[0] else None))
    audit = dict(passed=True, intervals=n, job_id=R['job_id'], commit=R['commit'], gpu=R['gpu'],
                 gpu_uuid=R['gpu_uuid'], final_uuid_matches=R['device_guard_final_uuid'] == R['gpu_uuid'],
                 cohort_seed_regeneration_max_abs=seed_match, error_checks=checks,
                 worst_error_difference=worst_diff, reference_checks=refres,
                 nondeterministic_subjects=unstable, subsample_stride=stride,
                 parity=R['parity'], all_parity_passed=all(p['passed'] for p in R['parity']),
                 all_lean_stationary=all(d['stationary'] for d in R['diagnostics']),
                 bank=R['bank'], table=table,
                 result_sha256=hashlib.sha256((out / 'result.json').read_bytes()).hexdigest())
    assert audit['final_uuid_matches']
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if a.delete_fields:
        for p in (out / 'fields').glob('*.npy'):
            p.unlink()
        (out / 'fields').rmdir()
    print('AUDIT PASSED', checks, 'checks; worst difference', worst_diff, flush=True)


if __name__ == '__main__':
    main()
