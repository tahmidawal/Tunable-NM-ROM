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
    assert R['matmul_precision'] == 'highest'
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
            assert rows[0]['field_sha256'] == h, (name, case)
            for x in rows:                      # recorded flag must agree with the real hash
                assert x['matches_saved_field'] == (x['field_sha256'] == h), (name, case)
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
    # ---- coverage: every declared subject, every case it was allowed, >= 5 repetitions ----
    limit = cfg.get('slow_subject_cases', {})
    coverage = []
    for name in R['declared_subjects']:
        want = min(limit.get(name, len(dev)), len(dev))
        got = table.get(name, dict(cases=0, repetitions_per_case=0))
        coverage.append(dict(name=name, cases=got['cases'], expected_cases=want,
                             repetitions=got['repetitions_per_case'],
                             ok=bool(got['cases'] == want
                                     and got['repetitions_per_case'] >= cfg['repetitions'])))

    # ---- protocol selections, on the FULL cohort only (restricted subjects are ineligible) ----
    full = {k: v for k, v in table.items() if v['cases'] == len(dev)}
    lean = 'lean64' if any(k.endswith('lean64') for k in full) else 'lean32'
    Rw = R['checkpoint']['R']

    def pick(names, bound):
        ok = [k for k in names if full[k]['worst_physical'] <= bound]
        return min(ok, key=lambda k: full[k]['median_total_ms']) if ok else None

    selections = []
    for suffix in ('', '-io32'):                 # f64-I/O contract, then the labelled f32-I/O one
        tail = '_io32' if suffix else ''
        for name in [k for k in full if full[k]['family'] in ('nm-rom' + suffix, 'linear-rom' + suffix)]:
            r = full[name]
            row = dict(rom=name, q=r['q'], io_contract='f32' if suffix else 'f64',
                       worst_same_grid=r['worst_same_grid'], worst_physical=r['worst_physical'],
                       median_total_ms=r['median_total_ms'], median_device_ms=r['median_device_ms'])
            for label, fam in (('named_cg_1e-2', None), ('fastest_cg_matched', ('cg' + suffix,)),
                               ('fastest_coarse_matched', ('coarse-dst' + suffix, 'coarse-cg' + suffix)),
                               ('dst_direct', None)):
                if label == 'named_cg_1e-2':
                    k = 'cg_0.01' + tail if 'cg_0.01' + tail in full else None
                elif label == 'dst_direct':
                    k = 'dst_direct' + tail if 'dst_direct' + tail in full else None
                else:
                    k = pick([c for c in full if full[c]['family'] in fam], r['worst_physical'])
                row[label] = None if k is None else dict(
                    comparator=k, worst_physical=full[k]['worst_physical'],
                    median_total_ms=full[k]['median_total_ms'],
                    speedup_total=full[k]['median_total_ms'] / r['median_total_ms'],
                    speedup_device=full[k]['median_device_ms'] / r['median_device_ms'])
            selections.append(row)
    head = next((x for x in selections if x['rom'] == f'rom_q256_{lean}'), None)
    verdict = None
    if head is not None and head['named_cg_1e-2'] is not None:
        verdict = dict(arm=head['rom'], comparator='cg_0.01',
                       worst_same_grid=head['worst_same_grid'],
                       speedup_total=head['named_cg_1e-2']['speedup_total'],
                       accuracy_bar_1pct=bool(head['worst_same_grid'] <= 0.01),
                       stretch_bar_0p5pct=bool(head['worst_same_grid'] <= 0.005),
                       speed_bar_5x=bool(head['named_cg_1e-2']['speedup_total'] >= 5.0))
        verdict['bar_met'] = bool(verdict['accuracy_bar_1pct'] and verdict['speed_bar_5x'])

    gates = dict(driver_gates=bool(R.get('gates')) and all(R['gates'].values()),
                 final_uuid_matches=R['device_guard_final_uuid'] == R['gpu_uuid'],
                 deterministic=not unstable,
                 coverage=bool(coverage) and all(c['ok'] for c in coverage),
                 parity_present_and_passed=bool(R['parity']) and all(p['passed'] for p in R['parity'] if p['passed'] is not None),
                 diagnostics_present_and_valid=bool(R['diagnostics'])
                                               and all(d['valid'] for d in R['diagnostics']),
                 verdict_computable=verdict is not None)
    audit = dict(passed=all(gates.values()), gates=gates, intervals=n, job_id=R['job_id'],
                 commit=R['commit'], gpu=R['gpu'], gpu_uuid=R['gpu_uuid'],
                 cohort_seed_regeneration_max_abs=seed_match, error_checks=checks,
                 worst_error_difference=worst_diff, reference_checks=refres,
                 nondeterministic_subjects=unstable, subsample_stride=stride,
                 coverage=coverage, parity=R['parity'], bank=R['bank'],
                 assembly_gate=R.get('assembly_gate'), device_memory=R.get('device_memory'),
                 headline_variant=lean, verdict=verdict, selections=selections, table=table,
                 result_sha256=hashlib.sha256((out / 'result.json').read_bytes()).hexdigest())
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if a.delete_fields and audit['passed']:          # failed gates keep the evidence
        for p in (out / 'fields').glob('*.npy'):
            p.unlink()
        (out / 'fields').rmdir()
    print('AUDIT', 'PASSED' if audit['passed'] else 'FAILED', gates, checks,
          'checks; worst difference', worst_diff, flush=True)
    print('VERDICT', verdict, flush=True)


if __name__ == '__main__':
    main()
