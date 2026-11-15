"""linear-query-speed, heat 2D (PROTOCOL.md amendment A3): the paper's two heat 2D arms (`lin_R128_cn`,
`lin_R32_cn`, vendored unmodified in `paper_code_heat/`) next to the compressed-reconstruction query.

One process: the selection mesh first (development cohort only -> mechanical freeze, written and hashed),
then the test on every mesh.  Timing scope is the paper's heat scope: initial field on the device ->
six output fields on the device, blocked.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PC = HERE / 'paper_code_heat'
sys.path.insert(0, str(PC))
import core as C            # noqa: E402  (paper code)
import hbk_core as K        # noqa: E402  (paper code)
import lqs_core as Q        # noqa: E402

ARMS = (('accurate', 128), ('fast', 32))
TAUS = (1e-4, 1e-6, 1e-8)
PRECS = ('f64', 'f32')
GATE = 1e-5
REPRO_TOL = 1e-9


def vname(tau, prec):
    return f'T_tau{tau:g}_{prec}'


def smi():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        return f'unavailable: {exc}'


def timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t


def make_kernel(p, q, Rp, nt, dtype):
    """device f64 initial field (n-1, n-1) -> device f64 fields (nt, n-1, n-1)."""
    def coefs_of(u0, S, I, J, P6):
        return (P6 @ (S.T @ u0 @ S)[I, J]).reshape(nt, Rp)

    def cores_of(coefs, C2):
        return C2 @ coefs.T.astype(dtype)                       # (p*q, nt)

    def fields_of(cores, U, Vt):
        m = U.shape[0]
        W = (U @ cores.reshape(p, q * nt)).reshape(m, q, nt)    # U core_t for every t in one product
        W = jnp.transpose(W, (2, 0, 1)).reshape(nt * m, q)
        return (W @ Vt).reshape(nt, m, m).astype(jnp.float64)

    @jax.jit
    def kernel(u0, S, I, J, P6, C2, U, Vt):
        return fields_of(cores_of(coefs_of(u0, S, I, J, P6), C2), U, Vt)

    @jax.jit
    def solve_only(u0, S, I, J, P6):
        return coefs_of(u0, S, I, J, P6)

    stages = (jax.jit(coefs_of), jax.jit(cores_of), jax.jit(fields_of))
    return kernel, solve_only, stages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--meshes', default='4096 2048 1024')
    ap.add_argument('--select-mesh', type=int, default=4096)
    ap.add_argument('--frozen', default=None, help='use this frozen file (A3c); no selection is run')
    ap.add_argument('--smoke', action='store_true', help='CPU debugging: small cohort, no stored-value gate')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend()
    print(f'jax_backend={backend}', flush=True)
    if not a.smoke:
        assert backend == 'gpu' and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    meshes = [int(x) for x in a.meshes.split()]
    assert a.frozen or meshes[0] == a.select_mesh, 'the selection mesh must come first'
    cfg = json.loads((PC / 'configs' / 'h2d.json').read_text())
    stored = json.loads((HERE / 'paper_reference' / 'h2d-stored.json').read_text())
    model = C.load_model(cfg['model'], PC / 'inputs'); d = model['d']; assert d == 2
    assert a.smoke or model['sha256'] == stored['model_sha256'], 'checkpoint differs from the paper run'
    prep_sha = hashlib.sha256((PC / cfg['prep']).read_bytes()).hexdigest()
    assert prep_sha == stored['prep_sha256']
    prep = np.load(PC / cfg['prep']); T, L = prep['T'], prep['L']
    times = cfg['times']; nu = cfg['diffusivity']; nt = len(times); dt = cfg['rom_defaults']['dt']
    edges = list(cfg['edges'])
    cohorts, draws = [], []
    for name, seed, count in cfg['cohorts']:
        dr = C.family(model['family'], seed, count); cohorts += [name] * len(dr); draws.append(dr)
    draws = np.concatenate(draws)
    dev = [i for i, c in enumerate(cohorts) if c.startswith('validation')]
    test = [i for i, c in enumerate(cohorts) if c.startswith('heldout')]
    if a.smoke:
        dev, test = dev[:2], test[:2]
    reps = 2 if a.smoke else int(cfg['repetitions']); burn = 0.001 if a.smoke else float(cfg['burn_seconds'])
    prop = C.make_propagate(d); tjx = jnp.asarray(times)
    R_ = dict(schema='linear-query-speed-h2d-v1', complete=False, source_commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'), nvidia_smi=smi(),
              backend=backend, jax=jax.__version__, x64=True, matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
              smoke=a.smoke, arms=dict(ARMS), gate=GATE, repro_tolerance=REPRO_TOL, repetitions=reps, burn_seconds=burn,
              timing_contract='paper heat scope: initial field on the device -> six output fields on the device, '
                              'blocked; no host copies; burn before every invocation; random order within a case; '
                              'phases paperA1 / variants / paperA2',
              development_cases=dev, test_cases=test, stored_reference_sha256=stored['sha256'], meshes=[])
    frozen = None
    if a.frozen:
        text = Path(a.frozen).read_text(); frozen = json.loads(text)
        R_['frozen'] = frozen; R_['frozen_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        R_['frozen_source'] = a.frozen
        (out / 'frozen-h2d.json').write_text(text)
        print('FROZEN-FROM-FILE', frozen['F']['variant'], R_['frozen_sha256'], flush=True)

    def save():
        (out / 'result.json').write_text(json.dumps(R_))

    for n in meshes:
        begin = time.perf_counter(); m = n - 1
        M_ = dict(intervals=n, unknowns=m * m)
        nested, _ = K.build_bank(model, n, T, edges, False)
        modes = C.mode_list(cfg['tests'], n, d); lam_m = C.mode_eigs(n, modes)
        rtri = K.nested_tsqr_r(nested); a_rot = K.nested_weak_matrix(nested, modes, n, d)
        M_['bank_seconds'] = time.perf_counter() - begin
        M_['bank_bytes'] = int(sum(b.size for row in nested for b in row) * 8)
        # ---- offline compression (same routine, probes and seed as Poisson 2D)
        G = jnp.concatenate([jnp.concatenate(row, axis=1) for row in nested]) if len(nested) > 1 \
            else jnp.concatenate(nested[0], axis=1)
        xs = 512
        rot = tuple((G[lo * m:min(lo + xs, m) * m],) for lo in range(0, m, xs)); del G
        U0, V0, C0, col2, cinfo = Q.compress_bank(rot, n); del rot
        M_['compression'] = cinfo
        print('MESH', n, 'bank', round(M_['bank_seconds'], 1), 'compress', round(cinfo['seconds'], 1),
              cinfo['range_finder_relative_residual'], flush=True)
        kmax = int(modes.max()); I = jnp.asarray(modes[:, 0] - 1); J = jnp.asarray(modes[:, 1] - 1)
        S = jnp.asarray(np.sin(np.pi * np.outer(np.arange(1, n), np.arange(1, kmax + 1)) / n))
        lam_d = C.eig_grid(n, d)

        grid = [(tau, prec) for tau in TAUS for prec in PRECS]
        paper, variants, solve_only, profs = {}, {}, {}, {}
        for label, Rp in ARMS:
            nb = edges.index(Rp)
            proj = (lambda b, v, nb=nb: K.nproject(b, v, nb)); expd = (lambda c, b, nb=nb: K.nexpand(c, b, nb))
            setup = dict(n=n, d=d, times=times, nu=nu, modes=modes, a=a_rot[:, :Rp], rtri=rtri[:Rp, :Rp], mode_lam=lam_m,
                         directions=L[:Rp] @ np.asarray(model['directions']))
            lin = K.make_linear(setup, 'moments', 'cn', dt, proj, expd)
            paper[label] = (lambda u, lin=lin: lin['query'](u, nested)[0]); profs[label] = lin
            # the same reduced algebra as `hbk_core.make_linear`, composed into one matrix
            aa = np.asarray(setup['a']); pinv = np.linalg.pinv(aa)
            stride = int(round((times[1] - times[0]) / dt))
            step = pinv @ (((1 - dt * nu * lam_m / 2) / (1 + dt * nu * lam_m / 2))[:, None] * aa)
            per = np.linalg.matrix_power(step, stride); maps = [np.eye(Rp)]
            for _ in range(nt - 1): maps.append(per @ maps[-1])
            P6 = jnp.asarray((np.stack(maps) @ pinv).reshape(nt * Rp, -1) * (2.0 / n ** 2))
            for tau, prec in grid:
                tk = Q.truncate(U0, V0, C0, col2, Rp, tau); dtype = jnp.float64 if prec == 'f64' else jnp.float32
                kern, so, stages = make_kernel(tk['p'], tk['q'], Rp, nt, dtype)
                args = (S, I, J, P6, jnp.asarray(tk['C'].reshape(tk['p'] * tk['q'], Rp), dtype=dtype),
                        jnp.asarray(tk['U'], dtype=dtype), jnp.asarray(np.ascontiguousarray(tk['V'].T), dtype=dtype))
                variants[(label, vname(tau, prec))] = dict(
                    call=(lambda u, kern=kern, args=args: kern(u, *args)), stages=stages, args=args,
                    info=dict(arm=label, variant=vname(tau, prec), tau=tau, precision=prec, p=tk['p'], q=tk['q'],
                              estimated_relative_residual=tk['estimated_relative_residual'],
                              stored_bytes_f64=tk['stored_bytes_f64']))
                solve_only[label] = (lambda u, so=so, args=args: so(u, *args[:4]))
        del U0, V0, C0
        M_['variants'] = [v['info'] for v in variants.values()]

        def case_fields(ci):
            u0 = C.block(C.initial_grid(n, d, draws[ci])); same = C.block(prop(u0, lam_d, tjx, nu))
            return u0, same

        def deviation(f, ref):
            return float(jnp.max(C.rel_errors(f, ref)))

        # moments identity (offline check of step (i)): sine-table moments == the paper's transform moments
        u0, _ = case_fields(dev[0])
        mom_p = C.moments(u0, modes, n, d); mom_s = (2.0 / n ** 2) * (S.T @ u0 @ S)[I, J]
        M_['moment_identity'] = float(jnp.linalg.norm(mom_s - mom_p) / jnp.linalg.norm(mom_p))
        assert M_['moment_identity'] < 1e-10, M_['moment_identity']

        def run_phase(label, subs, cases, reps, rows, seed):
            order = np.random.default_rng([20261114, n, seed])
            for rep in range(reps):
                for ci in cases:
                    u0 = C.block(C.initial_grid(n, d, draws[ci]))
                    for i in order.permutation(len(subs)):
                        name, call = subs[int(i)]
                        C.burn(burn); res, sec = timed(call, u0); del res
                        rows.append(dict(name=name, case=int(ci), rep=rep, phase=label, ms=1e3 * sec))
                print('PHASE', n, label, rep, round(time.perf_counter() - begin, 1), flush=True)

        def med(rows, name, phases=None):
            v = [r['ms'] for r in rows if r['name'] == name and (phases is None or r['phase'] in phases)]
            return float(np.median(v)) if v else None

        def dev_gate(names):
            rows = {k: [] for k in names}
            for ci in dev:
                u0, _ = case_fields(ci)
                for label, _Rp in ARMS:
                    fp = C.block(paper[label](u0))
                    for k in names:
                        if k[0] == label:
                            rows[k].append(deviation(variants[k]['call'](u0), fp))
                    del fp
            return [dict(arm=k[0], variant=k[1], max_deviation_from_paper=max(v), passes=bool(max(v) <= GATE),
                         deviations=v) for k, v in rows.items()]

        # ------------------------------------------------ selection (development cohort only)
        if n == a.select_mesh and frozen is None:
            for k, v in variants.items():                       # compile
                C.block(v['call'](u0))
            gate = dev_gate(list(variants))
            rows = []
            run_phase('select', [(f'{k[0]}__{k[1]}', v['call']) for k, v in variants.items()], dev,
                      2 if a.smoke else 3, rows, 0)
            cand = []
            for tau, prec in grid:
                vn = vname(tau, prec)
                g = [r for r in gate if r['variant'] == vn]
                cand.append(dict(variant=vn, tau=tau, precision=prec, passes=all(r['passes'] for r in g),
                                 max_deviation=max(r['max_deviation_from_paper'] for r in g),
                                 accurate_ms=med(rows, f'accurate__{vn}'), fast_ms=med(rows, f'fast__{vn}')))
            pool = [c for c in cand if c['passes']]
            assert pool, 'no variant passed the development gate'
            best = min(c['accurate_ms'] for c in pool)
            tied = [c for c in pool if c['accurate_ms'] <= 1.02 * best]
            pick = sorted(tied, key=lambda c: (c['precision'] != 'f64', c['tau']))[0]
            frozen = dict(rule='PROTOCOL.md A3 selection', selection_mesh=n, cohort='development only', candidates=cand,
                          F=dict(pick, tied=[c['variant'] for c in tied]), job_id=R_['job_id'], nvidia_smi=R_['nvidia_smi'])
            text = json.dumps(frozen, indent=1)
            (out / 'frozen-h2d.json').write_text(text)
            R_['frozen'] = frozen; R_['frozen_sha256'] = hashlib.sha256(text.encode()).hexdigest()
            M_['selection'] = dict(dev_gate=gate, invocations=rows)
            print('FROZEN', pick['variant'], R_['frozen_sha256'], flush=True)
            for c in cand:
                print('CAND', c['variant'], c['passes'], f"{c['max_deviation']:.3e}", round(c['accurate_ms'], 4),
                      round(c['fast_ms'], 4), flush=True)
            save()
        assert frozen is not None
        fname = frozen['F']['variant']
        keep = [fname] + [v for v in (vname(1e-8, 'f64'), vname(1e-8, 'f32')) if v != fname]
        subjects = [(label, v) for label, _ in ARMS for v in keep]

        # ------------------------------------------------ reproduction gate and errors (all cases)
        st = stored['meshes'].get(str(n))
        rep_rows, errs, devs = [], {}, {}
        for ci in dev + test:
            u0, same = case_fields(ci)
            for label, Rp in ARMS:
                fp = C.block(paper[label](u0)); e = np.asarray(C.rel_errors(fp, same))
                errs.setdefault(('paper', label), {})[ci] = float(e.max())
                if st is not None and not a.smoke:
                    assert np.allclose(draws[ci], st['draws'][ci], rtol=0, atol=0), 'cohort differs from the paper run'
                    ref = np.asarray(st['arms'][f'lin_R{Rp}_cn']['same'][ci])
                    rep_rows.append(dict(arm=label, case=int(ci), max_abs_difference=float(np.abs(e - ref).max())))
                for v in keep:
                    f = variants[(label, v)]['call'](u0)
                    errs.setdefault((v, label), {})[ci] = float(jnp.max(C.rel_errors(f, same)))
                    devs.setdefault((v, label), {})[ci] = deviation(f, fp); del f
                del fp
            del same
        M_['repro_gate'] = dict(rows=rep_rows, tolerance=REPRO_TOL,
                                max_abs_difference=max((r['max_abs_difference'] for r in rep_rows), default=None),
                                passed=bool(rep_rows) and all(r['max_abs_difference'] <= REPRO_TOL for r in rep_rows))
        print('REPRO', n, M_['repro_gate']['passed'], M_['repro_gate']['max_abs_difference'], flush=True)
        M_['dev_gate'] = [dict(arm=label, variant=v, max_deviation_from_paper=max(devs[(v, label)][ci] for ci in dev),
                               passes=bool(max(devs[(v, label)][ci] for ci in dev) <= GATE)) for label, v in subjects]

        # ------------------------------------------------ A-B-A timing on the test cohort
        psubs = [(f'paper__{label}', paper[label]) for label, _ in ARMS]
        vsubs = [(f'{label}__{v}', variants[(label, v)]['call']) for label, v in subjects] + \
                [(f'{label}__coefficients_only', solve_only[label]) for label, _ in ARMS]
        for _name, call in psubs + vsubs:
            C.block(call(u0))
        rows = []
        run_phase('paperA1', psubs, test, reps, rows, 1)
        run_phase('variants', vsubs, test, reps, rows, 2)
        run_phase('paperA2', psubs, test, reps, rows, 3)
        summary = []
        for label, Rp in ARMS:
            s = dict(arm=label, name=f'lin_R{Rp}_cn', kind='paper', median_ms=med(rows, f'paper__{label}'),
                     A1_ms=med(rows, f'paper__{label}', ('paperA1',)), A2_ms=med(rows, f'paper__{label}', ('paperA2',)),
                     worst_test_error=max(errs[('paper', label)][ci] for ci in test),
                     worst_dev_error=max(errs[('paper', label)][ci] for ci in dev))
            if st is not None:
                s['stored_test_median_ms'] = st['arms'][f'lin_R{Rp}_cn']['heldout_median_ms']
                s['stored_worst_test_error'] = st['arms'][f'lin_R{Rp}_cn']['heldout_worst_error']
            summary.append(s)
            for v in keep:
                summary.append(dict(arm=label, name=v, kind='frozen' if v == fname else 'ablation',
                                    median_ms=med(rows, f'{label}__{v}'),
                                    worst_test_error=max(errs[(v, label)][ci] for ci in test),
                                    worst_dev_error=max(errs[(v, label)][ci] for ci in dev),
                                    max_test_deviation_from_paper=max(devs[(v, label)][ci] for ci in test),
                                    **{k: variants[(label, v)]['info'][k] for k in ('p', 'q', 'stored_bytes_f64')}))
            summary.append(dict(arm=label, name='coefficients_only', kind='coefficients_only',
                                median_ms=med(rows, f'{label}__coefficients_only')))
        M_['summary'] = summary; M_['invocations'] = rows

        # ------------------------------------------------ stage profile (first test case)
        u0, _ = case_fields(test[0]); prof = {}
        for label, _Rp in ARMS:
            lin = profs[label]; rec = {k: [] for k in ('encode', 'evolve', 'decode', 'query')}
            v = variants[(label, fname)]; s1, s2, s3 = v['stages']; ar = v['args']
            rec2 = {k: [] for k in ('moments_and_coefficients', 'core_contraction', 'field_products', 'query')}
            for _ in range(reps + 2):
                C.burn(burn)
                (c0, m0), t0 = timed(lin['encode'], u0, nested); cf, t1 = timed(lin['evolve'], c0, m0)
                _f, t2 = timed(lin['decode'], cf, nested); del _f
                _f, t3 = timed(lin['query'], u0, nested); del _f
                for k, t in zip(rec, (t0, t1, t2, t3)): rec[k].append(1e3 * t)
                C.burn(burn)
                co, t0 = timed(s1, u0, *ar[:4]); cr, t1 = timed(s2, co, ar[4]); _f, t2 = timed(s3, cr, ar[5], ar[6]); del _f
                _f, t3 = timed(v['call'], u0); del _f
                for k, t in zip(rec2, (t0, t1, t2, t3)): rec2[k].append(1e3 * t)
            prof[label] = dict(paper={k: float(np.median(x[2:])) for k, x in rec.items()},
                               frozen={k: float(np.median(x[2:])) for k, x in rec2.items()})
        M_['profile_median_ms'] = prof
        try:
            M_['peak_device_bytes'] = int(jax.devices()[0].memory_stats()['peak_bytes_in_use'])
        except Exception:
            pass
        M_['elapsed_seconds'] = time.perf_counter() - begin
        R_['meshes'].append(M_); save()
        for s in summary:
            print('SUMMARY', n, s['arm'], s['name'], round(s['median_ms'], 4), s.get('worst_test_error'),
                  s.get('max_test_deviation_from_paper'), flush=True)
        print('PROFILE', n, json.dumps(prof), flush=True)
        del nested, paper, variants, solve_only, profs, S
        jax.clear_caches()
    R_['complete'] = True; save()
    (out / 'COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z\n'))
    print('HEAT 2D COMPLETE', flush=True)


if __name__ == '__main__':
    main()
