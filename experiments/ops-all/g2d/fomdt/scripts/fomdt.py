"""g2d fomdt: Burgers 2D lean Newton-BiCGStab at intermediate time steps, solver-only timing on dev6 (DESIGN.md).

    python fomdt.py --mesh 1024 --out <dir>      (run from experiments/burgers-bank-knob; g2d burgers PYTHONPATH)

Cohort, truth and timing are bankknob.py's (burgers-bank-knob @ 340627ca), reduced to the FOM:
* dev6 = params_draw(7090702,4) + params_draw(911702,2), hash asserted (108f12dc...); inputs e.initial(L, .).
* truth = fft_tight (iterative_paths.make_fom(L, 0.005, 'fft'), ntol 1e-6, ltol 1e-8), convergence asserted.
* settings = engines.make_fom(L, dt, target=L) (the 'lean' impl), dt in {1/100, 1/120, 1/140, 1/160, 1/200}
  x (ntol, ltol) in {(1e-2,1e-2), (3e-3,3e-3), (1e-3,1e-3)}; the two Table 1 anchors are members.
* error per case = max over evolved times of ||u - u_ref||_2 / ||u_0||_2; worst / mean / median over cases.
* timing = bankknob's 'main'/'slow' phases (threshold 1 s on the quick run), reps 5, fresh random order per (rep, case)
  (seed 20260923), GPU burn 0.25 s before each invocation, cooldown after slow ones, input device_put + block, GPU
  seconds = invocation to block_until_ready; every timed output hashed and compared with the quick run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip

DEV6_SHA = '108f12dc8f9e6a64daf94f55861c616a29810134a52726e8a683a2a43892dc8a'
DTS = [(0.01, 'dt01'), (1 / 120, 'dt1_120'), (1 / 140, 'dt1_140'), (0.00625, 'dt00625'), (0.005, 'dt005')]
TOLS = [(1e-2, 1e-2, 'nt1e-2_l1e-2'), (3e-3, 3e-3, 'nt3e-3_l3e-3'), (1e-3, 1e-3, 'nt1e-3_l1e-3')]


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--reps', type=int, default=5)
    a = p.parse_args()
    assert jax.default_backend() == 'gpu' and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print('jax_backend=gpu x64=True precision=highest', flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = a.mesh
    physical = np.concatenate((e.params_draw(7090702, 4), e.params_draw(911702, 2)))
    assert sha_array(physical) == DEV6_SHA or os.environ.get('G2D_SMOKE'), 'cohort differs'
    inputs_u = [e.initial(L, ph) for ph in physical]
    n0 = [float(np.linalg.norm(u)) for u in inputs_u]
    rep = dict(mesh=L, job_id=os.environ.get('SLURM_JOB_ID'), gpu=jax.devices()[0].device_kind,
               nvidia_smi=subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip(),
               cohort='dev6', cohort_sha256=DEV6_SHA, reps=a.reps, burn_seconds=.25, order_seed=20260923, settings={},
               invocations=[], truth=[])
    fn, pre = ip.make_fom(L, .005, 'fft')
    truth = []
    for c, u in enumerate(inputs_u):
        t0 = time.perf_counter()
        v = fn(jnp.asarray(u), float(physical[c, 4]), 1e-6, 1e-8, *pre)
        f = np.asarray(v[0])
        rn = np.asarray(v[2])
        assert np.isfinite(f).all() and rn.max() <= 1e-6 * (1 + 1e-9)
        truth.append(f)
        rep['truth'].append(dict(case=c, seconds=time.perf_counter() - t0, max_relative_residual=float(rn.max())))
    del fn, pre
    print('TRUTH done', flush=True)
    settings = {}
    for dt, dn in DTS:
        q = e.make_fom(L, dt, target=L)[0]
        for nt, lt, tn in TOLS:
            settings[f'lean_{tn}_{dn}'] = (q, dt, nt, lt)
    quick = {}
    for name, (q, dt, nt, lt) in settings.items():
        rows, secs = [], []
        for c, u in enumerate(inputs_u):
            t0 = time.perf_counter()
            v = q(jnp.asarray(u), float(physical[c, 4]), nt, lt)
            jax.block_until_ready(v)
            secs.append(time.perf_counter() - t0)
            f = np.asarray(v[0])
            rn = np.asarray(v[2])
            sg = [float(np.linalg.norm(x - y)) / n0[c] for x, y in zip(f, truth[c])]
            ok = np.isfinite(rn) & (rn <= nt * (1 + 1e-9))
            rows.append(dict(case=c, same_grid_per_time=sg, same_grid_evolved=max(sg[1:]), newton_total=int(np.sum(np.asarray(v[1]))),
                             stalled_steps=int(np.sum(~ok)), converged=bool(ok.all()), field_sha256=sha_array(f),
                             field_sha256_sub=sha_array(f[:, ::16, ::16])))
            quick[(name, c)] = rows[-1]
        ev = [r['same_grid_evolved'] for r in rows]
        rep['settings'][name] = dict(dt=dt, ntol=nt, ltol=lt, steps_per_output=int(round(.05 / dt)), cases=rows,
                                     worst_percent=100 * max(ev), mean_percent=100 * float(np.mean(ev)),
                                     median_percent=100 * float(np.median(ev)), converged=all(r['converged'] for r in rows),
                                     quick_seconds=float(np.median(secs[1:])))
        print('QUICK', name, round(100 * max(ev), 4), round(100 * float(np.mean(ev)), 4), round(float(np.median(secs[1:])), 4), flush=True)
    names = list(settings)
    for n_ in names:
        jax.block_until_ready(settings[n_][0](jnp.asarray(inputs_u[0]), float(physical[0, 4]), *settings[n_][2:]))
    phases = [('main', [n_ for n_ in names if rep['settings'][n_]['quick_seconds'] < 1.0]),
              ('slow', [n_ for n_ in names if rep['settings'][n_]['quick_seconds'] >= 1.0])]
    rng = np.random.default_rng(20260923)
    seq = 0
    for ph, arms in phases:
        prev = 0.
        for r_ in range(a.reps):
            for c in range(len(inputs_u)):
                for i in rng.permutation(len(arms)):
                    name = arms[int(i)]
                    q, dt, nt, lt = settings[name]
                    if prev >= .5:
                        time.sleep(min(2.0, prev))
                    e.burn(.25)
                    u = jax.device_put(np.array(inputs_u[c], copy=True))
                    jax.block_until_ready(u)
                    g0 = time.perf_counter()
                    v = q(u, float(physical[c, 4]), nt, lt)
                    jax.block_until_ready(v)
                    gs = time.perf_counter() - g0
                    prev = gs
                    f = np.asarray(v[0])
                    same = sha_array(f[:, ::16, ::16]) == quick[(name, c)]['field_sha256_sub']
                    rep['invocations'].append(dict(name=name, case=c, rep=r_, phase=ph, seq=seq, gpu_seconds=gs, identical_to_quick=bool(same)))
                    seq += 1
            print('TIMED', ph, r_, flush=True)
    for name in names:
        g = [x['gpu_seconds'] for x in rep['invocations'] if x['name'] == name]
        rep['settings'][name].update(median_gpu_ms=1e3 * float(np.median(g)), timed_invocations=len(g))
    rep['gates'] = dict(repetition_output_identical=all(x['identical_to_quick'] for x in rep['invocations']),
                        retained_repetitions=all(rep['settings'][n]['timed_invocations'] >= 6 * a.reps for n in names))
    rep['complete'] = True
    (out / 'result.json').write_text(json.dumps(rep, indent=1) + '\n')
    for n in names:
        s = rep['settings'][n]
        print('RESULT', n, f"{s['worst_percent']:.4f} {s['mean_percent']:.4f} {s['median_percent']:.4f} {s['median_gpu_ms']:.2f}ms", flush=True)
    print('FOMDT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
