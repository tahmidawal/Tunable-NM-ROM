"""Repair the 2n reference of an L-shape attempt whose audit rejected it (DESIGN A10).

The driver accepted its GPU-CG 2n reference on CG's RECURSIVE residual, which drifts from the
true residual at 12.6M unknowns (audit: true residual 4.6e-9 vs limit 6.4e-10). This script
restarts CG on the TRUE residual (x <- x + cg(b - A x)) until the true relative residual is at
the f64 evaluation floor, rewrites the reference files, recomputes `physical_error` for every
recorded invocation from the saved fields, and keeps the superseded values. No timed quantity
and no same-grid error is touched. The unchanged independent audit then decides acceptance.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import lsh_core as K_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    a = ap.parse_args()
    out = Path(a.out)
    assert jax.default_backend() == 'gpu' and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    R = json.loads((out / 'result.json').read_text())
    assert R['complete'] and 'fine_reference_repair' not in R
    shutil.copy(out / 'result.json', out / 'result.before-repair.json')
    cfg, n = R['config'], R['intervals']
    dev = np.asarray(R['cohort']['parameters'])
    g = K_.Geometry(2 * n, 'lshape')
    mask = jnp.asarray(g.mask)
    h2 = float((2 * n) ** 2)
    cg = K_.make_gpu_cg(g, cfg['cg_maxiter'])

    @jax.jit
    def true_residual(x, b):
        lap = 4.0 * x - (jnp.roll(x, 1, 0) + jnp.roll(x, -1, 0) + jnp.roll(x, 1, 1) + jnp.roll(x, -1, 1))
        return jnp.where(mask, b - lap * h2, 0.0)

    log = []
    for case, q in enumerate(dev):
        t0 = time.perf_counter()
        b = jnp.asarray(K_.source_full(g, q))
        bn = float(jnp.linalg.norm(b))
        x = jnp.asarray(np.load(out / 'fields' / f'reference_fine_full_case{case}.npy'))
        floor = float(np.finfo(float).eps * 8 * h2 * jnp.linalg.norm(x) / bn)
        history = []
        for sweep in range(8):
            r = true_residual(x, b)
            rel = float(jnp.linalg.norm(r)) / bn
            history.append(rel)
            if rel <= 1.2 * floor or (len(history) > 1 and rel > 0.7 * history[-2]):
                break
            e, it, _ = cg(r, 1e-4)
            x = x + e
        old = np.load(out / 'fields' / f'reference_fine_case{case}.npy')
        full = np.asarray(x)
        new = full[::2, ::2].copy()
        np.save(out / 'fields' / f'reference_fine_full_case{case}.npy', full)
        np.save(out / 'fields' / f'reference_fine_case{case}.npy', new)
        log.append(dict(case=case, true_residual_history=history, roundoff_floor=floor,
                        reference_change_relative=K_.relative(new, old), seconds=time.perf_counter() - t0))
        print('REPAIR', log[-1], flush=True)
        cache = {}
        for x_ in R['invocations']:
            if x_['case'] != case or not x_['matches_saved_field']:
                continue
            if x_['saved_field'] not in cache:
                cache[x_['saved_field']] = K_.relative(np.load(out / 'fields' / x_['saved_field']), new)
            x_['physical_error_before_repair'] = x_['physical_error']
            x_['physical_error'] = cache[x_['saved_field']]
        R['references'][case].update(fine_true_residual_after_repair=history[-1],
                                     fine_reference_change_relative=log[-1]['reference_change_relative'])
    unmatched = [x_ for x_ in R['invocations'] if 'physical_error_before_repair' not in x_]
    assert not unmatched, len(unmatched)
    R['fine_reference_repair'] = dict(
        reason='audit rejected the CG 2n reference: true residual above the round-off-aware limit (DESIGN A10)',
        job_id=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'), cases=log,
        worst_reference_change_relative=max(c['reference_change_relative'] for c in log),
        worst_physical_error_change=max(abs(x_['physical_error'] - x_['physical_error_before_repair'])
                                        for x_ in R['invocations']))
    K_.dump(out / 'result.json', R)
    print('REPAIR COMPLETE', R['fine_reference_repair']['worst_reference_change_relative'],
          R['fine_reference_repair']['worst_physical_error_change'], flush=True)


if __name__ == '__main__':
    main()
