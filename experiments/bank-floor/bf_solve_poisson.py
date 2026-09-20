"""Phase 2, Poisson 2D: full-bank (q = R) solved error and paired online cost per bank.

For a linear PDE the full-bank rung is a linear reduced model (p-linear, 2026-09-17), so it
is solved directly, in two forms, for every bank:

* `weak`     : the project's row-scaled weak least squares over the M = 4R lowest sine tests,
               thin QR of the M x R operator offline, projection + triangular solve + decode online;
* `galerkin` : Q^T A Q y = Q^T f, Cholesky offline; no transform of the source online.

Comparators in the SAME allocation with the identical timing scope (host source in -> host dense
field out): direct DST (labelled control), unpreconditioned CG at several tolerances (named FOM),
coarse-grid DST + bilinear prolongation (matched-accuracy control).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import iterative_core as IC
import pbh_core as P
import bf_core as K

F64 = jnp.float64


def load_basis(spec, here, xy):
    path = here / spec['file']
    if path.suffix == '.npy':
        Q = jnp.asarray(np.load(path)[:, :spec['R']])
        dev = float(jnp.max(jnp.abs(Q.T @ Q - jnp.eye(Q.shape[1], dtype=F64))))
        assert dev < 1e-10, dev
        info = dict(columns=int(Q.shape[1]), rank=int(Q.shape[1]), rank_valid=True,
                    orthonormality_deviation=dev)
    else:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        blocks = d['blocks'] if 'blocks' in d else [K.bank_block(d['params'])]
        Q, info = K.orth_basis(K.bank_of(blocks, xy))
    return Q, dict(info, file=spec['file'], sha256=K.sha_file(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    smoke = bool(cfg.get('smoke', False))
    N = cfg['intervals']
    n1 = N - 1
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    R_ = dict(lane='bank-floor', phase='solve', pde='poisson2d', config=cfg,
              source_commit=os.environ.get('SOURCE_COMMIT', 'unknown'),
              slurm_job=os.environ.get('SLURM_JOB_ID', 'local'), gpu=gpu, jax=jax.__version__,
              backend=jax.default_backend(), x64=True,
              matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'unset'),
              timing_contract='host source array in -> host dense nodal field out; burn-in before '
                              'every block; synchronised; medians over retained repetitions',
              subjects={}, complete=False)
    assert smoke or R_['backend'] == 'gpu'
    save = lambda: K.dump(out / 'result.json', R_)

    cohorts = {name: np.concatenate([C.source_params(s, c) for s, c in spec])
               for name, spec in cfg['cohorts'].items()}
    R_['cohorts'] = {k: dict(count=len(v), parameters_sha256=K.sha_array(v)) for k, v in cohorts.items()}
    sources = {k: [C.full_source(N, q) for q in v] for k, v in cohorts.items()}
    lam = jnp.asarray(C.eigenvalues(N))
    truth = {k: [np.asarray(C.dst_solve(jnp.asarray(s), lam)) for s in v] for k, v in sources.items()}
    xy = K.grid(N)

    # lowest-M sine tests, shared ordering
    lam_np = np.asarray(lam)
    order = np.argsort(lam_np.ravel(), kind='stable')

    def rel(field, ref):
        return float(np.linalg.norm(field - ref) / np.linalg.norm(ref))

    def run_subject(name, query, meta):
        rows = {}
        for coh, srcs in sources.items():
            errs, times, dev_t = [], [], []
            for i, s in enumerate(srcs):
                C.burn(cfg['burn_seconds'])
                query(s)                                          # warm, discarded
                reps = []
                for r in range(cfg['repetitions']):
                    field, info = query(s)
                    reps.append(info)
                errs.append(rel(field[1:-1, 1:-1], truth[coh][i][1:-1, 1:-1]))
                times.append([x['total_seconds'] for x in reps])
                dev_t.append([x.get('fused_device_seconds', x.get('solver_seconds')) for x in reps])
                if coh == cfg['save_fields_cohort']:
                    np.save(out / 'fields' / f'{name}_case{i}.npy', field)
            t = np.asarray(times)
            rows[coh] = dict(K.summarise(errs), per_case=errs,
                             median_total_ms=float(np.median(t) * 1e3),
                             median_device_ms=float(np.median(np.asarray(dev_t)) * 1e3),
                             total_seconds=t.tolist())
        R_['subjects'][name] = dict(meta, results=rows)
        save()
        p = rows[cfg['save_fields_cohort']]
        print(f'SUBJECT {name}: worst {p["worst"]:.4e} median {p["median"]:.4e} '
              f'total {p["median_total_ms"]:.3f} ms device {p["median_device_ms"]:.3f} ms', flush=True)

    def timed(kernel, *ops):
        def query(host_source):
            start = time.perf_counter()
            src = jax.device_put(host_source)
            src.block_until_ready()
            t1 = time.perf_counter()
            field = kernel(src, *ops)
            field.block_until_ready()
            t2 = time.perf_counter()
            field = np.asarray(jax.device_get(field))
            end = time.perf_counter()
            return field, dict(total_seconds=end - start, input_seconds=t1 - start,
                               fused_device_seconds=t2 - t1, output_seconds=end - t2)
        return query

    # ---- full-order comparators ----------------------------------------------------
    run_subject('dst_direct', lambda s: C.fom_query(s, lam), dict(kind='fom_control', label='direct DST'))
    cg = IC.make_cg(N, cfg['cg_maxiter'])
    for tol in cfg['cg_tolerances']:
        def q(s, tol=tol):
            f, info = IC.cg_query(s, cg, tol)
            assert info['cg_converged'], info
            return f, info
        run_subject(f'cg_{tol:g}', q, dict(kind='fom_named', label=f'unpreconditioned CG tol {tol:g}'))
    for coarse in cfg['coarse_intervals']:
        lc = jnp.asarray(C.eigenvalues(coarse))
        run_subject(f'coarse_dst_{coarse}', lambda s, lc=lc: C.coarse_query(s, lc),
                    dict(kind='fom_coarse', label=f'DST on {coarse} intervals + bilinear prolongation'))

    # ---- reduced subjects -------------------------------------------------------------
    @jax.jit
    def weak_kernel(src, I, J, lamM, Qt, Rr, Q):
        fm = C.dst2(src[1:-1, 1:-1])[I, J] / lamM
        y = jax.scipy.linalg.solve_triangular(Rr, Qt @ fm, lower=False)
        return jnp.pad((Q @ y).reshape(n1, n1), 1)

    @jax.jit
    def gal_kernel(src, Lc, Q):
        y = jax.scipy.linalg.cho_solve((Lc, True), Q.T @ src[1:-1, 1:-1].reshape(-1))
        return jnp.pad((Q @ y).reshape(n1, n1), 1)

    @jax.jit
    def proj_kernel(src, lam_, Q):
        u = C.dst_solve(src, lam_)[1:-1, 1:-1].reshape(-1)
        return jnp.pad((Q @ (Q.T @ u)).reshape(n1, n1), 1)

    dst_cols = jax.jit(jax.vmap(C.dst2, in_axes=0))

    def apply_A(V):                                               # V: (n, r) -> A V
        W = V.T.reshape(-1, n1, n1)
        Pd = jnp.pad(W, ((0, 0), (1, 1), (1, 1)))
        AW = N ** 2 * (4 * W - Pd[:, :-2, 1:-1] - Pd[:, 2:, 1:-1] - Pd[:, 1:-1, :-2] - Pd[:, 1:-1, 2:])
        return AW.reshape(W.shape[0], -1).T

    for spec in cfg['banks']:
        tag = spec['tag']
        t0 = time.perf_counter()
        Q, binfo = load_basis(spec, here, xy)
        Rk = int(Q.shape[1])
        M = int(min(cfg['test_multiplier'] * Rk, n1 * n1))
        sel = order[:M]
        I, J = np.unravel_index(sel, (n1, n1))
        I, J = jnp.asarray(I), jnp.asarray(J)
        lamM = lam[I, J]
        # B = Phi_M^T Q  (the row-scaled weak operator Lambda^-1 Phi^T A Q)
        parts = []
        for s in range(0, Rk, 128):
            parts.append(dst_cols(Q[:, s:s + 128].T.reshape(-1, n1, n1))[:, I, J].T)
        B = jnp.concatenate(parts, axis=1)
        Qb, Rr = jnp.linalg.qr(B)
        sv = np.asarray(jnp.linalg.svd(Rr, compute_uv=False))
        Ar = Q.T @ apply_A(Q)
        Ar = 0.5 * (Ar + Ar.T)
        Lc = jnp.linalg.cholesky(Ar)
        assert bool(jnp.isfinite(Lc).all()) and bool(jnp.isfinite(Rr).all())
        meta = dict(kind='rom', bank=tag, R=Rk, basis=binfo, M=M,
                    weak_operator_condition=float(sv[0] / sv[-1]),
                    galerkin_condition=float(np.linalg.cond(np.asarray(Ar))),
                    offline_seconds=time.perf_counter() - t0,
                    online_arrays_bytes=dict(weak=int(8 * (Qb.size + Rr.size + Q.size)),
                                             galerkin=int(8 * (Lc.size + Q.size))))
        run_subject(f'{tag}_floor', timed(proj_kernel, lam, Q),
                    dict(meta, form='projection of the true solution (floor; not a solver)'))
        run_subject(f'{tag}_weak', timed(weak_kernel, I, J, lamM, Qb.T, Rr, Q), dict(meta, form='weak'))
        run_subject(f'{tag}_galerkin', timed(gal_kernel, Lc, Q), dict(meta, form='galerkin'))
        del Q, B, Qb, Rr, Ar, Lc

    R_['complete'] = True
    save()
    print('SOLVE-DONE', flush=True)


if __name__ == '__main__':
    main()
