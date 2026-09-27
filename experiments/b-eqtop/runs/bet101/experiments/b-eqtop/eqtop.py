"""b-eqtop: certifying the top rungs of the Burgers EQ ladder (DESIGN.md).

`q-ridge` (job 3768168) certified reachable-state rules on the held-out bar
rho_max <= 0.116 only for q <= 64 at m = 1024; at q = 128 and q = 256 no rule with
m <= 2048 reached it. This module adds, on top of `eqcert.py` (collection, `certify`) and
`varpro.bounded_nnls` (the audited fitter, run unchanged in `fitworker.py`), exactly four
things:

  `build_design`   the fit design from bank coefficients, under the incumbent per-row unit
                   scaling ('row') or a per-state scaling ('state') that makes the least
                   squares objective sum_s rho(u_s)^2 -- the certification metric itself.
  `compress`       an exact QR compression of a design with more rows than candidates:
                   [D | b] = Q [R | c ; 0 r], so ||D w - b||^2 = ||R w - c||^2 + r^2 for every
                   w and the gradient D^T(b - D w) = R^T(c - R w). The fitter sees a square
                   design whatever the number of fit states.
  `run_chains`     ascending-m fit chains per (rung, arm) in CPU worker processes, each rule
                   certified on the GPU as it lands, with a declared stopping rule.
  `rho_of_field`   rho evaluated on a grid FIELD in NumPy (q-diag's definition); gated
                   bitwise against q-diag's archived numbers and against `eqcert.certify`.

Every large array is a jit ARGUMENT, never a closure.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import eqcert as EC

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ designs --

def candidate_pool(L, size, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice((L - 1) ** 2, min(int(size), (L - 1) ** 2), replace=False))


def build_design(G, Phi, L, coefficients, cand, scaling='row'):
    """Rows (state s, test mode j) x candidate points; targets Phi_j^T a(u_s).

    scaling 'row'   each row divided by its own norm over the candidates -- the incumbent.
    scaling 'state' every row of state s divided by ||Phi^T a(u_s)||, so that
                    ||D w - b||^2 = sum_s rho(u_s)^2 exactly.
    """
    P = jnp.asarray(Phi)
    cp = np.asarray(cand)
    Pc = np.asarray(Phi)[cp]
    adv = jax.jit(lambda g, c: e.spatial(g @ c, L)[0])
    rows, targets, tnorm = [], [], []
    for c in np.asarray(coefficients):
        n = adv(G, jnp.asarray(c))
        t = np.asarray(P.T @ n)
        targets.append(t)
        tnorm.append(float(np.linalg.norm(t)))
        rows.append(Pc.T * np.asarray(n)[cp])
    D = np.concatenate(rows)
    b = np.concatenate(targets)
    if scaling == 'row':
        scale = np.linalg.norm(D, axis=1) + 1e-300
    elif scaling == 'state':
        scale = np.repeat(np.asarray(tnorm), int(Phi.shape[1])) + 1e-300
    else:
        raise ValueError(scaling)
    D /= scale[:, None]
    b /= scale
    return D, b, dict(scaling=scaling, rows=int(D.shape[0]), candidates=int(len(cp)),
                      fit_states=int(len(coefficients)), target_norms=tnorm)


def compress(D, b, gram_columns=32, seed=0, where='gpu'):
    """Exact QR compression of an over-tall design; returns (R, c, info).

    Gate (recorded in `info`): R^T R reproduces D^T D on `gram_columns` random columns to
    1e-8 relative, and every entry is finite. The GB10's first f64 QR of a tall matrix has
    returned NaN before, so a non-finite or failed-gate GPU result falls back to the CPU.
    """
    n = int(D.shape[1])
    if D.shape[0] <= n:
        return D, b, dict(compressed=False, rows_in=int(D.shape[0]), rows_out=int(D.shape[0]),
                          unreachable_residual=0., b_norm=float(np.linalg.norm(b)))
    Db = np.concatenate((D, b[:, None]), axis=1)
    t0 = time.perf_counter()
    used = where
    Rf = None
    if where == 'gpu':
        try:
            Rf = np.asarray(jnp.linalg.qr(jnp.asarray(Db), mode='r'))
            if not np.isfinite(Rf).all():
                Rf, used = None, 'cpu_after_nonfinite_gpu'
        except Exception as ex:                      # noqa: BLE001 - recorded, then CPU
            Rf, used = None, f'cpu_after_gpu_error:{type(ex).__name__}'
    if Rf is None:
        import scipy.linalg
        Rf = scipy.linalg.qr(Db, mode='r', check_finite=False)[0]
    del Db
    R = np.ascontiguousarray(Rf[:n, :n])
    c = np.ascontiguousarray(Rf[:n, n])
    r_perp = float(abs(Rf[n, n])) if Rf.shape[0] > n else 0.
    del Rf
    j = np.random.default_rng(seed).choice(n, min(gram_columns, n), replace=False)
    ref = D.T @ D[:, j]
    gram = float(np.max(np.abs(R.T @ R[:, j] - ref)) / max(np.max(np.abs(ref)), 1e-300))
    assert np.isfinite(R).all() and np.isfinite(c).all(), 'non-finite compression'
    assert gram < 1e-8, f'compression gram check failed: {gram}'
    return R, c, dict(compressed=True, rows_in=int(D.shape[0]), rows_out=n, where=used,
                      unreachable_residual=r_perp, b_norm=float(np.linalg.norm(b)),
                      gram_check=gram, seconds=time.perf_counter() - t0)


# ------------------------------------------------------------------- rules ---

def make_rule(bank, Phi, L, nodes, weights):
    """(G5, Pq) exactly as `eqcert.fit_rule` builds them, from archived nodes and weights."""
    pos = np.asarray(nodes, dtype=int)
    w = np.asarray(weights, dtype=float)
    ij = np.stack(np.unravel_index(pos, (L - 1, L - 1)), 1) + 1
    return dict(G5=bank.stencil(ij, L), Pq=jnp.asarray(np.asarray(Phi)[pos] * w[:, None]))


def load_archived_rule(path):
    z = np.load(path)
    return np.asarray(z['nodes'], dtype=int), np.asarray(z['weights'], dtype=float)


def rho_of_field(u_interior, Phi, L, nodes, weights):
    """q-diag's rho on one grid field, NumPy: a functional of the field and the rule only."""
    p = np.pad(np.asarray(u_interior).reshape(L - 1, L - 1), 1)
    c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    av = (c * L * (np.where(c > 0, c - xm, xp - c) + np.where(c > 0, c - ym, yp - c))).reshape(-1)
    Phi = np.asarray(Phi)
    exact = Phi.T @ av
    approx = (Phi[nodes] * np.asarray(weights)[:, None]).T @ av[nodes]
    return float(np.linalg.norm(approx - exact) / max(np.linalg.norm(exact), 1e-300))


# ------------------------------------------------------------ fit chains -----

def launch_worker(python, design, b, m, seconds, block, out, sidecar, threads, log):
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), JAX_PLATFORMS='cpu')
    cmd = [python, str(HERE / 'fitworker.py'), str(design), str(b), str(m), str(seconds),
           str(block), str(out)]
    if sidecar is not None:
        cmd.append(str(sidecar))
    return subprocess.Popen(cmd, env=env, stdout=open(log, 'a'), stderr=subprocess.STDOUT)


def run_chains(chains, certify_fn, python, workers, threads, seconds, blocks, on_rule,
               submit_deadline=None, poll=5.):
    """Run ascending-m fit chains concurrently; certify each rule as it lands.

    `chains` is a list of dicts with keys: key, grid (ascending m), design, b, sidecar,
    tmpdir, adaptive (stop after the first primary-certified rule plus `confirm` more points),
    confirm. `certify_fn(chain, m, support, weights, info) -> rule record` runs on the GPU in
    this process and returns a record with `certified_primary`. `on_rule(record)` is called
    for every finished rule (incremental saving). The submission deadline stops NEW fits;
    running fits finish inside their own walltime cap. Returns the per-chain stop reasons.
    """
    state = {c['key']: dict(chain=c, i=0, certified=0, after=0, done=False, reason=None)
             for c in chains}
    running = {}          # key -> (proc, m, out, t0)

    def next_m(st):
        c = st['chain']
        if st['i'] >= len(c['grid']):
            return None, 'grid_exhausted'
        if c.get('adaptive') and st['certified'] >= 1 and st['after'] >= c.get('confirm', 1):
            return None, 'certified_and_confirmed'
        return int(c['grid'][st['i']]), None

    def submit(key):
        st = state[key]
        if submit_deadline is not None and time.perf_counter() > submit_deadline:
            st['done'], st['reason'] = True, 'submission_deadline'
            return
        m, why = next_m(st)
        if m is None:
            st['done'], st['reason'] = True, why
            return
        c = st['chain']
        out = Path(c['tmpdir']) / f'fit_{key}_m{m}.npz'
        log = Path(c['tmpdir']) / f'fit_{key}.log'
        proc = launch_worker(python, c['design'], c['b'], m, seconds, max(1, m // blocks),
                             out, c.get('sidecar'), threads, log)
        running[key] = (proc, m, out, time.perf_counter())
        st['i'] += 1
        print(f'  FIT launch {key} m={m} pid={proc.pid}', flush=True)

    keys = [c['key'] for c in chains]
    pending = list(keys)
    while pending or running:
        while pending and len(running) < workers:
            k = pending.pop(0)
            submit(k)
            if state[k]['done']:
                continue
        if not running:
            break
        time.sleep(poll)
        for key in list(running):
            proc, m, out, t0 = running[key]
            rc = proc.poll()
            if rc is None:
                continue
            del running[key]
            st = state[key]
            if rc != 0 or not out.exists():
                st['done'], st['reason'] = True, f'worker_failed_rc{rc}'
                print(f'  FIT FAILED {key} m={m} rc={rc}', flush=True)
                on_rule(dict(key=key, m_target=m, failed=True, rc=rc))
                continue
            z = np.load(out)
            info = json.loads(str(z['info']))
            rec = certify_fn(st['chain'], m, np.asarray(z['support']), np.asarray(z['weights']),
                             info)
            rec['wall_seconds'] = time.perf_counter() - t0
            on_rule(rec)
            # `after` counts every rule fitted after the first primary-certified one.
            if rec.get('certified_primary'):
                if st['certified'] >= 1:
                    st['after'] += 1
                st['certified'] += 1
            elif st['certified'] >= 1:
                st['after'] += 1
            if info.get('truncated'):
                st['done'], st['reason'] = True, 'truncated_at_walltime'
            if not st['done']:
                pending.append(key)
    return {k: dict(stop_reason=v['reason'], rules_fitted=v['i'], certified=v['certified'])
            for k, v in state.items()}


def certify_rule(bank, G, Phi, L, cert_states, nodes, weights, chunk=32):
    rule = make_rule(bank, Phi, L, nodes, weights)
    return rule, EC.certify(G, Phi, L, rule, cert_states, chunk=chunk)
