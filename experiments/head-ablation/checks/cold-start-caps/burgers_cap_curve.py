"""Burgers 2D, 256 intervals: error and cost against the LM iteration caps under cold starts.

Same frozen checkpoint, bank, head, EQ rule (m=256 on M=64 tests), dt, trust radius, tolerance
and the six development cases as the head-ablation job abl01. Two sweeps:
  A. per-step iteration cap, with the initial fit at its full budget, step start = previous
     step (incumbent) or z = 0 every step;
  B. initial-fit iteration cap, with steps at their full budget, initial start = nearest Gauss
     candidate (incumbent) or z = 0.
Budgets are traced arguments so each start policy compiles once. Local diagnostic only.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

ROOT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-head-ablation')
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))
sys.path.insert(0, str(ROOT / 'experiments/head-ablation'))
import engines as e            # noqa: E402
import iterative_paths as ip   # noqa: E402
import arms as A               # noqa: E402

CK = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
CFG = json.load(open(ROOT / 'experiments/head-ablation/config-ablation.json'))
OUT = Path(sys.argv[1])
L = int(sys.argv[2]) if len(sys.argv) > 2 else 256
STEP_CAPS = [1, 2, 3, 4, 6, 8, 12, 20, 180]
IC_CAPS = [1, 2, 4, 8, 16, 32, 64, 128, 400]
REPS = 3


def make_lm(fun, trust, gtol):
    """arms.make_stationary_lm with the budget as a traced argument."""
    solve = e.gj_solve

    def lm(z0, args, tol, budget):
        def evaluate(z):
            r = fun(z, *args)
            J = jax.jacfwd(fun)(z, *args)
            return r, J, jnp.linalg.norm(r)

        def grad(r, J):
            return jnp.linalg.norm(J.T @ r) / (jnp.linalg.norm(J) * jnp.linalg.norm(r) + 1e-300)

        r, J, rn = evaluate(z0)
        reason = jnp.where(jnp.isfinite(rn), jnp.where(grad(r, J) <= gtol, 4, jnp.where(rn <= tol, 1, 0)), 3).astype(jnp.int32)

        def body(s):
            z, r, J, rn, lam, it, reason = s
            H = J.T @ J
            g = J.T @ r
            dz = solve(H + lam * jnp.diag(jnp.diag(H) + 1e-30), -g)
            ok = jnp.all(jnp.isfinite(dz)) & (jnp.linalg.norm(dz) <= trust)
            zn = z + jnp.where(ok, dz, 0.)
            rn2 = jnp.linalg.norm(fun(zn, *args))
            accept = ok & jnp.isfinite(rn2) & (rn2 < rn)
            r2, J2, rn2 = jax.lax.cond(accept, lambda: evaluate(zn), lambda: (r, J, rn))
            gn = grad(r2, J2)
            tiny = ok & (jnp.linalg.norm(dz) <= 1e-14 * (1 + jnp.linalg.norm(z)))
            reason = jnp.where(gn <= gtol, 4, jnp.where(rn2 <= tol, 1, jnp.where(tiny, 2, jnp.where((~accept) & (lam >= 1e14), 3, 0)))).astype(jnp.int32)
            return (jnp.where(accept, zn, z), r2, J2, rn2,
                    jnp.where(accept, jnp.maximum(lam / 3, 1e-12), jnp.minimum(lam * 10, 1e14)), it + 1, reason)

        z, r, J, rn, lam, it, reason = jax.lax.while_loop(
            lambda s: (s[5] < budget) & (s[6] == 0), body, (z0, r, J, rn, jnp.asarray(1e-6), jnp.int32(0), reason))
        return z, rn, it, reason, grad(r, J)
    return lm


def make_query(head, K, L, dt, trust, gtol, ic_policy, step_policy):
    wk = A.weak_fn('eq')
    ic = make_lm(lambda z, y, R: R @ head(z) - y, np.inf, gtol)
    lm = make_lm(lambda z, p, nu, data: wk(z, p, nu, data, head, L, dt), trust, gtol)

    def query(u0, nu, data, cold, ic_budget, step_budget):
        xy, w, Q, R, Hrot, Hnorm, Zcand = cold
        ui = e.sample_field(u0, xy, L) * w
        y = Q.T @ ui
        idx = jnp.argmin(Hnorm - 2 * Hrot @ y)
        z0 = Zcand[idx] if ic_policy == 'gauss' else jnp.zeros((K,), dtype=Zcand.dtype)
        z, icrn, icit, icreason, icgn = ic(z0, (y, R), 0., ic_budget)
        scale = jnp.linalg.norm(ui) * jnp.sqrt(len(w))

        def step(carry, _):
            z, zprev = carry
            p = data['A'] @ head(z)
            if step_policy == 'prev':
                ze = z + (z - zprev)
                r0 = jnp.linalg.norm(wk(z, p, nu, data, head, L, dt))
                re = jnp.linalg.norm(wk(ze, p, nu, data, head, L, dt))
                zi = jnp.where(jnp.isfinite(re) & (re < r0), ze, z)
            else:
                zi = jnp.zeros((K,), dtype=z.dtype)
            z2, rn, it, reason, gn = lm(zi, (p, nu, data), 1e-9 * scale, step_budget)
            return (z2, z), (z2, rn, it, reason, gn)

        _, (zs, rn, it, reason, gn) = jax.lax.scan(step, (z, z), None, length=int(round(.25 / dt)))
        internal = jnp.concatenate((z[None], zs))
        Z = internal[::int(round(.05 / dt))]
        fields = jax.vmap(lambda z: e.output_field(data['G'] @ head(z), L, L))(Z)
        return fields, it, reason, icit, icreason, gn, icgn
    return jax.jit(query)


def main():
    assert jax.default_backend() == 'gpu'
    print('jax_backend=gpu', flush=True)
    dt, strict = CFG['dt'], CFG['strict']
    ck = pickle.load(open(CK, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Zold = np.asarray(ck['Z_tr'])
    K, R = Zold.shape[1], int(np.asarray(ck['params']['h_lin']).shape[1])
    head = A.neural_head(params)
    bank = A.CoordBank(params, K, R)
    physical = np.concatenate((e.params_draw(CFG['eval_seed'], CFG['eval_cases']),
                               e.params_draw(CFG['eval_fresh_seed'], CFG['eval_fresh_cases'])))
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    stride = max(1, len(Zold) // CFG['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    t0 = time.perf_counter()
    M = CFG['test_multiplier'] * K
    data, info = A.build_operators(bank, L, M, 'eq', Zcoef=Zold, m=CFG['quadrature_multiplier'] * M,
                                   eq_seed=CFG['eq_seed'], candidate_cap=CFG['candidate_cap'],
                                   fit_states=CFG['fit_states'], head=head)
    cold, cinfo = A.build_cold(bank, head, Zsub, CFG['cold_axis_points'])
    print(f'operators built in {time.perf_counter() - t0:.1f}s; M={M} m={info["m"]} eq_fit={info["eq_relative_fit"]:.3e} trust={trust:.4g}', flush=True)

    fom, pre = ip.make_fom(L, dt, 'fft')
    tight = next(f for f in CFG['fom_settings'] if f['name'] == 'fft_tight')
    loose = next(f for f in CFG['fom_settings'] if f['name'] == 'fft_loose')
    inputs = [jnp.asarray(e.initial(L, p)) for p in physical]
    truths, fom_ms = [], {}
    for name, fs in (('fft_tight', tight), ('fft_loose', loose)):
        tt = []
        for case, u in enumerate(inputs):
            nu = float(physical[case, 4])
            v = fom(u, nu, fs['ntol'], fs['ltol'], *pre)
            jax.block_until_ready(v)
            t1 = time.perf_counter()
            v = fom(u, nu, fs['ntol'], fs['ltol'], *pre)
            jax.block_until_ready(v)
            tt.append((time.perf_counter() - t1) * 1e3)
            if name == 'fft_tight':
                truths.append(np.asarray(v[0]))
        fom_ms[name] = float(np.median(tt))
    loose_err = []
    for case, u in enumerate(inputs):
        v = fom(u, float(physical[case, 4]), loose['ntol'], loose['ltol'], *pre)
        f = np.asarray(v[0])
        n0 = np.linalg.norm(truths[case][0])
        loose_err.append(max(np.linalg.norm(f[t] - truths[case][t]) / n0 for t in range(f.shape[0])))
    print(f"FOM tight {fom_ms['fft_tight']:.1f} ms; loose {fom_ms['fft_loose']:.1f} ms, worst same-grid {max(loose_err) * 100:.3f}%", flush=True)

    out = dict(L=L, K=K, R=R, M=M, m=info['m'], trust=trust, fom_ms=fom_ms, fom_loose_worst=float(max(loose_err)),
               step_caps=STEP_CAPS, ic_caps=IC_CAPS, rows=[])
    kernels = {}

    def run(ic_policy, step_policy, ic_budget, step_budget, sweep):
        key = (ic_policy, step_policy)
        if key not in kernels:
            kernels[key] = make_query(head, K, L, dt, trust, strict['gtol'], ic_policy, step_policy)
        q = kernels[key]
        errs, errs_ev, ms, exits, icits, its = [], [], [], [], [], []
        for case, u in enumerate(inputs):
            nu = jnp.asarray(physical[case, 4])
            args = (u, nu, data, cold, jnp.int32(ic_budget), jnp.int32(step_budget))
            jax.block_until_ready(q(*args))
            reps = []
            for _ in range(REPS):
                t1 = time.perf_counter()
                v = q(*args)
                jax.block_until_ready(v)
                reps.append((time.perf_counter() - t1) * 1e3)
            fields, it, reason, icit, icreason, gn, icgn = jax.device_get(v)
            f = np.asarray(fields)
            n0 = np.linalg.norm(truths[case][0])
            per_t = [np.linalg.norm(f[t] - truths[case][t]) / n0 for t in range(f.shape[0])]
            errs.append(float(max(per_t)))
            errs_ev.append(float(max(per_t[1:])))
            ms.append(float(np.median(reps)))
            exits.append(int(np.sum(np.asarray(reason) == 0)))
            icits.append(int(icit))
            its.append(float(np.median(np.asarray(it))))
            out['rows'].append(dict(sweep=sweep, ic_policy=ic_policy, step_policy=step_policy, ic_budget=ic_budget,
                                    step_budget=step_budget, case=case, worst_all=errs[-1], worst_evolved=errs_ev[-1],
                                    per_time=[float(x) for x in per_t], ms=ms[-1], step_budget_exits=exits[-1],
                                    ic_iterations=icits[-1], ic_reason=int(icreason), median_step_iters=its[-1],
                                    finite=bool(np.isfinite(f).all())))
        print(f"{sweep} ic={ic_policy:5s} step={step_policy:4s} ic_cap {ic_budget:3d} step_cap {step_budget:3d}: "
              f"worst-all {max(errs) * 100:8.3f}%  worst-evolved {max(errs_ev) * 100:8.3f}%  median-all {np.median(errs) * 100:7.3f}%  "
              f"ms {np.median(ms):7.1f}  step-budget-exits/300 {int(np.median(exits)):3d}  ic-iters med {np.median(icits):.0f}", flush=True)
        OUT.write_text(json.dumps(out, indent=1))

    for step_policy in ('prev', 'zero'):
        for cap in STEP_CAPS:
            run('gauss', step_policy, strict['ic_budget'], cap, 'A_step_cap')
    for ic_policy in ('gauss', 'zero'):
        for cap in IC_CAPS:
            run(ic_policy, 'prev', cap, strict['step_budget'], 'B_ic_cap')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
