"""ns2d_headfit.py -- DESIGN §A9, ns301: the head-only DIAGNOSIS on a FROZEN bank.

    CKPT=checkpoints/ckpt_K16_R256.pkl TRAIN_N=256 SUBSETS=128,256,512
    REGIMES=plain,reg,reg_small STEPS=100000 LR=1e-3 WD=1e-4 EVAL_EVERY=5000
    H_HIDDEN=512 H_LAYERS=3 H_SMALL_HIDDEN=128 H_SMALL_LAYERS=2
    REPORT_CASES=32 (dev cases 0..31 = reported)  SEL_CASES=32 (dev cases 32..63 = selection)
    TRAIN_ORACLE_CASES=64  ORACLE_TIMES=0,5,10,15,20,25 ORACLE_STARTS=8 ORACLE_BUDGET=300
    EXPECT_HASHES=configs/phase1-hashes.json DEV_REF=configs/dev8_eval_ref.npz SMOKE=0 OUT=output

The bank G (and its thin QR on the training grid) is taken from the checkpoint and never
updated.  With a frozen bank the field-space loss splits exactly,
    ||G h(z) - u||^2 = ||R_b (h(z) - c)||^2 + ||u - P u||^2,   c = R_b^{-1} Q_b^T u,
so a head + per-snapshot codes are trained in the (S, R) coefficient space, which makes one
training step O(S R^2) instead of O(S n R) and lets nine arms run in one job.  Every reported
error is nevertheless evaluated in FIELD space (DESIGN §A4) with the formula value beside it.

Arms: nested training subsets (the FIRST n trajectories of the gated 512-trajectory cohort)
x regimes {plain: the ns203 head recipe (512x3, Adam, warmup-cosine, STEPS steps, no weight
decay, last iterate); reg: same head, AdamW weight decay WD on the head weights, dev-SELECT
oracle early stopping (best of every EVAL_EVERY steps); reg_small: 128x2 head, otherwise reg}.
Selection uses dev cases SEL (32..63); every reported number is on dev cases REPORT (0..31),
so the early stopping never sees the reported states.  Also recorded: the oracle on 64
TRAINING trajectories at the same times (the held-out/training gap per arm) and POD-K of the
same subset on the same states.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                              # noqa: E402
import optax                                                         # noqa: E402

import ns2d_fom as F                                                 # noqa: E402
import ns2d_decoder as D                                             # noqa: E402
import ns2d_phase2 as P2                                             # noqa: E402  (helpers + shared env)
import sep_common as sc                                              # noqa: E402

E = os.environ.get
CKPT = E('CKPT', 'checkpoints/ckpt_K16_R256.pkl')
SUBSETS = [int(v) for v in E('SUBSETS', '128,256,512').split(',')]
REGIMES = E('REGIMES', 'plain,reg,reg_small').split(',')
STEPS, LR, WD = int(E('STEPS', '100000')), float(E('LR', '1e-3')), float(E('WD', '1e-4'))
EVAL_EVERY = int(E('EVAL_EVERY', '5000'))
H_HIDDEN, H_LAYERS = int(E('H_HIDDEN', '512')), int(E('H_LAYERS', '3'))
H_SMALL_HIDDEN, H_SMALL_LAYERS = int(E('H_SMALL_HIDDEN', '128')), int(E('H_SMALL_LAYERS', '2'))
REPORT_CASES, SEL_CASES = int(E('REPORT_CASES', '32')), int(E('SEL_CASES', '32'))
TRAIN_ORACLE_CASES = int(E('TRAIN_ORACLE_CASES', '64'))
SEL_STARTS, SEL_BUDGET = int(E('SEL_STARTS', '4')), int(E('SEL_BUDGET', '100'))
SEED = int(E('SEED', '0'))
TRAIN_N, N_TRAIN, N_DEV, NOUT, SMOKE, OUT = P2.TRAIN_N, P2.N_TRAIN, P2.N_DEV, P2.NOUT, P2.SMOKE, P2.OUT
ORACLE_TIMES, ORACLE_STARTS, ORACLE_BUDGET = P2.ORACLE_TIMES, P2.ORACLE_STARTS, P2.ORACLE_BUDGET
log, dump, gate, host, sha = P2.log, P2.dump, P2.gate, P2.host, P2.sha


def init_head(key, K, R, hidden, layers):
    kh, kl = jax.random.split(key)
    return dict(h=sc.init_mlp(kh, [K] + [hidden] * layers + [R]),
                h_lin=jax.random.normal(kl, (K, R), dtype=jnp.float64) * 0.3)


def train_head(key, C, Rb, n_pts, u_ms, K, R, hidden, layers, steps, lr, wd, eval_fn=None,
               eval_every=0, tag=''):
    """Head + codes on the whitened coefficient targets C (S, R).  Returns (params, Z, info);
    with eval_fn the best-scoring iterate (lower is better) is returned, else the last."""
    S = C.shape[0]
    key, kz, kp = jax.random.split(key, 3)
    p = init_head(kp, K, R, hidden, layers)
    Z = 0.1 * jax.random.normal(kz, (S, K), dtype=jnp.float64)
    Cd = jnp.asarray(C)
    RbT = jnp.asarray(Rb).T
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(500, steps // 10 + 1), steps, lr * 1e-2)
    if wd > 0:
        opt = optax.adamw(sched, weight_decay=wd, mask=(jax.tree_util.tree_map(lambda _: True, p), False))
    else:
        opt = optax.adam(sched)
    st = opt.init((p, Z))

    def loss_fn(pz, Ct):
        p_, z_ = pz
        err = (sc.head(p_, z_) - Ct) @ RbT
        return jnp.sum(err * err) / (S * n_pts * u_ms)          # = field rel-MSE minus the fixed perp term

    @jax.jit
    def step(pz, st, Ct):
        val, g = jax.value_and_grad(loss_fn)(pz, Ct)
        upd, st = opt.update(g, st, pz)
        return optax.apply_updates(pz, upd), st, val

    pz, t0, curve, best = (p, Z), time.time(), [], None
    for i in range(steps):
        pz, st, val = step(pz, st, Cd)
        if (i + 1) % 1000 == 0 or i == 0:
            print(f'   head[{tag}] step {i+1:6d}/{steps}  rel-MSE(coef) {float(val):.3e}  [{time.time()-t0:.0f}s]', flush=True)
        if eval_fn is not None and ((i + 1) % eval_every == 0 or i + 1 == steps):
            score = float(eval_fn(pz[0], pz[1]))
            curve.append(dict(step=i + 1, loss=float(val), select_score=score))
            print(f'   head[{tag}] eval step {i+1}: select-oracle median {score:.4e}', flush=True)
            if best is None or score < best[0]:
                best = (score, i + 1, host(pz[0]), np.asarray(pz[1]))
    if best is not None:
        p_out, Z_out, best_step = best[2], best[3], best[1]
    else:
        p_out, Z_out, best_step = host(pz[0]), np.asarray(pz[1]), steps
    info = dict(steps=steps, best_step=best_step, final_loss=float(val), lr=lr, wd=wd, hidden=hidden,
                layers=layers, seconds=time.time() - t0, curve=curve, n_snapshots=int(S),
                n_params=int(sum(int(np.prod(np.shape(x))) for x in jax.tree_util.tree_leaves(p_out))))
    return p_out, Z_out, info


def oracle_block(head_fn, Rb, G, Cst, Xst, perp, n0, Zc, starts, budget):
    """Best-found and single-start fits on the states Xst (S, n) with coefficients Cst; errors in
    field space with the whitened formula beside them (DESIGN §A4)."""
    Zo, rn, its, reasons = D.oracle_fit(head_fn, Rb, Cst, Zc, n_starts=starts, budget=budget)
    Z1, rn1, _, _ = D.oracle_fit(head_fn, Rb, Cst, Zc, n_starts=1, budget=budget)

    def field_err(Zs):
        H = jax.jit(jax.vmap(head_fn))(jnp.asarray(Zs))
        return np.asarray(jnp.linalg.norm(H @ G.T - Xst, axis=1)) / n0
    e_or, e_1 = field_err(Zo), field_err(Z1)
    e_form = np.sqrt(rn ** 2 + perp ** 2) / n0
    return dict(oracle_median=float(np.median(e_or)), oracle_worst=float(e_or.max()),
                single_start_median=float(np.median(e_1)), single_start_worst=float(e_1.max()),
                bank_floor_median=float(np.median(perp / n0)),
                oracle_iters_median=float(np.median(its)),
                oracle_reasons={str(k): int(v) for k, v in zip(*np.unique(reasons, return_counts=True))},
                budget_exits=int((reasons == 0).sum()), states=int(len(e_or)),
                formula_vs_field_worst_rel=float(np.max(np.abs(e_form - e_or) / e_or)),
                per_state_oracle=e_or.tolist(), per_state_single=e_1.tolist(),
                per_state_bank=(perp / n0).tolist()), Zo


def main():
    t_begin = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ck = pickle.load(open(CKPT, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    K, R = int(ck['cfg']['K']), int(ck['cfg']['R'])
    backend = jax.default_backend()
    report = dict(lane='ns2d', phase='2-headfit', commit=P2.git_commit(), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=backend, x64=bool(jax.config.jax_enable_x64),
                  matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
                  jax_version=jax.__version__, host=platform.node(), device=str(jax.devices()[0]),
                  smoke=bool(SMOKE),
                  checkpoint=dict(path=CKPT, sha256=hashlib.sha256(Path(CKPT).read_bytes()).hexdigest(),
                                  cfg={k: v for k, v in ck['cfg'].items()}),
                  config=dict(K=K, R=R, TRAIN_N=TRAIN_N, SUBSETS=SUBSETS, REGIMES=REGIMES, STEPS=STEPS, LR=LR,
                              WD=WD, EVAL_EVERY=EVAL_EVERY, H_HIDDEN=H_HIDDEN, H_LAYERS=H_LAYERS,
                              H_SMALL_HIDDEN=H_SMALL_HIDDEN, H_SMALL_LAYERS=H_SMALL_LAYERS,
                              REPORT_CASES=REPORT_CASES, SEL_CASES=SEL_CASES, TRAIN_ORACLE_CASES=TRAIN_ORACLE_CASES,
                              SEL_STARTS=SEL_STARTS, SEL_BUDGET=SEL_BUDGET, ORACLE_TIMES=ORACLE_TIMES,
                              ORACLE_STARTS=ORACLE_STARTS, ORACLE_BUDGET=ORACLE_BUDGET, SEED=SEED,
                              N_TRAIN=N_TRAIN, N_DEV=N_DEV, NMODES=P2.NMODES, SEEDS=P2.SEEDS),
                  gates={}, data={}, arms={}, summary={}, complete=False)
    dump(report)
    gate(report, 'S0', (backend == 'gpu' and jax.config.jax_enable_x64
                        and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest') or bool(SMOKE),
         backend=backend)
    gate(report, 'R-CKPT', (not ck['cfg'].get('smoke', True)) or bool(SMOKE), ckpt_smoke=ck['cfg'].get('smoke'))
    expect = json.loads(Path(P2.EXPECT).read_text()) if P2.EXPECT and Path(P2.EXPECT).exists() else {}
    ref = P2.load_ref()
    phys = dict(train=F.params_draw(P2.SEEDS['train'], N_TRAIN, P2.NMODES),
                dev=F.params_draw(P2.SEEDS['dev'], N_DEV, P2.NMODES))
    assert max(SUBSETS) <= N_TRAIN and REPORT_CASES + SEL_CASES <= N_DEV and TRAIN_ORACLE_CASES <= min(SUBSETS)

    # ---- data (dev first, its value gate certifies the node; then the gated base cohort)
    Udev, idev = P2.generate(TRAIN_N, phys['dev'])
    report['data'][f'dev_N{TRAIN_N}'] = idev
    dev_ok = P2.data_gate(report, expect, 'dev', TRAIN_N, Udev, idev, ref)
    Utr, itr = P2.generate(TRAIN_N, phys['train'])
    report['data'][f'train_N{TRAIN_N}'] = itr
    P2.data_gate(report, expect, 'train', TRAIN_N, Utr, itr, ref, dev_ok=dev_ok)
    dump(report)
    S_all = N_TRAIN * NOUT
    Uflat = Utr.reshape(S_all, -1)
    u_ms = float(np.mean(Uflat * Uflat))
    n_pts = TRAIN_N * TRAIN_N

    # ---- the frozen bank, its QR, coefficients of every training snapshot
    G = D.bank_on_grid(params, TRAIN_N)
    Qb, Rb = jnp.linalg.qr(G, mode='reduced')
    d = jnp.abs(jnp.diag(Rb))
    rank = int(jnp.sum(d > d.max() * 1e-12))
    gate(report, f'B-ORTH_N{TRAIN_N}', rank == R, rank=rank, cond_Rb=float(d.max() / d.min()),
         bank_sha256=sha(np.asarray(G)))
    assert rank == R, 'frozen bank must be full rank (DESIGN §A4)'

    def coefs(X):
        Xd = jnp.asarray(X)
        C = jnp.linalg.solve(Rb, Qb.T @ Xd.T).T
        perp = np.asarray(jnp.linalg.norm(Xd - (Xd @ Qb) @ Qb.T, axis=1))
        return np.asarray(C), perp
    C_tr = np.empty((S_all, R))
    perp_tr = np.empty(S_all)
    unorm_tr = np.linalg.norm(Uflat, axis=1)
    for s in range(0, S_all, 2048):
        C_tr[s:s + 2048], perp_tr[s:s + 2048] = coefs(Uflat[s:s + 2048])
    del Utr

    # ---- evaluation states: dev REPORT (cases 0..31), dev SELECT (32..63), TRAIN (traj 0..63)
    def states(U, cases):
        X = np.stack([U[c, t] for c in cases for t in ORACLE_TIMES])
        n0 = np.repeat(np.linalg.norm(U[cases][:, 0], axis=1), len(ORACLE_TIMES))
        C, perp = coefs(X)
        return dict(X=jnp.asarray(X), C=jnp.asarray(C), perp=perp, n0=n0, cases=list(cases))
    ev = dict(dev_report=states(Udev, range(REPORT_CASES)),
              dev_select=states(Udev, range(REPORT_CASES, REPORT_CASES + SEL_CASES)),
              train=states(Uflat.reshape(N_TRAIN, NOUT, -1), range(TRAIN_ORACLE_CASES)))
    del Udev
    report['bank'] = dict(rank=rank, u_ms=u_ms,
                          floors={k: dict(bank_floor_median=float(np.median(v['perp'] / v['n0'])),
                                          bank_floor_worst=float(np.max(v['perp'] / v['n0']))) for k, v in ev.items()})
    # POD-K of every SUBSET's field snapshots (the linear control at matched online dimension,
    # on the same states); computed before the fields are released
    podk = {}
    for n_traj in SUBSETS:
        Vk, sv = P2.pod_snapshots(Uflat[:n_traj * NOUT], K)
        podk[n_traj] = {k: (np.asarray(jnp.linalg.norm(v['X'] - (v['X'] @ Vk) @ Vk.T, axis=1)) / v['n0'])
                        for k, v in ev.items()}
        podk[n_traj]['energy_fraction_K'] = float((sv[:K] ** 2).sum() / (sv ** 2).sum())
        del Vk
    del Uflat
    jax.clear_caches()
    dump(report)

    # ---- arms: subsets x regimes
    regimes = dict(plain=dict(hidden=H_HIDDEN, layers=H_LAYERS, wd=0.0, select=False),
                   reg=dict(hidden=H_HIDDEN, layers=H_LAYERS, wd=WD, select=True),
                   reg_small=dict(hidden=H_SMALL_HIDDEN, layers=H_SMALL_LAYERS, wd=WD, select=True))
    sel = ev['dev_select']
    for n_traj in SUBSETS:
        S = n_traj * NOUT
        Csub = C_tr[:S]
        for name in REGIMES:
            rg = regimes[name]
            tag = f'n{n_traj}_{name}'
            t0 = time.time()

            def select_score(p_, Z_, sel=sel):
                head_fn = lambda z: sc.head(p_, z)                                    # noqa: E731
                _, rn, _, _ = D.oracle_fit(head_fn, Rb, sel['C'], np.asarray(Z_)[::max(1, len(Z_) // 4096)],
                                           n_starts=SEL_STARTS, budget=SEL_BUDGET)
                return np.median(np.sqrt(rn ** 2 + sel['perp'] ** 2) / sel['n0'])
            p_h, Z, tinfo = train_head(jax.random.PRNGKey(SEED), Csub, Rb, n_pts, u_ms, K, R, rg['hidden'],
                                       rg['layers'], STEPS, LR, rg['wd'],
                                       eval_fn=select_score if rg['select'] else None,
                                       eval_every=EVAL_EVERY, tag=tag)
            head_fn = lambda z, p_=p_h: sc.head(p_, z)                                # noqa: E731
            # training reconstruction from the trained codes (field space, own-norm)
            Hs = jax.jit(jax.vmap(head_fn))(jnp.asarray(Z))
            rn_tr = np.asarray(jnp.linalg.norm((Hs - jnp.asarray(Csub)) @ Rb.T, axis=1))
            rec = np.sqrt(rn_tr ** 2 + perp_tr[:S] ** 2) / unorm_tr[:S]
            Zc = np.asarray(Z)[::max(1, S // 8192)]
            arm = dict(n_traj=n_traj, regime=name, arch=dict(hidden=rg['hidden'], layers=rg['layers'], wd=rg['wd'],
                                                              select=rg['select']),
                       training=dict(recon_rel_l2_mean=float(rec.mean()), recon_rel_l2_median=float(np.median(rec)),
                                     recon_rel_l2_max=float(rec.max()), **tinfo),
                       eval={})
            for k, v in ev.items():
                blk, _ = oracle_block(head_fn, Rb, G, v['C'], v['X'], v['perp'], v['n0'], Zc, ORACLE_STARTS, ORACLE_BUDGET)
                e_p = podk[n_traj][k]
                blk['podK_median'], blk['podK_worst'] = float(np.median(e_p)), float(e_p.max())
                blk['per_state_podK'] = e_p.tolist()
                blk['ratio_podK_over_oracle'] = blk['podK_median'] / blk['oracle_median']
                blk['cases'], blk['times'] = v['cases'], ORACLE_TIMES
                arm['eval'][k] = blk
            arm['heldout_over_training_median'] = arm['eval']['dev_report']['oracle_median'] / arm['training']['recon_rel_l2_median']
            arm['heldout_over_train_oracle_median'] = arm['eval']['dev_report']['oracle_median'] / arm['eval']['train']['oracle_median']
            arm['seconds'] = time.time() - t0
            report['arms'][tag] = arm
            with open(OUT / f'head_{tag}.pkl', 'wb') as f:
                pickle.dump(dict(params=host(p_h), Z=np.asarray(Z), n_traj=n_traj, regime=name), f)
            log(f'ARM {tag}: dev-report oracle {arm["eval"]["dev_report"]["oracle_median"]:.4e} '
                f'(POD-{K} {arm["eval"]["dev_report"]["podK_median"]:.4e}, ratio {arm["eval"]["dev_report"]["ratio_podK_over_oracle"]:.3f}) '
                f'train-oracle {arm["eval"]["train"]["oracle_median"]:.4e} recon {arm["training"]["recon_rel_l2_median"]:.4e} '
                f'best_step {tinfo["best_step"]} [{arm["seconds"]:.0f}s]')
            dump(report)
            jax.clear_caches()

    # ---- pre-registered summary (DESIGN §A9): dev-report oracle median vs trajectory count
    for name in REGIMES:
        xs = np.log(np.asarray(SUBSETS, float))
        ys = np.log([report['arms'][f'n{n}_{name}']['eval']['dev_report']['oracle_median'] for n in SUBSETS])
        slope = float(np.polyfit(xs, ys, 1)[0]) if len(SUBSETS) > 1 else None
        report['summary'][name] = dict(
            n_traj=SUBSETS, dev_report_oracle_median=np.exp(ys).tolist(),
            podK_median=[report['arms'][f'n{n}_{name}']['eval']['dev_report']['podK_median'] for n in SUBSETS],
            ratio_podK_over_oracle=[report['arms'][f'n{n}_{name}']['eval']['dev_report']['ratio_podK_over_oracle'] for n in SUBSETS],
            train_oracle_median=[report['arms'][f'n{n}_{name}']['eval']['train']['oracle_median'] for n in SUBSETS],
            loglog_slope=slope,
            verdict=('needs-data' if slope is not None and slope <= -0.25 else
                     'head-limited' if slope is not None and slope >= -0.10 else 'ambiguous'))
        log(f'SUMMARY {name}: oracle {np.exp(ys)} slope {slope} -> {report["summary"][name]["verdict"]}')
    report['all_passed'] = all(g['passed'] for g in report['gates'].values())
    report['complete'] = not SMOKE
    report['seconds'] = time.time() - t_begin
    dump(report)
    log(f'HEADFIT done in {report["seconds"]:.0f}s')


if __name__ == '__main__':
    main()
