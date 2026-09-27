"""Train the Burgers head: data density, objective, latent dimension, bank rank.

The architecture never changes. What changes is how much data the head sees,
what it is asked to minimise, how many latent coordinates it has, and (in the
conditional arm) how wide the bank is. Every arm emits a FROZEN, hashed
checkpoint in the incumbent's own pickle format, which the separate evaluation
job then runs through the unchanged head-ablation arm (a) machinery.

Frozen-bank arms optimise the exact whitened objective of `sep_hfit` -- identity
(*), not an approximation of the field-space loss -- so the span floor is one
constant for every one of them and the only thing that moves is the head's reach
into its own bank. Joint bank+head arms move the bank, so they are graded
against their own re-derived bank algebra and their own span floor.

Predeclared protocol: DESIGN.md beside this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

import engines as e
import arms as A
import sep_hfit as hf
import common as C

F64 = jnp.float64


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False, default=float) + '\n')


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def local_index(global_index, sel, fallback_self=False):
    """Remap a global predecessor/neighbour index onto a selected subset."""
    sel = np.asarray(sel)
    pos = -np.ones(int(np.max(global_index)) + 2, dtype=np.int64)
    pos[sel] = np.arange(len(sel))
    g = np.asarray(global_index)[sel]
    loc = np.where(g >= 0, pos[np.maximum(g, 0)], -1)
    if fallback_self:
        loc = np.where(loc >= 0, loc, np.arange(len(sel)))
    return loc.astype(np.int32)


# ------------------------------------------------------------- the trainer ---

def fit_ext(key, A_tr, un2, fl2, k_lat, R, steps, lr, hidden, layers, batch,
            wp=None, prev=None, nbr=None, nu=None, beta_w=0., beta_t=0., gamma=0.,
            res_batch=256, L=256, dt=.005, log_every=20000, tag='', probe=0):
    """Adam on (q, Z) against  L_rec + beta_w L_W + beta_t L_T + gamma L_Z.

    With beta_w = beta_t = gamma = 0 this is `sep_hfit.fit` at its defaults,
    step for step and random draw for random draw; `smoke_train.py` asserts that
    equality, so an objective arm differs from a density arm in the added term
    and in nothing else.
    """
    S = A_tr.shape[0]
    key, kq, kz = jax.random.split(key, 3)
    qp = hf.init_head(kq, k_lat, R, hidden, layers)
    Z = 0.1 * jax.random.normal(kz, (S, k_lat), dtype=F64)
    inv = 1.0 / jnp.maximum(un2, 1e-300)
    sched = optax.warmup_cosine_decay_schedule(0., lr, min(500, steps // 10 + 1), steps, lr * 1e-2)
    opt = optax.adam(sched)
    state = opt.init((qp, Z))
    solve_aware = (beta_w > 0) or (beta_t > 0)
    nb = min(batch, S)
    prev_j = jnp.asarray(prev)
    nbr_j = jnp.asarray(nbr)
    nu_j = jnp.asarray(nu)
    valid_np = np.nonzero(np.asarray(prev) >= 0)[0]
    assert not solve_aware or len(valid_np) > 0, 'no consecutive state pair: stride must be 1'
    valid = jnp.asarray(valid_np if len(valid_np) else np.zeros(1, dtype=np.int64))
    nr = int(min(res_batch, len(valid_np))) if solve_aware else 1
    anorm2 = jnp.sum((A_tr @ wp['Aw'].T) ** 2, axis=1) if solve_aware else None

    def loss_fn(pz, idx, ridx):
        q, z = pz
        d = hf.head_apply(q, z[idx]) - A_tr[idx]
        rec = jnp.mean(inv[idx] * jnp.sum(d * d, axis=1))
        lw = lt = lz = 0.
        if solve_aware:
            qi = hf.head_apply(q, z[ridx])
            pj = prev_j[ridx]
            den = jnp.maximum(anorm2[ridx], 1e-300)
            if beta_w > 0:
                r = C.weak_residual(wp, qi, A_tr[pj] @ wp['Aw'].T, nu_j[ridx], L, dt)
                lw = jnp.mean(jnp.sum(r * r, axis=1) / den)
            if beta_t > 0:
                r = C.weak_residual(wp, qi, hf.head_apply(q, z[pj]) @ wp['Aw'].T,
                                    nu_j[ridx], L, dt)
                lt = jnp.mean(jnp.sum(r * r, axis=1) / den)
        if gamma > 0:
            p = prev_j[idx]
            pz_ = jnp.where(p[:, None] >= 0, z[jnp.maximum(p, 0)], z[idx])
            s2 = jnp.mean(jnp.sum((z - jnp.mean(z, axis=0)) ** 2, axis=1)) / z.shape[1]
            lz = (jnp.mean(jnp.sum((z[idx] - pz_) ** 2, axis=1))
                  + jnp.mean(jnp.sum((z[idx] - z[nbr_j[idx]]) ** 2, axis=1))) / jnp.maximum(s2, 1e-300)
        total = rec + beta_w * lw + beta_t * lt + gamma * lz
        return total, jnp.asarray([rec, lw, lt, lz], dtype=F64)

    @jax.jit
    def step(pz, st, kk):
        kk, k2 = jax.random.split(kk)
        idx = jax.random.choice(kk, S, shape=(nb,), replace=False)
        ridx = (valid[jax.random.choice(k2, valid.shape[0], shape=(nr,), replace=False)]
                if solve_aware else idx[:1])
        (val, parts), gr = jax.value_and_grad(loss_fn, has_aux=True)(pz, idx, ridx)
        upd, st = opt.update(gr, st, pz)
        return optax.apply_updates(pz, upd), st, val, parts

    pz = (qp, Z)
    t0 = time.time()
    val, parts, trace = jnp.inf, jnp.zeros(4), []
    for i in range(steps):
        key, kk = jax.random.split(key)
        pz, state, val, parts = step(pz, state, kk)
        if i < probe:
            trace.append(float(val))
        if (i + 1) % log_every == 0 or i == 0:
            r = hf.batched_rel(pz[0], pz[1], A_tr, fl2, un2)
            print(f'   fit[{tag}] {i + 1:7d}/{steps} loss {float(val):.4e} '
                  f'recon {float(jnp.mean(r)):.4e} [{time.time() - t0:.0f}s]', flush=True)
    info = dict(steps=int(steps), lr=lr, hidden=hidden, layers=layers, batch=int(nb),
                res_batch=int(nr) if solve_aware else 0, beta_w=float(beta_w),
                beta_t=float(beta_t), gamma=float(gamma), seconds=time.time() - t0,
                final_loss=float(val),
                final_parts=dict(zip(('rec', 'weak', 'traj', 'smooth'),
                                     [float(v) for v in np.asarray(parts)])))
    return pz[0], pz[1], info, trace


# ------------------------------------------------- joint bank + head arms ----

def widen(params, R_new, key):
    """Same feature family, wider: the g-MLP's output width and the head's output
    width grow together. The incumbent's R columns are kept exactly; the new ones
    are freshly initialised. The architecture is unchanged -- only the rank."""
    p = {k: v for k, v in params.items()}
    gw, gb = params['g'][-1]
    R_old = int(gw.shape[1])
    if int(R_new) == R_old:
        return p
    k1, k2, k3 = jax.random.split(key, 3)
    d = int(R_new) - R_old
    p['g'] = list(params['g'][:-1]) + [
        (jnp.concatenate([gw, jax.random.normal(k1, (gw.shape[0], d), dtype=F64)
                          * jnp.sqrt(2. / gw.shape[0])], 1),
         jnp.concatenate([gb, jnp.zeros(d, F64)]))]
    hw, hb = params['h'][-1]
    p['h'] = list(params['h'][:-1]) + [
        (jnp.concatenate([hw, jax.random.normal(k2, (hw.shape[0], d), dtype=F64)
                          * jnp.sqrt(2. / hw.shape[0])], 1),
         jnp.concatenate([hb, jnp.zeros(d, F64)]))]
    p['h_lin'] = jnp.concatenate(
        [params['h_lin'], jax.random.normal(k3, (params['h_lin'].shape[0], d), dtype=F64) * .3], 1)
    return p


def fit_joint(key, params0, xy, Upts, un2, n_full, k_lat, steps, lr, batch, log_every=20000,
              tag='', orth_fraction=0.1, Z0=None):
    """Joint Adam over (bank g, head h, codes Z) against the field-space loss on a
    fixed seeded subset of interior points -- the loss form
    `sep_common.train_autodecoder` uses, minibatched over snapshots because the
    full snapshot block does not fit.

    Two deliberate differences from that function, both recorded in DESIGN.md:
    the sum over the P sampled points is rescaled by n_full / P so the data term
    reads as the per-snapshot relative MSE the frozen-bank arms also minimise;
    and `lam_orth` is CALIBRATED rather than inherited. The feature-Gram
    orthonormality regulariser exists to condition a FRESHLY initialised bank;
    these arms warm start from an already-trained bank whose Gram is far from
    the identity, so the inherited constant 1e-4 makes that penalty orders of
    magnitude larger than the data term and a joint run under it is an
    orthonormalisation, not a refinement -- while switching it off entirely lets
    the bank's conditioning run away. It is therefore set, once, at the warm
    start, so the penalty contributes `orth_fraction` of the data term there,
    by the same scale rule the solve-aware objective weights use. The realised
    weight, the feature-Gram deviation before and after, and the resulting Gram
    condition number are all reported."""
    S = Upts.shape[0]
    key, kz = jax.random.split(key)
    Z = (jnp.asarray(Z0, dtype=F64) if Z0 is not None
         else 0.1 * jax.random.normal(kz, (S, k_lat), dtype=F64))
    assert Z.shape == (S, k_lat), (Z.shape, S, k_lat)
    # the sampled-point sum estimates (P / n_full) of the full-grid squared error,
    # so scaling it back by n_full / P makes `base` the per-snapshot relative MSE
    inv = (float(n_full) / Upts.shape[1]) / jnp.maximum(un2, 1e-300)
    sched = optax.warmup_cosine_decay_schedule(0., lr, min(500, steps // 10 + 1), steps, lr * 1e-2)
    opt = optax.adam(sched)
    state = opt.init((params0, Z))
    nb = min(batch, S)
    xy = jnp.asarray(xy)

    def orth_of(pp):
        G = A.sc.features(pp, xy)
        Cg = (G.T @ G) / (G.shape[0] * pp['out_scale'] ** 2)
        return jnp.mean((Cg - jnp.eye(Cg.shape[0], dtype=F64)) ** 2)

    def data_of(pp, z, idx):
        G = A.sc.features(pp, xy)
        d = A.sc.head(pp, z[idx]) @ G.T - Upts[idx]
        return jnp.mean(inv[idx] * jnp.sum(d * d, axis=1))

    orth0 = float(orth_of(params0))
    base0 = float(data_of(params0, Z, jnp.arange(nb)))
    lam_orth = float(orth_fraction) * base0 / max(orth0, 1e-300)

    def loss_fn(pz, idx):
        pp, z = pz
        G = A.sc.features(pp, xy)
        d = A.sc.head(pp, z[idx]) @ G.T - Upts[idx]
        base = jnp.mean(inv[idx] * jnp.sum(d * d, axis=1))
        Cg = (G.T @ G) / (G.shape[0] * pp['out_scale'] ** 2)
        return base + lam_orth * jnp.mean((Cg - jnp.eye(Cg.shape[0], dtype=F64)) ** 2), base

    @jax.jit
    def step(pz, st, kk):
        idx = jax.random.choice(kk, S, shape=(nb,), replace=False)
        (val, base), gr = jax.value_and_grad(loss_fn, has_aux=True)(pz, idx)
        gr[0]['out_scale'] = jnp.zeros_like(gr[0]['out_scale'])
        upd, st = opt.update(gr, st, pz)
        return optax.apply_updates(pz, upd), st, val, base

    pz = (params0, Z)
    t0 = time.time()
    val = base = jnp.inf
    for i in range(steps):
        key, kk = jax.random.split(key)
        pz, state, val, base = step(pz, state, kk)
        if (i + 1) % log_every == 0 or i == 0:
            print(f'   joint[{tag}] {i + 1:7d}/{steps} loss {float(val):.4e} '
                  f'data {float(base):.4e} [{time.time() - t0:.0f}s]', flush=True)
    return pz[0], pz[1], dict(steps=int(steps), lr=lr, batch=int(nb), points=int(xy.shape[0]),
                              n_full=int(n_full), lam_orth=float(lam_orth),
                              orth_fraction=float(orth_fraction),
                              data_term_at_warm_start=base0,
                              warm_start=Z0 is not None, seconds=time.time() - t0,
                              final_loss=float(val), final_data_loss=float(base),
                              feature_gram_deviation_start=orth0,
                              feature_gram_deviation_end=float(orth_of(pz[0])))


# ------------------------------------------------------- held-out grading ----

def train_encoder(key, Aw, Z, steps):
    """Small MLP a -> z on TRAINING pairs only; one of the two oracle inits, the
    protocol `sep_hfit_run` uses."""
    ep = hf.init_head(key, Aw.shape[1], Z.shape[1], hidden=256, layers=2, lin_scale=0.)
    sch = optax.warmup_cosine_decay_schedule(0., 1e-3, min(500, steps // 10 + 1), steps, 1e-5)
    o = optax.adam(sch)
    st = o.init(ep)
    zs = float(jnp.std(Z)) + 1e-30

    def ls(p, Ab, Zb):
        return jnp.mean((hf.head_apply(p, Ab) - Zb) ** 2) / zs ** 2

    @jax.jit
    def stp(p, st, kk):
        nbb = min(4096, Aw.shape[0])
        i = jax.random.choice(kk, Aw.shape[0], shape=(nbb,), replace=False)
        v, g = jax.value_and_grad(ls)(p, Aw[i], Z[i])
        u, st = o.update(g, st)
        return optax.apply_updates(p, u), st, v
    v = jnp.inf
    for _ in range(steps):
        key, kk = jax.random.split(key)
        ep, st, v = stp(ep, st, kk)
    return ep, float(v)


def stats(rel):
    rel = np.asarray(rel)
    return dict(mean=float(rel.mean()), max=float(rel.max()), median=float(np.median(rel)),
                n=int(len(rel)))


def grade(qp, Z, A_ho, fl2_ho, un2_ho, seed, iters, A_tr=None, enc_steps=0):
    """Representation oracle on states no arm ever fitted.

    `mean_only` uses the mean training code as the single initialisation and is
    computed for EVERY arm, so frozen-bank and joint arms are comparable on one
    column. `with_encoder` adds the training-only encoder init and exists only
    where the arm's own whitened training block is available."""
    zbar = jnp.mean(Z, axis=0)
    mo, _ = hf.oracle(qp, A_ho, fl2_ho, un2_ho,
                      jnp.tile(zbar[None, None], (A_ho.shape[0], 1, 1)), iters)
    out = dict(mean_only=stats(mo))
    if A_tr is not None and enc_steps:
        ep, ev = train_encoder(jax.random.PRNGKey(seed + 7), A_tr, Z, enc_steps)
        inits = jnp.stack([jnp.tile(zbar[None], (A_ho.shape[0], 1)),
                           hf.head_apply(ep, A_ho)], axis=1)
        we, _ = hf.oracle(qp, A_ho, fl2_ho, un2_ho, inits, iters)
        out['with_encoder'] = stats(we)
        out['encoder_final_loss'] = ev
    out['selection'] = out.get('with_encoder', out['mean_only'])
    return out


# ------------------------------------------------------------------ driver ---

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    L, dt = cfg['intervals'], cfg['dt']
    ck = pickle.load(open(a.checkpoint, 'rb'))
    params0 = jax.tree_util.tree_map(jnp.asarray, C.host(ck['params']))
    K0 = int(np.asarray(ck['Z_tr']).shape[1])
    R0 = int(np.asarray(ck['params']['h_lin']).shape[1])

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                  gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                  jax_version=jax.__version__, source_checkpoint_sha256=sha_file(a.checkpoint),
                  K_incumbent=K0, R_incumbent=R0, gates={}, data={}, weights={},
                  arms=[], emitted=[], selection={}, final_cohort_unopened=True, complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    # ------------------------------------------------------- frozen bank -----
    bank = A.CoordBank(params0, K0, R0)
    G = jax.block_until_ready(bank.on_grid(L))
    Gram, Lam, Linv = C.bank_algebra(G)
    project = C.make_projector(G, Lam)
    qp_ck = hf.to_q(dict(h=params0['h'], h_lin=params0['h_lin']), Lam)
    hp_rt = hf.to_h(qp_ck, Linv)
    rt = float(max(jnp.max(jnp.abs(hp_rt['h'][-1][0] - params0['h'][-1][0]))
                   / jnp.max(jnp.abs(params0['h'][-1][0])),
                   jnp.max(jnp.abs(hp_rt['h_lin'] - params0['h_lin']))
                   / jnp.max(jnp.abs(params0['h_lin']))))
    report['gates']['whitening_round_trip'] = rt
    assert rt < 1e-10, rt
    save()

    # ---------------------------------------------------------- the draws ----
    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    traj_all = C.incumbent_draw(cfg['canonical_trajectories'], cfg['extra_trajectories'],
                                cfg['canonical_seed'], cfg['extra_seed'])
    hold_physical = e.params_draw(cfg['holdout_seed'], cfg['holdout_trajectories'])
    C.assert_disjoint(traj_all, physical, 'training/eval')
    C.assert_disjoint(hold_physical, physical, 'holdout/eval')
    C.assert_disjoint(hold_physical, traj_all, 'holdout/training')
    report['data'].update(train_physical_sha256=C.sha_array(traj_all),
                          holdout_physical_sha256=C.sha_array(hold_physical),
                          eval_physical_sha256=C.sha_array(physical))
    save()

    densities = cfg['densities']
    tr, tinfo = C.generate_projected(L, dt, traj_all[:max(densities)], project,
                                     cfg['snapshot_ntol'], cfg['snapshot_ltol'],
                                     stride=cfg['state_stride'], progress=256)
    report['data']['training'] = tinfo
    save()
    print('TRAIN DATA', round(time.perf_counter() - begin, 1), flush=True)

    ho, hinfo = C.generate_projected(L, dt, hold_physical, project, cfg['snapshot_ntol'],
                                     cfg['snapshot_ltol'], stride=cfg['state_stride'],
                                     progress=16, points=np.arange((L - 1) ** 2))
    U_hold = ho.pop('points')
    report['data']['holdout'] = hinfo
    A_ho, fl2_ho, un2_ho = (jnp.asarray(ho['a']), jnp.asarray(ho['fl2']), jnp.asarray(ho['un2']))
    hold_floor = np.sqrt(ho['fl2'] / ho['un2'])
    report['data']['holdout_span_floor'] = dict(mean=float(hold_floor.mean()),
                                                max=float(hold_floor.max()))
    save()
    print('HOLDOUT DATA', round(time.perf_counter() - begin, 1), flush=True)

    prev_all = C.previous_index(tr['traj'], tr['t'])
    nbr_all = C.neighbour_index(traj_all[:max(densities)], tr['traj'], tr['t'])
    A_all = jnp.asarray(tr['a'])
    wps = {k: C.weak_pieces(G, Linv, L, cfg['test_multiplier'] * k) for k in cfg['latent_dims']}

    # gate: identity (*) against fields the FOM regenerates
    q, _ = e.make_fom(L, dt, None, .25, dt)
    dev = 0.
    for s in np.linspace(0, A_all.shape[0] - 1, cfg['identity_rows']).astype(int):
        ti, tj = int(tr['traj'][s]), int(tr['t'][s])
        f = np.asarray(q(jnp.asarray(e.initial(L, traj_all[ti])), float(traj_all[ti][4]),
                         cfg['snapshot_ntol'], cfg['snapshot_ltol'])[0])
        u = f[tj][1:-1, 1:-1].reshape(-1)
        c = np.asarray(jnp.asarray(tr['a'][s])[None] @ Linv)[0]
        lhs = float(np.linalg.norm(np.asarray(G @ jnp.asarray(c)) - u) ** 2)
        dev = max(dev, abs(lhs - float(tr['fl2'][s])) / max(float(tr['fl2'][s]), 1e-300))
    del q
    jax.clear_caches()
    report['gates']['identity_star_relative'] = dev
    assert dev < 1e-9, dev
    save()
    print('GATES', report['gates'], flush=True)

    # ------------------------------ objective weights, set once, before any arm
    rho = cfg['objective_fraction']
    cal = np.asarray([i for i in np.linspace(0, A_all.shape[0] - 1, 4 * cfg['calibration_states'])
                      .astype(int) if prev_all[i] >= 0])[:cfg['calibration_states']]
    cj, pj = jnp.asarray(cal), jnp.asarray(prev_all[cal])
    Zck = jnp.asarray(ck['Z_tr'])
    zc = Zck[np.linspace(0, Zck.shape[0] - 1, len(cal)).astype(int)]
    qc = hf.head_apply(qp_ck, zc)
    d = qc - A_all[cj]
    l_rec = float(jnp.mean(jnp.sum(d * d, axis=1) / jnp.maximum(jnp.asarray(tr['un2'])[cj], 1e-300)))
    wp0 = wps[K0]
    an2 = jnp.maximum(jnp.sum((A_all[cj] @ wp0['Aw'].T) ** 2, axis=1), 1e-300)
    rw = C.weak_residual(wp0, qc, A_all[pj] @ wp0['Aw'].T, jnp.asarray(tr['nu'])[cj], L, dt)
    l_w = float(jnp.mean(jnp.sum(rw * rw, axis=1) / an2))
    s2 = float(jnp.mean(jnp.sum((zc - jnp.mean(zc, 0)) ** 2, axis=1)) / K0)
    l_z = float(jnp.mean(jnp.sum((zc[1:] - zc[:-1]) ** 2, axis=1)) / max(s2, 1e-300))
    bw = rho * l_rec / max(l_w, 1e-300)
    gm = rho * l_rec / max(l_z, 1e-300)
    report['weights'] = dict(
        rule=f'weight = rho * L_rec / L_term at the INCUMBENT head on a training-cohort '
             f'calibration batch; rho = {rho}. No accuracy number is consulted.',
        calibration_states=int(len(cal)), L_rec=l_rec, L_weak=l_w, L_smooth=l_z,
        beta_w=bw, beta_t=bw, gamma=gm)
    save()
    print('WEIGHTS', report['weights'], flush=True)

    hidden, layers = cfg['head_hidden'], cfg['head_layers']

    def emit(name, params, Z, extra):
        newp = {kk: (np.asarray(vv) if not isinstance(vv, list)
                     else [(np.asarray(w), np.asarray(b)) for w, b in vv])
                for kk, vv in params.items()}
        meta = dict(ck.get('cfg', {}))
        meta.pop('hfit_pick', None)
        meta.update(extra)
        path = out / f'ckpt_{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(dict(params=newp, Z_tr=np.asarray(Z), cfg=meta), f)
        return path.name, sha_file(path)

    store = {}

    def record(name, spec, train_info, grades, floor, params_out, Z, recon=None, R_out=R0):
        if spec['bank'] == 'frozen':
            store[name] = (dict(h=params_out['h'], h_lin=params_out['h_lin']), np.asarray(Z))
        art, sha = emit(name, params_out, Z, dict(btrain_arm=name, btrain_spec=spec,
                                                  k=int(Z.shape[1]), r=int(R_out),
                                                  btrain_source=os.path.basename(a.checkpoint)))
        row = dict(arm=name, spec=spec, K=int(Z.shape[1]), R=int(R_out),
                   trajectories=int(spec['trajectories']), states=int(len(Z)),
                   train=train_info, recon_train=recon, holdout=grades,
                   holdout_over_floor=float(grades['selection']['mean'] / floor.mean()),
                   span_floor=dict(mean=float(floor.mean()), max=float(floor.max())),
                   checkpoint=art, checkpoint_sha256=sha,
                   train_gpu_hours=float(train_info['seconds'] / 3600.))
        report['arms'].append(row)
        report['emitted'].append(dict(arm=name, artifact=art, sha256=sha, K=int(Z.shape[1]),
                                      R=int(R_out)))
        save()
        print(f"ARM {name} holdout worst {grades['selection']['max']:.4e} mean "
              f"{grades['selection']['mean']:.4e} ({row['holdout_over_floor']:.1f}x floor) "
              f"[{round(time.perf_counter() - begin, 1)}s]", flush=True)
        return row

    def run_frozen(name, ntraj, K, beta_w=0., beta_t=0., gamma=0.):
        sel = np.nonzero(np.asarray(tr['traj']) < ntraj)[0]
        sj = jnp.asarray(sel)
        spec = dict(trajectories=int(ntraj), K=int(K), bank='frozen', rank=int(R0),
                    beta_w=float(beta_w), beta_t=float(beta_t), gamma=float(gamma),
                    hidden=hidden, layers=layers, steps=int(cfg['steps']),
                    objective=('rec' if not (beta_w or beta_t or gamma)
                               else 'w' if beta_w else 't' if beta_t else 'z'))
        qp, Z, ti, _ = fit_ext(jax.random.PRNGKey(cfg['seed'] + 11), A_all[sj],
                               jnp.asarray(tr['un2'][sel]), jnp.asarray(tr['fl2'][sel]), K, R0,
                               steps=cfg['steps'], lr=cfg['lr'], hidden=hidden, layers=layers,
                               batch=cfg['batch'], wp=wps[K], prev=local_index(prev_all, sel),
                               nbr=local_index(nbr_all, sel, fallback_self=True),
                               nu=tr['nu'][sel], beta_w=beta_w, beta_t=beta_t, gamma=gamma,
                               res_batch=cfg['res_batch'], L=L, dt=dt, tag=name)
        rec = np.asarray(hf.batched_rel(qp, Z, A_all[sj], jnp.asarray(tr['fl2'][sel]),
                                        jnp.asarray(tr['un2'][sel])))
        gr = grade(qp, Z, A_ho, fl2_ho, un2_ho, cfg['seed'], cfg['oracle_iters'],
                   A_tr=A_all[sj], enc_steps=cfg['encoder_steps'])
        hp = hf.to_h(qp, Linv)
        pout = {kk: vv for kk, vv in params0.items()}
        pout['h'], pout['h_lin'] = hp['h'], hp['h_lin']
        return record(name, spec, ti, gr, hold_floor, pout, Z,
                      recon=dict(mean=float(rec.mean()), max=float(rec.max())))

    # ------------------------------------------------------------ L1 ladder ---
    for n in densities:
        run_frozen(f'd{n}k{K0}rec', n, K0)
    # ------------------------------------------------------------ L3 latent ---
    for K in cfg['latent_dims']:
        if K != K0:
            for n in cfg['capacity_densities']:
                run_frozen(f'd{n}k{K}rec', n, K)
    # ------------------------------------------------------------ L2 objective
    bn = cfg['objective_density']
    run_frozen(f'd{bn}k{K0}w', bn, K0, beta_w=bw)
    run_frozen(f'd{bn}k{K0}t', bn, K0, beta_t=bw)
    run_frozen(f'd{bn}k{K0}z', bn, K0, gamma=gm)

    # ------------------------------------------- pre-registered selections ----
    def worst(name):
        return next(r for r in report['arms'] if r['arm'] == name)['holdout']['selection']['max']

    curve = [(n, worst(f'd{n}k{K0}rec')) for n in densities]
    monotone = all(curve[i][1] >= curve[i + 1][1] for i in range(len(curve) - 1))
    # the saturation rung is the SECOND density in the declared ladder (512 in
    # config-train.json); the comparison rungs are recorded so the verdict cannot
    # be read off the wrong pair
    i_mid = min(1, max(len(curve) - 2, 0))
    g_mid = 1. - curve[i_mid + 1][1] / curve[i_mid][1]
    g_top = 1. - curve[-1][1] / curve[i_mid][1]
    sat = cfg['saturation_fraction']
    verdict = ('data-limited' if (monotone and g_top > sat)
               else 'capacity/objective-limited' if g_mid < sat else 'mixed')
    best_n = min(curve, key=lambda x: x[1])[0]
    kc = [(K, worst(f'd{bn}k{K}rec')) for K in cfg['latent_dims']
          if any(r['arm'] == f'd{bn}k{K}rec' for r in report['arms'])]
    best_K = min(kc, key=lambda x: x[1])[0] if kc else K0
    oc = [(o, worst(f'd{bn}k{K0}{o}')) for o in ('rec', 'w', 't', 'z')]
    best_obj = min(oc, key=lambda x: x[1])[0]
    report['selection'] = dict(
        rule='held-out representation-oracle WORST, declared in DESIGN.md before any run',
        density_curve=curve, monotone=bool(monotone),
        saturation_rung=int(curve[i_mid][0]), next_rung=int(curve[i_mid + 1][0]),
        top_rung=int(curve[-1][0]), relative_gain_mid_to_next=float(g_mid),
        relative_gain_mid_to_top=float(g_top), saturation_fraction=sat,
        diagnostic_verdict=verdict, best_density=int(best_n), latent_curve=kc,
        best_K=int(best_K), objective_curve=oc, best_objective=best_obj)
    save()
    print('SELECTION', json.dumps(report['selection'], default=float), flush=True)

    combo = f'best_d{best_n}k{best_K}{best_obj}'
    already = next((r for r in report['arms']
                    if r['trajectories'] == best_n and r['K'] == best_K
                    and r['spec']['objective'] == best_obj and r['spec']['bank'] == 'frozen'), None)
    if already is None:
        run_frozen(combo, best_n, best_K, beta_w=bw if best_obj == 'w' else 0.,
                   beta_t=bw if best_obj == 't' else 0., gamma=gm if best_obj == 'z' else 0.)
        report['selection']['best_combination_arm'] = combo
    else:
        report['selection']['best_combination_arm'] = already['arm']
        report['selection']['best_combination_note'] = (
            'the selected combination coincides with an arm already run; not rerun')
    save()

    # ------------------------------------------------- joint bank + head ------
    if cfg['joint_steps']:
        rng = np.random.default_rng(cfg['point_seed'])
        pts = np.sort(rng.choice((L - 1) ** 2, cfg['joint_points'], replace=False))
        # the joint arms hold raw field values, so their density is capped: the
        # sampled-point block is (states x points) and grows with both
        joint_n = int(min(best_n, cfg['joint_density_cap']))
        jt, jinfo = C.generate_projected(L, dt, traj_all[:joint_n], project, cfg['snapshot_ntol'],
                                         cfg['snapshot_ltol'], stride=cfg['state_stride'],
                                         progress=256, points=pts)
        jinfo['density_cap'] = int(cfg['joint_density_cap'])
        jinfo['selected_density'] = joint_n
        report['data']['joint'] = jinfo
        Upts = jnp.asarray(jt['points'])
        save()
        ranks = [R0] + ([cfg['wide_rank']] if verdict != 'data-limited' else [])
        report['selection']['bank_rank_arms'] = ranks
        save()
        # the joint arms CONTINUE from the selected frozen-bank arm rather than
        # cold-starting: the question is what unfreezing the bank adds on top of
        # the best head-only fit, not whether a short joint run can rediscover it.
        src = report['selection']['best_combination_arm']
        warm_h, warm_Z = store[src]
        assert len(warm_Z) >= Upts.shape[0], (src, len(warm_Z), Upts.shape[0])
        report['selection']['joint_warm_start'] = src
        save()
        for Rn in ranks:
            name = f'joint_d{joint_n}k{best_K}r{Rn}'
            pin = {kk: vv for kk, vv in params0.items()}
            pin['h'], pin['h_lin'] = warm_h['h'], warm_h['h_lin']
            pin = widen(pin, Rn, jax.random.PRNGKey(cfg['seed'] + 31))
            pj_, Zj, ji = fit_joint(jax.random.PRNGKey(cfg['seed'] + 41), pin,
                                    e.coords(L)[pts], Upts, jnp.asarray(jt['un2']),
                                    (L - 1) ** 2, best_K, cfg['joint_steps'], cfg['lr'],
                                    cfg['batch'], tag=name,
                                    orth_fraction=cfg['joint_orth_fraction'],
                                    Z0=warm_Z[:Upts.shape[0]])
            bj = A.CoordBank(pj_, best_K, Rn)
            Gj = jax.block_until_ready(bj.on_grid(L))
            Gramj, Lamj, Linvj = C.bank_algebra(Gj)
            ji['bank_gram_condition'] = float(jnp.linalg.cond(Gramj))
            proj_j = C.make_projector(Gj, Lamj)
            aj, u2j, f2j = [], [], []
            for s in range(0, U_hold.shape[0], 256):
                x, y, z = proj_j(jnp.asarray(U_hold[s:s + 256]))
                aj.append(np.asarray(x)); u2j.append(np.asarray(y)); f2j.append(np.asarray(z))
            aj, u2j, f2j = np.concatenate(aj), np.concatenate(u2j), np.concatenate(f2j)
            floor_j = np.sqrt(f2j / u2j)
            qpj = hf.to_q(dict(h=pj_['h'], h_lin=pj_['h_lin']), Lamj)
            gr = grade(qpj, Zj, jnp.asarray(aj), jnp.asarray(f2j), jnp.asarray(u2j),
                       cfg['seed'], cfg['oracle_iters'])
            spec = dict(trajectories=int(joint_n), K=int(best_K), bank='joint', rank=int(Rn),
                        beta_w=0., beta_t=0., gamma=0., hidden=hidden, layers=layers,
                        steps=int(cfg['joint_steps']), objective='rec_field',
                        points=int(cfg['joint_points']), warm_start_from=src)
            record(name, spec, ji, gr, floor_j, pj_, Zj, R_out=Rn)
            del Gj, bj, proj_j
            jax.clear_caches()
        del Upts
        jax.clear_caches()

    report['elapsed_seconds'] = time.perf_counter() - begin
    report['source_checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['source_checkpoint_sha256'] == report['source_checkpoint_sha256_after']
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
