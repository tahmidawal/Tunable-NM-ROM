"""jcp-smooth-bank trainer (DESIGN.md §2-§3): the dn256b bank recipe with two levers, nothing else changed.

Data   = sep_burgers_r3.build_data_lean (copied below, POOL = all interior points): the canonical seed-0 draw of 576
         trajectories from deps/burgers2d_film.py (the byte-identical generator of the original job, sha256 c521b36a...),
         `nodes` points per axis INCLUDING the boundary (h = 1/(nodes-1)); the seed-0 state pick of 16384 states.
Train  = sep_solvers.train_autodecoder_v2 (copied below; deps/sep_solvers_reference.py is the r3a job's staged copy, git 5ae420414, sha 5af7056b) with
           * the random-Fourier-feature scale `sigma` (init_separable ff_scale; 4.0 = original),
           * an OPTIONAL Sobolev term  lam * mean ||grad u_hat - D_h u||^2 / mean ||D_h u||^2  on a random subset of
             `sob_states` states and the step's p_sub points (a separate key stream, so the value path and its random
             draws are bit-for-bit those of the original when lam = 0; lam = 0 skips the term at trace time).
         D_h = second-order central differences on the training mesh (boundary neighbours are the exact zeros).
Post   = least-squares coefficients of the training states on the bank (training mesh interior) and the importance
         rotation of burgers-bank-knob/make_rotation.py with h(z_i) replaced by those coefficients (DESIGN §2).

    python smoothtrain.py --arm base --sigma 4 --lam 0 --nodes 256 --out <dir> [--frozen <pkl>] [--steps N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'deps'))

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

import sep_common as sc            # deps/sep_common.py = the r3a job's staged copy (git 33322fda1, sha 1f1c2796)
import burgers2d_film as bf        # deps/burgers2d_film.py, byte-identical to the original job's staged generator

F64 = jnp.float64
# the original recipe (sep_burgers_r3 defaults + the dn256b arch overrides n_ff=128, g_hidden=1024, h_hidden=256)
RECIPE = dict(K=16, R=512, steps=300000, lr=1e-3, p_sub=4096, wd=1e-5, ema_decay=0.999, full_last=10000,
              lam_orth=1e-4, max_snaps=16384, t_early=5, seed=0, arch=dict(n_ff=128, g_hidden=1024, h_hidden=256))


def log(*a):
    print(*a, flush=True)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


# ------------------------------------------------------------------ data (sep_burgers_r3 copy) ----

def grid_coords(n):                                   # blat_common.grid_coords
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)


def interior_indices(n):                              # blat_common.interior_indices
    ii, jj = np.meshgrid(np.arange(1, n - 1), np.arange(1, n - 1), indexing="ij")
    return (ii * n + jj).reshape(-1)


def state_pick(n_traj, T, max_snaps, t_early, seed):
    """sep_burgers_r3.main's state pick (the rng's first draw); POOL=0 draws no point pool."""
    rng = np.random.default_rng(seed)
    n_states = n_traj * T
    tidx_of = np.arange(n_states) % T
    early = np.nonzero(tidx_of <= t_early)[0]
    rest = np.nonzero(tidx_of > t_early)[0]
    if early.size >= max_snaps:
        pick = np.sort(rng.choice(early, max_snaps, replace=False))
    else:
        extra = rng.choice(rest, min(max_snaps - early.size, rest.size), replace=False)
        pick = np.sort(np.concatenate([early, extra]))
    return pick, rng


def build_data(n, n_traj, pick, gen_chunk=64):
    """sep_burgers_r3.build_data_lean with pool = all interior points (as in the original job) and no full-row copy.
    Returns S_tr (n_pick, (n-2)^2) and the worst FOM relative residual (abort above 1e-8, as the original)."""
    interior = interior_indices(n)
    T = bf.NUM_STEPS + 1
    cx, cy, w, a, nu, _z = bf.sample_params(seed=bf.SEED)
    assert n_traj <= len(cx)
    rollout, res_fn = bf.make_rollout(n)
    chk = jax.jit(jax.vmap(lambda u1, u0, nu_: jnp.linalg.norm(res_fn(u1, u0, nu_)) / jnp.linalg.norm(u0)))
    pick_set = {int(v): i for i, v in enumerate(pick)}
    S_tr = np.zeros((pick.size, interior.size), dtype=np.float64)
    worst = 0.0
    fp_sum = fp_sumsq = 0.0
    t0 = time.time()
    for s in range(0, n_traj, gen_chunk):
        e = min(s + gen_chunk, n_traj)
        U0 = np.stack([bf.blob_ic(n, cx[i], cy[i], w[i], a[i]) for i in range(s, e)])
        nu_j = jnp.asarray(nu[s:e])
        snaps, res = rollout(jnp.asarray(U0), nu_j)
        cm = float(jnp.max(res))
        worst = cm if (not np.isfinite(cm) or cm > worst) else worst
        for k in range(bf.NUM_STEPS):
            wr = float(jnp.max(chk(snaps[k + 1], snaps[k], nu_j)))
            worst = wr if (not np.isfinite(wr) or wr > worst) else worst
        snaps_np = np.asarray(snaps)
        fp_sum += float(np.sum(snaps_np))                 # the original's data fingerprint
        fp_sumsq += float(np.sum(snaps_np * snaps_np))
        for b in range(e - s):
            for t in range(T):
                row = pick_set.get((s + b) * T + t)
                if row is not None:
                    S_tr[row] = snaps_np[t, b][interior]
        del snaps, snaps_np
        log(f'   gen n={n}: trajectories {e}/{n_traj} worst FOM rel residual {worst:.2e} [{time.time()-t0:.0f}s]')
    if not np.isfinite(worst) or worst > 1e-8:
        raise SystemExit(f'FOM residual {worst:.2e} > 1e-8: data not converged')
    return S_tr, worst, dict(sum=fp_sum, sumsq=fp_sumsq, shape=[int(n_traj), int(T), int(n * n)])


def neighbours(n):
    """(n_i2, 4) column indices of the x+, x-, y+, y- neighbours of every interior node in the interior ordering;
    a boundary neighbour maps to the sentinel column n_i2 (an appended zero column: exact Dirichlet zeros)."""
    m = n - 2
    k = np.arange(m * m)
    ii, jj = k // m, k % m                       # 0-based interior indices (x index first, as grid_coords 'ij')
    nb = np.full((m * m, 4), m * m, dtype=np.int64)
    nb[ii + 1 < m, 0] = (k + m)[ii + 1 < m]
    nb[ii - 1 >= 0, 1] = (k - m)[ii - 1 >= 0]
    nb[jj + 1 < m, 2] = (k + 1)[jj + 1 < m]
    nb[jj - 1 >= 0, 3] = (k - 1)[jj - 1 >= 0]
    return nb


def b_stats(B):
    """Quantiles of the Fourier-feature column norms ||B_j|| (cycles per unit length)."""
    nrm = np.linalg.norm(np.asarray(B), axis=0)
    return {q: float(np.quantile(nrm, float(q))) for q in ('0', '0.25', '0.5', '0.75', '0.9', '1')}


def fd_bias(U, n, chunk=256):
    """FD-target uncertainty (DESIGN A1.3): ||D2 u - D4 u|| / ||D4 u|| per state, second- vs fourth-order central
    differences of grad u at interior nodes at least two nodes from the wall (Dirichlet zeros padded)."""
    m, h = n - 2, 1. / (n - 1)

    @jax.jit
    def one(Ub):
        V = jnp.pad(Ub.reshape(-1, m, m), ((0, 0), (2, 2), (2, 2)))  # wall zeros + one ghost ring (unused region)
        c = slice(3, m + 1)                                         # interior nodes >= 2 from the wall
        def d2(ax):
            return (jnp.roll(V, -1, ax) - jnp.roll(V, 1, ax))[:, c, c] / (2 * h)
        def d4(ax):
            return (-jnp.roll(V, -2, ax) + 8 * jnp.roll(V, -1, ax) - 8 * jnp.roll(V, 1, ax) + jnp.roll(V, 2, ax))[:, c, c] / (12 * h)
        num = jnp.sum((d2(1) - d4(1)) ** 2 + (d2(2) - d4(2)) ** 2, axis=(1, 2))
        den = jnp.sum(d4(1) ** 2 + d4(2) ** 2, axis=(1, 2))
        return jnp.sqrt(num / den)
    r = np.concatenate([np.asarray(one(jnp.asarray(U[s:s + chunk]))) for s in range(0, len(U), chunk)])
    return dict(median=float(np.median(r)), p90=float(np.quantile(r, .9)), max=float(r.max()))


def feat_grad(p, X):
    """Bank values and exact x, y derivatives at points X (P, 2): three (P, R) arrays (forward mode, per-row)."""
    f = lambda X_: sc.features(p, X_)
    ex = jnp.zeros_like(X).at[:, 0].set(1.)
    ey = jnp.zeros_like(X).at[:, 1].set(1.)
    G, Gx = jax.jvp(f, (X,), (ex,))
    _, Gy = jax.jvp(f, (X,), (ey,))
    return G, Gx, Gy


# ------------------------------------------------------- trainer (train_autodecoder_v2 copy) ----

def train(key, coords, U, nb, h, k_lat, r_feat, steps, lr, lam_orth, weight_decay, p_sub, ema_decay,
          full_last, lam_sob=0.0, sob_states=2048, sob_seed=12345, log_every=5000, tag='', recon_chunk=2048,
          record_idx=None, check_hist=None, **arch):
    """sep_solvers.train_autodecoder_v2 (snap_norm=False, w_extra=None, z_polish=0, time_cap=0: the dn256b call),
    with the optional Sobolev term. Lines marked [SOB] are the only additions; everything else is the original's
    text for this call (rel = mean(err^2)/mean(U^2), as in the staged sep_solvers 5af7056b)."""
    coords = jnp.asarray(coords, dtype=F64)
    U = jnp.asarray(U, dtype=F64)
    S, n_pts = U.shape
    key, kz, kp = jax.random.split(key, 3)
    u_rms = float(jnp.sqrt(jnp.mean(U * U)))
    params = sc.init_separable(kp, k_lat, r_feat, out_scale=u_rms, **arch)
    Z = 0.1 * jax.random.normal(kz, (S, k_lat), dtype=F64)
    u_ms = jnp.mean(U * U)
    b_init = b_stats(params['B'])
    nb_j = jnp.asarray(nb)

    def fd(U_, sidx, idx):                                                   # [SOB] D_h u at (states, points)
        cols = nb_j[idx]                                                     # (P, 4); n_pts = boundary sentinel
        g = U_[sidx[:, None, None], jnp.minimum(cols, n_pts - 1)[None, :, :]]   # (Sg, P, 4)
        g = jnp.where((cols < n_pts)[None], g, 0.)                           # exact Dirichlet zeros
        return (g[..., 0] - g[..., 1]) / (2. * h), (g[..., 2] - g[..., 3]) / (2. * h)

    g_ms = 0.
    if lam_sob > 0:                                                          # [SOB] constant normaliser, all data
        allp = jnp.arange(n_pts)
        for s0 in range(0, S, 256):
            si = jnp.arange(s0, min(s0 + 256, S))
            dx, dy = fd(U, si, allp)
            g_ms += float(jnp.sum(dx * dx + dy * dy))
        g_ms /= S * n_pts

    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(500, steps // 10 + 1), steps, lr * 1e-2)

    def wd_mask(pz):
        p, z = pz
        mask_p = {}
        for k_, v in p.items():
            if k_ in ("g", "h"):
                mask_p[k_] = [(True, False) for _ in v]
            else:
                mask_p[k_] = jax.tree_util.tree_map(lambda _: False, v)
        return (mask_p, jax.tree_util.tree_map(lambda _: False, z))

    opt = optax.adamw(sched, weight_decay=weight_decay, mask=wd_mask) if weight_decay > 0.0 else optax.adam(sched)
    state = opt.init((params, Z))

    def loss_at(pz, U_, C_, sob):
        p, z = pz
        G = sc.features(p, C_)
        H = sc.head(p, z)
        err = H @ G.T - U_
        rel = jnp.mean(err * err) / u_ms
        C = (G.T @ G) / (G.shape[0] * p["out_scale"] ** 2)
        orth = jnp.mean((C - jnp.eye(C.shape[0], dtype=F64)) ** 2)
        total = rel + lam_orth * orth
        grel = jnp.zeros((), F64)
        if sob is not None:                                                  # [SOB]
            Ue, sidx, idx = sob
            _, Gx, Gy = feat_grad(p, coords[idx])
            Hs = H[sidx]
            dx, dy = fd(Ue, sidx, idx)
            ex, ey = Hs @ Gx.T - dx, Hs @ Gy.T - dy
            grel = jnp.mean(ex * ex + ey * ey) / g_ms
            total = total + lam_sob * grel
        return total, (rel, grel)

    def _apply(pz, st, ema, U_, C_, sob):
        (val, (rel, grel)), grads = jax.value_and_grad(loss_at, has_aux=True)(pz, U_, C_, sob)
        grads[0]["out_scale"] = jnp.zeros_like(grads[0]["out_scale"])
        upd, st = opt.update(grads, st, pz)
        pz = optax.apply_updates(pz, upd)
        ema = jax.tree_util.tree_map(lambda e, q: ema_decay * e + (1.0 - ema_decay) * q, ema, pz)
        return pz, st, ema, rel, grel

    def sob_args(ks, Ue):                                                    # [SOB] separate key stream
        if lam_sob <= 0:
            return None
        k1, k2 = jax.random.split(ks)
        sidx = jax.random.choice(k1, S, shape=(min(sob_states, S),), replace=False)
        idx = jax.random.choice(k2, n_pts, shape=(min(p_sub, n_pts),), replace=False)
        return (Ue, sidx, idx)

    @jax.jit
    def step_sub(pz, st, ema, k_, ks, U_all, C_all):
        pts_idx = jax.random.choice(k_, n_pts, shape=(p_sub,), replace=False)
        return _apply(pz, st, ema, U_all[:, pts_idx], C_all[pts_idx], sob_args(ks, U_all)) + (pts_idx[:8],)

    @jax.jit
    def step_full(pz, st, ema, ks, U_all, C_all):
        return _apply(pz, st, ema, U_all, C_all, sob_args(ks, U_all)) + (jnp.zeros((8,), jnp.int32),)

    pz = (params, Z)
    ema = pz
    t0 = time.time()
    rel = jnp.inf
    done = 0
    use_sub = 0 < p_sub < n_pts
    skey = jax.random.PRNGKey(sob_seed)                                      # [SOB]
    hist = []
    r2a = []
    for i in range(steps):
        ks = jax.random.fold_in(skey, i)                                     # [SOB] (unused when lam_sob = 0)
        if use_sub and i < steps - full_last:
            key, k_ = jax.random.split(key)
            pz, state, ema, rel, grel, pidx = step_sub(pz, state, ema, k_, ks, U, coords)
        else:
            pz, state, ema, rel, grel, pidx = step_full(pz, state, ema, ks, U, coords)
        done = i + 1
        if record_idx is not None and i < record_idx[0]:
            record_idx[1].append(np.asarray(pidx))
        if done % log_every == 0 or i == 0:
            hist.append(dict(step=done, rel_mse=float(rel), grad_rel_mse=float(grel), seconds=time.time() - t0))
            if not (np.isfinite(float(rel)) and np.isfinite(float(grel))):
                raise SystemExit(f'non-finite loss at step {done}')
            if check_hist and done in check_hist:                           # gate R2a (base only)
                want = check_hist[done]
                ok = f'{float(rel):.3e}' == want
                r2a.append(dict(step=done, got=f'{float(rel):.3e}', want=want, match=ok))
                log(f'   R2a step {done}: {float(rel):.3e} vs r3a {want} match={ok}')
                if done == 1 and not ok:                                     # step 1 is deterministic: code/data error
                    raise SystemExit(f'R2a failed at step 1: {float(rel):.3e} != {want}')
            log(f"   train2[{tag}] step {done:6d}/{steps}  rel-MSE {float(rel):.3e}  grad-rel-MSE {float(grel):.3e}"
                f"  [{time.time()-t0:.0f}s]")

    def recon_of(pz_):
        p, z = pz_
        G = sc.features(p, coords)
        H = sc.head(p, z)
        per = []
        for s in range(0, S, recon_chunk):
            e = min(s + recon_chunk, S)
            Uh = H[s:e] @ G.T
            Us = U[s:e]
            per.append(jnp.linalg.norm(Uh - Us, axis=1) / jnp.linalg.norm(Us, axis=1))
        per = jnp.concatenate(per)
        return float(jnp.mean(per)), float(jnp.max(per))

    raw_mean, raw_max = recon_of(pz)
    ema_mean, ema_max = recon_of(ema)
    use_ema = ema_mean < raw_mean
    params, Z = ema if use_ema else pz
    info = dict(final_rel_mse=float(rel), final_grad_rel_mse=float(grel), steps=steps, steps_done=done, lr=lr,
                lam_orth=lam_orth, weight_decay=weight_decay, p_sub=int(p_sub), ema_decay=ema_decay,
                full_last=full_last, used_ema=bool(use_ema), recon_raw_mean=raw_mean, recon_ema_mean=ema_mean,
                seconds=time.time() - t0, recon_rel_l2_mean=ema_mean if use_ema else raw_mean,
                recon_rel_l2_max=ema_max if use_ema else raw_max, n_snapshots=int(S), n_points=int(n_pts),
                lam_sob=lam_sob, sob_states=sob_states, grad_norm_ms=g_ms, history=hist,
                B_init=b_init, B_final=b_stats(params['B']), R2a=r2a)
    log(f"   train2[{tag}] done: recon mean {info['recon_rel_l2_mean']:.3e} max {info['recon_rel_l2_max']:.3e} "
        f"(used {'ema' if use_ema else 'raw'}) [{info['seconds']:.0f}s]")
    return params, np.asarray(Z), info


# ---------------------------------------------------------------- post: LS codes + rotation ----

def ls_rotation(params, n, U):
    """Least-squares coefficients of the training states U (S, (n-2)^2) on the bank at the training-mesh interior, and
    make_rotation's construction with h(z_i) -> c_i: A = rows (R_G c_i)/||R_G c_i||, SVD, T = R_G^-1 V_s, L = V_s^T R_G.
    Returns (npz dict, info). Also the gradient-free training LS floor per state."""
    coords = grid_coords(n)[interior_indices(n)]
    G = np.asarray(sc.SeparableDecoder(params, 16, 512).feat_at(coords, chunk=16384))   # (n_i2, R)
    assert np.isfinite(G).all(), 'bank not finite on the training mesh'
    Q, Rg = np.linalg.qr(G, mode='reduced')                                   # host f64 (deterministic)
    R = G.shape[1]
    A0, floor = [], []
    for s in range(0, len(U), 1024):
        Us = np.asarray(U[s:s + 1024])
        Y = Us @ Q                                                             # rows = (R_G c_i)^T = coords in Q
        A0.append(Y)
        floor.append(np.linalg.norm(Us - Y @ Q.T, axis=1) / np.linalg.norm(Us, axis=1))
    A0 = np.concatenate(A0)
    floor = np.concatenate(floor)
    C = np.linalg.solve(Rg, A0.T).T                                            # (S, R) LS coefficients c_i
    nrm0 = np.linalg.norm(A0, axis=1)
    if not (np.all(np.isfinite(A0)) and np.all(nrm0 > 0)):
        raise SystemExit('zero or non-finite training-coefficient row in the rotation')
    Am = A0 / nrm0[:, None]
    _, s, Vt = np.linalg.svd(Am, full_matrices=False)
    Vs = Vt.T
    T = np.linalg.solve(Rg, Vs)
    Lm = Vs.T @ Rg
    Gt = np.asarray(G) @ T
    sv_rg = np.linalg.svd(Rg, compute_uv=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    info = dict(construction='make_rotation (burgers-bank-knob) with h(z_i) replaced by the least-squares coefficients '
                             'of the training states on the training-mesh interior; per-row normalised',
                train_mesh_nodes=n, training_states=int(len(U)), R_G_condition_number=float(sv_rg[0] / sv_rg[-1]),
                L_times_T_identity_deviation=float(np.linalg.norm(Lm @ T - np.eye(R))),
                rotated_bank_orthonormality_deviation_at_train_mesh=float(np.linalg.norm(Gt.T @ Gt - np.eye(R))),
                cumulative_training_energy={str(r): float(energy[r - 1]) for r in (32, 64, 128, 256, 384, 512)},
                train_ls_floor=dict(mean=float(floor.mean()), median=float(np.median(floor)), max=float(floor.max())),
                trust_radius={str(r): float(.01 * np.max(np.linalg.norm(C @ Lm[:r].T - (C @ Lm[:r].T).mean(0), axis=1)))
                              for r in (128, 384)},
                T_sha256=sha_arr(T), L_sha256=sha_arr(Lm))
    return dict(T=T, L=Lm, singular_values=s, R_G=Rg, C=C), info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--sigma', type=float, required=True)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--nodes', type=int, default=256)
    ap.add_argument('--seed', type=int, default=RECIPE['seed'])
    ap.add_argument('--steps', type=int, default=RECIPE['steps'])
    ap.add_argument('--full-last', type=int, default=RECIPE['full_last'])
    ap.add_argument('--n-traj', type=int, default=576)
    ap.add_argument('--sob-states', type=int, default=2048)
    ap.add_argument('--frozen', default=None, help='also write the lane rotation of this frozen checkpoint')
    ap.add_argument('--out', required=True)
    ap.add_argument('--log-every', type=int, default=5000)
    ap.add_argument('--allow-cpu', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.allow_cpu:
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    dev = jax.devices()[0]
    log(f'jax_backend={jax.default_backend()} device={getattr(dev, "device_kind", dev)} arm={a.arm} sigma={a.sigma} '
        f'lam={a.lam} nodes={a.nodes} seed={a.seed} steps={a.steps}')
    rep = dict(arm=a.arm, sigma=a.sigma, lam_sob=a.lam, nodes=a.nodes, seed=a.seed, steps=a.steps, recipe=RECIPE,
               commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=getattr(dev, 'device_kind', str(dev)), jax_version=jax.__version__,
               precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
               generator_sha256=hashlib.sha256((HERE / 'deps/burgers2d_film.py').read_bytes()).hexdigest(),
               sep_common_sha256=hashlib.sha256((HERE / 'deps/sep_common.py').read_bytes()).hexdigest(),
               complete=False)
    save = lambda: (out / 'train.json').write_text(json.dumps(rep, indent=1, default=float))
    T = bf.NUM_STEPS + 1
    pick, _ = state_pick(a.n_traj, T, RECIPE["max_snaps"], RECIPE["t_early"], 0)   # data pick is fixed (seed 0) for every arm
    t0 = time.time()
    S_tr, worst, fp = build_data(a.nodes, a.n_traj, pick)
    rep['data'] = dict(n_traj=a.n_traj, T=T, n_states=int(pick.size), n_points=int(S_tr.shape[1]),
                       max_fom_rel_residual=worst, fingerprint=fp, pick_sha256=sha_arr(pick), data_sha256=sha_arr(S_tr),
                       seconds=time.time() - t0)
    if a.nodes == 256 and a.n_traj == 576:                 # gate R0: the r3a job's own fingerprint (push_r3a JSON)
        ok = abs(fp['sum'] / 200814620.48749176 - 1) < 1e-9 and abs(fp['sumsq'] / 102201588.76752055 - 1) < 1e-9
        rep['data']['R0_fingerprint_matches_r3a'] = bool(ok)
        if not ok:
            raise SystemExit(f'R0 failed: training data differ from the r3a job {fp}')
    log(f'  data: {S_tr.shape} worst residual {worst:.2e} fingerprint {fp} sha {rep["data"]["data_sha256"][:12]}')
    rep['data']['fd_target_uncertainty'] = fd_bias(S_tr, a.nodes)
    log(f"  FD-target uncertainty (D2 vs D4): {rep['data']['fd_target_uncertainty']}")
    save()
    n = a.nodes
    coords = grid_coords(n)[interior_indices(n)]
    arch = dict(RECIPE['arch'], ff_scale=a.sigma)
    # gate R2a: the original recipe (sigma 4, lam 0, seed 0, 256 nodes, 300k steps) must reproduce the r3a log
    orig = (a.sigma == 4.0 and a.lam == 0.0 and a.seed == 0 and a.nodes == 256 and a.steps == 300000 and a.n_traj == 576)
    check = {1: '2.402e+00', 5000: '2.125e-03'} if orig else None
    rep['R2a_checked'] = bool(orig)
    params, Z, tinfo = train(jax.random.PRNGKey(a.seed), coords, S_tr, neighbours(n), 1. / (n - 1), RECIPE['K'],
                             RECIPE['R'], a.steps, RECIPE['lr'], RECIPE['lam_orth'], RECIPE['wd'], RECIPE['p_sub'],
                             RECIPE['ema_decay'], min(a.full_last, a.steps), lam_sob=a.lam, sob_states=a.sob_states,
                             log_every=a.log_every, tag=a.arm, check_hist=check, **arch)
    rep['train'] = tinfo
    cfg = dict(pde='burgers2d', N=n, k=RECIPE['K'], r=RECIPE['R'], arch_overrides=arch, lam_sob=a.lam, arm=a.arm,
               seed=a.seed, steps=a.steps, lane='jcp-smooth-bank')
    sc.save_pkl(str(out / 'bank.pkl'), params, Z, cfg)
    rep['bank_sha256'] = hashlib.sha256((out / 'bank.pkl').read_bytes()).hexdigest()
    save()
    rot, rinfo = ls_rotation(jax.tree_util.tree_map(jnp.asarray, params), n, S_tr)
    np.savez(out / 'rotation.npz', **rot)
    rinfo['file_sha256'] = hashlib.sha256((out / 'rotation.npz').read_bytes()).hexdigest()
    rep['rotation'] = rinfo
    log(f'  rotation: {json.dumps({k: rinfo[k] for k in ("R_G_condition_number", "train_ls_floor")})}')
    if a.frozen:
        fp, _, _ = sc.load_pkl(a.frozen)
        frot, finfo = ls_rotation(fp, n, S_tr)
        np.savez(out / 'rotation_frozen.npz', **frot)
        finfo['file_sha256'] = hashlib.sha256((out / 'rotation_frozen.npz').read_bytes()).hexdigest()
        finfo['checkpoint_sha256'] = hashlib.sha256(Path(a.frozen).read_bytes()).hexdigest()
        rep['rotation_frozen'] = finfo
        log(f'  frozen rotation: {json.dumps({k: finfo[k] for k in ("R_G_condition_number", "train_ls_floor")})}')
    rep['complete'] = True
    rep['seconds_total'] = time.time() - t0
    save()
    log('DONE')


if __name__ == '__main__':
    main()
