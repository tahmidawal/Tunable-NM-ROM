"""Shared machinery for the GENERALIZABLE CASCADE NM-ROM experiment.

Design (2026-08-16, agreed with the owner):
  stage 0 : ENCODER E(inp) -> z  (inp = the PDE input the ROM legitimately
            knows: Poisson source f, Burgers initial condition u0, sampled on a
            fixed 16x16 lattice) + FiLM coord decoder D0(x; z), trained jointly
            in f64.  E is then FROZEN -> the same map at train and query time
            (this is what makes the cascade generalize to UNSEEN instances).
  stage k : frequency-scaled f64 RESIDUAL decoders D_k(x; z) with NO
            bottleneck, conditioned on the frozen z, fitting
            u - sum_{j<k} eps_j D_j (fixed target per stage) with the
            multistage-precision machinery (radial-mean f_d probe, half-cycle
            n_freq schedule, eps_k = residual RMS, inverse-energy weights).
  gate    : before every stage, June's POD-compressibility probe of the
            residual (eff. rank, POD-r reconstruction error) + the whitened
            nearest-neighbour-in-z correlation; stacking stops when the
            residual is incompressible / uncorrelated (thresholds are env).
  levers  : (a) K_EXTRA stage-specific extra conditioning c_k = E_k(inp)
            (small fresh encoder per stage, trained with that stage);
            (b) LAT_SMOOTH: encoder-Lipschitz proxy penalty during stage 0
            (pairs in the batch: ||z_i - z_j||^2 / ||inp_i - inp_j||^2).
  ROM     : Levenberg-Marquardt latent solve on the DISCRETE residual, init
            z = E(inp) (no held-out field ever touches the ROM path).

Everything numeric is f64.  We import the validated building blocks from the
multistage-precision worktree (ms_parametric: FiLM nets, coord features,
freq probe/schedule, metrics, weights, fit_stage; ms_autodecoder: lm_solve,
infer_latents_lm) instead of re-implementing them.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MSP_DIR = os.path.join(_WT, "2026-08-14-multistage-precision", "experiments",
                       "multistage-precision")
BURGERS_DIR = os.path.join(_WT, "2026-08-14-burgers2d-coord-rom", "experiments",
                           "burgers2d-coord-rom")
for d in (MSP_DIR, BURGERS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import ms_parametric as mp                       # noqa: E402
from ms_autodecoder import lm_solve, infer_latents_lm  # noqa: E402,F401

F64 = jnp.float64
LATTICE = int(os.environ.get("LATTICE", "16"))          # encoder input lattice
ENC_HIDDEN = int(os.environ.get("ENC_HIDDEN", "256"))
ENC_LAYERS = int(os.environ.get("ENC_LAYERS", "3"))
K_LAT = int(os.environ.get("K_LAT", "8"))
K_EXTRA = int(os.environ.get("K_EXTRA", "0"))           # lever (a)
LAT_SMOOTH = float(os.environ.get("LAT_SMOOTH", "0.0"))  # lever (b)
LAT_REG = float(os.environ.get("LAT_REG", "1e-4"))
T_SMOOTH = float(os.environ.get("T_SMOOTH", "1e-3"))    # burgers latent traj
ENC_LR = float(os.environ.get("ENC_LR", "1e-3"))
# stopping gate thresholds (report-only unless GATE=1)
GATE = int(os.environ.get("GATE", "0"))
GATE_EFFRANK = float(os.environ.get("GATE_EFFRANK", "64"))
GATE_NNCORR = float(os.environ.get("GATE_NNCORR", "0.2"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
SEED = mp.SEED

COMMON_CONFIG = dict(lattice=LATTICE, enc_hidden=ENC_HIDDEN, enc_layers=ENC_LAYERS,
                     k_lat=K_LAT, k_extra=K_EXTRA, lat_smooth=LAT_SMOOTH,
                     lat_reg=LAT_REG, t_smooth=T_SMOOTH, enc_lr=ENC_LR, gate=GATE,
                     gate_effrank=GATE_EFFRANK, gate_nncorr=GATE_NNCORR,
                     gn_iters=GN_ITERS, seed=SEED, x64=True)


# ------------------------------ lattice input ------------------------------

def lattice_points(m=LATTICE):
    """Fixed interior lattice (m x m) in (0,1)^2 — resolution independent."""
    t = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(t, t, indexing="ij")
    return X.reshape(-1), Y.reshape(-1)


def gaussian_on_lattice(cx, cy, w, a, m=LATTICE):
    """Evaluate a*exp(-((x-cx)^2+(y-cy)^2)/(2w^2)) on the lattice -> (n, m*m).
    Poisson: the SOURCE; Burgers: the initial condition (interior lattice, so
    the wall mask is 1)."""
    lx, ly = lattice_points(m)
    cx, cy, w, a = (np.asarray(v)[:, None] for v in (cx, cy, w, a))
    return a * np.exp(-((lx[None] - cx) ** 2 + (ly[None] - cy) ** 2) / (2 * w ** 2))


def standardize_fit(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-8
    return mu, sd


# ------------------------------ encoder ------------------------------

def init_encoder(key, d_in, k_out, hidden=ENC_HIDDEN, layers=ENC_LAYERS):
    keys = jax.random.split(key, layers + 1)
    ps = [mp.init_dense(keys[0], d_in, hidden)]
    for i in range(1, layers):
        ps.append(mp.init_dense(keys[i], hidden, hidden))
    out = mp.init_dense(keys[layers], hidden, k_out)
    out["W"] = out["W"] * 0.1
    return {"layers": ps, "out": out}


def enc_apply(params, x):
    h = x
    for lyr in params["layers"]:
        h = jax.nn.swish(h @ lyr["W"] + lyr["b"])
    return h @ params["out"]["W"] + params["out"]["b"]


# ------------------------------ cascade apply ------------------------------

def stage_z(stage, z, c):
    """Conditioning for one stage: z, or concat(z, c_k) when the stage has
    extra encoder features (lever a)."""
    if stage.get("k_extra", 0) > 0:
        return jnp.concatenate([z, c])
    return z


def cascade_apply(stages, z, cs, xy):
    """u(xy) = sum_k eps_k D_k(xy; z [, c_k]).  cs = list aligned with stages
    (None for stages without extras)."""
    tot = 0.0
    for s, c in zip(stages, cs):
        tot = tot + s["eps"] * mp.film_apply(s["params"], stage_z(s, z, c), xy,
                                             s["n_freq"], 0)
    return tot


def stage_extras(stages, inp_row):
    """Per-stage extra conditioning for ONE input row (lattice features)."""
    return [enc_apply(s["enc_extra"], inp_row) if s.get("k_extra", 0) > 0 else None
            for s in stages]


def predict_rows(stages, Z, C_list, coords, chunk=64):
    """Z (n, K); C_list: list per stage of (n, k_extra) or None."""
    n = Z.shape[0]
    outs = []
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(s0 + chunk, n))
        def one(zi, *ci):
            cs = []
            j = 0
            for s in stages:
                if s.get("k_extra", 0) > 0:
                    cs.append(ci[j]); j += 1
                else:
                    cs.append(None)
            return cascade_apply(stages, zi, cs, coords)
        cargs = [C[sl] for C, s in zip(C_list, stages) if s.get("k_extra", 0) > 0]
        outs.append(jax.vmap(one)(Z[sl], *cargs))
    return jnp.concatenate(outs, axis=0)


def stages_to_np(stages):
    out = []
    for s in stages:
        d = {"params": jax.tree_util.tree_map(np.asarray, s["params"]),
             "n_freq": int(s["n_freq"]), "eps": float(s["eps"]),
             "k_extra": int(s.get("k_extra", 0))}
        if s.get("k_extra", 0) > 0:
            d["enc_extra"] = jax.tree_util.tree_map(np.asarray, s["enc_extra"])
        out.append(d)
    return out


def stages_from_np(raw):
    out = []
    for s in raw:
        d = {"params": jax.tree_util.tree_map(jnp.asarray, s["params"]),
             "n_freq": s["n_freq"], "eps": s["eps"], "k_extra": s.get("k_extra", 0)}
        if d["k_extra"] > 0:
            d["enc_extra"] = jax.tree_util.tree_map(jnp.asarray, s["enc_extra"])
        out.append(d)
    return out


# ------------------------------ compressibility probe ------------------------------

def compressibility(E, ranks=(4, 8, 16, 32, 64)):
    """June's POD probe on the (centred) residual matrix E (n_rows, n_pts):
    effective rank (participation ratio), energy in top-r, POD-r relative
    reconstruction error (train-fitted, i.e. optimistic)."""
    E = np.asarray(E, dtype=np.float64)
    Ec = E - E.mean(0, keepdims=True)
    s = np.linalg.svd(Ec, full_matrices=False, compute_uv=False)
    s2 = s ** 2
    tot = s2.sum() + 1e-300
    eff = float(tot ** 2 / (np.sum(s2 ** 2) + 1e-300))
    out = {"eff_rank": eff, "n_rows": int(E.shape[0])}
    for r in ranks:
        r = min(r, len(s2))
        out[f"energy_top{r}"] = float(s2[:r].sum() / tot)
        out[f"pod{r}_rel_err"] = float(np.sqrt(max(0.0, 1.0 - s2[:r].sum() / tot)))
    return out


def whiten(Z):
    Zc = Z - Z.mean(0)
    C = np.cov(Zc.T) + 1e-12 * np.eye(Z.shape[1])
    L = np.linalg.cholesky(np.linalg.inv(C))
    return Zc @ L


def nn_corr(E, Z, k=5, max_rows=2048, seed=0):
    """Mean Pearson correlation of residual rows with their nearest neighbours
    in WHITENED latent space (nn1, mean over nn-k).  Fields are smooth in z
    (corr ~ 1); a residual with corr ~ 0 has no z-structure left to learn."""
    E = np.asarray(E); Z = np.asarray(Z)
    if E.shape[0] > max_rows:
        idx = np.random.default_rng(seed).choice(E.shape[0], max_rows, replace=False)
        E, Z = E[idx], Z[idx]
    Zw = whiten(Z)
    D = ((Zw[:, None, :] - Zw[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    Ec = E - E.mean(1, keepdims=True)
    Ec = Ec / (np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-300)
    c1 = np.array([Ec[i] @ Ec[nn[i, 0]] for i in range(len(E))])
    ck = np.array([np.mean([Ec[i] @ Ec[j] for j in nn[i]]) for i in range(len(E))])
    return float(c1.mean()), float(ck.mean())


def probe(E, Z):
    p = compressibility(E)
    c1, c5 = nn_corr(E, Z)
    p.update({"nn1_corr": c1, "nn5_corr": c5})
    p["stop_suggested"] = bool(p["eff_rank"] > GATE_EFFRANK or c5 < GATE_NNCORR)
    return p


# ------------------------------ stage-0: encoder + decoder ------------------------------

def fit_stage0(key, np_rng, coords, target, weights, n_freq, inp, row2inp,
               enc_rows_mask, k_lat, *, steps, prev_row=None, tag="A"):
    """Joint (encoder, decoder[, free latents]) fit of stage 0.

    target (n_rows, n_pts) ~RMS 1; weights (n_rows,) mean 1; inp (n_inp, d_in)
    standardized lattice features; row2inp (n_rows,) int; enc_rows_mask
    (n_rows,) bool — rows whose latent is E(inp[row2inp]) (all rows for
    Poisson; time-0 rows for Burgers).  Rows with mask False get FREE latents
    (lazy per-row Adam, cf. multistage-precision F5 fix) with a temporal
    smoothness penalty to prev_row (T_SMOOTH) — prev_row (n_rows,) int, -1 if
    none.  LAT_SMOOTH (lever b) penalizes ||z_i-z_j||^2/||inp_i-inp_j||^2 for
    encoded pairs in the batch.  Returns (dec_params, enc_params, Z_all,
    final_loss, secs) with Z_all filled with E(inp) on encoded rows."""
    n_rows, n_pts = target.shape
    d_in = inp.shape[1]
    kd, ke, kz = jax.random.split(key, 3)
    dec = mp.init_film_net(kd, n_freq, k_lat, 0)
    enc = init_encoder(ke, d_in, k_lat)
    Z = 0.1 * jax.random.normal(kz, (n_rows, k_lat), dtype=F64)
    params = {"dec": dec, "enc": enc}
    opt = optax.adamw(mp.make_lr_schedule(steps), weight_decay=1e-6)
    state = opt.init(params)
    enc_mask = jnp.asarray(enc_rows_mask.astype(np.float64))
    row2inp = jnp.asarray(row2inp)
    prev_row = jnp.asarray(prev_row if prev_row is not None else -np.ones(n_rows, int))
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    lat_lr = 5e-3
    lat_sched = optax.warmup_cosine_decay_schedule(0.0, lat_lr, max(1, steps // 20),
                                                   steps, end_value=1e-9)
    any_free = bool((~enc_rows_mask).any())

    def z_of_rows(ps, Zfree, bi):
        z_enc = enc_apply(ps["enc"], inp[row2inp[bi]])
        m = enc_mask[bi][:, None]
        return m * z_enc + (1 - m) * Zfree

    def loss_fn(ps, Zb, bi, t_b, w_b, pidx, Zprev):
        z_b = z_of_rows(ps, Zb, bi)
        pred = jax.vmap(lambda zi: mp.film_apply(ps["dec"], zi, coords[pidx],
                                                 n_freq, 0))(z_b)
        se = jnp.mean((pred - t_b[:, pidx]) ** 2, axis=1)
        loss = jnp.mean(w_b * se)
        # lever (b): encoder Lipschitz proxy on encoded pairs
        if LAT_SMOOTH > 0:
            zr = jnp.roll(z_b, 1, axis=0); ir = jnp.roll(inp[row2inp[bi]], 1, axis=0)
            mr = (enc_mask[bi] * jnp.roll(enc_mask[bi], 1))
            dz = jnp.sum((z_b - zr) ** 2, axis=1)
            di = jnp.sum((inp[row2inp[bi]] - ir) ** 2, axis=1) + 1e-6
            loss = loss + LAT_SMOOTH * jnp.sum(mr * dz / di) / (jnp.sum(mr) + 1e-12)
        # free latents: L2 reg + temporal smoothness to the previous row
        if any_free:
            free = (1 - enc_mask[bi])
            loss = loss + LAT_REG * jnp.mean(free[:, None] * z_b ** 2)
            has_prev = (prev_row[bi] >= 0).astype(F64)
            loss = loss + T_SMOOTH * jnp.sum(
                has_prev * free * jnp.sum((z_b - Zprev) ** 2, axis=1)) / (
                jnp.sum(has_prev * free) + 1e-12)
        return loss

    @jax.jit
    def step(ps, st, Z, m, v, cnt, gstep, bi, t_b, w_b, pidx):
        Zb = Z[bi]
        pr = prev_row[bi]
        pr_c = jnp.maximum(pr, 0)
        # previous-row latent (encoded if that row is an encoded row), no grad
        z_prev_enc = enc_apply(ps["enc"], inp[row2inp[pr_c]])
        Zprev = jax.lax.stop_gradient(
            enc_mask[pr_c][:, None] * z_prev_enc + (1 - enc_mask[pr_c][:, None]) * Z[pr_c])
        val, (gp, gz) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
            ps, Zb, bi, t_b, w_b, pidx, Zprev)
        up, st = opt.update(gp, st, ps)
        ps = optax.apply_updates(ps, up)
        if any_free:
            m_b = b1 * m[bi] + (1 - b1) * gz
            v_b = b2 * v[bi] + (1 - b2) * gz ** 2
            c_b = cnt[bi] + 1.0
            mhat = m_b / (1 - b1 ** c_b[:, None]); vhat = v_b / (1 - b2 ** c_b[:, None])
            z_new = Zb - lat_sched(gstep) * mhat / (jnp.sqrt(vhat) + eps_a)
            free = (1 - enc_mask[bi])[:, None]
            z_new = free * z_new + (1 - free) * Zb
            Z = Z.at[bi].set(z_new); m = m.at[bi].set(m_b); v = v.at[bi].set(v_b)
            cnt = cnt.at[bi].set(c_b)
        return ps, st, Z, m, v, cnt, val

    m = jnp.zeros_like(Z); v = jnp.zeros_like(Z)
    cnt = jnp.zeros((n_rows,), dtype=F64)
    all_pts = jnp.arange(n_pts)
    B = min(mp.BATCH, n_rows)
    t0 = time.time()
    for it in range(steps):
        bi = np.arange(n_rows) if B >= n_rows else np_rng.choice(n_rows, size=B, replace=False)
        pidx = (all_pts if mp.P_SUB <= 0 or mp.P_SUB >= n_pts else
                jnp.asarray(np_rng.choice(n_pts, size=mp.P_SUB, replace=False)))
        params, state, Z, m, v, cnt, val = step(
            params, state, Z, m, v, cnt, it, jnp.asarray(bi), target[bi], weights[bi], pidx)
        if it % 5000 == 0:
            print(f"  [{tag}] step {it:6d}  loss {float(val):.3e}  [{time.time()-t0:.0f}s]",
                  flush=True)
    # fill encoded rows
    z_enc_all = jax.vmap(lambda r: enc_apply(params["enc"], inp[r]))(jnp.arange(n_rows))
    Z = jnp.where(enc_mask[:, None] > 0, z_enc_all, Z)
    print(f"  [{tag}] {steps} steps in {time.time()-t0:.0f}s (final batch loss "
          f"{float(val):.3e})", flush=True)
    return params["dec"], params["enc"], Z, float(val), time.time() - t0


# ------------------------------ stage k: residual decoder ------------------------------

def fit_residual_stage(key, np_rng, coords, target, weights, n_freq, Z_frozen, inp,
                       row2inp, k_lat, *, k_extra=0, steps, tag=""):
    """Fixed-target residual stage.  If k_extra == 0 this is exactly
    mp.fit_stage(learn_latents=False).  Else a small fresh encoder E_k(inp)
    provides k_extra extra conditioning dims trained jointly (lever a)."""
    if k_extra <= 0:
        params, _, loss, secs = mp.fit_stage(
            key, np_rng, coords, target, weights, n_freq, Z_frozen, k_lat=k_lat,
            z_ff=0, learn_latents=False, steps=steps, tag=tag)
        return params, None, loss, secs
    n_rows, n_pts = target.shape
    kd, ke = jax.random.split(key)
    params = {"dec": mp.init_film_net(kd, n_freq, k_lat + k_extra, 0),
              "enc": init_encoder(ke, inp.shape[1], k_extra, hidden=64, layers=2)}
    opt = optax.adamw(mp.make_lr_schedule(steps), weight_decay=1e-6)
    state = opt.init(params)
    row2inp = jnp.asarray(row2inp)

    def loss_fn(ps, bi, t_b, w_b, pidx):
        c = enc_apply(ps["enc"], inp[row2inp[bi]])
        zc = jnp.concatenate([Z_frozen[bi], c], axis=1)
        pred = jax.vmap(lambda zi: mp.film_apply(ps["dec"], zi, coords[pidx], n_freq, 0))(zc)
        se = jnp.mean((pred - t_b[:, pidx]) ** 2, axis=1)
        return jnp.mean(w_b * se)

    @jax.jit
    def step(ps, st, bi, t_b, w_b, pidx):
        val, g = jax.value_and_grad(loss_fn)(ps, bi, t_b, w_b, pidx)
        up, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, up), st, val

    all_pts = jnp.arange(n_pts)
    B = min(mp.BATCH, n_rows)
    t0 = time.time()
    for it in range(steps):
        bi = np.arange(n_rows) if B >= n_rows else np_rng.choice(n_rows, size=B, replace=False)
        pidx = (all_pts if mp.P_SUB <= 0 or mp.P_SUB >= n_pts else
                jnp.asarray(np_rng.choice(n_pts, size=mp.P_SUB, replace=False)))
        params, state, val = step(params, state, jnp.asarray(bi), target[bi], weights[bi], pidx)
        if it % 5000 == 0:
            print(f"  [{tag}] step {it:6d}  loss {float(val):.3e}  [{time.time()-t0:.0f}s]",
                  flush=True)
    print(f"  [{tag}] {steps} steps in {time.time()-t0:.0f}s (final batch loss "
          f"{float(val):.3e})", flush=True)
    return params["dec"], params["enc"], float(val), time.time() - t0


# ------------------------------ misc ------------------------------

def rel_metrics(pred, U):
    return mp.rel_metrics(pred, U)


def per_rel(pred, U):
    return np.asarray(jnp.linalg.norm(pred - U, axis=1) / jnp.linalg.norm(U, axis=1))
