"""Bounded training round for the q16-conditioned transported Poisson tail.

Selection is based on validation FOM work (true-residual counting-CG iterations), with
energy norm as the deterministic tie breaker.  Relative L2 is persisted only as a diagnostic;
it neither trains nor selects the checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(HERE, "deps"),
             os.path.join(os.path.dirname(HERE), "rom-warmstart-fom"),
             os.path.abspath(os.path.join(HERE, "..", "..", "rom-warmstart-fom")),
             os.path.abspath(os.path.join(
                 HERE, "..", "..", "..", "2026-08-14-multistage-precision",
                 "experiments", "multistage-precision"))):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import ms_parametric as mp  # noqa: E402
import transport_arch as ta  # noqa: E402
import wsf_util as wu  # noqa: E402

OUTDIR = sys.argv[1]
N = int(os.environ.get("N", "64"))
N_TRAIN = int(os.environ.get("N_TRAIN", "512"))
N_VAL = int(os.environ.get("N_VAL", "64"))
Q = int(os.environ.get("Q_BASE", "16"))
K = int(os.environ.get("K_LAT", "6"))
RANK = int(os.environ.get("RANK", "8"))
WIDTH = int(os.environ.get("WIDTH", "24"))
DEPTH = int(os.environ.get("DEPTH", "2"))
STEPS = int(os.environ.get("STEPS", "12000"))
BATCH = int(os.environ.get("BATCH", "32"))
PEAK_LR = float(os.environ.get("PEAK_LR", "2e-3"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "1000"))
POST_CG_STEPS = int(os.environ.get("POST_CG_STEPS", "4"))
SEED = int(os.environ.get("SEED", "0"))
CG_TAU = float(os.environ.get("CG_TAU", "1e-6"))
RIDGES = [float(v) for v in os.environ.get(
    "INIT_RIDGES", "1e-10,1e-8,1e-6,1e-4,1e-2").split(",")]
F64 = jnp.float64


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sine_system(n):
    ni = n - 2
    idx = np.arange(1, ni + 1)
    S = np.sqrt(2.0 / (ni + 1)) * np.sin(np.pi * np.outer(idx, idx) / (ni + 1))
    dx = 1.0 / (n - 1)
    eig = 4.0 / dx**2 * np.sin(np.pi * idx / (2.0 * (ni + 1))) ** 2
    return jnp.asarray(S), jnp.asarray(eig[:, None] + eig[None, :])


def source_fields(n, seed, count):
    cx, cy, width, amp, p = mp.sample_params(seed=seed, m=count)
    F = np.stack([mp.source_interior(n, cx[i], cy[i], width[i], amp[i])
                  for i in range(count)])
    return jnp.asarray(F), np.asarray(p)


def parameter_latents(p):
    return np.column_stack([p, p[:, 0] * p[:, 2], p[:, 1] * p[:, 2]])


def fit_initializer_train_only(C, target):
    """Fixed train/calibration split chooses ridge before the 64 validation PDE cases."""
    rng = np.random.default_rng(73129)
    perm = rng.permutation(len(C))
    ncal = max(1, int(round(0.2 * len(C))))
    ical, ifit = perm[:ncal], perm[ncal:]

    def standardize(indices):
        mu = C[indices].mean(0)
        sd = C[indices].std(0) + 1e-12
        X = (C - mu) / sd
        return np.column_stack([np.ones(len(X)), X]), mu, sd

    X, _, _ = standardize(ifit)
    eye = np.eye(X.shape[1]); eye[0, 0] = 0.0
    trials = []
    best = None
    for ridge in RIDGES:
        W = np.linalg.solve(X[ifit].T @ X[ifit] + ridge * eye,
                            X[ifit].T @ target[ifit])
        mse = float(np.mean((X[ical] @ W - target[ical]) ** 2))
        trials.append(dict(ridge=ridge, calibration_latent_mse=mse))
        if best is None or (mse, ridge) < (best[0], best[1]):
            best = (mse, ridge)
    X, mu, sd = standardize(np.arange(len(C)))
    ridge = best[1]
    W = np.linalg.solve(X.T @ X + ridge * eye, X.T @ target)
    return dict(mean=mu, scale=sd, weights=W), dict(
        selection="fixed split of 512 training sources only",
        calibration_seed=73129,
        n_fit=int(len(ifit)), n_calibration=int(len(ical)),
        trials=trials, selected_ridge=ridge,
        refit_train_latent_mse=float(np.mean((X @ W - target) ** 2)),
    )


def apply_initializer(fit, C):
    X = (C - fit["mean"]) / fit["scale"]
    return np.column_stack([np.ones(len(X)), X]) @ fit["weights"]


def main():
    if jax.default_backend() != "gpu":
        raise SystemExit(f"training requires GPU, got {jax.default_backend()}")
    if K != 6:
        raise SystemExit("the bounded transported round is pre-registered at K=6")
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = ta.config(rank=RANK, width=WIDTH, depth=DEPTH, k_lat=K)
    S, lam = sine_system(N)
    Sq, lamq = S[:, :Q], lam[:Q, :Q]
    F, p = source_fields(N, SEED, N_TRAIN + N_VAL)
    coeff = jax.vmap(lambda f: Sq.T @ f @ Sq)(F)
    exact = jax.vmap(lambda f: S @ ((S.T @ f @ S) / lam) @ S.T)(F)
    base = jax.vmap(lambda c: Sq @ (c / lamq) @ Sq.T)(coeff)
    target_tail = exact - base
    # Orthonormal discrete sine coefficients grow like (N-1); this continuum-scaled
    # representation makes the train-N64 initializer deployable on finer meshes.
    C_np = (np.asarray(coeff) / (N - 1)).reshape(len(F), -1)
    init_fit, init_info = fit_initializer_train_only(
        C_np[:N_TRAIN], parameter_latents(p[:N_TRAIN])
    )
    Z = jnp.asarray(apply_initializer(init_fit, C_np))
    init_val_mse = float(np.mean(
        (np.asarray(Z[N_TRAIN:]) - parameter_latents(p[N_TRAIN:])) ** 2
    ))
    eps = float(jnp.sqrt(jnp.mean(jnp.square(target_tail[:N_TRAIN]))))
    x = jnp.linspace(0.0, 1.0, N, dtype=F64)
    op = lambda u: mp.neg_lap_interior(u, N)
    op_batch = jax.vmap(op)
    decode_batch = jax.vmap(ta.apply_grid, in_axes=(None, 0, None, None))

    def fixed_cg(rhs, steps):
        x0 = jnp.zeros_like(rhs)
        r0 = rhs

        def body(state, _):
            xx, rr, pp, rs = state
            Ap = op_batch(pp)
            alpha = rs / jnp.maximum(jnp.sum(pp * Ap, axis=(1, 2)), 1e-300)
            xx = xx + alpha[:, None, None] * pp
            rr = rr - alpha[:, None, None] * Ap
            rs2 = jnp.sum(rr * rr, axis=(1, 2))
            beta = rs2 / jnp.maximum(rs, 1e-300)
            pp = rr + beta[:, None, None] * pp
            return (xx, rr, pp, rs2), None

        rs0 = jnp.sum(r0 * r0, axis=(1, 2))
        return jax.lax.scan(body, (x0, r0, r0, rs0), None, length=steps)[0][0]

    def metrics(params, z, b, truth, f):
        pred = decode_batch(params, z, x, eps)[:, 1:-1, 1:-1]
        guess = b + pred
        err = guess - truth
        Ae = op_batch(err)
        denom_a = jnp.maximum(jnp.sum(truth * f, axis=(1, 2)), 1e-300)
        en = jnp.sum(err * Ae, axis=(1, 2)) / denom_a
        # Jacobi-preconditioned residual; the diagonal is constant for this operator.
        diag = 4.0 * (N - 1) ** 2
        jac = (jnp.sum(Ae * Ae / diag, axis=(1, 2))
               / jnp.maximum(jnp.sum(f * f / diag, axis=(1, 2)), 1e-300))
        correction = fixed_cg(-Ae, POST_CG_STEPS)
        post_err = err + correction
        Apost = op_batch(post_err)
        post = jnp.sum(post_err * Apost, axis=(1, 2)) / denom_a
        rel_l2 = (jnp.linalg.norm(err.reshape(len(err), -1), axis=1)
                  / jnp.maximum(jnp.linalg.norm(truth.reshape(len(truth), -1), axis=1),
                                1e-300))
        return en, jac, post, rel_l2, guess

    def loss_fn(params, z, b, truth, f):
        en, jac, post, _, _ = metrics(params, z, b, truth, f)
        return jnp.mean(0.35 * en + 0.05 * jac + 0.60 * post)

    warmup_steps = min(max(1, STEPS // 20), max(1, STEPS // 2))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=PEAK_LR / 20.0, peak_value=PEAK_LR,
        warmup_steps=warmup_steps, decay_steps=max(STEPS, warmup_steps + 1),
        end_value=PEAK_LR / 100.0,
    )
    opt = optax.adam(sched)
    key = jax.random.PRNGKey(SEED + 911)
    params = ta.init(key, cfg)
    state = opt.init(params)

    @jax.jit
    def step(params, state, idx):
        args = (Z[idx], base[idx], exact[idx], F[idx])
        val, grad = jax.value_and_grad(loss_fn)(params, *args)
        updates, state = opt.update(grad, state, params)
        return optax.apply_updates(params, updates), state, val

    cg = wu.make_cg(op, maxiter=20_000)
    zero = jnp.zeros((N - 2, N - 2), dtype=F64)
    report = dict(
        complete=False,
        config=dict(N=N, n_train=N_TRAIN, n_val=N_VAL, q_base=Q, K=K,
                    rank=RANK, width=WIDTH, depth=DEPTH, steps=STEPS, batch=BATCH,
                    peak_lr=PEAK_LR, post_cg_steps=POST_CG_STEPS, seed=SEED,
                    cg_tau=CG_TAU, dtype="f64",
                    matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "unset"),
                    architecture=cfg),
        provenance=dict(commit=os.environ.get("WSF_COMMIT", "unknown"),
                        job_id=os.environ.get("SLURM_JOB_ID", "local"),
                        backend=jax.default_backend(), device=str(jax.devices()[0]),
                        jax_version=jax.__version__,
                        train_source_sha256=sha256(__file__),
                        arch_source_sha256=sha256(ta.__file__)),
        initializer=dict(**init_info, validation_latent_mse=init_val_mse),
        tail_rms=eps,
        n_params=ta.parameter_count(params),
        evaluations=[],
    )
    rng = np.random.default_rng(SEED + 177)
    best_key = None
    best_params_np = None
    t0 = time.time()

    def evaluate(step_num, train_loss):
        zva, bva, uva, fva = (v[N_TRAIN:] for v in (Z, base, exact, F))
        en, jac, post, l2, guess = metrics(params, zva, bva, uva, fva)
        en, jac, post, l2, guess = map(np.asarray, (en, jac, post, l2, guess))
        iters, true_res = [], []
        for i in range(N_VAL):
            out = cg(fva[i], jnp.asarray(guess[i]), CG_TAU)
            iters.append(int(out[1])); true_res.append(float(out[2]))
        # Spectral-only q16 is evaluated with the exact same counting solver.
        base_iters = [int(cg(fva[i], bva[i], CG_TAU)[1]) for i in range(N_VAL)]
        row = dict(
            step=int(step_num), train_batch_loss=float(train_loss),
            val_energy_ratio_mean=float(np.mean(en)),
            val_energy_ratio_median=float(np.median(en)),
            val_jacobi_residual_ratio_mean=float(np.mean(jac)),
            val_post_fixed_cg_energy_ratio_mean=float(np.mean(post)),
            val_counting_cg_iters_mean=float(np.mean(iters)),
            val_counting_cg_iters_median=float(np.median(iters)),
            spectral_q16_counting_cg_iters_mean=float(np.mean(base_iters)),
            val_true_residual_max=float(np.max(true_res)),
            diagnostic_val_rel_l2_mean=float(np.mean(l2)),
            val_iters_per_case=iters,
            elapsed_seconds=time.time() - t0,
        )
        report["evaluations"].append(row)
        print("EVAL " + json.dumps(row), flush=True)
        return row

    for it in range(STEPS + 1):
        if it == 0:
            val = loss_fn(params, Z[:BATCH], base[:BATCH], exact[:BATCH], F[:BATCH])
        else:
            idx = jnp.asarray(rng.choice(N_TRAIN, size=BATCH, replace=False))
            params, state, val = step(params, state, idx)
        if it % EVAL_EVERY == 0 or it == STEPS:
            row = evaluate(it, val)
            key_sel = (row["val_counting_cg_iters_mean"], row["val_energy_ratio_mean"])
            if best_key is None or key_sel < best_key:
                best_key = key_sel
                best_params_np = jax.tree_util.tree_map(np.asarray, params)
                report["selected_step"] = it
                report["selection_key"] = list(key_sel)
            with open(os.path.join(OUTDIR, "transport_train.json"), "w") as f:
                json.dump(report, f, indent=1, allow_nan=False)

    checkpoint = dict(
        config=report["config"], params=best_params_np, eps=eps,
        initializer=jax.tree_util.tree_map(np.asarray, init_fit),
        z_train=np.asarray(Z[:N_TRAIN]),
        initializer_info=report["initializer"], selected_step=report["selected_step"],
        selection_key=report["selection_key"],
    )
    with open(os.path.join(OUTDIR, "transport_tail.pkl"), "wb") as f:
        pickle.dump(checkpoint, f)
    report["checkpoint_sha256"] = sha256(os.path.join(OUTDIR, "transport_tail.pkl"))
    report["complete"] = True
    with open(os.path.join(OUTDIR, "transport_train.json"), "w") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    print(f"DONE selected_step={report['selected_step']} key={best_key}", flush=True)


if __name__ == "__main__":
    main()
