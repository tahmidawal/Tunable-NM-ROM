"""ROUND 5 driver -- run h-refinement ARMS in coefficient space and report the
full accuracy ladder for each, from the extraction npz alone.

Every number here is a field-space relative L2 obtained through the exact
identity (*) documented in `sep_hfit.py`, so the rungs are directly comparable
with the r3/r4 cluster tables.  The bank is FROZEN throughout: an arm changes
only h (and the codes), never the span, so `span floor` is the same constant
for every arm and `oracle / span floor` is the headline the campaign turns on.

Oracle protocol, kept comparable with the cluster drivers: the latent LM is
started from the mean training code and from a small ENCODER trained on
TRAINING pairs only (the r3/r4 drivers use exactly these two inits).  A second
`oracle_nn` column additionally allows a nearest-training-state init chosen
using the test target -- that one is an upper bound on what better
initialisation could buy and is labelled as such, never quoted as the oracle.
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

import sep_hfit as hf

F64 = jnp.float64
log = hf.log

NPZ = os.environ["NPZ"]
CKPT = os.environ["CKPT"]
OUT = os.environ.get("OUT", "hfit.json")
ARMS = [a for a in os.environ.get("ARMS", "base").split(",") if a]
STEPS = int(os.environ.get("STEPS", "60000"))
BATCH = int(os.environ.get("BATCH", "4096"))
LR = float(os.environ.get("LR", "1e-3"))
ORACLE_ITERS = int(os.environ.get("ORACLE_ITERS", "150"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "12000"))
SEED0 = int(os.environ.get("SEED0", "0"))
TIME_CAP = float(os.environ.get("TIME_CAP", "0"))
EMIT = os.environ.get("EMIT", "")          # arm name -> write a new checkpoint
EMIT_PATH = os.environ.get("EMIT_PATH", "")
ORACLE_EVERY = int(os.environ.get("ORACLE_EVERY", "1"))
# the code-convergence diagnostic re-solves every training code exactly; a
# subsample is enough to see whether the codes had converged and keeps the
# per-arm evaluation short
CODEDIAG_N = int(os.environ.get("CODEDIAG_N", "2048"))
CODEDIAG_ITERS = int(os.environ.get("CODEDIAG_ITERS", "60"))

# ---------------------------------------------------------------------------
# arm table.  Each arm is a dict of overrides for `hf.fit` plus a few flags:
#   warm:   'ckpt' (start from the checkpoint's own h and codes) or None
#   k:      latent dimension (default = the checkpoint's K)
#   w:      'early' to up-weight the sharp early-time states
#   zinit:  'params' to initialise the codes from the standardised (mu, t)
# ---------------------------------------------------------------------------
ARM_SPECS = {
    # --- controls -----------------------------------------------------------
    "base":        dict(steps=0, warm="ckpt"),
    "refit_ctl":   dict(hidden=256, layers=2, warm="ckpt"),
    "refit_cold":  dict(hidden=256, layers=2),
    "glob_ctl":    dict(hidden=256, layers=2, warm="ckpt", norm="global"),
    # --- lever 1: h capacity and shape --------------------------------------
    "wide":        dict(hidden=1024, layers=3),
    "deep":        dict(hidden=512, layers=6),
    "huge":        dict(hidden=2048, layers=4),
    "glob":        dict(hidden=1024, layers=3, norm="global"),
    "wd":          dict(hidden=1024, layers=3, wd=1e-6),
    # --- lever 1b: h's FUNCTION CLASS (latent Fourier features) -------------
    "ffz64":       dict(hidden=1024, layers=3, h_ff=64, h_ff_scale=1.0),
    "ffz128":      dict(hidden=1024, layers=3, h_ff=128, h_ff_scale=2.0),
    "ffz256s4":    dict(hidden=1024, layers=3, h_ff=256, h_ff_scale=4.0),
    # --- lever 2: latent codes (round-5 base arm already says 1.000x) -------
    "zpolish":     dict(hidden=1024, layers=3, z_polish_every=10000),
    # --- lever 3: early-time weighting --------------------------------------
    "early":       dict(hidden=1024, layers=3, w="early"),
    # --- lever 4: K ---------------------------------------------------------
    "k8":          dict(hidden=1024, layers=3, k=8),
    "k24":         dict(hidden=1024, layers=3, k=24),
    "k32":         dict(hidden=1024, layers=3, k=32),
    "k32ffz":      dict(hidden=1024, layers=3, k=32, h_ff=128, h_ff_scale=2.0),
    # --- diagnostic: is the coefficient manifold a smooth function of the
    #     TRUE parameters at all?  codes initialised at standardised (mu, t).
    "paramcodes":  dict(hidden=1024, layers=3, k=6, zinit="params"),
}


def main():
    dev = jax.devices()[0]
    log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64}"
        f" HFIT arms={ARMS} steps={STEPS}")
    t_all = time.time()
    d = np.load(NPZ)
    with open(CKPT, "rb") as f:
        ck = pickle.load(f)
    p_ck = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    # codes come from the npz, which records exactly the codes aligned with
    # C_tr (the extraction job asserts that alignment against the pick)
    Z_ck = jnp.asarray(d["Z_ck"], dtype=F64)
    K0 = int(Z_ck.shape[1])
    Gram = jnp.asarray(d["Gram"], dtype=F64)
    R = Gram.shape[0]
    eps = 1e-12 * jnp.trace(Gram) / R
    L = jnp.linalg.cholesky(Gram + eps * jnp.eye(R, dtype=F64))
    Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(R, dtype=F64),
                                             lower=True)
    # whitened targets  a = L^T c*
    A_tr = jnp.asarray(d["C_tr"], dtype=F64) @ L
    A_te = jnp.asarray(d["C_te"], dtype=F64) @ L
    un2_tr = jnp.asarray(d["un2_tr"]); fl2_tr = jnp.asarray(d["fl2_tr"])
    un2_te = jnp.asarray(d["un2_te"]); fl2_te = jnp.asarray(d["fl2_te"])
    t_tr = np.asarray(d["t_tr"]); t_te = np.asarray(d["t_te"])
    traj_te = np.asarray(d["traj_te"])
    mu_tr = np.asarray(d["mu_tr"])
    S = A_tr.shape[0]
    fl_tr = np.asarray(jnp.sqrt(fl2_tr / un2_tr))
    fl_te = np.asarray(jnp.sqrt(fl2_te / un2_te))

    # whitening round-trip gate: to_h(to_q(h)) must return h to ~1e-12
    qp_ck = hf.to_q(dict(h=p_ck["h"], h_lin=p_ck["h_lin"]), L)
    hp_rt = hf.to_h(qp_ck, Linv)
    rt = float(max(
        jnp.max(jnp.abs(hp_rt["h"][-1][0] - p_ck["h"][-1][0]))
        / jnp.max(jnp.abs(p_ck["h"][-1][0])),
        jnp.max(jnp.abs(hp_rt["h_lin"] - p_ck["h_lin"]))
        / jnp.max(jnp.abs(p_ck["h_lin"]))))
    zt = jnp.asarray(Z_ck[:8])
    hv = hf.head_apply(dict(h=p_ck["h"], h_lin=p_ck["h_lin"]), zt)
    qv = hf.head_apply(qp_ck, zt)
    wt = float(jnp.max(jnp.abs(qv - hv @ L)) / jnp.max(jnp.abs(qv)))
    log(f"  whitening round-trip {rt:.3e}   q == L^T h deviation {wt:.3e}")
    assert rt < 1e-10 and wt < 1e-10, "whitening reparameterisation is not exact"

    report = dict(config=dict(
        npz=os.path.basename(NPZ), ckpt=os.path.basename(CKPT), arms=ARMS,
        steps=STEPS, batch=BATCH, lr=LR, oracle_iters=ORACLE_ITERS,
        enc_steps=ENC_STEPS, seed=SEED0, K_ckpt=K0, R=R, S=int(S),
        n_test_states=int(A_te.shape[0]), x64=True,
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        gates=dict(whitening_round_trip=rt, q_equals_LT_h=wt),
        floors=dict(train_span_floor_mean=float(fl_tr.mean()),
                    train_span_floor_max=float(fl_tr.max()),
                    test_span_floor_mean=float(fl_te.mean()),
                    test_span_floor_max=float(fl_te.max())),
        arms={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    log(f"  span LS floor: train {fl_tr.mean():.4e}  fresh test "
        f"{fl_te.mean():.4e}  (S={S}, R={R}, K_ckpt={K0})")

    # standardised (mu, t) for the parameter-code diagnostic
    T = int(t_te.max()) + 1
    P_tr = np.concatenate([mu_tr, (t_tr[:, None] / max(T - 1, 1))], axis=1)
    P_tr = (P_tr - P_tr.mean(0)) / (P_tr.std(0) + 1e-30)

    def train_encoder(key, A, Z, steps):
        """small MLP a -> z on TRAINING pairs only (init for the latent LM)."""
        k_ = Z.shape[1]
        ep = hf.init_head(key, A.shape[1], k_, hidden=256, layers=2,
                          lin_scale=0.0)
        sch = optax.warmup_cosine_decay_schedule(0.0, 1e-3,
                                                 min(500, steps // 10 + 1),
                                                 steps, 1e-5)
        o = optax.adam(sch)
        st = o.init(ep)
        zs = float(jnp.std(Z)) + 1e-30

        def ls(p, Ab, Zb):
            return jnp.mean((hf.head_apply(p, Ab) - Zb) ** 2) / zs ** 2

        @jax.jit
        def stp(p, st, kk, A_, Z_):
            nb = min(4096, A_.shape[0])
            i = jax.random.choice(kk, A_.shape[0], shape=(nb,), replace=False)
            v, g = jax.value_and_grad(ls)(p, A_[i], Z_[i])
            u, st = o.update(g, st)
            return optax.apply_updates(p, u), st, v
        v = jnp.inf
        for i in range(steps):
            key, kk = jax.random.split(key)
            ep, st, v = stp(ep, st, kk, A, Z)
        return ep, float(v)

    # nearest training state in whitened coefficient space (DIAGNOSTIC init)
    @jax.jit
    def nn_idx(Ate, Atr):
        d2 = (jnp.sum(Ate * Ate, 1)[:, None] - 2.0 * (Ate @ Atr.T)
              + jnp.sum(Atr * Atr, 1)[None, :])
        return jnp.argmin(d2, axis=1)
    nnj = np.asarray(nn_idx(A_te, A_tr))

    def evaluate(name, qp, Z, tinfo):
        ent = dict(spec=ARM_SPECS[name], train=tinfo, k=int(Z.shape[1]))
        rec = np.asarray(hf.batched_rel(qp, Z, A_tr, fl2_tr, un2_tr))
        ent["recon_train"] = dict(mean=float(rec.mean()), max=float(rec.max()))
        key = jax.random.PRNGKey(SEED0 + 7)
        ep, ev = train_encoder(key, A_tr, Z, ENC_STEPS)
        ent["encoder_final_loss"] = ev
        zbar = jnp.mean(Z, axis=0)
        z_enc = hf.head_apply(ep, A_te)
        inits = jnp.stack([jnp.tile(zbar[None], (A_te.shape[0], 1)), z_enc],
                          axis=1)
        o_rel, _ = hf.oracle(qp, A_te, fl2_te, un2_te, inits, ORACLE_ITERS)
        ent["oracle_test"] = dict(
            mean=float(o_rel.mean()), max=float(o_rel.max()),
            t0=float(o_rel[t_te == 0].mean()),
            early=float(o_rel[t_te <= 5].mean()),
            late=float(o_rel[t_te > 5].mean()),
            per_time=[float(o_rel[t_te == t].mean()) for t in range(T)],
            note="representation oracle, DIAGNOSTIC (uses test truth); inits "
                 "= mean training code + training-only encoder")
        ent["oracle_over_span_floor"] = float(o_rel.mean() / fl_te.mean())
        inits_nn = jnp.concatenate([inits, Z[jnp.asarray(nnj)][:, None]], axis=1)
        onn, _ = hf.oracle(qp, A_te, fl2_te, un2_te, inits_nn, ORACLE_ITERS)
        ent["oracle_test_nn"] = dict(
            mean=float(onn.mean()), max=float(onn.max()),
            note="UPPER BOUND on what better initialisation could buy: adds a "
                 "nearest-training-state init selected USING the test target")
        # code convergence: oracle on the TRAINING targets from the fitted codes
        sub = np.linspace(0, A_tr.shape[0] - 1,
                          min(CODEDIAG_N, A_tr.shape[0])).astype(int)
        sj = jnp.asarray(sub)
        o_tr, _ = hf.oracle(qp, A_tr[sj], fl2_tr[sj], un2_tr[sj],
                            Z[sj][:, None, :], CODEDIAG_ITERS, chunk=1024)
        rec_sub = float(rec[sub].mean())
        ent["oracle_train_from_codes"] = dict(
            mean=float(o_tr.mean()), max=float(o_tr.max()),
            n=int(len(sub)), recon_on_same_subset=rec_sub,
            gain_vs_recon=float(rec_sub / max(o_tr.mean(), 1e-300)),
            note="codes re-solved exactly at frozen h; ratio > 1 means the "
                 "codes had NOT converged during joint training")
        log(f"  ARM {name}: recon {rec.mean():.4e}  oracle {o_rel.mean():.4e}"
            f"  oracle/floor {ent['oracle_over_span_floor']:.1f}x"
            f"  oracle_nn {onn.mean():.4e}"
            f"  code-refit gain {ent['oracle_train_from_codes']['gain_vs_recon']:.3f}x")
        return ent

    emitted = None
    for name in ARMS:
        spec = dict(ARM_SPECS[name])
        t0 = time.time()
        k = int(spec.pop("k", K0))
        warm = spec.pop("warm", None)
        wsel = spec.pop("w", None)
        zinit = spec.pop("zinit", None)
        steps = int(spec.pop("steps", STEPS))
        w_state = None
        if wsel == "early":
            w_state = np.where(t_tr <= 5, 4.0, 1.0)
        Z0 = None
        qp0 = None
        if warm in ("ckpt", "z") and k == K0:
            Z0 = Z_ck
        if warm == "ckpt":
            if k != K0:
                raise SystemExit(f"arm {name}: warm='ckpt' needs k == {K0}")
            qp0 = qp_ck
        if zinit == "params":
            Z0 = jnp.asarray(P_tr[:, :k], dtype=F64)
            if k > P_tr.shape[1]:
                raise SystemExit(f"arm {name}: k={k} > {P_tr.shape[1]} params")
        if steps == 0:
            qp, Z, tinfo = qp0, Z0, dict(steps=0, seconds=0.0,
                                         note="checkpoint as-is, no refit")
        else:
            qp, Z, tinfo = hf.fit(
                jax.random.PRNGKey(SEED0 + 11), A_tr, un2_tr, fl2_tr, k, R,
                steps=steps, lr=LR, batch=BATCH, w_state=w_state, qp0=qp0,
                Z0=Z0, time_cap=TIME_CAP, tag=name, **spec)
        report["arms"][name] = evaluate(name, qp, Z, tinfo)
        report["arms"][name]["seconds"] = time.time() - t0
        save()
        if EMIT == name:
            hp = hf.to_h(qp, Linv)
            newp = {kk: vv for kk, vv in ck["params"].items()}
            newp["h"] = [(np.asarray(w), np.asarray(b)) for w, b in hp["h"]]
            newp["h_lin"] = np.asarray(hp["h_lin"])
            if "hB" in hp:
                newp["hB"] = np.asarray(hp["hB"])
            else:
                newp.pop("hB", None)
            cfg = dict(ck.get("cfg", {}))
            cfg["k"] = int(k)
            cfg["hfit_arm"] = name
            cfg["hfit_source_ckpt"] = os.path.basename(CKPT)
            cfg["hfit_spec"] = {kk2: vv2 for kk2, vv2
                                in ARM_SPECS[name].items()}
            path = EMIT_PATH or f"hfit_{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(dict(params=newp, Z_tr=np.asarray(Z), cfg=cfg), f)
            emitted = path
            log(f"  EMITTED refined decoder -> {path}")
    report["emitted"] = emitted
    report["total_seconds"] = time.time() - t_all
    report["complete"] = True
    save()
    log(f"HFIT done in {report['total_seconds']:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
