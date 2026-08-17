"""Stage 1 pilot: SPACE-TIME LSPG with the (z,t)-conditioned FiLM decoder from
the wave2d sweep (wave2d_film_N{N}.pkl, 5-dim true-parameter latent, f32 net).

The ROM knows u0 (and u_t(.,0)=0), the wave speed c and the PDE; it never sees
the held-out trajectory.  We solve for the 5-dim z by LM on:
  (a) ic     : misfit of u(.,0;z) to u0                       [IC-fit control]
  (b) resid  : the u-only Newmark residual of the decoded trajectory over the
               horizon at dt = DT_SNAP/S1_RS (interior rows + the FOM's
               boundary rows, since the sweep decoder is not hard-BC), the
               decoder supplying u(., t; z) at every time level -- a
               PDE-CONSISTENCY ABLATION: "no IC term in the OBJECTIVE".  Note
               (Codex SHOULD) that u0 still enters through the nearest-IC
               INITIALISATION, as it does in every arm; u0 is allowed
               information, so this is an objective ablation, not a
               u0-blind test.
  (c) both   : (a) and (b) stacked, IC block weight IC_W (default 1; the
               Burgers round found IC_W=1 best; sqrt(S) as a sensitivity arm)
and report the traj-RMS error vs the FOM next to the ORACLE (true z) error and
|z - z_true|.  Inits: family mean and the z of the training trajectory with the
nearest IC (legit: u0 known), both with the speed coordinate set from the KNOWN
c; best-of on the objective.  Test trajectories from TEST_SEED.

Usage: N=64 [S1_RS=10] [S1_BUDGET=60] [IC_W=1] python wlat_stage1.py <ckpt.pkl> <outdir>
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import wlat_common as wc
from wlat_common import wf, F64, log, lm_solve

CKPT = sys.argv[1]
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."
N = wc.N
T1 = wc.NUM_STEPS + 1
S1_RS = int(os.environ.get("S1_RS", "10"))            # residual time levels per snapshot interval
DT1 = wc.DT_SNAP / S1_RS
IC_W = float(os.environ.get("IC_W", "1.0"))
BUDGET = int(os.environ.get("S1_BUDGET", "60"))
# extra initialisations for `both`: the k nearest training ICs (legitimate -- u0 is
# known) and, as a LABELLED ORACLE DIAGNOSTIC, the true z.  The oracle-init arm
# answers the question a failed `both` raises: is the objective's minimiser not at
# z*, or is LM merely stuck in a local minimum on the way there?
N_MULTI = int(os.environ.get("S1_MULTI", "6"))
DO_ORACLE_INIT = int(os.environ.get("S1_ORACLE_INIT", "1"))
F32 = jnp.float32


def main():
    log(f"jax_backend={jax.default_backend()}  N={N}  n_freq={wf.N_FREQ} t_freq={wf.T_FREQ}  "
        f"S1_RS={S1_RS} dt={DT1:g} IC_W={IC_W} budget={BUDGET} "
        f"lev_chunk={os.environ.get('S1_LEV_CHUNK', '8')}")
    with open(CKPT, "rb") as f:
        params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=F32), pickle.load(f))
    rj = os.path.join(os.path.dirname(CKPT), f"wave2d_results_N{N}.json")
    if os.path.exists(rj):
        with open(rj) as f:
            rmeta = json.load(f)
        for k_, v_ in (("N", N), ("n_freq", wf.N_FREQ), ("t_freq", wf.T_FREQ), ("hidden", wf.HIDDEN),
                       ("n_train", wf.N_TRAIN), ("seed", wf.SEED), ("substeps", wf.SUBSTEPS)):
            if rmeta.get(k_) != v_:
                raise SystemExit(f"checkpoint manifest mismatch {k_}: {rmeta.get(k_)} vs {v_}")
        n_par = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
        if rmeta.get("params_film") != n_par:
            raise SystemExit(f"param count {n_par} != manifest {rmeta.get('params_film')}")
    else:
        log(f"  WARNING: no results JSON next to {CKPT}; architecture unverified")
    d = wc.build_data(N)
    U, z_all = d["U"], d["z"].astype(np.float64)
    U_te, z_te, c_te = d["U_test"], d["z_test"].astype(np.float64), d["c_test"]
    z_tr = z_all[:wc.N_TRAIN]
    U_tr0 = U[:wc.N_TRAIN, 0]
    coords = jnp.asarray(wc.grid_coords(N), dtype=F32)
    interior = np.asarray(wc.interior_indices(N))
    bnd = jnp.asarray(np.setdiff1d(np.arange(N * N), interior))
    n_lev = wc.NUM_STEPS * S1_RS + 1
    taus = jnp.asarray(np.arange(n_lev) / (n_lev - 1), dtype=F32)     # physical t in [0,1]
    taus_snap = jnp.asarray(np.arange(T1) / wc.NUM_STEPS, dtype=F32)

    # vmapping all n_lev time levels at once materialises n_lev x n^2 x HIDDEN
    # activations, and forward-mode AD over the 5 latents multiplies that by 6:
    # 25 GiB at N=64, S1_RS=10 (OOM'd job 2479742).  lax.map with a batch size
    # turns it into a scan over chunks, so peak memory is per-chunk and the
    # residual is unchanged.
    LEV_CHUNK = int(os.environ.get("S1_LEV_CHUNK", "8"))

    def field(z, tau):
        return wf.film_apply(params, jnp.asarray(z, F32), tau, coords).astype(F64)

    def traj_lev(z):
        return jax.lax.map(lambda t: field(z, t), taus, batch_size=LEV_CHUNK)

    def traj_snap(z):
        return jax.lax.map(lambda t: field(z, t), taus_snap, batch_size=LEV_CHUNK)

    def r_ic(z, u0):
        return field(z, 0.0) - u0

    def r_res(z, c):
        """u-only Newmark residual over all levels: interior rows (exact FOM
        stencil at dt = DT1) + boundary rows (u = 0 on the walls at every level)."""
        Uz = traj_lev(z)
        a = (0.5 * DT1 * c) ** 2
        L = jax.lax.map(lambda u: wc.lap_full_field(u, N), Uz, batch_size=LEV_CHUNK)
        first = (Uz[1] - Uz[0]) - a * (L[1] + L[0])
        gen = (Uz[2:] - 2.0 * Uz[1:-1] + Uz[:-2]) - a * (L[2:] + 2.0 * L[1:-1] + L[:-2])
        R = jnp.concatenate([first[None], gen], axis=0)[:, jnp.asarray(interior)]
        return jnp.concatenate([R.reshape(-1), Uz[:, bnd].reshape(-1)])

    def r_both(z, u0, c):
        return jnp.concatenate([IC_W * r_ic(z, u0), r_res(z, c)])

    def mk(fn):
        return (jax.jit(lambda z, u0, c: (fn(z, u0, c), jax.jacfwd(fn)(z, u0, c))),
                jax.jit(lambda z, u0, c: jnp.linalg.norm(fn(z, u0, c))))

    objs = {"ic": mk(lambda z, u0, c: r_ic(z, u0)),
            "resid": mk(lambda z, u0, c: r_res(z, c)),
            "both": mk(r_both)}
    traj_j = jax.jit(traj_snap)

    def traj_err(z, Ut):
        P = np.asarray(traj_j(jnp.asarray(z)))
        per_t, per_s, tr, sn = wc.traj_metrics(P, Ut)
        return tr, sn, per_t

    report = dict(config=dict(wc.CONFIG, s1_rs=S1_RS, dt1=DT1, ic_w=IC_W, budget=BUDGET,
                              ckpt=os.path.basename(CKPT), n_freq=wf.N_FREQ, t_freq=wf.T_FREQ),
                  backend=jax.default_backend(), data_fingerprint=wc.data_fingerprint(U))
    zmean = z_tr.mean(axis=0)
    orc = [traj_err(z_te[i], U_te[i]) for i in range(wc.N_TEST)]
    report["oracle_true_z"] = dict(traj_rel_mean=float(np.mean([o[0] for o in orc])),
                                   traj_rel_max=float(np.max([o[0] for o in orc])),
                                   snap_rel_mean=float(np.mean([o[1] for o in orc])))
    log(f"  ORACLE (true z) traj rel mean {report['oracle_true_z']['traj_rel_mean']:.3e}")
    # residual sanity: the same-dt Newmark FOM states through r_res's formula must vanish;
    # the sweep decoder at the true z gives the decoder's own PDE inconsistency
    st = wc.newmark_first_states(N, S1_RS, U_te[0, 0], float(c_te[0]), 3)
    a = (0.5 * DT1 * float(c_te[0])) ** 2
    Ls = [np.asarray(wc.lap_full_field(jnp.asarray(u), N)) for u in st]
    r0 = (st[1] - st[0]) - a * (Ls[1] + Ls[0]); r1 = (st[2] - 2 * st[1] + st[0]) - a * (Ls[2] + 2 * Ls[1] + Ls[0])
    report["newmark_states_residual_maxabs"] = float(max(np.abs(r0[interior]).max(), np.abs(r1[interior]).max()) / np.linalg.norm(st[0]))
    rt = [float(objs["resid"][1](jnp.asarray(z_te[i]), jnp.asarray(U_te[i, 0]), float(c_te[i]))
                / np.linalg.norm(U_te[i])) for i in range(wc.N_TEST)]
    report["resid_at_true_z_rel"] = float(np.mean(rt))
    log(f"  Newmark-state residual check {report['newmark_states_residual_maxabs']:.1e}; "
        f"decoder residual at true z (rel) {np.mean(rt):.3e}")

    arms = list(objs.items())
    if DO_ORACLE_INIT:
        arms.append(("both_oracleinit", objs["both"]))       # LABELLED ORACLE DIAGNOSTIC
        arms.append(("both_multistart", objs["both"]))
    for name, (rJ, rn) in arms:
        t0 = time.time()
        rows = []
        for i in range(wc.N_TEST):
            u0 = jnp.asarray(U_te[i, 0]); c = float(c_te[i])
            d_ic = np.linalg.norm(U_tr0 - U_te[i, 0], axis=1)
            j = int(np.argmin(d_ic))
            z_c = np.log(c) / np.log(2.0)                            # known c -> z[4]
            z0_mean = zmean.copy(); z0_mean[4] = z_c
            z0_near = z_tr[j].copy(); z0_near[4] = z_c
            inits = [("mean", z0_mean), ("nearest_ic", z0_near)]
            if name == "both_oracleinit":
                # the TRUE z as the only start: if LM stays there and the objective is
                # lower than the `both` optimum, `both` is a local-minimum failure;
                # if LM walks away to a LOWER objective, the residual+IC objective's
                # minimiser genuinely is not z*.  Not available to any ROM.
                inits = [("true_z", z_te[i].copy())]
            elif name == "both_multistart":
                for rank in range(1, N_MULTI):
                    jj = int(np.argsort(d_ic)[rank])
                    zk = z_tr[jj].copy(); zk[4] = z_c
                    inits.append((f"near{rank}", zk))
            best = None
            for iname, z0 in inits:
                z, r, info = lm_solve(lambda zz: rJ(zz, u0, c), lambda zz: rn(zz, u0, c),
                                      jnp.asarray(z0, F64), BUDGET)
                if best is None or r < best[1]:
                    best = (np.asarray(z), r, iname, info)
            z, r, iname, info = best
            e, es, per = traj_err(z, U_te[i])
            obj_at_true = float(rn(jnp.asarray(z_te[i], F64), u0, c))
            rows.append(dict(traj_rel=e, snap_rel=es, z_err=float(np.linalg.norm(z - z_te[i])),
                             init=iname, obj=float(r), obj_at_true_z=obj_at_true,
                             obj_ratio_found_over_true=float(r) / max(obj_at_true, 1e-300),
                             accepted=info["accepted"], reason=info["reason"],
                             per_time=per.tolist()))
        tr = np.array([r["traj_rel"] for r in rows])
        report[name] = dict(traj_rel_mean=float(tr.mean()), traj_rel_median=float(np.median(tr)),
                            traj_rel_max=float(tr.max()),
                            snap_rel_mean=float(np.mean([r["snap_rel"] for r in rows])),
                            z_err_mean=float(np.mean([r["z_err"] for r in rows])),
                            per_time_mean=np.mean([r["per_time"] for r in rows], axis=0).tolist(),
                            obj_ratio_found_over_true_mean=float(np.mean(
                                [r["obj_ratio_found_over_true"] for r in rows])),
                            n_found_below_true=int(sum(r["obj_ratio_found_over_true"] < 1.0
                                                       for r in rows)),
                            rows=rows, secs=time.time() - t0)
        log(f"  {name:16s}: traj rel mean {tr.mean():.3e} (med {np.median(tr):.3e}, "
            f"max {tr.max():.3e})  |z-z*| {report[name]['z_err_mean']:.3e}  obj(found)/obj(z*) "
            f"{report[name]['obj_ratio_found_over_true_mean']:.3f} "
            f"({report[name]['n_found_below_true']}/{len(rows)} below)  [{time.time()-t0:.0f}s]")

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"wlat_stage1_N{N}.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log("wrote stage1 json")


if __name__ == "__main__":
    main()
