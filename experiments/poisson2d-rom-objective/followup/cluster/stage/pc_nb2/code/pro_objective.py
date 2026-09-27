"""Cell 1 (+3): ROM-solve OBJECTIVE sweep at full collocation, several inits,
on a saved FiLM auto-decoder pkl (multistage-precision format).

For every (objective, init): LM latent solve on the held-out test sources
(the solver sees ONLY the source f and the decoder), then field error vs the
FD solution, compared with the ORACLE latent (same init, same LM budget, LM on
the data misfit — the finite-budget inferred-latent floor).  Per arm we log
the objective value AND the plain FD residual norm at both z_LM and z_oracle:
    obj(z_LM) <  obj(z_or) with err(z_LM) >> err(z_or)  => OBJECTIVE floor
    obj(z_LM) >  obj(z_or)                              => SOLVER floor (local min / budget)
Note: 'spec_a1_Mall' is the INTERIOR data misfit ||u_int(z) - u*_int|| (the
oracle/reported error also include the decoder's boundary values, whose norm is
recorded as boundary_block — ~1e-4 here, negligible; hard-BC decoders make the
two identical).  The 'nearest' init uses the source parameters (cx,cy,w,a) of
the test source — legitimate online input for this family (the source f IS the
input); the 'encoder' init is the opaque-f alternative.

Usage:
  PKL=<stages.pkl> [NS=1] [N_TEST=16] [GN_ITERS=60]   (hard-BC flag comes from the pkl)
  [OBJECTIVES=fd,spec_a0_M64,...] [INITS=mean,nearest,encoder] [ENC_STEPS=3000]
  python pro_objective.py <out.json>
Env N/N_TRAIN/N_VAL/HIDDEN/N_LAYERS/SEED must match the pkl (asserted).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pro_common as pc
from pro_common import mp

PKL = os.environ["PKL"]
OUT = sys.argv[1]
NS = int(os.environ.get("NS", "1"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "3000"))
DEFAULT_OBJ = ("fd,"
               "spec_a0_M64,spec_a0_M256,spec_a0_M1024,"
               "spec_a0.5_M64,spec_a0.5_M256,spec_a0.5_M1024,spec_a0.5_Mall,"
               "spec_a1_M64,spec_a1_M256,spec_a1_M1024,spec_a1_Mall,"
               "ritz,cg5,cg20,lowpass2,lowpass4,lowpass8")
OBJECTIVES = os.environ.get("OBJECTIVES", DEFAULT_OBJ).split(",")
INITS = os.environ.get("INITS", "mean,nearest,encoder").split(",")


def main():
    print(f"jax_backend={jax.default_backend()} x64={jax.config.jax_enable_x64}", flush=True)
    d, cfg, stages_all, Z_tr, HARD_BC = pc.load_pkl(PKL)
    K = cfg["K_LAT"]; N = mp.N; N_TRAIN = mp.N_TRAIN
    assert 1 <= NS <= len(stages_all), f"NS={NS} but pkl has {len(stages_all)} stages"
    assert 1 <= N_TEST <= mp.N_VAL
    stages = stages_all[:NS]
    dec = pc.make_decoder(stages, hard_bc=bool(HARD_BC))
    grid = pc.Grid(N)
    manifest = dict(pkl=os.path.basename(PKL), pkl_config=cfg, ns=NS, n_test=N_TEST,
                    gn_iters=GN_ITERS, hard_bc=HARD_BC, objectives=OBJECTIVES, inits=INITS,
                    enc_steps=ENC_STEPS, backend=jax.default_backend())
    print("MANIFEST " + json.dumps(manifest), flush=True)

    U, z_true_all, coords, fom_res = mp.build_snapshots(N)
    U_va = U[N_TRAIN:]
    zt = np.asarray(z_true_all)
    nn_idx = np.argmin(((zt[N_TRAIN:, None, :] - zt[None, :N_TRAIN, :]) ** 2).sum(-1), axis=1)
    cx, cy, w, a, _ = mp.sample_params()
    sl = slice(N_TRAIN, N_TRAIN + N_TEST)
    U_test = np.asarray(U_va[:N_TEST]); tn = np.linalg.norm(U_test, axis=1)
    F2d = [jnp.asarray(mp.source_interior(N, cx[i], cy[i], w[i], a[i]))
           for i in range(N_TRAIN, N_TRAIN + N_TEST)]
    z_mean = Z_tr.mean(0)

    # ---- inits ----
    inits = {"mean": np.tile(z_mean, (N_TEST, 1)), "nearest": Z_tr[nn_idx][:N_TEST]}
    enc_info = None
    if "encoder" in INITS:
        t0 = time.time()
        F_lat_tr = pc.lattice_source(cx[:N_TRAIN], cy[:N_TRAIN], w[:N_TRAIN], a[:N_TRAIN])
        F_lat_te = pc.lattice_source(cx[sl], cy[sl], w[sl], a[sl])
        enc, _, mse = pc.fit_encoder(jax.random.PRNGKey(mp.SEED + 7), F_lat_tr, Z_tr,
                                     steps=ENC_STEPS)
        Z_enc = enc(F_lat_te)
        inits["encoder"] = Z_enc
        # encoder plug-in error (no solve at all)
        pred = np.asarray(jax.vmap(lambda z: dec(z, coords))(jnp.asarray(Z_enc)))
        plug = np.linalg.norm(pred - U_test, axis=1) / tn
        enc_info = dict(train_mse=mse, plugin_rel_l2_mean=float(plug.mean()),
                        plugin_rel_l2_med=float(np.median(plug)), secs=time.time() - t0)
        print(f"ENCODER train-mse {mse:.2e}  plug-in held-out rel-L2 {plug.mean():.3e} "
              f"(med {np.median(plug):.3e})", flush=True)
    inits = {k: v for k, v in inits.items() if k in INITS}

    # ---- oracle: finite-budget inferred latents, same inits and budget ----
    # (make_data_misfit in ms_autodecoder uses mp.combined_apply directly; for
    #  hard-BC decoders we build the same thing through dec.)
    def infer(dec_, inits_):
        rJ = jax.jit(lambda z, u: (dec_(z, coords) - u, jax.jacfwd(lambda zz: dec_(zz, coords) - u)(z)))
        rn = jax.jit(lambda z, u: jnp.linalg.norm(dec_(z, coords) - u))
        out = {}
        for name, Z0 in inits_.items():
            Zs, rels = [], []
            for i in range(N_TEST):
                u = jnp.asarray(U_test[i])
                z, r, info = pc.lm_solve(lambda zz: rJ(zz, u), lambda zz: rn(zz, u),
                                         jnp.asarray(Z0[i]), GN_ITERS)
                Zs.append(np.asarray(z)); rels.append(r / tn[i])
            out[name] = {"Z": np.stack(Zs), "rel": np.asarray(rels)}
        return out
    t0 = time.time()
    oracle = infer(dec, inits)
    print("ORACLE (data-misfit LM, same budget): " + "  ".join(
        f"{k}={v['rel'].mean():.3e}" for k, v in oracle.items()) + f"  [{time.time()-t0:.0f}s]",
        flush=True)

    dec_full = jax.jit(lambda z: dec(z, coords))
    bnorm = jax.jit(lambda z: jnp.linalg.norm(dec(z, grid.bpts)))
    report = dict(manifest=manifest, fom_max_rel_residual=fom_res, encoder=enc_info,
                  oracle={k: dict(rel_l2_mean=float(v["rel"].mean()),
                                  rel_l2_med=float(np.median(v["rel"])),
                                  per_sample=[float(x) for x in v["rel"]])
                          for k, v in oracle.items()},
                  rows=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1)

    for oname in OBJECTIVES:
        spec = pc.parse_objective(oname)
        HgV, V, diag = pc.make_full_objective(dec, grid, spec)
        for iname, Z0 in inits.items():
            per = {k: [] for k in ("err", "err_or", "obj_lm", "obj_or", "fd_lm", "fd_or",
                                   "f_norm", "bnd", "acc", "rej", "att", "reason", "lam",
                                   "z_norm", "z_nn", "z_or_dist")}
            t0 = time.time()
            for i in range(N_TEST):
                f2d = F2d[i]
                z, val, info = pc.lm_generic(lambda zz: HgV(zz, f2d), lambda zz: V(zz, f2d),
                                             jnp.asarray(Z0[i]), GN_ITERS,
                                             use_rel_dec=(spec["kind"] != "ritz"))
                zo = jnp.asarray(oracle[iname]["Z"][i])
                fd_lm, obj_lm = diag(z, f2d); fd_or, obj_or = diag(zo, f2d)
                per["err"].append(float(np.linalg.norm(np.asarray(dec_full(z)) - U_test[i]) / tn[i]))
                per["err_or"].append(float(oracle[iname]["rel"][i]))
                per["obj_lm"].append(float(obj_lm)); per["obj_or"].append(float(obj_or))
                per["fd_lm"].append(float(fd_lm)); per["fd_or"].append(float(fd_or))
                per["f_norm"].append(float(jnp.linalg.norm(f2d)))
                per["bnd"].append(float(bnorm(z)))
                per["acc"].append(info["accepted"]); per["rej"].append(info["rejected"])
                per["att"].append(info["attempts"]); per["reason"].append(info["reason"])
                per["lam"].append(info["final_lambda"])
                zn = np.asarray(z)
                per["z_norm"].append(float(np.linalg.norm(zn)))
                per["z_nn"].append(float(np.min(np.linalg.norm(Z_tr - zn, axis=1))))
                per["z_or_dist"].append(float(np.linalg.norm(zn - np.asarray(zo))))
            e = np.asarray(per["err"])
            obj_better = int(np.sum(np.asarray(per["obj_lm"]) <= np.asarray(per["obj_or"])))
            n_modes = grid.n_modes(spec.get("M")) if spec["kind"] == "spec" else None
            row = dict(objective=oname, init=iname, ns=NS, budget=GN_ITERS, n_modes_retained=n_modes,
                       rom_rel_l2_mean=float(e.mean()), rom_rel_l2_med=float(np.median(e)),
                       rom_rel_l2_max=float(e.max()),
                       oracle_rel_l2_mean=float(np.mean(per["err_or"])),
                       obj_lm_med=float(np.median(per["obj_lm"])),
                       obj_oracle_med=float(np.median(per["obj_or"])),
                       n_obj_lm_le_oracle=obj_better,
                       fd_resid_lm_med=float(np.median(per["fd_lm"])),
                       fd_resid_oracle_med=float(np.median(per["fd_or"])),
                       f_norm_med=float(np.median(per["f_norm"])),
                       boundary_block_med=float(np.median(per["bnd"])),
                       lm_accepted_med=float(np.median(per["acc"])),
                       lm_rejected_med=float(np.median(per["rej"])),
                       lm_attempts_med=float(np.median(per["att"])),
                       lm_reasons={r: per["reason"].count(r) for r in set(per["reason"])},
                       z_norm_med=float(np.median(per["z_norm"])),
                       z_nn_dist_med=float(np.median(per["z_nn"])),
                       z_dist_to_oracle_med=float(np.median(per["z_or_dist"])),
                       per_sample_rom_rel_l2=[float(v) for v in e],
                       per_sample_oracle_rel_l2=[float(v) for v in per["err_or"]],
                       secs=time.time() - t0)
            report["rows"].append(row)
            print(f"RESULT obj={oname:16s} init={iname:8s} ROM {row['rom_rel_l2_mean']:.3e} "
                  f"(med {row['rom_rel_l2_med']:.3e} max {row['rom_rel_l2_max']:.3e}) "
                  f"oracle {row['oracle_rel_l2_mean']:.3e} | obj lm {row['obj_lm_med']:.3e} "
                  f"or {row['obj_oracle_med']:.3e} (lm<=or {obj_better}/{N_TEST}) | fd-res lm "
                  f"{row['fd_resid_lm_med']:.2e} or {row['fd_resid_oracle_med']:.2e} f "
                  f"{row['f_norm_med']:.2e} | acc/rej {row['lm_accepted_med']:.0f}/"
                  f"{row['lm_rejected_med']:.0f} {row['lm_reasons']} [{row['secs']:.0f}s]",
                  flush=True)
            save()
    report["complete"] = True
    save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
