"""Phase C only, from a saved auto-decoder pkl, with a (larger) LM budget —
decides whether the ROM floor is SOLVER-budget (error keeps falling with more
attempts) or OBJECTIVE (min-residual latent != min-field-error latent).
Same residual/collocation/init definitions as ms_autodecoder.py phase C.
Usage: K_LAT=.. GN_ITERS=300 [M_EQ=512] [N_TEST=16] python ms_rom_rerun.py <pkl> <out.json>
(env N/N_TRAIN/N_VAL/HIDDEN must match the pkl config — asserted)."""
import json, os, pickle, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ms_parametric as mp
from ms_autodecoder import lm_solve, infer_latents_lm

PKL, OUT = sys.argv[1], sys.argv[2]
GN_ITERS = int(os.environ.get("GN_ITERS", "300"))
M_EQ = int(os.environ.get("M_EQ", "512")); N_TEST = int(os.environ.get("N_TEST", "16"))
d = pickle.load(open(PKL, "rb")); cfg = d["config"]
for k in ("N", "n_train", "n_val", "seed", "hidden", "n_layers"):
    assert cfg[k] == mp.CONFIG[k], f"config mismatch {k}"
K = cfg["K_LAT"]; N, N_TRAIN = mp.N, mp.N_TRAIN
stages = mp.stages_from_np(d["stages"]); Z_tr = d["z_tr"]
print(f"jax_backend={jax.default_backend()} K={K} stages={len(stages)} GN_ITERS={GN_ITERS}", flush=True)
U, z_true_all, coords, _ = mp.build_snapshots(N)
U_va = U[N_TRAIN:]; zt = np.asarray(z_true_all)
nn_idx = np.argmin(((zt[N_TRAIN:, None, :] - zt[None, :N_TRAIN, :]) ** 2).sum(-1), axis=1)
z_mean = Z_tr.mean(0)
cx, cy, w, a, _ = mp.sample_params(); sl = slice(N_TRAIN, N_TRAIN + N_TEST)
cx, cy, w, a = cx[sl], cy[sl], w[sl], a[sl]
U_test = np.asarray(U_va[:N_TEST]); tn = np.linalg.norm(U_test, axis=1)
dx = 1.0 / (N - 1)
def stencil(ix, iy):
    pts, keep = [], []
    for ox, oy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
        jx, jy = ix+ox, iy+oy
        pts.append(np.stack([jx*dx, jy*dx], 1)); keep.append(~((jx==0)|(jx==N-1)|(jy==0)|(jy==N-1)))
    return jnp.asarray(np.stack(pts)), jnp.asarray(np.stack(keep).astype(float))
ii, jj = np.meshgrid(np.arange(1,N-1), np.arange(1,N-1), indexing="ij"); ixf, iyf = ii.reshape(-1), jj.reshape(-1)
sub = np.random.default_rng(mp.SEED + 12345).choice(len(ixf), size=min(M_EQ, len(ixf)), replace=False)
colls = {"full": (ixf, iyf), f"m{M_EQ}": (ixf[sub], iyf[sub])}
inits = {"nearest": Z_tr[nn_idx][:N_TEST], "mean": np.tile(z_mean, (N_TEST, 1))}
oracle = {ns: infer_latents_lm(stages[:ns], coords, U_va[:N_TEST], inits, GN_ITERS) for ns in range(1, len(stages)+1)}
rows = []
for cname, (ix, iy) in colls.items():
    pts, keep = stencil(ix, iy)
    for ns in range(1, len(stages)+1):
        st = stages[:ns]; dec = lambda z, xy: mp.combined_apply(st, z, xy)
        def residual(z, f):
            u = dec(z, pts.reshape(-1, 2)).reshape(5, -1) * keep
            return -(u[1]+u[2]+u[3]+u[4]-4.0*u[0])/(dx*dx) - f
        rJ = jax.jit(lambda z, f: (residual(z, f), jax.jacfwd(residual)(z, f)))
        rn = jax.jit(lambda z, f: jnp.linalg.norm(residual(z, f)))
        decf = jax.jit(lambda z: dec(z, coords))
        for iname in ("nearest", "mean"):
            errs, rl, ro, eo, acc, reasons, its = [], [], [], [], [], [], []
            t0 = time.time()
            for i in range(N_TEST):
                f = jnp.asarray(a[i]*np.exp(-((ix*dx-cx[i])**2+(iy*dx-cy[i])**2)/(2*w[i]**2)))
                z, r, info = lm_solve(lambda zz: rJ(zz, f), lambda zz: rn(zz, f), jnp.asarray(inits[iname][i]), GN_ITERS)
                errs.append(np.linalg.norm(np.asarray(decf(z)) - U_test[i]) / tn[i]); rl.append(r)
                zo = jnp.asarray(oracle[ns][iname]["Z"][i]); ro.append(float(rn(zo, f))); eo.append(float(oracle[ns][iname]["rel"][i]))
                acc.append(info["accepted"]); reasons.append(info["reason"]); its.append(info["attempts"])
            row = dict(colloc=cname, n_stages=ns, init=iname, budget_attempts=GN_ITERS,
                       rom_rel_l2_mean=float(np.mean(errs)), rom_rel_l2_med=float(np.median(errs)), rom_rel_l2_max=float(np.max(errs)),
                       oracle_rel_l2_mean=float(np.mean(eo)), resid_lm_med=float(np.median(rl)), resid_oracle_med=float(np.median(ro)),
                       lm_accepted_med=float(np.median(acc)), attempts_med=float(np.median(its)),
                       lm_reasons={r: reasons.count(r) for r in set(reasons)}, per_sample_rom_rel_l2=[float(v) for v in errs], secs=time.time()-t0)
            rows.append(row)
            print(f"RESULT K{K} colloc={cname:5s} stages={ns} init={iname:7s} ROM {row['rom_rel_l2_mean']:.3e} (med {row['rom_rel_l2_med']:.3e}) "
                  f"oracle {row['oracle_rel_l2_mean']:.3e} ||r|| lm {row['resid_lm_med']:.2e} oracle {row['resid_oracle_med']:.2e} "
                  f"acc {row['lm_accepted_med']:.0f}/{row['attempts_med']:.0f} {row['lm_reasons']}", flush=True)
            json.dump({"config": dict(cfg, gn_iters=GN_ITERS), "rom": rows, "complete": False}, open(OUT, "w"), indent=2)
json.dump({"config": dict(cfg, gn_iters=GN_ITERS), "rom": rows, "complete": True}, open(OUT, "w"), indent=2)
print("DONE", flush=True)
