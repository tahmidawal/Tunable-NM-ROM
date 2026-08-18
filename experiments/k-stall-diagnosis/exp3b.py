"""EXPERIMENT 3: the FIX.  Same decoders, same weak objective, same tau, same
budget, same initial guess -- only the LM's globalisation changes.

Arms
  base    : the current solver (lam0 = 1e-6, accept ANY decrease)
  lam0_1  : lam0 = 1.0
  tr      : base + a trust region ||dz|| <= TR * R_train, R_train = the radius of
            the TRAINING latent cloud (max_i ||Z_i - z_mean||).  The iterate can
            never jump out of the region the decoder was trained on.
  nielsen : the textbook gain-ratio LM
Reported: mean and median ROM/ceiling ratio per k, and the count of blown-up sources.
"""
import json, os, sys
import numpy as np, jax
import jax.numpy as jnp
import kcommon as kc, klm, klm2

NN = int(sys.argv[1]) if len(sys.argv) > 1 else 64
M = 64; TAU = 1e-3; BUDGET = 60
KS = [int(v) for v in os.environ.get("KS","6,8,12,16").split(",")]
TR = float(os.environ.get("TR", "1.0"))

grid, Fs, U, tn, nn_idx, par = kc.testbed(NN)
spec, pts, wq, PhiT, Wl = kc.weak_full(grid, M)
f_ms = [kc.fm(grid, spec, Fs[i]) for i in range(kc.N_TEST)]
print(f"jax_backend={jax.default_backend()} N={NN} M={M} modes={PhiT.shape[0]} "
      f"tau={TAU} budget={BUDGET} TR={TR}", flush=True)

CEIL = f"ceil_N{NN}.json"
ceil = json.load(open(CEIL)) if os.path.isfile(CEIL) else {}
out = []
for k in KS:
    dec, Z = kc.load(k)
    _, rJ, rn = klm.make(dec, k, pts, wq, PhiT, Wl)
    u_full = jax.jit(lambda z, _d=dec: _d(z, grid.coords_int))
    mis = lambda z, u: u_full(z) - u
    mrJ = jax.jit(lambda z, u: (mis(z, u), jax.jacfwd(mis)(z, u)))
    mrn = jax.jit(lambda z, u: jnp.linalg.norm(mis(z, u)))
    z_mean = jnp.asarray(Z.mean(0))
    R_train = float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))
    delta = TR * R_train
    res = {a: [] for a in ("base", "lam0_1", "tr", "nielsen")}
    cl = []
    for i in range(kc.N_TEST):
        key = f"{k}_{i}"
        if key not in ceil:
            ui = jnp.asarray(U[i].reshape(-1)); best = None
            for z0 in (z_mean, jnp.asarray(Z[nn_idx[i]])):
                zc, v, _, _, _, _, _ = klm.lm_trace(mrJ, mrn, k, z0, ui, 0.0, int(os.environ.get("CEILB","150")))
                if best is None or v < best[1]: best = (zc, v)
            ceil[key] = float(best[1]) / tn[i]
        cl.append(ceil[key])
        for arm, kw in (("base",   dict(variant="base", lam0=1e-6)),
                        ("lam0_1", dict(variant="base", lam0=1.0)),
                        ("tr",     dict(variant="tr",   lam0=1e-6, delta=delta)),
                        ("nielsen",dict(variant="nielsen", lam0=1e-2, rho_min=1e-4))):
            z, val, v0, att, acc, rsn = klm2.lm(rJ, rn, k, z_mean, f_ms[i], TAU,
                                                BUDGET, **kw)
            e = float(np.linalg.norm(np.asarray(u_full(jnp.asarray(z))).reshape(
                grid.n_i, grid.n_i) - U[i]) / tn[i])
            res[arm].append(dict(err=e, obj=val, red=val/v0, it=att, acc=acc, rsn=rsn))
    cl = np.array(cl)
    line = f"k={k:2d} R_train={R_train:.3f} ceiling mean {cl.mean():.3e} med {np.median(cl):.3e}"
    for arm in ("base", "lam0_1", "tr", "nielsen"):
        e = np.array([r["err"] for r in res[arm]])
        nb = int((e > 10*cl).sum())
        red = np.array([r["red"] for r in res[arm]])
        line += (f"\n    {arm:8s} err mean {e.mean():.3e} med {np.median(e):.3e} "
                 f"| ratio mean {e.mean()/cl.mean():5.2f} med {np.median(e/cl):5.2f} "
                 f"| blown {nb:2d}/16 | red {red.mean():.2e} "
                 f"| it {np.mean([r['it'] for r in res[arm]]):5.1f}")
        out.append(dict(N=NN, k=k, arm=arm, err_mean=float(e.mean()),
                        err_med=float(np.median(e)), ceil_mean=float(cl.mean()),
                        ceil_med=float(np.median(cl)),
                        ratio_mean=float(e.mean()/cl.mean()),
                        ratio_med=float(np.median(e/cl)), n_blown=nb,
                        iters=float(np.mean([r["it"] for r in res[arm]])),
                        red_mean=float(red.mean()),
                        err_per_source=[float(v) for v in e],
                        ceil_per_source=[float(v) for v in cl]))
    print(line, flush=True)
    json.dump(ceil, open(CEIL, "w"))
    json.dump(out, open(f"exp3_N{NN}.json", "w"), indent=1, default=float)
