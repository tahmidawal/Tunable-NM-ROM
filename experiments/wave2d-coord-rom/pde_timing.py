"""FOM-vs-FiLM-decoder online timing for the burgers2d/wave2d testbeds.

Measures, on ONE GPU sequentially (cross-N series must never span devices):
  - FOM: full implicit rollout wall time, batch 1, f64, at N in 16..512
  - net: full-trajectory FiLM reconstruction (51 slices) at native N, and
    natively on the 512^2 grid (mesh-transfer inference), f32, per train-N
Median of 5 timed runs after 1 warmup (jit compile excluded), block_until_ready.

This is SURROGATE-INFERENCE speedup (decoder conditioned on true z) -- the
GN/EQ ROM solve speedup is a separate (Phase 3) number.

Env: PDE=burgers2d|wave2d.  argv[1] = ckpt dir (film pkls), argv[2] = out json.
"""
import json
import os
import pickle
import statistics
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PDE = os.environ["PDE"]
mod_name = {"burgers2d": "burgers2d_film", "wave2d": "wave2d_film"}[PDE]
import importlib

CKPT_DIR = sys.argv[1]
OUT = sys.argv[2]
FOM_NS = [int(s) for s in os.environ.get(
    "FOM_NS", "16,32,64,128,256,512").split(",")]
NET_NS = [int(s) for s in os.environ.get(
    "NET_NS", "16,32,64,128,256").split(",")]
REPS = 5

print(f"jax_backend={jax.default_backend()}  PDE={PDE}", flush=True)


def timeit(fn):
    fn()                                   # warmup / compile
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


os.environ["N"] = "16"                     # module-level config; rollout takes n
base = importlib.import_module(mod_name)
cx, cy, w, a, p4, z = base.sample_params(seed=base.SEED,
                                         m=base.N_TRAIN + base.N_VAL)
iv = base.N_TRAIN                          # first val trajectory
z_val = np.asarray(z[iv], dtype=np.float32)
n_times = base.NUM_STEPS + 1
taus32 = jnp.asarray(np.arange(n_times) / base.NUM_STEPS, dtype=jnp.float32)

results = {"pde": PDE, "reps": REPS, "batch": 1,
           "fom_s": {}, "net_native_s": {}, "net_512_s": {},
           "speedup_native": {}, "speedup_vs_fom512": {}}

def build_ic(n, i):
    if hasattr(base, "blob_ic"):                     # burgers2d
        return base.blob_ic(n, cx[i], cy[i], w[i], a[i])
    x = np.linspace(0.0, 1.0, n)                     # wave2d: inline masked
    X, Y = np.meshgrid(x, x, indexing="ij")          # Gaussian (refgen recipe)
    mask = np.asarray(base.boundary_mask(n))
    return (a[i] * np.exp(-((X - cx[i]) ** 2 + (Y - cy[i]) ** 2)
                          / (2 * w[i] ** 2)) * mask).reshape(-1)


# ---------------- FOM timing ----------------
for n in FOM_NS:
    rollout, _ = base.make_rollout(n)
    U0 = jnp.asarray(build_ic(n, iv)[None])
    pb = jnp.asarray(p4[iv][None])
    t = timeit(lambda: jax.block_until_ready(rollout(U0, pb)))
    results["fom_s"][n] = t
    print(f"FOM N={n:4d}: {t*1e3:9.2f} ms/traj", flush=True)

# ---------------- net timing ----------------
def make_coords(n):
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1),
                       dtype=jnp.float32)

coords512 = make_coords(512)

for n in NET_NS:
    os.environ["N"] = str(n)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"m{n}", os.path.join(HERE, mod_name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"m{n}"] = mod
    spec.loader.exec_module(mod)
    ck = os.path.join(CKPT_DIR, f"{PDE}_film_N{n}.pkl")
    with open(ck, "rb") as f:
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    zj = jnp.asarray(z_val)

    apply = jax.jit(lambda tau, pts: mod.film_apply(params, zj, tau, pts))

    def traj(pts):
        outs = [apply(taus32[k], pts) for k in range(n_times)]
        return jax.block_until_ready(outs[-1])

    cn = make_coords(n)
    tn = timeit(lambda: traj(cn))
    t5 = timeit(lambda: traj(coords512))
    results["net_native_s"][n] = tn
    results["net_512_s"][n] = t5
    results["speedup_native"][n] = results["fom_s"].get(n, float("nan")) / tn
    results["speedup_vs_fom512"][n] = (
        results["fom_s"].get(512, float("nan")) / t5)
    print(f"net trainN={n:3d}: native {tn*1e3:8.2f} ms/traj "
          f"(speedup {results['speedup_native'][n]:7.2f}x) | on 512^2 "
          f"{t5*1e3:8.2f} ms/traj (vs FOM-512 "
          f"{results['speedup_vs_fom512'][n]:7.2f}x)", flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print(f"wrote {OUT}", flush=True)
