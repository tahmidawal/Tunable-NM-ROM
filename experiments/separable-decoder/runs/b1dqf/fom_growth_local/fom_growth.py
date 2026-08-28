"""Where does the 1D tridiagonal tolerance-Newton FOM's cost start to grow
with N?  Timing-only (no truth, one blob IC, nu=0.05), same
make_fom_tol_rollout instrument as sep_b1d_scale.py, 2 burn + 5 timed reps,
all reps persisted.  Run: jaxrun python fom_growth.py -> fom_growth.json"""
import json, os, time, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import b1d_common as b1
rows = []
for N in (512, 4096, 16384, 65536, 262144, 1048576):
    roll = b1.make_fom_tol_rollout(N)
    u0 = jnp.asarray(b1.blob_ic_1d(N, 0.5, 0.1, 1.0))
    for _ in range(2):
        tr, tot, worst = roll(u0, 0.05, 1e-3); tr.block_until_ready()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); tr, tot, worst = roll(u0, 0.05, 1e-3); tr.block_until_ready()
        ts.append((time.perf_counter() - t0) * 1e3)
    rows.append(dict(N=N, points=N - 2, ms_median=float(np.median(ts)), ms_raw=ts,
                     newton_per_step=int(tot) / 50, worst_rel_res=float(worst)))
    print(f"{N:>9} {N-2:>9} {np.median(ts):>8.2f} ms  newton/step {int(tot)/50:.1f}", flush=True)
dev = jax.devices()[0]
json.dump(dict(gpu=getattr(dev, "device_kind", str(dev)), backend=dev.platform, ntol=1e-3,
               nu=0.05, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fom_growth.json"), "w"), indent=1)
