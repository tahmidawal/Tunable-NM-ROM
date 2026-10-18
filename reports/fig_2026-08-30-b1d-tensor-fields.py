"""Field plots for the 1D Burgers tensor result: FOM truth vs tensor ROM vs
full-grid oracle ROM vs NNLS-32 ROM at several times, pointwise errors, the
per-step error curves, and |tensor - oracle| on its own scale.

Uses the committed N=512 checkpoint + node set on exp/2026-08-29-b1d-tensor and
the same test seed as the jobs (fc.gen_test).  Local, sub-minute:

  cd reports && source /etc/profile.d/jax-mem.sh && \
  JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun /home/tahmid/Dev/.venv/bin/python \
      fig_2026-08-30-b1d-tensor-fields.py            # env N=512 (default)

Writes figs/b1d-tensor-fields-n<N>.png, figs/b1d-tensor-errors-n<N>.png and
figs/b1d-tensor-fields-n<N>.json (the numbers in the captions).
"""
import json, os, sys, time
WT = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-29-b1d-tensor/experiments/separable-decoder"
sys.path.insert(0, WT)
os.environ.setdefault("JAX_ENABLE_X64", "1")
N = int(os.environ.get("N", "512"))
os.environ["N"] = str(N)
CKPT = os.environ.get("CKPT", f"{WT}/runs/b1dqf/b1ds_n{N}/out/sep_b1d_scale_n{N}.pkl")
NODES = os.environ.get("NODES", f"{WT}/runs/b1dqf/b1ds_n{N}/out/sep_b1d_scale_n{N}_nodes.npz")
os.environ.setdefault("CKPT_CACHE", CKPT)
os.environ.setdefault("NODES_NPZ", NODES)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
# ARMS: which ROM arms to draw (default all three); TAG: filename suffix so a
# reduced-arm rendering (e.g. ARMS=oracle,tensor TAG=-2arms) does not overwrite
# the full one.  The numbers computed are identical either way.
ARMS = tuple(a for a in os.environ.get("ARMS", "oracle,tensor,NNLS-32").split(",") if a)
TAG = os.environ.get("TAG", "")

import numpy as np
import jax, jax.numpy as jnp
import b1d_common as b1
import b1d_fast_common as fc
import b1d_tensor_common as tc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPT = dict(solver="gj", onepass=True, hoist=True, nocond=True, lean=True,
           nodot=True, unroll=1, scan_unroll=5, ic_unroll=1)
TIMES = [0, 10, 25, 50]

print(f"jax_backend={jax.default_backend()} N={N}")
su = fc.Setup(CKPT, N)
interior, n_i, dx = su.interior, su.n_i, su.dx
U_test, nu_test = fc.gen_test(N)
T = b1.NUM_STEPS + 1
x_int = np.asarray(su.coords_int)[:, 0]

Phi_np, G_np = np.asarray(su.Phi_j), np.asarray(su.G_int)
Q = tc.symmetrize(tc.build_T(Phi_np, G_np, dx, chunk=256))
arms_xw = fc.load_arms(NODES)
X_v, w_v = arms_xw["base_tight"]
ic_fast = fc.make_ic_fast(su, OPT)
ops = {
    "oracle": fc.make_device_fast(su, None, None, OPT),
    "tensor": fc.make_device_fast(su, None, None, OPT, Q=Q),
    "NNLS-32": fc.make_device_fast(su, X_v, w_v, OPT),
}

def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

fields, curves, summary = {}, {}, {}
for ti in range(fc.N_TEST):
    u0 = U_test[ti, 0]
    tol_abs = fc.STEP_TOL * float(np.sqrt(np.mean(u0[interior] ** 2))) * float(np.sqrt(n_i))
    z0, _ = ic_fast(jnp.asarray(u0[interior]))
    for arm, op in ops.items():
        R = op["rollout"](z0, float(nu_test[ti]), tol_abs, fc.GN_BUDGET)
        Z = np.asarray(R[0])
        F = np.asarray(su.decode_all(jnp.concatenate([np.asarray(z0)[None], Z], axis=0)))
        fields[(ti, arm)] = F
        curves[(ti, arm)] = [rel(F[t], U_test[ti, t][interior]) for t in range(T)]
        summary.setdefault(ti, {})[arm] = float(np.mean(curves[(ti, arm)]))
    summary[ti]["nu"] = float(nu_test[ti])
    summary[ti]["tensor_vs_oracle_field_max"] = float(max(
        np.max(np.abs(fields[(ti, "tensor")][t] - fields[(ti, "oracle")][t]))
        / np.max(np.abs(fields[(ti, "oracle")][t])) for t in range(T)))
    print(f"traj {ti}: nu={nu_test[ti]:.4f} " + " ".join(
        f"{a}={summary[ti][a]:.3e}" for a in ops) +
        f" |tensor-oracle|max={summary[ti]['tensor_vs_oracle_field_max']:.1e}")

errs_t = np.array([summary[ti]["tensor"] for ti in range(fc.N_TEST)])
pick = [int(np.argsort(errs_t)[len(errs_t) // 2]), int(np.argmax(errs_t))]
labels = ["median-error trajectory", "worst-error trajectory"]
colors = {"truth": "#222222", "oracle": "#2f5f8f", "tensor": "#c6531c", "NNLS-32": "#237a58"}
styles = {"truth": dict(lw=2.2, ls="-"), "oracle": dict(lw=1.6, ls="--"),
          "tensor": dict(lw=1.4, ls=":"), "NNLS-32": dict(lw=1.2, ls="-.")}

# ---- figure 1: fields at four times, two trajectories ----------------------
fig, axes = plt.subplots(2, len(TIMES), figsize=(4.2 * len(TIMES), 6.4), sharex=True)
for r, ti in enumerate(pick):
    for c, t in enumerate(TIMES):
        ax = axes[r, c]
        ax.plot(x_int, U_test[ti, t][interior], color=colors["truth"], label="FOM truth", **styles["truth"])
        for arm in ARMS:
            ax.plot(x_int, fields[(ti, arm)][t], color=colors[arm],
                    label={"oracle": "full-grid oracle ROM", "tensor": "tensor ROM (0 sample points)",
                           "NNLS-32": "NNLS-32 sampled ROM"}[arm], **styles[arm])
        ax.set_title(f"{labels[r]} (traj {ti}, ν={summary[ti]['nu']:.3f}) — step {t}", fontsize=9)
        ax.grid(alpha=0.25)
        if r == 1:
            ax.set_xlabel("x")
        if c == 0:
            ax.set_ylabel("u")
axes[0, 0].legend(fontsize=8, loc="upper right")
fig.suptitle(f"1D Burgers N={N}: FOM truth vs ROM arms (tensor ≡ oracle to the eye; per-trajectory mean rel-L2 in the JSON)", fontsize=11)
fig.tight_layout()
f1 = os.path.join(OUT, f"b1d-tensor-fields-n{N}{TAG}.png")
fig.savefig(f1, dpi=150); plt.close(fig)

# ---- figure 2: pointwise errors, |tensor-oracle|, and error curves -----------
fig, axes = plt.subplots(3, len(TIMES) - 1, figsize=(4.2 * (len(TIMES) - 1), 8.6))
ti = pick[1]
for c, t in enumerate(TIMES[1:]):
    ax = axes[0, c]
    for arm in ARMS:
        ax.plot(x_int, np.abs(fields[(ti, arm)][t] - U_test[ti, t][interior]), color=colors[arm], label=arm, **styles[arm])
    ax.set_title(f"|ROM − truth|, traj {ti}, step {t}", fontsize=9); ax.grid(alpha=0.25)
    ax = axes[1, c]
    ax.plot(x_int, np.abs(fields[(ti, "tensor")][t] - fields[(ti, "oracle")][t]), color=colors["tensor"], lw=1.4)
    ax.set_title(f"|tensor − oracle|, step {t}  (note the scale)", fontsize=9); ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[0, 0].legend(fontsize=8)
for c, ti2 in enumerate(pick + [None]):
    ax = axes[2, c]
    if ti2 is None:
        for arm in ARMS:
            m = np.mean([curves[(k, arm)] for k in range(fc.N_TEST)], axis=0)
            ax.semilogy(range(T), m, color=colors[arm], label=arm, **styles[arm])
        ax.set_title("rel-L2 error vs step, mean over 8 trajectories", fontsize=9)
    else:
        for arm in ARMS:
            ax.semilogy(range(T), curves[(ti2, arm)], color=colors[arm], label=arm, **styles[arm])
        ax.set_title(f"rel-L2 error vs step, traj {ti2}", fontsize=9)
    ax.set_xlabel("time step"); ax.grid(alpha=0.25, which="both")
axes[2, 0].legend(fontsize=8)
fig.suptitle(f"1D Burgers N={N}: pointwise errors and per-step error curves (tensor and oracle overlap; NNLS-32 differs)", fontsize=11)
fig.tight_layout()
f2 = os.path.join(OUT, f"b1d-tensor-errors-n{N}{TAG}.png")
fig.savefig(f2, dpi=150); plt.close(fig)

meta = dict(N=N, ckpt=CKPT, nodes=NODES, backend=jax.default_backend(), times=TIMES,
            picked=dict(median=pick[0], worst=pick[1]), per_traj=summary,
            mean_over_traj={a: float(np.mean([summary[k][a] for k in range(fc.N_TEST)])) for a in ops},
            figures=[f1, f2])
json.dump(meta, open(os.path.join(OUT, f"b1d-tensor-fields-n{N}{TAG}.json"), "w"), indent=1)
print(json.dumps(meta["mean_over_traj"]), "\nwrote", f1, f2)
