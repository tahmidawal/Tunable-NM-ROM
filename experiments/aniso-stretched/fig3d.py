"""Error-time picture for the 3D family (dev cohort curves; held-out points marked). Reads only saved JSON.
Usage: fig3d.py <pod_dev.json> <bank_dev.json> <out.png>"""
import sys, json, collections
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK, MUTED = "#1a1a19", "#6b6a63"
def solver_front(files, backend, solver):
    g = collections.defaultdict(dict)
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if r["backend"] == backend and r["solver"] == solver: g[(r["p"], r["N"], r["tol"])][r["case"]] = r
    pts = sorted((np.median([x["total_ms"] for x in d.values()]), np.mean([x["l2w"] for x in d.values()])) for d in g.values() if len(d) == 16)
    front, best = [], 1e9
    for t, e in pts:
        if e < best: front.append((t, e)); best = e
    return np.array(front)
def rom_curve(f):
    d = json.load(open(f)); g = collections.defaultdict(list)
    for r in d["rows"]: g[r["rank"]].append(r)
    return np.array([(np.median([x["total_ms"] for x in v]), np.mean([x["l2w"] for x in v]), k) for k, v in sorted(g.items())])
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160); fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
series = [("FD-PCG solver, GPU (float32)", solver_front(["runs/t3/dev_gpu.jsonl"], "gpu", "fd_pcg_f32"), "o"),
          ("FD-PCG solver, one CPU core", solver_front(["runs/t3/dev_cpujax1_dense.jsonl"], "cpu_jax_1core_dense", "fd_pcg_f32"), "s"),
          ("POD-Galerkin ROM, one CPU core", rom_curve(sys.argv[1]), "D"), ("Bank + Galerkin span ROM, one CPU core", rom_curve(sys.argv[2]), "^")]
for (name, a, mk), c in zip(series, C):
    ax.plot(a[:, 0], 100 * a[:, 1], "-", color=c, lw=2, marker=mk, ms=6, mec="#fcfcfb", mew=1, label=name)
for y in (3, 1, 0.3): ax.axhline(y, color=MUTED, lw=0.6, ls=":"); ax.text(ax.get_xlim()[0] if False else 0.021, y * 1.06, f"{y:g} % target", color=MUTED, fontsize=7.5)
pod = series[2][1]
for t, e, k in pod:
    if int(k) in (32, 96, 256, 512): ax.annotate(f"rank {int(k)}", (t, 100 * e), textcoords="offset points", xytext=(-6, -12), fontsize=7.5, color=INK, ha="right")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.02, 400); ax.set_ylim(0.04, 40)
ax.set_xlabel("warm time per query, ms (median over 16 dev cases)\nROM: reduced solve only; writing the field on the 103 823-node grid adds 2.5–13 ms", color=INK, fontsize=9); ax.set_ylabel("mean relative $L^2$ error, %", color=INK)
ax.set_title("3D anisotropic diffusion (anisotropy $10^{-4}$–$10^{-2}$, 6 parameters): solver vs ROM", fontsize=10, color=INK, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=8); ax.grid(True, which="major", color="#e4e3dc", lw=0.6); ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
fig.tight_layout(); fig.savefig(sys.argv[3]); print("wrote", sys.argv[3])
