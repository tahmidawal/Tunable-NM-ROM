"""PROJECTION plots for steps 1-3 of the full experiment plan (2026-10-17).

These are NOT results. Curves are illustrative shapes; solid filled markers are measured anchors taken
from existing lane reports (sources named next to each constant). Every figure is stamped PROJECTION.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "figure.dpi": 130})
BLUE, ORANGE, GREEN, PURPLE, GREY, RED = "#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#6b7280", "#dc2626"


def stamp(fig, note="PROJECTION: illustrative, not data"):
    fig.text(.5, .5, "PROJECTION", fontsize=64, color="#9ca3af", alpha=.13, ha="center", va="center",
             rotation=20, weight="bold", zorder=0)
    fig.text(.995, .005, note, fontsize=8, color=GREY, ha="right", va="bottom")


def save(fig, name):
    fig.tight_layout()
    stamp(fig)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- measured anchors (sources)
# 2D solve ms, held-out job t4096 (2D quadrature-study report, section T5): fast Fib1597 (r=128), acc Gauss96 (r=384)
SOLVE_128 = [13.3, 13.0, 13.4]
SOLVE_384 = [55.9, 55.5, 55.4]
MESH2 = [256, 1024, 4096]
# 2D median error vs E0 ST, LSPG-CN-R at dt0, mesh 1024^2 (combined report, accurate-equal-cost arms)
ERR_R = {128: 0.45, 256: 0.209, 384: 0.198, 512: 0.199}
# 2D common-baseline cost of those arms (combined report H2b)
COST_R = {128: 0.239, 256: 0.568, 384: 1.56, 512: 5.36}
# Measured 2D matched speed-up at 1024^2 vs CN/BDF2-in-time FOM (combined H5): selected scheme 13.6x
SPEED_1024 = 13.6

# ================================================================ STEP 1
# 1. cost vs mesh, lines per r
fig, ax = plt.subplots(figsize=(6.4, 4))
proj256 = [30, 30, 30]  # interpolated, projected
for r, ys, c, meas in [(128, SOLVE_128, GREEN, True), (256, proj256, BLUE, False), (384, SOLVE_384, PURPLE, True)]:
    ax.plot(MESH2, ys, "-o" if meas else "--o", color=c, mfc=c if meas else "white", label=f"k = {r}" + ("" if meas else " (projected)"))
ax.set_xscale("log", base=2); ax.set_xticks(MESH2); ax.set_xticklabels(["256²", "1024²", "4096²"])
ax.set_ylim(0, 70); ax.set_xlabel("mesh"); ax.set_ylabel("solve time per query (ms)")
ax.set_title("1 · Solve cost stays flat as the mesh grows")
ax.legend(frameon=False); save(fig, "p1_cost_vs_mesh.png")

# 2. cost vs r, per component (log-log), illustrative slopes 1,1,2,3
r = np.array([32, 64, 128, 256, 384, 512])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
comps = [("nonlinear term (∝ k)", 0.05, 1, BLUE), ("Jacobian (∝ k)", 0.12, 1, ORANGE),
         ("normal equations (∝ k²)", 0.0006, 2, GREEN), ("small linear solve (∝ k³)", 0.000004, 3, PURPLE)]
for name, a, p, c in comps:
    ax.loglog(r, 0.02 + a * (r / 32.0) ** p * 32 ** (p - 1) / 32 ** (p - 1), "--o", color=c, mfc="white", label=name)
ax.set_xticks(r); ax.set_xticklabels(r)
ax.set_xlabel("k (bank size)"); ax.set_ylabel("time per iteration (ms, illustrative)")
ax.set_title("2 · Cost grows with k, part by part (fixed tests and points)")
ax.legend(frameon=False, fontsize=9); save(fig, "p2_cost_vs_k.png")

# 3. k x mesh heat map: vertical contours
ks = np.array([128, 256, 384, 512]); ms = np.array([256, 1024, 4096])
cost = np.array([[COST_R[k]] * 3 for k in ks]).T
fig, ax = plt.subplots(figsize=(6.4, 3.6))
im = ax.imshow(np.log10(cost), aspect="auto", cmap="Blues", origin="lower")
for i in range(3):
    for j in range(4):
        ax.text(j, i, f"{cost[i, j]:.2f}×", ha="center", va="center", fontsize=9,
                color="white" if cost[i, j] > 1.2 else "black")
ax.set_xticks(range(4)); ax.set_xticklabels([f"k={k}" for k in ks])
ax.set_yticks(range(3)); ax.set_yticklabels(["256²", "1024²", "4096²"])
ax.grid(False); ax.set_title("3 · Cost depends on k (columns), not mesh (rows)")
fig.colorbar(im, ax=ax, label="log₁₀ relative cost"); save(fig, "p3_heatmap.png")

# 4. accuracy vs mesh per r (flat; 1024 measured)
fig, ax = plt.subplots(figsize=(6.4, 4))
for k, c in [(128, GREEN), (256, BLUE), (384, PURPLE)]:
    e = ERR_R[k]
    ax.plot(MESH2, [e * 1.02, e, e * 0.99], "--", color=c)
    ax.plot([256, 4096], [e * 1.02, e * 0.99], "o", color=c, mfc="white")
    ax.plot([1024], [e], "o", color=c, label=f"k = {k}")
ax.set_xscale("log", base=2); ax.set_xticks(MESH2); ax.set_xticklabels(["256²", "1024²", "4096²"])
ax.set_ylim(0, 0.55); ax.set_xlabel("mesh"); ax.set_ylabel("typical error (%)")
ax.set_title("4 · Accuracy stays flat across meshes (filled = measured at 1024²)")
ax.legend(frameon=False); save(fig, "p4_acc_vs_mesh.png")

# 5. accuracy vs r (measured at 1024) + cost
fig, ax = plt.subplots(figsize=(6.4, 4))
ks = list(ERR_R)
ax.plot(ks, [ERR_R[k] for k in ks], "-o", color=BLUE, label="typical error (measured, 1024²)")
ax.set_xticks(ks); ax.set_xlabel("k (bank size)"); ax.set_ylabel("typical error (%)", color=BLUE); ax.set_ylim(0, 0.5)
ax.annotate("big gain", (192, 0.33), color=GREY); ax.annotate("little or none", (400, 0.24), color=GREY)
ax.set_title("5 · Accuracy improves with k, then levels off")
ax.legend(frameon=False, loc="upper right"); save(fig, "p5_acc_vs_k.png")

# 6. memory vs k: off-mesh vs tensor (formula, m ~ projected)
kk = np.array([128, 256, 512, 768, 1024, 1536, 2048])
M = 4 * kk; m = 13824 * (kk / 512)  # assumed points grow ~ linearly with M (projection)
off = 8 * m * (2 * kk + M) / 1e9; ten = 8 * M * kk ** 2 / 1e9
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.semilogy(kk, ten, "-o", color=ORANGE, label="old exact tensor (8·M·k²)")
ax.semilogy(kk, off, "--o", color=BLUE, mfc="white", label="off-mesh tables (8·m·(2k+M)), m projected")
ax.axhline(141, ls=":", color=GREY); ax.text(140, 160, "one H200 GPU", color=GREY, fontsize=9)
ax.set_xlabel("k (bank size)"); ax.set_ylabel("memory (GB, log)")
ax.set_title("6 · Memory: off-mesh stays small, tensor explodes")
ax.legend(frameon=False, fontsize=9); save(fig, "p6_memory.png")

# ================================================================ STEP 2
# 7. speed-up vs mesh, per FOM type (bands)
fig, ax = plt.subplots(figsize=(6.6, 4.2))
x = np.array([256, 1024, 4096])
bands = [("first-order / CN-in-time simulation", [0.3, SPEED_1024, 60], [0.2, SPEED_1024, 25], [0.5, SPEED_1024, 120], ORANGE),
         ("fully second-order (central-CN)", [0.15, 4, 15], [0.1, 2, 5], [0.3, 8, 30], GREEN),
         ("spectral", [0.05, 0.5, 2], [0.02, 0.2, 0.5], [0.1, 1, 5], PURPLE)]
for name, mid, lo, hi, c in bands:
    ax.fill_between(x, lo, hi, color=c, alpha=.15)
    ax.plot(x, mid, "--o", color=c, mfc="white", label=name)
ax.plot([1024], [SPEED_1024], "o", color=ORANGE, ms=9, label=f"measured: {SPEED_1024}× at 1024²")
ax.axhline(1, color="black", lw=1); ax.text(260, 1.15, "break-even", fontsize=9)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["256²", "1024²", "4096²"])
ax.set_xlabel("mesh"); ax.set_ylabel("speed-up at equal accuracy (log)")
ax.set_title("7 · Speed-up against each kind of full simulation (2D)")
ax.legend(frameon=False, fontsize=8.5, loc="upper left"); save(fig, "p7_speedup_vs_mesh.png")

# 8. ours vs POD: error vs size
kk = np.array([32, 64, 128, 256, 384])
ours = np.array([2.5, 1.2, 0.45, 0.209, 0.198])
pod = ours * np.array([1.6, 1.8, 2.2, 2.5, 2.6])
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.semilogy(kk, ours, "--o", color=BLUE, mfc="white", label="ours (1024² measured at k ≥ 128)")
ax.semilogy([128, 256, 384], ours[2:], "o", color=BLUE)
ax.fill_between(kk, ours, pod * 1.6, color=ORANGE, alpha=.15)
ax.semilogy(kk, pod, "--s", color=ORANGE, mfc="white", label="POD + ECSW/DEIM (projected band)")
ax.set_xticks(kk); ax.set_xticklabels(kk); ax.set_xlabel("reduced size k"); ax.set_ylabel("typical error (%, log)")
ax.set_title("8 · Ours vs a standard POD reduced model")
ax.legend(frameon=False, fontsize=9); save(fig, "p8_vs_pod.png")

# 9. accuracy-cost curve
fig, ax = plt.subplots(figsize=(6.4, 4))
cks = [128, 256, 384, 512]
ax.plot([COST_R[k] for k in cks], [ERR_R[k] for k in cks], "-o", color=BLUE, label="new: second-order (measured, 1024²)")
old = {128: 1.10, 256: 0.983, 384: 0.978, 512: 0.979}  # LSPG-BE medians vs E0, combined report
ax.plot([COST_R[k] for k in cks], [old[k] for k in cks], "-s", color=GREY, mfc="white", label="old: backward Euler, same cost")
for k in cks:
    ax.annotate(f"k={k}", (COST_R[k], ERR_R[k]), textcoords="offset points", xytext=(4, -12), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("relative cost per query"); ax.set_ylabel("typical error (%)")
ax.set_title("9 · The accuracy-for-cost dial (Step 2 measured; other meshes projected)")
ax.legend(frameon=False, fontsize=9); save(fig, "p9_dial.png")

# ================================================================ STEP 3
# 10. points needed vs k, full vs reduced test set
kk = np.array([128, 256, 384, 512])
full = 9216 * kk / 384; red = full / 3
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(kk, full, "--o", color=ORANGE, mfc="white", label="all test functions (M = 4k)")
ax.plot(kk, red, "--o", color=GREEN, mfc="white", label="fewer mixed-frequency tests (~3× fewer points)")
ax.plot([384], [9216], "o", color=ORANGE, label="measured: 9,216 points at k = 384")
ax.set_xticks(kk); ax.set_xlabel("k (bank size)"); ax.set_ylabel("quadrature points needed")
ax.set_title("10 · Points needed grow with the number of tests")
ax.legend(frameon=False, fontsize=9); save(fig, "p10_points_vs_k.png")

# 11. rule families: error vs points (measured 2D 1024² worst rho for Gauss/Fib/Sobol; MC projected)
pts_g = [1024, 2304, 4096, 9216, 16384]; g = [1.6, .18, .019, .0046, .0021]
pts_f = [1597, 4181, 6765, 17711]; f = [.13, .033, .016, .0017]
pts_s = [4096, 16384, 65536]; s = [.25, .028, .006]
pts_mc = np.array([4096, 16384, 65536]); mc = 0.9 * np.sqrt(4096 / pts_mc)
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.loglog(pts_g, g, "-s", color=BLUE, label="Gauss (measured)")
ax.loglog(pts_f, f, "-o", color=ORANGE, label="Fibonacci lattice (measured)")
ax.loglog(pts_s, s, "-D", color=GREY, label="Sobol (measured)")
ax.loglog(pts_mc, mc, "--^", color=RED, mfc="white", label="Monte Carlo (projected, ∝ 1/√points)")
ax.axhline(.116, ls=":", color="black"); ax.text(1100, .14, "accuracy bar", fontsize=9)
ax.set_xlabel("quadrature points"); ax.set_ylabel("integral error ρ (worst, log)")
ax.set_title("11 · Every quadrature type, as full solves (2D, 1024²)")
ax.legend(frameon=False, fontsize=8.5); save(fig, "p11_rules.png")

# 12. error budget per k (stacked, illustrative fractions)
kk = ["k=128", "k=256", "k=384"]
time_ = np.array([.25, .14, .14]); floor = np.array([.36, .07, .03]); dyn = np.array([.17, .05, .02]); quad = np.array([.003, .002, .002])
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.bar(kk, time_, color=BLUE, label="time stepping"); ax.bar(kk, floor, bottom=time_, color=ORANGE, label="bank's own limit")
ax.bar(kk, dyn, bottom=time_ + floor, color=GREEN, label="reduced dynamics")
ax.bar(kk, quad, bottom=time_ + floor + dyn, color=PURPLE, label="quadrature + solver (tiny)")
ax.set_ylabel("error contribution (%, illustrative)"); ax.set_title("12 · Where the error comes from")
ax.legend(frameon=False, fontsize=9); save(fig, "p12_budget.png")

# 13. longer horizon
t = np.linspace(0, 1.0, 50)
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.fill_between(t, 0.2 + 0.3 * t, 0.2 + 2.5 * t ** 1.5, color=BLUE, alpha=.15, label="projected range")
ax.plot(t, 0.2 + 0.8 * t, "--", color=BLUE, label="most likely: slow, steady growth")
ax.axvline(0.25, ls=":", color=GREY); ax.text(0.26, 2.3, "current final time 0.25", fontsize=9, color=GREY)
ax.set_xlabel("simulated time"); ax.set_ylabel("error (%)"); ax.set_title("13 · Error over longer runs")
ax.legend(frameon=False, fontsize=9); save(fig, "p13_long_time.png")

print("wrote", sorted(os.listdir(OUT)))
