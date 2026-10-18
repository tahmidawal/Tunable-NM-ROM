"""Plots of the two 2026-10-01 off-mesh quadrature lanes, read straight from their run JSONs.

Run: /home/tahmid/Dev/.venv/bin/python make_plots.py
Sources (read-only):
  3D: worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/results/select-{val,ho}.json
  2D: worktrees/2026-10-01-quadrature-study/experiments/quadrature-study/checks/t{256,1024,4096}-summary.json
Writes figs/*.png and captions.json (every number quoted in a caption is computed here).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WT = "/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees"
D3 = f"{WT}/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/results"
D2 = f"{WT}/2026-10-01-quadrature-study/experiments/quadrature-study/checks"
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)

BLUE, VERM, AMBER, PINK, GREY = "#0072B2", "#D55E00", "#E69F00", "#CC79A7", "#5B5B5B"
INK, MUTED = "#1a2320", "#5a6862"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e3e8e4", "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2.2, "lines.markersize": 7})
CAP = {}
J = lambda p: json.load(open(p))
MESH3 = ["65", "129", "257"]; LAB3 = ["64³", "128³", "256³"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name + ".png"), dpi=170); plt.close(fig)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", color=color, fontsize=10.5,
                va="center", fontweight="bold")


# ======================================================================== 3D
sv, sh = J(f"{D3}/select-val.json"), J(f"{D3}/select-ho.json")

# --- 3D-1: error vs mesh (the headline)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, Rp in zip(axs, ["512", "256"]):
    ten = [100 * sh["meshes"][n]["R"][Rp]["tensor_row"]["refined_worst"] for n in MESH3]
    sel_name = sh["meshes"]["65"]["R"][Rp]["selected"]
    off = [100 * sh["meshes"][n]["R"][Rp]["selected_row"]["refined_worst"] for n in MESH3]
    fom = [100 * min(v["refined"] for v in sh["meshes"][n]["fom"].values()) for n in MESH3]
    x = np.arange(3)
    ax.plot(x, ten, "o-", c=VERM); label_end(ax, 2, ten[-1], "old: tensor", VERM, 6)
    ax.plot(x, off, "o-", c=BLUE); label_end(ax, 2, off[-1], f"new: off-mesh", BLUE, -6)
    ax.plot(x, fom, "s--", c=GREY, lw=1.6); label_end(ax, 2, fom[-1], "best full solver", GREY, -20)
    for i in range(3):
        ax.annotate(f"{ten[i]:.1f}%", (x[i], ten[i]), xytext=(0, 8), textcoords="offset points", ha="center", color=VERM, fontsize=9)
        ax.annotate(f"{off[i]:.2f}%", (x[i], off[i]), xytext=(0, -15), textcoords="offset points", ha="center", color=BLUE, fontsize=9)
    ax.set_xticks(x, LAB3); ax.set_xlim(-0.3, 2.9); ax.set_ylim(0, None)
    ax.set_title(f"bank width R′ = {Rp}  (new rule: {sel_name.split('_')[0]})", fontsize=11)
    ax.set_xlabel("mesh")
    CAP[f"3d_err_{Rp}"] = dict(tensor=ten, off=off, fom=fom, rule=sel_name)
axs[0].set_ylabel("worst error over 32 test cases (%)")
fig.suptitle("Burgers 3D: error of the reduced model as the mesh is refined", fontsize=12.5, x=0.02, ha="left")
save(fig, "3d_error_vs_mesh")

# --- 3D-2: quadrature error vs number of points (validation certification states), R'=512, three meshes
fam = [("lattice (CBC)", "lat", AMBER, "o-"), ("Gauss", "gl", BLUE, "s-")]
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, n, lab in zip(axs, MESH3, LAB3):
    rows = sv["meshes"][n]["R"]["512"]["rows"]
    for name, pre, c, st in fam:
        pts = sorted((r["m"], r["rho_cont"]["worst"]) for k, r in rows.items()
                     if k.startswith(pre) and k[len(pre)].isdigit() and r["m"] and r["m"] >= 4096)
        ax.loglog(*zip(*pts), st, c=c, label=name)
    for k, c, mk, nm in [("sob16384_R512", GREY, "D", "Sobol"), ("smol8_R512", PINK, "v", "Smolyak (control)"),
                         ("lat256_R512", GREY, "x", "lattice 256 (control)")]:
        r = rows[k]; ax.loglog([r["m"]], [r["rho_cont"]["worst"]], mk, c=c, ms=8, label=nm)
    ax.axhline(0.116, ls="--", c=INK, lw=1); ax.text(3e2, 0.14, "acceptance bar", fontsize=8.5, color=INK)
    ax.set_title(f"mesh {lab}", fontsize=11); ax.set_xlabel("number of quadrature points m")
axs[0].set_ylabel("worst quadrature error ρ"); axs[0].legend(fontsize=8.5, loc="lower left")
fig.suptitle("Burgers 3D: how accurately each rule computes the nonlinear term (lower is better, R′ = 512)",
             fontsize=12.5, x=0.02, ha="left")
save(fig, "3d_rule_ladder")

# --- 3D-3: time per query vs mesh, with the full solver used for the same-grid speedup
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, Rp in zip(axs, ["512", "256"]):
    m = sh["meshes"]; x = np.arange(3)
    ten = [m[n]["R"][Rp]["tensor_row"]["ms"] for n in MESH3]
    off = [m[n]["R"][Rp]["selected_row"]["ms"] for n in MESH3]
    fomk = [m[n]["R"][Rp]["selected_row"]["fom_same_rule"] for n in MESH3]
    fom = [m[n]["fom"][k]["ms"] for n, k in zip(MESH3, fomk)]
    ax.semilogy(x, fom, "s--", c=GREY, lw=1.6); label_end(ax, 2, fom[-1], "full solver\n(as accurate, same mesh)", GREY)
    ax.semilogy(x, ten, "o-", c=VERM); label_end(ax, 2, ten[-1], "old: tensor", VERM, 7)
    ax.semilogy(x, off, "o-", c=BLUE); label_end(ax, 2, off[-1], "new: off-mesh", BLUE, -7)
    ax.set_xticks(x, LAB3); ax.set_xlim(-0.3, 3.2); ax.set_title(f"bank width R′ = {Rp}", fontsize=11); ax.set_xlabel("mesh")
    CAP[f"3d_ms_{Rp}"] = dict(tensor=ten, off=off, fom=fom)
axs[0].set_ylabel("milliseconds per query (log scale)")
fig.suptitle("Burgers 3D: cost per query (lower is better)", fontsize=12.5, x=0.02, ha="left")
save(fig, "3d_time_vs_mesh")

# --- 3D-4: memory of the advection data
fig, ax = plt.subplots(figsize=(7.5, 3.6))
items = []
for Rp in ["512", "256"]:
    rows = sh["meshes"]["65"]["R"][Rp]
    sel = rows["selected"]
    items += [(f"tensor\nR′={Rp}", rows["tensor_row"]["bytes"], VERM),
              (f"{sel.split('_')[0]}\nR′={Rp}", rows["selected_row"]["bytes"], BLUE)]
y = np.arange(len(items))[::-1]
for yi, (lab, b, c) in zip(y, items):
    ax.barh(yi, b / 1e9, color=c, height=0.6)
    ax.text(b / 1e9, yi, f"  {b / 1e9:.2f} GB" if b > 1e9 else f"  {b / 1e6:.0f} MB", va="center", fontsize=10, color=INK)
ax.set_yticks(y, [i[0] for i in items]); ax.set_xlabel("memory for the nonlinear term (GB)"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(i[1] for i in items) / 1e9 * 1.25)
ax.set_title("Burgers 3D: memory (red = old tensor, blue = new off-mesh rule)", fontsize=12, loc="left")
save(fig, "3d_memory")

# ======================================================================== 2D
T = {n: J(f"{D2}/t{n}-summary.json") for n in (256, 1024, 4096)}
x = np.arange(3); LAB2 = ["256²", "1024²", "4096²"]

# --- 2D-1: error vs mesh, accurate and fast settings (test cohort, space+time reference)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, st, new in zip(axs, ["acc", "fast"], ["gauss96", "fib1597"]):
    def w(name):
        return [100 * T[n]["arms"][st][name]["test64"]["ref_ST_evolved"]["worst"] for n in (256, 1024, 4096)]
    def ncase(name):
        return [T[n]["arms"][st][name]["test64"]["ref_ST_evolved"]["n"] for n in (256, 1024, 4096)]
    d, l, o = w("dense"), w("lat64"), w(new)
    ax.plot(x, d, "o-", c=VERM); label_end(ax, 2, d[-1], "dense mesh solve", VERM, 8)
    ax.plot(x, l, "^--", c=AMBER, lw=1.8); label_end(ax, 2, l[-1], "old: 63² mesh lattice", AMBER, 20)
    ax.plot(x, o, "o-", c=BLUE); label_end(ax, 2, o[-1], f"new: {new.replace('gauss', 'Gauss ').replace('fib', 'Fibonacci ')}", BLUE, -8)
    nd = ncase("dense")
    if nd[-1] != nd[0]:
        ax.annotate(f"at 4096² the dense solve\nran on only {nd[-1]} cases", (2, d[-1]), xytext=(-150, -28), textcoords="offset points",
                    fontsize=8.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xticks(x, LAB2); ax.set_xlim(-0.3, 3.3); ax.set_ylim(0, max(d + l + o) * 1.12); ax.set_xlabel("mesh")
    ax.set_title({"acc": "accurate setting", "fast": "fast setting"}[st], fontsize=11)
    CAP[f"2d_err_{st}"] = dict(dense=d, lat=l, new=o, n_dense=nd)
axs[0].set_ylabel("worst error over test cases (%)"); axs[1].set_ylabel("worst error over test cases (%)")
fig.suptitle("Burgers 2D: error of the reduced model as the mesh is refined (64 test cases)", fontsize=12.5, x=0.02, ha="left")
save(fig, "2d_error_vs_mesh")

# --- 2D-2: solve time vs mesh (same H200 job, cross-mesh panel inside t4096)
sm = T[4096]["timing"]["solve_ms"]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, st, rules in zip(axs, ["acc", "fast"], [["gauss64", "gauss96", "gauss128", "fib17711"],
                                               ["fib1597", "gauss128", "fib17711"]]):
    for r in rules:
        ys = [sm[f"{st}|{n}|{r}"] for n in (256, 1024, 4096)]
        c = BLUE if r.startswith("gauss") else AMBER
        ax.plot(x, ys, "o-", c=c, lw=1.8, alpha=0.9); label_end(ax, 2, ys[-1], r.replace("gauss", "Gauss ").replace("fib", "Fib "), c)
    ys = [sm[f"{st}|{n}|lat64"] for n in (256, 1024, 4096)]
    ax.plot(x, ys, "^--", c=VERM, lw=1.8); label_end(ax, 2, ys[-1], "old: 63² lattice", VERM)
    ax.set_xticks(x, LAB2); ax.set_xlim(-0.3, 3.4); ax.set_ylim(0, None); ax.set_xlabel("mesh")
    ax.set_title({"acc": "accurate setting", "fast": "fast setting"}[st], fontsize=11)
axs[0].set_ylabel("solve time per query (ms)")
fig.suptitle("Burgers 2D: solve time stays flat as the mesh grows 256× in nodes", fontsize=12.5, x=0.02, ha="left")
save(fig, "2d_solve_time")

# --- 2D-3: quadrature error vs points at 1024², accurate setting (test states)
rr = T[1024]["rho"]["acc"]["rules"]
fig, ax = plt.subplots(figsize=(7.5, 4.3))
for pre, c, st, nm in [("gauss", BLUE, "s-", "Gauss"), ("fib", AMBER, "o-", "Fibonacci lattice"),
                       ("sobol", GREY, "D--", "Sobol"), ("smolyak", PINK, "v-", "Smolyak (control)")]:
    pts = sorted((r["m"], r["cont"]["max"]) for k, r in rr.items() if k.startswith(pre) and r.get("m"))
    if pts:
        ax.loglog(*zip(*pts), st, c=c, label=nm, lw=1.8)
lat = rr["lat64"]
ax.loglog([lat["m"]], [lat["mesh"]["max"]], "^", c=VERM, ms=10, label="old: 63² mesh lattice (vs its mesh target)")
ax.axhline(0.116, ls="--", c=INK, lw=1); ax.text(2e2, 0.14, "acceptance bar", fontsize=8.5)
ax.set_xlabel("number of quadrature points m"); ax.set_ylabel("worst quadrature error ρ")
ax.set_title("Burgers 2D at 1024², accurate setting: quadrature error by rule", fontsize=12, loc="left")
ax.legend(fontsize=8.5, loc="lower left")
save(fig, "2d_rule_ladder")

json.dump(CAP, open(os.path.join(HERE, "captions.json"), "w"), indent=1)
print(sorted(os.listdir(FIG)))
