"""Decoder-architecture comparison on Burgers 2D: bank floor / head fit (training logs), rho
ladders per rule for each decoder, end-to-end rollouts with Gauss / Fibonacci vs dense.
usage: python scripts/report_decoders.py burgers rff siren rbf spectral pod_cubic pod_linear"""
import sys, os, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
pde = sys.argv[1]
archs = sys.argv[2:]
LABEL = {"rff": "RFF coordinate MLP (paper)", "siren": "SIREN coordinate MLP", "rbf": "Gaussian RBF bank",
         "spectral": "fixed sine bank (no training)", "pod_cubic": "POD modes + cubic interp.", "pod_linear": "POD modes + bilinear interp."}
COL = {"rff": "#1f77b4", "siren": "#d62728", "rbf": "#2ca02c", "spectral": "#9467bd", "pod_cubic": "#ff7f0e", "pod_linear": "#8c564b"}
md = [f"# Decoder architectures with off-mesh quadrature ({pde} 2D)\n",
      "Every decoder is a bank G(x) (R = 128) with the same latent head (k = 32) and nested corrections; only the "
      "spatial representation changes. `partial decoding` = evaluating G and its gradient at the quadrature points.\n"]

def train_stats(arch):
    f = f"data/train_{pde}_{arch}.log" if arch != "rff" else f"data/train_{pde}.log"
    if not os.path.exists(f):
        return None
    t = open(f).read()
    m1 = re.search(r"projection floor: median ([\d.]+)%\s+worst ([\d.]+)%", t)
    m2 = re.search(r"head fit error median ([\d.]+)% worst ([\d.]+)%", t)
    return (m1.group(1), m1.group(2), m2.group(1) if m2 else "-", m2.group(2) if m2 else "-")

def quad_rows(arch):
    f = f"{OUT}/{pde}_quad_study_{arch}.json" if arch != "rff" else f"{OUT}/{pde}_quad_study_trainstates.json"
    return json.load(open(f)) if os.path.exists(f) else []

def roll_rows(arch):
    f = f"{OUT}/{pde}_rollout_study_{arch}.json" if arch != "rff" else f"{OUT}/{pde}_rollout_study_k32.json"
    return json.load(open(f)) if os.path.exists(f) else []

md.append("## Representation quality on the training mesh (256²)\n")
md.append("| decoder | bank floor median % | bank floor worst % | head fit median % | head fit worst % |\n|---|---|---|---|---|")
for a in archs:
    s = train_stats(a)
    if s:
        md.append(f"| {LABEL[a]} | {s[0]} | {s[1]} | {s[2]} | {s[3]} |")

md.append("\n## Quadrature error of the tested advection term (worst / median rho over 96 states, continuum target, accurate M)\n")
md.append("Reference check = agreement of 200² and 300² Gauss references (a large value means the integrand is not smooth enough for either to be converged).\n")
rules = [("gauss_tensor", 32), ("gauss_tensor", 48), ("gauss_tensor", 64), ("fibonacci", 17), ("fibonacci", 19), ("fibonacci", 20), ("sobol", 4096), ("smolyak_cc", 10)]
md.append("| decoder | " + " | ".join(f"{r} {p}" for r, p in rules) + " |\n|---|" + "---|" * len(rules))
for a in archs:
    rows = {(r["rule"], r["param"]): r for r in quad_rows(a) if r["target"] == "continuum" and r["form"] == "point"}
    cells = []
    for key in rules:
        r = rows.get(key)
        cells.append(f"{r['rho_acc_max']:.1e} / {r['rho_acc_med']:.1e}" if r else "-")
    md.append(f"| {LABEL[a]} | " + " | ".join(cells) + " |")
# reference checks from logs
def ref_check(a):
    f = f"{OUT}/{pde}_quad_study_{a}.log" if a != "rff" else f"{OUT}/{pde}_quad_study_trainstates.log"
    if not os.path.exists(f):
        return None
    m = re.search(r"GL200 vs GL300 rho max ([\d.e+-]+)", open(f).read())
    return m.group(1) if m else None
md.append("\nReference checks (GL200 vs GL300 worst rho): " + "; ".join(f"{LABEL[a]}: {ref_check(a)}" for a in archs if ref_check(a)) + ".\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ai, fam in enumerate(["gauss_tensor", "fibonacci"]):
    ax = axes[ai]
    for a in archs:
        pts = sorted([(r["m"], r["rho_acc_max"]) for r in quad_rows(a) if r["target"] == "continuum" and r["form"] == "point" and r["rule"] == fam])
        med = sorted([(r["m"], r["rho_acc_med"]) for r in quad_rows(a) if r["target"] == "continuum" and r["form"] == "point" and r["rule"] == fam])
        if pts:
            ax.loglog(*zip(*pts), "-o", color=COL[a], ms=4, label=f"{LABEL[a]} (worst)")
            ax.loglog(*zip(*med), "--", color=COL[a], lw=1, label=f"{LABEL[a]} (median)")
    ax.axhline(0.116, color="k", ls=":", lw=1)
    ax.set_xlabel("quadrature points m"); ax.set_ylabel("rho (accurate M)"); ax.set_title(f"{pde}: {fam} by decoder")
    ax.grid(True, which="both", alpha=0.3)
axes[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig7_decoders_rho.png", dpi=130); plt.close(fig)

md.append("\n## End-to-end rollouts (worst error vs refined reference %, and distance from the dense rollout of the same decoder %)\n")
md.append("| decoder | N | q | dense | Gauss 32² | Fibonacci 4181 | Sobol 4096 | Smolyak CC 8 |\n|---|---|---|---|---|---|---|---|")
for a in archs:
    rr = roll_rows(a)
    for N in sorted({r["N"] for r in rr}):
        for q in sorted({r["q"] for r in rr}):
            def cell(rule, par):
                r = [x for x in rr if x["N"] == N and x["q"] == q and x["rule"] == rule and x["param"] == par and x["form"] == "point"]
                if not r: return "-"
                r = r[0]; hr = r.get("hr_err_worst", float("nan"))
                return f"{r['worst_refined']*100:.2f} ({hr*100:.2f})"
            md.append(f"| {LABEL[a]} | {N} | {q} | {cell('dense',0)} | {cell('gauss_tensor',32)} | {cell('fibonacci',19)} | {cell('sobol',4096)} | {cell('smolyak_cc',8)} |")
open(f"{OUT}/DECODERS.md", "w").write("\n".join(md))
print("wrote", f"{OUT}/DECODERS.md")
