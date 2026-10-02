"""Figures + results/REPORT3D.md for the 3D Burgers study."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "results"; pde = "burgers3d"
COL = {"gauss_tensor": "#1f77b4", "smolyak_cc": "#d62728", "sobol": "#2ca02c", "mc": "#7f7f7f", "kuo_lattice": "#9467bd",
       "kuo_lattice_tent": "#8c564b", "korobov": "#e377c2", "cbc_lattice": "#17becf", "cbc_lattice_tent": "#ff7f0e", "mesh_lattice": "#bcbd22", "eq_nnls": "#000000", "dense": "#444444"}
MRK = {"gauss_tensor": "o", "smolyak_cc": "s", "sobol": "^", "mc": "x", "kuo_lattice": "P", "kuo_lattice_tent": "*", "korobov": "D", "cbc_lattice": "v", "cbc_lattice_tent": "<",
       "mesh_lattice": "h", "eq_nnls": "X", "dense": "_"}
md = [f"# {pde}\n"]
qf = f"{OUT}/{pde}_quad_study.json"
if os.path.exists(qf):
    rows = json.load(open(qf))
    lf = f"{OUT}/{pde}_quad_study_lattice.json"
    if os.path.exists(lf):
        rows = rows + json.load(open(lf))
    cont = [r for r in rows if r["target"] == "continuum"]
    forms = sorted({r["form"] for r in cont})
    fig, axes = plt.subplots(1, 2 * len(forms), figsize=(6 * 2 * len(forms), 4.5), squeeze=False)
    for fi, form in enumerate(forms):
        for ai, key in enumerate(["rho_acc_max", "rho_fast_max"]):
            ax = axes[0, 2 * fi + ai]
            for fam in COL:
                pts = sorted([(r["m"], r[key]) for r in cont if r["rule"] == fam and r["form"] == form])
                if pts: ax.loglog(*zip(*pts), marker=MRK[fam], color=COL[fam], label=fam, ms=5, lw=1.2)
            ax.axhline(0.116, color="k", ls=":", lw=1); ax.axhline(0.06, color="k", ls="--", lw=0.8)
            ax.set_xlabel("quadrature points m"); ax.set_ylabel("worst rho"); ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"{pde}: {'accurate' if ai == 0 else 'fast'} M, form={form}, continuum target")
    axes[0, 0].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig1_rho_continuum.png", dpi=130); plt.close(fig)
    meshes = sorted({r["N"] for r in rows if r["target"] == "mesh"})
    if meshes:
        fig, axes = plt.subplots(1, len(meshes), figsize=(5.5 * len(meshes), 4.5), squeeze=False, sharey=True)
        for ai, N in enumerate(meshes):
            ax = axes[0, ai]; sub = [r for r in rows if r["target"] == "mesh" and r["N"] == N]
            for fam in COL:
                pts = sorted([(r["m"], r["rho_acc_max"]) for r in sub if r["rule"] == fam])
                if pts: ax.loglog(*zip(*pts), marker=MRK[fam], color=COL[fam], label=fam, ms=5, lw=1.2)
            gap = [r for r in sub if r["rule"] == "continuum_gap"]
            if gap: ax.axhline(gap[0]["rho_acc_max"], color="#aaaaaa", lw=2, label="continuum-vs-upwind gap")
            ax.axhline(0.116, color="k", ls=":", lw=1); ax.set_title(f"{pde}: mesh target N={N}"); ax.set_xlabel("m"); ax.grid(True, which="both", alpha=0.3)
        axes[0, -1].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig2_rho_mesh.png", dpi=130); plt.close(fig)
    tim = [r for r in rows if r["target"] == "time"]
    md.append("## Quadrature error of the tested nonlinear term (worst rho over 96 projected training states)\n")
    md.append("| form | rule | param | m | rho max (acc M) | rho median | rho max (fast M) |\n|---|---|---|---|---|---|---|")
    for r in sorted(cont, key=lambda r: (r["form"], r["rule"], r["m"])):
        md.append(f"| {r['form']} | {r['rule']} | {r['param']} | {r['m']} | {r['rho_acc_max']:.2e} | {r['rho_acc_med']:.2e} | {r['rho_fast_max']:.2e} |")
    md.append("\nMesh target (dense upwind evaluation at N):\n\n| N | rule | param | m | rho max (acc M) | rho max (fast M) |\n|---|---|---|---|---|---|")
    for r in sorted([r for r in rows if r["target"] == "mesh"], key=lambda r: (r["N"], r["rule"], r["m"])):
        md.append(f"| {r['N']} | {r['rule']} | {r['param']} | {r['m']} | {r['rho_acc_max']:.2e} | {r['rho_fast_max']:.2e} |")
    if tim:
        md.append("\nms per residual + Jacobian evaluation:\n\n| q | rule | m | " + " | ".join(f"N={N}" for N in meshes) + " |\n|---|---|---|" + "---|" * len(meshes))
        for q in sorted({r["q"] for r in tim}):
            for lab in sorted({(r["rule"], r["param"]) for r in tim}):
                sub = {r["N"]: r for r in tim if (r["rule"], r["param"]) == lab and r["q"] == q}
                if sub: md.append(f"| {q} | {lab[0]} {lab[1]} | {'n' if lab[0]=='dense' else list(sub.values())[0]['m']} | " + " | ".join(f"{sub[N]['ms_eval']:.2f}" if N in sub else "-" for N in meshes) + " |")
rf = f"{OUT}/{pde}_rollout_study.json"
if os.path.exists(rf):
    rr = json.load(open(rf)); meshesR = sorted({r["N"] for r in rr}); qs = sorted({r["q"] for r in rr})
    fig, axes = plt.subplots(len(qs), len(meshesR), figsize=(5.5 * len(meshesR), 4.2 * len(qs)), squeeze=False)
    for qi, q in enumerate(qs):
        for ni, N in enumerate(meshesR):
            ax = axes[qi, ni]; sub = [r for r in rr if r["q"] == q and r["N"] == N]
            for r in sub:
                ax.scatter(r["ms_query"], r["worst_refined"] * 100, color=COL.get(r["rule"], "k"), marker=MRK.get(r["rule"], "o"), s=45, facecolors="none" if r["form"] == "flux" else None, label=r["rule"] + (" (flux)" if r["form"] == "flux" else ""))
                ax.annotate(str(r["param"]), (r["ms_query"], r["worst_refined"] * 100), fontsize=6, xytext=(3, 2), textcoords="offset points")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("ms per query"); ax.set_ylabel("worst rel. L2 error vs refined (%)"); ax.set_title(f"{pde}: N={N}, q={q}"); ax.grid(True, which="both", alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels(); u = dict(zip(l, h)); axes[0, 0].legend(u.values(), u.keys(), fontsize=6)
    fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig4_rollout_pareto.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, len(qs), figsize=(6 * len(qs), 4.2), squeeze=False)
    for qi, q in enumerate(qs):
        ax = axes[0, qi]
        for lab in sorted({(r["rule"], r["param"], r["form"]) for r in rr if r["q"] == q}):
            pts = sorted([(r["N"], r["ms_per_attempt"]) for r in rr if (r["rule"], r["param"], r["form"]) == lab and r["q"] == q])
            if len(pts) > 1: ax.loglog(*zip(*pts), marker=MRK.get(lab[0], "o"), color=COL.get(lab[0], "k"), ls="-" if lab[2] == "point" else "--", lw=1, ms=4, label=f"{lab[0]} {lab[1]}{' flux' if lab[2]=='flux' else ''}")
        ax.set_xlabel("mesh N (n=(N-1)^3)"); ax.set_ylabel("ms per LM attempt"); ax.set_title(f"{pde}: q={q}"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig5_ms_per_attempt_vs_N.png", dpi=130); plt.close(fig)
    md.append("\n## End-to-end reduced rollouts\n\n`vs dense` = worst distance from the dense-evaluation rollout of the same model.\n")
    md.append("| N | q | M | rule | param | form | m | err same-grid % | err refined % | FOM err refined % | vs dense % | ms/query | att/step | ms/att | exits |\n|" + "---|" * 15)
    for r in sorted(rr, key=lambda r: (r["N"], r["q"], r["kind"] != "dense", r["rule"], r["m"], r["form"])):
        md.append(f"| {r['N']} | {r['q']} | {r['M']} | {r['rule']} | {r['param']} | {r['form']} | {r['m']} | {r['worst_same_grid']*100:.3f} | {r['worst_refined']*100:.3f} | {r['fom_worst_refined']*100:.3f} | {r['hr_err_worst']*100:.3f} | {r['ms_query']:.1f} | {r['attempts_per_step']:.2f} | {r['ms_per_attempt']:.2f} | {r['exits']} |")
open(f"{OUT}/REPORT3D.md", "w").write("\n".join(md)); print("wrote", f"{OUT}/REPORT3D.md")
