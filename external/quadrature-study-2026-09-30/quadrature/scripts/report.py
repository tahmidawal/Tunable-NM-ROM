"""Figures and markdown tables from results/*.json.   usage: python scripts/report.py [pde ...]"""
import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
def pick(pde, kind):
    """Prefer tagged result files: quad -> _trainstates then plain; rollout -> _k32 then plain."""
    tags = {"quad_study": ["_trainstates", ""], "rollout_study": ["_k32", ""]}[kind]
    for t in tags:
        f = f"{OUT}/{pde}_{kind}{t}.json"
        if os.path.exists(f):
            return f
    return None

pdes = sys.argv[1:] or sorted({os.path.basename(f).split("_quad_study")[0] for f in glob.glob(f"{OUT}/*_quad_study*.json")})
FAM_COLOR = {"gauss_tensor": "#1f77b4", "smolyak_cc": "#d62728", "smolyak_gl_exp": "#ff7f0e", "smolyak_gl_lin": "#e377c2",
             "sobol": "#2ca02c", "halton": "#17becf", "mc": "#7f7f7f", "fibonacci": "#9467bd", "fibonacci_tent": "#8c564b",
             "mesh_lattice": "#bcbd22", "eq_nnls": "#000000", "dense": "#444444", "continuum_gap": "#aaaaaa"}
FAM_MARK = {"gauss_tensor": "o", "smolyak_cc": "s", "smolyak_gl_exp": "D", "smolyak_gl_lin": "d", "sobol": "^", "halton": "v",
            "mc": "x", "fibonacci": "P", "fibonacci_tent": "*", "mesh_lattice": "h", "eq_nnls": "X", "dense": "_"}
md = []

for pde in pdes:
    qf, rf = pick(pde, "quad_study"), pick(pde, "rollout_study")
    if qf is None:
        continue
    rows = json.load(open(qf))
    md.append(f"\n# {pde}\n")
    md.append(f"Quadrature study file: `{os.path.basename(qf)}`; rollout study file: `{os.path.basename(rf) if rf else 'none'}`.\n")
    # reached-state confirmation (paper's protocol) if available
    rfile = f"{OUT}/{pde}_quad_study_reached.json"
    if os.path.exists(rfile) and "_trainstates" in qf:
        rr_ = {(r["rule"], r["param"], r["form"]): r for r in json.load(open(rfile)) if r["target"] == "continuum"}
        tr_ = {(r["rule"], r["param"], r["form"]): r for r in rows if r["target"] == "continuum"}
        md.append("Reached-state check (states reached by the dense reduced solver on the evaluation cases, 256^2) vs the projected training snapshots used above, worst rho at the accurate M:\n")
        md.append("| rule | param | m | rho max, training states | rho max, reached states |\n|---|---|---|---|---|")
        for key in sorted(tr_, key=lambda k_: (k_[2], k_[0], tr_[k_]["m"])):
            if key in rr_ and key[2] == "point" and key[0] in ("gauss_tensor", "smolyak_cc", "sobol", "fibonacci", "fibonacci_tent", "halton"):
                md.append(f"| {key[0]} | {key[1]} | {tr_[key]['m']} | {tr_[key]['rho_acc_max']:.2e} | {rr_[key]['rho_acc_max']:.2e} |")
    # ---- Fig 1: rho vs m against the continuum reference (mesh independent)
    cont = [r for r in rows if r["target"] == "continuum"]
    forms = sorted({r["form"] for r in cont})
    fig, axes = plt.subplots(1, 2 * len(forms), figsize=(6 * 2 * len(forms), 4.5), squeeze=False)
    for fi, form in enumerate(forms):
        for ai, key in enumerate(["rho_acc_max", "rho_fast_max"]):
            ax = axes[0, 2 * fi + ai]
            for fam in FAM_COLOR:
                pts = sorted([(r["m"], r[key]) for r in cont if r["rule"] == fam and r["form"] == form])
                if pts:
                    ax.loglog(*zip(*pts), marker=FAM_MARK.get(fam, "o"), color=FAM_COLOR[fam], label=fam, ms=5, lw=1.2)
            ax.axhline(0.116, color="k", ls=":", lw=1, label="paper's primary bar 0.116")
            ax.axhline(0.06, color="k", ls="--", lw=0.8, label="paper's tight bar 0.06")
            ax.set_xlabel("quadrature points m"); ax.set_ylabel("worst rho over reached states")
            ax.set_title(f"{pde}: {'accurate M' if ai == 0 else 'fast M'}, form={form}, continuum target")
            ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig1_rho_continuum.png", dpi=130); plt.close(fig)

    # ---- Fig 2: rho vs m against the mesh (upwind) target at each N, off-mesh + mesh rules
    meshes = sorted({r["N"] for r in rows if r["target"] == "mesh"})
    fig, axes = plt.subplots(1, max(len(meshes), 1), figsize=(5.5 * max(len(meshes), 1), 4.5), squeeze=False, sharey=True)
    for ai, N in enumerate(meshes):
        ax = axes[0, ai]
        sub = [r for r in rows if r["target"] == "mesh" and r["N"] == N]
        for fam in FAM_COLOR:
            pts = sorted([(r["m"], r["rho_acc_max"]) for r in sub if r["rule"] == fam and fam != "continuum_gap"])
            if pts:
                ax.loglog(*zip(*pts), marker=FAM_MARK.get(fam, "o"), color=FAM_COLOR[fam], label=fam, ms=5, lw=1.2)
        gap = [r for r in sub if r["rule"] == "continuum_gap"]
        if gap:
            ax.axhline(gap[0]["rho_acc_max"], color="#aaaaaa", ls="-", lw=2, label=f"continuum-vs-upwind gap at N={N}")
        ax.axhline(0.116, color="k", ls=":", lw=1)
        ax.set_title(f"{pde}: mesh target N={N} (accurate M)"); ax.set_xlabel("m"); ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].set_ylabel("worst rho vs dense mesh evaluation"); axes[0, -1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig2_rho_mesh.png", dpi=130); plt.close(fig)

    # ---- Fig 3: ms per residual+Jacobian vs N
    tim = [r for r in rows if r["target"] == "time"]
    if tim:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ai, q in enumerate(sorted({r["q"] for r in tim})):
            ax = axes[ai]
            for lab in sorted({(r["rule"], r["param"]) for r in tim}):
                pts = sorted([(r["N"], r["ms_eval"]) for r in tim if (r["rule"], r["param"]) == lab and r["q"] == q])
                m = [r["m"] for r in tim if (r["rule"], r["param"]) == lab and r["q"] == q][0]
                ax.loglog(*zip(*pts), marker="o", color=FAM_COLOR.get(lab[0], "k"), label=f"{lab[0]} {lab[1]} (m={m})" if lab[0] != "dense" else "dense (m = n)")
            ax.set_xlabel("mesh N (n = (N-1)^2)"); ax.set_ylabel("ms per residual + Jacobian"); ax.set_title(f"{pde}: q={q}")
            ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig3_time_vs_N.png", dpi=130); plt.close(fig)

    # ---- tables of the quadrature study
    md.append("## Quadrature error of the tested nonlinear term (worst rho over the study states: projected training snapshots for `_trainstates` files, dense-solver reached states otherwise)\n")
    md.append("Continuum target (independent of the mesh):\n")
    md.append("| form | rule | param | m | rho max (accurate M) | rho median | rho max (fast M) |\n|---|---|---|---|---|---|---|")
    for r in sorted(cont, key=lambda r: (r["form"], r["rule"], r["m"])):
        md.append(f"| {r['form']} | {r['rule']} | {r['param']} | {r['m']} | {r['rho_acc_max']:.2e} | {r['rho_acc_med']:.2e} | {r['rho_fast_max']:.2e} |")
    md.append("\nMesh target (dense evaluation with the FOM stencil at N):\n")
    md.append("| N | rule | param | m | rho max (accurate M) | rho max (fast M) |\n|---|---|---|---|---|---|")
    for r in sorted([r for r in rows if r["target"] == "mesh"], key=lambda r: (r["N"], r["rule"], r["m"])):
        md.append(f"| {r['N']} | {r['rule']} | {r['param']} | {r['m']} | {r['rho_acc_max']:.2e} | {r['rho_fast_max']:.2e} |")
    if tim:
        md.append("\nCost of one residual + Jacobian evaluation (ms):\n")
        md.append("| q | rule | m | " + " | ".join(f"N={N}" for N in meshes) + " |\n|---|---|---|" + "---|" * len(meshes))
        for q in sorted({r["q"] for r in tim}):
            for lab in sorted({(r["rule"], r["param"]) for r in tim}):
                sub = {r["N"]: r for r in tim if (r["rule"], r["param"]) == lab and r["q"] == q}
                if sub:
                    m = list(sub.values())[0]["m"] if lab[0] != "dense" else "n"
                    md.append(f"| {q} | {lab[0]} {lab[1]} | {m} | " + " | ".join(f"{sub[N]['ms_eval']:.2f}" if N in sub else "-" for N in meshes) + " |")

    # ---- rollout study
    if rf is not None:
        rr = json.load(open(rf))
        meshesR = sorted({r["N"] for r in rr}); qs = sorted({r["q"] for r in rr})
        fig, axes = plt.subplots(len(qs), len(meshesR), figsize=(5 * len(meshesR), 4.2 * len(qs)), squeeze=False)
        for qi, q in enumerate(qs):
            for ni, N in enumerate(meshesR):
                ax = axes[qi, ni]
                sub = [r for r in rr if r["q"] == q and r["N"] == N]
                for r in sub:
                    fam = r["rule"]
                    ax.scatter(r["ms_query"], r["worst_refined"] * 100, color=FAM_COLOR.get(fam, "k"), marker=FAM_MARK.get(fam, "o"),
                               s=40 if r["form"] == "point" else 70, facecolors="none" if r["form"] == "flux" else None,
                               label=f"{fam}" if r["form"] == "point" else f"{fam} (flux)")
                    ax.annotate(str(r["param"]), (r["ms_query"], r["worst_refined"] * 100), fontsize=6, xytext=(3, 2), textcoords="offset points")
                fom = [r["fom_worst_refined"] for r in sub]
                if fom:
                    ax.axhline(fom[0] * 100, color="gray", ls="--", lw=1, label="FOM same-grid vs refined")
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel("ms per query"); ax.set_ylabel("worst rel. L2 error vs refined ref (%)")
                ax.set_title(f"{pde}: N={N}, q={q}"); ax.grid(True, which="both", alpha=0.3)
        h, l = axes[0, 0].get_legend_handles_labels(); uniq = dict(zip(l, h))
        axes[0, 0].legend(uniq.values(), uniq.keys(), fontsize=6)
        fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig4_rollout_pareto.png", dpi=130); plt.close(fig)

        # error vs N per rule (mesh independence of accuracy)
        fig, axes = plt.subplots(1, len(qs), figsize=(6 * len(qs), 4.2), squeeze=False)
        for qi, q in enumerate(qs):
            ax = axes[0, qi]
            for lab in sorted({(r["rule"], r["param"], r["form"]) for r in rr if r["q"] == q}):
                pts = sorted([(r["N"], r["ms_query"]) for r in rr if (r["rule"], r["param"], r["form"]) == lab and r["q"] == q])
                if len(pts) > 1:
                    ax.loglog(*zip(*pts), marker=FAM_MARK.get(lab[0], "o"), color=FAM_COLOR.get(lab[0], "k"),
                              ls="-" if lab[2] == "point" else "--", lw=1, ms=4, label=f"{lab[0]} {lab[1]}{' flux' if lab[2]=='flux' else ''}")
            ax.set_xlabel("mesh N"); ax.set_ylabel("ms per query"); ax.set_title(f"{pde}: query time vs mesh, q={q}")
            ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig5_query_time_vs_N.png", dpi=130); plt.close(fig)

        md.append("\n## End-to-end reduced rollouts\n")
        md.append("Worst relative L2 error over cases and output times (%), against the same-grid FOM and the refined reference; "
                  "ms per query (median over cases, compiled); LM attempts per step; exit counts [budget, stationary, tiny-step, damping-limit].\n")
        md.append("`vs dense` = worst relative distance of the rule's rollout from the dense-evaluation rollout of the same model (the hyper-reduction error).\n")
        md.append("| N | q | M | rule | param | form | m | err same-grid % | err refined % | FOM err refined % | vs dense % | ms/query | att/step | ms/att | exits |\n|" + "---|" * 15)
        for r in sorted(rr, key=lambda r: (r["N"], r["q"], r["kind"] != "dense", r["rule"], r["m"], r["form"])):
            hr = r.get("hr_err_worst", float("nan"))
            md.append(f"| {r['N']} | {r['q']} | {r['M']} | {r['rule']} | {r['param']} | {r['form']} | {r['m']} | {r['worst_same_grid']*100:.3f} | {r['worst_refined']*100:.3f} | "
                      f"{r['fom_worst_refined']*100:.3f} | {hr*100:.3f} | {r['ms_query']:.1f} | {r['attempts_per_step']:.2f} | {r['ms_per_attempt']:.2f} | {r['exits']} |")

# ---- clean timing runs
for pde in pdes:
    tf = f"{OUT}/{pde}_timing.json"
    if not os.path.exists(tf):
        continue
    tr = json.load(open(tf))
    meshesT = sorted({r["N"] for r in tr}); qs = sorted({r["q"] for r in tr})
    md.append(f"\n# {pde}: clean timing run (idle machine, medians of repeated calls)\n")
    md.append("ms per Levenberg--Marquardt time step (jitted while-loop, 3 attempts) / ms per Jacobian evaluation.\n")
    for q in qs:
        md.append(f"\nq = {q}, M = {4 * (tr[0]['M'] // 4 // (1 if True else 1)) if False else [r['M'] for r in tr if r['q'] == q][0]}:\n")
        md.append("| rule | param | m | " + " | ".join(f"N={N}" for N in meshesT) + " |\n|---|---|---|" + "---|" * len(meshesT))
        for lab in sorted({(r["rule"], r["param"]) for r in tr if r["q"] == q}, key=lambda x: (x[0] != "dense", x[0], x[1])):
            sub = {r["N"]: r for r in tr if (r["rule"], r["param"]) == lab and r["q"] == q}
            m = "n" if lab[0] == "dense" else list(sub.values())[0]["m"]
            md.append(f"| {lab[0]} | {lab[1]} | {m} | " + " | ".join(f"{sub[N]['ms_lm_step']:.2f} / {sub[N]['ms_jacobian']:.3f}" if N in sub else "-" for N in meshesT) + " |")
    fig, axes = plt.subplots(1, len(qs), figsize=(6 * len(qs), 4.4), squeeze=False)
    for qi, q in enumerate(qs):
        ax = axes[0, qi]
        for lab in sorted({(r["rule"], r["param"]) for r in tr if r["q"] == q}):
            pts = sorted([(r["N"], r["ms_lm_step"]) for r in tr if (r["rule"], r["param"]) == lab and r["q"] == q])
            m = [r["m"] for r in tr if (r["rule"], r["param"]) == lab and r["q"] == q][0]
            ax.loglog(*zip(*pts), marker=FAM_MARK.get(lab[0], "o"), color=FAM_COLOR.get(lab[0], "k"), lw=1.2, ms=5,
                      label=("dense (m = n)" if lab[0] == "dense" else f"{lab[0]} {lab[1]} (m={m})"))
        ax.set_xlabel("mesh N (n = (N-1)^2)"); ax.set_ylabel("ms per LM time step (3 attempts)")
        ax.set_title(f"{pde}: q={q}"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/{pde}_fig6_clean_timing.png", dpi=130); plt.close(fig)

open(f"{OUT}/REPORT.md", "w").write("\n".join(md))
print("wrote", f"{OUT}/REPORT.md", "and figures")
