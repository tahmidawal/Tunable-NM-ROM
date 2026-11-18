#!/home/tahmid/Dev/.venv/bin/python
"""Build the JCP evidence audit and separate measured/illustrative figures.

Initial execution freezes named, small local JSON records. Later executions read
those copies, validate their hashes, and write only this report's own artifacts.
No solver, training code, cluster client, or JAX module is imported.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports/2026-11-18-jcp-results-audit-assets"
EVIDENCE = ASSETS / "evidence"
REPORT = ROOT / "reports/2026-11-18-jcp-results-audit-and-priorities.md"
NOTES = ROOT / "research_notes/JCP results audit and priorities"
ASSETS.mkdir(exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)
NOTES.mkdir(exist_ok=True)

SOURCES = {
    "ns64": ("worktrees/2026-11-03-ns3d-pod-match/experiments/ns3d-pod-match/runs/ctest64/output/mesh/summary.json", "floors; rollouts; config; frozen_inputs_checked; timing"),
    "ns64_verify": ("worktrees/2026-11-03-ns3d-pod-match/experiments/ns3d-pod-match/runs/ctest64/output/mesh/verify.json", "passed; failed_checks"),
    "ns96": ("worktrees/2026-11-03-ns3d-pod-match/experiments/ns3d-pod-match/runs/ctest96/output/mesh/summary.json", "test; status; config; timing; failures"),
    "ns96_verify": ("worktrees/2026-11-03-ns3d-pod-match/experiments/ns3d-pod-match/runs/ctest96/output/mesh/verify.json", "passed; failed_checks"),
    "ns_final": ("worktrees/2026-11-03-ns3d-pod-match/experiments/ns3d-pod-match/reports/final_table.json", "test; dev; test[mesh].flags/derived/comparators"),
    "ns_dev_config": ("worktrees/2026-11-03-ns3d-pod-match/experiments/ns3d-pod-match/configs/confirm_dev64.json", "n; k; modes; eval_seed; eval_cases; ladder_dt; ladder_iters; horizon; coordnet_bank_sha256"),
    "hard_ns": ("worktrees/2026-11-13-manuscript-rebuttal/manuscript_tools/evidence/fresh_s0b.json", "meshes[128].arms[r128_head|r128_span].config/stats"),
    "heads": ("worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/runs/EXPORT.json", "training_acceptance; configs_1024[arm].Q1/Q2/Q3/jacobian; h_acc_k; lh_per_seed"),
    "dims": ("worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/runs/EXPORT.json", "A3_2d; B6_2d; B6_3d"),
    "reference": ("worktrees/2026-10-06-jcp-references/experiments/jcp-references/store/STATUS.json", "dimension_status; crosscheck; crosscheck_cases; store_version"),
    "time": ("worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/checks/analysis.json", "B4[configuration].label/cases"),
    "budget": ("worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/checks/budget2d-b1024a-1024.json", "members; gates; controls; coverage"),
    "mechanism": ("worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/report_numbers.json", "labels3d; labels2d; a2.panels"),
    "refinement": ("reports/2026-10-26-burgers3d-refinement/VALUES.json", "physical_error_controls; time_refinement_diagnostic; fixed_mode_mesh; test_space_fixed_quadrature"),
    "plateau": ("worktrees/2026-11-05-burgers-quadrature-plateau/experiments/burgers-quadrature-plateau/checks/p2d256b-summary.json", "plateau_ST; plateau_S; families[family].rungs"),
    "rules": ("worktrees/2026-11-15-jcp-sealed-closeout/experiments/jcp-sealed-closeout/results/four-rule-decisions.json", "[rule][ST|S].decision/by_mesh"),
    "scaling": ("worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/checks/analysis.json", "h_acc_n; p1g; timing; kernels_2d; kernels_3d; memory"),
    "tensor": ("worktrees/2026-11-05-ns3d-offmesh-quadrature/experiments/ns3d-offmesh-quadrature/runs/q96/output/summary.json", "timing.case/gate_passed/fast; tensor.head; ladder; mstar"),
    "cost": ("worktrees/2026-11-15-jcp-sealed-closeout/experiments/jcp-sealed-closeout/results/sealed-matched-fom-timing.json", "panels[name].qualified/subject_timings/confirm_rom_over_fom; scope; interpretation"),
    "crom_e": ("worktrees/2026-11-12-official-baselines/experiments/official-baselines/results/test/test-crom-E.json", "cohort_sha256; selection_sha256; results[*].errors/test"),
    "crom_i": ("worktrees/2026-11-12-official-baselines/experiments/official-baselines/results/test/test-crom-I.json", "cohort_sha256; selection_sha256; results[*].errors/test"),
    "colora": ("worktrees/2026-11-12-official-baselines/experiments/official-baselines/results/test/test-colora.json", "cohort_sha256; selection_sha256; results[*].selection/errors/test"),
    "operators": ("worktrees/2026-11-12-operator-scaling/experiments/operator-scaling/closeout/audit.json", "checkpoint_count; test_cases; target_median_percent; families; per_checkpoint"),
    "cylinder": ("reports/2026-11-06-cylinder-unsteady/report-numbers.json", "dev; heldout; qcheck.pass; timing_summary.scope"),
    "seeds2": ("worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/checks/e2-verdicts-2d.json", "seeds; rows; counts; spread"),
    "seeds3": ("worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/checks/e2-verdicts-3d.json", "seeds; rows; counts; spread"),
    "long": ("worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/checks/b7b8-1024.json", "b7[configuration].rows; b8"),
    "amortization": ("worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/EXPORT.json", "claim; labels; primary_ratio; breakeven; ours_offline_seconds; probe; sources"),
}

LEDGER = [
    ("Reference qualification", "DONE, scoped", "reference", "Independent reference checks and per-case uncertainty exist.", "Conditional numerical uncertainty, not a rigorous continuum bound.", "Keep each panel's admitted population; no generic rerun."),
    ("Hard-flow head/span comparison", "PARTIAL mechanism; comparison DONE", "hard_ns", "Held-out performance of selected head/span packages is saved.", "Tests and time steps differ; fixed iteration budgets do not establish converged solves. Different ranks use trained heads.", "P1: isolate the head with matched numerical controls."),
    ("Learned-bank fit versus dynamics", "DONE, diagnostic", "ns64", "Frozen-bank prefix fits, head fit and evolved trajectories are saved.", "Fits use the reference-centroid frame; head fit is finite-budget.", "Replot now; diagnose the evolution gap in P1."),
    ("NS held-out confirmation", "DONE, scoped", "ns_final", "The latest digest and raw audited tests supersede the earlier partial export.", "Coarse mesh has normalization/derivative flags; larger development mesh is not a test.", "Use admitted meshes and retain flags; no repeat confirmation yet."),
    ("Latent-dimension study", "DONE; accuracy trend inconclusive", "heads", "Multiple trained heads, seeds and numerical-qualification results exist.", "Some configurations fail; pooled summaries need not be a deployable checkpoint.", "Inspect failures; no blanket latent sweep."),
    ("Frozen ordered-prefix tunability", "DONE, scoped", "dims", "Fixed-bank rollout and test-count policies were swept.", "Rollout improvement saturates/reverses; this is not guaranteed monotonicity.", "Replot existing curves. Add ordering/retraining controls only for those claims."),
    ("Ordering versus unordered/retrained widths", "NOT FOUND as unified qualified comparison", "dims", "Existing ordering argument and prefix ladders supply part of the evidence.", "No single frozen selection protocol combining all these controls was located.", "Conditional P4 addition if ordering superiority or cost-free reuse is claimed."),
    ("Weak-test-count sensitivity", "PARTIAL", "dims", "Test-count sweeps and raw transition statistics already exist.", "Registered verdicts are blocked by the unqualified square control.", "P2: qualify overdetermined spaces and diagnose failed controls."),
    ("Solver accuracy/stationarity/parity", "DONE for selected settings", "heads", "Production/tight/tighter, stationarity and Jacobian checks are saved.", "Local convergence checks do not certify global minimization.", "Reuse instrumentation in P1; verify selected method only."),
    ("Conditioning and residual visibility", "PARTIAL", "heads", "Step-Jacobian condition summaries and flags are recorded.", "No systematic physical-tangent-normalized visibility/error study was located.", "P2: tangent scaling, independent tests and physical error together."),
    ("Temporal convergence", "DONE for tested settings", "time", "Qualified finite-range time-order labels and per-case checks exist.", "Not every head is covered; later step changes also alter residual weighting.", "Reuse; extend only if the selected final method differs."),
    ("Error-source interventions", "DONE, scoped", "budget", "Ingredient changes, pair interactions, projection and dynamics diagnostics exist.", "Intervention norms are nonadditive; some changes are coupled.", "Do not invent stacked additive error shares; close specific residual ambiguity."),
    ("Derivative/operator mechanism", "PARTIAL physical attribution; diagnostic DONE", "mechanism", "Mesh-node analytic derivatives reproduce the tested off-mesh change.", "Old physical references are preliminary; other dimensions show little separation.", "Use as mechanism evidence; refine the physical claim only if central."),
    ("Quadrature ladders and sealed rules", "DONE, mixed decisions", "rules", "Point-count ladders and frozen rule confirmation exist.", "Primary and secondary references produce different decisions.", "Replot and preserve both; no universal sufficiency claim."),
    ("Dense/sparse residual and Jacobian fidelity", "PARTIAL across final configurations", "refinement", "Operator, solver, quadrature and fixed-test controls exist.", "No unified reached-state residual/action audit for the final coupled method was located.", "Attach this diagnostic to P1/P2; do not rerun every rule."),
    ("Mesh, output and storage scaling", "DONE, scoped", "scaling", "Frozen/nominal tests, components, kernels and memory records exist.", "Reduced evolution, output, stored arrays and allocator peak differ.", "Replot separate contracts; extend only missing complete input/output work."),
    ("Quadrature versus exact tensors", "DONE, negative speed example", "tensor", "Sufficient quadrature can retain accuracy yet lose to tensor contraction.", "Timing uses one case; accuracy covers the development cohort.", "Retain the boundary of usefulness; no universal speed claim."),
    ("Matched classical/POD comparisons", "DONE, scoped", "cost", "Qualified same-allocation lattice/FOM confirmation exists; POD controls are in campaigns.", "Prepared inputs; some timings fail drift. Full baseline catalog is not exhausted.", "P3 only for the selected complete-query claim."),
    ("Official CROM/CoLoRA test accuracy", "DONE", "colora", "Released-code CROM and CoLoRA test records exist; old pending text is stale.", "CoLoRA-D gets parameters; CROM-I is an adapter; equation-driven CoLoRA absent.", "Update the paper; do not repeat expensive official tests first."),
    ("FNO/U-Net test accuracy", "DONE", "operators", "Frozen test selection and per-checkpoint audit exist.", "Training budgets differ; historical ROM was timed in another allocation.", "Use as scoped accuracy evidence; no cross-job speed ratio."),
    ("Training-seed/hardware robustness", "DONE for historical banks", "seeds2", "Recipe repetition and hardware-conditional checks already exist.", "Not replication of every later bank; some selections remain inconclusive.", "Repeat only the final architecture if it materially differs."),
    ("Long-time and physical fidelity", "PARTIAL", "long", "Energy/dissipation, error histories and longer-horizon runs exist.", "Longest reference and some horizon-specific quadratures are unqualified.", "Qualify the claimed horizon or restrict the claim."),
    ("Cylinder forces/divergence", "PARTIAL ranking; experiment DONE", "cylinder", "Field, force, phase and divergence diagnostics are saved.", "Neural bank misses a target; formal quadrature check fails; algorithm differs.", "Resolve normalization only if cylinder remains central; retain original verdict."),
    ("Lifetime cost and weak-residual theory", "PARTIAL cost; theorem NOT FOUND", "amortization", "Cost/break-even studies and algebraic ordering arguments exist.", "Some offline costs are projected; no developed solver stability theorem located.", "Measure final costs if claiming amortization; prove a conditional bound separately."),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze_sources():
    manifest_path = ASSETS / "source-manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    data, manifest = {}, {}
    for key, (relative, pointers) in SOURCES.items():
        frozen = EVIDENCE / f"{key}.json"
        if not frozen.exists():
            frozen.write_bytes((ROOT / relative).read_bytes())
        raw = frozen.read_bytes()
        digest = sha(raw)
        if key in old and old[key]["sha256"] != digest:
            raise ValueError(f"Frozen evidence hash changed: {key}")
        data[key] = json.loads(raw)
        manifest[key] = {"original_path": str(ROOT / relative), "snapshot": str(frozen.relative_to(ROOT)),
                         "sha256": digest, "bytes": len(raw), "inspected_fields": pointers}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return data, manifest


def link(key, label=None):
    return f"[{label or key}]({ROOT / SOURCES[key][0]})"


def asset_link(name, label):
    return f"[{label}]({ASSETS / name})"


def source_links(keys):
    return " · ".join(link(key) for key in keys)


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#66717C", "text.color": "#23313D",
                         "axes.labelcolor": "#23313D", "xtick.color": "#42515E",
                         "ytick.color": "#42515E", "figure.facecolor": "white",
                         "savefig.facecolor": "white", "svg.fonttype": "none"})


def save_figure(fig, stem, description):
    fig.savefig(ASSETS / f"{stem}.png", dpi=180, bbox_inches="tight", metadata={"Description": description})
    fig.savefig(ASSETS / f"{stem}.svg", bbox_inches="tight", metadata={"Description": description})
    plt.close(fig)


def measured_figures(data):
    ns = data["ns64"]
    widths = sorted(int(k.removeprefix("coordnet_R")) for k in ns["floors"] if k.startswith("coordnet_R"))
    fit = [100 * ns["floors"][f"coordnet_R{r}"]["evolved_worst"] for r in widths]
    evolved = [100 * ns["rollouts"][f"coordnet_span{r}"]["stats"]["evolved_worst"] for r in widths]
    all_values = [fit[-1], 100 * ns["floors"]["coordnet_head_k8"]["evolved_worst"],
                  evolved[-1], 100 * ns["rollouts"]["coordnet_head_k8"]["stats"]["evolved_worst"]]
    checks = {}
    for name, result in ns["rollouts"].items():
        recalculated = float(np.max(np.asarray(result["errors"], dtype=float)[:, 1:]))
        stored = result["stats"]["evolved_worst"]
        assert np.isclose(recalculated, stored, rtol=2e-13, atol=1e-15), name
        checks[name] = {"recomputed_worst": recalculated, "stored_worst": stored}
    assert ns["test"] and ns["status"] == "final"
    assert data["ns64_verify"]["passed"] and data["ns96_verify"]["passed"]
    assert set(data["ns_final"]["test"]) == {"32", "64", "96"}
    assert not np.all(np.diff(evolved) <= 0), "Preserve actual nonmonotone curve"
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.3), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("MEASURED EVIDENCE · learned-bank Navier–Stokes test", fontsize=15, fontweight="bold", y=1.02)
    ax[0].plot(widths, fit, "o-", color="#16798C", label="Reference-frame span fit")
    ax[0].plot(widths, evolved, "s-", color="#CB7440", label="Actual span trajectory")
    ax[0].set(xlabel="Active span width R′ (unknowns)", ylabel="Worst evolved relative error (%)", yscale="log",
              title="A. More capacity does not ensure better dynamics", xticks=widths)
    ax[0].grid(axis="y", alpha=.18)
    ax[0].legend(frameon=False, loc="upper right", fontsize=9)
    labels = ["Full-bank\nfit", "Head fit\n(best found)", "Span\ntrajectory", "Head\ntrajectory"]
    colors = ["#A2CAD0", "#16798C", "#CB7440", "#245B78"]
    bars = ax[1].bar(range(4), all_values, color=colors, width=.65)
    ax[1].set(xticks=range(4), xticklabels=labels, ylabel="Worst evolved relative error (%)", yscale="linear",
              title=f"B. Fixed full bank: R = {widths[-1]}, head k = {ns['config']['k']}")
    ax[1].set_ylim(0, max(all_values) * 1.22)
    for bar, value in zip(bars, all_values):
        ax[1].text(bar.get_x()+bar.get_width()/2, value*1.07, f"{value:.3f}%", ha="center", fontsize=9)
    ax[1].grid(axis="y", alpha=.18)
    footer = (f"Source: ctest64 · {ns['config']['eval_cases']} held-out cases · {ns['config']['n']}³ mesh · initial-field-normalized error.\n"
              "Fit diagnostics use the reference-centroid frame; head fit is finite-budget. Selected solver settings; not an isolated causal head ablation.")
    fig.text(.5, -.03, footer, ha="center", va="top", fontsize=9, color="#52616B")
    fig.tight_layout()
    save_figure(fig, "measured-ns3d", footer)

    qualified = [(name, p) for name, p in data["cost"]["panels"].items() if p["qualified"]]
    ratios, rows = [], []
    for name, p in qualified:
        rom_key = f"rom_lat|{p['quadrature_rule']}"
        fom_key = f"fom|{p['matched_fom']}"
        rom_ms = p["subject_timings"][rom_key]["confirm_ms"]
        fom_ms = p["subject_timings"][fom_key]["confirm_ms"]
        ratio = rom_ms / fom_ms
        assert np.isclose(ratio, p["confirm_rom_over_fom"], rtol=1e-13)
        ratios.append(ratio)
        rows.append({"panel": name, "gpu": p["gpu"], "fom": p["matched_fom"], "rom_ms": rom_ms,
                     "fom_ms": fom_ms, "rom_over_fom": ratio})
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    labels = [r["panel"].replace("sealed", "") + " · " + r["gpu"].replace("NVIDIA ", "").replace(" 80GB PCIe", "") for r in rows]
    yy = np.arange(len(rows))
    ax.barh(yy, ratios, color="#A95B3D", height=.58)
    ax.set(yticks=yy, yticklabels=labels, xlabel="Prepared-input ROM time / accuracy-matched FOM time", xlim=(0, max(ratios)*1.13))
    ax.invert_yaxis()
    ax.axvline(1, color="#293B49", linestyle="--", linewidth=1)
    for y, ratio in zip(yy, ratios): ax.text(ratio+.08, y, f"{ratio:.2f}×", va="center")
    ax.set_title("MEASURED COUNTEREXAMPLE · classical solver is faster on admitted panels", fontsize=14, fontweight="bold", pad=16)
    ax.grid(axis="x", alpha=.16)
    fig.text(.5, -.02, "Each ratio is paired within its own allocation; greater than one favors the FOM. Same scoring output, prepared inputs.\n"
             "ROM input preparation is unmeasured; no end-to-end ratio. Source: sealed-matched-fom-timing.json. Failed-drift panels excluded.",
             ha="center", va="top", fontsize=9, color="#52616B")
    fig.tight_layout()
    save_figure(fig, "measured-cost-counterexample", "Actual qualified within-allocation prepared-input cost ratios; not end-to-end timing.")
    measured = {"kind": "measured", "ns64": {"widths": widths, "fit_percent": fit, "rollout_percent": evolved,
                "four_quantities_percent": dict(zip(["full_bank_fit", "head_best_found_fit", "span_rollout", "head_rollout"], all_values)),
                "case_count": ns["config"]["eval_cases"], "mesh": ns["config"]["n"], "head_k": ns["config"]["k"]},
                "qualified_cost_rows": rows, "excluded_cost_panels": {name:p["reasons"] for name,p in data["cost"]["panels"].items() if not p["qualified"]}}
    (ASSETS / "measured-values.json").write_text(json.dumps(measured, indent=2)+"\n")
    return measured, checks


def hypothetical_figure():
    x = np.linspace(0, 1, 201)
    panels = {
        "head_dynamics": {"x": x, "Full-bank fit": x*0+.10, "Head fit": x*0+.14,
                          "Head trajectory": .17+.035*x, "Span trajectory": .17+.65*x**1.6},
        "test_visibility": {"x": x, "Physical error": .14+.72*np.exp(-4.2*x),
                            "Independent weak residual": .12+.75*np.exp(-4*x),
                            "Minimized weak residual": .055+.025*x},
        "quadrature": {"x": x, "Sparse model trajectory error": .19+.72*np.exp(-5*x),
                       "Dense model trajectory error": x*0+.19},
        "pareto": {"x": x, "Tunable candidate": .20+.90*np.exp(-5*x),
                   "Tuned baseline envelope": .08+.72*np.exp(-2.5*x)},
        "scaling": {"x": x, "Reduced evolution": x*0+.13, "Full-field output": .04+.65*x**1.4,
                    "Complete query": .20+.65*x**1.4},
        "prefix": {"x": x, "Candidate prefix outcomes": .23+.65*np.exp(-3*x)+.055*np.sin(5*np.pi*x),
                   "Ideal span-fit trend": .10+.52*np.exp(-4*x)},
    }
    titles = ["A. The head restrains trajectory drift", "B. Independent tests expose residual leakage",
              "C. Integration reaches the fixed-model plateau", "D. A bounded accuracy–cost advantage",
              "E. Full output still costs more as it grows", "F. Select among irregular prefix outcomes"]
    xlabels = ["Evolution time", "Admissible test-space richness", "Quadrature-point budget", "Complete-query cost", "Requested field size", "Frozen-bank prefix width"]
    ylabels = ["Physical error", "Error / residual indicator", "Physical error", "Physical error", "Query time", "Physical error"]
    palette = ["#8C969E", "#769CB1", "#166E88", "#B46F43"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.3))
    fig.suptitle("HYPOTHETICAL DESIRED OUTCOMES — NOT MEASURED OR FORECAST", fontsize=16, fontweight="bold", y=1.01)
    serializable = {"kind": "hypothetical", "empirical_calibration": False,
                    "purpose": "Optimistic figure-design aids only. No predicted percentages, runtimes, rates or probability of success.",
                    "axes": "Qualitative only; arrays are arbitrary deterministic plotting coordinates.", "panels": {}}
    for i, (name, values) in enumerate(panels.items()):
        ax = axes.flat[i]
        series = [(label,y) for label,y in values.items() if label != "x"]
        for j,(label,y) in enumerate(series):
            ax.plot(x, y, linestyle="--", linewidth=2, color=palette[j if len(series)>2 else j+2], label=label)
        ax.set(title=titles[i], xlabel=xlabels[i]+" →", ylabel=ylabels[i]+" →", xlim=(0,1), ylim=(0,1.12), xticks=[], yticks=[])
        ax.text(.02, .98, "Illustrative only", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color="#6B4B7A")
        ax.legend(frameon=False, fontsize=8.4, loc="upper right" if i in [0,4] else "upper right")
        ax.spines["bottom"].set_bounds(0,1)
        serializable["panels"][name] = {label:value.tolist() for label,value in values.items()}
    fig.text(.5, -.02, "Qualitative, formula-generated sketches with arbitrary plotting coordinates. They are hypotheses to test, not paper results.\n"
             "Actual prefix outcomes can be nonmonotonic; actual cost comparisons can favor the baseline. No uncertainty bands or numerical promises are implied.",
             ha="center", va="top", fontsize=10, color="#5B4A68")
    fig.tight_layout(h_pad=3, w_pad=2)
    save_figure(fig, "optimistic-schematics", serializable["purpose"])
    (ASSETS / "hypothetical-curves.json").write_text(json.dumps(serializable, indent=2)+"\n")


def build_report(data, measured):
    ledger_rows = []
    ledger_json = []
    for area,status,key,finding,limit,action in LEDGER:
        ledger_rows.append(f"| {area} | **{status}** | {finding} {link(key, 'Record')} | {limit} | {action} |")
        ledger_json.append({"analysis": area, "status": status, "source_id": key, "source_path": SOURCES[key][0],
                            "fields": SOURCES[key][1], "completed_finding": finding, "limitations": limit, "next_action": action})
    (ASSETS / "analysis-ledger.json").write_text(json.dumps(ledger_json, indent=2)+"\n")
    ledger = "\n".join(ledger_rows)
    rule_rows = []
    for rule,dd in data["rules"].items():
        st, ss = dd["ST"], dd["S"]
        rule_rows.append(f"| `{rule}` | {st['decision']} ({sum(v=='PASS' for v in st['by_mesh'].values())}/{len(st['by_mesh'])}) | {ss['decision']} ({sum(v=='PASS' for v in ss['by_mesh'].values())}/{len(ss['by_mesh'])}) |")
    ns_values = measured["ns64"]["four_quantities_percent"]
    ns_rows = "\n".join(f"| {label} | {ns_values[key]:.4f}% |" for key,label in [
        ("full_bank_fit", "Full-bank reference-frame fit"), ("head_best_found_fit", "Head best-found reference-frame fit"),
        ("span_rollout", "Full-span trajectory"), ("head_rollout", "Head trajectory")])
    cost_rows = "\n".join(f"| `{r['panel']}` | {r['rom_ms']:.2f} | {r['fom_ms']:.2f} | {r['rom_over_fom']:.2f}× |" for r in measured["qualified_cost_rows"])
    cfg = data["ns_dev_config"]
    cfg_rows = "\n".join(f"| {label} | `{cfg[key]}` |" for key,label in [
        ("n","Existing development mesh, points per direction"), ("k","Frozen head dimension"),
        ("eval_cases","Exposed development cases"), ("eval_seed","Development cohort seed"),
        ("modes","Inherited Fourier-mode setting, not assumed equal to scalar test count"),
        ("ladder_dt","Inherited reduced time step, starting value to qualify"),
        ("ladder_iters","Inherited iteration budget, to replace with demonstrated convergence"),
        ("horizon","Inherited integration horizon")])
    ns_coverage = ", ".join(f"${m}^3$" for m in data["ns_final"]["test"])
    hard = data["hard_ns"]["meshes"]["128"]["arms"]
    confounding_rows = "\n".join(f"| `{key}` | {hard[key]['config']['M']} | {hard[key]['config']['steps']} | {hard[key]['config']['iters']} |" for key in ["r128_head","r128_span"])
    source_index = "\n".join(f"| `{key}` | {link(key,'Original JSON')} · [frozen copy]({EVIDENCE / (key+'.json')}) | `{fields.replace('|', chr(92)+'|')}` |" for key,(_,fields) in SOURCES.items())
    (NOTES / "source-index.md").write_text("# JCP results audit source index\n\nSaved numerical records inspected for this audit. The source manifest supplies byte hashes; no new experiment was run.\n\n| ID | Record | Relevant fields |\n|---|---|---|\n"+source_index+"\n")
    (NOTES / "audit-findings.md").write_text("# Reconciled existing-results findings\n\nThe evidence ledger and source index distinguish completed experiments from qualified scientific conclusions.\n\n"+
        "Temporal convergence, nonadditive error-source interventions, training-seed robustness, long-horizon/physical diagnostics, official neural-field tests, and NS held-out confirmation already exist. Earlier pending statements are stale.\n\n"+
        "The latest NS digest and independently audited ctest96 record supersede the older partial confirmation export. Hard-flow head/span packages change numerical settings as well as representation. Weak-test and Jacobian studies exist, but a physically normalized visibility explanation remains missing.\n\n"+
        "The next work should isolate the head, explain test-space failures, and measure the selected complete-query contract. Hypothetical plots are stored separately and are not predictions.\n")

    template = r'''# Audit the evidence before new experiments

**Status: pre-campaign audit.** The [controlled-experiment report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments.md) records the subsequent measured head-mechanism, weak-test and complete-query studies. Its results supersede the unrun-task labels below within those studies' stated scopes.

This November 18, 2026 audit finds that **most broad analysis categories have already been run**. The priority is to isolate the head's contribution, explain weak-test failures, and complete the timing contract for the selected claim; the optimistic figures below are explicitly hypothetical, while measured figures and tables are generated from frozen saved JSONs.

“Done” means the experiment exists within its documented scope, including negative findings. “Partial” means an important qualification, control or interpretation remains unresolved. “Not found” is limited to the named manuscript, campaign exports and source index; it is not a claim that no historical file exists elsewhere. This audit rechecks saved summaries and selected arrays, not every solver run. The separate tree-embedding extension is outside the original-paper recommendation, and the older tree campaign remains paused.

## The missing-work list is smaller than the earlier overview suggested

Temporal-order studies, nonadditive error-budget interventions, training-seed replication, long-horizon tests, physical diagnostics and amortization analyses already exist. Official CROM/CoLoRA test results also exist. Treat their older “pending” status as superseded. The latest NS confirmation covers {{NS_COVERAGE}}; the larger development-only extension must not be described as a held-out test. The coarse test retains bank/derivative flags. ([Latest NS digest]({{NS_FINAL}}), [official test records]({{OFFICIAL}}))

The earlier figure blueprint remains useful, but its “missing” labels need this finer distinction. **Completed negative evidence is a finished analysis**, not a reason to rerun a study until it becomes positive. In particular, matched classical solvers and continuous POD already limit broad advantage claims.

| Analysis | Status | What is already done | What remains unresolved | Next action |
|---|---|---|---|---|
{{LEDGER}}

The source links identify the evidence behind each row. The [full source index](<{{SOURCE_INDEX}}>) gives exact JSON fields, and the [source manifest]({{MANIFEST}}) records immutable copies and hashes. The theory search refers to the [current manuscript]({{MANUSCRIPT}}); it contains the ordering argument and algebraic explanations, but no developed weak-residual error theorem was found. The official baselines' [results notes]({{OFFICIAL_NOTES}}) specify input, precision, implementation and allocation differences.

## Useful positive plots can already be made from actual results

The first figure is measured evidence from the original NS family, using a frozen learned bank. The left panel preserves the actual nonmonotonic prefix trajectory curve. The right panel shows that the head's trajectory stays much closer to its best-found fit than the full span's trajectory does under the recorded settings. This is encouraging evidence for studying constrained dynamics. It does not prove the causal mechanism, because frame choices, time discretization and solve convergence still need the matched control below.

![Measured learned-bank fit and trajectory results]({{MEASURED_NS}})

The following table is generated directly from the same source, in the initial-field-normalized norm. Fits are reference-centroid-frame diagnostics; the head fit is finite-budget and is not a globally optimal manifold distance.

| Saved quantity | Worst evolved relative error |
|---|---:|
{{NS_ROWS}}

There is also a strong completed counterexample. On the admitted Burgers panels, the accuracy-matched spectral solver is faster than the selected lattice model. Each ratio below is measured within one allocation, using a separate confirmation block. Inputs were prepared before timing, and the ROM's missing input-preparation cost prevents an end-to-end ratio. Failed-drift panels are excluded, not silently replaced.

![Measured prepared-input cost comparison]({{MEASURED_COST}})

| Saved panel ID | ROM confirmation time (ms) | FOM confirmation time (ms) | ROM / FOM |
|---|---:|---:|---:|
{{COST_ROWS}}

Panel IDs retain the source's spatial convention; they are not a common cross-job scaling coordinate. The figure labels GPU types, and only ratios within each panel are compared. The broader NS confirmation also distinguishes strict accuracy matching from near-error sensitivity: a relaxed comparator criterion is not the registered primary result.

Quadrature confirmation is completed but has mixed conclusions. The table below counts meshes with a PASS within each rule's separately defined primary space-and-time reference (ST) and secondary space-only reference (S). It establishes sufficiency relative to the dense reduced rule under the recorded criterion, not universal physical accuracy.

| Frozen rule | Primary ST decision (meshes passing) | Secondary S decision (meshes passing) |
|---|---|---|
{{RULE_ROWS}}

## Most proposed paper figures need assembly rather than a new campaign

| Earlier proposed figure | Existing material | What to do now |
|---|---|---|
| Representation versus dynamics | NS test fits/rollouts; Burgers head/cost controls; hard-flow head/span packages | **Replot now.** Add P1 only for a causal head claim. |
| Frozen-model tunability | Burgers fixed-test and scaled-test prefix ladders; NS frozen-bank prefixes | **Replot now.** Preserve reversals. Add unordered/retrained controls only if essential to novelty. |
| Weak-test reliability | Test-count sweeps, blocked square-control verdicts, Jacobian condition flags | **Targeted P2 needed.** Existing measurements are useful, but the explanation is incomplete. |
| Convergence and hyper-reduction | Time-order, ingredient/interactions, quadrature ladders and sealed decisions | **Assemble now.** A final-method time-step × quadrature check is conditional, not a repeat of everything. |
| Mesh transfer/output/storage | Frozen/nominal tests, kernels, fixed-output controls, array and allocator records | **Replot now.** Identify every rebuilt object; keep output work separate. |
| Cost at matched accuracy | Strong classical/POD controls, qualified paired timings, official accuracy comparisons | **Use scoped results now.** P3 closes the chosen full-input/full-output contract. |
| Physical behavior/robustness | Recipe seeds, energy/dissipation, held-out parameters, longer horizons and cylinder forces | **Use admitted portions now.** Qualify only the horizon/application retained in the paper. |

One important confound is visible in the harder-flow source. These are actual saved configurations, not proposed settings:

| Saved arm | Weak-test count | Time steps | Fixed nonlinear iterations |
|---|---:|---:|---:|
{{CONFOUNDING_ROWS}}

The stronger head result is therefore a comparison of selected solver packages. It does not isolate a change in representation alone. The largest rank also uses distinct bank assets, and trained heads across ranks are not a retraining-free width sweep. ([Saved hard-flow record]({{HARD_NS}}))

## Run the next work in this order

### P0 — Assemble and freeze the paper's existing evidence

This is analysis-only work. Rebuild the admitted figures, reconcile stale statuses, retain failed controls and select the paper's central claim. Keep the original-family learned-bank results distinct from the harder-flow POD-bank/head results. Preserve full-bank fit, best-found nonlinear fit and actual trajectory labels. The present report supplies the first measured panels and the complete audit ledger.

The decision at P0 is whether the article centers on head dynamics, reusable bank choices, or classical quadrature/storage. A generic “neural ROM is faster” claim is not supported by the existing comparisons. Do not make every optional comparator, new PDE or tree architecture a prerequisite for the original paper.

**Scope branch:** the priority sequence below assumes the full Tunable-method article. If the authors choose the narrower quadrature paper already discussed, the first new experiment should instead be a final-model time-step × quadrature interaction and dual-reference consistency check, using the existing ladders to investigate the differing ST/S decisions. A head-mechanism campaign is not a prerequisite for that narrower article.

### P1 — Isolate the head on a frozen, already trained model

Start with the saved original-family development configuration below, **not its already opened final-test cohort**. It provides a clean implementation control before repeating the decisive contrast on the harder-flow family. Load the existing bank and frozen head; do not train a new architecture. The source configuration is [confirm_dev64]({{DEV_CONFIG}}), and the frozen head is identified in the test record's `frozen_inputs_checked` by its content hash.

| Starting setting from the saved development configuration | Value |
|---|---|
{{CONFIG_ROWS}}

The bank's recorded SHA-256 is `{{BANK_HASH}}`; the frozen head SHA-256 is `{{HEAD_HASH}}`. These identify inputs, not a new execution authorization. The test-mode setting in the table is a Fourier-mode parameter: record the actual scalar residual-row count rather than assuming they are identical.

Compare the full bank span and its nonlinear head using the **same operator, physical initial field, frame rule, tests, row scaling and time grid**. Use a shared test space at least as rich as the inherited qualified spaces and qualify further enrichment; for harder-flow arms, start from the union of the inherited test spaces or the richer nested space, not merely equal scalar counts. Use the smaller inherited time step initially, then refine it to check sensitivity. Replace fixed iteration budgets with demonstrated trajectory insensitivity under tighter solves. Begin with the exact/dense nonlinear operator where available so quadrature is not another moving ingredient.

Persist per-case/per-time fields or reconstructible coefficients, best-found fits, physical errors, solver termination reasons, iteration histories, step acceptances, minimized and independently expanded weak residuals, frame coordinates, and operator/Jacobian-action checks. Record boundary/divergence/energy quantities appropriate to the PDE. Diagnose head-code fitting with multiple starts; do not call it a certified global minimum.

**Decision:** if the head remains better after matching controls and tightening solves, the paper gains evidence for a representation-induced dynamics benefit. If the gap disappears, attribute the previous contrast to the solver package and narrow the claim. If either arm fails numerical qualification, diagnose that failure before adding meshes or seeds. This is the first new GPU experiment; no GPU-hour forecast is claimed here.

### P2 — Explain what the weak test space can and cannot detect

Reuse P1's instrumentation and a selected existing Burgers checkpoint with recorded test-space problems. Freeze the representation, operator, time grid and a separately qualified quadrature. Sweep admissible overdetermined test spaces relative to the actual unknown count: latent dimension for the head, active bank width for direct coefficients. Retain the original square-control failure as historical evidence. A new protocol can ask a narrower question about qualified overdetermined spaces without retroactively changing that old verdict.

At reached states, orthonormalize physical decoder tangents in the mass norm, then measure the tested PDE Jacobian on those tangents. Save singular values, independent expanded-test residuals, physical error, solve tolerance sensitivity and cost. The current parameterized condition numbers remain useful, but are not invariant to arbitrary latent-coordinate scaling. Check that quadrature and Jacobian-action error are below the effect being diagnosed; positive weights and oversampling alone do not prove stability.

**Decision:** a reliable region should maintain physical accuracy and independent residual agreement as tests are enriched. Small minimized residual with poor physical accuracy exposes missed directions. If enrichment has no effect, examine approximation, operator consistency or optimization before increasing samples further. If time-step × quadrature interaction remains important on the selected final method, add that small factorial check here; the historical convergence studies need not be repeated wholesale.

### P3 — Measure one complete-query accuracy–cost frontier

Only after the scientific configuration is selected, compare Tunable head/span, the strongest eligible POD configuration and the accuracy-matched classical solver in one allocation. Use identical input information, requested field or quantity of interest, precision and error/reference requirements. Time input preparation, cold initialization, evolution, required recovery and output. Save repeated timings and score the field returned by each timed invocation. Qualify timing drift and report failures, medians and outliers. If claiming a quantity-of-interest advantage instead of a full-field advantage, define and measure that separate output contract explicitly.

**Decision:** a speed claim requires an admitted region of lower complete-query cost at matched accuracy. If the ROM loses, retain the comparison and center demonstrated memory, reuse or dynamics benefits instead. Repeating official CROM/CoLoRA tests is not the first priority; their accuracy evidence is already complete. A paired neural-field timing extension is justified only if that speed comparison is a retained contribution.

### P4 — Confirm the selected policy, then close only essential qualifications

Use development cases to select bank width, tests, quadrature and tolerances. Freeze that policy and its checkpoints before an authorized fresh confirmation. Previously opened test cases can be reused as historical diagnostic evidence, but cannot become untouched confirmation again after tuning. Repeat training seeds only if the final architecture differs materially from the recipes already replicated. Add unordered prefixes or separately trained widths only if the manuscript claims that ordering is superior or reuse sacrifices no accuracy.

For a cylinder-centered claim, independently justify the near-zero quadrature normalization treatment and retain the original failed verdict. For a longer-time claim, qualify the longest reference and horizon-specific quadrature; otherwise use the admitted horizon. For an amortization claim, measure actual final training/setup costs and the baseline's corresponding costs. More queries cannot repay an extra offline investment if there is no per-query saving. These are conditional closeouts, not reasons to restart all campaigns.

The theory task runs alongside P1/P2: formalize exact boundary enforcement, nested-span best approximation and linear-term preassembly, then state a conditional weak-residual perturbation bound. The crucial residual-visibility assumption must be explicit. Solver stationarity is not global residual optimality; sampled Jacobian singular values are diagnostics, not proof of a uniform stability constant. No new simulation is required merely to write these distinctions correctly.

## Optimistic plots show hypotheses, not promised results

The following **qualitative schematics are not measurements, numerical forecasts, confidence intervals or proposed paper-result figures**. Their arbitrary deterministic coordinates have no calibration to your experiments. Each panel keeps its label when copied out of this report.

![Hypothetical desired outcomes, not measured or forecast]({{SCHEMATICS}})

The favorable story would be: the head prevents drift under matched numerical controls; independent tests reveal and eventually control residual leakage; sparse integration approaches the dense-model result; and a limited accuracy–cost region favors the method. Full-field output still grows with requested resolution, and prefix rollout errors can remain irregular. The actual measured panels above determine what is already supported. These sketches only identify observations that would strengthen the argument if the proposed experiments produce them.

The paper does not need every favorable pattern. A careful account of where a useful mechanism operates, accompanied by retained counterexamples and explicit output contracts, is a coherent contribution even when the strongest classical method wins some comparisons.

## Reproduction and plain-language glossary

Run [the audit builder]({{BUILDER}}) with the repository's prescribed absolute Python environment. Its first execution freezes the named local JSONs; subsequent runs validate and use those copies. It imports no solver or cluster client. [Measured values]({{MEASURED_JSON}}), [hypothetical arrays]({{HYPOTHETICAL_JSON}}), [verification]({{VERIFICATION}}), [audit ledger]({{LEDGER_JSON}}) and [source manifest]({{MANIFEST}}) are separate. Numerical tables and actual plots are generated from saved measurements; narrative status judgments come from the bounded source audit.

**Status and table columns.** “Done” means a study exists within its scope, including negative results; “partial” identifies a missing qualification/control; “not found” refers to the inspected records. “What is already done” states the observed evidence; “what remains unresolved” limits the claim; “next action” points to the proposed priority. A saved arm is one model/solver configuration. Panel IDs name stored runs, not a universal resolution convention. ROM/FOM is a time ratio; above unity favors the full-order solver.

**ROM, FOM and bank.** A reduced-order model solves for fewer unknowns than a full-order model. A bank is a set of spatial functions; a linear span permits their arbitrary weighted combination. A nonlinear head maps fewer latent coordinates to those coefficients. A prefix keeps the first ordered functions. POD means proper orthogonal decomposition, a linear basis built from snapshots. A frozen model keeps learned weights unchanged.

**Fit, rollout and reference frame.** A fit measures how well a known solution can be represented; a rollout evolves a model through the PDE. The reference-centroid frame uses information from the known reference as a diagnostic, not an online prediction. Best-found nonlinear fit is an achieved error, not a certified optimum. The initial-field-normalized error divides by the initial reference norm; it differs from current-time normalization.

**Weak residual, tests, quadrature and tensor.** A residual measures failure to satisfy an equation. Weak tests observe weighted averages of that failure. Quadrature approximates integrals with weighted points; hyper-reduction reduces equation-evaluation work. A stored tensor is a preassembled multilinear operator. ST refines space and time for the reference; S refines space at the specified time treatment. Their eligible cases and sufficiency verdicts differ.

**Conditioning and consistency.** A Jacobian describes sensitivity; a physical tangent is a field change reachable from the decoder. Singular values describe how strongly those changes are detected. Mass normalization accounts for physical volume. Stationarity means a local gradient is small, not that the result is globally optimal or physically accurate. Consistency concerns approximating the intended equation. A formal gate is a predeclared qualification check; failing it restricts a claim even when a raw statistic looks favorable.

**Cohorts, baselines and cost.** Development cases choose settings; held-out cases test frozen choices. A seed fixes random choices. CROM and CoLoRA are neural-field ROM methods; the tested CoLoRA-D predicts from parameters, while CoLoRA-EQ denotes its equation-driven variant. FNO and U-Net are neural-operator architectures. Prepared-input timing excludes some input conversion; complete-query timing includes all work needed for the agreed output. Offline cost precedes queries; amortization spreads it over repeated use. A Pareto frontier identifies settings where improving accuracy requires more cost or vice versa. Divergence, energy, lift, drag and phase are physical diagnostics. JCP means Journal of Computational Physics.
'''
    replacements = {
        "NS_COVERAGE": ns_coverage, "NS_FINAL": str(ROOT/SOURCES['ns_final'][0]),
        "OFFICIAL": str(ROOT/SOURCES['colora'][0]), "LEDGER": ledger,
        "SOURCE_INDEX": str(NOTES/"source-index.md"), "MANIFEST": str(ASSETS/"source-manifest.json"),
        "MANUSCRIPT": str(ROOT/"worktrees/2026-11-13-manuscript-rebuttal/paper_revision/main.tex"),
        "OFFICIAL_NOTES": str(ROOT/"worktrees/2026-11-12-official-baselines/experiments/official-baselines/RESULTS.md"),
        "MEASURED_NS": str(ASSETS/"measured-ns3d.png"), "NS_ROWS": ns_rows,
        "MEASURED_COST": str(ASSETS/"measured-cost-counterexample.png"), "COST_ROWS": cost_rows,
        "RULE_ROWS": "\n".join(rule_rows), "CONFOUNDING_ROWS": confounding_rows,
        "HARD_NS": str(ROOT/SOURCES['hard_ns'][0]), "DEV_CONFIG": str(ROOT/SOURCES['ns_dev_config'][0]),
        "CONFIG_ROWS": cfg_rows, "BANK_HASH": cfg['coordnet_bank_sha256'],
        "HEAD_HASH": data['ns64']['frozen_inputs_checked']['frozen/coordnet_head_k8_n64.npz'],
        "SCHEMATICS": str(ASSETS/"optimistic-schematics.png"), "BUILDER": str(Path(__file__).resolve()),
        "MEASURED_JSON": str(ASSETS/"measured-values.json"), "HYPOTHETICAL_JSON": str(ASSETS/"hypothetical-curves.json"),
        "VERIFICATION": str(ASSETS/"verification.json"), "LEDGER_JSON": str(ASSETS/"analysis-ledger.json")}
    (ASSETS/"report-template.md").write_text(template)
    report = template
    for key,value in replacements.items(): report = report.replace("{{"+key+"}}", value)
    assert not re.search(r"\{\{[A-Z_]+\}\}", report)
    REPORT.write_text(report)


def main():
    data, manifest = freeze_sources()
    setup_style()
    measured, checks = measured_figures(data)
    hypothetical_figure()
    build_report(data, measured)
    missing_links = []
    for path in re.findall(r"\]\((<?/[^)]+)\)", REPORT.read_text()):
        local = re.sub(r":\d+$", "", path.strip("<>"))
        if not Path(local).exists() and not local.endswith("verification.json"):
            missing_links.append(local)
    assert not missing_links, missing_links
    validation = {"kind": "artifact_verification", "new_experiments": False,
                  "source_count": len(manifest), "source_hashes_validated": True,
                  "ns_rollout_worst_recomputed_from_case_arrays": checks,
                  "ns64_and_ns96_independent_audits_passed": True,
                  "ns_test_meshes": list(data['ns_final']['test']),
                  "cost_ratio_recomputed_from_confirm_ms": True,
                  "qualified_cost_panels": [r['panel'] for r in measured['qualified_cost_rows']],
                  "excluded_cost_panels": measured['excluded_cost_panels'],
                  "hypothetical_curves_separate_and_uncalibrated": True,
                  "local_link_failures": missing_links,
                  "report_word_count": len(REPORT.read_text().split()),
                  "visual_review": "Writer and parent inspected all three PNGs; parent requested and writer applied a zero-based linear bar axis. Labels readable, no clipped content, measured and hypothetical figures clearly separated."}
    (ASSETS/"verification.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps({"report":str(REPORT), "words":validation['report_word_count'],
                      "sources":len(manifest), "measured":measured}, indent=2))


if __name__ == "__main__":
    main()
