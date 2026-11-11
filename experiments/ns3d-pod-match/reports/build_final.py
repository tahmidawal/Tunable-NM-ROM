"""Final report of the ns3d-confirm lane: table (markdown), digest (JSON) and one figure.

Every number is read from runs/<run>/output/mesh/summary.json (keys `comparators`,
`coarse_comparators`, `timing.fast`, `coarse_fom`, `cnab2`); nothing is typed by hand.
CPU only. Usage:
  /home/tahmid/Dev/.venv/bin/python experiments/ns3d-pod-match/reports/build_final.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
COHORTS = {"dev": {32: "cdev32b", 64: "cdev64b", 96: "cdev96b", 128: "cdev128c"},
           "test": {32: "ctest32", 64: "ctest64", 96: "ctest96"}}
ARMS = ["coordnet_head_k8", "pod_head_k8", "interp_head_k8", "coordnet_span8", "coordnet_span16",
        "coordnet_span32", "coordnet_span48", "coordnet_span64", "pod_linear_R64"]
HEAD = "coordnet_head_k8"
NEAR = 1.25  # sensitivity factor on the arm's worst error (not the protocol rule)

pc = lambda x: "n/a" if x is None else f"{100 * x:.3f}"
ms = lambda x: "n/a" if x is None else f"{x:.2f}"
rx = lambda x: "n/a" if x is None else f"{x:.2f}x"


def spread(t, key):
    e = t.get(key)
    return None if e is None else {k: e[k] for k in ("min_ms", "median_ms", "max_ms")}


def digest(cohort, n, run):
    s = json.loads((RUNS / run / "output/mesh/summary.json").read_text())
    log = next((RUNS / run / "logs").glob("*.out")).read_text()
    verify = RUNS / run / "output/mesh/verify.json"
    t = s["timing"]["fast"]
    assert s["config"]["n"] == n
    coarse = s.get("coarse_fom", {})
    # every timed, stable solver candidate (coarse ladder end-to-end, plus same-mesh CNAB2)
    cand = [dict(name=c, worst=e["stats"]["evolved_worst"], ms=t[f"COARSE_{c}"]["median_ms"])
            for c, e in coarse.items() if f"COARSE_{c}" in t]
    cand += [dict(name=f"CNAB2_s{st}", worst=e["stats"]["evolved_worst"], ms=t[f"CNAB2_s{st}"]["median_ms"])
             for st, e in s["cnab2"].items() if e["stats"] is not None and not e["unstable"]]
    d = dict(run=run, job=s["job_id"], source_commit=s["source_commit"], gpu=s["gpu"], n=n, cohort=cohort,
             cohort_seed=s["config"]["eval_seed"], cases=s["config"]["eval_cases"], status=s["status"],
             flags=s["failures"], jax_backend_gpu="jax_backend=gpu" in log, all_exit_0="ALL_EXIT=0" in log,
             timing_gate=s["timing"]["gate_passed"],
             numpy_audit=json.loads(verify.read_text())["passed"] if verify.exists() else None,
             ladder_includes_same_mesh=any(e["mesh"] == n for e in coarse.values()),
             elapsed_seconds=s["elapsed_seconds"],
             comparators={}, coarse_comparators={}, timing_spread={}, derived={}, not_run=[])
    for a in ARMS:
        if a not in s["comparators"]:
            d["not_run"].append(a)
            continue
        c, k = s["comparators"][a], s["coarse_comparators"][a]
        d["comparators"][a] = {key: c.get(key) for key in ("arm_ms", "arm_worst", "comparator", "comparator_ms", "speedup")}
        d["comparators"][a]["eligible"] = len(c["eligible"]) if isinstance(c.get("eligible"), dict) else c.get("eligible")
        d["coarse_comparators"][a] = {key: k.get(key) for key in
                                      ("comparator", "comparator_ms", "comparator_worst", "ratio", "ratio_solve_only", "eligible")}
        d["coarse_comparators"][a]["note"] = k.get("note")
        d["timing_spread"][a] = dict(
            arm=spread(t, f"query_{a}"),
            same_mesh=spread(t, c["comparator"]) if c.get("comparator") else None,
            coarse=spread(t, f"COARSE_{k['comparator']}") if k.get("comparator") else None)
        # protocol rule over ALL candidates = the smaller of the two ratios saved in the summary
        both = [(k.get("ratio"), k.get("comparator")), (c.get("speedup"), c.get("comparator"))]
        both = [b for b in both if b[0] is not None]
        best = min(both, key=lambda b: b[0]) if both else (None, None)
        near = min((x for x in cand if x["worst"] <= NEAR * c["arm_worst"]), key=lambda x: x["ms"], default=None)
        d["derived"][a] = dict(
            any_candidate_ratio=best[0], any_candidate_comparator=best[1],
            near_comparator=None if near is None else near["name"],
            near_worst=None if near is None else near["worst"],
            near_ratio=None if near is None else near["ms"] / c["arm_ms"])
    return d


def sp(x):
    return "n/a" if x is None else f"{ms(x['min_ms'])} / {ms(x['median_ms'])} / {ms(x['max_ms'])}"


def table(d):
    out = [f"### {d['cohort']} {d['n']}^3 (run {d['run']}, job {d['job']}, seed {d['cohort_seed']}, {d['cases']} cases)", "",
           f"GPU {d['gpu']}; `jax_backend=gpu` in log: {d['jax_backend_gpu']}; `ALL_EXIT=0` in log: {d['all_exit_0']}; "
           f"timing gate: {d['timing_gate']}; NumPy audit: {d['numpy_audit']}; "
           f"solver ladder includes the ROM's own mesh (amendment A3): {d['ladder_includes_same_mesh']}; flags: {d['flags'] or 'none'}.", "",
           "| arm | worst err % | ROM ms | ROM ms min / med / max | same-mesh comparator | its ms | speedup | "
           "matched any-mesh comparator | its worst % | its ms | its ms min / med / max | ratio | ratio solve-only | eligible |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        if a in d["not_run"]:
            out.append(f"| {a} | not run at this mesh |" + " |" * 12)
            continue
        c, k, tsp = d["comparators"][a], d["coarse_comparators"][a], d["timing_spread"][a]
        out.append(f"| {a} | {pc(c['arm_worst'])} | {ms(c['arm_ms'])} | {sp(tsp['arm'])} | {c['comparator']} | {ms(c['comparator_ms'])} "
                   f"| {rx(c['speedup'])} | {k['comparator'] or 'none as accurate'} | {pc(k['comparator_worst'])} | {ms(k['comparator_ms'])} "
                   f"| {sp(tsp['coarse'])} | {rx(k['ratio'])} | {rx(k['ratio_solve_only'])} | {k['eligible'] if k['eligible'] is not None else 0} |")
    return out + [""]


def figure(groups):
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    meshes = [32, 64, 96, 128]
    pos = {n: i for i, n in enumerate(meshes)}
    series = [("speedup", "vs same-mesh CNAB2", blue), ("ratio", "vs accuracy-matched any-mesh solver", orange)]
    for cohort, filled, ls in (("test", True, "-"), ("dev", False, "--")):
        for key, label, col in series:
            xs, ys = [], []
            dx = -0.045 if filled else 0.045  # small dodge so coincident cohorts stay visible
            for n in sorted(groups[cohort]):
                d = groups[cohort][n]
                e = d["derived"][HEAD]
                y = d["comparators"][HEAD]["speedup"] if key == "speedup" else e["any_candidate_ratio"]
                if y is None:
                    continue
                xs.append(pos[n] + dx); ys.append(y)
                same = e["any_candidate_comparator"] == d["comparators"][HEAD]["comparator"]
                if key == "ratio" and same:
                    continue  # the matched solver IS the same-mesh one: one label serves both points
                txt = f"{y:.2f}x" + (" (both)" if key == "speedup" and same else "")
                ax.annotate(txt, (pos[n] + dx, y), textcoords="offset points", xytext=(-9 if filled else 9, 0),
                            ha="right" if filled else "left", va="center", fontsize=8, color=ink)
            ax.plot(xs, ys, ls, color=col, lw=2, marker="o" if filled else "s", ms=8,
                    mfc=col if filled else "#fcfcfb", mec=col, mew=2,
                    label=f"{label}, {'test cohort (sealed, 32 cases)' if filled else 'development cohort (16 cases)'}")
    ax.axhline(1.0, color=muted, lw=1.2, zorder=0)
    ax.text(3.47, 1.0, "ratio = 1\n(break-even)", va="center", ha="left", fontsize=8, color=muted)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16]); ax.set_yticklabels(["0.25x", "0.5x", "1x", "2x", "4x", "8x", "16x"])
    ax.minorticks_off()
    ax.set_xticks(range(len(meshes))); ax.set_xticklabels([f"{n}$^3$" for n in meshes])
    ax.set_xlim(-0.45, 3.45); ax.set_ylim(0.2, 20)
    ax.set_xlabel("mesh", color=muted); ax.set_ylabel("solver time / ROM time  (>1: ROM faster)", color=muted)
    ax.set_title("NS 3D, coordinate-bank head (k = 8), worst error about 0.2 %", fontsize=10, color=ink, loc="left")
    ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"final_head_ratio_vs_mesh.{ext}", facecolor=fig.get_facecolor(), bbox_inches="tight")


def main():
    groups = {c: {n: digest(c, n, r) for n, r in runs.items()} for c, runs in COHORTS.items()}
    L = ["# NS 3D confirmation lane: final table (generated; do not edit by hand)", "",
         "State: final. All seven runs are pulled; the sealed test cohort (seed 202609221, 32 cases) was opened once, "
         "one job per mesh. Generated by `experiments/ns3d-pod-match/reports/build_final.py` from the pulled "
         "`summary.json` files; protocol and amendments A1-A3 in `experiments/ns3d-pod-match/PROTOCOL.md`.", "",
         "Reading rule: a speedup or ratio above 1 means the reduced model is faster than the solver it is compared with.", "",
         "## Headline: coordinate-bank head (k = 8)", "",
         "![head ratios versus mesh](final_head_ratio_vs_mesh.png)", "",
         "| cohort | mesh | worst err % | ROM ms | same-mesh CNAB2 speedup | accuracy-matched ratio (comparator, its worst %) | solve-only ratio "
         f"| protocol rule over all timed candidates | sensitivity: fastest solver within {NEAR}x of the arm's error |",
         "|---|---|---|---|---|---|---|---|---|"]
    for cohort in ("test", "dev"):
        for n, d in groups[cohort].items():
            c, k, e = d["comparators"][HEAD], d["coarse_comparators"][HEAD], d["derived"][HEAD]
            L.append(f"| {cohort} | {n}^3 | {pc(c['arm_worst'])} | {ms(c['arm_ms'])} | {rx(c['speedup'])} ({c['comparator']}) "
                     f"| {rx(k['ratio'])} ({k['comparator'] or 'none as accurate'}, {pc(k['comparator_worst'])}) | {rx(k['ratio_solve_only'])} "
                     f"| {rx(e['any_candidate_ratio'])} ({e['any_candidate_comparator']}) "
                     f"| {rx(e['near_ratio'])} ({e['near_comparator']}, {pc(e['near_worst'])}) |")
    L += ["",
          "Caveats that belong next to these numbers:", "",
          "- The development runs at 32^3, 64^3 and 96^3 (`cdev*b`) ran before amendment A3, so their accuracy-matched ladder "
          "has only meshes coarser than the ROM's and no RK4 at the ROM's own mesh; the test runs and `cdev128c` include it. "
          "Where the table says \"none as accurate\", the protocol comparator is the same-mesh CNAB2 entry "
          "(column \"protocol rule over all timed candidates\"), and the figure plots that column.",
          "- The sensitivity column is not the protocol rule. It was added after the outcomes were seen and shows how the ratio "
          "moves if the solver may be up to 1.25x less accurate than the arm.",
          "- 128^3 is exploratory (development cohort only, the 96^3 head used zero-shot, no NumPy audit because fields were not stored).",
          "- The heads were trained in development jobs; only the test rows are held out.", ""]
    for cohort, title in (("test", "Test cohort (sealed, opened once)"), ("dev", "Development cohort")):
        L += [f"## {title}", ""]
        for d in groups[cohort].values():
            L += table(d)
    L += ["## Glossary", "",
          "- **arm**: one reduced-model configuration. `coordnet_head_k8`: coordinate-network basis bank with a learned 8-coefficient head "
          "(the accurate setting). `pod_head_k8`: the same with the paper's per-mesh POD basis. `interp_head_k8`: the 64^3 POD basis "
          "Fourier-resampled to the mesh, with the 64^3 head used without retraining. `coordnet_spanR`: the first R bank columns used as a "
          "plain linear subspace. `pod_linear_R64`: rank-64 linear POD.",
          "- **worst err %**: over all cases and evolved frames, the largest L2 error divided by the L2 norm of that case's initial field, "
          "against CNAB2 at the same mesh with time step 0.001.",
          "- **ROM ms**: median of 7 repeated timings of one query (after warm-up), with min / median / max beside it.",
          "- **same-mesh comparator, speedup**: the fastest CNAB2 step count at the ROM's mesh whose worst error is no larger than the arm's; "
          "speedup = its ms / ROM ms (`comparators.<arm>` in the JSON).",
          "- **matched any-mesh comparator, ratio**: the fastest stable pseudo-spectral run (CNAB2 or RK4, any mesh and step count on the "
          "ladder) whose worst error is no larger than the arm's, timed end-to-end (truncate the input, solve, pad the six frames back to "
          "the ROM's mesh); ratio = its ms / ROM ms (`coarse_comparators.<arm>`).",
          "- **ratio solve-only**: the same with truncation and padding excluded from the solver's time (more favourable to the solver).",
          "- **eligible**: how many timed ladder configurations were at least as accurate as the arm.",
          "- **CNAB2_sK / mM_rk4_sK / mM_cnab2_sK**: solver scheme (Crank-Nicolson/Adams-Bashforth 2 or Runge-Kutta 4), mesh M^3, K time steps.",
          "- **timing gate**: sentinel timings before and after the timing block agree within 10 %.",
          "- **NumPy audit**: an independent recomputation of the errors from the stored fields.",
          "- **development / test cohort**: 16 cases (seed 202609202) used while building; 32 disjoint cases (seed 202609221) opened once.", ""]
    (HERE / "final_table.md").write_text("\n".join(L) + "\n")
    (HERE / "final_table.json").write_text(json.dumps(groups, indent=1) + "\n")
    figure(groups)
    print("\n".join(L))
    print("GPU-hours (sum of elapsed_seconds of these 7 runs):", sum(d["elapsed_seconds"] for g in groups.values() for d in g.values()) / 3600)


if __name__ == "__main__":
    main()
