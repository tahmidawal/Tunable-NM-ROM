"""Generate reports/2026-11-11-ns3d-confirm.md and its digest JSON from the pulled
summaries of confirm_mesh.py. Every number in the report is read here from summary.json.
Usage: build_2026-11-11-ns3d-confirm.py dev:32=cdev32b dev:64=... test:96=... dev:128=..."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/ns3d-pod-match/runs"
pc = lambda x: "n/a" if x is None else f"{100 * x:.3f}"
ms = lambda x: "n/a" if x is None else f"{x:.2f}"
rx = lambda x: "n/a" if x is None else f"{x:.2f}x"


def arm_order(rows):
    def key(a):
        fam = 0 if a.startswith("coordnet") else 1 if a.startswith("pod") else 2
        head = 0 if "head" in a else 1
        num = int("".join(ch for ch in a.split("_")[-1] if ch.isdigit()) or 0)
        return (fam, head, num)
    return sorted(rows, key=key)


def digest(run):
    s = json.loads((RUNS / run / "output/mesh/summary.json").read_text())
    log = next((RUNS / run / "logs").glob("*.out")).read_text()
    t = s["timing"]["fast"]
    coarse = s.get("coarse_fom", {})
    cn = s["cnab2"]
    d = dict(run=run, job=s["job_id"], commit=s["source_commit"], gpu=s["gpu"], n=s["config"]["n"],
             cohort_seed=s["config"]["eval_seed"], cases=s["config"]["eval_cases"],
             status=s["status"], failures=s["failures"], jax_backend_gpu="jax_backend=gpu" in log,
             timing_gate=s["timing"]["gate_passed"], elapsed_s=s["elapsed_seconds"],
             floors={a: b["evolved_worst"] for a, b in s["floors"].items()},
             interp_bank=s.get("interp_pod_bank"), arms={}, cnab2={}, coarse_floor={}, coarse_front=[])
    verify = RUNS / run / "output/mesh/verify.json"
    d["audit_passed"] = json.loads(verify.read_text())["passed"] if verify.exists() else None
    for st, e in cn.items():
        d["cnab2"][f"s{st}"] = dict(worst=None if e["stats"] is None else e["stats"]["evolved_worst"],
                                    unstable=e["unstable"], **{k: t[f"CNAB2_s{st}"][k] for k in ("min_ms", "median_ms", "max_ms")})
    # every stable candidate of the mesh-free comparator: coarse (end-to-end / solve-only) and same-mesh
    cand = []
    for c, e in coarse.items():
        if f"COARSE_{c}" in t:
            cand.append(dict(name=c, mesh=e["mesh"], worst=e["stats"]["evolved_worst"],
                             ms=t[f"COARSE_{c}"]["median_ms"], solve_ms=t[f"COARSESOLVE_{c}"]["median_ms"]))
    for st, e in cn.items():
        if e["stats"] is not None and not e["unstable"]:
            v = t[f"CNAB2_s{st}"]["median_ms"]
            cand.append(dict(name=f"m{d['n']}_cnab2_s{st}", mesh=d["n"], worst=e["stats"]["evolved_worst"], ms=v, solve_ms=v))
    for m in sorted({e["mesh"] for e in coarse.values()}):
        ok = [e for e in coarse.values() if e["mesh"] == m and e["stats"] is not None and not e["unstable"]]
        if ok:
            best = min(ok, key=lambda e: e["stats"]["evolved_worst"])
            d["coarse_floor"][str(m)] = dict(worst=best["stats"]["evolved_worst"], at=f"{best['scheme']}_s{best['steps']}")
    front = []
    for c in sorted(cand, key=lambda c: c["ms"]):
        if not front or c["worst"] < front[-1]["worst"]:
            front.append(c)
    d["coarse_front"] = front
    d["candidates"] = cand
    for a in arm_order(s["rollouts"]):
        row = s["rollouts"][a]
        if not row["finite"]:
            d["arms"][a] = dict(finite=False)
            continue
        tt = t[f"query_{a}"]
        worst = row["stats"]["evolved_worst"]
        same = s["comparators"][a]
        elig = [c for c in cand if c["worst"] <= worst]
        e2e = min(elig, key=lambda c: c["ms"]) if elig else None
        so = min(elig, key=lambda c: c["solve_ms"]) if elig else None
        strict = [c for c in elig if c["mesh"] < d["n"]]
        d["arms"][a] = dict(
            finite=True, worst=worst, median=row["stats"]["evolved_median"],
            min_ms=tt["min_ms"], median_ms=tt["median_ms"], max_ms=tt["max_ms"],
            same_mesh=same.get("comparator"), same_mesh_ms=same.get("comparator_ms"),
            same_mesh_speedup=same.get("speedup"),
            free=None if e2e is None else e2e["name"], free_worst=None if e2e is None else e2e["worst"],
            free_ms=None if e2e is None else e2e["ms"],
            free_ratio=None if e2e is None else e2e["ms"] / tt["median_ms"],
            free_solve=None if so is None else so["name"], free_solve_ms=None if so is None else so["solve_ms"],
            free_solve_ratio=None if so is None else so["solve_ms"] / tt["median_ms"],
            coarser_exists=bool(strict))
    return d


def table(d):
    out = [f"### {d['n']}^3, cohort seed {d['cohort_seed']} ({d['cases']} cases), job {d['job']} ({d['run']})", "",
           f"GPU: {d['gpu']}; status {d['status']}; timing gate {d['timing_gate']}; NumPy audit {d['audit_passed']}; "
           f"jax_backend=gpu in log: {d['jax_backend_gpu']}; flags: {d['failures']}", "",
           "| arm | median err % | worst err % | ms min / med / max | same-mesh CNAB2 (ms) | speedup vs same-mesh | accuracy-matched any-mesh solver, end-to-end (worst %, ms) | ROM speedup vs it | same, solve-only (ms) | ROM speedup |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in d["arms"].items():
        if not r["finite"]:
            out.append(f"| {a} | nonfinite |" + " |" * 8)
            continue
        out.append(f"| {a} | {pc(r['median'])} | {pc(r['worst'])} | {ms(r['min_ms'])} / {ms(r['median_ms'])} / {ms(r['max_ms'])} "
                   f"| {r['same_mesh']} ({ms(r['same_mesh_ms'])}) | {rx(r['same_mesh_speedup'])} "
                   f"| {r['free']} ({pc(r['free_worst'])}, {ms(r['free_ms'])}) | {rx(r['free_ratio'])} "
                   f"| {r['free_solve']} ({ms(r['free_solve_ms'])}) | {rx(r['free_solve_ratio'])} |")
    out += ["", "Coarse-mesh spatial floor (best worst-case error over the step ladder, vs the same truth): "
            + ", ".join(f"m={m}: {pc(v['worst'])} % ({v['at']})" for m, v in d["coarse_floor"].items()), "",
            "Solver Pareto front (end-to-end median ms -> worst %): "
            + "; ".join(f"{c['name']} {ms(c['ms'])} ms -> {pc(c['worst'])}" for c in d["coarse_front"]), "",
            "Oracle floors (worst %): " + ", ".join(f"{a} {pc(b)}" for a, b in d["floors"].items()), ""]
    if d["interp_bank"]:
        b = d["interp_bank"]
        out += [f"Interpolated POD bank: energy removed by resampling+mask (worst column) {pc(b['resample_plus_mask_removed_worst'])} %, "
                f"orthonormality defect before Lowdin {b['orthonormality_before_lowdin']:.3g}.", ""]
    return out


def main():
    groups = {}
    for item in sys.argv[1:]:
        key, run = item.split("=")
        cohort, n = key.split(":")
        groups.setdefault(cohort, {})[int(n)] = digest(run)
    lines = ["# NS3D confirmation for the rebuttal (generated; do not edit by hand)", "",
             "Generated by `reports/build_2026-11-11-ns3d-confirm.py` from the pulled `summary.json` files. "
             "Protocol: `experiments/ns3d-pod-match/PROTOCOL.md` (fixed before outcomes; amendments A1, A2 there).", "",
             "Error = L2 difference / L2 norm of the case's initial field, vs CNAB2 at the same mesh with dt 0.001. "
             "\"Accuracy-matched any-mesh solver\" = fastest stable pseudo-spectral run (CNAB2 or RK4, any mesh m <= n, "
             "any step count on the ladder) whose worst error on the cohort is <= the arm's; end-to-end includes "
             "Fourier truncation of the input and zero-padding of the 6 output frames to the n-mesh; "
             "solve-only excludes both. Speedup > 1 means the ROM is faster.", ""]
    for cohort, title in (("test", "Test cohort (opened once)"), ("dev", "Development cohort")):
        if cohort in groups:
            lines += [f"## {title}", ""]
            for n in sorted(groups[cohort]):
                lines += table(groups[cohort][n])
    (ROOT / "reports/2026-11-11-ns3d-confirm.md").write_text("\n".join(lines) + "\n")
    (ROOT / "reports/2026-11-11-ns3d-confirm.json").write_text(json.dumps(groups, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
