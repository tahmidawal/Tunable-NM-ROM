"""Standalone scientific runtime/drift figure from native audited JSON."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("audit", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
review = json.loads(args.audit.read_text())
groups = {(g["intervals"], g["method"]): g for g in review["groups"]}
meshes = sorted(set(key[0] for key in groups))
tolerances = sorted(set(g["tolerance"] for g in groups.values() if g["tolerance"] is not None))
fig, axes = plt.subplots(2, len(meshes), figsize=(10, 7), constrained_layout=True)
colors = {"modular": "#205e9b", "compiled": "#bd4f25"}
for column, n in enumerate(meshes):
    x = np.arange(len(tolerances))
    for path in ("modular", "compiled"):
        rows = [groups[n, f"rom_{path}_gtol{tol:g}"] for tol in tolerances]
        axes[0, column].plot(x, [g["median_query_seconds"]*1000 for g in rows],
                             "o-", color=colors[path], label=path)
    baseline = groups[n, "fom_dst_exact_time"]["median_query_seconds"]*1000
    axes[0, column].axhline(baseline, color="#2d7142", linestyle="--", label="direct DST FOM")
    axes[0, column].set(title=f"{n} solver intervals", ylabel="Complete query median (ms)")
    rows = [groups[n, f"rom_compiled_gtol{tol:g}"] for tol in tolerances]
    drift = [max(rep["field_drift_from_strict"] for case in g["cases"] for rep in case["repetitions"]) for g in rows]
    axes[1, column].plot(x, drift, "o-", color=colors["compiled"], label="largest field change")
    axes[1, column].axhline(review["drift_ceiling"], color="#6c6262", linestyle="--", label="predeclared drift ceiling")
    axes[1, column].set_yscale("symlog", linthresh=1e-6)
    axes[1, column].set(ylabel="Field change / current reference norm", xlabel="Declared gradient tolerance")
    for row in range(2):
        axes[row, column].set_xticks(x, [f"{tol:g}" for tol in tolerances])
        axes[row, column].grid(alpha=.2)
        axes[row, column].legend(fontsize=8, loc="best")
fig.suptitle("Frozen heat checkpoint: faster tolerance, unchanged accuracy limitation\n"
             "Development cohort • shared observation grid • paired GPU timings", fontsize=13)
args.out.parent.mkdir(parents=True, exist_ok=True)
for suffix in ("png", "svg"):
    fig.savefig(args.out.with_suffix(f".{suffix}"), dpi=180)
args.out.with_suffix(".json").write_text(json.dumps(dict(
    source_audit=str(args.audit), source_audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
    observation_intervals=review["observation_intervals"], solver_intervals=meshes,
    gradient_tolerances=tolerances, note="No rigorous continuum bound; empirical refinement evidence only."), indent=2)+"\n")
