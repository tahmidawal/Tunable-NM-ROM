"""Reproducible scientific transfer figures from audited native records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(analysis):
    audit = json.loads((analysis/"audit.json").read_text())
    assert audit["passed"]
    groups = audit["groups"]; cohorts = ["original", "fresh", "union"]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .2, "savefig.dpi": 180})
    colors = {"expanded_seed790714": "#0072B2", "expanded_seed790715": "#D55E00"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True, sharex="col")
    for column, cohort in enumerate(cohorts):
        rows = [r for r in groups if r["cohort"] == cohort]
        ns = sorted({r["intervals"] for r in rows})
        for name, color in colors.items():
            arm = sorted((r for r in rows if r["method"] == name), key=lambda r: r["intervals"])
            x = [r["intervals"] for r in arm]
            axes[0, column].plot(x, [r["full_error"]*100 for r in arm], "o-", color=color, label=name.replace("expanded_seed790", "head "))
            axes[0, column].plot(x, [r["common_error"]*100 for r in arm], ":", color=color, label="common-grid norm" if name.endswith("714") else None)
            axes[1, column].plot(x, [r["cost_seconds"]*1000 for r in arm], "o-", color=color)
        samegrid = [next(r for r in rows if r["method"] == f"fom_dst_{n}" and r["intervals"] == n) for n in ns]
        axes[0, column].plot(ns, [r["full_error"]*100 for r in samegrid], "^-", color="#555555", label="same-grid DST")
        axes[1, column].plot(ns, [r["cost_seconds"]*1000 for r in samegrid], "^-", color="#555555")
        selected = sorted((s for s in audit["selections"] if s["cohort"] == cohort and s["norm"] == "full" and s["target"] == .05), key=lambda s: s["intervals"])
        axes[1, column].plot([s["intervals"] for s in selected], [s["fom_seconds"]*1000 if s["fom_seconds"] is not None else np.nan for s in selected], "s--", color="#009E73", label="cheapest FOM at 5%")
        axes[0, column].axhline(5, color="#009E73", ls="--", lw=.8, label="5% target")
        axes[0, column].set_title(cohort+" development cohort")
        for row in range(2):
            axes[row, column].set_xscale("log", base=2); axes[row, column].set_yscale("log")
            axes[row, column].set_xticks(ns, [str(n) for n in ns])
        axes[1, column].set_xlabel("Requested intervals per axis")
    axes[0, 0].set_ylabel("Worst current-relative physical error (%)")
    axes[1, 0].set_ylabel("Complete query (ms)")
    axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
    fig.suptitle("Frozen heat heads: full requested outputs and charged coarse-FOM envelope\nAll repetitions; case-median costs; empirical reference evidence, no strict certificate", fontsize=11)
    for suffix in ("png", "pdf"): fig.savefig(analysis/f"heat-transfer.{suffix}")
    plt.close(fig)
    # Complete-query decomposition exposes output and input costs as the mesh grows.
    rows = [r for r in groups if r["cohort"] == "union"]
    ns = sorted({r["intervals"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, method in zip(axes, [*colors, "fom_dst_16"]):
        arm = sorted((r for r in rows if r["method"] == method), key=lambda r: r["intervals"])
        bottom = np.zeros(len(arm))
        for key, label, color in (("input_restriction_transfer_seconds", "input/restriction", "#56B4E9"),
                                  ("device_solve_readout_seconds", "device solve/readout", "#E69F00"),
                                  ("full_host_output_seconds", "complete host output", "#009E73")):
            values = np.asarray([r[key] for r in arm])*1000
            ax.bar(np.arange(len(arm)), values, bottom=bottom, label=label, color=color); bottom += values
        ax.plot(np.arange(len(arm)), [r["cost_seconds"]*1000 for r in arm], "ko", label="full query median", ms=3)
        ax.set_xticks(np.arange(len(arm)), [r["intervals"] for r in arm]); ax.set_xlabel("Requested intervals")
        ax.set_title(method.replace("expanded_seed790", "head ")); ax.set_ylabel("Milliseconds")
    axes[0].legend(fontsize=7)
    fig.suptitle("Union cohort: component case-median costs (component medians need not sum to query median)")
    for suffix in ("png", "pdf"): fig.savefig(analysis/f"heat-transfer-components.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("analysis", type=Path); plot(p.parse_args().analysis)
