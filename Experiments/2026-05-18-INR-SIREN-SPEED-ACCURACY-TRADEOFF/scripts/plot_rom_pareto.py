#!/usr/bin/env python
"""Plot ROM Pareto: rel-L2 vs n_eq, and rel-L2 vs speedup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CP_REF = [
    ("CP best-acc",   3.23e-3, 183.0),
    ("CP best-speed", 2.00e-2, 324.0),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    obj = json.loads(Path(args.json).read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    by_decoder = defaultdict(list)
    for r in obj["results"]:
        by_decoder[r["decoder_label"]].append(r)
    for k in by_decoder:
        by_decoder[k].sort(key=lambda r: r["n_eq"])

    # Plot 1: rel-L2 vs n_eq.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, rs in by_decoder.items():
        ax.plot([r["n_eq"] for r in rs],
                [r["rom_relL2_median"] for r in rs],
                "o-", label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n_eq (EQ residual nodes)")
    ax.set_ylabel("ROM rel-L2 (vs CG FOM, median)")
    ax.set_title("ROM accuracy vs hyper-reduction budget")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "rom_relL2_vs_neq.png", dpi=120)
    plt.close(fig)

    # Plot 2: rel-L2 vs speedup.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, rs in by_decoder.items():
        ax.plot([r["speedup_median"] for r in rs],
                [r["rom_relL2_median"] for r in rs],
                "o-", label=label)
    for name, rL2, sp in CP_REF:
        ax.plot([sp], [rL2], "*", color="black", markersize=12)
        ax.annotate(name, (sp, rL2), fontsize=8,
                    textcoords="offset points", xytext=(6, 4))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Speedup over FOM CG (median)")
    ax.set_ylabel("ROM rel-L2 (median)")
    ax.set_title("ROM Pareto: accuracy vs speedup")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "rom_pareto.png", dpi=120)
    plt.close(fig)
    print(f"Plots saved to {args.out}/")


if __name__ == "__main__":
    main()
