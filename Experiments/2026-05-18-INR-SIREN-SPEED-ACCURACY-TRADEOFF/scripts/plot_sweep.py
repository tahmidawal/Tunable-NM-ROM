#!/usr/bin/env python
"""Plot the M-sweep comparison from eval_sweep_M.py output JSON.

Produces two plots:
  1. rel-L2 vs M (per decoder).
  2. rel-L2 vs total per-query FLOPs (per decoder, parametrised by M).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    obj = json.loads(Path(args.json).read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    # Label each line by the checkpoint file stem so multiple variants of the
    # same decoder_kind can be distinguished.
    def _label(d):
        ck = d.get("ckpt", "")
        stem = ck.rsplit("/", 1)[-1].replace(".pkl", "")
        return stem or d["decoder_kind"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in obj["decoders"]:
        Ms = [r["M"] for r in d["results"]]
        ls = [r["rel_l2"] for r in d["results"]]
        ax.plot(Ms, ls, "o-", label=_label(d))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("M (query points / snapshot)")
    ax.set_ylabel("val rel-L2 (off-mesh)")
    ax.set_title("rel-L2 vs sampling density")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "rel_l2_vs_M.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in obj["decoders"]:
        flops = [r["flops_per_query"] for r in d["results"]]
        ls = [r["rel_l2"] for r in d["results"]]
        ax.plot(flops, ls, "o-", label=_label(d))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FLOPs / query (incl. amortized snapshot work)")
    ax.set_ylabel("val rel-L2 (off-mesh)")
    ax.set_title("Pareto: accuracy vs per-query compute")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "pareto_relL2_vs_flops.png", dpi=120)
    plt.close(fig)

    print(f"Plots saved to {args.out}/")


if __name__ == "__main__":
    main()
