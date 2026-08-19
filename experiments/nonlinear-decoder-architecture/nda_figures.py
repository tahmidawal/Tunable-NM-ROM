"""Build publication-ready figures from ``summary.json`` only."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")


def load_summary():
    with open(os.path.join(HERE, "summary.json")) as f:
        return json.load(f)


def save(fig, stem):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, f"{stem}.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, f"{stem}.svg"), bbox_inches="tight")
    plt.close(fig)


def architecture_tradeoff(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), constrained_layout=True)

    poisson = [r for r in summary["poisson"] if r["seed"] == 0]
    marker = {"film": "s", "groupfilm": "o", "resfilm": "x"}
    color = {"film": "#4C78A8", "groupfilm": "#F58518", "resfilm": "#B9B9B9"}
    for row in poisson:
        axes[0].scatter(row["n_params"], row["eq"], s=55,
                        marker=marker[row["architecture"]],
                        color=color[row["architecture"]], zorder=3)
    selected = next(r for r in poisson if r["cell"] == "nda_pg98l4g2_r6")
    control = next(r for r in poisson if r["cell"] == "saved-control/fair-M128")
    axes[0].annotate("selected H98", (selected["n_params"], selected["eq"]),
                     xytext=(8, -20), textcoords="offset points")
    axes[0].annotate("saved control", (control["n_params"], control["eq"]),
                     xytext=(-98, 10), textcoords="offset points")
    axes[0].axhline(6e-3, color="#777777", linestyle="--", linewidth=1,
                    label="EQ reporting gate")
    axes[0].set_title("Poisson: size–accuracy screen")
    axes[0].set_xlabel("trainable parameters")
    axes[0].set_ylabel("held-out EQ ROM relative $L^2$ error")
    axes[0].set_yscale("log")
    axes[0].set_ylim(4e-3, 2.5e-2)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")

    burgers = summary["burgers_objectives"]
    trust0 = [r for r in burgers if r["seed"] == 0 and r["trust_factor"] == 0]
    xs = np.arange(len(trust0))
    labels = [f'M={r["M"]}, m={r["m"]}' for r in trust0]
    axes[1].bar(xs - 0.18, [r["full"] for r in trust0], width=0.36,
                color="#4C78A8", label="full weak")
    axes[1].bar(xs + 0.18, [r["eq"] for r in trust0], width=0.36,
                color="#F58518", label="NNLS-EQ weak")
    axes[1].axhline(1e-2, color="#777777", linestyle="--", linewidth=1,
                    label="1% accuracy target")
    axes[1].set_xticks(xs, labels)
    axes[1].set_title("Burgers H160: objective refinement")
    axes[1].set_ylabel("held-out trajectory relative $L^2$ error")
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[1].grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[1:] + handles[:1], legend_labels[1:] + legend_labels[:1],
                   frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    save(fig, "architecture_accuracy_tradeoff")


def decoder_speed(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), constrained_layout=True)
    for axis, bench in zip(axes, summary["benchmarks"]):
        for kernel, linestyle in (("forward", "-"), ("jacobian", "--")):
            rows = sorted((r for r in bench["rows"] if r["kernel"] == kernel),
                          key=lambda r: r["points"])
            x = [r["points"] for r in rows]
            axis.plot(x, [r["speedup"] for r in rows], marker="o",
                      linestyle=linestyle, label=f"{kernel}, raw")
            if all(r["cached_speedup"] is not None for r in rows):
                axis.plot(x, [r["cached_speedup"] for r in rows], marker="s",
                          linestyle=linestyle, label=f"{kernel}, cached")
        axis.axhline(1.0, color="#777777", linewidth=1)
        axis.set_xscale("log")
        axis.set_title(bench["pde"].title())
        axis.set_xlabel("decoder points")
        axis.set_ylabel("saved control / compact decoder median time")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    save(fig, "same_gpu_decoder_speedup")


def seed_variability(summary):
    available = summary["poisson_three_seed"] + summary["burgers_three_seed"]
    if not available:
        return
    groups = []
    for row in available:
        if row["name"] == "Poisson":
            label = f'Poisson H{row["hidden"]}'
        else:
            label = f'Burgers g{row["group_size"]} M{row["M"]}/m{row["m"]}'
        groups.append((label, row))
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(groups)), 4.5), constrained_layout=True)
    x = np.arange(len(groups))
    offsets = {"decoder": -0.24, "full": 0.0, "eq": 0.24}
    colors = {"decoder": "#54A24B", "full": "#4C78A8", "eq": "#F58518"}
    for metric in ("decoder", "full", "eq"):
        means = [row[metric]["mean"] for _, row in groups]
        stds = [row[metric]["sample_std"] for _, row in groups]
        ax.errorbar(x + offsets[metric], means, yerr=stds, fmt="o", capsize=4,
                    color=colors[metric], label=metric)
        for index, (_, row) in enumerate(groups):
            ax.scatter(np.repeat(x[index] + offsets[metric], 3), row[metric]["values"],
                       s=18, color=colors[metric], alpha=0.55)
    ax.set_xticks(x, [label for label, _ in groups])
    ax.set_ylabel("held-out relative $L^2$ error")
    ax.set_title("Three-seed variability (points) and mean ± sample std")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    save(fig, "three_seed_variability")


def main():
    summary = load_summary()
    architecture_tradeoff(summary)
    decoder_speed(summary)
    seed_variability(summary)
    print(f"wrote figures under {FIGURES}")


if __name__ == "__main__":
    main()
