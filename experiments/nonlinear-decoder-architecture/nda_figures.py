"""Build publication-ready figures from ``summary.json`` only."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
plt.rcParams["svg.hashsalt"] = "nonlinear-decoder-architecture"


def load_summary():
    with open(os.path.join(HERE, "summary.json")) as f:
        return json.load(f)


def save(fig, stem):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, f"{stem}.png"), dpi=220, bbox_inches="tight")
    svg_path = os.path.join(FIGURES, f"{stem}.svg")
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    with open(svg_path) as f:
        svg = f.read()
    with open(svg_path, "w") as f:
        f.write("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
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

    burgers = sorted(
        (r for r in summary["burgers_three_seed"]
         if r["hidden"] == 160 and r["group_size"] == 2 and r["M"] == 128),
        key=lambda r: r["m"])
    xs = [r["m"] for r in burgers]
    axes[1].errorbar(xs, [r["full"]["mean"] for r in burgers],
                     yerr=[r["full"]["sample_std"] for r in burgers],
                     marker="o", capsize=3, color="#4C78A8", label="full weak")
    axes[1].errorbar(xs, [r["eq"]["mean"] for r in burgers],
                     yerr=[r["eq"]["sample_std"] for r in burgers],
                     marker="s", capsize=3, color="#F58518", label="NNLS-EQ weak")
    recommended = summary["burgers_gate_audit"]["recommended"]
    if recommended is not None:
        chosen = next(r for r in burgers if r["m"] == recommended["m"])
        axes[1].scatter(chosen["m"], chosen["eq"]["mean"], s=145,
                        facecolors="none", edgecolors="#222222", linewidths=1.4,
                        zorder=4)
        axes[1].annotate(f'selected m={chosen["m"]}',
                         (chosen["m"], chosen["eq"]["mean"]),
                         xytext=(8, -24), textcoords="offset points", fontsize=8)
    axes[1].axhline(1e-2, color="#777777", linestyle="--", linewidth=1,
                    label="1% accuracy target")
    axes[1].set_title("Burgers H160/g2: quadrature boundary")
    axes[1].set_xlabel("NNLS quadrature points m (M=128)")
    axes[1].set_ylabel("held-out trajectory relative $L^2$ error")
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")
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
    b_all = summary["burgers_three_seed"]
    b_rec = summary["burgers_gate_audit"]["recommended"]
    b_focus = []
    if b_rec is not None:
        same_family = [r for r in b_all
                       if r["architecture"] == b_rec["architecture"] and
                       r["hidden"] == b_rec["hidden"] and
                       r["group_size"] == b_rec["group_size"] and
                       r["M"] == b_rec["M"]]
        selected = next(r for r in same_family if r["m"] == b_rec["m"])
        lower = max((r for r in same_family if r["m"] < b_rec["m"]),
                    key=lambda r: r["m"], default=None)
        b_focus.extend(r for r in (lower, selected) if r is not None)
    b_focus.extend(r for r in b_all
                   if r["M"] == 128 and r["m"] == 640 and
                   ((r["hidden"] == 159 and r["group_size"] == 3) or
                    (r["hidden"] == 160 and r["group_size"] == 4)))
    b_focus = sorted(b_focus, key=lambda r: (r["group_size"], r["m"]))
    panels = [
        ("Poisson H98/g2", 6e-3,
         sorted((r for r in summary["poisson_three_seed"]
                 if r["hidden"] == 98 and r["group_size"] == 2),
                key=lambda r: (r["m"], r["M"])),
         lambda r: f'M{r["M"]}\nm{r["m"]}'),
        ("Burgers architecture boundary", 1e-2,
         b_focus,
         lambda r: f'H{r["hidden"]}/g{r["group_size"]}\nM{r["M"]}/m{r["m"]}'),
    ]
    if not any(rows for _, _, rows, _ in panels):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    offsets = {"decoder": -0.24, "full": 0.0, "eq": 0.24}
    colors = {"decoder": "#54A24B", "full": "#4C78A8", "eq": "#F58518"}
    for ax, (title, gate, rows, label) in zip(axes, panels):
        x = np.arange(len(rows))
        for metric in ("decoder", "full", "eq"):
            means = [row[metric]["mean"] for row in rows]
            stds = [row[metric]["sample_std"] for row in rows]
            ax.errorbar(x + offsets[metric], means, yerr=stds, fmt="o", capsize=4,
                        color=colors[metric], label=metric)
            for index, row in enumerate(rows):
                ax.scatter(np.repeat(x[index] + offsets[metric], 3), row[metric]["values"],
                           s=18, color=colors[metric], alpha=0.55)
        ax.axhline(gate, color="#777777", linestyle="--", linewidth=1,
                   label="accuracy gate")
        ax.set_xticks(x, [label(row) for row in rows], fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("weak objective / quadrature")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("held-out relative $L^2$ error")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Three-seed values and mean ± sample standard deviation")
    save(fig, "three_seed_variability")


def main():
    summary = load_summary()
    architecture_tradeoff(summary)
    decoder_speed(summary)
    seed_variability(summary)
    print(f"wrote figures under {FIGURES}")


if __name__ == "__main__":
    main()
