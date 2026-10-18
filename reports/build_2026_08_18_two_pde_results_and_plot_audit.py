#!/home/tahmid/Dev/.venv/bin/python
"""Build the Poisson/Burgers result and plot audit from archived August 18 JSONs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "worktrees" / "2026-08-18-codex-handoff"
REPORT_STEM = "2026-08-18-two-pde-results-and-plot-audit"
MARKDOWN_OUT = ROOT / "reports" / f"{REPORT_STEM}.md"
ARTIFACT_OUT = ROOT / "reports" / f"{REPORT_STEM}.artifact.json"
FIGURE_DIR = ROOT / "reports" / "figs"
N2048_LEARNED_FIGURE = FIGURE_DIR / "two-pde-n2048-learned-speedup.png"
N2048_CLASSICAL_FIGURE = FIGURE_DIR / "two-pde-n2048-classical-speedup.png"


PATHS = {
    "poisson_objectives": HANDOFF
    / "experiments/poisson2d-rom-objective/runs/obj_K8_S1/obj_K8_S1.json",
    "poisson_accuracy": HANDOFF
    / "experiments/poisson2d-rom-objective/runs/followup/pk_K8/rom_K8.json",
    "poisson_pod": HANDOFF
    / "experiments/poisson2d-rom-objective/runs/followup/pp_pod/pod_ladder.json",
    "poisson_quadrature": HANDOFF
    / "experiments/poisson2d-rom-objective/runs/followup/pt_m/timing_m.json",
    "burgers_accuracy": HANDOFF
    / "experiments/burgers2d-rom-latent-stepping/runs/ad_n64_k8/blat_rom_N64_K8.json",
    "burgers_quadrature": HANDOFF
    / "experiments/burgers2d-rom-latent-stepping/runs/followup/bm_m/blat_rom_N64_K8.json",
    "optimizer": HANDOFF / "experiments/k-stall-diagnosis/exp3_N64.json",
    "cost": HANDOFF / "experiments/cost-to-tolerance/runs/pareto_points.json",
    "hybrid": HANDOFF / "experiments/rom-warmstart-fom/runs/hybrid_points.json",
    "hybrid_verifier": HANDOFF / "experiments/rom-warmstart-fom/wsf_verify.py",
    "poisson_n2048": ROOT
    / "worktrees/2026-08-20-poisson-hybrid-2048/experiments/poisson-hybrid-1024/"
    "runs/n2048final1/out/n2048final1.json",
    "poisson_n2048_audit": ROOT
    / "worktrees/2026-08-20-poisson-hybrid-2048/experiments/poisson-hybrid-1024/"
    "runs/n2048final1/INDEPENDENT-AUDIT.json",
    "burgers_n2048": ROOT
    / "worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/"
    "runs/final3/out/final.json",
    "burgers_n2048_audit": ROOT
    / "worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/"
    "runs/final3/INDEPENDENT-AUDIT.json",
    "burgers_n2048_learned": ROOT
    / "worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/"
    "runs/learned1/out/learned.json",
    "burgers_n2048_learned_audit": ROOT
    / "worktrees/2026-08-20-burgers-hybrid-2048/experiments/burgers-hybrid-2048/"
    "runs/learned1/LEARNED-AUDIT.json",
}


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files() -> None:
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required archived inputs:\n" + "\n".join(missing))


def repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sci(value: float) -> str:
    return f"{value:.3e}"


def dec(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def speed(value: float) -> str:
    return f"{value:.3f}x"


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def md_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def run_hybrid_verifier() -> dict[str, int]:
    result = subprocess.run(
        ["/home/tahmid/Dev/.venv/bin/python", str(PATHS["hybrid_verifier"])],
        cwd=PATHS["hybrid_verifier"].parent,
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"INDEPENDENT VERIFICATION: (\d+) checks passed, (\d+) failed", result.stdout)
    if not match:
        raise RuntimeError("hybrid verifier completed without its expected summary")
    return {"passed": int(match.group(1)), "failed": int(match.group(2))}


def build_accuracy() -> tuple[list[dict], list[dict]]:
    poisson = load(PATHS["poisson_accuracy"])
    poisson_pod = load(PATHS["poisson_pod"])
    burgers = load(PATHS["burgers_accuracy"])

    poisson_floor = poisson["oracle"]["nearest"]
    poisson_full = next(
        row for row in poisson["rows"] if row["scheme"] == "full" and row["init"] == "nearest"
    )["rom_rel_l2_mean"]
    poisson_eq = next(
        row for row in poisson["rows"] if row["scheme"] == "nnls" and row["init"] == "nearest"
    )["rom_rel_l2_mean"]
    poisson_pod_row = next(row for row in poisson_pod["rows"] if row["k"] == 8)

    burgers_floor = burgers["oracle_inferred_latent_test"]["traj_rel_mean"]
    burgers_full = burgers["rom"]["lspg:full:weak64"]["traj_rel_mean"]
    burgers_eq = burgers["rom"]["lspg:eq256:weak64"]["traj_rel_mean"]
    burgers_pod = burgers["pod_rom"]["k8:lspg:full:fd"]["traj_rel_mean"]
    burgers_pod_floor = burgers["oracle_pod_projection_floor_test"]["8"]

    nonlinear = []
    detail = []
    definitions = [
        ("Poisson-2D", poisson_floor, poisson_full, poisson_eq, poisson_pod_row["weak_a1_M64"]["mean"], poisson_pod_row["proj"]["mean"]),
        ("Burgers-2D", burgers_floor, burgers_full, burgers_eq, burgers_pod, burgers_pod_floor),
    ]
    for pde, floor, full, eq, pod, pod_floor in definitions:
        for route, error in (("Decoder ceiling", floor), ("Full-grid weak", full), ("EQ weak", eq)):
            row = {
                "pde": pde,
                "route": route,
                "mean_error": error,
                "ceiling": floor,
                "ratio_to_ceiling": error / floor,
                "pod_error": pod,
                "pod_floor": pod_floor,
            }
            nonlinear.append(row)
            detail.append(row)
        detail.append(
            {
                "pde": pde,
                "route": "POD-ROM",
                "mean_error": pod,
                "ceiling": floor,
                "ratio_to_ceiling": pod / floor,
                "pod_error": pod,
                "pod_floor": pod_floor,
            }
        )
    return nonlinear, detail


def build_objectives() -> list[dict]:
    payload = load(PATHS["poisson_objectives"])
    labels = {
        "fd": "Pointwise finite difference",
        "spec_a1_M64": "Weak / spectral M64",
        "cg20": "CG-filtered residual",
        "ritz": "Ritz energy",
        "lowpass4": "Low-pass residual",
    }
    rows = []
    for objective, label in labels.items():
        source = next(
            row for row in payload["rows"] if row["objective"] == objective and row["init"] == "nearest"
        )
        rows.append(
            {
                "objective": label,
                "objective_key": objective,
                "mean_error": source["rom_rel_l2_mean"],
                "median_error": source["rom_rel_l2_med"],
                "max_error": source["rom_rel_l2_max"],
                "n_sources": len(source["per_sample_rom_rel_l2"]),
            }
        )
    return rows


def build_quadrature() -> tuple[list[dict], list[dict]]:
    poisson = load(PATHS["poisson_quadrature"])
    burgers = load(PATHS["burgers_quadrature"])
    M = 64
    sizes = [64, 128, 256, 512, 1024, 3844]
    labels = {64: "1xM", 128: "2xM", 256: "4xM", 512: "8xM", 1024: "16xM", 3844: "full grid"}

    poisson_rows = []
    for m in sizes:
        pool = "full" if m == 3844 else "grid"
        source = next(row for row in poisson["rows"] if row["M"] == M and row["m"] == m and row["pool"] == pool)
        poisson_rows.append(
            {
                "pde": "Poisson-2D",
                "M": M,
                "m": m,
                "m_label": labels[m],
                "mean_error": source["rom_rel_l2_mean"],
                "time_ms": source["rom_solve_s"] * 1000.0,
                "eq_fit": source["eq_info"]["rel_fit"] if source["eq_info"] else None,
                "worst_row_error": source["eq_info"]["row_rel_max"] if source["eq_info"] else None,
            }
        )

    burgers_map = {}
    for method, values in burgers["rom"].items():
        if method == "lspg:full:weak64" or re.fullmatch(r"lspg:eq(64|128|256|512|1024):weak64", method):
            burgers_map[values["m"]] = values
    burgers_rows = []
    for m in sizes:
        source = burgers_map[m]
        burgers_rows.append(
            {
                "pde": "Burgers-2D",
                "M": M,
                "m": m,
                "m_label": labels[m],
                "mean_error": source["traj_rel_mean"],
                "time_ms": source["step_time_ms_median"],
                "eq_fit": source["eq_info"]["rel_fit"] if source["eq_info"] else None,
                "worst_row_error": source["eq_info"]["row_rel_max"] if source["eq_info"] else None,
            }
        )

    detail = poisson_rows + burgers_rows
    full_by_pde = {row["pde"]: row for row in detail if row["m"] == 3844}
    for row in detail:
        full = full_by_pde[row["pde"]]
        row["error_ratio_to_full"] = row["mean_error"] / full["mean_error"]
        row["time_ratio_to_full"] = row["time_ms"] / full["time_ms"]
    chart = [dict(row) for row in detail]
    return chart, detail


def build_optimizer() -> tuple[list[dict], list[dict]]:
    payload = load(PATHS["optimizer"])
    selected = [row for row in payload if row["arm"] in {"base", "tr"}]
    chart = []
    detail = []
    for row in selected:
        chart.append(
            {
                "k": row["k"],
                "k_label": str(row["k"]),
                "solver": "Base LM" if row["arm"] == "base" else "Trust region",
                "ratio_mean": row["ratio_mean"],
                "ratio_median": row["ratio_med"],
                "divergent_cases": row["n_blown"],
                "n_sources": len(row["err_per_source"]),
            }
        )
    for k in sorted({row["k"] for row in selected}):
        base = next(row for row in selected if row["k"] == k and row["arm"] == "base")
        trust = next(row for row in selected if row["k"] == k and row["arm"] == "tr")
        detail.append(
            {
                "k": k,
                "base_mean": base["ratio_mean"],
                "base_median": base["ratio_med"],
                "base_divergent": base["n_blown"],
                "trust_mean": trust["ratio_mean"],
                "trust_median": trust["ratio_med"],
                "trust_divergent": trust["n_blown"],
                "n_sources": len(base["err_per_source"]),
            }
        )
    return chart, detail


def build_cost() -> tuple[list[dict], dict[str, int]]:
    payload = load(PATHS["cost"])
    coord_rows = [
        row
        for row in payload
        if row["pde"] == "poisson2d"
        and row["arm"] == "consolidated"
        and row["method"] == "coord"
        and row["k"] == 16
        and row["tau"] == 0.01
    ]
    fom_rows = [
        row
        for row in payload
        if row["pde"] == "poisson2d"
        and row["arm"] == "fom_consolidated"
        and row["gpu"] == "NVIDIA A100 80GB PCIe"
    ]
    result = []
    for coord in sorted(coord_rows, key=lambda row: row["N"]):
        eligible = [
            row
            for row in fom_rows
            if row["N"] == coord["N"] and row["err_rel_l2"] <= coord["err_rel_l2"]
        ]
        if not eligible:
            raise RuntimeError(f"no accuracy-matched FOM row for Poisson N={coord['N']}")
        fom = min(eligible, key=lambda row: row["time_ms"])
        result.append(
            {
                "N": coord["N"],
                "N_label": str(coord["N"]),
                "coord_error": coord["err_rel_l2"],
                "coord_time_ms": coord["time_ms"],
                "fom_error": fom["err_rel_l2"],
                "fom_time_ms": fom["time_ms"],
                "speedup": fom["time_ms"] / coord["time_ms"],
                "gpu": coord["gpu"],
                "coord_censored": coord["censored"],
                "status": "Provisional: formal cell review incomplete",
            }
        )
    burgers_coord = [
        row
        for row in payload
        if row["pde"] == "burgers2d" and row["arm"] == "consolidated" and row["method"] == "coord"
    ]
    counts = {
        "burgers_total": len(burgers_coord),
        "burgers_uncensored": sum(not row["censored"] for row in burgers_coord),
    }
    return result, counts


def build_hybrid() -> tuple[list[dict], list[dict], dict[str, int]]:
    payload = load(PATHS["hybrid"])
    consolidated = [row for row in payload if row["run_role"] == "consolidated"]
    fom_tau = 1e-6
    selected = []
    for pde in ("poisson2d", "burgers2d"):
        pde_rows = [row for row in consolidated if row["pde"] == pde and row["fom_tau"] == fom_tau]
        for N in sorted({row["N"] for row in pde_rows}):
            candidates = [row for row in pde_rows if row["N"] == N]
            selected.append(max(candidates, key=lambda row: row["speedup_vs_fom"]))
    rows = []
    for source in selected:
        rows.append(
            {
                "pde": "Poisson-2D" if source["pde"] == "poisson2d" else "Burgers-2D",
                "N": source["N"],
                "N_label": str(source["N"]),
                "rom_tau": source["rom_tau"],
                "fom_tau": source["fom_tau"],
                "speedup": source["speedup_vs_fom"],
                "baseline_iters": source["iters_from_baseline"],
                "rom_iters": source["iters_from_rom"],
                "extrap_iters": source.get("iters_from_extrap"),
                "fom_time_ms": source["t_fom_baseline_ms"],
                "hybrid_time_ms": source["t_total_ms"],
                "selection": "Best observed ROM tolerance at fixed FOM tolerance",
            }
        )
    verifier = run_hybrid_verifier()
    return rows, rows, verifier


def build_n2048() -> dict:
    """Load and validate the final, independently audited N=2048 extension."""
    poisson = load(PATHS["poisson_n2048"])
    poisson_audit = load(PATHS["poisson_n2048_audit"])
    burgers = load(PATHS["burgers_n2048"])
    burgers_audit = load(PATHS["burgers_n2048_audit"])
    learned = load(PATHS["burgers_n2048_learned"])
    learned_audit = load(PATHS["burgers_n2048_learned_audit"])

    assert poisson["complete"] and poisson_audit["audit_pass"]
    assert sha256(PATHS["poisson_n2048"]) == poisson_audit["source_json_sha256"]
    assert poisson["config"]["ns"] == [2048]
    assert poisson["config"]["test_seed"] == 20260826
    assert poisson["provenance"]["jax_backend"] == "gpu"
    assert poisson["provenance"]["x64"]
    assert poisson["provenance"]["matmul_precision"] == "highest"
    assert poisson_audit["record_counts"] == {
        "learned": 576,
        "production": 1200,
        "outliers": 0,
    }

    assert burgers["complete"] and burgers_audit["status"] == "pass"
    assert sha256(PATHS["burgers_n2048"]) == burgers_audit["source_sha256"]
    assert burgers["config"]["ns"] == [2048]
    assert burgers["config"]["test_seed"] == 20260830
    assert burgers["config"]["classification"]["dynamic"].endswith(
        "not learned and not NM-ROM"
    )
    assert burgers["provenance"]["jax_backend"] == "gpu"
    assert burgers["provenance"]["x64"]
    assert burgers["provenance"]["matmul_precision"] == "highest"
    assert burgers_audit["timing_record_count"] == 288
    assert burgers_audit["burn_record_count"] == 144
    assert burgers_audit["reference"]["all_pass"]

    assert learned["complete"] and learned_audit["status"] == "complete"
    assert sha256(PATHS["burgers_n2048_learned"]) == learned_audit["source_sha256"]
    assert learned["config"]["N"] == 2048
    assert learned["config"]["test_seed"] == 20260828
    assert learned["config"]["classification"]["film_nmrom"].startswith(
        "genuine weak FiLM NM-ROM"
    )
    assert learned["provenance"]["jax_backend"] == "gpu"
    assert learned["provenance"]["x64"]
    assert learned["provenance"]["matmul_precision"] == "highest"
    assert sum(len(row["records"]) for row in learned["rows"]) == 576
    assert sum(len(row["burn_records"]) for row in learned["rows"]) == 288
    assert learned_audit["film_supported_vs_cubic_cells"] == 0
    assert learned_audit["film_supported_vs_dynamic_cells"] == 0

    poisson_learned = sorted(poisson_audit["learned_pair"], key=lambda row: -row["tau"])
    poisson_classical = sorted(
        (
            row
            for row in poisson_audit["production_controls"]
            if row["method"] == "spectral_q1024"
        ),
        key=lambda row: -row["tau"],
    )
    burgers_classical = sorted(burgers_audit["rows"], key=lambda row: -row["fom_tau"])
    burgers_learned = sorted(learned["rows"], key=lambda row: -row["fom_tau"])

    assert len(poisson_learned) == len(poisson_classical) == len(burgers_classical) == 3
    assert sum(row["supported"] for row in poisson_learned) == 1
    assert sum(row["supported_speedup"] for row in burgers_classical) == 1
    assert all(
        not summary["supported_film_speedup"]
        for row in burgers_learned
        for summary in row["summaries"].values()
    )
    assert all(
        row["learned_true_residual_max"] <= row["tau"]
        and row["zero_true_residual_max"] <= row["tau"]
        for row in poisson_learned
    )
    assert all(
        row["max_returned_residual"] <= row["fom_tau"] for row in burgers_classical
    )

    return {
        "poisson": poisson,
        "poisson_audit": poisson_audit,
        "poisson_learned": poisson_learned,
        "poisson_classical": poisson_classical,
        "burgers": burgers,
        "burgers_audit": burgers_audit,
        "burgers_classical": burgers_classical,
        "burgers_learned_raw": learned,
        "burgers_learned_audit": learned_audit,
        "burgers_learned": burgers_learned,
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(colors="#374151")


def _label_bars(ax: plt.Axes, bars, values: list[float], supported: list[bool]) -> None:
    span = max(values) if values else 1.0
    for bar, value, is_supported in zip(bars, values, supported, strict=True):
        marker = "*" if is_supported else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025 * span,
            f"{value:.3f}x{marker}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
            fontweight="bold" if is_supported else "normal",
        )


def render_n2048_plots(n2048: dict) -> None:
    """Render the two static Markdown figures from the audited N=2048 rows."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#1F2937",
            "axes.titlecolor": "#111827",
            "text.color": "#111827",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    blue = "#2F6BFF"
    orange = "#D97706"
    dark = "#1F2937"
    taus = ["1e-6", "1e-8", "1e-10"]
    x = np.arange(len(taus), dtype=float)

    poisson_learned = [row["speedup"] for row in n2048["poisson_learned"]]
    poisson_supported = [row["supported"] for row in n2048["poisson_learned"]]
    burgers_learned = [
        row["summaries"]["cubic"]["speedup_control_over_film_nmrom"]
        for row in n2048["burgers_learned"]
    ]
    burgers_supported = [
        row["summaries"]["cubic"]["supported_film_speedup"]
        for row in n2048["burgers_learned"]
    ]

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    width = 0.34
    p_bars = ax.bar(
        x - width / 2,
        poisson_learned,
        width,
        label="Poisson K8 vs counting CG",
        color=blue,
        edgecolor=dark,
        linewidth=0.8,
    )
    b_bars = ax.bar(
        x + width / 2,
        burgers_learned,
        width,
        label="Burgers FiLM vs cubic",
        color=orange,
        edgecolor=dark,
        linewidth=0.8,
        hatch="//",
    )
    ax.axhline(1.0, color=dark, linewidth=1.2, linestyle="--", label="Parity")
    _style_axis(ax)
    _label_bars(ax, p_bars, poisson_learned, poisson_supported)
    _label_bars(ax, b_bars, burgers_learned, burgers_supported)
    ax.set_xticks(x, taus)
    ax.set_xlabel("FOM tolerance")
    ax.set_ylabel("Baseline time / learned-hybrid time")
    ax.set_ylim(0.0, 1.16)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Genuine learned NM-ROM speedup at N=2048", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "Within-job ratios; values above 1 favor learning. Poisson: 8 A100 cases. Burgers: 4 H200 trajectories. * 95% supported.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.text(
        0.01,
        0.01,
        "Source: independently audited N=2048 Poisson and Burgers run JSONs; no cross-job wall-clock comparison.",
        fontsize=8,
        color="#6B7280",
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.86))
    fig.savefig(
        N2048_LEARNED_FIGURE,
        dpi=180,
        bbox_inches="tight",
        metadata={"Creator": "build_2026_08_18_two_pde_results_and_plot_audit.py"},
    )
    plt.close(fig)

    poisson_classical = [row["speedup_vs_zero"] for row in n2048["poisson_classical"]]
    burgers_classical = [row["speedup_cubic_over_dynamic"] for row in n2048["burgers_classical"]]
    burgers_classical_supported = [
        row["supported_speedup"] for row in n2048["burgers_classical"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8))
    p_bars = axes[0].bar(x, poisson_classical, 0.58, color=blue, edgecolor=dark, linewidth=0.8)
    axes[0].axhline(1.0, color=dark, linewidth=1.2, linestyle="--")
    _style_axis(axes[0])
    _label_bars(axes[0], p_bars, poisson_classical, [False, False, False])
    axes[0].set_xticks(x, taus)
    axes[0].set_xlabel("FOM tolerance")
    axes[0].set_ylabel("Zero-start CG time / spectral time")
    axes[0].set_ylim(0.0, max(poisson_classical) * 1.18)
    axes[0].set_title("Poisson spectral q=1024", fontsize=12, fontweight="bold")

    b_bars = axes[1].bar(
        x,
        burgers_classical,
        0.58,
        color=orange,
        edgecolor=dark,
        linewidth=0.8,
        hatch="//",
    )
    axes[1].axhline(1.0, color=dark, linewidth=1.2, linestyle="--")
    _style_axis(axes[1])
    _label_bars(axes[1], b_bars, burgers_classical, burgers_classical_supported)
    axes[1].set_xticks(x, taus)
    axes[1].set_xlabel("FOM tolerance")
    axes[1].set_ylabel("Cubic-history time / corrected time")
    axes[1].set_ylim(0.0, 1.72)
    axes[1].set_title("Burgers residual + Helmholtz correction", fontsize=12, fontweight="bold")

    fig.suptitle("Classical solver speedups at N=2048", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "Independent within-job panels with different vertical scales. Above 1 favors the classical warm start; * 95% supported.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.text(
        0.01,
        0.01,
        "Poisson q=1024 is eligible at all tolerances. Burgers wall time charges 50 residual evaluations and 50 exact Helmholtz inverses.",
        fontsize=8,
        color="#6B7280",
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.86), w_pad=2.4)
    fig.savefig(
        N2048_CLASSICAL_FIGURE,
        dpi=180,
        bbox_inches="tight",
        metadata={"Creator": "build_2026_08_18_two_pde_results_and_plot_audit.py"},
    )
    plt.close(fig)


def source_specs() -> list[dict]:
    common = {
        "engine": "local-json/python",
        "executed_at": "2026-08-20T00:00:00Z",
    }
    return [
        {
            "id": "accuracy_jsons",
            "label": "Archived Poisson and Burgers accuracy JSONs",
            "path": repo_relative(PATHS["poisson_accuracy"]),
            "query": {
                **common,
                "sql": "SELECT pde, route, mean_error, ceiling FROM archived_accuracy_json WHERE K = 8 AND N = 64",
                "description": "Selects K=8, N=64 decoder-ceiling, full weak, m=256 EQ weak, and POD comparison rows for both PDEs.",
                "tables_used": [repo_relative(PATHS["poisson_accuracy"]), repo_relative(PATHS["poisson_pod"]), repo_relative(PATHS["burgers_accuracy"])],
                "filters": ["Poisson-2D", "Burgers-2D", "K=8", "N=64"],
                "metric_definitions": {"ratio_to_ceiling": "mean ROM relative-L2 error divided by the corresponding nonlinear decoder oracle error"},
            },
        },
        {
            "id": "objective_json",
            "label": "Poisson residual-objective sweep",
            "path": repo_relative(PATHS["poisson_objectives"]),
            "query": {
                **common,
                "sql": "SELECT objective, rom_rel_l2_mean, rom_rel_l2_med, rom_rel_l2_max FROM poisson_objective_rows WHERE init = 'nearest'",
                "description": "Selects nearest-initialized pointwise, weak, filtered, Ritz, and low-pass objective rows.",
                "tables_used": [repo_relative(PATHS["poisson_objectives"])],
                "filters": ["init=nearest"],
                "metric_definitions": {"mean_error": "mean relative-L2 field error over the recorded test sources"},
            },
        },
        {
            "id": "quadrature_jsons",
            "label": "Poisson and Burgers quadrature timing sweeps",
            "path": repo_relative(PATHS["poisson_quadrature"]),
            "query": {
                **common,
                "sql": "SELECT pde, M, m, mean_error, time_ms, eq_fit FROM quadrature_rows WHERE M = 64 AND pool IN ('grid','full')",
                "description": "Selects M=64 grid-supported empirical-quadrature rows at m=M through m=16M and the full-grid control for both PDEs.",
                "tables_used": [repo_relative(PATHS["poisson_quadrature"]), repo_relative(PATHS["burgers_quadrature"])],
                "filters": ["M=64", "grid-supported EQ", "K=8"],
                "metric_definitions": {"error_ratio_to_full": "row mean error divided by the full-grid weak row error from the same timing cell"},
            },
        },
        {
            "id": "optimizer_json",
            "label": "Poisson latent-optimizer diagnostic",
            "path": repo_relative(PATHS["optimizer"]),
            "query": {
                **common,
                "sql": "SELECT k, arm, ratio_mean, ratio_med, n_blown FROM optimizer_rows WHERE N = 64 AND arm IN ('base','tr')",
                "description": "Compares the base latent LM and trust-region repair over the shared frozen source set.",
                "tables_used": [repo_relative(PATHS["optimizer"])],
                "filters": ["N=64", "arms=base,tr"],
                "metric_definitions": {"ratio_mean": "mean ROM error divided by mean decoder-ceiling error", "ratio_median": "median ROM error divided by median decoder-ceiling error"},
            },
        },
        {
            "id": "cost_json",
            "label": "Single-GPU cost-to-tolerance consolidation",
            "path": repo_relative(PATHS["cost"]),
            "query": {
                **common,
                "sql": "SELECT coord.N, coord.err_rel_l2, coord.time_ms, MIN(fom.time_ms) FROM consolidated_coord coord JOIN consolidated_fom fom ON fom.N = coord.N AND fom.err_rel_l2 <= coord.err_rel_l2 WHERE coord.pde = 'poisson2d' GROUP BY coord.N, coord.err_rel_l2, coord.time_ms",
                "description": "For each Poisson mesh, selects the consolidated K=16 coordinate ROM at tau=1e-2 and the cheapest same-GPU FOM row with error no larger than the ROM error.",
                "tables_used": [repo_relative(PATHS["cost"])],
                "filters": ["gpu=NVIDIA A100 80GB PCIe", "pde=poisson2d", "arm=consolidated/fom_consolidated"],
                "metric_definitions": {"speedup": "accuracy-matched FOM time divided by total coordinate-ROM time"},
            },
        },
        {
            "id": "hybrid_json",
            "label": "ROM-to-FOM warm-start consolidation and verifier",
            "path": repo_relative(PATHS["hybrid"]),
            "query": {
                **common,
                "sql": "SELECT pde, N, MAX(speedup_vs_fom) FROM hybrid_rows WHERE run_role = 'consolidated' AND fom_tau = 1e-6 GROUP BY pde, N",
                "description": "At the shared FOM tolerance, selects the best observed consolidated ROM tolerance per PDE and mesh; the independent verifier is rerun during report generation.",
                "tables_used": [repo_relative(PATHS["hybrid"]), repo_relative(PATHS["hybrid_verifier"])],
                "filters": ["run_role=consolidated", "fom_tau=1e-6"],
                "metric_definitions": {"speedup": "baseline FOM time divided by ROM construction plus FOM correction time"},
            },
        },
    ]


def build_markdown(data: dict) -> str:
    accuracy = data["accuracy_detail"]
    q = data["quadrature_detail"]
    opt = data["optimizer_detail"]
    cost = data["cost"]
    hybrid = data["hybrid_detail"]
    objective = data["objectives"]
    verifier = data["verifier"]
    counts = data["cost_counts"]
    n2048 = data["n2048"]

    p_full = next(row for row in accuracy if row["pde"] == "Poisson-2D" and row["route"] == "Full-grid weak")
    p_eq = next(row for row in accuracy if row["pde"] == "Poisson-2D" and row["route"] == "EQ weak")
    b_full = next(row for row in accuracy if row["pde"] == "Burgers-2D" and row["route"] == "Full-grid weak")
    b_eq = next(row for row in accuracy if row["pde"] == "Burgers-2D" and row["route"] == "EQ weak")
    p_m1 = next(row for row in q if row["pde"] == "Poisson-2D" and row["m"] == row["M"])
    p_m4 = next(row for row in q if row["pde"] == "Poisson-2D" and row["m"] == 4 * row["M"])
    b_m1 = next(row for row in q if row["pde"] == "Burgers-2D" and row["m"] == row["M"])
    b_m4 = next(row for row in q if row["pde"] == "Burgers-2D" and row["m"] == 4 * row["M"])
    total_base_divergent = sum(row["base_divergent"] for row in opt)
    total_trust_divergent = sum(row["trust_divergent"] for row in opt)
    best_cost = max(cost, key=lambda row: row["speedup"])
    best_p_hybrid = max(row["speedup"] for row in hybrid if row["pde"] == "Poisson-2D")
    best_b_hybrid = max(row["speedup"] for row in hybrid if row["pde"] == "Burgers-2D")
    pointwise = next(row for row in objective if row["objective_key"] == "fd")
    weak = next(row for row in objective if row["objective_key"] == "spec_a1_M64")

    lines: list[str] = []
    add = lines.append
    add("# Poisson-2D and Burgers-2D results and plot audit through August 18, with N=2048 hybrid extension")
    add("")
    add(
        "This generated technical report consolidates the archived Poisson-2D and Burgers-2D evidence available through August 18 and appends the final, independently audited N=2048 hybrid extension under the hybrid section. "
        "The accuracy and method conclusions are final within the archived cells; the old cost-to-tolerance curve is explicitly provisional, and the N=2048 results use separate fresh-seed confirmation panels."
    )
    add("")
    add(f"[Open the archived through-August-18 HTML companion]({REPORT_STEM}.html). The N=2048 extension and its static plots are embedded below in this Markdown report.")
    add("")
    add("## Technical summary")
    add("")
    add(
        f"The weak nonlinear ROM is close to its own decoder ceiling on both PDEs: full-grid weak is {p_full['ratio_to_ceiling']:.3f}x and EQ weak is {p_eq['ratio_to_ceiling']:.3f}x the Poisson ceiling; the corresponding Burgers ratios are {b_full['ratio_to_ceiling']:.3f}x and {b_eq['ratio_to_ceiling']:.3f}x. "
        f"For Poisson, replacing the pointwise residual with the weak objective reduces mean error from {sci(pointwise['mean_error'])} to {sci(weak['mean_error'])}."
    )
    add("")
    add(
        f"Quadrature is the sharpest shared implementation inconsistency. At m=M the error is {p_m1['error_ratio_to_full']:.2f}x full-grid for Poisson and {b_m1['error_ratio_to_full']:.2f}x for Burgers; at m=4M it is {p_m4['error_ratio_to_full']:.2f}x and {b_m4['error_ratio_to_full']:.2f}x. "
        f"The optimizer diagnostic also removes {total_base_divergent - total_trust_divergent} divergent source-solves while leaving {total_trust_divergent} after the trust-region repair, so the earlier latent-dimension stall plot must remain withdrawn."
    )
    add("")
    add(
        f"The cost stories do not yet reconcile. The provisional standalone Poisson curve reaches {best_cost['speedup']:.3f}x at N={best_cost['N']}, but after charging ROM construction and FOM correction the best observed original hybrid is only {best_p_hybrid:.3f}x for Poisson and {best_b_hybrid:.3f}x for Burgers. "
        f"Burgers has {counts['burgers_uncensored']} uncensored consolidated coordinate rows out of {counts['burgers_total']} in the archived strict cost cell."
    )
    p2048_supported = next(row for row in n2048["poisson_learned"] if row["supported"])
    b2048_supported = next(row for row in n2048["burgers_classical"] if row["supported_speedup"])
    add("")
    add(
        f"The audited N=2048 extension narrows the learned crossover rather than establishing a broad win. The genuine Poisson K=8 hybrid is supported only at tolerance {p2048_supported['tau']:.0e}, where it reaches {p2048_supported['speedup']:.3f}x over counting CG; the genuine Burgers FiLM arm has no supported win at any tested tolerance. "
        f"The practical Burgers result is classical: at tolerance {b2048_supported['fom_tau']:.0e}, the charged residual-plus-Helmholtz correction reaches {b2048_supported['speedup_cubic_over_dynamic']:.3f}x over cubic-history FOM."
    )
    add("")
    add("## Accuracy and objective evidence")
    add("")
    add("The nonlinear weak-form route nearly reaches its learned-manifold ceiling on both PDEs, while the matched POD routes remain much farther away. Ratios use each PDE's own decoder ceiling and are therefore comparable as optimization headroom, not as absolute field accuracy.")
    add("")
    lines.extend(
        md_table(
            ["PDE", "Route", "Mean relative-L2", "Ratio to nonlinear ceiling"],
            [[row["pde"], row["route"], sci(row["mean_error"]), f"{row['ratio_to_ceiling']:.3f}x"] for row in accuracy],
        )
    )
    add("")
    add("The Poisson objective sweep isolates the residual-definition issue:")
    add("")
    lines.extend(
        md_table(
            ["Objective", "Mean", "Median", "Maximum", "Sources"],
            [[row["objective"], sci(row["mean_error"]), sci(row["median_error"]), sci(row["max_error"]), str(row["n_sources"])] for row in objective],
        )
    )
    add("")
    add("## Quadrature tradeoff")
    add("")
    add("The common operating point is m around four times M. Below it, fit quality and state accuracy deteriorate sharply; above it, accuracy has largely saturated while online cost continues to rise.")
    add("")
    lines.extend(
        md_table(
            ["PDE", "m/M", "Mean error", "Error/full", "Online ms", "Time/full", "EQ fit"],
            [
                [
                    row["pde"], row["m_label"], sci(row["mean_error"]), f"{row['error_ratio_to_full']:.3f}x",
                    dec(row["time_ms"], 1), f"{row['time_ratio_to_full']:.3f}x", "--" if row["eq_fit"] is None else sci(row["eq_fit"]),
                ]
                for row in q
            ],
        )
    )
    add("")
    add("## Optimizer inconsistency")
    add("")
    add("Mean-only reporting created the apparent k-specific stalls. The medians stay near the decoder ceiling, while a few divergent base-LM solves dominate the means; trust-region globalization removes the recorded divergences.")
    add("")
    lines.extend(
        md_table(
            ["k", "Base mean", "Base median", "Base divergent", "Trust mean", "Trust median", "Trust divergent"],
            [[str(row["k"]), dec(row["base_mean"]), dec(row["base_median"]), str(row["base_divergent"]), dec(row["trust_mean"]), dec(row["trust_median"]), str(row["trust_divergent"])] for row in opt],
        )
    )
    add("")
    add("## Cost and hybrid evidence")
    add("")
    add("The standalone cost-to-tolerance curve is retained because it is same-GPU consolidated evidence, but it remains provisional: its formal cell review was not completed. The hybrid table is the stronger negative for the specific deployment claim because cost and correction come from the same solver invocation.")
    add("")
    lines.extend(
        md_table(
            ["Poisson N", "ROM error", "ROM ms", "Matched FOM error", "Matched FOM ms", "Speedup", "Status"],
            [[str(row["N"]), sci(row["coord_error"]), dec(row["coord_time_ms"]), sci(row["fom_error"]), dec(row["fom_time_ms"]), speed(row["speedup"]), "provisional"] for row in cost],
        )
    )
    add("")
    lines.extend(
        md_table(
            ["PDE", "N", "ROM tau", "FOM tau", "Baseline ms", "Hybrid ms", "Speedup", "Baseline/ROM/extrap iterations"],
            [
                [
                    row["pde"], str(row["N"]), sci(row["rom_tau"]), sci(row["fom_tau"]), dec(row["fom_time_ms"]), dec(row["hybrid_time_ms"]), speed(row["speedup"]),
                    f"{row['baseline_iters']:.2f}/{row['rom_iters']:.2f}/" + ("--" if row["extrap_iters"] is None else f"{row['extrap_iters']:.2f}"),
                ]
                for row in hybrid
            ],
        )
    )
    add("")
    add(f"The independent warm-start verifier was rerun during generation: {verifier['passed']} checks passed and {verifier['failed']} failed.")
    add("")
    add("### Audited N=2048 extension")
    add("")
    add(
        "The extension uses fresh seeds and balanced within-job timing. Poisson K=8 is a genuine learned NM-ROM warm start followed by counting CG. Burgers FiLM is also a genuine weak NM-ROM; the residual-plus-Helmholtz Burgers arm is classical, nonlearned, and not an NM-ROM. Speedup is always the matched within-job baseline time divided by candidate time."
    )
    add("")
    add(
        "The learned comparison below shows the central limitation: only the loose-tolerance Poisson row is statistically supported above parity. Burgers FiLM remains below cubic-history parity at all three tolerances, and its same-job comparisons against the classical correction also produce no supported FiLM win."
    )
    add("")
    add("![Genuine learned NM-ROM speedup at N=2048](figs/two-pde-n2048-learned-speedup.png)")
    add("")
    lines.extend(
        md_table(
            ["FOM tolerance", "Poisson K8 ms", "Poisson zero-CG ms", "Poisson speedup [95% CI]", "Poisson verdict"],
            [
                [
                    f"{row['tau']:.0e}",
                    dec(row["learned_median_ms"]),
                    dec(row["zero_median_ms"]),
                    f"{row['speedup']:.3f}x [{row['speedup_ci95'][0]:.3f}, {row['speedup_ci95'][1]:.3f}]",
                    "supported faster" if row["supported"] else "unsupported / inconclusive",
                ]
                for row in n2048["poisson_learned"]
            ],
        )
    )
    add("")
    lines.extend(
        md_table(
            ["FOM tolerance", "Cubic / FiLM ms", "Cubic/FiLM", "Dynamic / FiLM ms", "Dynamic/FiLM", "FiLM supported vs either"],
            [
                [
                    f"{row['fom_tau']:.0e}",
                    f"{row['summaries']['cubic']['control_median_ms']:.3f} / {row['summaries']['cubic']['film_nmrom_median_ms']:.3f}",
                    f"{row['summaries']['cubic']['speedup_control_over_film_nmrom']:.3f}x",
                    f"{row['summaries']['dynamic']['control_median_ms']:.3f} / {row['summaries']['dynamic']['film_nmrom_median_ms']:.3f}",
                    f"{row['summaries']['dynamic']['speedup_control_over_film_nmrom']:.3f}x",
                    "yes" if any(summary["supported_film_speedup"] for summary in row["summaries"].values()) else "no",
                ]
                for row in n2048["burgers_learned"]
            ],
        )
    )
    add("")
    add(
        "The production comparison is classical. The Poisson q=1024 spectral warm start is eligible and hundreds of times faster than same-block zero-start CG at every tolerance on this separable rectangle. The Burgers correction is supported only at 1e-6; its tighter rows are inconclusive and should not replace cubic history. The two panels deliberately use separate vertical scales."
    )
    add("")
    add("![Classical solver speedups at N=2048](figs/two-pde-n2048-classical-speedup.png)")
    add("")
    lines.extend(
        md_table(
            ["FOM tolerance", "Poisson q1024 ms", "Poisson zero-CG ms", "Poisson speedup", "Burgers cubic / corrected ms", "Burgers speedup", "Burgers paired saving ms [95% CI]", "Burgers verdict"],
            [
                [
                    f"{p_row['tau']:.0e}",
                    dec(p_row["median_ms"]),
                    dec(p_row["median_ms"] * p_row["speedup_vs_zero"]),
                    f"{p_row['speedup_vs_zero']:.1f}x",
                    f"{b_row['cubic_median_ms']:.3f} / {b_row['dynamic_median_ms']:.3f}",
                    f"{b_row['speedup_cubic_over_dynamic']:.3f}x",
                    f"{b_row['paired_saving_median_ms']:.3f} [{b_row['paired_saving_trajectory_cluster_95ci_ms'][0]:.3f}, {b_row['paired_saving_trajectory_cluster_95ci_ms'][1]:.3f}]",
                    "supported faster" if b_row["supported_speedup"] else "unsupported / inconclusive",
                ]
                for p_row, b_row in zip(n2048["poisson_classical"], n2048["burgers_classical"], strict=True)
            ],
        )
    )
    add("")
    p_records = n2048["poisson_audit"]["record_counts"]
    b_audit = n2048["burgers_audit"]
    b_ref = b_audit["reference"]
    add(
        f"The Poisson audit covers {p_records['learned']} learned/zero and {p_records['production']} production-control records with {p_records['outliers']} timing outliers. The Burgers classical audit covers {b_audit['timing_record_count']} timed records and {b_audit['burn_record_count']} burns. Its two exact-Helmholtz reference routes pass with worst residual {b_ref['max_actual_outer_relative_residual']:.3e}, maximum step disagreement {b_ref['max_step_relative_field_difference']:.3e}, and maximum trajectory disagreement {b_ref['max_trajectory_relative_field_difference']:.3e}."
    )
    add("")
    add(
        "These timings are warmed, compiled steady-state costs. Poisson used eight timed A100-80GB cases; each Burgers panel used four H200 trajectories on its own seed and job. Absolute milliseconds are never compared across those jobs or GPU types."
    )
    add("")
    add("## What to make consistent next")
    add("")
    add("1. Apply the same trust-region latent solver to the frozen Poisson and Burgers cases, then report mean, median, worst case, and divergent-count together.")
    add("2. Use one quadrature policy per comparison: M comfortably above k, m near four times M, decoder-output NNLS weights, and a hyper-reduced cold start.")
    add("3. Rebuild cost-to-tolerance with cost and accuracy from the same invocation, saved timing repetitions, GPU burn-in, and the FOM tolerance printed in every plot caption.")
    add("4. Compare learned warm starts with the strongest classical history/extrapolation arm in the same job; iteration savings alone are insufficient if construction cost dominates.")
    add("5. Finish the archived cost-cell audit before treating the standalone Poisson crossover as a claim, and do not draw a Burgers strict frontier until an uncensored point exists.")
    add("")
    add("## Scope, limitations, and source context")
    add("")
    add("This report preserves the archived evidence available through August 18 and adds only the final N=2048 hybrid extension requested above; it does not import the intervening N<=1024 architecture panels. Accuracy rows are not all from one executable and therefore support method-level consistency checks rather than a single pooled benchmark. The provisional archived Poisson cost curve is descriptive, not inferential; the N=2048 rows are separate fresh-seed confirmations with their own within-job baselines and clustered uncertainty.")
    add("")
    add("Every table, prose quantity, and static N=2048 plot above is generated by `reports/build_2026_08_18_two_pde_results_and_plot_audit.py` from the archived JSONs and the independently audited N=2048 raw/audit JSONs. The linked HTML companion remains the archived through-August-18 interactive surface; the requested N=2048 extension lives in this Markdown report.")
    add("")
    return "\n".join(lines)


def build_artifact(data: dict, sources: list[dict]) -> dict:
    accuracy = data["accuracy_detail"]
    opt = data["optimizer_detail"]
    cost = data["cost"]
    hybrid = data["hybrid_detail"]
    q = data["quadrature_detail"]
    verifier = data["verifier"]
    counts = data["cost_counts"]
    objective = data["objectives"]

    p_eq = next(row for row in accuracy if row["pde"] == "Poisson-2D" and row["route"] == "EQ weak")
    b_eq = next(row for row in accuracy if row["pde"] == "Burgers-2D" and row["route"] == "EQ weak")
    best_cost = max(cost, key=lambda row: row["speedup"])
    best_p_hybrid = max(row["speedup"] for row in hybrid if row["pde"] == "Poisson-2D")
    best_b_hybrid = max(row["speedup"] for row in hybrid if row["pde"] == "Burgers-2D")
    total_base_divergent = sum(row["base_divergent"] for row in opt)
    total_trust_divergent = sum(row["trust_divergent"] for row in opt)

    summary_rows = [{
        "poisson_eq_ceiling_ratio": p_eq["ratio_to_ceiling"],
        "burgers_eq_ceiling_ratio": b_eq["ratio_to_ceiling"],
        "poisson_provisional_speedup": best_cost["speedup"],
        "poisson_hybrid_speedup": best_p_hybrid,
        "burgers_hybrid_speedup": best_b_hybrid,
        "divergences_removed": total_base_divergent - total_trust_divergent,
        "burgers_uncensored": counts["burgers_uncensored"],
        "verifier_failed": verifier["failed"],
    }]

    charts = [
        {
            "id": "accuracy_headroom",
            "title": "Nonlinear ROM error relative to the decoder ceiling",
            "subtitle": "Values near one indicate that optimization and quadrature add little error beyond representation.",
            "headerMarkdown": "Both weak-form routes remain close to the learned-manifold ceiling; empirical quadrature adds modest headroom loss.",
            "type": "bar",
            "dataset": "accuracy_nonlinear",
            "sourceId": "accuracy_jsons",
            "encodings": {
                "x": {"field": "pde", "type": "nominal", "label": "PDE"},
                "y": {"field": "ratio_to_ceiling", "type": "quantitative", "label": "Error / decoder ceiling", "format": "number"},
                "color": {"field": "route", "type": "nominal", "label": "Route"},
                "tooltip": [
                    {"field": "mean_error", "type": "quantitative", "label": "Mean relative-L2", "format": "number"},
                    {"field": "ceiling", "type": "quantitative", "label": "Decoder ceiling", "format": "number"},
                ],
            },
            "referenceLines": [{"value": 1.0, "label": "Decoder ceiling"}],
            "valueFormat": "number",
        },
        {
            "id": "objective_comparison",
            "title": "Poisson error by residual objective",
            "subtitle": "The pointwise finite-difference residual is the outlier; smooth weak or filtered objectives approach the decoder floor.",
            "headerMarkdown": "The objective definition, not decoder capacity, explains the original Poisson failure.",
            "type": "horizontalBar",
            "dataset": "objectives",
            "sourceId": "objective_json",
            "encodings": {
                "x": {"field": "objective", "type": "nominal", "label": "Objective"},
                "y": {"field": "mean_error", "type": "quantitative", "label": "Mean relative-L2", "format": "number"},
                "tooltip": [
                    {"field": "median_error", "type": "quantitative", "label": "Median", "format": "number"},
                    {"field": "max_error", "type": "quantitative", "label": "Maximum", "format": "number"},
                    {"field": "n_sources", "type": "quantitative", "label": "Sources", "format": "number"},
                ],
            },
            "valueFormat": "number",
        },
        {
            "id": "quadrature_ratio",
            "title": "Quadrature error relative to the full-grid weak solve",
            "subtitle": "m=M is inadequate on both PDEs; accuracy is near saturation by m=4M.",
            "headerMarkdown": "A common m near four times M is the smallest defensible operating region in these archived sweeps.",
            "type": "bar",
            "dataset": "quadrature",
            "sourceId": "quadrature_jsons",
            "encodings": {
                "x": {"field": "m_label", "type": "ordinal", "label": "Quadrature size"},
                "y": {"field": "error_ratio_to_full", "type": "quantitative", "label": "Error / full-grid error", "format": "number"},
                "color": {"field": "pde", "type": "nominal", "label": "PDE"},
                "tooltip": [
                    {"field": "mean_error", "type": "quantitative", "label": "Mean error", "format": "number"},
                    {"field": "time_ms", "type": "quantitative", "label": "Online ms", "format": "number"},
                    {"field": "eq_fit", "type": "quantitative", "label": "EQ relative fit", "format": "number"},
                ],
            },
            "referenceLines": [{"value": 1.0, "label": "Full-grid accuracy"}],
            "valueFormat": "number",
        },
        {
            "id": "optimizer_mean",
            "title": "Mean error-to-ceiling ratio across latent dimension",
            "subtitle": "The base-LM spikes are caused by a few divergent source solves; trust-region globalization removes them.",
            "headerMarkdown": "This replaces the retracted k-stall plot; use the adjacent table for medians and divergent counts.",
            "type": "bar",
            "dataset": "optimizer_chart",
            "sourceId": "optimizer_json",
            "encodings": {
                "x": {"field": "k_label", "type": "ordinal", "label": "Latent dimension k"},
                "y": {"field": "ratio_mean", "type": "quantitative", "label": "Mean error / mean ceiling", "format": "number"},
                "color": {"field": "solver", "type": "nominal", "label": "Latent solver"},
                "tooltip": [
                    {"field": "ratio_median", "type": "quantitative", "label": "Median ratio", "format": "number"},
                    {"field": "divergent_cases", "type": "quantitative", "label": "Divergent sources", "format": "number"},
                    {"field": "n_sources", "type": "quantitative", "label": "Sources", "format": "number"},
                ],
            },
            "referenceLines": [{"value": 1.0, "label": "Decoder ceiling"}],
            "valueFormat": "number",
        },
        {
            "id": "cost_speedup",
            "title": "Provisional Poisson standalone speedup at matched accuracy",
            "subtitle": "Same A100-80GB consolidation; faster-than-FOM begins only at the larger meshes in this archived cell.",
            "headerMarkdown": "Treat this curve as provisional until the unfinished cost-cell audit is closed.",
            "type": "bar",
            "dataset": "cost",
            "sourceId": "cost_json",
            "encodings": {
                "x": {"field": "N_label", "type": "ordinal", "label": "Mesh N"},
                "y": {"field": "speedup", "type": "quantitative", "label": "Matched FOM time / ROM time", "format": "number"},
                "tooltip": [
                    {"field": "coord_error", "type": "quantitative", "label": "ROM error", "format": "number"},
                    {"field": "coord_time_ms", "type": "quantitative", "label": "ROM ms", "format": "number"},
                    {"field": "fom_error", "type": "quantitative", "label": "Matched FOM error", "format": "number"},
                    {"field": "fom_time_ms", "type": "quantitative", "label": "Matched FOM ms", "format": "number"},
                ],
            },
            "referenceLines": [{"value": 1.0, "label": "Parity"}],
            "valueFormat": "number",
        },
        {
            "id": "hybrid_speedup",
            "title": "Best observed original ROM-to-FOM hybrid speedup",
            "subtitle": "At the shared FOM tolerance, neither PDE crosses parity after charging ROM construction and correction.",
            "headerMarkdown": "Standalone ROM speed does not transfer automatically to end-to-end hybrid speed.",
            "type": "bar",
            "dataset": "hybrid",
            "sourceId": "hybrid_json",
            "encodings": {
                "x": {"field": "N_label", "type": "ordinal", "label": "Mesh N"},
                "y": {"field": "speedup", "type": "quantitative", "label": "Baseline FOM time / hybrid time", "format": "number"},
                "color": {"field": "pde", "type": "nominal", "label": "PDE"},
                "tooltip": [
                    {"field": "rom_tau", "type": "quantitative", "label": "ROM tolerance", "format": "number"},
                    {"field": "fom_time_ms", "type": "quantitative", "label": "Baseline FOM ms", "format": "number"},
                    {"field": "hybrid_time_ms", "type": "quantitative", "label": "Hybrid ms", "format": "number"},
                    {"field": "baseline_iters", "type": "quantitative", "label": "Baseline iterations", "format": "number"},
                    {"field": "rom_iters", "type": "quantitative", "label": "Post-ROM iterations", "format": "number"},
                ],
            },
            "referenceLines": [{"value": 1.0, "label": "Parity"}],
            "valueFormat": "number",
        },
    ]

    tables = [
        {
            "id": "accuracy_table",
            "title": "Accuracy detail",
            "subtitle": "Mean relative-L2 error and optimization headroom at K=8, N=64.",
            "dataset": "accuracy_detail",
            "sourceId": "accuracy_jsons",
            "density": "compact",
            "columns": [
                {"field": "pde", "label": "PDE"},
                {"field": "route", "label": "Route"},
                {"field": "mean_error", "label": "Mean error", "format": "number", "align": "right"},
                {"field": "ratio_to_ceiling", "label": "Error / ceiling", "format": "number", "align": "right"},
                {"field": "pod_floor", "label": "POD projection floor", "format": "number", "align": "right"},
            ],
        },
        {
            "id": "quadrature_table",
            "title": "Quadrature accuracy and online cost",
            "subtitle": "The two timing cells use their own full-grid controls; absolute milliseconds are not compared across PDEs.",
            "dataset": "quadrature",
            "sourceId": "quadrature_jsons",
            "density": "compact",
            "columns": [
                {"field": "pde", "label": "PDE"},
                {"field": "m_label", "label": "m / M"},
                {"field": "mean_error", "label": "Mean error", "format": "number", "align": "right"},
                {"field": "error_ratio_to_full", "label": "Error / full", "format": "number", "align": "right"},
                {"field": "time_ms", "label": "Online ms", "format": "number", "align": "right"},
                {"field": "time_ratio_to_full", "label": "Time / full", "format": "number", "align": "right"},
                {"field": "eq_fit", "label": "EQ fit", "format": "number", "align": "right"},
            ],
        },
        {
            "id": "optimizer_table",
            "title": "Mean, median, and divergent-count diagnostic",
            "subtitle": "All statistics are shown together to prevent mean-only artifacts.",
            "dataset": "optimizer_detail",
            "sourceId": "optimizer_json",
            "density": "compact",
            "columns": [
                {"field": "k", "label": "k", "format": "number", "align": "right"},
                {"field": "base_mean", "label": "Base mean", "format": "number", "align": "right"},
                {"field": "base_median", "label": "Base median", "format": "number", "align": "right"},
                {"field": "base_divergent", "label": "Base divergent", "format": "number", "align": "right"},
                {"field": "trust_mean", "label": "Trust mean", "format": "number", "align": "right"},
                {"field": "trust_median", "label": "Trust median", "format": "number", "align": "right"},
                {"field": "trust_divergent", "label": "Trust divergent", "format": "number", "align": "right"},
            ],
        },
        {
            "id": "cost_table",
            "title": "Provisional Poisson matched-accuracy rows",
            "subtitle": "The selected FOM row is the cheapest same-GPU row no less accurate than the coordinate ROM.",
            "dataset": "cost",
            "sourceId": "cost_json",
            "density": "compact",
            "columns": [
                {"field": "N", "label": "N", "format": "number", "align": "right"},
                {"field": "coord_error", "label": "ROM error", "format": "number", "align": "right"},
                {"field": "coord_time_ms", "label": "ROM ms", "format": "number", "align": "right"},
                {"field": "fom_error", "label": "FOM error", "format": "number", "align": "right"},
                {"field": "fom_time_ms", "label": "FOM ms", "format": "number", "align": "right"},
                {"field": "speedup", "label": "Speedup", "format": "number", "align": "right"},
                {"field": "status", "label": "Status"},
            ],
        },
        {
            "id": "hybrid_table",
            "title": "Original hybrid detail at the shared FOM tolerance",
            "subtitle": "Best observed ROM tolerance per PDE and mesh; descriptive envelope, not a preregistered selection.",
            "dataset": "hybrid",
            "sourceId": "hybrid_json",
            "density": "compact",
            "columns": [
                {"field": "pde", "label": "PDE"},
                {"field": "N", "label": "N", "format": "number", "align": "right"},
                {"field": "rom_tau", "label": "ROM tau", "format": "number", "align": "right"},
                {"field": "fom_time_ms", "label": "Baseline ms", "format": "number", "align": "right"},
                {"field": "hybrid_time_ms", "label": "Hybrid ms", "format": "number", "align": "right"},
                {"field": "speedup", "label": "Speedup", "format": "number", "align": "right"},
                {"field": "baseline_iters", "label": "Baseline iters", "format": "number", "align": "right"},
                {"field": "rom_iters", "label": "ROM iters", "format": "number", "align": "right"},
                {"field": "extrap_iters", "label": "Extrap iters", "format": "number", "align": "right"},
            ],
        },
    ]

    source_stubs = [{"id": source["id"], "label": source["label"], "path": source["path"]} for source in sources]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Poisson-2D and Burgers-2D results and plot audit through August 18\n\nArchived evidence only; later hybrid and architecture results are intentionally excluded."},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical summary\n\n"
                f"The weak nonlinear ROM stays close to its decoder ceiling on both PDEs: the EQ-to-ceiling ratios are **{p_eq['ratio_to_ceiling']:.3f}x** for Poisson and **{b_eq['ratio_to_ceiling']:.3f}x** for Burgers. "
                f"The trust-region diagnostic removes **{total_base_divergent - total_trust_divergent}** recorded divergent source-solves. "
                f"The provisional standalone Poisson curve reaches **{best_cost['speedup']:.3f}x**, yet the best observed original hybrid remains below parity at **{best_p_hybrid:.3f}x** for Poisson and **{best_b_hybrid:.3f}x** for Burgers."
            ),
        },
        {"id": "summary_metrics", "type": "metric-strip", "cardIds": ["poisson_headroom", "burgers_headroom", "poisson_cost", "poisson_hybrid", "burgers_hybrid"]},
        {"id": "accuracy_intro", "type": "markdown", "body": "## Accuracy and residual objectives\n\nThe weak form fixes the dominant accuracy failure. Ratios to the decoder ceiling isolate optimization and quadrature headroom from representation error."},
        {"id": "accuracy_chart_block", "type": "chart", "chartId": "accuracy_headroom"},
        {"id": "accuracy_table_block", "type": "table", "tableId": "accuracy_table"},
        {"id": "objective_chart_block", "type": "chart", "chartId": "objective_comparison"},
        {"id": "quadrature_intro", "type": "markdown", "body": "## Quadrature tradeoff\n\nThe shared inconsistency is undersampling: m=M is not a defensible operating point, while m near four times M is the first common region close to full-grid accuracy."},
        {"id": "quadrature_chart_block", "type": "chart", "chartId": "quadrature_ratio"},
        {"id": "quadrature_table_block", "type": "table", "tableId": "quadrature_table"},
        {"id": "optimizer_intro", "type": "markdown", "body": "## Optimizer robustness\n\nThe withdrawn k-stall conclusion came from mean-only aggregation. The replacement view keeps means, medians, and divergent counts together."},
        {"id": "optimizer_chart_block", "type": "chart", "chartId": "optimizer_mean"},
        {"id": "optimizer_table_block", "type": "table", "tableId": "optimizer_table"},
        {"id": "cost_intro", "type": "markdown", "body": "## Cost-to-tolerance and deployment\n\nThe standalone Poisson curve is provisional. The original hybrid result is the stronger deployment test through August 18 because it charges construction and correction in the same invocation."},
        {"id": "cost_chart_block", "type": "chart", "chartId": "cost_speedup"},
        {"id": "cost_table_block", "type": "table", "tableId": "cost_table"},
        {"id": "hybrid_chart_block", "type": "chart", "chartId": "hybrid_speedup"},
        {"id": "hybrid_table_block", "type": "table", "tableId": "hybrid_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations and robustness\n\n"
                f"The independent hybrid verifier was rerun during generation: **{verifier['passed']} checks passed and {verifier['failed']} failed**. "
                f"The strict Burgers cost cell contains **{counts['burgers_uncensored']} uncensored consolidated coordinate rows out of {counts['burgers_total']}**, so no Burgers strict frontier is shown. "
                "Absolute timings from different jobs are never compared across PDEs."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## What to make consistent next\n\n"
                "1. Apply the trust-region solver to both frozen PDE paths and always report mean, median, worst case, and divergent count together.\n"
                "2. Standardize M, m, the NNLS training snapshots, and cold-start hyper-reduction before comparing cost.\n"
                "3. Rebuild the cost frontier with same-invocation accuracy and time, saved repetition arrays, GPU burn-in, and the reference FOM tolerance in every caption.\n"
                "4. Compare learned warm starts with the strongest classical extrapolation or history arm within one job.\n"
                "5. Finish the archived cost-cell audit before promoting the provisional standalone Poisson crossover."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Does trust-region globalization close the remaining Burgers headroom without increasing online cost materially?\n"
                "- Which part of the hybrid overhead dominates after the cold start is hyper-reduced: latent solve, decode, or correction?\n"
                "- Does a fixed quadrature ratio remain adequate as resolution and latent dimension increase?\n"
                "- Can any learned warm start beat linear or higher-order history on the same trajectories?"
            ),
        },
        {
            "id": "scope",
            "type": "markdown",
            "body": "## Scope and source context\n\nThis report stops at the evidence available through August 18. Accuracy cells support method-level comparisons but are not pooled as one benchmark. All quantitative text, tables, and chart datasets were generated from the archived JSONs; the source controls expose their provenance.",
        },
    ]

    cards = [
        {"id": "poisson_headroom", "description": "EQ error divided by the nonlinear decoder ceiling.", "dataset": "summary", "sourceId": "accuracy_jsons", "metrics": [{"label": "Poisson EQ / ceiling", "field": "poisson_eq_ceiling_ratio", "format": "number"}]},
        {"id": "burgers_headroom", "description": "EQ error divided by the nonlinear decoder ceiling.", "dataset": "summary", "sourceId": "accuracy_jsons", "metrics": [{"label": "Burgers EQ / ceiling", "field": "burgers_eq_ceiling_ratio", "format": "number"}]},
        {"id": "poisson_cost", "description": "Provisional standalone matched-accuracy maximum.", "dataset": "summary", "sourceId": "cost_json", "metrics": [{"label": "Poisson standalone max", "field": "poisson_provisional_speedup", "format": "number"}]},
        {"id": "poisson_hybrid", "description": "Best observed original hybrid at the shared FOM tolerance.", "dataset": "summary", "sourceId": "hybrid_json", "metrics": [{"label": "Poisson hybrid max", "field": "poisson_hybrid_speedup", "format": "number"}]},
        {"id": "burgers_hybrid", "description": "Best observed original hybrid at the shared FOM tolerance.", "dataset": "summary", "sourceId": "hybrid_json", "metrics": [{"label": "Burgers hybrid max", "field": "burgers_hybrid_speedup", "format": "number"}]},
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Poisson-2D and Burgers-2D results and plot audit through August 18",
            "description": "Status-aware technical report from archived August 18 run JSONs.",
            "generatedAt": "2026-08-20T00:00:00Z",
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": source_stubs,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-20T00:00:00Z",
            "status": "ready",
            "datasets": {
                "summary": summary_rows,
                "accuracy_nonlinear": data["accuracy_nonlinear"],
                "accuracy_detail": accuracy,
                "objectives": objective,
                "quadrature": q,
                "optimizer_chart": data["optimizer_chart"],
                "optimizer_detail": opt,
                "cost": cost,
                "hybrid": hybrid,
            },
        },
        "sources": sources,
        "package_info": {"originUrl": f"artifact://{REPORT_STEM}"},
    }


def main() -> None:
    require_files()
    accuracy_nonlinear, accuracy_detail = build_accuracy()
    quadrature_chart, quadrature_detail = build_quadrature()
    optimizer_chart, optimizer_detail = build_optimizer()
    cost, cost_counts = build_cost()
    hybrid_chart, hybrid_detail, verifier = build_hybrid()
    n2048 = build_n2048()
    render_n2048_plots(n2048)
    data = {
        "accuracy_nonlinear": accuracy_nonlinear,
        "accuracy_detail": accuracy_detail,
        "objectives": build_objectives(),
        "quadrature_chart": quadrature_chart,
        "quadrature_detail": quadrature_detail,
        "optimizer_chart": optimizer_chart,
        "optimizer_detail": optimizer_detail,
        "cost": cost,
        "cost_counts": cost_counts,
        "hybrid_chart": hybrid_chart,
        "hybrid_detail": hybrid_detail,
        "verifier": verifier,
        "n2048": n2048,
    }
    sources = source_specs()
    MARKDOWN_OUT.write_text(build_markdown(data))
    ARTIFACT_OUT.write_text(json.dumps(build_artifact(data, sources), indent=2) + "\n")
    print(MARKDOWN_OUT)
    print(ARTIFACT_OUT)


if __name__ == "__main__":
    main()
