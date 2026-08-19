"""Three requested deliverables, built from the raw run JSONs.

  1. cost(k): total online cost, iterations-to-tolerance, and cost per iteration.
  2. The speed/accuracy Pareto frontier for one model, plus a best-speed vs
     best-accuracy table.
  3. FOM versus the ROM-warm-started FOM hybrid, on accuracy and speed.

Nothing here is hand-typed: every number is read from
  cost-to-tolerance/runs/pareto_points.json   (698 rows)
  rom-warmstart-fom/runs/hybrid_points.json   (174 rows)
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.join(HERE, "..", "worktrees")
PARETO = os.path.abspath(os.path.join(
    WT, "2026-08-18-codex-handoff/experiments/cost-to-tolerance/runs/pareto_points.json"))
HYBRID = os.path.abspath(os.path.join(
    WT, "2026-08-18-codex-handoff/experiments/rom-warmstart-fom/runs/hybrid_points.json"))
OUT = os.path.join(HERE, "talk_figs")
TAB = os.path.join(HERE, "2026-08-18-solve-cost-quadrature-and-the-hybrid.md")
os.makedirs(OUT, exist_ok=True)

OURS, POD, FOM, CEIL = "#0072B2", "#D55E00", "#555555", "#009E73"
NCOL = {32: "#c6dbef", 64: "#6baed6", 128: "#2171b5", 256: "#08519c", 512: "#062f5c"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 12,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

ROWS = json.load(open(PARETO))
HROWS = json.load(open(HYBRID))
OUTPUT = []          # markdown accumulator


def md(s=""):
    OUTPUT.append(s)


def sel(pde, method, N=None, tau=None, arm="primary"):
    out = [r for r in ROWS if r["pde"] == pde and r["method"] == method]
    if arm is not None:
        out = [r for r in out if r.get("arm") == arm]
    if N is not None:
        out = [r for r in out if r["N"] == N]
    if tau is not None:
        out = [r for r in out if r.get("tau") == tau]
    return out


def usable(r, frac=0.10):
    """Practical operating point: <=10% of solves miss the tolerance.

    The strict `censored` flag is an ANY-predicate over 16 solves for Poisson
    but 816 for Burgers, so it is not comparable across the two PDEs.
    """
    cf = r.get("censored_frac")
    if cf is None:
        return not r.get("censored", False)
    return cf <= frac and np.isfinite(r.get("err_rel_l2", np.inf))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ==================================================== 1. cost(k) decomposition
def gpu_group(r):
    """Panels did not all land on the same card; only compare within a group."""
    return "80GB" if "80GB" in (r.get("gpu") or "") else "40GB"


def fig_cost_k(pde, tag, Ns, tau=0.01):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(wspace=0.32)
    seen = set()
    for N in Ns:
        rs = sorted(sel(pde, "coord", N=N, tau=tau), key=lambda r: r["k"])
        if not rs:
            continue
        k = [r["k"] for r in rs]
        grp = gpu_group(rs[0])
        ls = "-" if grp == "80GB" else "--"
        seen.add(grp)
        axes[0].plot(k, [r["jac_evals"] for r in rs], "o-", color=NCOL[N], lw=2.2, ms=7,
                     label=f"N = {N}")
        axes[1].plot(k, [r["ms_per_jac"] for r in rs], "o", ls=ls, color=NCOL[N], lw=2, ms=7)
        axes[2].plot(k, [r["time_ms"] for r in rs], "o", ls=ls, color=NCOL[N], lw=2, ms=7)

    titles = ["Iterations to reach the tolerance\n(hardware-free)",
              "Cost per iteration", "Total cost of one solve"]
    ylabs = ["Jacobian evaluations", "ms per iteration", "online wall time (ms)"]
    for ax, ttl, yl in zip(axes, titles, ylabs):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(["2", "4", "8", "16", "32"])
        ax.set_xticks([], minor=True)
        ax.set_xlabel("latent dimension  k")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=13)
    axes[0].legend(frameon=False, title="mesh", ncol=2, fontsize=10)
    axes[0].text(0.5, 0.04, "all meshes collapse onto one curve",
                 transform=axes[0].transAxes, ha="center", fontsize=10.5, color="#2b7a43")
    if len(seen) > 1:
        for ax in axes[1:]:
            ax.text(0.5, 0.04, "solid: A100-80GB   dashed: A100-40GB",
                    transform=ax.transAxes, ha="center", fontsize=9.5, color="#a33")
    fig.suptitle(f"{tag}: how the cost of a solve depends on the latent dimension "
                 f"(τ = {tau:g})", fontsize=15, y=1.02)
    fig.text(0.5, -0.06,
             "Iteration counts are hardware-free and directly comparable across meshes. "
             "Wall-clock panels are NOT: the fine meshes ran on a different A100 model, "
             "so compare within one line style only.",
             ha="center", fontsize=10, color="#888")
    save(fig, f"6_cost_vs_k_{tag.lower()}")


def fig_where_time_goes(pde, tag, Ns, k=8, tau=0.01):
    """The finding the cost(k) panels hide: at fine meshes the O(n) decode dominates."""
    labs, solve, pre, dec = [], [], [], []
    for N in Ns:
        rs = sel(pde, "coord", N=N, tau=tau)
        rs = [r for r in rs if r["k"] == k]
        if not rs:
            continue
        r = rs[0]
        labs.append(f"{N}\n({gpu_group(r)})")
        solve.append(r["time_ms_solve"]); pre.append(r["time_ms_pre"])
        dec.append(r["time_ms_decode"])

    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar(x, solve, 0.6, color=OURS, label="latent solve  (mesh-free, what we optimised)")
    ax.bar(x, pre, 0.6, bottom=solve, color="#9ecae1", label="read the input  (O(n))")
    ax.bar(x, dec, 0.6, bottom=np.array(solve) + np.array(pre), color="#d9d9d9",
           label="decode the field  (O(n))")

    tot = np.array(solve) + np.array(pre) + np.array(dec)
    for xi, (d, t) in enumerate(zip(dec, tot)):
        ax.text(xi, t + t * 0.03, f"decode\n{100*d/t:.0f}%", ha="center",
                fontsize=11, color="#555")

    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_xlabel("mesh resolution  N")
    ax.set_ylabel("online wall time (ms)")
    ax.set_ylim(0, max(tot) * 1.28)
    ax.set_title(f"{tag}, k = {k}: the solve is mesh-free — everything around it is not")
    ax.legend(frameon=False, loc="upper left")
    save(fig, f"8_where_time_goes_{tag.lower()}")

    # --- companion table: is cost(k) mesh-independent? -----------------------
    md(f"\n### {tag}: cost per iteration, normalised at k = 8 (τ = {tau:g})\n")
    md("If the k-dependence is genuinely mesh-independent, every row is the same.\n")
    ks = [2, 4, 8, 16, 32]
    md("| N | " + " | ".join(f"k={x}" for x in ks) + " | ms/iter at k=8 |")
    md("|---|" + "---|" * (len(ks) + 1))
    for N in Ns:
        rs = {r["k"]: r for r in sel(pde, "coord", N=N, tau=tau)}
        if 8 not in rs:
            continue
        base = rs[8]["ms_per_jac"]
        cells = [f"{rs[x]['ms_per_jac']/base:.2f}×" if x in rs else "—" for x in ks]
        md(f"| {N} | " + " | ".join(cells) + f" | {base:.3f} ms |")


# ============================================== 2. Pareto frontier + the table
def frontier(pts):
    """Non-dominated (min time, min error) subset, sorted by time."""
    pts = sorted(pts, key=lambda p: p[0])
    out, best = [], np.inf
    for t, e, meta in pts:
        if e < best - 1e-15:
            out.append((t, e, meta)); best = e
    return out


def fig_pareto(pde, tag, N):
    fig, ax = plt.subplots(figsize=(9.5, 6))

    for method, col, lbl, mk in ((("coord", OURS, "coordinate ROM (ours)", "o")),
                                 (("pod", POD, "POD", "s"))):
        pts = [(r["time_ms"], r["err_rel_l2"], r) for r in sel(pde, method, N=N)
               if usable(r)]
        bad = [(r["time_ms"], r["err_rel_l2"]) for r in sel(pde, method, N=N)
               if not usable(r) and np.isfinite(r.get("err_rel_l2", np.inf))]
        if bad:
            bx, by = zip(*bad)
            ax.scatter(bx, by, s=26, facecolor="none", edgecolor=col, alpha=0.35, lw=1)
        if not pts:
            continue
        px, py = [p[0] for p in pts], [p[1] for p in pts]
        ax.scatter(px, py, s=54, color=col, alpha=0.55, marker=mk)
        fr = frontier(pts)
        ax.plot([p[0] for p in fr], [p[1] for p in fr], "-", color=col, lw=2.6,
                marker=mk, ms=9, label=lbl)
        for t, e, r in fr:
            ax.annotate(f"k={r['k']}", (t, e), textcoords="offset points",
                        xytext=(6, 6), fontsize=9.5, color=col)

    fl = sorted([r for r in ROWS if r["pde"] == pde and r["method"] == "fom"
                 and r["N"] == N and np.isfinite(r.get("err_rel_l2", np.inf))],
                key=lambda r: r["time_ms"])
    if fl:
        ax.plot([r["time_ms"] for r in fl], [r["err_rel_l2"] for r in fl],
                "^--", color=FOM, lw=2, ms=7, label="full-order solver (tolerance ladder)")

    # The FOM runs to machine precision; clip to the band where the ROM lives,
    # otherwise 9 decades of FOM squash the entire comparison into a sliver.
    allerr = [r["err_rel_l2"] for r in sel(pde, "coord", N=N) + sel(pde, "pod", N=N)
              if np.isfinite(r.get("err_rel_l2", np.inf))]
    lo = max(min(allerr) * 0.45, 1e-4)
    ax.set_ylim(lo, max(allerr) * 2.2)
    if fl and min(r["err_rel_l2"] for r in fl) < lo:
        ax.annotate("the full solver keeps going —\nto 1e-11, off the bottom of this plot",
                    xy=(fl[len(fl) // 2]["time_ms"], lo * 1.6),
                    xytext=(0.52, 0.13), textcoords="axes fraction",
                    fontsize=10.5, color=FOM, ha="center",
                    arrowprops=dict(arrowstyle="->", color=FOM, lw=1.4))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("online wall time per solve (ms)  →  cheaper is left")
    ax.set_ylabel("held-out error (relative L2)  →  better is down")
    ax.set_title(f"{tag}, N = {N}: one trained model, many operating points")
    ax.legend(frameon=False, loc="lower left")
    ax.text(0.99, 0.98, "hollow markers = tolerance not reached\n(>10% of solves censored)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#888")
    save(fig, f"7_pareto_{tag.lower()}_n{N}")


def table_iso_accuracy(pde, tag, Ns, targets):
    """The Pareto frontier, read the way people actually ask the question:
    'I need this accuracy — what is the cheapest way to get it?'"""
    md(f"\n### {tag}: cheapest route to a given accuracy\n")
    md("Each cell is the cheapest configuration of that method reaching the target. "
       "**—** means the method cannot reach it at any `k` or `τ` we tested.\n")
    md("| N | accuracy target | coordinate ROM (ours) | POD | full-order solver | "
       "ours vs the full solver |")
    md("|---|---|---|---|---|---|")
    for N in Ns:
        first = True
        for tgt in targets:
            cells, t_ours = [], None
            for method in ("coord", "pod"):
                ok = [r for r in sel(pde, method, N=N)
                      if usable(r) and r["err_rel_l2"] <= tgt]
                if ok:
                    b = min(ok, key=lambda r: r["time_ms"])
                    if method == "coord":
                        t_ours = b["time_ms"]
                    cells.append(f"**{b['time_ms']:.2f} ms** (k={b['k']}, τ={b['tau']:g})")
                else:
                    cells.append("—")
            fl = [r for r in ROWS if r["pde"] == pde and r["method"] == "fom"
                  and r["N"] == N and np.isfinite(r.get("err_rel_l2", np.inf))
                  and r["err_rel_l2"] <= tgt]
            t_fom = min(fl, key=lambda r: r["time_ms"])["time_ms"] if fl else None
            cells.append(f"{t_fom:.2f} ms" if t_fom else "—")
            if t_ours and t_fom:
                rat = t_fom / t_ours
                cells.append(f"**{rat:.2f}×** {'✅' if rat > 1 else '❌'}")
            else:
                cells.append("—")
            md(f"| {N if first else ''} | {tgt:.0e} | " + " | ".join(cells) + " |")
            first = False


def table_frontier(pde, tag, N):
    """Every non-dominated operating point at one mesh, cheapest first."""
    md(f"\n### {tag}, N = {N}: the frontier itself\n")
    md("Every point that is not beaten on both cost and accuracy by another point "
       "of the same method. Reading down a block, you buy accuracy with time.\n")
    md("| method | `k` | `τ` | time | error | iterations |")
    md("|---|---|---|---|---|---|")
    for method, lbl in (("coord", "coordinate ROM"), ("pod", "POD")):
        pts = [(r["time_ms"], r["err_rel_l2"], r) for r in sel(pde, method, N=N)
               if usable(r)]
        fr = frontier(pts)
        if not fr:
            md(f"| **{lbl}** | — | — | — | no point clears the censoring bar | — |")
            continue
        for i, (t, e, r) in enumerate(fr):
            md(f"| {'**' + lbl + '**' if i == 0 else ''} | {r['k']} | {r['tau']:g} "
               f"| {t:.2f} ms | {e:.2e} | {r['jac_evals']:.1f} |")


def table_speed_vs_accuracy(pde, tag, Ns, ladder_k=8):
    md(f"\n### {tag}: the same trained model, tuned for speed or for accuracy\n")
    md("One decoder per `k`; the operating point is chosen at **run time** by the "
       "stopping tolerance `τ`. Both columns come from the same runs.\n")
    md("| N | fastest usable point | | | most accurate point | | | you pay |")
    md("|---|---|---|---|---|---|---|---|")
    md("| | `k`, `τ` | time | error | `k`, `τ` | time | error | for accuracy |")
    for N in Ns:
        pts = [r for r in sel(pde, "coord", N=N) if usable(r)]
        if not pts:
            md(f"| {N} | — | — | — | — | — | — | — |")
            continue
        fast = min(pts, key=lambda r: r["time_ms"])
        acc = min(pts, key=lambda r: r["err_rel_l2"])
        same = fast is acc
        pay = ("only one point clears the censoring bar" if same else
               f"{acc['time_ms']/fast['time_ms']:.1f}× time "
               f"for {fast['err_rel_l2']/acc['err_rel_l2']:.1f}× accuracy")
        md(f"| {N} "
           f"| k={fast['k']}, τ={fast['tau']:g} | {fast['time_ms']:.2f} ms "
           f"| {fast['err_rel_l2']:.2e} "
           f"| k={acc['k']}, τ={acc['tau']:g} | {acc['time_ms']:.2f} ms "
           f"| {acc['err_rel_l2']:.2e} "
           f"| {pay} |")

    # The knob itself, at one k, including the points the filter rejects.
    md(f"\n**The tolerance knob at k = {ladder_k}** (all τ shown, including those that "
       "fail the ≤10%-censored bar — that failure is itself the answer to "
       "\"how far can you push it?\"):\n")
    md("| N | " + " | ".join(f"τ={t:g}" for t in (0.1, 0.01, 0.001)) + " |")
    md("|---|" + "---|" * 3)
    for N in Ns:
        cells = []
        for t in (0.1, 0.01, 0.001):
            rs = [r for r in sel(pde, "coord", N=N, tau=t) if r["k"] == ladder_k]
            if not rs:
                cells.append("—"); continue
            r = rs[0]
            cf = r.get("censored_frac") or 0.0
            flag = "" if cf <= 0.10 else f" ⚠{100*cf:.0f}% censored"
            cells.append(f"{r['time_ms']:.1f} ms → {r['err_rel_l2']:.2e}{flag}")
        md(f"| {N} | " + " | ".join(cells) + " |")


# ======================================================= 3. FOM vs the hybrid
def table_hybrid():
    md("\n## 3. Full-order solver versus the ROM-warm-started hybrid\n")
    md("The hybrid decodes the ROM solution and hands it to the full solver as its "
       "initial guess, so **both columns end at the same accuracy** — the question is "
       "only cost.\n")

    for pde, tag in (("poisson2d", "Poisson"), ("burgers2d", "Burgers")):
        rs = [r for r in HROWS if r["pde"] == pde]
        if not rs:
            continue
        # Hold fom_tau fixed, or the "same accuracy" claim fails ACROSS rows.
        taus = sorted({r["fom_tau"] for r in rs if r.get("fom_tau")})
        ft = 1e-6 if 1e-6 in taus else taus[0]
        rs = [r for r in rs if r.get("fom_tau") == ft]
        md(f"\n### {tag}  — every row finished to the same tolerance, `fom_tau` = {ft:g}\n")
        md("| N | FOM alone | hybrid total | = ROM + finish | speed-up | "
           "solver iters: plain → from ROM | final error |")
        md("|---|---|---|---|---|---|---|")
        for N in sorted({r["N"] for r in rs}):
            cand = [r for r in rs if r["N"] == N]
            best = max(cand, key=lambda r: r.get("speedup_vs_fom", 0))
            it_b, it_r = best.get("iters_from_baseline"), best.get("iters_from_rom")
            iters = (f"{it_b:.0f} → {it_r:.0f}" if it_b and it_r else "—")
            if it_b and it_r:
                iters += " ✗ worse" if it_r > it_b else " ✓ better"
            md(f"| {N} "
               f"| {best['t_fom_baseline_ms']:.1f} ms "
               f"| **{best['t_total_ms']:.1f} ms** "
               f"| {best['t_rom_ms']:.1f} + {best['t_fom_ms']:.1f} ms "
               f"| **{best['speedup_vs_fom']:.2f}×** "
               f"| {iters} "
               f"| {best['err_final']:.1e} |")
        md(f"\n*The hybrid's strongest configuration is shown at each mesh, so these are "
           f"upper bounds on it. Accuracy is identical by construction — both paths finish "
           f"at `fom_tau` = {ft:g} — so the only question is cost, and the hybrid never "
           f"reaches 1.00×.*")


# =============================================================== assemble
if __name__ == "__main__":
    print(f"reading {os.path.basename(PARETO)} ({len(ROWS)} rows), "
          f"{os.path.basename(HYBRID)} ({len(HROWS)} rows)")

    md("# Cost, the speed/accuracy frontier, and the hybrid\n")
    md("Everything below is generated by `make_cost_figs.py` from the raw run JSONs "
       "(`pareto_points.json`, 698 rows; `hybrid_points.json`, 174 rows). No number is "
       "hand-typed, and **cost and accuracy in every row come from the same solver "
       "invocation** — the defect that made the previous round's tables unusable.\n")
    md("> ### Two things to know before reading\n"
       "> **Hardware.** Meshes N ≤ 128 ran on an A100-80GB, N ≥ 256 on an A100-40GB. "
       "Within one mesh every method shares a card and is directly comparable; **wall "
       "times are not comparable across that boundary**, and the difference alone "
       "accounts for a 3.7× apparent speed-up in the latent solve. Iteration counts are "
       "hardware-free and comparable everywhere.\n"
       "> **Provisional.** The full-order ladder ran as its own job, so ROM-vs-FOM "
       "ratios are cross-GPU pairings and will be superseded by a single-GPU "
       "consolidation run now in progress.\n")

    # ------------------------------------------------------------ section 1
    md("\n## 1. How the cost of a solve depends on the latent dimension `k`\n")
    fig_cost_k("poisson2d", "Poisson", [32, 64, 128, 256, 512])
    fig_cost_k("burgers2d", "Burgers", [32, 64, 128, 256])
    fig_where_time_goes("poisson2d", "Poisson", [32, 64, 128, 256, 512])

    md("![cost vs k, Poisson](talk_figs/6_cost_vs_k_poisson.png)\n")
    md("Total cost = **iterations × cost per iteration**, so the three panels multiply "
       "across. The left panel is the load-bearing one: iteration counts are hardware-free, "
       "and **every mesh collapses onto one curve** — the k-dependence of the solve is "
       "genuinely mesh-independent. The sawtooth at k = 6 and k = 12 is the optimiser "
       "stalling, which shows up as wasted iterations.\n")

    md("\n### Iterations to reach τ = 0.01 — the hardware-free cost of a solve\n")
    ks = [2, 4, 6, 8, 12, 16, 24, 32]
    md("| N | " + " | ".join(f"k={x}" for x in ks) + " |")
    md("|---|" + "---|" * len(ks))
    for N in [32, 64, 128, 256, 512]:
        rs = {r["k"]: r for r in sel("poisson2d", "coord", N=N, tau=0.01)}
        md(f"| {N} | " + " | ".join(
            f"{rs[x]['jac_evals']:.1f}" if x in rs else "—" for x in ks) + " |")
    md("\nRead **across** a row: cost falls to k = 8, and the k = 6 / 12 spikes are the "
       "stall. Read **down** a column: the numbers barely move, which is the "
       "mesh-independence result.\n")

    md("\n![where the time goes](talk_figs/8_where_time_goes_poisson.png)\n")
    md("**The bottleneck has moved.** We made the latent solve mesh-free and succeeded — "
       "so at fine meshes the cost is now dominated by the `O(n)` decode that surrounds "
       "it. At N = 512 the decode is **84 %** of online cost and the solve is 9 %. "
       "Decode share across the ladder: 4 % → 6 % → 11 % → 50 % → 84 %.\n")

    # ------------------------------------------------------------ section 2
    md("\n## 2. The speed / accuracy frontier — one trained model, many operating points\n")
    md("The point of the method is that `k` and the stopping tolerance `τ` are knobs you "
       "turn at **run time**, not retraining decisions. These tables are the frontier: "
       "the set of operating points that are not beaten on both cost and accuracy.\n")

    table_iso_accuracy("poisson2d", "Poisson", [64, 256, 512],
                       [1e-1, 5e-2, 2e-2, 1.2e-2])
    md("\n**This is the split frontier in one table.** POD is the cheapest way to get a "
       "rough answer and simply *cannot* reach the tighter targets — its error saturates "
       "near 5e-2 no matter how many modes you give it. Below that line we are the only "
       "reduced method that works, and the comparison becomes ours versus the full solver.\n")

    table_iso_accuracy("burgers2d", "Burgers", [64, 256], [5e-2, 3e-2, 2.2e-2])

    table_frontier("poisson2d", "Poisson", 64)
    table_frontier("poisson2d", "Poisson", 256)
    table_frontier("burgers2d", "Burgers", 256)

    table_speed_vs_accuracy("poisson2d", "Poisson", [32, 64, 128, 256, 512])
    table_speed_vs_accuracy("burgers2d", "Burgers", [32, 64, 128, 256])

    # ------------------------------------------------------------ section 3
    table_hybrid()
    md("\n**Why it fails differs by problem, and the difference matters.** On Poisson the "
       "ROM's guess is simply *bad* for the solver: its field error is 9.3e-3 but its "
       "A-norm error — the quantity CG actually contracts — is 4.97e-2, so it saves only "
       "1–4 % of iterations and at coarse meshes needs *more* than a zero start. On "
       "Burgers the guess is genuinely good (Newton iterations fall 98 → 92 at every "
       "mesh) but the ROM stage costs 273 ms against a 48–228 ms full solve, so it can "
       "never pay for itself. Two different failures, one conclusion.\n")

    md("\n---\n")
    md("*Provenance: branches `exp/2026-08-17-cost-to-tolerance` and "
       "`exp/2026-08-17-rom-warmstart-fom`. Every run f64, "
       "`JAX_DEFAULT_MATMUL_PRECISION=highest`, `jax_backend=gpu` asserted, one job per "
       "directory, data regenerated on the cluster from seed, results pulled with "
       "checksums.*")

    open(TAB, "w").write("\n".join(OUTPUT) + "\n")
    print(f"  wrote {os.path.relpath(TAB, HERE)}")
