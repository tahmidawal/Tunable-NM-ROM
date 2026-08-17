"""Markdown tables + figures for the Burgers-2D FOLLOW-UP cells.

Reads the frozen runs/<cell>/ JSONs (K=4/8/16, seed 0) and the pulled
runs/followup/<cell>/ JSONs and writes followup/FOLLOWUP_TABLES.md plus PNG+PDF
figures into followup/figs/ and the shared Plots directory.  Nothing is
recomputed here; every number comes straight from a JSON.

Usage: python followup/fu_summarize.py [<experiment dir>]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
RUNS = os.path.join(EXP, "runs")
FU = os.path.join(RUNS, "followup")
FIGS = os.path.join(HERE, "figs")
PLOTS = "/home/tahmid/Dev/pod-ae-nmrom/Plots"
sys.path.insert(0, HERE)
import fu_style as fs  # noqa: E402

KS = [2, 4, 6, 8, 12, 16, 24, 32]
POD_KS = [2, 4, 6, 8, 12, 16, 24, 32, 64]
SEEDS = [0, 1, 2]
EQ = "lspg:eq256:weak64"
FULLW = "lspg:full:weak64"
FD = "lspg:full:fd"


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def first(*pats):
    for pat in pats:
        h = sorted(glob.glob(pat))
        if h:
            return load(h[0])
    return None


def fmt(x, d=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(x) else f"{x:.{d}e}"


def get(rep, path, default=None):
    cur = rep
    for k in path:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur


# ------------------------------------------------------------------ gather
def gather():
    D = dict(kladder=[], seeds=[], mlad=None, Mlad=None, tn=None, tk=None)
    for K in KS:
        rep = first(os.path.join(FU, f"bk_K{K}", f"blat_rom_N64_K{K}.json"),
                    os.path.join(RUNS, f"ad_n64_k{K}", f"blat_rom_N64_K{K}.json"))
        tr = first(os.path.join(FU, f"bk_K{K}", f"blat_ad_N64_K{K}_report.json"),
                   os.path.join(RUNS, f"ad_n64_k{K}", f"blat_ad_N64_K{K}_report.json"))
        if rep:
            D["kladder"].append(dict(K=K, rep=rep, train=tr))
    for S in SEEDS:
        if S == 0:
            rep = first(os.path.join(RUNS, "ad_n64_k8", "blat_rom_N64_K8.json"))
            tr = first(os.path.join(RUNS, "ad_n64_k8", "blat_ad_N64_K8_report.json"))
        else:
            rep = first(os.path.join(FU, f"bs_S{S}", f"blat_rom_N64_K8_S{S}.json"))
            tr = first(os.path.join(FU, f"bs_S{S}", f"blat_ad_N64_K8_S{S}_report.json"))
        if rep:
            D["seeds"].append(dict(seed=S, rep=rep, train=tr))
    D["mlad"] = first(os.path.join(FU, "bm_m", "blat_rom_N64_K8.json"))
    D["Mlad"] = first(os.path.join(FU, "bm_M", "blat_rom_N64_K8.json"))
    D["tn"] = first(os.path.join(FU, "bt_n", "timing_n.json"))
    D["tk"] = first(os.path.join(FU, "bt_k", "timing_k.json"))
    return D


# ------------------------------------------------------------------ tables
def tab_kladder(D, out):
    if not D["kladder"]:
        return
    out.append("### k ladder — coordinate ROM vs POD-LSPG at the same k "
               "(N=64, 16 held-out trajectories, trajectory rel-L2 = mean over the 51 slices)\n")
    out.append("| K | train recon | ORACLE inferred latent | IC fit (t=0) | `full:fd` | `full:weak64` | `eq256:weak64` | `eq512:weak64` | blow-ups | warm iters/step |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in D["kladder"]:
        r = e["rep"]; rom = r.get("rom", {})
        eq = rom.get(EQ, {})
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            e["K"], fmt(r.get("train_rel_mean")),
            fmt(get(r, ["oracle_inferred_latent_test", "traj_rel_mean"])),
            fmt(get(r, ["ic_fit", "rel_mean"])),
            fmt(get(rom, [FD, "traj_rel_mean"])), fmt(get(rom, [FULLW, "traj_rel_mean"])),
            fmt(eq.get("traj_rel_mean")), fmt(get(rom, ["lspg:eq512:weak64", "traj_rel_mean"])),
            sum(v.get("n_blowup", 0) for v in rom.values()),
            f"{eq.get('iters_warm_mean', float('nan')):.2f}" if eq else "—"))
    out.append("")
    pod = pod_table(D)
    if pod:
        out.append("POD-LSPG control (same solver, same objective, same TRAIN snapshots; the basis "
                   "does not depend on K, so it is computed once):\n")
        out.append("| POD k | projection floor | `full:fd` | `full:weak64` | `eq256:weak64` |")
        out.append("|---|---|---|---|---|")
        for k, row in sorted(pod.items()):
            out.append(f"| {k} | {fmt(row.get('proj'))} | {fmt(row.get(FD))} | "
                       f"{fmt(row.get(FULLW))} | {fmt(row.get(EQ))} |")
        out.append("")


def pod_table(D):
    """POD rows, preferring the K=2 follow-up cell (widest POD_KS) and falling
    back to any cell that carries the same k."""
    out = {}
    for e in D["kladder"]:
        r = e["rep"]
        floors = r.get("oracle_pod_projection_floor_test", {})
        for key, s in (r.get("pod_rom") or {}).items():
            k_s, var = key.split(":", 1)
            k = int(k_s[1:])
            out.setdefault(k, {})
            out[k].setdefault(var, s["traj_rel_mean"])
            out[k].setdefault("proj", floors.get(str(k), floors.get(k)))
    return out


def tab_seeds(D, out):
    if len(D["seeds"]) < 2:
        return
    out.append("### Multi-seed (K=8, N=64; TRAIN_SEED changes the net init, the latent init and "
               "the batch order only — the data draw, the split and the TEST_SEED test set are fixed)\n")
    keys = [("train recon (learned latents)", lambda r: r.get("train_rel_mean")),
            ("ORACLE inferred-latent floor (held out)", lambda r: get(r, ["oracle_inferred_latent_test", "traj_rel_mean"])),
            ("IC fit at t=0", lambda r: get(r, ["ic_fit", "rel_mean"])),
            ("ROM `full:weak64`", lambda r: get(r, ["rom", FULLW, "traj_rel_mean"])),
            ("ROM `eq256:weak64`", lambda r: get(r, ["rom", EQ, "traj_rel_mean"])),
            ("ROM `eq512:weak64`", lambda r: get(r, ["rom", "lspg:eq512:weak64", "traj_rel_mean"])),
            ("ROM `full:fd`", lambda r: get(r, ["rom", FD, "traj_rel_mean"]))]
    out.append("| quantity | " + " | ".join(f"seed {e['seed']}" for e in D["seeds"]) + " | mean ± std |")
    out.append("|" + "---|" * (len(D["seeds"]) + 2))
    for name, fn in keys:
        v = np.array([float(fn(e["rep"]) or np.nan) for e in D["seeds"]], dtype=float)
        out.append(f"| {name} | " + " | ".join(fmt(x) for x in v)
                   + f" | {np.nanmean(v):.2e} ± {np.nanstd(v, ddof=1):.1e} |")
    pk = {}
    for e in D["seeds"]:
        for key, s in (e["rep"].get("pod_rom") or {}).items():
            if key.startswith("k8:"):
                pk.setdefault(key, []).append(s["traj_rel_mean"])
    for key, vals in sorted(pk.items()):
        out.append(f"| POD control {key} | " + " | ".join(fmt(v) for v in vals)
                   + f" | {np.mean(vals):.2e} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.1e} |")
    out.append("\nThe POD basis is a deterministic function of the TRAIN snapshots, so the POD "
               "control carries no training-seed variance (any spread in its row is solver "
               "non-determinism only).\n")


def _mrows(rep, kind):
    """(m, variant, summary) rows of a given objective family from a blat_rom report."""
    out = []
    for var, s in (rep.get("rom") or {}).items():
        solver, col, obj = var.split(":")
        if not obj.startswith(kind):
            continue
        if col == "full":
            m, pool = "full", "grid"
        elif col.startswith("eqoff"):
            m, pool = int(col[5:]), "meshfree"
        else:
            m, pool = int(col[2:]), "grid"
        out.append((m, pool, obj, s))
    return out


def tab_ladders(D, out):
    if D["mlad"]:
        rep = D["mlad"]
        out.append("### m ladder at fixed (K=8, M=64) — NNLS-EQ nodes, grid vs meshfree pool\n")
        out.append("| objective / pool | " + " | ".join(f"m={m}" for m in (64, 128, 256, 512, 1024)) + " | full grid |")
        out.append("|" + "---|" * 7)
        for kind, pool, label in (("weak", "grid", "`weak64` (exact FOM operator), grid EQ"),
                                  ("weakc", "grid", "`weakc64` (continuum), grid EQ"),
                                  ("weakc", "meshfree", "`weakc64` (continuum), meshfree pool")):
            rows = {(m, p): s for m, p, o, s in _mrows(rep, kind)
                    if o.startswith("weakc") == (kind == "weakc")}
            vals = [rows.get((m, pool)) for m in (64, 128, 256, 512, 1024)]
            fullv = rows.get(("full", "grid"))
            out.append(f"| {label} | " + " | ".join(fmt(v["traj_rel_mean"]) if v else "—" for v in vals)
                       + f" | {fmt(fullv['traj_rel_mean']) if fullv else '—'} |")
            out.append("| ↳ ms per ROM step | " + " | ".join(
                f"{v['step_time_ms_median']:.1f}" if v else "—" for v in vals)
                + (f" | {fullv['step_time_ms_median']:.1f} |" if fullv else " | — |"))
            out.append("| ↳ NNLS relative fit | " + " | ".join(
                (f"{v['eq_info']['rel_fit']:.1e}" if v and v.get("eq_info") else "—") for v in vals) + " | 0 |")
        out.append(f"\nOracle inferred-latent floor on this cell: "
                   f"{fmt(get(rep, ['oracle_inferred_latent_test', 'traj_rel_mean']))}; "
                   f"blow-ups: {sum(v.get('n_blowup', 0) for v in rep['rom'].values())} in "
                   f"{len(rep['rom'])} x 16 rollouts.\n")
    if D["Mlad"]:
        rep = D["Mlad"]
        out.append("### M ladder at m ≈ 4M (K=8, grid EQ)\n")
        out.append("| M | full grid | NNLS-EQ m=4M | ms/step (full) | ms/step (EQ) |")
        out.append("|---|---|---|---|---|")
        for M in (16, 32, 64, 128, 256):
            f_ = get(rep, ["rom", f"lspg:full:weak{M}"]) or {}
            e_ = get(rep, ["rom", f"lspg:eq{4*M}:weak{M}"]) or {}
            out.append(f"| {M} | {fmt(f_.get('traj_rel_mean'))} | {fmt(e_.get('traj_rel_mean'))} | "
                       f"{f_.get('step_time_ms_median', float('nan')):.1f} | "
                       f"{e_.get('step_time_ms_median', float('nan')):.1f} |")
        out.append("")


def tab_timing(D, out):
    t = D["tn"]
    if t:
        out.append(f"### Online cost vs N on ONE GPU ({t['device']}, all N sequential in one process; "
                   f"K={get(t, ['ckpt', 'k'])}, M=64, m=256, median of {t['time_reps']} after 2 warm-ups, "
                   f"`block_until_ready`; the coordinate decoder is meshfree so the SAME N=64 "
                   f"checkpoint is used at every N, EQ weights refit per N)\n")
        out.append("| N | FOM rollout | ROM `eq256:weak64` | speedup | ms / ROM step | ms / Jacobian eval | "
                   "IC fit (python LM) | IC fit (jitted LM) | decode 51 slices | end-to-end speedup (jitted IC) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in t["rows"]:
            v = r["rom"].get(EQ, {})
            ic = r["ic_fit"]
            out.append(f"| {r['N']} | {r['fom_rollout_s']*1e3:.0f} ms | {v.get('rollout_s_median', float('nan'))*1e3:.0f} ms | "
                       f"**{v.get('speedup_rollout_only', float('nan')):.2f}x** | "
                       f"{v.get('rollout_s_median', float('nan'))*1e3/50:.2f} | "
                       f"{v.get('s_per_jacobian_eval', float('nan'))*1e3:.2f} | "
                       f"{ic['python_s']*1e3:.0f} ms | {ic['jit_s']*1e3:.1f} ms | "
                       f"{r['decode_all_slices_s']*1e3:.1f} ms | "
                       f"{v.get('speedup_end_to_end_jit_ic', float('nan')):.2f}x |")
        out.append("")
        out.append("Other variants (rollout ms / speedup): " + "; ".join(
            f"N={r['N']}: " + ", ".join(f"`{k.split(':',1)[1]}` {v['rollout_s_median']*1e3:.0f}/{v['speedup_rollout_only']:.2f}x"
                                        for k, v in r["rom"].items() if k != EQ)
            for r in t["rows"]) + ".\n")
        out.append("Accuracy of the N=64-trained ROM against the FOM at each N (test trajectory 0, "
                   "single trajectory — a transfer check, not the cell's error statistic): " + "; ".join(
            f"N={r['N']}: {fmt(r['rom'].get(EQ, {}).get('traj_rel_vs_fom_at_this_N'))}" for r in t["rows"]) + ".\n")
    t = D["tk"]
    if t:
        out.append(f"### Per-iteration cost and iterations vs k on ONE GPU ({t['device']}, N=64, "
                   f"FOM {t['fom_rollout_s']*1e3:.0f} ms)\n")
        out.append("| manifold | k | rollout `eq256:weak64` | speedup | Jacobian evals (50 steps) | ms / Jacobian eval | IC fit (jitted) |")
        out.append("|---|---|---|---|---|---|---|")
        for r in t["rows"]:
            v = r["rom"].get(EQ) or r["rom"].get(FD) or {}
            ic = (f"{get(r, ['ic_fit', 'jit_s'], float('nan'))*1e3:.0f} ms"
                  if r["kind"] == "coord" else "projection (exact)")
            out.append(f"| {'coordinate' if r['kind'] == 'coord' else 'POD'} | {r['K']} | "
                       f"{v.get('rollout_s_median', float('nan'))*1e3:.0f} ms | "
                       f"{v.get('speedup_rollout_only', float('nan')):.2f}x | {v.get('iters_total', '—')} | "
                       f"{v.get('s_per_jacobian_eval', float('nan'))*1e3:.2f} | {ic} |")
        out.append("")


# ------------------------------------------------------------------ figures
def fig_kladder(D):
    if not D["kladder"]:
        return None
    fs.use()
    import matplotlib.pyplot as plt
    pod = pod_table(D)
    ks = [e["K"] for e in D["kladder"]]
    full = [get(e["rep"], ["rom", FULLW, "traj_rel_mean"]) for e in D["kladder"]]
    eq = [get(e["rep"], ["rom", EQ, "traj_rel_mean"]) for e in D["kladder"]]
    orc = [get(e["rep"], ["oracle_inferred_latent_test", "traj_rel_mean"]) for e in D["kladder"]]
    pk = sorted(pod)
    pfd = [pod[k].get(FD) for k in pk]
    # the square Petrov-Galerkin case (M' test modes vs k >= M' unknowns) is unstable and
    # is documented as such; it is kept in the table and dropped from the figure
    pwk = [(v if (v is not None and v < 1.0) else np.nan) for v in (pod[k].get(FULLW) for k in pk)]
    ppr = [pod[k].get("proj") for k in pk]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    fs.clean(ax)
    ax.plot(ks, full, "o-", color=fs.C["blue"], label="coordinate ROM (full grid, weak64)", zorder=5)
    ax.plot(ks, eq, "s--", color=fs.C["blue"], alpha=0.75, label="coordinate ROM (NNLS-EQ, m=256)", zorder=4)
    ax.plot(ks, orc, ":", color=fs.MUTED, lw=1.6, label="coord. inferred-latent floor (oracle)", zorder=3)
    ax.plot(pk, pfd, "^-", color=fs.C["orange"], label="POD-LSPG (same solver)", zorder=5)
    if any(v is not None for v in pwk):
        ax.plot(pk, pwk, "v--", color=fs.C["orange"], alpha=0.7, label="POD, same weak objective", zorder=4)
    ax.plot(pk, ppr, ":", color=fs.C["orange"], lw=1.4, alpha=0.6, label="POD projection floor", zorder=3)
    ax.axvline(6, color=fs.MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    tk = sorted(set(ks) | set(pk))
    ax.set_xticks(tk); ax.set_xticklabels([str(k) for k in tk])
    fin = np.array([v for v in full + eq + orc + pfd + pwk + ppr
                    if v is not None and np.isfinite(v)], dtype=float)
    ax.set_ylim(fin.min() / 1.7, fin.max() * 1.5)
    ax.annotate("intrinsic dim 6\n(5 params + t)", (6, ax.get_ylim()[1]), xytext=(3, -14),
                textcoords="offset points", color=fs.INK2, fontsize=8, va="top")
    ax.set_xlabel("latent dimension k"); ax.set_ylabel("trajectory rel-L2 (16 held-out trajectories)")
    ax.set_title("Burgers 2D — where the nonlinear manifold stops paying", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    return fs.save(fig, FIGS, "burgers_k_ladder", (PLOTS,))


def fig_timing(D):
    t = D["tn"]
    if not t:
        return None
    fs.use()
    import matplotlib.pyplot as plt
    r = t["rows"]
    N = [x["N"] for x in r]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    fs.clean(ax)
    ax.plot(N, [x["fom_rollout_s"] * 1e3 for x in r], "o-", color=fs.C["red"],
            label="FOM (implicit BE rollout, 50 steps)")
    styles = [("s-", "blue"), ("^--", "aqua"), ("v--", "yellow"), ("d--", "violet")]
    vars_ = list(r[0]["rom"].keys())
    for (mk, col), v in zip(styles, vars_):
        y = [x["rom"][v]["rollout_s_median"] * 1e3 for x in r]
        ax.plot(N, y, mk, color=fs.C[col], label=f"ROM `{v.split(':',1)[1]}`")
    for x in r:
        s = x["rom"].get(EQ)
        if s:
            ax.annotate(f"{s['speedup_rollout_only']:.1f}x", (x["N"], s["rollout_s_median"] * 1e3),
                        xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, color=fs.INK2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel("mesh N ($N^2$ degrees of freedom)"); ax.set_ylabel("rollout wall time (ms)")
    ax.set_title("Burgers 2D — online cost is independent of the mesh", loc="left")
    ax.legend(loc="upper left")
    return fs.save(fig, FIGS, "burgers_cost_vs_N", (PLOTS,))


def fig_ladders(D):
    if not D["mlad"]:
        return None
    fs.use()
    import matplotlib.pyplot as plt
    rep = D["mlad"]
    ms = [64, 128, 256, 512, 1024]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = fs.clean(axes[0])
    series = [("weak", "grid", "`weak64` grid EQ (exact FOM operator)", "blue", "o"),
              ("weakc", "grid", "`weakc64` grid EQ (continuum)", "aqua", "s"),
              ("weakc", "meshfree", "`weakc64` meshfree pool", "yellow", "^")]
    for kind, pool, label, col, mk in series:
        rows = {(m, p): s for m, p, o, s in _mrows(rep, kind)
                if o.startswith("weakc") == (kind == "weakc")}
        y = [rows.get((m, pool)) for m in ms]
        ax.plot(ms, [v["traj_rel_mean"] if v else np.nan for v in y], mk + "-", color=fs.C[col], label=label)
        fv = rows.get(("full", "grid"))
        if fv and pool == "grid":
            ax.axhline(fv["traj_rel_mean"], color=fs.C[col], ls="--", lw=1.0, alpha=0.5)
    orc = get(rep, ["oracle_inferred_latent_test", "traj_rel_mean"])
    if orc:
        ax.axhline(orc, color=fs.MUTED, ls=":", lw=1.4, label="inferred-latent floor")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("quadrature points m"); ax.set_ylabel("trajectory rel-L2")
    ax.set_title("m ladder at M=64 (dashed: same objective, full grid)", loc="left")
    ax.legend(loc="upper right")
    ax = fs.clean(axes[1])
    if D["Mlad"]:
        Ms = [16, 32, 64, 128, 256]
        f_ = [get(D["Mlad"], ["rom", f"lspg:full:weak{M}", "traj_rel_mean"]) for M in Ms]
        e_ = [get(D["Mlad"], ["rom", f"lspg:eq{4*M}:weak{M}", "traj_rel_mean"]) for M in Ms]
        ax.plot(Ms, f_, "d-", color=fs.INK2, label="full grid")
        ax.plot(Ms, e_, "o-", color=fs.C["blue"], label="NNLS-EQ, m = 4M")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel("test modes M"); ax.set_ylabel("trajectory rel-L2")
    ax.set_title("M ladder at m ≈ 4M", loc="left")
    ax.legend(loc="upper right")
    fig.suptitle("Burgers 2D — the hyper-reduction knobs", x=0.005, ha="left", fontsize=10, color=fs.INK)
    fig.tight_layout()
    return fs.save(fig, FIGS, "burgers_eq_knobs", (PLOTS,))


def main():
    D = gather()
    out = ["# Burgers-2D follow-up — tables (generated by `followup/fu_summarize.py`)", ""]
    tab_kladder(D, out)
    tab_seeds(D, out)
    tab_ladders(D, out)
    tab_timing(D, out)
    p = os.path.join(HERE, "FOLLOWUP_TABLES.md")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", p)
    for f in (fig_kladder, fig_timing, fig_ladders):
        try:
            print("figure:", f(D))
        except Exception as e:
            print(f"  figure {f.__name__} skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
