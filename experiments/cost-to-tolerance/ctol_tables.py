"""Build `runs/pareto_points.json` from the two surface JSONs and regenerate
EVERY table in README.md.

Nothing in the README's tables is hand-typed: this script rewrites the blocks
delimited by

    <!-- BEGIN GENERATED: <name> -->  ...  <!-- END GENERATED: <name> -->

so a stale number cannot survive a rerun.  Prose outside those markers is
written by hand and is audited separately.

The shared cross-PDE Pareto schema (one object per (method, N, k, M, m, tau)
configuration) is exactly:

  {pde, method, N, k, M, m, tau, time_ms, err_rel_l2, iters, jac_evals,
   censored, n_sources, seed, gpu, jax_backend, commit}

with method in {"coord", "pod", "fom"}.  `time_ms` and `err_rel_l2` come from
the same run.  Diagnostic fields (time_ms_solve, censored_frac, eq_rel_fit, ...)
are carried alongside; the schema keys above are always present.

Usage: python ctol_tables.py [--runs runs] [--readme README.md]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = ["pde", "method", "N", "k", "M", "m", "tau", "time_ms", "err_rel_l2",
          "iters", "jac_evals", "censored", "n_sources", "seed", "gpu",
          "jax_backend", "commit"]
EXTRA = ["arm", "time_ms_solve", "time_ms_pre", "time_ms_decode", "censored_frac",
         "err_rel_l2_median", "err_rel_l2_max", "rel_reduction_mean",
         "fd_residual_rel_mean", "fom_residual_rel_mean", "eq_rel_fit",
         "ms_per_jac", "n_modes", "n_modes_le_k", "n_blowup", "speedup_e2e"]


def fmt(x, spec=".3e"):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float) and not math.isfinite(x):
        return "nan"
    try:
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def load(runs):
    """Merge every panel JSON (one job per (pde, mesh)) plus, when present, the
    single-GPU consolidation JSON.  Rows tagged arm='consolidated' are kept in a
    SEPARATE bucket: they are the only timings that may be compared ACROSS
    meshes, because the panel jobs are same-architecture but not the same
    physical GPU."""
    import glob
    out = {}
    for pde, pat in (("poisson2d", "ctol_poisson*.json"),
                     ("burgers2d", "ctol_burgers*.json")):
        files = sorted(glob.glob(os.path.join(runs, "**", pat), recursive=True))
        if not files:
            print(f"  (no {pat} under {runs})", file=sys.stderr)
            continue
        d = dict(config=None, panels=[], rows=[], fom=[], supplementary=[],
                 consolidated_rows=[], consolidated_fom=[], fom_baseline=[],
                 files=[], complete=True)
        ks, ns, taus = set(), set(), set()
        for f in files:
            j = json.load(open(f))
            cfg = j["config"]
            d["files"].append(os.path.relpath(f, runs))
            d["panels"].append(dict(cfg, _file=os.path.relpath(f, runs),
                                    _complete=bool(j.get("complete"))))
            d["complete"] = d["complete"] and bool(j.get("complete"))
            consolidated = bool(cfg.get("configs"))
            for r in j["rows"]:
                (d["consolidated_rows"] if consolidated else d["rows"]).append(r)
            for fo in j.get("fom", []):
                (d["consolidated_fom"] if consolidated else d["fom"]).append(fo)
            d["supplementary"] += j.get("supplementary", [])
            if not consolidated:
                d["fom_baseline"] += j.get("fom_baseline", [])
            if not consolidated:
                ks |= set(cfg["ks"]); ns |= set(cfg.get("ns_measured") or cfg["ns"])
                taus |= set(cfg["taus"])
                if d["config"] is None:
                    d["config"] = dict(cfg)
        if d["config"] is None:                      # only a consolidation file present
            d["config"] = dict(json.load(open(files[0]))["config"])
            ks = set(d["config"]["ks"]); ns = set(d["config"]["ns"]); taus = set(d["config"]["taus"])
        d["config"]["ks"] = sorted(ks)
        d["config"]["ns"] = sorted(ns)
        d["config"]["taus"] = sorted(taus, reverse=True)
        gpus = sorted({r.get("gpu") for r in d["rows"] if r.get("gpu")})
        nodes = sorted({r.get("node") for r in d["rows"] if r.get("node")})
        d["config"]["panel_gpus"] = gpus
        d["config"]["panel_nodes"] = nodes
        out[pde] = d
    return out


def to_points(data):
    """Flat Pareto-schema list: every ROM cell plus one FOM row per (pde, N)."""
    pts = []
    for pde, d in data.items():
        cfg = d["config"]
        for r in d["rows"]:
            o = {kk: r.get(kk) for kk in SCHEMA}
            o.update({kk: r.get(kk) for kk in EXTRA if kk in r})
            pts.append(o)
        for r in d.get("consolidated_rows", []):
            o = {kk: r.get(kk) for kk in SCHEMA}
            o.update({kk: r.get(kk) for kk in EXTRA if kk in r})
            o["arm"] = "consolidated"
            o["node"] = r.get("node")
            pts.append(o)
        # Each FOM entry is one rung of the ISO-ACCURACY ladder and carries its own
        # achieved error, so the FOM is a CURVE on the same axes as the ROMs rather
        # than a single exact point.  The rung that manufactured the truth has
        # err = 0 by construction and keeps the "price of exactness" vertical-line
        # role.
        for f, tag in ([(x, "fom") for x in d.get("fom", [])]
                       + [(x, "fom_consolidated") for x in d.get("consolidated_fom", [])]):
            t = f.get("fom_cg_s", f.get("fom_rollout_s"))
            pts.append(dict(pde=pde, method="fom", N=f["N"], k=None, M=None, m=None,
                            tau=None, time_ms=t * 1e3,
                            err_rel_l2=f.get("err_rel_l2", 0.0), iters=None,
                            jac_evals=None, censored=False, n_sources=f["n_sources"],
                            seed=cfg["seed"], gpu=f.get("gpu"),
                            jax_backend=f.get("jax_backend"), commit=f.get("commit"),
                            arm=tag, node=f.get("node"), time_ms_solve=t * 1e3,
                            fom_tol=f.get("fom_tol"),
                            fom_newton_iters=f.get("fom_newton_iters"),
                            fom_rule=f.get("fom_rule"),
                            exact_reference=f.get("exact_reference"),
                            achieved_rel_residual=f.get("achieved_rel_residual"),
                            fom_max_rel_residual=f.get("fom_max_rel_residual")))
    return pts


def usable_points(points, time_key="time_ms", err_key="err_rel_l2",
                  require_uncensored=True):
    """Points that may define a "cost at tau" operating point.

    A CENSORED cell stopped for a reason other than reaching its own tau, so its
    cost is the cost of running to termination, not the cost to the tolerance --
    admitting it would reintroduce exactly the defect this cell exists to remove.
    A cell with a blow-up has no usable error at all.  Both are still reported in
    every raw table and in the scatter; they simply cannot define the frontier.
    `require_uncensored=False` gives the AS-DEPLOYED view (set the knob, take
    whatever the solver reaches), which is reported alongside and labelled."""
    out = []
    for p in points:
        if p.get(time_key) is None or p.get(err_key) is None:
            continue
        if not (math.isfinite(p[time_key]) and math.isfinite(p[err_key])):
            continue
        if p.get("n_blowup"):
            continue
        if require_uncensored and (p.get("censored") or p.get("censored_frac")):
            continue
        out.append(p)
    return out


def nondominated(points, time_key="time_ms", err_key="err_rel_l2",
                 require_uncensored=True):
    """Non-dominated set under (minimise time, minimise error).  A point is
    dominated when another point is <= in BOTH coordinates and < in at least
    one."""
    usable = usable_points(points, time_key, err_key, require_uncensored)
    out = []
    for p in usable:
        dom = False
        for q in usable:
            if q is p:
                continue
            if (q[time_key] <= p[time_key] and q[err_key] <= p[err_key]
                    and (q[time_key] < p[time_key] or q[err_key] < p[err_key])):
                dom = True
                break
        if not dom:
            out.append(p)
    return sorted(out, key=lambda p: p[time_key])


def cheapest_reaching(points, target, time_key="time_ms", require_uncensored=True):
    """Cheapest UNCENSORED configuration whose error is <= target, or None."""
    ok = [p for p in usable_points(points, time_key, require_uncensored=require_uncensored)
          if p["err_rel_l2"] <= target]
    return min(ok, key=lambda p: p[time_key]) if ok else None


def md_table(header, rows):
    w = [len(h) for h in header]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for j, c in enumerate(r):
            w[j] = max(w[j], len(c))
    line = lambda cells: "| " + " | ".join(c.ljust(w[j]) for j, c in enumerate(cells)) + " |"
    out = [line(header), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += [line(r) for r in srows]
    return "\n".join(out)


# --------------------------------------------------------------------------
def build_blocks(data, pts):
    B = {}
    primary = [p for p in pts if p.get("arm") in (None, "primary", "fom")]

    # ---- provenance -------------------------------------------------------
    rows = []
    for pde, d in sorted(data.items()):
        for c in sorted(d["panels"], key=lambda x: (bool(x.get("configs")),
                                                    (x.get("ns_measured") or [0])[0])):
            rows.append([pde,
                         "consolidation" if c.get("configs") else
                         ",".join(str(v) for v in (c.get("ns_measured") or c["ns"])),
                         c.get("commit"), c.get("gpu"), c.get("node"), c.get("backend"),
                         c.get("slurm_job") or "--", c.get("matmul_precision"),
                         "yes" if c.get("x64") else "no", c.get("seed"),
                         c.get("n_test"), c.get("time_reps"), c.get("time_warm"),
                         "yes" if c.get("_complete") else "NO", c.get("_file")])
    B["provenance"] = md_table(
        ["pde", "panel (N)", "commit", "gpu", "node", "jax_backend", "slurm job",
         "matmul precision", "f64", "seed", "sources", "time reps", "warm-ups",
         "complete", "file"], rows)

    # ---- configuration ----------------------------------------------------
    rows = []
    for pde, d in sorted(data.items()):
        c = d["config"]
        rows.append([pde, ",".join(str(v) for v in c["ks"]), ",".join(str(v) for v in c["ns"]),
                     ",".join(f"{v:.0e}" for v in c["taus"]),
                     f"{c['M']} (k<{c['k_big']}), {c['M_big']} (k>={c['k_big']})",
                     c["m"], c["cand_cap"], c["eq_snaps"], c["eq_rows"],
                     c["time_ms_definition"]])
    B["configuration"] = md_table(
        ["pde", "k", "N", "tau", "M", "m", "EQ pool cap", "EQ snapshots", "EQ rows",
         "time_ms ="], rows)

    # ---- FOM baseline -----------------------------------------------------
    rows = []
    for pde, d in sorted(data.items()):
        for f in sorted(d.get("fom", []), key=lambda x: (x["N"], -x.get("fom_cg_s", 0)
                                                         - x.get("fom_rollout_s", 0))):
            t = f.get("fom_cg_s", f.get("fom_rollout_s"))
            knob = (f"tol={f['fom_tol']:.0e}" if f.get("fom_tol") is not None
                    else f"Newton={f.get('fom_newton_iters')}")
            rows.append([pde, f["N"], f["n_dof"], knob, fmt(t * 1e3, ".2f"),
                         fmt(f.get("err_rel_l2"), ".3e"),
                         fmt(f.get("achieved_rel_residual"), ".1e"),
                         "yes" if f.get("exact_reference") else ""])
    B["fom"] = md_table(["pde", "N", "interior DOF", "accuracy knob", "FOM ms",
                         "err vs exact", "achieved residual", "truth-manufacturing"], rows)

    # ---- cost vs k, per N (latent solve only) -----------------------------
    for pde, d in sorted(data.items()):
        for tau in d["config"]["taus"]:
            key = f"cost_k_{pde}_{tau:.0e}"
            ks = d["config"]["ks"]
            rows = []
            for method in ("coord", "pod"):
                for N in d["config"]["ns"]:
                    cells = {p["k"]: p for p in primary
                             if p["pde"] == pde and p["method"] == method
                             and p["N"] == N and p["tau"] == tau}
                    rows.append([method, N] + [fmt(cells[k]["time_ms_solve"], ".2f")
                                               if k in cells else "--" for k in ks])
            B[key] = md_table(["method", "N"] + [f"k={k}" for k in ks], rows)

            key = f"iters_k_{pde}_{tau:.0e}"
            rows = []
            for method in ("coord", "pod"):
                for N in d["config"]["ns"]:
                    cells = {p["k"]: p for p in primary
                             if p["pde"] == pde and p["method"] == method
                             and p["N"] == N and p["tau"] == tau}
                    rows.append([method, N] + [fmt(cells[k]["jac_evals"], ".1f")
                                               if k in cells else "--" for k in ks])
            B[key] = md_table(["method", "N"] + [f"k={k}" for k in ks], rows)

            key = f"err_k_{pde}_{tau:.0e}"
            rows = []
            for method in ("coord", "pod"):
                for N in d["config"]["ns"]:
                    cells = {p["k"]: p for p in primary
                             if p["pde"] == pde and p["method"] == method
                             and p["N"] == N and p["tau"] == tau}
                    rows.append([method, N] + [fmt(cells[k]["err_rel_l2"])
                                               if k in cells else "--" for k in ks])
            B[key] = md_table(["method", "N"] + [f"k={k}" for k in ks], rows)

    # ---- N-independence of the k dependence -------------------------------
    for pde, d in sorted(data.items()):
        ks, ns = d["config"]["ks"], d["config"]["ns"]
        rows = []
        for tau in d["config"]["taus"]:
            for method in ("coord", "pod"):
                for N in ns:
                    cells = {p["k"]: p for p in primary
                             if p["pde"] == pde and p["method"] == method
                             and p["N"] == N and p["tau"] == tau}
                    if 8 not in cells:
                        continue
                    base = cells[8]["time_ms_solve"]
                    rows.append([f"{tau:.0e}", method, N]
                                + [fmt(cells[k]["time_ms_solve"] / base, ".2f")
                                   if k in cells and base else "--" for k in ks])
        B[f"kshape_{pde}"] = md_table(["tau", "method", "N"] + [f"k={k}" for k in ks], rows)

    # ---- censoring --------------------------------------------------------
    for pde, d in sorted(data.items()):
        ks = d["config"]["ks"]
        rows = []
        for tau in d["config"]["taus"]:
            for method in ("coord", "pod"):
                for N in d["config"]["ns"]:
                    cells = {p["k"]: p for p in primary
                             if p["pde"] == pde and p["method"] == method
                             and p["N"] == N and p["tau"] == tau}
                    rows.append([f"{tau:.0e}", method, N]
                                + [(fmt(100 * cells[k].get("censored_frac", 0.0), ".0f")
                                    if k in cells else "--") for k in ks])
        B[f"censor_{pde}"] = md_table(["tau", "method", "N"] + [f"k={k}" for k in ks], rows)

    # ---- discrete-residual reference --------------------------------------
    for pde, d in sorted(data.items()):
        rk = "fd_residual_rel_mean" if pde == "poisson2d" else "fom_residual_rel_mean"
        ks = d["config"]["ks"]
        rows = []
        for tau in d["config"]["taus"]:
            for method in ("coord", "pod"):
                for N in d["config"]["ns"]:
                    cells = {p["k"]: p for p in primary
                             if p["pde"] == pde and p["method"] == method
                             and p["N"] == N and p["tau"] == tau}
                    rows.append([f"{tau:.0e}", method, N]
                                + [fmt(cells[k].get(rk), ".2e") if k in cells else "--"
                                   for k in ks])
        B[f"resid_{pde}"] = md_table(["tau", "method", "N"] + [f"k={k}" for k in ks], rows)

    # ---- iso-error Pareto frontier ---------------------------------------
    # STRICT frontier: uncensored, blow-up-free cells only -- a censored cell's
    # cost is "cost to termination", not "cost to tau", and admitting it would
    # reintroduce the defect this cell exists to remove.  The AS-DEPLOYED frontier
    # (all cells; set the knob and take what you get) is tabulated next to it.
    for pde, d in sorted(data.items()):
        for strict, key in ((True, "pareto"), (False, "paretodep")):
            rows = []
            for N in d["config"]["ns"]:
                fom = [p for p in primary if p["pde"] == pde and p["method"] == "fom"
                       and p["N"] == N]
                fom_ms = fom[0]["time_ms"] if fom else None
                for method in ("coord", "pod"):
                    sel = [p for p in primary if p["pde"] == pde
                           and p["method"] == method and p["N"] == N]
                    for p in nondominated(sel, require_uncensored=strict):
                        rows.append([N, method, p["k"], p["M"], p["m"], f"{p['tau']:.0e}",
                                     fmt(p["time_ms"], ".2f"), fmt(p["err_rel_l2"]),
                                     fmt(p["jac_evals"], ".1f"),
                                     fmt(100 * (p.get("censored_frac") or 0.0), ".0f"),
                                     fmt(fom_ms / p["time_ms"], ".1f") if fom_ms else "--"])
            B[f"{key}_{pde}"] = md_table(
                ["N", "method", "k", "M", "m", "tau", "time ms (e2e)", "err rel-L2",
                 "jac evals", "censored %", "x FOM"], rows)

    # ---- who owns the frontier -------------------------------------------
    for pde, d in sorted(data.items()):
        rows = []
        for N in d["config"]["ns"]:
            sel = [p for p in primary if p["pde"] == pde and p["N"] == N
                   and p["method"] in ("coord", "pod")]
            fr = nondominated(sel)
            n_c = sum(1 for p in fr if p["method"] == "coord")
            n_p = sum(1 for p in fr if p["method"] == "pod")
            usable = usable_points(sel)
            best_c = min([p for p in usable if p["method"] == "coord"],
                         key=lambda p: p["err_rel_l2"], default=None)
            best_p = min([p for p in usable if p["method"] == "pod"],
                         key=lambda p: p["err_rel_l2"], default=None)
            owner = ("--" if n_c + n_p == 0 else
                     "coord" if n_p == 0 else "pod" if n_c == 0 else "split")
            rows.append([N, n_c, n_p, owner,
                         fmt(best_c["err_rel_l2"]) if best_c else "--",
                         f"k={best_c['k']}" if best_c else "--",
                         fmt(best_p["err_rel_l2"]) if best_p else "--",
                         f"k={best_p['k']}" if best_p else "--"])
        B[f"owner_{pde}"] = md_table(
            ["N", "coord points on frontier", "pod points on frontier", "owner",
             "best coord err (uncensored)", "at", "best pod err (uncensored)", "at"],
            rows)

    # ---- scaling: cheapest time reaching an error target -------------------
    # The CONFIG is chosen from the panel grid (accuracy is GPU-independent); the TIME
    # is taken from the single-GPU consolidation run whenever it is available, because
    # this is the one table that compares timings ACROSS meshes.
    for pde, d in sorted(data.items()):
        cons = {(p["method"], p["N"], p["k"], p["M"], p["m"], p["tau"]): p
                for p in pts if p["pde"] == pde and p.get("arm") == "consolidated"}
        cons_fom = {p["N"]: p for p in pts if p["pde"] == pde
                    and p.get("arm") == "fom_consolidated"}
        src = "single-GPU consolidation run" if cons else "PANEL JOBS (different GPUs) -- NOT cross-N comparable"
        rows = []
        for target in TARGETS[pde]:
            for N in d["config"]["ns"]:
                r = [f"{target:.0e}", N]
                for method in ("coord", "pod"):
                    sel = [p for p in primary if p["pde"] == pde
                           and p["method"] == method and p["N"] == N]
                    best = cheapest_reaching(sel, target)
                    if best is None:
                        r.append("unreached")
                        continue
                    key = (method, N, best["k"], best["M"], best["m"], best["tau"])
                    t = cons.get(key, best)["time_ms"]
                    r.append(f"{t:.2f} (k={best['k']}, tau={best['tau']:.0e}"
                             + ("" if key in cons else ", PANEL TIME") + ")")
                # ISO-ACCURACY FOM: the cheapest CG tolerance / Newton length that
                # actually reaches the same target.  The exact rung is reported next
                # to it so the price of exactness stays visible.
                fsel = [p for p in pts if p["pde"] == pde and p["method"] == "fom"
                        and p["N"] == N and p.get("arm") in ("fom", "fom_consolidated")]
                fb = cheapest_reaching(fsel, target, require_uncensored=False)
                fex = next((p for p in fsel if p.get("exact_reference")), None)
                r.append(fmt(fb["time_ms"], ".2f") if fb else "unreached")
                r.append(fmt(fex["time_ms"], ".2f") if fex else "--")
                best_rom = min([x for x in (
                    cheapest_reaching([p for p in primary if p["pde"] == pde
                                       and p["method"] == mm and p["N"] == N], target)
                    for mm in ("coord", "pod")) if x], key=lambda p: p["time_ms"],
                    default=None)
                r.append(fmt(fb["time_ms"] / best_rom["time_ms"], ".2f")
                         if (fb and best_rom) else "--")
                rows.append(r)
        B[f"scaling_{pde}"] = (f"_timing source: {src}_\n\n" + md_table(
            ["target rel-L2", "N", "coord ms", "pod ms", "FOM ms (iso-accuracy)",
             "FOM ms (exact)", "best ROM speedup vs iso-accuracy FOM"], rows))

    # ---- consolidation cross-check: panel time vs single-GPU time -----------
    rows = []
    for pde, d in sorted(data.items()):
        cons = [p for p in pts if p["pde"] == pde and p.get("arm") == "consolidated"]
        for c in sorted(cons, key=lambda p: (p["method"], p["N"], p["k"])):
            pan = next((p for p in primary if p["pde"] == pde
                        and p["method"] == c["method"] and p["N"] == c["N"]
                        and p["k"] == c["k"] and p["M"] == c["M"] and p["m"] == c["m"]
                        and p["tau"] == c["tau"]), None)
            rows.append([pde, c["method"], c["N"], c["k"], f"{c['tau']:.0e}",
                         fmt(pan["time_ms"], ".2f") if pan else "--",
                         fmt(c["time_ms"], ".2f"),
                         fmt(c["time_ms"] / pan["time_ms"], ".2f") if pan else "--",
                         fmt(pan["err_rel_l2"]) if pan else "--",
                         fmt(c["err_rel_l2"]),
                         c.get("node")])
    B["consolidation"] = md_table(
        ["pde", "method", "N", "k", "tau", "panel ms", "consolidated ms", "ratio",
         "panel err", "consolidated err", "node"], rows) if rows else \
        "_(the single-GPU consolidation run has not landed yet; the scaling table falls "\
        "back to panel timings, which are NOT cross-N comparable)_"

    # ---- supplementary arms ----------------------------------------------
    rows = []
    for pde, d in sorted(data.items()):
        for s in d.get("supplementary", []):
            rows.append([pde, s["method"], s.get("N"), s.get("k", "--"),
                         s.get("M", "--"), s.get("m", "--"),
                         f"{s['tau']:.0e}" if s.get("tau") else "--",
                         fmt(s.get("time_ms"), ".3f"),
                         fmt(s.get("time_ms_solve"), ".3f"),
                         fmt(s.get("err_rel_l2")),
                         fmt(100 * (s.get("censored_frac") or 0.0), ".0f")
                         if s.get("censored_frac") is not None else "--",
                         fmt(s.get("eq_rel_fit"), ".2e"), s.get("arm")])
    if rows:
        B["supplementary"] = md_table(
            ["pde", "arm method", "N", "k", "M", "m", "tau", "e2e ms", "solve ms",
             "err rel-L2", "censored %", "EQ rel fit", "arm"], rows)
    else:
        B["supplementary"] = "_(no supplementary arms in this run)_"

    # ---- EQ fit quality ---------------------------------------------------
    for pde, d in sorted(data.items()):
        ks = d["config"]["ks"]
        rows = []
        for method in ("coord", "pod"):
            for N in d["config"]["ns"]:
                cells = {p["k"]: p for p in primary
                         if p["pde"] == pde and p["method"] == method and p["N"] == N
                         and p["tau"] == d["config"]["taus"][0]}
                rows.append([method, N] + [fmt(cells[k].get("eq_rel_fit"), ".2e")
                                           if k in cells else "--" for k in ks])
        B[f"eqfit_{pde}"] = md_table(["method", "N"] + [f"k={k}" for k in ks], rows)

    # ---- FOM baseline anchor, per mesh -------------------------------------
    rows = []
    for pde, d in sorted(data.items()):
        for b_ in d.get("fom_baseline", []):
            an = b_.get("anchor_vs_archived") or {}
            pct = 100 * an["rel_diff"] if an.get("rel_diff") is not None else None
            pms = PEER_RETIMED_MS.get((pde, b_["N"]))
            arch = an.get("archived_ms")
            mine = an.get("retimed_ms")
            peer = (100 * abs(pms - arch) / arch) if (pms and arch) else None
            cross = (100 * abs(mine - pms) / pms) if (pms and mine) else None
            oc_ach = b_.get("overconvergence_factor")
            v = anchor_verdict(pct, cross, oc_ach)
            eng = (b_.get("overconvergence_engineering") or {})
            eng_s = "  ".join(f"{kk}:{vv['factor']:.2f}x" for kk, vv in sorted(eng.items()))
            rows.append([pde, b_["N"], fmt(arch, ".2f"), fmt(mine, ".2f"),
                         fmt(pms, ".2f") if pms else "--",
                         fmt(pct, ".1f") if pct is not None else "--",
                         fmt(peer, ".1f") if peer is not None else "--",
                         fmt(cross, ".1f") if cross is not None else "--",
                         fmt(oc_ach, ".2f"), eng_s or "--", v])
    B["anchor"] = md_table(
        ["pde", "N", "archived ms", "re-timed (this cell)", "re-timed (peer cell)",
         "vs archive: this %", "vs archive: peer %", "CROSS-INSTRUMENT %",
         "over-conv (achieved)", "over-conv (engineering)", "verdict"], rows) if rows else \
        "_(no FOM anchor recorded in this run)_"

    # ---- Burgers denominator cross-check against rom-warmstart-fom ---------
    # The two cells ladder DIFFERENT knobs (fixed-length NEWTON_ITERS here, a
    # tolerance-based Newton loop there), so they are reconciled through the peer's
    # MEASURED per-step count rather than compared rung-for-rung.  Two known biases
    # act in OPPOSITE directions and are reported separately rather than netted:
    #   (a) their solve takes 1.97-1.99 steps/step, not 2.00, so an integer-2 rung
    #       here would be 1-3% too slow  -> removed by interpolating the ladder;
    #   (b) their loop performs an outer tolerance test (one residual evaluation per
    #       time step) that a fixed-length loop does not  -> their time carries a few
    #       percent with no counterpart here, so this cell should read slightly LOW.
    rows = []
    d = data.get("burgers2d")
    if d:
        rungs_by_N = {}
        for f in d.get("fom", []):
            if f.get("fom_newton_iters") is not None:
                rungs_by_N.setdefault(f["N"], []).append(f)
        for N, ref in sorted(PEER_BURGERS_TAU1EM6.items()):
            rungs = rungs_by_N.get(N, [])
            t_i, a_, b_, r2 = interp_newton_ladder(rungs, ref["steps_per_step"])
            r2rung = next((r for r in rungs if r["fom_newton_iters"] == 2), None)
            t2 = r2rung["fom_rollout_s"] * 1e3 if r2rung else None
            dev = (100 * (t_i - ref["t_ms"]) / ref["t_ms"]) if t_i else None
            rows.append([N, fmt(ref["t_ms"], ".1f"), ref["steps_per_step"],
                         fmt(t2, ".1f") if t2 else "--",
                         fmt(t_i, ".1f") if t_i else "--",
                         fmt(dev, "+.1f") if dev is not None else "--",
                         fmt(r2, ".4f") if r2 is not None else "--",
                         fmt(b_, ".2f") if b_ is not None else "--",
                         fmt(ref["achieved"], ".2e"),
                         fmt(r2rung.get("achieved_rel_residual"), ".2e") if r2rung else "--"])
    B["burgers_denominator"] = md_table(
        ["N", "peer tau=1e-6 ms", "peer steps/step", "this cell NEWTON_ITERS=2 ms",
         "this cell interpolated to peer steps", "deviation %", "ladder fit R2",
         "marginal ms per Newton step", "peer achieved res", "this cell achieved res"],
        rows) if rows else \
        "_(the Burgers panels have not landed yet; this cross-check is pre-registered "\
        "against the peer values recorded in ctol_tables.PEER_BURGERS_TAU1EM6)_"

    # ---- decoder ceiling per (N, k): the checkpoint's own oracle ------------
    # Accuracy is NON-MONOTONE in k because each k is a SEPARATELY TRAINED
    # checkpoint.  Reporting each checkpoint's own oracle inferred-latent error next
    # to the ROM error turns that confound into a measured quantity.
    for pde, d in sorted(data.items()):
        ks = d["config"]["ks"]
        rows = []
        for N in d["config"]["ns"]:
            cl = {c["k"]: c for c in d.get("supplementary", [])
                  if c.get("method") == "oracle_ceiling" and c.get("N") == N}
            if not cl:
                continue
            rows.append(["ceiling", N] + [fmt(cl[k]["err_rel_l2"]) if k in cl else "--"
                                          for k in ks])
            for tau in d["config"]["taus"]:
                cells = {p["k"]: p for p in primary
                         if p["pde"] == pde and p["method"] == "coord"
                         and p["N"] == N and p["tau"] == tau}
                rows.append([f"ROM tau={tau:.0e}", N]
                            + [fmt(cells[k]["err_rel_l2"]) if k in cells else "--"
                               for k in ks])
            tau_t = min(d["config"]["taus"])
            tight = {p["k"]: p for p in primary
                     if p["pde"] == pde and p["method"] == "coord"
                     and p["N"] == N and p["tau"] == tau_t}
            rows.append([f"ROM / ceiling (tau={tau_t:.0e})", N]
                        + [fmt(tight[k]["err_rel_l2"] / cl[k]["err_rel_l2"], ".2f")
                           if (k in tight and k in cl and cl[k]["err_rel_l2"]
                               and math.isfinite(tight[k]["err_rel_l2"])) else "--"
                           for k in ks])
            rows.append(["ceiling valid (ROM did not beat it)", N]
                        + [("no" if cl[k].get("rom_beat_ceiling") else "yes")
                           if k in cl else "--" for k in ks])
        B[f"ceiling_{pde}"] = (md_table(["quantity", "N"] + [f"k={k}" for k in ks], rows)
                               if rows else "_(no ceiling arm in this run)_")

    # ---- POD projection floor (oracle bound on the POD arm) ----------------
    rows = []
    for pde, d in sorted(data.items()):
        for sup in d.get("supplementary", []):
            if sup.get("method") != "pod_projection_floor":
                continue
            fl = sup.get("floors", {})
            rows.append([pde, sup["N"], sup.get("n_snapshots"),
                         fmt(sup.get("orthonormality_dev"), ".1e")]
                        + [fmt(fl.get(str(k))) for k in sorted(int(x) for x in fl)])
    if rows:
        ks_f = sorted(int(x) for x in next(
            sup["floors"] for d in data.values() for sup in d.get("supplementary", [])
            if sup.get("method") == "pod_projection_floor"))
        B["podfloor"] = md_table(["pde", "N", "snapshots", "orthonorm dev"]
                                 + [f"k={k}" for k in ks_f], rows)
    else:
        B["podfloor"] = "_(no POD projection floors in this run)_"
    return B


TARGETS = {"poisson2d": [2e-2, 1e-2], "burgers2d": [5e-2, 2e-2]}

# The ANCHOR is a PER-MESH property, not a global licence.  Re-timing the archived
# baseline function agrees to single-digit percent at the fine meshes and fails badly
# at the coarse end, because a coarse solve is dominated by per-iteration kernel-launch
# overhead, which does not transfer between environments or driver versions.  Two
# independent re-timings (this cell and rom-warmstart-fom) found the same thing:
#   N=32   archived 5.591 ms   this 4.74   peer 3.548  -> 15% / 37%   FAILS
#   N=64   archived 7.786 ms   this 7.144  peer 7.124  ->  8% /  8.5% marginal
#   N=128  archived 15.145 ms  this 14.795 peer 14.860 ->  2% /  1.9% ok
#   N=256  archived 31.135 ms              peer 29.610 ->        4.9% ok
#   N=512  archived 96.010 ms              peer 93.070 ->        3.1% ok
# Note the two re-timings agree with EACH OTHER at N=64 to 0.3% while both sit ~8%
# below the archive: at that mesh the archive is the outlier, not the re-timings.
ANCHOR_OK_PCT = 10.0          # "single-digit percent" per the shared rule
# The rom-warmstart-fom cell's own re-timing of the SAME archived function, in ms.
# Having a SECOND instrument is what separates "the archive is stale" from "nobody
# has a trustworthy number": those two look identical if you only compare against
# the archive.
PEER_RETIMED_MS = {("poisson2d", 32): 3.548, ("poisson2d", 64): 7.124,
                   ("poisson2d", 128): 14.860, ("poisson2d", 256): 29.610,
                   ("poisson2d", 512): 93.070}


# The rom-warmstart-fom cell's tolerance-based Burgers FOM at tau=1e-6 (mean over 4
# held-out trajectories, A100 80GB, job 2511371, burn-in on, paired timing).  Recorded
# BEFORE this cell's Burgers panels landed, so the cross-check is pre-registered.
PEER_BURGERS_TAU1EM6 = {
    32:  dict(t_ms=47.6,  steps_per_step=1.97, achieved=9.71e-07),
    64:  dict(t_ms=85.6,  steps_per_step=1.97, achieved=9.83e-07),
    128: dict(t_ms=228.1, steps_per_step=1.98, achieved=9.94e-07),
    256: dict(t_ms=449.9, steps_per_step=1.99, achieved=9.57e-07),
}


def interp_newton_ladder(rungs, target_steps):
    """Interpolate this cell's fixed-length Newton ladder to a FRACTIONAL per-step
    count.

    The peer's tolerance solve converges in 1.97-1.99 Newton steps per time step,
    not 2.00, so comparing their time against this cell's integer NEWTON_ITERS=2
    rung carries a systematic 1-3% bias.  The ladder {1,2,3,4,6,8} pins a linear
    cost model t(k) = a + b*k -- `a` the fixed per-rollout overhead, `b` the marginal
    cost of one Newton step per time step -- which can be evaluated at their exact
    count instead.  Returns (t_at_target_ms, a_ms, b_ms, r2)."""
    pts = sorted((r["fom_newton_iters"], r["fom_rollout_s"] * 1e3) for r in rungs
                 if r.get("fom_newton_iters") is not None)
    if len(pts) < 2:
        return None, None, None, None
    xs = [float(x) for x, _ in pts]
    ys = [float(y) for _, y in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None, None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return a + b * target_steps, a, b, r2


def anchor_verdict(pct, cross_pct, overconv_achieved):
    """THREE states, not two.

    A two-state rule (anchor tight -> time, else iterations) cannot tell a STALE
    ARCHIVE from an UNMEASURABLE MESH, because both present as a large archive
    disagreement.  The discriminator is whether a second, independent re-timing
    agrees with ours:

      anchor tight                          -> "time"
      anchor loose, re-timings AGREE        -> the archive is stale; the re-timed
                                               value is the better baseline and the
                                               "correction" is a re-measurement
      anchor loose, re-timings DISAGREE     -> "uncorrected"

    Overlaid on all three: where the achieved-residual over-convergence factor is
    exactly 1.00 there is nothing to correct in the first place."""
    if pct is None or not math.isfinite(pct):
        return "unknown"
    nothing = (overconv_achieved is not None
               and math.isfinite(overconv_achieved)
               and abs(overconv_achieved - 1.0) < 1e-9)
    if pct < ANCHOR_OK_PCT:
        base = "time"
    elif cross_pct is not None and math.isfinite(cross_pct) and cross_pct < ANCHOR_OK_PCT:
        base = "archive stale -> use the re-timed baseline"
    else:
        base = "UNCORRECTED (no instrument agrees)"
    return base + (" [nothing to correct: over-conv 1.00]" if nothing else "")


def rewrite(readme, blocks):
    if not os.path.isfile(readme):
        print(f"  (no README at {readme}; skipping rewrite)")
        return 0
    txt = open(readme).read()
    n = 0
    for name, body in blocks.items():
        pat = re.compile(r"(<!-- BEGIN GENERATED: " + re.escape(name)
                         + r" -->\n).*?(<!-- END GENERATED: " + re.escape(name) + r" -->)",
                         re.S)
        new, cnt = pat.subn(lambda mo: mo.group(1) + body + "\n" + mo.group(2), txt)
        if cnt:
            txt, n = new, n + cnt
    missing = [name for name in blocks
               if f"<!-- BEGIN GENERATED: {name} -->" not in txt]
    open(readme, "w").write(txt)
    print(f"  rewrote {n} generated blocks in {os.path.basename(readme)}"
          + (f"; {len(missing)} blocks have no marker: {missing[:8]}" if missing else ""))
    return n


# The grid this cell is SPECIFIED to deliver.  Checking coverage against the union
# of what actually arrived would never notice a panel that failed to land at all.
EXPECTED = {
    "poisson2d": dict(ns=[32, 64, 128, 256, 512], ks=[2, 4, 6, 8, 12, 16, 24, 32],
                      taus=[1e-1, 1e-2, 1e-3]),
    "burgers2d": dict(ns=[32, 64, 128, 256], ks=[2, 4, 6, 8, 12, 16, 24, 32],
                      taus=[1e-1, 1e-2, 1e-3]),
}


def audit(data, allow_incomplete=False):
    """Refuse to build tables from a partial surface.

    The drivers save incrementally and pull.sh copies the output of a job that
    failed, so a crashed panel would otherwise become a quietly incomplete
    surface.  Every panel must have finished, every expected
    (N, k, M, m, method, tau) primary cell must be present exactly once, and the
    configuration/backend must agree across panels."""
    problems = []
    for pde, d in sorted(data.items()):
        for c in d["panels"]:
            if not c.get("_complete"):
                problems.append(f"{pde}: panel {c.get('_file')} did not finish "
                                f"(no `complete: true`)")
            if c.get("backend") != "gpu":
                problems.append(f"{pde}: panel {c.get('_file')} ran on "
                                f"backend={c.get('backend')!r}, not gpu")
            if c.get("matmul_precision") != "highest":
                problems.append(f"{pde}: panel {c.get('_file')} had "
                                f"JAX_DEFAULT_MATMUL_PRECISION={c.get('matmul_precision')!r}")
            if not c.get("x64"):
                problems.append(f"{pde}: panel {c.get('_file')} did not run in f64")
        seen = {}
        for r in d["rows"]:
            key = (r["N"], r["k"], r["M"], r["m"], r["method"], r["tau"], r.get("arm"))
            seen[key] = seen.get(key, 0) + 1
        dup = [k for k, v in seen.items() if v > 1]
        if dup:
            problems.append(f"{pde}: {len(dup)} duplicated cells, e.g. {dup[:3]}")
        cfg = d["config"]
        kb, mb, mq = cfg["k_big"], cfg["M_big"], cfg["m"]
        m4m = cfg.get("mq_4m", 4 * mb)
        exp = EXPECTED.get(pde, dict(ns=cfg["ns"], ks=cfg["ks"], taus=cfg["taus"]))
        for miss in sorted(set(exp["ns"]) - set(cfg["ns"])):
            problems.append(f"{pde}: mesh N={miss} is missing entirely (no panel landed)")
        for miss in sorted(set(exp["ks"]) - set(cfg["ks"])):
            problems.append(f"{pde}: k={miss} is missing entirely")
        for N in sorted(set(cfg["ns"]) & set(exp["ns"])):
            for k in exp["ks"]:
                M = mb if k >= kb else cfg["M"]
                # m is capped by the interior itself: at N=32 there are only 900
                # interior nodes, so an m=1024 request yields m=900 and the cell is
                # present, not missing.
                mexp = min(m4m if k >= kb else mq, (N - 2) ** 2)
                for method in ("coord", "pod"):
                    for tau in exp["taus"]:
                        if (N, k, M, mexp, method, tau, "primary") not in seen:
                            problems.append(f"{pde}: missing primary cell "
                                            f"N={N} k={k} M={M} m={mexp} {method} "
                                            f"tau={tau:.0e}")
    if problems:
        head = f"{len(problems)} surface-integrity problem(s):\n  " + "\n  ".join(problems[:40])
        if not allow_incomplete:
            raise SystemExit(head + "\n(pass --allow-incomplete to build provisional "
                                    "tables from a partial surface)")
        print("  WARNING (--allow-incomplete): " + head, file=sys.stderr)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--readme", default=os.path.join(HERE, "README.md"))
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="build provisional tables from a partial surface, stamping the "
                         "problems into the README's integrity block")
    args = ap.parse_args()
    data = load(args.runs)
    if not data:
        raise SystemExit("no surface JSONs found")
    problems = audit(data, args.allow_incomplete)
    pts = to_points(data)
    out = os.path.join(args.runs, "pareto_points.json")
    json.dump(pts, open(out, "w"), indent=1)
    print(f"  wrote {out}: {len(pts)} points "
          f"({sum(1 for p in pts if p['method']=='coord')} coord, "
          f"{sum(1 for p in pts if p['method']=='pod')} pod, "
          f"{sum(1 for p in pts if p['method']=='fom')} fom)")
    # schema guard: every point must carry every schema key
    for p in pts:
        missing = [k for k in SCHEMA if k not in p]
        if missing:
            raise SystemExit(f"pareto point missing schema keys {missing}: {p}")
        if p["method"] not in ("coord", "pod", "fom"):
            raise SystemExit(f"pareto point has method '{p['method']}' outside the schema")
    blocks = build_blocks(data, pts)
    blocks["integrity"] = ("_no integrity problems: every panel complete, every expected "
                           "primary cell present exactly once, every panel on GPU in f64 at "
                           "matmul precision `highest`._"
                           if not problems else
                           "**PROVISIONAL -- the surface is incomplete:**\n\n"
                           + "\n".join(f"* {t}" for t in problems[:40]))
    rewrite(args.readme, blocks)


if __name__ == "__main__":
    main()
