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
                 consolidated_rows=[], consolidated_fom=[], files=[], complete=True)
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
        for f, tag in ([(x, "fom") for x in d.get("fom", [])]
                       + [(x, "fom_consolidated") for x in d.get("consolidated_fom", [])]):
            t = f.get("fom_cg_s", f.get("fom_rollout_s"))
            pts.append(dict(pde=pde, method="fom", N=f["N"], k=None, M=None, m=None,
                            tau=None, time_ms=t * 1e3, err_rel_l2=0.0, iters=None,
                            jac_evals=None, censored=False, n_sources=f["n_sources"],
                            seed=cfg["seed"], gpu=f.get("gpu"),
                            jax_backend=f.get("jax_backend"), commit=f.get("commit"),
                            arm=tag, node=f.get("node"), time_ms_solve=t * 1e3,
                            fom_max_rel_residual=f.get("fom_max_rel_residual")))
    return pts


def nondominated(points, time_key="time_ms", err_key="err_rel_l2"):
    """Non-dominated set under (minimise time, minimise error).  A point is
    dominated when another point is <= in BOTH coordinates and < in at least
    one.  Points with a non-finite error are never on the frontier (they are
    kept in the scatter and reported, but they are not a usable operating
    point)."""
    usable = [p for p in points
              if p.get(time_key) is not None and p.get(err_key) is not None
              and math.isfinite(p[time_key]) and math.isfinite(p[err_key])]
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


def cheapest_reaching(points, target, time_key="time_ms"):
    """Cheapest configuration whose error is <= target, or None."""
    ok = [p for p in points if p.get("err_rel_l2") is not None
          and math.isfinite(p["err_rel_l2"]) and p["err_rel_l2"] <= target
          and p.get(time_key) is not None and math.isfinite(p[time_key])]
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
        for f in d.get("fom", []):
            t = f.get("fom_cg_s", f.get("fom_rollout_s"))
            rows.append([pde, f["N"], f["n_dof"], fmt(t * 1e3, ".2f"),
                         fmt(f.get("fom_max_rel_residual"), ".1e")])
    B["fom"] = md_table(["pde", "N", "interior DOF", "FOM ms", "FOM rel residual"], rows)

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
    for pde, d in sorted(data.items()):
        rows = []
        for N in d["config"]["ns"]:
            fom = [p for p in primary if p["pde"] == pde and p["method"] == "fom"
                   and p["N"] == N]
            fom_ms = fom[0]["time_ms"] if fom else None
            for method in ("coord", "pod"):
                sel = [p for p in primary if p["pde"] == pde and p["method"] == method
                       and p["N"] == N]
                fr = nondominated(sel)
                for p in fr:
                    rows.append([N, method, p["k"], f"{p['tau']:.0e}",
                                 fmt(p["time_ms"], ".2f"), fmt(p["err_rel_l2"]),
                                 fmt(p["jac_evals"], ".1f"),
                                 "yes" if p["censored"] else "no",
                                 fmt(fom_ms / p["time_ms"], ".1f") if fom_ms else "--"])
        B[f"pareto_{pde}"] = md_table(
            ["N", "method", "k", "tau", "time ms (e2e)", "err rel-L2", "jac evals",
             "censored", "x FOM"], rows)

    # ---- who owns the frontier -------------------------------------------
    for pde, d in sorted(data.items()):
        rows = []
        for N in d["config"]["ns"]:
            sel = [p for p in primary if p["pde"] == pde and p["N"] == N
                   and p["method"] in ("coord", "pod")]
            fr = nondominated(sel)
            n_c = sum(1 for p in fr if p["method"] == "coord")
            n_p = sum(1 for p in fr if p["method"] == "pod")
            best_c = min([p for p in sel if p["method"] == "coord"
                          and math.isfinite(p["err_rel_l2"])],
                         key=lambda p: p["err_rel_l2"], default=None)
            best_p = min([p for p in sel if p["method"] == "pod"
                          and math.isfinite(p["err_rel_l2"])],
                         key=lambda p: p["err_rel_l2"], default=None)
            owner = ("coord" if n_p == 0 else "pod" if n_c == 0 else "split")
            rows.append([N, n_c, n_p, owner,
                         fmt(best_c["err_rel_l2"]) if best_c else "--",
                         f"k={best_c['k']}" if best_c else "--",
                         fmt(best_p["err_rel_l2"]) if best_p else "--",
                         f"k={best_p['k']}" if best_p else "--"])
        B[f"owner_{pde}"] = md_table(
            ["N", "coord points on frontier", "pod points on frontier", "owner",
             "best coord err", "at", "best pod err", "at"], rows)

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
                fom = cons_fom.get(N) or next(
                    (p for p in primary if p["pde"] == pde and p["method"] == "fom"
                     and p["N"] == N), None)
                r.append(fmt(fom["time_ms"], ".2f") if fom else "--")
                rows.append(r)
        B[f"scaling_{pde}"] = (f"_timing source: {src}_\n\n" + md_table(
            ["target rel-L2", "N", "coord ms", "pod ms", "FOM ms"], rows))

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
            rows.append([pde, s["method"], s["N"], s["k"], s.get("M"), s.get("m"),
                         f"{s['tau']:.0e}" if s.get("tau") else "--",
                         fmt(s.get("time_ms_solve"), ".3f"),
                         fmt(s.get("err_rel_l2")), s.get("arm")])
    if rows:
        B["supplementary"] = md_table(
            ["pde", "arm method", "N", "k", "M", "m", "tau", "solve ms", "err rel-L2",
             "arm"], rows)
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

    return B


TARGETS = {"poisson2d": [2e-2, 1e-2], "burgers2d": [5e-2, 2e-2]}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--readme", default=os.path.join(HERE, "README.md"))
    args = ap.parse_args()
    data = load(args.runs)
    if not data:
        raise SystemExit("no surface JSONs found")
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
    rewrite(args.readme, blocks)


if __name__ == "__main__":
    main()
