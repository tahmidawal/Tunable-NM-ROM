"""Generate the N=128 scaling-round summary tables FROM THE RUN JSONs.

Never hand-type a number: this script is the only path from runs/*/out/*.json
to the tables quoted in reports and the lab log.

Usage:  python summarize_n128.py [runs_glob ...]   (default: runs/sepdec_n128_*/out)
Writes SUMMARY-N128.md next to this script and prints it.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(patterns):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            with open(p) as f:
                d = json.load(f)
            d["_path"] = os.path.relpath(p, HERE)
            out.append(d)
    return out


def fmt(v, spec="{:.3e}"):
    if v is None:
        return "--"
    try:
        if isinstance(v, float) and not np.isfinite(v):
            return "nan"
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


def poisson_tables(docs, lines):
    docs = [d for d in docs if d["config"]["pde"] == "poisson2d"]
    if not docs:
        return
    lines.append("\n## Poisson 2D (stationary weak-EQ solve)\n")
    lines.append("### Per-cell training / oracle / gate\n")
    lines.append("| job | K | R | steps | train s | recon mean | oracle held | "
                 "oracle fresh | gate0 | EQ rel fit | EQ row p95 | EQ row max |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        c, t = d["config"], d.get("train", {})
        orc = d.get("oracle_test_rel_l2", {})
        held = next((v["mean"] for k, v in orc.items() if "held" in k), None)
        fresh = next((v["mean"] for k, v in orc.items() if "fresh" in k), None)
        eq = d.get("eq", {})
        lines.append(
            f"| {os.path.basename(os.path.dirname(os.path.dirname(d['_path'])))} "
            f"| {c['k']} | {c['r']} | {c['steps']} | {fmt(t.get('seconds'), '{:.0f}')} "
            f"| {fmt(t.get('recon_rel_l2_mean'))} | {fmt(held)} | {fmt(fresh)} "
            f"| {fmt(d.get('gate0_max_rel_dev'), '{:.1e}')} "
            f"| {fmt(eq.get('rel_fit'), '{:.1e}')} | {fmt(eq.get('row_rel_p95'), '{:.1e}')} "
            f"| {fmt(eq.get('row_rel_max'), '{:.1e}')} |")
    lines.append("\n### Solve rows (errors and counters from the LAST TIMED "
                 "invocation; time = median of per-source medians, raw reps in "
                 "the JSONs)\n")
    lines.append("| K | cohort | method | tau/tol | time ms | err mean | err max "
                 "| jac | censored | stop reasons |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        for r in d["rows"]:
            par = r.get("tau", r.get("fom_tol"))
            lines.append(
                f"| {d['config']['k']} | {r.get('cohort','--')} | {r['method']} "
                f"| {fmt(par, '{:.0e}')} | {fmt(r.get('time_ms'), '{:.3f}')} "
                f"| {fmt(r.get('err_rel_l2'))} | {fmt(r.get('err_rel_l2_max'))} "
                f"| {fmt(r.get('jac_evals'), '{:.1f}')} "
                f"| {fmt(r.get('censored_frac'), '{:.0%}') if r.get('censored_frac') is not None else '--'} "
                f"| {r.get('stop_reasons', '--')} |")


def burgers_tables(docs, lines):
    docs = [d for d in docs if d["config"]["pde"] == "burgers2d"]
    if not docs:
        return
    lines.append("\n## Burgers 2D (50-step implicit rollout, END-TO-END: "
                 "IC fit + rollout + full-grid decode)\n")
    lines.append("### Per-cell training / gate\n")
    lines.append("| job | K | R | steps | train s | recon mean | gate0 | "
                 "EQ rel fit | EQ row p95 | EQ row max | IC check (incumbent vs batched) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        c, t = d["config"], d.get("train", {})
        eq = d.get("eq", {})
        icc = d.get("ic_fit_check", [])
        ics = "; ".join(f"t{e['traj']}: {e['incumbent_rel']:.2e} vs "
                        f"{e['batched_rel']:.2e}" for e in icc)
        lines.append(
            f"| {os.path.basename(os.path.dirname(os.path.dirname(d['_path'])))} "
            f"| {c['k']} | {c['r']} | {c['steps']} | {fmt(t.get('seconds'), '{:.0f}')} "
            f"| {fmt(t.get('recon_rel_l2_mean'))} "
            f"| {fmt(d.get('gate0_max_rel_dev'), '{:.1e}')} "
            f"| {fmt(eq.get('rel_fit'), '{:.1e}')} | {fmt(eq.get('row_rel_p95'), '{:.1e}')} "
            f"| {fmt(eq.get('row_rel_max'), '{:.1e}')} | {ics} |")
    lines.append("\n### Rollout / baseline rows (per-trajectory raw reps and "
                 "splits in the JSONs)\n")
    lines.append("| K | method | ntol | err mean | err max | e2e ms | ic ms | "
                 "roll ms | dec ms | jac | ic_rel mean/max | blowups | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        for r in d["rows"]:
            icr = (f"{fmt(r.get('ic_rel_mean'), '{:.2e}')}/"
                   f"{fmt(r.get('ic_rel_max'), '{:.2e}')}"
                   if r.get("ic_rel_mean") is not None else "--")
            lines.append(
                f"| {d['config']['k']} | {r['method']} "
                f"| {fmt(r.get('ntol'), '{:.0e}')} "
                f"| {fmt(r.get('err_traj_rel_mean'))} | {fmt(r.get('err_traj_rel_max'))} "
                f"| {fmt(r.get('e2e_ms_median'), '{:.2f}')} "
                f"| {fmt(r.get('ic_ms_median'), '{:.2f}')} "
                f"| {fmt(r.get('roll_ms_median'), '{:.2f}')} "
                f"| {fmt(r.get('dec_ms_median'), '{:.2f}')} "
                f"| {fmt(r.get('jac_total_mean'), '{:.0f}')} | {icr} "
                f"| {r.get('n_blowups', '--')} | {r.get('label', '')} |")
    # stop-reason roll-up per cell/method
    lines.append("\n### Stop-reason distributions (sep arms, summed over "
                 "trajectories and steps)\n")
    lines.append("| K | method | reasons |")
    lines.append("|---|---|---|")
    for d in docs:
        for r in d["rows"]:
            if not r["method"].startswith("sep_"):
                continue
            agg = {}
            for e in r.get("per_traj", []):
                for k_, v in (e.get("stop_reasons") or {}).items():
                    agg[k_] = agg.get(k_, 0) + v
            lines.append(f"| {d['config']['k']} | {r['method']} | {agg} |")


def main():
    pats = sys.argv[1:] or [os.path.join(HERE, "runs", "sepdec_n128_*", "out",
                                         "*.json")]
    pats = [p if p.endswith(".json") else os.path.join(p, "*.json")
            for p in pats]
    docs = load(pats)
    lines = ["# N=128 separable-decoder scaling round -- generated summary",
             "",
             f"Generated by summarize_n128.py from {len(docs)} run JSONs. "
             "Do not edit numbers by hand.",
             "", "Sources:"]
    lines += [f"- `{d['_path']}` (complete={d.get('complete')})" for d in docs]
    poisson_tables(docs, lines)
    burgers_tables(docs, lines)
    text = "\n".join(lines) + "\n"
    out = os.path.join(HERE, "SUMMARY-N128.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f"[written {out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
