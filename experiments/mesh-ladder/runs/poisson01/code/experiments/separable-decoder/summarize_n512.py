"""Generate the N=512 scaling-round summary tables FROM THE RUN JSONS.

Never hand-type a number: this script is the only path from
runs/sepdec_n512_*/out/*.json to the reported tables.

Usage:  python summarize_n512.py [runs_glob ...]   (default runs/sepdec_n512_*/out)
Writes markdown to stdout.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def fmt(x, spec=".3e"):
    if x is None:
        return "--"
    try:
        if not np.isfinite(x):
            return "nan"
    except TypeError:
        return str(x)
    return format(x, spec)


def load(patterns):
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(pat, "*.json"))
    out = []
    for f in sorted(files):
        try:
            d = json.load(open(f))
        except Exception as e:                                    # noqa: BLE001
            print(f"<!-- SKIP {f}: {e} -->")
            continue
        d["_file"] = f
        out.append(d)
    return out


def poisson_table(runs):
    rows = []
    for d in runs:
        c = d.get("config", {})
        if c.get("pde") != "poisson2d":
            continue
        tr = d.get("train", {})
        orc = d.get("oracle_test_rel_l2", {})
        orf = d.get("oracle_fresh_rel_l2", {})
        eq = d.get("eq", {})
        for r in d.get("rows", []):
            if r.get("method") != "sep_cached":
                continue
            rows.append(dict(
                K=c["k"], R=c["r"], nff=c.get("n_ff"), steps=c.get("steps"),
                cohort=r.get("cohort"), tau=r["tau"],
                train_s=tr.get("seconds"), recon=tr.get("recon_rel_l2_mean"),
                oracle=orc.get("mean"), oracle_fresh=orf.get("mean"),
                err=r.get("err_rel_l2"), err_max=r.get("err_rel_l2_max"),
                cens=r.get("censored_frac"), jac=r.get("jac_evals"),
                rom_ms=r.get("time_ms"),
                gate0=d.get("gate0_max_rel_dev"),
                eq_relfit=eq.get("rel_fit"), eq_rowmax=eq.get("row_rel_max"),
                complete=d.get("complete"), file=os.path.basename(d["_file"])))
    if not rows:
        return
    print("\n## Poisson 2D, N=512 (sep_cached arm)\n")
    print("| K | R | steps | cohort | tau | train s | recon | oracle(held/fresh) |"
          " solve err (max) | cens | jac | ROM ms | gate0 | EQ relfit / rowmax |")
    print("|--:|--:|--:|:--|--:|--:|--:|:--|:--|--:|--:|--:|--:|:--|")
    for r in sorted(rows, key=lambda x: (x["K"], x["R"], x["cohort"], x["tau"])):
        print(f"| {r['K']} | {r['R']} | {r['steps']} | {r['cohort']} | "
              f"{fmt(r['tau'],'.0e')} | {fmt(r['train_s'],'.0f')} | "
              f"{fmt(r['recon'])} | {fmt(r['oracle'])} / {fmt(r['oracle_fresh'])} | "
              f"{fmt(r['err'])} ({fmt(r['err_max'])}) | "
              f"{fmt(r['cens'],'.0%') if r['cens'] is not None else '--'} | "
              f"{fmt(r['jac'],'.1f')} | {fmt(r['rom_ms'],'.3f')} | "
              f"{fmt(r['gate0'],'.1e')} | "
              f"{fmt(r['eq_relfit'],'.1e')} / {fmt(r['eq_rowmax'],'.1e')} |")
    # CG ladder (one per job; identical across cells -- take from each file)
    seen = set()
    print("\n### Same-job FOM CG ladder (timed, balanced schedule)\n")
    print("| file | tol | err | err max | ms |")
    print("|:--|--:|--:|--:|--:|")
    for d in runs:
        if d.get("config", {}).get("pde") != "poisson2d":
            continue
        key = os.path.basename(d["_file"])
        for fr in d.get("fom", []):
            print(f"| {key} | {fmt(fr['fom_tol'],'.0e')} | {fmt(fr['err_rel_l2'])} | "
                  f"{fmt(fr['err_rel_l2_max'])} | {fmt(fr.get('time_ms'),'.3f')} |")
        seen.add(key)


def burgers_table(runs):
    rows = []
    for d in runs:
        c = d.get("config", {})
        if c.get("pde") != "burgers2d":
            continue
        tr = d.get("train", {})
        eq = d.get("eq", {})
        for r in d.get("rows", []):
            rows.append(dict(
                K=c["k"], R=c["r"], nff=c.get("n_ff"), steps=c.get("steps"),
                method=r["method"], err=r.get("err_traj_rel_mean"),
                err_max=r.get("err_traj_rel_max"),
                jac=r.get("jac_total_mean"), blow=r.get("n_blowups"),
                train_s=tr.get("seconds"), recon=tr.get("recon_rel_l2_mean"),
                e2e_ms=r.get("e2e_ms_median"), ic_ms=r.get("icfit_ms_median"),
                rd_ms=r.get("rollout_decode_ms_median"),
                gate0=d.get("gate0_max_rel_dev"),
                eq_relfit=eq.get("rel_fit"), eq_rowmax=eq.get("row_rel_max"),
                ics=[t.get("ic_rel") for t in r.get("per_traj", [])],
                reasons=_merge([t.get("stop_reasons", {})
                                for t in r.get("per_traj", [])]),
                complete=d.get("complete"), file=os.path.basename(d["_file"])))
    if not rows:
        return
    print("\n## Burgers 2D, N=512 (end-to-end: IC fit + 50-step rollout + full decode)\n")
    print("| K | R | steps | arm | train s | recon | traj err (max) | IC fits |"
          " jac | e2e ms (ic + roll/dec) | stop reasons | gate0 | EQ relfit/rowmax |")
    print("|--:|--:|--:|:--|--:|--:|:--|:--|--:|:--|:--|--:|:--|")
    for r in sorted(rows, key=lambda x: (x["K"], x["R"], x["method"])):
        ics = " ".join(fmt(v, ".1e") for v in r["ics"]) if r["ics"] else "--"
        split = (f"{fmt(r['e2e_ms'],'.1f')} ({fmt(r['ic_ms'],'.1f')} + "
                 f"{fmt(r['rd_ms'],'.1f')})" if r["ic_ms"] is not None
                 else fmt(r["e2e_ms"], ".1f"))
        print(f"| {r['K']} | {r['R']} | {r['steps']} | {r['method']} | "
              f"{fmt(r['train_s'],'.0f')} | {fmt(r['recon'])} | "
              f"{fmt(r['err'])} ({fmt(r['err_max'])}) | {ics} | "
              f"{fmt(r['jac'],'.0f')} | {split} | {r['reasons']} | "
              f"{fmt(r['gate0'],'.1e')} | "
              f"{fmt(r['eq_relfit'],'.1e')} / {fmt(r['eq_rowmax'],'.1e')} |")
    print("\n### Same-job classical Burgers baselines (per job)\n")
    print("| file | method | ntol | lin | err mean (max) | median ms |")
    print("|:--|:--|--:|--:|:--|--:|")
    for d in runs:
        if d.get("config", {}).get("pde") != "burgers2d":
            continue
        key = os.path.basename(d["_file"])
        meds = d.get("timing", {}).get("summary", {})
        tg = meds.get("fom_truth_newton8", {})
        print(f"| {key} | fom_truth_newton8 (OVER-SOLVED truth gen) | -- | -- | "
              f"0 by construction | {fmt(tg.get('median_ms'),'.1f')} |")
        for br in d.get("fom_newton_tol", []):
            nm = f"fom_newton_ntol{br['ntol']:g}_lin{br['lin_tol']:g}"
            t = meds.get(nm, {})
            print(f"| {key} | fom_newton_tol | {fmt(br['ntol'],'.0e')} | "
                  f"{fmt(br['lin_tol'],'.0e')} | {fmt(br['err_traj_rel_mean'])} "
                  f"({fmt(br['err_traj_rel_max'])}) | "
                  f"{fmt(t.get('median_ms'),'.1f')} |")


def _merge(hists):
    out = {}
    for h in hists:
        for k, v in (h or {}).items():
            out[k] = out.get(k, 0) + v
    return out


def main():
    pats = sys.argv[1:] or sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs", "sepdec_n512_*", "out")))
    runs = load(pats)
    print(f"<!-- generated by summarize_n512.py from {len(runs)} JSONs: "
          f"{[os.path.basename(d['_file']) for d in runs]} -->")
    incomplete = [os.path.basename(d["_file"]) for d in runs
                  if not d.get("complete")]
    if incomplete:
        print(f"\n**WARNING: incomplete cells (numbers provisional): {incomplete}**")
    poisson_table(runs)
    burgers_table(runs)


if __name__ == "__main__":
    main()
