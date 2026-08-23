#!/usr/bin/env python3
"""Iso-accuracy ROM-vs-classical comparison FROM THE RUN JSONS.

For every ROM row, find the CHEAPEST same-job classical arm whose mean error
is <= the ROM's mean error (iso-accuracy-or-better), and report the time
ratio.  A ratio > 1 means the ROM is faster than the matched classical arm;
the honest headline is the CACHED arm vs that matched arm, with censoring
stated.  Never compares against the over-solved truth generator.

Usage: python runs/crossover_n256.py runs/<job>/out/*.json
"""
import json
import sys

import numpy as np


def poisson(d, name):
    foms = [f for f in d.get("fom", [])]
    for r in d.get("rows", []):
        if not r["method"].startswith("sep_"):
            continue
        cands = [f for f in foms if f["cohort"] == r["cohort"]
                 and f["err_rel_l2"] <= r["err_rel_l2"]]
        if not cands:
            print(f"{name} {r['method']} tau={r['tau']:.0e} [{r['cohort']}]: "
                  f"err {r['err_rel_l2']:.3e} in {r['time_ms']:.3f} ms -- NO CG "
                  f"rung reaches this error (ROM err above whole ladder?)")
            continue
        best = min(cands, key=lambda f: f["time_ms"])
        ratio = best["time_ms"] / r["time_ms"]
        print(f"{name} {r['method']} tau={r['tau']:.0e} [{r['cohort']}]: "
              f"err {r['err_rel_l2']:.3e} in {r['time_ms']:.3f} ms vs CG "
              f"tol={best['fom_tol']:.0e} (err {best['err_rel_l2']:.3e}) "
              f"{best['time_ms']:.3f} ms -> ROM is {ratio:.2f}x "
              f"{'FASTER' if ratio > 1 else 'slower'} "
              f"(cens {r['censored_frac']*100:.0f}%)")


def burgers(d, name):
    bases = [r for r in d.get("rows", []) if r["method"] == "fom_newton_tol"]
    for r in d.get("rows", []):
        if not r["method"].startswith("sep_"):
            continue
        err = r["err_traj_rel_mean"]
        cands = [b for b in bases if b["err_traj_rel_mean"] <= err]
        t_rom = r["e2e_ms_median"]
        if not cands:
            print(f"{name} {r['method']}: err {err:.3e} in {t_rom:.2f} ms "
                  f"(e2e) -- no tol-Newton rung is this loose; loosest rung "
                  f"err {max(b['err_traj_rel_mean'] for b in bases):.3e} at "
                  f"{min(b['time_ms_median'] for b in bases):.1f} ms")
            continue
        best = min(cands, key=lambda b: b["time_ms_median"])
        ratio = best["time_ms_median"] / t_rom
        print(f"{name} {r['method']}: err {err:.3e} in {t_rom:.2f} ms e2e "
              f"(ic {r['ic_ms_median']:.2f} + roll {r['rollout_ms_median']:.2f} "
              f"+ dec {r['decode_ms_median']:.2f}) vs tol-Newton "
              f"ntol={best['newton_tol']:.0e} (err "
              f"{best['err_traj_rel_mean']:.3e}) {best['time_ms_median']:.1f} ms "
              f"-> ROM is {ratio:.2f}x {'FASTER' if ratio > 1 else 'slower'} "
              f"(stops {r.get('stop_reasons')})")


def main(paths):
    for p in sorted(paths):
        d = json.load(open(p))
        name = p.split("/")[-1].replace(".json", "")
        (poisson if d["config"]["pde"] == "poisson2d" else burgers)(d, name)


if __name__ == "__main__":
    main(sys.argv[1:])
