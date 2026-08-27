"""Markdown table for the nodes-at-m=M transfer experiment (N=256/1024).

Usage:  python gen_cdmm.py [runs/cdmm-dir]

Reads every runs/cdmm/*/out/sep_codesign_*.json and prints one row per
(N, m, variant), plus the N=64 pilot rows (cd_n_m*/ in the pilot layout) if a
second argument points at the pilot runs dir.  Every number comes from the
JSONs; nothing is typed by hand.  'base' is the frozen-decoder + NNLS-grid-
nodes certification each job performs with the same instrument; 'cot' is the
learned-nodes variant (arm n: decoder frozen, only node positions trained).
"""
import glob
import json
import os
import sys


def e(v):
    return "—" if v is None else f"{v:.3e}"


def c(v):
    return "—" if v is None else f"{v:.4f}"


def rows_from(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p))
        cf = d["config"]
        base_roll = (d.get("variants", {}).get("base") or {}).get(
            "rollout_err_mean")
        for vname, v in d.get("variants", {}).items():
            ho = v.get("heldout", {})
            roll = v.get("rollout_err_mean")
            dlt = (None if vname == "base" or not base_roll or roll is None
                   else 100.0 * (roll - base_roll) / base_roll)
            rows.append(dict(
                N=cf.get("N"), m=v.get("m"), variant=vname,
                b=ho.get("b"), c1=ho.get("c1"), c1_cos=ho.get("c1_cos"),
                recon=v.get("held_recon_rel"), roll=roll, delta=dlt,
                trip=d.get("tripwire_fired"), complete=d.get("complete")))
    return rows


def main(cdmm_root, pilot_root=None):
    rows = rows_from(os.path.join(cdmm_root, "*", "out",
                                  "sep_codesign_*.json"))
    if pilot_root:
        rows += rows_from(os.path.join(pilot_root, "cd_n_m*", "out",
                                       "sep_codesign_*.json"))
    print("| N | m | variant | held (b) | held (c1) | (c1) cos | held recon "
          "| rollout err | vs base | trip | done |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r_: (r_["N"] or 0, r_["m"] or 0,
                                          r_["variant"])):
        d = "—" if r["delta"] is None else f"{r['delta']:+.1f}%"
        print(f"| {r['N']} | {r['m']} | {r['variant']} | {e(r['b'])} | "
              f"{e(r['c1'])} | {c(r['c1_cos'])} | {e(r['recon'])} | "
              f"{e(r['roll'])} | {d} | {'Y' if r['trip'] else 'n'} | "
              f"{'Y' if r['complete'] else 'PARTIAL'} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cdmm",
         sys.argv[2] if len(sys.argv) > 2 else None)
