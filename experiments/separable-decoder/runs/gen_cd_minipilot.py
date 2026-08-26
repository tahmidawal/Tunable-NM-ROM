"""Markdown table for the N=64 co-design mini-pilot.

Usage:  python gen_cd_minipilot.py [runs-dir]

Reads every runs/cd_*/out/sep_codesign_*.json and prints one row per
(arm, m, variant).  Every number comes from the JSONs; nothing is typed by
hand.  'base' rows are the frozen-decoder + NNLS-nodes certification that
every run performs with the same instrument; they must agree across arms at
equal m (same seed, same fit) -- disagreement means a wiring change leaked.
"""
import glob
import json
import os
import sys


def e(v):
    return "—" if v is None else f"{v:.3e}"


def c(v):
    return "—" if v is None else f"{v:.4f}"


def main(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "cd_*", "out",
                                           "sep_codesign_*.json"))):
        d = json.load(open(p))
        cf = d["config"]
        arm = p.split(os.sep)[-3].replace("cd_", "")
        for vname, v in d.get("variants", {}).items():
            ho = v.get("heldout", {})
            rows.append(dict(
                arm=arm, variant=vname, m=v.get("m"),
                steps=cf.get("steps"), lr=cf.get("lr"),
                flags=f"H{cf.get('train_h')}N{cf.get('train_nodes')}"
                      f" s{cf.get('samp_rel'):g} j{cf.get('jac_rel'):g}"
                      f" g{cf.get('sob_rel'):g}",
                b=ho.get("b"), c1=ho.get("c1"), c1_cos=ho.get("c1_cos"),
                c3_cos=ho.get("c3_cos"),
                recon=v.get("held_recon_rel"),
                roll=v.get("rollout_err_mean"),
                trip=d.get("tripwire_fired"),
                move=(d.get("node_stats") or {}).get("mean_move"),
                complete=d.get("complete")))
    print("| arm | variant | m | flags | held (b) | held (c1) | (c1) cos | "
          "(c3) cos | held recon | rollout err | node move | trip | done |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r_: (r_["m"] or 0, r_["arm"],
                                          r_["variant"])):
        print(f"| {r['arm']} | {r['variant']} | {r['m']} | {r['flags']} | "
              f"{e(r['b'])} | {e(r['c1'])} | {c(r['c1_cos'])} | "
              f"{c(r['c3_cos'])} | {e(r['recon'])} | {e(r['roll'])} | "
              f"{e(r['move'])} | {'Y' if r['trip'] else 'n'} | "
              f"{'Y' if r['complete'] else 'PARTIAL'} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.dirname(os.path.abspath(__file__)))
