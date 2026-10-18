"""Tables for reports/2026-08-26-codesign-minipilot.md.

Usage:
  python reports/gen_2026-08-26-codesign-minipilot.py

Reads every cd_*/out/sep_codesign_*.json on the exp/2026-08-26-codesign
worktree and prints markdown.  Every number comes from the JSONs; nothing is
typed by hand.
"""
import glob
import json
import os

import numpy as np

RUNS = ("/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/"
        "2026-08-26-codesign/experiments/separable-decoder/runs")


def e(v):
    return "—" if v is None else f"{v:.3e}"


def c(v):
    return "—" if v is None else f"{v:.4f}"


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS, "cd_*", "out",
                                           "sep_codesign_*.json"))):
        d = json.load(open(p))
        cf = d["config"]
        arm = p.split(os.sep)[-3].replace("cd_", "")
        if arm == "smoke":
            continue
        for vname, v in d.get("variants", {}).items():
            ho = v.get("heldout", {})
            rows.append(dict(
                arm=arm, variant=vname, m=v.get("m"),
                rec_w=cf.get("rec_w"), train_h=cf.get("train_h"),
                jac=cf.get("jac_rel"), sob=cf.get("sob_rel"),
                b=ho.get("b"), c1=ho.get("c1"), c1_cos=ho.get("c1_cos"),
                c3_cos=ho.get("c3_cos"), recon=v.get("held_recon_rel"),
                roll=v.get("rollout_err_mean"),
                trip=d.get("tripwire_fired"), gpu=cf.get("gpu"),
                backend=cf.get("backend"), steps=cf.get("steps"),
                complete=d.get("complete")))

    base = {r["m"]: r for r in rows if r["variant"] == "base"}
    print("### T-C1. All arms, held-out rungs and test rollouts\n")
    print("N=64, K=16, R=64, M=64; 2000 steps, LR 3e-5; 4 fresh test "
          "trajectories; base = frozen decoder + advection-only NNLS nodes "
          "(identical across arms at equal m by construction).\n")
    print("| m | arm | trained | REC_W | held (b) | held (c1) | (c1) cos | "
          "(c3) cos | held recon | rollout err | vs base | trip |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    seen_base = set()
    for r in sorted(rows, key=lambda r_: (r_["m"], r_["variant"] != "base",
                                          r_["arm"])):
        if r["variant"] == "base":
            if r["m"] in seen_base:
                continue
            seen_base.add(r["m"])
            label, tr = "base (NNLS)", "—"
        else:
            label = r["arm"]
            tr = ("nodes" if not r["train_h"] else
                  "h+nodes" + ("+jac" if r["jac"] else "")
                  + ("+sob" if r["sob"] else ""))
        d_roll = (r["roll"] / base[r["m"]]["roll"] - 1.0) * 100
        is_b = r["variant"] == "base"
        print(f"| {r['m']} | {label} | {tr} | "
              f"{'—' if is_b else f'{r['rec_w']:g}'} | {e(r['b'])} | "
              f"{e(r['c1'])} | {c(r['c1_cos'])} | {c(r['c3_cos'])} | "
              f"{e(r['recon'])} | {e(r['roll'])} | "
              f"{'—' if is_b else f'{d_roll:+.1f}%'} | "
              f"{'—' if is_b else ('Y' if r['trip'] else 'n')} |")

    print("\n### T-C2. Provenance\n")
    print("| arm | backend | gpu | steps | complete |")
    print("|---|---|---|---|---|")
    done = set()
    for r in rows:
        if r["variant"] != "cot" or r["arm"] in done:
            continue
        done.add(r["arm"])
        print(f"| {r['arm']} | {r['backend']} | {r['gpu']} | {r['steps']} | "
              f"{'Y' if r['complete'] else 'NO'} |")
    print("\nSources: runs/cd_*/out/sep_codesign_*.json on branch "
          "exp/2026-08-26-codesign; base rows deduplicated (bit-identical "
          "across arms at equal m).")
    # consistency check, not printed as data: bases must agree at equal m
    for m_ in sorted({r["m"] for r in rows}):
        bs = [r["roll"] for r in rows if r["variant"] == "base"
              and r["m"] == m_]
        assert np.allclose(bs, bs[0]), f"base rows disagree at m={m_}"


if __name__ == "__main__":
    main()
