"""Tables for reports/2026-08-27-nodes-at-mM-transfer.md.

Usage:  python gen_2026-08-27-nodes-mM-transfer.py

Reads the run JSONs from the two experiment worktrees (nodes-mm cluster runs
+ the N=64 pilot arm-n runs) and prints the report's tables T-N1 and T-N2.
Every number comes from the JSONs; nothing is typed by hand.
"""
import glob
import json
import os

WTS = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees"
CDMM = os.path.join(WTS, "2026-08-27-nodes-mm",
                    "experiments", "separable-decoder", "runs", "cdmm")
PILOT = os.path.join(WTS, "2026-08-26-codesign",
                     "experiments", "separable-decoder", "runs")


def e(v):
    return "—" if v is None else f"{v:.3e}"


def c(v):
    return "—" if v is None else f"{v:.4f}"


def load():
    cells = {}
    pats = (glob.glob(os.path.join(CDMM, "*", "out", "sep_codesign_*.json"))
            + glob.glob(os.path.join(PILOT, "cd_n_m*", "out",
                                     "sep_codesign_*.json")))
    for p in sorted(pats):
        d = json.load(open(p))
        assert d.get("complete"), f"incomplete run: {p}"
        N = d["config"]["N"]
        for vname, v in d["variants"].items():
            cells[(N, v["m"], vname)] = dict(
                b=v["heldout"].get("b"), c1=v["heldout"].get("c1"),
                c1_cos=v["heldout"].get("c1_cos"),
                recon=v.get("held_recon_rel"),
                roll=v.get("rollout_err_mean"),
                trip=d.get("tripwire_fired"))
    return cells


def main():
    cells = load()
    Ns = sorted({k[0] for k in cells})

    print("**T-N1 — per-budget: learned nodes vs the NNLS grid-node baseline"
          " (same m).**\n")
    print("| N | m | variant | held (b) | held (c1) | (c1) cos | held recon "
          "| rollout err | vs base | trip |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for N in Ns:
        for m in sorted({k[1] for k in cells if k[0] == N}):
            base = cells[(N, m, "base")]
            for vname in ("base", "cot"):
                r = cells[(N, m, vname)]
                d = ("—" if vname == "base" else
                     f"{100*(r['roll']-base['roll'])/base['roll']:+.1f}%")
                print(f"| {N} | {m} | {vname} | {e(r['b'])} | {e(r['c1'])} | "
                      f"{c(r['c1_cos'])} | {e(r['recon'])} | {e(r['roll'])} "
                      f"| {d} | {'Y' if r['trip'] else 'n'} |")

    print("\n**T-N2 — the matched-accuracy question: do learned m=M nodes "
          "reach the NNLS m=4M baseline?**\n")
    print("| N | learned nodes, m=M rollout | NNLS baseline, m=4M rollout | "
          "learned m=M vs NNLS m=4M |")
    print("|---|---|---|---|")
    for N in Ns:
        a = cells[(N, 64, "cot")]["roll"]
        b = cells[(N, 256, "base")]["roll"]
        print(f"| {N} | {e(a)} | {e(b)} | {100*(a-b)/b:+.1f}% |")


if __name__ == "__main__":
    main()
