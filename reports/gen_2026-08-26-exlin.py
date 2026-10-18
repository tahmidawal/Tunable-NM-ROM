"""Tables for reports/2026-08-26-exact-linear-terms-and-gradient-eq.md.

Usage:
  python reports/gen_2026-08-26-exlin.py <worktree-exp-dir>

where <worktree-exp-dir> is .../worktrees/2026-08-26-eq-learned/experiments/
separable-decoder.  Reads the run JSONs of that branch (and the 2026-08-25
ladder JSONs from the eq-fidelity-ladder worktree for the old-vs-new ladder
comparison) and prints markdown.  Every number comes from the JSONs; nothing
is typed by hand.
"""
import glob
import json
import os
import sys

import numpy as np

LAD_OLD = ("/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/"
           "2026-08-25-eq-fidelity-ladder/experiments/separable-decoder/runs")


def e(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1e}"


def e3(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3e}"


def c(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def mean(rs, k):
    v = [r[k] for r in rs if k in r]
    return float(np.mean(v)) if v else float("nan")


def recs(d, kind, name):
    return [r for r in d["records"] if r["kind"] == kind and r["eq"] == name]


def roll(d, name):
    v = [r["err_mean"] for r in d["rollout"] if r["eq"] == name]
    return float(np.mean(v)) if v else float("nan")


def load(p):
    with open(p) as f:
        return json.load(f)


def main(root):
    R = lambda *a: os.path.join(root, *a)

    print("### T-X1. Stage 1 — the round-4/5 speed protocol, incumbent residual vs exact-linear residual\n")
    print("Same checkpoint, same protocol, same GPU type per row pair; paired = median of AB/BA "
          "interleaved end-to-end runs at matched accuracy.  `base` = the un-shortcut ROM arm's "
          "mean rollout error over 8 test trajectories.\n")
    print("| N | residual | job | GPU | base rollout err | champion err | paired ROM ms | paired FOM ms | ratio |")
    print("|---|---|---|---|---|---|---|---|---|")
    for N, po, pn in ((256, R("runs/dn256b/out/sep_speed_r5_dense_mid_N256_dense.json"),
                       R("runs/xs256dm/out/sep_burgers_exlin_dense_mid_N256_dense_exlin.json")),
                      (1024, R("runs/dn1024/out/sep_speed_r5_dense_mid_N1024_dense.json"),
                       R("runs/xs1024dm/out/sep_burgers_exlin_dense_mid_N1024_dense_exlin.json"))):
        for tag, p in (("incumbent", po), ("exact-linear", pn)):
            d = load(p)
            ma = d["matched_accuracy"]
            pa = ma.get("paired") or {}
            base = [v for v in d["variants"] if v["name"] == "base"][0]
            print(f"| {N} | {tag} | {d['config']['slurm_job']} | {d['config']['gpu']} | "
                  f"{e3(base['err_traj_rel_mean'])} | {e3(ma['rom_err'])} | "
                  f"{pa.get('rom_ms', float('nan')):.1f} | {pa.get('base_ms', float('nan')):.1f} | "
                  f"{pa.get('base_ms', float('nan'))/pa.get('rom_ms', float('nan')):.2f}x |")

    print("\n### T-X2. Stage 1 — the ladder, incumbent vs exact-linear + advection-only nodes (same checkpoints, same states)\n")
    print("Solver-path = mean over every LM iterate of 4 test-trajectory rollouts; oracle = mean over "
          "the full-grid LS codes of the truth states.  `lin` is the linear share of (b) — with the "
          "exact-linear residual it is zero by construction (~1e-13).\n")
    print("| ckpt | N | K | set | residual | (b) path | of which lin | (c1) cos path | (c3) cos path | (b) oracle | (c1) cos oracle | rollout err |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    lad_pairs = [
        ("r3a", 256, 16, os.path.join(LAD_OLD, "lad256k16/out/sep_eq_ladder_N256_K16_r3a.json"),
         R("runs/xl256r3a/out/sep_eq_ladder_N256_K16_r3a_exlin.json")),
        ("dense_mid", 256, 16, os.path.join(LAD_OLD, "lad256dm/out/sep_eq_ladder_N256_K16_dense_mid.json"),
         R("runs/xl256dm/out/sep_eq_ladder_N256_K16_dense_mid_exlin.json")),
        ("r3d", 1024, 16, os.path.join(LAD_OLD, "lad1024k16/out/sep_eq_ladder_N1024_K16_r3d.json"),
         R("runs/xl1024k16/out/sep_eq_ladder_N1024_K16_r3d_exlin.json")),
        ("r4a6", 1024, 32, os.path.join(LAD_OLD, "lad1024k32/out/sep_eq_ladder_N1024_K32_r4a6.json"),
         R("runs/xl1024k32/out/sep_eq_ladder_N1024_K32_r4a6_exlin.json")),
    ]
    for ck, N, K, po, pn in lad_pairs:
        for tag, p in (("incumbent", po), ("exlin+adv", pn)):
            if not os.path.exists(p):
                print(f"| {ck} | {N} | {K} | — | {tag} | (missing: {os.path.basename(p)}) |")
                continue
            d = load(p)
            for name in d["eq"]:
                sp = recs(d, "solver", name)
                oc = recs(d, "oracle", name)
                print(f"| {ck} | {N} | {K} | {name} | {tag} | "
                      f"{e(mean(sp,'b_resid'))} | {e(mean(sp,'b_lin'))} | {c(mean(sp,'c1_cos'))} | "
                      f"{c(mean(sp,'c3_cos'))} | {e(mean(oc,'b_resid'))} | {c(mean(oc,'c1_cos'))} | "
                      f"{e3(roll(d, name))} |")

    print("\n### T-X3. Stage 2 — four quadratures at fixed m, exact-linear residual everywhere\n")
    print("inc = incumbent two-block fit at training codes; adv = advection-only at training codes; "
          "path = advection rows at off-manifold LM iterates of TRAINING rollouts; grad = path + "
          "gradient-teacher rows (full-grid Jacobian frozen).  held-out = fit-side iterates excluded "
          "from every row build; test-path = LM iterates of the 4 test rollouts.\n")
    print("| arm | N | K | M/m | set | held-out (b) | held-out (c1) | (c1) cos | (c3) cos | test-path (b) | oracle (b) | rollout err |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(glob.glob(R("runs/gf*/out/sep_eq_gradfit_*.json"))):
        d = load(p)
        cf = d["config"]
        arm = p.split("/runs/")[1].split("/")[0]
        for name in d["eq"]:
            hh = recs(d, "heldout", name)
            sp = recs(d, "solver", name)
            oc = recs(d, "oracle", name)
            print(f"| {arm} | {cf['N']} | {cf['k']} | {cf['eq_M']}/{4*cf['eq_M']} | {name} | "
                  f"{e(mean(hh,'b_resid'))} | {e(mean(hh,'c1_grad'))} | {c(mean(hh,'c1_cos'))} | "
                  f"{c(mean(hh,'c3_cos'))} | {e(mean(sp,'b_resid'))} | {e(mean(oc,'b_resid'))} | "
                  f"{e3(roll(d, name))} |")

    print("\n### T-X4. Stage 3 — learned continuous nodes vs the convex baselines\n")
    print("node = m continuous positions optimized against the frozen full-grid teacher "
          "(inner NNLS re-solving the weights; loss = ladder rungs b + c1).  Success bar: beat "
          "`grad` on held-out (c1)/(c3) AND on test rollout error at the same m.\n")
    print("| arm | N | K | M/m | set | held-out (b) | held-out (c1) | (c1) cos | (c3) cos | test-path (b) | rollout err | opt loss init→final | mean node move |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(glob.glob(R("runs/nf*/out/sep_eq_nodefit_*.json"))):
        d = load(p)
        cf = d["config"]
        arm = p.split("/runs/")[1].split("/")[0]
        no = d.get("node_opt", {})
        for name in d["eq"]:
            hh = recs(d, "heldout", name)
            sp = recs(d, "solver", name)
            opt = (f"{no.get('loss_init', float('nan')):.2e}→"
                   f"{no.get('loss_final_refit', float('nan')):.2e}"
                   if name == "node" else "—")
            mv = e(no.get("mean_move")) if name == "node" else "—"
            print(f"| {arm} | {cf['N']} | {cf['k']} | {cf['eq_M']}/{4*cf['eq_M']} | {name} | "
                  f"{e(mean(hh,'b_resid'))} | {e(mean(hh,'c1_grad'))} | {c(mean(hh,'c1_cos'))} | "
                  f"{c(mean(hh,'c3_cos'))} | {e(mean(sp,'b_resid'))} | {e3(roll(d, name))} | "
                  f"{opt} | {mv} |")

    print("\n### T-X5. Provenance and gates\n")
    print("| run | N | GPU | job | backend | gate 0 | gate F | gate L | gate A | gate C |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    pats = ["runs/xl*/out/sep_eq_ladder_*.json", "runs/xs*/out/sep_burgers_exlin_*.json",
            "runs/gf*/out/sep_eq_gradfit_*.json", "runs/nf*/out/sep_eq_nodefit_*.json"]
    for pat in pats:
        for p in sorted(glob.glob(R(pat))):
            d = load(p)
            cf = d["config"]
            arm = p.split("/runs/")[1].split("/")[0]
            g0 = max((v.get("gate0") or 0) for v in d["eq"].values())
            gF = [v.get("gateF") for v in d["eq"].values() if v.get("gateF") is not None]
            gF = max(gF) if gF else (d.get("gates") or {}).get("gateF")
            gL = [v.get("gateL") for v in d["eq"].values() if v.get("gateL") is not None]
            gA = [v.get("gateA") for v in d["eq"].values() if v.get("gateA") is not None]
            gC = (d.get("gates") or {}).get("gateC")
            print(f"| {arm} | {cf['N']} | {cf['gpu']} | {cf['slurm_job']} | {cf['backend']} | "
                  f"{e(g0)} | {e(gF)} | {e(max(gL) if gL else None)} | "
                  f"{e(max(gA) if gA else None)} | {e(gC)} |")
    print("\nSources: every `runs/*/out/*.json` matched above, on branch exp/2026-08-26-eq-learned; "
          "old-ladder JSONs on branch exp/2026-08-25-eq-fidelity-ladder.")


if __name__ == "__main__":
    main(sys.argv[1])
