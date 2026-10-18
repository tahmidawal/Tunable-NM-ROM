"""Compact EQ-fidelity-ladder tables for reports/2026-08-25-eq-fidelity-ladder.md.

Usage:
  python reports/gen_2026-08-25-eq-ladder.py <sep_eq_ladder_*.json ...>

Prints markdown.  Every number comes from the JSONs (written by
experiments/separable-decoder/sep_eq_ladder.py on branch
exp/2026-08-25-eq-fidelity-ladder); nothing is typed by hand.  The full
per-time-bucket tables are in experiments/separable-decoder/EQ-LADDER.md on that
branch (runs/gen_eq_ladder.py).
"""
import json
import sys
import numpy as np


def mean(rs, k):
    v = [r[k] for r in rs if k in r]
    return float(np.mean(v)) if v else float("nan")


def worst(rs, k):
    v = [r[k] for r in rs if k in r]
    if not v:
        return float("nan")
    return float(np.min(v)) if "cos" in k else float(np.max(v))


def e(v):
    return f"{v:.1e}"


def c(v):
    return f"{v:.2f}"


def main(paths):
    ds = [json.load(open(p)) for p in paths]
    print("### T-L1. Where the quadrature error sits: residual, split into the part that could be exact and the part that cannot\n")
    print("Solver-path states (every LM iterate of a real ROM rollout, 4 fresh test trajectories, all 50 steps). "
          "Relative to the full-grid residual norm. `lin` = mass + previous state + Laplacian terms (all `Φᵀu`, exactly computable as `(ΦᵀG)h(z)`); "
          "`adv` = `ΦᵀN(u)` (sign-upwind, needs sampling).\n")
    print("| decoder | N | K | EQ set (M/m) | NNLS rel fit | (b) ‖R_s−R_f‖/‖R_f‖ | of which lin | of which adv | (b) worst |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in ds:
        cf = d["config"]
        for name, eq in d["eq"].items():
            rs = [r for r in d["records"] if r["kind"] == "solver" and r["eq"] == name]
            M = eq["m"] // cf["eq_m_factor"]
            print(f"| {cf['ckpt'].replace('.pkl','')} | {cf['N']} | {cf['k']} | {name} ({M}/{eq['m']}) | "
                  f"{e(eq['rel_fit'])} | {e(mean(rs,'b_resid'))} | {e(mean(rs,'b_lin'))} | {e(mean(rs,'b_adv'))} | {e(worst(rs,'b_resid'))} |")

    print("\n### T-L2. The ladder: residual → gradient → Hessian → step (solver-path states, means; cos = direction agreement)\n")
    print("| decoder | N | K | EQ set | (b) resid | (c1) grad | (c1) cos | (c1) abs | (c2) Hess | (c3) LM step | (c3) cos |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in ds:
        cf = d["config"]
        for name in d["eq"]:
            rs = [r for r in d["records"] if r["kind"] == "solver" and r["eq"] == name]
            print(f"| {cf['ckpt'].replace('.pkl','')} | {cf['N']} | {cf['k']} | {name} | "
                  f"{e(mean(rs,'b_resid'))} | {e(mean(rs,'c1_grad'))} | {c(mean(rs,'c1_cos'))} | {e(mean(rs,'c1_abs'))} | "
                  f"{e(mean(rs,'c2_hess'))} | {e(mean(rs,'c3_step'))} | {c(mean(rs,'c3_cos'))} |")

    print("\n### T-L3. The same ladder AT THE SOLUTION (oracle states: full-grid LS code of the truth state, t ∈ {0,1,2,3,5,10,25,50})\n")
    print("Here `R_f` is the true weak residual of the best on-manifold state; a sampled residual/gradient much larger than it means the sampled objective's minimum is somewhere else.\n")
    print("| decoder | N | K | EQ set | (b) resid | (b) at t=0 | (b) at t≥5 | (c1) grad | (c1) cos | (c1) abs | (c2) Hess | (c3) step | (c3) cos |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in ds:
        cf = d["config"]
        for name in d["eq"]:
            rs = [r for r in d["records"] if r["kind"] == "oracle" and r["eq"] == name]
            r0 = [r for r in rs if r["t"] == 0]
            r5 = [r for r in rs if r["t"] >= 5]
            print(f"| {cf['ckpt'].replace('.pkl','')} | {cf['N']} | {cf['k']} | {name} | "
                  f"{e(mean(rs,'b_resid'))} | {e(mean(r0,'b_resid'))} | {e(mean(r5,'b_resid'))} | "
                  f"{e(mean(rs,'c1_grad'))} | {c(mean(rs,'c1_cos'))} | {e(mean(rs,'c1_abs'))} | "
                  f"{e(mean(rs,'c2_hess'))} | {e(mean(rs,'c3_step'))} | {c(mean(rs,'c3_cos'))} |")

    print("\n### T-L4. Rung (a): is the L2 error on the m sample points representative of the full-grid error? (oracle states)\n")
    print("| decoder | N | K | EQ set | recon err, full grid | recon err on nodes (w-weighted RMS) | ratio |")
    print("|---|---|---|---|---|---|---|")
    for d in ds:
        cf = d["config"]
        for name in d["eq"]:
            rs = [r for r in d["records"] if r["kind"] == "oracle" and r["eq"] == name]
            f_, p_ = mean(rs, "a_recon_full"), mean(rs, "a_recon_pts")
            print(f"| {cf['ckpt'].replace('.pkl','')} | {cf['N']} | {cf['k']} | {name} | {e(f_)} | {e(p_)} | {c(p_/f_)} |")

    print("\n### T-L5. Provenance and gates\n")
    print("| decoder | N | GPU | job | backend | bank-vs-meshfree | gate 0 (ctrl / M256) | gate F | rollout err (ctrl / M256, mean of 4) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in ds:
        cf = d["config"]
        g0 = " / ".join(e(v["gate0"]) for v in d["eq"].values())
        gF = " / ".join("—" if v.get("gateF") is None else e(v["gateF"]) for v in d["eq"].values())
        ro = {}
        for r in d["rollout"]:
            ro.setdefault(r["eq"], []).append(r["err_mean"])
        ros = " / ".join(e(float(np.mean(v))) for v in ro.values())
        print(f"| {cf['ckpt'].replace('.pkl','')} | {cf['N']} | {cf['gpu']} | {cf['slurm_job']} | {cf['backend']} | "
              f"{e(d['gates']['eq_bank_vs_meshfree'])} | {g0} | {gF} | {ros} |")
    print("\nSources:\n")
    for p in paths:
        print(f"- `{p}`")


if __name__ == "__main__":
    main(sys.argv[1:])
