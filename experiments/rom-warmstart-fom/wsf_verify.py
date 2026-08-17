"""Independent re-derivation of every headline claim, straight from the per-cell JSONs.

This is deliberately NOT built on wsf_summarize/wsf_facts: those produce the tables and
the README, so checking them with their own machinery proves nothing.  The logic here is
written separately, reads `runs/*/out/*.json` directly, and then checks the result against
the strings actually present in the RENDERED `README.md` -- so it catches derivation
errors, template/substitution errors and column misalignment alike.

It is a complement to the Codex results audit, not a substitute: it verifies the claims I
knew to check, which is exactly the blind spot an outside auditor covers.

Usage: python wsf_verify.py        # exits non-zero if any check fails
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

R = open("README.md").read()
def infile(s):  # is this exact string present in the rendered README?
    return s in R
raw = {}
for p in glob.glob("runs/*/out/*.json"):
    d = json.load(open(p))
    assert d.get("complete") is True, p
    raw[p.split("/")[1]] = d
P = raw["wsp_cons"]["rows"]; B = raw["wsb_cons"]["rows"]
ok = fail = 0
def chk(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1
    else: fail += 1; print(f"  MISMATCH: {name} {detail}")

# 1. no crossover
best = max(r["speedup_vs_fom"] for r in P)
chk("no Poisson crossover", best <= 1.0, f"max speedup {best:.4f}")
chk("best value in README", infile(f"{best:.3f}x"), f"{best:.3f}x")
bl = [r for r in P if r["speedup_vs_fom"] == best][0]
chk("best location", infile(f"N={bl['N']}, rom_tau={bl['rom_tau']:g}, tau_FOM={bl['fom_tau']:g}"))

# 2. burgers win counts
n = len(B)
chk("burgers newton wins", infile(f"{sum(r['iters_from_rom']<r['iters_from_baseline'] for r in B)} of {n}"))
chk("burgers lin wins 0", sum(r["lin_iters_from_rom"] < r["lin_iters_from_baseline"] for r in B) == 0)
chk("burgers time wins 0", sum(r["speedup_vs_fom"] > 1.0 for r in B) == 0)
chk("extrap wins all", sum(r["iters_from_extrap"] < r["iters_from_baseline"] for r in B) == n)

# 3. negative-saving count
neg = sum(r.get("iter_saving_frac", 0) < 0 for r in P)
chk("negative savings count", infile(f"{neg} of {len(P)} measured configurations"), f"{neg}/{len(P)}")

# 4. direct-solver ratio at N=512
r512 = [r for r in P if r["N"] == 512 and r["fom_tau"] == 1e-10][0]
ratio = r512["t_fom_baseline_ms"] / r512["t_fom_direct_ms"]
chk("direct ratio N=512", infile(f"{ratio:.0f}x"), f"{ratio:.0f}x")

# 5. burgers over-convergence factors
for r in B:
    if r["fom_tau"] != 1e-10: continue
    f = r["t_fom_testbed_ms"] / r["t_fom_baseline_ms"]
    chk(f"burgers ovconv N={r['N']}", abs(f - r["overconvergence_factor"]) < 1e-9)
    chk(f"burgers factor in README N={r['N']}", infile(f"{f:.2f}x"), f"{f:.2f}x")

# 6. health
chk("zero breakdowns", sum(r["bicgstab_breakdowns"] for r in B) == 0)
chk("zero newton flags", sum(r["newton_flags_nonzero"] for r in B) == 0)
for r in B:
    chk(f"burgers residual<=tau N={r['N']} tau={r['fom_tau']:g}",
        max(r["max_rel_newton_residual"].values()) <= r["fom_tau"])
for r in P:
    chk(f"poisson residual<=tau N={r['N']}", r["final_rel_residual"] <= r["fom_tau"])

# 7. every row is A100 and gpu backend
for k, d in raw.items():
    chk(f"{k} gpu", d["provenance"]["jax_backend"] == "gpu")
    chk(f"{k} a100", "A100" in d["provenance"]["gpu_kind"])
# 8. consolidated rows all from one job each
chk("poisson one job", len({r["slurm_job_id"] for r in P}) == 1)
chk("burgers one job", len({r["slurm_job_id"] for r in B}) == 1)
print(f"\nINDEPENDENT VERIFICATION: {ok} checks passed, {fail} failed")
if fail:
    sys.exit(1)
