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

# 4. direct-solver ratio at N=512 -- SEMANTIC: the ratio and BOTH its operands must appear
#    together, aggregated the same way.  An earlier version checked only that "494x" occurred
#    somewhere, which is why it passed a sentence whose quoted operands did not divide to it.
grp = [r for r in P if r["N"] == 512 and r["fom_tau"] == 1e-10]
num = sum(r["t_fom_baseline_ms"] for r in grp) / len(grp)
den = sum(r["t_fom_direct_ms"] for r in grp) / len(grp)
ratio = num / den
chk("direct ratio N=512 value", infile(f"{ratio:.0f}x"), f"{ratio:.0f}x")
chk("direct ratio numerator quoted", infile(f"{num:.4g} ms"), f"{num:.4g}")
chk("direct ratio denominator quoted", infile(f"{den:.4g} ms"), f"{den:.4g}")
chk("direct ratio operands divide to the quoted value",
    abs(float(f"{num:.4g}") / float(f"{den:.4g}") - ratio) / ratio < 0.01)

# 4b. the commit reported must be the WRAPPER commit from the sbatch logs, and the bogus
#     git-discovered one must not be presented as the run commit.
import re as _re, glob as _g
wrap = set()
for lg in _g.glob("runs/*/logs/*.out"):
    m = _re.search(r"commit=([0-9a-f]{40})", open(lg, errors="ignore").read())
    if m:
        wrap.add(m.group(1))
chk("one wrapper commit across all logs", len(wrap) == 1, str(wrap))
wrap_commit = sorted(wrap)[0] if len(wrap) == 1 else None
if wrap:
    w = sorted(wrap)[0]
    chk("wrapper commit in README", infile(w[:12]))
    bogus = raw["wsp_cons"]["provenance"]["commit"]
    chk("bogus discovered commit is named as invalid, not as the run commit",
        (bogus[:12] not in R) or ("not an object in" in R))

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

# 6b. the ERODED-not-diluted claim: the absolute saving must FALL as tau tightens.
#     (An earlier draft asserted the opposite; this check exists so that cannot recur.)
nmax = max(r["N"] for r in P)
abs_by_tau = {}
for ft in sorted({r["fom_tau"] for r in P}, reverse=True):
    sub = [r for r in P if r["N"] == nmax and r["fom_tau"] == ft]
    b = max(sub, key=lambda r: r["iter_saving_frac"])
    abs_by_tau[ft] = b["iters_from_baseline"] - b["iters_from_rom"]
seq = [abs_by_tau[t] for t in sorted(abs_by_tau, reverse=True)]
chk("absolute saving falls as tau tightens", all(a > b for a, b in zip(seq, seq[1:])),
    f"{seq}")
chk("absolute savings quoted in README", infile(f"{seq[0]:.3g} iterations out of"))

# 4c. THE PROVENANCE DEFECT IS METADATA ONLY: prove it by content hash rather than by
#     trusting the (broken) git-discovery field.  The per-file sha256 recorded at RUN TIME
#     on the cluster must equal the files as they stand at the wrapper commit from the log.
#     Note the current working-tree files DIFFER -- they carry post-run audit fixes -- so
#     this must be checked against git history, not against the checkout.
import hashlib as _h, subprocess as _sp
_recs = raw["wsp_cons"]["provenance"]["source_sha256"]
if wrap_commit:
    for fn in ("wsf_poisson.py", "wsf_burgers.py", "wsf_util.py"):
        try:
            blob = _sp.check_output(
                ["git", "show", f"{wrap_commit}:experiments/rom-warmstart-fom/{fn}"],
                stderr=_sp.DEVNULL)
            chk(f"{fn} that RAN == wrapper commit",
                _h.sha256(blob).hexdigest()[:16] == _recs[fn])
        except Exception as e:
            chk(f"{fn} recoverable at wrapper commit", False, str(e))

# 4d. THE ENGINEERING FACTOR DEPENDS ON THE ASSUMED TOLERANCE, and the README must show
#     all three.  A version of this README headlined only the tightest (1e-10) factor,
#     understating the correction by ~34%; this check stops that recurring.
facs = {}
for ft in sorted({r["fom_tau"] for r in P}, reverse=True):
    vals = []
    for n in sorted({r["N"] for r in P}):
        grp = [r for r in P if r["N"] == n and r["fom_tau"] == ft
               and r.get("t_fom_baseline_native_ms")]
        if grp:
            tn = sum(r["t_fom_baseline_native_ms"] for r in grp) / len(grp)
            vals.append(grp[0]["t_fom_testbed_ms"] / tn)
    if vals:
        facs[ft] = (min(vals), max(vals), sum(vals) / len(vals))
chk("factor grows as the assumed tolerance loosens",
    facs[1e-6][2] > facs[1e-8][2] > facs[1e-10][2],
    str({k: round(v[2], 3) for k, v in facs.items()}))
for ft, (lo, hi, mn) in facs.items():
    chk(f"factor range at tau={ft:g} quoted in README", infile(f"{lo:.2f}-{hi:.2f}x"),
        f"{lo:.2f}-{hi:.2f}x")
chk("the consumer-tolerance (1e-6) factor appears, not only the tightest",
    infile(f"{facs[1e-6][2]:.2f}x"))

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
