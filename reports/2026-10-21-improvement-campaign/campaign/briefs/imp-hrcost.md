# Lane imp-hrcost — E6 (ideation W3): a cheaper nonlinear term (hyper-reduction cost)

Worktree `worktrees/2026-10-21-imp-hrcost`, branch `exp/2026-10-21-imp-hrcost`, lane dir
`experiments/imp-hrcost/`, namespace `/cluster/tufts/paralab/tawal01/imp-hrcost/`.
**GPU-hour cap: 3. Lane concurrent-job cap: 1.** Wave 1. Read `COMMON-BRIEF.md` first (binding).

## Problem it answers
Problem 2 (POD + ECSW dominates). Ideation F2 + F3 (§4) and §5 row W3: reduce the quadrature size $m$ and the
per-point cost of the nonlinear term.

## Arms (r = 384, full Q2 contract, E0 errors, paired timing against E5's precontracted arm)
- F2: **bank-tested Galerkin with classical rules** (test functions = bank, not sines); find the smallest
  rule size $m^*$ that passes Q2.
- F3: **precontracted sine-tested Galerkin with fitted continuous ECSW-style weights** on our off-mesh points,
  sparsity/compression factor κ.
- both together (F2 + F3).
- optional: sketching control (F3) if budget allows.

## Stop rules (verbatim; pre-register them)
- Stop F2 if $m^* > 16384$, accuracy is worse, or it is not faster.
- Stop F3 if no κ ≤ 16 passes Q2 at equal worst error with a ≥ 1.5× gain.

## Dependency on E5
You consume lane imp-faststep's precontracted Galerkin arm. Do the design, its audit, and all code that does
not depend on it now. Poll `worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/HANDOFF-E6.md`
(read-only) every ≤ 30 min with an overall deadline of 48 h; until it exists set your heartbeat
`waiting_on` to `"handoff: imp-faststep HANDOFF-E6.md"`. Copy (do not import across worktrees) the arm's code
into your lane with a provenance note (source commit + sha256). If E5's precontracted arm fails its own stop
rule, still run F2/F3 against it as the reference arm and say so.

## Build on
jcp-hr-baselines (ECSW / NNLS `nnls.py`, `hrcore.py`), quadrature-study / jcp-mechanism (classical rules,
Q2 machinery), jcp-qualify (`qcert.py`), jcp-robust (rule bank-specificity finding), jcp-references.
