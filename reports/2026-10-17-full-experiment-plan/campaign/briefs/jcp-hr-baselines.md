# Lane brief: jcp-hr-baselines (C3 — POD-based hyper-reduced ROM baselines)

Read `briefs/COMMON-BRIEF.md` first; it binds you. This brief adds your lane specifics.

## Where
- Worktree `worktrees/2026-10-06-jcp-hr-baselines` (exists; scaffolding only: vendor + sync script +
  README, commit `56a1f9f0f`, forked at `21c175a1b`). Branch `exp/2026-10-06-jcp-hr-baselines`.
  Lane dir `experiments/jcp-hr-baselines`. Namespace `/cluster/tufts/paralab/tawal01/jcp-hrb/`.
- Concurrent-job cap: **2**. GPU-hour cap: C3-minimal is part of priority 2 (60–110 GPU-h total for the
  priority, shared): **your cap is 30 GPU-h** for C3-minimal including contingency.

## Phasing (important)
- **Phase I (now, in parallel with priority 1):** DESIGN.md → Codex design audit → code → Codex code
  audit → real-size smoke (local `jaxrun`, at most one, or a single cluster smoke job ≤ 1 GPU-h) where
  every control fires. **No production jobs.** When Phase I is done, write your heartbeat with step
  "phase I done; awaiting coordinator release" and **hand back** (final message) with a Phase-I report.
  The coordinator will resume you (SendMessage) when priority 1 is reported, to run Phase II.
- **Phase II (after release):** job audits → production → results audit → report → report audit → hand back.

## Items owned (plan V2 Group C3, V3-5, V4-5)
**C3-minimal (priority 2):** POD-LSPG at rank r and smaller ranks, hyper-reduced with **ECSW** and
**Q-DEIM**, each with its native solver, then a **common-solver ablation** (same Gauss–Newton/LM solver as
our method). Both 2D and 3D viscous Burgers, on the same dev/val cohorts and the same E0 ST scoring as our
method. **C3-full (priority 4, later release):** GNAT, POD + exact tensor (quadratic nonlinearity
precomputed), best-practice EQ on our bank, continuum ECSW, CROM-style sampler — design them now only
as an appendix; implement in phase I only if cheap.
- **Fairness:** equal tuning budgets on validation (counted in GPU-hours); offline cost reported
  (snapshot generation, SVD, sampling/weights); same time stepping options as our method where
  applicable (BE Δt₀ and LSPG-CN-R Δt₀ in 2D; the 3D deployed schemes); report each baseline's actual
  residual and sampling formulation; snapshots from the **training** distribution (not dev/val/test).
  Reuse the training-trajectory generator the bank was trained on, or its stored data; say which.
- **Controls:** an independent residual/Jacobian finite-difference check for each baseline (must pass);
  POD at full rank with all mesh points must reproduce the FOM to solver tolerance (positive gate); an
  under-sampled ECSW/Q-DEIM (too few points) must show degraded accuracy (negative control).
- **Scoring:** the common scoring contract; Q1 qualification applies to the baselines too.

## Decision rule (H-base; V3-5 as amended by V4-5) — freeze in DESIGN.md
Objectives and directions: online time per query (lower better), offline cost (lower), median E0 error
(lower), worst E0 error (lower). Per mesh and dimension, against our qualified configurations from
priority 1 / speed3d:
- "Baseline dominates ours" = better in online time **and** worst error with non-overlapping intervals
  (paired per-case where applicable), offline cost and median error reported beside it.
- "Matches" only via TOST on paired per-case differences with an explicit equivalence margin; overlapping
  intervals give **"unresolved"**, never "matches".
- Crossing fronts → **"neither dominates"**, never "matches or beats".
Timing: paired A–B–A in one job on one GPU against our method's same-mesh query (full-field contract).

## Inputs
- 2D: combined lane (`worktrees/2026-10-13-jcp-combined/experiments/jcp-combined`), FOM code `fom2.py`,
  E0 references lane `refcore.py`, `refstore.py`. 3D: speed3d/combined `c3common.py`, `fomcore.py`.
- Historical POD-LSPG exists elsewhere in the repo (no hyper-reduction, rank 16 only): look, reuse ideas,
  but write fresh audited code in your lane dir.
- Heartbeat at `experiments/jcp-hr-baselines/HEARTBEAT.json`.
