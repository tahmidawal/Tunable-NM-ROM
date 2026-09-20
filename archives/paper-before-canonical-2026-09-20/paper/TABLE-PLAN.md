# Table plan — ICLR 2027 submission

One table per experiment, with the reviewer requirement it answers and its status on disk. Written
2026-09-17 during the nine-lane campaign; status column updated as lanes close. Reviewer requirement
codes refer to the NeurIPS 2026 reviews (`REVIEWER-RESPONSE-MAP.md`): R1 one checkpoint per curve,
R2 competitive full-order solvers, R3 tuned baselines on shared data, R4 SMA cold start, R5
advection case with a linear POD baseline under the same knobs, R6 seeds with variance, R7 whether
the family is really free from one model including quadrature refits, R8 timing protocol, R9 precise
PDE specification, R10 harder domains, R11 the intrusive/non-intrusive asymmetry, R12 no hand-typed
numbers, R13 a correct time-step derivation, R14 when the nonlinear manifold is needed.

**Status vocabulary.** `done` = audited run JSONs on disk and a generator reads them. `running` = job
in flight. `failed` = job failed, diagnosis or resubmission pending. `not run` = nothing on disk.

| # | Table | Answers | Lane | Status |
|---|---|---|---|---|
| T1 | Problem specification per PDE: equation, domain, boundary treatment, source or initial-condition family, mesh, time step, reference solver and tolerance, cohort sizes, seeds | R9, R13 | all | done (configs + generators) |
| T2 | Provenance and protocol per result: checkpoint SHA, K, R, commit, job id, GPU, precision, burn-in, repetitions, sync policy | R1, R8, R12 | all | done |
| T3 | The tunability curve, Burgers 256² and 1024²: rungs q, dense and both EQ rule sets (256²) / transferred rules (1024²), both error metrics, t=0 compression, median GPU ms, same-job full-order controls | R1, R7 | b-panel, b-eqtop | **done** — bpn301 (job 3789570) replaces bpn101 at 256²; bpn203 (job 3789572, H200) at 1024²; T3b/T5b generated |
| T4 | Rank versus test count, crossed q × M, dense | the confound objection | b-qxm | **done** — fixed-M=1088 ladder passes the bar alone (2.44× error, 5.16× cost), pin dropped at lane commit b4e38103; round-2 extensions: q=512 unconverged (not plotted), M-saturation at q=256 reported in one sentence |
| T5 | Final same-job panel at 256² and 1024²: ROM rungs, POD-LSPG ranks, FNO, Newton at several tolerances, exact direct solve | R2, R3, R5, R11, R14 | b-panel | **done** — 256²: 0 of 39 admissible reduced non-dominated; 1024²: 5 of 20 on the evolved metric only, 0 on all-times; 512² panel bpn401 (job 3805065) queued to bracket |
| T6 | Head ablation at matched latent dimension, both PDEs | R5, R14 | inherited (head-ablation) | done |
| T7 | Three-layer decomposition: bank floor, best-found, solved | R14 | inherited + all lanes | done |
| T8 | Solver knobs at a fixed checkpoint: cap, tolerance, quadrature count, cold start | fxe8 F7/F8, 5mgh M11 | inherited | done |
| T9 | Quadrature rule certification: NNLS fit versus held-out error on reachable states, per rung | R7, 5mgh M8/M9 | b-eqtop | **done** — replication landed (job 3783811): rules confirmed at q ≤ 32, marginal above; top rungs are never called certified |
| T10 | Mesh ladder, frozen checkpoint, 64 → 1024 | R2, sub-cubic scaling | inherited (mesh-ladder) | done |
| T11 | Where the family collapses: Poisson ladder with POD and the direct solve; heat linear bank; wave; Navier–Stokes phase-2 gate (T11e, **final**: four settings — K=16, K=32, family dimension 8 (ns303, 3808498), 4x data / 2048 trajectories (ns302, 3808495); H-ORACLE fails in all four, ratios 1.15–1.46; lane closed at 9830d202, no phase-3 job ever submitted); head-only data scaling (T11f); exploratory q-ladder on the failed-gate K=32 manifold with matched POD-LSPG and the FOM tolerance ladder (T11g, T11h — ns304, job 3808502, lane commit 46650a1e, labelled exploratory after a failed phase-2 gate, not phase 3) | R14, GwrW G5 | p-linear, w-ladder, ns2d | **done** — Poisson 256²/1024²; wave 64²/256²/1024²; heat from the 2026-09-10 cell; NS CLOSED as a negative at both K=16 (job 3787319, 1.19× vs bar 2.0) and K=32 (job 3787320, ~1.15×); oracle values are upper bounds (some LM budget exits, counts in summary); lane commit 50bf36da |
| T12 | Seeds: per-rung mean and spread over three training seeds, monotone-seed count | R6 | b-seeds | **done** (development cohort; seed variability, T13 is the headline) — jobs 3783776/3783777/3783778, lane commit e533b48e; T12b per-seed verdicts |
| T13 | Sealed cohort: development versus sealed worst error per rung, difficulty-normalised | R6, pre-registration | b-seeds | **done** — sealed job 3804465 (A100-40GB), lane commit be9415ab; T13 + T13b; all four checkpoints monotone and meet the knob bar; C2/C2n fail at q=0 for the incumbent alone (wrong-branch solve on one sealed case), C3 fails on seed2 sealed q=64 |
| T14 | Neural operators on shared data: FNO, U-Net, Transolver capacities, matched cohort against the ROM and the full-order solver | R3, 5mgh M1 | no-second | **done** — U-Net and Transolver beat the ROM; accuracy claim withdrawn |
| T15 | Speed at parity: fused residual and Jacobian, folded head, block solve | R8 | inherited (b-speed) | done |
| T16 | Training study (appendix): data ladder, objective, K, smoothness penalty, with the reproduction caveat | 5mgh originality | inherited (b-head-train) | done (negative) |
| T17 | SMA-NM-ROM under the same cold-start protocol | R4 | none | **not run** — state as a limitation |
| T18 | L-shaped Poisson: signed-distance boundary factor, sparse-direct and preconditioned-CG controls, where the direct solver stops applying | R10 | lshape | **done** — solves at 64²/128²/256²/512² (jobs 3784662, 3784663, 3789568), lane commit dc762ed3; head q=64 non-dominated from 256² (2.77×) to 512² (7.60× cheaper than SuperLU) |
| T20, T20b | Burgers at ten times lower viscosity (appendix cell): fixed-M=1088 neural ladder, M=256 control, POD-LSPG at matched dimension, free bank, full-order tolerance/step grid; three-layer collapse ratios against the incumbent cell | structural-condition discussion, F2/F4 pre-registration | b-lowvisc | **done** — panel job 3817807 (A100-40GB), gate 3789639, training 3804337, lane commit df92e40d; F4 applies (mesh under-resolved, converged operator 19.3–20.9 % from the 4096² reference), so every number is reduced-vs-reduced only; criterion P fails (no reduced subject non-dominated); F2 does not fire (linear floors collapse 12–19x, head 3.3–4.5x); every neural rung beats every POD-LSPG rank, q>=64 on the reduced-only frontier; knob bar not met (1.36x) |

## What the tables must not claim

- No accuracy superiority over neural operators. T14 overturned it.
- No speedup over an efficient full-order solver in 2D. T5 shows zero of 27 reduced arms non-dominated
  at 256².
- No cost ratio across jobs or GPUs. Same-allocation only, per the timing protocol.
- No quadrature rule certified by its NNLS fitting residual; only held-out error on reachable states.
- No performance ratio in the abstract until T12 and T13 exist (both exist since be9415ab; the abstract carries the sealed top-rung range).

## Decisions still with the user

1. Burgers headline metric: worst over evolved times with the t=0 compression reported separately,
   against worst over all times. The fixed-M ladder is monotone on both, so this now moves the
   headline number rather than the ladder's shape.
2. The code-only GitHub mirror push, prepared and dry-run, not run.
3. Merge or archive the campaign worktrees.
