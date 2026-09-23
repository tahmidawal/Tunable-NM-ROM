# heat-bank-knob — design and pre-registration (written before any cluster job)

Branch `exp/2026-09-23-heat-bank-knob` (sparse worktree forked from `exp/2026-09-20-hires-heat` @ 4fb12a6d; local
commits only, never pushed). Lane directory `experiments/heat-bank-knob/`. Cluster namespace
`/cluster/tufts/paralab/tawal01/hbank_20260923/<job>/`, one directory per job, never reused. Sibling lane:
`exp/2026-09-23-poisson-bank-knob` (same construction on Poisson 2D).

## Question

Is a **nested bank truncation** $R' \le R$ of the frozen heat banks a real accuracy/cost knob that needs no
retraining — and what heat rows of Table 1 follow from it under a rule fixed now?

## Models (frozen, byte-identical to the paper rows; `inputs/SHA256SUMS`)

| | bank | head | training | paper rows |
|---|---|---|---|---|
| 2D wide | `wide2d/bank.pkl` R=128 (93b0ec7a…) | `head_K8.pkl` K=8 (49c2ae73…) | 512 draws seed 791000 at $64^2$ | h2d-final04 (job 4060928), sealed 791099 (16) |
| 3D new | `vp_R320/bank.pkl` R=320 (edd8b0a3…) | `head_K32.pkl` K=32 (abe56bfa…) | 2048 draws seed 921000 at $32^3$ | final01 (4153878) + final256c (4175680), sealed 921099 (64) |

## Construction (offline, training data only; `hbk_core.make_rotation`, `make_prep.py`)

At the training mesh $G = Q_G R_G$. Rows $a_i = [Q_G^\top u_i;\ R_G h(z_i)] / \lVert u_i\rVert$ over the training fields
$u_i$ (all six output times of every training draw) and the stored training codes $z_i$. SVD $A = U S V_s^\top$,
$T = R_G^{-1} V_s$, $L = V_s^\top R_G = T^{-1}$. The rotated bank $G' = GT$ orders its columns by training energy.
Truncation $R'$: $u = G'_{:,1:R'}\, L_{1:R',:}\,(h(z) + D_q y)$ — the parent model with $h \to L_{R'}h$,
$D \to L_{R'} D$ in $R'$-dimensional coefficients. Weak matrix $a' = \Phi^\top G'_{:,1:R'}$ (DST of rotated columns),
triangular factor = leading block of the rotated bank's factor (nested QR). Elimination of $y$, nearest-code starts,
LM, CN stepping and batched fit are the parent code (`core.make_stages` of heat3d-bank @5f1b048d, ported verbatim
with the two bank reads injected). The rotated bank is stored as row blocks × nested column blocks so a query
reads exactly $R'$ columns. Pinned: `prep_2d.npz` (T sha 8974c18b…), `prep_3d.npz` (T sha 619641d0…), built on the
local GB10 from training draws whose hash equals the one recorded in each `training.json`.

Training-fit diagnostics (from `prep_*.json`, training fields, worst over fields): 2D floor at
R'=128/96/64/48/32/16: 0.144/0.152/0.18/0.30/1.28/9.0 %; 3D at 320/256/192/128/64/32: 0.027/0.11/0.24/0.93/5.3/16.7 %.

## Arms (every mesh, one allocation)

- `nmrom_R{R'}_q{q}_{cn|bf}`: 2D ladder {128,96,64,48,32,16}, q ∈ {0, 32, R'−8} (where valid); 3D ladder
  {320,256,192,128,64,32}, q ∈ {0, (R'−32)/2 rounded down to 16, R'−32}.
  `cn` = the paper's CN stepping arm (2D: moments init, 3D: field init; Δt 0.025, 4 starts, tol 1e-6);
  `bf` = the paper's batched fit (field init, exact-propagated moment targets, tol 1e-4, Cholesky).
- `lin_R{R'}_{cn|bf}`: the linear rung q = R' (head dropped, free coefficients in $G'_{R'}$), same init and stepping
  family (CN: the reduced weak CN step composed offline into one map per output; bf: each output fitted to the exact
  propagated moments).
- `parent_q{q}_{fam}`, `parent_lin_{fam}`: unrotated parent model at R'=R (parity; 2D all meshes; 3D ≤ 128³).
- FOM: CN–CG grid Δt ∈ {0.025, 0.05, 0.1} × rtol ∈ {1e-2, 1e-3, 1e-4} plus the tight named rtol 1e-6 (Δt 0.025);
  DST exact transform as a labelled control.

Cohorts: **validation** (2D seed 791001, 16 draws; 3D seed 921777, first 16 draws — training-time validation
seeds) for selection; **held-out** = the sealed cohorts already opened once for Table 1 (2D 791099 ×16, 3D 921099 ×64).
Draws are asserted disjoint from training. Meshes: 2D 1024², 2048², 4096² (job h2d); 3D 32³, 64³, 128³ (job h3d);
3D 256³ (job h3d256). H200, 240 G.

## Timing contract

GPU query (supplied full field on device → six full fields on device, blocked), f64, `highest` precision. Per case:
an untimed pass (compile, errors, fingerprint), then **phase 1** (all model arms) and **phase 2** (FOM + DST) — slow
arms in their own phase — each with 5 retained repetitions, a fresh random permutation of the arms per repetition,
0.1 s GPU burn-in before every invocation. Determinism: every timed output's fingerprint equals the untimed one.

## Gates (a job's numbers are used only if all pass)

1. **Parity** at R'=R: rotated vs unrotated parent fields, worst relative difference ≤ 1e-10 (every case, every
   paired arm, full fields).
2. **Determinism**: zero fingerprint mismatches.
3. **Order-effect (neighbour) gate**: each model arm re-timed right after a CN–CG (Δt 0.05, rtol 1e-3) solve on the
   first 3 held-out cases × 3 repetitions; median ≤ 1.10 × its phase-1 median on the same cases (arms ≥ 1 ms;
   |Δ| ≤ 0.1 ms below 1 ms).
4. **Independent NumPy audit** (`hbk_audit_np.py`, in-job): every saved error recomputed to 1e-10 with a SciPy DST
   reference; random-node full-grid estimates within 5 %; swapped-case and perturbed-error controls detected.
5. Solver health: arms with non-stationary LM exits or unconverged CG are ineligible for selection / as comparator.

## Pre-registered setting rule (fixed now; applied by `hbk_summarize.py` to VALIDATION rows only)

Per mesh and per stepping family (`cn`, `bf`; pool = `nmrom_*` + `lin_*` of that family, no failures):
- **accurate** = the arm with the lowest worst all-times same-grid error;
- **fast** = the cheapest arm (median GPU ms) whose worst error ≤ that of the current paper fast setting
  (`nmrom_R{R}_q0_{fam}`, q=0 at full R) on the same cohort.
The chosen settings are frozen and then read on the held-out cohort. **Table-1 speedup** = time of the fastest tested
CN–CG setting (no failures) whose held-out worst error ≤ the accurate arm's held-out worst error (one FOM per row) /
arm time, same job. Every arm is also reported against the fastest FOM at least as accurate as itself. The same rule
is also reported across both families pooled.

**Knob verdict** (held-out): worst error non-decreasing as R' falls, per family, for q=0, for q=R'−K and for the
linear rung; and the GPU time of the linear rung falls ≥ 2× from R'=R to the cheapest R' that meets the fast rule at
the largest mesh.

**Cost profile** at the largest mesh (4096², 256³; also 128³): encode $G'^\top u$ / init / evolve / decode $G'c$ per arm.

## Known limitations, stated in advance

The held-out cohorts were opened before (Table 1); nothing is trained or tuned on them here. Timing noise between
jobs is not a factor in any ratio (every ratio divides two times of one job).
