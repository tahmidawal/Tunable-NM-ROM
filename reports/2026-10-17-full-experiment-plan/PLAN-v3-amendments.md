# Full experiment plan, v3: amendments to v2 after Codex audit 2

**Status:** DRAFT v3, 2026-10-17. Not pre-registered. This file amends `PLAN-v2.md` and supersedes it wherever they differ. It answers `codex-plan-audit-2.md` (NEEDS-RESTATEMENT, six ranked blockers plus ten new problems). The coordinator verified the two largest new findings against the files:
- 3D Δt₀ = 0.01 (`jcp-combined/c3common.py:35`), while the E0 space-only reference S uses BE at 0.005 (`make_e0_registry.py:49–51`), so 3D S is unmatched and unused.
- Combined's finalist table marks qualification per configuration: R512 selected arms are qualified, and R256/acc/fast accurate arms are not.

## V3-1. Scoring and qualification contract (blocker 1; replaces v2 §3 "References" and Q1)

- **The S reference applies only to BE at physical Δt = 0.005.** In 2D that is the deployed Δt₀. In 3D the deployed BE step is 0.01, so **3D S is unmatched and never used**. Every other S comparison is labelled "unmatched diagnostic". This mirrors `make_e0_registry.py:36–51`.
- **Eligibility is queried per case-kind from the E0 store**, never applied as a fixed count.
  - Historical cohorts: 13 non-rankable case-kinds (12 S, 1 ST).
  - Sealed 2D (for later): 21 S and 5 ST case-kinds failed QC.
- **Qualification is per configuration, not per rank.** The key is: bank and rotation hash, r, the actual test-mode set (M), rule and points, mesh, scheme, Δt, solver and tolerances, initialisation, test weighting, and output metric.
- **Starting status comes from the combined lane's finalist table** (`jcp-combined/report.md` finalist section):
  - 2D selected arms qualified at R128, R384 and R512;
  - not qualified at R256, acc and fast for the accurate-equal-cost arm;
  - the fast-equal-accuracy fallbacks qualified (fib4181, gauss128).
- **Qualification is separate from usefulness.** A qualified rung that is less accurate (R512) stays on the qualified-but-dominated list; it is not called "diagnostic".
- **Q1 numerical bars**, adopted from speed3d `DESIGN.md:122–125`:
  - an arm qualifies if, **on every case**, its distance to a verified adaptive tight solve is ≤ 2.5e-4 (relative, in the scoring metric), **and** its non-stationary-step fraction is at or below the speed3d stationarity criterion;
  - otherwise it is reported as failing for that configuration.
- **Q3 is part of priority 1** (conditioning of A, the weighted J and $J^\top J$ for every priority-1 configuration).

## V3-2. Priority 1 made concrete and resource-bounded (blocker 2; replaces v2's priority-1 tranche)

**Hard cap: 20 GPU-hours *including* contingency.** It is preceded by a smoke-derived cost certificate following speed3d `resources.py`. Projected cost per job is computed from the smoke's measured setup, compile and per-trajectory times, and no production job is submitted without a certificate under its allocation.

**Matrix (2D only in priority 1; 3D comes from speed3d plus a kernel panel)**

| Panel | Ranks r | Meshes | Cases | Schemes | What is timed or scored |
|---|---|---|---|---|---|
| P1a kernel (2D) | 32, 64, 128, 256, 384, 512 | 1024² | 2 states per case from dev6 (reached states saved by combined) | n/a (single residual/Jacobian/normal/Cholesky evaluations) | each component separately; panels: fixed (M, m) = (1536, 9216); fixed m = 9216 with M = 4r; nominal M = 4r with the deployed rule per r |
| P1b kernel (3D) | 64, 128, 256, 512, 768, 1024 (W1024 prefixes) | 65³ | 2 saved states × 6 cases (923801) | n/a | same components; fixed (M, m) and nominal panels; actual shell M recorded |
| P1c mesh-cost rollout (2D) | 128, 256, 384 (+512 diagnostic at 1024² only) | 256², 1024², 4096² | timing: dev6 (6 cases × 3 repeats, A–B–A) | BE Δt₀; LSPG-CN-R Δt₀ | solve (evolution only), IC projection, decode and full query, all timed separately |
| P1d accuracy (2D) | 128, 256, 384 | 256², 4096² new; 1024² **reused** from combined | 256²: val32 + dev6 (38); 4096²: dev6 + val32[0:10] (16, labelled subset) | BE Δt₀; LSPG-CN-R Δt₀ | E0 ST error; Q1 and Q2 gates; per-case eligibility |
| P1e quadrature slice | r = 256 (2D), r = 512 (3D, from speed3d grid) | 1024² / 65³ | dev6 / 6 cases | BE | failing / deployed / finer rule, own reached states, both bars |
| P1f memory and offline | all above | all above | — | — | peak device memory per panel (measured); offline-cost inventory from logs |

- **Rollout count:** about 3 ranks × 2 meshes × 2 schemes × 38 (256²) plus 3 × 2 × 16 (4096²) ≈ 552, plus tight-solve and finer-rule checks on the same cases. The certificate decides the exact count.
- **4096² memory pilot first.** The full-grid R512 bank alone is 4095² × 512 × 8 B ≈ 64 GiB (audit 2 §7). P1c and P1d at 4096² run on an **H200 (141 GB)** after a pilot that measures peak memory. If the pilot exceeds 120 GB, then 4096² switches to a **chunked bank** (blocked construction with only the needed prefix retained, r ≤ 384, which is 48 GiB) and is labelled so. **No 4096² run happens without the pilot.**
- **Missing code (to write, audit, then smoke):**
  1. a kernel benchmark harness (separate component timers, fixed-(M, m) control, drift and synchronisation checks);
  2. a split query in `t2core.make_query`, so initialisation, evolution and decode are separately callable and timed, with an equivalence check against the fused query;
  3. a cross-mesh orchestration job;
  4. E0 scoring wired to the run path (the combined E0 registry and loader, reused);
  5. an own-state indicator for the quadrature slice.
- **Stop and defer order if the cap is reached** (the last item is dropped first):
  P1f-offline → P1e 3D → P1b → R512 diagnostic → P1d 4096² subset → P1c 4096².
  P1a, P1c at 256²/1024² and P1d at 256² are never dropped. Dropped items are reported as deferred, not silently omitted.

## V3-3. Scaling estimands (blocker 3; amends H-cost-N, H-cost-r, A1, B6)

- **Fixed test modes.** "Fixed M" means a fixed set of test modes. In 2D this is exact (M = 4r sine modes). In 3D, shell completion makes M mesh-dependent (W512: 2052 vs 2049), so fixed-M panels **reuse one explicit mode list across meshes**, and nominal-ratio panels are labelled as "deployment policy".
- **Query contract.** P1c measures the **full-field hybrid query**: full-field IC projection and full-field decode on the mesh. Speed3d's mesh-free query (quadrature IC projection, scoring-lattice output) is a **different contract**, reported separately and never substituted.
- **Timing statistic and decision rule (H-cost-N).**
  - Per configuration and mesh: the median over 6 cases × 3 repeats of paired A–B–A solve time.
  - Ratio R = max over meshes ÷ min over meshes, with a 95 % interval from a case-level bootstrap.
  - **Pass** if the upper bound ≤ 1.25; **fail** if the lower bound > 1.25; otherwise **inconclusive**.
  - Must-fail: an injected 1.5× timing fault on one mesh must be flagged as fail.
- **H-cost-r.** The expected slopes are pre-declared per panel. At fixed (M, m): $C_N$ about 0 in r beyond the $mr$ term, $C_{J_N}$ about 1, $J^\top J$ about 2, Cholesky about 3. Slopes are fitted over r ≥ 128 (small-matrix overhead excluded and reported), as a log–log least-squares fit with a bootstrap interval. **Disagreement** means the expected slope is outside the 95 % interval. Fixed overhead is reported as the fitted intercept.
- **B6 in 3D.** The M = r control breaks the shell (an explicit lowest-r mode list) and is labelled; nominal ratio 1 with shell completion is reported separately.

## V3-4. Speed3d reuse mapping (blocker 4)

| Speed3d output (after its audits) | Satisfies | Contract label |
|---|---|---|
| grid g1–g3 (QN, FS-K1/K2, predictors; W256–W1024, M2_512) | Q1 for those 3D arms (its ≤ 2.5e-4 bar), part of A2 (3D component profile) | full-field 65³ rollout |
| f1 FOM ladder | 3D C1 comparator set (upwind/BE, central-CN, spectral) | finite ladder: upper bound on FOM efficiency |
| m1 mesh panel | 3D A1 (cost against N), C2 | **mesh-free / scoring-lattice**, labelled; not the full-field contract |
| p1 profile and batching | A7, A8 | mesh-free |
| Not supplied | 3D fixed-(M, m) kernel panel (→ P1b); 3D Q2 own-state checks beyond its grid; 2D anything | — |

Speed3d's reduced-rule arms are not assumed to be quadrature-equivalent; each needs its own Q2 check before entering a front.

## V3-5. Falsifiable accuracy and competition claims (blocker 5)

- **Rank range registered now:** 2D r ∈ {128, 256, 384, 512}; 3D r ∈ {256, 512, 768, 1024}. Qualification is decided by V3-1 gates, independent of observed accuracy.
- **H-acc-r (monotone test).** For each consecutive pair, count qualified cases that improve, worsen or tie (ties within 2ε_ref).
  - "Monotone improvement" holds for a step if improvements > 50 % of resolved cases **and** the median paired change < 0.
  - The claim "accuracy improves with r" is made **only for the steps that pass**.
  - The Pareto front is reported separately and is not a monotonicity test.
- **H-acc-N.** Per case and per qualified configuration, the maximum over meshes of |e(N) − e(N′)| is compared against max(2ε_ref, 0.1 × e(1024²)).
  - Pass if ≥ 90 % of eligible cases are within the margin and no monotone trend in N (sign test across the three meshes, p > 0.05).
  - Otherwise fail. The mesh-dependent linear terms are the declared suspect.
- **H-comp.** Evaluated **separately per dimension**, against **each FOM family** (upwind/BE, central-CN, spectral) and against their union, for each qualified setting at the largest fixed mesh.
  - The claim "≥ 2× faster" is stated only for the (dimension, family, setting) cells that pass.
  - K2 (governing plan) is applied as written: for every linear setting at the largest mesh.
  - All matches carry the finite-ladder upper-bound label.
- **H-base.** Objectives are online time per query, offline cost, and worst and median E0 error.
  - "Dominated" means the baseline is better in both time and worst error, with non-overlapping intervals.
  - "Matches" means the intervals overlap in both.
  - When fronts cross, the result is reported as **"neither dominates"**, never as "matches or beats".
  - Tuning budget is counted in GPU-hours on validation.

## V3-6. Freeze roster, dependencies and budget (blocker 6, plus new problems 8–10)

- **Authoritative roster for F1:** jcp-references (E0), jcp-mechanism, jcp-time2, jcp-wide-bank, jcp-smooth-bank, jcp-combined, jcp-speed3d, this plan's lanes (priorities 1–4) and the C3 baseline lane.
- **Explicitly excluded from the opening unless they freeze first:** D2–D5 and C4. If any of them joins later, it needs its own fresh sealed cohort (sealing is global).
- **F1 freezes:** bank and rotation hashes; mode lists and M; initialisation; metric and output times; cohort membership; the lane roster and exclusions; code versions (commit hashes); every threshold, fallback and failure rule; the comparator sets; and indicator calibration.
- **B7 (longer horizon) and B8 (sampling adequacy) get their own budget lines.** They need new references beyond T = 0.25 (B7) and a finer observation route than the 257²/63³ scoring arrays (B8): **+20–40 GPU-h**, priority 3.
- **A6 label fix:** W1024 training **job total 1.85 h, bank fit 1.14 h**. Existing shared costs are reported as sunk; per-query amortisation is computed only for new deployments.
- **D5 restated:** the domain must lack the relevant fast separable transform, **demonstrated** by showing that the spectral or FFT-preconditioned FOM cannot be applied. Candidates: an L-shaped or curved domain; mixed BCs alone do not qualify.
- **Budget (base, then with 25 % contingency on priorities 2–4 and F):**
  - P1 is charged its full 20 GPU-h, contingency included.
  - P2 60–110; P3 80–160 (now including B7/B8); P4 120–220; F 30–50.
  - Base total: **310–560 GPU-h**. With contingency on P2–F: **20 + 1.25 × (290–540) ≈ 383–695 GPU-h**.
  - Occupancy: ≈ 4.0 days (383 GPU-h on 4 GPUs) to ≈ 9.7 days (695 on 3). Calendar time is longer; the critical path is P1 → Q → C3-minimal → B5/C2 → F1.
  - D2–D5 are priced separately.

## V3-7. Positive gates vs negative controls (audit 2 §4 "Controls")

Each lane design lists three kinds of check:
- **Positive gates**, which must pass: reproduce a trusted FOM; a manufactured temporal-order check at the expected order; the split-query equivalence check; the fixed-mode-list reuse check.
- **Negative controls**, which must fail: an injected timing fault; doubled work; wrong-mesh table rejection; an under-resolved rule; a wrong-gradient solver; case and time swaps; a bad lattice vector caught by an independent family.
- **Sensitivity probes**, reported only: seed, shift, hardware.

For each, the design states which claim it blocks if it does not behave as required.
