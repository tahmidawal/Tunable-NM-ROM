# Full experiment plan, v4: amendments after Codex audit 3

**Status:** DRAFT v4, 2026-10-17. This file amends `PLAN-v2.md` + `PLAN-v3-amendments.md` and supersedes them where they differ. It answers `codex-plan-audit-3.md` (NEEDS-RESTATEMENT). Blockers 1–4 (priority 1) are fixed here. Blockers 5–6 get concrete replacements here, to be frozen in the lane designs.

**Coordinator note: a real finding, not just a plan defect.** Audit 3 recomputed the 2D test-mode ordering, which sorts mesh-dependent discrete Laplacian eigenvalues (`hops.py:35–41`). At M = 1536 (the accurate setting, R384) the 256² and 1024² mode sets differ by 4 modes; at M = 2048, by 8. M ≤ 1024 is identical. So earlier cross-mesh 2D comparisons in the accurate setting used **slightly different test spaces at different meshes**. The effect on the published "flat" results is expected to be small but is **unmeasured**, and must be checked (V4-1).

## V4-1. Frozen mode lists in both dimensions (blocker 1)

- **Correction to v3:** 2D mode sets are **not** mesh-invariant at M ≥ 1536 under the existing builder.
- **Fixed-mode panels** (P1a, P1b, P1c's fixed-mode arm) use **one explicit mode list per (dimension, M)**: the 2D list taken at 1024² and the 3D list at 65³. The lists are hashed and reused at every mesh. Configurations built this way are new and need Q1/Q2 qualification.
- **Nominal-policy panels** keep the builder's per-mesh lists and are labelled "deployment policy (mode set varies with mesh)". The symmetric difference against the frozen list is reported for every (M, mesh).
- **New check (P1g, cheap, CPU):** for the deployed 2D accurate setting, re-run the existing 256² and 4096² accuracy comparisons with the frozen 1024² mode list, and report how much the earlier flat-accuracy numbers move. Reported, not hidden.

## V4-2. Q1 inheritance and three separate labels (blocker 2)

- **Q1 adopts the full speed3d contract** (`DESIGN.md:96–98,122–125,306–313`, A2.3 scoring-lattice distance):
  - every run valid (finite; no non-finite step; no damping saturation);
  - stationarity ratio $\gamma = \|J^\top r\| / (\|J\|_F\|r\| + 10^{-300})$, where a step is non-stationary if $\gamma > 10^{-3}$, and non-stationary steps are ≤ 1 % **pooled over steps**;
  - distance to the adaptive tight solve ≤ 2.5e-4 on **every** case;
  - tight/tighter anchor agreement ≤ 2.5e-5 on the prescribed subset.
- **Certification is computed in-job, at every step** (fresh Jacobian). Saved output states are not enough. Every P1 rollout configuration, BE included, is certified fresh. Combined's finalist labels are **prior numerical evidence only**, not a Q1 certificate.
- **Three labels, kept separate:**
  1. **Numerical qualification** (Q1 + Q2 pass).
  2. **Role qualification** (meets a pre-declared cost or role criterion, such as "equal cost").
  3. **Frontier usefulness** (not dominated).

  For example, combined's accurate-equal-cost fallbacks are numerically fine but miss the role criterion (cost 1.31–1.56×).

## V4-3. Run inventory, frozen rules and fixtures, cumulative cap (blocker 3)

- **Frozen rule per rank** (2D, numerically passing in combined's finalist table): R128 fib4181; R256 gauss128; R384 gauss128; R512 gauss192. Each is re-certified under V4-2 at every mesh.
- **Kernel fixtures** (labelled "kernel fixtures, not reached states of the smaller-r solver"):
  - **P1a:** output indices 1 and 5 of the LSPG-CN-R Δt₀ trajectories of R384 at 1024² for dev6 cases 0–5 (`W_R384.npz`), truncated to each r.
  - **P1b:** BE states, output indices 1 and 5, cases 0–5 of 923801 from `W_grid.npz` (W1024), truncated to each r.
  - **The fixed 3D pair is (M, m) = (the frozen 65³ mode list for r = 512, Gauss 24³ = 13,824 points).**
- **Run inventory, priority 1:**

| Item | Count |
|---|---|
| P1d new accuracy rollouts | 3 ranks × 2 schemes × (38 at 256² + 16 at 4096²) = **324** (+228 reused at 1024²) |
| P1d Q1 anchors | 324 adaptive tight solves + tighter solves on a 6-case subset per configuration |
| P1c timed rollouts | 3 ranks × 3 meshes × 2 schemes × 6 cases × 3 repeats = **324** (+36 for the R512 diagnostic at 1024²) |
| P1g frozen-mode recheck | 2 meshes × 38 cases × 2 schemes at R384 = 152 |
| P1a/P1b kernel evaluations | 6 ranks × 3 panels × 12 fixtures × components (seconds each) |

- **Cumulative cap.** A ledger records the measured GPU-hours of every finished job. A new job is submitted only if ledger spent + certified projection ≤ 20 GPU-h.
- **If the core** (P1a, P1c at 256²/1024², P1d at 256²) **cannot finish within the cap,** the outcome is **"P1 incomplete"**: no verdict, and the shortfall is reported.
- **If 4096² is dropped,** H-cost-N and H-acc-N are reported as "two-mesh evidence, registered three-mesh verdict not reached". They are never silently kept as pass.

## V4-4. Initialisation contract and the scaling model (blocker 4)

- **Correction to v3:** the existing 2D query does **not** project the full field. It samples the supplied full-field IC at the cold quadrature points and solves the reduced projection (`t2core.py:259–263`). P1c **keeps this contract** so the reuse stays valid, and labels it "full-field input sampled at quadrature points; full-field decode". A true full-grid projection would be a separate, labelled arm.
- **Scaling model.** The "intercept = overhead" reading is withdrawn.
  - Model per component: $t(r) = a + b\,r^{p}$, fitted by nonlinear least squares over r = 32–512 (2D) and 64–1024 (3D).
  - The pre-declared $p$ at fixed (M, m) is: nonlinear term about 1 (the m·r part), Jacobian 1, normal equations 2, Cholesky 3.
  - **Disagreement** means the pre-declared $p$ lies outside the 95 % case-bootstrap interval.
  - $a$ is reported as fitted fixed cost, cross-checked by the r = 32 kernel.
- **Paired timing mechanics:**
  - A = a fixed reference configuration (r = 384, 1024², LSPG-CN-R), interleaved A–B–A with each tested configuration in the same job.
  - B/A ratios are the timed quantity.
  - The bootstrap resamples the 6 cases with replacement, 10,000 times, percentile intervals.
  - With 6 cases the intervals are coarse, so "inconclusive" is an expected, legitimate outcome.

## V4-5. Statistical rules to freeze in the lane designs (blocker 5)

- **H-acc-N (replaces the degenerate three-point sign test).**
  - Per eligible case, fit the slope of error against log N over the 3 meshes.
  - Run a Wilcoxon signed-rank test of the slopes across cases (n ≈ 38 at 256²/1024²; the 4096² subset is 16, so use the 16-case panel for three-mesh slopes, labelled).
  - **No-trend equivalence** is a TOST on per-case paired differences e(N) − e(1024²) with a declared margin max(2ε_ref, 0.1 × e(1024²)). Absolute errors are reported beside it, since the margin is relative stability, not accuracy quality.
- **H-acc-r.** Report **every** rank transition, never just the passing ones.
  - A transition counts only if resolved coverage is ≥ 50 % of eligible cases.
  - "Improves" requires both > 50 % of resolved cases improving and a negative median paired change.
  - "Monotone over the registered range" requires every transition to improve; otherwise the claim is limited to named transitions.
- **H-base matching.** Paired per-case differences with an explicit equivalence margin (TOST). Overlapping intervals give **"unresolved"**, not "matches". Dominance uses the declared objectives: online time, offline cost, median and worst error, with each one's direction stated.

## V4-6. Competition and final confirmation (blocker 6)

- **Bound label corrected.** A finite FOM ladder gives an **upper bound on the optimal FOM cost**, and therefore an **upper bound on the claimed ROM speed-up** (`S/DESIGN.md:373–376`). V3's "upper bound on FOM efficiency" is withdrawn.
- **Speed3d reuse inherits its mesh-specific restrictions** (`S/DESIGN.md:457–462,537–543`): matched labels do not transfer across meshes. The g-grid does not supply the separated component profile; that comes from P1b.
- **The H200 trigger uses actual device bytes** (143,771 MiB as logged), not "141 GB".
- **F1** freezes exact immutable lane identities (branch, commit, config hashes); this is finalised at design stage.

## Budget change

P1g adds about 1–2 GPU-h inside the 20 GPU-h cap. Because it checks earlier published claims, it joins the **never-dropped core**. Everything else in the budget is unchanged.

## V4-7. Scope decision (user, 2026-10-17): add Navier–Stokes 2D and 3D

- The user added **incompressible Navier–Stokes in 2D and 3D** to the paper's scope, alongside viscous Burgers 2D/3D.
- **Existing assets:**
  - NS 3D has a coordinate-network bank and a mesh ROM (`exp/2026-10-01-ns3d-coordnet-bank`). Off-mesh quadrature has not been implemented for it.
  - A 2D incompressible NS appears in the historical roadmap. No current off-mesh lane exists.
- **What NS needs** (each its own lane, after the Burgers work):
  - divergence-free handling and pressure;
  - the co-moving/translation treatment used by the NS 3D lane;
  - order-verified references;
  - a spectral FOM comparator.

  The plan predicts **no accuracy gain against a spectral FOM** (governing plan, E6b), so the NS claims are cost and mesh-independence. Burgers priorities 1–4 are unchanged.
