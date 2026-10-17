# Full experiment plan, v2 (restated after the Codex audit)

**Status:** DRAFT v2, 2026-10-17. Not pre-registered. It restates `PLAN.md` (v1) item by item to answer `codex-plan-audit-1.md` (verdict NEEDS-RESTATEMENT). Each lane built from it still needs its own `DESIGN.md`, Codex audit, code audit and real-size smoke under `reports/2026-10-08-jcp-lane-agent-rules.md`. Accuracy numbers it cites are E0-conditional or provisional, as labelled in their source reports.

## 0. Changes from v1 (the audit's top 10, and how each is handled)

| # | Audit problem | v2 fix |
|---|---|---|
| 1 | Sealed counts were doubled | Corrected: **2D 128 physical cases (256 case-kinds), 3D 64 physical cases (128 case-kinds)**, from the E0 report's QC table. Opening stays contingent on the global freeze (§F) and on the unresolved custody and opening tool. |
| 2 | Claims assumed the outcome | §1 states every claim as a **hypothesis with a falsifying outcome**. Mesh-invariance at fixed r and the r-dial are separate hypotheses. The known R'=512 regression in 2D and the 4–7 % 3D worst errors are recorded as prior evidence against unconditional versions. |
| 3 | Solver and quadrature qualification came too late | New **Group Q (qualification) runs first**: solver stationarity, quadrature convergence and conditioning for every configuration that enters a frontier. Unqualified configurations are reported, never ranked. |
| 4 | A1 confounded r, M, m; the 5 % bar was unsupported | A1 is split into fixed-(M, m), fixed-m and deployment-policy panels. The flatness bar is **pre-declared as max/min ≤ 1.25** (the existing registered bar), with the statistic, repeats and interval defined. A tighter equivalence margin is optional, declared before data. |
| 5 | C1 omitted the FOM mesh | C1 selects the comparator over **Δt × tolerances × FOM mesh**. Speed-up is reported both against the same-mesh FOM and against the **cheapest FOM on any mesh** meeting the accuracy criterion. |
| 6 | Existing assets confused with validated configurations | §2 inventory separates **exists / runs / qualified** for every asset. Bank prefixes below the qualified range are kernel-only until rollout-qualified. |
| 7 | E0 treated too uniformly | §3 scoring contract: E0 ST primary; S only for BE at Δt₀; the 13 non-rankable 2D case-kinds are shown but never ranked; ties within 2ε_ref; ε_ref labelled conditional; post-hoc labels (E0 A10, combined A10/A11, retracted A15) carried forward. |
| 8 | Global freeze and OOD custody under-specified | §F defines one global freeze manifest across all lanes and baselines. Reserved OOD cohorts are treated like sealed ones. New regimes (D1) use newly drawn **development** cases only. |
| 9 | Memory and setup cost underestimated | A5 measures **peak memory** (not table scalars); A6 inventories shared versus per-prefix versus per-mesh offline cost. Budgets include qualification, compilation and setup, plus a 25 % contingency. Large-table builds need a memory pilot first (Gauss 112³ OOMed at 42.9 GiB). |
| 10 | Duplicated speed3d; baselines delayed | Speed3d owns A7, A8 and C1/C2 in 3D, plus the 3D part of E3; this plan consumes its audited results only. **C3 (a minimal POD-LSPG with ECSW/Q-DEIM/GNAT baseline) moves to priority 2**, because it can falsify the framing. |

## 1. Hypotheses (each with what would falsify it)

Notation: **r = R'** (bank coefficients solved for); **K** = nonlinear head latent size; M = number of test functions (actual count reported; 3D shells make M only approximately 4r); m = quadrature points; N = mesh points per axis.

- **H-cost-N.** At fixed (r, M, m, scheme, Δt, tolerances), solve time per query is invariant in N: max/min ≤ 1.25 across N, with a pre-declared repeat count and interval. *Falsified* if any qualified configuration exceeds it. The full query (IC projection plus decode) is **expected to grow** with N, and is reported separately.
- **H-cost-r.** Per-iteration cost follows the leading-order model below, within fitted constants over the tested range. *Falsified* if the measured slopes disagree beyond the fit uncertainty. Iteration counts are reported, since they can depend on r and tolerances.
- **H-acc-N.** At fixed r, error against E0 ST is approximately invariant in N: within 2ε_ref plus a pre-declared tolerance. *Falsified* by a systematic trend. Known caveat: the linear terms (A, Λ) and the IC projection are mesh-dependent in the deployed hybrid.
- **H-acc-r.** Error decreases with r **up to a qualified limit**. Prior evidence against monotonicity: 2D R'=384 → 512 worsened on 22 of 38 cases (combined H1). *Falsified* claim = "monotone"; the reported object is the qualified Pareto front.
- **H-comp.** For qualified configurations, a speed-up ≥ 2× exists against the best validation-selected FOM (any mesh) at the largest meshes. *Falsified* as in kill criterion K2 of the governing plan. Unaudited f1 evidence already shows a spectral FOM winning at 64³.
- **H-base.** The method matches or beats POD-based hyper-reduced ROMs at equal tuning budgets. *Falsified* if a POD baseline is both cheaper and more accurate.

**Leading-order cost model** (from `offmesh.py:84,129–134`):

$$C_N = O(mr + mM),\quad C_{J_N} = O(mMr),\quad C_{J^\top J} = O(Mr^2),\quad C_{\mathrm{chol}} = O(r^3).$$

This is a per-operation model, not a rollout theorem. Deployed paths may obtain $N$ from $\tfrac12 J_N c$ (3D quadratic form), fixed-sweep solvers start with adaptive steps, and small dense kernels need not reach their asymptotic slopes.

## 2. Asset inventory: exists / runs / qualified

| Asset | Exists | Runs end to end | Qualified (rollout, gates) |
|---|---|---|---|
| 2D bank R512, ordered prefixes r ≤ 512 | yes (`qcore.py:141–145`) | r = 128–512 (wide-bank) | r = 128, 256, 384 (E0 combined); 512 diagnostic only |
| 2D prefixes r = 32, 64 | yes | not run | **kernel-only** until qualified |
| 3D M2 bank (R512), W1024 bank | yes (`c3common.py:36–39`) | r = 256–1024 | post-hoc rules (combined A10/A11); worst errors 4–7 % |
| 3D prefixes r = 64, 128 | yes | not run | kernel-only |
| Configurable M/r | wide-bank drivers yes; combined hard-codes `4*Rp` (`c2common.py:121,130`), 3D `complete_M(n,4r)` | M/r = 2, 3, 4 (wide) | requires driver change and requalification |
| Rules: Gauss, lattice, Sobol, Smolyak | yes, 2D and 3D | yes | per configuration |
| Halton | 2D generator and generic arm (`qcore.py:83–100`, `qstudy.py:270–292`) | ρ ladder only | no |
| Monte Carlo; 3D Halton | **no** generator in production dispatch | no | no |
| Central+CN-R and sine pseudo-spectral FOMs | yes, 2D and 3D (E0 `refcore.py`, `pscore.py`; ported in speed3d) | yes | via speed3d audits |
| QN and fixed-sweep solvers | yes (speed3d, combined) | yes | fixed CN K=2 fails on 7–15/64 3D cases |
| POD + DEIM/ECSW/GNAT | **no** (hr-baselines lane is scaffolding only) | no | no |
| Allen–Cahn | prototype in Hari's package (`pdes.py:106–141`) | small CPU only | no |
| Exponential PDE | only **steady Bratu** (`pdes.py:167–174`) | — | time-dependent F-K would be new work |
| NS 3D | bank and mesh ROM exist; off-mesh not implemented | mesh ROM only | no |
| Non-box nonlinear geometry | no | no | no |

## 3. Scoring and integrity contract (applies to every group)

- **Selection** only on development and validation cohorts (2D dev6/val32; 3D 923801). Historical test cohorts (2D test64; 3D 923901) are replication-only. **No new sealed or reserved-OOD cohort is touched before §F.**
- **References.** E0 ST is primary. S is used only for BE at Δt₀ and labelled "unmatched" otherwise. The 13 non-rankable 2D case-kinds are shown but never ranked. Two arms rank on a case only if they differ by more than 2ε_ref. ε_ref is a conditional Richardson estimate, so label it so. Keep a spectral cross-check where central-CN comparisons share E0's discretisation.
- **Post-hoc labels** carry forward unchanged: E0 A10; combined A10, A11 and the A15 retraction.
- **Statistics.** Cases are the unit. Time snapshots are not bootstrapped as independent. Report failures, not just survivors. Repeated selection, seeds and lattice draws get their own intervals.
- **Timing.** Paired A–B–A within one job on one physical GPU, warm JIT, burn-in, synchronisation checks. Hardware conclusions are compared as dimensionless ratios only.
- **Must-fail controls per group** (each demonstrated on a real-size smoke before any verdict):

| Group | Controls |
|---|---|
| Q | zero-update and wrong-gradient solver; under-resolved rule; rank-deficient or ill-conditioned fixture |
| A | injected timing fault beyond the drift threshold; doubled-work timing; wrong-mesh table rejection; component-sum vs fused-query consistency |
| B | under-resolved Gauss/lattice/Smolyak; case, time and reference-kind swaps; inadequate fixed-sweep solver; manufactured temporal-order check |
| C | reproduce a trusted FOM; reject a non-converged or too-loose comparator; reject a falsely matched accuracy pair; independent residual/Jacobian check for each ROM baseline |
| D | PDE-specific manufactured checks; Allen–Cahn no-advection-gap prediction; stability checks for new nonlinearities |
| E | bad lattice generating vector caught by an independent rule family; permuted-viscosity batching |
| F | reject changed hashes, wrong reference version, restricted cohorts, ineligible rankings |

## Group Q: qualification (runs first; gates everything that enters a frontier)

- **Q1. Solver qualification.** For every (r, M, rule, scheme, Δt, solver) entering B5, C1 or C2: stationarity rate, distance to a verified adaptive tight solve, and iteration counts. Fixed-sweep arms that depart from the adaptive solve on any case are disqualified for that configuration and reported. The 3D part is reused from speed3d after its audit.
- **Q2. Quadrature qualification.** "Converged" means agreement between successive rules for **the bank, solver, Δt and own reached states**, on both the ρ and the rollout-distance bars. Large rules need a memory pilot first.
- **Q3. Conditioning.** Singular values and condition numbers of A, of the weighted J, and of $J^\top J$, against r. Fixes such as orthogonalisation or regularisation are chosen on validation only, and labelled as changing the problem when they do.

## Group A: cost

- **A1. Mesh invariance of cost (H-cost-N).** Three separate panels:
  - (a) **fixed-work kernel panel**: fixed (r, M, m), fixed iteration count, timed per component;
  - (b) **fixed-configuration rollout panel**: adaptive solver, iteration counts reported;
  - (c) **deployment-policy panel**: each r with its qualified rule.

  Each spans N = 256²/1024²/4096² (2D) and 64³/128³/256³ (3D, from speed3d). Solve time, complete query, IC projection and decode are timed separately. Bar: max/min ≤ 1.25 as in §1.
- **A2. Per-component scaling (H-cost-r).** Time $C_N$, $C_{J_N}$, $J^\top J$ formation and Cholesky separately, against r. Fit slopes over the tested range with uncertainty. Report FLOP and byte counts (roofline). Note where the deployed path fuses operations.
- **A3. Factorial r × M × m.** Panels at fixed m, at fixed M, and at fixed (M, m), so representation size is separated from test-space and quadrature growth. Uses the wide-bank drivers (configurable r and M).
- **A4. m\*(r, M).** Defined over a discrete candidate list per rule family, on own reached states, with **both** bars (ρ and rollout distance). Selection can return "none" (as combined showed); report it.
- **A5. Memory.** Peak device memory (measured), table scalars (m(2r+M) at 8 bytes in f64), temporaries, the mesh bank, compilation and concurrently live rules, against r. The tensor's $Mr^2$ is shown as a computed line.
- **A6. Offline cost.** Shared bank training (W1024: 1.85 H200-h), prefix-specific setup, per-mesh setup, rule selection, compilation. Break-even query count is computed only against an accuracy-qualified competitor with a positive per-query saving.
- A7 and A8 (batching, mesh-free online path): **owned by speed3d** in 3D. A 2D analogue only if speed3d's design is reused unchanged.

## Group B: accuracy

- **B1. Error against r × N (H-acc-N, H-acc-r).** Qualified configurations only, under the §3 contract. R'=512 (2D) is a diagnostic rung.
- **B2. Error budget.** Separate one-ingredient ablations (floor, quadrature, time, test space, solver, IC projection, scoring) **plus interactions**. It is not an additive decomposition of norms.
- **B3. Rule families.** Staged:
  - (i) existing families (Gauss, lattice, Sobol, Smolyak control, 2D Halton) as full solves on own reached states;
  - (ii) MC and 3D Halton after their generators pass a code audit.

  Random families use repeated draws.
- **B4. Time order.** Observed order against a same-form converged anchor, with solver and quadrature errors verified to be smaller. Order is never awarded by method name: LSPG-CN is not established as second order.
- **B5. Qualified Pareto front.** Error against cost parameterised by r at each mesh. Only Q-qualified configurations go on the headline front. Failed ones stay visible in a separate panel. The front need not be monotone in r.
- **B6. Test space.** M/r ∈ {2, 3, 4, 8}, plus **M = r as a deliberately weak control**. Galerkin is a different solve equation ($Q^\top \mathcal R = 0$), not just a different test list. A low mixed-frequency test set needs new selection logic plus its own stability checks.
- **B7 (new). Longer horizons.** Run past T = 0.25 to measure error accumulation and stability.
- **B8 (new). Physical diagnostics.** Boundary error, energy and dissipation, overshoots, IC-projection and output-sampling error (does the scoring lattice miss fine features?).

## Group C: competition

- **C1. FOM ladder.** Upwind+BE, central+CN-R, sine pseudo-spectral, over Δt × tolerances × **FOM mesh**. The comparator is chosen on validation using pre-declared median **and** worst criteria with ε_ref, then confirmed by paired timing. The 3D part comes from speed3d; this plan adds 2D only.
- **C2. Crossover.** Report both the same-mesh crossover and the best accuracy-matched crossover (FOM on any mesh), for each opponent.
- **C3. Standard ROMs (moved to priority 2).**
  - **Minimal version first:** POD-LSPG at r and smaller ranks, with ECSW and Q-DEIM, using native solvers, then a common-solver ablation.
  - **Later:** GNAT, POD + exact tensor, best-practice EQ on our bank, continuum ECSW, CROM-style sampler.
  - **Fairness:** equal tuning budgets on validation; offline cost reported.
- **C4. Decoder ablation.** Equal data, training budget, retained rank and normalisation, with derivative accuracy and the representation floor reported in identical norms.
- **C5 (new). Linear span vs nonlinear head.** The headline results use the linear span. The claim for the head (K) is either repaired under pre-registration (governing plan E3c) or the paper drops "nonlinear manifold". If the advisor's "k" means K, a K ∈ {4, 8, 16, 32} sweep goes here.

## Group D: generality (each is its own development project, with its own budget)

- **D1. Parameters.** Pre-declared ν ranges and IC families, separating parameter extrapolation, IC-family change and stiffness. New regimes use newly drawn development cases. Reserved OOD cohorts stay closed.
- **D2. Allen–Cahn at full scale.** The prototype exists; needs a bank, references and production code. Predicted: **no advection-stencil gap**. Note that $Mr^3$ is the storage of a direct cubic tensor, not a lower bound for all alternatives.
- **D3. Exponential nonlinearity.** **Re-scoped**: time-dependent Frank–Kamenetskii is new work (problem, data, bank, references, stability regime). A steady Bratu study using the existing prototype is an optional smaller step.
- **D4. NS 3D off-mesh.** Off-mesh transfer is not implemented. Needs divergence and pressure checks and representation controls; the bank floor is already worse than POD-64.
- **D5. Geometry.** **Motivation corrected**: the zero-Dirichlet box is handled well by the sine spectral FOM, so the reason for a new geometry is a domain **without a fast transform** (non-box or mixed boundary conditions). Entirely new scope.

## Group E: robustness

- **E1. Randomised rules.** Shifts for lattices and scrambles for Sobol (not applicable to Gauss or Smolyak). Indicator coverage is checked for the **deployed single draw**. A bad generating vector must be caught by an independent rule family.
- **E2. Seeds.** Repeat the complete recipe (training, ordering, selection) 3–5 times, not just initialisation.
- E3 (solver) and E4 (conditioning) are now **Q1 and Q3**.
- **E5. Hardware.** Paired comparisons within each physical GPU and job; dimensionless conclusions compared across A100 and H200. Record clocks, precision and software versions.

## Group F: final confirmation

- **F1. Global freeze manifest** across all lanes. It includes:
  - every rule, r, scheme, Δt, solver and tolerance;
  - every **baseline** and FOM comparator;
  - every threshold, fallback rule and failure-handling rule;
  - the indicator calibration, the scoring code and the reference version, plus the OOD policy.

  It is hashed, and Codex-audited before access.
- **F2. Open the sealed cohorts once:** 2D 128 physical cases (256 case-kinds) and 3D 64 physical cases (128 case-kinds). Contingent on F1, on resolving key custody (the key currently sits on the shared account) and on building the opening tool (E0 A3.8, specified, not built).

## Priority, dependencies and budget

```mermaid
flowchart LR
  P1[Priority 1: manifest, kernel panels, 2D mesh-cost panel, Q on those rungs, quadrature slice, memory and offline inventory]:::p1
  S3[speed3d results, after audit]:::ext
  P2[Priority 2: C3 minimal POD baseline; Q1-Q3 full; B1, B4, B5 on qualified rungs; C1/C2 2D]:::p2
  P3[Priority 3: A3, A4, B2, B3 staged, B6, B7, B8, C5]:::p3
  P4[Priority 4: E1, E2, E5, C3 full, C4, D1]:::p4
  P5[Priority 5: D2, D3, D4, D5 as separate projects]:::p5
  F[F1 global freeze, then F2 opening]:::f
  P1 --> P2 --> P3 --> P4 --> F
  S3 --> P2
  P4 -.-> P5
  classDef p1 fill:#dbeafe,stroke:#1e40af; classDef p2 fill:#e0f2fe,stroke:#0369a1; classDef p3 fill:#ecfeff,stroke:#0e7490;
  classDef p4 fill:#f0fdf4,stroke:#166534; classDef p5 fill:#f3f4f6,stroke:#6b7280; classDef f fill:#fee2e2,stroke:#991b1b; classDef ext fill:#fef3c7,stroke:#92400e;
```

| Priority | Content | GPU-h | Notes |
|---|---|---|---|
| 1 (pilot) | see "Priority-1 tranche" below | **≤ 20 (cap)** | measured costs then re-price everything below |
| 2 | C3 minimal; Q1–Q3; B1, B4, B5; 2D C1/C2 | 60–110 | C3 needs implementation time, not only GPU-hours |
| 3 | A3, A4, B2, B3, B6, B7, B8, C5 | 60–120 | MC and 3D Halton generators audited first |
| 4 | E1, E2, E5, C3 full, C4, D1 | 120–220 | E1 cost depends on draws × rollouts |
| 5 | D2–D5 | separate budgets | each needs data, a bank and references |
| F | freeze and opening | 30–50 | depends on the frozen arm count |

Totals (priorities 1–4 plus F, from the listed ranges): **about 270–520 GPU-h** (0–20 + 60–110 + 60–120 + 120–220 + 30–50), plus 25 % contingency, so about **340–650 GPU-h**. That is excluding D2–D5. At 3–4 continuously busy GPUs it is roughly 3.5–9 days of occupancy; wall time is longer because of queues, implementation and audits. The critical path is priority 1 → Q → C3 minimal → B5/C2 → F1.

## Priority-1 tranche (existing assets only; ≤ 20 GPU-h)

1. **Development manifest.** Record bank and rotation hashes, actual M per r and dimension, E0 eligibility per case-kind, and post-hoc status. List reusable saved results from combined, wide-bank and time2 before rerunning anything.
2. **Kernel panel (A1a, A2, part of A3).**
   - 2D prefixes r = 32, 64, 128, 256, 384, 512; 3D W1024 prefixes r = 64, 128, 256, 512, 768, 1024.
   - Components timed separately, in three panels: fixed m, fixed M, and nominal M ≈ 4r (actual M reported).
   - Small prefixes are labelled kernel-only.
3. **2D mesh-cost panel (A1b/c).**
   - r = 128, 256, 384 at 256², 1024², 4096², one frozen rule per r across meshes, plus R512 as a diagnostic.
   - Solve, complete query, IC projection and decode timed separately, with paired repeats and iteration counts.
4. **Qualification of those rungs (Q1, Q2) and E0 accuracy (B1 slice).**
   - Use the existing BE and LSPG-CN-R implementations, solver-distance checks, and finer-rule checks.
   - Failed rungs are reported and excluded from any front.
5. **Fixed-r quadrature slice (A3/A4 slice).** At one middle r per dimension: a failing rule, the deployed rule and a finer existing rule.
6. **Memory and offline inventory (A5, A6).** Measured peaks added to the timing panels; offline costs taken from existing logs.
7. **Reuse speed3d (3D).** Its results are consumed only after its own audits, with no duplicate runs.

**Not in priority 1:** retraining, new PDEs, new test spaces, MC generators, sealed or OOD opening, Gauss 112³.

## Open decisions (user)

- Which "k" the advisor means: r = R' (assumed) or the head's K. If K, C5's sweep moves up.
- Custody of the sealed-cohort key, and whether to revisit the 2D floor check before opening.
- Whether D2–D5 are in scope for this paper, or deferred.
