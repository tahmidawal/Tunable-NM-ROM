**NEEDS-RESTATEMENT.** The design contains a false reuse claim, an impossible kernel fixture, an undefined quadrature gate, incomplete verdict rules, and controls that need not detect their injected faults. Several defects are inherited directly from v4; faithfully copying them does not make the lane executable.

I made no file changes and submitted no jobs. I inspected the archived NPZ metadata and ran only small, read-only CPU calculations.

For compact file:line citations below:

- **P** = [campaign/P1-RESTATEMENT.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/P1-RESTATEMENT.md)
- **V2/V3/V4/A3** = `PLAN-v2.md`, `PLAN-v3-amendments.md`, `PLAN-v4-amendments.md`, `codex-plan-audit-3.md` in that plan directory.
- **B** = `campaign/briefs/COMMON-BRIEF.md`.
- **C/** = `worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/`.
- **S/** = `worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/`.

**Section-by-section verdicts**

| Section/item | Verdict | Evidence and finding |
|---|---|---|
| §0 question and notation | CORRECT | P:13–15 distinguishes bank width r from head dimension K. However, its promise that all four hypotheses have pass/fail/inconclusive outcomes is not fulfilled below. |
| P1a, 2D kernels | WRONG | P:21 includes r=512, but P:51–52 supplies only an R384 vector “truncated to each r.” An array with 384 coefficients cannot supply 512 coefficients by truncation. |
| P1b, 3D kernels | NEEDS-RESTATEMENT | Rank range, W1024 fixtures and fixed pair are consistent with V3:33 and V4:34–35. The nominal panel has no frozen quadrature-rule mapping: P:57–58 specifies M but not m/rule. |
| P1c, mesh-cost rollout | NEEDS-RESTATEMENT | Matrix matches V3:34; the timing statistic is substantially specified at P:62–70. Missing drift disposition, precise A-pair denominator, and qualification/replay accounting remain. |
| P1d, accuracy | WRONG | P:24 reuses `c2grid` at 1024², while P:37 changes R256 from its archived `gauss96` to `gauss128`. C/configs/c2grid.json:20–24. |
| P1e, quadrature slice | NEEDS-RESTATEMENT | P:25 does not freeze its failing/deployed/finer rules, especially in 3D. The claimed second Q2 bar is undefined; see below. |
| P1f, memory/offline inventory | CORRECT | P:26 preserves measured peak memory and sunk/new accounting, as required by V3:37,101. Implementation must distinguish per-panel peaks from process-lifetime peaks. |
| P1g, historical recheck | NEEDS-RESTATEMENT | P:27 preserves the never-dropped check. It does not identify the exact historical configurations/artifacts being compared, or distinguish overlapping P1d work from additional work. |
| Q3 | NEEDS-RESTATEMENT | P:28 includes all three required matrices. It does not freeze which reached states/steps supply J, or the aggregation of conditioning across them. |
| §2 frozen configurations | WRONG | Correct bank, initialization and mode-list direction; false blanket reuse assertion and impossible R512 fixture. P:35–58. |
| §3 decision rules | NEEDS-RESTATEMENT | Individual findings below. |
| §4 controls | WRONG | Several “must-fail” assertions have counterexamples; applicable controls and claim-blocking consequences are also missing. P:112–123. |
| §5 resources | NEEDS-RESTATEMENT | Defer/core policy is faithful, but the workload omits required certification work and concurrent reservations are absent. P:125–136. |
| §6 missing code | NEEDS-RESTATEMENT | P:138–144 identifies useful work, but omits the complete reuse/configuration manifest, control validators and corrected Q2/statistical specifications. |
| §7 deliverables | CORRECT | P:146–149 requires generated reporting, audits, ledger and machine-readable outcomes. Those outcomes first need complete definitions. |

**Amendment coverage**

“Present” below includes requirements explicitly inherited through the common brief; I have not treated every abbreviated restatement as an omission.

| Requirement | Verdict | Coverage and remaining defect |
|---|---|---|
| V4-1: frozen modes, nominal labels, P1g | NEEDS-RESTATEMENT | Present at P:39–47,27. The explicit-list approach is sound; the claim that it establishes configuration identity is false. Historical P1g comparator provenance remains unspecified. V4:9–12. |
| V4-2: complete Q1 and separate labels | NEEDS-RESTATEMENT | P:103 explicitly inherits B:49–60, which contains validity, fresh-Jacobian stationarity, pooled 1%, all-case distance and tight/tighter agreement. P:110 preserves three labels. But the inventory only provides 324 new P1d anchors and does not implement fresh certification of reused 1024² configurations. |
| V4-3: rules, fixtures, inventory, cap | WRONG | Rules and most counts are copied faithfully. V4 itself contains the impossible R384→512 fixture and incompatible R256 reuse. P:51–54,134–136; V4:31–43. |
| V4-4: initialization and scaling | NEEDS-RESTATEMENT | Initialization is correct against C/t2core.py:259–267. Additive timing model and paired bootstrap are present. Non-fixed-panel expected exponents and fit-failure handling are absent. |
| V4-5: statistical rules | NEEDS-RESTATEMENT | Per-case slopes, TOST, resolved coverage and all-transition reporting are present, P:85–101. TOST with case-specific margins is not operationally defined; H-acc-r lacks complete three-way outcomes. |
| V4-6: reuse restrictions, hardware, freeze | NEEDS-RESTATEMENT | P:30–32 references the restrictions and P:128 uses logged capacity. Explicitly carry the mesh-free/scoring-lattice contract into the output schema; g-grid is not a component profile. S/DESIGN.md:457–462,537–543. FOM competition and final F1 identities are later-stage obligations, not reasons to add P1 competition work. |
| V3-1: scoring/qualification | NEEDS-RESTATEMENT | B:35–60 correctly carries ST, case-kind eligibility, S restrictions and three labels. Full configuration identity from V3:13 is not enforced by the mode-hash check. Q1 replay gap remains. |
| V3-2: bounded P1 matrix | NEEDS-RESTATEMENT | Scope/defer order preserved at P:19–28,125–136, with v4’s corrected count and added P1g. Required work is not fully inventoried. |
| V3-3: scaling estimands | NEEDS-RESTATEMENT | Later corrections to modes, query contract and additive overhead are included. Nominal/fixed-m cost-model predictions and timing failure treatment remain incomplete. V3:51–61; P:55–83. |
| V3-4: speed3d map | NEEDS-RESTATEMENT | Audited-only reuse is correct, P:30–32. State explicitly which imported result supplies which hypothesis and contract, rather than calling everything “3D cost-vs-mesh.” V3:65–73. |
| V3-5: accuracy claims | NEEDS-RESTATEMENT | P:96–101 requires every transition, but demotes 384→512 to diagnostic despite V3:77 registering through 512. Any primary monotonicity claim must explicitly end at 384 and separately report the full registered ladder. Competition/baseline clauses are later priorities. |
| V3-6: freeze/dependencies/budget | NEEDS-RESTATEMENT | P1 cap and sunk/new costs are present. P:36 refers to a development manifest “item 0 below,” but no such item exists. Exact identity/cohort/metric/tolerance manifests remain necessary. The campaign-wide F1 roster and later budgets need not be reproduced here. |
| V3-7: controls | WRONG | Categories are present, but the design lacks the required per-control statement of which claim is blocked, and several faults need not fire. V3:112–117; P:112–123. |

**The 1024² reuse claim is false**

P:40–42 reasons that matching the frozen mode list makes the reused combined result the same configuration. Mode equality establishes only one part of configuration identity.

The decisive counterexample is R256:

- Proposed configuration: `gauss128`, P:37.
- Archived `c2grid` configuration: `gauss96`, C/configs/c2grid.json:20–24.
- Separate R256 `gauss128` fallback results exist in the report, but those are explicitly `R256fb`, C/report.md:354–357. They do not turn `W_R256.npz` from `c2grid` into gauss128 data.

Direct NPZ inspection confirmed that `W_R256.npz` has shape `(38,39,6,256)`, includes the requested production BE/CNR runs, and belongs to the archived gauss96 configuration. The save path uses the setting’s `main` rule: C/c2grid.py:117–121,160–161.

Thus **76 of the claimed 228 reused P1d trajectories—38 cases × two schemes at R256—are not reusable as specified**. Locate matching artifacts with complete provenance or rerun them.

R128 and R384 have compatible rule selections, but still need full configuration equality and new Q1 certification. The archive’s extra tight/tighter trajectories are GAL-BDF2 time anchors, not the required same-scheme BE/CNR solver anchors. A mode hash cannot certify solver, tolerances, weighting, initialization, metric or reference version; V3:13 explicitly lists these.

The mode-list correction itself is justified by C/vendor/quad2d/vendor/hops.py:35–41. Freeze mode identities/order, while retaining the appropriate mesh-dependent eigenvalues and projections; C/vendor/quad2d/qcore.py:185–189 currently builds these together.

**Decision rules and controls**

| Rule | Verdict | Can it return the promised outcomes? |
|---|---|---|
| H-cost-N | NEEDS-RESTATEMENT | **Yes**, its interval rule can return pass, fail and inconclusive. P:65–68. Its must-fail control is not guaranteed, and missing/drift-invalid panels need an explicit outcome. |
| H-cost-r | NEEDS-RESTATEMENT | **No**, P:79–83 defines “consistent / disagrees (op-model) / disagrees (hardware),” not pass/fail/inconclusive. A wide or unidentified exponent interval can become “consistent.” |
| H-acc-N | NEEDS-RESTATEMENT | **Nominally yes**, P:91–93 defines all three. But the TOST statistic and several edge cases are unspecified, so results are not uniquely implementable. |
| H-acc-r | NEEDS-RESTATEMENT | **No**, P:98–101 defines “improves” and a monotonicity conjunction without mapping insufficient coverage, failed qualification, or non-improvement into fail versus inconclusive. |

For **H-cost-N**, I constructed this six-case example, with identical repeats:

- Largest-median mesh: `[1,1,1,101,101,101]`.
- Each other mesh: `[2,2,2,99,99,99]`.

The first mesh’s median is 51; the others’ are 50.5. After multiplying the first mesh by two, the full-sample ratio exceeds two. Nevertheless, the specified joint case bootstrap gives a 95% interval approximately **[1.000, 2.040]**, hence **inconclusive**, not fail.

This directly refutes P:71–74. Selecting the largest observed median does not ensure that it remains largest in bootstrap samples. Use a separately frozen deterministic fault fixture with a demonstrably separated interval; do not change the inference procedure until an arbitrary production-data mutation rejects.

Also freeze the A denominator—speed3d uses the mean of its two flanking A timings—and drift exclusion/retry rules. S/DESIGN.md:105–111 supplies those details; P:62–74 does not.

For **H-cost-r**:

- The fixed-(M,m) exponents are reasonable leading-order predictions: C/vendor/quad2d/qcore.py:262–266 and C/t2core.py:76–78.
- P:82 promises verdicts for every panel, but expected exponents are specified only for fixed-(M,m). At fixed m with M=4r, the Jacobian/normal formation predictions change; nominal changing m is not generally one power law.
- Freeze parameter constraints, fit failures, boundary solutions and an identifiability criterion. An interval containing the expected exponent establishes compatibility, not positive agreement.
- The r=32 overhead cross-check does not exist for the stated 3D panel, whose minimum is 64. P:76–80.

For **H-acc-N**:

- A conventional TOST has a specified estimand and equivalence bounds. P:88–90 supplies a different margin for each case without defining whether differences are normalized, whether mean lower/upper slacks are tested, or another construction. Those choices test different hypotheses.
- Specify zero slopes, all ties, missing meshes, eligible-case intersections, and the two populations used for the 90% condition.
- A substantial symmetric middle-mesh excursion can have zero three-mesh slope. TOST may reject equivalence while Wilcoxon does not reject a trend, producing inconclusive indefinitely. That is consistent with the written logic, but limits what “fail” detects.
- V4:48 requires the “three-mesh verdict not reached” label for **both** cost and accuracy if 4096² is dropped. P:70 states it only for H-cost-N.

Its synthetic trend control is also **not guaranteed**. If the original errors follow
\(e(N)=c(N/1024)^{-0.2}\), multiplying by the proposed factor produces constant errors. The injected control can pass equivalence. Large reference-based margins can also absorb the perturbation. P:120–121 needs a fixed synthetic dataset, explicit margins, and known expected outcome.

For **H-acc-r**, retain separate outcomes for insufficient resolution and demonstrated non-improvement. Also label the result accurately: the rule establishes **majority improvement on sufficiently resolved cases**, not improvement for every case or in worst error. Reporting the worst change at P:101 is useful, but does not make the majority criterion a worst-case guarantee.

**Q1 and Q2**

**Q1: NEEDS-RESTATEMENT.** The inherited numerical contract is correct. Its execution inventory is not.

P:103–105 promises fresh every-step certification for every configuration, yet only counts 324 new P1d anchors. Reused 1024² arms, additional P1g cases, nominal-policy arms and the diagnostic rung require an explicit certification/replay disposition. Freeze tight/tighter tolerances, budgets and the six case IDs. C/c2common.py:32–36 provides candidate tolerances; mentioning “tight” alone is insufficient.

**Q2: WRONG.** P:107–108 asks for “the combined lane’s δ_quad finalist bar for ρ.” These are different quantities:

- Combined’s finalist check is a **trajectory-distance** check, C/DESIGN.md:450–453.
- Its converged-rule anchor has an additional check against a larger rule, C/DESIGN.md:442–445.
- A ρ indicator compares tested nonlinear terms on states; for example C/vendor/quad2d/vendor/hops.py:163–174.

The restatement therefore has **no defined numerical ρ threshold**, normalization, state schedule or aggregation. It cannot implement “both bars” unambiguously. Freeze those independently and state whether the finer rule itself needs a convergence check.

The proposed finer rules **do exist and are memory-plausible**:

| Rule/configuration | Points | Retained point tables, computed |
|---|---:|---:|
| fib6765, r=128, M=512 | 6,765 | 39.64 MiB |
| gauss160, r=256, M=1024 | 25,600 | 300 MiB |
| gauss160, r=384, M=1536 | 25,600 | 450 MiB |
| gauss256, r=512, M=2048 | 65,536 | 1,536 MiB |

These calculations use \(8m(2r+M)\) bytes for `Gq`, `Gs`, `Psi`. Dispatch accepts arbitrary Gauss orders and Fibonacci counts: C/vendor/quad2d/qcore.py:75–100. Construction and retained arrays are at :209–227.

These are **table-storage estimates, not peak-memory certificates**. Construction temporaries, concatenation, simultaneously live rules and the mesh bank still need the pilot. The large 4096² mesh bank is the larger concern.

**Remaining controls**

| Control | Verdict | Reason |
|---|---|---|
| Split-query equivalence | CORRECT | Appropriate positive gate, P:113–114; current fused path is C/t2core.py:269–274. |
| Frozen-mode hash equality | NEEDS-RESTATEMENT | Useful mode check; insufficient configuration/reuse check. P:114. |
| Timed/scored output equality | CORRECT | Essential integrity gate, P:115. |
| Reproduce R384 archived scores | CORRECT | Useful scoring-path gate, but does not validate R256 reuse. P:115–116. |
| Manufactured CN-R order | NEEDS-RESTATEMENT | Freeze the problem, refinements and acceptance interval; “≈2” is not an executable predicate. P:116. |
| Doubled work | NEEDS-RESTATEMENT | Specify two consumed sequential evolutions and matched baseline; otherwise compilation can eliminate duplicate work or the denominator can be wrong. P:73–74. |
| Wrong-mesh table rejection | NEEDS-RESTATEMENT | Requires explicit provenance validation: reduced A/λ tables can have identical shapes across meshes. C/c2common.py:120–132. |
| gauss32 must fail Q2 | NEEDS-RESTATEMENT | Plausibly under-resolved, but no guarantee for every chosen state; zero/sufficiently simple states can agree. Freeze a discriminating fixture and a defined Q2 gate. P:118–119. |
| Perturbed Jacobian must fail stationarity | WRONG | A nonzero scalar multiple of the correct Jacobian has the same stationary points; the normalized stationarity ratio is invariant to that scaling. Specify an independently certified, demonstrably discriminating mutation. P:119; B:53–54. |
| Case/time swaps | NEEDS-RESTATEMENT | Freeze identity/hash assertions and unequal fixtures. Aggregate errors need not reveal a permutation. P:120. |
| Sensitivity probes | NEEDS-RESTATEMENT | Hardware/repeat seed are listed; explain applicable seed/shift probes and exclusions. V3:115; P:122. |

Applicable V2 controls are also missing: zero-update solver, rank-deficient/ill-conditioned fixture, reference-kind swap, and component-time sum versus fused timing. V2:68–70. The required claim-blocking map is absent, V3:117. A blanket “shown to fire” does not define these tests.

**Budget, defer order and honest unsupported-pass paths**

The **20 GPU-h cap, defer order and never-dropped core are correctly copied**, including P1g. P:126–133 matches V3:26,47–49 and V4:85. The explicit incomplete-core outcome is correct.

The enforcement and inventory need repair:

- With four concurrent jobs, `spent + next projection ≤ 20` is insufficient. Several individually admissible jobs can jointly exceed the cap. Reserve projected cost for queued/running jobs. P:10,126–127.
- Count smoke, memory pilots, compilation, failures, fresh certification and anchors for all reused/additional configurations.
- P1g overlaps P1d at R384: both describe frozen-list runs on 38 lower-mesh cases and 16 higher-mesh cases. Identify shared executions versus independent rechecks rather than adding all counts blindly.
- P1g remains mandatory at 4096² even after other 4096² items are deferred. A failed memory feasibility check must yield incomplete, not silent P1g removal.
- P:36’s missing “item 0” manifest leaves critical provenance to inference.

Honest implementation could otherwise report unsupported success by using gauss96 R256 results as gauss128 results; accepting reused trajectories without fresh Q1; treating an arbitrary δ_quad number as a ρ threshold; calling an unidentified cost exponent “consistent”; excluding the known regressing upper rank from a broadly worded monotonicity claim; or carrying a two-mesh result forward as the registered three-mesh pass.

**Ranked blockers**

1. **Repair configuration identity and reuse:** replace the R256 `c2grid` reuse, freeze the complete configuration manifest, and inventory fresh certification.
2. **Define Q2 correctly:** separate ρ from trajectory distance, supply executable bars and own-state rules, and freeze P1e configurations.
3. **Repair kernel fixtures:** supply a genuine 512-dimensional 2D fixture or explicitly define a different construction; freeze nominal 3D quadrature.
4. **Complete all four verdict procedures:** define heterogenous-margin TOST, unresolved/invalid outcomes, non-fixed-panel cost models, and registered rank scope.
5. **Replace unreliable controls:** use fixed discriminating fixtures, independent checks, and explicit claim-blocking consequences.
6. **Reconcile resources and provenance:** reserve concurrent costs, include all qualification work, deduplicate shared runs, identify P1g historical comparators, and apply the missing-three-mesh restriction to accuracy too.

**Overall verdict: NEEDS-RESTATEMENT.**