**NEEDS-RESTATEMENT.** The design preserves most of the amended experiment matrix and fixes the earlier fixture, reuse and rule-selection errors. It does **not** yet freeze all delegated work without loss. Several mandatory controls can reject correct implementations, statistical edge cases remain undefined, and qualification can transfer to a different hardware execution without a numerical acceptance gate.

I made no file changes and submitted no jobs. Checks were read-only, including CPU recomputation of mode lists, NPZ metadata inspection and small statistical counterexamples.

Citation aliases:

- **D** = [lane DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/DESIGN.md)
- **A** = [P1-RESTATEMENT-AMENDMENTS.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/P1-RESTATEMENT-AMENDMENTS.md)
- **B** = [COMMON-BRIEF.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/COMMON-BRIEF.md)
- **C/** = `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/`
- **S/** = `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/`
- **R/** = `/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/`

**Section and amendment verdicts**

| DESIGN item | Verdict | Evidence |
|---|---|---|
| §0 question and notation | CORRECT | D:19–32 distinguishes bank width \(r\) from head dimension \(K\), and names the four hypotheses. |
| §1 model, initialization, tolerances | CORRECT | D:34–52 matches C/t2core.py:35–39,125–145,259–267 and C/c2common.py:32–36. |
| §2.1 manifest; AM-1 | CORRECT | D:58–66 separates numerical identity from provenance, as amended by A:176–177. Fresh 1024² runs replace archive reuse. |
| §2.2 frozen rules and ladders | CORRECT | D:75–82 implements the amended Fibonacci/Gauss successors, including R512’s gauss320 check. |
| §2.3 frozen modes | CORRECT | Independently recomputed from C/vendor/quad2d/vendor/hops.py:35–41. Counts and ordering claims agree; clarify the direction of “out/in.” |
| §2.4 kernel fixtures; AM-3 | CORRECT | Both specified archive entries exist with the required coefficient widths. Details below. |
| P1a/P1b matrix | CORRECT | D:118–124 preserves ranks, fixed panels and nominal rule mapping. Statistical and timing procedures still need repairs. |
| P1c | NEEDS-RESTATEMENT | Matrix and same-job pairing are correct, D:125–126,182–195. Invalid-bootstrap handling and hardware qualification transfer remain open. |
| P1d | CORRECT | D:127–128,341–343 includes fresh 1024² arms, the labelled 4096² subset and R512 diagnostics. |
| P1e | NEEDS-RESTATEMENT | Rules and historical 3D evidence are correct, D:129–134. §5’s population and certification specifications are written for 2D and do not explicitly resolve the six-case 3D slice. |
| P1f | CORRECT | D:135–138 distinguishes live allocations, compiled memory and process-lifetime peak; sunk/new costs are separated. |
| §2.6 P1g; AM-6/AM-7 | NEEDS-RESTATEMENT | Required gauss96 and gauss128 arms are present. H1’s cited rules are explicitly not rerun, so their moved historical numbers cannot be supplied: D:148,154–161 versus A:119–121. |
| §3 scoring | CORRECT | D:165–176 matches E0’s lattice, evolved-time maximum, fixed-reference normalization and eligibility API; refstore.py:10–16,132–155. |
| §4.1 H-cost-N | NEEDS-RESTATEMENT | Main rule is disjoint and executable on complete valid data. Some allowed missing-data bootstrap samples have no statistic. |
| §4.2 H-cost-r | NEEDS-RESTATEMENT | Boundary, fitting, operation-model and synthetic-fixture details do not uniquely determine all results. |
| §4.3 H-acc-N | WRONG | D:238–239 falsely says \(n<6\) cannot reject a one-sided Wilcoxon test at 0.05. Zero margins and all-zero shifted samples remain undefined. |
| §4.4 H-acc-r; AM-7 | NEEDS-RESTATEMENT | Precedence and range aggregation are correct for nonempty populations. Empty eligible populations are not handled. |
| §5.1 Q1 | NEEDS-RESTATEMENT | Fresh autodiff certification is valid in principle. Pure-relative agreement gates and blanket budget-exit invalidation add unjustified rejection paths. |
| §5.2 Q2; AM-2 | NEEDS-RESTATEMENT | Core bars, successors, own-state population and labels are present. Zero denominators, invalid successor solves and P1e scope need dispositions. |
| §5.3 Q3 | CORRECT | D:302–305 freezes matrices, own-trajectory output indices, aggregation and rank tolerance. Jacobian expression matches C/t2core.py:139–143. |
| §6 controls; AM-5/AM-8 | NEEDS-RESTATEMENT | Several improvements are correct, but independent fixture establishment and component accounting remain incomplete. |
| §7 timing/jobs | NEEDS-RESTATEMENT | Same-H200 design is sound. Kernel burn-in, raw timing retention and smoke coverage are incomplete. |
| §8 deliverables/imports | CORRECT | D:368–379 retains generated results and separates speed3d’s mesh-free contract from this lane’s full-field contract. The offline numerical agreement tolerance needs repair. |
| §9 reservation/defer/core | CORRECT | D:385–405 implements A:113–126, including never-dropped P1g at 4096² and “P1 incomplete.” Certificate implementation must count the full execution tree. |
| §§10–11 implementation/integrity | CORRECT | D:409–422 preserves provenance and cluster-only GPU evidence; governing operating rules are incorporated. |
| §12 claim that delegated work is frozen | WRONG | D:428–436 overstates completion: numerical edge cases, independent aggregate controls and component accounting remain unresolved. |

Thus AM-1 and AM-3 are substantially implemented correctly; AM-2, AM-4, AM-5, AM-6, AM-7 and AM-8 are present but retain the specific losses or ambiguities below.

**Control-by-control verdicts**

| Control, DESIGN line | Verdict | Finding |
|---|---|---|
| Synthetic timing fault, 314 | CORRECT | Fixed small-noise synthetic data distinguishes pass from fail. |
| Production timing fault, 315 | CORRECT | Properly demoted to a probe. |
| Doubled work, 316 | CORRECT | Two separate synchronized dispatches from the same state implement AM-8 and avoid the earlier elision/workload problems. State explicitly that the burn is outside the measured interval. |
| Component sum versus fused, 317 | WRONG | Invocation accounting omits history work and the initial gradient; separate-dispatch costs also need not lie within the mandatory interval. |
| Kernel dispatch synchronization, 318 | NEEDS-RESTATEMENT | Two dispatches are appropriate. The smoke’s ranks do not include the largest ranks required by this control. |
| Split-query equivalence, 319 | CORRECT | Same fields and iteration counts are suitable acceptance criteria. |
| Timed = scored, 320 | NEEDS-RESTATEMENT | In-job hashes establish trajectory identity. Configuration identity alone does not establish cross-hardware numerical qualification. |
| Reproduction, 321 | CORRECT | Matches AM-1 and the archive’s A100 80GB PCIe hardware. Matching GPU type still does not guarantee identical arithmetic across software versions. |
| Wrong-mesh table, 322 | CORRECT | Explicit metadata rejection tests the intended interface. |
| Under-resolved rule, 323 | WRONG | Independent evidence concerns another arm’s states and only \(\rho\); it does not independently establish the mandatory rollout-distance failure. |
| Q1 stationarity, 324 | NEEDS-RESTATEMENT | Independent check uses one case while the mutation names dev6. The pooled denominator is ambiguous; budget zero also does not necessarily return the quadratic predictor. |
| Sign-flipped Jacobian, 325 | CORRECT | Properly a probe. |
| Zero-update solver, 326 | CORRECT | Verified tight motion with the same initialization and normalization establishes the intended distance failure. |
| Rank-deficient fixture, 327 | CORRECT | Duplicated column is a discriminating rank-deficiency fixture. |
| Record swaps, 328 | CORRECT | Identity assertions address the claimed failure modes. |
| Synthetic accuracy trend, 329 | CORRECT | Fixed synthetic errors avoid the earlier data-dependent trend-control problem. |
| H-acc-r precedence, 330 | CORRECT | Expected outcomes match the amended precedence. Add an empty-population test. |
| H-cost-r fit, 331 | NEEDS-RESTATEMENT | \(a,b\), noise construction, seed and fixture grouping are unspecified. The stated model family does not guarantee pass/fail rather than boundary inconclusive. |
| CN-R temporal order, 332 | NEEDS-RESTATEMENT | Asymptotic, anchor and solver checks are useful. Difference of error magnitudes is not a bound on quadrature trajectory contamination. |
| Sensitivity probes, 333 | NEEDS-RESTATEMENT | “Untimed seconds” from separate accuracy jobs are not a matched A–B–A hardware-sensitivity experiment. Frozen Fibonacci realization is correct. |

**Ranked blockers**

**1. The under-resolved and component controls still can reject correct implementations.**

D:323 independently computes gauss32-versus-gauss64 \(\rho\) on the **gauss128 CN-R trajectory**, then mandates both rollout failure and excessive \(\rho\) for the **gauss32 trajectory**.

Neither implication follows:

- A large discrepancy at a gauss128 state need not occur on gauss32’s own trajectory.
- A large tested-advection discrepancy need not produce a trajectory distance above \(2.5\times10^{-4}\); damping or weak sensitivity can suppress its solution effect.
- Gauss64 is not independently established here as the converged comparator required by AM-7’s original fixture obligation.

AM-8 additionally requires failure against the actual successor; it does not withdraw the requirement for independently established expected results. Establish the actual own-state and rollout predicates independently, or register a fixture-selection procedure that can return “fixture unsuitable” without blaming correct Q2 code.

The component formula at D:317 also omits executed work:

- C/t2core.py:170–174 computes history projections and, on CN steps, \(F(c_n)\).
- C/t2core.py:68–70 performs an initial residual/Jacobian evaluation **and gradient multiplication**.
- D:317 includes the initial `evalJ`, but its `g` term is multiplied only by the iteration count.
- Predictor evaluations are batched by `vmap`, C/t2core.py:178–181; three isolated residual-dispatch timings need not equal that batched work.

Even complete arithmetic accounting does not prove the mandatory \([0.9,1.6]\) wall-time ratio. For example, separately dispatched components can each pay launch/synchronization overhead that the fused evolution avoids. A correct implementation can exceed 1.6. Freeze a measured, independently validated accounting fixture or make this discrepancy diagnostic rather than an unconditional correctness gate.

**2. The verdict functions are not total, uniquely executable procedures.**

For **H-cost-N**, the pass/fail/inconclusive inequalities are mutually exclusive, and all three outcomes are reachable. Missing meshes and excessive invalid triplets have explicit dispositions.

But consider two cases whose three repeats are invalid on one mesh. That is \(6/18=1/3\), so D:186 does **not** invalidate the panel. A bootstrap resample containing only those two cases has no valid timing on that mesh. Its median and max/min ratio are undefined. D:188–191 does not say whether to invalidate the panel, drop that resample, resample again or propagate NaN. Those choices produce different intervals.

For **H-acc-N**, the normalized signed-rank TOST is a coherent declared procedure; it is not a mean-error equivalence test. The main outcome predicates do not overlap. However:

- **The minimum-sample assertion is false.** With five cases and \(d_i=0\), the two one-sided tests have \(p=2^{-5}=0.03125\). Both reject. I reproduced this using local SciPy 1.17.1. D:238–239 instead prohibits rejection.
- If \(e_i(1024^2)=\varepsilon_{\mathrm{ref},i}=0\), D:229 divides by zero.
- If all \(d_i=1\), the upper shifted sample is all zero. The specified local SciPy call returns NaN. Define that side as non-rejection explicitly, rather than relying on incidental NaN comparisons.
- `method='auto'` is not a complete frozen numerical specification without a pinned version or explicit tie/zero method.

For **H-acc-r**, nonempty populations have correct precedence and reachable improves/does-not-improve/inconclusive outcomes. Empty eligibility is undefined: the percentage and majority denominators vanish, while the written “resolved < 50%” count comparison can become \(0<0\), which is false. Add an explicit empty-population inconclusive outcome.

For **H-cost-r**:

- `least_squares` cannot directly encode the open bound \(b>0\); freeze the transformation or numerical lower bound, initialization, scaling and failure criteria.
- Boundary handling names \(p\) and \(a\), but not a near-zero \(b\). Nearly constant times can leave \(p\) unidentified.
- D:331’s control permits \(a=0\), which D:214 explicitly classifies as inconclusive. It therefore does not guarantee its mandatory pass/fail outcomes.
- Operation-count fits use the “same fit,” but exactly linear FLOP models can have \(a=0\), a declared boundary solution. Specify whether those fits retain an exponent despite the boundary.
- Parenthesize D:218. As written, the Boolean scope of “or … and” and “that departing model exponent” is ambiguous.

The secondary label also changes the amended contract: A:64–65 calls a failure “operation-model” when the operation model departs from the declared exponent. D:217–219 additionally requires that departure to lie inside the measured interval. If declared \(p=3\), the byte model gives 2, and measured scaling is 1, the amendment and design can assign different labels. Freeze the intended classification explicitly.

**3. Q1 adds rejection conditions not justified by the contract, and qualification transfers across hardware too easily.**

A forward-mode autodiff Jacobian reconstructed from **every saved internal step**, within the same job, is a valid fresh-Jacobian certificate. It need not execute inside the timed evolution loop. It must reconstruct the **weighted** residual, actual startup/history and returned state; six output snapshots alone would not suffice. D:266–270 has the right architecture.

The agreement gate is problematic. D:271 requires \(10^{-6}\) **relative** agreement even when \(\gamma\) is near zero. Independent but correct arithmetic can produce, for example, \(10^{-14}\) versus \(2\times10^{-14}\): both safely stationary, but the gate fails. The same issue recurs in D:369’s \(10^{-9}\) relative offline agreement.

Use a declared absolute-plus-relative tolerance. The inherited speed3d precedent already does this: S/DESIGN.md:312–313 uses
\[
|\gamma_{\mathrm{cert}}-\gamma_{\mathrm{rec}}|
\le 10^{-9}\gamma_{\mathrm{rec}}+10^{-12}.
\]

Budget exhaustion is not itself non-finiteness or damping saturation. B:50–56 does not declare every budget exit invalid. A tight solve can exhaust its budget above its requested \(10^{-7}\) tolerance while satisfying the production stationarity threshold \(10^{-3}\), trajectory-distance and tight/tighter agreement bars. D:273 rejects it categorically. This is a stricter additional contract, even though the inherited combined solver’s own verification also rejects budget exits at C/t2core.py:189. State and justify the amendment or preserve the governing qualification predicate.

D:320 correctly ensures timed and scored coefficient trajectories agree **within the timing job**. That satisfies the central AM-5 output-identity requirement, provided those coefficients are decoded with the same tables and scoring contract.

It does **not** justify inheriting A100 numerical qualification merely from configuration identity. D:320 only reports the H200-versus-accuracy-job distance; it imposes no acceptance threshold. A different numerical trajectory could therefore receive the other job’s Q1/Q2 label. Require qualification of the timing-job trajectory, or a frozen numerical transfer gate with a failure disposition.

**4. The Q1 negative fixture and temporal-order gate do not fully establish their claimed predicates.**

D:324 names six cases but independently evaluates only dev6|0. One non-stationary step among that case’s 50 steps is 2%; among the six-case population’s 300 steps it is only 0.333%. Specify the denominator and establish **more than three** non-stationary steps if all six cases define the control population, or independently evaluate the complete pool.

Also, budget zero returns the solver’s selected initialization. C/t2core.py:176–183 chooses the lowest-residual candidate among current, linear-extrapolated and quadratic-extrapolated states. It does not necessarily return the quadratic predictor.

The CN-R order check includes the delegated asymptotic and anchor safeguards. Its quadrature check, however, uses
\[
|e_q-e_{q^+}|.
\]
Equal error magnitudes can conceal different trajectories. If two errors relative to an anchor are \(v\) and \(-v\), their magnitudes agree while their trajectory distance is \(2\|v\|\). Thus the proposed check can certify negligible quadrature contamination when contamination is substantial. Use a direct trajectory-distance bound under the same normalization, with verified finer-rule convergence.

**5. P1g’s historical mapping is factual, but its promised historical recheck is incomplete.**

The cited files and numbers exist:

| Historical item | Verification |
|---|---|
| H1 | R/2026-10-01-burgers2d-offmesh-quadrature.md:106–108 gives 0.012, 0.014 and 0.012 pp for gref, gauss64 and fib6765. |
| H2 | Same report:77 gives acc gauss96 test64 max/min 1.0034. |
| H3 | Same report:110–112 and122–124 confirms that the quoted 0.020/0.011 pp maxima come from `fast`. |
| H4 | Mechanism report:94 gives acc gauss96 and gref spreads of 0.008 pp. |
| H5 | R/2026-10-06-jcp-offmesh-paper-plan.md:27 and its evidence inventory:17,78 contain the stated summaries. Theory:100 contains the omitted-test-set assertion; theory:210 is the related proposed scaling check. |

But D:161 explicitly declines to rerun **every rule in H1**, while D:160 promises results next to H1 and A:119–121 requires the moved historical numbers next to the claims being rechecked. A gauss96 delta cannot supply a gauss64, gref or fib6765 correction.

Either narrow the registered historical claims explicitly to those actually rechecked, marking the others unresolved, or add their required comparisons and cost. Also establish vendor-versus-generic equivalence for the relevant new configurations: combined G1a is evidence for its tested setup, not a universal \(10^{-8}\) identity.

**6. Timing and resource execution details remain insufficiently frozen.**

Holding all meshes in one H200 job **does satisfy** same-job, same-GPU A–B–A, assuming the physical device is recorded and unchanged. Co-residency is a resource choice, not a pairing violation.

Remaining issues:

- D:356–358 provides three warm calls for kernels but no explicit GPU burn before timed kernel blocks. Warm compilation calls are not the required clock burn-in.
- It says the median is recorded, without requiring retention of all 30 timing repetitions. Raw repetition arrays must be persisted.
- The smoke promises every control, D:339, but its kernel ranks are only 128 and384. D:318 requires the largest fixed-panel ranks, 512 and1024.
- The reservation formula and defer order are correct. The certificate must expand all A flanks, retries, warm/scoring calls, burns, split/fused calls, independent controls and extra certification—not merely production rollout counts.
- A projection is not a hard runtime bound. To enforce the literal 20 GPU-hour cap, reserve enforceable allocations or define a runtime stop/reconciliation rule when projections are exceeded.

**Verified factual details**

The mode recomputation gave:

| \(M\) | 256² versus frozen 1024²: local-only / frozen-only counts | 4096² set difference | Order identical: 256² / 4096² |
|---:|---:|---:|---|
| 128 | 0 / 0 | 0 | yes / yes |
| 256 | 0 / 0 | 0 | yes / yes |
| 512 | 0 / 0 | 0 | no / yes |
| 1024 | 0 / 0 | 0 | no / yes |
| 1536 | 2 / 2 | 0 | no / no |
| 2048 | 4 / 4 | 0 | no / no |

At \(M=1536\), 256²-only modes are \((1,45),(44,9)\); frozen-only modes are \((28,35),(35,28)\). D:97’s “out/in” matches changing the frozen list into the nominal 256² list; label that direction explicitly.

Other checks:

- `W_R512.npz`: `W.shape == (38,39,6,512)`; run index15 is exactly `LSPG|CNR|0.005|prod`; the first six cases are dev6|0–5.
- Its archived setup confirms R512, M2048, gauss192 and mesh1024.
- `W_grid.npz` contains `W1024|lat65536|fs|BE|0.01|K1`, shape `(64,6,1024)`, with case IDs0–5 available.
- C/checks/SELECTION-3D-POSTHOC.json:1255–1266 records W512 gl56 convergence and lat8192 FS-BE distance **0.004212277353005088**, agreeing with D:131–133.
- S/DESIGN.md:51–54 confirms the nominal W256/W512/W768/W1024 rule mapping.
- All stated LSPG/GAL tolerance tuples match C/c2common.py:32–36.
- The 0.116 threshold is correctly attributed to the quadrature-study DESIGN.md:234, and correctly distinguished from its original certificate.
- The prescribed Fibonacci and Gauss successor chains are consistent with AM-2.

The principal unsupported-pass routes are the quadrature order check, cross-hardware qualification inheritance, unidentified cost exponents, and undefined invalid-data handling. The principal false-rejection routes are the component-sum gate, incompletely established negative fixtures and pure-relative certification agreement.

**Overall verdict: NEEDS-RESTATEMENT.**