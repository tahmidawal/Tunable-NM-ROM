**Overall verdict: NEEDS-RESTATEMENT.** A2 resolves most of audit 2. The remaining blockers concern reference-probe validity and the new budget-dependent eligibility procedure. The 2D design is close, but those rules still permit materially different implementations.

Read-only audit completed: no files changed, jobs submitted, or scratch files created. I read the complete DESIGN, both previous audits and both binding briefs. Code inspection was limited to checking the design’s implementability.

Below, **D** denotes [experiments/jcp-regimes/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/DESIGN.md).

| Audit-2 required fix | Verdict | Evidence and assessment |
|---|---|---|
| 1. Complete duplicate checks and correct RNG claim | **CORRECT** | D:456–465 explicitly covers N0s, all single-bump rows, and within/across two-bump rows. The implementation supports that contract: `rcohorts.py:257–275`; gates call it at `test_regimes.py:69–73`. The stream-disjointness overclaim is removed. |
| 2. Limit calibration claims; require valid probes | **NEEDS-RESTATEMENT** | D:467–472 correctly limits the conclusion to observed coarse-level agreement and retains conditional Richardson language. However, only the **tight** solve is explicitly required to pass `steps_ok`. The production member must also be valid; otherwise a failed production solve can produce a passing agreement probe. The draft makes this omission concrete at `refjob.py:199–203`. |
| 3. Missing σ, stiff-control coverage, OOD wording | **CORRECT** | D:474–477 defines unavailable σ and separate N0/N0s coverage, with a labelled N0-only fallback. D:462–465 replaces the unsupported “cannot approximate OOD” claim and accurately limits the realised N0s coverage. |
| 4. Pair/baseline coverage, zeros, Wilcoxon, primary labels | **CORRECT** | D:479–488 supplies the missing rules: coverage on both sides, positive finite errors, eight complete pairs, two-sided Wilcoxon, zero handling and method selection. N0s is explicitly descriptive. |
| 5. Pilot identities and control predicates | **CORRECT** | D:490–498 freezes the reference/calibration/Q-control cases and requires specifically `order-fail` for R-upwind, with earlier checks passing. NumPy recomputation confirms the stated indices. Pairing and N0s gates are present at `test_regimes.py:53–54,77–89`. |
| 6. Device-specific, complete reservations and escalation rules | **NEEDS-RESTATEMENT** | D:500–513 substantially improves budgeting: H200 only, all 120 regime references counted, explicit reservations and open-job accounting. But the tight allowance is not invariant to job partitioning/restarts, and the reduced-C2 branch does not specify how already-run pilot cases enter analysis. Details below. |
| 7. FOM ordering/no-match behavior; deferred 3D | **CORRECT** | D:515–519 defines a nondominated matching set under an explicit resolution/step partial order and specifies no-match output. The comparison remains accuracy-only. D:445–448 and 518–519 correctly defer 3D pending its own specification. |

The requested numerical checks:

- **Frozen pilots:** CPU NumPy regeneration from `rcohorts.py` gives C2 c9, \(s=953.414319759\), followed by c11, \(s=934.249240111\); B2 c15, \(s=138.457391463\), followed by c0, \(s=105.807356022\). A+’s largest viscosity is c13, \(\nu=0.383427422508\). All match A2.
- **48 primary labels:** correct: eight LSPG configurations × six regimes. N0s, GAL and paired contrasts remain additional exploratory outputs, not part of that count. A1’s no-family-wise-control qualification still applies. If C2 is deferred, its labels must be marked deferred rather than silently omitted.
- **Wilcoxon:** sufficiently specified for implementation. It tests signed ranks of \(\log q\) around zero, corresponding to \(q=1\); the bootstrap’s practical degradation threshold remains \(q=2\). These answer different questions. The small-sample approximation with ties is a declared methodological choice, not a blocker.
- **H200 timing evidence:** parsing the requested E0 logs yields **102 ST cases: 80 H200 and 22 A100**. H200 median is **145.5 s**; A100 median is **326 s**. Thus 150 s is a reasonable H200 planning rate, but “102 H200 cases” would be incorrect. No ST tight attempts appear in those logs.
- The **693 s refined case is the entire case**, including its base triple. Its added 12288-level solve took 526.0 s, plus compilation/overhead. Therefore A2’s \(150+693\), rounded to 850 s, conservatively double-counts the ordinary portion; it is not an observed incremental refinement cost. See [5221490.out:40](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/runs/fd2dtest64p2/logs/5221490.out:40).
- A2’s arithmetic is consistent: 3.6 + 13.6 + 4.5 + 2 + 1 + 2 = **26.7 GPU-h**, and ×1.5 gives **40.05 GPU-h**. These remain preliminary projections, not certificates that the new regimes will finish.

**Remaining required fixes**

1. **Require validity of both calibration-probe members.**  
   At D:467–472, require production and tight trajectories to pass the relevant per-step/finite-output checks. Specify whether a successful retry replaces a failed production attempt for this comparison. An invalid member must yield “probe invalid,” irrespective of numerical agreement. A tight/tighter anchor is unnecessary if the conclusion stays explicitly limited to observed agreement.

2. **Make the tight allowance a deterministic cohort-level policy and disclose its selection effect.**  
   D:504–508 currently allocates allowance per *job*. Splitting A+ across jobs, restarting a job, or changing case order can change which cases obtain eligible references. Freeze the allocation—for example, one allowance across the entire A+ cohort, assigned to the first floor failure in ascending case index, with consumption preserved across restarts.

   Also state the terminal behavior when a tight base triple subsequently fails order/Péclet: D:508 prohibits a refined tight triple, superseding A1’s unchanged escalation procedure. This must be an explicit stop-and-record branch. The draft currently illustrates the trap: `refjob.py:171–174` can still request a tight 12288 solve.

   **Skipping escalation can bias the eligible population**, with direction unknown. Floor failures need not preferentially identify hard ROM cases; they can also identify very small discretization differences. A2 discloses skipped cases and ineligibility, while A1 limits HOLDS to eligible cases, but neither explicitly says eligibility now depends on unequal resource policies across regimes. Require that disclosure beside accuracy comparisons, separate skipped-escalation counts, and distinguish “unresolved under this allowance” from failure of the full E0 procedure.

3. **Complete the budget-reduction branch and its analysis population.**  
   D:512–513 cuts C2 to c0–c7, but the mandatory smoke has already run C2 c9 and uses c11 for controls. State explicitly whether these remain diagnostic-only and excluded from the reduced primary cohort. Freeze the reduced reference, ROM, Q1/Q2 and PS populations together; otherwise “every sampled case” and “first eight cases” conflict.

   Before submission, the replacement certificate must also map every required workload to a reservation: reference reproduction, upwind triple, calibration probes—including A1’s conditional extra probes—ROM controls, retries and compilation. Preserve the brief’s defer order by explicitly disposing of optional FOM and 3D work before reducing C2, and specify stopping for amendment if the remaining mandatory workload cannot fit.

Minor points, not additional blockers:

- Describe the timing source as 80 H200 cases within the 102-case record; call 850 s a conservative planning allowance.
- Clarify that Wilcoxon “ties” means ties in **absolute nonzero log differences**.
- Apply ratio coverage after zero/non-finite exclusions.
- Explicitly identify whether the seven PS cases exclude N0s.

Once these three fixes are frozen, the design should be complete enough to code against for **2D**. This verdict does not certify the draft implementation or authorize the deferred 3D phase.