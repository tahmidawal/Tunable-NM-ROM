**The amendments resolve four of the six round-1 blockers. Two remain: unreliable mandatory controls and overlapping H-acc-r verdicts.** These need small coordinator-level corrections; most remaining implementation details can safely go into the lane’s audited `DESIGN.md`.

I made no file changes, submitted no jobs, and did not write to `LAB-LOG.md`. Checks included source inspection, archived NPZ metadata, and small read-only CPU calculations.

Citations use these path abbreviations:

- **A** = [P1-RESTATEMENT-AMENDMENTS.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/P1-RESTATEMENT-AMENDMENTS.md)
- **P** = [P1-RESTATEMENT.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/P1-RESTATEMENT.md)
- **B** = `reports/2026-10-17-full-experiment-plan/campaign/briefs/COMMON-BRIEF.md`
- **V2/V3/V4** = `PLAN-v2.md`, `PLAN-v3-amendments.md`, `PLAN-v4-amendments.md` in the plan directory.
- **C/** = `worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/`
- **S/** = `worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/`
- **Q/** = `worktrees/2026-10-01-quadrature-study/experiments/quadrature-study/`

**Disposition of the six round-1 blockers**

| Blocker | Round-2 disposition | Evidence |
|---|---|---|
| 1. Configuration identity, reuse and fresh certification | **RESOLVED** | A:7–22 replaces reuse with 228 fresh rollouts and a full manifest. A:124–126 includes tight/tighter anchors for additional configurations. R256’s archived `gauss96` is correctly distinguished from proposed `gauss128`; C/configs/c2grid.json:20–24. |
| 2. Undefined Q2 and P1e | **RESOLVED at coordinator level** | A:25–43 separately defines trajectory distance, own-state ρ, thresholds, sampling and finer-rule checks. The remaining 3D rule selection is explicitly delegated to the lane before data. Qualification must retain its stated validation-population scope. |
| 3. Impossible R512 fixture and missing nominal 3D rules | **RESOLVED** | A:46–52 supplies a genuine R512 archive and identifies the source of the nominal 3D mapping. Direct archive inspection confirmed the requested fixtures. |
| 4. Incomplete verdict procedures | **PARTIALLY** | A:55–79 makes the cost and mesh-accuracy procedures operational at coordinator level. A:81–88 still permits simultaneous “improves” and “inconclusive” outcomes. |
| 5. Unreliable controls | **PARTIALLY** | Fixed synthetic tests and the claim-blocking map are improvements, A:90–110. The production timing mutation remains demonstrably unreliable; doubled adaptive evolution and wrong-gradient stationarity also lack their asserted guarantees. |
| 6. Resources and provenance | **RESOLVED at coordinator level** | A:113–128 reserves concurrent work, counts previously omitted costs, deduplicates P1g, preserves its mandatory status and specifies Q3 sampling. Exact inventory and historical comparator mapping belong in the lane design. |

**Disposition of every round-1 section and amendment-coverage finding**

Repeated findings are kept here so none disappears merely because its underlying blocker was discussed elsewhere.

| Round-1 item | Disposition | Evidence / remaining issue |
|---|---|---|
| P1a: 2D kernels | **RESOLVED** | A:46–48; genuine 512-coefficient fixtures. |
| P1b: 3D kernels | **RESOLVED** | A:49–52; source mapping is adequate for the lane to freeze exact configs. |
| P1c: mesh-cost rollout | **RESOLVED** | A:55–59 defines denominator, drift retries and invalid/missing-panel outcomes; A:113–126 covers certification costs. |
| P1d: accuracy/reuse | **RESOLVED** | A:7–22 replaces incompatible reuse with fresh runs. |
| P1e: quadrature slice | **RESOLVED at coordinator level** | A:25–43; exact 3D failing/finer rules remain a lane-design obligation. |
| P1g: historical recheck | **RESOLVED at coordinator level** | A:116–123 defines shared/additional runs and explicitly requires historical file:line mapping. |
| Q3 | **RESOLVED** | A:127–128 freezes reached-state indices and aggregation. |
| §2 frozen configurations | **RESOLVED** | A:15–21,46–52 repair manifest and fixture defects. |
| §3 decision rules | **PARTIALLY** | A:81–84 has overlapping outcomes. |
| §4 controls | **PARTIALLY** | A:94–95,101–103 retain unsupported mandatory outcomes. |
| §5 resources | **RESOLVED** | A:113–126. |
| §6 missing code | **PARTIALLY** | Manifest and validators are now specified, but the statistics/control implementations cannot resolve contradictory requirements themselves. |
| V4-1: frozen modes/configuration identity | **RESOLVED** | A:15–21; recomputing eigenvalues/projections preserves the distinction in C/vendor/quad2d/qcore.py:185–189. |
| V4-2: complete Q1 and labels | **RESOLVED** | A:7–9,124–126 plus B:49–60. |
| V4-3: rules, fixtures, inventory, cap | **RESOLVED** | A:7–22,46–52,113–126 supersedes the inherited defects. |
| V4-4: initialization/scaling | **RESOLVED** | P:48–50 matches C/t2core.py:259–267; A:52,61–66 repairs smallest-rank and fit/verdict issues. |
| V4-5: statistics | **PARTIALLY** | Normalized signed-rank equivalence is specified, A:68–79; rank-verdict overlap remains, A:81–84. |
| V4-6: speed3d contracts/hardware | **RESOLVED** | A:129–133 distinguishes imports and adds a cost-contract field; consistent with S/DESIGN.md:457–462,537–543. |
| V3-1: identity and Q1 replay | **RESOLVED** | A:15–21,124–126. |
| V3-2: bounded matrix/inventory | **RESOLVED** | A:113–126; exact arithmetic belongs in the certificate. |
| V3-3: scaling estimands | **RESOLVED** | A:55–66 explicitly makes non-fixed panels descriptive. This is a scope amendment, not an omitted exponent specification. |
| V3-4: speed3d mapping | **RESOLVED** | A:129–133. |
| V3-5: registered rank range | **RESOLVED** | A:85–88 restores the 128–512 claim at 1024² and separately identifies the shorter primary range. |
| V3-6: manifest/dependencies/budget | **RESOLVED** | A:15–21,113–126. |
| V3-7: controls and blocking map | **PARTIALLY** | Map supplied at A:91–110; several predicates remain unreliable. |

**Can the amended verdict procedures return all three outcomes?**

| Procedure | Answer | Assessment |
|---|---|---|
| H-cost-N | **Yes** | The interval rule at P:65–68 admits pass, fail and inconclusive. A:55–59 adds invalid/missing-panel outcomes. The problem is its production control, not the verdict rule. |
| H-cost-r | **Yes** | A:61–66 admits all three outcomes and prevents wide intervals from counting as agreement. Boundary/failure precedence is explicit. |
| H-acc-N | **Yes** | A:68–79 defines normalized differences and two one-sided signed-rank tests. Equivalence, clear departure and unresolved evidence can produce the three outcomes. |
| H-acc-r | **Not uniquely** | A:82–84 defines overlapping predicates without precedence. |

For H-acc-r, take **38 eligible cases**, with **10 lacking qualification** and the other **28 resolved and improving**. Coverage is \(28/38>50\%\), the improvement share exceeds 50%, and the median change is negative: **“improves.”** But \(10/38>25\%\) lack qualification: **“inconclusive.”** Reversing the 28 changes similarly produces both “does not improve” and “inconclusive.”

The repair is straightforward: apply the qualification/coverage exclusions first, then classify improvement. Also propagate an inconclusive transition to an inconclusive whole-range monotonicity verdict unless another transition already establishes failure. The existing conjunction at P:99–100 does not supply that three-way aggregation.

For H-acc-N, these details can remain with the lane:

- Compute percentages using the actual eligible populations; **35/38 and 15/16 are examples**, not fixed thresholds after exclusions.
- Use the three-mesh eligible intersection for slopes.
- Freeze Wilcoxon tie/exact-method handling, zero denominators and empty populations.
- Label the normalized signed-rank equivalence estimand accurately; it is not a test of the arithmetic mean error difference.

**Adversarial checks of the amended controls**

| Control | Round-2 assessment |
|---|---|
| Synthetic timing fault, A:93 | **RESOLVED.** With NumPy `default_rng(0)`, six cases, three meshes and three repeats, the unmodified fixture gave approximately **[1.0010, 1.0097]**, and the doubled mesh **[1.9921, 2.0077]**. These produce pass and fail respectively. Freeze the RNG and shape in the lane. |
| Production timing fault, A:94 | **NOT RESOLVED.** Multiplying every case on one mesh by three still does not guarantee a failing bootstrap interval. Counterexample below. |
| Doubled work, A:95 | **PARTIALLY.** Sequential dependency prevents elimination, but changing the initial state changes adaptive workload. Two evolutions need not cost twice one evolution. |
| Component sum versus fused, A:96 | **PARTIALLY.** The bounds are executable, but the component list must cover the same workload. Single-kernel component times cannot be compared directly with an entire adaptive query. The lane can define matching scopes. |
| Split-query equivalence, A:97 | **RESOLVED.** Appropriate positive gate against C/t2core.py:269–272. |
| Timed/scored equality, A:98 | **RESOLVED.** |
| Reproduction gate, A:99 | **RESOLVED in intent.** Separate numerical configuration equality from provenance/scoring metadata, as discussed below. |
| Wrong-mesh table, A:100 | **RESOLVED.** Explicit metadata checks address equal-shaped tables. |
| Under-resolved rule, A:101 | **PARTIALLY.** A concrete population is supplied, but \(m<M\) does not prove either required numerical failure. The specified fixture needs independent evidence that it separates the bars. |
| Wrong-gradient solver, A:102 | **NOT RESOLVED as a guaranteed stationarity control.** A non-scalar Jacobian mutation can still converge to a correctly stationary solution. |
| Zero-update solver, A:103 | **PARTIALLY.** It needs a frozen trajectory with independently established motion exceeding the distance bar; a stationary or nearly stationary trajectory is not discriminating. |
| Rank-deficient fixture, A:104 | **RESOLVED.** A duplicated column establishes rank deficiency for the intended tall matrix. Freeze the numerical rank tolerance in the lane. |
| Case/time/reference-kind swaps, A:105 | **RESOLVED.** Identity assertions detect mismatched records; use distinct records. |
| Synthetic accuracy trend, A:106 | **RESOLVED.** The small CPU example produced failure for the trend and equivalence for equal errors with noise. |
| CN-R temporal order, A:107 | **PARTIALLY.** Executable, but a previously observed order does not guarantee this finite-step fixture lies in the asymptotic regime. The lane must verify solver/anchor contamination before interpreting the gate. |
| Sensitivity probes, A:108 | **PARTIALLY.** “No randomised rules” is factually false. A fixed-realization scope can nevertheless be legitimate. |

The production timing counterexample uses identical repeats:

```text
Mesh A before mutation: [1, 1, 1, 101, 101, 101]
Mesh B:                 [3, 3, 3,  99,  99,  99]
Mesh C:                 [3, 3, 3,  99,  99,  99]
```

Multiply mesh A by three. Joint case bootstrapping gives a 95% interval approximately **[1.000, 3.061]**: **inconclusive**, not fail. Bootstrap samples dominated by the first three cases have equal medians across meshes. Increasing the multiplier did not fix the round-1 defect.

The doubled-work control has a different problem. If the first adaptive evolution costs \(t_1\), and the smoother continuation costs \(0.2t_1\), the measured ratio is approximately **1.2**, despite both executions being correct and consumed. C/t2core.py:165–188 shows the state-dependent predictors and convergence decisions. Use matched fixed work, or compare the sequential total with separately measured constituent work.

For the gradient control, even distinct correct and incorrect Jacobians can annihilate the same residual. For example,

\[
J=(1,1,0)^\top,\qquad
\widetilde J=(1,-1,0)^\top,\qquad
r=(0,0,1)^\top
\]

give \(J^\top r=\widetilde J^\top r=0\). More fundamentally, using a wrong derivative does not prevent an iteration from finding a genuine residual root. The mutation resembles the proposed “linear part intact, nonlinear part sign-flipped” structure at C/t2core.py:138–143.

The coordinator should require a **demonstrably discriminating fixture with an independent expected result**, rather than require this mutation to cause a particular production stationarity failure. Its exact construction can then be left to the lane.

**New factual claims checked**

- **R512 archive claim: correct.** `C/runs/c2grid/archive/output/W_R512.npz` contains `W` with shape **(38,39,6,512)**. Cases 0–5 are `dev6|0` through `dev6|5`; run index **15** is `LSPG|CNR|0.005|prod`. Output indices 1 and 5 are finite. The archive metadata records mesh **1024**, R512, M2048 and `gauss192`. Source corroboration: C/configs/c2grid.json:35–38,51–54,64–66; C/c2grid.py:110–121,160–161.

- **`tol_of` claim: correct.** LSPG tight/tighter are `(1e-7,0)` and `(1e-8,0)`; GAL tight/tighter are `(0,1e-13)` and `(0,1e-15)`. C/c2common.py:32–36 matches A:126.

- **The 0.116 citation: correct, with limited inheritance.** Q/DESIGN.md:234 registers that numerical bar. Its original population and target were different: internal states and a continuum target, Q/DESIGN.md:178–188. A:29–35 defines a new successor-rule/output-state check borrowing the threshold; it does not reproduce the historical B5 certificate.

- **2D rule availability: correct.** The Gauss orders are accepted by generic dispatch, and 4181, 6765 and 10946 are valid Fibonacci counts. C/vendor/quad2d/qcore.py:75–100. Availability does not certify peak memory.

- **Nominal 3D mapping: available.** S/DESIGN.md:50–54 supplies W256 `gl24`, W512/W768 `lat32768`, W1024 `lat65536`. W512 is confirmed in S/configs/g2.json:15–23.

- **“Next Gauss order” needs clarification in the lane.** W512’s deployed rule is a **lattice**, not Gauss. Therefore A:40–43 does not itself identify a successor. The lane can freeze an explicit Gauss comparator and convergence check; S/DESIGN.md:415–418 identifies historical W512 `gl56` evidence.

- **“No randomised rules”: false.** P1’s Fibonacci rules call `fibonacci_lattice(..., seed=0)`, C/vendor/quad2d/qcore.py:88–89. That implementation adds a random uniform shift, C/vendor/quad2d/vendor/hari_quadrature.py:107–116. Correct the statement to “one frozen seeded realization; shift sensitivity outside P1 scope,” if that is intended. No additional sweep is automatically necessary.

- **Literal full-manifest equality needs a sensible scope.** A:15–21 includes the code commit and E0 reference version. A fresh implementation/scoring run cannot literally share every provenance field with the historical archive. Compare the numerical configuration for trajectory reproduction, while recording old/new provenance separately. This is safe to specify in the lane; it does not undermine the fresh-run remedy.

**Details that can safely remain in the lane’s audited DESIGN.md**

Exact historical P1g report/table mapping, rule/configuration hashes, 3D P1e comparator selection, memory projections, solver budgets, bootstrap RNG details, numerical edge cases, and the deduplicated cost certificate do not require another coordinator-level expansion.

Two scope qualifications must accompany that work:

- A:37 restricts Q2 checks to **16 cases**. Report that validation population explicitly; do not describe the other 22 cases as individually Q2-tested. Include a clear qualification disposition for nominal-policy and R512 arms if they support claims.
- P1g must distinguish historical `acc`/`gauss96` from `R384`/`gauss128`—both exist in C/configs/c2grid.json:25–33. A generic new R384 comparison is not automatically an isolated correction to every historical accurate-setting result.

**Ranked remaining blockers**

1. **Repair mandatory controls that can reject correct implementations.** Remove the guaranteed-fail requirement for the arbitrary production timing mutation; make doubled work workload-matched; require independently discriminating numerical fixtures instead of assuming the wrong-gradient mutation must fail stationarity. Evidence: A:94–95,101–103.
2. **Make H-acc-r outcomes mutually exclusive and define whole-range aggregation.** Qualification/coverage exclusions need precedence over improvement/non-improvement. Evidence: A:81–88 and P:99–100.

**Overall verdict: NEEDS-RESTATEMENT.**