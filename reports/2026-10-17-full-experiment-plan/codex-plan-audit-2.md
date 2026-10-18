**Overall verdict: NEEDS-RESTATEMENT.** V2 fixes much of the first audit, but it still misstates qualification, contains a concrete 3D reference mismatch, and does not establish that priority 1 can finish within 20 GPU-hours using existing drivers.

I made no filesystem changes, ran no experiments, and did not access sealed solutions. Findings concern the local repository, not live scheduler status.

“CORRECT” below means supported by the repository or sound as a proposed experiment—not that its hypothesis will pass.

**Source key**

Paths in subsequent citations expand through this table; `C/c2common.py:121` means line 121 of that file under directory C.

| Key | Exact path |
|---|---|
| V2 | [PLAN-v2.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/PLAN-v2.md) |
| OLD | [Governing plan](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan.md) |
| INV | [Evidence inventory](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan/01-evidence-inventory.md) |
| C | [Combined lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined) |
| W | [Wide-bank lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank) |
| T | [Time2 lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2) |
| R | [E0 lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references) |
| S | [Speed3d lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d) |
| Q2 | [Original 2D quadrature code](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/experiments/quadrature-study) |
| Q3 | [Original 3D quadrature code](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d) |
| PDE | [Imported pdes.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/external/quadrature-study-2026-09-30/quadrature/nmrom/pdes.py) |
| HR | [Baseline-lane README](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/README.md) |

**1. Disposition of the first audit’s ten problems—and verification of §0**

| # | Verdict | Evidence |
|---|---|---|
| 1. Doubled sealed counts | **RESOLVED** | V2:9,147 correctly says 128 physical 2D cases and 64 physical 3D cases. R/report.md:245–258 contains four 32-case 2D partitions and two 32-case 3D partitions, each with S and ST. These are generated counts, not counts of eligible references. |
| 2. Claims assume outcomes | **PARTIALLY** | Separate cost/accuracy hypotheses are an improvement. The 22/38 regression is correct: C/report.md:104. The 3D worst errors are 4.41–6.70% for the selected arms: C/report.md:209. But H-acc-r’s “up to a qualified limit” leaves the stopping point selectable after seeing errors; H-cost-r lacks a specified statistical alternative. |
| 3. Qualification too late | **PARTIALLY** | Group Q now precedes frontiers. However Q1’s qualification key omits mesh, and “depart … on any case” lacks a distance tolerance. Existing speed3d permits distance ≤2.5e-4, not exact equality: S/DESIGN.md:122–125. Priority 1 explicitly includes Q1/Q2 but omits Q3: V2:188. |
| 4. Confounded scaling and unsupported 5% bar | **PARTIALLY** | The 1.25 bar is supported by S/DESIGN.md:180–181. Fixed-work versus adaptive panels are now distinguished. But §0’s claim that repeats and interval are “defined” is false: V2:24 merely promises to predeclare them. Priority 1 also omits the explicit fixed-(M,m) panel present in A3. |
| 5. Comparator mesh omitted | **RESOLVED** | C1/C2 now permit any FOM mesh. This matches speed3d’s amended cross-resolution selection, S/DESIGN.md:373–379. Preserve its finite-ladder upper-bound qualification. |
| 6. Assets confused with qualification | **PARTIALLY** | Small prefixes and new PDE development are correctly separated. But the “qualified” inventory still misclassifies 2D configurations: R256’s deployed selected arms fail, while R512’s selected arms pass their finalist checks. C/report.md:246–255. Qualification cannot be attached to rank alone. |
| 7. Uniform treatment of E0 | **PARTIALLY** | Conditional uncertainty, historical exclusions and post-hoc labels are retained. **The S matching statement is wrong in 3D:** R/refjob.py:43–44 uses BE 0.005 in both dimensions; C/c3common.py:35 defines 3D Δt₀=0.01. C/make_e0_registry.py:49–51 explicitly says the 3D S reference is unmatched and unused. |
| 8. Freeze/OOD custody | **PARTIALLY** | Reserved OOD protection and an opening-tool dependency are correct. F1 lacks the authoritative lane roster and explicit exclusions required by R/DESIGN.md:652–658. The graph allows F while priority-5 projects remain outside its dependency chain; their inclusion/exclusion must be explicit. |
| 9. Memory/setup/budget | **PARTIALLY** | Peak memory and setup categories are improved, and the total arithmetic is correct. The 42.9 GiB failed allocation is real: C/report.md:485. But no resource certificate supports priority 1, and A6 calls the 1.85-hour total training job “bank training”; the bank fit itself was 1.14 hours: W/report.md:111. |
| 10. Speed3d duplication/baselines delayed | **PARTIALLY** | Moving minimal C3 to priority 2 resolves the sequencing concern. Ownership is improved, but speed3d’s cross-mesh experiment uses mesh-free projection and scoring-lattice output, not the full-field hybrid contract implied by A1. S/DESIGN.md:156–179,358–362. Its reduced-rule arms also are not necessarily quadrature-equivalent: lines 415–418. |

**2. Hypotheses and cost model**

| Item | Verdict | Evidence and required correction |
|---|---|---|
| H-cost-N | **NEEDS-RESTATEMENT** | Fix bank, input/output contract, actual test set, physical cases and statistic as well as `(r,M,m,scheme,Δt,tolerances)`. Define repetitions, interval and pass/fail/inconclusive rule. Existing 3D shell completion changes M across meshes: W/report.md:194,200 gives 2052 versus 2049 for W512. |
| H-cost-r | **NEEDS-RESTATEMENT** | The operation model is sound, but “slopes disagree beyond fit uncertainty” does not specify an expected slope when M and m vary, a fit range, or handling of overhead. Q3/offmesh.py:84–87,129–134 identifies the actual operations. |
| H-acc-N | **NEEDS-RESTATEMENT** | Specify casewise versus aggregate invariance, an absolute/relative tolerance and a trend test. E0 uncertainty concerns error relative to the reference; it does not by itself define an equivalence test between meshes. Mesh-dependent operators are real: C/c3common.py:121–138. |
| H-acc-r | **NEEDS-RESTATEMENT** | Register the rank range and qualification independently of observed accuracy. Otherwise “up to a qualified limit” can discard the first regression. C/report.md:104 establishes the R384→R512 regression; lines 246–247 show R512 nevertheless passes finalist qualification. A Pareto front is not a test of monotonicity. |
| H-comp | **NEEDS-RESTATEMENT** | Specify whether ≥2× must hold in both dimensions, for at least one setting, and against each FOM family or their union. OLD:135 defines K2 over every linear setting at the largest fixed meshes. Any finite comparator ladder yields an upper bound relative to unrestricted FOM tuning: S/DESIGN.md:373–376. |
| H-base | **NEEDS-RESTATEMENT** | “Matches or beats” is stronger than the proposed falsifier. Absence of strict POD dominance does not establish that claim when fronts cross. Define uncertainty-aware dominance, failures, offline/online objectives and tuning-budget units. HR:3–7 confirms this baseline remains new work. |
| Leading-order model | **CORRECT** | `B@c`, `D@c`, projection and Jacobian contraction yield the stated orders; normal equations and Cholesky are explicit. Q3/offmesh.py:84–92,129–134. V2 correctly limits this to operations and acknowledges fusion/adaptive startup. |

**3. Every §2 inventory row**

| Inventory row | Verdict | Repository check |
|---|---|---|
| 2D R512 bank and prefixes | **WRONG qualification column** | Prefix support exists: Q2/qcore.py:123–145. Checkpoint and rotation files also exist locally. But C/report.md:250–251 marks R256 selected arms **not qualified**; lines 246–247 marks R512 selected arms **qualified**. “Diagnostic because accuracy regressed” is different from “numerically unqualified.” |
| 2D r=32,64 | **CORRECT with scope** | Existing rotation can supply these prefixes, Q2/qcore.py:141–156. They have no established production rollout qualification in the cited campaign. “Not run” should mean that campaign, not every historical smoke/kernel. |
| 3D M2/W1024 | **NEEDS-RESTATEMENT** | Both checkpoint files exist; C/c3common.py:36–39,95–100 loads them. W1024 supports the 256–1024 ladder; M2 stops at 512. Post-hoc selection and selected-arm worst errors are correct, C/report.md:181–209. List bank-specific configurations and gates. |
| 3D r=64,128 | **CORRECT with scope** | Prefix construction is supported by the rotation slicing in C/c3common.py:135–138; no cited campaign qualification establishes these small production rungs. |
| Configurable M/r | **CORRECT** | W/w2d.py:310–323 accepts independent M; W/w3d.py:14 uses configurable nominal ratio with shell completion. Combined hard-codes 4r, C/c2common.py:109–140 and C/c3common.py:124. |
| Gauss/lattice/Sobol/Smolyak | **CORRECT** | Dispatch/generation exists: Q2/qcore.py:83–100; Q3/rules.py:173–186. Smolyak is an intentional control, not an established accurate deployment family. |
| Halton | **CORRECT** | Q2/qcore.py:92–93 supplies the generator; Q2/qstudy.py:270–292 supplies a generic full-solve arm. Historical ladder availability is not accepted rollout qualification. |
| MC and 3D Halton absent | **CORRECT within production dispatch** | Neither appears in Q2/qcore.py:83–100 or Q3/rules.py’s production generation at 173–186. This is appropriately narrower than claiming no implementation exists anywhere. |
| Central+CN-R and sine spectral FOMs | **NEEDS-RESTATEMENT** | Both dimensions are implemented: R/refcore.py:102–109; R/pscore.py:111 onward. E0 has its own gates, R/report.md:86–98. “Qualified via speed3d audits” cannot cover 2D competitive timing/matching, and speed3d result acceptance remains a separate condition. |
| QN/fixed-sweep solvers | **NEEDS-RESTATEMENT** | Implementations exist, S/DESIGN.md:69–85. The 7–15/64 failures specifically concern fixed CN K=2 at 2.5Δt₀, C/report.md:155–169—not every K=2 configuration. |
| POD+DEIM/ECSW/GNAT | **CORRECT for the proposed production comparison** | HR:3–7 remains a scaffold. Historical POD-LSPG exists but has no hyper-reduction and only matched rank 16, INV:24. |
| Allen–Cahn | **CORRECT** | PDE:115–141 contains time-dependent cubic-reaction prototypes. CPU-only, reduced-scale evidence and serious accuracy limitations are recorded at INV:20,56. |
| Exponential PDE | **CORRECT** | PDE:173–174 explicitly sets Bratu `steady=True`, `T=0`. Time-dependent Frank–Kamenetskii is new work. |
| NS3D | **CORRECT within this campaign** | Mesh-bank ROM exists; an off-mesh result does not. INV:25,54 records this distinction and the weaker bank floor. |
| Non-box nonlinear geometry | **CORRECT within this campaign** | Historical L-shaped Poisson is linear and supplies no nonlinear off-mesh implementation, INV:53. |

**4. §3 scoring/integrity contract and Group Q**

| Item | Verdict | Evidence and correction |
|---|---|---|
| Selection cohorts | **CORRECT** | The named validation/replication distinction agrees with R/report.md:270 and S/DESIGN.md:404–409. Add an explicit freeze-before-replication dependency. |
| References | **WRONG as written** | Replace “BE at Δt₀” with **BE at physical Δt=0.005**, and state that deployed 3D BE at 0.01 is unmatched. C/make_e0_registry.py:36–50 already contains the correct distinction. |
| Non-rankable counts | **NEEDS-RESTATEMENT** | Thirteen is the **historical** count. It comprises twelve S case-kinds and one ST case-kind, R/report.md:171. Sealed 2D additionally has five failed ST and 21 failed S references, lines 173,245–252. Eligibility must be queried per case-kind, not implemented as a fixed exclusion count. |
| Post-hoc labels | **CORRECT** | C/report.md:181–191,449–450 and R/report.md:177–186 substantiate the named caveats. Retain configuration-specific fallback status too. |
| Statistics | **CORRECT as policy** | Cases are the right independent unit; failures and draw/seed variability must remain visible. Define aggregate uncertainty propagation as already done in C/report.md:48 and S/DESIGN.md:118–120. |
| Timing | **NEEDS-RESTATEMENT** | Pairing/warmup/burn-in are correct. Also require timed-output equivalence to scored outputs and persisted repetitions; S/DESIGN.md:105–111 provides an existing specification. Ratios alone do not make differently defined queries comparable. |
| Controls | **NEEDS-RESTATEMENT** | Good coverage, but not every listed item is a must-fail control: reproducing a trusted FOM and a correct manufactured-order test are positive gates. Distinguish positive gates, negative controls and sensitivity probes, as R/report.md:103–105 does. Define which failed control blocks which claim. |
| Q1 | **NEEDS-RESTATEMENT** | Include mesh, bank/rotation, initialization, test weighting and output metric in the qualification key. “Depart” needs a numerical threshold; exact equality would reject acceptable approximations. Existing speed3d uses ≤2.5e-4 on every case and a stationarity-rate criterion, S/DESIGN.md:122–125. |
| Q2 | **CORRECT** | Own-state, scheme-specific and rollout-distance checks address the actual failure mechanism. C/report.md:137–191,344–355 supports this requirement. Numerical bars and escalation limits belong in the lane design. |
| Q3 | **CORRECT** | Conditioning belongs before interpretation. Distinguish an exact coordinate change from regularization or a change of residual weighting/objective. Existing weighting is explicit in T/report.md:29–31. |

**5. Every experiment item**

| Item | Verdict | Evidence and required correction |
|---|---|---|
| A1 | **NEEDS-RESTATEMENT** | Fixed-work/adaptive/policy separation is good. Fix actual M/test modes across N or label shell changes as a policy comparison. Speed3d’s mesh-free scoring-output query is not the full-field query in C/t2core.py:252–274; S/DESIGN.md:177–181,358–362. |
| A2 | **CORRECT** | Separate operation timing and fusion disclosure are appropriate. Q3/offmesh.py:84–87,129–134 supplies kernels; speed3d currently times normal formation plus Cholesky together, S/s3modes.py:270–302. Separate measurements require instrumentation. |
| A3 | **CORRECT** | Independent r/M/m panels isolate the desired effects. W/w2d.py:310–323 supports M control. Combined needs changes, C/c2common.py:109–140. |
| A4 | **CORRECT** | Discrete candidates, both bars, own states and a “none” outcome match the evidence. C/report.md:173–189 demonstrates why “none” is necessary. |
| A5 | **CORRECT** | `8m(2r+M)` correctly counts point-form advection arrays, Q2/qcore.py:209–227; it is not peak memory. W/report.md:236 documents 1.61 GB tables versus 75.76 GB sampled footprint. |
| A6 | **NEEDS-RESTATEMENT** | Cost categories and break-even condition are correct. Change “training 1.85 H200-h” to **training-job total 1.85 h; bank fit 1.14 h**, W/report.md:111. Include failed setup attempts and define whether existing shared costs are sunk or amortized. |
| A7/A8 ownership | **NEEDS-RESTATEMENT** | Reuse is appropriate, but a 2D implementation cannot reuse the 3D projection/query design literally unchanged: compare Q2/qcore.py:314–320 with S/DESIGN.md:156–159. Preserve equivalent experimental contracts, not identical dimension-specific code. |
| B1 | **NEEDS-RESTATEMENT** | Fix qualification by configuration rather than rank and define the accuracy statistic/margin. C/report.md:246–255 contradicts the inventory’s rank-level labels. |
| B2 | **CORRECT** | Separate ablations plus interactions avoid adding norms incorrectly. C/report.md:119–125 shows the earlier cancellation explanation failed; W/report.md:113 requires consistent norms for floors. |
| B3 | **CORRECT as staged development** | Existing generic paths support stage i; generators/configuration/repeated draws remain necessary for stage ii. Q2/qcore.py:83–100; Q2/qstudy.py:270–292; Q3/rules.py:173–186. |
| B4 | **CORRECT** | Same-form anchors and independently smaller solver/quadrature error are necessary. T/report.md:17–23 confirms that CN/CN-R/BDF2 labels alone do not establish order. |
| B5 | **CORRECT, dependent on repaired Q** | A qualified frontier need not be monotone. C/report.md:100–105 supplies a concrete regression. Preserve dominated but qualified rungs separately from failed ones. |
| B6 | **NEEDS-RESTATEMENT** | GAL distinction and weak control are correct, T/report.md:31. In 3D nominal ratio one need not produce exact M=r because of shell completion, C/c3common.py:124. Specify whether the control breaks a shell or tests approximate equality. |
| B7 | **NEEDS-RESTATEMENT** | Scientifically useful but requires longer-horizon references and output/scoring changes. C/t2core.py:13 fixes six outputs; C/score.py:59–62 fixes times through 0.25. A horizon argument alone does not supply a complete longer-time experiment. |
| B8 | **NEEDS-RESTATEMENT** | Add a reference/observation route for detecting features missed by the scoring lattice. Existing E0 arrays cover 257² or 63³ points, C/score.py:59–62. Scoring those same points again cannot validate their sampling adequacy. |
| C1 | **NEEDS-RESTATEMENT** | Any-mesh selection is fixed. Define common prepared-input/output contracts and carry finite-ladder/boundary-match qualifications. S/DESIGN.md:358–379 provides the necessary precedent. |
| C2 | **NEEDS-RESTATEMENT** | Define crossover in ROM resolution/output workload while allowing FOM resolution to differ. It depends on mesh-specific accuracy qualification; speed3d A6 explicitly restricts transferring matched labels to other meshes, S/DESIGN.md:537–543. |
| C3 | **CORRECT as new development** | Smaller ranks, native solvers, common-solver ablation and equal tuning budgets address audit 1. HR:3–7 confirms there is no ready production implementation. Specify each baseline’s actual residual/sampling formulation in its design. |
| C4 | **CORRECT** | Matched training/data/rank/norms and derivative/floor measurements are appropriate. W/report.md:113 explicitly warns that the existing old/new bank comparison does not isolate rank. |
| C5 | **CORRECT** | The headline method is a linear span; repair or remove the nonlinear-manifold claim. INV:38; OLD:45,87. The proposed K sweep needs new matched heads, not just slicing a trained K=16 input. |
| D1 | **CORRECT as separately designed development** | Protecting reserved OOD while drawing new development regimes is appropriate, R/report.md:237–258. Numerical distributions/reference requirements remain lane-design work. |
| D2 | **CORRECT as new development** | Prototype and no-advection-gap prediction are supported, PDE:123–125; INV:56. The direct cubic-tensor storage qualification is appropriate. |
| D3 | **CORRECT as re-scoped development** | Steady Bratu is available; time-dependent F-K requires new assets. PDE:173–174. |
| D4 | **CORRECT as new development** | Off-mesh NS transfer remains unimplemented and bank quality is a known limitation, INV:25,54. Its design must also preserve the co-moving/translation treatment. |
| D5 | **NEEDS-RESTATEMENT** | The Dirichlet-box correction is right, R/report.md:47–55. But **mixed boundary conditions do not automatically eliminate fast transforms**. Choose and demonstrate a specific operator/domain lacking the relevant separable transform. “Non-box or mixed” is insufficient as a selection rule. |
| E1 | **CORRECT, implementation required** | Single-draw coverage and independent-family controls are right. Existing 2D dispatch hard-codes seed zero, Q2/qcore.py:85–93; expose draw/seed configuration before running repetitions. |
| E2 | **CORRECT** | Repeating training, ordering and selection addresses recipe uncertainty. W/report.md:111–113 identifies the present W1024 evidence as single-seed and confounded across bank recipes. |
| E5 | **CORRECT** | Within-device pairing is appropriate. Keep identical query contracts and scored outputs across hardware, following S/DESIGN.md:105–111,358–362. |
| F1 | **NEEDS-RESTATEMENT** | Explicitly freeze bank/rotation hashes, M/test modes, initialization, metric/output times, cohort membership, lane roster/exclusions and implementation versions. R/DESIGN.md:652–658 requires an authoritative roster with explicit exclusions. |
| F2 | **CORRECT as a gated future operation** | Counts and missing-tool/custody conditions are correct. R/report.md:237,245–258; R/DESIGN.md:652–658. Eligibility must remain per case-kind after opening. |

**6. Budget arithmetic and day estimates**

The arithmetic in V2:176 is correct:

| Calculation | Exact result |
|---|---:|
| Lower total: 0 + 60 + 60 + 120 + 30 | 270 GPU-h |
| Upper total: 20 + 110 + 120 + 220 + 50 | 520 GPU-h |
| Add 25% | 337.5–650 GPU-h |
| Lower occupancy, four GPUs | 337.5 / 96 = **3.516 days** |
| Upper occupancy, three GPUs | 650 / 72 = **9.028 days** |

Thus “about 340–650 GPU-h” and “roughly 3.5–9 days” are arithmetically sound.

**Budget feasibility: NEEDS-RESTATEMENT.**

- Zero GPU-hours is a mathematical lower endpoint for a cap, not an estimate for completing priority 1. Charging its full 20 hours gives **290–520 base**, or **362.5–650 with contingency**.
- Clarify whether the pilot’s 20-hour hard cap **includes** contingency. Applying 25% to that row separately would make it 25 hours.
- These totals exclude D2–D5, as stated. They also do not clearly price new references for B7/B8 or distinguish incremental cost from the existing speed3d campaign.
- The occupancy estimate assumes continuously busy GPUs. The dependency graph and implementation/audit work provide no calendar-time guarantee.
- Historical timings cannot certify the new matrix: W/report.md:16,23 records 2.28 hours for the 1024² 2D grid and 2.64 hours for the 3D dial, under their own configurations.

**7. Can priority 1 actually run with existing code/assets in ≤20 GPU-h?**

**Not established. As an unchanged-driver, existing-code-only tranche: INFEASIBLE. As a bounded pilot after modest implementation: plausible, but unpriced.** I found no evidence proving it must exceed 20 hours; equally, the stated cap is not evidence that all listed work fits.

| Tranche item | Exact reusable code/assets | Missing work |
|---|---|---|
| Development manifest | C/make_e0_registry.py:16–55; C/score.py:34–43; existing checkpoints, rotations and saved `checks/` results | Consolidated configuration-level qualification/reuse manifest; fix the 3D S contract. The registry generator is not itself a complete cross-lane freeze manifest. |
| 2D kernel panel | Q2/qcore.py:141–146,185–227; `tested_value`/`tested_jac` used at C/t2core.py:256 | New benchmark harness for separate residual, Jacobian, normal matrix and factorization timings; fixed-(M,m) controls; uncertainty/drift controls. |
| 3D kernel panel | Q3/offmesh.py:73–92,129–134; C/c3common.py:95–100; S/s3modes.py:231–314 | Arbitrary-r/M/m harness. Existing speed3d profile combines normal formation and Cholesky and profiles selected trajectories, not this factorial prefix matrix. |
| 2D mesh-cost panel | C/c2grid.py; C/c2common.py:98–159; C/t2core.py:252–274; W/w2d.py | Same-job cross-mesh orchestration and separately callable initialization/evolution/decode. `t2core.make_query` currently returns only the fused jitted query. |
| Q1/Q2 and E0 scoring | C/c2bud.py; C/c2common.py:32–36; C/score.py; C/make_e0_registry.py | Frozen arms/rules/case counts, exact bars, ported controls, and resource-limited qualification schedule. Current run setup still loads historical references through C/c2common.py:72–95; E0 rescoring is a separate path. |
| Fixed-r quadrature slice | W/w2d.py; W/w3d.py:14–22; C/c2bud.py | Specify the two middle ranks, three rules, schemes, populations and own-state indicator computation. W/w3d.py’s existing indicator population is the converged-rule trajectory, not automatically each candidate’s own reached states. |
| Memory/offline inventory | W/w2d.py:101–129,245–248; C/c2grid.py:157–159,336–338; W/report.md:111 | Per-panel peak attribution and lifetime/isolation policy; training-job versus bank-fit cost distinction. |
| Speed3d reuse | S/s3run.py, S/s3modes.py, `select_s3.py`, `resources.py` | Audited accepted outputs and an explicit contract mapping. It cannot silently supply full-field hybrid measurements from scoring-lattice mesh-free results. |

Two concrete feasibility issues remain.

First, **the 4096² bank is large even before quadrature qualification**. C/c2common.py:105–108 appends the full R512 edge, and Q2/qcore.py:166–182 retains all resulting blocks. The stored bank alone is

\[
4095^2 \times 512 \times 8
= 63.969\ \text{GiB}.
\]

That excludes temporary evaluations, operators, reference arrays, full outputs and simultaneously live rules. Blocking the construction does not eliminate the final retained bank. Specify an appropriate GPU and a measured peak-memory pilot.

Second, **the run count is unspecified**. If the R512 diagnostic is included at all three meshes and both named schemes run on all 38 development/validation cases, the base accuracy matrix already contains

\[
4\ \text{ranks}\times3\ \text{meshes}\times2\ \text{schemes}\times38
=912
\]

trajectories—before tight/tighter solves, finer rules, warmups, A–B–A repetitions, controls or kernel panels. This is an illustrative count, not a runtime prediction.

A credible ≤20-hour design needs an explicit inventory and smoke-derived cost projection, including setup, compilation, certification and burn-ins. Speed3d already provides that pattern in S/resources.py:1–12. It also needs a deterministic stop/defer order that says which outputs may remain incomplete when the cap is reached.

**8. New problems or misleading statements in v2**

1. **Blanket 3D S matching.** V2 turns a dimension-specific rule into a global contract. The correct mismatch is already documented in C/make_e0_registry.py:49–51. This was also missed by audit 1.
2. **Qualification is confused with accuracy utility.** R512 is treated as diagnostic while R256 is called qualified without identifying its fallback. Numerical qualification and monotone-r improvement are separate axes.
3. **§0 overstates completion of the timing specification.** Repeats and intervals are promised, not defined.
4. **Q1’s “any departure” rule lacks a tolerance.** Literal equality conflicts with the existing approximate-solver qualification.
5. **Fixed-M invariance conflicts with mesh-dependent shell completion.** This matters even when nominal M/r stays four.
6. **Speed3d reuse changes the query being measured.** Full-field hybrid and mesh-free scoring-lattice paths cannot be substituted without labels.
7. **The priority-1 implementation omits the explicit fixed-(M,m) panel and Q3**, despite the broader plan claiming these controls.
8. **B7/B8 introduce reference and observation work not covered by the “existing assets” framing or explicit budget lines.**
9. **The training-cost label conflates fit time and total job time.**
10. **The freeze graph leaves optional new-PDE lanes outside the opening dependency without an explicit exclusion decision.**

**Overall verdict: NEEDS-RESTATEMENT. Ranked remaining blockers:**

1. **Correct the scoring and qualification contract:** physical S timestep, per-case-kind eligibility, configuration-level qualification, and numerical solver-distance thresholds.
2. **Make priority 1 concrete and resource-bounded:** exact matrix, drivers/adaptations, memory pilot, measured cost certificate and cap stop policy.
3. **Define the scaling estimands:** fixed actual M/test modes, fixed-(M,m) panel, query/output contract, repetitions and uncertainty-based decisions.
4. **Reconcile speed3d reuse with Group Q:** identify which audited arms satisfy which gates and which query contract.
5. **Freeze falsifiable accuracy/competition claims:** independent rank qualification, explicit dominance/matching rules and finite-ladder limits.
6. **Complete the freeze roster and dependency accounting:** optional-lane exclusions, longer-horizon/sampling references, and incremental versus reused costs.