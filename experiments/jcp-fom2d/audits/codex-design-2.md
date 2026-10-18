**NOT READY.** A1 resolves several round-1 defects, but C2 selection/inference and the K2 dominance shortcut remain insufficiently specified. The draft C1 code also retains contradictions with A1.

Read-only audit: no files changed, no cluster commands or jobs, no arrays copied, and no scratch created.

Evidence abbreviations:

- **D** = [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/DESIGN.md).
- **B** = [lane brief](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/jcp-fom2d.md).
- **V3** = [PLAN-v3-amendments.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/PLAN-v3-amendments.md).
- **R** = E0’s `experiments/jcp-references` in `2026-10-06-jcp-references`.
- Code filenames below refer to this lane’s `experiments/jcp-fom2d/`.

**Disposition of the six round-1 blocking fixes**

| Round-1 blocking fix | A1 verdict | Evidence and remaining issue |
|---|---|---|
| 1. Repair C2 inference | **NEEDS-RESTATEMENT** | D:276–284 correctly aligns the statistic and bootstrap, specifies common-valid cases, and introduces subject-level vetoes. Fresh stage-2 measurements avoid reusing individual selecting measurements. However, union selection, selection-block validity, and the interpretation of case-bootstrap inference after selecting on those same cases remain unresolved. |
| 2. Screen using full-field costs | **NEEDS-RESTATEMENT** | D:269–275 fixes the output-contract mismatch. Screening two cases once and confirming only three candidates does not establish “cheapest among … members of the tested ladder” at D:235. Describe the result as the comparator selected by this screening procedure, or establish a stronger minimum-cost claim. |
| 3. Freeze populations/identities; restore K2 | **NEEDS-RESTATEMENT** | D:290–297 fixes configuration correspondence and separates crossover populations from H-comp populations. D:298–303 expands K2’s intended universe, but its dominance shortcut is not established for screened comparators or confidence bounds. |
| 4. Repair controls | **CORRECT at design level, with qualifications** | D:305–332 repairs K-syn and gives a valid K-false construction for \(n\ge3\). Fixtures and execution stages are now specified. Explicitly enforce the eligible-population minimum; distinguish unchanged raw median from the adjusted median. Draft code still implements the old controls. |
| 5. Same-mesh PS/union and FRONT fallback | **NEEDS-RESTATEMENT** | D:260–266 declares a meaningful same-resolution convention and available-family union. D:340–341 correctly makes fallback partial evidence. The high-resolution PS exclusion is transparently labelled, but its cost rationale is not an adequate demonstration of unaffordability or irrelevance. |
| 6. Boundaries, PS anchors, resources | **NEEDS-RESTATEMENT** | D:239–258 and 334–338 add the requested policies and combined-workload certificate. Remaining ambiguities concern extension selection, anchor scheduling, recovery qualification, and the omitted cheapest-admissible-member boundary flag. |

**Disposition of every round-1 NEEDS-RESTATEMENT/WRONG finding**

| Round-1 item | A1 verdict | Assessment |
|---|---|---|
| 1: Same-mesh PS and union | **NEEDS-RESTATEMENT** | Convention and restricted union supplied, D:260–266; excluded cells remain an incomplete part of B:28–29’s requested crossover deliverable. |
| 1: “Cheapest” unsupported | **NEEDS-RESTATEMENT** | Full-field screening fixes the contract mismatch, D:269–275, but not exhaustive minimum-cost identification. |
| 1/6: K2 narrowed to front members | **NEEDS-RESTATEMENT** | Intended universe expanded, D:298–303; coverage and dominance inference need repair below. |
| 1/8: FRONT fallback changes deliverable | **CORRECT** | D:340–341 explicitly leaves FRONT C2/H-comp/K2 outstanding. |
| 3: Explicit \(N/L\) interpolation factor | **NEEDS-RESTATEMENT** | D:66 still only says “same sine interpolation”; A1 never states the general factor. Implementation correctly uses \(M/L\), `fomrun2.py`:58–67. Minor documentation omission. |
| 3: Prepared-input fairness/scope | **NEEDS-RESTATEMENT** | D:71–72 preserves asymmetric preparation exclusion, but A1 does not explicitly name the resulting statistic “prepared-input latency.” This remains a defensible workload, provided it is labelled. |
| 3: PS “has no mesh” justification | **CORRECT replacement convention** | D:260–261 defines equal state resolution. The exclusion justification is assessed separately below. |
| 4: Coarse/large-step ladder omissions | **CORRECT** | D:243–245 widens all relevant families. This is broader coverage, not proof of an optimum. |
| 4: Bounded extension | **NEEDS-RESTATEMENT** | D:253–256 bounds rounds and resolution, but does not fully determine which added resolution is chosen when the trigger is a time-step edge or several targets trigger it. |
| 4: Calibration versus unrestricted tolerance optimization | **CORRECT** | D:235–236 explicitly distinguishes them. |
| 4: Production recovery | **NEEDS-RESTATEMENT** | D:257–258 adds one tighter retry, but must require that recovered level to pass calibration agreement as well as production convergence. “Tighter” alone does not prove this. |
| 4: PS stability qualification | **CORRECT distinction; NEEDS-RESTATEMENT details** | D:247–252 explicitly makes the label diagnostic. The lattice proxy is not generally a guaranteed lower bound on a maximum over a different, non-nested collocation grid. Anchor scheduling is also circular as written. |
| 5: Lattice-cost top-three selection | **CORRECT contract repair** | D:269–272 screens full fields. Remaining minimum-cost wording is unresolved. |
| 5: Cross-mesh population changes | **CORRECT policy** | D:293–297 requires common-population rematching. Both population-specific candidate selections must actually enter C2. |
| 5: Cross-mesh configuration identity | **CORRECT** | D:290–292 defines identity and missing-qualification labels. |
| 5: Cheapest-member boundary flag | **NEEDS-RESTATEMENT** | D:253–256 addresses coordinate edges only. The inherited cheapest-admissible-member flag remains absent from A1, although `match2.py`:195–201 implements it. |
| 5: “Most accurate” means lowest worst error | **NEEDS-RESTATEMENT** | D:117 already contains the parenthetical definition, but A1 leaves the broad label unchanged. Use “lowest-worst-error unmatched comparator.” |
| 6: Point estimate/bootstrap mismatch | **CORRECT** | D:276–281 uses the same ratio-of-medians statistic in each paired draw. |
| 6: Joint resampling/common-valid policy | **CORRECT** | D:278–281 explicitly pairs IDs and requires at least five common-valid cases. |
| 6: Selection/confirmation reuse | **NEEDS-RESTATEMENT** | D:271–275 separates timing measurements, but does not completely settle union selection or case-population inference after selection on dev6. |
| 6: Identity-veto scope | **CORRECT** | D:282–284 vetoes the whole subject, and the whole panel if A fails. |
| 6: Crossover endpoints/reversals | **CORRECT** | D:285–287 explicitly reports both endpoint classifications and reversals. |
| 7: G-PS1 fixture | **CORRECT design specification** | D:320 fixes cases, resolution and steps. Discrimination still must be demonstrated in smoke; it is not established merely by specifying the fixture. |
| 7: G-PS2 explicit parameters | **CORRECT** | D:321–322 gives the stored run, step, IC quadrature and synthesis convention. |
| 7: G-CT coverage | **CORRECT** | D:323–324 covers all three resolution relations and 4096² output. |
| 7: K-loose rejecting predicate | **CORRECT design specification** | D:314–317 specifically requires agreement failure. The smaller-resolution CPU observation is supporting evidence, not proof for the production fixture. |
| 7: K-syn zero-cost confound | **CORRECT** | D:311–313 uses positive cost and requires rejection reasons. |
| 7: K-false construction | **CORRECT for \(n\ge3\)** | D:306–310; proof and small-population limitation below. |
| 7: K-swap fixture and interpretation | **CORRECT** | D:318–319 freezes pair/permutation and calls it sensitivity, not metadata validation. |
| 7: T-dbl/T-fault dispatch | **CORRECT** | D:326–327 requires separately dispatched, synchronized executions and verified fault separation. |
| 7: T-id scope | **CORRECT** | D:328 invokes subject-level veto. |
| 7: Control execution stage | **CORRECT** | D:329–332 distinguishes ladder, offline, and C2 smoke controls. |
| 8: Complete resource accounting | **NEEDS-RESTATEMENT, small omission** | D:334–338 substantially fixes it, but should explicitly include input preparation, anchors/extensions, recovery, and both population-specific selections. |
| 8: Combined memory/object lifetimes | **CORRECT design requirement** | D:335–336 specifies the combined workload, shared bank and cache release. Actual feasibility remains smoke-gated. |
| 8: Deferred versus unmatched | **CORRECT** | D:338 explicitly distinguishes them. |

The inherited-fact clarifications are also **CORRECT**: D:234–237 distinguishes P1 evolution times and combined H5’s rematching of existing timings. These agree with [P1 report:183](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/report.md:183) and [combined report:129](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/report.md:129).

**Adversarial checks of A1**

**1. C2 is improved, but not yet fully defined or selection-bias-free.**

The corrected statistic is coherent:

\[
S_{Y,X}
=
\frac{\operatorname{median}_{i\in I}r_{Y,i}}
     {\operatorname{median}_{i\in I}r_{X,i}},
\]

with the same common-valid set \(I\) and paired resampled IDs. D:276–281 correctly implements that definition.

Four gaps remain:

- **Union selection:** D:272 selects per family, while inherited D:151–152 selects the cheapest confirmed family comparator. Explicitly select and freeze the union winner using stage 1. Choosing the union minimum from stage 2 reintroduces the very selection bias A1 is meant to remove.
- **Selection-block validity:** stage 1 has one repetition per case, but the only new validity threshold is “at least two of three” for stage 2. Define stage-1 missing-case/retry rules, a common selection population across candidates, deterministic ties, and the outcome when no valid selection exists.
- **Fresh timings are not fresh cases:** stage 1 and stage 2 both use dev6. Independent timing repetitions address selection on timing noise; they do not remove selection on case-specific workload advantages. A case-bootstrap interval therefore cannot automatically be presented as selection-adjusted population inference. Declare the conclusion conditional on the fixed dev6 workload, or use independent inference cases/selection-aware population inference.
- **Anchor dependency:** D:251–252 requests anchors for selected PS comparators, but D:269–272 places anchors in stage 0 before stage 1 selects them. Run anchors for a predetermined candidate set, or place selected-comparator anchors after selection with no resulting reselection.

Also freeze **selected comparator IDs and union winners**, not merely “candidate sets,” before stage 2.

**2. K2’s inequality direction is correct; its proposed operational shortcut is not established.**

Let

\[
U_{\mathrm{med}}(T)=\operatorname{median}(e_T+\eta),
\qquad
U_{\max}(T)=\max(e_T+\eta).
\]

If both upper endpoints for \(Y\) are no larger than those for \(X\), then every FOM matching \(Y\) also matches \(X\). Thus

\[
\mathcal M_Y\subseteq\mathcal M_X.
\]

For **exact minimum costs over those sets**, and positive ROM costs with \(c_X\ge c_Y\),

\[
\frac{\min_{F\in\mathcal M_X}c_F}{c_X}
\le
\frac{\min_{F\in\mathcal M_Y}c_F}{c_Y}.
\]

So the direction at D:301–302 is **correct**, including the use of the \(+\eta\) target endpoints.

But A1 selects separate top-three shortlists; those selected costs need not obey the minimum-cost inequality. For example, a screened comparator costing 10 for \(X\), versus 1 for \(Y\), with ROM costs 2 and 1, yields selected speed-ups 5 and 1 despite the mathematical nesting of admissible sets.

Furthermore:

- Observed cost dominance does not automatically imply dominance of bootstrap upper bounds.
- Pair-specific valid-case sets can differ.
- A FRONT file containing only nondominated members cannot itself certify that every other qualified configuration has been enumerated.

**Fix:** require a complete qualification manifest and directly time all qualified configurations, or define a common-population, uncertainty-aware witness argument using a shared matching FOM and justified ROM-cost ordering. Missing coverage must remain inconclusive, as D:303 already requires. V3:85–87 remains the governing quantifier.

**3. K-false now works—including tied maxima—but explain the right median.**

For the worst-failure fixture,

\[
e'_j-\eta_j=U_T+\delta>U_T,
\]

so worst matching certainly fails.

Choosing \(j\) with maximal **raw** target error ensures the raw median remains unchanged for \(n\ge3\), even when several cases share that maximum. The uncertainty-adjusted median can change, however.

Its matching result still passes: for every \(i\ne j\),

\[
e_{T,i}-\eta_i\le e_{T,j}\le e_{T,j}+\eta_j.
\]

Both the original target upper endpoint at \(j\) and the replacement lower endpoint sit above all remaining lower endpoints. For \(n\ge3\), the median excludes that single top endpoint; monotonicity of the other endpoints gives the required median inequality.

For the median-failure fixture,

\[
e'-\eta=e_T+\eta+\delta,
\]

so both median and maximum exceed the target upper endpoints by \(\delta\).

**Small-\(n\) caveat:** with \(n=2\), \(e_T=(1,1)\), \(\eta=0\), raising one value makes the median exceed 1. Enforce \(n\ge3\) **after eligibility/intersection**, and register a deterministic tie-break for \(j\). Do not claim the construction handles every population size.

**4. High-resolution same-resolution PS exclusion is not adequately justified as written.**

D:262–264 turns E0’s conservative stability heuristic into a required step restriction, despite D:250 explicitly not requiring that heuristic for this lane’s PS admissibility.

The cited E0 46-second record exists at [R/store/2d/val32/PS/c000_a0.json:21](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/store/2d/val32/PS/c000_a0.json:21). But [R/refjob.py:420–441](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:420) shows that timer includes IC preparation, quadrature/contour checks and the reference execution. It is not this lane’s warm fused prepared-input latency.

Consequently, the 20-second/20-minute projections are not demonstrated lower bounds. Nor is “far above every matched comparator” established before this lane runs.

A budget-based omission is permissible as a **declared partial deliverable**. Retain the unavailable-cell and restricted-union labels, remove unsupported comparative claims, and either supply an explicit cost model/pilot or state that these cells are deferred by scope rather than proven irrelevant.

**5. Extension policy is bounded in intent, but needs a deterministic definition.**

D:253–256 limits it to one round and caps coarsening. Specify:

- one global extension round after all required target populations are known;
- the precise added resolution per family, including when only the time-step edge triggers;
- deduplication when “next coarser” is already tested but inadmissible;
- inclusion of extension results in calibration, screening and frozen selection;
- whether P1 fallback targets trigger the same rule.

Keep the cheapest-admissible-member flag separate from coordinate-edge flags. A boundary label need not imply further unbounded searching.

**6. Crossover versus H-comp populations are now conceptually consistent.**

D:293–297 correctly makes crossover use a common cross-mesh population and H-comp use each mesh’s published population.

The operational key must nevertheless include **population identity**. At a given mesh, the crossover and H-comp versions of the same ROM can have different matched FOM sets. Both must receive stage-0 screening, stage-1 selection and stage-2 inference. Also define the intersection when an identity is unqualified or absent at one mesh, and refuse matching on an empty or undersized eligible intersection.

**Design-level glance at the drafted C1 code**

These are contradictions to resolve before the separate code audit, not an exhaustive code review.

| File | Finding |
|---|---|
| [fomrun2.py:130](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/fomrun2.py:130), :144–146 | FD solve and output mapping are separate dispatched compiled calls. D:58 and :136 promise one jitted query. Either fuse them or amend and justify the timed-program contract. |
| [psjax2d.py:48](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/psjax2d.py:48), [fomrun2.py:153](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/fomrun2.py:153) | PS discards nonlinear-stage maxima and validates output finiteness only. D:93 also requires finite running maxima. The diagnostic lattice label in `ladder.py`:208–212 does not supply that evidence. |
| [gates.py:38](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/gates.py:38) | G-FD1 checks source hashes, not the CPU execution equality specified at D:167. Hash provenance is useful but is a different gate. |
| [ladder.py:265](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/ladder.py:265), [match2.py:113](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/match2.py:113) | Production recovery changes the level, but offline validation requires the original calibrated chosen level. Recovered records are therefore rejected. Recovery agreement qualification is also missing. |
| [ladder.py:291](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/ladder.py:291) | K-loose `fires` accepts either convergence failure or distance failure; A1 requires the agreement predicate specifically to fail. |
| [match2.py:198](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/match2.py:198) | Same-mesh matching still excludes PS, contrary to A1.2 at 256². |
| [match2.py:215](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/match2.py:215) | K-false still starts from selected real FOMs and uses old perturbations. Its pass aggregation omits the median-still-matches requirement and can pass vacuously when no family has a match. |
| [match2.py:165](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/match2.py:165) | K-syn uses positive cost correctly, but the final control checks rejection without enforcing A1’s intended rejection reasons/admissibility distinction. |
| [targets.py:61](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/targets.py:61) | FRONT loading is explicitly unimplemented. That is acceptable for a draft, but population-specific targets and complete K2 inventory remain required interfaces. |

**Ranked remaining blocking fixes**

1. **Complete C2 selection and inference:** freeze family and union selections before inference; specify selection-block validity/ties; resolve the same-case selection/bootstrap interpretation; schedule PS anchors without circularity.
2. **Repair K2 coverage and dominance inference:** require the complete qualified inventory; directly evaluate every setting or supply a valid common-population witness argument with uncertainty.
3. **Align minimum-cost claims and population-specific execution:** distinguish screening-selected comparators from established ladder minima; independently select crossover and H-comp comparators where populations differ.
4. **Resolve high-resolution PS scope honestly:** substantiate the resource exclusion or explicitly defer those crossover cells without claiming they are necessarily slower.
5. **Finish deterministic extension/recovery rules and reconcile C1 code:** especially calibration after recovery, K-false/K-syn assertions, PS validity evidence, and the promised fused FD query.

The paired estimand, K-false algebra for \(n\ge3\), and direction of the match-set dominance inequality do **not** require reversal. Their surrounding inference and implementation contracts still require amendment.