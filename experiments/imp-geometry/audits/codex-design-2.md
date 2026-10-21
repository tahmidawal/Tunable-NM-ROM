**OVERALL: NOT-READY.** A1 resolves several important issues, but reference eligibility, qualification, attribution, controls, and A2’s evidence claims still block production. This audit was read-only; no files changed and no jobs were submitted.

The ten verdicts below follow the **numbered required-fixes list at the end of the round-1 audit**, rather than its eleven discussion items.

| # | Round-1 required fix | Verdict | Round-2 finding |
|---|---|---|---|
| 1 | Competitor minimum, aggregation, attribution, confirmation | **WRONG** | R-PODG is now included; accuracy, qualification and speed confirmation are added. But attribution does not specify validation versus confirmation ratios, box censoring is incorrectly converted into zero speed-up, and the outcome rules omit some mixed/failure cases. |
| 2 | Physical-interaction diagnostic | **WRONG** | A diagnostic exists, but it combines changed initial conditions with changed boundaries. Geometry selection used its validation values, and the adopted aligned geometry lacks matching diagnostic evidence. |
| 3 | Corner-aware references; no wins from unqualified fallback | **WRONG** | Positive-time checks, temporal anchors and cross-check escalation improve matters. However, A1.4 still permits a physical WIN with eight ineligible references using an unjustified last-grid-difference allowance. |
| 4 | Stronger and fully specified competitors | **CORRECT** | A1.5 supplies a second SEM solver, mesh/integration/startup details, MG and spectral ladders, ECSW fitting details, and independent competitor stepper tuning. This fixes the design specification; actual competitiveness remains subject to smoke and code audits. |
| 5 | Reduced-space/operator mathematics | **CORRECT** | A1.6 freezes the basis transformation, uses rule-specific mass matrices and projections, acknowledges test-space saturation, adds an independent residual diagnostic, and handles nonsymmetric diffusion explicitly. |
| 6 | Scoring compatibility, search and timing | **WRONG** | Grid compatibility, separate statistics, raw timings and replay identity are improved. The new size-search stopping justification is invalid, and finalist selection still needs reconciliation with the new timing statistic. |
| 7 | Complete Q1/Q2 | **WRONG** | ETDRK4 finiteness/growth checks do not implement the requested stage/update-consistency certificate. Linearised CN still lacks an explicit every-step certificate. Bank Q2 remains limited to a validation subset, and the ECSW exception has not been established as an approved narrowing of the binding contract. |
| 8 | Detectable fault controls | **WRONG** | Several controls remain contingent. The supplied evidence actually contains failures of MF-FD and MF-MATCH-real. |
| 9 | Full-size cost certificates and fair deferral | **NEEDS-RESTATEMENT** | A1.10 correctly makes an audited cost certificate a prerequisite and protects competitor strength. That prerequisite remains outstanding: no `runs/cost/projection.json` appears in the supplied files. The design can be conditionally acceptable on budget; production cannot yet be released. |
| 10 | Finer-grid, feature, decay and missing-evidence safeguards | **WRONG** | Decay and zero-field safeguards are good additions. Finer-grid scoring is optional for some FD candidates, the feature diagnostic is confounded, and the verdict logic is not exhaustive. |

**A. Geometry selection and pre-registration**

Choosing a problem using a physical diagnostic before comparative performance analysis is legitimate **developmental design selection**. It does not invalidate a subsequently frozen performance experiment. A1.2 explicitly anticipated another redesign if interaction was weak.

It is nevertheless not an untouched, outcome-independent registration:

- Variants and horizons were examined on `val[0:16]`, and selection explicitly sought passage of the interaction threshold.
- Those cases are reused in the proposed val32 interaction gate. Passing that gate is therefore partly selected evidence, not independent confirmation.
- Geometry selection also changes the accepted descriptor distribution and the initial conditions—quantities that can affect compressibility and solver cost even without examining speed-ups.

A2 should call this **diagnostic-guided problem development**, disclose all tried variants, and freeze the final problem before further performance selection. Confirm the interaction claim on untouched cases, naturally the existing test-once cohort, with no subsequent redesign.

The chronology is not documented accurately enough. [A2’s heading and explanation](experiments/imp-geometry/DESIGN.md:512) say “no ROM or competitor result exists” and “No solver, ROM or timing result entered this choice.” Yet the supplied smoke contains:

- FD/RK4 errors and execution times in [the ladder](experiments/imp-geometry/runs/smoke/smoke2-cellL/ladder.jsonl);
- bank reconstruction measurements in the smoke log;
- real ROM rollout and qualification-control measurements in [cell controls](experiments/imp-geometry/runs/smoke/smoke2-cellL/controls_L.json).

Their existence does **not** prove they influenced the choice. It does disprove the categorical nonexistence claim. Replace it with an accurate account of what had been generated, what was inspected, and what selection used. The interaction diagnostic itself is also a solver result; distinguish it from comparative performance results.

**B. Cutoff, rejection zone and corner move**

The cutoff is a defensible way to construct compatible initial data. Its fourth-order vanishing removes the original boundary-value mismatch; the stated limited regularity at the L corner is appropriately acknowledged. Applying the same function to every method within a geometry is fair.

It does **not**, however, make the problem neutral with respect to approximation method.

The cutoff introduces a prescribed transition scale, \(\delta=0.05\), comparable to the Gaussian widths. The rejection clearance is only \(0.03\). For a locally distance-like cutoff argument, at that clearance,
\[
1-\exp[-(0.03/0.05)^4]\approx0.122.
\]
Thus accepted centres can lie where the bump is strongly attenuated. The resulting initial data can contain substantial boundary-shaped gradients and finite-regularity corner structure. These can change high-order convergence and learned-basis compressibility. The direction of the net performance effect is unmeasured; it is not valid to assume either advantage or disadvantage.

More importantly, excluding \(t=0\) from \(D_{\rm geo}\) does not eliminate the initial-condition confound. An initial difference propagates to positive times.

A read-only NumPy evaluation of the specified ICs on the \(1/128\) lattice, using the same first sixteen validation descriptors, found:

| Geometry variant; L wide S1 | Median initial relative difference from box |
|---|---:|
| v4, corner 0.35 | 0.1400 |
| v4a, corner 22/64 | 0.1853 |

These are initial-function calculations, not new PDE runs. For comparison, the recorded v4 positive-time interaction median for that cell is 0.13293. This does not establish that its entire effect comes from initialization; it establishes that the current diagnostic cannot separate the two causes.

For attribution, add a paired box evolution initialized with the **same non-box IC extended by zero into the removed region**. Compare that evolution on the common domain. Retain the existing geometry-specific box comparison as a combined problem contrast. Also report initial mismatch and cutoff-resolution diagnostics for the high-order arm.

The shared rejection is sensible for ensuring admissible centres and pairing methods, but it changes “wide family” into a **conditional distribution on a nonrectangular support**. It is no longer independent uniform coordinates over a \(W=0.4\) square. The narrow family also moves to a different location, so narrow-versus-wide is not a width-only comparison. State those facts explicitly and report acceptance rates and descriptor hashes.

The corner alignment is mathematically correct for the fine FD ladder: \(22n/64\) is integral when \(n\) is a multiple of 64. Its documentation and evidence are not yet correct:

- A2 says the adopted variant is `geom.VARIANTS['v4']`; code assigns 0.35 to `v4` and \(22/64\) to **`v4a`**.
- A2’s table gives exclusion threshold 0.32; its later paragraph and code give **0.31375**.
- All eight recorded v4 interaction medians exceed 0.05, but there is no v4a interaction JSON.
- Changing rejection can change descriptors in every geometry, including H and B. In my check the first sixteen validation rows happened to remain identical, but the L IC differences changed substantially.

The old evidence remains evidence for v4 only. Run the frozen v4a diagnostic and relevant geometry controls before treating the adopted problem as validated. The supplied L control’s nearly zero observed FD orders are an unresolved failure, not evidence that alignment has already fixed it.

**C. Win and attribution rules**

A1 substantially improves the original rules, but they remain ambiguous enough to change conclusions:

1. **Reference fallback still supports physical wins.** A1.4 allows 24 qualified cases and eight ineligible cases, assigning the latter a last-grid difference. That difference is not a justified error bound: a slowly converging or biased sequence can have a small final difference. Require qualified references for every case entering a worst-case physical claim, or independently justified uncertainty bounds for the others. Otherwise label the target UNRESOLVED. Apply the same policy to test confirmation.

2. **Attribution’s evidence split is undefined.** Specify whether \(S_{\rm box}\) and \(S_{\rm geom}\) are validation or test ratios. A confirmed attributed win should use a frozen box control, renewed accuracy/qualification checks, and a defined confirmation comparison.

3. **Censored box bank does not imply \(S_{\rm box}=0\).** It means no matched bank configuration was found in its ladder. Report that categorical result; attribution remains unestablished unless a distinct, explicitly justified categorical rule is registered.

4. **The outcomes are not exhaustive.** For example, one level can win and the other lose; or a validation win can fail test confirmation. Neither necessarily satisfies “at every τ … \(S<2\),” nor is every such outcome missing evidence. Write a complete decision table, preserving “not confirmed” separately from reference failure and measured loss.

5. **The timing definitions differ.** Original §4 selects by \(t^*\) and defines a ratio of selected times; A1.7 defines a median of paired ratios. These are different statistics. Specify finalist selection, competitor minimization, tie-breaking and the reported ratio explicitly.

6. **The size-stop rationale is false.** Two sizes meeting 1% do not establish that larger sizes cannot win with fewer steps. Running the step ladder at smaller sizes does not test that possibility at omitted larger sizes. Either run the registered size ladder or call this an explicitly limited search heuristic and remove the claimed coverage guarantee.

The factor-two attribution threshold is now honestly labelled a lane choice. Multiple targets are also acceptable if frozen, all reported, and aggregated exactly as registered.

**D. Other production blockers**

[Qualification A1.8](experiments/imp-geometry/DESIGN.md:462) needs another amendment. A finite ETD update with bounded growth can still use incorrect coefficients or an incorrect update. Specify the every-step algebraic consistency checks. Linearised CN needs its own residual certificate. Define Q2 rollout coverage on every required case and the small-denominator policy precisely. Merely saying an ungated ECSW exception “favours the competitor” does not resolve its relationship to the binding qualification contract.

[Controls A1.9](experiments/imp-geometry/DESIGN.md:476) still overstates determinism. An arbitrary iterate perturbation need not violate the stationarity threshold; a largest tensor contribution need not produce the stipulated rollout difference; a coarse FD run can genuinely meet 1%. Indeed, the supplied real matcher control reports **`meets`**, with error approximately 0.007014 and allowance 0.001098.

Keep convergence demonstrations distinct from guaranteed fault-detection controls. For faults, establish an independently measured violation first, then require the checker to detect it. Do not repeatedly alter a valid coarse solver until it loses.

Finally, the new corner affects coarse MG geometry as well as fine-grid alignment. The production checks must establish that its coarse-grid treatment remains consistent. Cost certification must cover that solver, actual production sizes, preprocessing, qualification, compilation and confirmation—not extrapolate solely from the small smoke.

**OVERALL: NOT-READY. Required fixes — blocking only:**

1. Correct A2’s chronology and freeze one authoritative **v4a** specification, exclusion rule, descriptor-generation contract and hashes; disclose diagnostic-guided selection.
2. Separate boundary interaction from initial-condition changes, document the conditional family and cutoff effects, and validate the final geometry with an untouched confirmation gate.
3. Prohibit physical wins based on unjustified uncertainty allowances for ineligible reference cases; specify reference qualification for validation and confirmation.
4. Freeze an exhaustive win/withdraw/confirmation/attribution decision table, including box censoring and the exact timing/selection statistic.
5. Complete the binding Q1/Q2 contracts for every selectable arm and stepper, or obtain an explicit contract amendment.
6. Replace remaining contingent must-fail expectations with verified detectable faults; resolve the recorded FD and matcher-control failures and rerun affected controls on the final geometry.
7. Correct the size-search coverage claim and provide a common finer-lattice diagnostic for every matched method.
8. Supply and audit the required full-size time/memory cost certificate before production release.