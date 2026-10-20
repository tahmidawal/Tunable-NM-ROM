**Overall verdict: NEEDS-RESTATEMENT.** The design covers much of the intended campaign, and the factorial arithmetic is correct. The blockers are qualification, control logic, implementation scope and memory—not the basic experimental question.

Read-only audit completed: no files changed, no jobs submitted, no large arrays copied, and no scratch files created. Small CPU checks verified the factorial sets, mode-prefix nesting, shell counts and table-storage arithmetic.

References below use:

- **D** = [jcp-dims/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/DESIGN.md).
- **Q/** = files under `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/`.
- **S/** = files under `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/`.
- **P/** = governing documents under `/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/`.
- **V/** = `experiments/jcp-dims/vendor/` in this worktree.

1. **Coverage of the lane brief — NEEDS-RESTATEMENT**

   A3, the 2D A4 ladders, the first 16 exposed 3D candidates, paired qualified re-timing, both B3 stages and five random draws are represented. The explicit shell-breaking 3D \(M=r\) control is correctly labelled. Evidence: D:68–158, 204–209.

   Three coverage problems remain:

   - **3D GAL is excluded unilaterally.** D:64 says “LSPG only” is a scope statement, whereas `P/campaign/briefs/jcp-dims.md:30–33` requires GAL versus LSPG at each ratio without that exemption. Moreover, “No 3D Galerkin solver exists” is too strong: `S/c3common.py:160–163,180–188` already exposes an adaptive GAL path. It may need integration and certification, but that is different from inexistence.
   - **The defer order does not actually save the advertised resources.** D:371–373 drops only reporting for the 3D fixed-\(m\) panel. That preserves its computational cost.
   - **Dropping \(\kappa=8\) cells conflicts with the never-dropped factorial.** D:373–375 makes all Gauss ladders of \(\mathcal F_2\) core, yet permits dropping their \(\kappa=8\) cells. Some are shared: \(r=256,M=2048\) is also in the fixed-\(M\) representation panel.

   **Fix:** implement and certify the 3D GAL comparison, or obtain an explicit scope amendment. Define deferrals over unique configurations and their dependencies; shared core cells must remain. Distinguish computational deferral from omitted reporting.

2. **Q1/Q2 inheritance and next-rung replay — NEEDS-RESTATEMENT**

   **Using the next production rung as \(q^+\) is valid.** The predecessor itself runs \(q^+\) at production tolerances: `Q/qacc.py:215–226`; 3D uses the same solver specification: `Q/q3run.py:39–65`. There is no requirement to duplicate an identical invocation. The manifest must include the entire numerical configuration, including initialization, weighting, mode list, tolerances and draw.

   The accuracy-job Q1 bars and expanded 2D Q2 population are sound: D:171–185. However:

   - D:179–186 does not freeze precedence for invalid own/successor/check runs, missing populations, nonfinite indicators or zero denominators.
   - The predecessor already distinguishes these: `Q/analyze.py:550–587`; `Q/p1common.py:323–331` handles \(0/0\), zero denominator and nonfinite values.
   - “Ladder top” must mean **not qualified**, with a precise unavailable/unverified status. It cannot be treated as a failed quadrature experiment or a qualifying endpoint.
   - The 3D tight/tighter check must apply to each new rule and M configuration. A deployed-rule anchor does not automatically validate every new ladder rung; speed3d explicitly added reduced-rule finalist checks in `S/DESIGN.md:479–482`.
   - The timing exception in D:199–203 contradicts the in-job certificate requirement; see item 7.

   **Smolyak:** signed weights are valid for the tested integral. They must not be clipped or renormalized. Boundary removal means their sum need not equal one: `V/quad2d/qcore.py:83–85`, `V/quad2d/vendor/hari_quadrature.py:53–78`. In 3D, `S/s3common.py:118–123` unconditionally constructs a weighted Gram Cholesky; signed weights can make it indefinite even though the mesh-projection query does not need it.

   **Fix:** inherit an explicit qualification state machine, including exact populations and exceptional values. Separate mesh-path tables from optional mesh-free projection tables so an unused Smolyak Cholesky cannot invalidate the scientific experiment.

3. **Decision rules — NEEDS-RESTATEMENT**

   Several parts are correct:

   - The transition rule and range precedence are coherent for nonempty eligible populations: D:229–243.
   - All Shapley corner configurations are present in the stated factorial: D:75–83,245–251.
   - The symmetric 0.8 point-efficiency test cannot award both families “fewer points”: D:267–269.
   - The 3% winner rule matches `S/analyze.py:208–215`. The accuracy matching rule should explicitly preserve `S/analyze.py:48–65`: aggregate the **per-case endpoints**, rather than adding one aggregate uncertainty afterward.

   The unresolved cases are consequential:

   - **Empty populations:** zero eligible cases pass the written “resolved <50%” check vacuously. Require a nonempty population before any verdict.
   - **Missing versus failed ladders:** D:255–262 defines “not reached” only for an unrun ladder. A partially completed ladder cannot establish the smallest qualifying rung or stability. Missing, invalid and unverified evidence must remain distinct.
   - **Random-family summaries:** median/range are undefined when some draws have no finite \(m^*\). Report qualifying-draw counts and censored outcomes; do not silently omit failed draws.
   - **Width \(m^*\):** taking the larger of the two spec-specific minima is unsafe when qualification is non-monotone. That larger rung need not qualify for both specs. Select the smallest candidate in the intersection of their qualifying sets.
   - **Cross-family gate:** D:270–274 does not specify what happens if the finest rung or target is invalid or solver-unqualified. Numerical agreement with an invalid target is not a pass.
   - **Shapley:** freeze a common eligible-case population across all corners and separate the accuracy bootstrap from the six-case timing bootstrap. Specify whether \(m^*\) is frozen during bootstrap; otherwise selection uncertainty is ambiguous.
   - **H2q:** D:310–315 needs invalid-screening exclusion, comparator availability and invalid-confirmation precedence. As written, “fail iff \(S_C<1/0.6\)” overlaps “inconclusive (panel invalid).” The predecessor checks comparator selectability first: `S/analyze.py:218–229`.

   **Fix:** write executable decision tables for these cases before production. Use the predecessor’s matching equations and screening validity rules explicitly.

4. **Controls — WRONG as acceptance gates**

   **GAL=LSPG at \(M=r\) is an exact-root equivalence, not a production-output guarantee.** D:294–295 requires agreement within \(10^{-8}\) wherever both runs are merely valid.

   For square full-rank \(A\), \(Q\) is invertible, so \(Q^\mathsf T\mathcal R=0\) means \(\mathcal R=0\). Invertible \(D\) preserves those roots. But production LSPG minimizes a weighted residual and may terminate at a stationary non-root; invertible \(A\) does not establish a nonsingular nonlinear step Jacobian or a unique root.

   The implementation also chooses predictors using the respective residual norms and uses different stopping rules: `p1core.py:131–153,185–198`; `Q/p1common.py:40–44`. Those differences can produce valid trajectories more than \(10^{-8}\) apart.

   **Fix:** use an independently verified, well-conditioned root fixture with tight residual bounds and the same LMM. Treat production agreement as measured evidence, not a universal positive gate.

   Other control problems:

   - **Bad-vector fixture:** D:335 does not independently establish that these particular ROM rollouts exceed the cross-family distance threshold. A lattice confined to a shifted diagonal is geometrically bad, but that alone does not prove this rollout predicate fires. Freeze the bank, M, scheme, case, rule and independent reference evidence first. AM-7 requires a discriminating fixture: `P/campaign/P1-RESTATEMENT-AMENDMENTS.md:146–156`.
   - **Smolyak:** being registered as “must-fail” elsewhere is not evidence that the new Q1/Q2 predicate fails for the specified configuration: D:334.
   - **\(M=r\) control:** D:287–290 treats *any* failure to qualify as successful degradation. Missing \(q^{++}\), invalid anchors or broken certification would therefore count as supporting the scientific claim. Square-system stationarity ratios also have the root-related pathology documented in `Q/DESIGN.md:507–515`.
   - **3D M override:** reproducing the default M cannot detect an implementation that ignores the override: D:331.
   - **Cross-GPU reproduction:** the \(10^{-10}\) threshold is a legitimate proposed gate, but its success is not established merely by allowing that tolerance. Failure needs diagnosis, not automatic attribution to a broken solver.

   **Fix:** separate classifier fixtures from scientific outcomes. A valid \(M=r\) result that does not degrade must remain an adverse result. If the brief requires blocking B6, label that outcome honestly; never “repair” it by adjusting the criterion. Add nondefault-M shape, mode-hash and independently evaluated residual checks.

5. **Factorial arithmetic and 2D rule names — CORRECT, with manifest clarifications**

   Independent enumeration confirms:

   | Quantity | Count |
   |---|---:|
   | R128 / R256 / R384 / R512 M sets | 7 / 6 / 8 / 5 |
   | Total \((r,M)\) cells | 26 |
   | LSPG-CN-R, eight Gauss rungs | 208 |
   | GAL-CN | 160 |
   | LSPG-CN | 160 |
   | BE | 60 |

   Evidence: D:75–93. The frozen 4096-mode list’s first 2048 entries also match the independently generated 2048-mode list.

   The Fibonacci sizes are valid Fibonacci numbers. The Sobol sizes are powers of two; Halton accepts the proposed sizes. The integer Smolyak levels are valid generator inputs. The listed 2D candidate sets all have two successor rungs.

   **Clarifications needed:** `sobol2^k` is mathematical shorthand, not a parser name. Draw suffixes and MC require the new dispatcher already acknowledged in D:109–111,409–410; the existing parser accepts only unsuffixed integer names (`V/quad2d/qcore.py:83–97`). BE has only one qualifying candidate, gauss256, so its reported \(m^*\) must be labelled a minimum over that singleton list.

   **Fix:** generate and audit an explicit configuration manifest with concrete names, counts, successor links and per-scheme candidate lists.

6. **3D rounds, ladders and explicit M — WRONG in implementation scope; otherwise substantially sound**

   The round-selection procedure matches `Q/front3d.py:98–109` and `Q/DESIGN.md:474–482`. The existing 436-candidate universe uses rules covered by the proposed successors. Testing the 16 exposed candidates first and retaining the “qualified subset” label is correct.

   Extending to `lat262144` is legitimate prospective work. It uses the same CBC construction as `S/rules3d/make_rules.py:35–39`; it does not retroactively qualify previous results.

   **The explicit-M path is not supplied by changing the driver.** `S/s3common.py:105–107,115` obtains M from `Tables`; `S/c3common.py:121–138` hard-codes `complete_M(n,4*r)` and uses that mapping in both linear and quadrature tables.

   CPU evaluation of the existing shell code gives:

   | Requested M | Completed M |
   |---:|---:|
   | 512 | 513 |
   | 1024 | 1027 |
   | 1536 | 1538 |
   | 2048 | 2052 |
   | 4096 | 4097 |

   Thus W512 \(\kappa=8\) requires **4097 modes**, not 4096. A W512-only default table has only 2052. The shell-breaking control must use exactly 512.

   **Fix:** carry explicit mode lists through `Ctx`, `Setting`, `Tables.lean`, `rule_data`, caches and identities. Derive Mmax from the requested lists, and test nondefault overrides independently.

   Further restatements:

   - Freeze concrete 3D Smolyak levels, point counts and candidate/check roles; “levels with \(m\le262144\)” lacks an explicit lower bound and candidate map: D:155–158.
   - Specify how a previously unverified configuration gets reconsidered when its missing check rule becomes available.
   - The proposed \(C_w^q\) comparison is underprovided. D:142–143 runs comparator Q2 checks, but does not supply full BE ladders to find their “smallest qualified rule” in D:317–318.
   - Labelling comparisons against Q2-unqualified deployed comparators is honest. Preserve their Q1, evidence and timing requirements; exemption from Q2 is not exemption from all qualification.

7. **Timing and elasticity — WRONG for certification; NEEDS-RESTATEMENT for fitting**

   **The archive-output shortcut is insufficient.** D:199–203 replaces timing-job certification with agreement at six saved outputs because both jobs use H200. Equal outputs do not establish equal internal trajectories, stopping decisions or fresh per-step certificates.

   The inherited protocol explicitly certifies the timing job’s recording run and hashes internal states, exits and iteration counts: `Q/qtime.py:5–12,42–44`. The common brief explicitly says saved outputs are insufficient: `P/campaign/briefs/COMMON-BRIEF.md:49–56`; `P/PLAN-v4-amendments.md:21`.

   **Fix:** retain in-job certification on the untimed scoring invocation and the timed/scored internal-state identity check. Any proposed reuse of anchor/Q2 evidence needs a separately stated contract; GPU model equality is not its justification.

   The 3D screening and confirmation sizes are correct: D:204–209 matches `S/DESIGN.md:318–338`. Require complete expected invocation populations, valid screening panels, comparator evidence and same-job scored/timed identity.

   The elasticity regression is identifiable on the full factorial, but potentially not on the **qualified subset**. Qualification may leave too few cells or collinear predictors. D:214–218 provides no rank or fit-failure rule.

   **Fix:** require positive valid costs, full design-matrix rank, a condition diagnostic and sufficient residual degrees of freedom. Resample the same timing cases jointly across cells. Otherwise report “fit unavailable.” Keep these as descriptive full-query elasticities, not estimates of individual kernel exponents.

8. **Budget, reservation and memory — NEEDS-RESTATEMENT; inherited timing cache is infeasible**

   The budget line totals **25.5 GPU-h**, reasonably rounded to 26. The reservation inequality is structurally sound: D:364–369. The estimate remains provisional until the smoke certificate includes compilation, certification, tight/tighter solves, failed attempts and controls.

   The 2D scheme inventory alone has 588 cells: **22,344 production rollouts**, plus the same number of tight solves and **3,528 tighter solves**, before family panels or timing.

   Individual table payloads look plausible:

   - R512, M4096, gauss384: approximately **6.04 GB** for \(G_q,G_s,\Psi\).
   - W1024, M4097, gl64 or lat262144: approximately **12.89 GB** for the corresponding \(m(2r+M)\) payload.

   These are not peak-memory estimates.

   **Concrete blocker:** the inherited timing code retains every setting’s main tables (`Q/qtime.py:81–100`). Retaining the 208 primary factorial tables requires approximately **156.08 GB**, before the mesh bank, solver buffers, compiler memory and other forms. This already exceeds the 120 GB ceiling. D:345’s largest-single-cell pilot cannot detect it.

   **Fix:** specify bounded table lifetimes and timing blocks within the same physical-GPU job, preserving the paired reference. Pilot the actual simultaneously live footprint—including reference tables, successors, caches and compiled queries. Apply the same audit to `S/s3common.py:108–124`, which caches rules.

   Also price local rule generation separately: CBC at 262144 uses the quadratic search in `V/quad3d/rules.py:116–135`. It is mathematically legitimate but not a negligible preprocessing step.

9. **Other ways results could mislead — NEEDS-RESTATEMENT**

   - **Absence of a winner is not absence of a qualified speed-up in the full universe.** Five more bounded rounds can still leave open candidates. D:132–138 correctly acknowledges incompleteness; carry that limitation into every H2q absence statement.
   - **A selected winner failing confirmation does not prove every candidate fails.** The 3% accuracy tie-break can select a slightly slower candidate. Report “selected candidate failed confirmation,” unless a preregistered further-confirmation procedure establishes broader absence.
   - **Do not make unexpected science fail the infrastructure audit.** The blanket requirement in D:324–325 risks forcing the \(M=r\), Smolyak or bad-vector outcomes into conformity. Qualify already withdrew that pattern for production GAL configurations: `Q/DESIGN.md:492–498`.
   - **Selection-conditioned decomposition:** Shapley describes differences among selected configurations; it is not a decomposition of the physical error norm. D:245–251 should state that limit and distinguish contributions from normalized shares, especially when the total change is near zero.
   - **Preserve the campaign’s negative findings.** D:27–32,319–320 appropriately retains K2 and the unqualified historical winners. Nothing in H2q should revive a claim of beating the best FOM; `Q/report.md:470–471` specifically preserves measured speed while retracting qualification.

   **Fix:** use bounded-search and selected-candidate wording throughout, retain failed and incomplete configurations visibly, and distinguish implementation checks from falsifiable scientific predictions.

**Ranked blockers**

1. Restore timing-job Q1 certification; six-output archive agreement is insufficient.
2. Replace the unsupported production GAL=LSPG gate and prevent qualification failures from automatically satisfying the \(M=r\) scientific control.
3. Specify and audit the complete 3D explicit-mode path, including Mmax = 4097 and nondefault override checks.
4. Fix table lifetimes and pilot the actual job footprint; inherited caching exceeds available memory.
5. Freeze qualification/decision precedence for invalid, missing, unverified, empty-population and partially completed evidence.
6. Resolve 3D GAL coverage and the conflicting core/defer definitions.
7. Complete the 3D family/comparator candidate manifests and independently establish the new negative-control fixtures.

**Overall verdict: NEEDS-RESTATEMENT.**