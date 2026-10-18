**Overall verdict: NEEDS-RESTATEMENT.** A1 substantially improves the design, and the under-resolved fixture’s numerical evidence checks out. Remaining problems include a broken fit initializer, an unsupported mandatory timing band, and incomplete qualification edge cases.

Read-only audit: no files changed, no jobs submitted. CPU checks used NumPy without running the fixture script’s writing entry point.

Line references below are to [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/DESIGN.md).

**Round-1 blocker dispositions**

| Round-1 blocker | Disposition | Evidence |
|---|---|---|
| 1. Under-resolved and component controls | PARTIALLY | Under-resolved evidence repaired, 445–460. Component accounting improved, but mandatory band remains unsupported, 462–472. |
| 2. Verdict procedures not total | PARTIALLY | Most statistical edge cases repaired, 521–542. H-cost-r initialization is broken; Q2 empty-indicator handling remains undefined. |
| 3. Q1 rejection conditions and hardware transfer | PARTIALLY | Agreement tolerances and budget exits repaired, 474–483. Output-distance transfer alone does not establish Q1/Q2, 485–493. |
| 4. Q1 negative fixture and temporal-order check | RESOLVED | Correct 300-step population and selected initialization, 495–500; error-vector contamination check, 502–506. Independent fixture establishment remains a prerequisite to using it. |
| 5. Historical recheck incomplete | PARTIALLY | Missing rules added, 508–519. H4’s historical two-mesh statistic is still conflated with a three-mesh spread. |
| 6. Timing and resource protocol | RESOLVED | Burn boundaries, raw repetitions, largest-rank smoke coverage and allocation reservations specified, 544–556. |

**Per-item verdicts for every previously non-CORRECT item**

Grouped rows cover both section findings and their corresponding control findings.

| Round-1 item | Disposition | Round-2 verdict | DESIGN evidence |
|---|---|---|---|
| P1c; timed=scored control | PARTIALLY | NEEDS-RESTATEMENT | 485–493: transfer gate insufficient for inherited certification |
| P1e | RESOLVED | CORRECT | 558–562: explicit descriptive 3D population and metric |
| §2.6 P1g | PARTIALLY | NEEDS-RESTATEMENT | 508–519: scope explicit, historical statistic mismatch remains |
| §4.1 H-cost-N | RESOLVED | CORRECT | 522–524: undefined-resample disposition fixed |
| §4.2 H-cost-r; synthetic fit control | PARTIALLY | WRONG | 532–542: initializer necessarily encounters zero |
| §4.3 H-acc-N | RESOLVED | CORRECT | 525–530: margin floor, zero handling, pinned test procedure |
| §4.4 H-acc-r | RESOLVED | CORRECT | 531: empty population explicitly inconclusive |
| §5.1 Q1 | RESOLVED | CORRECT | 474–483: absolute-plus-relative comparisons; contract-compatible validity |
| §5.2 Q2 | PARTIALLY | NEEDS-RESTATEMENT | 564–565: exclusions lack an empty-population disposition |
| §6 controls overall | PARTIALLY | NEEDS-RESTATEMENT | Component gate remains a false-rejection route |
| Component-sum control | PARTIALLY | NEEDS-RESTATEMENT | 462–472 |
| Kernel synchronization control | RESOLVED | CORRECT | 544–548: smoke includes governed maximum ranks |
| Under-resolved control | RESOLVED | CORRECT | 445–460, conditional on the required vendor bridge |
| Q1 stationarity control | RESOLVED | CORRECT | 495–500 |
| CN-R temporal-order control | RESOLVED | CORRECT | 502–506 |
| Hardware sensitivity probe | RESOLVED | CORRECT | 548–549: matched A–B–A comparisons |
| §7 timing/jobs | RESOLVED | CORRECT | 544–556 |
| §12 completeness claim | PARTIALLY | NEEDS-RESTATEMENT | 567–568 withdraw the old wording; remaining defects prevent completeness |

**Ranked remaining blockers**

1. **H-cost-r’s prescribed initialization cannot execute as written.**  
   Lines 533–534 set \(a_0=\min t\), then obtain \(\beta_0\) from a log-linear fit of \(t-a_0\). At least one observation is exactly zero. On the registered noiseless synthetic curve, CPU recomputation produced `-inf` in the log data and `[nan, nan]` from the linear fit. Noise does not eliminate the minimum.

   Specify a strictly interior initializer or an explicit positive-data fitting procedure, including constant-data handling. The current identifiability rule also detects only a small power-law contribution relative to the intercept; it does not generally establish exponent identifiability. A nearly constant power law with \(a=0\) escapes that rule.

   The revised **AM-4 secondary label is now literal and unambiguous**, including its explicit 0.25 departure threshold. That repair does not rescue the fit.

2. **The component-sum band can still reject a correct implementation.**  
   Lines 465–469 repair the missing initial gradient and batched predictor accounting. But restricting to \(r=384,512\) and widening the interval to \([0.5,2]\) supplies no independent evidence that correct separate dispatches must satisfy it.

   The stated accounting also omits evolution bookkeeping present in `t2core.make_evolve`: post-solve projections, alternation diagnostics, statistics and output updates. Compiler reuse across blocks and separate-dispatch overhead further prevent arithmetic completeness from implying a wall-time bound.

   Make this a reported diagnostic, or establish an independently validated fixture and acceptance interval before treating an out-of-band ratio as incorrect solver accounting.

3. **The transfer gate establishes output proximity, not transferred Q1/Q2.**  
   The \(2.5\times10^{-5}\) threshold at 486–489 is reasonable as a reproducibility tolerance. It does not establish fresh-Jacobian stationarity at every internal step, validity flags, or own-state \(\rho\).

   Even the distance predicate is not preserved at its boundary: an accuracy trajectory at distance \(2.49\times10^{-4}\) from its tight anchor can pass Q1, while a timing trajectory displaced by another \(2\times10^{-5}\) passes transfer but exceeds the Q1 bar.

   Certify the timing job’s untimed scoring trajectory directly, or distinguish “close to a qualified trajectory” from numerical qualification. A margin-aware distance argument alone still cannot transfer internal-step stationarity.

4. **Q2 remains non-total when all denominators vanish.**  
   Lines 564–565 exclude zero-denominator \(\rho\) values but do not define the result when none remain. An implementation could produce an empty maximum, NaN, or a vacuous pass.

   This also treats \(N_{q^+}=0,\ N_q\ne0\) identically to both vectors being zero. Specify separate dispositions and an explicit no-valid-indicators outcome. Invalid-successor handling is otherwise repaired.

5. **P1g still needs an exact historical-statistic map.**  
   H4 is explicitly a **256²-versus-1024²** spread at line 151. A1.8, lines 513–514, prescribes a spread over the newly listed meshes and places it beside the historical 0.008 pp value. A three-mesh spread cannot replace that two-mesh statistic without changing the question.

   Report H4 on its historical mesh pair, with any three-mesh extension separate. The no-Q2 scope is candid and acceptable for a **descriptive within-rule comparison**, provided these arms never receive qualification or frontier labels. However, “rule error common to both arms” is too strong: changing the test space can change quadrature error. Same rule does not imply cancellation.

**Independent numerical and contractual checks**

- **Under-resolved fixture: verified.** The archived calculation uses the same 257² restriction and physical-initial-field normalization for both distances. The maximum over evolved times is a norm, so the reverse-triangle bound is valid even when the maxima occur at different times.
- Recomputed all six lower bounds: minimum **0.0010465160601401803**, maximum **0.03359912617034971**, both matching the JSON and comfortably exceeding \(2.5\times10^{-4}\).
- Recomputed all twelve own-state \(\rho\) values: range **1.2727756313871847–1.5816420465831023**, matching the JSON. Both trajectory SHA256 values match.
- Checkpoint and rotation hashes match the historical job. The ordered 1024², \(M=1536\) mode list matches the archived builder.
- Vendor BE and generic LSPG-BE have matching mathematical residuals and initialization architecture. Their numerical equivalence remains a **required smoke bridge**, not something the historical JSON itself proves. Run that bridge before attributing an unexpected control outcome specifically to Q2 code.
- **H-acc-N:** the amended Wilcoxon procedure resolves the earlier five-case error, zero margins, ties and all-zero shifted samples. **H-cost-N:** conditioning the interval on defined resamples is now an explicit, reproducible rule.
- **A1.4:** excluding budget exhaustion from automatic invalidity matches COMMON-BRIEF §3. Such steps still face the stationarity and distance predicates.
- **A1.11:** reserving single-GPU time limits is the correct allocation-based repair. Enforcement must retain reservations until accounting is reconciled and count every execution attempt, including requeues; a time-limit increase requires a new reservation check.
- The gauss640 table estimate recomputes to **7.5497472 GB**. This verifies the table arithmetic, not total peak device memory.

The decisive remaining false-rejection route is the mandatory component band; the decisive unsupported-pass route is output-only qualification transfer. The broken H-cost-r initializer independently prevents readiness.