**JOB GATE: SUBMIT `c2fbgrid`; SUBMIT `c2fbbud`.** The configs and resource certificates support running both jobs. This does **not** qualify any fallback in advance. A14’s qualification wording needs clarification before reporting outcomes.

1. **CORRECT — A14.1 finalist outcomes.** Both selected CN-R steps have the stated verdicts. Worst $\delta_{\rm quad}$, against $\tau=2.5\times10^{-4}$:

   | Setting | $\Delta t_0$ | $2\Delta t_0$ | Both |
   |---|---:|---:|---|
   | fast | 7.60661e−4 | 7.18304e−4 | Fail |
   | acc | 2.85582e−4 | 4.12991e−4 | Fail |
   | R256 | 2.69594e−4 | 2.74024e−4 | Fail |
   | R128 | 1.00756e−4 | 1.30269e−4 | Pass |
   | R384 | 1.10476e−4 | 1.48810e−4 | Pass |
   | R512 | 1.07656e−4 | 1.33530e−4 | Pass |

   Every finalist has zero unavailable cases across 38 cases. All corresponding Gauss320/Gauss640 checks pass; the largest distance is 1.24246e−5, below 2.5e−5.

2. **CORRECT — model equivalence and same-job timing provenance; qualification is not automatic.** Frozen first fallbacks match R128 and R384 exactly: shared frozen bank, respectively $R'=128,M=512,\mathrm{fib4181}$ and $R'=384,M=1536,\mathrm{gauss128}$. Their selected forms and steps also match.

   The `c2grid` common panel is valid, from job **5209526**. Dividing its median baseline-normalized ratios gives:

   | Fallback versus original deployed BE | $\Delta t_0$ | $2\Delta t_0$ |
   |---|---:|---:|
   | fast → R128 | 1.30865 | 0.76217 |
   | acc → R384 | 1.56126 | 0.86423 |

   Consequently, **both accurate-equal-cost fallbacks fail the original ≤1 cost requirement**, despite passing quadrature. Describe this estimator as a *ratio of median paired ratios*, not a newly measured direct fallback/original-BE paired median.

3. **CORRECT — both configs implement the proposed runs.** `c2fbgrid` has the two specified rules, BE/CN-R at both steps, anchors, empty FOM lists, and original R256 BE as common baseline. `c2fbbud` preserves the frozen R256 arm list exactly and uses Gauss320/Gauss640 with Gauss768 successor. Its frozen SHA matches `FROZEN-SELECTION.json` exactly.

4. **CORRECT for these jobs — inventory changes and certificates.** R256fb grid pricing uses the same-$R'$ R256 smoke measurements scaled by $128^2/96^2=16/9$. Empty production FOM lists avoid the empty minimum/mean failures. Both certificates pass the repository’s verifier against **HEAD**, including source, config, smoke hashes and smoke gates.

   | Job | Projected hours, including margin | Projected device memory |
   |---|---:|---:|
   | c2fbgrid | 0.47534 | 17.5686 GB |
   | c2fbbud | 23.10052 | 40.8207 GB |

   Both fit the certified 48-hour A100-80GB allocations. These are resource projections, not measured production runtimes. The empty-FOM change still requires calibration timings in the **source smoke**; the supplied smoke has them.

5. **CORRECT — local empty-FOM execution completes.** `local2dfb` records GPU, x64, highest precision, `complete=true`, empty FOM outputs, and completion of all timing panels. **Its timing panels are invalid**, with worst drift factors 1.10241, 1.10706 and 1.12292. This establishes machinery completion only; it supplies no accepted timing qualification. The production certificates use separate cluster smokes.

6. **NEEDS-RESTATEMENT — A14’s final “iff” sentence.** The numerical thresholds and frozen fallback order are unchanged, but that sentence incompletely states the inherited gates. Qualification must retain verified solver/accuracy runs, valid timing, all-case finalist/reference checks, and applicable reproduction gates. The fast arm’s requirement is **accuracy feasibility**, specifically all four worst/median ST/TX comparisons against the original deployed BE under A3.2’s tie policy—not a “cost criterion.”

   Preserve those requirements explicitly. Passing quadrature alone cannot qualify either accurate fallback above, and passing the new jobs cannot silently replace the frozen selections.

No files were created, modified or appended; no jobs were submitted.