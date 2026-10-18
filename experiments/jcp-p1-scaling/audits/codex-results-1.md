**Overall: NEEDS-FIX.** The registered primary verdicts reproduce. The blockers are misleading population comparisons and reporting defects, not evidence that the primary simulations or hypothesis calculations are wrong.

I made no file changes, copied no arrays, and submitted no jobs. Recomputation used CPU only, reading archives in place.

1. **H-cost-N — CORRECT, with an identity-audit limitation.**

   Independently recomputed valid-triplet medians and the registered 10,000-case bootstrap:

   | Subject | Medians: 256² / 1024² / 4096² | R | 95% interval |
   |---|---|---:|---|
   | R128 BE | 0.254258 / 0.252104 / 0.252203 | 1.008544 | [1.003832, 1.040386] |
   | R128 CNR | 0.254964 / 0.250939 / 0.252855 | 1.016041 | [1.004788, 1.026720] |
   | R256 BE | 0.596543 / 0.597077 / 0.605589 | 1.015164 | [1.002294, 1.024243] |
   | R256 CNR | 0.592788 / 0.593692 / 0.602398 | 1.016212 | [1.003946, 1.034243] |
   | R384 BE | 1.008110 / 1.006478 / 1.015497 | 1.008961 | [1.005335, 1.028545] |
   | R384 CNR | 1.006132 / 1.015766 / 1.016168 | 1.009975 | [1.005524, 1.030018] |

   Both nominal-policy verdicts also reproduce as passes. All 432 final evolution triplets are valid; one drift failure required a retry. One of 432 descriptive fused triplets is invalid. Doubled-work medians are **1.996353** and **1.956543**; split/fused field difference is zero with equal iterations.

   All 24 subjects pass the recomputed timing-job distance and stationarity predicates; their accuracy-job labels also pass. All 11 job stdout logs contain `jax_backend=gpu`, `x64=True`, and `precision=highest`; stderr scans found no relevant errors.

   **Limitation:** all **144 scoring-reference hashes** reconstruct exactly, including the padded step buffers. Individual timed executions retain comparison booleans, not their actual hashes/arrays. Every saved identity comparison passes, including retries, but independent rehashing of every timed execution is impossible from this archive.

2. **H-acc-N / H-acc-r — CORRECT primary calculations; WRONG presentation of some descriptive populations.**

   Independently scored the primary saved fields against loader-verified E0 references and reproduced all primary transition counts and statistics. All 38 ST case-kinds are eligible.

   P256 uses 38 cases; P4096 and three-mesh trends correctly use **16**, including R384 despite its larger archive. Every primary pair has 100% within margin. Both TOST p-values are **3.637979e-12** for P256 and **1.525879e-5** for P4096.

   R128-CNR’s trend rejects at **p = 0.00628662**. Its pass follows the registered precedence: practical equivalence can coexist with a detectable small trend. It must not be described as evidence of literally zero mesh dependence.

   At 1024², R384→512 reproduces:

   | Scheme | Improve / worsen / tie / unqualified | Outcome |
   |---|---|---|
   | BE | 4 / 16 / 18 / 0 | does not improve |
   | CNR | 6 / 22 / 10 / 0 | does not improve |

   The primary 128→384 ranges are correctly “monotone” under **majority improvement among resolved cases**, not every-case monotonicity.

   **Presentation defect:** [make_report.py:101](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/make_report.py:101) joins 16-case R128/R256 summaries to **38-case R384** summaries at 4096². Its mesh plot then labels 4096² as a 16-case subset. For R384-CNR, the actual 16-case median is **0.247929%**, versus the plotted 38-case **0.197624%**. For BE: **1.210462%** versus **0.977786%**. These curves confound population changes with rank/mesh changes.

3. **Q1/Q2 — CORRECT for the checked configurations.**

   Independently decoded coefficients and checked validity, pooled stationarity, tight distance, tight/tighter, successor distance, and finer-rule convergence:

   | Configuration | Tight | Tight/tighter | q→q+ | q+→q++ |
   |---|---:|---:|---:|---:|
   | 256² R128 BE | 5.028e-5 | 1.745e-9 | 1.977e-4 | 3.639e-6 |
   | 1024² R384 CNR | 4.455e-5 | 3.878e-10 | 1.425e-4 | 1.292e-6 |
   | 1024² R512 CNR | 4.152e-5 | 5.475e-10 | 1.231e-4 | 3.064e-7 |
   | 4096² R256 BE | 1.063e-5 | 7.237e-10 | 9.230e-5 | 2.513e-6 |

   Each has zero nonstationary steps and passes the recorded/certified γ agreement. Independent NumPy recomputation of R128’s six dev6|0 ρ values agrees within **2.12e-16**. No configuration supporting a primary verdict is labelled unqualified. Historical arms remain descriptive and lack Q2 certification.

4. **H-cost-r — CORRECT under literal A1.9/AM-4.**

   Independently reconstructed fixture medians from the 30 timings and reproduced the registered bootstrap intervals:

   | Panel/component | Fitted p | 95% interval | Verdict |
   |---|---:|---|---|
   | 2D JTJ | 1.924877 | [1.896909, 1.954618] | fail, operation-model |
   | 2D chol | 1.235084 | [1.222893, 1.243874] | fail, operation-model |
   | 3D JTJ | 1.326162 | [1.314488, 1.338436] | fail, operation-model |
   | 3D chol | 1.034708 | [1.031289, 1.037489] | fail, operation-model |

   Dispatch-sync medians are within **1.980–2.007**; fixed-panel drift ratios are **0.998547** and **0.998220**.

   The byte-model exponents are approximately **1.164/1.225** for JTJ and **2** for chol, satisfying A1.9’s departure rule. Thus “operation-model” is correct **even though those exponents lie outside the measured intervals**. It is a classification rule, not proof that memory traffic explains the measurements.

5. **P1g — CORRECT numbers; historical interpretation NEEDS-RESTATEMENT.**

   Independent saved-field calculations reproduce the 256² frozen–nominal maximum distances:

   - R384 BE: **5.233939e-4**.
   - R384 CNR: **9.219106e-4**.
   - Historical gauss96 BE: **5.285928e-4**.

   Historical gauss96’s two-mesh first-order spread changes from **0.008222 pp nominal** to **0.014504 pp frozen**. Its nominal errors reproduce dv256/dv1024/dv4096 within approximately **3.6e-16**.

   The old numerical result therefore reproduces, but its interpretation as a fixed-test-space comparison does not. The nominal construction changed the test space. Historical-rule differences are descriptive total effects under those rules, not independently certified quadrature-free effects.

6. **P1e, reproduction, pilot, smoke, accounting — CORRECT calculations; P1e report labels WRONG.**

   - **2D:** independently decoded P1e distances reproduce. The supposedly “failing” gauss64 arm actually passes both bars: distance **4.863348e-5**, ρ **0.0553124** against gauss96.
   - **3D:** judgments recomputed from raw job rows: lat8192→lat32768 fails distance (**0.004210079**) while passing ρ (**0.0563712**). The subsequent pairs pass both bars. Recorded evidence has zero nonstationary, saturated, or reason-3 steps among **600**.
   - **Reproduction:** independently checked all 38 cases for all four arms. Worst coefficient differences: R128 BE **2.854e-15**, CNR **1.688e-15**; both R384 arms **zero**.
   - **Pilot:** **113,469,241,344 bytes < 120 GB**; full-bank decision follows.
   - **Smoke:** first acceptance contains the stated five failures; corrected acceptance passes **82/82** required checks. Raw smkB/smkC sync ratios reproduce the documented correction.
   - **Accounting:** live read-only `sacct` returns 11 completed one-GPU jobs totalling **6,888 seconds = 1.913333 GPU-hours**, agreeing with ledger rounding.

   [report.md:135](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/report.md:135) nevertheless calls gauss64 “failing” and prints **`None`** for deployed/finer/check rules. Those are gauss128/160/192.

7. **Additional eligibility and scope defects — NEEDS-RESTATEMENT.**

   [analyze.py:353](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/analyze.py:353) computes S-reference medians without excluding or identifying the three ineligible case-kinds: dev6|1, val32|21, val32|24. At 1024² R384 BE, the displayed all-case median is **0.033746%**; the eligible-only median is **0.051261%**. Primary ST verdicts are unaffected, but the S column needs eligibility disclosure or filtering.

   The report/export also omit A3.2’s required timing qualification: **solver with internal-state recording**. H-cost-N establishes evolution-time equivalence for that executable, not constant full-query cost.

**Ranked blockers before accepting the deliverable:**

1. Correct the mixed-population accuracy plots and their false 16-case label.
2. Correct or explicitly label the S-reference summaries containing ineligible cases.
3. Repair P1e rule names and state that the intended 2D failing arm passed.
4. Restore the recorded-executable timing scope and disclose the per-execution hash audit limitation.

The primary registered verdicts survive the audit. The current report/export package does not yet warrant **RESULTS-CONFIRMED** without these corrections.