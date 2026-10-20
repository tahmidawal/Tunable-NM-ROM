**Overall verdict: the numerical results pass this audit; the broad headline needs qualification.** The report’s error tables, order verdicts, H1/H2 failures and selections reproduce correctly. The equal-cost gain belongs to **LSPG CN, whose order is not established**, rather than to a demonstrated second-order arm.

I used CPU-only NumPy, independently recomputed summaries from `result.json`, and reconstructed errors and trajectory differences from `W_fast.npz`. Across all 8,816 trajectories, the largest reference-error discrepancy was \(2.53\times10^{-12}\); no mismatches were found. All three archived output checksums match. **No files were modified.**

**Headline audit**

| Statement | Verdict | What the data support |
|---|---|---|
| **(a)** Second-order stepping reduces median ST error from ≈1.0% to ≈0.42% at equal cost | **NEEDS-RESTATEMENT** | LSPG BE → LSPG CN at \(\Delta t_0=0.005\): **1.034805100% → 0.418560418%**, a **2.472296×** reduction. Paired time ratio **0.993340861**, IQR **0.979702871–1.010414508**. Approximately equal measured cost is supported, but calling this arm “second order” violates A1.2. GAL CN establishes order two, but costs **1.354353620×** the comparator. |
| **(b)** Cohort-worst ST error rises, so H1/H2 fail for fast | **CORRECT** | At \(\Delta t_0\), LSPG CN raises worst ST error **4.637343247% → 5.171401077%**, **11.516461%** relatively. Every CN/CN-R/BDF2 arm at this step has worse cohort-worst ST error. Both hypotheses fail independently under their exact rules. |
| **(c)** ROM time error drops ≈27×, and CN at \(5\Delta t_0\) has less time error than BE at \(\Delta t_0\) | **NEEDS-RESTATEMENT** | For **LSPG CN**, median anchor discrepancy drops **0.993721423% → 0.036053914%**, a **27.562096× ratio of medians**. The median casewise improvement factor is **23.117744×**. At \(5\Delta t_0\), CN’s median is **0.838593326%**, below BE’s **0.993721423%**, but only **32/38 cases improve**. These are resolved discrepancies to the numerical GAL-BDF2 anchor, supporting a conditional time-error estimate—not certified exact time errors. |
| **(d)** LSPG CN/BDF2 are not second order on real data, while GAL is, consistent with A1.2 | **NEEDS-RESTATEMENT** | Say **“second order is not established for LSPG CN/CN-R/BDF2 on these refinement triples; GAL establishes order two.”** The data are consistent with A1.2, but also **do not establish order one** for those LSPG arms. |

For (c), LSPG CN at \(5\Delta t_0\) also reduces **cohort-worst** anchor discrepancy, **3.773796701% → 3.485793778%**. This does not make the improvement uniform across cases. If “CN” means GAL CN instead, its median is **0.873683490%**, but its worst is **4.174354166%**, *above* the deployed BE comparator.

**Accuracy and comparator**

I checked all **56 production rows × six ST/S/TX summaries**, including their printed rounding: no discrepancies. Here are the \(\Delta t_0\) results, all **provisional reference errors**, in percent; entries are **worst / median**.

| Form | Scheme | ST | S | TX |
|---|---|---|---|---|
| LSPG | BE | 4.637343 / 1.034805 | 3.445118 / 0.412150 | 4.748191 / 1.090569 |
| LSPG | CN | 5.171401 / 0.418560 | 5.311295 / 1.114095 | 5.200344 / 0.414619 |
| LSPG | CN-R | 5.144121 / 0.419565 | 5.236475 / 0.939992 | 5.176487 / 0.443798 |
| LSPG | BDF2 | 5.035756 / 0.421461 | 5.106143 / 1.025951 | 5.070425 / 0.429486 |
| GAL | BE | 4.783191 / 1.029214 | 3.978068 / 0.410979 | 4.882264 / 1.085066 |
| GAL | CN | 5.620807 / 0.422590 | 5.856777 / 1.114281 | 5.640113 / 0.417261 |
| GAL | CN-R | 5.585314 / 0.422700 | 5.777137 / 0.940658 | 5.607886 / 0.445979 |
| GAL | BDF2 | 5.583592 / 0.425027 | 5.791530 / 1.025941 | 5.605067 / 0.431576 |

The correct preregistered comparator is **`main|LSPG|BE|0.005|prod`**, not GAL BE, the old lattice arm, or vendor runtime. Its pooled baseline median is **25.323511800 ms**.

The vendor query agrees numerically but has paired runtime ratio **0.879173382**. Thus “equal cost” must explicitly mean **relative to the generic comparator implementation**, not the faster vendor implementation.

LSPG CN improves ST error in **34/38 cases** at \(\Delta t_0\). Both its worst case and BE’s worst case are **val32 case 19**: the worsening is not merely a change in which case attains the maximum.

**Order rules and controls**

I applied the prescribed rules:

- Primary: \((\Delta t_0/2,\Delta t_0/4,\Delta t_0/8)\).
- Adjacent: \((\Delta t_0,\Delta t_0/2,\Delta t_0/4)\).
- Each difference must satisfy \(d(h)>10(s_h+s_{h/2})\), with all participating tight/tighter trajectories verified.
- At least 80% primary validity and 80% primary band membership; with 38 jointly valid cases here, at least **31/38 must have both orders inside the band**.
- Bands: **[1.7, 2.3]** and **[0.8, 1.25]**. Neither coarse nor extra-fine triples replace the primary test.

Every arm has **38/38 valid primary and adjacent triples**.

| Form | Scheme | Primary median \(p\) | Adjacent median \(p\) | Both in order-2 band | Both in order-1 band | Verdict |
|---|---|---:|---:|---:|---:|---|
| LSPG | BE | 0.970950 | 0.948934 | 0/38 | 36/38 | Order 1 |
| LSPG | CN | 0.919562 | 1.118999 | 3/38 | 17/38 | Not established |
| LSPG | CN-R | 1.728323 | 1.869251 | 20/38 | 2/38 | Not established |
| LSPG | BDF2 | 1.429014 | 1.817616 | 12/38 | 4/38 | Not established |
| LSPG | TH06 | 0.940103 | 0.897248 | 0/38 | 34/38 | Order 1 |
| GAL | BE | 0.975199 | 0.951947 | 0/38 | 37/38 | Order 1 |
| GAL | CN | 2.000289 | 2.001149 | 38/38 | 0/38 | Order 2 |
| GAL | CN-R | 1.983353 | 1.962683 | 38/38 | 0/38 | Order 2 |
| GAL | BDF2 | 2.022471 | 2.045483 | 38/38 | 0/38 | Order 2 |
| GAL | TH06 | 0.967702 | 0.950673 | 0/38 | 38/38 | Order 1 |

The extra-fine GAL medians are **2.000072321** for CN and **2.012248156** for BDF2, reported only.

**All requested controls pass:**

- BE and TH06: **38/38 primary orders outside [1.7, 2.3]**, for both forms.
- G1a: coefficient, initial-coefficient and field discrepancies **zero**, with matching iteration counts across all 38 cases.
- G4: **738 A–B–A records**, 41 subjects × 18 samples; all recorded output hashes match their expected hashes; compilation caches remain unchanged.
- All **8,816 trajectories verified**. Archived logs explicitly show GPU backend, x64 and highest precision.
- The supplied audit records all five data-corruption rejection checks passing; the manufactured-test certificate records all six mutation checks detected. I did not rerun those mutation suites.

**Anchor resolution**

I recomputed \(U_{\rm anc}=d_{8,16}+s_8+s_{16}\), without dividing the difference by three, and applied discrepancy \(\ge3U_{\rm anc}\).

All production arms are resolved on **38/38 cases**, except:

| Arm | \(\Delta t/\Delta t_0\) | Resolved |
|---|---:|---:|
| LSPG CN | 1/8, 1/4 | 27, 29 |
| LSPG CN-R | 1/8 | 30 |
| LSPG BDF2 | 1/8 | 27 |
| GAL CN | 1/8, 1/4, 1/2 | 0, 0, 36 |
| GAL CN-R | 1/8, 1/4 | 1, 37 |
| GAL BDF2 | 1/8 | 0 |

These counts match the report. In particular, the tiny finest-step GAL discrepancies must not be advertised as resolved time-error measurements.

**Hypotheses, selections and timing**

- **H1 fails:** all six CN-family/BDF2-family candidates at \(\Delta t_0\) pass the median-improvement requirement but fail the worst-error requirement.
- **H2 fails:** the smallest candidate worst ST error at steps \(\ge2\Delta t_0\) is LSPG BDF2 at \(2\Delta t_0\): **4.695607135%**, above BE’s **4.637343247%**.
- **ST selections:** fast-equal-accuracy retains BE; accurate-equal-cost selects LSPG CN at \(\Delta t_0\).
- **TX sensitivity:** fast-equal-accuracy selects LSPG BDF2 at \(2\Delta t_0\); accurate-equal-cost still selects LSPG CN.

The TX fast selection is extremely marginal in worst error: **4.747556227% versus 4.748190927%**, only **0.000634701 percentage points** better. Its median TX error is **0.616214343%**, and paired runtime ratio **0.604802519**. The reference-dependent label is essential.

All 41 timing subjects’ recomputed ratio medians, IQRs and ratio-outlier counts match the analysis. LSPG CN at \(5\Delta t_0\) has ratio **0.452348222**, but that speed does not rescue its failed ST accuracy constraint. Timing conclusions cover **six development cases, three repetitions each**, not 38 timed cases.

**Report issues to fix**

The numerical tables and existing ST/S/TX provisional labels are sound. These textual issues remain:

1. **“Only the time discretisation changes” is too broad.** [Opening paragraph](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report.md:3): the GAL arm also changes the projection/step equation relative to deployed LSPG. Restrict that statement to comparisons within a form.

2. **The preregistration citation is stale.** The opening cites A1–A5; the document now includes A1–A9. Identify the amendments applicable to this job; A9 concerns the separate FOM comparison.

3. **The glossary incompletely states the order rule.** [Lines 226–227](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report.md:226) should specify the **sum of two solve-sensitivity indicators** for each difference and the **joint adjacent-band requirement**. The implementation gets these right.

4. **“Prod vs tight” is not established solver error.** [Line 220](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report.md:220) should call it a **production-to-tight solve-sensitivity indicator**, not “solver error at production settings.”

5. **Anchor interpretation needs an explicit scope restriction.** These discrepancies concern the fixed reduced model and quadrature rule. They do not measure total PDE error or isolate all spatial, representation and quadrature errors. The current conditional wording is useful and should accompany any “27× time-error reduction” headline.

6. **Presentation omissions:** the cost plot should explicitly identify its **dev6 timing cohort**; the right-hand title in `error_vs_dt_fast.png` is clipped. The “accurate-equal-cost” selection should explicitly state that it optimizes **median** error without a worst-error constraint.

**Accept the results as provisional development/validation evidence. Do not accept an unqualified “second-order accuracy at equal cost” headline or a claim of uniformly improved accuracy.**