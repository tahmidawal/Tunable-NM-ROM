**Overall verdict: the numerical results and pre-registered H1/H2 passes survive. The headlines need tighter wording about equal cost, reference accuracy, and observed order.** Both selected arms reproduce exactly, but their timing-based selection is fragile.

I recomputed from `result.json` using CPU-only NumPy. No files were modified. All 56 production arms’ ST/S/TX worst and median values match the analysis and report; all anchor summaries match. Both raw-file checksums match the archive. I did not repeat the existing audit’s full coefficient-to-field reconstruction.

**Headline verdicts**

| Statement | Verdict | Findings |
|---|---|---|
| **(a)** CN at \(dt_0\): 0.906% → 0.098% median, 2.747% → 1.796% worst, “at equal cost” | **NEEDS-RESTATEMENT** | Errors are correct: **0.906333% → 0.097998%**, **2.746514% → 1.795643%**. Paired cost ratio is **1.000063858**, which **fails** the registered ≤1 cutoff. Say “at approximately the same measured cost,” not that CN qualifies for accurate-equal-cost selection. |
| **(b)** CN at \(2dt_0\) is more accurate and 1.75× faster; H2 passes | **CORRECT for the registered aggregate comparison** | ST worst **1.802938%**, median **0.258577%**; ratio **0.570096301**, reciprocal **1.75409×**. H2 passes. This compares against **generic main-rule LSPG-BE**, and “more accurate” means cohort worst/median—not every case. |
| **(c)** “Time error drops 37×” | **NEEDS-RESTATEMENT** | Median **anchor discrepancy** drops **0.978156% → 0.026083%**, a **37.5021× ratio of medians**. Both are resolved on **38/38** cases. This supports a ROM time-discretization interpretation under the anchor assumptions; it is not a 37× reduction in total PDE error or a median of per-case improvement factors. |
| **(d)** LSPG CN-R/BDF2 second order, CN not established; “GAL all second order” | **NEEDS-RESTATEMENT** | Correct for **observed orders on the prescribed triples**, with “GAL all” restricted to **CN, CN-R, BDF2**. GAL BE and TH06 are first order. These results do not establish general asymptotic second-order accuracy of the LSPG methods. |
| **(e)** CN-R selection is decided by a \(6\times10^{-5}\) timing margin | **CORRECT, with qualification** | CN exceeds the cutoff by **0.000063858**, or **0.006386%**. Admitting CN would displace CN-R. But BDF2 at \(dt_0\), with an even smaller ST median **0.095667%**, is also excluded by ratio **1.001969609**. The selection is a deterministic threshold outcome, not evidence that CN-R is reliably cheaper. |

CN at \(2dt_0\) actually worsens ST error on three cases:

| Case | BE at \(dt_0\) | CN at \(2dt_0\) |
|---|---:|---:|
| val32/10 | 0.572842% | 1.505360% |
| val32/21 | 0.475823% | 0.917876% |
| val32/24 | 0.956159% | 1.091620% |

**Recomputed accuracy**

Below, each entry is **worst / median**, in percent. Every row has 38 verified cases.

| Arm | Step | ST | S | TX |
|---|---:|---:|---:|---:|
| LSPG BE | \(dt_0\) | 2.746514 / 0.906333 | 1.063627 / 0.050754 | 2.918509 / 0.967210 |
| LSPG CN | \(dt_0\) | 1.795643 / 0.097998 | 3.116146 / 1.001490 | 1.786026 / 0.060339 |
| LSPG CN-R | \(dt_0\) | 1.760715 / 0.135480 | 3.006648 / 0.846912 | 1.761274 / 0.194760 |
| LSPG BDF2 | \(dt_0\) | 1.710326 / 0.095667 | 2.989088 / 0.918894 | 1.710537 / 0.127759 |
| GAL BE | \(dt_0\) | 2.729659 / 0.906343 | 1.086644 / 0.048853 | 2.900918 / 0.967221 |
| GAL CN | \(dt_0\) | 1.851264 / 0.098562 | 3.260229 / 1.001733 | 1.838926 / 0.059830 |
| GAL CN-R | \(dt_0\) | 1.819853 / 0.135449 | 3.148813 / 0.847123 | 1.823019 / 0.194744 |
| GAL BDF2 | \(dt_0\) | 1.843520 / 0.092884 | 3.178708 / 0.919160 | 1.844795 / 0.127739 |
| LSPG CN | \(2dt_0\) | 1.802938 / 0.258577 | 3.149525 / 1.167902 | 1.789957 / 0.205487 |
| LSPG CN-R | \(2dt_0\) | 1.738133 / 0.657643 | 2.751673 / 0.586793 | 1.836344 / 0.715204 |
| LSPG BDF2 | \(2dt_0\) | 1.999551 / 0.448229 | 2.746627 / 0.773228 | 2.052698 / 0.500048 |

The improvement survives switching ST to TX. It does **not** survive switching to S, which retains the coarse BE time error. The report correctly labels these reference-based accuracy results provisional.

**Order and anchor checks**

All primary and adjacent triples are valid on all 38 cases. Applying A2.12/A3.3/A4.1 gives:

| Arm | Primary median \(p\) | Adjacent median \(p\) | Primary in second-order band | Both in band | Claim |
|---|---:|---:|---:|---:|---|
| LSPG CN | 1.5694 | 1.9317 | 16/38 | 15/38 | Not established |
| LSPG CN-R | 1.9706 | 1.9499 | 36/38 | 36/38 | Order 2 |
| LSPG BDF2 | 1.9916 | 2.0228 | 35/38 | 35/38 | Order 2 |
| GAL CN | 2.0007 | 2.0038 | 38/38 | 36/38 | Order 2 |
| GAL CN-R | 1.9836 | 1.9627 | 38/38 | 38/38 | Order 2 |
| GAL BDF2 | 2.0235 | 2.0472 | 38/38 | 38/38 | Order 2 |

BE and TH06 pass the first-order criteria in both forms. The extra finer GAL-BDF2 triple has median **2.0150**, supporting the anchor interpretation.

Anchor uncertainty is **0.001293% median**, **0.006637% maximum**. At \(dt_0\), BE and CN discrepancies exceed three times the indicator on every case. Fine-step CN discrepancies are often unresolved: **5/38** resolved at \(dt_0/8\), **16/38** at \(dt_0/4\). Thus the smallest plotted discrepancies should not be presented as fully resolved time errors.

**Hypotheses, selections, and timing**

- **H1 passes:** all six CN/CN-R/BDF2 arms at \(dt_0\), across both forms.
- **H2 passes:** LSPG CN, CN-R, and BDF2 at \(2dt_0\). The amended criterion concerns the scheme family; it does not require CN’s observed-order claim to pass. CN-R and BDF2 independently sustain H2.
- **Fast-equal-accuracy:** BDF2 at \(2dt_0\), ratio **0.558140074**, versus CN-R’s **0.558452854**. The difference is only **0.000312780**. CN-R actually has the lower unpaired median milliseconds, **74.797 versus 75.056**; the prescribed paired-ratio selection still correctly chooses BDF2.
- **Accurate-equal-cost:** CN-R at \(dt_0\), ratio **0.996460093**. Both selections are unchanged under TX.

There are **738 A–B–A records**, 41 candidates × 18 samples. Recorded hashes agree, compilation caches remain unchanged, and the log confirms GPU/f64/highest precision.

CN at \(dt_0\) has ratio IQR **[0.996646, 1.004745]**, with **5/18** MAD outliers. CN-R’s IQR is **[0.992568, 1.001684]**. These spreads dwarf CN’s cutoff margin. CN at \(2dt_0\) has IQR **[0.564211, 0.624438]**, with **6/18** outliers; its substantial speed advantage is much less fragile.

**Controls and report issues**

All **8,816** raw trajectories are recorded finite and verified. G1a reports exact generic/vendor agreement; G1b differs by only **\(6.60\times10^{-16}\)**. Higher quadrature preserves the headline CN worst errors: **1.793452%** at \(dt_0\), **1.800057%** at \(2dt_0\). Neither headline CN arm is flagged alternating.

I found **no incorrect accuracy-table numbers, order decisions, or hypothesis/selection calculations** in [the report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report.md:207). Its presentation has three weaknesses:

1. **Comparator naming:** “deployed” means the generic implementation using the main quadrature rule. The actual vendor query is faster—paired ratio **0.941745**, median **125.885 ms**, versus the generic baseline’s **133.770 ms**. The reported 1.754× is against the generic comparator.
2. **Rounding hides selection logic:** printing CN’s ratio as **1.000** conceals why it is excluded. Show sufficient precision beside the selection.
3. **Plot resolution wording:** “hollow = unresolved” means fewer than half the cases resolve; filled markers do not guarantee all cases resolve. State that aggregation rule.

The defensible result is a substantial, provisional ST/TX accuracy improvement and an approximately **1.75× paired speedup at doubled step size**, against the registered generic BE comparator. Exact equal-cost claims and robust superiority of the selected timing winners are not established.