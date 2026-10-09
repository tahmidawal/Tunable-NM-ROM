**The displayed J2 numerical tables are CORRECT.** The main issues are incomplete secondary-tolerance reporting and wording that still implies per-case eligibility. No files were modified.

I independently recomputed JSON aggregates, both-tolerance selections and timing ratios, verified all **207 archived output checksums**, and inspected all four plots. The full NumPy field audit was reviewed, not rerun; its saved result reports 6,144 error comparisons and 5,760 distance comparisons passing, with both fault injections detected.

1. **CORRECT — errors, floors and tensor results.**

   Recomputed percentages below use each case’s maximum over the five evolved output times, then the cohort maximum/median. All agree with the displayed rounding in [report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md:116).

   | \(R'\) | \(\kappa\) | Deployed worst / median (%) | Converged worst / median (%) | Tensor worst (%) |
   |---:|---:|---:|---:|---:|
   | 512 | 4 | 3.165024 / 1.117490 | 3.165598 / 1.117490 | 10.218572 |
   | 512 | 3 | 3.292371 / 1.124505 | 3.293255 / 1.124506 | 10.221626 |
   | 512 | 2 | 4.717849 / 1.184479 | 4.721930 / 1.184428 | 10.230441 |
   | 256 | 4 | 4.702516 / 1.251184 | 4.702632 / 1.251304 | 10.438115 |
   | 256 | 3 | 5.358007 / 1.253292 | 5.357898 / 1.253411 | 10.434127 |
   | 256 | 2 | 7.781636 / 1.299757 | 7.785770 / 1.299678 | 10.424913 |

   Projection floors, worst / median: **2.252091 / 0.158088%** at \(R'=512\); **4.009585 / 0.493525%** at \(R'=256\). Both floor checks pass.

2. **CORRECT selections; NEEDS-RESTATEMENT — report omits secondary \(m^\star\).**

   Independent pooled-eligibility selection reproduces every stored choice:

   | \(R'\) | \(\kappa\) | Primary \(m^\star\), lattice / Gauss | Secondary \(m^\star\), lattice / Gauss |
   |---:|---:|---:|---:|
   | 512 | 4 | 32768 / 32768 | 16384 / 13824 |
   | 512 | 3 | 32768 / 32768 | 16384 / 13824 |
   | 512 | 2 | 32768 / 32768 | 16384 / 32768 |
   | 256 | 4 | 16384 / 13824 | 4096 / 8000 |
   | 256 | 3 | 16384 / 13824 | 4096 / 8000 |
   | 256 | 2 | 16384 / 13824 | 8192 / 8000 |

   Primary \(\tau=2.5\times10^{-4}\); secondary \(\tau=10^{-3}\). A0-3b requires both to be reported, but the 3D report displays only primary.

3. **CORRECT — H3 table and negative verdicts.**

   | \(R'\) | Trim \(\kappa\) | Worst change (pp) | Median change (pp) | Setting ratio | Final paired ratio | Acceptable / useful |
   |---:|---:|---:|---:|---:|---:|---|
   | 512 | 3 | +0.127347 | +0.007016 | 0.791893 | missing | False / False |
   | 512 | 2 | +1.552825 | +0.066990 | 0.659241 | missing | False / False |
   | 256 | 3 | +0.655491 | +0.002108 | 0.856776 | 0.851828 | False / False |
   | 256 | 2 | +3.079120 | +0.048573 | 0.788533 | 0.795268 | False / False |

   Final ratios were recomputed by taking repetition medians within each case/phase, forming trim/base ratios, then taking their median. Every trim fails the worst-error allowance; \(512,\kappa=2\) also fails the median allowance.

4. **CORRECT — as-run versus amended deployments and explicit post-hoc disclosure. NEEDS-RESTATEMENT — residual chronology/eligibility wording.**

   The as-run deployments are unavailable at all three \(R'=512\) settings; A7 selects `lat32768`, `lat32768`, `gl32`. All \(R'=256\) deployments remain `gl24`. Preserved `as_run` records match the original JSON exactly, and all per-arm numerical records are unchanged.

   The [report’s chronology paragraph](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md:103) correctly discloses that first-setting errors/distances had been seen before A7. The original log supports the triggering observation: converged case 36 has one non-stationary step—**1/1600 = 0.0625% pooled**, but **1/25 = 4% per rollout**. Human observation timing remains a documented account, not something the result JSON proves.

   Two wording defects remain:
   - A7’s heading, “before any 3D selection was read,” is overbroad: the as-run `deployed None` had been read. Its body gives the necessary qualification: **before pooled-rule selection**.
   - The report glossary still says eligibility “on every validation case.” For 3D, the non-stationary fraction is pooled over the cohort, with certification pooled separately. The introduction’s “A0–A5” reference is also stale.

5. **CORRECT — gates pass, controls fire, K-time is valid.**

   - G1–G4 pass under A6. The corner-point second-order FD diagnostic remains \(1.868\times10^{-6}\); the registered replacement checks pass.
   - Across six settings, K-conv distances are \(4.226\times10^{-7}\)–\(9.657\times10^{-6}\), below \(2.5\times10^{-5}\).
   - K-target discrepancies are \(1.609\times10^{-8}\)–\(1.942\times10^{-7}\), below \(10^{-5}\).
   - Both controls fail **both** distance thresholds and the \(\rho\) threshold everywhere. Their distances span 0.339–1.605; \(\rho\) spans 1.756–45.540.
   - All seven timing panels pass independently recomputed drift/determinism checks. Maximum symmetric drift is **1.054719**, below 1.10; every recorded timed-output difference is zero. Logs confirm GPU, f64 and highest precision.

6. **WRONG — interpreting lower \(M\) as reducing required quadrature points.**

   At fixed \(R'\), primary \(m^\star\) is unchanged across all three \(\kappa\). At secondary tolerance, trimming sometimes **increases** it. Thus J2 contradicts the proposed point-count reduction mechanism on this ladder.

   The \(m^\star\)-versus-\(M\) plot is numerically correct, but combines two widths without distinguishing \(R'\). Its overall upward pattern cannot establish an effect of \(M\): at **the same \(M=1027\)**, the two widths need different \(m^\star\). The cost/error plots match the data; the memory curve is explicitly a nominal tensor-storage formula, not measured peak job memory.

7. **CORRECT — the off-mesh/tensor gap is consistent with the October 1 validation lane.**

   The historical [validation table](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/2026-10-01-burgers3d-offmesh-quadrature.md) reports, at \(64^3\):

   - \(R'=512\): **3.16% / 10.22%** selected off-mesh/tensor; J2 \(\kappa=4\): **3.1650% / 10.2186%**.
   - \(R'=256\): **4.71% / 10.44%**; J2: **4.7025% / 10.4381%**.

   These agree closely despite different selected rules and primary tolerances. This is consistency on the same validation cohort, not independent generalization evidence. No cross-job timing comparison is justified.

8. **WRONG if stated as “one-case artifact” — case 5 dominates, but does not explain away failure.**

   Case 5 is worst in every setting. Removing it still leaves these worst-error increases:

   | \(R'\) | Trim \(\kappa\) | Worst increase excluding case 5 (pp) | Cases worsening by \(>0.05\) pp |
   |---:|---:|---:|---:|
   | 512 | 3 | +0.108021 | 4 |
   | 512 | 2 | +1.056940 | 27 |
   | 256 | 3 | +0.140196 | 2 |
   | 256 | 2 | +1.141392 | 29 |

   Every trim still fails the worst-error allowance after removing case 5.

9. **CORRECT — final-panel confirmation is missing for \(R'=512\).**

   None of its three amended deployments appears in the final cross-setting panel. The report explicitly discloses this. “Missing/not measured” would be clearer than “withdrawn.” Valid setting-panel timings do not supply the missing confirmation, although the accuracy failures already settle H3 negatively.

**Statements safe to report:**

- Under post-hoc A7 selection, no tested 3D trim meets H3’s benchmark-relative accuracy requirement.
- Trimming \(M\) lowers measured setting-panel cost but does not lower primary \(m^\star\) at fixed \(R'\).
- The accuracy deterioration survives removal of the worst case.
- The off-mesh/tensor accuracy gap reproduces the earlier validation pattern.
- Results remain provisional against first-order references; \(R'=512\) lacks final-panel timing confirmation.