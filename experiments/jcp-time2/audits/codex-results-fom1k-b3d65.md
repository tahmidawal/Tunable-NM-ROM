**Overall verdict: the tables reproduce correctly, but the proposed headlines overstate the scope.** Statement 2 survives; 1, 3 and 5 need qualifications; 4 is wrong across both 3D settings.

CPU-only, no files modified. Both audit hashes match the current result files. I independently recomputed all 96 FOM/ROM aggregate rows, 80 matched comparisons, 96 3D production rows, calibration choices, order classifications and H1/H2 outcomes from `result.json`. The two report sections reproduce without text/numerical mismatches. I did not rerun the solvers or the full NPZ field-rescoring audits.

1. **NEEDS-RESTATEMENT — CN is faster and more accurate here, but its observed second order is not established.**

   | FOM at $\Delta t_0$ | Worst ST | Median ST | Median time |
   |---|---:|---:|---:|
   | BE | 3.213957% | 0.981733% | 456.133 ms |
   | CN | 1.174236% | 0.316314% | 285.708 ms |

   The numerical comparison is correct. Using BE at this step as the sole baseline inflates speed-ups relative to available, more accurate comparators.

   However, **A11 explicitly prohibits calling this FOM-CN result established second order**: dev6 case 0 has orders **2.382590, 5.320926, 3.483377**, outside the registered band; case 2 gives **2.009301, 2.003120, 2.000822**. “Consistent with a stiff-mode transient” is an interpretation, not a demonstrated cause.

   Safer headline: “The CN FOM at $\Delta t_0$ is faster and has lower provisional ST error than BE; its observed second-order convergence is not established on both check cases.” CN-R provides an established-order alternative: **1.291440% worst ST, 365.069 ms** at $\Delta t_0$.

2. **CORRECT — with the matched FOMs named explicitly.**

   | Acc-setting LSPG ROM | ROM worst ST | ROM ms | Cheapest qualifying FOM | FOM worst ST | FOM ms | Speed-up |
   |---|---:|---:|---|---:|---:|---:|
   | BE, $\Delta t_0$ | 2.746514% | 134.745 | CN-R, $2\Delta t_0$ | 1.937881% | 236.727 | **1.756847×** |
   | CN, $2\Delta t_0$ | 1.802938% | 78.305 | CN, $\Delta t_0$ | 1.174236% | 285.708 | **3.648670×** |

   These satisfy A9.4 over the **listed configurations**, within job 5017631 on one A100. “With BE” describes the ROM, not its matched FOM.

   These are ratios of median times, with accuracy over 38 cases and timing over six dev6 cases. Matching cohort-worst error does **not** establish equal-or-better accuracy case by case. Both comparisons remain provisional because of the reference.

3. **NEEDS-RESTATEMENT — correct for LSPG, `gl24_R512`, using median anchor discrepancies.**

   - Deployed BE at $\Delta t_0$: **1.870207%** median anchor discrepancy.
   - Adaptive LSPG-CN at $\Delta t_0$: **0.089058%**, a **20.999943×** reduction.
   - Adaptive LSPG-CN at $2.5\Delta t_0$: **0.532919%**, a **3.509365×** reduction.
   - Its median paired cost ratio to deployed BE is **1.093756**, with IQR **[0.978274, 1.195076]**. Median times are **37.838 versus 35.516 ms**; their quotient is **1.065347**, a different statistic.
   - Adaptive **LSPG-BE** at equal step costs **1.699818×** deployed BE. This is not generic to GAL: GAL-BE costs **2.505×**.

   All 16 cases resolve the headline anchor discrepancies. Anchor uncertainty indicators span **0.001484–0.014026%**, median **0.005455%**. At finer steps, unresolved values must remain censored.

   Call this **distance to the discrete-ROM GAL-BDF2 anchor**, an estimate of temporal error—not certified error against the time-continuous PDE. At `lat4096_R256`, the corresponding reductions are **21.9985× / 3.4438×**, with coarse-CN cost ratio **1.1240**.

4. **WRONG as a statement about both 3D settings.**

   All primary and adjacent triples are valid for all 16 cases. Thus **13 cases must pass the band checks**; the “adjacent unresolved below 8” exception never applies.

   Counts below are cases with **both** orders in $[1.7,2.3]$; primary in-band counts happen to be identical.

   | Scheme/form | `gl24_R512` | `lat4096_R256` | Registered conclusion |
   |---|---:|---:|---|
   | GAL CN | 16/16 | 16/16 | Order 2, both |
   | GAL CN-R | 16/16 | 16/16 | Order 2, both |
   | GAL BDF2 | 16/16 | 16/16 | Order 2, both |
   | LSPG CN-R | 15/16 | 14/16 | Order 2, both |
   | LSPG BDF2 | 14/16 | **11/16** | Order 2 only at `gl24_R512` |
   | LSPG CN | **10/16** | **2/16** | Not established, either |

   Lattice LSPG-BDF2’s primary median **1.919015** and adjacent median **1.948525** look second order, but **five outliers defeat the registered threshold**. The report correctly withholds this claim.

5. **NEEDS-RESTATEMENT — worse cohort-worst reference error, not uniformly worse accuracy.**

   At $\Delta t_0$, `gl24_R512` LSPG results are:

   | Scheme | Worst reference error | Median reference error |
   |---|---:|---:|
   | BE | 3.164073% | 1.153241% |
   | CN | 5.107576% | **0.951254%** |
   | CN-R | 4.879265% | **0.747746%** |
   | BDF2 | 4.800516% | **0.840435%** |

   The second-order-family schemes look worse by **worst-case** error but better by **median** error. The same distinction holds at the lattice setting.

   This is consistent with A13’s reference-limitation warning. It does **not** isolate cancellation of BE temporal errors as the cause.

   **H1-3D and H2-3D fail for both settings.** No eligible arm satisfies every required inequality. These failures remain reference-dependent; they do not negate the anchor-discrepancy improvements.

Additional report findings:

- **Calibration passes:** all 16 choices reproduce the registered strict **<1% of tight-solve ST error** rule. Newton tolerance is $10^{-4}$ for every $5\Delta t_0$ configuration and CN at $2\Delta t_0$; otherwise $10^{-6}$. Linear tolerance is $10^{-8}$. No fallback or unresolved calibration.
- **Missing order outlier counts:** the 3D tables show valid counts and medians but omit in-band/outlier counts required by A1.3. This omission conceals why lattice LSPG-BDF2 fails.
- **Timing labels need precision:** distinguish median paired ratios from ratios of median milliseconds. They are visibly different here.
- **3D plots omit valid fine-step accuracy data:** `report_extra.py` filters plotted rows through timing eligibility, dropping $\Delta t_0/4$ and $\Delta t_0/8$ even when their accuracy/anchor data are valid. Label the plots as the timed subset or include those points.
- The 3D reference-error columns should explicitly carry **%**. “Same two-step methods” in the FOM introduction should read “same time-integration methods”; BE and CN are one-step methods.

**Accept the numerical results and registered outcomes. Revise the headline scope before publication; do not promote FOM-CN or lattice LSPG-BDF2 to established second order.**