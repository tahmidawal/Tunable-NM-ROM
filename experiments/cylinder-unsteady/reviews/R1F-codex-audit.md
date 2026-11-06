
codex
1. **INFO — Retention verified.** Files: `retention/*/results/*/{snapshots_*.npz,*state files}`, corresponding `stages.json` and `REMOTE-SHA256SUMS`. Independently hashed **107 snapshot chunks and 13 states**, totaling **17,150,243,824 bytes**. Every file matches both recorded hashes; no declared snapshot/state is missing. This verifies agreement with the saved cluster manifests; live cluster bytes were not re-read.

2. **INFO — R1b passes at dt₀ = 1/3200.** Files: `artifacts/r1b-compare01/results/compare/*.json` and underlying `r1b-study-a2`/`r1b-study-b2` histories. Independent coarse-grid recomputation reproduces:

   | Comparison | e_D | e_L |
   |---|---:|---:|
   | 800/1600 | 1.071361212e-4 | 8.506924739e-4 |
   | 1600/3200 | 2.697100983e-5 | 2.141850789e-4 |
   | 3200/6400 | 6.917273452e-6 | 5.373716959e-5 |
   | Startup 800 | 6.870396011e-6 | 1.850992774e-4 |
   | Startup 1600 | 1.702504307e-6 | 4.608204557e-5 |
   | Startup 3200 | 4.302073789e-7 | 1.149970252e-5 |
   | Roundoff 1600 | 6.382773371e-14 | 1.624256285e-13 |

   Force grids contain exactly 279, 559 or 1119 eligible samples. All comparisons have 141 saves, the same initial-state hash, and the same 0.35-second window. Field-error series are finite, nonnegative, and their maxima equal the reported maxima. Independent gate logic selects 3200; applicable empirical orders are **1.963–1.998**. Field norms themselves were not recomputed because FE mass matrices are unavailable.

3. **MINOR — Narrative omits coarse startup failures; decision unaffected.** File: coordinator `DESIGN-G5.md`, amendment G5-A6, and corresponding lab-log summary. The description that coarser candidates “fail only the lift tolerance” omits that **both also fail the startup lift criterion**: 1.85099e-4 and 4.60820e-5 exceed its 4e-5 threshold. The saved gate JSON correctly records these failures.

4. **INFO — R1c passes.** File: `artifacts/r1c-space02/results/compare/space_3200.json` and underlying histories. Recomputed **e_D = 4.676099827e-6**, **e_L = 2.326091784e-5**. Field-series consistency checks give e_u,max = 3.202422670e-5 and e_p,max = 4.495562174e-5. All four meet their tolerances; identity and coverage checks pass.

5. **INFO — R1d passes.** Files: `r1-record-re100/results/record_re100_p4/history.npz`, `artifacts/R1DE.json`, coordinator `sources/dfg2d2_reference.json`. Independent lift-minimum detection and quadratic fitting recover **four complete cycles**. Recomputed per-cycle statistics and aggregate mean/min/max/count agree with `R1DE.json` within 2e-10. Means and Featflow discrepancies are:

   | Statistic | Recomputed mean | Discrepancy |
   |---|---:|---:|
   | C_D,max | 3.227736510803 | 1.06520e-4 relative |
   | C_L,max | 0.986987343182 | 4.04360e-4 absolute |
   | St | 0.301842758306 | 5.80365e-6 relative |
   | Δp, Featflow convention | 2.483351374182 | 5.31506e-5 relative |

   All pass. The required 1996 ranges also pass, including Δp = 2.485251863950 under that source’s convention. The lift maximum remains outside its explicitly non-gating historical range.

6. **INFO — R1e passes for all five Reynolds numbers.** Files: `tasks/population.json`, each `r1-record-re*/results/record_re*_p4/`, and linked settle states. Every record completed with **681 saves at 1/400 seconds** and **171 diagnostics**. Configurations match P4/P3, dt = 1/3200, ν = 0.1/Re, `frozen_oseen`, K = 40 and float64. All continuation, cross-Re predecessor and settle→record hashes match. Cumulative settle durations are **5, 5.75, 1.5, 5.75 and 4.5 seconds** for Re 90/95/100/105/110. Maximum divergence across these quality checks is **3.90442e-4**, mass mismatch **9.47752e-16**, and linear residual **4.58032e-16**. Unsealed settle periodicity independently passes; Re 105’s completion evidence was checked without computing cycle statistics.

7. **INFO — No additional physical or array anomaly found within the permitted scope.** Unsealed retained arrays are finite. Unsealed histories contain only the documented unavailable startup reaction-force entries; no energy blow-up or material boundary discontinuity was found. The largest boundary observation mismatch is 4.44e-16. **Re 105 histories and field arrays were not opened**; only permitted quality checks were performed. Repository files remained unchanged; scratch output stayed in the authorized directory.

R1f: PASS
tokens used
80,049
1. **INFO — Retention verified.** Files: `retention/*/results/*/{snapshots_*.npz,*state files}`, corresponding `stages.json` and `REMOTE-SHA256SUMS`. Independently hashed **107 snapshot chunks and 13 states**, totaling **17,150,243,824 bytes**. Every file matches both recorded hashes; no declared snapshot/state is missing. This verifies agreement with the saved cluster manifests; live cluster bytes were not re-read.

2. **INFO — R1b passes at dt₀ = 1/3200.** Files: `artifacts/r1b-compare01/results/compare/*.json` and underlying `r1b-study-a2`/`r1b-study-b2` histories. Independent coarse-grid recomputation reproduces:

   | Comparison | e_D | e_L |
   |---|---:|---:|
   | 800/1600 | 1.071361212e-4 | 8.506924739e-4 |
   | 1600/3200 | 2.697100983e-5 | 2.141850789e-4 |
   | 3200/6400 | 6.917273452e-6 | 5.373716959e-5 |
   | Startup 800 | 6.870396011e-6 | 1.850992774e-4 |
   | Startup 1600 | 1.702504307e-6 | 4.608204557e-5 |
   | Startup 3200 | 4.302073789e-7 | 1.149970252e-5 |
   | Roundoff 1600 | 6.382773371e-14 | 1.624256285e-13 |

   Force grids contain exactly 279, 559 or 1119 eligible samples. All comparisons have 141 saves, the same initial-state hash, and the same 0.35-second window. Field-error series are finite, nonnegative, and their maxima equal the reported maxima. Independent gate logic selects 3200; applicable empirical orders are **1.963–1.998**. Field norms themselves were not recomputed because FE mass matrices are unavailable.

3. **MINOR — Narrative omits coarse startup failures; decision unaffected.** File: coordinator `DESIGN-G5.md`, amendment G5-A6, and corresponding lab-log summary. The description that coarser candidates “fail only the lift tolerance” omits that **both also fail the startup lift criterion**: 1.85099e-4 and 4.60820e-5 exceed its 4e-5 threshold. The saved gate JSON correctly records these failures.

4. **INFO — R1c passes.** File: `artifacts/r1c-space02/results/compare/space_3200.json` and underlying histories. Recomputed **e_D = 4.676099827e-6**, **e_L = 2.326091784e-5**. Field-series consistency checks give e_u,max = 3.202422670e-5 and e_p,max = 4.495562174e-5. All four meet their tolerances; identity and coverage checks pass.

5. **INFO — R1d passes.** Files: `r1-record-re100/results/record_re100_p4/history.npz`, `artifacts/R1DE.json`, coordinator `sources/dfg2d2_reference.json`. Independent lift-minimum detection and quadratic fitting recover **four complete cycles**. Recomputed per-cycle statistics and aggregate mean/min/max/count agree with `R1DE.json` within 2e-10. Means and Featflow discrepancies are:

   | Statistic | Recomputed mean | Discrepancy |
   |---|---:|---:|
   | C_D,max | 3.227736510803 | 1.06520e-4 relative |
   | C_L,max | 0.986987343182 | 4.04360e-4 absolute |
   | St | 0.301842758306 | 5.80365e-6 relative |
   | Δp, Featflow convention | 2.483351374182 | 5.31506e-5 relative |

   All pass. The required 1996 ranges also pass, including Δp = 2.485251863950 under that source’s convention. The lift maximum remains outside its explicitly non-gating historical range.

6. **INFO — R1e passes for all five Reynolds numbers.** Files: `tasks/population.json`, each `r1-record-re*/results/record_re*_p4/`, and linked settle states. Every record completed with **681 saves at 1/400 seconds** and **171 diagnostics**. Configurations match P4/P3, dt = 1/3200, ν = 0.1/Re, `frozen_oseen`, K = 40 and float64. All continuation, cross-Re predecessor and settle→record hashes match. Cumulative settle durations are **5, 5.75, 1.5, 5.75 and 4.5 seconds** for Re 90/95/100/105/110. Maximum divergence across these quality checks is **3.90442e-4**, mass mismatch **9.47752e-16**, and linear residual **4.58032e-16**. Unsealed settle periodicity independently passes; Re 105’s completion evidence was checked without computing cycle statistics.

7. **INFO — No additional physical or array anomaly found within the permitted scope.** Unsealed retained arrays are finite. Unsealed histories contain only the documented unavailable startup reaction-force entries; no energy blow-up or material boundary discontinuity was found. The largest boundary observation mismatch is 4.44e-16. **Re 105 histories and field arrays were not opened**; only permitted quality checks were performed. Repository files remained unchanged; scratch output stayed in the authorized directory.

R1f: PASS
