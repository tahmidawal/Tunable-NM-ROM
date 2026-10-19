**Overall: RESULTS-CONFIRMED.** The eight 256² domination conclusions reproduce independently. The corrected 1024² accounting and dimension-level budget wording are correct.

1. **CORRECT — A: eligibility, provenance and X errors.** Verified verdict, frozen-membership and tuning/Stage-B hashes, all 46 baseline coefficient files, and 63 staged-source manifest entries. Independently checked the 38 reference metadata files against their manifest. Eligible cases are exactly `dev6|0–5` and `val32|0–31`; ε_ref ranges from `8.669938305e-7` to `9.519552440e-6`. Every reported X maximum and median reproduces from `p1_analysis.json`.

2. **CORRECT — A: member Q1 and replay.** Independently reaggregated **3,588 production/anchor certificates**, including stepwise stationarity, exit reasons and damping. Production has zero nonstationary steps. Q1 gives **44 passes and two failures**:

   | Failed member | Maximum production–tight distance |
   |---|---:|
   | ECSW r32/κ4, CNR/common | 0.000283290852 |
   | Q-DEIM r32/p32, CNR/common | 0.000299835236 |

   Both exceed `2.5e-4`. Largest passing distance is `0.000245270480`; largest tight–tighter distance is `2.387911180e-8`.

   Reconstructed distances from saved coefficients using POD orthonormality and reference initial norms. Maximum discrepancies from recorded distances are `6.10e-17` and `2.48e-17`, with no gate disagreement. This relies on the recorded orthonormality certificate, `3.22e-15`; the offloaded full basis was not reloaded.

   All 46 coefficient replays against tuning reproduce; maximum discrepancy is `1.887595282e-13`. All eight Stage-B G-ours gates pass; maximum error-replay discrepancy is `2.130041639e-13`.

3. **CORRECT — A: timing contract and pre-A44.1 code.** Checked **352 pairs, 6,336 triples**, six cases and three repeats per pair. Every attempt passes; no retries were needed. Maximum drift is `0.090528948`; all recorded timed-output discrepancies are zero.

   Measurements share job **6003086**, A100 80GB PCIe, UUID `GPU-ef0906da-ee15-780e-c19b-39842e183c5c`. Logs confirm GPU execution and successful completion; the batch script enables f64/highest precision. Staged timing code warms both queries, burns before every measurement and synchronizes outputs.

   The staged `stageb.py` differs by the missing dt canonicalization only. **All 46 members use 0.005**, and `float("0.005") == 0.05/10`. No member was affected.

4. **CORRECT — A: fronts, tie-breaks, role picks and statistics.** Independent NumPy calculations—not calls to `analyze.py` or `stats.py`—reproduce every X-specific front, registered pick, conditional envelope, 10,000-resample seed-0 bootstrap interval, equivalence calculation and pair label.

   Here E256/4-BE means ECSW r256/κ4/BE; other E entries use CNR. Q512 means Q-DEIM r512/p1024/CNR. All picks use common solvers, central snapshots and Δt = 0.005.

   | X | Front size | B_acc | Time ratio [95% CI] | B_cost | Time ratio [95% CI] |
   |---|---:|---|---|---|---|
   | R128 BE | 18 | E256/4-BE | .719352 [.662187, .798415] | E256/8 | .906020 [.881983, .944568] |
   | R128 CNR | 17 | E256/4-BE | .719781 [.656861, .791041] | E256/8 | .893676 [.875451, .930725] |
   | R256 BE | 20 | E256/4 | .209306 [.202933, .214647] | Q512 | .577318 [.572703, .591781] |
   | R256 CNR | 19 | E384/4 | .308406 [.299615, .316746] | Q512 | .583215 [.573890, .595002] |
   | R384 BE | 20 | E256/8 | .134285 [.128672, .136619] | Q512 | .312965 [.308447, .317990] |
   | R384 CNR | 19 | E384/4 | .165857 [.162455, .169390] | Q512 | .313055 [.310747, .319532] |
   | R384nom BE | 20 | E256/8 | .134736 [.129647, .136796] | Q512 | .312400 [.308250, .317376] |
   | R384nom CNR | 20 | E384/4 | .166042 [.162530, .168856] | Q512 | .312217 [.309729, .319003] |

   Both picks dominate in every row. Across **122 unique assessed pairs**, labels are **59 baseline dominates and 63 neither dominates**; no matches, ours-dominates or unresolved outcomes. Per-X assessed counts are `19, 20, 18, 11, 16, 11, 16, 11`. All eight cell lines reproduce.

5. **CORRECT — A: restriction propagation; NEEDS-RESTATEMENT — its interpretation.** Saturation fails at r384:

   `0.000895807224 / 0.000688355291 = 1.301373340 > 1.10`.

   The 256² cell then installs the prescribed **all-step basis**, using 29,376 snapshots. Its recorded adequacy ratios pass, maximum `0.951397073`. All **164 tuning configurations**, including **80 ECSW configurations**, are complete; recorded ECSW qualities are `ok`.

   A35.1 nevertheless requires carrying the failed-saturation restriction and withholding favourable/neutral conclusions. Thus the 63 “neither dominates” outcomes remain withheld; the eight observed domination conclusions remain reportable.

   **The restriction must not imply that the installed 256² basis is the failed 12-step basis.** That snapshot limitation applies to 1024². The 256² failure triggered the registered remedy; it does not invalidate the observed baseline dominations.

6. **CORRECT — A: V/T accounting.**

   | Quantity | Hours |
   |---|---:|
   | Training-only construction T | 0.462041941 |
   | Tuning V | 0.535609306 |
   | Stage-B V | 1.938114576 |
   | Cumulative V | **2.473723883** |
   | Allocation upper bound | **3.0094** |

   Both attempts finished, with no active phase or extra attempt. `V_cumulative_is_lower_bound=false` is correct. The limiting comparator is separately recorded at `0.014527297 h`.

7. **CORRECT — B: corrected 1024² accounting; numerical results unchanged.** Independent phase sums give:

   - Selected-attempt V: `2.022398995 h`.
   - Cancelled `s2d1024a` completed V: `1.034590999 h`.
   - **Cumulative V ≥ 3.056989995 h**, conventionally rounded to ≥3.057 h.
   - **Allocation upper bound: 5.2174 h**.

   The cancelled attempt retains an unfinished `aba` phase, so the cumulative lower-bound flag is correct.

   Compared against the pre-correction verdict at commit `8ad7e649f`: **every X block and eligible-case entry is exactly unchanged**. Only accounting changed; fronts, picks, errors, intervals and labels match results audit 1’s tables.

8. **CORRECT — C: dimension-level budget statements.**

   | Dimension | Baseline V bounds, h | Ours V, h | Ratio bounds | Correct wording |
   |---|---:|---:|---:|---|
   | 2D | [5.530713877, 8.2268] | 4.767744444 | [1.160027334, 1.725511947] | **Unresolved** |
   | 3D | [8.835878088, 11.4395] | 17.2983 | [.510794592, .661307759] | **Unequal-budget, descriptive** |

   The 2D total includes both cells and the cancelled 1024² attempt. Ours’ totals reproduce from `ours_V.json` item sums. The 2D interval crosses the upper band boundary; both 3D bounds lie below 0.8. The report’s wording follows the specified logic.

9. **NEEDS-RESTATEMENT — audit and headline scope.** The report’s blanket statement that Q1 distances were not reconstructed is now outdated for 256²: this audit reconstructed them through coefficient norms, subject to the recorded orthonormality certificate. Conclusions remain **developmental, in-sample, unadjusted and conditional on ε_ref**. “Dominates” concerns query time and worst-case error; winners were selected on Stage-B outcomes.

No files were changed, jobs launched, GPU work performed or scratch files created.