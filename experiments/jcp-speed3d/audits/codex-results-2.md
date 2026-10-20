**Overall: NOT READY.** A9 gets the G2 totals and numerical significance right, but misidentifies which configurations failed. That restatement needs correction.

1. **m1 calibration — CORRECT.**  
   `select_s3.py` regenerates m1 byte-for-byte with `calib_large=[0,1,2,3]`. Compared with the round-1 configuration at `ebca2342d`, this is the **only content change**. [s3modes.py:448](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:448) passes that population to calibration at L=256/384. The certificate’s config hash matches:
   `3335f632469dd619def624efd5d10dc4ff8d07ee30e89198f4ad76f36b0bd8fb`.

2. **G2 — NEEDS-RESTATEMENT.**  
   Recomputed **15 failures / 288,944 fixed-phase FS steps**, maximum gamma **3.336924e-12**, maximum difference **1.889957e-12**. No failing comparison changes stationarity classification. However, A9’s configuration attribution is wrong:

   | Job | Setting / rule | Scheme / step / sweeps | Failures |
   |---|---|---|---:|
   | g2 | W768 / lat32768 | BDF2 / .01 / K2 | 3 |
   | g2 | W768 / lat32768 | BDF2 / .0125 / K2 | 1 |
   | g2 | W768 / lat16384 | BDF2 / .01 / K2 | 1 |
   | g3 | W1024 / lat32768 | CN / .01 / K2 | 5 |
   | g3 | W1024 / lat65536 | CN / .01 / K2 | 3 |
   | g3 | W1024 / lat65536 | CN / .0125 / K2 | 2 |

   The eligibility statement is correct: [s3ctx.py:157](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3ctx.py:157) derives `nonstationary` from certified gamma; [analyze.py:112](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze.py:112) uses that count for E-b. Every stored count matches independent recomputation. Recorded `steps.gn` does not determine E-b.

3. **H1.1 — CORRECT.**  
   Raw replay records give **71 mixed + 1 baseline already stationary**. The exception is W256 case 5, distance-only, baseline gamma **0.0005209058390663031**. All 71 mixed records satisfy the stated sweep/trust successes and damping-reset failure.

4. **A8 resources and selection integrity — CORRECT, as estimates.**  
   Calling the committed projection functions reproduces **p1 7.037496971 h** and **m1 10.202453419 h**, exactly matching their certificates; both config hashes match. Replacing compile-dominated L128 extrapolation with the measured L384 **12.678041379 s** is a reasonable budgeting proxy, although monotonic runtime is not guaranteed. A8 explicitly disclaims a demonstrated upper bound. The old L⁴ description remains in the function docstring—a nonblocking documentation inconsistency.

   A8/A9 change resource estimation and restore the registered calibration population. **They do not change screening selections or verdicts:** analysis code, analysis results, selection JSON, FOM matches, and p1 configuration are unchanged.

**Required correction:** replace A9’s inaccurate G2 configuration attribution with the table above, retaining the literal-failure qualification.

Audit was read-only and CPU-only. No repository files, LAB-LOG, or main were changed; no jobs were submitted.