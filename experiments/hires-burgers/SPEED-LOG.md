# SPEED-LOG — hires-burgers

Running log of the profile → hypothesis → fix → re-measure loop (protocol §"Standing mandate").
One row per hypothesis. "Kept" requires a parity gate against the unoptimised path and a
same-job before/after timing; numbers are copied from the named `summary.json`, never typed
from memory. Status `pending` means the measuring job has not landed.

## Starting point (orientation from audited earlier panels, not new claims)

Burgers 2D, frozen checkpoint $K=16$, $R=512$. b-panel `bpn301` (A100, $256^2$): $q=0$ EQ
43.4 ms at tol $10^{-3}$; $q=256$, $M=1088$ EQ (b-eqtop rule, $m=2560$) 466 ms at $10^{-3}$ /
746 ms at $10^{-6}$; dense 3940 ms. b-speed: the query is kernel-launch bound, ≈17 µs per step
+ 141 µs per LM iteration at $q=0$; `fuse` 1.25×, `lean` fold 1.15×, block Gauss–Jordan loses.

Static reading of the audited $q>0$ path (`topfix.make_query` → `varpro.make_block_lm`), before
any measurement — this is the hypothesis list the first job tests:

| # | where the accurate rung's time can go | evidence before measuring |
|---|---|---|
| H1 | forward-mode Jacobian pushes $K+q=272$ tangents through the $R=512$-wide stencil bank: $m\cdot5\cdot R\cdot(K+q) \approx 1.8$ GFLOP per Jacobian, more than the $M\times m\times(K+q)=0.76$ GFLOP test projection | code reading |
| H2 | every accepted LM iteration evaluates the residual twice and the Jacobian once under `lax.cond` (b-speed `fuse`) | b-speed measured 1.25× at $q=0$ |
| H3 | `blocks()` per time step: two extra forward-mode Jacobians ($z$ and $y$ blocks) purely for a stationarity diagnostic that equals the LM's own exit gradient | code reading; ≈ +1 Jacobian per step on 2–3 iterations per step |
| H4 | QR of $R_cC$ ($2304\times q$) recomputed inside every query's initial fit | code reading |
| H5 | LU of the $272\times272$ damped normal matrix (cuSOLVER launch-bound) — Cholesky is the cheaper factorisation of an SPD matrix | to measure |
| H6 | decode as six separate $G h$ matvecs reads $G$ six times: 17 GB at $2048^2$, 68 GB at $4096^2$ | bandwidth arithmetic: ≈6×68 GB / 4.8 TB/s ≈ 85 ms at $4096^2$ vs ≈15–20 ms fused |
| H7 | test count: $M=544$ at $q=256$ halves the projection and the Gram cost; b-qxm measured 0.757 % (vs 0.519 % at $M=1088$) at $256^2$ | b-qxm `bqx201` |
| H8 | iterations per step (2–3 at tol $10^{-3}$, 5 at $10^{-6}$): predictor quality | b-panel rows |
| H9 | trust-radius REJECTIONS cost a full iteration each (25–114 per query, 15–26 % of all iterations) | hb2k01 quick rows |
| H10 | every time step restarts the damping at 1e-6 and re-learns it | code reading + hb2k01 retries |

## Log

| date | job | hypothesis | change | parity gate | before → after (same job) | kept? |
|---|---|---|---|---|---|---|
| 2026-09-20 | hb2k01 (4054951, H200, 2048²) | H1–H4, H6 | `q0_M64_xfer_g0p001_fast`: b-speed C1 vs audited `q0_M64_xfer_g0p001_base` | fields 7.7e-13, integers identical True | 32.04 → 25.10 ms (1.28×) | **kept** |
| 2026-09-20 | hb2k01 (4054951, H200, 2048²) | H1–H4, H6 | `q0_M64_xfer_g1em06_fast`: b-speed C1 vs audited `q0_M64_xfer_g1em06_base` | fields 7.9e-13, integers identical True | 39.76 → 29.49 ms (1.35×) | **kept** |
| 2026-09-20 | hb2k01 (4054951, H200, 2048²) | H1–H4, H6 | `q128_M576_xfer_g0p001_fast`: hfast fold+ajac+fuse+hoist+nodiag+decfuse vs audited `q128_M576_xfer_g0p001_base` | fields 5.4e-14, integers identical True | 133.92 → 115.21 ms (1.16×) | **kept** |
| 2026-09-20 | hb2k01 (4054951, H200, 2048²) | H1–H4, H6 | `q128_M576_xfer_g1em06_fast`: hfast fold+ajac+fuse+hoist+nodiag+decfuse vs audited `q128_M576_xfer_g1em06_base` | fields 5.4e-14, integers identical True | 168.09 → 145.17 ms (1.16×) | **kept** |
| 2026-09-20 | hb2k01 (4054951, H200, 2048²) | H1–H4, H6 | `q256_M544_xfer_g0p001_fast`: hfast fold+ajac+fuse+hoist+nodiag+decfuse vs audited `q256_M544_xfer_g0p001_base` | fields 6.2e-14, integers identical True | 352.31 → 314.51 ms (1.12×) | **kept** |
| 2026-09-20 | hb2k01 (4054951, H200, 2048²) | H1–H4, H6 | `q256_M544_xfer_g1em06_fast`: hfast fold+ajac+fuse+hoist+nodiag+decfuse vs audited `q256_M544_xfer_g1em06_base` | fields 9.6e-14, integers identical True | 602.57 → 541.54 ms (1.11×) | **kept** |
| 2026-09-20 | hb2k01 (4054951, H200, 2048²) | H1–H4, H6 | `q256_M1088_xfer_g0p001_fast`: hfast fold+ajac+fuse+hoist+nodiag+decfuse vs audited `q256_M1088_xfer_g0p001_base` | fields 6.2e-14, integers identical True | 310.57 → 270.03 ms (1.15×) | **kept** |
| 2026-09-20 | hb2k01 (4054951, H200, 2048²) | H1–H4, H6 | `q256_M1088_xfer_g1em06_fast`: hfast fold+ajac+fuse+hoist+nodiag+decfuse vs audited `q256_M1088_xfer_g1em06_base` | fields 7.2e-14, integers identical True | 489.18 → 426.33 ms (1.15×) | **kept** |
| 2026-09-20 | hb2k01 | H5 | Cholesky instead of LU on the 272×272 damped normal matrix, `q256_M1088_xfer_g0p001_fast_chol` | fields 8.4e-14 vs audited base, integers identical True | 270.03 → 161.47 ms (1.67×) | **kept** |
| 2026-09-20 | hb2k01 | H5 | Cholesky instead of LU on the 272×272 damped normal matrix, `q256_M1088_lat64_g0p001_fast_chol` | same error to 4 digits (0.5980 % vs 0.5980 %); no audited twin for this rule | 304.07 → 195.18 ms (1.56×) | **kept** |
| 2026-09-20 | hb2k01 | H5 | Cholesky instead of LU on the 272×272 damped normal matrix, `q256_M1088_scaled_g0p001_fast_chol` | same error to 4 digits (0.5915 % vs 0.5915 %); no audited twin for this rule | 255.58 → 147.94 ms (1.73×) | **kept** |
| 2026-09-20 | hb2k01 | H5 | Cholesky instead of LU on the 272×272 damped normal matrix, `q256_M1088_xfer_g1em06_fast_chol` | fields 7.3e-14 vs audited base, integers identical True | 426.33 → 244.92 ms (1.74×) | **kept** |
| 2026-09-20 | hb2k01 | H5 | Cholesky instead of LU on the 272×272 damped normal matrix, `q256_M1088_lat64_g1em06_fast_chol` | same error to 4 digits (0.5979 % vs 0.5979 %); no audited twin for this rule | 478.36 → 295.21 ms (1.62×) | **kept** |
| 2026-09-20 | hb2k01 | H5 | Cholesky instead of LU on the 272×272 damped normal matrix, `q256_M1088_scaled_g1em06_fast_chol` | same error to 4 digits (0.5915 % vs 0.5915 %); no audited twin for this rule | 404.33 → 225.03 ms (1.80×) | **kept** |
| 2026-09-20 | hb2k01 | H7 | $M=544$ instead of $1088$ at $q=256$ (lat64, tol 1e-3, LU) | accuracy measured: 0.8896 % vs 0.5980 % | 304.07 → 332.09 ms (0.92×): SLOWER and less accurate (more LM iterations: 342 vs 285 median) | **reverted** |
| 2026-09-20 | hb2k01 | H8 | tolerance 1e-3 instead of 1e-6 (lat64, LU) | error 0.5980 % vs 0.5979 %; zero stalled exits in both | 478.36 → 304.07 ms (1.57×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H5 | Cholesky: `q128_M576_lat64_g0p001_fast` → `q128_M576_lat64_g0p001_fast_chol` | factorisation only; identical integers: error 1.0748 → 1.0748 %, stalled 0 → 0, rejected trial steps 164 → 164, median LM iterations 166 → 166 | 131.97 → 102.73 ms (1.28×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H9 | `clip` (shorten an over-long z-step onto the trust radius instead of rejecting it): `q128_M576_lat64_g0p001_fast_chol` → `q128_M576_lat64_g0p001_fast_chol_clip` | algorithmic arm, error measured directly: error 1.0748 → 1.0748 %, stalled 0 → 0, rejected trial steps 164 → 0, median LM iterations 166 → 124 | 102.73 → 74.35 ms (1.38×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H10 | `lamcarry` ALONE: `q128_M576_lat64_g0p001_fast_chol` → `q128_M576_lat64_g0p001_fast_chol_lamcarry` | algorithmic arm: error 1.0748 → 1.0748 %, stalled 0 → 0, rejected trial steps 164 → 158, median LM iterations 166 → 159 | 102.73 → 97.62 ms (1.05×) | **reverted (more rejections without clip)** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H10 | `lamcarry` on top of `clip`: `q128_M576_lat64_g0p001_fast_chol_clip` → `q128_M576_lat64_g0p001_fast_chol_clip_lamcarry` | algorithmic arm: error 1.0748 → 1.0748 %, stalled 0 → 0, rejected trial steps 0 → 0, median LM iterations 124 → 124 | 74.35 → 74.35 ms (1.00×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H5 | Cholesky: `q256_M1088_lat64_g0p001_fast` → `q256_M1088_lat64_g0p001_fast_chol` | factorisation only; identical integers: error 0.5980 → 0.5980 %, stalled 0 → 0, rejected trial steps 378 → 378, median LM iterations 285 → 285 | 302.74 → 193.79 ms (1.56×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H9 | `clip` (shorten an over-long z-step onto the trust radius instead of rejecting it): `q256_M1088_lat64_g0p001_fast_chol` → `q256_M1088_lat64_g0p001_fast_chol_clip` | algorithmic arm, error measured directly: error 0.5980 → 0.5980 %, stalled 0 → 0, rejected trial steps 378 → 0, median LM iterations 285 → 181 | 193.79 → 137.27 ms (1.41×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H10 | `lamcarry` ALONE: `q256_M1088_lat64_g0p001_fast_chol` → `q256_M1088_lat64_g0p001_fast_chol_lamcarry` | algorithmic arm: error 0.5980 → 0.5980 %, stalled 0 → 0, rejected trial steps 378 → 565, median LM iterations 285 → 338 | 193.79 → 222.11 ms (0.87×) | **reverted (more rejections without clip)** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H10 | `lamcarry` on top of `clip`: `q256_M1088_lat64_g0p001_fast_chol_clip` → `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry` | algorithmic arm: error 0.5980 → 0.5980 %, stalled 0 → 0, rejected trial steps 0 → 0, median LM iterations 181 → 168 | 137.27 → 130.69 ms (1.05×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H5 | Cholesky: `q256_M2176_lat64_g0p001_fast` → `q256_M2176_lat64_g0p001_fast_chol` | factorisation only; identical integers: error 0.4538 → 0.4538 %, stalled 0 → 0, rejected trial steps 343 → 343, median LM iterations 230 → 230 | 271.01 → 183.57 ms (1.48×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H9 | `clip` (shorten an over-long z-step onto the trust radius instead of rejecting it): `q256_M2176_lat64_g0p001_fast_chol` → `q256_M2176_lat64_g0p001_fast_chol_clip` | algorithmic arm, error measured directly: error 0.4538 → 0.4538 %, stalled 0 → 0, rejected trial steps 343 → 10, median LM iterations 230 → 170 | 183.57 → 147.23 ms (1.25×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H10 | `lamcarry` ALONE: `q256_M2176_lat64_g0p001_fast_chol` → `q256_M2176_lat64_g0p001_fast_chol_lamcarry` | algorithmic arm: error 0.4538 → 0.4538 %, stalled 0 → 0, rejected trial steps 343 → 452, median LM iterations 230 → 232 | 183.57 → 180.82 ms (1.02×) | **reverted (more rejections without clip)** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H10 | `lamcarry` on top of `clip`: `q256_M2176_lat64_g0p001_fast_chol_clip` → `q256_M2176_lat64_g0p001_fast_chol_clip_lamcarry` | algorithmic arm: error 0.4538 → 0.4538 %, stalled 0 → 0, rejected trial steps 10 → 10, median LM iterations 170 → 157 | 147.23 → 138.80 ms (1.06×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H9+H10 | q=0 through hfast with clip+lamcarry: `q0_M64_scaled_g0p001_fast` → `q0_M64_scaled_g0p001_fast_clip_lamcarry` | algorithmic arm: error 2.3721 → 2.3721 %, stalled 0 → 0, rejected trial steps 77 → 0, median LM iterations 133 → 114 | 33.19 → 29.48 ms (1.13×) | **kept** |
| 2026-09-20 | hb2k02 (4071616, H200, 2048², dev6) | H7b | M=544 with clip vs M=1088 with clip (q=256): `q256_M1088_lat64_g0p001_fast_chol_clip` → `q256_M544_lat64_g0p001_fast_chol_clip` | accuracy measured: error 0.5980 → 0.8653 %, stalled 0 → 0, rejected trial steps 0 → 0, median LM iterations 181 → 188 | 137.27 → 124.15 ms (1.11×) | **kept as the cheaper ≤1 % rung only; M=1088 stays the accurate rung** |

| 2026-09-21 | hb4k03 (4071625, H200, 4096², dev6) | H5 | Cholesky: `q256_M1088_lat64_g0p001_fast` → `q256_M1088_lat64_g0p001_fast_chol` | algorithmic arm, error measured directly: error 0.6043 → 0.6043 %, stalled 0 → 0, rejected trial steps 378 → 378, median LM iterations per query 284 → 284, worst exit stationarity 9.93e-04 → 9.93e-04, certified True → True | 315.78 → 207.01 ms (1.53×); host-inclusive 559.4 → 450.1 ms | **kept** (replicates hb2k02 at 4096²) |
| 2026-09-21 | hb4k03 (4071625, H200, 4096², dev6) | H9 | `clip` (shorten an over-long z-step onto the trust radius): `q256_M1088_lat64_g0p001_fast_chol` → `q256_M1088_lat64_g0p001_fast_chol_clip` | algorithmic arm, error measured directly: error 0.6043 → 0.6043 %, stalled 0 → 0, rejected trial steps 378 → 0, median LM iterations per query 284 → 181, worst exit stationarity 9.93e-04 → 9.94e-04, certified True → True | 207.01 → 151.08 ms (1.37×); host-inclusive 450.1 → 393.3 ms | **kept** (replicates hb2k02 at 4096²) |
| 2026-09-21 | hb4k03 (4071625, H200, 4096², dev6) | H10 | `lamcarry` (carry the damping between steps): `q256_M1088_lat64_g0p001_fast_chol_clip` → `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry` | algorithmic arm, error measured directly: error 0.6043 → 0.6043 %, stalled 0 → 0, rejected trial steps 0 → 0, median LM iterations per query 181 → 168.5, worst exit stationarity 9.94e-04 → 1.00e-03, certified True → True | 151.08 → 144.66 ms (1.04×); host-inclusive 393.3 → 385.0 ms | **kept** (replicates hb2k02 at 4096²) |
| 2026-09-21 | hb4k03 (4071625, H200, 4096², dev6) | H5+H9+H10 | chol+clip+lamcarry cumulative: `q256_M1088_lat64_g0p001_fast` → `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry` | algorithmic arm, error measured directly: error 0.6043 → 0.6043 %, stalled 0 → 0, rejected trial steps 378 → 0, median LM iterations per query 284 → 168.5, worst exit stationarity 9.93e-04 → 1.00e-03, certified True → True | 315.78 → 144.66 ms (2.18×); host-inclusive 559.4 → 385.0 ms | **kept** (replicates hb2k02 at 4096²) |
| 2026-09-21 | hb4k03 (4071625, H200, 4096², dev6) | H9+H10 | clip+lamcarry: `q0_M64_scaled_g0p001_fast` → `q0_M64_scaled_g0p001_fast_clip_lamcarry` | algorithmic arm, error measured directly: error 2.4157 → 2.4158 %, stalled 0 → 0, rejected trial steps 77 → 0, median LM iterations per query 133 → 113.5, worst exit stationarity 9.84e-04 → 9.80e-04, certified True → True | 46.49 → 43.04 ms (1.08×); host-inclusive 288.5 → 285.2 ms | **kept** (replicates hb2k02 at 4096²) |

| 2026-09-21 | hb4k04 (4079320, H200, 4096², dev6) | H11 | `pred2` (quadratic-extrapolation guard, one batched residual): `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry` → `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2` | algorithmic arm, error measured directly: error 0.6043 → 0.6043 %, stalled 0 → 0, rejected trial steps 0 → 7, median LM iterations per query 168.5 → 136.5, worst exit stationarity 1.00e-03 → 9.97e-04, certified True → True | 145.73 → 127.18 ms (1.15×); host-inclusive 393.6 → 373.0 ms | **kept** (same error, fewer iterations) |
| 2026-09-21 | hb4k04 (4079320, H200, 4096², dev6) | H11 | `pred2` (quadratic-extrapolation guard, one batched residual): `q256_M544_lat64_g0p001_fast_chol_clip_lamcarry` → `q256_M544_lat64_g0p001_fast_chol_clip_lamcarry_pred2` | algorithmic arm, error measured directly: error 0.8754 → 0.8754 %, stalled 0 → 0, rejected trial steps 0 → 122, median LM iterations per query 171 → 133, worst exit stationarity 9.97e-04 → 9.96e-04, certified True → True | 129.48 → 112.11 ms (1.15×); host-inclusive 372.9 → 358.1 ms | **kept** (same error, fewer iterations) |
| 2026-09-21 | hb4k04 (4079320, H200, 4096², dev6) | H11 | `pred2` (quadratic-extrapolation guard, one batched residual): `q128_M576_lat64_g0p001_fast_chol_clip_lamcarry` → `q128_M576_lat64_g0p001_fast_chol_clip_lamcarry_pred2` | algorithmic arm, error measured directly: error 1.0907 → 1.0906 %, stalled 0 → 0, rejected trial steps 0 → 0, median LM iterations per query 123.5 → 96.5, worst exit stationarity 9.87e-04 → 9.96e-04, certified True → True | 86.08 → 77.35 ms (1.11×); host-inclusive 331.0 → 323.1 ms | **kept** (same error, fewer iterations) |
| 2026-09-21 | hb4k04 (4079320, H200, 4096², dev6) | H11 | `pred2` (quadratic-extrapolation guard, one batched residual): `q0_M64_scaled_g0p001_fast_clip_lamcarry` → `q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2` | algorithmic arm, error measured directly: error 2.4158 → 2.4150 %, stalled 0 → 0, rejected trial steps 0 → 0, median LM iterations per query 113.5 → 84.5, worst exit stationarity 9.80e-04 → 9.95e-04, certified True → True | 42.55 → 40.73 ms (1.04×); host-inclusive 287.7 → 286.8 ms | **kept** (same error, fewer iterations) |
| 2026-09-21 | hb4k04 (4079320, H200, 4096², dev6) | H8b | looser LM stationarity tolerance (labelled tolerance arm): `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2` → `q256_M1088_lat64_g0p003_fast_chol_clip_lamcarry_pred2` | algorithmic arm, error measured directly: error 0.6043 → 0.6042 %, stalled 0 → 0, rejected trial steps 7 → 7, median LM iterations per query 136.5 → 118, worst exit stationarity 9.97e-04 → 2.90e-03, certified True → True | 127.18 → 114.99 ms (1.11×); host-inclusive 373.0 → 361.4 ms | labelled tolerance arm: kept as a reported arm; exit stationarity 3e-3, not 1e-3 |
| 2026-09-21 | hb4k04 (4079320, H200, 4096², dev6) | H8b | looser LM stationarity tolerance (labelled tolerance arm): `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2` → `q256_M1088_lat64_g0p01_fast_chol_clip_lamcarry_pred2` | algorithmic arm, error measured directly: error 0.6043 → 0.6042 %, stalled 0 → 0, rejected trial steps 7 → 7, median LM iterations per query 136.5 → 107, worst exit stationarity 9.97e-04 → 9.98e-03, certified True → True | 127.18 → 107.51 ms (1.18×); host-inclusive 373.0 → 353.1 ms | labelled tolerance arm: kept as a reported arm; exit stationarity 1e-2, not 1e-3 |

### Profile, hb2k01, `q256_M1088_lat64_g0p001_fast`, case 0 (the slowest case; LU)

whole query 506.0 ms = initial fit 18.4 + evolve 479.9 + decode 4.3. Evolve: 494 LM iterations over 50 steps, 114 rejected trial steps, 0.972 ms per iteration; isolated kernels: $(r,J)$ 0.761 ms, residual 0.348 ms, Gram 0.158 ms, Gram + LU solve 0.780 ms. **The LU solve of the 272×272 system (≈0.6 ms) is the largest single item per iteration, larger than the whole residual-and-Jacobian evaluation** — hence H5. From `result.json` (quick rows): the FIRST time step takes 41–88 of the 170–494 LM iterations of a query at tol 1e-3 (37–315 at 1e-6), most of the rejected trial steps are there, and steady state is 2–3 iterations per step. The start of step 1 is the initial-fit state and the $z$-step is capped at 1 % of the code radius, so the solver creeps and every over-long step is rejected at the price of a full $(r,J)$ evaluation and a solve. → H9, H10.

All numbers above are generated from `checks/hb2k01-summary.json` by the snippet recorded in the commit that added them; medians over 5 repetitions × 6 cases, same allocation.

### Cumulative, accurate rung $q=256$, $M=1088$, lattice rule, tol 1e-3, 2048², dev6 (hb2k02)

`q256_M1088_lat64_g0p001_fast` 302.74 ms → `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry` 130.69 ms: **2.32×** at unchanged error (0.5980 → 0.5980 %), zero stalled exits, on top of hfast's 1.15× over the audited path (hb2k01). Profile of the final arm, case 0: whole query 146.8 ms = initial fit 16.6 + evolve 125.5 + decode 4.3; 191 LM iterations, 0 rejected, 0.657 ms per iteration, of which $(r,J)$ 0.719 ms and Gram + Cholesky-free LU probe 0.784 ms (the probe times the LU solve; the arm runs Cholesky). What is left is ≈3.8 LM iterations per time step at ≈0.65 ms; the remaining ideas are a second-order predictor and a looser first-step policy (H8), not yet measured.

### Replication at 4096² (hb4k03) and what is left

The hb2k02 changes replicate at 4096² at unchanged error (rows above, generated by
`checks/speedlog_rows.py` from `checks/hb4k03-summary.json`). Profile of the kept arm at 4096², case 0:
whole query 163.7 ms = initial fit 18.3 + evolve 127.2 + decode 16.7; 191 LM iterations, 0 rejected,
0.666 ms per iteration; per-step iterations 20, 8, 8, 6, 6, 10, 10, 6, 5, 4, then 2–4. Decode is the
read of the 64 GiB f64 bank (≈13 ms at H200 bandwidth), so it cannot shrink without a precision arm.
Complete-query (host-inclusive) time adds ≈240 ms to every arm at 4096² (six f64 fields, 806 MB) and
dominates any GPU-side gain in that scope.

**H11 (measured, hb4k04, rows above):** the steady steps take 2–3 iterations because the linear predictor is
$O(\Delta t^2)$ wrong; `pred2` (quadratic extrapolation, one batched residual guard). Local 64² smoke
(GB10, not a result): 105 → 72 and 102 → 59 LM iterations per query at unchanged error.
**Not pursued (local evidence only, no job spent):** XLA command buffers for `while` loops — GB10
micro-benchmark 0.146 vs 0.146 ms per iteration; initial-fit tolerance 1e-4 — 81 → 78 iterations.

**H11 outcome (hb4k04, 4096², dev6, same job).** `pred2` removes 19–32 % of LM iterations at
unchanged error at every rung, but the time gain is smaller than the iteration gain (1.15× at the
accurate rung) because the steady steps were already cheap and the first steps, the initial fit
(18.4 ms) and the decode (16.7 ms) are untouched. Profile, case 0: 164.2 → 147.6 ms whole query,
191 → 161 iterations. With the stationarity tolerance loosened to $10^{-2}$ the accurate rung
reaches 107.5 ms at 0.6042 % (130 iterations on case 0) = 4.87× the relaxed passing FOM: **still
below 5**. What remains per query at 4096² is ≈ 18 ms initial fit + ≈ 17 ms decode + ≈ 0.7 ms per
LM iteration × ≈ 2 iterations per step × 50 steps; the fixed part alone (≈ 35 ms) caps the ratio
against a 524 ms FOM near 15× even with zero iterations, but the iteration floor is the binding one.
