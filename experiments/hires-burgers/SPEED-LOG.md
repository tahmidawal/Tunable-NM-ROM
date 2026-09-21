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
| 2026-09-20 | hb2k02 / hb4k03 (pending) | H9 | `clip`: shorten an over-long $z$-step onto the trust radius instead of rejecting it | algorithmic arm: error measured directly | pending | pending |
| 2026-09-20 | hb2k02 / hb4k03 (pending) | H10 | `lamcarry`: next step starts from this step's final damping | algorithmic arm | pending | pending |

### Profile, hb2k01, `q256_M1088_lat64_g0p001_fast`, case 0 (the slowest case; LU)

whole query 506.0 ms = initial fit 18.4 + evolve 479.9 + decode 4.3. Evolve: 494 LM iterations over 50 steps, 114 rejected trial steps, 0.972 ms per iteration; isolated kernels: $(r,J)$ 0.761 ms, residual 0.348 ms, Gram 0.158 ms, Gram + LU solve 0.780 ms. **The LU solve of the 272×272 system (≈0.6 ms) is the largest single item per iteration, larger than the whole residual-and-Jacobian evaluation** — hence H5. From `result.json` (quick rows): the FIRST time step takes 41–88 of the 170–494 LM iterations of a query at tol 1e-3 (37–315 at 1e-6), most of the rejected trial steps are there, and steady state is 2–3 iterations per step. The start of step 1 is the initial-fit state and the $z$-step is capped at 1 % of the code radius, so the solver creeps and every over-long step is rejected at the price of a full $(r,J)$ evaluation and a solve. → H9, H10.

All numbers above are generated from `checks/hb2k01-summary.json` by the snippet recorded in the commit that added them; medians over 5 repetitions × 6 cases, same allocation.
