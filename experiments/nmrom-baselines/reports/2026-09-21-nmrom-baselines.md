# Kim et al. masked-autoencoder NM-LSPG versus the project NM-ROM on the shared Burgers 2D family

Reproduction gate: **PASSED**. **Adapted** reproduction: passed on attempt 3 of the pre-registered three (activation sigmoid), not on the paper-default attempt 1; median 1.45 % against the 1.5 % bar (published < 1 %), i.e. above the published figure. Hyper-reduction was NOT reproduced (HR gate failed on every attempt), so every HR arm is exploratory. Kim-baseline rows are admissible only where marked (same code hashes and the gate activation).
All numbers are generated from run JSONs by `reports/gen_report.py`; state: final for this lane.

## 1. Reproduction gate (Kim et al. 2022, Section 6.2; published NM-LSPG < 1 %, NM-LSPG-HR 0.93–0.98 %, LS-LSPG-HR 34–38 %)

Pass rule (pre-registered): median of three seeds ≤ 1.5 % and a valid LS-LSPG control ≥ 10 %; HR: median at 55/58 ≤ 2 %.

| job | attempt | activation / scaling | NM-LSPG per seed | median | autoencode-only per seed | LS-LSPG control | HR 55/58 per seed (basis) | gate | HR gate |
|---|---|---|---|---|---|---|---|---|---|
| 4059370 | 1 | swish / feature | 1.67 %, 1.45 %, 1.73 % | 1.67 % | 1.04 %, 0.85 %, 1.21 % | 31.60 % | 131.69 %, 318.89 %, 154.03 % (residual) | FAIL | FAIL |
| 4072224 | 2 | swish / global | 709.38 %, 11.22 %, 10.89 % | 11.22 % | 5.60 %, 3.81 %, 3.86 % | 31.60 % | 9006.73 %, 13143.74 %, 2050.67 % (sns) | FAIL | FAIL |
| 4072224 | 1 (weights reloaded) | swish / feature | 1.67 %, 1.45 %, 1.73 % | 1.67 % | 1.04 %, 0.85 %, 1.21 % | 31.60 % | 386.56 %, 13005.47 %, 18777.20 % (sns) | FAIL | FAIL |
| 4077574 | 3 | sigmoid / feature | 1.45 %, 1.83 %, 1.44 % | 1.45 % | 1.58 %, 1.79 %, 1.49 % | 31.60 % | diverged, diverged, diverged (sns) | pass | FAIL |

## 2. Headline: worst evolved same-grid error, query time and memory per method, mesh and latent dimension

Validation cohort (32 held-out cases) for every row. Times: median GPU query, supplied initial field on GPU to six dense fields on GPU, all arms interleaved in one allocation per mesh. "× FOM" = named-FOM time / arm time (> 1 means faster than the full-order solve). Kim rows marked INADMISSIBLE used an activation that failed the reproduction gate.

**Cohort.** Every error in this table is on the shared split's 32 held-out *validation* cases (worst over cases and the five evolved times). It is not the 6 development cases used by other lanes; the project ROM's q = 0 error is larger on this cohort than on those, consistent with the bank-floor lane's held-out finding. Accuracy comparisons within this table are like-for-like.

**Which query path "ours" is.** The project-ROM rows run the unoptimised *dense-residual reference path*: vendored `topfix.make_query(..., quadrature='dense', arm='base')` from `exp/2026-09-17-b-panel` @ 25434a27, M = 4(K+q) sine weak tests (64 at q = 0, 1088 at q = 256), Gauss–Newton/LM stationarity tolerance 1e-6 for the steps and the initial fit, Gauss–Jordan linear solve at q = 0 and LU at q = 256, no b-speed kernels; compilation excluded (every subject gets one warm call before timing). It is **not** the optimised query (certified empirical quadrature, looser tolerance, fused/fast kernels) measured by the hires-burgers and b-panel lanes; no time from another job is used here. The fair speed comparison in this table is therefore same-path: our dense path against Kim NM-LSPG without HR (also a dense full-residual Gauss–Newton) and against dense POD-LSPG; Kim NM-LSPG-HR is the hyper-reduced analogue of the optimised query, which this job did not time.

| mesh | method | solved unknowns | worst evolved | median evolved | query ms | × FOM | compiled-query MB | admissible |
|---|---|---|---|---|---|---|---|---|
| 128² | Kim NM-LSPG K=8 [swish] | 8 | 50.57 % | 21.53 % | 301.1 | 0.24 | 4252 | NO |
| 128² | Kim NM-LSPG-HR K=8 (exploratory) [swish] | 8 | 54.16 % | 26.59 % | 79.9 | 0.91 | 4258 | NO |
| 128² | POD-LSPG K=8 (better reference) | 8 | 54.95 % | 22.34 % | 56.2 | 1.29 | 4 | yes |
| 128² | Kim NM-LSPG K=16 [swish] | 16 | 43.42 % | 19.49 % | 518.9 | 0.14 | 4275 | NO |
| 128² | Kim NM-LSPG-HR K=16 (exploratory) [swish] | 16 | 44.75 % | 19.03 % | 132.5 | 0.55 | 4321 | NO |
| 128² | POD-LSPG K=16 (better reference) | 16 | 40.85 % | 9.93 % | 93.6 | 0.78 | 8 | yes |
| 128² | Kim NM-LSPG K=32 [swish] | 32 | 40.84 % | 14.99 % | 945.6 | 0.08 | 4322 | NO |
| 128² | Kim NM-LSPG-HR K=32 (exploratory) [swish] | 32 | 58.15 % | 20.22 % | 166.5 | 0.44 | 4328 | NO |
| 128² | POD-LSPG K=32 (better reference) | 32 | 23.35 % | 4.20 % | 176.3 | 0.41 | 14 | yes |
| 128² | ours fast (q=0), K=16 — dense reference path, M=64 | 16 | 3.41 % | 0.59 % | 109.2 | 0.67 | 190 | yes |
| 128² | ours accurate (q=256), K=16 — dense reference path, M=1088 | 272 | 0.51 % | 0.08 % | 1353.7 | 0.05 | 504 | yes |
| 128² | FOM loose (1e-4) | — | 0.05 % | 0.01 % | 32.2 | 2.26 | 4 | yes |
| 128² | FOM named / reference | — | 0.00 % | 0.00 % | 72.8 | 1.00 | 4 | yes |
| 256² | Kim NM-LSPG K=8 [sigmoid] | 8 | 163.93 % | 43.34 % | 1300.8 | 0.06 | 2480 | yes |
| 256² | Kim NM-LSPG-HR K=8 (exploratory) [sigmoid] | 8 | 115.60 % | 51.78 % | 64.7 | 1.25 | 2452 | yes |
| 256² | POD-LSPG K=8 (better reference) | 8 | 55.52 % | 22.72 % | 138.0 | 0.59 | 18 | yes |
| 256² | Kim NM-LSPG K=16 [sigmoid] | 16 | 145.40 % | 43.87 % | 2384.4 | 0.03 | 2567 | yes |
| 256² | Kim NM-LSPG-HR K=16 (exploratory) [sigmoid] | 16 | 202.87 % | 52.51 % | 126.2 | 0.64 | 2506 | yes |
| 256² | Kim NM-LSPG K=16, data-matched (576 traj.) [sigmoid] | 16 | 120.33 % | 38.71 % | 3885.9 | 0.02 | 2567 | yes |
| 256² | POD-LSPG K=16 (better reference) | 16 | 41.53 % | 10.45 % | 258.8 | 0.31 | 30 | yes |
| 256² | Kim NM-LSPG K=32 [sigmoid] | 32 | 127.51 % | 43.61 % | 3845.7 | 0.02 | 2742 | yes |
| 256² | Kim NM-LSPG-HR K=32 (exploratory) [sigmoid] | 32 | 122.68 % | 41.98 % | 474.9 | 0.17 | 2794 | yes |
| 256² | POD-LSPG K=32 (better reference) | 32 | 24.78 % | 5.65 % | 518.8 | 0.16 | 55 | yes |
| 256² | ours fast (q=0), K=16 — dense reference path, M=64 | 16 | 6.79 % | 0.66 % | 262.2 | 0.31 | 625 | yes |
| 256² | ours accurate (q=256), K=16 — dense reference path, M=1088 | 272 | 0.88 % | 0.09 % | 3746.3 | 0.02 | 1769 | yes |
| 256² | FOM loose (1e-4) | — | 0.06 % | 0.01 % | 33.9 | 2.39 | 14 | yes |
| 256² | FOM named / reference | — | 0.00 % | 0.00 % | 81.0 | 1.00 | 14 | yes |
| 512² | Kim NM-LSPG K=8 [sigmoid] | 8 | 175.71 % | 55.48 % | 746.1 | 0.11 | 9959 | yes |
| 512² | Kim NM-LSPG-HR K=8 (exploratory) [sigmoid] | 8 | 273.27 % | 53.08 % | 29.6 | 2.67 | 9813 | yes |
| 512² | POD-LSPG K=8 (better reference) | 8 | 55.82 % | 24.08 % | 313.5 | 0.25 | 71 | yes |
| 512² | Kim NM-LSPG K=16 [sigmoid] | 16 | 174.99 % | 58.56 % | 1496.6 | 0.05 | 10511 | yes |
| 512² | Kim NM-LSPG-HR K=16 (exploratory) [sigmoid] | 16 | 185.17 % | 58.52 % | 61.7 | 1.28 | 10099 | yes |
| 512² | POD-LSPG K=16 (better reference) | 16 | 42.01 % | 13.83 % | 617.2 | 0.13 | 122 | yes |
| 512² | Kim NM-LSPG K=32 [sigmoid] | 32 | 252.33 % | 56.72 % | 2059.3 | 0.04 | 10879 | yes |
| 512² | Kim NM-LSPG-HR K=32 (exploratory) [sigmoid] | 32 | 194.47 % | 56.00 % | 138.5 | 0.57 | 10569 | yes |
| 512² | POD-LSPG K=32 (better reference) | 32 | 25.63 % | 7.74 % | 383.4 | 0.21 | 222 | yes |
| 512² | ours fast (q=0), K=16 — dense reference path, M=64 | 16 | 7.72 % | 0.69 % | 404.2 | 0.20 | 2368 | yes |
| 512² | ours accurate (q=256), K=16 — dense reference path, M=1088 | 272 | 1.05 % | 0.08 % | 6268.5 | 0.01 | 6891 | yes |
| 512² | FOM loose (1e-4) | — | 0.05 % | 0.01 % | 33.2 | 2.38 | 55 | yes |
| 512² | FOM named / reference | — | 0.00 % | 0.00 % | 79.0 | 1.00 | 55 | yes |

### Where a baseline beats the project NM-ROM

Every admissible reduced baseline row that is better than one of our two settings on the same mesh, on either axis. **Every speed line below compares against our dense reference path** (see above); lines marked [HR vs dense] compare a hyper-reduced baseline with it and do not say how the baseline compares with our optimised (empirical-quadrature) query, which this job did not measure. Accuracy lines are unaffected.

- 128² [dense vs dense]: POD-LSPG K=8 (better reference) is faster (56.2 ms vs 109.2 ms for `ours_q0`, dense reference path) at 54.95 % vs 3.41 % worst evolved error.
- 128² [dense vs dense]: POD-LSPG K=8 (better reference) is faster (56.2 ms vs 1353.7 ms for `ours_q256`, dense reference path) at 54.95 % vs 0.51 % worst evolved error.
- 128² [dense vs dense]: POD-LSPG K=16 (better reference) is faster (93.6 ms vs 109.2 ms for `ours_q0`, dense reference path) at 40.85 % vs 3.41 % worst evolved error.
- 128² [dense vs dense]: POD-LSPG K=16 (better reference) is faster (93.6 ms vs 1353.7 ms for `ours_q256`, dense reference path) at 40.85 % vs 0.51 % worst evolved error.
- 128² [dense vs dense]: POD-LSPG K=32 (better reference) is faster (176.3 ms vs 1353.7 ms for `ours_q256`, dense reference path) at 23.35 % vs 0.51 % worst evolved error.
- 128²: the named FOM itself (72.8 ms) is faster than `ours_q0` (109.2 ms), so our dense reference path at that setting is not a speed-up over the full-order solve here (says nothing about the optimised query, not timed in this job).
- 128²: the named FOM itself (72.8 ms) is faster than `ours_q256` (1353.7 ms), so our dense reference path at that setting is not a speed-up over the full-order solve here (says nothing about the optimised query, not timed in this job).
- 256² [dense vs dense]: Kim NM-LSPG K=8 is faster (1300.8 ms vs 3746.3 ms for `ours_q256`, dense reference path) at 163.93 % vs 0.88 % worst evolved error.
- 256² [HR vs dense]: Kim NM-LSPG-HR K=8 (exploratory) is faster (64.7 ms vs 262.2 ms for `ours_q0`, dense reference path) at 115.60 % vs 6.79 % worst evolved error.
- 256² [HR vs dense]: Kim NM-LSPG-HR K=8 (exploratory) is faster (64.7 ms vs 3746.3 ms for `ours_q256`, dense reference path) at 115.60 % vs 0.88 % worst evolved error.
- 256² [dense vs dense]: POD-LSPG K=8 (better reference) is faster (138.0 ms vs 262.2 ms for `ours_q0`, dense reference path) at 55.52 % vs 6.79 % worst evolved error.
- 256² [dense vs dense]: POD-LSPG K=8 (better reference) is faster (138.0 ms vs 3746.3 ms for `ours_q256`, dense reference path) at 55.52 % vs 0.88 % worst evolved error.
- 256² [dense vs dense]: Kim NM-LSPG K=16 is faster (2384.4 ms vs 3746.3 ms for `ours_q256`, dense reference path) at 145.40 % vs 0.88 % worst evolved error.
- 256² [HR vs dense]: Kim NM-LSPG-HR K=16 (exploratory) is faster (126.2 ms vs 262.2 ms for `ours_q0`, dense reference path) at 202.87 % vs 6.79 % worst evolved error.
- 256² [HR vs dense]: Kim NM-LSPG-HR K=16 (exploratory) is faster (126.2 ms vs 3746.3 ms for `ours_q256`, dense reference path) at 202.87 % vs 0.88 % worst evolved error.
- 256² [dense vs dense]: POD-LSPG K=16 (better reference) is faster (258.8 ms vs 262.2 ms for `ours_q0`, dense reference path) at 41.53 % vs 6.79 % worst evolved error.
- 256² [dense vs dense]: POD-LSPG K=16 (better reference) is faster (258.8 ms vs 3746.3 ms for `ours_q256`, dense reference path) at 41.53 % vs 0.88 % worst evolved error.
- 256² [HR vs dense]: Kim NM-LSPG-HR K=32 (exploratory) is faster (474.9 ms vs 3746.3 ms for `ours_q256`, dense reference path) at 122.68 % vs 0.88 % worst evolved error.
- 256² [dense vs dense]: POD-LSPG K=32 (better reference) is faster (518.8 ms vs 3746.3 ms for `ours_q256`, dense reference path) at 24.78 % vs 0.88 % worst evolved error.
- 256²: the named FOM itself (81.0 ms) is faster than `ours_q0` (262.2 ms), so our dense reference path at that setting is not a speed-up over the full-order solve here (says nothing about the optimised query, not timed in this job).
- 256²: the named FOM itself (81.0 ms) is faster than `ours_q256` (3746.3 ms), so our dense reference path at that setting is not a speed-up over the full-order solve here (says nothing about the optimised query, not timed in this job).
- 512² [dense vs dense]: Kim NM-LSPG K=8 is faster (746.1 ms vs 6268.5 ms for `ours_q256`, dense reference path) at 175.71 % vs 1.05 % worst evolved error.
- 512² [HR vs dense]: Kim NM-LSPG-HR K=8 (exploratory) is faster (29.6 ms vs 404.2 ms for `ours_q0`, dense reference path) at 273.27 % vs 7.72 % worst evolved error.
- 512² [HR vs dense]: Kim NM-LSPG-HR K=8 (exploratory) is faster (29.6 ms vs 6268.5 ms for `ours_q256`, dense reference path) at 273.27 % vs 1.05 % worst evolved error.
- 512² [dense vs dense]: POD-LSPG K=8 (better reference) is faster (313.5 ms vs 404.2 ms for `ours_q0`, dense reference path) at 55.82 % vs 7.72 % worst evolved error.
- 512² [dense vs dense]: POD-LSPG K=8 (better reference) is faster (313.5 ms vs 6268.5 ms for `ours_q256`, dense reference path) at 55.82 % vs 1.05 % worst evolved error.
- 512² [dense vs dense]: Kim NM-LSPG K=16 is faster (1496.6 ms vs 6268.5 ms for `ours_q256`, dense reference path) at 174.99 % vs 1.05 % worst evolved error.
- 512² [HR vs dense]: Kim NM-LSPG-HR K=16 (exploratory) is faster (61.7 ms vs 404.2 ms for `ours_q0`, dense reference path) at 185.17 % vs 7.72 % worst evolved error.
- 512² [HR vs dense]: Kim NM-LSPG-HR K=16 (exploratory) is faster (61.7 ms vs 6268.5 ms for `ours_q256`, dense reference path) at 185.17 % vs 1.05 % worst evolved error.
- 512² [dense vs dense]: POD-LSPG K=16 (better reference) is faster (617.2 ms vs 6268.5 ms for `ours_q256`, dense reference path) at 42.01 % vs 1.05 % worst evolved error.
- 512² [dense vs dense]: Kim NM-LSPG K=32 is faster (2059.3 ms vs 6268.5 ms for `ours_q256`, dense reference path) at 252.33 % vs 1.05 % worst evolved error.
- 512² [HR vs dense]: Kim NM-LSPG-HR K=32 (exploratory) is faster (138.5 ms vs 404.2 ms for `ours_q0`, dense reference path) at 194.47 % vs 7.72 % worst evolved error.
- 512² [HR vs dense]: Kim NM-LSPG-HR K=32 (exploratory) is faster (138.5 ms vs 6268.5 ms for `ours_q256`, dense reference path) at 194.47 % vs 1.05 % worst evolved error.
- 512² [dense vs dense]: POD-LSPG K=32 (better reference) is faster (383.4 ms vs 404.2 ms for `ours_q0`, dense reference path) at 25.63 % vs 7.72 % worst evolved error.
- 512² [dense vs dense]: POD-LSPG K=32 (better reference) is faster (383.4 ms vs 6268.5 ms for `ours_q256`, dense reference path) at 25.63 % vs 1.05 % worst evolved error.
- 512²: the named FOM itself (79.0 ms) is faster than `ours_q0` (404.2 ms), so our dense reference path at that setting is not a speed-up over the full-order solve here (says nothing about the optimised query, not timed in this job).
- 512²: the named FOM itself (79.0 ms) is faster than `ours_q256` (6268.5 ms), so our dense reference path at that setting is not a speed-up over the full-order solve here (says nothing about the optimised query, not timed in this job).

## 3. Where each Kim configuration stops fitting or training

| mesh | arm | outcome |
|---|---|---|
| 128² | `kim_K16_base` | trained 869 epochs in 1201 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 7.18e-06 |
| 128² | `kim_K16_refzero` | trained 869 epochs in 1201 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 8.71e-06 |
| 128² | `kim_K16_global` | trained 869 epochs in 1201 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 4.17e-06 |
| 128² | `kim_K16_refzero_global` | trained 869 epochs in 1201 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 4.25e-06 |
| 128² | `kim_K16_b50` | trained 1174 epochs in 1200 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 1.10e-05 |
| 128² | `kim_K16_b200` | trained 895 epochs in 1201 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 6.88e-06 |
| 128² | `kim_K16_M1_4096` | trained 2077 epochs in 1200 s, stop = wall_budget, M1 = 4096, 112 fit trajectories, best validation-snapshot MSE 4.45e-06 |
| 128² | `kim_K16_M1_1024` | trained 2480 epochs in 1190 s, stop = early_stop, M1 = 1024, 112 fit trajectories, best validation-snapshot MSE 4.64e-06 |
| 128² | `kim_K16_pat50` | trained 334 epochs in 461 s, stop = early_stop, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 2.56e-05 |
| 128² | `kim_K16_lr3e4_pat50` | trained 869 epochs in 1201 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 4.80e-06 |
| 128² | `kim_K16_f64` | trained 454 epochs in 1200 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 9.83e-06 |
| 128² | `kim_final_K8` | trained 870 epochs in 1200 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 8.20e-06 |
| 128² | `kim_final_K32` | trained 864 epochs in 1200 s, stop = wall_budget, M1 = 32258, 112 fit trajectories, best validation-snapshot MSE 2.83e-06 |
| 256² | `sig_K16_published_M1` | exceeds_device_memory_precheck: needs 136 GB for weights + gradient + Adam state > device 77 GB (M1 = 130050); not attempted |
| 256² | `sig_K16_ic_feature` | trained 846 epochs in 2400 s, stop = wall_budget, M1 = 4096, 112 fit trajectories, best validation-snapshot MSE 4.08e-06 |
| 256² | `sig_K16_zero_feature` | trained 846 epochs in 2401 s, stop = wall_budget, M1 = 4096, 112 fit trajectories, best validation-snapshot MSE 2.68e-05 |
| 256² | `sig_K16_zero_global` | trained 847 epochs in 2402 s, stop = wall_budget, M1 = 4096, 112 fit trajectories, best validation-snapshot MSE 3.75e-06 |
| 256² | `kim_final_K8` | trained 852 epochs in 2401 s, stop = wall_budget, M1 = 4096, 112 fit trajectories, best validation-snapshot MSE 3.96e-05 |
| 256² | `kim_final_K32` | trained 837 epochs in 2402 s, stop = wall_budget, M1 = 4096, 112 fit trajectories, best validation-snapshot MSE 2.89e-05 |
| 256² | `kim_final_K16_fit576` | trained 540 epochs in 7205 s, stop = wall_budget, M1 = 4096, 576 fit trajectories, best validation-snapshot MSE 1.93e-05 |
| 512² | `sig_K16_published_M1` | exceeds_device_memory_precheck: needs 2183 GB for weights + gradient + Adam state > device 135 GB (M1 = 522242); not attempted |
| 512² | `sig_K8_sel` | trained 231 epochs in 4810 s, stop = wall_budget, M1 = 4096, 112 fit trajectories, best validation-snapshot MSE 2.97e-04 |
| 512² | `sig_K16_sel` | trained 230 epochs in 4814 s, stop = wall_budget, M1 = 4096, 112 fit trajectories, best validation-snapshot MSE 2.48e-04 |
| 512² | `sig_K32_sel` | trained 229 epochs in 4814 s, stop = wall_budget, M1 = 4096, 112 fit trajectories, best validation-snapshot MSE 3.09e-04 |

## 4. Tuning effort given to the Kim baseline

- **128²** (job 4073272): 11 sweep candidates scored on the tuning subset, 13 autoencoders trained, 4.1 GPU-hours of training; selected `kim_K16_refzero_global`. Candidates (tune worst evolved): `kim_K16_base` 83.52 %, `kim_K16_refzero` 43.62 %, `kim_K16_global` 83.73 %, `kim_K16_refzero_global` 35.95 %, `kim_K16_b50` 84.56 %, `kim_K16_b200` 69.78 %, `kim_K16_M1_4096` 69.54 %, `kim_K16_M1_1024` 68.97 %, `kim_K16_pat50` 69.77 %, `kim_K16_lr3e4_pat50` 70.02 %, `kim_K16_f64` 84.39 %
- **256²** (job 4095408): 3 sweep candidates scored on the tuning subset, 6 autoencoders trained, 5.3 GPU-hours of training; selected `sig_K16_zero_feature`. Candidates (tune worst evolved): `sig_K16_ic_feature` 84.93 %, `sig_K16_zero_feature` 70.59 %, `sig_K16_zero_global` 76.47 %
- **512²** (job 4107272): 0 sweep candidates scored on the tuning subset, 3 autoencoders trained, 4.0 GPU-hours of training; selected `—`. Candidates (tune worst evolved): 

## Shared Burgers family, 128² intervals (n = 16129), job 4073272, NVIDIA A100 80GB PCIe, GPU-9732c808-acc9-5c43-f0ce-4a81a2bf1719

Kim rows: **code matches a passed gate; a Kim row is admissible only if its activation is the gate activation (sigmoid)**. Other Kim hyper-parameters are tuned on the family (DESIGN s.3) and printed in the `act` column and variant. Cohort = 32 held-out validation cases unless the column says tune (16 training-side cases used for every choice). NumPy audit: all audited rows agree.

| arm | family | solved unknowns | worst evolved (validation) | median evolved | worst evolved (tune) | autoencode-only worst | GN cap hits | query GPU ms (median) | compiled-query memory MB | training s (epochs, stop) |
|---|---|---|---|---|---|---|---|---|---|---|
| `pod_lspg_zero_K8` | pod_lspg | 8 | 54.95 % | 22.34 % | — | — | 0 | 56.2 | 4 | 0 (—, —) |
| `pod_lspg_zero_K16` | pod_lspg | 16 | 40.85 % | 9.93 % | — | — | 0 | 93.6 | 8 | 0 (—, —) |
| `pod_lspg_zero_K32` | pod_lspg | 32 | 23.35 % | 4.20 % | — | — | 1 | 176.3 | 14 | 0 (—, —) |
| `pod_lspg_zero_K64` | pod_lspg | 64 | 12.27 % | 1.41 % | — | — | 1 | 355.7 | 26 | 0 (—, —) |
| `pod_lspg_zero_K128` | pod_lspg | 128 | 3.57 % | 0.30 % | — | — | 0 | 679.6 | 51 | 0 (—, —) |
| `pod_lspg_ic_K8` | pod_lspg | 8 | 100.35 % | 55.88 % | — | — | 0 | 64.7 | 4 | 0 (—, —) |
| `pod_lspg_ic_K16` | pod_lspg | 16 | 66.79 % | 39.25 % | — | — | 0 | 108.7 | 8 | 0 (—, —) |
| `pod_lspg_ic_K32` | pod_lspg | 32 | 45.93 % | 23.56 % | — | — | 1 | 205.9 | 14 | 0 (—, —) |
| `pod_lspg_ic_K64` | pod_lspg | 64 | 33.67 % | 5.07 % | — | — | 2 | 414.8 | 26 | 0 (—, —) |
| `pod_lspg_ic_K128` | pod_lspg | 128 | 10.73 % | 1.13 % | — | — | 0 | 716.9 | 51 | 0 (—, —) |
| `kim_K16_base` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 84.39 % | 33.20 % | 83.52 % | 40.76 % | 46 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_refzero` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 43.04 % | 19.68 % | 43.62 % | 30.06 % | 17 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_global` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 84.63 % | 30.44 % | 83.73 % | 40.30 % | 28 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_refzero_global` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 43.42 % | 19.49 % | 35.95 % | 29.73 % | 5 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_b50` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 85.30 % | 34.93 % | 84.56 % | 38.26 % | 19 | not timed | — | 1200 (1174, wall_budget) |
| `kim_K16_b200` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 68.45 % | 29.96 % | 69.78 % | 40.45 % | 25 | not timed | — | 1201 (895, wall_budget) |
| `kim_K16_M1_4096` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 67.92 % | 29.52 % | 69.54 % | 34.81 % | 30 | not timed | — | 1200 (2077, wall_budget) |
| `kim_K16_M1_1024` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 66.74 % | 28.65 % | 68.97 % | 38.52 % | 18 | not timed | — | 1190 (2480, early_stop) |
| `kim_K16_pat50` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 63.81 % | 27.79 % | 69.77 % | 43.25 % | 17 | not timed | — | 461 (334, early_stop) |
| `kim_K16_lr3e4_pat50` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 70.64 % | 31.66 % | 70.02 % | 38.26 % | 29 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_f64` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 84.57 % | 32.24 % | 84.39 % | 38.78 % | 36 | not timed | — | 1200 (454, wall_budget) |
| `kim_final_K8` [swish] INADMISSIBLE | kim_nm_lspg | 8 | 50.57 % | 21.53 % | 44.40 % | 34.66 % | 11 | 301.1 | 4252 | 1200 (870, wall_budget) |
| `kim_final_K8_hr` [swish] INADMISSIBLE | kim_nm_lspg_hr (exploratory HR) | 8 | 54.16 % | 26.59 % | — | — | — | 79.9 | 4258 | 0 (—, —) |
| `kim_final_K16` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 43.42 % | 19.49 % | 35.95 % | 29.73 % | 5 | 518.9 | 4275 | 0 (—, —) |
| `kim_final_K16_hr` [swish] INADMISSIBLE | kim_nm_lspg_hr (exploratory HR) | 16 | 44.75 % | 19.03 % | — | — | — | 132.5 | 4321 | 0 (—, —) |
| `kim_final_K32` [swish] INADMISSIBLE | kim_nm_lspg | 32 | 40.84 % | 14.99 % | 35.11 % | 21.28 % | 9 | 945.6 | 4322 | 1200 (864, wall_budget) |
| `kim_final_K32_hr` [swish] INADMISSIBLE | kim_nm_lspg_hr (exploratory HR) | 32 | 58.15 % | 20.22 % | — | — | — | 166.5 | 4328 | 0 (—, —) |
| `ours_q0` | ours | 16 | 3.41 % | 0.59 % | — | — | — | 109.2 | 190 | 0 (—, —) |
| `ours_q256` | ours | 272 | 0.51 % | 0.08 % | — | — | — | 1353.7 | 504 | 0 (—, —) |
| `fom_fft_tight` | fom | — | 0.00 % | 0.00 % | — | — | — | 72.8 | 4 | 0 (—, —) |
| `fom_nt1e4_dt005` | fom | — | 0.05 % | 0.01 % | — | — | — | 32.2 | 4 | 0 (—, —) |

Selection rule: smallest worst evolved NM-LSPG error on the 16-case tuning subset (train cases 112-127). Selected: `kim_K16_refzero_global`.


## Shared Burgers family, 256² intervals (n = 65025), job 4095408, NVIDIA A100 80GB PCIe, GPU-7ea87052-7ee2-b522-615d-be66f44238c3

Kim rows: **code matches a passed gate; a Kim row is admissible only if its activation is the gate activation (sigmoid)**. Other Kim hyper-parameters are tuned on the family (DESIGN s.3) and printed in the `act` column and variant. Cohort = 32 held-out validation cases unless the column says tune (16 training-side cases used for every choice). NumPy audit: all audited rows agree.

| arm | family | solved unknowns | worst evolved (validation) | median evolved | worst evolved (tune) | autoencode-only worst | GN cap hits | query GPU ms (median) | compiled-query memory MB | training s (epochs, stop) |
|---|---|---|---|---|---|---|---|---|---|---|
| `pod_lspg_zero_K8` | pod_lspg | 8 | 55.52 % | 22.72 % | — | — | 0 | 138.0 | 18 | 0 (—, —) |
| `pod_lspg_zero_K16` | pod_lspg | 16 | 41.53 % | 10.45 % | — | — | 0 | 258.8 | 30 | 0 (—, —) |
| `pod_lspg_zero_K32` | pod_lspg | 32 | 24.78 % | 5.65 % | — | — | 0 | 518.8 | 55 | 0 (—, —) |
| `pod_lspg_zero_K64` | pod_lspg | 64 | 12.73 % | 1.78 % | — | — | 0 | 1184.2 | 105 | 0 (—, —) |
| `pod_lspg_zero_K128` | pod_lspg | 128 | 4.33 % | 0.41 % | — | — | 0 | 2190.1 | 205 | 0 (—, —) |
| `pod_lspg_ic_K8` | pod_lspg | 8 | 139.16 % | 62.84 % | — | — | 0 | 148.0 | 18 | 0 (—, —) |
| `pod_lspg_ic_K16` | pod_lspg | 16 | 97.91 % | 49.21 % | — | — | 1 | 316.2 | 30 | 0 (—, —) |
| `pod_lspg_ic_K32` | pod_lspg | 32 | 85.98 % | 30.79 % | — | — | 0 | 600.9 | 55 | 0 (—, —) |
| `pod_lspg_ic_K64` | pod_lspg | 64 | 34.79 % | 8.42 % | — | — | 1 | 1386.5 | 105 | 0 (—, —) |
| `pod_lspg_ic_K128` | pod_lspg | 128 | 11.25 % | 1.87 % | — | — | 1 | 2319.1 | 205 | 0 (—, —) |
| `sig_K16_ic_feature` [sigmoid] | kim_nm_lspg | 16 | 90.47 % | 63.91 % | 84.93 % | 92.94 % | 128 | not timed | — | 2400 (846, wall_budget) |
| `sig_K16_zero_feature` [sigmoid] | kim_nm_lspg | 16 | 145.40 % | 43.87 % | 70.59 % | 61.76 % | 252 | not timed | — | 2401 (846, wall_budget) |
| `sig_K16_zero_global` [sigmoid] | kim_nm_lspg | 16 | 117.40 % | 37.47 % | 76.47 % | 37.37 % | 199 | not timed | — | 2402 (847, wall_budget) |
| `kim_final_K8` [sigmoid] | kim_nm_lspg | 8 | 163.93 % | 43.34 % | 79.22 % | 60.32 % | 206 | 1300.8 | 2480 | 2401 (852, wall_budget) |
| `kim_final_K8_hr` [sigmoid] | kim_nm_lspg_hr (exploratory HR) | 8 | 115.60 % | 51.78 % | — | — | — | 64.7 | 2452 | 0 (—, —) |
| `kim_final_K16` [sigmoid] | kim_nm_lspg | 16 | 145.40 % | 43.87 % | 70.59 % | 61.76 % | 252 | 2384.4 | 2567 | 0 (—, —) |
| `kim_final_K16_hr` [sigmoid] | kim_nm_lspg_hr (exploratory HR) | 16 | 202.87 % | 52.51 % | — | — | — | 126.2 | 2506 | 0 (—, —) |
| `kim_final_K32` [sigmoid] | kim_nm_lspg | 32 | 127.51 % | 43.61 % | 70.37 % | 55.59 % | 229 | 3845.7 | 2742 | 2402 (837, wall_budget) |
| `kim_final_K32_hr` [sigmoid] | kim_nm_lspg_hr (exploratory HR) | 32 | 122.68 % | 41.98 % | — | — | — | 474.9 | 2794 | 0 (—, —) |
| `kim_final_K16_fit576` [sigmoid] | kim_nm_lspg | 16 | 120.33 % | 38.71 % | 55.90 % | 30.16 % | 475 | 3885.9 | 2567 | 7205 (540, wall_budget) |
| `ours_q0` | ours | 16 | 6.79 % | 0.66 % | — | — | — | 262.2 | 625 | 0 (—, —) |
| `ours_q256` | ours | 272 | 0.88 % | 0.09 % | — | — | — | 3746.3 | 1769 | 0 (—, —) |
| `fom_fft_tight` | fom | — | 0.00 % | 0.00 % | — | — | — | 81.0 | 14 | 0 (—, —) |
| `fom_nt1e4_dt005` | fom | — | 0.06 % | 0.01 % | — | — | — | 33.9 | 14 | 0 (—, —) |

Selection rule: smallest worst evolved NM-LSPG error on the 16-case tuning subset (train cases 112-127). Selected: `sig_K16_zero_feature`.


Dropped arms: `sig_K16_published_M1` (exceeds_device_memory_precheck)


## Shared Burgers family, 512² intervals (n = 261121), job 4107272, NVIDIA H200, GPU-50404ee7-1d50-1d0c-6780-806cf66543b8

Kim rows: **code matches a passed gate; a Kim row is admissible only if its activation is the gate activation (sigmoid)**. Other Kim hyper-parameters are tuned on the family (DESIGN s.3) and printed in the `act` column and variant. Cohort = 32 held-out validation cases unless the column says tune (16 training-side cases used for every choice). NumPy audit: all audited rows agree.

| arm | family | solved unknowns | worst evolved (validation) | median evolved | worst evolved (tune) | autoencode-only worst | GN cap hits | query GPU ms (median) | compiled-query memory MB | training s (epochs, stop) |
|---|---|---|---|---|---|---|---|---|---|---|
| `pod_lspg_zero_K8` | pod_lspg | 8 | 55.82 % | 24.08 % | — | — | 0 | 313.5 | 71 | 0 (—, —) |
| `pod_lspg_zero_K16` | pod_lspg | 16 | 42.01 % | 13.83 % | — | — | 0 | 617.2 | 122 | 0 (—, —) |
| `pod_lspg_zero_K32` | pod_lspg | 32 | 25.63 % | 7.74 % | — | — | 0 | 383.4 | 222 | 0 (—, —) |
| `pod_lspg_zero_K64` | pod_lspg | 64 | 13.52 % | 2.22 % | — | — | 0 | 1023.4 | 422 | 0 (—, —) |
| `pod_lspg_zero_K128` | pod_lspg | 128 | 5.41 % | 0.56 % | — | — | 0 | 1458.0 | 823 | 0 (—, —) |
| `pod_lspg_ic_K8` | pod_lspg | 8 | 149.51 % | 69.16 % | — | — | 1 | 322.1 | 71 | 0 (—, —) |
| `pod_lspg_ic_K16` | pod_lspg | 16 | 167.60 % | 57.08 % | — | — | 0 | 778.0 | 122 | 0 (—, —) |
| `pod_lspg_ic_K32` | pod_lspg | 32 | 177.37 % | 38.24 % | — | — | 0 | 441.7 | 222 | 0 (—, —) |
| `pod_lspg_ic_K64` | pod_lspg | 64 | 35.76 % | 12.18 % | — | — | 0 | 1207.4 | 422 | 0 (—, —) |
| `pod_lspg_ic_K128` | pod_lspg | 128 | 11.98 % | 3.09 % | — | — | 0 | 1563.3 | 823 | 0 (—, —) |
| `sig_K8_sel` [sigmoid] | kim_nm_lspg | 8 | 175.71 % | 55.48 % | 74.64 % | 275.23 % | 387 | 746.1 | 9959 | 4810 (231, wall_budget) |
| `sig_K8_sel_hr` [sigmoid] | kim_nm_lspg_hr (exploratory HR) | 8 | 273.27 % | 53.08 % | — | — | — | 29.6 | 9813 | 0 (—, —) |
| `sig_K16_sel` [sigmoid] | kim_nm_lspg | 16 | 174.99 % | 58.56 % | 76.77 % | 248.02 % | 244 | 1496.6 | 10511 | 4814 (230, wall_budget) |
| `sig_K16_sel_hr` [sigmoid] | kim_nm_lspg_hr (exploratory HR) | 16 | 185.17 % | 58.52 % | — | — | — | 61.7 | 10099 | 0 (—, —) |
| `sig_K32_sel` [sigmoid] | kim_nm_lspg | 32 | 252.33 % | 56.72 % | 67.49 % | 227.92 % | 227 | 2059.3 | 10879 | 4814 (229, wall_budget) |
| `sig_K32_sel_hr` [sigmoid] | kim_nm_lspg_hr (exploratory HR) | 32 | 194.47 % | 56.00 % | — | — | — | 138.5 | 10569 | 0 (—, —) |
| `ours_q0` | ours | 16 | 7.72 % | 0.69 % | — | — | — | 404.2 | 2368 | 0 (—, —) |
| `ours_q256` | ours | 272 | 1.05 % | 0.08 % | — | — | — | 6268.5 | 6891 | 0 (—, —) |
| `fom_fft_tight` | fom | — | 0.00 % | 0.00 % | — | — | — | 79.0 | 55 | 0 (—, —) |
| `fom_nt1e4_dt005` | fom | — | 0.05 % | 0.01 % | — | — | — | 33.2 | 55 | 0 (—, —) |

Dropped arms: `sig_K16_published_M1` (exceeds_device_memory_precheck)


## Glossary

- **NM-LSPG**: nonlinear-manifold least-squares Petrov–Galerkin; each backward-Euler step minimises the full discrete residual over the latent vector. **HR**: hyper-reduction (residual evaluated at sampled rows only).
- **worst evolved**: largest, over cases and the five output times after t = 0, of the field error against the same-grid full-order solve, divided by the norm of the initial field.
- **autoencode-only**: encode then decode the true states; not a lower bound on the manifold error.
- **solved unknowns**: number of unknowns in the online nonlinear solve (K, or K+q for the project ROM with q corrections).
- **tune**: 16 cases carved from the training split, used for every choice; **validation**: 32 held-out cases, never used to choose.
- **GN cap hits**: time steps whose Gauss–Newton solve hit the 20-iteration cap. **compiled-query memory**: XLA memory analysis (arguments + outputs + temporaries) of the jitted query.
- **POD-LSPG zero / ic**: linear basis with zero reference or with the initial field as reference. **ours_q0 / ours_q256**: frozen project checkpoint without / with 256 corrections.
- **FOM**: full-order model on the same grid; `fom_fft_tight` is the reference itself (error 0 by construction) and is the named FOM for "× FOM"; `fom_nt1e4_dt005` is the same solver with loose tolerances.
- **× FOM**: named-FOM median query time divided by the arm's median query time, same allocation; below 1 the reduced model is slower than solving the full problem.
- **median evolved**: median over the 32 cases of each case's worst evolved-time error.
- **data-matched**: the Kim autoencoder trained on 576 trajectories (the count the project bank was trained on) instead of 112.
- **admissible**: a Kim row counts as the validated method only if it ran the code and activation that passed the reproduction gate.
- **precheck**: before training, weights + gradient + two Adam moments of the dense encoder are compared with device memory; if larger, the arm is recorded as not fitting and not attempted.
- **epochs / wall budget**: training passes over the fit snapshots; the wall budget is a per-arm time limit added by this lane (the paper allows up to 10 000 epochs).
