# Stored ablation rows of the paper's method (no new runs)

Every number below is read from a saved JSON of an earlier lane (path relative to `worktrees/`). Cohorts differ
from row group to row group and are stated; development-cohort rows are **not** test results.

## A. What the decoder contributes (Burgers 2D, 256 mesh, development cohort of 6 cases)

Source: `2026-09-14-head-ablation/experiments/head-ablation/artifacts/abl01/audit.json`, `checks.arm_table[]`
with `intervals == 256`; Slurm job 3711424, NVIDIA A100 80GB PCIe; cohort = 4 opened + 2 fresh development cases
(`params_draw(7090702,4) + params_draw(911702,2)`). Error = `worst_same_grid_percent`, time = `median_gpu_ms`.

| variant (latent size) | arm key | worst error % | ms, all nodes | ms, hyper-reduced |
|---|---|---|---|---|
| bank + neural head (16) | `a_neural_dense` / `a_neural_eq` | 2.563 | 281.7 | 47.6 |
| bank + linear decoder (16) | `b_linear_dec_dense` / `_eq` | 56.93 | 160.9 | 19.9 |
| bank + quadratic decoder (16) | `c_quad_dec_dense` / `_eq` | 31.83 | 189.5 | 28.0 |
| POD-LSPG, no bank (16) | `e_pod16_dense` / `_eq` | 61.65 | 46.0 | 16.3 |
| POD-LSPG, no bank (128) | `e_pod128_dense` | 10.12 | 328.3 | — |
| bank coefficients free, no head (512) | `d_freebank_dense` | 0.603 | 2417.6 | — |

Reading: at latent size 16 the nonlinear head is what makes the error small (2.6 % against 57 % linear, 32 %
quadratic, 62 % POD); the hyper-reduced evaluation changes the time (281.7 → 47.6 ms, ×5.9) and not the error
(identical to the printed digits).

## B. What the precomputed quadrature contributes (span solver, 1024 mesh, development cohort of 6 cases)

Source: `2026-09-24-burgers-eq-tol-knobs/experiments/burgers-eq-tol-knobs/checks/e1024-summary.json`,
`ladder.ladders`; job 4303886, NVIDIA A100 80GB PCIe, cohort dev6.

| span size | all-node evaluation, ms | lattice quadrature, ms | speed-up | worst error %, all-node / lattice |
|---|---|---|---|---|
| R' = 384 | 1262.7 | 49.3 | ×25.6 | 0.21075 / 0.21082 |
| R' = 96 | 298.9 | 17.9 | ×16.7 | 2.78703 / 2.78703 |

## C. Linear and quadratic subspaces on the test cohort (1024 mesh, test64)

Source: `2026-09-25-t3-burgers-test/experiments/t3-burgers-test/checks/t31024-summary.json`, `arms[]`; job 4327211,
NVIDIA A100 80GB PCIe, cohort test64 (same sha256 as this lane's). Worst / median error %, ms.

| method | arm key | worst % | median % | ms |
|---|---|---|---|---|
| POD rank 16, all nodes | `pod16_M64_dense` | 41.56 | 13.04 | 186.5 |
| quadratic manifold, 16 | `qman16_quad_M64` | 34.14 | 7.96 | 758.0 |
| bank + nonlinear head, 16 | `q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2` | 8.42 | 0.709 | 35.3 |
| bank + span, R' = 384 (all-node arm of that file) | `bank384_M1536_lat64_g1em06` | 1.25 | 0.029 | 316.9 |

No POD or quadratic-manifold arm is stored for the 256 mesh on test64 (`t256-summary.json` has none); group A is the
only 256-mesh evidence for those and it is a development cohort.

## D. What is not stored

No stored run removes the coordinate bank while keeping a neural head *and* hyper-reduction on the test cohort;
the closest are POD rows (no bank, linear) and this lane's CROM-style arm (no bank, nonlinear neural-field decoder,
collocation sub-sampling).
