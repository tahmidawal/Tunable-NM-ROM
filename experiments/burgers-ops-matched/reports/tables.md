### Table 1. Test errors and inference time at $256^2$ (test64, A100 80GB, batch 1)

| method | training trajectories | seed | params | worst % | median % | time ms (median [p05, p95; min, max] over 320 timed queries) | stopped by the rule |
|---|---|---|---|---|---|---|---|
| ROM accurate (`R384`) | 576 bank + 4,608 head | — | — | 0.875 | 0.0264 | 45.2 | n/a |
| ROM fast (`R96`) | 576 bank + 4,608 head | — | — | 3.85 | 0.529 | 14.2 | n/a |
| ROM extra (`R128`) | 576 bank + 4,608 head | — | — | 2.55 | 0.268 | 18.3 | n/a |
| FOM at the ROM-accurate tolerance | — | — | — | 0.109 | 0.0299 | 17.4 | n/a |
| ROM head only, k=16 (stored, job 4327396) | 4,608 head | — | — | 4.10 | 0.672 | 32.4 | n/a |
| FNO, paper row (128 traj., 3000 s cap; stored, job 4327396) | 128 | 0 | — | 8.76 | 2.59 | 7.45 | no (wall cap) |
| U-Net, paper row (128 traj., 3000 s cap; stored, job 4327396) | 128 | 0 | — | 5.98 | 2.00 | 10.3 | no (wall cap) |
| Transolver, paper row (128 traj., 3000 s cap; stored, job 4327396) | 128 | 0 | — | 6.83 | 2.26 | 11.5 | no (wall cap) |
| FNO F0 | 128 | 0 | 17,877,317 | 9.78 | 2.61 | 4.06 [4.05, 4.09; min 4.05, max 5.07] | yes |
| FNO F0 | 576 | 0 | 17,877,317 | 4.03 | 0.402 | 4.05 [4.04, 4.08; min 4.04, max 4.16] | **no** (time_limit) |
| FNO F0 | 4,608 | 0 | 17,877,317 | 0.477 | 0.22 | 4.06 [4.05, 4.08; min 4.04, max 5.04] | **no** (time_limit) |
| FNO F0 | 4,608 | 1 | 17,877,317 | 0.435 | 0.195 | 4.06 [4.05, 4.08; min 4.04, max 4.44] | **no** (time_limit) |
| FNO F0 | 4,608 | 2 | 17,877,317 | 0.492 | 0.233 | 4.06 [4.05, 4.08; min 4.04, max 4.85] | **no** (time_limit) |
| FNO F1 | 4,608 | 0 | 39,373,125 | 0.463 | 0.199 | 4.27 [4.25, 4.29; min 4.25, max 4.38] | **no** (time_limit) |
| FNO F2 | 4,608 | 0 | 40,222,181 | 0.483 | 0.189 | 6.60 [6.59, 6.64; min 6.58, max 6.68] | **no** (time_limit) |
| U-Net U0 | 128 | 0 | 17,462,021 | 5.01 | 1.23 | 7.39 [7.36, 7.54; min 7.35, max 7.60] | yes |
| U-Net U0 | 576 | 0 | 17,462,021 | 0.741 | 0.27 | 7.38 [7.36, 7.40; min 7.36, max 7.42] | **no** (time_limit) |
| U-Net U0 | 4,608 | 0 | 17,462,021 | 0.453 | 0.213 | 7.40 [7.38, 7.44; min 7.37, max 7.63] | **no** (time_limit) |
| U-Net U0 | 4,608 | 1 | 17,462,021 | 0.502 | 0.224 | 7.41 [7.37, 7.44; min 7.36, max 7.55] | yes |
| U-Net U0 | 4,608 | 2 | 17,462,021 | 0.506 | 0.203 | 7.38 [7.37, 7.41; min 7.36, max 7.42] | **no** (time_limit) |
| U-Net U1 | 4,608 | 0 | 31,038,469 | 0.459 | 0.216 | 10.1 [10.0, 10.4; min 9.99, max 10.4] | **no** (time_limit) |
| U-Net U2 | 4,608 | 0 | 17,462,021 | 0.682 | 0.32 | 7.40 [7.38, 7.43; min 7.37, max 7.53] | **no** (time_limit) |

### Table 2. Training record of every run

Round 1 = first training; round 2 = the single continuation of amendment A5 (cumulative steps). A round-1 row marked superseded was continued and is not evaluated.

| run | job dir | round | rung | seed | steps | epochs | best step | val. mean % | val. worst % | final lr | gain of best val. over last 10 / 20 validations % | validations since best | stop | stopped by the rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FNO F0 | tr6 | 1 | 128 | 0 | 28,000 | 2000.0 | 28,000 | 3.788 | 12.6 | 6.3e-05 | 0.0991 / 0.0991 | 0 | early_stopping | yes |
| FNO F0 | tr6 | 1 (superseded) | 576 | 0 | 31,000 | 484.4 | 29,500 | 0.6399 | 1.64 | 0.00025 | 5.76 / 10.5 | 3 | time_limit | **no** |
| FNO F0 | c3 | 2 | 576 | 0 | 123,500 | 1929.7 | 121,500 | 0.5051 | 1.40 | 1e-05 | 0.226 / 0.649 | 4 | time_limit | **no** |
| FNO F0 | tr1 | 1 (superseded) | 4,608 | 0 | 89,000 | 163.6 | 89,000 | 0.2593 | 0.913 | 1e-05 | 0.938 / 1.16 | 0 | time_limit | **no** |
| FNO F0 | c2 | 2 | 4,608 | 0 | 123,000 | 226.1 | 121,000 | 0.2496 | 0.878 | 1e-05 | 0.463 / 1.13 | 4 | time_limit | **no** |
| FNO F0 | tr2 | 1 (superseded) | 4,608 | 1 | 61,000 | 112.1 | 61,000 | 0.2433 | 0.843 | 3.1e-05 | 6.02 / 6.70 | 0 | time_limit | **no** |
| FNO F0 | c2 | 2 | 4,608 | 1 | 95,000 | 174.6 | 92,500 | 0.2268 | 0.83 | 1.6e-05 | 1.17 / 1.64 | 5 | time_limit | **no** |
| FNO F0 | tr3 | 1 (superseded) | 4,608 | 2 | 60,000 | 110.3 | 60,000 | 0.2917 | 0.9 | 1.6e-05 | 1.59 / 2.33 | 0 | time_limit | **no** |
| FNO F0 | c2 | 2 | 4,608 | 2 | 94,000 | 172.8 | 91,500 | 0.2718 | 0.861 | 1e-05 | 0.482 / 1.06 | 5 | time_limit | **no** |
| FNO F1 | tr4b | 1 (superseded) | 4,608 | 0 | 56,000 | 102.9 | 56,000 | 0.2669 | 0.858 | 3.1e-05 | 2.50 / 4.41 | 0 | time_limit | **no** |
| FNO F1 | c3 | 2 | 4,608 | 0 | 147,500 | 271.1 | 146,500 | 0.2256 | 0.788 | 1e-05 | 0.357 / 0.744 | 2 | time_limit | **no** |
| FNO F2 | tr5 | 1 (superseded) | 4,608 | 0 | 38,000 | 69.9 | 38,000 | 0.247 | 0.709 | 6.3e-05 | 15.0 / 16.9 | 0 | time_limit | **no** |
| FNO F2 | c1 | 2 | 4,608 | 0 | 59,500 | 109.4 | 59,500 | 0.2119 | 0.638 | 3.1e-05 | 2.60 / 4.33 | 0 | time_limit | **no** |
| U-Net U0 | tr6 | 1 | 128 | 0 | 56,000 | 4000.0 | 46,000 | 1.825 | 5.03 | 1e-05 | 0 / 0 | 20 | early_stopping | yes |
| U-Net U0 | tr6 | 1 (superseded) | 576 | 0 | 56,500 | 882.8 | 53,000 | 0.3874 | 1.15 | 1e-05 | 1.76 / 3.21 | 7 | time_limit | **no** |
| U-Net U0 | c2 | 2 | 576 | 0 | 84,500 | 1320.3 | 84,500 | 0.3303 | 1.01 | 1e-05 | 1.72 / 3.49 | 0 | time_limit | **no** |
| U-Net U0 | tr1 | 1 (superseded) | 4,608 | 0 | 79,500 | 146.1 | 76,000 | 0.2586 | 0.63 | 1e-05 | 1.39 / 3.83 | 7 | time_limit | **no** |
| U-Net U0 | c1 | 2 | 4,608 | 0 | 106,500 | 195.8 | 99,500 | 0.2312 | 0.596 | 1e-05 | 0 / 4.40 | 14 | time_limit | **no** |
| U-Net U0 | tr2 | 1 | 4,608 | 1 | 107,500 | 197.6 | 98,500 | 0.2656 | 0.717 | 1e-05 | 0 / 0.0397 | 18 | early_stopping | yes |
| U-Net U0 | tr3 | 1 (superseded) | 4,608 | 2 | 111,500 | 205.0 | 108,000 | 0.2534 | 0.636 | 1e-05 | 2.29 / 2.71 | 7 | time_limit | **no** |
| U-Net U0 | c1 | 2 | 4,608 | 2 | 138,500 | 254.6 | 138,500 | 0.2342 | 0.612 | 1e-05 | 2.35 / 3.87 | 0 | time_limit | **no** |
| U-Net U1 | tr4b | 1 (superseded) | 4,608 | 0 | 74,500 | 136.9 | 73,500 | 0.288 | 0.772 | 1e-05 | 2.52 / 4.34 | 2 | time_limit | **no** |
| U-Net U1 | c3 | 2 | 4,608 | 0 | 120,500 | 221.5 | 119,500 | 0.2472 | 0.639 | 1e-05 | 1.58 / 2.39 | 2 | time_limit | **no** |
| U-Net U2 | tr5 | 1 (superseded) | 4,608 | 0 | 110,000 | 202.2 | 110,000 | 0.3746 | 1.15 | 1e-05 | 1 / 1.73 | 0 | time_limit | **no** |
| U-Net U2 | c1 | 2 | 4,608 | 0 | 140,500 | 258.3 | 135,500 | 0.359 | 1.12 | 1e-05 | 0 / 1.00 | 10 | time_limit | **no** |

### Table 3. $1024^2$ test (networks trained at $256^2$, no retraining)

| method | rung | seed | zero-shot worst % | zero-shot median % | zero-shot ms | coarse+bilinear worst % | coarse+bilinear median % | coarse+bilinear ms |
|---|---|---|---|---|---|---|---|---|
| ROM accurate (`R384`) | — | — | 1.25 | 0.0292 | 49.1 | | | |
| ROM fast (`R96`) | — | — | 4.25 | 0.558 | 19.1 | | | |
| FOM at the ROM-accurate tolerance | — | — | 0.129 | 0.0325 | 74.1 | | | |
| FNO, paper row (trained at $1024^2$, 128 traj.; stored, job 4327389) | 128 | 0 | 14.8 | 4.12 | 100.7 | | | |
| U-Net, paper row (trained at $1024^2$, 128 traj.; stored, job 4327389) | 128 | 0 | 11.5 | 2.62 | 53.9 | | | |
| Transolver, paper row (trained at $1024^2$, 128 traj.; stored, job 4327389) | 128 | 0 | 14.9 | 4.53 | 135.1 | | | |
| FNO F0 | 4,608 | 0 | 3.17 | 0.753 | 52.1 | 3.00 | 0.742 | 4.19 |
| FNO F0 | 4,608 | 1 | 2.99 | 0.7 | 52.2 | 2.85 | 0.717 | 4.17 |
| FNO F0 | 4,608 | 2 | 3.04 | 0.758 | 52.2 | 2.89 | 0.761 | 4.17 |
| FNO F1 | 4,608 | 0 | 2.91 | 0.747 | 52.5 | 2.74 | 0.746 | 4.38 |
| FNO F2 | 4,608 | 0 | 2.98 | 0.717 | 85.0 | 2.84 | 0.756 | 6.75 |
| U-Net U0 | 4,608 | 0 | 53.5 | 19.4 | 90.6 | 2.95 | 0.703 | 7.50 |
| U-Net U0 | 4,608 | 1 | 54.0 | 21.0 | 90.6 | 2.86 | 0.736 | 7.50 |
| U-Net U0 | 4,608 | 2 | 53.4 | 19.6 | 90.7 | 2.84 | 0.708 | 7.53 |
| U-Net U1 | 4,608 | 0 | 53.0 | 19.2 | 134.7 | 2.86 | 0.73 | 10.1 |
| U-Net U2 | 4,608 | 0 | 52.2 | 21.0 | 90.6 | 3.04 | 0.747 | 7.50 |

### Table 4. Verdict inputs (PROTOCOL section 6)

| family | validation scores of the grid at 4,608 (seed 0), % | selected arm | seeds | worst % min / median / max | median % min / median / max | time ms | worst: ROM advantage | ratio to ROM accurate | median: ROM advantage | ratio | dominates ROM fast (worst) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FNO | F0 0.25, F1 0.226, F2 0.212 | F2 | [0] | 0.483 / 0.483 / 0.483 | 0.189 / 0.189 / 0.189 | 6.60 | lost | 0.552× | survives | 7.16× | yes |
| U-Net | U0 0.231, U1 0.247, U2 0.359 | U0 | [0, 1, 2] | 0.453 / 0.502 / 0.506 | 0.203 / 0.213 / 0.224 | 7.40 | lost | 0.573× | survives | 8.07× | yes |

Ratios are operator / ROM accurate setting (above 1: the operator is less accurate). Time is the median over the seeds of the per-run median.
