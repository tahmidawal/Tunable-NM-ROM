### Table 1. Every run of this lane: size, data, cost, validation (test not yet evaluated)

| family | arm | params | training trajectories (+256 validation) | checkpoint | steps (best) | stop | converged | training GPU-h (GPU) | val. mean % | val. worst % | test median % | test worst % | query ms (p05–p95) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FNO | Fw128m64L4 | 277,025,413 | 4,352 | final best | 158,500 (154,000) | running | **no** | 7.0 (RTX PRO 6000) | 0.1728 | 0.695 |  |  |  |
| FNO | Fw128m64L4 | 277,025,413 | 18,176 | final best | 200,000 (199,000) | step_cap | **no** | 5.6 (H200) | 0.1436 | 0.481 |  |  |  |
| FNO | Fw128m64L4 (cosine, A1) | 277,025,413 | 36,608 | final best | 198,500 (160,500) | running | **no** | 5.6 (H200) | 0.1683 | 0.39 |  |  |  |
| FNO | Fw128m64L4 | 277,025,413 | 36,608 | final best | 245,000 (242,500) | running | **no** | 6.9 (H200) | 0.1469 | 0.489 |  |  |  |
| U-Net | Ub96 | 69,825,029 | 4,352 | final best | 19,500 (19,500) | running | **no** | 0.7 (RTX PRO 6000) | 0.6544 | 1.53 |  |  |  |
| U-Net | Ub128 | 124,123,141 | 4,352 | final best | 113,000 (103,000) | early_stopping | yes | 6.2 (RTX PRO 6000) | 0.309 | 1.11 |  |  |  |
| U-Net | Ub192 | 279,254,021 | 4,352 | final best | 41,000 (39,000) | running | **no** | 6.0 (H200) | 0.5535 | 1.45 |  |  |  |
| U-Net | Ub128 | 124,123,141 | 18,176 | final best | 10,500 (8,000) | running | **no** | 0.7 (H200) | 1.905 | 4.69 |  |  |  |
| U-Net | Ub128 | 124,123,141 | 36,608 | final best | 97,500 (95,000) | running | **no** | 6.8 (H200) | 0.2763 | 0.848 |  |  |  |

Reference (same test64, same metric, A100 80GB): ROM accurate median 0.0264 %, worst 0.875 %, 45.2 ms; ROM fast median 0.529 %, worst 3.85 %, 14.2 ms; full-order solver at the ROM-accurate tolerance median 0.0299 %, worst 0.109 %, 17.4 ms. Gain columns of Table 2 show whether a run was still improving.

### Table 2. Training state at the end of each run

| run | final lr | gain of best val. over last 10 / 20 validations % | ms/step | job |
|---|---|---|---|---|
| a1F/Fw128m64L4-n36864-s0-cosine | 0.00051 | 0 / 0 | — | 7158479 |
| p1F/Fw128m64L4-n36864-s0 | 1e-05 | 0.446 / 1.84 | — | 7157811 |
| p1U/Ub128-n36864-s0 | 1e-05 | 3.51 / 5.53 | — | 7157812 |
| p3/Fw128m64L4-n18432-s0 | 1e-05 | 1 / 1.18 | 101 | 7157948 |
| p3/Ub128-n18432-s0 | 0.0003 | 30.3 / 75.1 | — | 7157948 |
| szF2/Fw128m64L4-n4608-s0 | 1e-05 | 0.482 / 1.36 | — | 7157808 |
| szU192/Ub192-n4608-s0 | 1e-05 | 2.57 / 14.3 | — | 7157949 |
| szU2/Ub128-n4608-s0 | 1e-05 | 0 / 0 | 198 | 7157810 |
| szU2/Ub96-n4608-s0 | 7.5e-05 | 36.1 / 67.3 | — | 7157810 |

### Table 3. Template-lane rows (4,608 rung, earlier budget: 8.1 h on shared GPUs), same test64 and timing

| arm | seed | params | test median % | test worst % | ms |
|---|---|---|---|---|---|
| F0 | 0 | 17,877,317 | 0.22 | 0.477 | 4.06 |
| F0 | 1 | 17,877,317 | 0.195 | 0.435 | 4.06 |
| F0 | 2 | 17,877,317 | 0.233 | 0.492 | 4.06 |
| F1 | 0 | 39,373,125 | 0.199 | 0.463 | 4.27 |
| F2 | 0 | 40,222,181 | 0.189 | 0.483 | 6.6 |
| U0 | 0 | 17,462,021 | 0.213 | 0.453 | 7.4 |
| U0 | 1 | 17,462,021 | 0.224 | 0.502 | 7.41 |
| U0 | 2 | 17,462,021 | 0.203 | 0.506 | 7.38 |
| U1 | 0 | 31,038,469 | 0.216 | 0.459 | 10.1 |
| U2 | 0 | 17,462,021 | 0.32 | 0.682 | 7.4 |

### Table 4. Log–log fits of validation error (validation mean/worst % — interim, not the protocol fit)

| fit | points | slope [95 % CI] | R² | resid. SE (log10) | extrapolation allowed (protocol §7) | size/data needed for median 0.0264 % [95 % CI] |
|---|---|---|---|---|---|---|
| FNO data (Fw128m64L4), median | 3 | -0.084 [-0.588, 0.420] | 0.817 | 0.026 | no (n_ge_4=no, r2_ge_0p9=no, slope_ci_below_0=no, monotone=no) |  |
| FNO data (Fw128m64L4), worst | 3 | -0.178 [-1.057, 0.700] | 0.869 | 0.046 | n/a (worst case) |  |
| U-Net size, median | 3 | -0.066 [-7.203, 7.071] | 0.013 | 0.240 | no (n_ge_4=no, r2_ge_0p9=no, slope_ci_below_0=no, monotone=no) |  |
| U-Net size, worst | 3 | -0.014 [-3.210, 3.181] | 0.003 | 0.108 | n/a (worst case) |  |
| U-Net data (Ub128), median | 3 | 0.143 [-12.415, 12.701] | 0.020 | 0.659 | no (n_ge_4=no, r2_ge_0p9=no, slope_ci_below_0=no, monotone=no) |  |
| U-Net data (Ub128), worst | 3 | 0.043 [-10.717, 10.802] | 0.003 | 0.565 | n/a (worst case) |  |

### Table 5. Cost

- Training: 45.6 GPU-hours over 9 runs (wall time on the run's own GPU).
- Extra data: 32,256 trajectories in 28 chunks, solver time 0.69 GPU-hours (median 0.0723 s per trajectory; jobs 7157491, 7157492; max relative Newton residual 1e-06). Template pool (4,608 trajectories): 0.12 GPU-hours.

### Table 6. Headline per family (validation-selected over all runs) and closest test median

| family | validation-selected run | params | training traj. | test median % | test worst % | ms | runs at or below 0.0264 % median | lowest test median of any run (run, ms) |
|---|---|---|---|---|---|---|---|---|
| FNO | p3/Fw128m64L4-n18432-s0 | 277,025,413 | 18,176 |  |  |  | 0 |  |
| U-Net | p1U/Ub128-n36864-s0 | 124,123,141 | 36,608 |  |  |  | 0 |  |
