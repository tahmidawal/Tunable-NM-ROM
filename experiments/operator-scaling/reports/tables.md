# Held-out FNO and U-Net scaling on 2D viscous Burgers

The registered single-precision operators were scored against regenerated double-precision reference fields. Held-out errors and timings are final for the saved checkpoints in this allocation. Rows marked interrupted were stopped during training and preserve their last validation-selected best weights.

### Table 1. Every run of this lane: size, data, cost, validation and test (test64, NVIDIA A100 80GB PCIe, batch 1)

| family | arm | params | training trajectories (+256 validation) | checkpoint | steps (best) | stop | converged | training GPU-h (GPU) | val. mean % | val. worst % | test median % | test worst % | query ms (p05–p95) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FNO | Fw128m32L4 | 71,504,517 | 4,352 | final best | 193,500 (191,000) | interrupted_by_cancellation | **no** | 4.7 (H200) | 0.1582 | 0.594 | 0.137 | 0.346 | 9.09 (9.03–9.14) |
| FNO | Fw128m64L4 (cosine, A1) | 277,025,413 | 4,352 | final best | 170,500 (168,000) | interrupted_by_cancellation | **no** | 7.5 (RTX PRO 6000) | 0.2183 | 0.634 | 0.195 | 0.421 | 11.1 (11.1–11.2) |
| FNO | Fw128m64L4 | 277,025,413 | 4,352 | final best | 200,000 (200,000) | step_cap | **no** | 8.8 (RTX PRO 6000) | 0.1632 | 0.678 | 0.135 | 0.367 | 11.2 (11.2–11.3) |
| FNO | Fw192m48L4 | 354,343,877 | 4,352 | final best | 115,500 (113,000) | interrupted_by_cancellation | **no** | 7.5 (RTX PRO 6000) | 0.1593 | 0.666 | 0.135 | 0.29 | 17.3 (17.2–17.4) |
| FNO | Fw128m64L4 | 277,025,413 | 8,960 | final best | 200,000 (200,000) | step_cap | **no** | 5.6 (H200) | 0.1365 | 0.465 | 0.124 | 0.289 | 11.2 (11.2–11.3) |
| FNO | Fw128m64L4 (cosine, A1) | 277,025,413 | 18,176 | final best | 167,500 (151,000) | interrupted_by_cancellation | **no** | 7.4 (RTX PRO 6000) | 0.1994 | 0.491 | 0.176 | 0.4 | 11.1 (11.1–11.2) |
| FNO | Fw128m64L4 | 277,025,413 | 18,176 | final best | 200,000 (199,000) | step_cap | **no** | 5.6 (H200) | 0.1436 | 0.481 | 0.131 | 0.305 | 11.2 (11.2–11.3) |
| FNO | Fw128m64L4 (cosine, A1) | 277,025,413 | 36,608 | final best | 400,000 (398,500) | step_cap | **no** | 11.2 (H200) | 0.07289 | 0.219 | 0.0654 | 0.175 | 11 (11–11.1) |
| FNO | Fw128m64L4 (cosine, A1) | 277,025,413 | 36,608 | final best | 137,000 (132,500) | interrupted_by_cancellation | **no** | 3.8 (H200) | 0.2592 | 0.633 | 0.225 | 0.512 | 11.1 (11.1–11.2) |
| FNO | Fw128m64L4 | 277,025,413 | 36,608 | final best | 332,000 (322,000) | early_stopping | yes | 9.4 (H200) | 0.1344 | 0.455 | 0.12 | 0.311 | 11.2 (11.2–11.3) |
| FNO | Fw128m64L4 | 277,025,413 | 36,608 | best ≤200k | 332,000 (198,500) | early_stopping | yes | 9.4 (H200) | 0.1554 | 0.531 | 0.141 | 0.36 | 11.2 (11.2–11.3) |
| FNO | Fw192m48L4 (cosine, A1) | 354,343,877 | 36,608 | final best | 6,500 (6,000) | interrupted_by_cancellation | **no** | 0.3 (H200) | 1.04 | 2.7 | 0.965 | 2.64 | 17.2 (17.1–17.3) |
| U-Net | Ub48 | 17,462,021 | 4,352 | final best | 85,500 (84,500) | interrupted_by_cancellation | **no** | 1.1 (RTX PRO 6000) | 0.2556 | 0.649 | 0.236 | 0.462 | 7.39 (7.37–7.55) |
| U-Net | Ub64 | 31,038,469 | 4,352 | final best | 181,000 (176,000) | early_stopping | yes | 3.5 (RTX PRO 6000) | 0.212 | 0.541 | 0.177 | 0.375 | 10 (10–10.3) |
| U-Net | Ub96 | 69,825,029 | 4,352 | final best | 144,500 (134,500) | early_stopping | yes | 5.4 (RTX PRO 6000) | 0.2391 | 0.621 | 0.216 | 0.536 | 18.7 (18.7–19.2) |
| U-Net | Ub128 | 124,123,141 | 4,352 | final best | 113,000 (103,000) | early_stopping | yes | 6.2 (RTX PRO 6000) | 0.309 | 1.11 | 0.282 | 0.643 | 29.5 (29.5–29.5) |
| U-Net | Ub192 | 279,254,021 | 4,352 | final best | 105,000 (103,500) | interrupted_by_cancellation | **no** | 15.3 (H200) | 0.3847 | 1.04 | 0.35 | 0.829 | 57 (56.8–57.4) |
| U-Net | Ub128 | 124,123,141 | 8,960 | final best | 50,500 (50,000) | interrupted_by_cancellation | **no** | 3.5 (H200) | 0.3699 | 1.07 | 0.325 | 0.769 | 29.5 (29.5–29.5) |
| U-Net | Ub128 | 124,123,141 | 18,176 | final best | 144,500 (141,500) | interrupted_by_cancellation | **no** | 10.0 (H200) | 0.2587 | 0.863 | 0.241 | 0.575 | 29.5 (29.5–29.5) |
| U-Net | Ub128 (cosine, A1) | 124,123,141 | 36,608 | final best | 107,000 (106,000) | interrupted_by_cancellation | **no** | 7.4 (H200) | 0.3015 | 0.637 | 0.267 | 0.511 | 29.5 (29.5–30) |
| U-Net | Ub128 | 124,123,141 | 36,608 | final best | 180,000 (170,000) | early_stopping | yes | 12.6 (H200) | 0.216 | 0.634 | 0.196 | 0.461 | 29.5 (29.5–29.5) |

Reference (same test64, same metric, historical separate allocations): ROM accurate median 0.0264 %, worst 0.875 %, 45.2 ms; ROM fast median 0.529 %, worst 3.85 %, 14.2 ms; full-order solver at the ROM-accurate tolerance median 0.0299 %, worst 0.109 %, 17.4 ms. Gain columns of Table 2 show whether a run was still improving.

### Table 2. Training state at the end of each run

| run | final lr | gain of best val. over last 10 / 20 validations % | ms/step | job |
|---|---|---|---|---|
| a1F/Fw128m64L4-n36864-s0-cosine | 1e-06 | 0.111 / 0.2 | 101 | 7158479 |
| a1U/Ub128-n36864-s0-cosine | 0.00013 | 2.41 / 5.47 | — | 7163215 |
| a2F18/Fw128m64L4-n18432-s0-cosine | 0.00063 | 0 / 0 | — | 7163213 |
| a2F4/Fw128m64L4-n4608-s0-cosine | 0.00062 | 6.72 / 6.72 | — | 7163214 |
| a3a/Fw128m64L4-n36864-s0-cosine | 0.00095 | 15.5 / 15.5 | — | 7164142 |
| a3b/Fw192m48L4-n36864-s0-cosine | 0.001 | 48 / — | — | 7175422 |
| p1F/Fw128m64L4-n36864-s0 | 1e-05 | 0 / 0 | 101 | 7157811 |
| p1F/Fw128m64L4-n9216-s0 | 1e-05 | 0.215 / 0.458 | 101 | 7157811 |
| p1U/Ub128-n36864-s0 | 1e-05 | 0 / 0 | 252 | 7157812 |
| p1U/Ub128-n9216-s0 | 1e-05 | 6.46 / 15.3 | — | 7157812 |
| p3/Fw128m64L4-n18432-s0 | 1e-05 | 1 / 1.18 | 101 | 7157948 |
| p3/Ub128-n18432-s0 | 1e-05 | 2.31 / 2.83 | — | 7157948 |
| szF2/Fw128m64L4-n4608-s0 | 1e-05 | 0.817 / 1.22 | 158 | 7157808 |
| szF2/Fw192m48L4-n4608-s0 | 1e-05 | 0.87 / 2.15 | — | 7157808 |
| szF3/Fw128m32L4-n4608-s0 | 1e-05 | 0.225 / 0.469 | — | 7163881 |
| szU192/Ub192-n4608-s0 | 1e-05 | 2.47 / 2.47 | — | 7157949 |
| szU2/Ub128-n4608-s0 | 1e-05 | 0 / 0 | 198 | 7157810 |
| szU2/Ub48-n4608-s0 | 1e-05 | 4.64 / 4.84 | — | 7157810 |
| szU2/Ub64-n4608-s0 | 1e-05 | 0 / 0.0681 | 69.7 | 7157810 |
| szU2/Ub96-n4608-s0 | 1e-05 | 0 / 0 | 134 | 7157810 |

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

### Table 4. Log–log fits of test error (test median/worst %)

| fit | points | slope [95 % CI] | R² | resid. SE (log10) | extrapolation allowed (protocol §7) | size/data needed for median 0.0264 % [95 % CI] |
|---|---|---|---|---|---|---|
| FNO size, median | 3 | -0.008 [-0.017, 0.000] | 0.993 | 0.000 | no (n_ge_4=no, r2_ge_0p9=yes, slope_ci_below_0=no, monotone=yes) |  |
| FNO size, worst | 3 | -0.058 [-1.698, 1.582] | 0.168 | 0.068 | n/a (worst case) |  |
| FNO data (Fw128m64L4), median | 4 | 0.023 [-0.135, 0.181] | 0.167 | 0.025 | no (n_ge_4=yes, r2_ge_0p9=no, slope_ci_below_0=no, monotone=no) |  |
| FNO data (Fw128m64L4), worst | 4 | -0.001 [-0.396, 0.393] | 0.000 | 0.063 | n/a (worst case) |  |
| FNO data, cosine (Fw128m64L4), median | 3 | -0.448 [-4.621, 3.725] | 0.650 | 0.219 | no (n_ge_4=no, r2_ge_0p9=no, slope_ci_below_0=no, monotone=yes) |  |
| FNO data, cosine (Fw128m64L4), worst | 3 | -0.356 [-3.908, 3.196] | 0.619 | 0.186 | n/a (worst case) |  |
| U-Net size, median | 5 | 0.184 [-0.088, 0.457] | 0.607 | 0.082 | no (n_ge_4=yes, r2_ge_0p9=no, slope_ci_below_0=no, monotone=no) |  |
| U-Net size, worst | 5 | 0.251 [0.034, 0.467] | 0.819 | 0.065 | n/a (worst case) |  |
| U-Net data (Ub128), median | 4 | -0.194 [-0.600, 0.212] | 0.679 | 0.065 | no (n_ge_4=yes, r2_ge_0p9=no, slope_ci_below_0=no, monotone=no) |  |
| U-Net data (Ub128), worst | 4 | -0.181 [-0.633, 0.271] | 0.597 | 0.072 | n/a (worst case) |  |

Query time against parameters (measured, A100 80GB, batch 1): FNO: slope 0.285, R² 0.402; U-Net: slope 0.744, R² 0.996

### Table 5. Cost

- Training: 136.9 GPU-hours over 20 runs (wall time on the run's own GPU).
- Extra data: 32,256 trajectories in 28 chunks, solver time 0.69 GPU-hours (median 0.0723 s per trajectory; jobs 7157491, 7157492; max relative Newton residual 1e-06). Template pool (4,608 trajectories): 0.12 GPU-hours.

### Table 6. Headline per family (validation-selected over all runs) and closest test median

| family | validation-selected run | params | training traj. | test median % | test worst % | ms | runs at or below 0.0264 % median | lowest test median of any run (run, ms) |
|---|---|---|---|---|---|---|---|---|
| FNO | a1F/Fw128m64L4-n36864-s0-cosine | 277,025,413 | 36,608 | 0.0654 | 0.175 | 11 | 0 | 0.0654 (a1F/Fw128m64L4-n36864-s0-cosine, 11 ms) |
| U-Net | szU2/Ub64-n4608-s0 | 31,038,469 | 4,352 | 0.177 | 0.375 | 10 | 0 | 0.177 (szU2/Ub64-n4608-s0, 10 ms) |

All query times in the test table were measured on the same GPU for this study. Historical reference and template rows came from separate jobs and cannot supply a paired speed ratio. The lowest test median is descriptive. The validation-selected row is the headline. Missing training rungs and interruptions limit size and data fits, including any extrapolation.

## Glossary

FNO means Fourier neural operator. U-Net is an encoder–decoder with skip connections. A rung is one training-set size. A parameter is one real-valued model weight, counting both parts of a complex number. A checkpoint stores weights. Final best means the weights with the lowest validation score during the available training. Validation selects a model. Test cases measure it after selection. The error for each case is the largest relative field error over its five evolved times, normalized by the initial field norm. Median, mean and worst summarize those case errors. A timed query starts with device-resident inputs and ends with six device-resident fields. p05/p95 are the fifth and ninety-fifth percentiles of repeated batch-one latency. GPU-hours measure training or generation elapsed time on the allocated accelerator. LR is the learning rate. A step is one optimizer update. Regression slopes use log coordinates. R² measures explained variation, SE is standard error and CI is a confidence interval.
