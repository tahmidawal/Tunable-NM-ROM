### Table 1. Every run of this lane: size, data, cost, validation (test not yet evaluated)

| family | arm | params | training trajectories (+256 validation) | checkpoint | steps (best) | stop | converged | training GPU-h (GPU) | val. mean % | val. worst % | test median % | test worst % | query ms (p05–p95) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Reference (same test64, same metric, A100 80GB): ROM accurate median 0.0264 %, worst 0.875 %, 45.2 ms; ROM fast median 0.529 %, worst 3.85 %, 14.2 ms; full-order solver at the ROM-accurate tolerance median 0.0299 %, worst 0.109 %, 17.4 ms. Gain columns of Table 2 show whether a run was still improving.

### Table 2. Training state at the end of each run

| run | final lr | gain of best val. over last 10 / 20 validations % | ms/step | job |
|---|---|---|---|---|

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

### Table 5. Cost

- Training: 0.0 GPU-hours over 0 runs (wall time on the run's own GPU).
- Extra data: 32,256 trajectories in 28 chunks, solver time 0.69 GPU-hours (median 0.0723 s per trajectory; jobs 7157491, 7157492; max relative Newton residual 1e-06). Template pool (4,608 trajectories): 0.12 GPU-hours.

### Table 6. Headline per family (validation-selected over all runs) and closest test median

| family | validation-selected run | params | training traj. | test median % | test worst % | ms | runs at or below 0.0264 % median | lowest test median of any run (run, ms) |
|---|---|---|---|---|---|---|---|---|
| FNO | — | 0 | 0 |  |  |  | 0 |  |
| U-Net | — | 0 | 0 |  |  |  | 0 |  |
