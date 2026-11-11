### Table 1. Test errors and inference time at $256^2$ (test64, A100 80GB, batch 1)

| method | training trajectories | seed | params | worst % | median % | time ms (median [p05, p95]) | converged |
|---|---|---|---|---|---|---|---|
| ROM accurate (`R384`) | 576 bank + 4,608 head | — | — | 0.875 | 0.0264 | 45.2 | n/a |
| ROM fast (`R96`) | 576 bank + 4,608 head | — | — | 3.85 | 0.529 | 14.2 | n/a |
| ROM extra (`R128`) | 576 bank + 4,608 head | — | — | 2.55 | 0.268 | 18.3 | n/a |
| FOM at the ROM-accurate tolerance | — | — | — | 0.109 | 0.0299 | 17.4 | n/a |
| FNO, paper (128 traj., 3000 s cap) | 128 | 0 | — | 8.76 | — | — | no (wall cap) |
| U-Net, paper (128 traj., 3000 s cap) | 128 | 0 | — | 5.98 | — | — | no (wall cap) |
| Transolver, paper (128 traj., 3000 s cap) | 128 | 0 | — | 6.83 | — | — | no (wall cap) |

### Table 2. Training record of every run

| run | rung | seed | steps | epochs | best step | val. mean % | val. worst % | final lr | gain of best val. over last 10 validations % | stop | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FNO F0 | 128 | 0 | 28,000 | 2000.0 | 28,000 | 3.79 | 12.6 | 6.3e-05 | 0.0991 | early_stopping | yes |
| FNO F0 | 576 | 0 | 31,000 | 484.4 | 29,500 | 0.64 | 1.64 | 0.00025 | 5.76 | time_limit | **no** |
| FNO F0 | 4,608 | 0 | 89,000 | 163.6 | 89,000 | 0.259 | 0.913 | 1e-05 | 0.938 | time_limit | **no** |
| FNO F0 | 4,608 | 1 | 61,000 | 112.1 | 61,000 | 0.243 | 0.843 | 3.1e-05 | 6.02 | time_limit | **no** |
| FNO F0 | 4,608 | 2 | 60,000 | 110.3 | 60,000 | 0.292 | 0.9 | 1.6e-05 | 1.59 | time_limit | **no** |
| FNO F1 | 4,608 | 0 | 56,000 | 102.9 | 56,000 | 0.267 | 0.858 | 3.1e-05 | 2.50 | time_limit | **no** |
| FNO F2 | 4,608 | 0 | 38,000 | 69.9 | 38,000 | 0.247 | 0.709 | 6.3e-05 | 15.0 | time_limit | **no** |
| U-Net U0 | 128 | 0 | 56,000 | 4000.0 | 46,000 | 1.82 | 5.03 | 1e-05 | 0 | early_stopping | yes |
| U-Net U0 | 576 | 0 | 56,500 | 882.8 | 53,000 | 0.387 | 1.15 | 1e-05 | 1.76 | time_limit | **no** |
| U-Net U0 | 4,608 | 0 | 79,500 | 146.1 | 76,000 | 0.259 | 0.63 | 1e-05 | 1.39 | time_limit | **no** |
| U-Net U0 | 4,608 | 1 | 107,500 | 197.6 | 98,500 | 0.266 | 0.717 | 1e-05 | 0 | early_stopping | yes |
| U-Net U0 | 4,608 | 2 | 111,500 | 205.0 | 108,000 | 0.253 | 0.636 | 1e-05 | 2.29 | time_limit | **no** |
| U-Net U1 | 4,608 | 0 | 74,500 | 136.9 | 73,500 | 0.288 | 0.772 | 1e-05 | 2.52 | time_limit | **no** |
| U-Net U2 | 4,608 | 0 | 110,000 | 202.2 | 110,000 | 0.375 | 1.15 | 1e-05 | 1 | time_limit | **no** |

### Table 3. $1024^2$ test (networks trained at $256^2$, no retraining)

| method | rung | seed | zero-shot worst % | zero-shot median % | zero-shot ms | coarse+bilinear worst % | coarse+bilinear median % | coarse+bilinear ms |
|---|---|---|---|---|---|---|---|---|
| ROM accurate (`R384`) | — | — | 1.25 | 0.0292 | 49.1 | | | |
| ROM fast (`R96`) | — | — | 4.25 | 0.558 | 19.1 | | | |
| FOM at the ROM-accurate tolerance | — | — | 0.129 | 0.0325 | 74.1 | | | |
| FNO, paper (trained at $1024^2$, 128 traj.) | 128 | 0 | 14.8 | — | — | | | |
| U-Net, paper (trained at $1024^2$, 128 traj.) | 128 | 0 | 11.5 | — | — | | | |
| Transolver, paper (trained at $1024^2$, 128 traj.) | 128 | 0 | 14.9 | — | — | | | |

### Table 4. Verdict inputs (PROTOCOL section 6)

| family | validation scores of the grid at 4,608 (seed 0), % | selected arm | seeds | worst % min / median / max | median % min / median / max | time ms | worst: ROM advantage | ratio to ROM accurate | median: ROM advantage | ratio | dominates ROM fast (worst) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FNO | F0 0.259, F1 0.267, F2 0.247 | F2 | not evaluated | | | | | | | | |
| U-Net | U0 0.259, U1 0.288, U2 0.375 | U0 | not evaluated | | | | | | | | |
