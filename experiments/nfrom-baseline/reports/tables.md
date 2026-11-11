### Test cohort (test64, 256 mesh, NVIDIA A100 80GB PCIe; one evaluation, job 7128546,7128549,7128551)

| method | worst error % | median error % | time per query, ms (median) | p5–p95 ms | latent-only ms | validation worst % | source |
|---|---|---|---|---|---|---|---|
| CROM-style k=16 (C16-w256-lr3, 600k steps; collocation LM, M=2048, 4 it/step; float64; accurate) | 22.435 | 0.426 | 2817.5 | 2816.0–2818.6 | 2797.8 | 2.047 | job 7128546, 3 warm-up, 1x64 timed |
| CROM-style k=16 (C16-w256-lr3, 600k steps; collocation LM, M=2048, 4 it/step; float32; accurate-float32) | 22.435 | 0.426 | 2055.1 | 2053.7–2056.4 | 2039.6 | 2.048 | job 7128546, 20 warm-up, 5x64 timed |
| CROM-style k=16 (C16-w128, 120k steps; collocation LM, M=2048, 2 it/step; float64; fast) | 16.896 | 1.109 | 506.9 | 505.7–508.3 | 500.2 | 3.136 | job 7128546, 20 warm-up, 5x64 timed |
| CROM-style k=16 (C16-w128, 120k steps; collocation LM, M=2048, 2 it/step; float32; fast-float32) | 16.897 | 1.109 | 306.4 | 305.2–307.3 | 302.6 | 3.136 | job 7128546, 20 warm-up, 5x64 timed |
| CROM-style k=32 (C32-w512, 600k steps; collocation LM, M=2048, 4 it/step, refined z0; float64; accurate) | 58.243 | 1.749 | 15196.1 | 15162.8–15230.1 | 15137.9 | 25.549 | job 7128549, 3 warm-up, 1x64 timed |
| CROM-style k=32 (C32-w512, 600k steps; collocation LM, M=2048, 4 it/step, refined z0; float32; accurate-float32) | 58.217 | 1.916 | 12048.2 | 12020.6–12073.0 | 12001.0 | 26.600 | job 7128549, 3 warm-up, 1x64 timed |
| CoLoRA-style k=16 (L16-w512, 600k steps; hyper-network; float64; accurate) | 0.240 | 0.104 | 49.9 | 49.6–50.2 | 0.1 | 0.243 | job 7128551, 20 warm-up, 5x64 timed |
| CoLoRA-style k=16 (L16-w512, 600k steps; hyper-network; float32; accurate-float32) | 0.240 | 0.104 | 49.0 | 48.7–49.3 | 0.1 | 0.243 | job 7128551, 20 warm-up, 5x64 timed |
| CoLoRA-style k=32 (L32-w512, 600k steps; hyper-network; float64; accurate) | 0.214 | 0.106 | 49.5 | 49.3–50.0 | 0.1 | 0.283 | job 7128551, 20 warm-up, 5x64 timed |
| CoLoRA-style k=32 (L32-w512, 600k steps; hyper-network; float32; accurate-float32) | 0.214 | 0.106 | 48.7 | 48.5–49.1 | 0.1 | 0.283 | job 7128551, 20 warm-up, 5x64 timed |
| CoLoRA-style decoder, online residual solve k=16 (L16-w256, 600k steps; collocation LM, M=512, 2 it/step; float64; accurate) | 31.367 | 1.152 | 305.4 | 303.9–306.6 | 288.0 | 17.073 | job 7128551, 20 warm-up, 5x64 timed |
| CoLoRA-style decoder, online residual solve k=16 (L16-w256, 600k steps; collocation LM, M=512, 2 it/step; float32; accurate-float32) | 31.375 | 1.151 | 305.9 | 304.5–307.2 | 288.9 | 17.075 | job 7128551, 20 warm-up, 5x64 timed |
| paper: bank + span, R'=384 (accurate) | 0.875 | 0.026 | 45.2 | | | | job 4319328 (stored) |
| paper: bank + span, R'=128 | 2.547 | 0.268 | 18.3 | | | | job 4319328 (stored) |
| paper: bank + span, R'=96 (fast) | 3.846 | 0.529 | 14.2 | | | | job 4319328 (stored) |
| paper: bank + nonlinear head, k=16 | 4.097 | 0.672 | 32.4 | | | | job 4327396 (stored) |
| full-order solver (fastest at <= 0.875 %) | 0.109 | 0.030 | 17.4 | | | | job 4319328 (stored) |

| method | cases above 0.875 % | above 3.85 % | above 10 % | 90th percentile % | case index of the worst |
|---|---|---|---|---|---|
| CROM-style k=16 (C16-w256-lr3, 600k steps; collocation LM, M=2048, 4 it/step; float64; accurate) | 11 / 64 | 7 / 64 | 2 / 64 | 2.459 | 1 |
| CROM-style k=16 (C16-w256-lr3, 600k steps; collocation LM, M=2048, 4 it/step; float32; accurate-float32) | 11 / 64 | 7 / 64 | 2 / 64 | 2.465 | 1 |
| CROM-style k=16 (C16-w128, 120k steps; collocation LM, M=2048, 2 it/step; float64; fast) | 42 / 64 | 7 / 64 | 2 / 64 | 3.437 | 36 |
| CROM-style k=16 (C16-w128, 120k steps; collocation LM, M=2048, 2 it/step; float32; fast-float32) | 42 / 64 | 7 / 64 | 2 / 64 | 3.436 | 36 |
| CROM-style k=32 (C32-w512, 600k steps; collocation LM, M=2048, 4 it/step, refined z0; float64; accurate) | 46 / 64 | 26 / 64 | 20 / 64 | 29.550 | 60 |
| CROM-style k=32 (C32-w512, 600k steps; collocation LM, M=2048, 4 it/step, refined z0; float32; accurate-float32) | 46 / 64 | 26 / 64 | 20 / 64 | 29.589 | 60 |
| CoLoRA-style k=16 (L16-w512, 600k steps; hyper-network; float64; accurate) | 0 / 64 | 0 / 64 | 0 / 64 | 0.145 | 27 |
| CoLoRA-style k=16 (L16-w512, 600k steps; hyper-network; float32; accurate-float32) | 0 / 64 | 0 / 64 | 0 / 64 | 0.145 | 27 |
| CoLoRA-style k=32 (L32-w512, 600k steps; hyper-network; float64; accurate) | 0 / 64 | 0 / 64 | 0 / 64 | 0.153 | 36 |
| CoLoRA-style k=32 (L32-w512, 600k steps; hyper-network; float32; accurate-float32) | 0 / 64 | 0 / 64 | 0 / 64 | 0.153 | 36 |
| CoLoRA-style decoder, online residual solve k=16 (L16-w256, 600k steps; collocation LM, M=512, 2 it/step; float64; accurate) | 39 / 64 | 21 / 64 | 8 / 64 | 10.321 | 36 |
| CoLoRA-style decoder, online residual solve k=16 (L16-w256, 600k steps; collocation LM, M=512, 2 it/step; float32; accurate-float32) | 39 / 64 | 20 / 64 | 7 / 64 | 9.844 | 36 |

| method | beats accurate setting (R'=384) | beats fast setting (R'=96) | beats head k=16 | error vs accurate | time vs accurate | time vs fast |
|---|---|---|---|---|---|---|
| CROM-style k=16 (C16-w256-lr3, 600k steps; collocation LM, M=2048, 4 it/step; float64; accurate) | no | no | no | ×25.64 | ×62.28 | ×197.80 |
| CROM-style k=16 (C16-w256-lr3, 600k steps; collocation LM, M=2048, 4 it/step; float32; accurate-float32) | no | no | no | ×25.64 | ×45.43 | ×144.28 |
| CROM-style k=16 (C16-w128, 120k steps; collocation LM, M=2048, 2 it/step; float64; fast) | no | no | no | ×19.31 | ×11.21 | ×35.59 |
| CROM-style k=16 (C16-w128, 120k steps; collocation LM, M=2048, 2 it/step; float32; fast-float32) | no | no | no | ×19.31 | ×6.77 | ×21.51 |
| CROM-style k=32 (C32-w512, 600k steps; collocation LM, M=2048, 4 it/step, refined z0; float64; accurate) | no | no | no | ×66.57 | ×335.92 | ×1066.82 |
| CROM-style k=32 (C32-w512, 600k steps; collocation LM, M=2048, 4 it/step, refined z0; float32; accurate-float32) | no | no | no | ×66.54 | ×266.33 | ×845.82 |
| CoLoRA-style k=16 (L16-w512, 600k steps; hyper-network; float64; accurate) | no | no | no | ×0.27 | ×1.10 | ×3.50 |
| CoLoRA-style k=16 (L16-w512, 600k steps; hyper-network; float32; accurate-float32) | no | no | no | ×0.27 | ×1.08 | ×3.44 |
| CoLoRA-style k=32 (L32-w512, 600k steps; hyper-network; float64; accurate) | no | no | no | ×0.24 | ×1.10 | ×3.48 |
| CoLoRA-style k=32 (L32-w512, 600k steps; hyper-network; float32; accurate-float32) | no | no | no | ×0.24 | ×1.08 | ×3.42 |
| CoLoRA-style decoder, online residual solve k=16 (L16-w256, 600k steps; collocation LM, M=512, 2 it/step; float64; accurate) | no | no | no | ×35.85 | ×6.75 | ×21.44 |
| CoLoRA-style decoder, online residual solve k=16 (L16-w256, 600k steps; collocation LM, M=512, 2 it/step; float32; accurate-float32) | no | no | no | ×35.86 | ×6.76 | ×21.47 |

### Stored rows, same cohort and mesh (job 4327396, NVIDIA A100 80GB PCIe)

| method | worst % | median % | ms |
|---|---|---|---|
| fno-large | 8.761 | 2.588 | 7.4 |
| unet-refine | 5.018 | 2.008 | 6.0 |
| tsol-small | 5.269 | 2.237 | 11.5 |
| don-small | 59.051 | 12.241 | 3.9 |

### Offline training (validation = 64 held-out training-pool trajectories, reconstruction (C) / prediction (L) error)

| config | steps | arm | k | parameters | lr | val mean % | val evolved worst % | train time, min | steps/s (last) | plateau | GPU | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C16-w128 | 120,000 | C | 16 | 1,972,241 | 0.001 | 0.591 | 1.944 | 3.2 | 701 | no (3.2 % over last tenth) | NVIDIA H200 | 7121932 |
| C16-w256 | 120,000 | C | 16 | 2,220,945 | 0.001 | 0.452 | 1.327 | 5.7 | 370 | no (5.0 % over last tenth) | NVIDIA H200 | 7121932 |
| C16-w256-lr3 | 120,000 | C | 16 | 2,220,945 | 0.0003 | 0.570 | 1.496 | 5.4 | 370 | no (4.6 % over last tenth) | NVIDIA H200 | 7121932 |
| C16-w512 | 120,000 | C | 16 | 3,078,545 | 0.001 | 0.378 | 1.272 | 13.3 | 155 | no (4.4 % over last tenth) | NVIDIA H200 | 7121932 |
| C16-w128 | 600,000 | C | 16 | 1,972,241 | 0.001 | 0.457 | 2.744 | 14.6 | 701 | no (2.5 % over last tenth) | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | C | 16 | 2,220,945 | 0.001 | 0.353 | 1.240 | 27.4 | 369 | no (3.5 % over last tenth) | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | C | 16 | 2,220,945 | 0.0003 | 0.251 | 0.833 | 27.5 | 368 | no (3.8 % over last tenth) | NVIDIA H200 | 7122345 |
| C16-w512 | 600,000 | C | 16 | 3,078,545 | 0.001 | 0.331 | 1.441 | 64.8 | 155 | no (2.7 % over last tenth) | NVIDIA H200 | 7122345 |
| C32-w128 | 600,000 | C | 32 | 1,982,497 | 0.001 | 0.382 | 1.261 | 14.6 | 699 | no (2.1 % over last tenth) | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | C | 32 | 2,233,249 | 0.001 | 0.338 | 1.650 | 22.6 | 455 | no (3.4 % over last tenth) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 7122582 |
| C32-w512 | 600,000 | C | 32 | 3,094,945 | 0.001 | 0.289 | 0.854 | 65.1 | 155 | no (3.0 % over last tenth) | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | L | 16 | 475,153 | 0.001 | 0.121 | 0.307 | 27.7 | 365 | no (6.4 % over last tenth) | NVIDIA H200 | 7122346 |
| L16-w256-lr3 | 600,000 | L | 16 | 475,153 | 0.0003 | 0.135 | 0.315 | 27.4 | 368 | no (12.8 % over last tenth) | NVIDIA H200 | 7122346 |
| L16-w512 | 600,000 | L | 16 | 1,736,465 | 0.001 | 0.120 | 0.341 | 60.7 | 165 | no (5.1 % over last tenth) | NVIDIA H200 | 7122346 |
| L32-w256 | 600,000 | L | 32 | 487,457 | 0.001 | 0.123 | 0.310 | 27.2 | 371 | no (7.6 % over last tenth) | NVIDIA H200 | 7122346 |
| L32-w512 | 600,000 | L | 32 | 1,761,057 | 0.001 | 0.114 | 0.371 | 60.9 | 165 | no (2.9 % over last tenth) | NVIDIA H200 | 7122346 |

Total training time of all runs listed: 7.80 GPU-hours.

### Validation knob grid, solver v2 (used for selection)

First 32 validation trajectories; times are rough batch-1 timings on the GPU named in the row, used for selection only.

| decoder | steps | kind | M | LM it/step | refined z0 | precision | worst % | median % | rough ms | non-finite | GPU | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C16-w128 | 120,000 | residual | 2048 | 2 | no | float64 | 3.136 | 1.225 | 246.7 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 2048 | 2 | yes | float64 | 3.347 | 1.107 | 248.8 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 2048 | 1 | no | float64 | 7.521 | 1.496 | 132.2 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 2048 | 4 | no | float64 | 2.773 | 1.241 | 475.2 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 512 | 1 | no | float64 | 10.714 | 1.690 | 41.1 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 512 | 2 | no | float64 | 5.565 | 1.492 | 73.2 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 512 | 4 | no | float64 | 5.584 | 1.496 | 137.2 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 8192 | 1 | no | float64 | 6.625 | 1.441 | 343.6 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 8192 | 2 | no | float64 | 4.869 | 1.092 | 643.2 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 8192 | 4 | no | float64 | 2.705 | 1.056 | 1238.7 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 8192 | 4 | no | float32 | 2.705 | 1.056 | 1024.2 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 120,000 | residual | 2048 | 2 | no | float32 | 3.136 | 1.225 | 146.1 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 2048 | 2 | no | float64 | 3.314 | 1.262 | 643.6 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 2048 | 2 | yes | float64 | 3.279 | 1.198 | 652.9 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 2048 | 1 | yes | float64 | 5.013 | 1.239 | 348.3 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 2048 | 4 | yes | float64 | 3.308 | 1.186 | 1257.9 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 512 | 1 | yes | float64 | 5.889 | 2.188 | 93.6 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 512 | 2 | yes | float64 | 6.872 | 2.105 | 168.3 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 512 | 4 | yes | float64 | 3.813 | 2.050 | 317.8 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 8192 | 1 | yes | float64 | 5.435 | 1.186 | 981.9 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 8192 | 2 | yes | float64 | 3.566 | 1.045 | 1829.5 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 8192 | 4 | yes | float64 | 3.471 | 1.027 | 3488.9 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 2048 | 2 | yes | float32 | 3.278 | 1.199 | 460.8 | 0 | NVIDIA H200 | 7123239 |
| C16-w256-lr3 | 120,000 | residual | 512 | 4 | yes | float32 | 3.952 | 2.059 | 253.1 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 2048 | 2 | no | float64 | 7.112 | 0.876 | 639.6 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 2048 | 2 | yes | float64 | 7.528 | 0.854 | 645.8 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 2048 | 1 | no | float64 | 15.319 | 1.388 | 340.4 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 2048 | 4 | no | float64 | 5.404 | 0.765 | 1244.3 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 512 | 1 | no | float64 | 25.045 | 1.331 | 92.0 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 512 | 2 | no | float64 | 24.168 | 1.315 | 166.4 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 512 | 4 | no | float64 | 24.129 | 1.293 | 315.5 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 8192 | 1 | no | float64 | 11.381 | 1.422 | 962.4 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 8192 | 2 | no | float64 | 6.174 | 0.936 | 1805.1 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 8192 | 4 | no | float64 | 5.898 | 0.825 | 3488.9 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 120,000 | residual | 2048 | 4 | no | float32 | 5.404 | 0.765 | 881.4 | 0 | NVIDIA H200 | 7123239 |
| C16-w512 | 120,000 | residual | 2048 | 2 | no | float64 | 11.385 | 1.855 | 1338.6 | 0 | NVIDIA H200 | 7123239 |
| C16-w512 | 120,000 | residual | 2048 | 2 | yes | float64 | 12.210 | 1.523 | 1373.1 | 0 | NVIDIA H200 | 7123239 |
| C16-w512 | 120,000 | residual | 2048 | 1 | no | float64 | 16.525 | 2.266 | 725.6 | 0 | NVIDIA H200 | 7123239 |
| C16-w512 | 120,000 | residual | 2048 | 4 | no | float64 | 6.699 | 0.918 | 2590.5 | 0 | NVIDIA H200 | 7123239 |
| C16-w512 | 120,000 | residual | 512 | 1 | no | float64 | 19.857 | 2.627 | 235.9 | 0 | NVIDIA H200 | 7123239 |
| C16-w512 | 120,000 | residual | 512 | 2 | no | float64 | 13.884 | 2.018 | 436.5 | 0 | NVIDIA H200 | 7123239 |
| C16-w512 | 120,000 | residual | 512 | 4 | no | float64 | 16.393 | 1.546 | 831.1 | 0 | NVIDIA H200 | 7123239 |
| C16-w512 | 120,000 | residual | 2048 | 4 | no | float32 | 6.699 | 0.918 | 2445.3 | 0 | NVIDIA H200 | 7123239 |
| C16-w128 | 600,000 | residual | 2048 | 2 | no | float64 | 26.259 | 1.461 | 246.3 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 2048 | 2 | yes | float64 | 25.206 | 1.961 | 248.6 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 2048 | 1 | yes | float64 | 26.280 | 3.029 | 134.5 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 2048 | 4 | yes | float64 | 25.151 | 0.860 | 476.6 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 512 | 1 | yes | float64 | 22.300 | 2.178 | 42.2 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 512 | 2 | yes | float64 | 19.165 | 1.489 | 74.4 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 512 | 4 | yes | float64 | 17.714 | 1.346 | 138.7 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 8192 | 1 | yes | float64 | 32.168 | 2.274 | 348.7 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 8192 | 2 | yes | float64 | 28.439 | 1.413 | 648.0 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 8192 | 4 | yes | float64 | 25.905 | 0.564 | 1239.1 | 0 | NVIDIA H200 | 7122345 |
| C16-w128 | 600,000 | residual | 512 | 4 | yes | float32 | 17.733 | 1.592 | 100.2 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 2048 | 2 | no | float64 | 5.859 | 0.697 | 638.7 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 2048 | 2 | yes | float64 | 6.378 | 0.585 | 645.8 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 2048 | 1 | no | float64 | 11.764 | 1.128 | 341.2 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 2048 | 4 | no | float64 | 2.047 | 0.356 | 1238.3 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 512 | 1 | no | float64 | 11.638 | 0.974 | 92.2 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 512 | 2 | no | float64 | 11.921 | 0.687 | 167.1 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 512 | 4 | no | float64 | 11.933 | 0.588 | 316.7 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 8192 | 1 | no | float64 | 22.382 | 1.189 | 962.2 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 8192 | 2 | no | float64 | 6.049 | 0.525 | 1795.4 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 8192 | 4 | no | float64 | 5.576 | 0.376 | 3467.8 | 0 | NVIDIA H200 | 7122345 |
| C16-w256-lr3 | 600,000 | residual | 2048 | 4 | no | float32 | 2.048 | 0.356 | 880.8 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 2048 | 2 | no | float64 | 32.641 | 2.498 | 636.6 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 2048 | 2 | yes | float64 | 43.182 | 2.485 | 645.9 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 2048 | 1 | no | float64 | 43.831 | 4.723 | 339.7 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 2048 | 4 | no | float64 | 32.597 | 1.123 | 1240.3 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 512 | 1 | no | float64 | 20.543 | 4.331 | 92.0 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 512 | 2 | no | float64 | 18.440 | 2.870 | 165.5 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 512 | 4 | no | float64 | 15.690 | 2.403 | 313.0 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 8192 | 1 | no | float64 | 40.214 | 3.444 | 960.6 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 8192 | 2 | no | float64 | 41.653 | 2.081 | 1798.9 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 8192 | 4 | no | float64 | 42.421 | 0.982 | 3476.3 | 0 | NVIDIA H200 | 7123239 |
| C16-w256 | 600,000 | residual | 512 | 4 | no | float32 | 15.675 | 2.407 | 254.0 | 0 | NVIDIA H200 | 7123239 |
| C16-w512 | 600,000 | residual | 2048 | 2 | no | float64 | 27.110 | 1.304 | 1322.6 | 0 | NVIDIA H200 | 7122345 |
| C16-w512 | 600,000 | residual | 2048 | 2 | yes | float64 | 19.177 | 2.506 | 1332.7 | 0 | NVIDIA H200 | 7122345 |
| C16-w512 | 600,000 | residual | 2048 | 1 | yes | float64 | 24.854 | 5.918 | 718.1 | 0 | NVIDIA H200 | 7122345 |
| C16-w512 | 600,000 | residual | 2048 | 4 | yes | float64 | 16.620 | 0.994 | 2562.1 | 0 | NVIDIA H200 | 7122345 |
| C16-w512 | 600,000 | residual | 512 | 1 | yes | float64 | 57.517 | 4.974 | 237.8 | 0 | NVIDIA H200 | 7122345 |
| C16-w512 | 600,000 | residual | 512 | 2 | yes | float64 | 51.285 | 4.337 | 433.9 | 0 | NVIDIA H200 | 7122345 |
| C16-w512 | 600,000 | residual | 512 | 4 | yes | float64 | 30.361 | 2.427 | 819.2 | 0 | NVIDIA H200 | 7122345 |
| C16-w512 | 600,000 | residual | 2048 | 4 | yes | float32 | 16.604 | 0.994 | 2458.2 | 0 | NVIDIA H200 | 7122345 |
| C32-w128 | 600,000 | residual | 2048 | 2 | no | float64 | 29.510 | 4.568 | 465.3 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 2048 | 2 | yes | float64 | 30.438 | 3.988 | 468.6 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 2048 | 1 | no | float64 | 31.357 | 5.708 | 241.4 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 2048 | 4 | no | float64 | 28.662 | 1.907 | 912.0 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 512 | 1 | no | float64 | 34.868 | 6.581 | 66.1 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 512 | 2 | no | float64 | 41.379 | 2.728 | 123.1 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 512 | 4 | no | float64 | 36.006 | 1.950 | 237.3 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 8192 | 1 | no | float64 | 37.954 | 7.478 | 732.3 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 8192 | 2 | no | float64 | 33.485 | 3.945 | 1424.9 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 8192 | 4 | no | float64 | 37.773 | 1.454 | 2792.6 | 0 | NVIDIA H200 | 7123239 |
| C32-w128 | 600,000 | residual | 2048 | 4 | no | float32 | 28.665 | 1.907 | 474.2 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 2048 | 2 | no | float64 | 40.801 | 4.721 | 1174.2 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 2048 | 2 | yes | float64 | 34.176 | 3.743 | 1188.7 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 2048 | 1 | yes | float64 | 36.317 | 8.573 | 620.7 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 2048 | 4 | yes | float64 | 33.132 | 1.874 | 2330.3 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 512 | 1 | yes | float64 | 39.207 | 6.392 | 154.8 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 512 | 2 | yes | float64 | 37.241 | 5.758 | 291.5 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 512 | 4 | yes | float64 | 43.132 | 3.707 | 563.1 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 8192 | 1 | yes | float64 | 42.772 | 9.942 | 2070.4 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 8192 | 2 | yes | float64 | 39.348 | 4.161 | 4016.1 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 8192 | 4 | yes | float64 | 34.148 | 2.330 | 7835.5 | 0 | NVIDIA H200 | 7123239 |
| C32-w256 | 600,000 | residual | 2048 | 4 | yes | float32 | 33.126 | 1.872 | 1656.4 | 0 | NVIDIA H200 | 7123239 |
| C32-w512 | 600,000 | residual | 2048 | 2 | no | float64 | 39.120 | 7.771 | 2626.1 | 0 | NVIDIA H200 | 7123239 |
| C32-w512 | 600,000 | residual | 2048 | 2 | yes | float64 | 26.908 | 7.760 | 2655.1 | 0 | NVIDIA H200 | 7123239 |
| C32-w512 | 600,000 | residual | 2048 | 1 | yes | float64 | 36.431 | 13.641 | 1388.6 | 0 | NVIDIA H200 | 7123239 |
| C32-w512 | 600,000 | residual | 2048 | 4 | yes | float64 | 25.549 | 3.655 | 5186.1 | 0 | NVIDIA H200 | 7123239 |
| C32-w512 | 600,000 | residual | 512 | 1 | yes | float64 | 30.062 | 8.110 | 415.3 | 0 | NVIDIA H200 | 7123239 |
| C32-w512 | 600,000 | residual | 512 | 2 | yes | float64 | 27.562 | 6.436 | 785.0 | 0 | NVIDIA H200 | 7123239 |
| C32-w512 | 600,000 | residual | 512 | 4 | yes | float64 | 27.418 | 4.514 | 1523.2 | 0 | NVIDIA H200 | 7123239 |
| C32-w512 | 600,000 | residual | 2048 | 4 | yes | float32 | 26.600 | 3.623 | 4629.8 | 0 | NVIDIA H200 | 7123239 |
| L16-w256-lr3 | 600,000 | hyper | — | — | no | float64 | 0.302 | 0.125 | 7.2 | 0 | NVIDIA H200 | 7122346 |
| L16-w256-lr3 | 600,000 | hyper | — | — | no | float32 | 0.302 | 0.125 | 7.1 | 0 | NVIDIA H200 | 7122346 |
| L16-w256 | 600,000 | hyper | — | — | no | float64 | 0.307 | 0.110 | 7.2 | 0 | NVIDIA H200 | 7122346 |
| L16-w256 | 600,000 | hyper | — | — | no | float32 | 0.307 | 0.110 | 7.1 | 0 | NVIDIA H200 | 7122346 |
| L16-w512 | 575,000 | hyper | — | — | no | float64 | 0.243 | 0.120 | 18.1 | 0 | NVIDIA H200 | 7122346 |
| L16-w512 | 575,000 | hyper | — | — | no | float32 | 0.243 | 0.120 | 18.0 | 0 | NVIDIA H200 | 7122346 |
| L32-w256 | 600,000 | hyper | — | — | no | float64 | 0.310 | 0.105 | 7.2 | 0 | NVIDIA H200 | 7122346 |
| L32-w256 | 600,000 | hyper | — | — | no | float32 | 0.310 | 0.105 | 7.2 | 0 | NVIDIA H200 | 7122346 |
| L32-w512 | 600,000 | hyper | — | — | no | float64 | 0.283 | 0.113 | 18.1 | 0 | NVIDIA H200 | 7122346 |
| L32-w512 | 600,000 | hyper | — | — | no | float32 | 0.283 | 0.113 | 17.9 | 0 | NVIDIA H200 | 7122346 |

### Validation knob grid, arm L with the online residual solve instead of the hyper-network (solver v2; not used for selection of hyper rows)

First 32 validation trajectories; times are rough batch-1 timings on the GPU named in the row, used for selection only.

| decoder | steps | kind | M | LM it/step | refined z0 | precision | worst % | median % | rough ms | non-finite | GPU | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L16-w256 | 600,000 | residual | 2048 | 2 | no | float64 | 20.153 | 0.851 | 482.9 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 2048 | 2 | yes | float64 | 20.746 | 0.924 | 486.2 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 2048 | 1 | no | float64 | 41.340 | 2.475 | 262.9 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 2048 | 4 | no | float64 | 20.912 | 0.536 | 921.2 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 512 | 1 | no | float64 | 22.595 | 2.415 | 79.9 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 512 | 2 | no | float64 | 17.073 | 0.949 | 140.5 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 512 | 4 | no | float64 | 17.082 | 0.726 | 261.8 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 8192 | 1 | no | float64 | 52.849 | 2.735 | 1000.6 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 8192 | 2 | no | float64 | 21.679 | 1.078 | 1868.6 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 8192 | 4 | no | float64 | 23.802 | 0.487 | 3605.3 | 0 | NVIDIA H200 | 7123239 |
| L16-w256 | 600,000 | residual | 512 | 2 | no | float32 | 17.075 | 0.947 | 142.5 | 0 | NVIDIA H200 | 7123239 |

### Validation knob grid, solver v1 (superseded by amendment A3; kept for the record)

First 32 validation trajectories; times are rough batch-1 timings on the GPU named in the row, used for selection only.

| decoder | steps | kind | M | LM it/step | refined z0 | precision | worst % | median % | rough ms | non-finite | GPU | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C16-w128 | 120,000 | residual | 2048 | 2 | no | float64 | 4.094 | 1.222 | 233.5 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 2048 | 2 | yes | float64 | 4.094 | 1.098 | 236.0 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 2048 | 1 | no | float64 | 8.561 | 1.422 | 119.5 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 2048 | 4 | no | float64 | 2.777 | 1.241 | 460.6 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 512 | 1 | no | float64 | 81.940 | 1.918 | 36.9 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 512 | 2 | no | float64 | 81.940 | 1.492 | 69.1 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 512 | 4 | no | float64 | 5.584 | 1.496 | 133.4 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 8192 | 1 | no | float64 | 7.860 | 1.466 | 300.4 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 8192 | 2 | no | float64 | 4.175 | 1.088 | 594.6 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 8192 | 4 | no | float64 | 2.705 | 1.056 | 1184.0 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 8192 | 4 | no | float32 | 2.705 | 1.056 | 995.3 | 0 | NVIDIA H200 | 7121932 |
| C16-w128 | 120,000 | residual | 2048 | 4 | no | float32 | 2.779 | 1.241 | 270.2 | 0 | NVIDIA H200 | 7121932 |
| C16-w256 | 600,000 | residual | 2048 | 2 | no | float64 | 75.325 | 3.089 | 605.9 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 2048 | 2 | yes | float64 | 75.166 | 3.067 | 612.4 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 2048 | 1 | yes | float64 | 75.166 | 5.604 | 314.3 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 2048 | 4 | yes | float64 | 75.166 | 1.140 | 1209.8 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 512 | 1 | yes | float64 | 84.487 | 4.350 | 85.5 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 512 | 2 | yes | float64 | 84.487 | 3.967 | 160.1 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 512 | 4 | yes | float64 | 75.164 | 3.308 | 309.8 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 8192 | 1 | yes | float64 | 75.171 | 6.256 | 875.6 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 8192 | 2 | yes | float64 | 75.171 | 2.643 | 1713.9 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 8192 | 4 | yes | float64 | 75.171 | 0.907 | 3395.5 | 0 | NVIDIA H200 | 7122345 |
| C16-w256 | 600,000 | residual | 512 | 4 | yes | float32 | 75.164 | 3.375 | 247.8 | 0 | NVIDIA H200 | 7122345 |

