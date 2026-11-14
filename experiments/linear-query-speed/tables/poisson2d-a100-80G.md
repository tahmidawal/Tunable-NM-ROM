# Poisson 2D query time, GPU type `a100-80G` (job results `results`)

Frozen from development cases only: F = `T_tau0.0001_f64_full`, S = `T_tau0.0001_f64_sub256` (selection meshes [256, 2048], timing mesh 2048).

## Headline: full-field output, full-source input (configuration F), test cohort of 32 cases

Times are the median GPU-query in ms (device float64 nodal source -> device float64 nodal field of the whole mesh, synced).  S (source read on a 256-interval sub-lattice) and solve-only (reduced coefficients, no field) are separate measurements and are not the headline.

| mesh | arm | worst err, paper path | worst err, F | paper stored (A100) | paper path, this job | F, this job | cuDSS solve (quoted) | AmgX solve (quoted) | F vs paper path | F vs cuDSS | S (sampled input) | solve only (no field) | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256^2 | accurate | 0.008379 | 0.008379 | 0.565 | 1.042 | **0.152** | 0.623 | 10.700 | 6.9x | 4.1x | - | 0.114 | 7220235 |
| 256^2 | fast | 0.040668 | 0.040668 | 0.381 | 0.791 | **0.136** | 0.623 | 10.700 | 5.8x | 4.6x | - | 0.111 | 7220235 |
| 512^2 | accurate | 0.008356 | 0.008356 | - | 1.615 | **0.171** | 1.520 | 19.000 | 9.5x | 8.9x | 0.160 | 0.131 | 7220235 |
| 512^2 | fast | 0.040594 | 0.040594 | - | 1.070 | **0.149** | 1.520 | 19.000 | 7.2x | 10.2x | 0.141 | 0.125 | 7220235 |
| 1024^2 | accurate | 0.008350 | 0.008350 | 3.103 | 3.703 | **0.210** | 4.960 | 34.600 | 17.6x | 23.6x | 0.181 | 0.152 | 7220235 |
| 1024^2 | fast | 0.040575 | 0.040575 | 1.177 | 1.633 | **0.170** | 4.960 | 34.600 | 9.6x | 29.2x | 0.154 | 0.134 | 7220235 |
| 2048^2 | accurate | 0.008348 | 0.008348 | 11.663 | 11.927 | **0.346** | 16.500 | 96.600 | 34.5x | 47.7x | 0.237 | 0.249 | 7220235 |
| 2048^2 | fast | 0.040571 | 0.040571 | 3.405 | 3.673 | **0.295** | 16.500 | 96.600 | 12.5x | 56.0x | 0.203 | 0.221 | 7220235 |
| 4096^2 | accurate | 0.008348 | 0.008348 | 41.916 | 44.348 | **0.778** | 66.800 | 333.0 | 57.0x | 85.9x | 0.429 | 0.541 | 7221131 |
| 4096^2 | fast | 0.040570 | 0.040570 | 12.110 | 12.955 | **0.667** | 66.800 | 333.0 | 19.4x | 100.1x | 0.364 | 0.471 | 7221131 |

## Query time and error (test cohort, 32 cases; GPU-query = device f64 source -> device f64 full field)

| mesh | arm | path | worst err | mean err | max field dev. from paper path | ms (median) | paper A1/A2 ms | cuDSS ms | AmgX ms | ours / cuDSS |
|---|---|---|---|---|---|---|---|---|---|---|
| 256^2 | accurate | paper path | 0.008379 | 0.001982 | 0 | 1.042 | 1.061/1.040 | 0.623 | 10.700 | 1.67 |
| 256^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008379 | 0.001982 | 1.4e-06 | 0.152 | - | 0.623 | 10.700 | 0.24 |
| 256^2 | accurate | T_tau1e-08_f64_full (p=110, q=108) | 0.008379 | 0.001982 | 2.4e-10 | 0.174 | - | 0.623 | 10.700 | 0.28 |
| 256^2 | accurate | T_tau1e-08_f32_full (p=110, q=108) | 0.008379 | 0.001982 | 3.0e-07 | 0.162 | - | 0.623 | 10.700 | 0.26 |
| 256^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.114 | - | 0.623 | 10.700 | not comparable |
| 256^2 | fast | paper path | 0.040668 | 0.008042 | 0 | 0.791 | 0.815/0.782 | 0.623 | 10.700 | 1.27 |
| 256^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040668 | 0.008042 | 8.8e-06 | 0.136 | - | 0.623 | 10.700 | 0.22 |
| 256^2 | fast | T_tau1e-08_f64_full (p=94, q=95) | 0.040668 | 0.008042 | 1.1e-09 | 0.142 | - | 0.623 | 10.700 | 0.23 |
| 256^2 | fast | T_tau1e-08_f32_full (p=94, q=95) | 0.040668 | 0.008042 | 3.2e-07 | 0.141 | - | 0.623 | 10.700 | 0.23 |
| 256^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.111 | - | 0.623 | 10.700 | not comparable |
| 512^2 | accurate | paper path | 0.008356 | 0.001977 | 0 | 1.615 | 1.683/1.515 | 1.520 | 19.000 | 1.06 |
| 512^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008356 | 0.001977 | 1.4e-06 | 0.171 | - | 1.520 | 19.000 | 0.11 |
| 512^2 | accurate | T_tau0.0001_f64_sub256 [S] (p=65, q=66) | 0.008356 | 0.001977 | 4.2e-06 | 0.160 | - | 1.520 | 19.000 | 0.11 |
| 512^2 | accurate | T_tau1e-08_f64_full (p=110, q=109) | 0.008356 | 0.001977 | 2.4e-10 | 0.191 | - | 1.520 | 19.000 | 0.13 |
| 512^2 | accurate | T_tau1e-08_f32_full (p=110, q=109) | 0.008356 | 0.001977 | 2.8e-07 | 0.186 | - | 1.520 | 19.000 | 0.12 |
| 512^2 | accurate | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.125 | - | 1.520 | 19.000 | not comparable |
| 512^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.131 | - | 1.520 | 19.000 | not comparable |
| 512^2 | fast | paper path | 0.040594 | 0.008027 | 0 | 1.070 | 1.115/1.018 | 1.520 | 19.000 | 0.70 |
| 512^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040594 | 0.008027 | 8.8e-06 | 0.149 | - | 1.520 | 19.000 | 0.10 |
| 512^2 | fast | T_tau0.0001_f64_sub256 [S] (p=50, q=51) | 0.040594 | 0.008027 | 8.8e-06 | 0.141 | - | 1.520 | 19.000 | 0.09 |
| 512^2 | fast | T_tau1e-08_f64_full (p=95, q=95) | 0.040594 | 0.008027 | 1.1e-09 | 0.154 | - | 1.520 | 19.000 | 0.10 |
| 512^2 | fast | T_tau1e-08_f32_full (p=95, q=95) | 0.040594 | 0.008027 | 3.3e-07 | 0.157 | - | 1.520 | 19.000 | 0.10 |
| 512^2 | fast | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.112 | - | 1.520 | 19.000 | not comparable |
| 512^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.125 | - | 1.520 | 19.000 | not comparable |
| 1024^2 | accurate | paper path | 0.008350 | 0.001976 | 0 | 3.703 | 3.657/3.722 | 4.960 | 34.600 | 0.75 |
| 1024^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008350 | 0.001976 | 1.4e-06 | 0.210 | - | 4.960 | 34.600 | 0.04 |
| 1024^2 | accurate | T_tau0.0001_f64_sub256 [S] (p=65, q=66) | 0.008350 | 0.001976 | 5.2e-06 | 0.181 | - | 4.960 | 34.600 | 0.04 |
| 1024^2 | accurate | T_tau1e-08_f64_full (p=110, q=109) | 0.008350 | 0.001976 | 2.3e-10 | 0.237 | - | 4.960 | 34.600 | 0.05 |
| 1024^2 | accurate | T_tau1e-08_f32_full (p=110, q=109) | 0.008350 | 0.001976 | 3.1e-07 | 0.227 | - | 4.960 | 34.600 | 0.05 |
| 1024^2 | accurate | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.123 | - | 4.960 | 34.600 | not comparable |
| 1024^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.152 | - | 4.960 | 34.600 | not comparable |
| 1024^2 | fast | paper path | 0.040575 | 0.008023 | 0 | 1.633 | 1.567/1.676 | 4.960 | 34.600 | 0.33 |
| 1024^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040575 | 0.008023 | 8.8e-06 | 0.170 | - | 4.960 | 34.600 | 0.03 |
| 1024^2 | fast | T_tau0.0001_f64_sub256 [S] (p=50, q=51) | 0.040575 | 0.008023 | 8.8e-06 | 0.154 | - | 4.960 | 34.600 | 0.03 |
| 1024^2 | fast | T_tau1e-08_f64_full (p=95, q=95) | 0.040575 | 0.008023 | 1.1e-09 | 0.185 | - | 4.960 | 34.600 | 0.04 |
| 1024^2 | fast | T_tau1e-08_f32_full (p=95, q=95) | 0.040575 | 0.008023 | 2.8e-07 | 0.188 | - | 4.960 | 34.600 | 0.04 |
| 1024^2 | fast | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.114 | - | 4.960 | 34.600 | not comparable |
| 1024^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.134 | - | 4.960 | 34.600 | not comparable |
| 2048^2 | accurate | paper path | 0.008348 | 0.001976 | 0 | 11.927 | 11.929/11.913 | 16.500 | 96.600 | 0.72 |
| 2048^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008348 | 0.001976 | 1.4e-06 | 0.346 | - | 16.500 | 96.600 | 0.02 |
| 2048^2 | accurate | T_tau0.0001_f64_sub256 [S] (p=65, q=66) | 0.008348 | 0.001976 | 5.4e-06 | 0.237 | - | 16.500 | 96.600 | 0.01 |
| 2048^2 | accurate | T_tau1e-08_f64_full (p=110, q=108) | 0.008348 | 0.001976 | 2.6e-10 | 0.388 | - | 16.500 | 96.600 | 0.02 |
| 2048^2 | accurate | T_tau1e-08_f32_full (p=110, q=108) | 0.008348 | 0.001976 | 3.5e-07 | 0.387 | - | 16.500 | 96.600 | 0.02 |
| 2048^2 | accurate | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.136 | - | 16.500 | 96.600 | not comparable |
| 2048^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.249 | - | 16.500 | 96.600 | not comparable |
| 2048^2 | fast | paper path | 0.040571 | 0.008022 | 0 | 3.673 | 3.658/3.676 | 16.500 | 96.600 | 0.22 |
| 2048^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040571 | 0.008022 | 8.8e-06 | 0.295 | - | 16.500 | 96.600 | 0.02 |
| 2048^2 | fast | T_tau0.0001_f64_sub256 [S] (p=50, q=51) | 0.040571 | 0.008022 | 8.8e-06 | 0.203 | - | 16.500 | 96.600 | 0.01 |
| 2048^2 | fast | T_tau1e-08_f64_full (p=95, q=95) | 0.040571 | 0.008022 | 1.0e-09 | 0.319 | - | 16.500 | 96.600 | 0.02 |
| 2048^2 | fast | T_tau1e-08_f32_full (p=95, q=95) | 0.040571 | 0.008022 | 2.8e-07 | 0.340 | - | 16.500 | 96.600 | 0.02 |
| 2048^2 | fast | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.127 | - | 16.500 | 96.600 | not comparable |
| 2048^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.221 | - | 16.500 | 96.600 | not comparable |
| 4096^2 | accurate | paper path | 0.008348 | 0.001975 | 0 | 44.348 | 44.274/44.373 | 66.800 | 333.0 | 0.66 |
| 4096^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008348 | 0.001975 | 1.4e-06 | 0.778 | - | 66.800 | 333.0 | 0.01 |
| 4096^2 | accurate | T_tau0.0001_f64_sub256 [S] (p=65, q=66) | 0.008348 | 0.001975 | 5.5e-06 | 0.429 | - | 66.800 | 333.0 | 0.01 |
| 4096^2 | accurate | T_tau1e-08_f64_full (p=110, q=109) | 0.008348 | 0.001975 | 2.5e-10 | 0.873 | - | 66.800 | 333.0 | 0.01 |
| 4096^2 | accurate | T_tau1e-08_f32_full (p=110, q=109) | 0.008348 | 0.001975 | 3.4e-07 | 0.935 | - | 66.800 | 333.0 | 0.01 |
| 4096^2 | accurate | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.189 | - | 66.800 | 333.0 | not comparable |
| 4096^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.541 | - | 66.800 | 333.0 | not comparable |
| 4096^2 | fast | paper path | 0.040570 | 0.008022 | 0 | 12.955 | 12.939/12.964 | 66.800 | 333.0 | 0.19 |
| 4096^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040570 | 0.008022 | 8.8e-06 | 0.667 | - | 66.800 | 333.0 | 0.01 |
| 4096^2 | fast | T_tau0.0001_f64_sub256 [S] (p=50, q=51) | 0.040570 | 0.008022 | 8.8e-06 | 0.364 | - | 66.800 | 333.0 | 0.01 |
| 4096^2 | fast | T_tau1e-08_f64_full (p=95, q=95) | 0.040570 | 0.008022 | 9.8e-10 | 0.740 | - | 66.800 | 333.0 | 0.01 |
| 4096^2 | fast | T_tau1e-08_f32_full (p=95, q=95) | 0.040570 | 0.008022 | 3.5e-07 | 0.827 | - | 66.800 | 333.0 | 0.01 |
| 4096^2 | fast | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.173 | - | 66.800 | 333.0 | not comparable |
| 4096^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.471 | - | 66.800 | 333.0 | not comparable |

## Stage medians, ms (separately jitted and synced; they do not sum to the fused time)

| mesh | arm | path | projection + solve | core contraction | reconstruction / field products | host->device / device->host copy |
|---|---|---|---|---|---|---|
| 256^2 | accurate | paper path | 0.813 | - | 0.323 | - |
| 256^2 | accurate | T_tau0.0001_f64_full | 0.126 | 0.060 | 0.070 | 0.423/1.098 |
| 256^2 | accurate | T_tau1e-08_f64_full | 0.138 | 0.099 | 0.062 | 0.418/1.079 |
| 256^2 | accurate | T_tau1e-08_f32_full | 0.132 | 0.087 | 0.082 | 0.419/1.046 |
| 256^2 | fast | paper path | 0.797 | - | 0.145 | - |
| 256^2 | fast | T_tau0.0001_f64_full | 0.166 | 0.064 | 0.057 | 0.420/1.049 |
| 256^2 | fast | T_tau1e-08_f64_full | 0.128 | 0.058 | 0.054 | 0.421/1.002 |
| 256^2 | fast | T_tau1e-08_f32_full | 0.194 | 0.071 | 0.083 | 0.427/1.020 |
| 512^2 | accurate | paper path | 0.490 | - | 0.867 | - |
| 512^2 | accurate | T_tau0.0001_f64_full | 0.159 | 0.237 | 0.090 | 0.676/1.271 |
| 512^2 | accurate | T_tau0.0001_f64_sub256 | 0.125 | 0.079 | 0.080 | 0.670/1.297 |
| 512^2 | accurate | T_tau1e-08_f64_full | 0.142 | 0.108 | 0.097 | 0.676/1.288 |
| 512^2 | accurate | T_tau1e-08_f32_full | 0.136 | 0.094 | 0.108 | 0.668/1.278 |
| 512^2 | fast | paper path | 0.171 | - | 0.620 | - |
| 512^2 | fast | T_tau0.0001_f64_full | 0.131 | 0.064 | 0.058 | 0.670/1.243 |
| 512^2 | fast | T_tau0.0001_f64_sub256 | 0.133 | 0.064 | 0.056 | 0.670/1.218 |
| 512^2 | fast | T_tau1e-08_f64_full | 0.125 | 0.057 | 0.069 | 0.672/1.255 |
| 512^2 | fast | T_tau1e-08_f32_full | 0.141 | 0.062 | 0.084 | 0.672/1.235 |
| 1024^2 | accurate | paper path | 0.419 | - | 2.870 | - |
| 1024^2 | accurate | T_tau0.0001_f64_full | 0.197 | 0.091 | 0.099 | 1.730/2.139 |
| 1024^2 | accurate | T_tau0.0001_f64_sub256 | 0.216 | 0.094 | 0.101 | 1.743/2.195 |
| 1024^2 | accurate | T_tau1e-08_f64_full | 0.182 | 0.111 | 0.115 | 1.739/2.157 |
| 1024^2 | accurate | T_tau1e-08_f32_full | 0.190 | 0.100 | 0.126 | 1.734/2.124 |
| 1024^2 | fast | paper path | 0.560 | - | 0.872 | - |
| 1024^2 | fast | T_tau0.0001_f64_full | 0.171 | 0.070 | 0.088 | 1.741/2.147 |
| 1024^2 | fast | T_tau0.0001_f64_sub256 | 0.123 | 0.059 | 0.085 | 1.733/2.126 |
| 1024^2 | fast | T_tau1e-08_f64_full | 0.175 | 0.072 | 0.110 | 1.735/2.110 |
| 1024^2 | fast | T_tau1e-08_f32_full | 0.174 | 0.070 | 0.123 | 1.735/2.122 |
| 2048^2 | accurate | paper path | 0.820 | - | 10.901 | - |
| 2048^2 | accurate | T_tau0.0001_f64_full | 0.283 | 0.089 | 0.172 | 4.295/10.149 |
| 2048^2 | accurate | T_tau0.0001_f64_sub256 | 0.163 | 0.090 | 0.170 | 4.338/10.285 |
| 2048^2 | accurate | T_tau1e-08_f64_full | 0.274 | 0.106 | 0.194 | 4.316/10.157 |
| 2048^2 | accurate | T_tau1e-08_f32_full | 0.277 | 0.094 | 0.212 | 4.323/10.227 |
| 2048^2 | fast | paper path | 0.658 | - | 2.940 | - |
| 2048^2 | fast | T_tau0.0001_f64_full | 0.331 | 0.074 | 0.125 | 4.312/10.170 |
| 2048^2 | fast | T_tau0.0001_f64_sub256 | 0.171 | 0.060 | 0.121 | 4.327/10.246 |
| 2048^2 | fast | T_tau1e-08_f64_full | 0.249 | 0.072 | 0.174 | 4.307/10.176 |
| 2048^2 | fast | T_tau1e-08_f32_full | 0.249 | 0.070 | 0.199 | 4.314/10.170 |
| 4096^2 | accurate | paper path | 1.127 | - | 43.086 | - |
| 4096^2 | accurate | T_tau0.0001_f64_full | 0.537 | 0.089 | 0.320 | 16.207/36.237 |
| 4096^2 | accurate | T_tau0.0001_f64_sub256 | 0.187 | 0.091 | 0.321 | 16.106/36.077 |
| 4096^2 | accurate | T_tau1e-08_f64_full | 0.543 | 0.121 | 0.402 | 16.134/36.174 |
| 4096^2 | accurate | T_tau1e-08_f32_full | 0.530 | 0.101 | 0.463 | 16.093/36.133 |
| 4096^2 | fast | paper path | 0.823 | - | 11.603 | - |
| 4096^2 | fast | T_tau0.0001_f64_full | 0.472 | 0.072 | 0.244 | 16.133/36.055 |
| 4096^2 | fast | T_tau0.0001_f64_sub256 | 0.172 | 0.068 | 0.244 | 16.118/36.130 |
| 4096^2 | fast | T_tau1e-08_f64_full | 0.476 | 0.083 | 0.354 | 16.110/36.251 |
| 4096^2 | fast | T_tau1e-08_f32_full | 0.477 | 0.071 | 0.431 | 16.077/36.244 |

## Gates

| mesh | GPU | job | stored value exists | repro gate | max abs diff | dev gate (all) |
|---|---|---|---|---|---|---|
| 256^2 | NVIDIA A100 80GB PCIe | 7220235 | True | True | 5.204170427930421e-18 | True |
| 512^2 | NVIDIA A100 80GB PCIe | 7220235 | False | None | 0 | True |
| 1024^2 | NVIDIA A100 80GB PCIe | 7220235 | True | True | 6.938893903907228e-18 | True |
| 2048^2 | NVIDIA A100 80GB PCIe | 7220235 | True | True | 0 | True |
| 4096^2 | NVIDIA A100 80GB PCIe | 7221131 | True | True | 0 | True |

## Notes

- 256^2 accurate: stored paper run (job 4319401, A100 80GB): {"worst_error": 0.008378866617227667, "invocations": 320, "median_fused_device_seconds": 0.0005648075602948666, "median_total_seconds": 0.0011927976738661528}
- 256^2 fast: stored paper run (job 4319401, A100 80GB): {"worst_error": 0.040667799754386565, "invocations": 320, "median_fused_device_seconds": 0.00038066995330154896, "median_total_seconds": 0.0010810208041220903}
- 256^2 offline: bank build + compression 6.0283435909077525 s for the compression; range-finder residual {'128': 1.4543064970192752e-07, '512': 1.5136900991128923e-07}; device memory {'peak_bytes_in_use': 1589837824, 'bytes_limit': 80841015296}
- 512^2 offline: bank build + compression 11.195324710570276 s for the compression; range-finder residual {'128': 1.4081261968644108e-07, '512': 1.4916927049661354e-07}; device memory {'peak_bytes_in_use': 3194412288, 'bytes_limit': 80841015296}
- 1024^2 accurate: stored paper run (job 4319399, A100 80GB): {"worst_error": 0.008349808939792404, "invocations": 320, "median_fused_device_seconds": 0.0031026711221784353, "median_total_seconds": 0.00644827401265502}
- 1024^2 fast: stored paper run (job 4319399, A100 80GB): {"worst_error": 0.040575494932428326, "invocations": 320, "median_fused_device_seconds": 0.001176533056423068, "median_total_seconds": 0.004541890695691109}
- 1024^2 offline: bank build + compression 12.110190724022686 s for the compression; range-finder residual {'128': 1.4314025860303998e-07, '512': 1.5136840926582595e-07}; device memory {'peak_bytes_in_use': 7910124032, 'bytes_limit': 80841015296}
- 2048^2 accurate: stored paper run (job 4319396, A100 80GB): {"worst_error": 0.008348356150836013, "invocations": 320, "median_fused_device_seconds": 0.011662853648886085, "median_total_seconds": 0.025085158180445433}
- 2048^2 fast: stored paper run (job 4319396, A100 80GB): {"worst_error": 0.04057088749316021, "invocations": 320, "median_fused_device_seconds": 0.003404956543818116, "median_total_seconds": 0.016794663621112704}
- 2048^2 offline: bank build + compression 12.743010644800961 s for the compression; range-finder residual {'128': 1.4543064816576098e-07, '512': 1.5281694155698855e-07}; device memory {'peak_bytes_in_use': 23690599936, 'bytes_limit': 80841015296}
- 4096^2 accurate: stored paper run (job 4319394, A100 80GB): {"worst_error": 0.008347992954697186, "invocations": 320, "median_fused_device_seconds": 0.041915921960026026, "median_total_seconds": 0.09369846805930138}
- 4096^2 fast: stored paper run (job 4319394, A100 80GB): {"worst_error": 0.04056973574919411, "invocations": 320, "median_fused_device_seconds": 0.012110298499464989, "median_total_seconds": 0.06364867906086147}
- 4096^2 offline: bank build + compression 14.548099318519235 s for the compression; range-finder residual {'128': 1.3923924485213606e-07, '512': 1.5136840691108616e-07}; device memory {'peak_bytes_in_use': 75238372096, 'bytes_limit': 80841015296}
