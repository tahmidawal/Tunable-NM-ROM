# Poisson 2D query time, GPU type `h200` (job results `results`)

Frozen from development cases only: F = `T_tau0.0001_f64_full`, S = `T_tau0.0001_f64_sub256` (selection meshes [256, 4096], timing mesh 4096).

## Headline: full-field output, full-source input (configuration F), test cohort of 32 cases

Times are the median GPU-query in ms (device float64 nodal source -> device float64 nodal field of the whole mesh, synced).  S (source read on a 256-interval sub-lattice) and solve-only (reduced coefficients, no field) are separate measurements and are not the headline.

| mesh | arm | worst err, paper path | worst err, F | paper stored (A100) | paper path, this job | F, this job | cuDSS solve (quoted) | AmgX solve (quoted) | F vs paper path | F vs cuDSS | S (sampled input) | solve only (no field) | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256^2 | accurate | 0.008379 | 0.008379 | 0.565 | 1.335 | **0.175** | 0.448 | 13.900 | 7.6x | 2.6x | - | 0.126 | 7220234 |
| 256^2 | fast | 0.040668 | 0.040668 | 0.381 | 1.014 | **0.171** | 0.448 | 13.900 | 5.9x | 2.6x | - | 0.124 | 7220234 |
| 512^2 | accurate | 0.008356 | 0.008356 | - | 1.541 | **0.183** | 1.000 | 27.200 | 8.4x | 5.5x | 0.177 | 0.157 | 7220234 |
| 512^2 | fast | 0.040594 | 0.040594 | - | 1.315 | **0.175** | 1.000 | 27.200 | 7.5x | 5.7x | 0.175 | 0.125 | 7220234 |
| 1024^2 | accurate | 0.008350 | 0.008350 | 3.103 | 2.213 | **0.199** | 2.960 | 23.900 | 11.1x | 14.9x | 0.186 | 0.164 | 7220234 |
| 1024^2 | fast | 0.040575 | 0.040575 | 1.177 | 1.399 | **0.188** | 2.960 | 23.900 | 7.5x | 15.8x | 0.180 | 0.158 | 7220234 |
| 2048^2 | accurate | 0.008348 | 0.008348 | 11.663 | 5.408 | **0.254** | 9.360 | 48.400 | 21.3x | 36.8x | 0.213 | 0.197 | 7220234 |
| 2048^2 | fast | 0.040571 | 0.040571 | 3.405 | 2.327 | **0.235** | 9.360 | 48.400 | 9.9x | 39.9x | 0.202 | 0.188 | 7220234 |
| 4096^2 | accurate | 0.008348 | 0.008348 | 41.916 | 17.872 | **0.449** | 36.500 | 140.0 | 39.8x | 81.2x | 0.310 | 0.316 | 7220234 |
| 4096^2 | fast | 0.040570 | 0.040570 | 12.110 | 5.593 | **0.408** | 36.500 | 140.0 | 13.7x | 89.5x | 0.295 | 0.297 | 7220234 |

## Query time and error (test cohort, 32 cases; GPU-query = device f64 source -> device f64 full field)

| mesh | arm | path | worst err | mean err | max field dev. from paper path | ms (median) | paper A1/A2 ms | cuDSS ms | AmgX ms | ours / cuDSS |
|---|---|---|---|---|---|---|---|---|---|---|
| 256^2 | accurate | paper path | 0.008379 | 0.001982 | 0 | 1.335 | 1.259/1.409 | 0.448 | 13.900 | 2.98 |
| 256^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008379 | 0.001982 | 1.4e-06 | 0.175 | - | 0.448 | 13.900 | 0.39 |
| 256^2 | accurate | T_tau1e-08_f64_full (p=110, q=108) | 0.008379 | 0.001982 | 2.4e-10 | 0.190 | - | 0.448 | 13.900 | 0.43 |
| 256^2 | accurate | T_tau1e-08_f32_full (p=110, q=108) | 0.008379 | 0.001982 | 3.1e-07 | 0.185 | - | 0.448 | 13.900 | 0.41 |
| 256^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.126 | - | 0.448 | 13.900 | not comparable |
| 256^2 | fast | paper path | 0.040668 | 0.008042 | 0 | 1.014 | 0.861/1.054 | 0.448 | 13.900 | 2.26 |
| 256^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040668 | 0.008042 | 8.8e-06 | 0.171 | - | 0.448 | 13.900 | 0.38 |
| 256^2 | fast | T_tau1e-08_f64_full (p=94, q=95) | 0.040668 | 0.008042 | 1.1e-09 | 0.176 | - | 0.448 | 13.900 | 0.39 |
| 256^2 | fast | T_tau1e-08_f32_full (p=94, q=95) | 0.040668 | 0.008042 | 3.2e-07 | 0.179 | - | 0.448 | 13.900 | 0.40 |
| 256^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.124 | - | 0.448 | 13.900 | not comparable |
| 512^2 | accurate | paper path | 0.008356 | 0.001977 | 0 | 1.541 | 1.504/1.628 | 1.000 | 27.200 | 1.54 |
| 512^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008356 | 0.001977 | 1.4e-06 | 0.183 | - | 1.000 | 27.200 | 0.18 |
| 512^2 | accurate | T_tau0.0001_f64_sub256 [S] (p=65, q=66) | 0.008356 | 0.001977 | 4.2e-06 | 0.177 | - | 1.000 | 27.200 | 0.18 |
| 512^2 | accurate | T_tau1e-08_f64_full (p=110, q=109) | 0.008356 | 0.001977 | 2.4e-10 | 0.196 | - | 1.000 | 27.200 | 0.20 |
| 512^2 | accurate | T_tau1e-08_f32_full (p=110, q=109) | 0.008356 | 0.001977 | 3.6e-07 | 0.190 | - | 1.000 | 27.200 | 0.19 |
| 512^2 | accurate | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.123 | - | 1.000 | 27.200 | not comparable |
| 512^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.157 | - | 1.000 | 27.200 | not comparable |
| 512^2 | fast | paper path | 0.040594 | 0.008027 | 0 | 1.315 | 1.285/1.347 | 1.000 | 27.200 | 1.32 |
| 512^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040594 | 0.008027 | 8.8e-06 | 0.175 | - | 1.000 | 27.200 | 0.18 |
| 512^2 | fast | T_tau0.0001_f64_sub256 [S] (p=50, q=51) | 0.040594 | 0.008027 | 8.8e-06 | 0.175 | - | 1.000 | 27.200 | 0.17 |
| 512^2 | fast | T_tau1e-08_f64_full (p=95, q=95) | 0.040594 | 0.008027 | 1.1e-09 | 0.181 | - | 1.000 | 27.200 | 0.18 |
| 512^2 | fast | T_tau1e-08_f32_full (p=95, q=95) | 0.040594 | 0.008027 | 3.3e-07 | 0.182 | - | 1.000 | 27.200 | 0.18 |
| 512^2 | fast | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.124 | - | 1.000 | 27.200 | not comparable |
| 512^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.125 | - | 1.000 | 27.200 | not comparable |
| 1024^2 | accurate | paper path | 0.008350 | 0.001976 | 0 | 2.213 | 2.194/2.258 | 2.960 | 23.900 | 0.75 |
| 1024^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008350 | 0.001976 | 1.4e-06 | 0.199 | - | 2.960 | 23.900 | 0.07 |
| 1024^2 | accurate | T_tau0.0001_f64_sub256 [S] (p=65, q=66) | 0.008350 | 0.001976 | 5.2e-06 | 0.186 | - | 2.960 | 23.900 | 0.06 |
| 1024^2 | accurate | T_tau1e-08_f64_full (p=110, q=109) | 0.008350 | 0.001976 | 2.3e-10 | 0.213 | - | 2.960 | 23.900 | 0.07 |
| 1024^2 | accurate | T_tau1e-08_f32_full (p=110, q=109) | 0.008350 | 0.001976 | 3.7e-07 | 0.210 | - | 2.960 | 23.900 | 0.07 |
| 1024^2 | accurate | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.124 | - | 2.960 | 23.900 | not comparable |
| 1024^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.164 | - | 2.960 | 23.900 | not comparable |
| 1024^2 | fast | paper path | 0.040575 | 0.008023 | 0 | 1.399 | 1.372/1.436 | 2.960 | 23.900 | 0.47 |
| 1024^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040575 | 0.008023 | 8.8e-06 | 0.188 | - | 2.960 | 23.900 | 0.06 |
| 1024^2 | fast | T_tau0.0001_f64_sub256 [S] (p=50, q=51) | 0.040575 | 0.008023 | 8.8e-06 | 0.180 | - | 2.960 | 23.900 | 0.06 |
| 1024^2 | fast | T_tau1e-08_f64_full (p=95, q=95) | 0.040575 | 0.008023 | 1.1e-09 | 0.192 | - | 2.960 | 23.900 | 0.06 |
| 1024^2 | fast | T_tau1e-08_f32_full (p=95, q=95) | 0.040575 | 0.008023 | 3.3e-07 | 0.197 | - | 2.960 | 23.900 | 0.07 |
| 1024^2 | fast | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.124 | - | 2.960 | 23.900 | not comparable |
| 1024^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.158 | - | 2.960 | 23.900 | not comparable |
| 2048^2 | accurate | paper path | 0.008348 | 0.001976 | 0 | 5.408 | 5.366/5.471 | 9.360 | 48.400 | 0.58 |
| 2048^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008348 | 0.001976 | 1.4e-06 | 0.254 | - | 9.360 | 48.400 | 0.03 |
| 2048^2 | accurate | T_tau0.0001_f64_sub256 [S] (p=65, q=66) | 0.008348 | 0.001976 | 5.4e-06 | 0.213 | - | 9.360 | 48.400 | 0.02 |
| 2048^2 | accurate | T_tau1e-08_f64_full (p=110, q=108) | 0.008348 | 0.001976 | 2.6e-10 | 0.272 | - | 9.360 | 48.400 | 0.03 |
| 2048^2 | accurate | T_tau1e-08_f32_full (p=110, q=108) | 0.008348 | 0.001976 | 3.5e-07 | 0.277 | - | 9.360 | 48.400 | 0.03 |
| 2048^2 | accurate | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.128 | - | 9.360 | 48.400 | not comparable |
| 2048^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.197 | - | 9.360 | 48.400 | not comparable |
| 2048^2 | fast | paper path | 0.040571 | 0.008022 | 0 | 2.327 | 2.259/2.374 | 9.360 | 48.400 | 0.25 |
| 2048^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040571 | 0.008022 | 8.8e-06 | 0.235 | - | 9.360 | 48.400 | 0.03 |
| 2048^2 | fast | T_tau0.0001_f64_sub256 [S] (p=50, q=51) | 0.040571 | 0.008022 | 8.8e-06 | 0.202 | - | 9.360 | 48.400 | 0.02 |
| 2048^2 | fast | T_tau1e-08_f64_full (p=95, q=95) | 0.040571 | 0.008022 | 1.0e-09 | 0.249 | - | 9.360 | 48.400 | 0.03 |
| 2048^2 | fast | T_tau1e-08_f32_full (p=95, q=95) | 0.040571 | 0.008022 | 3.3e-07 | 0.265 | - | 9.360 | 48.400 | 0.03 |
| 2048^2 | fast | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.126 | - | 9.360 | 48.400 | not comparable |
| 2048^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.188 | - | 9.360 | 48.400 | not comparable |
| 4096^2 | accurate | paper path | 0.008348 | 0.001975 | 0 | 17.872 | 17.795/17.971 | 36.500 | 140.0 | 0.49 |
| 4096^2 | accurate | T_tau0.0001_f64_full [F] (p=65, q=66) | 0.008348 | 0.001975 | 1.4e-06 | 0.449 | - | 36.500 | 140.0 | 0.01 |
| 4096^2 | accurate | T_tau0.0001_f64_sub256 [S] (p=65, q=66) | 0.008348 | 0.001975 | 5.5e-06 | 0.310 | - | 36.500 | 140.0 | 0.01 |
| 4096^2 | accurate | T_tau1e-08_f64_full (p=110, q=109) | 0.008348 | 0.001975 | 2.5e-10 | 0.475 | - | 36.500 | 140.0 | 0.01 |
| 4096^2 | accurate | T_tau1e-08_f32_full (p=110, q=109) | 0.008348 | 0.001975 | 3.8e-07 | 0.534 | - | 36.500 | 140.0 | 0.01 |
| 4096^2 | accurate | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.168 | - | 36.500 | 140.0 | not comparable |
| 4096^2 | accurate | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.316 | - | 36.500 | 140.0 | not comparable |
| 4096^2 | fast | paper path | 0.040570 | 0.008022 | 0 | 5.593 | 5.560/5.638 | 36.500 | 140.0 | 0.15 |
| 4096^2 | fast | T_tau0.0001_f64_full [F] (p=50, q=51) | 0.040570 | 0.008022 | 8.8e-06 | 0.408 | - | 36.500 | 140.0 | 0.01 |
| 4096^2 | fast | T_tau0.0001_f64_sub256 [S] (p=50, q=51) | 0.040570 | 0.008022 | 8.8e-06 | 0.295 | - | 36.500 | 140.0 | 0.01 |
| 4096^2 | fast | T_tau1e-08_f64_full (p=95, q=95) | 0.040570 | 0.008022 | 9.8e-10 | 0.436 | - | 36.500 | 140.0 | 0.01 |
| 4096^2 | fast | T_tau1e-08_f32_full (p=95, q=95) | 0.040570 | 0.008022 | 3.5e-07 | 0.496 | - | 36.500 | 140.0 | 0.01 |
| 4096^2 | fast | SOLVE ONLY, no field (solve_only_sub256) | - | - | - | 0.154 | - | 36.500 | 140.0 | not comparable |
| 4096^2 | fast | SOLVE ONLY, no field (solve_only_full) | - | - | - | 0.297 | - | 36.500 | 140.0 | not comparable |

## Stage medians, ms (separately jitted and synced; they do not sum to the fused time)

| mesh | arm | path | projection + solve | core contraction | reconstruction / field products | host->device / device->host copy |
|---|---|---|---|---|---|---|
| 256^2 | accurate | paper path | 0.676 | - | 0.414 | - |
| 256^2 | accurate | T_tau0.0001_f64_full | 0.177 | 0.080 | 0.072 | 0.449/1.383 |
| 256^2 | accurate | T_tau1e-08_f64_full | 0.184 | 0.114 | 0.073 | 0.456/1.412 |
| 256^2 | accurate | T_tau1e-08_f32_full | 0.180 | 0.085 | 0.080 | 0.456/1.321 |
| 256^2 | fast | paper path | 0.424 | - | 0.139 | - |
| 256^2 | fast | T_tau0.0001_f64_full | 0.174 | 0.079 | 0.072 | 0.449/1.216 |
| 256^2 | fast | T_tau1e-08_f64_full | 0.194 | 0.081 | 0.073 | 0.463/1.172 |
| 256^2 | fast | T_tau1e-08_f32_full | 0.178 | 0.079 | 0.080 | 0.461/1.194 |
| 512^2 | accurate | paper path | 0.581 | - | 0.578 | - |
| 512^2 | accurate | T_tau0.0001_f64_full | 0.192 | 0.087 | 0.104 | 0.804/1.385 |
| 512^2 | accurate | T_tau0.0001_f64_sub256 | 0.145 | 0.084 | 0.100 | 0.787/1.415 |
| 512^2 | accurate | T_tau1e-08_f64_full | 0.197 | 0.277 | 0.106 | 0.793/1.376 |
| 512^2 | accurate | T_tau1e-08_f32_full | 0.178 | 0.080 | 0.077 | 0.796/1.393 |
| 512^2 | fast | paper path | 0.442 | - | 0.265 | - |
| 512^2 | fast | T_tau0.0001_f64_full | 0.198 | 0.094 | 0.080 | 0.799/1.384 |
| 512^2 | fast | T_tau0.0001_f64_sub256 | 0.155 | 0.085 | 0.075 | 0.786/1.368 |
| 512^2 | fast | T_tau1e-08_f64_full | 0.203 | 0.104 | 0.105 | 0.804/1.358 |
| 512^2 | fast | T_tau1e-08_f32_full | 0.197 | 0.095 | 0.116 | 0.796/1.357 |
| 1024^2 | accurate | paper path | 0.631 | - | 1.296 | - |
| 1024^2 | accurate | T_tau0.0001_f64_full | 0.196 | 0.084 | 0.102 | 1.669/1.994 |
| 1024^2 | accurate | T_tau0.0001_f64_sub256 | 0.158 | 0.089 | 0.119 | 1.657/1.983 |
| 1024^2 | accurate | T_tau1e-08_f64_full | 0.195 | 0.116 | 0.130 | 1.663/2.047 |
| 1024^2 | accurate | T_tau1e-08_f32_full | 0.200 | 0.087 | 0.139 | 1.663/2.004 |
| 1024^2 | fast | paper path | 0.455 | - | 0.594 | - |
| 1024^2 | fast | T_tau0.0001_f64_full | 0.197 | 0.083 | 0.077 | 1.673/2.053 |
| 1024^2 | fast | T_tau0.0001_f64_sub256 | 0.145 | 0.079 | 0.076 | 1.667/2.012 |
| 1024^2 | fast | T_tau1e-08_f64_full | 0.197 | 0.088 | 0.107 | 1.654/2.019 |
| 1024^2 | fast | T_tau1e-08_f32_full | 0.191 | 0.082 | 0.111 | 1.665/1.964 |
| 2048^2 | accurate | paper path | 0.790 | - | 4.381 | - |
| 2048^2 | accurate | T_tau0.0001_f64_full | 0.236 | 0.084 | 0.147 | 5.315/8.751 |
| 2048^2 | accurate | T_tau0.0001_f64_sub256 | 0.182 | 0.086 | 0.151 | 5.317/8.702 |
| 2048^2 | accurate | T_tau1e-08_f64_full | 0.238 | 0.117 | 0.155 | 5.304/8.711 |
| 2048^2 | accurate | T_tau1e-08_f32_full | 0.244 | 0.085 | 0.177 | 5.318/8.752 |
| 2048^2 | fast | paper path | 0.519 | - | 1.278 | - |
| 2048^2 | fast | T_tau0.0001_f64_full | 0.231 | 0.086 | 0.116 | 5.312/8.714 |
| 2048^2 | fast | T_tau0.0001_f64_sub256 | 0.151 | 0.082 | 0.113 | 5.310/8.685 |
| 2048^2 | fast | T_tau1e-08_f64_full | 0.239 | 0.093 | 0.153 | 5.322/8.723 |
| 2048^2 | fast | T_tau1e-08_f32_full | 0.236 | 0.082 | 0.175 | 5.320/8.755 |
| 4096^2 | accurate | paper path | 0.996 | - | 17.041 | - |
| 4096^2 | accurate | T_tau0.0001_f64_full | 0.341 | 0.086 | 0.247 | 20.051/32.599 |
| 4096^2 | accurate | T_tau0.0001_f64_sub256 | 0.154 | 0.084 | 0.248 | 20.061/32.595 |
| 4096^2 | accurate | T_tau1e-08_f64_full | 0.400 | 0.120 | 0.259 | 20.078/32.619 |
| 4096^2 | accurate | T_tau1e-08_f32_full | 0.342 | 0.088 | 0.313 | 20.050/32.620 |
| 4096^2 | fast | paper path | 0.772 | - | 4.719 | - |
| 4096^2 | fast | T_tau0.0001_f64_full | 0.351 | 0.083 | 0.193 | 20.064/32.595 |
| 4096^2 | fast | T_tau0.0001_f64_sub256 | 0.161 | 0.085 | 0.183 | 20.079/32.589 |
| 4096^2 | fast | T_tau1e-08_f64_full | 0.321 | 0.086 | 0.239 | 20.084/32.600 |
| 4096^2 | fast | T_tau1e-08_f32_full | 0.342 | 0.090 | 0.317 | 20.086/32.584 |

## Gates

| mesh | GPU | job | stored value exists | repro gate | max abs diff | dev gate (all) |
|---|---|---|---|---|---|---|
| 256^2 | NVIDIA H200 | 7220234 | True | True | 6.938893903907228e-18 | True |
| 512^2 | NVIDIA H200 | 7220234 | False | None | 0 | True |
| 1024^2 | NVIDIA H200 | 7220234 | True | True | 1.3877787807814457e-17 | True |
| 2048^2 | NVIDIA H200 | 7220234 | True | True | 8.673617379884035e-17 | True |
| 4096^2 | NVIDIA H200 | 7220234 | True | True | 5.637851296924623e-17 | True |

## Notes

- 256^2 accurate: stored paper run (job 4319401, A100 80GB): {"worst_error": 0.008378866617227667, "invocations": 320, "median_fused_device_seconds": 0.0005648075602948666, "median_total_seconds": 0.0011927976738661528}
- 256^2 fast: stored paper run (job 4319401, A100 80GB): {"worst_error": 0.040667799754386565, "invocations": 320, "median_fused_device_seconds": 0.00038066995330154896, "median_total_seconds": 0.0010810208041220903}
- 256^2 offline: bank build + compression 6.107827805913985 s for the compression; range-finder residual {'128': 1.4543064970192752e-07, '512': 1.5136900991128923e-07}; device memory {'peak_bytes_in_use': 1619197952, 'bytes_limit': 142606336000}
- 512^2 offline: bank build + compression 10.812132546678185 s for the compression; range-finder residual {'128': 1.4081261968644108e-07, '512': 1.4842897178062417e-07}; device memory {'peak_bytes_in_use': 3221675520, 'bytes_limit': 142606336000}
- 1024^2 accurate: stored paper run (job 4319399, A100 80GB): {"worst_error": 0.008349808939792404, "invocations": 320, "median_fused_device_seconds": 0.0031026711221784353, "median_total_seconds": 0.00644827401265502}
- 1024^2 fast: stored paper run (job 4319399, A100 80GB): {"worst_error": 0.040575494932428326, "invocations": 320, "median_fused_device_seconds": 0.001176533056423068, "median_total_seconds": 0.004541890695691109}
- 1024^2 offline: bank build + compression 11.435807411558926 s for the compression; range-finder residual {'128': 1.4314025860303998e-07, '512': 1.5136840926582595e-07}; device memory {'peak_bytes_in_use': 7939484160, 'bytes_limit': 142606336000}
- 2048^2 accurate: stored paper run (job 4319396, A100 80GB): {"worst_error": 0.008348356150836013, "invocations": 320, "median_fused_device_seconds": 0.011662853648886085, "median_total_seconds": 0.025085158180445433}
- 2048^2 fast: stored paper run (job 4319396, A100 80GB): {"worst_error": 0.04057088749316021, "invocations": 320, "median_fused_device_seconds": 0.003404956543818116, "median_total_seconds": 0.016794663621112704}
- 2048^2 offline: bank build + compression 11.218928924761713 s for the compression; range-finder residual {'128': 1.4543064816576098e-07, '512': 1.5281694155698855e-07}; device memory {'peak_bytes_in_use': 23750298880, 'bytes_limit': 142606336000}
- 4096^2 accurate: stored paper run (job 4319394, A100 80GB): {"worst_error": 0.008347992954697186, "invocations": 320, "median_fused_device_seconds": 0.041915921960026026, "median_total_seconds": 0.09369846805930138}
- 4096^2 fast: stored paper run (job 4319394, A100 80GB): {"worst_error": 0.04056973574919411, "invocations": 320, "median_fused_device_seconds": 0.012110298499464989, "median_total_seconds": 0.06364867906086147}
- 4096^2 offline: bank build + compression 11.522875222377479 s for the compression; range-finder residual {'128': 1.3923924485213606e-07, '512': 1.5136840691108616e-07}; device memory {'peak_bytes_in_use': 80979113728, 'bytes_limit': 142606336000}
