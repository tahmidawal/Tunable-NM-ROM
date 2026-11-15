# Heat 2D query time, a100-80G (stored paper values are H200) (job 7228875, node pax106, `NVIDIA A100 80GB PCIe, GPU-c692ba0c-a68c-8e5c-5811-dbbe618d443c, 81920 MiB, 575.57.08`)

Complete: True.  Frozen from the development cohort only at 2048^2: F = `T_tau1e-06_f64` (frozen file SHA256 1e4220ef2b0e6b1e..., tied candidates ['T_tau1e-06_f64']).

Times: median GPU-query in ms over 16 test cases x 5 repetitions; initial field on the device -> six output fields on the device, blocked (the paper's heat scope).  Errors: worst over the test cohort of the largest same-grid relative L2 error over the six output times.  "Coefficients only" returns the six reduced coefficient vectors and no field; it is a separate measurement, not the headline.  cuDSS / AmgX columns are the sparse-solvers lane's arithmetic estimates for 20 Crank-Nicolson solves (quoted, not measured trajectories).

## Headline (full six-field output)

| mesh | arm | worst err, paper path | worst err, F | paper stored | paper path, this job (A1/A2) | F, this job | F vs paper path | cuDSS 20 solves, factor reused (est.) | AmgX setup + 20 solves (est.) | cuDSS est. / F | coefficients only (no field) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024^2 | accurate (`lin_R128_cn`) | 0.001332 | 0.001332 | 0.636 | 1.263 (1.263/1.262) | **0.255** | 4.9x | 99 | 486 | 389x | 0.164 |
| 1024^2 | fast (`lin_R32_cn`) | 0.010823 | 0.010823 | 0.336 | 0.525 (0.524/0.525) | **0.238** | 2.2x | 99 | 486 | 417x | 0.164 |
| 2048^2 | accurate (`lin_R128_cn`) | 0.001332 | 0.001332 | 1.964 | 4.635 (4.633/4.636) | **0.528** | 8.8x | 334 | 1455 | 633x | 0.241 |
| 2048^2 | fast (`lin_R32_cn`) | 0.010823 | 0.010823 | 0.829 | 1.759 (1.758/1.759) | **0.490** | 3.6x | 334 | 1455 | 682x | 0.242 |

## All subjects

| mesh | arm | subject | ranks p, q | worst test err | max test field dev. from paper path | ms |
|---|---|---|---|---|---|---|
| 1024^2 | accurate | paper path | - | 0.001332 | - | 1.263 |
| 1024^2 | accurate | T_tau1e-06_f64 [F] | 49, 52 | 0.001332 | 3.0e-08 | 0.255 |
| 1024^2 | accurate | T_tau1e-08_f64 (ablation) | 64, 66 | 0.001332 | 2.9e-10 | 0.275 |
| 1024^2 | accurate | T_tau1e-08_f32 (ablation) | 64, 66 | 0.001332 | 5.5e-07 | 0.300 |
| 1024^2 | accurate | COEFFICIENTS ONLY, no field | - | - | - | 0.164 |
| 1024^2 | fast | paper path | - | 0.010823 | - | 0.525 |
| 1024^2 | fast | T_tau1e-06_f64 [F] | 41, 43 | 0.010823 | 2.1e-07 | 0.238 |
| 1024^2 | fast | T_tau1e-08_f64 (ablation) | 54, 58 | 0.010823 | 3.0e-09 | 0.253 |
| 1024^2 | fast | T_tau1e-08_f32 (ablation) | 54, 58 | 0.010823 | 5.3e-07 | 0.280 |
| 1024^2 | fast | COEFFICIENTS ONLY, no field | - | - | - | 0.164 |
| 2048^2 | accurate | paper path | - | 0.001332 | - | 4.635 |
| 2048^2 | accurate | T_tau1e-06_f64 [F] | 49, 52 | 0.001332 | 3.0e-08 | 0.528 |
| 2048^2 | accurate | T_tau1e-08_f64 (ablation) | 64, 67 | 0.001332 | 3.0e-10 | 0.596 |
| 2048^2 | accurate | T_tau1e-08_f32 (ablation) | 64, 67 | 0.001332 | 6.9e-07 | 0.695 |
| 2048^2 | accurate | COEFFICIENTS ONLY, no field | - | - | - | 0.241 |
| 2048^2 | fast | paper path | - | 0.010823 | - | 1.759 |
| 2048^2 | fast | T_tau1e-06_f64 [F] | 41, 43 | 0.010823 | 2.1e-07 | 0.490 |
| 2048^2 | fast | T_tau1e-08_f64 (ablation) | 54, 57 | 0.010823 | 3.0e-09 | 0.524 |
| 2048^2 | fast | T_tau1e-08_f32 (ablation) | 54, 57 | 0.010823 | 5.2e-07 | 0.662 |
| 2048^2 | fast | COEFFICIENTS ONLY, no field | - | - | - | 0.242 |

## Stage medians (ms, first test case)

| mesh | arm | paper: moments (transform) | paper: evolve | paper: six-field bank product | paper: whole | F: moments + coefficients | F: core contraction | F: field products | F: whole |
|---|---|---|---|---|---|---|---|---|---|
| 1024^2 | accurate | 0.217 | 0.066 | 1.167 | 1.244 | 0.212 | 0.080 | 0.181 | 0.212 |
| 1024^2 | fast | 0.226 | 0.064 | 0.396 | 0.486 | 0.154 | 0.061 | 0.149 | 0.191 |
| 2048^2 | accurate | 0.520 | 0.067 | 4.239 | 4.603 | 0.225 | 0.080 | 0.339 | 0.457 |
| 2048^2 | fast | 0.518 | 0.062 | 1.356 | 1.726 | 0.221 | 0.067 | 0.318 | 0.409 |

## Gates and offline cost

| mesh | reproduction gate (max abs diff of paper-path errors vs stored, tol 1e-09) | development gate (max dev., F, both arms; limit 1e-05) | moment identity | compression s | range-finder residual | peak device GB |
|---|---|---|---|---|---|---|
| 1024^2 | True (4.6e-17) | True (2.7e-07) | 4.3e-16 | 6.3 | 9.9e-08 | 13.7 |
| 2048^2 | True (6.2e-17) | True (2.7e-07) | 5.0e-16 | 6.9 | 1.0e-07 | 13.7 |

## Selection candidates (development cohort, 2048^2)

| variant | passes gate | max dev. | accurate ms | fast ms |
|---|---|---|---|---|
| T_tau0.0001_f64 | False | 2.1e-05 | 0.493 | 0.420 |
| T_tau0.0001_f32 | False | 2.1e-05 | 0.609 | 0.565 |
| T_tau1e-06_f64 | True | 2.7e-07 | 0.531 | 0.491 |
| T_tau1e-06_f32 | True | 8.0e-07 | 0.649 | 0.623 |
| T_tau1e-08_f64 | True | 3.0e-09 | 0.594 | 0.525 |
| T_tau1e-08_f32 | True | 8.0e-07 | 0.694 | 0.664 |
