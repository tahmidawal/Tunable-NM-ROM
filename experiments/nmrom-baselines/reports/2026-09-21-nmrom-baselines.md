# Kim et al. masked-autoencoder NM-LSPG versus the project NM-ROM on the shared Burgers 2D family

Reproduction gate: **PASSED**. **Adapted** reproduction: passed on attempt 3 of the pre-registered three (activation sigmoid), not on the paper-default attempt 1; median 1.45 % against the 1.5 % bar (published < 1 %), i.e. above the published figure. Hyper-reduction was NOT reproduced (HR gate failed on every attempt), so every HR arm is exploratory. Kim-baseline rows are admissible only where marked (same code hashes and the gate activation).
All numbers are generated from run JSONs by `reports/gen_report.py`; state: final for this lane.

## 1. Reproduction gate (Kim et al. 2022, Section 6.2; published NM-LSPG < 1 %, NM-LSPG-HR 0.93–0.98 %, LS-LSPG-HR 34–38 %)

Pass rule (pre-registered): median of three seeds ≤ 1.5 % and a valid LS-LSPG control ≥ 10 %; HR: median at 55/58 ≤ 2 %.

| job | attempt | activation / scaling | NM-LSPG per seed | median | autoencode-only per seed | LS-LSPG control | HR 55/58 per seed (basis) | gate | HR gate |
|---|---|---|---|---|---|---|---|---|---|
| 4059370 | 1 | swish / feature | 1.67 %, 1.45 %, 1.73 % | 1.67 % | 1.04 %, 0.85 %, 1.21 % | 31.60 % | 131.69 %, 318.89 %, 154.03 % (residual) | FAIL | FAIL |
| 4072224 | 2 | swish / global | 709.38 %, 11.22 %, 10.89 % | 11.22 % | 5.60 %, 3.81 %, 3.86 % | 31.60 % | 9006.73 %, 13143.74 %, 2050.67 % (sns) | FAIL | FAIL |
| 4072224 | 1 (weights reloaded) | swish / feature | 1.67 %, 1.45 %, 1.73 % | 1.67 % | 1.04 %, 0.85 %, 1.21 % | 31.60 % | 386.56 %, 13005.47 %, 18777.20 % (sns) | FAIL | FAIL |
| 4077574 | 3 | sigmoid / feature | 1.45 %, 1.83 %, 1.44 % | 1.45 % | 1.58 %, 1.79 %, 1.49 % | 31.60 % | diverged, diverged, diverged (sns) | pass | FAIL |

## Shared Burgers family, 128² intervals (n = 16129), job 4073272, NVIDIA A100 80GB PCIe, GPU-9732c808-acc9-5c43-f0ce-4a81a2bf1719

Kim rows: **code matches a passed gate; a Kim row is admissible only if its activation is the gate activation (sigmoid)**. Other Kim hyper-parameters are tuned on the family (DESIGN s.3) and printed in the `act` column and variant. Cohort = 32 held-out validation cases unless the column says tune (16 training-side cases used for every choice). NumPy audit: all audited rows agree.

| arm | family | solved unknowns | worst evolved (validation) | median evolved | worst evolved (tune) | autoencode-only worst | GN cap hits | query GPU ms (median) | compiled-query memory MB | training s (epochs, stop) |
|---|---|---|---|---|---|---|---|---|---|---|
| `pod_lspg_zero_K8` | pod_lspg | 8 | 54.95 % | 22.34 % | — | — | 0 | 56.2 | 4 | 0 (—, —) |
| `pod_lspg_zero_K16` | pod_lspg | 16 | 40.85 % | 9.93 % | — | — | 0 | 93.6 | 8 | 0 (—, —) |
| `pod_lspg_zero_K32` | pod_lspg | 32 | 23.35 % | 4.20 % | — | — | 1 | 176.3 | 14 | 0 (—, —) |
| `pod_lspg_zero_K64` | pod_lspg | 64 | 12.27 % | 1.41 % | — | — | 1 | 355.7 | 26 | 0 (—, —) |
| `pod_lspg_zero_K128` | pod_lspg | 128 | 3.57 % | 0.30 % | — | — | 0 | 679.6 | 51 | 0 (—, —) |
| `pod_lspg_ic_K8` | pod_lspg | 8 | 100.35 % | 55.88 % | — | — | 0 | 64.7 | 4 | 0 (—, —) |
| `pod_lspg_ic_K16` | pod_lspg | 16 | 66.79 % | 39.25 % | — | — | 0 | 108.7 | 8 | 0 (—, —) |
| `pod_lspg_ic_K32` | pod_lspg | 32 | 45.93 % | 23.56 % | — | — | 1 | 205.9 | 14 | 0 (—, —) |
| `pod_lspg_ic_K64` | pod_lspg | 64 | 33.67 % | 5.07 % | — | — | 2 | 414.8 | 26 | 0 (—, —) |
| `pod_lspg_ic_K128` | pod_lspg | 128 | 10.73 % | 1.13 % | — | — | 0 | 716.9 | 51 | 0 (—, —) |
| `kim_K16_base` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 84.39 % | 33.20 % | 83.52 % | 40.76 % | 46 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_refzero` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 43.04 % | 19.68 % | 43.62 % | 30.06 % | 17 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_global` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 84.63 % | 30.44 % | 83.73 % | 40.30 % | 28 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_refzero_global` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 43.42 % | 19.49 % | 35.95 % | 29.73 % | 5 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_b50` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 85.30 % | 34.93 % | 84.56 % | 38.26 % | 19 | not timed | — | 1200 (1174, wall_budget) |
| `kim_K16_b200` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 68.45 % | 29.96 % | 69.78 % | 40.45 % | 25 | not timed | — | 1201 (895, wall_budget) |
| `kim_K16_M1_4096` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 67.92 % | 29.52 % | 69.54 % | 34.81 % | 30 | not timed | — | 1200 (2077, wall_budget) |
| `kim_K16_M1_1024` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 66.74 % | 28.65 % | 68.97 % | 38.52 % | 18 | not timed | — | 1190 (2480, early_stop) |
| `kim_K16_pat50` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 63.81 % | 27.79 % | 69.77 % | 43.25 % | 17 | not timed | — | 461 (334, early_stop) |
| `kim_K16_lr3e4_pat50` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 70.64 % | 31.66 % | 70.02 % | 38.26 % | 29 | not timed | — | 1201 (869, wall_budget) |
| `kim_K16_f64` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 84.57 % | 32.24 % | 84.39 % | 38.78 % | 36 | not timed | — | 1200 (454, wall_budget) |
| `kim_final_K8` [swish] INADMISSIBLE | kim_nm_lspg | 8 | 50.57 % | 21.53 % | 44.40 % | 34.66 % | 11 | 301.1 | 4252 | 1200 (870, wall_budget) |
| `kim_final_K8_hr` [swish] INADMISSIBLE | kim_nm_lspg_hr (exploratory HR) | 8 | 54.16 % | 26.59 % | — | — | — | 79.9 | 4258 | 0 (—, —) |
| `kim_final_K16` [swish] INADMISSIBLE | kim_nm_lspg | 16 | 43.42 % | 19.49 % | 35.95 % | 29.73 % | 5 | 518.9 | 4275 | 0 (—, —) |
| `kim_final_K16_hr` [swish] INADMISSIBLE | kim_nm_lspg_hr (exploratory HR) | 16 | 44.75 % | 19.03 % | — | — | — | 132.5 | 4321 | 0 (—, —) |
| `kim_final_K32` [swish] INADMISSIBLE | kim_nm_lspg | 32 | 40.84 % | 14.99 % | 35.11 % | 21.28 % | 9 | 945.6 | 4322 | 1200 (864, wall_budget) |
| `kim_final_K32_hr` [swish] INADMISSIBLE | kim_nm_lspg_hr (exploratory HR) | 32 | 58.15 % | 20.22 % | — | — | — | 166.5 | 4328 | 0 (—, —) |
| `ours_q0` | ours | 16 | 3.41 % | 0.59 % | — | — | — | 109.2 | 190 | 0 (—, —) |
| `ours_q256` | ours | 272 | 0.51 % | 0.08 % | — | — | — | 1353.7 | 504 | 0 (—, —) |
| `fom_fft_tight` | fom | — | 0.00 % | 0.00 % | — | — | — | 72.8 | 4 | 0 (—, —) |
| `fom_nt1e4_dt005` | fom | — | 0.05 % | 0.01 % | — | — | — | 32.2 | 4 | 0 (—, —) |

Selection rule: smallest worst evolved NM-LSPG error on the 16-case tuning subset (train cases 112-127). Selected: `kim_K16_refzero_global`.


## Glossary

- **NM-LSPG**: nonlinear-manifold least-squares Petrov–Galerkin; each backward-Euler step minimises the full discrete residual over the latent vector. **HR**: hyper-reduction (residual evaluated at sampled rows only).
- **worst evolved**: largest, over cases and the five output times after t = 0, of the field error against the same-grid full-order solve, divided by the norm of the initial field.
- **autoencode-only**: encode then decode the true states; not a lower bound on the manifold error.
- **solved unknowns**: number of unknowns in the online nonlinear solve (K, or K+q for the project ROM with q corrections).
- **tune**: 16 cases carved from the training split, used for every choice; **validation**: 32 held-out cases, never used to choose.
- **GN cap hits**: time steps whose Gauss–Newton solve hit the 20-iteration cap. **compiled-query memory**: XLA memory analysis (arguments + outputs + temporaries) of the jitted query.
- **POD-LSPG zero / ic**: linear basis with zero reference or with the initial field as reference. **ours_q0 / ours_q256**: frozen project checkpoint without / with 256 corrections.
- **FOM**: full-order model on the same grid; `fom_fft_tight` is the reference itself (error 0 by construction).
