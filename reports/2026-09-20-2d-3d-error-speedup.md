# Relative L2 error and FOM speed: both dimensions

This generated inventory covers completed NM-ROM comparisons in both dimensions and the matched three-dimensional operator panels. Development results remain provisional; no final three-dimensional cohort has been opened.

Errors are percentages. Each row uses its explicitly stated norm and reference convention; they must not be treated as one common cross-PDE metric. Speedup is $T_{\mathrm{FOM}}/T_{\mathrm{method}}$: above one means faster, below one means slower.

Fastest listed same-job tested FOM whose reported error is no greater than the displayed method. Burgers includes time-step and coarse-grid controls. Separable linear PDEs use the direct same-grid FOM; L-shaped Poisson includes sparse-direct and iterative controls. This is a finite tested comparator set, not proof of an optimal FOM.

Costs are median device times except L-shaped Poisson, which uses complete-query times to include its CPU direct solver. Every ratio is within a job. Source error medians, timing arrays/outlier counts where available, and full source metadata are preserved in the JSON. Summary-only sources do not supply a new outlier count here.

Burgers and Navier–Stokes retain the paper’s initial-normalized evolved-time L2 metric. Heat uses current-normalized evolved-time error. Poisson is steady-state current-normalized L2. Wave rows use current-normalized displacement L2; the energy and velocity errors remain separate, and an all-state pass is not claimed. Except as explicitly disclosed, comparisons use the same numerical-grid reference.

The 2D heat entry comes from the earlier audited bank/head experiment, not a new overnight retraining. For linear PDEs, unrestricted linear endpoints are listed separately from the nonlinear solves. Sealed Burgers2D accuracy from the independent final cohort is not attached to development timings.

No current 3D reflective-wave, L-shaped-Poisson, or low-viscosity-Burgers counterpart is available. The wider matched 2D operator programme remains incomplete; the existing Burgers2D U-Net/Transolver accuracy study lacks same-allocation FOM timings and is not assigned a speedup here.

**NM-ROM**

| Problem | Mesh | Method | Worst L2 (%) | Method ms | FOM | FOM L2 (%) | FOM ms | Speedup |
|---|---|---|---:|---:|---|---:|---:|---:|
| Burgers 2D | 256² intervals | `q0_M64_eqcert_g1em06_fastL4` | 1.8891 | 40.3586 | `nt1e-3_dt01` | 1.5179 | 19.7243 | 0.4887× |
| Burgers 2D | 256² intervals | `q256_M1088_eqtop_g1em06` | 0.5129 | 746.0200 | `nt1e-3_dt005` | 0.0489 | 31.7877 | 0.0426× |
| Burgers 2D | 512² intervals | `q0_M64_eqxfer_g1em06_fastL4` | 2.1400 | 40.4917 | `nt1e-3_dt01` | 1.5893 | 33.7158 | 0.8327× |
| Burgers 2D | 512² intervals | `q256_M1088_eqtopxfer_g1em06` | 0.5510 | 783.3277 | `nt1e-3_dt005` | 0.0516 | 52.9153 | 0.0676× |
| Burgers 2D | 1024² intervals | `q0_M64_eqxfer_g1em06` | 2.2913 | 40.2699 | `nt1e-4_dt01` | 1.6287 | 64.0352 | 1.5902× |
| Burgers 2D | 1024² intervals | `q256_M1088_dense_g1em06` | 0.5861 | 22053.8524 | `nt1e-4_dt005` | 0.0343 | 81.3113 | 0.0037× |
| Poisson 2D | 1024² intervals | `q0_m4@new_K32` | 3.1495 | 3.9406 | `dst_direct` | 0.0000 | 0.2369 | 0.0601× |
| Poisson 2D | 1024² intervals | `q256_m4@new_K32` | 0.9648 | 4.3188 | `dst_direct` | 0.0000 | 0.2369 | 0.0549× |
| Heat 2D | 1024² intervals | `nmrom` | 4.5555 | 12.3188 | `fom_same_grid` | 0.0000 | 1.0346 | 0.0840× |
| Reflective wave 2D | 1024² intervals | `head_q0` | 6.1066 | 192.4624 | `dst` | 0.0000 | 17.1028 | 0.0889× |
| Reflective wave 2D | 1024² intervals | `nested_q32` | 2.8094 | 2529.4450 | `dst` | 0.0000 | 17.1028 | 0.0068× |
| L-shaped Poisson 2D | 256² intervals | `neural_q0@head_sdf_R512_K16` | 3.8608 | 2.8787 | `fom_splu` | 0.0000 | 8.3862 | 2.9132× |
| L-shaped Poisson 2D | 256² intervals | `neural_q64@head_sdf_R512_K16` | 2.1305 | 3.0283 | `fom_splu` | 0.0000 | 8.3862 | 2.7693× |
| L-shaped Poisson 2D | 512² intervals | `neural_q0@head_sdf_R512_K16` | 3.8521 | 4.7161 | `fom_cg_gpu_r0.01` | 0.3845 | 29.1270 | 6.1761× |
| L-shaped Poisson 2D | 512² intervals | `neural_q64@head_sdf_R512_K16` | 2.1233 | 4.8008 | `fom_cg_gpu_r0.01` | 0.3845 | 29.1270 | 6.0670× |
| Navier–Stokes 2D | 256² periodic points | `neural_q0` | 68.7654 | 13387.5380 | `fom_ntol0.003` | 25.5895 | 263.1617 | 0.0197× |
| Navier–Stokes 2D | 256² periodic points | `neural_q512` | 6.0140 | 52592.3382 | `fom_ntol0.001` | 0.0041 | 421.2694 | 0.0080× |
| Low-viscosity Burgers 2D | 256² intervals | `q0_M1088_dense_g1em06` | 9.0500 | 557.4455 | `nt1e-2_dt01` | 3.8950 | 13.9836 | 0.0251× |
| Low-viscosity Burgers 2D | 256² intervals | `q256_M1088_dense_g1em06` | 6.6718 | 1810.2724 | `nt1e-2_dt01` | 3.8950 | 13.9836 | 0.0077× |
| Burgers 3D | 33³ nodes | `rom_q0` | 8.9875 | 223.3924 | `fom_n33_dt0.025_nt3e-02_lt5e-01` | 3.9285 | 4.6709 | 0.0209× |
| Burgers 3D | 33³ nodes | `rom_q192` | 2.4345 | 379.6557 | `fom_n33_dt0.01_nt1e-02_lt5e-01` | 1.8820 | 8.7436 | 0.0230× |
| Heat 3D | 32³ intervals | `nmrom_K32_q0_dense` | 6.0134 | 34.7709 | `dst_exact` | 0.0000 | 1.4283 | 0.0411× |
| Heat 3D | 32³ intervals | `nmrom_K32_q96_dense` | 1.3791 | 114.3431 | `dst_exact` | 0.0000 | 1.4283 | 0.0125× |
| Poisson 3D | 32³ intervals | `nmrom_K16_q0_dense` | 0.9704 | 2.4874 | `dst_exact` | 0.0000 | 0.1807 | 0.0726× |
| Poisson 3D | 32³ intervals | `nmrom_K16_q96_dense` | 0.1765 | 2.5886 | `dst_exact` | 0.0000 | 0.1807 | 0.0698× |
| Navier–Stokes 3D | 32³ periodic points | `nmrom_q0` | 50.4758 | 779.2309 | `fom_dt0.008` | 0.3598 | 3.8932 | 0.0050× |
| Navier–Stokes 3D | 32³ periodic points | `nmrom_q128` | 45.3348 | 1865.9994 | `fom_dt0.008` | 0.3598 | 3.8932 | 0.0021× |

**Linear endpoints**

| Problem | Mesh | Method | Worst L2 (%) | Method ms | FOM | FOM L2 (%) | FOM ms | Speedup |
|---|---|---|---:|---:|---|---:|---:|---:|
| Poisson 2D | 1024² intervals | `d_linear_qr_m4@new_K32` | 0.7421 | 1.8300 | `dst_direct` | 0.0000 | 0.2369 | 0.1295× |
| Heat 2D | 1024² intervals | `linear_weak_exact` | 1.2315 | 0.5617 | `fom_same_grid` | 0.0000 | 1.0346 | 1.8420× |
| Reflective wave 2D | 1024² intervals | `linear_bank64` | 2.8192 | 4.5624 | `dst` | 0.0000 | 17.1028 | 3.7487× |
| Heat 3D | 32³ intervals | `linear_bank_galerkin_exact` | 1.3324 | 0.1868 | `dst_exact` | 0.0000 | 1.4283 | 7.6463× |
| Poisson 3D | 32³ intervals | `linear_bank_galerkin` | 0.1443 | 0.2182 | `dst_exact` | 0.0000 | 0.1807 | 0.8281× |

**Neural operators**

| Problem | Mesh | Method | Worst L2 (%) | Method ms | FOM | FOM L2 (%) | FOM ms | Speedup |
|---|---|---|---:|---:|---|---:|---:|---:|
| Burgers 2D | 256² intervals | `fno-large` | 7.4164 | 7.1826 | `nt1e-2_dt01` | 3.1999 | 9.0092 | 1.2543× |
| Burgers 2D | 512² intervals | `fno-large` | 6.6169 | 22.4756 | `nt1e-2_dt01` | 3.3550 | 13.8515 | 0.6163× |
| Burgers 2D | 1024² intervals | `fno-large` | 6.2657 | 58.9173 | `nt1e-2_dt01` | 3.4582 | 17.7243 | 0.3008× |
| Burgers 3D | 33³ nodes | `deeponet3d` | 14.9530 | 2.6047 | `fom_n33_dt0.05_nt3e-02_lt5e-01` | 8.9982 | 3.9598 | 1.5202× |
| Burgers 3D | 33³ nodes | `fno3d` | 2.4995 | 6.0651 | `fom_n33_dt0.01_nt1e-02_lt5e-01` | 1.8820 | 8.7436 | 1.4416× |
| Burgers 3D | 33³ nodes | `transolver3d` | 2.1487 | 3.5753 | `fom_n33_dt0.01_nt1e-02_lt5e-01` | 1.8820 | 8.7436 | 2.4455× |
| Burgers 3D | 33³ nodes | `unet3d` | 2.6575 | 2.1744 | `fom_n33_dt0.01_nt1e-02_lt5e-01` | 1.8820 | 8.7436 | 4.0211× |
| Heat 3D | 32³ intervals | `deeponet3d_r128_w16` | 21.7692 | 3.8889 | `dst_exact` | 0.0000 | 1.4283 | 0.3673× |
| Heat 3D | 32³ intervals | `fno3d_w16_m6` | 0.3947 | 10.6383 | `dst_exact` | 0.0000 | 1.4283 | 0.1343× |
| Heat 3D | 32³ intervals | `transolver3d_w48_s32` | 1.4826 | 6.2822 | `dst_exact` | 0.0000 | 1.4283 | 0.2274× |
| Heat 3D | 32³ intervals | `unet3d_w8` | 0.3185 | 4.3730 | `dst_exact` | 0.0000 | 1.4283 | 0.3266× |
| Poisson 3D | 32³ intervals | `deeponet3d_r128_w16` | 3.5683 | 2.7780 | `dst_exact` | 0.0000 | 0.1807 | 0.0650× |
| Poisson 3D | 32³ intervals | `fno3d_w24_m8` | 0.2366 | 11.2086 | `dst_exact` | 0.0000 | 0.1807 | 0.0161× |
| Poisson 3D | 32³ intervals | `transolver3d_w48_s32` | 0.7639 | 4.1623 | `dst_exact` | 0.0000 | 0.1807 | 0.0434× |
| Poisson 3D | 32³ intervals | `unet3d_w16` | 0.2310 | 3.8710 | `dst_exact` | 0.0000 | 0.1807 | 0.0467× |
| Navier–Stokes 3D | 32³ periodic points | `deeponet3d_projected` | 44.8450 | 4.3721 | `fom_dt0.008` | 0.3598 | 3.8932 | 0.8905× |
| Navier–Stokes 3D | 32³ periodic points | `fno3d_projected` | 1.5121 | 8.9695 | `fom_dt0.008` | 0.3598 | 3.8932 | 0.4341× |
| Navier–Stokes 3D | 32³ periodic points | `transolver3d_projected` | 2.0366 | 3.3980 | `fom_dt0.008` | 0.3598 | 3.8932 | 1.1457× |
| Navier–Stokes 3D | 32³ periodic points | `unet3d_projected` | 4.2830 | 2.0937 | `fom_dt0.008` | 0.3598 | 3.8932 | 1.8595× |
| Navier–Stokes 3D, augmented training | 32³ periodic points | `deeponet3d_increment_projected` | 64.3294 | 4.2740 | `fom_dt0.02` | 49.2171 | 1.8829 | 0.4406× |
| Navier–Stokes 3D, augmented training | 32³ periodic points | `fno3d_increment_projected` | 0.4703 | 9.0895 | `fom_dt0.008` | 0.3598 | 3.7006 | 0.4071× |
| Navier–Stokes 3D, augmented training | 32³ periodic points | `transolver3d_increment_projected` | 2.6909 | 3.2767 | `fom_dt0.01` | 0.6284 | 3.1017 | 0.9466× |
| Navier–Stokes 3D, augmented training | 32³ periodic points | `unet3d_increment_projected` | 2.0441 | 1.9085 | `fom_dt0.01` | 0.6284 | 3.1017 | 1.6252× |

**Row qualifications and provenance**

- Burgers 2D: Development. Evolved-time advantage at the largest mesh disappears when initial compression is included. EQ construction status is retained in source_metrics. [Source JSON](../worktrees/2026-09-17-b-panel/experiments/b-panel/reports/summary.json).
- Burgers 2D: Paired FNO from the Burgers panel. This cohort/reference differs from the separate U-Net/Transolver operator study. [Source JSON](../worktrees/2026-09-17-b-panel/experiments/b-panel/reports/summary.json).
- Poisson 2D: Development. Iterative q=512 failed its stopping rule; the separately timed direct linear endpoint is listed instead. [Source JSON](../worktrees/2026-09-17-p-linear/experiments/p-linear/reports/summary.json).
- Heat 2D: Earlier audited bank/head experiment. Same-grid error here; the prior paper table instead reported physical-reference all-times error. [Source JSON](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/linear05/archive/outputs/results.json).
- Reflective wave 2D: Displacement only. Velocity and energy-state errors are separate source metrics; this does not establish an all-state accuracy pass. [Source JSON](../worktrees/2026-09-17-w-ladder/experiments/w-ladder/reports/summary.json).
- L-shaped Poisson 2D: Complete-query clock includes transfers and the CPU sparse-direct control. Development. [Source JSON](../worktrees/2026-09-17-lshape/experiments/lshape/reports/summary.json).
- Navier–Stokes 2D: Exploratory after the representation gate failed; q=512 reaches full-bank capacity. Initial-fit convergence is not implied by zero evolution budget exits. [Source JSON](../worktrees/2026-09-17-ns2d/experiments/ns2d/artifacts/ns304/result.json).
- Low-viscosity Burgers 2D: Fixed-test ladder. Under-resolved mesh: this is a comparison to the same discrete equation, not a resolved physical-accuracy claim. [Source JSON](../worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/reports/summary.json).
- Burgers 3D: Provisional development results; final cohort unopened. First of two independently trained checkpoints. Spatial refinement misses the physical-reference target. [Source JSON](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d005/collected/out/seed0/result.json).
- Heat 3D: Provisional development results; final cohort unopened. [Source JSON](../worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/runs/extra03/archive/out/result.json).
- Poisson 3D: Provisional development results; final cohort unopened. [Source JSON](../worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/seed04/archive/out/result.json).
- Navier–Stokes 3D: Provisional development results; final cohort unopened. Representation target failed. New paired fields pass numerical auditing; the separately retained strict cross-run replay gate failed. [Source JSON](../worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/extra03/collected/output/result.json).
- Navier–Stokes 3D, augmented training: Provisional development. Later training-coverage experiment with its own paired FOM. No new NM-ROM rollout passed the representation gate in this attempt. [Source JSON](../worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/coverage04/collected/output/result.json).

**Glossary**

- Relative L2: Euclidean field-difference norm divided by the named reference-field norm. Initial normalization divides by the initial state; current normalization divides by the reference at the evaluated time.
- Worst: maximum over evaluated cases and the stated output times, retaining bad cases. Median error, where available in the JSON, summarizes cases instead.
- Evolved times: requested output times after the supplied initial state. All times also scores initial-state compression.
- FOM: full-order numerical solver. DST: direct discrete sine transform solver. CG: conjugate gradients; its tolerance appears in the method name. Sparse direct uses a matrix factorization.
- NM-ROM: nonlinear-manifold reduced-order model. Head: compressed nonlinear map into a learned spatial bank. Linear endpoint: unrestricted solution in that bank.
- q: number of correction directions. K: nonlinear latent dimension. M: weak test count. R: bank rank. EQ: empirical quadrature, a sampled weighted residual evaluation. Dense: evaluates all grid points.
- Method/FOM ms: median milliseconds for the stated timing scope. Complete query includes transfers. GPU timing includes resident initialization, solution and requested dense outputs but excludes offline training.
- Speedup: FOM milliseconds divided by method milliseconds; each row uses its displayed comparator. A value below one is a slowdown.
- FNO, U-Net, DeepONet and Transolver: the four trained operator architectures tested. Projected: velocity output is projected to satisfy the discrete divergence-free constraint.
- Nodes / intervals / periodic points: different mesh-size conventions, stated explicitly. Steady state: one solution rather than a trajectory.
- Development: cases used while choosing or tuning configurations. Final cohort: separate cases reserved for evaluation after choices are frozen.
- Representation gate: required accuracy check before promoting a compressed model. Exploratory: a diagnostic experiment conducted without passing that gate.
- Same-grid reference: a converged solve of the discrete equation. Physical reference: a finer or analytic reference. Under-resolved: grid error remains too large for the stated physical target.
- Timing outlier: retained repetition taking more than one and a half times its method median. Source hash: checksum identifying the exact JSON used to generate this inventory.
