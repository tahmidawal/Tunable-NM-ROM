# NM-ROM versus iterative CG: collected comparisons

Collected development measurements. Heat is an earlier independently audited checkpoint; current 3D CG measurements are pending. This generated table changes the named full-order comparator, not the trained models or the measured outputs.

Fastest retained same-job passing iterative CG whose displayed error is no greater than the method error; no cross-job ratio. This is selection from a finite measured tolerance/time-step set.

Speedup is $T_{\mathrm{CG}}/T_{\mathrm{method}}$. Values above one mean the displayed method is faster than the named CG implementation. These are not speedups over every available full-order solver; the direct-transform comparison remains in the companion inventory.

[Direct-FOM inventory](2026-09-20-2d-3d-error-speedup.md). Costs below are medians of synchronized GPU query measurements, including reduced-model initialization and requested dense device outputs. Offline training and host transfers are excluded.

| Problem | Intervals/axis | Method | L2 error (%) | Method ms | CG setting | CG error (%) | CG ms | Speedup |
|---|---:|---|---:|---:|---|---:|---:|---:|
| Poisson2D | 256 | `q0_m4@new_K32` | 3.1567 | 6.7031 | `cg_0.01` | 0.1527 | 22.2898 | 3.3253× |
| Poisson2D | 256 | `q256_m4@new_K32` | 0.9689 | 7.1213 | `cg_0.01` | 0.1527 | 22.2898 | 3.1300× |
| Poisson2D | 256 | `d_linear_qr_m4@new_K32` | 0.7459 | 1.6928 | `cg_0.01` | 0.1527 | 22.2898 | 13.1671× |
| Poisson2D | 1024 | `q0_m4@new_K32` | 3.1495 | 3.9406 | `cg_0.01` | 0.0722 | 60.0170 | 15.2302× |
| Poisson2D | 1024 | `q256_m4@new_K32` | 0.9648 | 4.3188 | `cg_0.01` | 0.0722 | 60.0170 | 13.8968× |
| Poisson2D | 1024 | `d_linear_qr_m4@new_K32` | 0.7421 | 1.8300 | `cg_0.01` | 0.0722 | 60.0170 | 32.7964× |
| Reflective wave2D | 64 | `head_q0` | 6.3109 | 184.5351 | `cgdt_0.005_tol_0.01` | 2.7812 | 41.8936 | 0.2270× |
| Reflective wave2D | 64 | `nested_q32` | 2.6815 | 2500.0601 | `cgdt_0.005_tol_1e-06` | 1.5967 | 69.3737 | 0.0277× |
| Reflective wave2D | 64 | `linear_bank64` | 2.6913 | 1.1685 | `cgdt_0.005_tol_1e-06` | 1.5967 | 69.3737 | 59.3705× |
| Reflective wave2D | 256 | `head_q0` | 6.1035 | 183.5155 | `cg_1e-06` | 0.4008 | 142.5111 | 0.7766× |
| Reflective wave2D | 256 | `nested_q32` | 2.8017 | 2520.1766 | `cg_1e-06` | 0.4008 | 142.5111 | 0.0565× |
| Reflective wave2D | 256 | `linear_bank64` | 2.8115 | 1.4277 | `cg_1e-06` | 0.4008 | 142.5111 | 99.8166× |
| Reflective wave2D | 1024 | `head_q0` | 6.1066 | 192.4624 | `cgdt_0.005_tol_0.01` | 5.1413 | 583.0259 | 3.0293× |
| Reflective wave2D | 1024 | `nested_q32` | 2.8094 | 2529.4450 | `cg_1e-06` | 0.4010 | 1043.3681 | 0.4125× |
| Reflective wave2D | 1024 | `linear_bank64` | 2.8192 | 4.5624 | `cg_1e-06` | 0.4010 | 1043.3681 | 228.6899× |
| Heat2D | 64 | `nmrom_cholesky` | 4.5593 | 11.2268 | `fom_cg_cn_tol1e2` | 1.5860 | 2.5038 | 0.2230× |
| Heat2D | 64 | `linear_weak_exact` | 1.6758 | 0.1083 | `fom_cg_cn_tol1e2` | 1.5860 | 2.5038 | 23.1173× |
| Heat2D | 256 | `nmrom_cholesky` | 4.5557 | 11.3414 | `fom_cg_cn_tol1e2` | 0.9255 | 6.9389 | 0.6118× |
| Heat2D | 256 | `linear_weak_exact` | 1.6758 | 0.1382 | `fom_cg_cn_tol1e2` | 0.9255 | 6.9389 | 50.2269× |
| Heat2D | 1024 | `nmrom_cholesky` | 4.5555 | 12.2062 | `fom_cg_cn_tol1e2` | 0.7707 | 59.1780 | 4.8482× |
| Heat2D | 1024 | `linear_weak_exact` | 1.6758 | 0.8381 | `fom_cg_cn_tol1e2` | 0.7707 | 59.1780 | 70.6127× |

**Definitions and qualifications**

- Poisson2D: Worst current-relative L2 versus the same-grid reference, steady state. Current bank/head development experiment. Unpreconditioned CG on the full finite-difference grid; constant-diagonal Jacobi is only a scalar rescaling. The direct linear endpoint is a separate method. [Source JSON](../worktrees/2026-09-17-p-linear/experiments/p-linear/reports/summary.json).
- Reflective wave2D: Worst current-relative displacement L2. Velocity and energy are separate metrics. Development. CG solves the full-grid implicit-midpoint system. Candidates include the retained tolerance/time-step ladder and must pass the full-state gate. These NM-ROM configurations still fail their full-state accuracy gate. [Source JSON](../worktrees/2026-09-17-w-ladder/experiments/w-ladder/reports/summary.json).
- Heat2D: Worst current-relative physical-reference L2 over all cases and all requested times. Earlier audited frozen-checkpoint experiment, measured separately from the newest direct-FOM table. CG and NM-ROM use matching Crank–Nicolson steps. All displayed CG solves converged; the nonlinear model passed its stopping checks. [Source JSON](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/iterative_cg09/archive/outputs/results.json).

Heat timing outlier counts are retained in the CSV/JSON. The Poisson/wave source summaries do not provide outlier counts; the original repetition records remain upstream. No count is inferred from a median.

The Heat3D and Poisson3D CG columns require new same-allocation measurements with the frozen NM-ROM and operators. Their direct-solver timings cannot be replaced by CG timings from another job. Burgers already uses iterative Newton–BiCGStab. For Navier–Stokes, plain CG does not apply to the full nonsymmetric nonlinear system; the existing solver families remain explicitly identified.

**Glossary**

- CG: conjugate gradients, an iterative method for symmetric positive-definite linear systems. A tolerance sets its residual stopping criterion; it is not the same as field error.
- NM-ROM: nonlinear-manifold reduced-order model. Head: the compressed nonlinear decoder. Bank: learned spatial functions; its unrestricted linear endpoint has no nonlinear latent constraint.
- q / K: correction rank / nonlinear latent dimension. The method identifiers name the actual stored configurations.
- Relative L2: field-difference norm divided by the stated reference norm. Worst error is the maximum over the evaluated cases and specified times. Current-relative divides by the current reference field norm.
- Same-grid reference: a converged solution of the discrete equation. Physical reference: the separately verified finer/continuum reference. Displacement is one part of the wave state; velocity error is separate.
- Crank–Nicolson / implicit midpoint: second-order time discretizations that require a linear solve for these linear PDEs.
- GPU ms: median milliseconds for the resident-device query. Speedup: named CG median divided by method median, within the same job.
- Development: cases available during configuration selection, rather than untouched final cases. A passing gate is a recorded accuracy or solver check; it does not imply global optimality.
- Source checksum: fingerprint of the input JSON used to generate the table. Outlier: a retained timing repetition classified by the upstream audit. Newton–BiCGStab: nonlinear Newton iterations with an iterative solver for their nonsymmetric linear systems.
