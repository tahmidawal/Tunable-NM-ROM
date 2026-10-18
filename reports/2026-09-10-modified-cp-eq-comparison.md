# Modified CP with empirical quadrature: Burgers and waves

This report compares the original CP decoder, latent-modulated CP factors, and a FiLM coordinate decoder. Completed single-seed pilot; accuracy and speed claims are limited to the declared families and cohorts.

None of the tested decoder configurations established a validation-selected, evaluation-qualified operating point at the declared accuracy targets. This pilot therefore establishes no ROM speedup over a full solver at matched target accuracy. Diagnostic timings remain useful for understanding cost, with their measured errors shown alongside them.

| Case | Intervals/axis | CP | Modified CP | FiLM | Full solver (5% selection) |
| --- | --- | --- | --- | --- | --- |
| burgers2d | 256 | 45.29% / 29.46 ms | 34.58% / 55.77 ms | 9.839% / 108.8 ms | Newton–BiCGStab: 3.561% / 27.93 ms |
| burgers2d | 512 | 45.27% / 29.58 ms | 34.55% / 56.7 ms | 12.11% / 161.1 ms | Newton–BiCGStab: 3.203% / 43.46 ms |
| wave_reflective | 256 | 209% / 413.3 ms | 230.1% / 642 ms | 3.296e+04% / 1026 ms | CG: 2.118% / 109.1 ms |
| wave_reflective | 512 | 208.5% / 413.1 ms | 204.5% / 835.9 ms | 1.173e+07% / 1992 ms | CG: 4.056% / 184.5 ms |
| wave_absorbing | 256 | 28.77% / 357.5 ms | 61.54% / 1656 ms | 4724% / 1593 ms | CG: 1.407% / 134.6 ms |
| wave_absorbing | 512 | 21.31% / 359.1 ms | 63% / 1624 ms | 1959% / 1598 ms | CG: 1.447% / 271.5 ms |

Each cell gives worst trajectory error / median complete-query time on the untouched cohort. Worst error includes all cases, stored times, repetitions, and the largest physical component for waves, using fixed initial reference scales. Decoder columns use the best-error diagnostic setting frozen during validation; the full-solver column uses its validation-selected setting for the largest declared target. These errors differ, so the table is a cost-and-error comparison, not a matched-accuracy speedup. Finite numerical completion and weak stationarity do not establish physical accuracy or stability. The detailed tables also include the direct, spectral, and explicit wave controls.

## Frozen validation settings

| Case | Intervals/axis | Method | Selection role | Configuration | Median case error | Worst error | Median query ms | Failed cases | Nonstationary cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burgers2d | 256 | cp | best-error diagnostic | cp_cap10_dt0.005_q4_tol0.0001 | 11.3794% | 56.4571% | 42.2665 | 0 | 0 |
| burgers2d | 256 | modcp | best-error diagnostic | modcp_cap30_dt0.005_q4_tol0.0001 | 7.08915% | 50.3952% | 63.2725 | 0 | 0 |
| burgers2d | 256 | film | best-error diagnostic | film_cap10_dt0.005_q4_tol0.0001 | 4.42208% | 23.2942% | 116.999 | 0 | 0 |
| burgers2d | 256 | newton_bicgstab | 5% target | newton_bicgstab_dt0.005_ltol0.1_ntol0.0001 | 1.11828% | 2.82875% | 31.1702 | 0 | — |
| burgers2d | 512 | cp | best-error diagnostic | cp_cap10_dt0.005_q4_tol0.0001 | 11.4134% | 56.4477% | 38.9502 | 0 | 0 |
| burgers2d | 512 | modcp | best-error diagnostic | modcp_cap30_dt0.005_q4_tol0.0001 | 7.29469% | 50.3541% | 65.4887 | 0 | 2 |
| burgers2d | 512 | film | best-error diagnostic | film_cap10_dt0.005_q4_tol0.0001 | 5.31831% | 23.2758% | 171.912 | 0 | 0 |
| burgers2d | 512 | newton_bicgstab | 1% target | newton_bicgstab_dt0.00125_ltol0.1_ntol0.0001 | 0.236118% | 0.979637% | 105.46 | 0 | — |
| burgers2d | 512 | newton_bicgstab | 5% target | newton_bicgstab_dt0.005_ltol0.1_ntol0.0001 | 1.01474% | 2.4852% | 47.8512 | 0 | — |
| wave_reflective | 256 | cg | 1% target | cg_tol1e-06_dt0.0025 | 0.345611% | 0.492034% | 112.823 | 0 | — |
| wave_reflective | 256 | cg | 5% target | cg_tol1e-06_dt0.005 | 1.4076% | 1.99855% | 116.755 | 0 | — |
| wave_reflective | 256 | cg | best-error diagnostic | cg_tol1e-06_dt0.00125 | 0.0895051% | 0.130182% | 223.661 | 0 | — |
| wave_reflective | 256 | cn_direct | 1% target | cn_direct_dt0.0025 | 0.427052% | 0.642471% | 79.0639 | 0 | — |
| wave_reflective | 256 | cn_direct | 5% target | cn_direct_dt0.005 | 1.53236% | 2.07292% | 40.904 | 0 | — |
| wave_reflective | 256 | cn_direct | best-error diagnostic | cn_direct_dt0.00125 | 0.114382% | 0.189008% | 154.843 | 0 | — |
| wave_reflective | 256 | cp | best-error diagnostic | cp_eq8_cap10_tol1e-06_dt0.005 | 54.548% | 149.353% | 438.878 | 0 | 0 |
| wave_reflective | 256 | film | best-error diagnostic | film_eq4_cap10_tol0.0001_dt0.005 | 264.517% | 5492.6% | 1030.56 | 0 | 3 |
| wave_reflective | 256 | modcp | best-error diagnostic | modcp_eq4_cap10_tol0.0001_dt0.005 | 51.2244% | 203.214% | 661.841 | 0 | 0 |
| wave_reflective | 256 | rk4 | 1% target | rk4_cfl0.6 | 0.00173957% | 0.00548189% | 19.9187 | 0 | — |
| wave_reflective | 256 | rk4 | 5% target | rk4_cfl0.6 | 0.00173957% | 0.00548189% | 19.9187 | 0 | — |
| wave_reflective | 256 | rk4 | best-error diagnostic | rk4_cfl0.12 | 3.31264e-06% | 1.13588e-05% | 89.5232 | 0 | — |
| wave_reflective | 256 | spectral | 1% target | spectral_exact_semidiscrete | 2.42878e-12% | 2.69494e-12% | 3.85803 | 0 | — |
| wave_reflective | 256 | spectral | 5% target | spectral_exact_semidiscrete | 2.42878e-12% | 2.69494e-12% | 3.85803 | 0 | — |
| wave_reflective | 256 | spectral | best-error diagnostic | spectral_exact_semidiscrete | 2.42878e-12% | 2.69494e-12% | 3.85803 | 0 | — |
| wave_reflective | 512 | cg | 1% target | cg_tol1e-06_dt0.0025 | 0.324275% | 0.451669% | 240.913 | 0 | — |
| wave_reflective | 512 | cg | 5% target | cg_tol0.0001_dt0.005 | 2.70656% | 4.07971% | 192.402 | 0 | — |
| wave_reflective | 512 | cg | best-error diagnostic | cg_tol1e-06_dt0.00125 | 0.0900604% | 0.131966% | 271.418 | 0 | — |
| wave_reflective | 512 | cn_direct | 1% target | cn_direct_dt0.0025 | 0.428908% | 0.646223% | 118.114 | 0 | — |
| wave_reflective | 512 | cn_direct | 5% target | cn_direct_dt0.005 | 1.53511% | 2.07655% | 60.9755 | 0 | — |
| wave_reflective | 512 | cn_direct | best-error diagnostic | cn_direct_dt0.00125 | 0.115201% | 0.191794% | 232.618 | 0 | — |
| wave_reflective | 512 | cp | best-error diagnostic | cp_eq4_cap30_tol1e-06_dt0.005 | 55.1062% | 149.972% | 440.087 | 0 | 0 |
| wave_reflective | 512 | film | best-error diagnostic | film_eq8_cap10_tol1e-06_dt0.005 | 163.707% | 13224.1% | 2043.84 | 0 | 4 |
| wave_reflective | 512 | modcp | best-error diagnostic | modcp_eq4_cap30_tol1e-06_dt0.005 | 50.4701% | 156.559% | 859.745 | 0 | 0 |
| wave_reflective | 512 | rk4 | 1% target | rk4_cfl0.6 | 0.000155215% | 0.000529328% | 91.2215 | 0 | — |
| wave_reflective | 512 | rk4 | 5% target | rk4_cfl0.6 | 0.000155215% | 0.000529328% | 91.2215 | 0 | — |
| wave_reflective | 512 | rk4 | best-error diagnostic | rk4_cfl0.12 | 2.71717e-07% | 9.51927e-07% | 437.66 | 0 | — |
| wave_reflective | 512 | spectral | 1% target | spectral_exact_semidiscrete | 4.63612e-12% | 5.04003e-12% | 5.56623 | 0 | — |
| wave_reflective | 512 | spectral | 5% target | spectral_exact_semidiscrete | 4.63612e-12% | 5.04003e-12% | 5.56623 | 0 | — |
| wave_reflective | 512 | spectral | best-error diagnostic | spectral_exact_semidiscrete | 4.63612e-12% | 5.04003e-12% | 5.56623 | 0 | — |
| wave_absorbing | 256 | cg | 1% target | cg_tol1e-06_dt0.005 | 0.278138% | 0.418369% | 190.416 | 0 | — |
| wave_absorbing | 256 | cg | 5% target | cg_tol0.0001_dt0.005 | 0.87584% | 1.38656% | 147.748 | 0 | — |
| wave_absorbing | 256 | cg | best-error diagnostic | cg_tol1e-08_dt0.00125 | 0.0174146% | 0.036865% | 381.612 | 0 | — |
| wave_absorbing | 256 | cp | best-error diagnostic | cp_eq8_cap30_tol0.0001_dt0.005 | 15.5855% | 23.8452% | 370.722 | 0 | 1 |
| wave_absorbing | 256 | film | best-error diagnostic | film_eq8_cap10_tol0.0001_dt0.005 | 123.292% | 639.774% | 1560.59 | 0 | 2 |
| wave_absorbing | 256 | modcp | best-error diagnostic | modcp_eq8_cap10_tol1e-06_dt0.005 | 11.5137% | 41.2187% | 1730.78 | 0 | 0 |
| wave_absorbing | 256 | rk4 | 1% target | rk4_cfl0.6 | 0.000468192% | 0.00139524% | 75.3403 | 0 | — |
| wave_absorbing | 256 | rk4 | 5% target | rk4_cfl0.6 | 0.000468192% | 0.00139524% | 75.3403 | 0 | — |
| wave_absorbing | 256 | rk4 | best-error diagnostic | rk4_cfl0.12 | 7.29609e-07% | 2.41785e-06% | 361.045 | 0 | — |
| wave_absorbing | 512 | cg | 1% target | cg_tol1e-06_dt0.00125 | 0.10126% | 0.131512% | 383.901 | 0 | — |
| wave_absorbing | 512 | cg | 5% target | cg_tol0.0001_dt0.005 | 0.918639% | 1.20216% | 285.021 | 0 | — |
| wave_absorbing | 512 | cg | best-error diagnostic | cg_tol1e-08_dt0.00125 | 0.0179195% | 0.0382139% | 566.177 | 0 | — |
| wave_absorbing | 512 | cp | best-error diagnostic | cp_eq8_cap30_tol0.0001_dt0.005 | 15.625% | 23.7572% | 372.586 | 0 | 1 |
| wave_absorbing | 512 | film | best-error diagnostic | film_eq4_cap10_tol0.0001_dt0.005 | 90.0567% | 328.668% | 1666.06 | 0 | 2 |
| wave_absorbing | 512 | modcp | best-error diagnostic | modcp_eq8_cap10_tol1e-06_dt0.005 | 11.5008% | 44.4693% | 1706.86 | 0 | 0 |
| wave_absorbing | 512 | rk4 | 1% target | rk4_cfl0.6 | 3.8372e-05% | 0.000119694% | 212.96 | 0 | — |
| wave_absorbing | 512 | rk4 | 5% target | rk4_cfl0.6 | 3.8372e-05% | 0.000119694% | 212.96 | 0 | — |
| wave_absorbing | 512 | rk4 | best-error diagnostic | rk4_cfl0.12 | 5.72626e-08% | 1.92226e-07% | 1037.08 | 0 | — |

These are the frozen settings carried forward from validation. Best-error diagnostics did not necessarily meet either accuracy target. Errors and times above use the complete validation cohort, with one paired measured invocation per case and setting; timing-based selection used the separately retained seven repetitions of the predetermined proxy case. Evaluation repeats every frozen setting on every untouched case. Rows for different targets may reuse one configuration. These validation times are not combined with evaluation times from another allocation.


## Evaluation of validation-selected configurations

Queries start with full GPU-resident initial fields and return full GPU-resident output trajectories. Timing includes initialization, evolution, and reconstruction. Compilation, offline setup, and host transfers are excluded.

| Case | Intervals/axis | Method | Target | Configuration | Median query ms | Median case error | Worst error | Outlier cases | Failed cases | Nonstationary cases | Timing outliers | Target attained vs numerical reference | Reference interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burgers2d | 256 | cp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (16/16 flagged) |
| burgers2d | 256 | cp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (7/16 flagged) |
| burgers2d | 256 | modcp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (16/16 flagged) |
| burgers2d | 256 | modcp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (7/16 flagged) |
| burgers2d | 256 | film | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (16/16 flagged) |
| burgers2d | 256 | film | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (7/16 flagged) |
| burgers2d | 256 | newton_bicgstab | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (16/16 flagged) |
| burgers2d | 512 | cp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (16/16 flagged) |
| burgers2d | 512 | cp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (5/16 flagged) |
| burgers2d | 512 | modcp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (16/16 flagged) |
| burgers2d | 512 | modcp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (5/16 flagged) |
| burgers2d | 512 | film | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (16/16 flagged) |
| burgers2d | 512 | film | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | provisional continuum (5/16 flagged) |
| wave_reflective | 256 | cp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 256 | cp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 256 | film | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 256 | film | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 256 | modcp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 256 | modcp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 512 | cp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 512 | cp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 512 | film | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 512 | film | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 512 | modcp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_reflective | 512 | modcp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 256 | cp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 256 | cp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 256 | film | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 256 | film | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 256 | modcp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 256 | modcp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 512 | cp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 512 | cp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 512 | film | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 512 | film | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 512 | modcp | 1% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| wave_absorbing | 512 | modcp | 5% | No validation-qualified setting | — | — | — | — | — | — | — | no | semidiscrete only |
| burgers2d | 256 | cp | diagnostic | cp_cap10_dt0.005_q4_tol0.0001 | 29.4558 | 13.7722% | 45.2893% | 15 | 0 | 1 | 0 | no | diagnostic |
| burgers2d | 256 | modcp | diagnostic | modcp_cap30_dt0.005_q4_tol0.0001 | 55.774 | 6.72179% | 34.5799% | 12 | 0 | 2 | 0 | no | diagnostic |
| burgers2d | 256 | film | diagnostic | film_cap10_dt0.005_q4_tol0.0001 | 108.789 | 5.51731% | 9.83892% | 9 | 0 | 2 | 0 | no | diagnostic |
| burgers2d | 256 | newton_bicgstab | 5% | newton_bicgstab_dt0.005_ltol0.1_ntol0.0001 | 27.9348 | 1.21461% | 3.56112% | 0 | 0 | — | 0 | yes | provisional continuum (10/16 flagged) |
| burgers2d | 512 | cp | diagnostic | cp_cap10_dt0.005_q4_tol0.0001 | 29.5804 | 14.0145% | 45.2739% | 15 | 0 | 0 | 0 | no | diagnostic |
| burgers2d | 512 | modcp | diagnostic | modcp_cap30_dt0.005_q4_tol0.0001 | 56.6995 | 7.59602% | 34.5494% | 13 | 0 | 2 | 0 | no | diagnostic |
| burgers2d | 512 | film | diagnostic | film_cap10_dt0.005_q4_tol0.0001 | 161.124 | 6.42367% | 12.1124% | 9 | 0 | 2 | 0 | no | diagnostic |
| burgers2d | 512 | newton_bicgstab | 5% | newton_bicgstab_dt0.005_ltol0.1_ntol0.0001 | 43.4578 | 0.985373% | 3.20343% | 0 | 0 | — | 0 | yes | provisional continuum (7/16 flagged) |
| burgers2d | 512 | newton_bicgstab | 1% | newton_bicgstab_dt0.00125_ltol0.1_ntol0.0001 | 100.214 | 0.30707% | 1.32707% | 1 | 0 | — | 0 | no | provisional continuum (16/16 flagged) |
| wave_reflective | 256 | cp | diagnostic | cp_eq8_cap10_tol1e-06_dt0.005 | 413.326 | 46.5836% | 208.966% | 16 | 0 | 0 | 0 | no | diagnostic |
| wave_reflective | 256 | modcp | diagnostic | modcp_eq4_cap10_tol0.0001_dt0.005 | 641.983 | 43.0144% | 230.11% | 16 | 0 | 0 | 0 | no | diagnostic |
| wave_reflective | 256 | film | diagnostic | film_eq4_cap10_tol0.0001_dt0.005 | 1025.9 | 155.061% | 32956.3% | 16 | 0 | 4 | 0 | no | diagnostic |
| wave_reflective | 256 | cg | 5% | cg_tol1e-06_dt0.005 | 109.062 | 1.21176% | 2.11762% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 256 | cg | 1% | cg_tol1e-06_dt0.0025 | 114.496 | 0.304069% | 0.496033% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 256 | cg | diagnostic | cg_tol1e-06_dt0.00125 | 226.624 | 0.0769731% | 0.126725% | 0 | 0 | — | 0 | no | diagnostic |
| wave_reflective | 256 | rk4 | 1% | rk4_cfl0.6 | 20.772 | 0.00138608% | 0.00384732% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 256 | rk4 | 5% | rk4_cfl0.6 | 20.772 | 0.00138608% | 0.00384732% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 256 | rk4 | diagnostic | rk4_cfl0.12 | 91.3218 | 2.5675e-06% | 7.59101e-06% | 0 | 0 | — | 0 | no | diagnostic |
| wave_reflective | 256 | spectral | 1% | spectral_exact_semidiscrete | 4.41521 | 2.51344e-12% | 2.73149e-12% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 256 | spectral | 5% | spectral_exact_semidiscrete | 4.41521 | 2.51344e-12% | 2.73149e-12% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 256 | spectral | diagnostic | spectral_exact_semidiscrete | 4.41521 | 2.51344e-12% | 2.73149e-12% | 0 | 0 | — | 0 | no | diagnostic |
| wave_reflective | 256 | cn_direct | 5% | cn_direct_dt0.005 | 42.099 | 1.34438% | 2.23648% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 256 | cn_direct | 1% | cn_direct_dt0.0025 | 81.0727 | 0.364054% | 0.565493% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 256 | cn_direct | diagnostic | cn_direct_dt0.00125 | 158.248 | 0.100307% | 0.149331% | 0 | 0 | — | 0 | no | diagnostic |
| wave_reflective | 512 | cp | diagnostic | cp_eq4_cap30_tol1e-06_dt0.005 | 413.117 | 46.7885% | 208.466% | 16 | 0 | 0 | 0 | no | diagnostic |
| wave_reflective | 512 | modcp | diagnostic | modcp_eq4_cap30_tol1e-06_dt0.005 | 835.934 | 42.4925% | 204.541% | 16 | 0 | 0 | 0 | no | diagnostic |
| wave_reflective | 512 | film | diagnostic | film_eq8_cap10_tol1e-06_dt0.005 | 1991.53 | 287.895% | 1.17277e+07% | 16 | 0 | 6 | 0 | no | diagnostic |
| wave_reflective | 512 | cg | 5% | cg_tol0.0001_dt0.005 | 184.549 | 2.55509% | 4.05641% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 512 | cg | 1% | cg_tol1e-06_dt0.0025 | 212.822 | 0.292315% | 0.506292% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 512 | cg | diagnostic | cg_tol1e-06_dt0.00125 | 272.809 | 0.0773389% | 0.140878% | 0 | 0 | — | 0 | no | diagnostic |
| wave_reflective | 512 | rk4 | 1% | rk4_cfl0.6 | 91.8363 | 0.000121411% | 0.000355485% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 512 | rk4 | 5% | rk4_cfl0.6 | 91.8363 | 0.000121411% | 0.000355485% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 512 | rk4 | diagnostic | rk4_cfl0.12 | 439.121 | 2.11075e-07% | 6.29009e-07% | 0 | 0 | — | 0 | no | diagnostic |
| wave_reflective | 512 | spectral | 1% | spectral_exact_semidiscrete | 6.24066 | 4.62102e-12% | 5.07857e-12% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 512 | spectral | 5% | spectral_exact_semidiscrete | 6.24066 | 4.62102e-12% | 5.07857e-12% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 512 | spectral | diagnostic | spectral_exact_semidiscrete | 6.24066 | 4.62102e-12% | 5.07857e-12% | 0 | 0 | — | 0 | no | diagnostic |
| wave_reflective | 512 | cn_direct | 5% | cn_direct_dt0.005 | 62.2681 | 1.3458% | 2.23756% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 512 | cn_direct | 1% | cn_direct_dt0.0025 | 119.773 | 0.365718% | 0.565897% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_reflective | 512 | cn_direct | diagnostic | cn_direct_dt0.00125 | 235.113 | 0.102335% | 0.150942% | 0 | 0 | — | 0 | no | diagnostic |
| wave_absorbing | 256 | cp | diagnostic | cp_eq8_cap30_tol0.0001_dt0.005 | 357.515 | 15.8756% | 28.7683% | 16 | 0 | 0 | 0 | no | diagnostic |
| wave_absorbing | 256 | modcp | diagnostic | modcp_eq8_cap10_tol1e-06_dt0.005 | 1655.82 | 11.4881% | 61.5366% | 16 | 0 | 0 | 0 | no | diagnostic |
| wave_absorbing | 256 | film | diagnostic | film_eq8_cap10_tol0.0001_dt0.005 | 1593.05 | 117.266% | 4724.05% | 16 | 0 | 6 | 0 | no | diagnostic |
| wave_absorbing | 256 | cg | 5% | cg_tol0.0001_dt0.005 | 134.603 | 0.93919% | 1.40683% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_absorbing | 256 | cg | 1% | cg_tol1e-06_dt0.005 | 174.282 | 0.260804% | 0.356975% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_absorbing | 256 | cg | diagnostic | cg_tol1e-08_dt0.00125 | 357.758 | 0.0165701% | 0.0253553% | 0 | 0 | — | 0 | no | diagnostic |
| wave_absorbing | 256 | rk4 | 1% | rk4_cfl0.6 | 74.8719 | 0.000399281% | 0.000974151% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_absorbing | 256 | rk4 | 5% | rk4_cfl0.6 | 74.8719 | 0.000399281% | 0.000974151% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_absorbing | 256 | rk4 | diagnostic | rk4_cfl0.12 | 359.741 | 5.93062e-07% | 1.59695e-06% | 0 | 0 | — | 0 | no | diagnostic |
| wave_absorbing | 512 | cp | diagnostic | cp_eq8_cap30_tol0.0001_dt0.005 | 359.125 | 15.2465% | 21.3109% | 16 | 0 | 0 | 0 | no | diagnostic |
| wave_absorbing | 512 | modcp | diagnostic | modcp_eq8_cap10_tol1e-06_dt0.005 | 1623.64 | 11.5001% | 63% | 16 | 0 | 0 | 0 | no | diagnostic |
| wave_absorbing | 512 | film | diagnostic | film_eq4_cap10_tol0.0001_dt0.005 | 1597.97 | 120.127% | 1959.48% | 16 | 0 | 5 | 0 | no | diagnostic |
| wave_absorbing | 512 | cg | 5% | cg_tol0.0001_dt0.005 | 271.532 | 0.939099% | 1.44656% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_absorbing | 512 | cg | 1% | cg_tol1e-06_dt0.00125 | 351.906 | 0.106894% | 0.138252% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_absorbing | 512 | cg | diagnostic | cg_tol1e-08_dt0.00125 | 545.469 | 0.0170969% | 0.0280348% | 0 | 0 | — | 0 | no | diagnostic |
| wave_absorbing | 512 | rk4 | 1% | rk4_cfl0.6 | 212.688 | 3.19937e-05% | 8.1633e-05% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_absorbing | 512 | rk4 | 5% | rk4_cfl0.6 | 212.688 | 3.19937e-05% | 8.1633e-05% | 0 | 0 | — | 0 | yes | semidiscrete only |
| wave_absorbing | 512 | rk4 | diagnostic | rk4_cfl0.12 | 1036.36 | 4.58004e-08% | 1.26646e-07% | 0 | 0 | — | 0 | no | diagnostic |

Worst error is the maximum over evaluation cases, stored times, and recorded repetitions; for waves it is also the maximum over displacement, velocity, and energy-state errors. All expected cases and repetitions must be present for a target to qualify. Failure counts retain numerical breakdowns and incomplete trajectories. Diagnostic settings still count physical outliers against the largest declared target, even though those settings cannot establish a target-qualified speedup. Iteration-capped or small-step exits may attain a physical accuracy target, but are separately counted as nonstationary and never described as converged PDE solves. Stationarity concerns the weak least-squares objective; physical accuracy and full-residual diagnostics are assessed separately.

These errors compare against the declared numerical reference. Any unresolved reference uncertainty keeps the corresponding continuum-accuracy interpretation provisional. Reference flags use the evaluation cohort for evaluated settings and the validation cohort for settings that never qualified for evaluation. They do not change the frozen numerical-reference qualification or configuration selection. No configuration is chosen using evaluation accuracy or timing.

![Validation and evaluation error versus query time](2026-09-10-modified-cp-eq-comparison-frontiers.png)

Configurations with nonfinite errors have no finite position on the logarithmic axes; their failures remain in the summary tables and raw records.

## Initial fitting and subsequent evolution

| Case | Intervals/axis | Method | Configuration | Worst initial error | Worst final error | Worst trajectory error |
| --- | --- | --- | --- | --- | --- | --- |
| burgers2d | 256 | cp | cp_cap10_dt0.005_q4_tol0.0001 | 45.2893% | 12.1189% | 45.2893% |
| burgers2d | 256 | modcp | modcp_cap30_dt0.005_q4_tol0.0001 | 34.5799% | 9.92624% | 34.5799% |
| burgers2d | 256 | film | film_cap10_dt0.005_q4_tol0.0001 | 9.78651% | 5.78702% | 9.83892% |
| burgers2d | 256 | newton_bicgstab | newton_bicgstab_dt0.005_ltol0.1_ntol0.0001 | 0% | 2.44539% | 3.56112% |
| burgers2d | 512 | cp | cp_cap10_dt0.005_q4_tol0.0001 | 45.2739% | 13.0943% | 45.2739% |
| burgers2d | 512 | modcp | modcp_cap30_dt0.005_q4_tol0.0001 | 34.5494% | 9.92003% | 34.5494% |
| burgers2d | 512 | film | film_cap10_dt0.005_q4_tol0.0001 | 9.76452% | 7.91317% | 12.1124% |
| burgers2d | 512 | newton_bicgstab | newton_bicgstab_dt0.005_ltol0.1_ntol0.0001 | 0% | 1.86119% | 3.20343% |
| burgers2d | 512 | newton_bicgstab | newton_bicgstab_dt0.00125_ltol0.1_ntol0.0001 | 0% | 1.03702% | 1.32707% |
| wave_reflective | 256 | cp | cp_eq8_cap10_tol1e-06_dt0.005 | 23.1219% | 208.966% | 208.966% |
| wave_reflective | 256 | modcp | modcp_eq4_cap10_tol0.0001_dt0.005 | 18.1353% | 210.17% | 230.11% |
| wave_reflective | 256 | film | film_eq4_cap10_tol0.0001_dt0.005 | 23.0667% | 32956.3% | 32956.3% |
| wave_reflective | 256 | cg | cg_tol1e-06_dt0.005 | 0% | 2.11762% | 2.11762% |
| wave_reflective | 256 | cg | cg_tol1e-06_dt0.0025 | 0% | 0.496033% | 0.496033% |
| wave_reflective | 256 | cg | cg_tol1e-06_dt0.00125 | 0% | 0.126725% | 0.126725% |
| wave_reflective | 256 | rk4 | rk4_cfl0.6 | 0% | 0.00384732% | 0.00384732% |
| wave_reflective | 256 | rk4 | rk4_cfl0.12 | 0% | 7.59101e-06% | 7.59101e-06% |
| wave_reflective | 256 | spectral | spectral_exact_semidiscrete | 2.20354e-12% | 2.4956e-12% | 2.73149e-12% |
| wave_reflective | 256 | cn_direct | cn_direct_dt0.005 | 0% | 2.23648% | 2.23648% |
| wave_reflective | 256 | cn_direct | cn_direct_dt0.0025 | 0% | 0.565493% | 0.565493% |
| wave_reflective | 256 | cn_direct | cn_direct_dt0.00125 | 0% | 0.149331% | 0.149331% |
| wave_reflective | 512 | cp | cp_eq4_cap30_tol1e-06_dt0.005 | 23.1406% | 208.466% | 208.466% |
| wave_reflective | 512 | modcp | modcp_eq4_cap30_tol1e-06_dt0.005 | 18.2295% | 188.114% | 204.541% |
| wave_reflective | 512 | film | film_eq8_cap10_tol1e-06_dt0.005 | 23.0847% | 1.17277e+07% | 1.17277e+07% |
| wave_reflective | 512 | cg | cg_tol0.0001_dt0.005 | 0% | 4.05641% | 4.05641% |
| wave_reflective | 512 | cg | cg_tol1e-06_dt0.0025 | 0% | 0.506292% | 0.506292% |
| wave_reflective | 512 | cg | cg_tol1e-06_dt0.00125 | 0% | 0.140878% | 0.140878% |
| wave_reflective | 512 | rk4 | rk4_cfl0.6 | 0% | 0.000355485% | 0.000355485% |
| wave_reflective | 512 | rk4 | rk4_cfl0.12 | 0% | 6.29009e-07% | 6.29009e-07% |
| wave_reflective | 512 | spectral | spectral_exact_semidiscrete | 3.76026e-12% | 4.78021e-12% | 5.07857e-12% |
| wave_reflective | 512 | cn_direct | cn_direct_dt0.005 | 0% | 2.23756% | 2.23756% |
| wave_reflective | 512 | cn_direct | cn_direct_dt0.0025 | 0% | 0.565897% | 0.565897% |
| wave_reflective | 512 | cn_direct | cn_direct_dt0.00125 | 0% | 0.150942% | 0.150942% |
| wave_absorbing | 256 | cp | cp_eq8_cap30_tol0.0001_dt0.005 | 19.0925% | 24.6805% | 28.7683% |
| wave_absorbing | 256 | modcp | modcp_eq8_cap10_tol1e-06_dt0.005 | 10.2143% | 60.7605% | 61.5366% |
| wave_absorbing | 256 | film | film_eq8_cap10_tol0.0001_dt0.005 | 10.1752% | 4724.05% | 4724.05% |
| wave_absorbing | 256 | cg | cg_tol0.0001_dt0.005 | 0% | 0.982223% | 1.40683% |
| wave_absorbing | 256 | cg | cg_tol1e-06_dt0.005 | 0% | 0.0172414% | 0.356975% |
| wave_absorbing | 256 | cg | cg_tol1e-08_dt0.00125 | 0% | 0.00173537% | 0.0253553% |
| wave_absorbing | 256 | rk4 | rk4_cfl0.6 | 0% | 0.000128929% | 0.000974151% |
| wave_absorbing | 256 | rk4 | rk4_cfl0.12 | 0% | 5.23194e-07% | 1.59695e-06% |
| wave_absorbing | 512 | cp | cp_eq8_cap30_tol0.0001_dt0.005 | 19.1187% | 18.7373% | 21.3109% |
| wave_absorbing | 512 | modcp | modcp_eq8_cap10_tol1e-06_dt0.005 | 10.3545% | 60.528% | 63% |
| wave_absorbing | 512 | film | film_eq4_cap10_tol0.0001_dt0.005 | 10.1802% | 1959.48% | 1959.48% |
| wave_absorbing | 512 | cg | cg_tol0.0001_dt0.005 | 0% | 0.925562% | 1.44656% |
| wave_absorbing | 512 | cg | cg_tol1e-06_dt0.00125 | 0% | 0.107369% | 0.138252% |
| wave_absorbing | 512 | cg | cg_tol1e-08_dt0.00125 | 0% | 0.00117961% | 0.0280348% |
| wave_absorbing | 512 | rk4 | rk4_cfl0.6 | 0% | 3.34252e-06% | 8.1633e-05% |
| wave_absorbing | 512 | rk4 | rk4_cfl0.12 | 0% | 6.5709e-09% | 1.26646e-07% |

Each column takes its own maximum over the complete evaluation cohort and physical components, so the maximizing case may differ between columns. Every time uses the same initial-reference normalization. These are measured field discrepancies; local snapshot-fitting diagnostics do not establish a mathematical best-approximation floor.

| Wave case | Intervals/axis | Method | Configuration | Worst displacement error | Worst velocity error | Worst energy-state error |
| --- | --- | --- | --- | --- | --- | --- |
| wave_reflective | 256 | cp | cp_eq8_cap10_tol1e-06_dt0.005 | 113.845% | 154.106% | 208.966% |
| wave_reflective | 256 | modcp | modcp_eq4_cap10_tol0.0001_dt0.005 | 72.3932% | 143.005% | 230.11% |
| wave_reflective | 256 | film | film_eq4_cap10_tol0.0001_dt0.005 | 7787.81% | 18067.4% | 32956.3% |
| wave_reflective | 256 | cg | cg_tol1e-06_dt0.005 | 1.0031% | 1.65636% | 2.11762% |
| wave_reflective | 256 | cg | cg_tol1e-06_dt0.0025 | 0.283839% | 0.394598% | 0.496033% |
| wave_reflective | 256 | cg | cg_tol1e-06_dt0.00125 | 0.0875244% | 0.102295% | 0.126725% |
| wave_reflective | 256 | rk4 | rk4_cfl0.6 | 0.000121436% | 0.00273252% | 0.00384732% |
| wave_reflective | 256 | rk4 | rk4_cfl0.12 | 2.22548e-07% | 5.33234e-06% | 7.59101e-06% |
| wave_reflective | 256 | spectral | spectral_exact_semidiscrete | 4.61789e-14% | 1.84108e-12% | 2.73149e-12% |
| wave_reflective | 256 | cn_direct | cn_direct_dt0.005 | 0.976265% | 1.73283% | 2.23648% |
| wave_reflective | 256 | cn_direct | cn_direct_dt0.0025 | 0.244156% | 0.437408% | 0.565493% |
| wave_reflective | 256 | cn_direct | cn_direct_dt0.00125 | 0.0610494% | 0.110088% | 0.149331% |
| wave_reflective | 512 | cp | cp_eq4_cap30_tol1e-06_dt0.005 | 103.213% | 163.395% | 208.466% |
| wave_reflective | 512 | modcp | modcp_eq4_cap30_tol1e-06_dt0.005 | 85.248% | 146.732% | 204.541% |
| wave_reflective | 512 | film | film_eq8_cap10_tol1e-06_dt0.005 | 2.20564e+06% | 3.34646e+06% | 1.17277e+07% |
| wave_reflective | 512 | cg | cg_tol0.0001_dt0.005 | 1.98688% | 3.2059% | 4.05641% |
| wave_reflective | 512 | cg | cg_tol1e-06_dt0.0025 | 0.270894% | 0.400681% | 0.506292% |
| wave_reflective | 512 | cg | cg_tol1e-06_dt0.00125 | 0.0792876% | 0.11365% | 0.140878% |
| wave_reflective | 512 | rk4 | rk4_cfl0.6 | 9.14002e-06% | 0.000248231% | 0.000355485% |
| wave_reflective | 512 | rk4 | rk4_cfl0.12 | 1.58733e-08% | 4.32659e-07% | 6.29009e-07% |
| wave_reflective | 512 | spectral | spectral_exact_semidiscrete | 5.0462e-14% | 3.03021e-12% | 5.07857e-12% |
| wave_reflective | 512 | cn_direct | cn_direct_dt0.005 | 0.976181% | 1.73372% | 2.23756% |
| wave_reflective | 512 | cn_direct | cn_direct_dt0.0025 | 0.244144% | 0.43833% | 0.565897% |
| wave_reflective | 512 | cn_direct | cn_direct_dt0.00125 | 0.061045% | 0.110389% | 0.150942% |
| wave_absorbing | 256 | cp | cp_eq8_cap30_tol0.0001_dt0.005 | 21.7903% | 22.0943% | 28.7683% |
| wave_absorbing | 256 | modcp | modcp_eq8_cap10_tol1e-06_dt0.005 | 34.9739% | 44.1067% | 61.5366% |
| wave_absorbing | 256 | film | film_eq8_cap10_tol0.0001_dt0.005 | 4485.64% | 2977.03% | 4724.05% |
| wave_absorbing | 256 | cg | cg_tol0.0001_dt0.005 | 1.05789% | 1.05005% | 1.40683% |
| wave_absorbing | 256 | cg | cg_tol1e-06_dt0.005 | 0.130666% | 0.251983% | 0.356975% |
| wave_absorbing | 256 | cg | cg_tol1e-08_dt0.00125 | 0.00881605% | 0.0178979% | 0.0253553% |
| wave_absorbing | 256 | rk4 | rk4_cfl0.6 | 2.81216e-05% | 0.000687801% | 0.000974151% |
| wave_absorbing | 256 | rk4 | rk4_cfl0.12 | 4.41601e-08% | 1.12541e-06% | 1.59695e-06% |
| wave_absorbing | 512 | cp | cp_eq8_cap30_tol0.0001_dt0.005 | 14.1222% | 15.8343% | 21.3109% |
| wave_absorbing | 512 | modcp | modcp_eq8_cap10_tol1e-06_dt0.005 | 34.5959% | 44.7123% | 63% |
| wave_absorbing | 512 | film | film_eq4_cap10_tol0.0001_dt0.005 | 867.047% | 1155.4% | 1959.48% |
| wave_absorbing | 512 | cg | cg_tol0.0001_dt0.005 | 0.640789% | 1.32485% | 1.44656% |
| wave_absorbing | 512 | cg | cg_tol1e-06_dt0.00125 | 0.138252% | 0.0989558% | 0.128509% |
| wave_absorbing | 512 | cg | cg_tol1e-08_dt0.00125 | 0.00836595% | 0.0197935% | 0.0280348% |
| wave_absorbing | 512 | rk4 | rk4_cfl0.6 | 1.96705e-06% | 5.76752e-05% | 8.1633e-05% |
| wave_absorbing | 512 | rk4 | rk4_cfl0.12 | 3.02506e-09% | 8.94635e-08% | 1.26646e-07% |

| Wave case | Intervals/axis | Method | Configuration | Energy discrepancy / reference initial energy | Reflective energy drift / reference initial energy | Absorbing invariant error | Absorbing invariant drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wave_reflective | 256 | cp | cp_eq8_cap10_tol1e-06_dt0.005 | 2.93724 | 2.92736 | — | — |
| wave_reflective | 256 | modcp | modcp_eq4_cap10_tol0.0001_dt0.005 | 4.05875 | 4.04319 | — | — |
| wave_reflective | 256 | film | film_eq4_cap10_tol0.0001_dt0.005 | 108615 | 108615 | — | — |
| wave_reflective | 256 | cg | cg_tol1e-06_dt0.005 | 0.000139489 | 0.000139489 | — | — |
| wave_reflective | 256 | cg | cg_tol1e-06_dt0.0025 | 8.52322e-05 | 8.52322e-05 | — | — |
| wave_reflective | 256 | cg | cg_tol1e-06_dt0.00125 | 2.12614e-05 | 2.12614e-05 | — | — |
| wave_reflective | 256 | rk4 | rk4_cfl0.6 | 2.69198e-08 | 2.69198e-08 | — | — |
| wave_reflective | 256 | rk4 | rk4_cfl0.12 | 9.59491e-12 | 9.5951e-12 | — | — |
| wave_reflective | 256 | spectral | spectral_exact_semidiscrete | 6.69049e-16 | 4.46033e-16 | — | — |
| wave_reflective | 256 | cn_direct | cn_direct_dt0.005 | 2.76552e-13 | 2.76552e-13 | — | — |
| wave_reflective | 256 | cn_direct | cn_direct_dt0.0025 | 1.2766e-12 | 1.2766e-12 | — | — |
| wave_reflective | 256 | cn_direct | cn_direct_dt0.00125 | 5.11293e-12 | 5.11278e-12 | — | — |
| wave_reflective | 512 | cp | cp_eq4_cap30_tol1e-06_dt0.005 | 2.73614 | 2.72657 | — | — |
| wave_reflective | 512 | modcp | modcp_eq4_cap30_tol1e-06_dt0.005 | 3.06092 | 3.04483 | — | — |
| wave_reflective | 512 | film | film_eq8_cap10_tol1e-06_dt0.005 | 1.37539e+10 | 1.37539e+10 | — | — |
| wave_reflective | 512 | cg | cg_tol0.0001_dt0.005 | 0.00276013 | 0.00276013 | — | — |
| wave_reflective | 512 | cg | cg_tol1e-06_dt0.0025 | 5.5277e-05 | 5.5277e-05 | — | — |
| wave_reflective | 512 | cg | cg_tol1e-06_dt0.00125 | 2.13949e-05 | 2.13949e-05 | — | — |
| wave_reflective | 512 | rk4 | rk4_cfl0.6 | 9.23478e-10 | 9.23478e-10 | — | — |
| wave_reflective | 512 | rk4 | rk4_cfl0.12 | 3.21823e-13 | 3.21823e-13 | — | — |
| wave_reflective | 512 | spectral | spectral_exact_semidiscrete | 7.08134e-16 | 5.00191e-16 | — | — |
| wave_reflective | 512 | cn_direct | cn_direct_dt0.005 | 3.6012e-13 | 3.59963e-13 | — | — |
| wave_reflective | 512 | cn_direct | cn_direct_dt0.0025 | 1.65282e-12 | 1.65271e-12 | — | — |
| wave_reflective | 512 | cn_direct | cn_direct_dt0.00125 | 5.43363e-12 | 5.43347e-12 | — | — |
| wave_absorbing | 256 | cp | cp_eq8_cap30_tol0.0001_dt0.005 | 0.0827462 | — | 0.0928989 | 0.0894562 |
| wave_absorbing | 256 | modcp | modcp_eq8_cap10_tol1e-06_dt0.005 | 0.375533 | — | 0.397775 | 0.397869 |
| wave_absorbing | 256 | film | film_eq8_cap10_tol0.0001_dt0.005 | 2231.69 | — | 64.8609 | 64.8634 |
| wave_absorbing | 256 | cg | cg_tol0.0001_dt0.005 | 0.004806 | — | 0.00708658 | 0.00708658 |
| wave_absorbing | 256 | cg | cg_tol1e-06_dt0.005 | 0.00206145 | — | 0.000338449 | 0.000338449 |
| wave_absorbing | 256 | cg | cg_tol1e-08_dt0.00125 | 0.000111914 | — | 2.07231e-05 | 2.07231e-05 |
| wave_absorbing | 256 | rk4 | rk4_cfl0.6 | 7.03221e-08 | — | 2.22045e-16 | 1.52656e-16 |
| wave_absorbing | 256 | rk4 | rk4_cfl0.12 | 1.14118e-10 | — | 2.22045e-16 | 1.52656e-16 |
| wave_absorbing | 512 | cp | cp_eq8_cap30_tol0.0001_dt0.005 | 0.0566191 | — | 0.0802108 | 0.0904051 |
| wave_absorbing | 512 | modcp | modcp_eq8_cap10_tol1e-06_dt0.005 | 0.393802 | — | 0.422537 | 0.422655 |
| wave_absorbing | 512 | film | film_eq4_cap10_tol0.0001_dt0.005 | 383.955 | — | 15.2596 | 15.2622 |
| wave_absorbing | 512 | cg | cg_tol0.0001_dt0.005 | 0.00357037 | — | 0.00541529 | 0.00541529 |
| wave_absorbing | 512 | cg | cg_tol1e-06_dt0.00125 | 0.000371278 | — | 0.00118354 | 0.00118354 |
| wave_absorbing | 512 | cg | cg_tol1e-08_dt0.00125 | 0.000117647 | — | 1.41673e-05 | 1.41673e-05 |
| wave_absorbing | 512 | rk4 | rk4_cfl0.6 | 4.45338e-09 | — | 2.22045e-16 | 1.66533e-16 |
| wave_absorbing | 512 | rk4 | rk4_cfl0.12 | 7.1528e-12 | — | 2.22045e-16 | 1.66533e-16 |

These independently reconstructed diagnostics distinguish energy discrepancies from the energy norm of state error. Energy drift measures change from the prediction's own initial energy; energy discrepancy includes its initial mismatch. The absorbing signed invariant is $I(u,v)=\int_\Omega v\,dx+c\int_{\partial\Omega}u\,ds$, using the discrete area and edge weights with both corner contributions. Invariant error and drift are absolute quantities, not percentages. The table takes the largest absolute defect across the complete cohort, stored times, and repetitions. Close energies alone do not prove that the remaining state discrepancy is a phase error; no boundary-flux accuracy claim is inferred from coarse observation times.

## Matched-accuracy full-solver comparisons

No paired evaluation configurations currently qualify at a common declared target.

Each ratio uses the same owner job and GPU and two validation-selected configurations that both attain the target on the untouched cohort. A ratio above unity means a smaller median ROM query time. Numerical completion and latent convergence remain separate.

## Trained models and online work

| Case | Decoder | Training intervals/axis | Latent dimension | CP rank | Decoder parameters | Completed training updates |
| --- | --- | --- | --- | --- | --- | --- |
| burgers2d | cp | 256 | 16 | 64 | 120577 | 16000 |
| burgers2d | modcp | 256 | 16 | 64 | 135553 | 16000 |
| burgers2d | film | 256 | 16 | — | 30386 | 16000 |
| wave_reflective | cp | 256 | 32 | 64 | 177154 | 16000 |
| wave_reflective | modcp | 256 | 32 | 64 | 204546 | 16000 |
| wave_reflective | film | 256 | 32 | — | 40388 | 16000 |
| wave_absorbing | cp | 256 | 32 | 64 | 177154 | 16000 |
| wave_absorbing | modcp | 256 | 32 | 64 | 204546 | 16000 |
| wave_absorbing | film | 256 | 32 | — | 40388 | 16000 |

Each PDE/boundary has separately trained weights, frozen for both evaluation meshes. Training jointly optimizes decoder weights and a latent code for each training snapshot; this pilot does not reproduce the older ViT encoder training pipeline. CP and modified CP share the same initial CP training stage. FiLM uses the full update budget from its own initialization. Training update budgets match; parameter counts and training costs differ. This pilot compares the declared architectures without a parameter-matched or exhaustive tuning claim.

For each output component, the CP family represents

$$\widetilde u(z;x,y)=m(x,y)\left[\beta+\sum_{r=1}^{R}c_r(z)a_r(x;z)b_r(y;z)\right].$$

Here $z$ is the solved latent state, $R$ is the CP rank, $c_r$ is the nonlinear coefficient head with a linear skip, and $m$ enforces the boundary. Original CP uses factors independent of $z$. Modified CP adds a latent-conditioned nonlinear correction to each one-dimensional factor; its zero correction exactly recovers the shared CP initialization. FiLM instead conditions a two-dimensional coordinate network on the latent state and includes its own linear latent skip. Coordinate-only features are cached offline for all three decoders. Dirichlet mesh transfer uses a boundary strip that preserves training-node values and avoids interpolating untrained masked endpoint parameters into new interior nodes.

Each reduced implicit step minimizes the scaled weak discrete residual,

$$z_{n+1}\approx\operatorname*{arg\,min}_z\frac12\left\|W_{\mathrm{EQ}}\,\mathcal R_n(\widetilde u(z);\widetilde u(z_n),\mu)\right\|_2^2.$$

$\mathcal R_n$ is the fully discrete PDE residual, $\mu$ contains physical parameters, and $W_{\mathrm{EQ}}$ applies the smooth test functions, fitted quadrature, and fixed scaling. The weak residual and its latent Jacobian use JAX automatic differentiation; damped Gauss–Newton solves small dense systems and warm-starts each time step from the preceding latent state. The initial latent fit uses the supplied field at sampled locations and stored starting codes. The full-solver CG comparison belongs to the SPD wave discretization; nonlinear Burgers uses Newton–BiCGStab.

CP precontracts its fixed spatial factors with the selected EQ weights for weak linear terms. The quadrature approximation is preserved. Burgers still evaluates the nonlinear upwind term on its sampled stencil. Modified CP and FiLM retain state-dependent spatial evaluation. On waves, CP's contracted evolution dimensions stay fixed when EQ count changes; EQ count still affects approximation and sampled initialization. Solver iteration limits, tolerance, and time step continue to change work. Full-field output cost grows with the requested mesh for every decoder. This implementation uses EQ-fitted operators, and does not establish an exact quadrature-free operator claim.

Burgers validation preceded the equivalent CP mass contraction and GPU scalar preloading. Its final evaluation keeps those validation-selected settings and uses the optimized runner, whose numerical parity was checked separately. The final timing ratios come entirely from that evaluation allocation. The older validation timings do not establish the fastest configuration for the optimized implementation.

## Frozen quadrature rules

| Case | Intervals/axis | Decoder | Test modes/component | Requested volume nodes | Stored volume nodes | Positive volume weights | Stored boundary entries | Volume fit rows | Relative volume fit defect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burgers2d | 256 | cp | 64 | 256 | 256 | 256 | 0 | 4097 | 0.0227011 |
| burgers2d | 256 | cp | 64 | 512 | 512 | 512 | 0 | 4097 | 0.0046808 |
| burgers2d | 512 | cp | 64 | 256 | 256 | 256 | 0 | 4097 | 0.0196647 |
| burgers2d | 512 | cp | 64 | 512 | 512 | 512 | 0 | 4097 | 0.00435434 |
| burgers2d | 256 | film | 64 | 256 | 256 | 256 | 0 | 4097 | 0.00743065 |
| burgers2d | 256 | film | 64 | 512 | 512 | 512 | 0 | 4097 | 0.000624868 |
| burgers2d | 512 | film | 64 | 256 | 256 | 256 | 0 | 4097 | 0.00760549 |
| burgers2d | 512 | film | 64 | 512 | 512 | 512 | 0 | 4097 | 0.000591494 |
| burgers2d | 256 | modcp | 64 | 256 | 256 | 256 | 0 | 4097 | 0.0125679 |
| burgers2d | 256 | modcp | 64 | 512 | 512 | 512 | 0 | 4097 | 0.00273991 |
| burgers2d | 512 | modcp | 64 | 256 | 256 | 256 | 0 | 4097 | 0.0123842 |
| burgers2d | 512 | modcp | 64 | 512 | 512 | 512 | 0 | 4097 | 0.0024364 |
| wave_reflective | 256 | cp | 128 | 512 | 512 | 512 | 0 | 8193 | 0.0013806 |
| wave_reflective | 256 | cp | 128 | 1024 | 1024 | 1024 | 0 | 8193 | 0.00070251 |
| wave_reflective | 256 | modcp | 128 | 512 | 512 | 512 | 0 | 8193 | 0.00134954 |
| wave_reflective | 256 | modcp | 128 | 1024 | 1024 | 1024 | 0 | 8193 | 0.000678396 |
| wave_reflective | 256 | film | 128 | 512 | 512 | 512 | 0 | 8193 | 0.00102815 |
| wave_reflective | 256 | film | 128 | 1024 | 1024 | 1024 | 0 | 8193 | 2.31713e-05 |
| wave_reflective | 512 | cp | 128 | 512 | 512 | 512 | 0 | 8193 | 0.00114255 |
| wave_reflective | 512 | cp | 128 | 1024 | 1024 | 1024 | 0 | 8193 | 0.000512249 |
| wave_reflective | 512 | modcp | 128 | 512 | 512 | 512 | 0 | 8193 | 0.00121078 |
| wave_reflective | 512 | modcp | 128 | 1024 | 1024 | 1024 | 0 | 8193 | 0.000531042 |
| wave_reflective | 512 | film | 128 | 512 | 512 | 512 | 0 | 8193 | 0.00095046 |
| wave_reflective | 512 | film | 128 | 1024 | 1024 | 1024 | 0 | 8193 | 2.6365e-05 |
| wave_absorbing | 256 | cp | 128 | 512 | 512 | 512 | 192 | 8193 | 0.00309395 |
| wave_absorbing | 256 | cp | 128 | 1024 | 1024 | 1024 | 384 | 8193 | 0.0012016 |
| wave_absorbing | 256 | modcp | 128 | 512 | 512 | 512 | 192 | 8193 | 0.0030623 |
| wave_absorbing | 256 | modcp | 128 | 1024 | 1024 | 1024 | 384 | 8193 | 0.0010959 |
| wave_absorbing | 256 | film | 128 | 512 | 512 | 512 | 192 | 8193 | 0.00239486 |
| wave_absorbing | 256 | film | 128 | 1024 | 1024 | 1024 | 384 | 8193 | 2.00712e-05 |
| wave_absorbing | 512 | cp | 128 | 512 | 512 | 512 | 192 | 8193 | 0.00270325 |
| wave_absorbing | 512 | cp | 128 | 1024 | 1024 | 1024 | 384 | 8193 | 0.00091571 |
| wave_absorbing | 512 | modcp | 128 | 512 | 512 | 512 | 192 | 8193 | 0.00265974 |
| wave_absorbing | 512 | modcp | 128 | 1024 | 1024 | 1024 | 384 | 8193 | 0.000895284 |
| wave_absorbing | 512 | film | 128 | 512 | 512 | 512 | 192 | 8193 | 0.00193615 |
| wave_absorbing | 512 | film | 128 | 1024 | 1024 | 1024 | 384 | 8193 | 1.97224e-05 |

Quadrature is fitted offline using decoded training snapshots. Each mesh and test space has its own frozen rule; evaluation imports the exact weights used for validation. Requested nodes, stored support, positive weights, and fitting-system rows are different quantities. Boundary entries count separate physical faces, including both corner contributions; they are not a count of unique decoder coordinates. Burgers also evaluates the neighbors required by its exact sign-upwind stencil. The offline fit defect is not an unseen-case quadrature error bound.

## Diagnosis of the Burgers validation failure

| Validation case | Decoder | Intervals/axis | Online initial error | Best tested initial fit | Best tested final-snapshot fit | Online final error | CP affine-image initial floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | cp | 256 | 56.4571% | 56.4571% | 2.55919% | 4.54492% | 38.0449% |
| 3 | modcp | 256 | 50.3952% | 50.3952% | 1.79197% | 5.39246% | not established |
| 3 | film | 256 | 23.2942% | 23.2942% | 1.32608% | 3.80169% | not established |
| 9 | film | 256 | 2.77472% | 2.77472% | 6.34099% | 10.2763% | not established |

For validation case 3, independent full-grid least squares over the 64 learned CP spatial products gives an initial error floor of 38.0449%. Its factor matrix has condition number 45.5976. This limits this frozen checkpoint even with freely chosen spatial coefficients; changing only quadrature or nonlinear iteration settings cannot overcome it. The same conclusion is not established for modified CP or FiLM. Their tested stationary snapshot fits are achieved reconstruction errors, which can exceed the unknown global minimum. Later-time snapshot fitting uses the reference solution and is excluded from online timings, initialization, and configuration selection. These cases were chosen to diagnose validation failures.

![Independent CP reconstruction diagnostic](2026-09-10-modified-cp-span-audit.png)

## Wave snapshot and weak-residual diagnostics

| Wave case | Intervals/axis | Validation case | Diagnostic configuration | Initial snapshot fit error | Worst sampled later fit error | Final snapshot fit error | Minimum tangent singular-value ratio | Nonstationary fits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wave_reflective | 256 | 0 | cp_eq8_cap30_tol1e-06_dt0.00125 | 15.6493% | 13.5464% | 13.5464% | 0.0955905 | 0 |
| wave_reflective | 256 | 0 | modcp_eq8_cap30_tol1e-06_dt0.00125 | 12.3569% | 10.4414% | 10.4414% | 0.0688921 | 0 |
| wave_reflective | 256 | 0 | film_eq8_cap30_tol1e-06_dt0.00125 | 15.3746% | 21.1337% | 17.3075% | 0.0385205 | 0 |
| wave_reflective | 512 | 0 | cp_eq8_cap30_tol1e-06_dt0.00125 | 15.6778% | 13.5568% | 13.5568% | 0.0955881 | 0 |
| wave_reflective | 512 | 0 | modcp_eq8_cap30_tol1e-06_dt0.00125 | 12.4794% | 10.5094% | 10.5094% | 0.0688884 | 0 |
| wave_reflective | 512 | 0 | film_eq8_cap30_tol1e-06_dt0.00125 | 15.361% | 21.0744% | 17.2924% | 0.0385225 | 0 |
| wave_absorbing | 256 | 0 | cp_eq8_cap30_tol1e-06_dt0.00125 | 12.3351% | 6.74973% | 1.09668% | 0.0181893 | 0 |
| wave_absorbing | 256 | 0 | modcp_eq8_cap30_tol1e-06_dt0.00125 | 7.60346% | 3.71782% | 0.729309% | 0.0112007 | 0 |
| wave_absorbing | 256 | 0 | film_eq8_cap30_tol1e-06_dt0.00125 | 4.9836% | 2.82826% | 0.576205% | 0.0107422 | 0 |
| wave_absorbing | 512 | 0 | cp_eq8_cap30_tol1e-06_dt0.00125 | 12.3746% | 6.74524% | 1.09626% | 0.0181859 | 0 |
| wave_absorbing | 512 | 0 | modcp_eq8_cap30_tol1e-06_dt0.00125 | 7.72078% | 3.73628% | 0.731197% | 0.011199 | 0 |
| wave_absorbing | 512 | 0 | film_eq8_cap30_tol1e-06_dt0.00125 | 4.98583% | 2.82633% | 0.576629% | 0.0107422 | 0 |

Each snapshot is fitted separately to a known reference state. Errors take the largest physical component on the same fixed initial normalization as the rollout. These local achieved errors are not global reconstruction floors. The tangent ratio is the smallest singular value divided by the largest for the sampled weak decoder map; it does not certify global conditioning.

| Wave case | Intervals/axis | Validation case | Diagnostic configuration | Maximum full weak residual norm | Maximum EQ weak residual norm | Maximum full-minus-EQ residual norm | Maximum mass-moment discrepancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wave_reflective | 256 | 0 | cp_eq8_cap30_tol1e-06_dt0.00125 | 0.00243639 | 0.0024377 | 6.21818e-05 | 0.000489412 |
| wave_reflective | 256 | 0 | modcp_eq8_cap30_tol1e-06_dt0.00125 | 0.00244265 | 0.00244217 | 5.51201e-05 | 0.000485319 |
| wave_reflective | 256 | 0 | film_eq8_cap30_tol1e-06_dt0.00125 | 0.188231 | 0.189266 | 0.00490535 | 0.0395269 |
| wave_reflective | 512 | 0 | cp_eq8_cap30_tol1e-06_dt0.00125 | 0.00244201 | 0.00244533 | 4.94927e-05 | 0.000385623 |
| wave_reflective | 512 | 0 | modcp_eq8_cap30_tol1e-06_dt0.00125 | 0.00242498 | 0.002424 | 5.06875e-05 | 0.000385899 |
| wave_reflective | 512 | 0 | film_eq8_cap30_tol1e-06_dt0.00125 | 0.567305 | 0.56714 | 0.0222526 | 0.183683 |
| wave_absorbing | 256 | 0 | cp_eq8_cap30_tol1e-06_dt0.00125 | 0.000744374 | 0.000744608 | 6.51071e-05 | 0.000714106 |
| wave_absorbing | 256 | 0 | modcp_eq8_cap30_tol1e-06_dt0.00125 | 0.000542544 | 0.00054075 | 4.60081e-05 | 0.000532669 |
| wave_absorbing | 256 | 0 | film_eq8_cap30_tol1e-06_dt0.00125 | 0.0135835 | 0.0135684 | 6.87826e-05 | 0.000750154 |
| wave_absorbing | 512 | 0 | cp_eq8_cap30_tol1e-06_dt0.00125 | 0.0007448 | 0.000746131 | 5.28468e-05 | 0.000549278 |
| wave_absorbing | 512 | 0 | modcp_eq8_cap30_tol1e-06_dt0.00125 | 0.000543151 | 0.000544143 | 4.12718e-05 | 0.000443165 |
| wave_absorbing | 512 | 0 | film_eq8_cap30_tol1e-06_dt0.00125 | 0.00573532 | 0.00573629 | 1.25644e-05 | 0.000196506 |

These checks use actual consecutive integration states from the matching validation rollout. Each column takes its own maximum over the audited times; the maxima need not occur together. Residual and moment values use the stored fixed weak scaling, not the physical error norm. The diagnostics cover the predetermined first validation case, use owner-computed full-grid quantities from verified source, and are excluded from online cost and model selection. They cannot certify quadrature accuracy or explain all failures elsewhere in the cohort.

## Provenance and independent review

| Case | Campaign status | Job ID | GPU | Source commit | Full-field audits |
| --- | --- | --- | --- | --- | --- |
| burgers2d | validation_frozen | 3500795 | NVIDIA A100-PCIE-40GB | b229e8521fa94d7229bbd745290beebbb7c89ae6 | 2592 |
| wave_reflective | validation_frozen | 3503439 | ['NVIDIA A100 80GB PCIe'] | df737ba306f1edac21054939b9241f4f3f221320 | 0 |
| wave_absorbing | validation_frozen | 3503457 | ['NVIDIA A100-PCIE-40GB'] | df737ba306f1edac21054939b9241f4f3f221320 | 0 |
| burgers2d | complete | 3518411 | NVIDIA A100-PCIE-40GB | 890a67b5ef313001d546735f8b32101f75849910 | 144 |
| wave_reflective | complete | 3518421 | ['NVIDIA A100 80GB PCIe'] | 7a571324213c7c761c7f6e3de19937a167fbd729 | 384 |
| wave_absorbing | complete | 3518429 | ['NVIDIA A100-PCIE-40GB'] | 7a571324213c7c761c7f6e3de19937a167fbd729 | 256 |

Raw repetition records, validation sweeps, selection declarations, source hashes, and field-audit results are indexed in [2026-09-10-modified-cp-eq-comparison.json](2026-09-10-modified-cp-eq-comparison.json). Timing ratios must use the same job and GPU, and an FOM configuration meeting the same accuracy target. This report does not substitute timings from separate jobs. Every evaluation panel is bound to the same saved global validation freeze, including its unchanged cohort seed, checkpoint identities, selected settings, and quadrature file hashes. Reference arrays must agree across every method and repetition for each case.

The [source audit](2026-09-10-modified-cp-source-audit.json) compares collected code with immutable Git objects. The [raw archive manifest](2026-09-10-modified-cp-raw-artifacts.json) identifies retained field archives and explains checksum verification and extraction. Large raw fields are retained outside Git history.

Wave validation retains full-grid metrics and output hashes but only bounded observation fields. Its audit checks source, reference operators, selection records, and saved observations; the unsaved full-grid validation errors cannot be independently recomputed from those observations. Every finite final evaluation invocation is instead checked against full-grid fields, with independently recomputed errors and matching output hashes.

## Numerical reference checks

| Case | Cohort | Intervals/axis | Reference | Worst temporal difference | Worst nested space/time difference | Largest per-case empirical indicator | Continuum target interpretation | Worst energy balance defect | Worst invariant drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burgers2d | validation | 256 | nested_finer_reference | 0.00273575 | 0.0158586 | 0.0176588 | 1%: provisional continuum (16/16 flagged); 5%: provisional continuum (7/16 flagged) | — | — |
| burgers2d | validation | 512 | nested_finer_reference | 0.0029296 | 0.00874686 | 0.0106544 | 1%: provisional continuum (16/16 flagged); 5%: provisional continuum (5/16 flagged) | — | — |
| wave_reflective | validation | 256 | exact_semidiscrete_sine | — | — | — | semidiscrete only | 1.07479e-15 | — |
| wave_reflective | validation | 512 | exact_semidiscrete_sine | — | — | — | semidiscrete only | 7.78349e-16 | — |
| wave_absorbing | validation | 256 | fresh_RK4_with_balance | 3.56623e-08 | — | — | semidiscrete only | 3.2316e-11 | 1.25741e-16 |
| wave_absorbing | validation | 512 | fresh_RK4_with_balance | 2.97842e-09 | — | — | semidiscrete only | 2.12567e-12 | 1.05601e-16 |
| burgers2d | evaluation | 256 | nested_finer_reference | 0.00360827 | 0.0194247 | 0.0230329 | 1%: provisional continuum (16/16 flagged); 5%: provisional continuum (10/16 flagged) | — | — |
| burgers2d | evaluation | 512 | nested_finer_reference | 0.00387735 | 0.0109849 | 0.0148622 | 1%: provisional continuum (16/16 flagged); 5%: provisional continuum (7/16 flagged) | — | — |
| wave_reflective | evaluation | 256 | exact_semidiscrete_sine | — | — | — | semidiscrete only | 1.02504e-15 | — |
| wave_reflective | evaluation | 512 | exact_semidiscrete_sine | — | — | — | semidiscrete only | 9.51988e-16 | — |
| wave_absorbing | evaluation | 256 | fresh_RK4_with_balance | 3.1318e-08 | — | — | semidiscrete only | 3.60739e-11 | 1.11022e-16 |
| wave_absorbing | evaluation | 512 | fresh_RK4_with_balance | 2.57882e-09 | — | — | semidiscrete only | 2.36684e-12 | 1.02132e-16 |

The scalar Burgers equation is $\partial_t u+u(\partial_xu+\partial_yu)=\nu\Delta u$ on the unit square with homogeneous Dirichlet boundaries and localized Gaussian initial fields. The full solver uses backward Euler, sign-dependent upwinding, and Newton–BiCGStab. The wave system is $\partial_t u=v$, $\partial_t v=c^2\Delta u$, with reflective $u=0$ or absorbing $\partial_t u+c\partial_nu=0$ boundaries and the fresh localized Gaussian-core family. Its implicit comparison uses a symmetric positive-definite Crank–Nicolson elimination solved by CG. Direct and explicit wave controls are reported separately. Exact parameter generators and recorded cohort parameters remain in the source artifacts indexed above.

Burgers is scored against a finer-grid trajectory restricted to the output grid; its nested difference also contains spatial discretization error. Wave errors use the same-grid discrete system: reflective propagation is exact for that system, and absorbing references have temporal refinement and energy/boundary balance checks. These checks do not establish a rigorous continuum error bound. Differences and energy balance defects are dimensionless; invariant drift is an absolute signed-moment magnitude.

The Burgers empirical indicator is each case's nested space/time difference plus its temporal refinement difference; the table reports the maximum of these per-case sums. A case is flagged when its indicator exceeds one tenth of the target. These flags identify reference-sensitive continuum interpretations, not proven error bounds. Missing reference records are labeled unassessed, and an unflagged target still has no certified continuum bound.

The new wave decoder represents displacement and velocity jointly. Its latent dimension is not the phase-state dimension of the earlier displacement-manifold experiments; changes relative to those earlier results do not isolate decoder architecture.

The representative figures share one color scale across methods. Their captions identify any symmetric logarithmic scale used to display large excursions together with the reference without clipping. This display choice does not change the fields or numerical errors.

![burgers2d: reference and decoder fields](2026-09-10-modified-cp-eq-comparison-burgers2d-fields.png)

![wave_reflective: reference and decoder fields](2026-09-10-modified-cp-eq-comparison-wave_reflective-fields.png)

![wave_absorbing: reference and decoder fields](2026-09-10-modified-cp-eq-comparison-wave_absorbing-fields.png)

## Glossary

- **CP:** a sum of products of learned one-dimensional spatial factors.
- **Modified CP:** CP factors with small nonlinear changes conditioned on the solved latent state.
- **FiLM / INR:** feature-wise modulation of a neural coordinate-to-field decoder.
- **EQ:** empirical quadrature, an offline-selected set of spatial samples and nonnegative integration weights.
- **Test modes/component:** smooth spatial functions against which each state component's residual is integrated.
- **Requested / stored / positive volume nodes:** respectively the target sample count, stored quadrature entries, and entries with strictly positive weight.
- **Stored boundary entries:** the sum of samples on separate absorbing faces, counting a shared corner once per face.
- **Volume fit rows / relative fit defect:** the number of offline fitting constraints and their relative residual on the fitted decoder snapshots.
- **FOM:** the full-order numerical PDE solver used as a speed comparison.
- **Weak residual:** the PDE mismatch integrated against smooth spatial test functions.
- **Latent state:** the small vector of unknowns solved inside the decoder.
- **CP rank:** the number of spatial product terms, separate from the latent dimension.
- **Linear skip:** a direct linear dependence on latent coordinates added to the nonlinear decoder head.
- **FiLM conditioning:** latent-dependent scales and shifts applied to hidden coordinate features.
- **Jacobian / automatic differentiation:** respectively derivatives with respect to latent coordinates and their computation by differentiating the implemented numerical operations.
- **Gauss–Newton / damping:** local least-squares iteration using that Jacobian, with regularization and acceptance checks to control the step.
- **CG / SPD:** conjugate gradients and the symmetric positive-definite matrix property required by that solver.
- **Newton–BiCGStab:** nonlinear Newton iteration with a Krylov linear solver that allows nonsymmetric systems.
- **Crank–Nicolson / backward Euler:** implicit time-stepping rules used here for waves and Burgers respectively.
- **Decoder parameters:** trained weights in the field decoder, excluding training-only snapshot codes.
- **Training updates:** optimizer steps completed before validation and evaluation.
- **Affine spatial image:** the fixed CP bias plus every linear combination of its learned spatial products.
- **Reconstruction floor:** the smallest error in the specified fixed linear/affine space, giving a lower bound for a decoder restricted to that space.
- **Snapshot fit:** a local latent optimization against one known reference field; its achieved error is not a proof of the best possible decoder error.
- **Tangent singular-value ratio:** the smallest divided by the largest singular value of the sampled weak decoder Jacobian at a fitted state; a local rank/conditioning diagnostic.
- **Mass-moment discrepancy:** the fixed-scale difference between full-grid and quadrature integration of decoder fields against smooth test functions.
- **Full-minus-EQ weak residual norm:** the norm of the difference between full-grid and quadrature weak residual vectors at the same actual integration step.
- **Intervals/axis:** subdivisions of the unit domain; the number of stored nodes depends on boundary conditions.
- **Validation-selected configuration:** solver and quadrature settings frozen before evaluation fields are examined.
- **Target / target attained:** the declared error ceiling, and whether every expected invocation completes below it.
- **Median query ms:** median across cases of each case's median recorded duration, in milliseconds.
- **Median case error:** median across the declared cases of each case's worst error across times, components, and repetitions; nonfinite errors and incomplete repetition coverage enter as infinite errors rather than being dropped.
- **Worst error:** the largest fixed-initial-normalized error across the reported cases, times, state components, and repetitions.
- **Outlier cases:** cases with any error above the target or invalid error values; diagnostic settings use the largest declared target for this count.
- **Failed cases:** cases with any incomplete/nonfinite solve or missing trajectory.
- **Nonstationary cases:** ROM cases with any latent fit or weak time step lacking the declared gradient condition; accurate capped rollouts remain labeled. Classical methods show a dash because they use their own completion checks.
- **Timing outliers:** invocations taking more than twice their own case's repetition median; retained in all summaries.
- **Temporal difference:** discrepancy after refining the reference time step, on fixed initial physical scales.
- **Nested space/time difference:** discrepancy against a finer spatial grid and time step, restricted back to the reported grid.
- **Empirical indicator / reference flag:** a Burgers case's summed nested space/time and temporal differences, and whether that sum exceeds one tenth of a target; a sensitivity check, not a certified error bound.
- **Reference interpretation:** whether the result is limited to a semidiscrete reference, has flagged continuum sensitivity, or lacks an assessment; separate from attaining the target against the stored numerical reference.
- **Energy balance defect:** relative failure of reference energy conservation, or energy plus outgoing boundary flux conservation.
- **Invariant drift:** change in the absorbing reference's area integral of velocity plus speed times its boundary integral of displacement.
- **Semidiscrete / continuum:** respectively the spatially discretized PDE and the original PDE before spatial discretization.
- **Median-time ratio FOM/ROM:** the full solver's median query duration divided by the ROM's, for paired qualifying configurations.
- **Energy-state error:** the physical energy norm of the displacement/velocity error, scaled by the initial reference energy.
- **Energy discrepancy / reflective drift:** respectively the absolute prediction-minus-reference energy difference and change from the prediction's own start, divided by reference initial energy.
- **Absorbing invariant error / drift:** absolute difference of the conserved area-plus-boundary moment from the reference or from the prediction's own start.
- **Full-field audit:** independent NumPy recomputation from a saved full-grid prediction and reference.
- **Symmetric logarithmic color scale:** a color mapping that is linear near zero and logarithmic for larger positive or negative values.
- **Campaign status / job ID / GPU / source commit:** completion state and identifiers of the recorded scientific execution.
- **Single-seed pilot:** an initial comparison using one training random seed, without a training-variance claim.
