### q-ridge — the evolved-times regression is the empirical-quadrature RULE; certifying it on states the ROM actually reaches removes up to 5.86x of it (at $q=256$) at no extra cost, but leaves 1 violation ($q=128\to256$) because no constructible $m$ certifies the top rungs; neither a ridge on the corrections nor more tests fixes it either

The coordinator first asked whether the Burgers $256^2$ correction ladder's non-monotonicity on the worst-over-evolved-times metric ($q=16$ worse than $q=0$) is the extra unknowns **overfitting the $M$ weak test equations**, with two remedies — a field-metric ridge on the corrections (R1) and more tests at fixed $q$ (R2) — and a held-out-residual control (R3). Mid-flight, after `q-diag` reported that the cause is the quadrature, the lane was **re-scoped**: R1 and R2 were demoted to controls and the primary became **EQ rule certification** — refit the $m$-point rule on states the ROM actually reaches, grow $m$ at fixed $M=4(K+q)$, certify every rule by its held-out $\rho$ rather than by its NNLS fit residual, and rebuild the ladder with the cheapest certified rule per rung. Predeclared protocol and every amendment, including the re-scope as §A3: `experiments/q-ridge/DESIGN.md`. Nothing was merged or pushed.

Worktree `worktrees/2026-09-16-q-ridge`, branch `exp/2026-09-16-q-ridge`, forked from `exp/2026-09-16-b-ladder-top` at `b8efd5b4`. Namespace `/cluster/tufts/paralab/tawal01/q_ridge_20260916/`. R1 job `3757235` (`qrg101`) on `NVIDIA A100 80GB PCIe`, source `5fc6099127d4`, elapsed 5389.4 s; R2 job `3757237` (`qrg201`) on `NVIDIA A100 80GB PCIe`, source `5fc6099127d4`, elapsed 6834.0 s; EQCERT job `3768168` (`qrg304`) on `NVIDIA A100 80GB PCIe`, source `467a4670f179`, elapsed 10502.3 s. All three printed `jax_backend=gpu`, ran float64 at highest matmul precision, and were checksum-collected, independently NumPy-audited and archived before their exact remote attempt directories were removed. Two earlier submissions (3757043, 3757044) died in the sbatch preamble on a placeholder collision, before the GPU preflight and with zero GPU work; they are recorded in `DESIGN.md` §A1 and are not counted against the three-job cap.

**R1 (qrg101) gates.** `artifacts_present` yes; `backend_gpu` yes; `bank_frozen` yes; `bank_sha256_consistent` yes; `checkpoint_unchanged` yes; `complete` yes; `cross_job_fidelity` yes; `decoded_fields_match_saved_outputs` yes; `directions_hash_matches_cclad01` yes; `directions_rank_covers_ladder` yes; `eq_rules_untruncated` yes; `evaluation_cohort_bitwise_abl01` yes; `every_eq_rule_reports_validity` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `r3_mode_blocks_complete` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_fields_bitwise_match_a_source` yes; `reference_residuals` yes; `repetition_output_identical` yes; `reproduces_q0_m4_dense_l0` yes; `reproduces_q0_m4_eq_l0_ret` yes; `reproduces_q16_m4_dense_l0` yes; `reproduces_q256_m2_dense_l0` yes; `reproduces_q64_m4_dense_l0` yes; `same_grid_baseline_present` yes; `step_budget_600` yes; `t0_field_invariant_in_lambda` yes; `x64` yes.

**R2 (qrg201) gates.** `artifacts_present` yes; `backend_gpu` yes; `bank_frozen` yes; `bank_sha256_consistent` yes; `checkpoint_unchanged` yes; `complete` yes; `cross_job_fidelity` yes; `decoded_fields_match_saved_outputs` yes; `directions_hash_matches_cclad01` no; `directions_rank_covers_ladder` yes; `eq_rules_untruncated` yes; `evaluation_cohort_bitwise_abl01` yes; `every_eq_rule_reports_validity` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `r3_mode_blocks_complete` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_fields_bitwise_match_a_source` yes; `reference_residuals` yes; `repetition_output_identical` yes; `reproduces_q0_m4_dense_l0` yes; `reproduces_q0_m4_eq_l0_ret` yes; `reproduces_q16_m4_dense_l0` yes; `reproduces_q64_m4_dense_l0` yes; `same_grid_baseline_present` yes; `step_budget_600` yes; `t0_field_invariant_in_lambda` —; `x64` yes.

**EQCERT (qrg304) gates.** `artifacts_present` yes; `backend_gpu` yes; `bank_frozen` yes; `bank_sha256_consistent` yes; `checkpoint_unchanged` yes; `complete` yes; `cross_job_fidelity` yes; `decoded_fields_match_saved_outputs` yes; `directions_hash_matches_cclad01` no; `evaluation_cohort_bitwise_abl01` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_rule_archived` yes; `every_rule_reports_rho` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `fit_and_certification_trajectories_disjoint` yes; `no_rule_truncated` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `r3_mode_blocks_complete` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_residuals` yes; `repetition_output_identical` yes; `reproduces_q0_m4_eqold_ret` yes; `reproduces_q128_m2_eqold_bnd` yes; `rule_grid_complete` yes; `same_grid_baseline_present` yes; `step_budget_600` yes; `x64` yes.

**Cross-job fidelity.**

| job | arm | source | comparator | tolerance | rel. diff (all-times) | rel. diff (evolved) | passed |
|---|---|---|---|---|---|---|---|
| qrg101 | `q0_m4_dense_l0` | btq101 | `q0_m4_dense_base` | 1e-09 | 9.5e-15 | 2.4e-13 | yes |
| qrg101 | `q0_m4_eq_l0_ret` | btq101 | `q0_m4_eq_base` | 1e-09 | 9.5e-15 | 4.1e-13 | yes |
| qrg101 | `q16_m4_dense_l0` | cclad01 | `q16_m4_dense_block` | 1e-09 | 2.4e-14 | 6.5e-15 | yes |
| qrg101 | `q256_m2_dense_l0` | btq102 | `q256_m2_dense_base_b600` | 1e-03 | 1.2e-13 | 3.6e-13 | yes |
| qrg101 | `q64_m4_dense_l0` | cclad01 | `q64_m4_dense_block` | 1e-09 | 9.5e-15 | 8.9e-14 | yes |
| qrg201 | `q0_m4_dense_l0` | btq101 | `q0_m4_dense_base` | 1e-09 | 0.0e+00 | 2.4e-13 | yes |
| qrg201 | `q0_m4_eq_l0_ret` | btq101 | `q0_m4_eq_base` | 1e-09 | 0.0e+00 | 5.1e-14 | yes |
| qrg201 | `q16_m4_dense_l0` | cclad01 | `q16_m4_dense_block` | 1e-03 | 3.6e-14 | 2.1e-14 | yes |
| qrg201 | `q64_m4_dense_l0` | cclad01 | `q64_m4_dense_block` | 1e-03 | 3.3e-14 | 8.0e-14 | yes |
| qrg304 | `q0_m4_eqold_ret` | btq101 | `q0_m4_eq_base` | 1e-09 | 0.0e+00 | 5.0e-13 | yes |
| qrg304 | `q128_m2_eqold_bnd` | btq101 | `q128_m2_eq_base` | 1e-03 | 2.5e-10 | 1.0e-09 | yes |


**EQ rule certification — every rule, its NNLS fit and its held-out $\rho$** (bar $\rho^\star = 0.116$, declared before the job ran from `q-diag`'s $q=0$ incumbent measurement).

| $q$ | population | $m$ target | $m$ | $m/M$ | NNLS fit | $\rho_{max}$ | $\rho_{95}$ | certified | truncated | fit (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | reachable | 1024 | 1024 | 16.00 | 4.58e-05 | 0.0153 | 0.0088 | yes | no | 71.5 |
| 0 | reachable | 2048 | 1521 | 23.77 | 7.83e-06 | 0.0261 | 0.0083 | yes | no | 116.7 |
| 0 | static | 256 | 256 | 4.00 | 6.85e-03 | 0.2407 | 0.2073 | no | no | 6.0 |
| 16 | reachable | 1024 | 1024 | 8.00 | 1.90e-04 | 0.0935 | 0.0273 | yes | no | 146.5 |
| 16 | reachable | 2048 | 2048 | 16.00 | 1.92e-05 | 0.0452 | 0.0041 | yes | no | 641.6 |
| 16 | static | 512 | 512 | 4.00 | 1.54e-03 | 0.1279 | 0.0523 | no | no | 48.0 |
| 32 | reachable | 1024 | 1024 | 5.33 | 2.89e-04 | 0.0533 | 0.0226 | yes | no | 154.7 |
| 32 | reachable | 2048 | 2048 | 10.67 | 2.66e-05 | 0.0668 | 0.0120 | yes | no | 650.7 |
| 32 | static | 768 | 768 | 4.00 | 6.27e-04 | 0.1482 | 0.0214 | no | no | 99.6 |
| 64 | reachable | 1024 | 1024 | 3.20 | 7.12e-04 | 0.0531 | 0.0230 | yes | no | 159.9 |
| 64 | reachable | 2048 | 2048 | 6.40 | 7.54e-05 | 0.1120 | 0.0172 | yes | no | 656.1 |
| 64 | static | 1280 | 1280 | 4.00 | 2.43e-04 | 0.1439 | 0.0235 | no | no | 275.3 |
| 128 | reachable | 1024 | 1024 | 1.78 | 1.62e-03 | 0.2786 | 0.2246 | no | no | 238.2 |
| 128 | reachable | 2048 | 2048 | 3.56 | 1.04e-04 | 0.1908 | 0.0518 | no | no | 781.1 |
| 128 | static | 2048 | 2048 | 3.56 | 7.16e-05 | 0.2194 | 0.0677 | no | no | 840.8 |
| 256 | reachable | 1024 | 1024 | 0.94 | 4.01e-02 | 0.5731 | 0.5370 | no | no | 100.1 |
| 256 | reachable | 2048 | 2048 | 1.88 | 3.08e-04 | 0.1678 | 0.0957 | no | no | 962.1 |
| 256 | static | 2048 | 2048 | 1.88 | 1.49e-04 | 0.4621 | 0.4275 | no | no | 1485.1 |


**The rule chosen at each rung.**

| $q$ | $M$ | chosen $m$ | basis | $\rho_{max}$ | $\rho_{95}$ |
|---|---|---|---|---|---|
| 0 | 64 | 1024 | primary | 0.0153 | 0.0088 |
| 16 | 128 | 1024 | primary | 0.0935 | 0.0273 |
| 32 | 192 | 1024 | primary | 0.0533 | 0.0226 |
| 64 | 320 | 1024 | primary | 0.0531 | 0.0230 |
| 128 | 576 | 2048 | secondary | 0.1908 | 0.0518 |
| 256 | 1088 | 2048 | secondary | 0.1678 | 0.0957 |


**What the certification buys, rung by rung** (worst evolved-times error, incumbent static-population rule against the cheapest certified reachable-state rule, and the dense twin where this job ran one).

| $q$ | $M$ | old rule $m$ / evolved % | certified $m$ / evolved % | improvement | dense evolved % | cost vs old rule |
|---|---|---|---|---|---|---|
| 0 | 64 | 256 / 3.4166 | 1024 / 1.8891 | 1.81x | 1.8890 | 1.129x |
| 16 | 128 | 512 / 2.1427 | 1024 / 1.4270 | 1.50x | — | 1.106x |
| 32 | 192 | 768 / 1.5026 | 1024 / 1.2493 | 1.20x | — | 1.063x |
| 64 | 320 | 1280 / 1.1454 | 1024 / 1.2275 | 0.93x | 1.0843 | 0.957x |
| 128 | 576 | 2048 / 0.8928 | 2048 / 0.8925 | 1.00x | — | 1.006x |
| 256 | 1088 | 2048 / 6.0676 | 2048 / 1.0361 | 5.86x | 0.5194 | 1.015x |


**The rebuilt ladder, both metrics.**

| ladder | q | worst evolved % | worst all-times % | median GPU ms | monotone | converged | cost <=2x old rule | passes |
|---|---|---|---|---|---|---|---|---|
| EQ, cheapest certified rule per rung | 0, 16, 32, 64, 128, 256 | 1.8891 / 1.4270 / 1.2493 / 1.2275 / 0.8925 / 1.0361 | 2.5629 / 2.4806 / 2.3534 / 2.1489 / 1.8116 / 1.0361 | 57 / 79 / 96 / 119 / 247 / 699 | no | yes | yes | no |
| EQ, incumbent static-population rule | 0, 16, 32, 64, 128, 256 | 3.4166 / 2.1427 / 1.5026 / 1.1454 / 0.8928 / 6.0676 | 3.4166 / 2.4806 / 2.3534 / 2.1489 / 1.8116 / 6.0676 | 51 / 72 / 90 / 124 / 245 / 689 | no | yes | — | — |
| dense (exact) quadrature | 0, 64, 256 | 1.8890 / 1.0843 / 0.5194 | 2.5629 / 2.1489 / 0.9053 | 282 / 620 / 3945 | yes | yes | — | — |


No $m$ in the declared grid reaches the bar at $q=256$; a log-log extrapolation of $\rho_{max}$ against $m$ puts the crossing at $m \approx 2522$ (slope -1.772).


**R1 — the field-metric ridge (demoted to a control).**


*dense quadrature:*

| setting | worst evolved % at q = 0, 16, 64, 256 | worst all-times % | median GPU ms | monotone | converged | cost <=1.5x | all-times not raised | passes |
|---|---|---|---|---|---|---|---|---|
| $\lambda_{rel}=0$ | 1.8890 / 1.3985 / 1.0843 / 0.5194 | 2.5629 / 2.4806 / 2.1489 / 0.9053 | 295 / 368 / 627 / 3912 | yes | yes | yes | yes | yes |
| $\lambda_{rel}=0.0001$ | 1.8890 / 1.3983 / 1.0842 / 0.5127 | 2.5629 / 2.4806 / 2.1489 / 0.9053 | 295 / 369 / 625 / 4021 | yes | yes | yes | yes | yes |
| $\lambda_{rel}=0.001$ | 1.8890 / 1.3966 / 1.0830 / 0.4727 | 2.5629 / 2.4806 / 2.1489 / 0.9053 | 295 / 369 / 627 / 3738 | yes | yes | yes | yes | yes |
| $\lambda_{rel}=0.01$ | 1.8890 / 1.3838 / 1.0745 / 0.4308 | 2.5629 / 2.4806 / 2.1489 / 0.9053 | 295 / 370 / 630 / 3033 | yes | yes | yes | yes | yes |
| $\lambda_{rel}=0.1$ | 1.8890 / 1.3750 / 1.0877 / 0.6251 | 2.5629 / 2.4806 / 2.1489 / 0.9053 | 295 / 369 / 634 / 2862 | yes | yes | yes | yes | yes |
| $\lambda_{rel}=1$ | 1.8890 / 1.4297 / 1.2140 / 1.1011 | 2.5629 / 2.4806 / 2.1489 / 1.1011 | 295 / 367 / 634 / 2658 | yes | yes | yes | no | no |


*eq quadrature:*

| setting | worst evolved % at q = 0, 16, 64, 256 | worst all-times % | median GPU ms | monotone | converged | cost <=1.5x | all-times not raised | passes |
|---|---|---|---|---|---|---|---|---|
| $\lambda_{rel}=0$ | 2.0659 / 2.1811 / 1.1408 / 3.5204 | 2.5629 / 2.4806 / 2.1489 / 3.5204 | 52 / 72 / 127 / 715 | no | yes | yes | yes | no |
| $\lambda_{rel}=0.0001$ | 2.0659 / 2.1809 / 1.1406 / 3.5155 | 2.5629 / 2.4806 / 2.1489 / 3.5155 | 52 / 73 / 127 / 732 | no | yes | yes | yes | no |
| $\lambda_{rel}=0.001$ | 2.0659 / 2.1786 / 1.1396 / 3.4804 | 2.5629 / 2.4806 / 2.1489 / 3.4804 | 52 / 74 / 127 / 669 | no | yes | yes | yes | no |
| $\lambda_{rel}=0.01$ | 2.0659 / 2.1604 / 1.1332 / 3.2834 | 2.5629 / 2.4806 / 2.1489 / 3.2834 | 52 / 74 / 128 / 518 | no | yes | yes | yes | no |
| $\lambda_{rel}=0.1$ | 2.0659 / 2.1247 / 1.1623 / 2.7294 | 2.5629 / 2.4806 / 2.1489 / 2.7294 | 52 / 73 / 132 / 485 | no | no | yes | yes | no |
| $\lambda_{rel}=1$ | 2.0659 / 2.1720 / 1.3026 / 2.5988 | 2.5629 / 2.4806 / 2.1489 / 2.5988 | 52 / 74 / 130 / 452 | no | yes | yes | yes | no |


**R2 — more tests at fixed $q$ (demoted to a control).**


*dense quadrature:*

| setting | worst evolved % at q = 0, 16, 64 | worst all-times % | median GPU ms | monotone | converged | cost <=1.5x | all-times not raised | passes |
|---|---|---|---|---|---|---|---|---|
| $M=16(K+q)$ | 1.2710 / 1.2204 / 1.0588 | 2.5629 / 2.4806 / 2.1489 | 343 / 545 / 1302 | yes | yes | no | yes | no |
| $M=4(K+q)$ | 1.8890 / 1.3985 / 1.0843 | 2.5629 / 2.4806 / 2.1489 | 284 / 358 / 621 | yes | yes | yes | yes | yes |
| $M=8(K+q)$ | 1.4695 / 1.2305 / 1.0619 | 2.5629 / 2.4806 / 2.1489 | 283 / 419 / 896 | yes | yes | yes | yes | yes |


*eq quadrature:*

| setting | worst evolved % at q = 0, 16, 64 | worst all-times % | median GPU ms | monotone | converged | cost <=1.5x | all-times not raised | passes |
|---|---|---|---|---|---|---|---|---|
| $M=16(K+q)$ | 1.3186 / 1.2202 / 5.2481 | 2.5629 / 2.4806 / 5.2481 | 58 / 109 / 184 | no | yes | no | no | no |
| $M=4(K+q)$ | 2.0659 / 2.1811 / 1.1408 | 2.5629 / 2.4806 / 2.1489 | 50 / 71 / 124 | no | yes | yes | yes | no |
| $M=8(K+q)$ | 1.5756 / 1.8066 / 1.1243 | 2.5629 / 2.4806 / 2.1489 | 49 / 80 / 157 | no | yes | yes | yes | no |


**R3 — the weak residual on held-out test modes** (worst over cases and steps, per-mode RMS normalised by the per-node RMS of the previous state; the common held-out block is the 512 modes ranked 1537-2048, beyond every arm's $M$ in any of the three jobs). `q-diag` already measured this on the dense fixed-$M=256$ arms; what is new here is the same measurement under a ridge and under larger $M$.

| job | arm | $q$ | quadrature | $\lambda_{rel}$ | $M$ | $m$ | in-space | held-out (common) | held/in |
|---|---|---|---|---|---|---|---|---|---|
| qrg101 | `q0_m4_dense_l0` | 0 | dense | 0 | 64 | — | 1.570e-01 | 1.302e-01 | 0.829 |
| qrg101 | `q0_m4_eq_l0` | 0 | eq | 0 | 64 | 256 | 2.919e-01 | 1.329e-01 | 0.455 |
| qrg101 | `q0_m4_eq_l0_ret` | 0 | eq | 0 | 64 | 256 | 1.964e-01 | 1.337e-01 | 0.681 |
| qrg101 | `q16_m4_dense_l0` | 16 | dense | 0 | 128 | — | 1.270e-01 | 1.262e-01 | 0.994 |
| qrg101 | `q16_m4_dense_l0p0001` | 16 | dense | 1e-04 | 128 | — | 1.270e-01 | 1.262e-01 | 0.994 |
| qrg101 | `q16_m4_dense_l0p001` | 16 | dense | 1e-03 | 128 | — | 1.270e-01 | 1.262e-01 | 0.994 |
| qrg101 | `q16_m4_dense_l0p01` | 16 | dense | 1e-02 | 128 | — | 1.271e-01 | 1.263e-01 | 0.994 |
| qrg101 | `q16_m4_dense_l0p1` | 16 | dense | 1e-01 | 128 | — | 1.275e-01 | 1.269e-01 | 0.995 |
| qrg101 | `q16_m4_dense_l1` | 16 | dense | 1e+00 | 128 | — | 1.438e-01 | 1.266e-01 | 0.880 |
| qrg101 | `q16_m4_eq_l0` | 16 | eq | 0 | 128 | 512 | 1.353e-01 | 1.410e-01 | 1.043 |
| qrg101 | `q16_m4_eq_l0p0001` | 16 | eq | 1e-04 | 128 | 512 | 1.353e-01 | 1.410e-01 | 1.043 |
| qrg101 | `q16_m4_eq_l0p001` | 16 | eq | 1e-03 | 128 | 512 | 1.353e-01 | 1.410e-01 | 1.043 |
| qrg101 | `q16_m4_eq_l0p01` | 16 | eq | 1e-02 | 128 | 512 | 1.352e-01 | 1.411e-01 | 1.043 |
| qrg101 | `q16_m4_eq_l0p1` | 16 | eq | 1e-01 | 128 | 512 | 1.362e-01 | 1.415e-01 | 1.039 |
| qrg101 | `q16_m4_eq_l1` | 16 | eq | 1e+00 | 128 | 512 | 1.550e-01 | 1.408e-01 | 0.908 |
| qrg101 | `q256_m2_dense_l0` | 256 | dense | 0 | 544 | — | 2.847e-02 | 6.786e-02 | 2.383 |
| qrg101 | `q256_m4_dense_l0` | 256 | dense | 0 | 1088 | — | 3.127e-02 | 3.628e-02 | 1.160 |
| qrg101 | `q256_m4_eq_l0` | 256 | eq | 0 | 1088 | 2048 | 6.044e-02 | 5.764e-02 | 0.954 |
| qrg101 | `q64_m4_dense_l0` | 64 | dense | 0 | 320 | — | 9.560e-02 | 1.078e-01 | 1.127 |
| qrg101 | `q64_m4_eq_l0` | 64 | eq | 0 | 320 | 1280 | 9.560e-02 | 1.199e-01 | 1.255 |
| qrg201 | `q0_m16_dense_l0` | 0 | dense | 0 | 256 | — | 1.233e-01 | 1.153e-01 | 0.936 |
| qrg201 | `q0_m16_eq_l0` | 0 | eq | 0 | 256 | 1024 | 1.233e-01 | 1.216e-01 | 0.987 |
| qrg201 | `q0_m4_dense_l0` | 0 | dense | 0 | 64 | — | 1.570e-01 | 1.302e-01 | 0.829 |
| qrg201 | `q0_m4_eq_l0` | 0 | eq | 0 | 64 | 256 | 2.919e-01 | 1.329e-01 | 0.455 |
| qrg201 | `q0_m4_eq_l0_ret` | 0 | eq | 0 | 64 | 256 | 1.964e-01 | 1.337e-01 | 0.681 |
| qrg201 | `q0_m8_dense_l0` | 0 | dense | 0 | 128 | — | 1.373e-01 | 1.298e-01 | 0.945 |
| qrg201 | `q0_m8_eq_l0` | 0 | eq | 0 | 128 | 512 | 1.518e-01 | 1.363e-01 | 0.898 |
| qrg201 | `q16_m16_dense_l0` | 16 | dense | 0 | 512 | — | 1.011e-01 | 9.439e-02 | 0.933 |
| qrg201 | `q16_m16_eq_l0` | 16 | eq | 0 | 512 | 2048 | 1.065e-01 | 1.068e-01 | 1.003 |
| qrg201 | `q16_m4_dense_l0` | 16 | dense | 0 | 128 | — | 1.270e-01 | 1.262e-01 | 0.994 |
| qrg201 | `q16_m4_eq_l0` | 16 | eq | 0 | 128 | 512 | 1.353e-01 | 1.410e-01 | 1.043 |
| qrg201 | `q16_m8_dense_l0` | 16 | dense | 0 | 256 | — | 1.182e-01 | 1.150e-01 | 0.973 |
| qrg201 | `q16_m8_eq_l0` | 16 | eq | 0 | 256 | 1024 | 1.235e-01 | 1.318e-01 | 1.068 |
| qrg201 | `q64_m16_dense_l0` | 64 | dense | 0 | 1280 | — | 8.044e-02 | 6.841e-02 | 0.850 |
| qrg201 | `q64_m16_eq_l0` | 64 | eq | 0 | 1280 | 2048 | 9.599e-02 | 8.135e-02 | 0.848 |
| qrg201 | `q64_m4_dense_l0` | 64 | dense | 0 | 320 | — | 9.560e-02 | 1.078e-01 | 1.127 |
| qrg201 | `q64_m4_eq_l0` | 64 | eq | 0 | 320 | 1280 | 9.560e-02 | 1.199e-01 | 1.255 |
| qrg201 | `q64_m8_dense_l0` | 64 | dense | 0 | 640 | — | 8.627e-02 | 8.624e-02 | 1.000 |
| qrg201 | `q64_m8_eq_l0` | 64 | eq | 0 | 640 | 2048 | 9.428e-02 | 1.009e-01 | 1.070 |
| qrg304 | `q0_m4_dense` | 0 | dense | 0 | 64 | — | 1.570e-01 | 1.302e-01 | 0.829 |
| qrg304 | `q0_m4_eqcert` | 0 | eq | 0 | 64 | 1024 | 1.574e-01 | 1.275e-01 | 0.810 |
| qrg304 | `q0_m4_eqold_ret` | 0 | eq | 0 | 64 | 256 | 1.964e-01 | 1.337e-01 | 0.681 |
| qrg304 | `q0_m4_eqstatic` | 0 | eq | 0 | 64 | 256 | 1.853e-01 | 1.442e-01 | 0.778 |
| qrg304 | `q128_m2_eqold_bnd` | 128 | eq | 0 | 288 | 1152 | 8.089e-02 | 1.204e-01 | 1.488 |
| qrg304 | `q128_m4_eqcert` | 128 | eq | 0 | 576 | 2048 | 7.340e-02 | 9.763e-02 | 1.330 |
| qrg304 | `q128_m4_eqstatic` | 128 | eq | 0 | 576 | 2048 | 7.244e-02 | 9.613e-02 | 1.327 |
| qrg304 | `q16_m4_eqcert` | 16 | eq | 0 | 128 | 1024 | 1.271e-01 | 1.301e-01 | 1.023 |
| qrg304 | `q16_m4_eqstatic` | 16 | eq | 0 | 128 | 512 | 1.337e-01 | 1.382e-01 | 1.034 |
| qrg304 | `q256_m4_dense` | 256 | dense | 0 | 1088 | — | 3.127e-02 | 3.628e-02 | 1.160 |
| qrg304 | `q256_m4_eqcert` | 256 | eq | 0 | 1088 | 2048 | 5.536e-02 | 5.924e-02 | 1.070 |
| qrg304 | `q256_m4_eqstatic` | 256 | eq | 0 | 1088 | 2048 | 5.988e-02 | 4.404e-02 | 0.735 |
| qrg304 | `q32_m4_eqcert` | 32 | eq | 0 | 192 | 1024 | 1.190e-01 | 1.295e-01 | 1.088 |
| qrg304 | `q32_m4_eqstatic` | 32 | eq | 0 | 192 | 768 | 1.197e-01 | 1.351e-01 | 1.129 |
| qrg304 | `q64_m4_dense` | 64 | dense | 0 | 320 | — | 9.560e-02 | 1.078e-01 | 1.127 |
| qrg304 | `q64_m4_eqcert` | 64 | eq | 0 | 320 | 1024 | 9.911e-02 | 1.226e-01 | 1.237 |
| qrg304 | `q64_m4_eqstatic` | 64 | eq | 0 | 320 | 1280 | 9.561e-02 | 1.194e-01 | 1.249 |


Source-generated report: `experiments/q-ridge/reports/2026-09-16-q-ridge.md` (SHA256 `8a9935b75951efd7b9e8532b762d8b5434c079129a9d3d8102275f86d7232167`) with its LaTeX twin, its two figures and its generator beside it.

Raw archive `qrg101` Git-tracked as bounded chunks: whole SHA256 `63a97e38f7c1b6982ca47fb5e6209375d1c0e47799369c383a5bc9374d94f024` (18 chunks). `output/bank_G.npz` is excluded by design and its SHA256 recorded beside them.

Raw archive `qrg201` Git-tracked as bounded chunks: whole SHA256 `5e58dd7b13c70b8a925bb7ffec20c8a1031b3ba431a954ee7300176365d394e6` (10 chunks). `output/bank_G.npz` is excluded by design and its SHA256 recorded beside them.

Raw archive `qrg304` Git-tracked as bounded chunks: whole SHA256 `2df8074de63f0c309310dfd6371d42d2bd9dd7298d5345fab21489646b39dd62` (9 chunks). `output/bank_G.npz` is excluded by design and its SHA256 recorded beside them.

