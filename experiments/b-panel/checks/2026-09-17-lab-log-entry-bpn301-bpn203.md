## 2026-09-17
### b-panel — the panel at both meshes: 0 of 39 reduced subjects non-dominated at 256², 5 of 20 at 1024²; b-eqtop's rules make the 256² EQ ladder monotone; the transferred top rungs stay uncertified even with the fit-state cap removed

Worktree `worktrees/2026-09-17-b-panel`, branch `exp/2026-09-17-b-panel` at `4d32b9699509`, forked from `exp/2026-09-16-q-ridge`. Namespace `/cluster/tufts/paralab/tawal01/b_panel_20260917/`. Predeclared protocol and amendments: `experiments/b-panel/DESIGN.md`. Frozen inputs (qtd02 directions, qrg304 certified rules, b-speed kernels) with SHA256 provenance: `experiments/b-panel/inputs/PROVENANCE.json`. Cluster jobs used: 5 of the cap of 8. Every job printed `jax_backend=gpu`, ran float64 at highest matmul precision, and was checksum-collected, independently NumPy-audited and Git-archived as bounded chunks before its exact remote attempt directory was removed.

**256² — job `3789570` (`bpn301`) on `NVIDIA A100 80GB PCIe`**, source `f3c5fbed8a21`, elapsed 1780.3 s, 47 timed subjects, failed gates: none; dropped by the OOM rule: none.

| subject | family | q / k′ | quad. | rule set | basis | rule status | tol | worst all % | worst evolved % | t=0 % | vs ref % | GPU ms | complete ms | med it | budget exits | converged | strict | admissible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `q0_M256_dense_g1em06` | rom | 0 | dense | — | — | — | 1e-06 | 2.5629 | 1.2710 | 2.5629 | 4.0637 | 341.479 | 343.401 | 3.0 | 0 | yes | yes | yes |
| `q0_M64_dense_g1em06` | rom | 0 | dense | — | — | — | 1e-06 | 2.5629 | 1.8890 | 2.5629 | 4.5575 | 283.913 | 285.872 | 3.0 | 0 | yes | yes | yes |
| `q0_M64_eqcert_g0p001` | rom | 0 | eq | eqcert | primary | confirmed (3/3) | 1e-03 | 2.5628 | 1.8898 | 2.5628 | 4.5521 | 43.636 | 45.797 | 2.0 | 0 | yes | yes | yes |
| `q0_M64_eqcert_g1em06` | rom | 0 | eq | eqcert | primary | confirmed (3/3) | 1e-06 | 2.5629 | 1.8891 | 2.5629 | 4.5529 | 59.463 | 61.722 | 3.0 | 0 | yes | yes | yes |
| `q0_M64_eqtop_g0p001` | rom | 0 | eq | eqtop | primary | confirmed (3/3) | 1e-03 | 2.5628 | 1.8898 | 2.5628 | 4.5521 | 43.411 | 45.468 | 2.0 | 0 | yes | yes | yes |
| `q0_M64_eqtop_g1em06` | rom | 0 | eq | eqtop | primary | confirmed (3/3) | 1e-06 | 2.5629 | 1.8891 | 2.5629 | 4.5529 | 58.573 | 60.596 | 3.0 | 0 | yes | yes | yes |
| `q16_M128_dense_g1em06` | rom | 16 | dense | — | — | — | 1e-06 | 2.4806 | 1.3985 | 2.4806 | 4.1065 | 360.770 | 362.766 | 3.0 | 0 | yes | yes | yes |
| `q16_M128_eqcert_g0p001` | rom | 16 | eq | eqcert | primary | confirmed (3/3) | 1e-03 | 2.4806 | 1.4272 | 2.4806 | 4.1136 | 63.303 | 65.284 | 2.0 | 0 | yes | yes | yes |
| `q16_M128_eqcert_g1em06` | rom | 16 | eq | eqcert | primary | confirmed (3/3) | 1e-06 | 2.4806 | 1.4270 | 2.4806 | 4.1145 | 81.702 | 83.776 | 3.0 | 0 | yes | yes | yes |
| `q16_M128_eqtop_g0p001` | rom | 16 | eq | eqtop | primary | confirmed (3/3) | 1e-03 | 2.4806 | 1.4272 | 2.4806 | 4.1136 | 62.090 | 64.207 | 2.0 | 0 | yes | yes | yes |
| `q16_M128_eqtop_g1em06` | rom | 16 | eq | eqtop | primary | confirmed (3/3) | 1e-06 | 2.4806 | 1.4270 | 2.4806 | 4.1145 | 81.092 | 83.330 | 3.0 | 0 | yes | yes | yes |
| `q32_M192_dense_g1em06` | rom | 32 | dense | — | — | — | 1e-06 | 2.3534 | 1.2336 | 2.3534 | 4.0814 | 434.699 | 436.969 | 3.0 | 0 | yes | yes | yes |
| `q32_M192_eqcert_g0p001` | rom | 32 | eq | eqcert | primary | confirmed (2/2) | 1e-03 | 2.3534 | 1.2497 | 2.3534 | 4.0805 | 73.695 | 75.670 | 2.0 | 0 | yes | yes | yes |
| `q32_M192_eqcert_g1em06` | rom | 32 | eq | eqcert | primary | confirmed (2/2) | 1e-06 | 2.3534 | 1.2493 | 2.3534 | 4.0818 | 96.403 | 98.369 | 3.0 | 0 | yes | yes | yes |
| `q32_M192_eqtop_g0p001` | rom | 32 | eq | eqtop | primary | confirmed (2/2) | 1e-03 | 2.3534 | 1.2497 | 2.3534 | 4.0805 | 74.104 | 76.157 | 2.0 | 0 | yes | yes | yes |
| `q32_M192_eqtop_g1em06` | rom | 32 | eq | eqtop | primary | confirmed (2/2) | 1e-06 | 2.3534 | 1.2493 | 2.3534 | 4.0818 | 97.807 | 99.792 | 3.0 | 0 | yes | yes | yes |
| `q64_M320_dense_g1em06` | rom | 64 | dense | — | — | — | 1e-06 | 2.1489 | 1.0843 | 2.1489 | 4.1013 | 621.103 | 622.971 | 3.0 | 0 | yes | yes | yes |
| `q64_M320_eqcert_g0p001` | rom | 64 | eq | eqcert | primary | marginal (b-eqtop superseded list) | 1e-03 | 2.1489 | 1.2278 | 2.1489 | 4.0836 | 90.828 | 92.633 | 2.0 | 0 | yes | yes | yes |
| `q64_M320_eqcert_g1em06` | rom | 64 | eq | eqcert | primary | marginal (b-eqtop superseded list) | 1e-06 | 2.1489 | 1.2275 | 2.1489 | 4.0855 | 119.651 | 121.781 | 3.0 | 0 | yes | yes | yes |
| `q64_M320_eqtop_g0p001` | rom | 64 | eq | eqtop | primary | confirmed (2/2) | 1e-03 | 2.1489 | 1.0841 | 2.1489 | 4.1008 | 113.887 | 116.014 | 2.0 | 0 | yes | yes | yes |
| `q64_M320_eqtop_g1em06` | rom | 64 | eq | eqtop | primary | confirmed (2/2) | 1e-06 | 2.1489 | 1.0840 | 2.1489 | 4.1027 | 148.946 | 151.027 | 3.0 | 0 | yes | yes | yes |
| `q128_M576_dense_g1em06` | rom | 128 | dense | — | — | — | 1e-06 | 1.8116 | 0.8930 | 1.8116 | 4.0797 | 1186.926 | 1189.022 | 3.0 | 0 | yes | yes | yes |
| `q128_M576_eqcert_g0p001` | rom | 128 | eq | eqcert | secondary | — | 1e-03 | 1.8116 | 0.8926 | 1.8116 | 4.0791 | 189.334 | 191.524 | 2.0 | 0 | yes | yes | yes |
| `q128_M576_eqcert_g1em06` | rom | 128 | eq | eqcert | secondary | — | 1e-06 | 1.8116 | 0.8925 | 1.8116 | 4.0809 | 246.534 | 248.633 | 3.0 | 0 | yes | yes | yes |
| `q128_M576_eqtop_g0p001` | rom | 128 | eq | eqtop | primary | certified in one draw | 1e-03 | 1.8116 | 0.8932 | 1.8116 | 4.0787 | 205.126 | 207.152 | 2.0 | 0 | yes | yes | yes |
| `q128_M576_eqtop_g1em06` | rom | 128 | eq | eqtop | primary | certified in one draw | 1e-06 | 1.8116 | 0.8931 | 1.8116 | 4.0806 | 273.483 | 275.327 | 3.0 | 0 | yes | yes | yes |
| `q256_M1088_dense_g1em06` | rom | 256 | dense | — | — | — | 1e-06 | 0.9053 | 0.5194 | 0.9053 | 4.0391 | 3939.764 | 3941.992 | 5.0 | 0 | yes | yes | yes |
| `q256_M1088_eqcert_g0p001` | rom | 256 | eq | eqcert | secondary | — | 1e-03 | 1.0324 | 1.0324 | 0.9053 | 4.0322 | 429.560 | 431.550 | 3.0 | 0 | yes | yes | yes |
| `q256_M1088_eqcert_g1em06` | rom | 256 | eq | eqcert | secondary | — | 1e-06 | 1.0361 | 1.0361 | 0.9053 | 4.0332 | 697.644 | 699.693 | 5.0 | 0 | yes | yes | yes |
| `q256_M1088_eqtop_g0p001` | rom | 256 | eq | eqtop | primary | certified in one draw | 1e-03 | 0.9053 | 0.5129 | 0.9053 | 4.0355 | 466.448 | 468.562 | 3.0 | 0 | yes | yes | yes |
| `q256_M1088_eqtop_g1em06` | rom | 256 | eq | eqtop | primary | certified in one draw | 1e-06 | 0.9053 | 0.5129 | 0.9053 | 4.0365 | 746.020 | 748.389 | 5.0 | 0 | yes | yes | yes |
| `q0_M64_eqcert_g1em06_fastL4` | fast | 0 | eq | eqcert | primary | confirmed (3/3) | 1e-06 | 2.5629 | 1.8891 | 2.5629 | 4.5529 | 40.359 | 42.515 | 3.0 | 0 | yes | yes | yes |
| `pod16_M64_dense` | pod | 16 | dense | — | — | — | 1e-06 | 61.6503 | 28.7250 | 61.6503 | 61.6503 | 46.776 | 48.762 | 2.0 | 0 | yes | yes | yes |
| `pod32_M128_dense` | pod | 32 | dense | — | — | — | 1e-06 | 47.0681 | 18.7995 | 47.0681 | 47.0681 | 80.988 | 82.890 | 2.0 | 0 | yes | no | yes |
| `pod64_M256_dense` | pod | 64 | dense | — | — | — | 1e-06 | 19.8156 | 7.0835 | 19.8156 | 19.8156 | 145.747 | 147.685 | 2.0 | 0 | yes | no | yes |
| `pod128_M512_dense` | pod | 128 | dense | — | — | — | 1e-06 | 10.1198 | 1.9464 | 10.1198 | 10.1198 | 331.585 | 333.542 | 2.0 | 0 | yes | no | yes |
| `pod256_M1024_dense` | pod | 256 | dense | — | — | — | 1e-06 | 3.7698 | 0.7109 | 3.7698 | 4.0362 | 898.769 | 901.239 | 2.0 | 0 | yes | no | yes |
| `pod512_M2048_dense` | pod | 512 | dense | — | — | — | 1e-06 | 0.6125 | 0.2184 | 0.6125 | 4.0266 | 2790.828 | 2794.292 | 2.0 | 0 | yes | no | yes |
| `free512_M1024_dense` | free | 512 | dense | — | — | — | 1e-06 | 0.6027 | 0.4471 | 0.6027 | 4.0423 | 2439.551 | 2441.921 | 3.0 | 0 | yes | no | yes |
| `fno-large` | fno | — | — | — | — | — | — | 7.4164 | 7.4164 | 0.0000 | 5.7495 | 7.183 | 7.427 | — | — | — | — | yes |
| `dense_tight` | fom | — | — | — | — | — | — | 0.0000 | 0.0000 | 0.0000 | 4.0265 | 63.878 | 65.849 | 2.0 | — | — | — | yes |
| `fft_tight` | fom | — | — | — | — | — | — | 0.0000 | 0.0000 | 0.0000 | 4.0265 | 90.405 | 92.537 | 2.0 | — | — | — | yes |
| `nt1e-2_dt005` | fom | — | — | — | — | — | — | 3.7127 | 3.7127 | 0.0000 | 2.4737 | 15.656 | 17.749 | 1.0 | — | — | — | yes |
| `nt1e-2_dt01` | fom | — | — | — | — | — | — | 3.1999 | 3.1999 | 0.0000 | 3.6168 | 9.009 | 10.982 | 1.0 | — | — | — | yes |
| `nt1e-3_dt005` | fom | — | — | — | — | — | — | 0.0489 | 0.0489 | 0.0000 | 4.0399 | 31.788 | 34.227 | 1.0 | — | — | — | yes |
| `nt1e-3_dt01` | fom | — | — | — | — | — | — | 1.5179 | 1.5179 | 0.0000 | 5.1761 | 19.724 | 21.666 | 1.0 | — | — | — | yes |
| `nt1e-4_dt005` | fom | — | — | — | — | — | — | 0.0338 | 0.0338 | 0.0000 | 4.0320 | 36.999 | 38.980 | 1.0 | — | — | — | yes |
| `nt1e-4_dt01` | fom | — | — | — | — | — | — | 1.5109 | 1.5109 | 0.0000 | 5.1552 | 27.769 | 30.047 | 1.0 | — | — | — | yes |

Non-dominated (median_gpu_ms, worst_all_times_percent), admissible: `fno-large`, `nt1e-2_dt01`, `nt1e-3_dt01`, `nt1e-4_dt01`, `nt1e-3_dt005`, `nt1e-4_dt005`, `dense_tight`, `fft_tight`.
Non-dominated (median_gpu_ms, worst_evolved_percent), admissible: `fno-large`, `nt1e-2_dt01`, `nt1e-3_dt01`, `nt1e-4_dt01`, `nt1e-3_dt005`, `nt1e-4_dt005`, `dense_tight`, `fft_tight`.
Non-dominated (median_host_ms, worst_all_times_percent), admissible: `fno-large`, `nt1e-2_dt01`, `nt1e-3_dt01`, `nt1e-4_dt01`, `nt1e-3_dt005`, `nt1e-4_dt005`, `dense_tight`, `fft_tight`.
Non-dominated (median_host_ms, worst_evolved_percent), admissible: `fno-large`, `nt1e-2_dt01`, `nt1e-3_dt01`, `nt1e-4_dt01`, `nt1e-3_dt005`, `nt1e-4_dt005`, `dense_tight`, `fft_tight`.

Ladder `dense`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [1.889, 1.3985, 1.2336, 1.0843, 0.893, 0.5194]; GPU ms [283.9, 360.8, 434.7, 621.1, 1186.9, 3939.8]; monotone evolved yes, all converged yes, converged non-dominated points 6, error span 3.637, cost span 13.877.
Ladder `eq_eqcert_g0p001`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [1.8898, 1.4272, 1.2497, 1.2278, 0.8926, 1.0324]; GPU ms [43.6, 63.3, 73.7, 90.8, 189.3, 429.6]; monotone evolved no, all converged yes, converged non-dominated points 5, error span 2.117, cost span 4.339.
Ladder `eq_eqcert_g1em06`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [1.8891, 1.427, 1.2493, 1.2275, 0.8925, 1.0361]; GPU ms [59.5, 81.7, 96.4, 119.7, 246.5, 697.6]; monotone evolved no, all converged yes, converged non-dominated points 5, error span 2.117, cost span 4.146.
Ladder `eq_eqtop_g0p001`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [1.8898, 1.4272, 1.2497, 1.0841, 0.8932, 0.5129]; GPU ms [43.4, 62.1, 74.1, 113.9, 205.1, 466.4]; monotone evolved yes, all converged yes, converged non-dominated points 6, error span 3.684, cost span 10.745.
Ladder `eq_eqtop_g1em06`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [1.8891, 1.427, 1.2493, 1.084, 0.8931, 0.5129]; GPU ms [58.6, 81.1, 97.8, 148.9, 273.5, 746.0]; monotone evolved yes, all converged yes, converged non-dominated points 6, error span 3.683, cost span 12.737.
Ladder `pod`: rungs [16, 32, 64, 128, 256, 512]; worst evolved % [28.725, 18.7995, 7.0835, 1.9464, 0.7109, 0.2184]; GPU ms [46.8, 81.0, 145.7, 331.6, 898.8, 2790.8]; monotone evolved yes, all converged yes, converged non-dominated points 6, error span 131.536, cost span 59.664.

Cross-job fidelity: `q0_M64_dense_g1em06` vs `q0_M64_dense` (qtd02) 4.9e-13 [first tier, pass]; `q0_M256_dense_g1em06` vs `q0_M256_dense_g1em06` (btq201) 1.8e-13 [first tier, pass]; `q0_M64_eqcert_g1em06` vs `q0_m4_eqcert` (qrg304) 6.5e-14 [first tier, pass]; `q16_M128_dense_g1em06` vs `old_q16_M128_dense` (qtd02) 9.1e-13 [first tier, pass]; `q32_M192_dense_g1em06` vs `old_q32_M192_dense` (qtd02) 6.1e-13 [first tier, pass]; `q64_M320_dense_g1em06` vs `old_q64_M320_dense` (qtd02) 2.2e-13 [first tier, pass]; `q128_M576_dense_g1em06` vs `old_q128_M576_dense` (qtd02) 2.8e-13 [first tier, pass]; `q256_M1088_dense_g1em06` vs `old_q256_M1088_dense` (qtd02) 6.9e-13 [first tier, pass]; `q64_M320_eqcert_g1em06` vs `q64_m4_eqcert` (qrg304) 1.9e-09 [first tier, pass]; `q256_M1088_eqcert_g1em06` vs `q256_m4_eqcert` (qrg304) 3.5e-09 [first tier, pass].

FNO `fno-large` in the same allocation: pooled device query median 7.183 ms; second process in the same Slurm allocation on the same GPU, after the JAX phase exited; the source protocol times all models in one process.

Gates: `complete` yes; `backend_gpu` yes; `x64` yes; `precision_highest` yes; `bank_frozen` yes; `checkpoint_unchanged` yes; `final_cohort_unopened` yes; `step_budget_600` yes; `evaluation_cohort_bitwise_abl01` yes; `directions_file_sha256` yes; `directions_prefix_hashes` yes; `repetition_output_identical` yes; `fft_tight_converged_everywhere` yes; `direct_reproduces_fft_tight` yes; `fast_parity` yes; `fno_cohort_disjoint_from_training` yes; `reference_residuals` yes; `every_rule_hash_matches_provenance` yes; `matched_rule_files_bitwise` yes; `no_subject_dropped` yes; `every_subject_case_has_all_reps` yes; `every_invocation_paired` yes; `overdetermined_weak_system` yes; `every_rom_carries_exit_and_stationarity` yes; `host_time_covers_gpu_time` yes; `artifacts_present` yes; `same_grid_baseline_present` yes; `recorded_errors_recomputed_from_saved_fields` yes; `matched_rule_files_bitwise_recomputed` yes; `fno_returns_supplied_field_at_t0` yes; `fno_timed_in_same_allocation` yes; `convergence_flags_reproduced` yes; `reproduces_q0_M64_dense_g1em06` yes; `reproduces_q0_M256_dense_g1em06` yes; `reproduces_q0_M64_eqcert_g1em06` yes; `reproduces_q16_M128_dense_g1em06` yes; `reproduces_q32_M192_dense_g1em06` yes; `reproduces_q64_M320_dense_g1em06` yes; `reproduces_q128_M576_dense_g1em06` yes; `reproduces_q256_M1088_dense_g1em06` yes; `reproduces_q64_M320_eqcert_g1em06` yes; `reproduces_q256_M1088_eqcert_g1em06` yes.

**1024² — job `3789572` (`bpn203`) on `NVIDIA H200`**, source `f3c5fbed8a21`, elapsed 5559.7 s, 29 timed subjects, failed gates: none; dropped by the OOM rule: none.

| subject | family | q / k′ | quad. | rule set | basis | rule status | tol | worst all % | worst evolved % | t=0 % | vs ref % | GPU ms | complete ms | med it | budget exits | converged | strict | admissible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `q0_M64_dense_g1em06` | rom | 0 | dense | — | — | — | 1e-06 | 3.8562 | 2.2886 | 3.8562 | 3.8562 | 1551.732 | 1562.390 | 3.0 | 0 | yes | yes | yes |
| `q0_M64_eqxfer_g0p001` | rom | 0 | eq | eqxfer | primary | confirmed (3/3) | 1e-03 | 3.8562 | 2.2925 | 3.8562 | 3.8562 | 32.230 | 44.384 | 2.0 | 0 | yes | yes | yes |
| `q0_M64_eqxfer_g1em06` | rom | 0 | eq | eqxfer | primary | confirmed (3/3) | 1e-06 | 3.8562 | 2.2913 | 3.8562 | 3.8562 | 40.270 | 51.411 | 3.0 | 0 | yes | yes | yes |
| `q16_M128_dense_g1em06` | rom | 16 | dense | — | — | — | 1e-06 | 3.8071 | 1.5398 | 3.8071 | 3.8071 | 1949.125 | 1960.572 | 3.0 | 0 | yes | yes | yes |
| `q16_M128_eqxfer_g0p001` | rom | 16 | eq | eqxfer | primary | confirmed (3/3) | 1e-03 | 3.8071 | 1.5277 | 3.8071 | 3.8071 | 48.351 | 60.676 | 2.0 | 0 | yes | yes | yes |
| `q16_M128_eqxfer_g1em06` | rom | 16 | eq | eqxfer | primary | confirmed (3/3) | 1e-06 | 3.8071 | 1.5273 | 3.8071 | 3.8071 | 59.585 | 70.490 | 3.0 | 0 | yes | yes | yes |
| `q32_M192_dense_g1em06` | rom | 32 | dense | — | — | — | 1e-06 | 3.7140 | 1.4482 | 3.7140 | 3.7140 | 2382.864 | 2393.457 | 3.0 | 0 | yes | yes | yes |
| `q32_M192_eqxfer_g0p001` | rom | 32 | eq | eqxfer | primary | confirmed (2/2) | 1e-03 | 3.7140 | 1.4488 | 3.7140 | 3.7140 | 57.585 | 69.248 | 2.0 | 0 | yes | yes | yes |
| `q32_M192_eqxfer_g1em06` | rom | 32 | eq | eqxfer | primary | confirmed (2/2) | 1e-06 | 3.7140 | 1.4487 | 3.7140 | 3.7140 | 72.025 | 83.041 | 3.0 | 0 | yes | yes | yes |
| `q64_M320_dense_g1em06` | rom | 64 | dense | — | — | — | 1e-06 | 3.5773 | 1.2739 | 3.5773 | 3.5773 | 3565.635 | 3581.765 | 3.0 | 0 | yes | yes | yes |
| `q64_M320_eqxfer_g0p001` | rom | 64 | eq | eqxfer | none | marginal (b-eqtop superseded list) | 1e-03 | 3.5773 | 1.2763 | 3.5773 | 3.5773 | 64.433 | 76.499 | 2.0 | 0 | yes | yes | no |
| `q64_M320_eqxfer_g1em06` | rom | 64 | eq | eqxfer | none | marginal (b-eqtop superseded list) | 1e-06 | 3.5773 | 1.2762 | 3.5773 | 3.5773 | 83.970 | 94.548 | 3.0 | 0 | yes | yes | no |
| `q128_M576_dense_g1em06` | rom | 128 | dense | — | — | — | 1e-06 | 3.2858 | 1.0444 | 3.2858 | 3.2858 | 6895.231 | 6909.123 | 3.0 | 0 | yes | yes | yes |
| `q128_M576_eqxfer_g0p001` | rom | 128 | eq | eqxfer | secondary | — | 1e-03 | 3.2858 | 1.0443 | 3.2858 | 3.2858 | 128.250 | 139.230 | 2.0 | 0 | yes | yes | yes |
| `q128_M576_eqxfer_g1em06` | rom | 128 | eq | eqxfer | secondary | — | 1e-06 | 3.2858 | 1.0442 | 3.2858 | 3.2858 | 163.684 | 175.414 | 3.0 | 0 | yes | yes | yes |
| `q256_M1088_dense_g1em06` | rom | 256 | dense | — | — | — | 1e-06 | 2.7974 | 0.5861 | 2.7974 | 2.7974 | 22053.852 | 22066.298 | 5.0 | 0 | yes | yes | yes |
| `q256_M1088_eqxfer_g0p001` | rom | 256 | eq | eqxfer | none | — | 1e-03 | 2.7974 | 1.8655 | 2.7974 | 2.7974 | 277.226 | 290.052 | 2.0 | 0 | yes | yes | no |
| `q256_M1088_eqxfer_g1em06` | rom | 256 | eq | eqxfer | none | — | 1e-06 | 2.7974 | 1.8664 | 2.7974 | 2.7974 | 502.207 | 515.150 | 5.0 | 0 | yes | yes | no |
| `pod16_M64_dense` | pod | 16 | dense | — | — | — | 1e-06 | 61.9285 | 29.2793 | 61.9285 | 61.9285 | 198.177 | 209.595 | 2.0 | 0 | yes | yes | yes |
| `pod32_M128_dense` | pod | 32 | dense | — | — | — | 1e-06 | 47.6410 | 19.4088 | 47.6410 | 47.6410 | 391.216 | 402.508 | 2.0 | 0 | yes | no | yes |
| `pod64_M256_dense` | pod | 64 | dense | — | — | — | 1e-06 | 20.5994 | 7.4077 | 20.5994 | 20.5994 | 741.581 | 752.921 | 2.0 | 0 | yes | no | yes |
| `pod128_M512_dense` | pod | 128 | dense | — | — | — | 1e-06 | 10.7339 | 2.1609 | 10.7339 | 10.7339 | 1760.633 | 1773.032 | 2.0 | 0 | yes | no | yes |
| `pod256_M1024_dense` | pod | 256 | dense | — | — | — | 1e-06 | 4.1416 | 1.1017 | 4.1416 | 4.1416 | 4897.627 | 4911.441 | 2.0 | 0 | yes | no | yes |
| `free512_M1024_dense` | free | 512 | dense | — | — | — | 1e-06 | 2.4466 | 0.5108 | 2.4466 | 2.4466 | 13514.566 | 13528.993 | 3.0 | 0 | yes | no | yes |
| `fno-large` | fno | — | — | — | — | — | — | 6.2657 | 6.2657 | 0.0000 | 5.6472 | 58.917 | 71.029 | — | — | — | — | yes |
| `fft_tight` | fom | — | — | — | — | — | — | 0.0000 | 0.0000 | 0.0000 | 2.1416 | 210.898 | 222.236 | 2.0 | — | — | — | yes |
| `nt1e-2_dt005` | fom | — | — | — | — | — | — | 4.2628 | 4.2628 | 0.0000 | 2.3899 | 31.116 | 42.734 | 1.0 | — | — | — | yes |
| `nt1e-2_dt01` | fom | — | — | — | — | — | — | 3.4582 | 3.4582 | 0.0000 | 2.8919 | 17.724 | 29.300 | 1.0 | — | — | — | yes |
| `nt1e-4_dt005` | fom | — | — | — | — | — | — | 0.0343 | 0.0343 | 0.0000 | 2.1281 | 81.311 | 92.541 | 1.0 | — | — | — | yes |
| `nt1e-4_dt01` | fom | — | — | — | — | — | — | 1.6287 | 1.6287 | 0.0000 | 3.7562 | 64.035 | 76.278 | 2.0 | — | — | — | yes |

Non-dominated (median_gpu_ms, worst_all_times_percent), admissible: `nt1e-2_dt01`, `nt1e-4_dt01`, `nt1e-4_dt005`, `fft_tight`.
Non-dominated (median_gpu_ms, worst_evolved_percent), admissible: `nt1e-2_dt01`, `q0_M64_eqxfer_g0p001`, `q0_M64_eqxfer_g1em06`, `q16_M128_eqxfer_g0p001`, `q32_M192_eqxfer_g0p001`, `q32_M192_eqxfer_g1em06`, `nt1e-4_dt005`, `fft_tight`.
Non-dominated (median_host_ms, worst_all_times_percent), admissible: `nt1e-2_dt01`, `nt1e-4_dt01`, `nt1e-4_dt005`, `fft_tight`.
Non-dominated (median_host_ms, worst_evolved_percent), admissible: `nt1e-2_dt01`, `q0_M64_eqxfer_g0p001`, `q0_M64_eqxfer_g1em06`, `q16_M128_eqxfer_g0p001`, `q32_M192_eqxfer_g0p001`, `q32_M192_eqxfer_g1em06`, `nt1e-4_dt005`, `fft_tight`.

Ladder `dense`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [2.2886, 1.5398, 1.4482, 1.2739, 1.0444, 0.5861]; GPU ms [1551.7, 1949.1, 2382.9, 3565.6, 6895.2, 22053.9]; monotone evolved yes, all converged yes, converged non-dominated points 6, error span 3.905, cost span 14.212.
Ladder `eq_eqxfer_g0p001`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [2.2925, 1.5277, 1.4488, 1.2763, 1.0443, 1.8655]; GPU ms [32.2, 48.4, 57.6, 64.4, 128.3, 277.2]; monotone evolved no, all converged yes, converged non-dominated points 4, error span 2.195, cost span 3.979.
Ladder `eq_eqxfer_g1em06`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [2.2913, 1.5273, 1.4487, 1.2762, 1.0442, 1.8664]; GPU ms [40.3, 59.6, 72.0, 84.0, 163.7, 502.2]; monotone evolved no, all converged yes, converged non-dominated points 4, error span 2.194, cost span 4.065.
Ladder `pod`: rungs [16, 32, 64, 128, 256]; worst evolved % [29.2793, 19.4088, 7.4077, 2.1609, 1.1017]; GPU ms [198.2, 391.2, 741.6, 1760.6, 4897.6]; monotone evolved yes, all converged yes, converged non-dominated points 5, error span 26.577, cost span 24.713.

Transferred rules (`eqxfer`): q=0 m=934 ρmax 0.0180 ρ95 0.0161 → primary; q=16 m=918 ρmax 0.0594 ρ95 0.0562 → primary; q=32 m=972 ρmax 0.0307 ρ95 0.0135 → primary; q=64 m=942 ρmax 0.1275 ρ95 0.1169 → none; q=128 m=1713 ρmax 0.1702 ρ95 0.1020 → secondary; q=256 m=1658 ρmax 0.3239 ρ95 0.2931 → none.

FNO `fno-large` in the same allocation: pooled device query median 58.917 ms; second process in the same Slurm allocation on the same GPU, after the JAX phase exited; the source protocol times all models in one process.

Gates: `complete` yes; `backend_gpu` yes; `x64` yes; `precision_highest` yes; `bank_frozen` yes; `checkpoint_unchanged` yes; `final_cohort_unopened` yes; `step_budget_600` yes; `evaluation_cohort_bitwise_abl01` yes; `directions_file_sha256` yes; `directions_prefix_hashes` yes; `repetition_output_identical` yes; `fft_tight_converged_everywhere` yes; `fno_cohort_disjoint_from_training` yes; `transfer_fit_cert_disjoint` yes; `reference_residuals` yes; `every_rule_hash_matches_provenance` yes; `matched_rule_files_bitwise` —; `no_subject_dropped` yes; `every_subject_case_has_all_reps` yes; `every_invocation_paired` yes; `overdetermined_weak_system` yes; `every_rom_carries_exit_and_stationarity` yes; `host_time_covers_gpu_time` yes; `artifacts_present` yes; `same_grid_baseline_present` yes; `recorded_errors_recomputed_from_saved_fields` yes; `fno_returns_supplied_field_at_t0` yes; `fno_timed_in_same_allocation` yes; `convergence_flags_reproduced` yes.

**What was wrong and is retracted.** Nothing numerical retracted in this session. bpn201 (3783817) and bpn202 (3787247) remain retracted from earlier sessions and produced no timed number; bpn101's 256² table is superseded wholesale by bpn301's (DESIGN §A5.1) but is not withdrawn. DESIGN §A5.2's prediction is scored here: its OUTCOME held (q=128 secondary, q=256 uncertified at 1024²) but its MECHANISM (fit-state starvation) is refuted — the uncapped 64-state refit misses the same bar.

**Open.** bpn401 (512²) is now the measurement that would locate the crossover between 256² (0 reduced non-dominated) and 1024² (5); run it on A100-80G so two meshes share hardware. The cause of the transferred top rungs' failure at 1024² is open: support attrition (918-1713 of 1024/2048 nodes keep positive weight) and the 1024² reachable population are the remaining candidates. Codex report audit still owed after 2026-09-19 11:33.

Source-generated report: `experiments/b-panel/reports/2026-09-17-b-panel.md` (SHA256 `2be2f56217bab3aecf816cc9e2eadc8bd0291114ce2240412ac4e35bf461f2d1`) with `summary.json`, the envelope figures and their point JSONs beside it; generator `reports/generate_panel.py` reads only the audit JSONs.
Raw archive `bpn101` Git-tracked as bounded chunks: whole SHA256 `db2fcb6f1fd1c75024455c963eda8619ac2ba38ecdbae97986cfa3abf2cfd1ab` (15 chunks).
Raw archive `bpn203` Git-tracked as bounded chunks: whole SHA256 `dedab45db81d35dbeaa47e22b95cece13352217cdd9d417390bec7bf6e118c68` (185 chunks).
Raw archive `bpn301` Git-tracked as bounded chunks: whole SHA256 `4c788e051fe7813086a86ad28369373a1ad4188938c15cfb5fb2814555a75c42` (19 chunks).
