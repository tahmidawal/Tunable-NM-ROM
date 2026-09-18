## 2026-09-17
### b-panel — bpn401 (512², A100-80G) lands: 0 of 31 reduced subjects non-dominated, so the crossover lies between 512² and 1024²; shared-hardware ratios 4.480×→2.923× (vs cheapest FOM) and 0.446×→0.257× (vs fft_tight) from 256² to 512²; all 39 reduced subjects sit above the 2.7025 % discretisation error

Worktree `worktrees/2026-09-17-b-panel`, branch `exp/2026-09-17-b-panel` at `cc44abe158ae`, forked from `exp/2026-09-16-q-ridge`. Namespace `/cluster/tufts/paralab/tawal01/b_panel_20260917/`. Predeclared protocol and amendments: `experiments/b-panel/DESIGN.md`. Frozen inputs (qtd02 directions, qrg304 certified rules, b-speed kernels) with SHA256 provenance: `experiments/b-panel/inputs/PROVENANCE.json`. Cluster jobs used: 6 of the cap of 8. Every job printed `jax_backend=gpu`, ran float64 at highest matmul precision, and was checksum-collected, independently NumPy-audited and Git-archived as bounded chunks before its exact remote attempt directory was removed.

**512² — job `3805065` (`bpn401`) on `NVIDIA A100 80GB PCIe`**, source `7a57f02e2bf4`, elapsed 5469.6 s, 47 timed subjects, failed gates: matched_rule_files_bitwise, matched_rule_files_bitwise_recomputed; dropped by the OOM rule: none.

| subject | family | q / k′ | quad. | rule set | basis | rule status | tol | worst all % | worst evolved % | t=0 % | vs ref % | GPU ms | complete ms | med it | budget exits | converged | strict | admissible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `q0_M256_dense_g1em06` | rom | 0 | dense | — | — | — | 1e-06 | 3.2836 | 1.3761 | 3.2836 | 3.2836 | 1175.712 | 1179.724 | 3.0 | 0 | yes | yes | yes |
| `q0_M64_dense_g1em06` | rom | 0 | dense | — | — | — | 1e-06 | 3.2836 | 2.1376 | 3.2836 | 3.7578 | 923.215 | 927.304 | 3.0 | 0 | yes | yes | yes |
| `q0_M64_eqtopxfer_g0p001` | rom | 0 | eq | eqtopxfer | primary | confirmed (3/3) | 1e-03 | 3.2836 | 2.1387 | 3.2836 | 3.7448 | 43.823 | 47.497 | 2.0 | 0 | yes | yes | yes |
| `q0_M64_eqtopxfer_g1em06` | rom | 0 | eq | eqtopxfer | primary | confirmed (3/3) | 1e-06 | 3.2836 | 2.1378 | 3.2836 | 3.7449 | 59.080 | 62.925 | 3.0 | 0 | yes | yes | yes |
| `q0_M64_eqxfer_g0p001` | rom | 0 | eq | eqxfer | primary | confirmed (3/3) | 1e-03 | 3.2836 | 2.1409 | 3.2836 | 3.7681 | 44.135 | 48.273 | 2.0 | 0 | yes | yes | yes |
| `q0_M64_eqxfer_g1em06` | rom | 0 | eq | eqxfer | primary | confirmed (3/3) | 1e-06 | 3.2836 | 2.1400 | 3.2836 | 3.7681 | 58.009 | 61.925 | 3.0 | 0 | yes | yes | yes |
| `q16_M128_dense_g1em06` | rom | 16 | dense | — | — | — | 1e-06 | 3.2248 | 1.5071 | 3.2248 | 3.2248 | 1190.822 | 1194.891 | 3.0 | 0 | yes | yes | yes |
| `q16_M128_eqtopxfer_g0p001` | rom | 16 | eq | eqtopxfer | primary | confirmed (3/3) | 1e-03 | 3.2248 | 1.4538 | 3.2248 | 3.2248 | 62.452 | 66.326 | 2.0 | 0 | yes | yes | yes |
| `q16_M128_eqtopxfer_g1em06` | rom | 16 | eq | eqtopxfer | primary | confirmed (3/3) | 1e-06 | 3.2248 | 1.4537 | 3.2248 | 3.2248 | 79.214 | 83.143 | 3.0 | 0 | yes | yes | yes |
| `q16_M128_eqxfer_g0p001` | rom | 16 | eq | eqxfer | primary | confirmed (3/3) | 1e-03 | 3.2248 | 1.4917 | 3.2248 | 3.2248 | 62.504 | 66.734 | 2.0 | 0 | yes | yes | yes |
| `q16_M128_eqxfer_g1em06` | rom | 16 | eq | eqxfer | primary | confirmed (3/3) | 1e-06 | 3.2248 | 1.4914 | 3.2248 | 3.2248 | 79.574 | 83.654 | 3.0 | 0 | yes | yes | yes |
| `q32_M192_dense_g1em06` | rom | 32 | dense | — | — | — | 1e-06 | 3.1172 | 1.3666 | 3.1172 | 3.1172 | 1502.392 | 1506.082 | 3.0 | 0 | yes | yes | yes |
| `q32_M192_eqtopxfer_g0p001` | rom | 32 | eq | eqtopxfer | secondary | confirmed (2/2) | 1e-03 | 3.1172 | 1.4060 | 3.1172 | 3.1172 | 72.140 | 75.900 | 2.0 | 0 | yes | yes | yes |
| `q32_M192_eqtopxfer_g1em06` | rom | 32 | eq | eqtopxfer | secondary | confirmed (2/2) | 1e-06 | 3.1172 | 1.4055 | 3.1172 | 3.1172 | 93.874 | 97.523 | 3.0 | 0 | yes | yes | yes |
| `q32_M192_eqxfer_g0p001` | rom | 32 | eq | eqxfer | primary | confirmed (2/2) | 1e-03 | 3.1172 | 1.3673 | 3.1172 | 3.1172 | 74.392 | 78.572 | 2.0 | 0 | yes | yes | yes |
| `q32_M192_eqxfer_g1em06` | rom | 32 | eq | eqxfer | primary | confirmed (2/2) | 1e-06 | 3.1172 | 1.3672 | 3.1172 | 3.1172 | 96.697 | 100.783 | 3.0 | 0 | yes | yes | yes |
| `q64_M320_dense_g1em06` | rom | 64 | dense | — | — | — | 1e-06 | 2.9565 | 1.2015 | 2.9565 | 3.0133 | 2221.735 | 2225.724 | 3.0 | 0 | yes | yes | yes |
| `q64_M320_eqtopxfer_g0p001` | rom | 64 | eq | eqtopxfer | none | confirmed (2/2) | 1e-03 | 2.9565 | 1.1995 | 2.9565 | 3.0056 | 115.103 | 119.116 | 2.0 | 0 | yes | yes | no |
| `q64_M320_eqtopxfer_g1em06` | rom | 64 | eq | eqtopxfer | none | confirmed (2/2) | 1e-06 | 2.9565 | 1.1994 | 2.9565 | 3.0069 | 150.222 | 154.019 | 3.0 | 0 | yes | yes | no |
| `q64_M320_eqxfer_g0p001` | rom | 64 | eq | eqxfer | none | marginal (b-eqtop superseded list) | 1e-03 | 2.9565 | 1.2488 | 2.9565 | 3.0135 | 89.783 | 93.704 | 2.0 | 0 | yes | yes | no |
| `q64_M320_eqxfer_g1em06` | rom | 64 | eq | eqxfer | none | marginal (b-eqtop superseded list) | 1e-06 | 2.9565 | 1.2485 | 2.9565 | 3.0149 | 120.838 | 125.105 | 3.0 | 0 | yes | yes | no |
| `q128_M576_dense_g1em06` | rom | 128 | dense | — | — | — | 1e-06 | 2.6066 | 0.9890 | 2.6066 | 2.9236 | 4541.310 | 4545.276 | 3.0 | 0 | yes | yes | yes |
| `q128_M576_eqtopxfer_g0p001` | rom | 128 | eq | eqtopxfer | primary | certified in one draw | 1e-03 | 2.6066 | 0.9902 | 2.6066 | 2.9224 | 200.064 | 204.058 | 2.0 | 0 | yes | yes | yes |
| `q128_M576_eqtopxfer_g1em06` | rom | 128 | eq | eqtopxfer | primary | certified in one draw | 1e-06 | 2.6066 | 0.9901 | 2.6066 | 2.9237 | 262.366 | 266.311 | 3.0 | 0 | yes | yes | yes |
| `q128_M576_eqxfer_g0p001` | rom | 128 | eq | eqxfer | none | — | 1e-03 | 2.6066 | 0.9895 | 2.6066 | 2.9188 | 188.391 | 192.382 | 2.0 | 0 | yes | yes | no |
| `q128_M576_eqxfer_g1em06` | rom | 128 | eq | eqxfer | none | — | 1e-06 | 2.6066 | 0.9894 | 2.6066 | 2.9200 | 241.968 | 245.916 | 3.0 | 0 | yes | yes | no |
| `q256_M1088_dense_g1em06` | rom | 256 | dense | — | — | — | 1e-06 | 2.0195 | 0.5625 | 2.0195 | 2.7611 | 16219.871 | 16224.390 | 5.0 | 0 | yes | yes | yes |
| `q256_M1088_eqtopxfer_g0p001` | rom | 256 | eq | eqtopxfer | primary | certified in one draw | 1e-03 | 2.0195 | 0.5510 | 2.0195 | 2.7581 | 486.341 | 490.449 | 3.0 | 0 | yes | yes | yes |
| `q256_M1088_eqtopxfer_g1em06` | rom | 256 | eq | eqtopxfer | primary | certified in one draw | 1e-06 | 2.0195 | 0.5510 | 2.0195 | 2.7602 | 783.328 | 787.776 | 5.0 | 0 | yes | yes | yes |
| `q256_M1088_eqxfer_g0p001` | rom | 256 | eq | eqxfer | none | — | 1e-03 | 2.0195 | 1.8179 | 2.0195 | 2.7920 | 395.669 | 399.906 | 2.5 | 0 | yes | yes | no |
| `q256_M1088_eqxfer_g1em06` | rom | 256 | eq | eqxfer | none | — | 1e-06 | 2.0195 | 1.8204 | 2.0195 | 2.7969 | 686.643 | 690.736 | 4.0 | 0 | yes | yes | no |
| `q0_M64_eqxfer_g1em06_fastL4` | fast | 0 | eq | eqxfer | primary | confirmed (3/3) | 1e-06 | 3.2836 | 2.1400 | 3.2836 | 3.7681 | 40.492 | 44.414 | 3.0 | 0 | yes | yes | yes |
| `pod16_M64_dense` | pod | 16 | dense | — | — | — | 1e-06 | 61.8330 | 29.0916 | 61.8330 | 61.8330 | 127.522 | 131.465 | 2.0 | 0 | yes | yes | yes |
| `pod32_M128_dense` | pod | 32 | dense | — | — | — | 1e-06 | 47.4440 | 19.1988 | 47.4440 | 47.4440 | 247.828 | 251.734 | 2.0 | 0 | yes | yes | yes |
| `pod64_M256_dense` | pod | 64 | dense | — | — | — | 1e-06 | 20.3198 | 7.2900 | 20.3198 | 20.3198 | 488.515 | 492.463 | 2.0 | 0 | yes | no | yes |
| `pod128_M512_dense` | pod | 128 | dense | — | — | — | 1e-06 | 10.5346 | 2.0372 | 10.5346 | 10.5346 | 1206.731 | 1210.684 | 2.0 | 0 | yes | no | yes |
| `pod256_M1024_dense` | pod | 256 | dense | — | — | — | 1e-06 | 4.0010 | 0.9578 | 4.0010 | 4.0010 | 3488.159 | 3492.766 | 2.0 | 0 | yes | no | yes |
| `pod512_M2048_dense` | pod | 512 | dense | — | — | — | 1e-06 | 0.6579 | 0.3328 | 0.6579 | 2.7209 | 11933.903 | 11941.127 | 2.0 | 0 | yes | no | yes |
| `free512_M1024_dense` | free | 512 | dense | — | — | — | 1e-06 | 1.8046 | 0.4912 | 1.8046 | 2.7232 | 10019.307 | 10024.221 | 3.0 | 0 | yes | no | yes |
| `fno-large` | fno | — | — | — | — | — | — | 6.6169 | 6.6169 | 0.0000 | 5.6812 | 22.476 | 23.221 | — | — | — | — | yes |
| `dense_tight` | fom | — | — | — | — | — | — | 0.0000 | 0.0000 | 0.0000 | 2.7025 | 147.433 | 151.423 | 2.0 | — | — | — | yes |
| `fft_tight` | fom | — | — | — | — | — | — | 0.0000 | 0.0000 | 0.0000 | 2.7025 | 157.688 | 161.759 | 2.0 | — | — | — | yes |
| `nt1e-2_dt005` | fom | — | — | — | — | — | — | 4.0614 | 4.0614 | 0.0000 | 2.0638 | 24.371 | 28.254 | 1.0 | — | — | — | yes |
| `nt1e-2_dt01` | fom | — | — | — | — | — | — | 3.3550 | 3.3550 | 0.0000 | 3.3046 | 13.851 | 17.840 | 1.0 | — | — | — | yes |
| `nt1e-3_dt005` | fom | — | — | — | — | — | — | 0.0516 | 0.0516 | 0.0000 | 2.7116 | 52.915 | 56.513 | 1.0 | — | — | — | yes |
| `nt1e-3_dt01` | fom | — | — | — | — | — | — | 1.5893 | 1.5893 | 0.0000 | 4.1814 | 33.716 | 37.692 | 1.0 | — | — | — | yes |
| `nt1e-4_dt005` | fom | — | — | — | — | — | — | 0.0341 | 0.0341 | 0.0000 | 2.6975 | 61.194 | 65.395 | 1.0 | — | — | — | yes |
| `nt1e-4_dt01` | fom | — | — | — | — | — | — | 1.5869 | 1.5869 | 0.0000 | 4.1611 | 48.432 | 52.335 | 1.5 | — | — | — | yes |

Non-dominated (median_gpu_ms, worst_all_times_percent), admissible: `nt1e-2_dt01`, `nt1e-3_dt01`, `nt1e-4_dt01`, `nt1e-3_dt005`, `nt1e-4_dt005`, `dense_tight`, `fft_tight`.
Non-dominated (median_gpu_ms, worst_evolved_percent), admissible: `nt1e-2_dt01`, `nt1e-3_dt01`, `nt1e-4_dt01`, `nt1e-3_dt005`, `nt1e-4_dt005`, `dense_tight`, `fft_tight`.
Non-dominated (median_host_ms, worst_all_times_percent), admissible: `nt1e-2_dt01`, `nt1e-3_dt01`, `nt1e-4_dt01`, `nt1e-3_dt005`, `nt1e-4_dt005`, `dense_tight`, `fft_tight`.
Non-dominated (median_host_ms, worst_evolved_percent), admissible: `nt1e-2_dt01`, `nt1e-3_dt01`, `nt1e-4_dt01`, `nt1e-3_dt005`, `nt1e-4_dt005`, `dense_tight`, `fft_tight`.

Ladder `dense`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [2.1376, 1.5071, 1.3666, 1.2015, 0.989, 0.5625]; GPU ms [923.2, 1190.8, 1502.4, 2221.7, 4541.3, 16219.9]; monotone evolved yes, all converged yes, converged non-dominated points 6, error span 3.800, cost span 17.569.
Ladder `eq_eqtopxfer_g0p001`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [2.1387, 1.4538, 1.406, 1.1995, 0.9902, 0.551]; GPU ms [43.8, 62.5, 72.1, 115.1, 200.1, 486.3]; monotone evolved yes, all converged yes, converged non-dominated points 5, error span 3.882, cost span 11.098.
Ladder `eq_eqtopxfer_g1em06`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [2.1378, 1.4537, 1.4055, 1.1994, 0.9901, 0.551]; GPU ms [59.1, 79.2, 93.9, 150.2, 262.4, 783.3]; monotone evolved yes, all converged yes, converged non-dominated points 5, error span 3.880, cost span 13.259.
Ladder `eq_eqxfer_g0p001`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [2.1409, 1.4917, 1.3673, 1.2488, 0.9895, 1.8179]; GPU ms [44.1, 62.5, 74.4, 89.8, 188.4, 395.7]; monotone evolved no, all converged yes, converged non-dominated points 3, error span 1.566, cost span 1.686.
Ladder `eq_eqxfer_g1em06`: rungs [0, 16, 32, 64, 128, 256]; worst evolved % [2.14, 1.4914, 1.3672, 1.2485, 0.9894, 1.8204]; GPU ms [58.0, 79.6, 96.7, 120.8, 242.0, 686.6]; monotone evolved no, all converged yes, converged non-dominated points 3, error span 1.565, cost span 1.667.
Ladder `pod`: rungs [16, 32, 64, 128, 256, 512]; worst evolved % [29.0916, 19.1988, 7.29, 2.0372, 0.9578, 0.3328]; GPU ms [127.5, 247.8, 488.5, 1206.7, 3488.2, 11933.9]; monotone evolved yes, all converged yes, converged non-dominated points 6, error span 87.420, cost span 93.583.

Transferred rules (`eqxfer`): q=0 m=922 ρmax 0.0185 ρ95 0.0166 → primary; q=16 m=911 ρmax 0.0499 ρ95 0.0370 → primary; q=32 m=973 ρmax 0.0152 ρ95 0.0126 → primary; q=64 m=937 ρmax 0.1303 ρ95 0.1215 → none; q=128 m=1644 ρmax 0.1406 ρ95 0.1170 → none; q=256 m=1663 ρmax 0.3253 ρ95 0.2925 → none; q=0 m=914 ρmax 0.0186 ρ95 0.0175 → primary; q=16 m=879 ρmax 0.0825 ρ95 0.0350 → primary; q=32 m=848 ρmax 0.1319 ρ95 0.0765 → secondary; q=64 m=1898 ρmax 0.1791 ρ95 0.1647 → none; q=128 m=2152 ρmax 0.0532 ρ95 0.0259 → primary; q=256 m=2438 ρmax 0.0364 ρ95 0.0231 → primary.

FNO `fno-large` in the same allocation: pooled device query median 22.476 ms; second process in the same Slurm allocation on the same GPU, after the JAX phase exited; the source protocol times all models in one process.

Gates: `complete` yes; `backend_gpu` yes; `x64` yes; `precision_highest` yes; `bank_frozen` yes; `checkpoint_unchanged` yes; `final_cohort_unopened` yes; `step_budget_600` yes; `evaluation_cohort_bitwise_abl01` yes; `directions_file_sha256` yes; `directions_prefix_hashes` yes; `repetition_output_identical` yes; `fft_tight_converged_everywhere` yes; `direct_reproduces_fft_tight` yes; `fast_parity` yes; `fno_cohort_disjoint_from_training` yes; `transfer_fit_cert_disjoint` yes; `reference_residuals` yes; `every_rule_hash_matches_provenance` yes; `matched_rule_files_bitwise` no; `no_subject_dropped` yes; `every_subject_case_has_all_reps` yes; `every_invocation_paired` yes; `overdetermined_weak_system` yes; `every_rom_carries_exit_and_stationarity` yes; `host_time_covers_gpu_time` yes; `artifacts_present` yes; `same_grid_baseline_present` yes; `recorded_errors_recomputed_from_saved_fields` yes; `matched_rule_files_bitwise_recomputed` no; `fno_returns_supplied_field_at_t0` yes; `fno_timed_in_same_allocation` yes; `convergence_flags_reproduced` yes.

**What was wrong and is retracted.** Nothing numerical retracted. One pre-registered gate FAILED and is reported as failed: matched_rule_files_bitwise on all six same-file pairs (q=0,16,32, both tolerances) — mis-scoped for TRANSFERRED sets, whose weights are refit on independent fit-state draws; its failure measures transfer draw variance (q=32: same source file, eqxfer rho_max 0.0152 primary vs eqtopxfer 0.1319 secondary). The §7 falsification clause is not triggered (DESIGN A12).

**Open.** The 512²–1024² gap is where the crossover sits; a 768² or a 1024² A100-80G re-run would place it on shared hardware. Transfer draw variance at fixed source rule is now measured (~9× in rho_max at q=32) and unexplained; a future two-set transfer job should share one fit-state draw per shared file. Codex report audit still owed after 2026-09-19 11:33. Jobs used 6 of 8; nothing submitted.

Source-generated report: `experiments/b-panel/reports/2026-09-17-b-panel.md` (SHA256 `e48997e0b94fe2b73bf34c349d74f313339cc3acad17fd82cb9dc6575add48da`) with `summary.json`, the envelope figures and their point JSONs beside it; generator `reports/generate_panel.py` reads only the audit JSONs.
Raw archive `bpn101` Git-tracked as bounded chunks: whole SHA256 `db2fcb6f1fd1c75024455c963eda8619ac2ba38ecdbae97986cfa3abf2cfd1ab` (15 chunks).
Raw archive `bpn203` Git-tracked as bounded chunks: whole SHA256 `dedab45db81d35dbeaa47e22b95cece13352217cdd9d417390bec7bf6e118c68` (185 chunks).
Raw archive `bpn301` Git-tracked as bounded chunks: whole SHA256 `4c788e051fe7813086a86ad28369373a1ad4188938c15cfb5fb2814555a75c42` (19 chunks).
Raw archive `bpn401` Git-tracked as bounded chunks: whole SHA256 `0e92d4291ccc88e33d5ae79214924d71684f6c32325c5ed36fafb8e71351ebcb` (73 chunks).
