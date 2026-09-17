## 2026-09-17

### b-eqtop — INTERIM (bet301 pending) — every rung primary-certified and the primary EQ ladder monotone on evolved times in this single draw

**Interim entry.** `bet301` (job 3783811, DESIGN A2 draw replication, four independent pool/fit-state draws at q in {64,128,256}) is still running. DESIGN §A2 showed that the same rule construction at the same $m$ moves $\rho_{\max}$ by $0.11\times$–$2.3\times$ under an independent draw of (candidate pool, fit-state subset) and flips certification at $q = 64$, so every certified flag below is a single draw and is **provisional** until that job bounds the spread. No certified ladder is claimed here; a closing entry follows when it lands.

Worktree `worktrees/2026-09-17-b-eqtop`, branch `exp/2026-09-17-b-eqtop`, forked from `exp/2026-09-16-q-ridge` at `7dc970fc`. Namespace `/cluster/tufts/paralab/tawal01/b_eqtop_20260917`. Jobs: `bet101` = 3780164 (a100:1 constraint a100-80G, 16 cpus, 180G, 20h); `bet201` = 3780165 (a100:1 (any), 16 cpus, 180G, 20h); `bet301` = 3783811 (a100:1 (any), 16 cpus, 180G, 20h). Job 1 ran on `NVIDIA A100 80GB PCIe`, source `ace8936e42b3558ccda39ab9cc6ee38a8dec407d`, elapsed 16352 s; job 2 on `NVIDIA A100 80GB PCIe`, source `ace8936e42b3558ccda39ab9cc6ee38a8dec407d`, elapsed 6977 s. Both printed `jax_backend=gpu`, float64, highest matmul precision; checksum-collected, NumPy-audited, archived as Git chunks, remote directories deleted. Design and amendments: `experiments/b-eqtop/DESIGN.md`. Codex was unavailable (quota) for the pre-job audit; a written self-audit stands in (`experiments/b-eqtop/reports/self-audit-design.md`, DESIGN A1).

**Failed blocking gates:** job 1 none; job 2 none.

**Certification at the top rungs** [provisional: `bet301` 3783811 pending] (held-out $\rho_{\max}$ over 512 reachable states, bar 0.116 primary / 0.06 tight; this job's rules only; `qrg304` rules re-certified to 1.1e-09 relative):

| $q$ | arm | $m$ | fit states | NNLS fit | $\rho_{\max}$ | $\rho_{95}$ | primary | tight | fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| 128 | fs64 | 2048 | 64 | 3.52e-04 | 0.0669 | 0.0214 | yes | no | 2920 |
| 128 | fs64 | 2560 | 64 | 1.82e-04 | 0.0292 | 0.0111 | yes | yes | 3471 |
| 128 | rhow64 | 2048 | 64 | 3.54e-04 | 0.0472 | 0.0176 | yes | yes | 2698 |
| 128 | rhow64 | 2319 | 64 | 2.52e-04 | 0.0277 | 0.0102 | yes | yes | 2747 |
| 128 | std | 1024 | 14 | 1.49e-03 | 0.2793 | 0.1615 | no | no | 469 |
| 128 | std | 2048 | 14 | 7.98e-05 | 0.2373 | 0.0520 | no | no | 1339 |
| 128 | std | 2560 | 14 | 3.14e-05 | 0.2040 | 0.0308 | no | no | 1936 |
| 128 | std | 3055 | 14 | 1.40e-05 | 0.1193 | 0.0197 | no | no | 2370 |
| 128 | std | 3088 | 14 | 1.35e-05 | 0.1091 | 0.0183 | yes | no | 1888 |
| 128 | std | 3128 | 14 | 1.33e-05 | 0.1487 | 0.0161 | no | no | 1797 |
| 256 | fs64 | 2048 | 64 | 2.41e-03 | 0.1074 | 0.0878 | yes | no | 2650 |
| 256 | fs64 | 2560 | 64 | 9.35e-04 | 0.0647 | 0.0314 | yes | no | 3993 |
| 256 | rhow64 | 2048 | 64 | 1.44e-03 | 0.1299 | 0.0836 | no | no | 3020 |
| 256 | rhow64 | 2560 | 64 | 6.23e-04 | 0.1018 | 0.0380 | yes | no | 3793 |
| 256 | rhow64 | 3072 | 64 | 3.37e-04 | 0.0963 | 0.0229 | yes | no | 4552 |
| 256 | std | 1024 | 8 | 1.61e-02 | 0.4977 | 0.4450 | no | no | 310 |
| 256 | std | 2048 | 8 | 1.75e-04 | 0.2958 | 0.1822 | no | no | 2056 |
| 256 | std | 2560 | 8 | 5.84e-05 | 0.1956 | 0.1052 | no | no | 2775 |
| 256 | std | 3072 | 8 | 2.32e-05 | 0.1774 | 0.0722 | no | no | 3117 |
| 256 | std | 3370 | 8 | 1.45e-05 | 0.1308 | 0.0685 | no | no | 3010 |
| 256 | std | 3410 | 8 | 1.43e-05 | 0.1761 | 0.0836 | no | no | 2330 |

**Laws $\rho_{\max}\propto m^{-\alpha}$ and the $m$ each bar needs:**

| $q$ | arm | $\alpha$ | $m$ for 0.116 (law) | cheapest certified $m$ | $m$ for 0.06 (law) | job |
|---|---|---|---|---|---|---|
| 128 | fs64 | 3.718 | 1766 | 2048 | 2109 | 3780164 |
| 128 | rhow64 | 4.295 | 1661 | 2048 | 1937 | 3780164 |
| 128 | std | 0.738 | 3978 | 3088 | 9725 | 3780164 |
| 256 | fs64 | 2.268 | 1979 | 2048 | 2647 | 3780164 |
| 256 | rhow64 | 0.751 | 2307 | 2560 | 5552 | 3780164 |
| 256 | std | 1.003 | 4565 | — | 8809 | 3780164 |
| 0 | fs64 | 2.126 | 638 | 1024 | 870 | 3780165 |
| 0 | rhow64 | 15.822 | 589 | 609 | 614 | 3780165 |
| 0 | std | 1.830 | 449 | 512 | 643 | 3780165 |
| 16 | fs64 | 2.344 | 851 | 1024 | 1127 | 3780165 |
| 16 | rhow64 | 0.894 | 468 | 946 | 978 | 3780165 |
| 16 | std | 2.207 | 440 | 512 | 594 | 3780165 |
| 32 | fs64 | 0.739 | 514 | 1024 | 1253 | 3780165 |
| 32 | rhow64 | 0.860 | 275 | 1024 | 591 | 3780165 |
| 32 | std | 2.120 | 637 | 1024 | 869 | 3780165 |
| 64 | fs64 | 0.062 | 6592 | — | 260093657 | 3780165 |
| 64 | rhow64 | 1.646 | 958 | 1024 | 1430 | 3780165 |
| 64 | std | 1.507 | 978 | 2048 | 1515 | 3780165 |

**EQ, cheapest primary-certified rule per rung (secondary fallback where none)** [provisional: `bet301` 3783811 pending] — monotone evolved **yes**, all-times yes, converged yes, EQ cheaper than dense twin yes; q = [0, 16, 32, 64, 128, 256]; evolved % = 1.8891 / 1.4270 / 1.2493 / 1.2275 / 0.8936 / 0.5389; all-times % = 2.5629 / 2.4806 / 2.3534 / 2.1489 / 1.8116 / 0.9053; median GPU ms = 59 / 81 / 98 / 121 / 247 / 722; m = [1024, 1024, 1024, 1024, 2048, 2048]; $\rho_{\max}$ = 0.0153 / 0.0935 / 0.0533 / 0.0531 / 0.0669 / 0.1074.

**EQ, cheapest tight-certified rule per rung** [provisional: `bet301` 3783811 pending] — monotone evolved **yes**, all-times yes, converged yes, EQ cheaper than dense twin yes; q = [0, 16, 32, 64, 128]; evolved % = 1.8891 / 1.4259 / 1.2493 / 1.2275 / 0.8944; all-times % = 2.5629 / 2.4806 / 2.3534 / 2.1489 / 1.8116; median GPU ms = 59 / 96 / 98 / 121 / 251; m = [1024, 2048, 1024, 1024, 2048]; $\rho_{\max}$ = 0.0153 / 0.0452 / 0.0533 / 0.0531 / 0.0472.

**hybrid: primary-certified EQ where it exists, dense elsewhere** [provisional: `bet301` 3783811 pending] — monotone evolved **yes**, all-times yes, converged yes, EQ cheaper than dense twin yes; q = [0, 16, 32, 64, 128, 256]; evolved % = 1.8891 / 1.4270 / 1.2493 / 1.2275 / 0.8936 / 0.5389; all-times % = 2.5629 / 2.4806 / 2.3534 / 2.1489 / 1.8116 / 0.9053; median GPU ms = 59 / 81 / 98 / 121 / 247 / 722; m = [1024, 1024, 1024, 1024, 2048, 2048]; $\rho_{\max}$ = 0.0153 / 0.0935 / 0.0533 / 0.0531 / 0.0669 / 0.1074.

**dense (exact) quadrature twins** [provisional: `bet301` 3783811 pending] — monotone evolved **yes**, all-times yes, converged yes, EQ cheaper than dense twin no; q = [0, 64, 128, 256]; evolved % = 1.8890 / 1.0843 / 0.8930 / 0.5194; all-times % = 2.5629 / 2.1489 / 1.8116 / 0.9053; median GPU ms = 293 / 625 / 1190 / 3915; m = [None, None, None, None]; $\rho_{\max}$ = — / — / — / —.

**Top rung $q=256$:** [provisional: `bet301` 3783811 pending] primary arm $m=2048$, $\rho_{\max}=0.1074$, evolved 0.5389 % vs dense 0.5194 %. $\rho$ against evolved error at $q=256$: q256_eq_primary m=2048 rho_max=0.1074 -> 0.5389 %; q256_eq_qrg304_m2048 m=2048 rho_max=0.1678 -> 1.0361 %.

**Cross-job fidelity vs qrg304:** `q0_dense` passed (3.6e-13 evolved); `q0_eq_primary` passed (7.3e-14 evolved); `q128_eq_qrg304_m2048` passed (1.9e-10 evolved); `q16_eq_primary` passed (1.0e-10 evolved); `q256_dense` passed (6.2e-09 evolved); `q256_eq_qrg304_m2048` passed (8.5e-10 evolved); `q32_eq_primary` passed (6.1e-11 evolved); `q64_dense` passed (1.5e-10 evolved); `q64_eq_primary` passed (2.5e-10 evolved).

**Gates, job 1:** `archived_rules_recertify` yes; `arm_aliases` —; `artifacts_present` yes; `backend_gpu` yes; `bank_frozen` yes; `bank_sha256_consistent` yes; `certification_flags_recomputed` yes; `checkpoint_unchanged` yes; `collection_pools_bitwise_qrg304` no; `complete` yes; `cross_job_fidelity` yes; `decoded_fields_match_saved_outputs` yes; `directions_hash_matches_qrg304` no; `evaluation_cohort_bitwise_abl01` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_rule_archived` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `fit_and_certification_trajectories_disjoint` yes; `fit_phase_complete` yes; `no_arm_dropped` yes; `no_worker_failed` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_residuals` yes; `repetition_output_identical` yes; `reproduces_q0_dense` yes; `reproduces_q0_eq_primary` yes; `reproduces_q128_eq_qrg304_m2048` yes; `reproduces_q16_eq_primary` yes; `reproduces_q256_dense` yes; `reproduces_q256_eq_qrg304_m2048` yes; `reproduces_q32_eq_primary` yes; `reproduces_q64_dense` yes; `reproduces_q64_eq_primary` yes; `same_grid_baseline_present` yes; `step_budget_600` yes; `x64` yes.

**Gates, job 2:** `archived_rules_recertify` yes; `backend_gpu` yes; `bank_frozen` yes; `certification_flags_recomputed` yes; `collection_pools_bitwise_qrg304` no; `complete` yes; `directions_hash_matches_qrg304` no; `evaluation_cohort_bitwise_abl01` yes; `every_rule_archived` yes; `final_cohort_unopened` yes; `fit_and_certification_trajectories_disjoint` yes; `fit_phase_complete` yes; `no_worker_failed` yes; `precision_highest` yes; `step_budget_600` yes; `x64` yes.

**What was wrong and retracted / corrected:**

- `q-ridge`'s log–log extrapolation that the $q = 256$ primary crossing lies at $m \approx 2522$ (and $\alpha = 1.772$) is retracted as a prediction: it was fitted through two points whose draw noise `bet201` measured at $0.11\times$–$2.3\times$. The binding constraint at the top rungs was the fit-state count, not $m$: with 64 fit states the rule certifies at $m = 2048$ at both $q = 128$ ($\rho_{\max}$ 0.0669) and $q = 256$ (0.1074); with the incumbent 8 states it never does at $q = 256$ in 8 reachable rules up to $m = 3410$ (best 0.1308).
- The incumbent fit-state convention $\mathrm{clip}(8192/M, 8, 64)$ inherited from `q-ridge` is withdrawn for $q \ge 128$: 14 and 8 states do not see the tail that $\rho_{\max}$ measures.
- DESIGN §9 sized the `std` chains to $m = 6144$ by a walltime model. The walltime was never binding: the fitter stopped on `gradient` (no improving candidate) at $m \approx 3055$–$3128$ ($q = 128$, 8064 rows) and $3370$–$3410$ ($q = 256$, 8704 rows), so the declared targets 4096 and 6144 were unreachable for a design with that few rows. The rules are untruncated and count; the grid, not the science, was mis-sized.
- Within one chain $\rho_{\max}$ is not monotone in $m$ ($q = 128$ `std`: 0.1091 at $m = 3088$, certified; 0.1487 at $m = 3128$, not). The single `std` certification at $q = 128$ is therefore marginal and is not the rule the ladder ran.
- `bet201` at $q = 64$: the archived $m = 1024$ rule certifies (0.0531) but the same construction re-drawn gives 0.1220 (not certified), and the 64-fit-state `fs64` chain never certified there (0.1303, 0.1248) while `rhow64` did (0.1039, 0.0518). '64 fit states' is not uniformly better; the draw is a first-order effect, which is why every certified flag here is provisional until `bet301`.
- Nothing in this lane's own numbers was retracted; no job was retracted (3 of 8 used).

**Open:**

- `bet301` (job 3783811): four independent draws at $q \in \{64, 128, 256\}$ for `std` ($m = 1024$) and `fs64` ($m = 2048$). If `fs64` at $q = 256$ certifies in fewer than 4/4 draws, the top rung is reported as *marginal at $m = 2048$* and the paper should cite $m = 2560$ (0.0647) or the draw count; the exported rule set's status field flips only then.
- No tight-certified rule at $q = 256$ (best 0.0647 at `fs64` $m = 2560$; the law puts the crossing near $m \approx 2650$, an order-of-magnitude statement). One `fs64` fit at $m = 3072$ (~1.5 h CPU, one job) would settle it.
- The sampling noise of $\rho_{\max}$ over the 512 held-out states (a second held-out draw) is still unmeasured — self-audit weakness (b).
- At $q = 64$ the certified EQ rung costs $+0.14$ pp evolved error against its dense twin (1.2275 vs 1.0843 %), the largest EQ-vs-dense gap on the ladder; at $q = 256$ it is $+0.02$ pp. Whether a tighter bar at $q = 64$ closes it is untested.
- Codex report audit after 2026-09-19 11:33 if the lane is still open; the written self-audit stands in until then.
- The `b-panel` lane can drop `experiments/b-eqtop/certified-rules/` into its re-run as a second rule set; the status field must be carried with it.

Source-generated report: `experiments/b-eqtop/reports/2026-09-17-b-eqtop.md` (SHA256 `2ab56ab9fbb3f55cd447aa619d2588b42f88568e629dc93b272f1028c067c066`) with `summary.json` and its generator beside it; audits `experiments/b-eqtop/checks/bet101-audit.json`, `bet201-audit.json`; self-audit of the report in place of Codex: `experiments/b-eqtop/reports/self-audit-report.md`; archives `experiments/b-eqtop/artifacts/bet101`, `bet201`; exported rule set for `b-panel`: `experiments/b-eqtop/certified-rules/` (`PROVENANCE.json`, `SHA256SUMS`).
