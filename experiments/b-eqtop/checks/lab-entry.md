## 2026-09-17

### b-eqtop — CLOSED — the primary EQ ladder is monotone as built, but its certification does not survive the draw replication: at $q = 256$ the ladder's rule is 1 of 5 draws of its construction to meet the bar (marginal at m=2048 (1/5)); $q = 0$ confirmed (3/3); $q = 16$ confirmed (3/3); $q = 32$ confirmed (2/2); $q = 64$ marginal at m=1024 (2/6); $q = 128$ marginal at m=2048 (4/5)

**Closing entry.** `bet301` (job 3783811, DESIGN §A2, four independent draws of (candidate pool, fit-state subset) at $q \in \{64, 128, 256\}$ for the incumbent construction at $m = 1024$ and the 64-fit-state construction at $m = 2048$) landed: `NVIDIA A100-PCIE-40GB`, source `b2844c607290d08d255092ba780ff8ae8a9a0a48`, elapsed 7776 s, `jax_backend=gpu`, no blocking gate failed, the 18 `qrg304` rules re-certified to 9.1e-09 relative on that card; checksum-collected, NumPy-audited, archived (`experiments/b-eqtop/artifacts/bet301`), the whole cluster namespace deleted. The rule DESIGN §A2 pre-registered is applied literally: a construction that certifies on some draws and not others is *marginal at that $m$*.

**Verdict per rung** (the rule the timed ladder ran in `bet101`; draws = every independent draw of its construction across `qrg304`, `bet101`, `bet201`, `bet301`; the exported rule is `certified-rules/` under the §A4 policy):

| $q$ | ladder rule | draws certifying | $\rho_{\max}$ min / median / max | status | exported rule | its status |
|---|---|---|---|---|---|---|
| 0 | qrg304 reachable (pool 8192) $m$=1024, 64 st., 0.0153 | 3/3 | 0.0153 / 0.0255 / 0.0424 | **confirmed (3/3)** | qrg304 reachable (pool 8192) $m$=1024, 64 st., 0.0153 | confirmed (3/3) |
| 16 | qrg304 reachable (pool 8192) $m$=1024, 64 st., 0.0935 | 3/3 | 0.0118 / 0.0751 / 0.0935 | **confirmed (3/3)** | qrg304 reachable (pool 8192) $m$=1024, 64 st., 0.0935 | confirmed (3/3) |
| 32 | qrg304 reachable (pool 8192) $m$=1024, 42 st., 0.0533 | 2/2 | 0.0481 / 0.0507 / 0.0533 | **confirmed (2/2)** | qrg304 reachable (pool 8192) $m$=1024, 42 st., 0.0533 | confirmed (2/2) |
| 64 | qrg304 reachable (pool 8192) $m$=1024, 25 st., 0.0531 | 2/6 | 0.0531 / 0.1320 / 0.1587 | **marginal at m=1024 (2/6)** | bet201 std $m$=2048, 25 st., 0.0583 | confirmed (2/2) |
| 128 | bet101 fs64 $m$=2048, 64 st., 0.0669 | 4/5 | 0.0426 / 0.0669 / 0.2040 | **marginal at m=2048 (4/5)** | bet101 rhow64 $m$=2319, 64 st., 0.0277 | certified in one draw |
| 256 | bet101 fs64 $m$=2048, 64 st., 0.1074 | 1/5 | 0.1074 / 0.1820 / 0.2421 | **marginal at m=2048 (1/5)** | bet101 fs64 $m$=2560, 64 st., 0.0647 | certified in one draw |

**The pre-registered replication itself** (`bet301`; four draws per configuration, seed order):

| $q$ | $m$ | fit states | each $\rho_{\max}$ | min / max | spread | certify primary | certify tight |
|---|---|---|---|---|---|---|---|
| 128 | 1024 | 14 | 0.3523, 0.2098, 0.3357, 0.4609 | 0.2098 / 0.4609 | 2.20× | 0/4 | 0/4 |
| 128 | 2048 | 64 | 0.0497, 0.0426, 0.0788, 0.2040 | 0.0426 / 0.2040 | 4.79× | 3/4 | 2/4 |
| 256 | 1024 | 8 | 0.6743, 0.5737, 0.7319, 0.5075 | 0.5075 / 0.7319 | 1.44× | 0/4 | 0/4 |
| 256 | 2048 | 64 | 0.1820, 0.1899, 0.1299, 0.2421 | 0.1299 / 0.2421 | 1.86× | 0/4 | 0/4 |
| 64 | 1024 | 25 | 0.1420, 0.1509, 0.1587, 0.0670 | 0.0670 / 0.1587 | 2.37× | 1/4 | 0/4 |
| 64 | 2048 | 64 | 0.0459, 0.1036, 0.0134, 0.0108 | 0.0108 / 0.1036 | 9.57× | 4/4 | 3/4 |

**Exported rule set** (`experiments/b-eqtop/certified-rules/`, status: final after bet301 (job 3783811): q<=64 confirmed constructions; q=128 and q=256 single-draw rules above the marginal m=2048 (DESIGN A4)): $q=0$ `rule_q0_m1024_qrg304_reachable.npz` $m=1024$, $\rho_{\max}=0.0153$ [confirmed (3/3)]; $q=16$ `rule_q16_m1024_qrg304_reachable.npz` $m=1024$, $\rho_{\max}=0.0935$ [confirmed (3/3)]; $q=32$ `rule_q32_m1024_qrg304_reachable.npz` $m=1024$, $\rho_{\max}=0.0533$ [confirmed (2/2)]; $q=64$ `rule_q64_m2048_bet201_std.npz` $m=2048$, $\rho_{\max}=0.0583$ [confirmed (2/2)]; $q=128$ `rule_q128_m2319_bet101_rhow64.npz` $m=2319$, $\rho_{\max}=0.0277$ [certified in one draw]; $q=256$ `rule_q256_m2560_bet101_fs64.npz` $m=2560$, $\rho_{\max}=0.0647$ [certified in one draw]. Superseded and removed: `rule_q128_m2048_bet101_fs64.npz`, `rule_q256_m2048_bet101_fs64.npz`, `rule_q64_m1024_qrg304_reachable.npz`.

Worktree `worktrees/2026-09-17-b-eqtop`, branch `exp/2026-09-17-b-eqtop`, forked from `exp/2026-09-16-q-ridge` at `7dc970fc`. Namespace `/cluster/tufts/paralab/tawal01/b_eqtop_20260917`. Jobs: `bet101` = 3780164 (a100:1 constraint a100-80G, 16 cpus, 180G, 20h); `bet201` = 3780165 (a100:1 (any), 16 cpus, 180G, 20h); `bet301` = 3783811 (a100:1 (any), 16 cpus, 180G, 20h). Job 1 ran on `NVIDIA A100 80GB PCIe`, source `ace8936e42b3558ccda39ab9cc6ee38a8dec407d`, elapsed 16352 s; job 2 on `NVIDIA A100 80GB PCIe`, source `ace8936e42b3558ccda39ab9cc6ee38a8dec407d`, elapsed 6977 s. Both printed `jax_backend=gpu`, float64, highest matmul precision; checksum-collected, NumPy-audited, archived as Git chunks, remote directories deleted. Design and amendments: `experiments/b-eqtop/DESIGN.md`. Codex was unavailable (quota) for the pre-job audit; a written self-audit stands in (`experiments/b-eqtop/reports/self-audit-design.md`, DESIGN A1).

**Failed blocking gates:** job 1 none; job 2 none.

**Certification at the top rungs (bet101 draw)** (held-out $\rho_{\max}$ over 512 reachable states, bar 0.116 primary / 0.06 tight; this job's rules only; `qrg304` rules re-certified to 1.1e-09 relative):

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

**EQ, cheapest primary-certified rule per rung (secondary fallback where none)** — monotone evolved **yes**, all-times yes, converged yes, EQ cheaper than dense twin yes; q = [0, 16, 32, 64, 128, 256]; evolved % = 1.8891 / 1.4270 / 1.2493 / 1.2275 / 0.8936 / 0.5389; all-times % = 2.5629 / 2.4806 / 2.3534 / 2.1489 / 1.8116 / 0.9053; median GPU ms = 59 / 81 / 98 / 121 / 247 / 722; m = [1024, 1024, 1024, 1024, 2048, 2048]; $\rho_{\max}$ = 0.0153 / 0.0935 / 0.0533 / 0.0531 / 0.0669 / 0.1074.

**EQ, cheapest tight-certified rule per rung** — monotone evolved **yes**, all-times yes, converged yes, EQ cheaper than dense twin yes; q = [0, 16, 32, 64, 128]; evolved % = 1.8891 / 1.4259 / 1.2493 / 1.2275 / 0.8944; all-times % = 2.5629 / 2.4806 / 2.3534 / 2.1489 / 1.8116; median GPU ms = 59 / 96 / 98 / 121 / 251; m = [1024, 2048, 1024, 1024, 2048]; $\rho_{\max}$ = 0.0153 / 0.0452 / 0.0533 / 0.0531 / 0.0472.

**hybrid: primary-certified EQ where it exists, dense elsewhere** — monotone evolved **yes**, all-times yes, converged yes, EQ cheaper than dense twin yes; q = [0, 16, 32, 64, 128, 256]; evolved % = 1.8891 / 1.4270 / 1.2493 / 1.2275 / 0.8936 / 0.5389; all-times % = 2.5629 / 2.4806 / 2.3534 / 2.1489 / 1.8116 / 0.9053; median GPU ms = 59 / 81 / 98 / 121 / 247 / 722; m = [1024, 1024, 1024, 1024, 2048, 2048]; $\rho_{\max}$ = 0.0153 / 0.0935 / 0.0533 / 0.0531 / 0.0669 / 0.1074.

**dense (exact) quadrature twins** — monotone evolved **yes**, all-times yes, converged yes, EQ cheaper than dense twin no; q = [0, 64, 128, 256]; evolved % = 1.8890 / 1.0843 / 0.8930 / 0.5194; all-times % = 2.5629 / 2.1489 / 1.8116 / 0.9053; median GPU ms = 293 / 625 / 1190 / 3915; m = [None, None, None, None]; $\rho_{\max}$ = — / — / — / —.

**Top rung $q=256$:** primary arm $m=2048$, $\rho_{\max}=0.1074$, evolved 0.5389 % vs dense 0.5194 %. $\rho$ against evolved error at $q=256$: q256_eq_primary m=2048 rho_max=0.1074 -> 0.5389 %; q256_eq_qrg304_m2048 m=2048 rho_max=0.1678 -> 1.0361 %.

**Cross-job fidelity vs qrg304:** `q0_dense` passed (3.6e-13 evolved); `q0_eq_primary` passed (7.3e-14 evolved); `q128_eq_qrg304_m2048` passed (1.9e-10 evolved); `q16_eq_primary` passed (1.0e-10 evolved); `q256_dense` passed (6.2e-09 evolved); `q256_eq_qrg304_m2048` passed (8.5e-10 evolved); `q32_eq_primary` passed (6.1e-11 evolved); `q64_dense` passed (1.5e-10 evolved); `q64_eq_primary` passed (2.5e-10 evolved).

**Gates, job 1:** `archived_rules_recertify` yes; `arm_aliases` —; `artifacts_present` yes; `backend_gpu` yes; `bank_frozen` yes; `bank_sha256_consistent` yes; `certification_flags_recomputed` yes; `checkpoint_unchanged` yes; `collection_pools_bitwise_qrg304` no; `complete` yes; `cross_job_fidelity` yes; `decoded_fields_match_saved_outputs` yes; `directions_hash_matches_qrg304` no; `evaluation_cohort_bitwise_abl01` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_rule_archived` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `fit_and_certification_trajectories_disjoint` yes; `fit_phase_complete` yes; `no_arm_dropped` yes; `no_worker_failed` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_residuals` yes; `repetition_output_identical` yes; `reproduces_q0_dense` yes; `reproduces_q0_eq_primary` yes; `reproduces_q128_eq_qrg304_m2048` yes; `reproduces_q16_eq_primary` yes; `reproduces_q256_dense` yes; `reproduces_q256_eq_qrg304_m2048` yes; `reproduces_q32_eq_primary` yes; `reproduces_q64_dense` yes; `reproduces_q64_eq_primary` yes; `same_grid_baseline_present` yes; `step_budget_600` yes; `x64` yes.

**Gates, job 2:** `archived_rules_recertify` yes; `backend_gpu` yes; `bank_frozen` yes; `certification_flags_recomputed` yes; `collection_pools_bitwise_qrg304` no; `complete` yes; `directions_hash_matches_qrg304` no; `evaluation_cohort_bitwise_abl01` yes; `every_rule_archived` yes; `final_cohort_unopened` yes; `fit_and_certification_trajectories_disjoint` yes; `fit_phase_complete` yes; `no_worker_failed` yes; `precision_highest` yes; `step_budget_600` yes; `x64` yes.

**Gates, job 3:** `archived_rules_recertify` yes; `backend_gpu` yes; `bank_frozen` yes; `certification_flags_recomputed` yes; `collection_pools_bitwise_qrg304` no; `complete` yes; `directions_hash_matches_qrg304` no; `evaluation_cohort_bitwise_abl01` yes; `every_rule_archived` yes; `final_cohort_unopened` yes; `fit_and_certification_trajectories_disjoint` yes; `fit_phase_complete` yes; `no_worker_failed` yes; `precision_highest` yes; `step_budget_600` yes; `x64` yes.

**What was wrong and retracted / corrected:**

- `q-ridge`'s log–log extrapolation that the $q = 256$ primary crossing lies at $m \approx 2522$ (and $\alpha = 1.772$) is retracted as a prediction: it was fitted through two points whose draw noise `bet201` measured at $0.11\times$–$2.3\times$ and `bet301` at $1.4\times$–$9.6\times$ within four draws of one construction. The binding constraint at the top rungs was the fit-state count, not $m$, but 64 fit states at $m = 2048$ do not *reliably* certify either: at $q = 256$ that construction met the bar in 1 of 5 draws.
- The incumbent fit-state convention $\mathrm{clip}(8192/M, 8, 64)$ inherited from `q-ridge` is withdrawn for $q \ge 128$: 14 and 8 states do not see the tail that $\rho_{\max}$ measures.
- DESIGN §9 sized the `std` chains to $m = 6144$ by a walltime model. The walltime was never binding: the fitter stopped on `gradient` (no improving candidate) at $m \approx 3055$–$3128$ ($q = 128$, 8064 rows) and $3370$–$3410$ ($q = 256$, 8704 rows), so the declared targets 4096 and 6144 were unreachable for a design with that few rows. The rules are untruncated and count; the grid, not the science, was mis-sized.
- Within one chain $\rho_{\max}$ is not monotone in $m$ ($q = 128$ `std`: 0.1091 at $m = 3088$, certified; 0.1487 at $m = 3128$, not). The single `std` certification at $q = 128$ is therefore marginal and is not the rule the ladder ran.
- `bet201` at $q = 64$: the archived $m = 1024$ rule certifies (0.0531) but the same construction re-drawn gives 0.1220 (not certified), and the 64-fit-state `fs64` chain never certified there (0.1303, 0.1248) while `rhow64` did (0.1039, 0.0518). '64 fit states' is not uniformly better; the draw is a first-order effect, which is why every certified flag here is provisional until `bet301`.
- **The interim entry's and `bet101`'s "primary-certified rule at every rung" is withdrawn as a statement about the construction.** The top-rung rule (`fs64` $m = 2048$, $\rho_{\max} = 0.1074$) is the only one of five draws of its recipe to meet the primary bar: the four `bet301` re-draws give 0.1820, 0.1899, 0.1299, 0.2421. Per DESIGN §A2 the rung is *marginal at $m = 2048$*; the paper may cite $m = 2560$ (0.0647, one draw, `fs64`) only as a single-draw rule. At $q = 128$ the recipe met the bar in 4 of 5 draws (0.0426–0.2040); at $q = 64$ the rule the ladder ran (incumbent, $m = 1024$, 0.0531) is 1 of 4 re-draws / 2 of 6 draws (0.0531–0.1587). Only $q \le 32$ are confirmed (2 or 3 of 2 or 3 draws).
- The exported rule set `certified-rules/` of the interim entry (the `bet101` in-job choice: archived $m = 1024$ at $q \le 64$, `fs64` $m = 2048$ at $q = 128, 256$) is superseded: the $q = 64$, $128$, $256$ files are removed and replaced under the §A4 policy by `bet201` `std` $m = 2048$ (0.0583; confirmed 2/2), `bet101` `rhow64` $m = 2319$ (0.0277; one draw) and `bet101` `fs64` $m = 2560$ (0.0647; one draw). The timed-ladder errors in the report were measured with the superseded rules; `b-panel`'s re-run with the new set is the measurement that replaces them.
- The monotone primary ladder ($1.8891 \to 0.5389$ %) stands *as measured with the rules as built*, but it was measured only with draws that passed the bar. Whether a failing draw ($\rho_{\max} \approx 0.2$ at $q = 256$) also gives a monotone ladder was not measured; the ladder claim in the paper must be tied to the specific rules (by SHA256) and not to "certified EQ".
- Operational: `bet301` landed on a 40 GB A100 (the sbatch had no 80 GB constraint because the job has no timed phase) and the `.err` carries four XLA BFC-allocator `ran out of memory trying to allocate 8.50GiB` warnings during certification; all were recovered (every rule certified, the archive re-certified to $9.1\times10^{-9}$, `ALL-DONE`). Recorded so that the checklist's "no OOMs" line is read against the facts, not assumed.
- No job was retracted (3 of 8 used: `bet101`, `bet201`, `bet301`).

**Open:**

- **What would settle the top rung** (not run; the lane closed at 3 of 8 jobs): (a) four draws of `fs64` at $q = 256$, $m = 2560$ and $3072$, and of `rhow64` at $q = 128$, $m = 2560$ — the exported single-draw rules' constructions have no measured re-draw; (b) a timed ladder that runs a *failing* draw at the top rung (e.g. `bet301` draw 4, $\rho_{\max} = 0.2421$) beside the passing one, to learn whether the bar predicts the evolved error at all — the $\rho$-vs-error table has only three points at $q = 256$ (0.1678 → 1.0361 %, 0.1074 → 0.5389 %, dense → 0.5194 %); (c) a second held-out draw of the 512 certification states, still unmeasured.
- No tight-certified rule at $q = 256$ (best 0.0647 at `fs64` $m = 2560$, one draw). One `fs64` fit at $m = 3072$ would say whether the tight bar is reachable; it would still be one draw.
- At $q = 64$ the ladder's certified EQ rung costs $+0.14$ pp evolved error against its dense twin (1.2275 vs 1.0843 %), the largest EQ-vs-dense gap on the ladder, and its rule is now marginal; the exported $m = 2048$ rule (0.0583) has not been run in a timed ladder.
- Codex report audit after 2026-09-19 11:33 if anyone reopens the lane; `reports/self-audit-report.md` (21 checks, the replication counts recounted from the raw `result.json`) stands in.
- `b-panel` drops `experiments/b-eqtop/certified-rules/` into its full $256^2$ re-run; the `construction.status` field must travel with each rule and the paper must say which rungs are single-draw.

Source-generated report: `experiments/b-eqtop/reports/2026-09-17-b-eqtop.md` (SHA256 `fcfb36bf6ea4f0a5f68229ca7fde2f5a8e415834f38c0c86a924415748463347`) with `summary.json` and its generator beside it; audits `experiments/b-eqtop/checks/bet101-audit.json`, `bet201-audit.json`, `bet301-audit.json`; self-audit of the report in place of Codex: `experiments/b-eqtop/reports/self-audit-report.md`; archives `experiments/b-eqtop/artifacts/bet101`, `bet201`, `bet301`; exported rule set for `b-panel`: `experiments/b-eqtop/certified-rules/` (`PROVENANCE.json`, `SHA256SUMS`); the draw bookkeeping shared by report, export, self-audit and this entry: `experiments/b-eqtop/draws.py`. Jobs used: 3 of 8, none retracted.
