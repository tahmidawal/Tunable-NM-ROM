## 2026-09-16
### b-ladder-top — the three solver fixes are INERT at $q=256$; the rung converges only by raising the iteration budget, for free; and on the evolved-times metric the ladder is not even monotone

The coordinator asked two questions in order. **Q1:** make the $q=256$ rung of the
fixed-weight correction ladder converge, with three candidate fixes each isolated — a
coarse-to-fine cascade warm start, column equilibration of the augmented Jacobian, and a
damping/trust schedule for the correction block decoupled from the latent block — and a valid
empirical-quadrature rule per rung. **Q2:** price the whole inference-time envelope in ONE job:
the $q$ ladder crossed with the quadrature and the evolution tolerance, against same-job
full-order controls, POD-LSPG and the trained FNO, reporting BOTH the worst-over-all-times and
the worst-over-evolved-times error with the $t=0$ compression separated out. A third job, Q1-B,
was predeclared after Q1 landed to answer the *"at a stated cost"* half of Q1's target.
Predeclared protocol and every amendment: `experiments/b-ladder-top/DESIGN.md`. Nothing was
merged.

Worktree `worktrees/2026-09-16-b-ladder-top`, branch `exp/2026-09-16-b-ladder-top`. Namespace `/cluster/tufts/paralab/tawal01/b_ladder_top_20260916/`. Q1 job `3745589` (`btq101`) on `NVIDIA A100 80GB PCIe`, source `b03fc2e07aa3ed98cacaccaacbd640f078d19f3d`, elapsed 5641.8 s. Q1-B job `3749039` (`btq102`) on `NVIDIA A100 80GB PCIe`, source `4eccb76e3a9f908ad853f467d9043a4db30d0d00`, elapsed 3292.0 s. Q2 job `3747245` (`btq201`) on `NVIDIA A100-PCIE-40GB`, source `a9b99c50cb2e5b274cd02cf5b53dc62cc9eb8c75`, elapsed 6557.5 s. Every job printed `jax_backend=gpu`, ran float64 with highest matmul precision, and were checksum-collected, independently NumPy-audited and archived before their exact remote attempt directories were removed.

**Q1 gates.** `artifacts_present` yes; `backend_gpu` yes; `bank_frozen` yes; `cclad01_fidelity` yes; `checkpoint_unchanged` yes; `complete` yes; `directions_hash_matches_cclad01` yes; `directions_rank_covers_ladder` yes; `every_eq_rule_reports_validity` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `q0_arms_bitwise` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_fields_bitwise_match_cclad01` no; `reference_residuals` yes; `repetition_output_identical` yes; `reproduces_q0_m4_dense_base` yes; `reproduces_q0_m4_eq_base` yes; `reproduces_q128_m2_dense_base` yes; `same_grid_baseline_present` yes; `x64` yes.

**Q1-B gates.** `artifacts_present` yes; `backend_gpu` yes; `bank_frozen` yes; `cclad01_fidelity` yes; `checkpoint_unchanged` yes; `complete` yes; `directions_hash_matches_cclad01` no; `directions_rank_covers_ladder` yes; `every_eq_rule_reports_validity` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_fields_bitwise_match_cclad01` no; `reference_residuals` yes; `repetition_output_identical` yes; `reproduces_q256_m2_dense_base` yes; `same_grid_baseline_present` yes; `x64` yes.

**Q2 gates.** `artifacts_present` yes; `backend_gpu` yes; `bank_frozen` yes; `cclad01_fidelity` yes; `checkpoint_unchanged` yes; `complete` yes; `directions_hash_matches_cclad01` no; `directions_rank_covers_ladder` yes; `every_eq_rule_reports_validity` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `fno_cohort_disjoint_from_training` yes; `fno_returns_supplied_field_at_t0` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_fields_bitwise_match_cclad01` yes; `reference_residuals` yes; `repetition_output_identical` yes; `reproduces_q0_M256_dense_g1em06` yes; `reproduces_q128_M256_dense_g1em06` yes; `same_grid_baseline_present` yes; `x64` yes.

**Fidelity against the cheap-corrections job.**

Q1:

| arm | reproduces | tolerance | relative difference (reference) | relative difference (same-grid) | passed |
|---|---|---|---|---|---|
| `q0_m4_dense_base` | `q0_m4_dense_block` | 1e-09 | 2e-13 | 0e+00 | yes |
| `q0_m4_eq_base` | `q0_m4_eq_varpro` | 1e-09 | 1e-14 | 0e+00 | yes |
| `q128_m2_dense_base` | `q128_m2_dense_block` | 1e-09 | 4e-13 | 1e-13 | yes |

Q1-B:

| arm | reproduces | tolerance | relative difference (reference) | relative difference (same-grid) | passed |
|---|---|---|---|---|---|
| `q256_m2_dense_base` | `q256_m2_dense_block` | 1e-09 | 5e-13 | 9e-14 | yes |

Q2:

| arm | reproduces | tolerance | relative difference (reference) | relative difference (same-grid) | passed |
|---|---|---|---|---|---|
| `q0_M256_dense_g1em06` | `q0_m256_dense_block` | 1e-09 | 5e-13 | 0e+00 | yes |
| `q128_M256_dense_g1em06` | `q128_m256_dense_block` | 1e-03 | 4e-10 | 6e-10 | yes |

**Conditioning of the augmented normal equations** (offline probe, $\lambda=10^{-6}$):

| q | unknowns | column norm ratio | $\kappa$ unscaled | $\kappa$ equilibrated | improvement |
|---|---|---|---|---|---|
| 0 | 16 | 4.231e+00 | 2.968e+05 | 2.582e+05 | 1.149 |
| 64 | 80 | 9.204e+01 | 3.475e+05 | 2.271e+05 | 1.530 |
| 128 | 144 | 8.277e+01 | 6.080e+05 | 4.342e+05 | 1.400 |
| 256 | 272 | 1.066e+02 | 6.139e+06 | 3.492e+06 | 1.758 |
| 512 | 528 | 9.735e+01 | 6.865e+10 | 3.494e+07 | 1964.597 |

**Q1 convergence sweep** (dense quadrature):

| arm | q | fix | median iters/step | budget exits | worst joint gradient | worst same-grid all % | worst evolved % | median GPU ms | cascade total ms | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| `q0_m4_dense_base` | 0 | `base` | 3.0 | 0 | 9.970e-07 | 2.5629 | 1.8890 | 284.991 | — | yes |
| `q0_m4_dense_pre` | 0 | `pre` | 3.0 | 0 | 9.970e-07 | 2.5629 | 1.8890 | 285.330 | — | yes |
| `q0_m4_dense_predamp` | 0 | `predamp` | 3.0 | 0 | 9.970e-07 | 2.5629 | 1.8890 | 284.855 | — | yes |
| `q64_m2_dense_base` | 64 | `base` | 3.0 | 0 | 9.909e-07 | 2.1489 | 1.4480 | 535.900 | — | yes |
| `q64_m2_dense_predamp` | 64 | `predamp` | 3.0 | 0 | 9.909e-07 | 2.1489 | 1.4480 | 534.841 | — | yes |
| `q128_m2_dense_base` | 128 | `base` | 3.0 | 0 | 9.983e-07 | 1.8116 | 0.9799 | 913.684 | — | yes |
| `q128_m2_dense_casc` | 128 | `casc` | 3.0 | 63 | 5.339e-02 | 1.8116 | 0.9799 | 1043.120 | 1577.962 | no |
| `q128_m2_dense_damp` | 128 | `damp` | 3.0 | 0 | 9.983e-07 | 1.8116 | 0.9799 | 913.827 | — | yes |
| `q128_m2_dense_pre` | 128 | `pre` | 3.0 | 0 | 9.983e-07 | 1.8116 | 0.9799 | 914.715 | — | yes |
| `q128_m2_dense_predamp` | 128 | `predamp` | 3.0 | 0 | 9.983e-07 | 1.8116 | 0.9799 | 914.730 | — | yes |
| `q256_m2_dense_base` | 256 | `base` | 5.0 | 15 | 1.191e-03 | 0.9053 | 0.7566 | 3398.977 | — | no |
| `q256_m2_dense_casc` | 256 | `casc` | 7.0 | 180 | 2.212e-02 | 0.9053 | 0.7566 | 12301.559 | 13216.289 | no |
| `q256_m2_dense_damp` | 256 | `damp` | 5.0 | 15 | 1.191e-03 | 0.9053 | 0.7566 | 3413.226 | — | no |
| `q256_m2_dense_pre` | 256 | `pre` | 5.0 | 15 | 1.191e-03 | 0.9053 | 0.7566 | 3399.543 | — | no |
| `q256_m2_dense_predamp` | 256 | `predamp` | 5.0 | 15 | 1.191e-03 | 0.9053 | 0.7566 | 3410.992 | — | no |
| `q512_m2_dense_base` | 512 | `base` | 2.0 | 0 | 1.497e-01 | 0.6027 | 0.4343 | 2402.103 | — | no |
| `q512_m2_dense_casc` | 512 | `casc` | 2.0 | 0 | 1.732e-01 | 0.6027 | 0.4343 | 2372.593 | 5783.585 | no |
| `q512_m2_dense_damp` | 512 | `damp` | 2.0 | 0 | 1.497e-01 | 0.6027 | 0.4343 | 2431.989 | — | no |
| `q512_m2_dense_pre` | 512 | `pre` | 2.0 | 0 | 1.497e-01 | 0.6027 | 0.4343 | 2404.754 | — | no |
| `q512_m2_dense_predamp` | 512 | `predamp` | 2.0 | 0 | 1.497e-01 | 0.6027 | 0.4343 | 2424.614 | — | no |

**Q1 empirical quadrature per rung:**

| arm | q | M | m | relative fit | truncated | rule valid | fit seconds | worst same-grid all % | median GPU ms | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| `q0_m4_eq_base` | 0 | 64 | 256 | 0.005159 | no | yes | 13.3 | 2.5629 | 48.907 | yes |
| `q128_m2_eq_base` | 128 | 288 | 1152 | 0.000288 | no | yes | 217.5 | 1.8116 | 182.093 | yes |
| `q128_m2_eq_predamp` | 128 | 288 | 1152 | 0.000288 | no | yes | 217.5 | 1.8116 | 184.949 | yes |
| `q256_m2_eq_base` | 256 | 544 | 2048 | 0.000083 | no | yes | 841.7 | 0.9053 | 842.368 | no |
| `q256_m2_eq_predamp` | 256 | 544 | 2048 | 0.000083 | no | yes | 841.7 | 0.9053 | 853.113 | no |
| `q512_m2_eq_base` | 512 | 1056 | 2048 | 0.000206 | no | yes | 1005.9 | 3.6505 | 396.541 | no |
| `q512_m2_eq_predamp` | 512 | 1056 | 2048 | 0.000206 | no | yes | 1005.9 | 3.6505 | 410.159 | no |

**Q1-B, one relaxed contract constant at a time, $q=256$ only:**

| arm | per-step iteration budget | latent trust radius | quadrature | budget exits | worst joint gradient | worst same-grid all % | worst evolved % | median GPU ms | cost vs control | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| `q256_m2_dense_base` | 180 | x1 | dense | 15 | 1.191e-03 | 0.9053 | 0.7566 | 3662.173 | 1.000x | no |
| `q256_m2_dense_base_b2000` | 2000 | x1 | dense | 0 | 9.710e-07 | 0.9053 | 0.7566 | 3652.751 | 0.997x | yes |
| `q256_m2_dense_base_b600` | 600 | x1 | dense | 0 | 9.710e-07 | 0.9053 | 0.7566 | 3618.621 | 0.988x | yes |
| `q256_m2_dense_base_b600t10` | 600 | x10 | dense | 0 | 9.710e-07 | 0.9053 | 0.7566 | 2179.486 | 0.595x | yes |
| `q256_m2_dense_base_t10` | 180 | x10 | dense | 3 | 6.266e-03 | 0.9053 | 0.7566 | 2192.885 | 0.599x | no |
| `q256_m2_eq_base` | 180 | x1 | eq | 15 | 1.978e-03 | 0.9053 | 0.7580 | 843.870 | 0.230x | no |
| `q256_m2_eq_base_b2000` | 2000 | x1 | eq | 0 | 9.849e-07 | 0.9053 | 0.7580 | 848.014 | 0.232x | yes |

**Q2 envelope, every subject, both metrics:**

| subject | family | q / k' | quadrature | evolution tol | worst all times % | worst evolved % | t0 compression % | median GPU ms | converged |
|---|---|---|---|---|---|---|---|---|---|
| `fno-large` | fno | None | — | — | 7.4164 | 7.4164 | 0.0000 | 11.206 | — |
| `fft_tight` | fom | None | — | — | 0.0000 | 0.0000 | 0.0000 | 92.903 | — |
| `nt1e-2_dt01` | fom | None | — | — | 3.1999 | 3.1999 | 0.0000 | 9.312 | — |
| `nt1e-4_dt005` | fom | None | — | — | 0.0338 | 0.0338 | 0.0000 | 37.699 | — |
| `pod16_dense` | pod | 16 | dense | 1e-06 | 61.6503 | 28.7250 | 61.6503 | 50.835 | yes |
| `pod32_dense` | pod | 32 | dense | 1e-06 | 47.0681 | 18.7995 | 47.0681 | 91.176 | no |
| `pod64_dense` | pod | 64 | dense | 1e-06 | 19.8156 | 7.0835 | 19.8156 | 172.769 | no |
| `pod128_dense` | pod | 128 | dense | 1e-06 | 10.1198 | 1.9464 | 10.1198 | 392.033 | no |
| `q0_M256_dense_g0p001` | rom | 0 | dense | 0.001 | 2.5628 | 1.2705 | 2.5628 | 311.286 | no |
| `q0_M256_dense_g1em06` | rom | 0 | dense | 1e-06 | 2.5629 | 1.2710 | 2.5629 | 422.825 | yes |
| `q0_M256_eq_g0p001` | rom | 0 | eq | 0.001 | 2.5628 | 1.3182 | 2.5628 | 45.762 | no |
| `q0_M256_eq_g1em06` | rom | 0 | eq | 1e-06 | 2.5629 | 1.3186 | 2.5629 | 60.627 | yes |
| `q16_M256_eq_g0p001` | rom | 16 | eq | 0.001 | 2.4806 | 1.8070 | 2.4806 | 66.350 | no |
| `q16_M256_eq_g1em06` | rom | 16 | eq | 1e-06 | 2.4806 | 1.8066 | 2.4806 | 84.494 | yes |
| `q32_M256_eq_g0p001` | rom | 32 | eq | 0.001 | 2.3534 | 1.3052 | 2.3534 | 77.793 | no |
| `q32_M256_eq_g1em06` | rom | 32 | eq | 1e-06 | 2.3534 | 1.3048 | 2.3534 | 101.934 | yes |
| `q64_M256_eq_g0p001` | rom | 64 | eq | 0.001 | 2.1489 | 1.5742 | 2.1489 | 94.373 | no |
| `q64_M256_eq_g1em06` | rom | 64 | eq | 1e-06 | 2.1489 | 1.5738 | 2.1489 | 126.211 | yes |
| `q128_M256_dense_g1em06` | rom | 128 | dense | 1e-06 | 1.8116 | 1.0418 | 1.8116 | 1183.578 | yes |
| `q128_M256_eq_g0p001` | rom | 128 | eq | 0.001 | 1.8116 | 1.3517 | 1.8116 | 153.748 | no |
| `q128_M256_eq_g1em06` | rom | 128 | eq | 1e-06 | 1.8116 | 1.3517 | 1.8116 | 188.358 | yes |
| `q256_M544_eq_g0p001` | rom | 256 | eq | 0.001 | 0.9053 | 0.7580 | 0.9053 | 496.671 | no |
| `q256_M544_eq_g1em06` | rom | 256 | eq | 1e-06 | 0.9053 | 0.7580 | 0.9053 | 912.092 | no |
| `q512_M1056_eq_g1em06` | rom | 512 | eq | 1e-06 | 3.6505 | 3.6505 | 0.6027 | 459.722 | no |

**Non-dominated over (median GPU ms, worst error), all output times:**

| subject | family | q / k' | worst error % | median GPU ms | converged |
|---|---|---|---|---|---|
| `nt1e-2_dt01` | fom | None | 3.1999 | 9.312 | — |
| `nt1e-4_dt005` | fom | None | 0.0338 | 37.699 | — |
| `fft_tight` | fom | None | 0.0000 | 92.903 | — |

**Non-dominated over (median GPU ms, worst error), evolved times only:**

| subject | family | q / k' | worst error % | median GPU ms | converged |
|---|---|---|---|---|---|
| `nt1e-2_dt01` | fom | None | 3.1999 | 9.312 | — |
| `nt1e-4_dt005` | fom | None | 0.0338 | 37.699 | — |
| `fft_tight` | fom | None | 0.0000 | 92.903 | — |

**Pre-registered criterion for calling $q$ a knob:**

| metric | monotone at fixed M | monotone over every rung | converged non-dominated points | cost span | error span | passes |
|---|---|---|---|---|---|---|
| evolved | no | no | 4 | 19.522 | 1.266 | no |
| all_times | yes | no | 5 | 3.107 | 1.415 | no |

**FNO in the same allocation.** `fno-large`, 17,877,317 real parameters, checkpoint SHA256 `208d9002cd8e8567…`, pooled device query 11.206 ms median over 18 retained repetitions.

Source-generated report: `experiments/b-ladder-top/reports/2026-09-16-b-ladder-top.md` (SHA256 `d1d64a4766c876cb33583a6163bb315e767706ec276a41db10305c11cb5d06b4`) with its envelope figure and generator beside it.

Raw archive `btq101` Git-tracked as bounded chunks: whole SHA256 `c81aa8c2a01d46527321250433e3ccd08fdc05358b61a3795eeeececba0584e3` (12 chunks).
Raw archive `btq102` Git-tracked as bounded chunks: whole SHA256 `5b83355915231e0211fb702113930dd975626f761815c5635d08843ebf02e6b0` (5 chunks).
Raw archive `btq201` Git-tracked as bounded chunks: whole SHA256 `14e6820c401014a30a8fec4bf6fdea5e9dcd5a245efe2776adee10aa0c6e0760` (11 chunks).

**Q1 — the three fixes are inert, and the two failing rungs fail for different
reasons.** At $q=256$ the `base`, `pre`, `damp` and `predamp` arms are indistinguishable to four
decimals in error, in median iterations per step, in budget-exit count and in worst gradient;
the cascade arm is much worse. The failure localises to **five time steps of ONE case out of
six** (case 0, steps 0-3 and 31 of 50). At $q=512$ there is no solver failure at all: at $q=R$
the corrections span the whole bank, the supplied field is fitted to $10^{-16}$ relative, and
the normalized gradient $\|J^\top r\|/(\|J\|\,\|r\|)$ becomes a $0/0$ ratio whose reported
value means nothing, while every one of the rung's 900 time steps exits on the gradient
criterion. That is the same degenerate endpoint the Poisson $q=R$ rung showed.

Fix (b) does exactly what it was predicted to do to the conditioning and nothing to the answer:
it cuts $\kappa$ of the damped normal equations at $q=512$ by a factor of 1965 and changes no
reported error anywhere. The conditioning was never the binding constraint. Fix (c) changes
nothing measurable. Fix (a) is actively harmful.

**Q1-B — the rung converges at a raised iteration budget, and the relaxation is free.** Raising
the per-step budget from the retained 180 to 600, changing nothing else, gives zero budget
exits and a worst gradient inside $10^{-6}$ at 0.988x the control's median GPU time, because
the extra iterations land on five of nine hundred steps. Its worst same-grid error is strictly
below the $q=128$ rung's, so the pre-registered Q1 accuracy requirement holds — the target
PASSES under that one declared relaxation and FAILS under the frozen contract. Ten times the
latent trust radius is a genuine 1.67x **cost** lever at unchanged error and does not converge
the rung on its own. Every arm reports the same error to four decimals whether it converged or
not: **the convergence failure was a stopping-rule fact, not an accuracy fact.**

**Empirical quadrature per rung is now constructible everywhere.** Raising the bounded fitter's
refit block from $m/64$ to $m/16$ made every rule in this lane reach full target support with
none truncated, including $m=2048$ at $q=512$ in 1006 s.

**Q2 — the ladder's monotonicity is the $t=0$ compression term.** On the worst-over-all-times
metric the fixed-test-count rungs fall monotonically and the criterion fails only on error
span. On the worst-over-**evolved**-times metric the same rungs are **not monotone** ($q=16$ is
worse than $q=0$), and the converged non-dominated set spans 1.266x in error across 19.5x in
cost. "$q$ is a knob" therefore fails on both metrics, for a different reason on each. The
decoder's compression of the supplied field is what makes the all-times ladder look monotone.

**Q2 — nothing but the full-order solver is on the envelope.** On both metrics the non-dominated
set over every subject in the job — ladder rungs, POD-LSPG ranks, the trained FNO and the FOM
controls — contains only same-job full-order controls. The FNO ran in the same Slurm allocation
on the same GPU with the neural-operator lane's `timing.py` protocol replicated, its six cases
verified disjoint from its training draw, and it returns the supplied field bitwise at $t=0$,
which is exactly why the all-times metric flatters it and both metrics are reported.

**Retracted from the cheap-corrections cell:** its speculation that the broken
$q=512$ empirical-quadrature arm (27.8 % same-grid error) was the bounded fitter's **walltime
cap**. With the larger refit block that rule now converges to full support with a relative fit
an order of magnitude better, and the rung still gives 3.65 % same-grid error against 0.60 %
for its dense twin. The cap was not the suspect; the failure is in the rung.

**Retracted within this cell:** the design's own framing that $q=512$ "does not converge" as a
solver fact — see the degenerate-endpoint finding above — and the working hypothesis behind all
three fixes, that the $q=256$ failure lives in the correction block's damping, trust radius or
conditioning. It lives in none of them.

**Recorded negatives rather than dropped:** the cascade warm start (fix a) turns a cleanly
converged $q=128$ rung into one with 63 budget exits and the $q=256$ rung into one with 180 at
3.6x the cost, even though its three-way guard cannot start a step from a worse residual; and
the `q0_arms_bitwise` gate failed vacuously on Q1-B (which has no $q=0$ arms) before being
changed to report "not applicable", which is recorded in `DESIGN.md` rather than silently fixed.

**Cross-job reproduction, for the record.** Every substantive fidelity gate passes at or inside
$10^{-9}$: Q1 reproduces `cclad01`'s $q=0$ dense, $q=0$ EQ and $q=128$ dense rows, Q1-B
reproduces its $q=256$ dense row, and Q2 reproduces its $q=0$ and $q=128$ fixed-$M$ dense rows.
The directions hash reproduced `cclad01` bitwise in Q1 (same GPU model) and did not in Q2 or
Q1-B; the 4096-interval reference fields did the opposite, matching `qlad01` bitwise in Q2 and
not in Q1. Neither hash is a gate and both are reported as probes.

**Open.** (1) The $q=16$ regression on the evolved-times metric is unexplained and
reproducible across both quadratures and both evolution tolerances; nothing here says why one
extra correction direction makes the trajectory worse while making the supplied-field fit
better. (2) The $q=512$ empirical-quadrature failure is measured, not explained; the hypothesis
on offer — that at $q=R$ the reachable states include fields the $m$-point rule was never
fitted on — was not tested. (3) The five stubborn $q=256$ steps are all in the first four steps
after the initial fit plus one late step of one case; a cheaper fix than more iterations (a
sub-step or a continuation start) was not tried. (4) Whether the $t=0$ compression should be
charged to the ROM at all is a framing decision for the paper: the FNO's output contract returns
the supplied field, the ROM's decodes it, and that single difference moves the headline number
by a factor of two. **Next session:** decide (4) before any of these numbers enter the
manuscript, and note that the campaign's standing conclusion is unchanged and now measured on
one GPU in one job — the tolerance-matched FOM beats every reduced-order and operator subject
on both axes at once. One mesh, one checkpoint, one training seed, six opened development cases;
final cohorts sealed, no new case opened, nothing merged.
