
### hires-burgers — hb2k01 (2048², job 4054951, H200) audited: corrected accuracy SURVIVES frozen-weight transfer; the lane bar is NOT met at 2048² (accurate rung 3.85× the tight Newton–BiCGStab, 0.84× the relaxed passing one); hb4k02 (4096², job 4059827) FAILED with no ROM number

Worktree `worktrees/2026-09-20-hires-burgers` @ `a8eff327`; job source `b66a59bd`; GPU 0: NVIDIA H200 (UUID: GPU-50404ee7-1d50-1d0c-6780-806cf66543b8); `jax_backend=gpu`, f64, highest matmul precision; elapsed 5654 s; checksum-collected, independently NumPy-audited (`experiments/hires-burgers/audit_hires.py`: every restricted field recomputed, full-grid fields of the audit case recomputed to ≤1e-12, failed gates: none), remote directory deleted. Summary `experiments/hires-burgers/checks/hb2k01-summary.json`, report `experiments/hires-burgers/reports/2026-09-20-hires-burgers.md` (+ `summary.json`), both source-generated. Six development cases, one checkpoint ($K=16$, $R=512$, trained at $256^2$), sealed cohort unopened. Error = $\lVert u-u^{tight}\rVert_2/\lVert u_0\rVert_2$ vs the same-job `fft_tight`; medians of 5 repetitions × 6 cases.

| arm | worst evolved % | worst all-times % | GPU ms | stalled / steps | ρ_max held-out / deployed | certified | S vs `lean_tight` | S vs `lean_nt1e-3_l1e-3_dt005` |
|---|---|---|---|---|---|---|---|---|
| `q0_M64_xfer_g0p001_fast` | 2.3822 | 4.1823 | 25.10 | 0 / 300 | 0.0328 / 0.0337 | True | 29.92 | 6.52 |
| `q128_M576_lat64_g0p001_fast` | 1.0748 | 3.6604 | 133.90 | 0 / 300 | 0.0698 / 0.0719 | True | 5.61 | 1.22 |
| `q256_M544_lat64_g0p001_fast` | 0.8896 | 3.2190 | 332.09 | 0 / 300 | 0.0634 / 0.0624 | True | 2.26 | 0.49 |
| `q256_M1088_lat64_g0p001_fast_chol` | 0.5980 | 3.2190 | 195.18 | 0 / 300 | 0.1000 / 0.0980 | True | 3.85 | 0.84 |
| `q256_M1088_xfer_g0p001_fast_chol` | 0.5886 | 3.2190 | 161.47 | 0 / 300 | 0.2052 / 0.2192 | False | 4.65 | 1.01 |
| `q256_M1088_scaled_g0p001_fast_chol` | 0.5915 | 3.2190 | 147.94 | 0 / 300 | 0.2549 / 0.2927 | False | 5.08 | 1.11 |
| `q256_M1088_bad0_g0p001_fast_chol` | 1.0359 | 3.2190 | 138.60 | 0 / 300 | 0.4840 / 0.4841 | False | 5.42 | 1.18 |
| `fft_tight` | 0.0000 | 0.0000 | 758.48 | 0 / 300 | — | — | — | — |
| `lean_tight` | 0.0000 | 0.0000 | 750.81 | 0 / 300 | — | — | — | — |
| `lean_nt1e-4_dt005` | 0.0334 | 0.0334 | 280.05 | 0 / 300 | — | — | — | — |
| `lean_nt1e-3_l1e-3_dt005` | 0.0543 | 0.0543 | 163.62 | 0 / 300 | — | — | — | — |
| `lean_nt1e-2_l1e-2_dt005` | 2.1615 | 2.1615 | 100.43 | 0 / 300 | — | — | — | — |
| `nt1e-2_dt01` | 3.5129 | 3.5129 | 56.74 | 0 / 150 | — | — | — | — |
| `c1024_nt1e-4_dt005` | 0.4173 | 1.2598 | 78.38 | 0 / 300 | — | — | — | — |
| `c512_nt1e-4_dt005` | 1.2170 | 2.3569 | 35.82 | 0 / 300 | — | — | — | — |

**Bar verdict (pre-registered rule: cheapest certified arm with evolved error ≤ 1 %).** `q256_M1088_lat64_g0p001_fast_chol`: 0.5980 % evolved (3.2190 % all-times, the $t=0$ compression), 195.18 ms, 0 stalled exits of 300 steps: 3.85× vs `lean_tight`, 0.84× vs the relaxed passing `lean_nt1e-3_l1e-3_dt005` — **not met** on either. Stretch ≤0.5 %: no arm. Fast setting `q0_M64_xfer_g0p001_fast`: 2.3822 %, 25.10 ms, 29.92× / 6.52×, 3.82× vs the fastest tested FOM at least as accurate.

**Dense truth (exact advection, same solver, tol 1e-6), worst evolved over six cases:** `q0_M64_dense` 2.3717 %; `q128_M576_dense` 1.0747 %; `q256_M544_dense` 0.8655 %; `q256_M1088_dense` 0.5985 %. The deployed lattice arm reproduces its dense truth, so the error is representation, not quadrature.

**What does not flatter the ROM.** (1) Same-grid truth vs the 8192² refined reference is 1.995 % at 2048²; the coarse-grid FOM `c1024_nt1e-4_dt005` is 0.4173 % same-grid at 78.38 ms and 2.208 % vs the refined reference, against the accurate ROM rung's 2.112 % at 195.18 ms: **a half-resolution FOM is cheaper and as physical as the accurate rung**. (2) The relaxed FOM with a matched inner tolerance (`lean_nt1e-3_l1e-3_dt005`, added on the Codex audit's advice) is 0.054 % at 163.62 ms — 1.7× cheaper than the `nt1e-4_dt005` the earlier panels called relaxed. (3) The all-times error is 3.2–4.2 % for every ROM arm.

**Quadrature.** Only the uniform 63×63 lattice (no fit, no draw, equal weights) certifies at $q=256$, $M=1088$ (ρ_max 0.100 held-out, 0.098 deployed; bar 0.116). b-eqtop's rule with weights scaled by $(L/256)^2$ FAILS (0.255 / 0.293) and so does the NNLS weight refit on 128 states (0.205 / 0.219; fit-state ρ 0.0018 — over-fit), yet all three give the same evolved error to 0.01 pp: the certificate is conservative here; the uncertified arms are not eligible for the verdict. Control `bad0` fails as it must (0.484; error 1.04 % vs 0.59 %). At $q=128$ lattice and scaled certify, refit does not (0.139 deployed).

**Speed loop (same job, parity-gated; `SPEED-LOG.md`).** `hfast` (analytic structured Jacobian, affine folds, fused $(r,J)$, hoisted QR, no redundant diagnostic) 1.11–1.16× at fields ≤1e-13 and identical integers; **Cholesky instead of LU on the 272×272 normal matrix 1.56–1.80× at parity** — the LU solve (≈0.6 ms) was the largest item per LM iteration, larger than the whole residual+Jacobian (0.5–0.76 ms); tolerance 1e-3 vs 1e-6 1.57× at unchanged error; $M=544$ at $q=256$ REVERTED (slower — more iterations — and 0.89 % vs 0.60 %). Finding that drives the next jobs: the FIRST time step takes 41–88 of a query's 170–494 LM iterations with most of the 25–114 rejected trial steps (z-step capped at 1 % of the code radius, start = initial-fit state).

**What was wrong / retracted.** `hb4k02` (job 4059827, H200, source `77db1832`): the in-place 64 GiB bank built, the FOM arms ran once untimed, then every product with the bank failed (`Autotuning failed ... f64[6,16769025]`: XLA's Triton gemm cannot handle > 2^31 elements) and the driver's OOM filter silently recorded every ROM arm as dropped before the job died in `dense_targets`. No ROM number; its untimed FOM log lines are not results. Record `experiments/hires-burgers/artifacts/hb4k02-failed/`; remote dir deleted. Fix: the bank is a tuple of row blocks below the limit. The HANDOFF's earlier 'early answer' figures were unaudited log lines; the audited table above supersedes them (they agree). Jobs used 3 of 8.

**Open.** `hb2k02` (2048², lattice rules, Cholesky, trust clipping `clip`, damping carry-over `lamcarry`, $M=2176$ stretch rung, one more FOM tolerance) and `hb4k03` (4096², same arms, blocked bank) are staged next. At 4096² the FOM should cost ≈4× while the ROM solve is mesh-flat, so the tight-comparator bar may be met there; the relaxed-passing and coarse-grid comparators will be shown beside it regardless.
