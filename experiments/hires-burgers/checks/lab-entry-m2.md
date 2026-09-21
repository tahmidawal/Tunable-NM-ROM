
## 2026-09-21

### hires-burgers — milestone 2: hb2k02 (2048²), hb4k03 (4096²) and hb2kh64 (2048², 64 held-out cases) audited; bar met vs the TIGHT FOM on dev6 at both meshes, NOT vs the relaxed passing FOM, and NOT on held-out cases (accurate rung 1.31 %)

Worktree `worktrees/2026-09-20-hires-burgers` (branch `exp/2026-09-20-hires-burgers`); namespace `hires_b_20260920`; all three jobs H200, `jax_backend=gpu`, f64, highest precision, checksum-collected, NumPy-audited by `audit_hires.py`, remote dirs deleted. Summaries `experiments/hires-burgers/checks/{hb2k02,hb4k03,hb2kh64}-summary.json`; report and verdict matrix `experiments/hires-burgers/reports/2026-09-20-hires-burgers.md` + `reports/summary.json`, generated. Agent handover: the Fable 5.1 agent stopped on usage credits after committing hb2k02; this entry is by its Opus 5 successor.

- `hb2k02` job 4071616, 2048², cohort dev6, source `ae700dfd`, elapsed 2647 s, failed gates: none.
- `hb4k03` job 4071625, 4096², cohort dev6, source `ae700dfd`, elapsed 4697 s, failed gates: none.
- `hb2kh64` job 4077566, 2048², cohort hold64 (64 cases), source `7627da25`, elapsed 1210 s, failed gates: restricted_recomputation_tracks_full_grid.

| mesh | cohort | role | arm | worst evolved % | stalled | GPU ms | host ms | S tight GPU/host | S relaxed passing GPU/host | S coarse GPU/host |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048² | dev6 | chosen on dev6 | `q256_M544_lat64_g0p001_fast_chol_clip` | 0.865 | 0/300 | 124.2 | 187.2 | 6.07/4.36 (`lean_tight`) | 0.99/0.99 (`lean_nt3e-3_l3e-3_dt005`) | 0.63/0.75 (`c1024_nt1e-4_dt005`) |
| 2048² | dev6 | accurate rung q256/M1088 | `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry` | 0.598 | 0/300 | 130.7 | 192.4 | 5.76/4.24 (`lean_tight`) | 0.94/0.96 (`lean_nt3e-3_l3e-3_dt005`) | 0.60/0.73 (`c1024_nt1e-4_dt005`) |
| 2048² | dev6 | fast q=0 | `q0_M64_scaled_g0p001_fast_clip_lamcarry` | 2.372 | 0/300 | 29.5 | 91.8 | 25.54/8.88 (`lean_tight`) | 4.16/2.01 (`lean_nt3e-3_l3e-3_dt005`) | 1.21/1.07 (`c512_nt1e-4_dt005`) |
| 2048² | hold64 | accurate rung q256/M1088 | `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry` | 1.308 | 0/3200 | 114.9 | 177.5 | 6.48/4.53 (`lean_tight`) | 6.48/4.53 (`lean_tight`) | 0.71/0.80 (`c1024_nt1e-4_dt005`) |
| 2048² | hold64 | fast q=0 | `q0_M64_scaled_g0p001_fast_clip_lamcarry` | 9.026 | 0/3200 | 29.2 | 90.9 | 25.51/8.85 (`lean_tight`) | 25.51/8.85 (`lean_tight`) | 2.79/1.56 (`c1024_nt1e-4_dt005`) |
| 4096² | dev6 | chosen on dev6 | `q256_M544_lat64_g0p001_fast_chol_clip` | 0.875 | 0/300 | 137.8 | 381.0 | 23.64/9.18 (`lean_tight`) | 3.79/1.99 (`lean_nt3e-3_l3e-3_dt005`) | 0.58/0.85 (`c1024_nt1e-4_dt005`) |
| 4096² | dev6 | accurate rung q256/M1088 | `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry` | 0.604 | 0/300 | 144.7 | 385.0 | 22.52/9.09 (`lean_tight`) | 3.61/1.97 (`lean_nt3e-3_l3e-3_dt005`) | 1.94/1.36 (`c2048_nt1e-4_dt005`) |
| 4096² | dev6 | fast q=0 | `q0_M64_scaled_g0p001_fast_clip_lamcarry` | 2.416 | 0/300 | 43.0 | 285.2 | 75.69/12.27 (`lean_tight`) | 12.13/2.66 (`lean_nt3e-3_l3e-3_dt005`) | 1.85/1.13 (`c1024_nt1e-4_dt005`) |

**Speed.** Cholesky + trust clipping + damping carry-over (hb2k02, SPEED-LOG) replicate at 4096²: q256/M1088 315.8 → 144.7 ms (2.18×) at unchanged error 0.6043 → 0.6043 %, rejected trial steps 378 → 0. The ROM's speedup against the relaxed passing FOM grows with mesh (the FOM scales with n, the ROM does not, except its decode), but stays below 5 in both scopes. At 4096² the complete query adds ≈240 ms of output transfer to every arm, which caps any complete-query ratio.

**The coarse-grid FOM beats the ROM at 4096².** `c1024_nt1e-4_dt005` runs in 79.7 ms at 0.627 % same-grid (refined-reference error 2.21 % vs the ROM's 2.02 % and the tight FOM's 1.90 %): about as accurate as the accurate rung and faster. Against that comparator the ROM has no speed story at these meshes.

**Held-out cases (hb2kh64).** The 0.598 % dev6 accuracy does not generalise: on hold64 the accurate rung is 1.308 % worst evolved (median 0.120 %), M=2176 1.109 % (rule uncertified), q=0 9.03 %; zero stalled exits. No certified ROM arm is ≤ 1 % on held-out cases at 2048², so the lane bar is not met on unseen cases at any speed. This matches the bank-floor lane (R=512 bank floor far higher on hold64 than on dev6).

**What was wrong / caveats.** hb2kh64 fails one audit gate, `restricted_recomputation_tracks_full_grid` (threshold 5 %): on 5 of 384 rows, all low-error q=256 cases, the 256² restricted sample reads the full-grid error 5–7 % low (worst gap 0.073); the cohort-worst per arm agrees within 1.2 % and the full-grid recomputation of case 0 is exact, so the verdict numbers stand, but the gate failed and is reported. hold64 timings are one repetition. On hold64 no same-mesh FOM other than the tight one passes the 0.1 % "relaxed passing" definition, so "relaxed passing" = tight there by rule. The HANDOFF line "worst evolved 1.31 % on hold64" was an unaudited log line; the audit confirms it.

**Running / next.** Budget 8/8 now spent: `hb4k04` (job 4079320, 4096², dev6, 5 reps) tests the `pred2` quadratic-extrapolation predictor (H11) and labelled tolerance arms; `hb4kh64` (job 4079321, 4096², hold64) reports the same arms on held-out cases. Selection stays on dev6 (DESIGN addendum A1). After them the lane stops; the next lever is accuracy, not speed: refit the head + recertify EQ on the bank-floor lane's better banks (out of this lane's budget).
