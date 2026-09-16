## 2026-09-16

### q-trajdirs — redirected mid-flight: the trajectory-direction hypothesis was killed by the q-diag lane before this lane's own job landed, and the DENSE correction ladder PASSES the redirected knob criterion

The lane opened on one question — does fitting the correction directions $C_q$ to the ROM's **trajectory** error, instead of the head's **static** reconstruction residual, make the Burgers $256^2$ ladder monotone on the worst-over-evolved-times metric? It was pre-registered in full, implemented, smoked and submitted. **While that job was running, the `q-diag` lane answered the underlying question from data that already existed: the $q=16$ evolved-metric regression is the EMPIRICAL QUADRATURE, not the directions.** The coordinator redirected the lane; both halves are recorded below and the redirect itself is section 11 of `experiments/q-trajdirs/DESIGN.md`, appended without editing anything above it. Nothing was merged and the branch was **not** pushed.

`q-diag`'s diagnosis, cited because it is what redirected this lane: `worktrees/2026-09-16-q-diag/experiments/q-diag/reports/2026-09-16-q16-regression-diagnosis.md`, SHA256 `c0bf57626d9e0a42…`. Its verdict: 12 of 12 dense ladder/metric combinations monotone across four jobs, only empirical-quadrature ladders regressing, and cause (1) — trajectory-blind directions — refuted on both of its pre-registered criteria, the incumbent directions' per-interval step map improving with $q$ at 0.95 / 0.87 / 0.73 / 0.65.

Worktree `worktrees/2026-09-16-q-trajdirs`, branch `exp/2026-09-16-q-trajdirs`, forked from `exp/2026-09-16-b-ladder-top` at `b8efd5b4`. Namespace `/cluster/tufts/paralab/tawal01/q_trajdirs_20260916/`. Two A100 jobs, one attempt directory each, `squeue` checked before and after every submission, the three-job cap leaving one unused: the redirected primary `3757505` (`qtd02`) on `NVIDIA A100-PCIE-40GB`, source `f064b8825c57fad2a6c3c62715ad41568ddce828`, elapsed 3152.1 s; and the control `3756800` (`qtd01`) on `NVIDIA A100-PCIE-40GB`, source `59fa69b1ad0c88140fc9519d6dad74fa0570ad9c`, **FAILED** before writing its elapsed time. Both printed `jax_backend=gpu`, ran float64 with highest matmul precision, and were checksum-collected, independently NumPy-audited and archived before their exact remote attempt directories were removed; the namespace is now empty.

**The redirected verdict.** On the `dense_m4` ladder — incumbent directions, DENSE quadrature, $M=4(K+q)$, per-step budget 600, one allocation:

| criterion | holds | measured |
|---|---|---|
| 1. worst evolved error non-increasing in $q$, every rung converged | yes | monotone yes, all converged yes |
| 2. converged non-dominated set spans $\ge2\times$ in error AND $\ge2\times$ in cost, evolved metric | yes | 6 points, error span 3.637x, cost span 13.070x |
| (reported) worst all-times error non-increasing in $q$ | yes | — |


**Overall: the redirected criterion PASSES.** Converged non-dominated arms on the evolved metric: `old_q128_M576_dense`, `old_q16_M128_dense`, `old_q256_M1088_dense`, `old_q32_M192_dense`, `old_q64_M320_dense`, `q0_M64_dense`.


**Part 1, no GPU — the dense ladder in data that already existed.** The four comparator archives (`cclad01`, `btq101`, `btq102`, `btq201`) were restored from their Git-tracked chunks and every same-grid error recomputed in NumPy from the saved fields, reproducing each job's archived reference error to 5.76e-16 relative:

| ladder | rungs | monotone evolved | monotone all times | every rung converged | non-dominated points | evolved error span | cost span | passes redirected criterion |
|---|---|---|---|---|---|---|---|---|
| `btq101_m2` | 64, 128, 256, 512 | yes | yes | no | 2 | 1.478 | 1.705 | no |
| `cclad01_m2` | 0, 16, 32, 64, 128, 256, 512 | yes | yes | no | 5 | 3.975 | 3.040 | no |
| `cclad01_m256` | 0, 16, 32, 64, 128 | yes | yes | yes | 5 | 1.220 | 2.438 | no |
| `cclad01_m4` | 0, 16, 32, 64, 128, 256, 512 | yes | yes | no | 5 | 2.115 | 4.188 | no |


**8 of 8** dense ladder/metric combinations in the archives are monotone — the `q-diag` census reproduced independently from the fields rather than from its tables.


**Part 2 — the redirected job, every dense rung at budget 600 in one allocation.**


`dense_m4` — incumbent directions, DENSE quadrature, M = 4(K+q); monotone evolved yes, monotone all times yes, every rung converged yes:

| $q$ | $M$ | $m$ | all % | evolved % | $t_0$ % | best-found % | median GPU ms | budget exits | converged |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 64 | — | 2.5629 | 1.8890 | 2.5629 | 2.5447 | 333.192 | 0 | yes |
| 16 | 128 | — | 2.4806 | 1.3985 | 2.4806 | 2.4615 | 417.939 | 0 | yes |
| 32 | 192 | — | 2.3534 | 1.2336 | 2.3534 | 2.3325 | 496.922 | 0 | yes |
| 64 | 320 | — | 2.1489 | 1.0843 | 2.1489 | 2.1386 | 701.655 | 0 | yes |
| 128 | 576 | — | 1.8116 | 0.8930 | 1.8116 | 1.8105 | 1330.433 | 0 | yes |
| 256 | 1088 | — | 0.9053 | 0.5194 | 0.9053 | 0.9016 | 4354.902 | 0 | yes |


`dense_fixedM` — incumbent directions, DENSE quadrature, fixed M = 256; monotone evolved yes, monotone all times yes, every rung converged yes:

| $q$ | $M$ | $m$ | all % | evolved % | $t_0$ % | best-found % | median GPU ms | budget exits | converged |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 256 | — | 2.5629 | 1.2710 | 2.5629 | 2.5447 | 404.245 | 0 | yes |
| 16 | 256 | — | 2.4806 | 1.2305 | 2.4806 | 2.4615 | 490.820 | 0 | yes |
| 32 | 256 | — | 2.3534 | 1.1869 | 2.3534 | 2.3325 | 544.416 | 0 | yes |
| 64 | 256 | — | 2.1489 | 1.1255 | 2.1489 | 2.1386 | 643.200 | 0 | yes |
| 128 | 256 | — | 1.8116 | 1.0418 | 1.8116 | 1.8105 | 915.975 | 0 | yes |


Same-job full-order controls:

| control | worst all times % | worst evolved % | median GPU ms |
|---|---|---|---|
| `fft_loose` | 3.7127 | 3.7127 | 15.366 |
| `fft_tight` | 0.0000 | 0.0000 | 88.630 |
| `nt1e-2_dt01` | 3.1999 | 3.1999 | 8.883 |


Non-dominated over (median GPU ms, error), every subject: all times `fft_tight`, `nt1e-2_dt01`; evolved times `fft_tight`, `nt1e-2_dt01`.


**Gates, `qtd02`.** `artifacts_present` yes; `backend_gpu` yes; `bank_frozen` yes; `btq201_envelope_probe` yes; `checkpoint_unchanged` yes; `complete` yes; `cross_job_fidelity` no; `direction_cohorts_disjoint_from_evaluation` yes; `direction_prefixes_recomputed_from_artifact` yes; `directions_hashed_and_saved` yes; `directions_rank_covers_ladder` yes; `evaluation_cohort_bitwise_abl01` yes; `every_eq_rule_reports_validity` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `nested_prefix_consistent` yes; `no_eq_rule_truncated` yes; `old_directions_hash_matches_comparator` no; `overdetermined_weak_system` yes; `precision_highest` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_fields_bitwise_match_comparator` yes; `reference_residuals` yes; `repetition_output_identical` yes; `reproduces_old_q128_M256_dense` yes; `reproduces_old_q128_M576_dense` yes; `reproduces_old_q16_M128_dense` yes; `reproduces_old_q16_M256_dense` yes; `reproduces_old_q256_M1088_dense` no; `reproduces_old_q32_M192_dense` yes; `reproduces_old_q32_M256_dense` yes; `reproduces_old_q64_M256_dense` no; `reproduces_old_q64_M320_dense` no; `reproduces_q0_M256_dense` yes; `reproduces_q0_M64_dense` yes; `same_grid_baseline_present` yes; `step_budget_is_600` yes; `x64` yes.


**Cross-job fidelity, `qtd02`.**

| arm | reproduces | job | tolerance | worst relative difference | passed |
|---|---|---|---|---|---|
| `old_q128_M256_dense` | `q128_M256_dense_g1em06` | btq201 | 1.00e-09 | 7.67e-10 | yes |
| `old_q128_M576_dense` | `q128_m4_dense_block` | cclad01 | 1.00e-09 | 1.14e-10 | yes |
| `old_q16_M128_dense` | `q16_m4_dense_block` | cclad01 | 1.00e-09 | 5.59e-11 | yes |
| `old_q16_M256_dense` | `q16_m256_dense_block` | cclad01 | 1.00e-09 | 5.59e-11 | yes |
| `old_q256_M1088_dense` | `q256_m4_dense_block` | cclad01 | 1.00e-09 | 3.03e-09 | no |
| `old_q32_M192_dense` | `q32_m4_dense_block` | cclad01 | 1.00e-09 | 3.91e-11 | yes |
| `old_q32_M256_dense` | `q32_m256_dense_block` | cclad01 | 1.00e-09 | 3.91e-11 | yes |
| `old_q64_M256_dense` | `q64_m256_dense_block` | cclad01 | 1.00e-09 | 1.07e-09 | no |
| `old_q64_M320_dense` | `q64_m4_dense_block` | cclad01 | 1.00e-09 | 1.07e-09 | no |
| `q0_M256_dense` | `q0_m256_dense_block` | cclad01 | 1.00e-09 | 7.64e-13 | yes |
| `q0_M64_dense` | `q0_m4_dense_block` | cclad01 | 1.00e-09 | 3.91e-13 | yes |


**The control, `qtd01` — the trajectory-fitted directions, run because it was already running.** It **did not reach its timed phase**, so it has no ladders and no verdict; what it did produce is below. Three direction sets differing only in which residual matrix is decomposed:

| set | residual decomposed | rows | rank | fit s | `directions_sha256` |
|---|---|---|---|---|---|
| `old` | static reconstruction residual, 1024 snapshots | 1024 | 512 | 1546.9 | `104887e76be1e80c…` |
| `traj` | trajectory error, 32 trajectories x 51 internal steps | 1632 | 512 | 27.0 | `06f85e4493050c5f…` |
| `prac` | trajectory error, 6 trajectories x 51 internal steps | 306 | 306 | 8.8 | `44cfa21138a127d6…` |


Cross-capture $\kappa_q(P,C)$, per cent of residual matrix $P$'s whitened energy that direction set $C$ reaches at rung $q$:

| residual $P$ | directions $C$ | $q=16$ | $q=32$ | $q=64$ | $q=128$ | $q=256$ |
|---|---|---|---|---|---|---|
| `old` | `old` | 45.28 | 62.01 | 80.26 | 94.70 | 99.83 |
| `old` | `traj` | 11.41 | 22.76 | 39.98 | 64.82 | 92.68 |
| `old` | `prac` | 8.64 | 15.57 | 28.60 | 53.65 | 90.56 |
| `traj` | `old` | 11.72 | 22.05 | 38.99 | 66.87 | 97.82 |
| `traj` | `traj` | 73.61 | 86.31 | 95.74 | 99.61 | 100.00 |
| `traj` | `prac` | 9.86 | 18.89 | 34.44 | 59.94 | 94.25 |
| `prac` | `old` | 4.53 | 10.60 | 26.17 | 51.33 | 94.86 |
| `prac` | `traj` | 48.11 | 57.80 | 68.92 | 83.95 | 97.45 |
| `prac` | `prac` | 94.37 | 99.37 | 100.00 | 100.00 | 100.00 |


Principal angles between the direction subspaces (field metric, degrees):

| pair | $q$ | overlap | min | median | max |
|---|---|---|---|---|---|
| `prac_vs_old` | 16 | 0.07913 | 56.97 | 77.98 | 89.14 |
| `prac_vs_old` | 64 | 0.30809 | 25.88 | 60.56 | 89.63 |
| `prac_vs_old` | 256 | 0.82904 | 0.00 | 4.89 | 89.71 |
| `traj_vs_old` | 16 | 0.17578 | 38.18 | 70.17 | 88.69 |
| `traj_vs_old` | 64 | 0.44116 | 11.74 | 48.81 | 89.89 |
| `traj_vs_old` | 256 | 0.85741 | 0.00 | 4.11 | 89.08 |
| `traj_vs_prac` | 16 | 0.14263 | 26.37 | 74.96 | 89.84 |
| `traj_vs_prac` | 64 | 0.31203 | 11.63 | 59.87 | 89.55 |
| `traj_vs_prac` | 256 | 0.80031 | 0.00 | 6.15 | 89.36 |


The static manifold floor (best-found on the supplied field) each rule buys, worst over the six cases; a dash is a rung this job did not build for that rule:

| directions | $q=0$ % | $q=16$ % | $q=32$ % | $q=64$ % | $q=128$ % | $q=256$ % |
|---|---|---|---|---|---|---|
| `old` | 2.5447 | 2.4615 | — | 2.1386 | — | — |
| `traj` | 2.5447 | 2.4780 | 2.2711 | 2.0764 | 1.8172 | 1.0681 |
| `prac` | 2.5447 | 2.4241 | 2.2857 | 2.0669 | 1.7430 | 0.9492 |


**Its timed ladders do not exist.** It built all 31 reduced arms and every empirical-quadrature rule, ran every offline diagnostic above, and then died in the compile warm-up of its 21st subject with `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: [0] Failed to load in-memory CUBIN (compiled for a different GPU?).: CUDA_ERROR_OUT_OF_MEMORY: out of memory [executable_name='jit__lambda']` on a 40 GB A100 it was sized past. Its offline output is collected, checksum-verified and audited in a `--partial` mode; it was **not** resubmitted, because the redirect said not to spend another job on new directions, and the third job of the cap is unused.


**Gates, `qtd01`** (the `complete` gate fails by construction for a partial job). `backend_gpu` yes; `bank_frozen` yes; `complete` no; `direction_cohorts_disjoint_from_evaluation` yes; `direction_prefixes_recomputed_from_artifact` yes; `directions_hashed_and_saved` yes; `directions_rank_covers_ladder` yes; `evaluation_cohort_bitwise_abl01` yes; `every_declared_rung_has_a_reconstruction` yes; `every_eq_rule_reports_validity` yes; `final_cohort_unopened` yes; `nested_prefix_consistent` yes; `no_eq_rule_truncated` yes; `old_directions_hash_matches_comparator` no; `overdetermined_weak_system` yes; `precision_highest` yes; `reference_fields_bitwise_match_comparator` yes; `reference_residuals` yes; `x64` yes.


Source-generated report: `experiments/q-trajdirs/reports/2026-09-16-dense-correction-ladder.md` (SHA256 `da86a865f7dc254ed0515714d9c369ead5c5d1249d9dcf0ee974d6ca12a7ff56`) with its figure and its generator beside it; the generator reads only the audited JSONs, so no number in it is hand-typed. The archive-only Part 1 is `experiments/q-trajdirs/checks/dense-from-archives.json`, produced by `dense_from_archives.py` with no GPU.


Raw archive `qtd01` Git-tracked as bounded chunks: whole SHA256 `8768d2fd24e7d5e29e1d31ff131c3a4a0d84daa92acf1e1aadf22ed77b487add` (2 chunks).


Raw archive `qtd02` Git-tracked as bounded chunks: whole SHA256 `188fbf88a8d3562399f5d2b5a2f9efb18f6c3bbfb38759d1f861f2b7431828e1` (7 chunks).


