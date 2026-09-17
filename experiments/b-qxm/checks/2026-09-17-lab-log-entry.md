## 2026-09-17
### b-qxm — both factors matter; the pure-rank ladder at fixed M = 1088 spans 2.44x and the scheduled ladder's log-span is 69 % rank, 31 % test count (corner path)

The coordinator asked whether the passing Burgers $256^2$ dense correction ladder — which raises the test count $M = 4(K+q)$ together with the rank $q$ — owes its $3.6\times$ evolved-error span to the rank, to the test count, or to both. This lane ran the two factors **crossed**: rows $q \in \{0, 16, 32, 64, 128, 256\}$ against fixed $M \in \{256, 1088\}$, scheduled $M \in \{4, 8, 16\}(K+q)$, the bridge cells $(q_k, M_{k+1})$, a solver control, and a saturation sweep in $M$ at $q = 0$ and $q = 64$; dense quadrature, budget 600, the frozen incumbent checkpoint, the six opened development cases, same-job FOM controls, three timed repetitions. Pre-registered protocol, decomposition and falsification clause: `experiments/b-qxm/DESIGN.md` (amendments A0–A1 and later, appended). Nothing merged, nothing pushed.

Worktree `worktrees/2026-09-17-b-qxm`, branch `exp/2026-09-17-b-qxm`, forked from `exp/2026-09-16-q-ridge` at `7dc970fc`. Namespace `/cluster/tufts/paralab/tawal01/b_qxm_20260917/`. Jobs: G1 `bqx101` = 3780175 on `NVIDIA A100 80GB PCIe`, source `ed431edb498a`, 2528 s, failed gates: none; G2 `bqx201` = 3780177 on `NVIDIA A100-PCIE-40GB`, source `ed431edb498a`, 3715 s, failed gates: none; S1 `bqx301` = 3780178 on `NVIDIA A100-PCIE-40GB`, source `ed431edb498a`, 3217 s, failed gates: none. All printed `jax_backend=gpu`, float64, highest matmul precision; checksum-collected, independently NumPy-audited, archived as Git chunks, remote attempt directories deleted.

**Verdict against the pre-registered rules (DESIGN §6).**

| rule | value |
|---|---|
| headline the paper should carry | fixed-M ladder (M = 1088, pure rank), scheduled ladder reported beside it |
| fixed $M=1088$ ladder monotone / every rung converged | yes / yes |
| $q$-span at fixed $M=1088$ ($q = 0 \to 256$) | 2.437x |
| $q$-span at fixed $M=256$ ($q = 0 \to 128$) | 1.220x |
| rank claim false ($<1.5\times$ at every fixed $M$) | no |
| H(rank) false (no saturation in $M$) | yes |
| H(tests) false (rank share $\ge 0.5$) | yes |
| corner-path shares: rank / test count | 0.690 / 0.310 |
| rung-path test-count share | 0.344 |
| the two paths disagree by $>0.15$ | no |

**Spans (worst evolved %, same-grid, converged cells only).**

| ladder | rungs | values | monotone | span |
|---|---|---|---|---|
| fixed $M=256$ | $q$ = 0, 16, 32, 64, 128 | 1.2710 / 1.2305 / 1.1869 / 1.1255 / 1.0418 | yes | 1.220x |
| fixed $M=1088$ | $q$ = 0, 16, 32, 64, 128, 256 | 1.2657 / 1.2204 / 1.1629 / 1.0593 / 0.8711 / 0.5194 | yes | 2.437x |
| scheduled `4x` | (0,64), (16,128), (32,192), (64,320), (128,576), (256,1088) | 1.8890 / 1.3985 / 1.2336 / 1.0843 / 0.8930 / 0.5194 | yes | 3.637x |
| scheduled `8x` | (0,128), (16,256), (32,384), (64,640), (128,1152), (256,2176) | 1.4695 / 1.2305 / 1.1665 / 1.0619 / 0.8689 / 0.3736 | yes | 3.933x |

**The pure-rank ladder as an operating-point family, inside one job** (`G2`, 3780177, so the costs are comparable): rungs $q$ = 0, 64, 128, 256 at fixed $M = 1088$, worst evolved 1.2657 / 1.0593 / 0.8711 / 0.5194 %, median GPU 848 / 1359 / 1886 / 4378 ms — error span **2.437x**, cost span **5.163x**, 4 non-dominated points, monotone yes. It **passes** the project's standing tunability bar (monotone, $\ge 3$ non-dominated points, $\ge 2\times$ in both error and cost, nothing early-stopped) **with the test count held fixed**, which is what the audit's objection asked for.

| fixed $q$ | $M$ | values | monotone in $M$ | span (max/min) | span over $M \ge 2(K+q)$ |
|---|---|---|---|---|---|
| 0 | 64, 128, 256, 512, 1088, 2048, 4096 | 1.8890 / 1.4695 / 1.2710 / 1.2662 / 1.2657 / 1.2652 / 1.2649 | yes | 1.493x | 1.493x |
| 16 | 128, 192, 256, 512, 1088 | 1.3985 / 1.2690 / 1.2305 / 1.2204 / 1.2204 | yes | 1.146x | 1.146x |
| 32 | 192, 256, 320, 384, 768, 1088 | 1.2336 / 1.1869 / 1.1717 / 1.1665 / 1.1634 / 1.1629 | yes | 1.061x | 1.061x |
| 64 | 128, 256, 320, 512, 576, 640, 1088, 1280, 2048, 4096 | 1.8032 / 1.1255 / 1.0843 / 1.0647 / 1.0633 / 1.0619 / 1.0593 / 1.0588 / 1.0594 / 1.0620 | no | 1.703x | 1.063x |
| 128 | 256, 576, 1088, 1152 | 1.0418 / 0.8930 / 0.8711 / 0.8689 | yes | 1.199x | 1.028x |
| 256 | 544, 1088, 2176 | 0.7566 / 0.5194 / 0.3736 | yes | 2.025x | 2.025x |

**Decomposition of the scheduled $4(K+q)$ ladder.** Corner path (0,64) -> (0,1088) [M] -> (256,1088) [q]: $\log S = 1.2912$ ($S = 3.637\times$) $= \Delta_M + \Delta_q = 0.4004 + 0.8907$, shares test count 31.0 %, rank 69.0 %.
Rung path through the bridge cells: test count 34.4 %, rank 65.6 %; per rung (0→16: 84 % tests), (16→32: 77 % tests), (32→64: 40 % tests), (64→128: 10 % tests), (128→256: 5 % tests).
Variance shares on the balanced $5 \times 2$ sub-grid: rank 85.1 %, test count 6.1 %, interaction 8.8 %.

**Saturation (S1, 3780178).** $q=0$: $M^\star = 256$, error 1.2710 %, cost 1.210x the $M = 64$ cell (within-job), cost-neutral claim no, monotone in $M$: yes; curve 1.8890 / 1.4695 / 1.2710 / 1.2662 / 1.2657 / 1.2652 / 1.2649 % at $M$ = 64, 128, 256, 512, 1088, 2048, 4096. $q=64$: $M^\star = 256$, error 1.1255 %, cost 0.922x the $M = 320$ cell (within-job), cost-neutral claim yes, monotone in $M$: no; curve 1.8032 / 1.1255 / 1.0843 / 1.0647 / 1.0593 / 1.0594 / 1.0620 % at $M$ = 128, 256, 320, 512, 1088, 2048, 4096.

**Cross-job anchors.** 9 cells appear in two jobs; worst relative difference on either metric 1.44e-09; all within their declared tolerance: yes. No cost ratio was formed across jobs.

**Gates.** G1 (bqx101): `artifacts_present` yes; `at_least_two_unconditional_reproductions_at_1e-9` yes; `backend_gpu` yes; `bank_frozen` yes; `checkpoint_unchanged` yes; `complete` yes; `cross_job_fidelity` yes; `dense_quadrature_everywhere` yes; `directions_hash_matches_btq101` no; `directions_hash_matches_btq102` no; `directions_hash_matches_btq201` no; `directions_hash_matches_cclad01` no; `directions_hash_matches_qrg201` no; `directions_hash_matches_qtd02` no; `directions_rank_covers_ladder` yes; `evaluation_cohort_bitwise_abl01` yes; `every_declared_cell_timed` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `operators_shared_bitwise` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_fields_bitwise_match_btq101` no; `reference_fields_bitwise_match_btq102` yes; `reference_fields_bitwise_match_btq201` no; `reference_fields_bitwise_match_cclad01` no; `reference_residuals` yes; `repetition_output_identical` yes; `same_grid_baseline_present` yes; `step_budget_600` yes; `t0_field_bitwise_in_M` yes; `t0_field_invariant_in_M` yes; `t0_field_invariant_in_M_1e-12` yes; `x64` yes. G2 (bqx201): `artifacts_present` yes; `at_least_two_unconditional_reproductions_at_1e-9` yes; `backend_gpu` yes; `bank_frozen` yes; `checkpoint_unchanged` yes; `complete` yes; `cross_job_fidelity` yes; `dense_quadrature_everywhere` yes; `directions_hash_matches_btq101` no; `directions_hash_matches_btq102` no; `directions_hash_matches_btq201` no; `directions_hash_matches_cclad01` no; `directions_hash_matches_qrg201` no; `directions_hash_matches_qtd02` no; `directions_rank_covers_ladder` yes; `evaluation_cohort_bitwise_abl01` yes; `every_declared_cell_timed` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `operators_shared_bitwise` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_fields_bitwise_match_btq101` no; `reference_fields_bitwise_match_btq102` no; `reference_fields_bitwise_match_btq201` yes; `reference_fields_bitwise_match_cclad01` no; `reference_residuals` yes; `repetition_output_identical` yes; `same_grid_baseline_present` yes; `step_budget_600` yes; `t0_field_bitwise_in_M` yes; `t0_field_invariant_in_M` yes; `t0_field_invariant_in_M_1e-12` yes; `x64` yes. S1 (bqx301): `artifacts_present` yes; `at_least_two_unconditional_reproductions_at_1e-9` yes; `backend_gpu` yes; `bank_frozen` yes; `checkpoint_unchanged` yes; `complete` yes; `cross_job_fidelity` yes; `dense_quadrature_everywhere` yes; `directions_hash_matches_btq101` no; `directions_hash_matches_btq102` no; `directions_hash_matches_btq201` no; `directions_hash_matches_cclad01` no; `directions_hash_matches_qrg201` no; `directions_hash_matches_qtd02` no; `directions_rank_covers_ladder` yes; `evaluation_cohort_bitwise_abl01` yes; `every_declared_cell_timed` yes; `every_invocation_paired` yes; `every_rom_carries_exit_and_stationarity` yes; `every_subject_case_has_all_reps` yes; `final_cohort_unopened` yes; `operators_shared_bitwise` yes; `overdetermined_weak_system` yes; `precision_highest` yes; `recorded_errors_recomputed_from_saved_fields` yes; `reference_fields_bitwise_match_btq101` no; `reference_fields_bitwise_match_btq102` no; `reference_fields_bitwise_match_btq201` yes; `reference_fields_bitwise_match_cclad01` no; `reference_residuals` yes; `repetition_output_identical` yes; `same_grid_baseline_present` yes; `step_budget_600` yes; `t0_field_bitwise_in_M` yes; `t0_field_invariant_in_M` yes; `t0_field_invariant_in_M_1e-12` yes; `x64` yes.

**Cross-job fidelity.**

| job | arm | source / comparator | unconditional | tolerance | achieved | passed |
|---|---|---|---|---|---|---|
| bqx101 | `q0_M128_dense` | qrg201 `q0_m8_dense_l0` | yes | 1e-09 | 3.42e-13 | yes |
| bqx101 | `q0_M256_dense` | cclad01 `q0_m256_dense_block` | yes | 1e-09 | 1.48e-14 | yes |
| bqx101 | `q0_M256_dense` | qrg201 `q0_m16_dense_l0` | yes | 1e-09 | 5.44e-13 | yes |
| bqx101 | `q0_M256_dense` | qtd02 `q0_M256_dense` | yes | 1e-09 | 7.58e-13 | yes |
| bqx101 | `q0_M64_dense` | btq101 `q0_m4_dense_base` | yes | 1e-09 | 5.31e-13 | yes |
| bqx101 | `q0_M64_dense` | cclad01 `q0_m4_dense_block` | yes | 1e-09 | 6.95e-13 | yes |
| bqx101 | `q0_M64_dense` | qrg201 `q0_m4_dense_l0` | yes | 1e-09 | 4.62e-13 | yes |
| bqx101 | `q0_M64_dense` | qtd02 `q0_M64_dense` | yes | 1e-09 | 6.78e-13 | yes |
| bqx101 | `q16_M128_dense` | cclad01 `q16_m4_dense_block` | no | 1e-03 | 1.09e-10 | yes |
| bqx101 | `q16_M128_dense` | qrg201 `q16_m4_dense_l0` | no | 1e-03 | 1.09e-10 | yes |
| bqx101 | `q16_M128_dense` | qtd02 `old_q16_M128_dense` | no | 1e-03 | 2.60e-10 | yes |
| bqx101 | `q16_M256_dense` | cclad01 `q16_m256_dense_block` | no | 1e-03 | 5.16e-11 | yes |
| bqx101 | `q16_M256_dense` | qrg201 `q16_m8_dense_l0` | no | 1e-03 | 5.16e-11 | yes |
| bqx101 | `q16_M256_dense` | qtd02 `old_q16_M256_dense` | no | 1e-03 | 1.07e-10 | yes |
| bqx101 | `q16_M512_dense` | qrg201 `q16_m16_dense_l0` | no | 1e-03 | 5.16e-11 | yes |
| bqx101 | `q32_M192_dense` | cclad01 `q32_m4_dense_block` | no | 1e-03 | 2.64e-11 | yes |
| bqx101 | `q32_M192_dense` | qtd02 `old_q32_M192_dense` | no | 1e-03 | 9.81e-11 | yes |
| bqx101 | `q32_M256_dense` | cclad01 `q32_m256_dense_block` | no | 1e-03 | 4.96e-11 | yes |
| bqx101 | `q32_M256_dense` | qtd02 `old_q32_M256_dense` | no | 1e-03 | 4.74e-11 | yes |
| bqx201 | `q0_M256_dense` | cclad01 `q0_m256_dense_block` | yes | 1e-09 | 1.64e-13 | yes |
| bqx201 | `q0_M256_dense` | qrg201 `q0_m16_dense_l0` | yes | 1e-09 | 7.14e-13 | yes |
| bqx201 | `q0_M256_dense` | qtd02 `q0_M256_dense` | yes | 1e-09 | 9.28e-13 | yes |
| bqx201 | `q128_M256_dense` | cclad01 `q128_m256_dense_block` | no | 1e-03 | 6.66e-10 | yes |
| bqx201 | `q128_M256_dense` | qtd02 `old_q128_M256_dense` | no | 1e-03 | 6.44e-09 | yes |
| bqx201 | `q128_M576_dense` | cclad01 `q128_m4_dense_block` | no | 1e-03 | 2.95e-09 | yes |
| bqx201 | `q128_M576_dense` | qtd02 `old_q128_M576_dense` | no | 1e-03 | 3.96e-09 | yes |
| bqx201 | `q256_M1088_dense` | cclad01 `q256_m4_dense_block` | no | 1e-03 | 3.02e-09 | yes |
| bqx201 | `q256_M1088_dense` | qtd02 `old_q256_M1088_dense` | no | 1e-03 | 9.16e-09 | yes |
| bqx201 | `q256_M544_dense` | btq102 `q256_m2_dense_base_b600` | no | 1e-03 | 5.88e-09 | yes |
| bqx201 | `q64_M1280_dense` | qrg201 `q64_m16_dense_l0` | no | 1e-03 | 1.20e-09 | yes |
| bqx201 | `q64_M256_dense` | cclad01 `q64_m256_dense_block` | no | 1e-03 | 1.46e-09 | yes |
| bqx201 | `q64_M256_dense` | qtd02 `old_q64_M256_dense` | no | 1e-03 | 2.27e-09 | yes |
| bqx201 | `q64_M320_dense` | cclad01 `q64_m4_dense_block` | no | 1e-03 | 1.20e-09 | yes |
| bqx201 | `q64_M320_dense` | qrg201 `q64_m4_dense_l0` | no | 1e-03 | 1.20e-09 | yes |
| bqx201 | `q64_M320_dense` | qtd02 `old_q64_M320_dense` | no | 1e-03 | 2.27e-09 | yes |
| bqx201 | `q64_M640_dense` | qrg201 `q64_m8_dense_l0` | no | 1e-03 | 1.20e-09 | yes |
| bqx301 | `q0_M128_dense` | qrg201 `q0_m8_dense_l0` | yes | 1e-09 | 2.06e-14 | yes |
| bqx301 | `q0_M256_dense` | cclad01 `q0_m256_dense_block` | yes | 1e-09 | 5.35e-13 | yes |
| bqx301 | `q0_M256_dense` | qrg201 `q0_m16_dense_l0` | yes | 1e-09 | 1.49e-14 | yes |
| bqx301 | `q0_M256_dense` | qtd02 `q0_M256_dense` | yes | 1e-09 | 2.29e-13 | yes |
| bqx301 | `q0_M64_dense` | btq101 `q0_m4_dense_base` | yes | 1e-09 | 5.22e-13 | yes |
| bqx301 | `q0_M64_dense` | cclad01 `q0_m4_dense_block` | yes | 1e-09 | 3.26e-13 | yes |
| bqx301 | `q0_M64_dense` | qrg201 `q0_m4_dense_l0` | yes | 1e-09 | 2.80e-13 | yes |
| bqx301 | `q0_M64_dense` | qtd02 `q0_M64_dense` | yes | 1e-09 | 1.25e-13 | yes |
| bqx301 | `q64_M256_dense` | cclad01 `q64_m256_dense_block` | no | 1e-03 | 2.07e-11 | yes |
| bqx301 | `q64_M256_dense` | qtd02 `old_q64_M256_dense` | no | 1e-03 | 1.05e-09 | yes |
| bqx301 | `q64_M320_dense` | cclad01 `q64_m4_dense_block` | no | 1e-03 | 1.94e-11 | yes |
| bqx301 | `q64_M320_dense` | qrg201 `q64_m4_dense_l0` | no | 1e-03 | 1.94e-11 | yes |
| bqx301 | `q64_M320_dense` | qtd02 `old_q64_M320_dense` | no | 1e-03 | 1.05e-09 | yes |

**What was wrong and got retracted.**

- **My own first reading of the local smoke was wrong, and it nearly became a code bug.** `smoke_xm.py`'s
  gate 1 — the retained $q=0$ path at 64 intervals against the consolidated audited fixture — returned
  $2.14\times10^{-7}$ on two consecutive runs, five orders above its $10^{-12}$ bar, while the *identical*
  construction in `smoke_gate1.py` returned $1.56\times10^{-14}$ three times in the same minutes. GPU
  contention was ruled out (the passing runs were at 96 % utilisation). The difference is the initial-condition
  Levenberg–Marquardt fit landing at a different point *inside* its own $10^{-6}$ stationarity tolerance —
  111 iterations in the failing process against 86 in the passing one — which an ulp-level GEMM difference is
  enough to cause when XLA's GPU autotuner picks a different algorithm on a contended shared box. **Nothing
  in the lane's code was wrong**; what was wrong was the gate's premise that a $10^{-12}$ field bar is
  reachable across processes on the shared GB10. Consequences, all applied before submission: the $t=0$
  invariance gate is blocking at $10^{-6}$ with $10^{-12}$ and bitwise as probes (DESIGN §9 A0); gate 1 now
  records its IC iteration count and `XLA_FLAGS` and is asserted at the end so the other gates still produce
  data. On the cluster — one process, one exclusively allocated GPU — the question is settled empirically by
  the in-job fidelity gates, not assumed.
- **"The arrays and peak workspace this lane needs ($\le$ ~10 GB, measured by `probe_memory.py`)" was an
  extrapolation stated as a measurement**, and the pre-submission audit caught it. At the time it was written
  the probe had measured 6.6 GB at $(256, 2176)$ and had not run its largest cell. It finished later at
  11.8 GB for $(64, 4096)$ — *above* the claimed bound. The sentence was corrected to say what was measured
  before any job was staged.
- **The memory probe itself was a process error.** It held one of three shared local GPU slots for 27 minutes
  against a sub-minute rule for local runs, with eight other lanes queued behind it, and GB10 numbers under a
  36 GB unified-memory cgroup do not predict an A100 anyway. It was not restarted; memory was handled by
  design instead (the grid is split into three jobs by $q$, each with its own attempt directory and its own
  FOM controls). Recorded as a deviation in DESIGN §9 A0.
- **Codex was unavailable** (account over its usage limit until 2026-09-19), so the protocol's independent
  audit of DESIGN.md was run by a fresh Claude Opus agent with no lane context, read-only. That is a weaker
  control than a different model family: it is the same family as the author. Its 14 findings and their
  disposition are in `reports/design-audit.md`; two were blocking and both were fixed before the first
  submission (S1 was missing the $(64, 320)$ cell its own cost ratios needed; the memory claim above). The
  final report has **not** had a cross-family audit, and this lane does not claim one.
- **Four design defects the audit found would each have produced a wrong or unfalsifiable number**, and are
  recorded because they were in the pre-registered design, not in a draft: the fixed-$M$ span silently
  dropped a non-converged rung and quoted the span over the shortened ladder (now reported *unavailable*);
  a cross-job fidelity pair whose comparator carried a null metric would have passed while comparing nothing,
  and counted toward the "two unconditional reproductions" gate (now requires two compared metrics); the
  H(rank) falsification lever could have been flipped by the single near-square $(64, 128)$ cell (now reads
  only cells with $M \ge 2(K+q)$); and the inner linear solve switches from Gauss–Jordan to LU at exactly the
  rung where the fixed-$M$ ladder crosses between jobs, confounding solver with rank (now controlled by a
  duplicate $(32, 1088)$ arm run with LU in the same job as its Gauss–Jordan twin).
- **The report generator's first within-job cost ladder was wrong, and it under-reported the lane's own
  headline.** It searched only each cell's *primary* job, so for the fixed-$M=1088$ column it found G1's
  three rungs ($q = 0, 16, 32$: error span $1.088\times$, cost span $1.291\times$, **fails** the tunability
  bar) and never saw that G2 holds four rungs of the same column ($q = 0, 64, 128, 256$) because G2 runs the
  $q = 0$ cell as an in-job anchor — which is exactly why the anchors were pre-registered. Corrected to
  search every appearance: the ladder is error span $2.437\times$, cost span $5.163\times$, four
  non-dominated points, and it **passes**. The wrong version existed for one generator run and never left
  the worktree, but it would have inverted the lane's recommendation.
- **No cross-family audit of the final report exists.** Codex remained over quota, so the numbers were
  checked by `checks/selfaudit.py` — an independent recomputation sharing no code with the generator, which
  re-derives three cells' errors from the archived field tarballs and verifies the decomposition identities
  — and all 27 checks agree. That is a self-check by the same author, not the protocol's independent
  auditor, and the lane says so rather than implying otherwise.
- **I hand-typed a number into DESIGN §A2 and it was wrong.** The solver control's
  evolved-metric agreement was written as $4.7\times10^{-14}$ from memory; read from
  `summary.json` it is exactly 0.0 — the Gauss–Jordan and LU solves of the $(32, 1088)$
  cell agree to the last bit. Corrected in place, and the paragraph now quotes the generated
  value. No other prose number in this lane was typed by hand; every table is generated.

Source-generated report: `experiments/b-qxm/reports/2026-09-17-b-qxm.md` (SHA256 `e7b96d09101236761aff2f0ede853aad9bd469304eeccd6c3150a05273061363`) with `summary.json`, `analysis.json`, its two figures and its generator beside it; the design audit is `experiments/b-qxm/reports/design-audit.md`.
Raw archive `bqx101` Git-tracked as bounded chunks: whole SHA256 `fb4a39e21b6e3f7e21c839e6edc7250cc03dbfd975305afaa97382ff296878e5` (9 chunks).
Raw archive `bqx201` Git-tracked as bounded chunks: whole SHA256 `b447b3ab0bdd88d3aa6ef8ec7fb60449db34e8a47a64ead1d8f3d115cc165a97` (8 chunks).
Raw archive `bqx301` Git-tracked as bounded chunks: whole SHA256 `721d56b1dcf82b2b22343c737f62e741e9099c61eb36022f840bf2add569f469` (8 chunks).

