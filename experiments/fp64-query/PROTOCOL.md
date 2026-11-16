# fp64-query: the Burgers 2D query-speed claim with every array operation in float64

Lane `worktrees/2026-11-15-fp64-query`, branch `exp/2026-11-15-fp64-query`, forked from `main` @ 274a3a9af.
Cluster namespace `/cluster/tufts/paralab/tawal01/fp64-query-20261115/`. Written 2026-11-15, before any job of this
lane was submitted and before any outcome of this lane existed.

## 1. Question

The b2d-query-speed lane (`exp/2026-11-14-burgers2d-query-speed` @ c8bd352aa, RESULTS.md there; test64 job 7220630,
fresh64 job 7220632, A100 80GB) reports Burgers 2D 256^2 single-query latency 5.6-10x below the unchanged baseline at
unchanged median error, and width 176 at 0.1207 % median in 2.74 ms against FNO 0.22 % / 4.07 ms and U-Net
0.21 % / 7.37 ms. The JCP readiness audit (`reports/2026-11-15-jcp-submission-readiness.md`, main) notes that the
fast variants use single precision for the stencil. Question: with the SAME frozen settings and no re-tuning, what
are the errors and latencies when every array operation is float64, measured in one allocation next to the original
mixed-precision settings, the unchanged baseline and the operators?

## 2. Precision audit from the code (done before any job)

Source: `src/qsfast.py`, `src/qsdrive.py`, `src/frozen.json`, `src/ops/opnet.py` (byte-identical copies of c8bd352aa)
and the vendored tree `vendor/experiments/` (iso-error vendor = burgers2d-test @ 1b7b7f96).

- **Every frozen qs setting is mixed precision.** `frozen.json` sets `"stencil32": true` for every frozen setting:
  "accurate" at widths 96, 128, 144, 160, 176, 192, 256, 384, 512 and "fast" at 256, 384, 512. There is no frozen
  setting without it. So every speed number of the query-speed lane for "accurate" and "fast" comes from a
  mixed-precision solver.
- **What runs in float32** (`qsfast.adv_of`, `stencil32=True` branch; `qsdrive.build_arm` adds
  `data['G5d32'] = QS.stencil_diff32(data['G5'])`): the stencil columns (c, c - xm, xp - c, c - ym, yp - c) are
  differenced in float64 once per arm and stored as float32; per correction, w is cast to float32, the contraction
  `einsum('msr,r->ms', G5d32, w32)` (m quadrature nodes x 5 x R'), and the pointwise upwind product
  c L (upwind x-difference + upwind y-difference) run in float32; the result is cast back to float64.
  This is the advection evaluation inside every correction: in the fused form (`form='fused'`, all "accurate"
  settings) `w <- Kw w + Cp w_prev - Qt adv(G5 w)`; in the pseudo-inverse form (`form='P'`, the "fast" settings) the
  residual `A w - S.(A w_prev) + dt S.Pq^T adv(G5 w)`.
- **What stays float64 in those settings:** the reference Jacobian (`jac` uses the float64 `G5` and the float64
  derivative of `advect_points`), its Gram matrix and Cholesky factorisation, the pseudo-inverse and fused step
  matrices (Kw, Cp, Qt), the predictor, every update of w, the projection Pq^T, the initial fit and the decoder
  (`bank_apply`, float64 bank).
- **The unchanged baseline** (`b2fast.make_linear_query`, eng/chol path) contains no float32 cast (grep of the
  vendored tree: the only float32 code is `b-panel/speed/fast.py` `decode='lean_f32'`, not imported on this path).
- **Operators** (`ops/opnet.py`): FNO and U-Net networks run in float32 with TF32 disabled
  (`torch.backends.cuda.matmul.allow_tf32 = False`, `cudnn.allow_tf32 = False`, float32 matmul precision
  'highest'); inputs and outputs are converted from and to float64 around the network. This is their registered
  precision and is kept.
- `_graphs` in an arm name is a compile mode (XLA command buffers / CUDA graphs), not a precision setting.
- JAX runs with `jax_enable_x64` and `JAX_DEFAULT_MATMUL_PRECISION=highest` (float32 contractions are full float32,
  no TF32).

The code audit is confirmed in the job by a traced-program audit (section 6, Q6): for every ROM arm the jaxpr of the
query is walked and every variable is counted by dtype.

## 3. Cohorts and the status of re-evaluation

- test64 = `params_draw(20260916, 64)`, SHA cd058fd2...6044527c. It was opened by the query-speed lane (job 7220630)
  after its freeze. It is evaluated again here with the frozen settings unchanged. **This is not tuning**: no setting,
  width, tolerance, or selection is chosen from it, before or after; the float64 arm differs from the frozen setting
  only in the precision of the stencil evaluation, fixed by this protocol.
- fresh64 = `params_draw(2026111402, 64)`, opened once by the query-speed lane (job 7220632). The query-speed lane
  scored it, so it is scored again here, with the same caveat (frozen settings re-evaluated, nothing selected).
- No development cohort is used for any decision. A rehearsal on six dev38 cases (`p64dev`, widths 176 and 384)
  checks the pipeline (audit, timing, operator step) only; its numbers are not reported and do not change any setting.
  Only code defects found by it may be fixed (recorded in section 8).

## 4. Arms (fixed now)

Per width R' in {96, 128, 176, 192, 256, 384} (512 dropped: not part of the rebuttal claim):
- the unchanged baseline of `src/configs/ftest64.json` (same tolerance, compile mode);
- the frozen "accurate" setting, unchanged (mixed precision; the control);
- the same setting with `qs.stencil32 = false` (`<role>_fp64`): the solver's own float64 path,
  `advect_points(einsum('msr,r->ms', G5, w))` with the float64 stencil; nothing else changes (same k, s, form,
  extra early corrections, predictor, ridge, compile mode);
- at 256 and 384 also the frozen "fast" setting and its float64 copy.
22 ROM arms per job. Configs are produced by `make_fp64_config.py` from the byte-identical query-speed final configs;
all other fields (cohort, truth solver `fft_tight`, timing rule, gates, order seed) are those configs unchanged.
Operators: the six frozen 4,608-trajectory checkpoints (F0 FNO seeds 0-2, U0 U-Net seeds 0-2) listed in
`src/ops/selection_manifest.json`, re-timed and scored in the same job on the same GPU by the same `evaluate.py`
(batch 1, 20 burn-in calls, 5 repetitions per case, synchronised); test64 reference fields from
`rebuttal-burgers-ops/data/test` (as 7220630), fresh64 references generated in the job by `gen_cohort.py`
(as 7220632). Nothing is retrained.

## 5. Jobs, metric, timing

- `p64test` (test64) and `p64fresh` (fresh64): one job each, A100 80GB (`--constraint=a100-80G`, same GPU type as
  7220630), each a single allocation holding all 22 ROM arms and the six operators. The test64 job carries the
  headline. Wall 1 h each.
- Error per case: max over the five evolved output times of ||u - u_ref|| / ||u_ref(0)|| against `fft_tight`
  (computed in the job); median and worst over the 64 cases.
- Timing: the query-speed rule unchanged (A-B-A, randomised order seed 20260924, 0.1 s burn, cooldown between phases,
  block_until_ready, warm-up excluded, 2 x 2 repetitions per case, median over invocations; spread = 5-95 % range of
  per-case medians and min/max; drift and neighbour gates at 1.1).
- Derived: speedup vs the baseline of the same width (baseline ms / arm ms); float64 slowdown =
  ms(fp64 arm) / ms(original mixed arm); error ratios fp64 / original and fp64 / baseline.

## 6. Code changes (all committed before the first job)

- `src/qsdrive.py`: Q6 precision audit (`precision_audit` config flag; `require_fp64` on every float64 arm aborts the
  job if any reduced-precision dtype appears in the traced program or the inputs). No other change.
- `make_fp64_config.py` (new), `cluster/*.sh` (query-speed scripts with lane paths, namespace and job prefix changed;
  `post-ops.sh` with the ops path changed), `summarise_fp64.py` (new, generates RESULTS.md from the job JSON).

## 7. Pre-registered headline rule

**The rebuttal's speed headline becomes whatever the float64 numbers of `p64test` support**, whatever they are:
- Speedup headline = range over the six widths of (baseline ms / fp64 "accurate" ms) in p64test, reported with the
  fp64 errors measured in the same job. "Unchanged median error" may be said only for widths where the fp64
  accurate median is within 2 % of the same-job baseline median (the selection bar); otherwise the measured errors
  are stated.
- Operator comparison = fp64 "accurate" at width 176 (the frozen equal-error width; not re-chosen) against the FNO
  and U-Net seed-medians measured in p64test: the claim "more accurate and faster" stands only if both the median
  error and the median ms are lower; if the float64 ms is above an operator's, the headline says so.
- If the float64 numbers are worse than the mixed-precision ones, the mixed-precision numbers may be quoted only as
  a separately labelled mixed-precision result, never as the headline.
- fresh64 is reported beside test64 and may only weaken, not strengthen, the headline.

## 8. Stopping rule and failures

One job per cohort. An identical resubmission is allowed once only for an infrastructure failure (preflight exit 42,
node fault, out-of-memory, wall time) or a failed drift gate; both attempts are then reported. No resubmission for
an outcome. A float64 arm that is non-finite is recorded as rejected (query-speed amendment A2 rule) and reported.

## 9. Amendments



**A1 (2026-11-15, before any test64/fresh64 job).** Rehearsal job 7250249 (`p64dev`, six dev cases) stopped in the
Q6 audit before any ROM arm ran: `jax.make_jaxpr` around the arm's query nests a jit that carries `compiler_options`
(the `_graphs` compile mode), which JAX refuses. Fix: the audit traces the arm's own top-level jit with `.trace(...)`
(same program, options kept). No setting changed. Rehearsal resubmitted as `p64dev2`; log of 7250249 kept in
`results/p64dev-failed/`.

**A2 (2026-11-16, after p64test 7250285 and p64fresh 7250286).** Presentation only: `summarise_fp64.py` gained,
after the outcomes were seen, (a) the largest per-case relative error change between each float64 arm and its
mixed-precision original and (b) the generated headline / rebuttal-sentence section applying section 7. No arm,
cohort, metric or rule changed; both jobs passed both gates and are the single evaluation per cohort.
