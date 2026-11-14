# iso-error: accuracy against query time across span widths (Burgers 2D, 256², test64)

Written 2026-11-13 before any width other than the existing three (384, 96, 128) was run on the test cases, and
before the reproduction job's result was read. Amendments, if any, are appended at the end with time and reason.

## 0. Question

The rebuttal table compares one row of ours (span width 384: 0.026 % median / 0.875 % worst / 45.2 ms) with neural
operators that are about ten times less accurate in the median. That is not a like-for-like comparison. This lane
measures our method's own accuracy-against-time curve by changing only the span width, and then reads off, for fixed
median-error levels, our cheapest passing width next to the fastest operator that reaches the same level.

## 1. Provenance of the existing row, and the reproduction check

- Source: lane `burgers2d-test`, attempt `t256`, Slurm job 4319328 (A100 80GB PCIe, node pax105), source commit
  1b7b7f96bcf5808d397d467b5e99ae287f775d07, driver `b2test.py`, config `config-t256.json`.
  Stored values (vendor/ref/t256-summary.json): width 384 median 0.02641921794844461 %, worst 0.8748597576173374 %,
  45.23743106983602 ms; width 96: 0.5293 % / 3.8456 % / 14.244 ms; width 128: 0.2675 % / 2.5469 % / 18.279 ms.
- The staged job tree of that attempt is vendored byte-identically under `vendor/experiments` (`diff -r` identical
  to the original staged tree; MANIFEST-t256.sha256).
- Reproduction job `repro256`: the same driver, config, checkpoint, rotation and sbatch text (only job name,
  namespace and wall time differ), on the same GPU type.
- Acceptance, fixed here: for widths 384, 96 and 128, median and worst-case error equal the stored values to the
  printed digits (four significant digits); the arm's median time is within ±10 % of the stored time (the lane's own
  drift/neighbour gate limit is 1.1; a different node of the same GPU type is expected inside it); all gates of the
  driver pass; the log shows `jax_backend=gpu`. If the errors do not match, the lane stops and reports why.

## 2. What is frozen (nothing is tuned on the test cohort)

Everything except the span width R' is the Table-1 setting of the existing row:

| item | value | where fixed |
|---|---|---|
| checkpoint | `dn256b` sep_hfit_dense_mid_N256_dense.pkl, sha256 18f0266a…d6b589 | burgers2d-test DESIGN.md |
| rotation (bank order) | rotation_R512.npz, sha256 51149166…6426772; 512 rotated columns | burgers-bank-knob |
| arm | linear rung: span of the first R' rotated bank columns, head dropped, R' unknowns | burgers-bank-knob |
| test modes | M = 4 R' sine modes | construction rule `lin(L, Rp)` in burgers2d-speed/make_configs.py (dev-time) |
| quadrature | `lat64`: 63×63 interior lattice, 3969 points, equal weights | DESIGN.md |
| solver | Levenberg–Marquardt, Cholesky, step clip, damping carry, quadratic predictor; no cap; no exact first step | DESIGN.md |
| tolerances | gtol 1e-2, step budget 600, inner damping 1e-10 | config-t256.json |
| implementation | `eng` | DESIGN.md |
| time step, outputs | dt 0.005, six output times 0…0.25 | config-t256.json |

Per-width settings. The only setting that was chosen per width at development time is the compile mode (`graphs`
= XLA command-buffer options, or default). It does not change the computed fields beyond floating-point reordering.
- Widths 96, 128, 192, 256, 384: the mode frozen on the six development cases (dev6 panel, burgers2d-speed
  `checks/b256-summary.json`, job 4241033, field `knobs[*].timed_arm`; copy in vendor/ref/dev6-b256-summary.json):
  384 graphs, 256 default, 192 default, 128 graphs, 96 graphs. `make_config.py` asserts this against that file.
- Widths 32, 48, 64, 512: there is no development record at 256². Rule: hold the width-384 setting (graphs).
- Both modes are run and timed for every width. The **primary** time of a width is the mode fixed by the rule above.
  The other mode is reported in a secondary column and is never used to pick a number for the tables. If the other
  mode is faster at some width, that is stated; the primary stays.

Widths: 32, 48, 64, 96, 128, 192, 256, 384, 512 (the bank has 512 columns, so 512 is included). The spans are
nested: width R' uses the first R' columns of the same rotated bank.

## 3. The sweep job

- Driver `b2iso.py` = `b2test.py` with two additions that do not touch the query or the A–B–A timing: it keeps each
  arm's six retained coefficient states, and after the A–B–A phases it times a reconstruction-only kernel (section 5).
- Config `config-sweep256.json` = `config-t256.json` with exactly these fields changed (`make_config.py` asserts the
  list): `arms` (9 widths × 2 modes), `ladder` (48 inserted: the driver can only cut the bank at block edges; this
  changes the column-block boundaries of the stored bank, not its columns), `audit_arms`, `test_arms`, `attempt`,
  `purpose`, `save_restricted` (off; disk), `recon_reps` (2). Cohort, reference, full-order ladder, timing rule and
  gates are unchanged.
- Consistency check inside the sweep: widths 384, 96, 128 are re-run in the sweep job. Their errors must again equal
  the stored values to the printed digits (the block boundaries can change the last floating-point bits, not more),
  and their times are compared with the reproduction job and the stored row. A disagreement is reported, not hidden.
- GPU: A100 80GB PCIe (`--gres=gpu:a100:1 --constraint=a100-80G --exclude=pax007`), `gpu` partition, one GPU,
  8 CPUs, 128 G, x64, matmul precision highest — the sbatch text of job 4319328.

## 4. Cohort, reference, error

- Cohort: test64 = `params_draw(20260916, 64)`, physical sha256 cd058fd2…6044527c (the driver asserts it). The same
  64 cases as the existing row and as the operator test evaluations.
- Reference: `fft_tight` full-order solve on the same 256² mesh (Newton tolerance 1e-6).
- Error per case: max over the five evolved output times of ‖u − u_ref‖ / ‖u_0‖, in percent.
- Reported per width over the 64 cases: median, mean, 90th percentile (numpy linear interpolation), worst case.
  Also: Levenberg–Marquardt iterations per query (median, max) and the number of stalled time-step exits.

## 5. Time

- Rule: DESIGN.md §4 of burgers2d-test, unchanged. Every subject warmed up; A–B–A (ROM phase A1: 2 reps × 64 cases,
  full-order phase B, ROM phase A2: 2 reps × 64 cases); order randomised within a case (seed 20260924); 5 s cool-down
  and 2 s dummy kernel between phases; 0.1 s burn before each invocation; `block_until_ready` on both sides; the time
  runs from the dense input field resident on the GPU to six dense output fields on the GPU.
- Reported per width: median over the 256 A1 ∪ A2 invocations (the statistic of the existing row), and the minimum
  and maximum over the same invocations; per-phase medians and the drift and neighbour gates as in the parent.
  Note: the per-invocation spread includes case-to-case differences in iteration count, not only timing noise.
- More subjects share the round-robin than in job 4319328 (18 reduced arms instead of 3). The neighbour gate and the
  comparison of widths 384/96/128 against the reproduction job measure whether that matters.
- Split. The query is one fused compiled function (initial fit → 50 implicit time steps → reconstruction of six
  fields); the code does not expose a split. Method fixed here: a separately compiled reconstruction-only kernel
  with the query's own reconstruction text (bank prefix applied to the six retained coefficient vectors, zero
  padding), same compile mode as the arm, timed after the A–B–A phases with the same conventions (warm-up, burn,
  `block_until_ready`, random order, 2 reps × 64 cases). Reported: "field reconstruction" = median of that kernel;
  "reduced solve (incl. initial fit)" = query median − reconstruction median. The second number is **derived**, not
  measured directly, and inside the fused function the compiler may schedule reconstruction differently; it is
  labelled so.

## 6. Iso-error table

- Levels (median relative error on test64): 0.2 %, 0.1 %, 0.05 %, 0.026 %.
- Ours: the smallest width in the sweep whose test median is ≤ the level, with its primary time. No interpolation
  between widths. The 0.026 % level is the printed median of the existing row; it is met by a median that prints as
  ≤ 0.026 % at three decimals (i.e. < 0.0265 %), which is how the operator lane's target (≤ 0.0264 %) is phrased.
- This is a read-off of the test curve at fixed levels, not a selection of a model for deployment; no setting is
  chosen from it. It is stated as such in RESULTS.md.
- Operators: finished numbers only, read from
  `…/scratchpad/nmrom/push/burgers-ops-status.md` and `…/push/operator-scaling-status.md`, cited by file and line and
  by the time the file was read. An operator meets a level only with a **test64** evaluation and a query time under
  the same timing rule. Where none does: "not reached yet (best so far: X % validation, no test evaluation)".
  No operator time is estimated or extrapolated, and this lane evaluates no operator.
- The worst-case error is reported next to every entry. On worst case the operators currently lead
  (0.43–0.51 % against our 0.875 % at width 384); the table says so whatever the sweep shows.

## 7. The nonlinear-head row

Cited, not re-run: 16 coordinates, 4.10 % worst / 0.672 % median / 32.4 ms, job 4327396
(`experiments/burgers-ops-matched/reports/tables.md` line 9). It is a different arm family and timing job; it is
placed beside the width curve for context only.

## 8. Reporting rules

- All numbers in RESULTS.md are generated by `summarise.py` from the saved `result.json` files, not typed.
- Every width is reported, including any that is slower and less accurate than a neighbour, stalls, or fails.
- If a gate of the driver fails (cohort hash, operator parity, drift, neighbour, repetition identity), the affected
  numbers are reported with the failure named.
- Known expectation, stated before measuring: at the 0.2 % level the operators (0.19–0.22 % median at 4–7 ms) are
  probably faster than our narrowest passing width (width 128 was 0.27 % at 18.3 ms in job 4319328, so the passing
  width is likely 192 or wider). If so, RESULTS.md says so plainly.
- Budget: at most 2 GPU jobs of this lane queued or running, one GPU each, short wall times.
