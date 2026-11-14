# b2d-worstcase: reduce the worst-case error of the linear rung on Burgers 2D (256²)

Lane `exp/2026-11-14-burgers2d-worstcase` (from `exp/2026-11-13-iso-error` @ 2fd48d20f). Cluster namespace
`/cluster/tufts/paralab/tawal01/b2d-worstcase-20261114/`. Written and committed **before any outcome of this lane
was produced or read**. Amendments are appended at the end with their reason and the time they were made.

## 1. Question

On the 64 held-out cases (test64) the deployed linear rung (span width 384) has median 0.026 % and worst case
0.875 % at 45.2 ms; the operators trained on the same 4,608 trajectories have worst case 0.43–0.51 %, the CoLoRA
re-implementation 0.240 %. Can the worst case of *our* method be brought below 0.43 % (ideally below 0.24 %) by
improving our own training, hyper-parameters or online solve, keeping the median advantage and a comparable query
time? Comparators are not touched.

## 2. Cohorts

| name | cases | role |
|---|---|---|
| train | `P[0:4352]`, `P = params_draw(0,576) ++ params_draw(1000,4032)` | every fit of this lane (bank, ordering, trust radius) |
| validation | `P[4352:4608]` (256 cases; = `params_draw(1000,4032)[3776:4032]`) | all diagnosis, tuning, model selection |
| test64 | `params_draw(20260916, 64)`, sha256 `cd058fd2…` | opened once, after the freeze |
| fresh64 | `params_draw(20261114, 64)` (seed fixed here, before drawing) | opened once, after the freeze |

- train/validation is the split of both baseline lanes (operators: 4,352 + 256 carved from the pool; CoLoRA:
  `P[0:4352]` / `P[4352:4608]`). Nothing outside `P` is used for training; if that changes it is declared in an
  amendment with the amount.
- Training fields: the baseline lanes' pool (`rebuttal-nfrom/data/pool26.fields.npy`, 26 time levels
  t = 0, 0.01, …, 0.25 per trajectory, `fft_tight`, read-only). Validation / test / fresh references are recomputed
  in-job by the reference solver (`fft_tight`, Newton 1e-6, linear 1e-8, same 256 mesh, float64).
- Declared leak in favour of the *baseline* on validation only: the deployed checkpoint's head was fitted on all of
  `P[0:4608]`, so the baseline's column ordering and trust radius have seen the validation trajectories (its bank
  has not: it was trained on `P[0:576]`). Anything new in this lane is fitted on `P[0:4352]` only.
- Before the fresh cohort is evaluated the driver asserts that no fresh case coincides with any case of `P`, test64,
  or the earlier development draws.

## 3. Metric

Per case: max over t = 0.05, …, 0.25 of ‖u − u_ref‖₂ / ‖u_0‖₂ on the 257² grid (the paper's metric). Reported per
cohort: median, 95th percentile, worst case, in percent; query time in ms.

## 4. Step 1 — diagnosis (validation only, deployed checkpoint unchanged)

For widths 192 / 256 / 384 / 512, per validation case and output time:
- (a) projection error: best approximation of the reference in the span (least squares on the grid);
- (b) reduced-solve error: the deployed solve, and the same solve with one ingredient changed at a time — dense
  advection for all steps (no quadrature), finer lattice (127²), more / fewer test modes (M = 2R', 8R'), tighter
  stopping tolerance (1e-4, 1e-6);
- location: correlation of the per-case error with ν, width, amplitude, centre and with the output time.
The diagnosis decides which levers of step 2 are pursued. It is reported whatever it shows.

## 5. Step 2 — levers (validation only)

Projection-limited: retrain / fine-tune the bank on `P[0:4352]` with a worst-case-aware objective (power mean or
log-sum-exp of per-snapshot relative projection errors), all 26 levels, longer schedule, width / depth / Fourier
frequencies; column ordering from the training fields with worst-case weights. Solve-limited: quadrature, number
of test modes, stopping tolerance, residual weighting, regularisation of the wide span.

## 6. Selection rule (fixed now)

- Score of a configuration = worst case over the 256 validation cases of the full reduced solve (not the projection).
- Admissible: validation median ≤ 0.10 % (below the operators' and CoLoRA's test medians, 0.19 % and 0.104 %) and
  no failed solve on validation.
- **Primary** = admissible configuration with the lowest validation worst case among those whose validation query
  time is ≤ 1.25 × the deployed width-384 setting measured in the same job. Ties within 5 % relative go to the
  faster one.
- **Alternative (speed)**, optional = the fastest admissible configuration whose validation worst case is below
  0.43 %; dropped if it coincides with the primary or none exists.
- If no configuration improves the validation worst case over the deployed setting, the deployed setting stays and
  the negative result is reported.

## 7. Final evaluation (once)

After the choice is frozen in `selection.json` (committed, with file hashes), one job per cohort on an A100 80GB
runs, in the same allocation: the deployed baseline (width 384, unchanged), the primary, and the alternative if any;
on test64 and on fresh64. Timing by the iso-error lane's driver conventions (dense initial field on the GPU to six
dense output fields on the GPU, A-B-A round robin). On fresh64 the operators (selected FNO, U-Net) and CoLoRA are
evaluated unchanged from their lanes' checkpoints and drivers if those run without modification; otherwise this is
reported as not done. No configuration is changed after a test or fresh number has been seen. Reported: median,
95th percentile, worst, ms; extra offline cost in GPU-hours; any extra data.

## 8. Limits

At most 4 GPU jobs of this lane queued or running; `gpu` partition; A100 80GB; no local GPU work.

## Amendments
