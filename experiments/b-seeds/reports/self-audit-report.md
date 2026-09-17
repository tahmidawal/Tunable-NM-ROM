# Written self-audit of the b-seeds report (Codex substitute)

The protocol asks for an audit of the finished report by a different model family. The account's
Codex quota is exhausted until 2026-09-19 11:33 (coordinator notice, `LANE-PROTOCOL.md`), so this
written self-audit stands in its place, as `DESIGN.md` A1 and A3 record. It is **not** an
independent model: when the quota returns, the Codex audit of
`reports/2026-09-17-b-seeds.md` runs and is appended there as a dated addendum, retracting
anything it overturns.

Everything below is a claim the report makes, the JSON field it rests on, and the check actually
run. The machine-readable result is `checks/selfaudit-recheck.json`, written by
`reports/selfaudit_recheck.py`.

## The check that carries the most weight

`reports/selfaudit_recheck.py` re-derives every development-cohort number **from the retained
solution fields**, without reading a single audit JSON and without importing the report
generator. For each arm it loads the `.npz` field of every case, subtracts the same-job converged
`fft_tight` field of that case, normalises by the reference solution's initial-state norm, and
takes the worst over cases at $t>0$, over all times, and at $t=0$; it takes the median GPU
millisecond over all 18 invocations of the arm; it reads the bank floor and best-found layers out
of the reconstruction block; and it then compares all of that to `reports/summary.json` **and to
the mean ± std actually printed in the report's T12 markdown**.

**Result: 360 of 360 comparisons agree, the error and cost rows to $10^{-12}$ relative and the
printed T12 means to $5\times10^{-5}$ absolute; zero failures.** That covers 11 arms × 3 metrics ×
(3 seeds + 3 incumbent blocks) plus costs, both layers, and the 18 printed mean ± std cells.

## Claim by claim

| claim in the report | where the number comes from | check run | outcome |
|---|---|---|---|
| T12 per-rung worst evolved / all-times / $t{=}0$ errors, every checkpoint | `ladders.*.rungs[].evolved / all_times / t0` in each audit JSON, themselves recomputed by `audit_seeds.py` from the saved fields | recomputed a third time from the `.npz` fields by `selfaudit_recheck.py`, compared to `summary.json` | agree to $10^{-12}$; 0 failures |
| T12 median GPU ms | `ladders.*.rungs[].gpu_ms` | recomputed as the median over all 18 invocations of the arm from `gpu_seconds` | agree to $10^{-12}$ |
| T12 mean ± std over the three seeds | computed by `generate_b_seeds.py` from the per-seed rows | the printed markdown cells parsed back out of the report and compared to means of the field-derived values | agree to $5\times10^{-5}$ (the printed precision) |
| three layers 0.36–0.38 / 2.54–2.86 / 2.54–2.86 % per seed | `reconstruction[q=0].cases[].bank_projection_max / best_found_max` | worst over the six cases recomputed from the raw result | agree to $10^{-12}$ |
| "monotone on 3 of 3 seeds, both metrics, both ladders" | `verdict.monotone_evolved_counts`, `monotone_all_times_counts` | the field-derived per-rung errors are non-increasing in $q$ for all three seeds on both ladders; checked directly against the recomputed table | holds |
| "converged on 2 of 3 seeds" and C1 = 2 of 3 | `ladders.dense_m4.rungs[].converged`, `budget_exits` | every ROM invocation's `converged` flag re-aggregated per arm; the six failing invocations enumerated (seed 3, `old_q256_M1088_dense`, cases 2 and 3, all three reps, `budget_exits` = 1, joint stationarity 1.52e-06 / 5.83e-06 vs `gtol` 1e-06) | holds; the exception is named in the report, `DESIGN.md` A4 and `checks/retractions.md` |
| the incumbent reproduces qtd02 within $10^{-3}$, probe $10^{-9}$ | `checks.incumbent_reproduces_qtd02`, `..._at_1e-9` against `comparators/qtd02-audit.json` | the gate passed in all three jobs (33 of 33 arms); the worst deviation anywhere is 1.34e-08, five orders inside the gate; 20 of 33 arms also meet $10^{-9}$, and the report prints the value per arm | holds, and the weakening is stated in the report body per A3 |
| TR: 3 of 3 seeds reproduce the recipe | `three_layer.best_found_percent`, `bank_floor_percent` against the bars read from `comparators/qtd02-audit.json` | best-found 2.86 / 2.54 / 2.57 % against the 3.82 % bar, floors 0.36 / 0.38 / 0.35 % against 0.78 % | holds |
| training completed to the recipe | `training.*` gates in the `ladder_seed` audits | 300000 bank steps and 200000 head steps on every seed, `time_capped` false, the 131072-state pick with 27648 early states, stage A/B data fingerprints matching the incumbent's own job JSONs to $10^{-9}$ | holds (43 of 43 gates) |
| the sealed cohort was still sealed when the seed numbers were read | `PROVENANCE.json` and `MANIFEST.sha256` of each seed attempt; `final_cohort_unopened` in each `result.json` | zero occurrences of `sealed` in all six files; the flag is `true` in all six seed blocks | holds |
| EQ certification per seed | `rules[].certification.rho_max`, `certified_primary` | the q-ridge EQ audit passed 28 of 28 gates on each seed; the report prints $\rho_{\max}$ and the certification outcome per rule rather than the NNLS fit residual, which certifies nothing (`LANE-PROTOCOL.md` §14) | holds |

## What this audit does **not** establish

- It is one model family checking its own work. The failure modes it is least able to see are the
  ones built into the design: the choice of the qtd02 configuration as the frozen comparator, the
  reading of C1 as "the method, not the checkpoint", and the TR bar's 1.5× factor, which is a
  heuristic precondition rather than a measurement.
- It does not revisit the physics: the FOM residual, the reference-grid convergence and the
  identity gates are checked inside the job and by `audit_seeds.py`, not here.
- The sealed-cohort claims (C2, C2n, C3) are not audited because they do not exist yet; job
  3804465 is pending.
- Two defects found in this lane's own tooling *after* the numbers landed — EQ rows keyed by file
  name instead of the seed label, and a lab-log helper that crashed on those null labels — were
  generator-side and invisible to every in-job gate. Both are fixed and the report regenerated;
  they are the reason this recheck parses the published markdown rather than trusting the
  generator.
