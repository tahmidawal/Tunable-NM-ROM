# Self-audit of the 2026-09-17 lshape solve report

Codex is unavailable until 2026-09-19 11:33 (coordinator-verified), which is after this lane's
reporting window, so this written self-audit substitutes for the independent-auditor pass that
DESIGN.md §9 requires. The substitution is recorded as DESIGN.md §A5 and §A8.

Every claim below names the JSON field it rests on and the check that was run against it. The
checks are executed by `../checks/verify_report_2026-09-17.py`, which does **not** import the
report generator, and their verdicts are stored in `../checks/verify_report_2026-09-17.json`.
The four per-job `audit.json` files come from `../lsh_audit_np.py`, which imports neither JAX nor
any driver module and recomputes the operator, the reference solutions and every reported error
from the saved fields.

## Claims and their evidence

| # | claim in the report | field it rests on | check | verdict |
|---|---|---|---|---|
| 1 | every timed number belongs to one identified job and one test-mode block | `summary.json` rows `job_id`, `test_modes`, `source_sha` | `every_timed_row_is_keyed_by_job_and_block`, `no_duplicate_timed_rows` | pass |
| 2 | the non-dominated sets are what the report says | `result.json:invocations[].same_grid_error`, `.total_seconds` | `nondominated_sets_match_independent_audit` — the sets recomputed by `lsh_audit_np.py` equal the report's, for all four (job, mesh) pairs | pass |
| 3 | every worst error and median complete-query time in the report is the raw data | `result.json:invocations` | `report_values_recomputed_from_raw_invocations`, worst relative difference $0$ | pass |
| 4 | every reported error is reproducible from the stored output fields | the per-invocation `.npz` artifacts | audit check `same_grid_errors_recomputed_from_saved_fields` (and `physical_...`), worst difference $0.0$ in all three solve jobs | pass |
| 5 | the free rung is a linear model that runs no nonlinear solve | `invocations[].iterations`, `arm_setup[].free_rung` | `free_rung_runs_no_lm`, max iterations $0$ | pass |
| 6 | the free rung sits on its own bank floor | `reconstruction[].bank_projection.worst` | `free_rung_reaches_its_bank_floor`, ratio $1.003$ | pass |
| 7 | the free rung is head-independent | the two heads' output fields | `free_rung_is_head_independent_to_roundoff`, worst relative field difference $4.06\times10^{-14}$; **not** bitwise, correcting §A6 (now §A9) | pass, with the correction |
| 8 | timing repetitions measure the same computation | `invocations[].field_sha256` | `timed_repetitions_are_bit_identical` | pass |
| 9 | every operator gate passed in every job | `gates[].passed` | `all_gates_passed`, 12 gate records | pass |
| 10 | every job ran on a GPU in float64 at `highest` and finished | `backend`, `x64`, `matmul_precision`, `complete` | `all_jobs_gpu_x64_highest_complete` | pass |
| 11 | the validation-versus-development gap quoted beside the headline is the training job's own | `lsh02:head_arms[].best_found_validation/development` | `validation_gap_matches_training_json` | pass |
| 12 | the independent NumPy audits pass | `artifacts/*/audit.json` | `independent_numpy_audits_pass`, 13/13 checks on each solve job and 19/19 on the training job | pass |

## What this audit does not cover

- **One training seed, one development cohort of 32.** Nothing here estimates seed variance, and
  the validation gap in the report (worst 10.7–16.7 % on 461 sources against 3.2–6.8 % on 32)
  says directly that the worst-over-32 numbers are the optimistic end of the range. A 461-source
  *solve* sweep has not been run; only the untimed reconstruction bound has.
- **Cross-block costs.** The report refuses these by construction, and the generator now aborts
  rather than merge two jobs in one block, but nothing checks that a future reader obeys the
  refusal.
- **$N=512$.** `lsh07` (job 3789568) was running when this report was generated. Its numbers are
  absent, and the mesh trend in the report is therefore three meshes wide, not four.
- **The $M=257$ free rung.** Not constructible ($M < R$), so the $M=1024$ rung has no same-block
  $M=257$ twin and the two cannot be compared on cost.

## Claims that were weakened or retracted during this audit

1. **"Byte-identical output from the two heads on the free rung"** (DESIGN.md §A6(b), from an
   $N=32$, $R=32$ smoke) is **false on the real banks** and is corrected in §A9. The agreement is
   $4.06\times10^{-14}$ relative, which supports every reported digit but is not byte-identity.
2. **The $N=128$ "non-dominated reduced model" is a cheapness-only membership** and the report now
   says so in the generated prose: all three reduced front members there are POD rungs at
   7.7–34.6 % worst error, 1.08x cheaper than a direct solve that is exact. Reporting them as
   "a reduced model is non-dominated" without that qualifier would have been misleading.
