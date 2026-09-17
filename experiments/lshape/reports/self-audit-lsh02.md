# Self-audit of the `lsh02` training layer (Codex substitute)

`DESIGN.md` §9 requires an independent Codex audit of the report; the Codex quota is exhausted
until 2026-09-19 11:33, after this lane's window, so §A5 substitutes this written audit. Each
row names a claim in `reports/2026-09-17-lshape.md`, the `result.json` field it rests on, and
the check actually run against it. Status of the numbers: the bank and head layers are final
for job `3783786`; the solve layer does not exist yet.

The standing rule from the campaign memory — *controls must fail on real data* — is applied
throughout: a check that cannot fail is recorded as such rather than counted as evidence.

| # | claim in the report | field it rests on | check run | outcome |
|---|---|---|---|---|
| 1 | every number comes from the job, none typed | — | the report and `summary.json` are emitted by `reports/generate_lshape.py` from `artifacts/lsh02/result.json` alone; the file is regenerated, not edited | pass |
| 2 | the job really ran on a GPU in f64 at `highest` | `backend`, `gpu`, `x64`, `matmul_precision` | audit asserts all four; the log contains `jax_backend=gpu` and `ALL-DONE`; `sacct` shows `COMPLETED 0:0`, 51m45s, `pax050` | pass |
| 3 | the operator is the L-shape 5-point Laplacian | `gates` | the audit **re-assembles it by the Kronecker route the driver does not use** and compares entrywise (`nnz` difference 0), checks exact symmetry, and recomputes $\lambda_1$ by its own shift-invert Lanczos (3.20e-4 / 1.32e-4 relative to the Trefethen–Betcke value at $N=256/512$) | pass |
| 4 | the three cohorts are disjoint and centre-rejected | `cohorts` | the audit redraws all three from the seeds with its own NumPy generator and compares 4-vectors; rejection counts 1161 / 126 / 14 reproduce | pass |
| 5 | the bank floors are what the driver says | `bank_arms[].floors[]` | the audit rebuilds each bank from the saved weights **in NumPy from the equations**, re-forms the QR projector, re-solves every reference field by SuperLU, and recomputes every per-case floor; worst absolute difference 4.723e-8 against a 1e-6 bound (§A4) | pass |
| 6 | the *relative* floor difference (5.236e-6) is conditioning, not error | `bank_arms[].floors[]` | `checks/floor_diag.py` recomputes the identical floor by a second NumPy route (SVD instead of QR): the two agree to 1.14e-14, so the quantity is not cancellation-limited. `checks/floor_sens.py` perturbs the features by one $\varepsilon$ per entry: the resulting band is 8e-15 at cond 1.3e4 and 1.0e-10 at cond 8.9e9, and the observed driver-vs-audit difference is a near-constant 1.4e4–2.0e5 multiple of it at **every** arm across four decades | pass, and the conclusion is recorded in §A4 rather than hidden |
| 7 | `sdf_R512` is the selected bank | `selection.bank` | the audit re-applies the selection key (worst, then median, then smaller $R$, then table order) to the arms and reproduces the winner; the report also prints that the development ranking **disagrees**, as the honesty clause requires | pass |
| 8 | the bank target is met | `floors[N=512].floor_dev.worst` | best arm 0.6650 % against the pre-registered 1.0 % — and the bar was set in §4 before any number existed | pass |
| 9 | the falsification clause is not met | same | it needs *every* smooth arm above 1.5 % at $N=512$ **and** enrichment helping by more than 1.5×; measured 1.506 / 0.709 / 2.539 % and a 1.067× enrichment ratio, so **both** conjuncts fail, not just one | pass |
| 10 | the head misses its 1.2× target | `head_arms[].head_floor_over_bank_floor` | recomputed from `best_found_development.worst / bank_floor_dev`: 2.24–5.77× on the seven arms; the miss is reported as a miss on every arm, including both primaries | pass (as a recorded miss) |
| 11 | every development solve is stationary | `weak_solved_development.exit_reasons` | 32/32 exit reason 4 on all seven arms, and the audit re-reads the raw list rather than the summary count | pass |
| 12 | the correction bases are orthonormal | `basis.orthogonality_*` | the audit forms $W = G\,\text{directions}$ on the full grid and applies its **own** absolute 1e-6 bound, separate from the driver's per-column gate; worst 2.824e-8 | pass |
| 13 | the checkpoints reported are the checkpoints shipped | `checkpoints[].sha256` | every `.pkl` and `-basis.npz` is re-hashed by the audit, by `make_models.py` before staging, and by `stage.py` into `PROVENANCE.json`; the same hashes appear in all three | pass |

## Checks that would have caught a failure, and what they say about the ones that passed

- Check 3 **has fired before**: the same class of independent-assembly check is what the
  cell's §A1 self-audit used to find the `atan2` branch bug in the singular columns, before
  any job. It is not a check that cannot fail.
- Check 5 **did fire on this job**: it is the check that failed at the inherited 1e-8 relative
  bound and produced §A4. The investigation in check 6 is what turned the failure into a
  diagnosis rather than a silenced gate, and the tight relative number is still printed.
- Check 12 **did fire on the previous attempt**: the same quantity aborted `lsh01` (§A3). It
  is the most sensitive check in the set.
- Check 11 is the weakest: an all-32 pass is what a correct solver and a broken
  always-report-4 solver both produce. It is corroborated only indirectly, by the solved error
  agreeing with the independently computed multistart best-found to four digits on every arm —
  a solver that exited without converging would not land there. The solve jobs will test it
  properly, because they re-run the same solve under timing with iteration counts retained.

## Claims the report does NOT make, deliberately

- **No speedup, and no cost number of any kind.** The solve jobs `3784662` (N=64,128),
  `3784663` (N=256) and `3784664` (N=512) are still queued. The headline non-dominated-set
  table is emitted empty, and the report's status line says so.
- **No claim that the head trains acceptably on the corner.** It does not: every arm misses
  the 1.2× bar, worst-case, by 2.2–5.8×.
- **No claim from the validation worst-case that is not also qualified.** Validation worst
  (10.7–16.7 %) is far above development worst (3.5–6.8 %) because it maximises over 461 cases
  instead of 32. The two are not comparable and the report must not be read as if they were.
- **Nothing from `lsh01`**, which is retracted (§A3).
