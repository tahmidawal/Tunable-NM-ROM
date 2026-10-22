**Overall: NEEDS-FIX.** Audited `d149f94e0..56fea6ea7`. No files changed, no jobs submitted, no lab-log writes.

1. **A8.1 — CORRECT, with a qualification.**  
   `M_rel = M_dev / sqrt(R)` correctly computes the stated relative Frobenius norm; both checks use `1e-8` over all retained modes. This is compatible with A2.3, although A2.3’s successive-rule Gram check does **not imply** either exact-operator bound, particularly stiffness. The amendment explicitly discloses its post-smoke origin. Collected artifacts contain no final selection result, consistent with the stated chronology. Call this a disclosed tolerance relaxation; “benign round-off” remains an explanation, not something established by the norm alone.

2. **Unbatched cohort and screen timing — CORRECT.**  
   The new loop matches the existing official unbatched evaluator, preserves case ordering, statistics and pooled validity predicates, and reuses the query for screening. `time_one` blocks on a warm call before starting its timers.  
   A tiny CPU AD probe found identical statistics and a maximum batched/unbatched output difference of `2.78e-17`. Thus equivalent semantics, not guaranteed bitwise equality across batching strategies.

3. **Persistent cache — NEEDS-FIX for the executable-sharing claim; no stale-executable defect found.**  
   In local JAX 0.10.1, lowering the actual AD query embeds each tolerance as a distinct HLO literal. Merely moving `jnp.asarray(float(tol))` outside `q` does **not** make it an executable parameter. Even enabling simplified constants leaves the scalar embedded under the tested defaults. See [spec.py:194](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/spec.py:194).

   Captured representation arrays—including ROM operators/eigenvectors, POD tables and shifted-decoder coefficients—may likewise be embedded depending on JAX configuration. **Embedded data do not cause configuration collisions:** JAX hashes the serialized computation, including literals. Hoisted arrays instead receive their current values as arguments. I found no custom cache key bypassing this distinction.

   Cache loads occur before screen and paired timers. The default `code/.jaxcache` is excluded by collection’s `code` pruning and archive scope. However, unrestricted `IMPBOX_JAX_CACHE` can place it under `out/cell`, defeating that exclusion.

   **Required:** use explicit executable arguments for tolerance if cross-tolerance sharing is required; verify sharing on cluster JAX 0.10.2. Constrain the cache directory if exclusion is meant to be guaranteed.

4. **Tolerance arithmetic — CORRECT.**  
   The production comparison and coarsening threshold retain their f64 arithmetic. Tiny CPU comparisons against the old implementation at two tolerances produced exactly matching outputs and statistics. This passes independently of the failed sharing claim.

5. **Relative convergence stop — WRONG.**  
   [ladder.py:162](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/ladder.py:162) does not establish domination.

   A simple second-order convergence example, with identical case errors:

   - Previous error: `0.03041`
   - Current error: `0.03001`; trajectory change `0.00040`
   - Next error: `0.02991`; trajectory change `0.00010`

   The absolute stop does not fire at the current rung, but the new relative stop does. With target `0.03` and allowance `U=0.00001`, the current rung does not meet the target; the next does. This example requires neither instability nor unusual convergence.

   Runtime monotonicity is also unproved for iterative ROMs: finer steps can reduce Newton/LM iterations enough to offset their increased count. **Remove this stop or explicitly register it as heuristic search truncation with incomplete-search labels. It cannot support “only dominated rows removed.”**

6. **Report generator — NEEDS-FIX.**  
   Conditional-reference and provisional labels exist, as does a glossary. However:

   - [make_report.py:181](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/make_report.py:181) exports `finalists[0]` beside `S`, although the max–min timing result identifies its actual winner in `bank_key`. These can differ.
   - Missing FOM finalists are universally labelled “FOM censored,” conflating construction failures, incomplete search and qualification failures with genuine censoring.
   - Missing device-memory measurements become **zero GiB**, contradicting `record_peak`’s explicit unknown-versus-zero distinction.
   - The glossary hand-types empirical front width `~0.002` and fixes the validation population at 32 despite eligible-subset scoring. Not every reported number comes from JSON.
   - Discovery still falls back to old `smk-*`, omits `smk2-*`, and excludes cancelled-smoke stage costs because it requires `final.json`. That does not implement A9’s accounting promise.
   - The glossary leaves printed concepts/statuses such as κ, `n`, boundary labels and partial selection undefined.

7. **Other integration — NEEDS-FIX.**  
   [resources.py:67](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/resources.py:67) omits `converged-relative` from its accepted termination reasons. Any smoke exercising the new stop is therefore rejected as a cost certificate. Resolve this consistently with item 5.

Validation was limited to source inspection, installed-JAX lowering inspection and tiny CPU checks; no GPU timing or cluster cache-hit claim is validated.