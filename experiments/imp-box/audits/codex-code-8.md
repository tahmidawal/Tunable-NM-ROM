**Overall: NEEDS-FIX.** Audited `d149f94e0..10e0a3987`, including the A9 addendum. No files changed, jobs submitted, or lab-log writes.

1. **A8.1 — CORRECT, with qualification.**

   `M_dev / sqrt(R)` implements the stated relative Frobenius norm; both operator checks use `1e-8` across all retained modes. This is compatible with A2.3, although successive-rule convergence does **not imply** the exact mass/stiffness bounds.

   The change is a disclosed, post-smoke tolerance relaxation. Collected artifacts contain no final selection results, consistent with the stated chronology. “Benign round-off” remains an interpretation, not something established by the norm alone.

2. **Unbatched cohort and screen timing — CORRECT.**

   The loop matches the official unbatched evaluator, preserving ordering, scoring and pooled validity predicates. `time_one` completes a warm call before starting its timers.

   Tiny CPU AD checks at two tolerances produced identical batched/unbatched statistics; maximum output difference was `1.46e-16`. Equivalent semantics are supported; bitwise identity with the former batched execution is not guaranteed.

3. **Persistent cache — NEEDS-FIX for unconditional containment; executable correctness passes inspection.**

   I found no stale-executable collision mechanism. Captured ROM operators, tensors, eigenvectors, POD/translation tables and bank constants can become HLO literals. Installed JAX hashes the serialized computation—including literals—plus compiler/backend configuration. Constants passed as executable arguments instead receive their current values. Neither case implies reuse of another configuration’s data.

   The addendum correctly retracts cross-tolerance executable sharing. Cache loads occur during warm calls, outside screen and paired timing regions. Rebuilt-query cache hits on cluster JAX remain unverified.

   **Remaining loophole:** [cell.py:430](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cell.py:430) exempts `--smoke` from path enforcement. Such a run can place its cache inside `out/cell`, where collection and archival include it. The proposed full-size `smk2` bodies do not pass `--smoke`, so their normal path is protected. Enforce containment for every collected run. Also correct the nearby blanket claim that captured constants are executable parameters.

4. **Hoisted AD tolerance — CORRECT.**

   Production comparisons and coarsening thresholds retain their f64 arithmetic. Against the audited implementation, small CPU checks at tolerances `1e-3` and `1e-5` gave exactly matching outputs and statistics.

5. **Replacement stopping rule — WRONG as a domination guarantee.**

   Restricting the rule to explicit steppers removes the iterative-solver objection, but [ladder.py:180](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/ladder.py:180) still does not bound remaining refinement error.

   A CPU check using the actual `target_stable` predicate gives:

   | Rung | Median and worst error | Classification at τ = 0.03, U = 0.00001 |
   |---|---:|---|
   | Previous | 0.03101 | misses |
   | Current | 0.03080 | misses |
   | Next | 0.02990 | meets |

   The last change is `0.00021`, so the absolute stop does not fire. Improving the current error by twice that change leaves `0.03038`, still missing: **the new stop fires**. Nevertheless, the next rung meets the target. These values fit a smooth error expansion \(a+b h^4+c h^6\); the code checks no contraction assumption that excludes this example.

   There is another independent gap: stopping happens before Q1/Q2 qualification. A finer rung may qualify where the stopped rung fails, even without changing its error classification.

   Fixed-step operation counts grow with step count; total measured query time is not *exactly proportional* because initialization, coefficient construction and decoding also cost time.

   **Required:** remove the new stop, or explicitly treat it as heuristic search truncation and propagate incomplete/conditional-search semantics through selection, stop decisions and cost certification. A termination name alone does not establish domination.

6. **Report generator — NEEDS-FIX.**

   The previous wrong-winner export, missing-FOM status conflation, missing device-memory zero, empirical front-width literal and cancelled-smoke accounting issues are repaired. A glossary and conditional-reference labels exist.

   Remaining issues in [make_report.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/make_report.py:130):

   - Neither report nor export carries `ladder_termination`. Thus the new conditional search assumption disappears from reported speedups.
   - Failed or missing controls are mentioned in prose, but numeric speedups still appear and are exported without a controls-acceptance field or an adjacent invalid/unaccepted label.
   - FOM partial-search status and timing exclusions are not consistently labelled beside the affected speedups.
   - Strictly, “every number from JSON” remains false: widths, targets, thresholds, population text and glossary constants are embedded in the script. Source these from recorded metadata, or explicitly distinguish registered definitions from measured results.
   - The glossary still leaves printed statuses such as `arm-unqualified`, `search-incomplete` and `timing invalid` undefined.

7. **Other integration — NEEDS-FIX.**

   `resources.py` now recognizes `smk2-*` and the new termination name, but accepts heuristic-truncated searches as complete cost evidence. `stoprule.py` likewise does not account for this truncation. Resolve both with item 5.

   Additionally, [ledger.py:49](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/ledger.py:49) reads preserved GPU metadata **before** acquiring its lock. Concurrent updates for the same job can overwrite newly recorded allocation metadata with a stale record. Move that read/merge inside the locked transaction.

Validation used source inspection, installed-JAX cache-key inspection and small CPU-only checks. No GPU timing or cluster cache-hit performance was validated.