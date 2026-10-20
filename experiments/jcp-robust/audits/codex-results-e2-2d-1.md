**Verdict: REJECT E2 2D as currently analyzed.** The selected-rule accuracy findings reproduce, but the front calculation violates the registered dominance rule and changes reported seed-dependence. This is an analysis defect; the evidence does not require new GPU runs.

1. **WRONG — front membership uses an unregistered rule.**  
   DESIGN §3.2 requires qualify’s dominance rule. [qstats.py:15](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/qstats.py:15) includes a 5% cost margin and reference-aware error margins, with separate median-error and worst-error fronts. [e2analyze.py:156](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2analyze.py:156) instead applies exact inequalities jointly to cost, median error and worst error.

   **Actual counterexample, seed 2:** R256 BE factor 1 is labelled `member`. R256 CN-R factor 1 has cost **0.357655 versus 0.357188**, median error **0.002086 versus 0.009811**, and worst error **0.026171 versus 0.033418**. Its approximately 0.13% greater cost is within the registered margin: BE is dominated on both registered fronts.

   Recomputing the registered fronts from the stored points removes the reported **R256 front seed-dependence**. Some **R384 worst-error memberships remain seed-dependent**. Front labels, historical comparisons and aggregate verdict counts must therefore be regenerated.

2. **CORRECT — the three retrains completed the same registered recipe.**  
   `e2t2d/archive` records identical recipe/configuration values apart from seeds and checkpoint paths: **300,000 Stage-A steps and 200,000 mid-head steps** for every seed. Mid-head times were approximately **332, 365 and 338 seconds**, below the 1,500-second cap; no recorded recipe deviation occurred. Registered checkpoint and rotation files match `BANK.json`.

   All **721 locally listed output files** across training and evaluation archives passed checksum verification. Each absent `sep_coeff` file has the matching digest in `LARGE.sha256`. Remote bytes were not inspected.

3. **NEEDS-FIX — qualify any claim of byte-identical training data.**  
   Stage-A state-pick hashes and aggregate data fingerprints agree, but seed 2’s `data_sha256` differs from seeds 1 and 3. The evidence establishes identical data-generation settings, not byte-identical regenerated arrays. Numerical nondeterminism is plausible, but the retained summaries do not quantify the difference. Do not describe this as independently demonstrated bitwise-fixed data.

4. **CORRECT — evaluation execution and bank binding check out.**  
   All **20 evaluation jobs** have complete outputs, `ALL-DONE`, and logs asserting `jax_backend=gpu x64=True precision=highest`. Bank/checkpoint/rotation identities agree with their registrations. Staged source hashes match their recorded commits; comparisons of banks 2–3 against bank 1 support the diff-check claims. All **2,196 timing triplets** pass the recorded identity/drift checks; independently reconstructed ratios and front costs agree.

5. **CORRECT — the committed analyses reproduce.**  
   CPU reruns of all four selections, all four bank analyses and the roll-up completed in scratch:
   - Selection differences: source-path spelling and numerical roundoff no larger than **8.7×10⁻¹⁹**.
   - Bank analyses 1–3: **byte-identical**.
   - Bank 0: identical existing values; current code additionally emits empty `target_unverified_ranks`.
   - Cross-seed roll-up: **byte-identical**.

   [Replay diffs](/tmp/jcp-robust-audit-scratch/diff.txt). Reproduction confirms the implementation’s output, including its erroneous front rule.

6. **CORRECT — selection outcomes and gate handling.**  
   Independent checking of **168 candidate/scheme cells** agrees with the stored qualification predicates and first-qualified selection order.

   | Rank | Incumbent | Seed 1 | Seed 2 | Seed 3 |
   |---|---|---|---|---|
   | R128 | fib4181 | gauss96 | fib6765 | fib6765 |
   | R256 | fib10946 | fib10946 | gauss96 | gauss96 |
   | R384 | gauss128 | gauss160 | gauss160 | gauss128 |

   All three selected-rule labels are seed-dependent. **G-E2.1 fails at incumbent R256**, which selects `fib10946` rather than deployed `gauss128`; this is explicitly non-blocking. **G-E2.3 passes all eight registered replay cells**, with maximum discrepancy **3.21×10⁻¹⁵**. Every selection target is certified, so no actual A2.5 diagnostic-pick qualifier applies.

7. **CORRECT — selected-rule accuracy survives the registered seed comparison.**  
   All **six H-acc-N cells pass**; all **12 H-acc-r transitions improve**, with all six range summaries monotone. These labels are seed-robust across seeds 1–3 and agree with the incumbent. Independent SciPy calculations also reproduce all **96 paired TOST calculations**.

   Scope matters: these are registered validation populations and configurations, including the **16-case 4096² subset**, not universal accuracy claims.

8. **CORRECT — deployed-rule qualification does not survive retraining uniformly.**  
   At 1024², deployed R128 `fib4181` fails successor-rollout qualification under both schemes on **all three retrained banks**. Deployed R384 `gauss128` fails for seed 1 CN-R and seed 2 BE/CN-R. These are qualification failures, not evidence that the selected-rule accuracy verdicts fail.

9. **CORRECT — A2.7 attribution, spread and intervals; avoid overinterpreting them.**  
   The classifier uses seeds **1–3 only**, with historical incumbent agreement separate. Independent calculations reproduce all spread extrema, log-scale Student-t intervals and spread ratios.

   Selected-rule BE **256→384** spread ratios are **2.531, 4.561 and 2.585** at 256², 1024² and 4096² respectively. All exceed one, while every seed still satisfies “improves.” These statements are compatible. The back-transformed t intervals concern the mean log-error statistic with **n=3, df=2**; they are not prediction intervals for future seeds or intervals on the spread ratio.

10. **NEEDS-FIX — the report is stale and the headline must distinguish the accepted findings.**  
    [report.md:99](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/report.md:99) still says only banks 0–1 have analyses and cross-seed labels are pending. Regenerate it **after correcting the fronts**. Retain the selected-rule accuracy, selection-dependence and deployed-qualification findings; withdraw the current R256 front-dependence claim. Carry the fixed-data, three-seed and conditional-reference qualifications beside the conclusions.

No repository or lab-log changes were made, no cluster commands were run, and scratch usage was approximately **488 KB**. E2 3D was excluded.