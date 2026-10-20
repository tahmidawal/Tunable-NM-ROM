**NOT READY for the complete `smk2` workflow.** The failed-transfer deletion bug is fixed, but collection still treats a failed queue query as evidence that the job has finished.

This was a targeted review of `8547ba38e` against Round 9. No files changed, jobs submitted, or GPU work run. The selected CPU G-nnls test passed; all **17 inherited hashes and 16 source-commit comparisons matched**. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **MAJOR — WRONG: terminal-job verification fails open.**

   **Evidence:** `cluster/collect.sh:13` puts the SSH/`squeue` pipeline inside an `if`. A timeout, SSH failure, or scheduler error makes the condition false and execution continues. `set -euo pipefail` does not abort this conditional. At `:20–22`, `sacct` output is printed without checking its state, then `--delete` permits deletion.

   An ordinary transient queue-query failure can therefore allow collection and deletion while a job remains active. Checksums can match during a quiet interval before subsequent writes. A read-only shell reproduction confirmed that exit 255 from the query reaches the collection path. **This is not forgery-only.**

   **Fix:** capture the queue result in a separately checked assignment; abort on any query failure. Require an explicit terminal accounting state for the exact numeric job ID before collecting, and recheck before deletion.

2. **MAJOR — CORRECT: unsuccessful transfer no longer authorizes deletion.**

   **Evidence:** `cluster/collect.sh:14–18` retrieves a fresh manifest covering logs and outputs, requires successful `rsync`, and verifies local hashes before proceeding to `:22`. The former `|| true` is gone.

   **Fix:** retain these changes. Also execute remote manifest generation under explicit `bash -o pipefail`: the local setting does not propagate into the remote `find | sort | xargs` pipeline at `:14`.

3. **MINOR — CORRECT: randomized-POD sketch lifetime is reduced.**

   **Evidence:** `build.py:122–124` releases `Y` and `Om` before power iterations; `:128–132` releases `Qy` and `Bm` after forming the basis. The saturated-prefix explanation is corrected at `:88–90`.

   **Fix:** none for this patch. Maximal-shape feasibility still requires measured GPU peaks; deleting Python references does not establish the full asynchronous allocation peak.

4. **MINOR — NEEDS-RESTATEMENT: new NNLS tests cover checkpoints and dependent columns, but do not demonstrate banning.**

   **Evidence:** `tests/test_core.py:324–336` checks support checkpoints, positive weights, directly recomputed residuals, and agreement with SciPy on duplicated columns. These checks passed. However, the duplicated-column run returned **`dup_banned=0`**; `:337–338` only records that count.

   **Fix:** describe this as dependent-column coverage. Add a deterministic repeated-drop fixture that distinguishes the third-drop rule in `nnls.py:172–174` from immediate banning and verifies the resulting all-candidate KKT behavior.

5. **MINOR — CORRECT: A15.6 resolves the identified documentary discrepancies.**

   **Evidence:** `DESIGN.md:1155–1162` corrects the smk1 orthonormality, stationarity, failing-expression and incomplete-G-fd statements; registers first-crossing support sizes and continuation without rebuilding earlier checkpoints; and permits certified saturated prefixes.

   **Fix:** none for those restatements. The terminal-job guarantee at `:1163–1164` remains inaccurate until finding 1 is fixed. Phase II must implement the registered path continuation explicitly.

6. **MINOR — CORRECT, noted: remaining forgery-only hardening is nonblocking.**

   **Evidence:** `smoke_accept.py:234–243` still uses saved support summaries and partly trusts the KKT `ok` flag. These remain the previously noted evidence-hardening limitations under `DESIGN.md:1050–1051`.

   **Fix:** optionally persist checkpoint weights/supports and independently compare both KKT bounds. No new blocker follows from these concerns.

Phase II still needs production orchestration, live-memory/resource accounting, preserved NNLS continuation, qualification and eligibility enforcement, frozen selection, and verified evidence collection. I did not certify the regenerating `checks/test_core.json`; its committed contents must match the staged sources.

**Overall: NOT READY. Top blocker: collection can delete an active job’s evidence after a failed queue query. No additional numerical blocker was identified in the targeted A14/A15 changes.**