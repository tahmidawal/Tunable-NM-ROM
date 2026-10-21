**NOT READY to submit.** Audited HEAD `87614085bac7e4bc1d671580a6e8e516e17daa25`. The Galerkin construction is sound, but C11, provenance, submission safety, and budget enforcement need fixes.

Read-only throughout: no files changed, no jobs submitted, no cluster commands executed, no large arrays copied.

1. **Galerkin target and matrix layout — CORRECT.**

   [ecswjob.py:44](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/ecswjob.py:44) computes
   \[
   c_s=\Phi_r^\top u_s,\qquad N_s=N_h(\Phi_rc_s),\qquad b_s=\Phi_r^\top N_s.
   \]
   Each candidate column contains \(\Phi_r[i,:]N_{s,i}\), stacked by state and divided by the same block norm as the target. Columns are candidate nodes, as `ecsw_weights` expects. The central operator uses zero Dirichlet boundaries; no LSPG-gradient target or extra mesh factor enters.

   A small, in-memory NumPy surrogate executing the extracted construction matched an independently assembled stencil and matrix: maximum differences \(2.8\times10^{-17}\) for \(G\), \(1.1\times10^{-16}\) for \(b\). This checks arithmetic/layout, not GPU execution.

2. **Importance sampling, floors, dropped blocks — WRONG at edge cases; correct for nondegenerate production inputs.**

   [ecswjob.py:48–68](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/ecswjob.py:48) correctly uses
   \(\sum_s |N_{s,i}|\|\Phi_r[i,:]\|=\sum_s\|g_i(s)\|\), seed `20261018`, half uniform sampling, and importance sampling without replacement from the remainder. The floor and retained-block scaling match the reused implementation.

   Two reproduced failures:

   - Unlike `build.ecsw_contrib`, it lacks the `n <= n_cand` all-node branch. Small meshes fail with “larger sample than population.”
   - Zero importance mass produces NaN probabilities before the intended all-blocks-dropped construction failure. Non-finite inputs also lack an explicit rejection.

   **Fix:** restore the all-node branch; validate finite norms/importance; detect all-dropped blocks before sampling; explicitly handle zero remaining importance mass. Persist the floor, block norms, and retained-block mask for reproducible C11 evaluation.

3. **NNLS budgets and unavailable checkpoints — CORRECT semantics; WRONG evidence persistence.**

   The limits are correctly `240/360/600` seconds **per attempt**. A banning-triggered retry receives another full limit. A mocked time-stop/retry test confirmed both calls receive the configured limit and unreached support checkpoints remain `None`. The writer skips their weight arrays.

   Certified KKT optima can fill larger κ checkpoints with duplicate rules. That is permitted by **A3.1**, which supersedes a blanket interpretation that every unreached support size must be absent. Duplicate-rule handling belongs downstream.

   However, [ecswjob.py:155](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/ecswjob.py:155) saves `list(summ['attempts'])`: only `"initial"`/`"retry"` names. It discards the nonselected attempt’s elapsed time, termination reason, KKT status, checkpoint results, and orthogonality diagnostics.

   **Fix:** serialize the full attempt-summary dictionary. Preserve actual total NNLS elapsed time and per-attempt overshoot/finalization evidence. The configured 2,400 seconds is a budget sum, not a hard runtime ceiling.

4. **C11 aggregate and uniform witness — WRONG.**

   [ecswjob.py:125–148](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/ecswjob.py:125):

   - Aggregate recomputation uses the correct stacked, scaled \(G,b\).
   - The agreement test adds `1e-14` absolute slack, violating the registered \(10^{-8}\) relative criterion. For example, a reported residual of \(10^{-12}\) with a \(5\times10^{-15}\) discrepancy passes despite **0.5% relative disagreement**.
   - The witness samples uniformly from **`cand`**, an importance-biased subset, rather than uniformly from all \(n\) nodes. Weights \(n/|E|\) therefore do not implement the registered uniform-node witness.
   - Failure or absence of the witness does not prevent `complete=True` and `COMPLETE`. I found no downstream C11 enforcement in the lane code searched.

   **Fix:** sample nodes from `range(n)` and recompute their contributions using the same retained blocks/scales. Enforce the registered relative comparison, with an explicitly specified zero-residual case. Persist sampled nodes and detector outcomes. Block acceptance/production when agreement fails or the required witness is unexercised.

   The κ preference—8, otherwise largest available—and the \(|E|\), then \(|E|/4\) sizes are correct.

5. **Host/device memory — NEEDS-RESTATEMENT, no evident capacity blocker.**

   With 120 retained states, \(r=384\), and 32,768 candidates:

   | Allocation | Decimal GB |
   |---|---:|
   | \(G\), each host/device copy | 12.08 |
   | Fixed QR buffer \(Q\), capacity 8,192 | 3.02 |
   | QR buffer \(R\) | 0.54 |
   | Device POD prefix, 384 columns | 3.21 |
   | Cached nonlinear vectors, 120 states | 1.00 |

   Additional memory includes retained FOM output arrays, the decompressed 512-column basis, gather/QR temporaries, compiled executables, and allocator retention. Thus “12 GB host + device” must mean **about 12 GB each**, not 12 GB combined.

   Static sizes plausibly fit H200 and `200G` host RAM. They do not establish actual peak usage. **Fix:** record peak device/host memory in this pilot and retain the 1.25-hour reservation; do not present these estimates as measured capacity certification.

6. **Staging, imports, sbatch environment, provenance — WRONG, with several correct components.**

   [stage.py:59](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/cluster/stage.py:59) stages Git blobs at the recorded commit and rejects differing checked-out copies. The ECSW import dependency chain appears complete by static inspection. I did not run staging or its import subprocess.

   Correct: absolute cluster venv, f64/highest precision, GPU preflight exit 42, `JCPREF_STORE`, pre-execution manifest verification, bounded driver timeout, output checksums, default H200/`200G`.

   Required fixes:

   - **External basis unpinned:** `Phi.npz` is loaded from the archive without recording its SHA256, contrary to §3.5. The staged manifest does not cover it. Hash it in-job and verify a pinned expected digest.
   - **Driver failures become successful batch exits:** lines 111–117 print the timeout/driver exit code, then finish successfully. Save the status, generate checksums, and return that status; require the expected completion artifact on success.
   - **Local import check bypasses the campaign slot guard:** CPU-only JAX imports still require that guard under COMMON-BRIEF §6.
   - `--hours` is unrestricted; the ECSW config itself does not enforce the registered **1.25-hour** reservation. Validate the concrete submission parameters.

7. **Submission lock, directory exclusivity, duplicate refusal — WRONG.**

   [submit.sh:10–24](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/cluster/submit.sh:10) uses the required campaign lock and propagates exit 75. Normal successful submission includes `squeue` before and after. Shell syntax checks pass.

   Concrete failure paths:

   - Two simultaneous invocations can both pass the local receipt and remote-directory checks, then submit sequentially under the campaign lock into the **same directory**. The lock does not recheck directory ownership/submission state.
   - Exit 75 leaves the uploaded directory behind. Retrying immediately refuses because that directory exists.
   - If the post-submit `squeue` call fails, `set -e` exits **before writing `SUBMIT.json`**. The accepted job then disappears from ledger discovery.
   - `jobs_in_dir` is recorded but never enforced; the lane concurrency cap of two is not checked.
   - Receipt writing uses forbidden bare `python3`.

   **Fix:** make directory claiming, submission-state checking, and admission atomic; support retrying an identical, verified, never-submitted stage. Persist the validated job ID immediately after acceptance, before optional queue diagnostics. Handle ambiguous SSH outcomes through reconciliation. Enforce lane concurrency and use the absolute interpreter path.

8. **Ledger reservation, accounting, cap — WRONG as an admission system.**

   [ledger.py:22–49](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/cluster/ledger.py:22) correctly reserves full time limits for known nonterminal jobs and counts elapsed time for recognized terminal jobs. Its four-field `sacct -P` parsing is appropriate for ordinary single-allocation output; unknown states remain conservatively reserved.

   But:

   - `submit.sh` never invokes the ledger check.
   - Jobs lacking `SUBMIT.json` are invisible, including the post-submit failure above.
   - Check and reservation are not atomic; concurrent submissions can independently pass.
   - SSH return status is ignored, so “check” can proceed with incomplete accounting.
   - `check` without hours exits successfully; negative hours are accepted.
   - Admission can use an estimate smaller than the new job’s reserved time limit.

   **Fix:** require valid positive finite admission inputs; reconcile accepted jobs; reject incomplete accounting during admission; and check/reserve the rounded sbatch time limit atomically with submission. Preserve sufficient `sacct` identity/state data to verify each allocation.

**Overall: NOT READY.** Correct the C11 witness, input provenance, failure propagation, submission races/receipts, and atomic budget admission, then re-audit before submitting the ECSW pilot.