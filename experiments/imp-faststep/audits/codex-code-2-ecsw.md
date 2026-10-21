**NOT READY to submit.** Audited commit `526114b6eb10c106bfef301b8aea5c53037d43cd` against the previous report and governing amendments. Most numerical fixes close their findings; submission and accounting still have failure paths that bypass safeguards.

Read-only throughout: no files changed, no jobs submitted, no cluster commands executed, no arrays copied to disk. Tests used small in-memory fixtures and mocked external operations.

1. **Galerkin target and contribution layout — CORRECT.**

   `ecswjob.py:44–90` constructs exactly
   $$
   b_s=\Phi_r^\top N_h(\Phi_r\Phi_r^\top u_s),
   \qquad g_i(s)=\Phi_r[i,:]^\top N_{h,i}.
   $$
   Candidate nodes are columns; retained state blocks and targets receive identical scaling. The operator is central advection with zero Dirichlet boundaries.

   Re-executing the extracted construction with NumPy and an independently assembled stencil gave maximum differences of **\(5.6\times10^{-17}\)** for both \(G\) and \(b\). This validates arithmetic/layout, not JAX execution at production size.

2. **Importance sampling, floors and dropped blocks — CORRECT. Previous findings closed.**

   The implementation retains the specified importance mass, half-uniform/half-importance sampling and seed. It now handles the all-node case, rejects non-finite contributions, rejects all-dropped targets before sampling, and explicitly rejects zero remaining importance mass.

   Floor, block norms and retained-block mask are persisted. Small fixtures confirmed the all-node branch, unique sampled nodes, dropped zero blocks, and explicit rejection of zero/non-finite inputs.

3. **NNLS budgets and unavailable checkpoints — CORRECT. Previous persistence finding closed.**

   The limits remain **240/360/600 seconds per attempt**. A mocked banning-triggered retry received the same full budget twice; unreached checkpoints remained `None`. Their weight arrays are skipped.

   Both attempt summaries now retain termination reason, elapsed time, KKT diagnostics, orthogonality evidence and checkpoint summaries. Total NNLS elapsed time is also recorded. KKT-optimum duplicates remain permitted by A3.1. The configured budget sum remains distinct from a hard runtime ceiling, as required by A3.5.

4. **C11 aggregate and uniform witness — WRONG at the zero-residual edge; principal previous fixes closed.**

   Corrected: strict relative aggregate agreement, sampling from **all mesh nodes**, weights \(n/|E|\), retained-block reconstruction, κ preference, registered escalation, and failure exit without a `COMPLETE` marker.

   **Remaining defect:** [ecswjob.py:192](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/ecswjob.py:192) assigns `ratio = inf` whenever the reference residual is not positive. I reproduced `fires=True` for a zero reference residual with either a **zero mutant residual** or a **NaN mutant residual**. Neither demonstrates the required detector.

   **Fix:** validate finite, nonnegative residuals before comparison. Specify the zero-reference case explicitly: zero/zero is unexercised, non-finite values fail, and only a finite positive mutant residual demonstrates deterioration from an exact fit.

   Also, line 208 still writes `complete=True` before C11 acceptance. The exit status and marker now block batch success, but the JSON is ambiguous. Set acceptance completion only after C11 passes, or distinguish `construction_complete` from `accepted`. Persisting witness nodes themselves would close the previous evidence request more directly than their hash alone.

5. **Memory — NEEDS-RESTATEMENT; no evident capacity blocker.**

   At 120 retained states, \(r=384\), and 32,768 candidates, \(G\) occupies **12.08 decimal GB**.

   The patch changed assembly: it now constructs blocks and concatenates them on the **device**, rather than assembling a full host matrix first. Thus the previous “12 GB host plus 12 GB device” description no longer describes this implementation. Device assembly can temporarily retain both blocks and concatenated buffers; NNLS QR storage, basis storage, cached nonlinear vectors and solver temporaries remain additional.

   Static sizes still plausibly fit H200 and `200G` host RAM. Device peak and host maximum RSS are now recorded, closing the instrumentation request. Treat actual capacity and runtime as pilot measurements; reject unavailable peak telemetry rather than presenting `-1` as a measurement.

6. **Staging, imports, batch environment and provenance — CORRECT for the audited ECSW configuration.**

   I verified all **35 selected staging inputs** exist in Git and match their HEAD blobs. The ECSW dependency chain is covered by the staged files. The local import check now runs through the shared slot guard; I did not execute staging or that JAX subprocess.

   Previous fixes are present:

   - The external basis is hashed and checked against the configured digest before loading.
   - The reservation comes from the committed config: **1.25 hours**.
   - Absolute cluster venv, f64/highest precision, GPU preflight exit 42, `JCPREF_STORE`, manifest verification and bounded timeout are retained.
   - Driver status survives checksum generation; success requires `COMPLETE`.
   - Staged source comes from Git blobs, with working-copy equality checks.

   Minor manifest coverage issue: `stage.py:131` excludes **every** file named `MANIFEST.sha256`, including the nested reference-store manifest. Exclude only the root manifest being generated. This does not affect ECSW’s numerical input path.

7. **Submission locking, directory ownership and duplicate refusal — WRONG; partially fixed.**

   The local lane lock and remote directory check under the campaign lock close the ordinary simultaneous-submit race. Campaign/lane caps are checked, exit 75 occurs before claiming the final directory, and the receipt precedes post-submit queue diagnostics. The absolute interpreter replaces bare `python3`.

   **Remaining defects:**

   - **Queue failures permit submission.** [submit.sh:25](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/cluster/submit.sh:25) runs the remote shell without pipeline failure enforcement. A failed `squeue` becomes zero through `wc -l` or `grep -c`. A mocked execution with both queue queries failing still reached `sbatch`.
   - **Ambiguous acceptance is not durably reconciled.** Lines 30–37 consult only `squeue`. An accepted job that finishes before reconciliation disappears from that query; SSH failure during reconciliation is also suppressed. No receipt or pending reservation is guaranteed.
   - **Definite submission failure strands the directory.** The directory is moved before `sbatch`; rejection leaves it present, so the next invocation refuses before reconciliation.

   **Fix:** explicitly require successful queue queries inside the lock before counting. Persist submission intent and its reservation before invoking `sbatch`; record a validated numeric ID remotely immediately after acceptance. Reconcile through that receipt and accounting, retaining unresolved intent conservatively. Permit recovery of a verified never-submitted directory without allowing a second allocation into an already-used directory.

8. **Ledger accounting, reservation and cap — WRONG under ambiguous submission; ordinary accounting improved.**

   Corrected: submission invokes admission under the local lane lock; admission uses the staged rounded time limit; SSH errors raise; missing/negative/non-finite admission amounts cannot succeed; receipt-known nonterminal jobs reserve their full limits. Ordinary seven-field accounting rows are parsed appropriately, and receiptless `ifs_` jobs are discovered once accounting exposes them.

   **Remaining gap:** [ledger.py:38](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/cluster/ledger.py:38) has no durable record of an unresolved submission. If acceptance loses its receipt and accounting has not exposed the allocation yet, a successful empty accounting response produces **zero reservation**. I confirmed that behavior with the extracted function and an in-memory ledger. A subsequent submission can therefore pass admission without charging the accepted job.

   **Fix:** include unresolved submission intents in reserved hours until positively reconciled. Use the same lock for every ledger-writing entry point and atomic file replacement; standalone `ledger.py update` currently bypasses the submission lock and writes directly. Validate returned allocation identity and actual time limit against the reservation.

**Overall: NOT READY.** Close the C11 degenerate-case acceptance, queue-query failure handling, and durable submission/reservation reconciliation before submitting the ECSW pilot.