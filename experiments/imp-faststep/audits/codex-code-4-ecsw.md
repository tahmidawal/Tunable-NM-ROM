**NOT READY to submit.** Audited commit `80520a70d416c4c344fac14e639f66fd4ea6afed`, including `git diff HEAD~1`. The numerical fixes close audit 2’s principal findings. Submission ambiguity and ledger mismatch enforcement remain blockers.

Read-only throughout: no files changed, jobs submitted, cluster commands executed, or arrays copied to disk. Tests used small NumPy fixtures and mocked external operations; no JAX imports or production-size execution.

1. **Galerkin target and contribution layout — CORRECT.**

   `ecswjob.py:46–90` constructs
   $$
   b_s=\Phi_r^\top N_h(\Phi_r\Phi_r^\top u_s),\qquad
   g_i(s)=\Phi_r[i,:]^\top N_{h,i}.
   $$
   Candidate nodes are columns. Retained state blocks and their targets receive identical scaling. `adv_central` supplies the specified central operator with zero Dirichlet boundaries.

   Re-executing the extracted construction with NumPy against an independently assembled stencil gave maximum differences of **\(5.6\times10^{-17}\)** for \(G\) and **\(1.1\times10^{-16}\)** for \(b\). This checks arithmetic and layout, not GPU execution.

2. **Importance sampling, floors and dropped blocks — CORRECT.**

   The importance mass is exactly \(\sum_s |N_{s,i}|\|\Phi_r[i,:]\|\). Candidate sampling, seed, positive floor, retained-block mask, all-node branch, and explicit degenerate-input rejection remain consistent with the specification and reused implementation.

   Small fixtures confirmed unique candidates, dropping a zero block, selecting every node on a small mesh, and rejecting all-zero targets. No regression from audit 2 found.

3. **NNLS budgets and unavailable checkpoints — CORRECT.**

   Limits remain **240/360/600 seconds per attempt** for ranks 128/256/384. The unchanged `ecsw_weights` passes the full limit to each attempt, including a banning-triggered retry.

   Unavailable checkpoints remain `None`; their weights are not saved. Certified KKT-optimum duplicates remain permitted by A3.1. Both attempt summaries and total NNLS elapsed time are persisted. A3.5 correctly governs the interpretation: configured budgets do not bound overshoot and finalization.

4. **C11 aggregate and uniform witness — CORRECT. Audit 2’s fixes closed.**

   Aggregate recomputation still uses the same stacked, scaled \(G,b\). The witness samples from all mesh nodes, uses \(n/|E|\), prefers κ=8, and follows the registered fallback and escalation.

   Extracted decision tests confirmed:

   | Reference residual | Mutant residual | Fires |
   |---:|---:|:---|
   | 0 | 0 | No |
   | 0 | NaN | No |
   | 0 | 1 | Yes |
   | 1 | 9 | No |
   | 1 | 10 | Yes |

   Witness node arrays are now saved. `construction_complete` and `accepted` replace the ambiguous `complete=True`; failed C11 still exits unsuccessfully without creating `COMPLETE`.

5. **Memory — NEEDS-RESTATEMENT; no evident static capacity blocker.**

   At 120 retained states, rank 384 and 32,768 candidates, \(G\) alone occupies **12.08 decimal GB**.

   Current assembly is on the **device**, so “12 GB host + device” does not describe this implementation. Block arrays and concatenated output can coexist temporarily. QR buffers, basis arrays, nonlinear vectors, retained training outputs, compilation and solver temporaries add to that footprint.

   Static sizes plausibly fit H200 and `200G` host RAM; peak usage remains unverified. Device peak and host maximum RSS instrumentation remain present.

   **Required interpretation:** use the real-size pilot to establish capacity. A device telemetry value of `-1` means unavailable, not a measured peak.

6. **Staging, imports, batch environment and provenance — CORRECT for the audited ECSW configuration.**

   All **35 selected inputs** exist in Git and match their HEAD blobs. Static dependency inspection found the ECSW import chain covered. The staged import check uses the shared slot guard; I did not execute it.

   Retained safeguards include committed-blob staging, working-copy equality checks, pinned external basis digest, the configured **1.25-hour** reservation, absolute cluster interpreter, f64/highest precision, GPU preflight exit 42, `JCPREF_STORE`, manifest verification, driver timeout, output checksums, and failure-status propagation.

   The nested-manifest omission is fixed: only the root manifest being generated is excluded.

   The new `REMOTE_SUBMIT` has a blocking failure path, covered next.

7. **Submission locking, directory ownership and duplicate refusal — WRONG. Partially fixed.**

   **Closed:** failed queue queries now exit 70 before `sbatch`; campaign and lane caps use the successful query under the campaign lock. Local serialization, duplicate refusal, pre-submit intent, remote receipt, and post-submit queue diagnostics are present. Both shell syntax checks pass.

   **Remaining blocker:** [stage.py:49](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/cluster/stage.py:49) treats **every nonzero `sbatch` return as definite non-submission**:
   ```bash
   jid=$(sbatch --parsable run.sbatch) || { cd /; rm -rf "$REMOTE"; exit 72; }
   ```
   [submit.sh:37](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/cluster/submit.sh:37) then deletes the intent for exit 72.

   A submission communication failure can leave acceptance unresolved. This path can therefore delete an accepted job’s working directory and release its reservation before accounting exposes it. A mocked accepted-but-reply-lost failure reproduced **directory deletion and exit 72**.

   Also, the `JOBID` write at `stage.py:50` is unchecked: the script can print success after receipt persistence fails.

   **Fix:** preserve the directory and intent after an unsuccessful `sbatch` invocation until non-acceptance is positively established. Reconcile through receipt, queue and accounting, conservatively retaining unresolved reservations. Validate the returned ID and check atomic remote receipt persistence before reporting success. Only demonstrably pre-submission failures should automatically release the intent.

8. **Ledger reservation, accounting and cap — WRONG. Most audit 2 fixes closed.**

   **Closed:** unresolved intents are reserved; standalone updates take the lane lock; ledger replacement is atomic; SSH failures raise; positive finite admission input is required. Ordinary queued/running allocations retain full reservations, while recognized terminal allocations count elapsed time.

   **Remaining blocker:** [ledger.py:78](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/cluster/ledger.py:78) records `slurm_time_limit_hours` and `identity_ok`, but neither affects admission. Reservation still uses the original local limit.

   An in-memory accounting fixture reproduced:

   | Local reservation | Actual running limit | `identity_ok` | New request | Decision |
   |---:|---:|:---|---:|:---|
   | 1.25 h | 3 h | False | 2 h | **Admit** |

   Actual reserved demand would be **5 h against the 4 h cap**. Job-name mismatches likewise set a flag without blocking admission.

   **Fix:** reject admission on unresolved identity or time-limit mismatches. Conservatively reserve at least the larger verified time limit while reconciling. Enforce the recorded allocation’s one-GPU assumption as well. Preserve unresolved intents through the ambiguous submission path identified above.

**Overall: NOT READY.** Close the destructive handling of ambiguous `sbatch` failures and enforce accounting mismatches during admission before submitting the ECSW pilot.