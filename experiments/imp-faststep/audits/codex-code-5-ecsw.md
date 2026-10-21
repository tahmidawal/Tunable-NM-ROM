**READY for the ECSW pilot.** Audited commit `45ba429962edae291cf568dbb4ea117d683539da`, including the changes since `HEAD~1`. Audit 4’s blocking findings are closed. No new blocking defect found.

Read-only throughout: no file changes, cluster commands, submissions, JAX execution, or disk scratch. Checks used extracted functions, small NumPy fixtures, and mocked accounting/submission operations.

1. **Galerkin target and contributions — CORRECT.**

   The construction computes
   $$
   b_s=\Phi_r^\top N_h(\Phi_r\Phi_r^\top u_s),\qquad
   g_i(s)=\Phi_r[i,:]^\top N_{h,i}.
   $$
   Candidate nodes are columns; retained state blocks and targets receive identical scaling. Central advection uses zero Dirichlet boundaries.

   An independent stencil fixture matched the extracted construction to **\(2.22\times10^{-16}\)** for both \(G\) and \(b\).

2. **Importance distribution, sampling, floors and dropped blocks — CORRECT.**

   Importance mass is \(\sum_s|N_{s,i}|\|\Phi_r[i,:]\|\). Sampling retains the registered seed and uniform/importance split. Floors, retained-block scaling, all-node handling and explicit degenerate-input rejection remain correct.

   Fixtures confirmed unique candidates, dropping a zero block, selecting every node when appropriate, and rejecting all-zero targets.

3. **NNLS budgets and unavailable checkpoints — CORRECT.**

   Limits remain **240/360/600 seconds per attempt** for ranks 128/256/384. A retry receives the full limit. Unavailable checkpoints remain `None` and produce no weight arrays; certified KKT-optimum duplicates remain permitted by A3.1.

   Both attempt summaries and total NNLS elapsed time are persisted. A3.5 correctly distinguishes configured budgets from runtime including overshoot and finalization.

4. **C11 aggregate recomputation and uniform witness — CORRECT.**

   Recomputed residuals use the same stacked, scaled \(G,b\). The witness samples all mesh nodes with weights \(n/|E|\), prefers κ=8, and follows the registered fallback and escalation.

   Zero/zero and non-finite witness residuals cannot pass. Witness nodes are saved. Failed C11 yields `accepted=False`, exits unsuccessfully, and creates no `COMPLETE` marker.

5. **Memory — NEEDS-RESTATEMENT, non-blocking.**

   At 120 retained states, rank 384 and 32,768 candidates, \(G\) occupies **12,079,595,520 bytes**.

   Assembly is on the **device**, with possible overlap between individual blocks and concatenated output. “12 GB host + device” is therefore inaccurate. Basis storage, retained states, QR buffers and temporaries add to the footprint.

   Static sizes plausibly fit H200 and `200G` host RAM; actual peak capacity remains a pilot measurement. Treat telemetry `-1` as unavailable.

6. **Staging, imports, batch environment and provenance — CORRECT.**

   All **35 selected inputs** match their HEAD blobs. Static inspection found the ECSW import chain covered; the guarded staged import check remains present but was not executed.

   Committed-blob staging, working-copy equality checks, pinned basis digest, nested-manifest coverage, absolute venv, f64/highest precision, GPU preflight exit 42, `JCPREF_STORE`, 1.25-hour reservation, driver timeout, output checksums and failure propagation remain intact.

7. **Submission safeguards — CORRECT. Audit 4 blocker closed.**

   Ambiguous `sbatch` failure now preserves the remote directory and local intent. Returned IDs are validated, and remote receipt persistence is checked before success is reported.

   Mocked executions confirmed:

   | Failure | Exit | Directory deletion |
   |---|---:|---|
   | Queue query fails | 70 | No |
   | `sbatch` outcome ambiguous | 72 | No |
   | Invalid returned ID | 74 | No |

   Campaign locking, cap exit 75, lane serialization, directory exclusivity, duplicate refusal and before/after queue queries remain present. Both shell syntax checks pass.

   **Non-blocking wording fix:** `submit.sh`’s header still says rejected `sbatch` removes the directory. Update it to describe preservation pending reconciliation. Recovery remains conservative and requires reconciliation before retry.

8. **Ledger reservation, accounting and cap — CORRECT for this one-GPU submission path. Audit 4 blocker closed.**

   Admission now rejects recorded name/time-limit/GPU-count mismatches and unreconciled jobs. Nonterminal reservations use the larger local/accounting time limit and account for reported GPU multiplicity.

   Mocked accounting results:

   | Fixture | Reserved | Admission of another 2 h |
   |---|---:|---|
   | Normal running allocation | 1.25 h | Admit |
   | Actual limit increased to 3 h | 3 h | Refuse |
   | Job-name mismatch | 1.25 h | Refuse |
   | Two allocated GPUs | 2.5 GPU-h | Refuse |

   Pending allocations retain their reservation; ordinary terminal allocations contribute elapsed time. Durable unresolved intents, atomic ledger replacement, locking and SSH-error rejection remain intact.

**Overall: READY to submit the ECSW pilot.** This clears the code gate; real-size memory, runtime and C11 acceptance still require the pilot’s evidence. Concurrent edits to `DESIGN.md` appeared during this audit; staging correctly refuses that dirty state until resolved.