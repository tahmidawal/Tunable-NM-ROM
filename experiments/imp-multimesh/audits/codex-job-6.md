**J0: FIX-FIRST. J5: FIX-FIRST.** The staged configurations are largely correct, but neither job is ready to submit.

| Check | J0 | J5 |
|---|---|---|
| H200, 240 GB host RAM, f64, highest precision, GPU preflight | Correct | Correct |
| Command budgets versus allocation | 6,600 / 6,900 s | 9,000 / 9,300 s |
| Resource certificate config hash | Matches | Matches |
| H200 feasibility | Plausible; empirical acceptance missing | Plausible; empirical acceptance missing |
| Required outputs | Correct on successful completion; persistence defective | Correct on successful completion; qualification defective |

**Blocking for both jobs**

The certificates cite historical estimates, not the required smoke-derived phase measurements × counts × 1.5. The selected smoke is `s1b`, but its local `pulled/output/ACCEPT.json` is absent. Obtain accepted controls and the device/host peak evidence required by A5.3, then regenerate the certificates before submission. I did not query live cluster status.

The allocations fit their outer command budgets, but this does **not** establish that the workloads finish within those budgets.

**J0 — blocking fixes**

1. **Persist source artifacts and terminal states incrementally.** In [staged mmbuild.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/runs/j0/experiments/imp-multimesh/mmbuild.py:125), JSON is saved after each completed rank, but `Phi.npy`, `C.npy`, `Phi_lat.npy`, `rules.npz`, and `trust.json` are written only after the entire rank loop. A later timeout/OOM loses the usable artifacts from earlier ranks. If **no rank certifies**, the loop never saves `src{L}` at all, yet the driver can set `complete=true`. Save certificates and basis artifacts before ECSW; update rules after each rank; preserve unfinished constructions as `UNRESOLVED`.

2. **Reconcile path budgets with the job budget.** The paths correctly target κ=4 and κ=8, and `max_seconds=2700` reaches NNLS. However, that allowance applies **per attempt**, including a possible retry. Six initial paths alone permit 16,200 s, against the 6,600 s outer timeout. Use measured projections and an explicit remaining-budget policy so the outer timeout does not destroy useful work.

3. **Add burn-in before the measured construction blocks.** `build_source` starts its snapshot, POD, code, state, and ECSW timers without explicit burn-in. These are cost-model measurements, so the repository’s timing requirement applies.

Memory does not reveal an inevitable OOM: the snapshot arrays are **15.28 GB at 256** and **57.87 GB at 1024**, and the 256 Gram matrix is **6.90 GB**. Eigensolver workspaces, dense ECSW Jacobians, and NNLS temporaries still require empirical peak evidence.

On successful completion, J0 supplies the requested artifacts and certified ranks. Its `snap_seconds`, `pod_seconds`, `codes_seconds`, `states_seconds`, and per-rank ECSW `seconds` match what `analysis.py::comp_inputs` consumes; contribution and NNLS subtimers are also retained.

Two avoidable costs remain: `basis_from` already computes projection codes, which `build_source` recomputes; `cell.set_basis` builds full Laplacian/derivative tables that this source path’s ECSW builder computes again. These waste time and memory.

**J5 — blocking fixes**

1. **Preserve unverified finer-rule convergence as `UNRESOLVED`.** [Staged oursq.py:88](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/runs/j5/experiments/imp-multimesh/oursq.py:88) treats a failed successor/check convergence test as Q2 `FAIL`. A tiny CPU fixture with Q1 passing, deployed-rule ρ and rollout bars passing, and all six finer-rule distances equal to 1 returned:

   ```text
   Q1 PASS Q2 FAIL NOT-QUALIFIED
   ```

   The inherited `p1score` explicitly labels this **“unverified (finer rule not converged)”**. E7 must return `UNRESOLVED`; otherwise it can remove X from the comparison and help create a vacuous stop-rule result.

2. **Complete the acceptance and identity checks.** `oursq` checks case coverage, mesh/Rp/scheme metadata, and W hashes, but takes qualification counts and distances from JSON rows rather than recomputing the predicates from saved evidence as COMMON-BRIEF requires. It also does not implement A2.5’s full descriptor identity check. Add these checks while preserving byte-identical numerical driver code.

The configurations themselves are **correct** on all three meshes:

| Setting | Deployed | Successor | Check |
|---|---|---|---|
| R128, BE/CNR | fib4181 | fib6765 | fib10946 |
| R256/R384, BE/CNR | gauss128 | gauss160 | gauss192 |

Every setting specifies frozen modes, `cases=all38`, and `q2_cases=all38`. The nominal Q1/Q2 thresholds, pooled stationarity denominator, and six dev6 anchors are correct. Successful execution writes all six `R_*.npz`, `result.json`, and `oursq.json` per mesh.

At 4096, the bank occupies **68.69 GB**, with approximately **5.10 GB** more for 38 resident initial fields, before query/setup temporaries. Blocked bank construction makes H200 execution plausible, but the bank size alone is not a peak-memory certificate.

Each mesh runs **756 trajectories**, or **2,268 total**. The configs also leave optional Q3 Jacobian SVD diagnostics enabled; these are not needed for E7’s Q1/Q2 deliverable and can be disabled through configuration to save time.

Validation: shell syntax passed; both certificate hashes matched; all checked manifest files ≤2 MB matched; staged `p1acc`, `p1common`, and `p1core` match the worktree copies byte-for-byte. No files changed, no jobs submitted, no large arrays loaded, and no lab-log or main writes.