## Round-1 disposition

1. **PARTIAL** — D3, ROLL, per-trajectory E1, NaN handling, and the C1 threshold were repaired, but `complete` and row authorization still omit contractual prerequisites; `sep_b3d_tensor.py:1290-1301`, `runs/b3dtensor/gen_tables.py:71-78`.
2. **UNRESOLVED** — the full oracle still independently reconstructs the production residual instead of sharing its decomposition; `b3d_common.py:335-342`, `sep_b3d_tensor.py:717-720`.
3. **RESOLVED** — flattening, axis offsets, stencils, and sampled seven-point ordering remain consistent; `b3d_common.py:62-74`, `b3d_common.py:278-298`, `sep_b3d_tensor.py:671-673`.
4. **RESOLVED** — the defect ladder now accurately documents “up to k accepted corrections” and records stalls; `b3d_common.py:608-618`, `b3d_common.py:653-665`.
5. **RESOLVED** — TA uses the algebraic denominator, TB removes the actual final chunk, and T0 covers four slices of every test trajectory; `sep_b3d_tensor.py:768-826`.
6. **PARTIAL** — exact D4/IC field misfits and P1 row flags are repaired, but disallowed/censored arms are still printed as result rows; `sep_b3d_tensor.py:570-575`, `sep_b3d_tensor.py:882`, `sep_b3d_tensor.py:1298-1301`, `runs/b3dtensor/gen_tables.py:72-78`.
7. **RESOLVED** — D4 starts, POD snapshots, and metrics now derive exclusively from training data, with validation used as targets; `sep_b3d_tensor.py:314-315`, `sep_b3d_tensor.py:420-434`, `sep_b3d_tensor.py:583-635`.
8. **RESOLVED** — `S_tr`, `Z_tr`, `S_pod`, encoder pairs, multistarts, and EQ picks are training-only; PILOT does not load the test table; `sep_b3d_tensor.py:312-315`, `sep_b3d_tensor.py:397-475`, `sep_b3d_tensor.py:661-665`, `sep_b3d_tensor.py:865-868`.
9. **PARTIAL** — bank-sized JIT arguments are now explicit, but M1 limits remain unevaluated and M1 declares completion unconditionally; `sep_b3d_tensor.py:551-576`, `sep_b3d_tensor.py:968-980`.
10. **RESOLVED** — the original C1 omissions—count, common dimensions, eight trajectories, 1.25 threshold, attempts, and completion—were repaired; `sep_b3d_kernels.py:75-104`, `runs/b3dtensor/gen_tables.py:135-143`.
11. **PARTIAL** — pilot staging, output preservation, and result hashing improved, but submission is not idempotent under concurrent pushes or an interrupted `JID.tmp` transaction; `cluster/push_b3dtensor.sh:14-22`.

## Round-2 findings

12. **CORRECT — training split.** No validation state reaches `S_tr`, `Z_tr`, `S_pod`, encoder pairs, multistart codes, EQ selection, or training-head diagnostics; validation is used only as D4 targets and for the mandated F3/F5 checks. Fix: none. `sep_b3d_tensor.py:314-315`, `sep_b3d_tensor.py:420-475`, `sep_b3d_tensor.py:583-584`, `sep_b3d_tensor.py:661-665`, `sep_b3d_tensor.py:865-868`.

13. **NEEDS-RESTATEMENT — oracle exactness is correct but not cheap.** `Jᵀr` and `‖GJ_h‖_F` are mathematically correct, and the eight-start `vmap` does not materialize `8×n_i`; the field residual is formed only after selecting one start. However, `Gb.T @ Gb` is recomputed for every target despite the already available Gram matrix. Fix: pass the exact precomputed Gram into `solve` and use it for `Jᵀr`/Frobenius normalization. `sep_b3d_tensor.py:545-575`.

14. **CORRECT — bank-sized JIT capture.** `G_int`, `G_pool`, `G_all`, and `Phi_j` enter compiled functions as explicit arguments; only model parameters and grid-independent reduced arrays remain closed over. Fix: none. `sep_b3d_tensor.py:556`, `sep_b3d_tensor.py:656-657`, `sep_b3d_tensor.py:717-721`, `sep_b3d_tensor.py:787-797`, `sep_b3d_tensor.py:883-910`.

15. **WRONG — final contract validation remains incomplete.** D4 is included indirectly through `gates_all_pass`, but row authorization omits bracket, phase-0 evidence, pilot promotion, C1, and N=129 M1; `complete` also ignores P1. E1 failure is a declared negative outcome and TR concern is only a caveat, but neither positive oracle-equivalence claims nor caveat propagation are enforced. The table generator prints rows even when `row allowed=False`. Fix: perform a post-C1 cross-artifact validation and suppress disallowed rows. `sep_b3d_tensor.py:1290-1301`, `runs/b3dtensor/gen_tables.py:69-78`.

16. **CORRECT — TB partial-chunk control.** `last_c = n_i-(n_chunks-1)T_CHUNK` returns the remainder for a partial chunk and exactly `T_CHUNK` when divisible. Fix: none. `sep_b3d_tensor.py:768-775`.

17. **WRONG — TR audit recomputes `rJ_T` twice.** The two calls used to obtain `ra` duplicate both residual and Jacobian work at every step. Fix: evaluate once or call a residual-norm-only function. `sep_b3d_tensor.py:1133-1136`.

18. **WRONG — cluster submission is still race-prone.** The BEFORE check is advisory, the test-and-create of `JID` is non-atomic, an SSH loss after `sbatch` but before `mv` can resubmit, and the preceding `rsync --delete` can delete an existing `JID`/`JID.tmp`. Fix: lock the entire remote sync-and-submit transaction atomically, preserve submission metadata from rsync deletion, and use a unique attempt directory. `cluster/push_b3dtensor.sh:14-22`.

19. **NEEDS-RESTATEMENT — NO_TEST staging works but is not enforced by the interface.** The corrected staged pilot manifests contain only the train table and the driver refuses to open test data, but invoking the documented stage command without externally setting `NO_TEST_TABLE` still ships test tables. Fix: make pilot/no-test an explicit, fail-closed staging mode. `cluster/stage_b3dtensor.sh:7-18`, `cluster/stage/pilot33a/MANIFEST.sha256:6`.

20. **WRONG — NaNs can still be hidden in the custom training stream.** Unlike `build_truth` and the validation stream, the training loop reduces `res/snaps` with Python `max/min` without first rejecting nonfinite values. Fix: explicitly test both `snaps` and `res` before every reduction and set failure state immediately. `sep_b3d_tensor.py:404-419`.

21. **WRONG — C1 accepts duplicate checkpoints.** It requires three entries and common `(K,R,M)` but not the exact distinct set `{33,65,129}`; repeated N values collide in the timing dictionary and can yield a false ratio of 1. Fix: assert distinct expected N values and eight-element `z0`, `nu`, and tolerance arrays. `sep_b3d_kernels.py:75-79`.

22. **WRONG — ROLL’s decoded-field gate samples only every seventh state.** The reported field maximum is over eight sampled rollout states, not all 50 required states. Fix: evaluate decoded-field discrepancy at every rollout step. `sep_b3d_tensor.py:953-965`.

23. **WRONG — the table generator treats withdrawn pilots as current.** Its unconditional glob currently includes `pilot33a/b` without a superseded marker. Fix: require artifact status/provenance metadata and move withdrawn runs to a separately labelled table. `runs/b3dtensor/gen_tables.py:152-158`, `B3D-NOTES.md:81-88`.

24. **WRONG — A1 cannot be evaluated from the emitted contract.** Test-oracle values are stored, but same-state ROM excess, the within-3× ratio, and the absolute-usefulness decision are never computed or enforced. Fix: compare ROM errors and oracle errors at the same test states/times and emit an explicit A1 verdict. `sep_b3d_tensor.py:644-650`, `sep_b3d_tensor.py:1027-1064`, `sep_b3d_tensor.py:1290-1301`.

| item | verdict | one-line fix |
|---|---|---|
| 1 | PARTIAL | Complete the cross-artifact contract validator and table filtering. |
| 2 | UNRESOLVED | Share one mutation-free production residual decomposition. |
| 3 | RESOLVED | None. |
| 4 | RESOLVED | None. |
| 5 | RESOLVED | None. |
| 6 | PARTIAL | Suppress censored/disallowed rows rather than merely labelling them. |
| 7 | RESOLVED | None. |
| 8 | RESOLVED | None. |
| 9 | PARTIAL | Enforce M1 memory, free-memory, RSS, and projected-time limits. |
| 10 | RESOLVED | None for the original item. |
| 11 | PARTIAL | Replace the JID transaction with atomic remote synchronization. |
| 12 | CORRECT | None. |
| 13 | NEEDS-RESTATEMENT | Pass the precomputed exact Gram matrix into the oracle. |
| 14 | CORRECT | None. |
| 15 | WRONG | Validate phase 0, promotion, bracket, C1, M1, P1, and claim eligibility after all artifacts exist. |
| 16 | CORRECT | None. |
| 17 | WRONG | Compute `rJ_T` once for `ra`. |
| 18 | WRONG | Atomically lock sync/submission and preserve JID metadata. |
| 19 | NEEDS-RESTATEMENT | Make no-test pilot staging explicit and fail closed. |
| 20 | WRONG | Reject nonfinite training chunks before Python reductions. |
| 21 | WRONG | Require distinct N={33,65,129} artifacts and complete trajectory arrays. |
| 22 | WRONG | Check ROLL field discrepancy at all 50 steps. |
| 23 | WRONG | Exclude or explicitly label superseded pilot artifacts. |
| 24 | WRONG | Compute and enforce same-state A1 oracle excess and usefulness. |

NO — the single blocking defect is the non-atomic, rsync-deletable remote JID guard, which can submit multiple jobs into the same output directory.