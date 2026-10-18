**NOT READY for the phase-I smoke.** I found reproducible CPU failures and several acceptance paths that do not implement A1–A5.

All paths below are relative to `experiments/jcp-hr-baselines/`. I made no file changes, submitted no jobs, and ran no GPU work. CPU checks used the prescribed interpreter, `JAX_PLATFORMS=cpu`, highest precision, and disabled bytecode writes.

1. **BLOCKER — WRONG: NNLS crashes when checking KKT.**  
   **Evidence:** `nnls.py:156–158`. `np.asarray(G.T @ res_)` produces a read-only NumPy view; assigning `wfull[S] = -np.inf` raises `ValueError: assignment destination is read-only`. The existing **G-nnls test actually failed here** during this audit.  
   **Fix:** make an explicit writable copy—or compute support/complement statistics without mutation. Then rerun G-nnls, including paths terminating by KKT.

2. **BLOCKER — WRONG: the registered ECSW N-under fixture cannot run through native GN.**  
   **Evidence:** `smoke.py:296–301` constructs supports capped at \(r/4\) and \(r/8\). `rom.py:126–127` applies reduced QR followed by a square triangular solve. With fewer residual rows than coefficients, \(R\) is rectangular.  
   A CPU reproduction with a \(2\times8\) Jacobian raised the corresponding dimension-mismatch `ValueError`. The first ECSW level therefore aborts the entire N-under phase, including the subsequent Q-DEIM ladder.  
   **Fix:** implement a minimum-norm least-squares step for underdetermined Jacobians, with explicit rank-deficiency handling, and test the actual registered support shapes.

3. **BLOCKER — WRONG: offline acceptance is neither complete nor independently recomputed.**  
   **Evidence:** `smoke_accept.py:40–54` trusts `orth_ok`, `optimality_ok`, and `solve_ok`; line 102 trusts the saved timing `ratio`. Line 40 checks only phases present in the output, without an expected-phase inventory. Missing N-swap produces no failed gate (`:88–99`). Missing values passed to `put` become `None` and are excluded from `ALL` (`:24–25,133`). No acceptance predicate checks the complete grid, Q1 evidence, or the CPU-gate results.  
   **Fix:** define an explicit required-evidence schema for both cells and every control. Missing, skipped, errored, non-finite, or unavailable required evidence must prevent acceptance. Recompute predicates from numerical evidence, not saved pass flags. Keep optional diagnostics separate from missing required gates.

4. **MAJOR — WRONG: Q1 can qualify without the prescribed tighter anchors.**  
   **Evidence:** `pipeline.py:201–213` accepts any `tighter` dictionary, including an empty one. CPU reproduction returned `q1=True` with no tighter anchors. `smoke.py:222–223` supplies only case 0, although the registered subset is the first **two** cases.  
   **Fix:** require the expected cohort explicitly, exact production/tight coverage, and both prescribed tighter cases. Run and persist both smoke tighter anchors.

5. **MAJOR — WRONG: certification does not reject damping saturation.**  
   **Evidence:** `rom.py:275–282` checks coefficients and reason codes but ignores `out['stats']['lam']`. The inherited LM updates damping after computing its reason (`t2core.py:96–100`), so the returned damping can reach \(10^{14}\) without reason 3.  
   A synthetic CPU record with reason 4, gamma 0, and damping \(10^{14}\) was accepted as both valid and anchor-valid.  
   **Fix:** explicitly reject non-finite or saturated damping at every live step. Preserve this evidence in smoke outputs.

6. **MAJOR — WRONG: the timing statistic differs from A3.9, and incomplete timing coverage is assessed.**  
   **Evidence:** `stats.py:33–37` pools all repeats, both for the point estimate and bootstrap. A3.9 requires a median within each case, then a median across cases. `pair_label` defaults to three timing cases (`:65,72`), rather than all six.  
   CPU counterexamples:
   - Case ratios `[.5,.5,.5], [.5,.5,.5], [.5], [2], [2], [2]` give **0.5** in the implementation versus the registered **1.25**.
   - Three timing cases can receive **“baseline dominates.”**  
   Also, `log(median(rho))` is not generally `median(log(rho))` for an even sample.  
   **Fix:** implement the specified case-level estimators directly, bootstrap cases, and require the exact six timing-case identities.

7. **MAJOR — WRONG: picks still use the superseded absolute-time rule.**  
   **Evidence:** `stats.py:151–159` selects B_cost using `c['t'] <= X['t']` and B_acc using absolute time. It has no explicit paired-ratio input or X-specific-front construction. This does not implement A4.8/A5.3.  
   **Fix:** introduce explicit per-X paired-ratio statistics and fronts. Use them for role selection, and make Stage B include their union with the absolute-time front and near-front additions.

8. **MAJOR — WRONG: the timing harness loses required evidence and does not validate N-time as registered.**  
   **Evidence:** `smoke.py:75–91` discards rejected measurements when retrying; A4.10 requires every rejected triple and retry count. N-time runs only one case (`:379–388`), includes invalid triples in its reported median, and never checks the “X faster” label. Timed outputs are discarded; neither A–B–A nor the warm screening pass verifies equality with the scored/certified trajectory (`:214–216,381–406`). `pipeline.py:173–181` has no burn.  
   **Fix:** retain every attempt and its outputs or reproducibility evidence, filter valid triples, exercise all six cases, recompute the registered statistic and label offline, and assert timed/scored trajectory agreement. Burn before every measured screening block as well as A–B–A calls.

9. **MAJOR — WRONG: N-under lacks complete Q1 and can count solver failure as degradation.**  
   **Evidence:** `smoke.py:288–311` records only production errors, validity, and nonstationary counts. It runs no own-problem tight/tighter anchors. `smoke_accept.py:69–73` counts any finite doubled error, ignoring even the saved validity fields.  
   **Fix:** compute complete Q1 at every ladder level, including distances to that level’s own anchors. Preserve and report solver failures separately; do not credit an invalid solve as demonstrated sampling-induced accuracy degradation.

10. **MAJOR — WRONG: N-grad does not exercise the trajectory certifier.**  
    **Evidence:** `smoke.py:321–345` directly computes `RM.ratio(J,r)` and compares it with autodifferentiation. It never invokes `make_certifier` or the certification-label path on perturbed tuples. Consequently, bugs in the actual certifier’s history indexing or labels can survive this control. It also does not establish that the base fixture is certified-valid and silently substitutes `none` when ECSW is absent.  
    **Fix:** expose a shared isolated-tuple certification function used by the production certifier, exercise that function, validate the frozen ECSW fixture, and record tuple histories and actual flags. Treat unavailable fixtures according to A5.4.

11. **MAJOR — WRONG: N-swap does not implement A4.4/A5.1.**  
    **Evidence:** `smoke.py:361–366` chooses the first native ECSW grid entry without checking certified validity. It does not implement the certified-valid `none` fallback or record fixture-unavailable status. `smoke_accept.py:94` silently accepts a missing ROM fixture. The time-shift implementation (`score.py:78–80`) also includes an extra unshifted final-time comparison, whereas the registered shift compares indices \(1,\ldots,4\).  
    **Fix:** select the exact registered ID, check validity, attempt the registered fallback, and explicitly report unavailability. Implement the shift on the registered four comparisons. Require the full kind-control inventory, rather than deriving expectations only from controls present.

12. **MAJOR — WRONG: important real-size G-fd checks are absent.**  
    **Evidence:** `smoke.py:241–275` checks directional derivatives and autodifferentiation, plus independently assembled ECSW residual rows. It does **not** implement the real-mesh independent raw stencil/Jacobian check or the Q-DEIM full-field norm/gradient comparison, including the cancellation candidate.  
    The small tests contain some corresponding checks, but `tests/test_core.py:215–227` uses residual/gradient-relative denominators instead of A2.1’s cancellation-safe \(\Sigma\) bounds and does not record the cancellation ratio.  
    **Fix:** implement the missing real-size checks with sampled independent stencil derivatives where needed, and use the registered mixed tolerances and cancellation evidence.

13. **MAJOR — WRONG: POD construction failures are reported but not enforced.**  
    **Evidence:** `pipeline.py:81–87,119–131` records certificates yet returns and uses the basis regardless. `qdeim` likewise returns data even when `solve_ok=False`; `arm_data` does not enforce constructibility. The fallback-selection condition (`:76,113`) rejects the entire fallback if it loses any previously passing rank, potentially rejecting a rank that passed one of the two bases—contrary to A5.2.  
    `certificate_ok` (`:196–198`) also omits the explicit saturated-rank rule: a CPU check accepted \(e=1.5\times10^{-6}\), \(\tau=0.9\times10^{-6}\), although A5.2 requires \(e\le10^{-6}\) in that regime.  
    **Fix:** enforce certificates, orthonormality, finiteness, and DEIM backward-error gates before construction/use. Resolve fallback selection per rank while preserving the registered basis-prefix contract, or explicitly restate conflicting fallback outcomes.

14. **MAJOR — WRONG: NNLS’s final KKT check omits support stationarity.**  
    **Evidence:** `nnls.py:157–161` masks the support and tests only the largest off-support positive gradient. A5.2 also requires the two-sided support bound. `build.py:327–328` drops banned-index counts and KKT diagnostics from persisted path summaries.  
    **Fix:** check nonnegative weights, finite residuals, two-sided support stationarity, and the off-support bound including banned indices. Persist all diagnostics and distinguish iteration exhaustion from actual KKT termination. Exact residual verification at recorded tau checkpoints (`nnls.py:101–106`) should be retained.

15. **MAJOR — WRONG: the smoke does not measure all required maximal workloads or preserve their progress reliably.**  
    **Evidence:** maximal Q-DEIM and ECSW phases exist (`smoke.py:164–192`), but all online grid runs use `rs` (`:195–212`); the registered 2D `none` pilot at \(r=512\) is absent. JSON is written only after an entire cell returns (`:432–440`), despite the progressive-output claim. A wall-time termination during the first expensive cell can lose its completed phase evidence.  
    **Fix:** add the maximal `none` query/QR pilot, synchronize and record peak memory at each required phase, and checkpoint after every phase. Measure rather than assume feasibility of the full one-hour schedule.

16. **MAJOR — WRONG: CPU controls omit or alter registered fixtures.**  
    **Evidence:** `tests/test_core.py:269` tests only four of the twelve 3D options in G-full. G-lmm calls `smooth_state` with its default `seed=1` (`:49–51,353`), so it uses case 1 rather than the frozen case 0. It requires both order estimates to pass and omits A4.6’s fifth-level refinement (`:374–378`). N-dom omits both A4.2 identical-arm regimes and tests only two missing timing cases, which misses the three-to-five-case bug (`:394–403`).  
    **Fix:** use the exact fixtures, all registered options, the bounded refinement policy, both degeneracy regimes, and explicit failures with three, four, and five timing cases.

17. **MAJOR — WRONG: the STATUS pin can check a different file from the loader.**  
    **Evidence:** `score.py:28–30` checks the loader’s manifest version but hashes the hard-coded local STATUS path. The inherited loader reads STATUS from `JCPREF_META` when set (`e0/refstore.py:61–80`). `cluster/stage.py:89–93` does not clear or pin that variable. Thus an alternate STATUS with the same manifest version can alter eligibility without failing this pin.  
    **Fix:** hash the exact metadata paths used by the loader and explicitly pin the metadata directory in the job environment.

18. **MINOR — NEEDS-RESTATEMENT: phase-II use needs stronger interfaces and resource accounting.**  
    **Evidence:** `pipeline.py:190–193` copies every decoded full field to the host before selecting lattice nodes. `build.py:294–307` retains all contribution blocks while constructing scaled blocks and concatenating them. `cluster/submit.sh:17–23` serializes this lane’s submissions, but its lane-local lock cannot make the account-wide cap atomic against other lanes.  
    **Fix:** select lattice rows on-device; include actual simultaneous buffers and compilation shapes in `manifest.py`/`resources.py`; coordinate account-wide submission locking if the cap must be strict. Phase II also needs audited Q1+Q2 exports, replay gates, complete cohort eligibility, X-specific fronts, budget accounting, and fail-closed coverage reporting. Their absence as unwritten production drivers is not itself a phase-I defect.

The core mathematics fared better than the acceptance machinery. The inspected residuals, Jacobians, FOM history updates, CNR/BDF2 startup handling, ECSW training histories/scaling/all-node targets, and untruncated Q-DEIM \(R_B\) construction match the governing formulas. CPU G-fd-a/b/c, G-lmm, small Q-DEIM interpolation/pivot comparisons, and the existing N-dom tests passed. The pivot implementation is projection-based rather than LAPACK Householder QR; its two tested random fixtures agreed, which does not establish universal finite-precision pivot identity.

All **17** `INHERITED.json` hashes matched; all **16** entries claiming source-commit identity matched those Git blobs byte-for-byte. The remaining entry is explicitly a shim. The cluster scripts include GPU preflight exit 42, x64/highest precision, paralab staging, manifest verification, unique attempt directories, queue checks, and staged-import isolation. I found no large captured-array closure in the lane’s principal jitted ROM/FOM/contribution kernels; maximal GPU memory remains unverified.

**Overall verdict: NOT READY.** Top blockers are the reproducible NNLS KKT crash, the guaranteed underdetermined-GN failure in N-under, and an acceptance program that can omit required evidence or trust in-job flags. Fix those, the Q1 omissions, and the timing estimand before spending the phase-I GPU allocation.