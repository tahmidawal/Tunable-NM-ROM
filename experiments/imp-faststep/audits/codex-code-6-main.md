**NOT READY for the real-size smoke.** Most audit-3 fixes are implemented, but the two-rank smoke has a new timing-guard failure, and acceptance, native POD scoring, and evidence persistence still have blocking gaps.

I inspected the supplied CPU artifacts read-only and ran small NumPy/Python checks using extracted production functions. No files were changed and no jobs were submitted.

The CPU run reports **446.9 seconds**, zero offline disagreements, and only C11 false. The real archived ECSW result has `accepted=true`, `c11_ok=true`, and GPU preflight evidence; its three rule files match `inputs/gecsw/` byte-for-byte. **The fixture’s C11 failure is not a blocker.**

1. **CORRECT — precontraction, steppers, and coordinate maps.**

   The QR/eigenbasis transformation, projected nonlinear force, CN residual and Newton Jacobian, tolerance proportional to $\|z_n\|$, ETDRK4 recurrence, and one-pass linearised CN equation remain consistent with the specification. The contour coefficients match `pscore.etdrk4_coefs`.

   A5’s independent initial-coordinate and output-map checks are implemented. Production Galerkin initial coefficients are compared bitwise with arm 1’s.

   **Non-blocking restatement:** the independent phi evaluator uses 40 Taylor terms below magnitude 0.5 rather than A1.3’s stated 12 terms below 0.2. Update the specification to describe the implementation.

2. **CORRECT for the main Galerkin certificates — audit-3’s two reproduced holes are closed.**

   [fsrun.py:158](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:158) gates the eigen-residual and orthogonality against original tables, fingerprints the transformed tables, and independently reconstructs initial coordinates and returned coefficients.

   Consequently, the previous altered-diffusion and zero-output-map mutations can no longer pass through the normal production path. CN, ETD, and linearised-CN certificates cover every saved step of every configured case.

   This does **not** close native POD field/scoring verification; see items 5–6.

3. **WRONG — Q1/Q2 execution is substantially repaired, but A5.4 evidence remains incomplete.**

   Closed findings:

   - LSPG anchors receive fresh autodiff certificates without the former `100 * gtol` exception.
   - CN budget exits above tolerance invalidate production runs.
   - The offline checker computes pooled main-run certificate statistics.
   - Successor rollouts cover every configured case; own-state force ratios use main-rule states.
   - LSPG step-halving runs are present.
   - Saved initial coefficients are populated.

   However, [fsrun.py:315](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:315) still discards required anchor evidence:

   - LSPG tight/tighter states and residual arrays are not saved.
   - CN tight/tighter certificate arrays are explicitly removed.
   - CN tighter states are not saved.

   I confirmed these omissions in the supplied NPZ files. Stored gamma alone cannot independently establish A5.2’s residual-tolerance alternative.

   **Fix:** persist both anchor trajectories, fresh certificate arrays, and the validity evidence needed to recompute their decisions. Check required keys and shapes before accepting an output directory.

4. **WRONG overall — A4’s classifier is correct, but refinement certification is incomplete.**

   `fsselect.classify_growth` implements the exhaustive A4 rules correctly. All six prescribed boundary fixtures returned `unresolved` in my check. Accepted-step growth and ETD stage finiteness are monitored, and the former eight-run truncation is gone.

   But [fsrun.py:431](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:431) declares LSPG refinements acceptable using only finite output coefficients and the exit counter. It never freshly certifies their stationarity. An uncertified refinement can therefore receive “growth present/absent,” contrary to A3.2/A4.

   The offline alternation calculation also only handles bank files containing `Z`; LSPG files containing `states` receive no diffusion-coordinate alternation diagnostic.

   **Fix:** certify the actual used portion of each LSPG refinement trajectory and require its registered certificate before classification. Compute LSPG alternation after transforming saved coefficients into diffusion coordinates.

5. **WRONG overall — bank/POD-Galerkin offline scoring is repaired; native 5L scoring remains unchecked. Cohort restriction is CORRECT.**

   The lane selects only dev6/val32. The scorer checks the supplied case/time metadata, pinned store and metric, and records the reference digest. Independent offline decoding now checks bank and POD-Galerkin errors.

   However, [checks.py:291](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/checks.py:291) copies native 5L errors directly from JSON without independently decoding or rescoring them.

   **Concrete surviving mutation:** permute native decoded output times before scoring while leaving native coefficients and certificates unchanged. The separately constructed `rec_of(ci)` metadata still passes. Native timing compares scored coefficients, then checks fused outputs against their own warm-up; it does not compare scored fields. The offline native branch never detects the incorrect score.

   **Fix:** save native output coefficients and independently decode/rescore every native case, using the same agreement gate as the other arms.

6. **WRONG overall — POD Galerkin mechanics are correct; native replay evidence remains incomplete.**

   Closed findings:

   - Basis and native artifact digest pins are enforced.
   - Diffusion/table/map gates are implemented.
   - ECSW and full-node Galerkin providers and their twins match the central operator.
   - Successors skip duplicate rules and end at `none`.
   - Duplicate members are run/timed once with their κ aliases retained.
   - Q2-POD checks successor validity, stability and reconstruction; trivial indicators receive the registered label.
   - Native tight/tighter solves now run with the explicitly permitted `rom.certify` semantics.

   Nevertheless, [podarm.py:272](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/podarm.py:272) discards native anchor states and gamma arrays. The supplied native NPZ contains only `C`, `gamma`, and `case_ids`: **A5.12 is not implemented as written.**

   Native replay qualification is also not fully summarized: the checker reports validity/stability and anchor booleans, but omits the recorded stationarity and anchor-distance predicates.

   **Fix:** retain native production outputs, both anchor trajectories and certificates, and emit the descriptive native Q1 evidence explicitly. Add independent native scoring as above.

7. **WRONG — a new two-rank timing failure blocks `smk1`; best-stepper ties use the superseded rule.**

   **Blocking runtime failure:** [fsrun.py:175](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:175) defines `check_tab` inside the rank loop. Deferred timing subjects retain this function, but its `manifest` and `st` closure cells advance to the final rank.

   Thus, after constructing rank 384, the rank-128 Galerkin timing subject checks its tables against the rank-384 manifest and raises:

   ```text
   ValueError: precontracted table differs from the manifest
   ```

   I reproduced this using the production guard body in a two-rank closure fixture: rank 128 fails; rank 384 passes. The supplied single-rank CPU run cannot expose it.

   **Fix:** bind each rank’s manifest and setting through a factory or default arguments. Exercise deferred timing preparation for both ranks before the GPU smoke.

   **Selection error:** [fsrun.py:573](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/fsrun.py:573) retains A2’s priority `E4b, E4a, R1, CN`. A3.4 supersedes that with lower error, then lexicographic configuration ID. An equal-time/equal-error production-selector fixture chose `G-E4b`; A3.4 requires `G-CN`.

   The remaining timing repairs are sound: scored evolve hashes, fused full-tree hashes, identity propagation across discarded retries, E/F/D ratio directions, drift retries, and direct triplets are present.

8. **WRONG — control coverage is much improved, but the acceptance gate omits correctness failures.**

   C1–C24 now have implementations/imported evidence, including the previously missing controls. Positive order thresholds and C2 anchor checks participate in the in-job summary. The real C11 evidence is available.

   However, [checks.py:364](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/checks.py:364) accepts using only execution, control-summary success, and offline error agreement. It does **not** require:

   - a timing block free of identity failures;
   - agreement between recomputed and in-job selection.

   Executing that exact acceptance expression produced:

   | Timing status | Selection agrees | Accepted |
   |---|---:|---:|
   | `ok` | true | true |
   | `failed (identity)` | true | true |
   | `ok` | false | true |

   These are operational correctness failures, not merely forged inputs. A timing identity failure in a non-incumbent subject need not fail C9.

   **Fix:** make timing identity and selection agreement explicit acceptance gates. Recompute required control predicates from their recorded witnesses rather than trusting only `summary.all_ok`.

9. **NEEDS-RESTATEMENT — resource readiness remains a smoke measurement, not an established result.**

   I found no new obvious captured large-array constants in the reviewed principal JIT paths. Large bank/POD tables remain arguments.

   Still unpriced on H200: full-bank fingerprint scans, retained timing subjects across ranks, POD rotated bases and full-node evaluations, compilation, certification, and conditional refinements. The small CPU run does not establish real-size memory or runtime.

   **Action:** retain the mandatory real-size memory/runtime pilot after fixing item 7. Record peak device memory and setup/compile/certification/timing costs separately. `time_limit_hours` in the JSON is not itself an enforced driver deadline.

10. **WRONG — smoke and global timing-failure outcomes can still be mislabeled.**

   Execution completion is now correctly distinguished from acceptance, and the checker reports separate worst-error predicates and population-specific matching.

   Two decision-layer holes remain in [checks.py:247](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/checks.py:247):

   - `headline` is recorded but not required by the E5 speed predicate. A sufficiently fast two-rank smoke subset can produce an E5 success despite A5.8 forbidding smoke verdicts.
   - Global timing identity failure fires E5 speed conditions, but F5 gain and matched direct comparisons are still evaluated independently and can remain successful. A5.7 requires **every** speed condition to fire.

   **Fix:** suppress scientific verdicts for smoke subsets; require registered headline coverage before evaluating them. Propagate global timing identity failure into F5 and every comparison.

The immediate smoke blocker is the **late-bound rank guard**. Before resubmission, also repair the acceptance gates, native scoring/persistence, missing anchor evidence, refinement certification, and superseded tie rule. These are substantive implementation gaps, so **READY would be premature**.