**NOT READY for the real-size smoke.** Audit 6’s principal fixes are present, but an output-schedule mutation still escapes certification, refinement validity is incomplete, and two controls do not exercise the production implementation.

Reviewed commit `e6437d484`, including A5’s implementation amendments. I inspected the supplied artifacts and ran small NumPy/AST probes; no solver code changed and no jobs were submitted.

The two-rank CPU fixture confirms:
- Timing identity status `ok`, selection agreement, and zero offline scoring disagreements.
- The missing bank/native anchor arrays are now saved.
- Acceptance remains correctly false: controls were disabled, and rank-256 C9 returned **1.13864**, outside `[0.9, 1.1]`. This CPU timing result does not establish an H200 defect.

1. **CORRECT — precontraction, steppers and coordinate maps.**

   The QR/eigenbasis transformation, CN residual and Newton Jacobian, tolerance proportional to $\|z_n\|$, ETDRK4 recurrence, and one-pass linearised CN match the amended equations. Coefficient construction and coordinate maps remain inside evolution.

   **Non-blocking restatement:** `phi_np` still uses 40 Taylor terms below magnitude 0.5, whereas A1.3 specifies 12 below 0.2. Document the implemented evaluator.

2. **WRONG — certificates still permit incorrectly scheduled implicit-arm outputs.**

   The main Galerkin checks retain the repaired diffusion, initial-coordinate, output-map and per-step reconstruction gates.

   However, [podarm.py:260](experiments/imp-faststep/podarm.py#L260) certifies native trajectory `C`, while scoring uses `W`; neither it nor the offline native branch verifies that `W` equals the scheduled rows of `C`.

   **Concrete mutation:** swap native output rows 1 and 2 before the query decodes `W`, leaving `C` unchanged. On the saved dev6[0] fixture:

   | Check | Original | Swapped outputs |
   |---|---:|---:|
   | Maximum difference between `W` and scheduled `C` | 0 | 6.46516 |
   | Maximum certified gamma | 0.000911740 | unchanged |
   | Nonstationary steps | 0 | unchanged |

   Applying the same output permutation to production and anchors preserves their maximum-over-time distances. Offline decoding agrees with the incorrectly scheduled fields, and timing repeats the same incorrect program. This is an implementation mutation, not merely forged JSON.

   Bank LSPG similarly lacks a `W`-versus-scheduled-`states` gate.

   **Fix:** independently check scheduled outputs against certified trajectories for bank LSPG and native POD, including anchors and successors. Exercise those checks with schedule mutations.

3. **CORRECT for bank Q1/Q2 execution; NEEDS-RESTATEMENT for independent evidence verification.**

   All configured cases receive successor rollouts; force ratios use main-rule states. Production configs request all 38 cases. Fresh anchors, pooled production certificates, CN budget invalidity, and bank anchor persistence are implemented.

   The checker still trusts anchor booleans instead of recomputing them from the newly saved arrays. Native anchor certification additionally depends on exit reasons and damping values that are discarded.

   **Fix:** recompute anchor predicates and distances from retained evidence; save native `reason` and `lam` arrays so `rom.certify`’s stall/saturation decisions can be reconstructed.

4. **WRONG in the refinement callers; A4’s classifier itself is CORRECT.**

   All six prescribed boundary fixtures returned `unresolved`. Production growth and ETD stage-finiteness monitoring are present.

   Two caller defects remain in [fsrun.py:454–462](experiments/imp-faststep/fsrun.py#L454):

   - LSPG refinements accept `gamma <= GAMMA OR residual <= tolerance`. The production stationarity certificate has no residual exception; that exception belongs to anchors.
   - CN refinements call `g_valid` **without their certificate**, bypassing `budget_unconverged`. One bad budget exit among 400 steps can satisfy the pooled 1% certificate allowance while violating run validity, allowing a resolved growth classification.

   **Fix:** use the registered production stationarity predicate; compute each CN certificate once and pass it into `g_valid`. Invalid refinements must produce `unresolved`.

5. **CORRECT for independent E0 decoding and cohort restriction; output identity remains blocked by item 2.**

   Native outputs are now independently decoded and rescored, closing audit 6’s field-only permutation hole. Bank and POD-Galerkin rescoring remain present, with store/metric checks.

   The driver admits only dev6/val32. `refstore` itself allows historical test cohorts, but the lane selector cannot select them; sealed/OOD cohorts are explicitly rejected.

   Independent decoding does not establish output-time identity when both decoders consume the same incorrectly scheduled coefficients.

6. **CORRECT — POD Galerkin mechanics and successor policy; WRONG — native verification remains incomplete.**

   ECSW and full-node providers, central Galerkin diffusion, un-precontracted twins, distinct-rule successors, duplicate aliases, and Q2-POD successor certification match the amendments. The configs contain the amended POD ladder and native picks.

   Native outputs and both anchor trajectories/gamma arrays are now persisted. Remaining fixes are the schedule gate in item 2 and reconstructible native validity evidence in item 3.

7. **CORRECT — timing mechanics and repaired in-job selection.**

   The rank-bound guard repair is supported by the two-rank execution. Best-stepper ties now use `FS.pick_min`, implementing lower error then configuration ID.

   E/F/D subjects, ratio directions, warm-up identity checks, retry-wide identity propagation, headline coverage checks and direct matched/F5 triplets are present. Acceptance now requires timing identity and selection agreement.

   Native timing cannot detect the consistently wrong output schedule described in item 2.

8. **WRONG — C19/C21 test a parallel helper, not production successor construction.**

   Control coverage and witness-based recomputation are substantially improved. C1–C24 are represented, and missing controls prevent acceptance.

   But [checks.py:367](experiments/imp-faststep/checks.py#L367) exercises `FS.pod_successors`, while [podarm.py:152](experiments/imp-faststep/podarm.py#L152) constructs successors separately. Mutating that production loop leaves C19/C21 passing. C19 also does not reproduce qualification labels after checkpoint removal as registered.

   **Fix:** share the actual production successor builder with the controls, and exercise unavailable/duplicate checkpoints through the production labeling path.

9. **WRONG — resource certificate; NEEDS-RESTATEMENT — H200 readiness.**

   No obvious captured full-bank/POD constants were found in the principal JIT paths. Nested bank prefixes share blocks. Actual peak memory and runtime still require the registered H200 pilot.

   [resources.py:23](experiments/imp-faststep/resources.py#L23) cannot currently certify production cost:

   - It ignores target ranks, arms and timing-subject counts.
   - `smk4096` omits rank 128 and controls; `f4096` adds both without pricing them.
   - Dev-only order blocks are not scaled to six cases.
   - Rank setup and stability-classification costs are omitted.
   - Setup already includes bank construction, which is added again.
   - No mesh/device compatibility check prevents using an unsuitable smoke.

   My arithmetic probe produced an **identical estimate after removing a target rank and five target arms**.

   **Fix:** price the target inventory explicitly, include unmeasured mandatory work conservatively, scale dev-only blocks, and reject unsupported extrapolations. The JSON time limits are reservations, not validated projections.

10. **WRONG — remaining reporting gaps.**

   E5 smoke verdict suppression and global timing-identity failure propagation are repaired. However:

   - [checks.py:273](experiments/imp-faststep/checks.py#L273) still emits F5 gain predicates for smoke subsets without requiring headline coverage.
   - Direct comparison statistics likewise remain `ok` on smoke subsets rather than explicitly non-headline.
   - The registered same-rank/shared-integrator diagnostic POD pick—smallest available κ within `1.1 × P-E4-none`, with its fallback—is not implemented in the decision outputs.

   **Fix:** label smoke F5/comparison outputs descriptive and suppress their verdicts. Implement the diagnostic pick and comparison before the final scientific report.

The decisive pre-smoke fixes are **implicit-arm output identity, refinement validity, and controls exercising production successor logic**. The resource estimator also needs correction before it can authorize production. The canonical lab log records this audit outcome.