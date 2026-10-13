**DO NOT SUBMIT.** Audit items **10 and 21 remain NOT RESOLVED**. The principal blockers are incomplete resource certification and missing A6.2 qualification data for the 3D selection ladder.

Read-only review completed: no files modified, no project modules imported, no jobs submitted. Checks used source inspection, JSON/NPZ reads, and isolated in-memory execution of the certificate verifier.

1. **BLOCKER — Resource-time projection can still underestimate production work. Item 10: NOT RESOLVED.**

   [inventory.py:82](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:82) includes more work than the previous estimator, but does not enumerate and price the actual invocation graph.

   | Previous item-10 defect | Status |
   |---|---|
   | Missing FOM phases and vendor queries | **NOT RESOLVED fully:** approximations added, but calibration uses evaluation timings multiplied by two instead of its recorded tight/tolerance-specific timings. |
   | Missing common timing panels | **RESOLVED as an omission:** included now; their costing remains approximate. |
   | Missing warmups, setup, reconstruction, operator checks, extra decodes and serialization | **NOT RESOLVED:** warmups are approximated; other named work remains unpriced. |
   | Rule/static-signature identity discarded | **NOT RESOLVED:** measurements initially group by rule, then collapse into solver-family rates. |
   | Production memory projection | **NOT RESOLVED:** see finding 2. |
   | Smoke completion and compatibility | **NOT RESOLVED fully:** completion/driver checks added; accepted gates, shape/source compatibility and exact GPU capacity are not required. |

   Specific weaknesses:

   - 3D rates scale entirely with \(mR'(M+2R')\). This does not bound projection, decoding, dense solves, compilation, or case-dependent adaptive iteration counts. The `K=2` smoke measurements come from lattice rules, whereas production includes large Gauss rules and `lat256`.
   - Family timing prices every A/B invocation using the mean of rollout costs. Actual panels repeatedly run the large converged-rule BE baseline; that baseline must be priced separately from each ladder subject.
   - Rule construction/reconstruction and the full-cohort operator checks are absent.
   - The 2D estimator ignores recorded calibration timings and lacks production-case runtime bounds.
   - Measurement values are not comprehensively required to be finite, positive and complete.

   A 1.5× multiplier does not establish conservativeness for these omissions. This is **not evidence that the requested limits will necessarily be exceeded**; it is evidence that the certificates do not establish the promised gate.

   **Fix:** enumerate actual A/B/warmup, setup, operator, decode and persistence work; retain rule/shape/static-signature measurements; validate extrapolations or use explicit conservative bounds. Regenerate all certificates after the ladder fix below.

2. **BLOCKER — Production residency is not certified. Item 10: NOT RESOLVED.**

   [inventory.py:127](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:127) assigns **both escalation jobs exactly the smoke maximum: 73.896 GB**, with no production adjustment. The additional-memory branch runs only for `mode == 'grid'`.

   Production changes relevant to memory include:

   - Operator checks grow from **10 to 320 states**. At `gl80`, one state-by-point intermediate grows from approximately **41 MB to 1.31 GB**.
   - `c3descB` constructs and retains tables for **both banks**; the escalation smoke contained only W1024.
   - Additional rule shapes, settings and executables are introduced.
   - The 2D projection copies the smoke’s **17.249 GB** peak despite more resident initial conditions and additional production configurations. Host-memory growth is not projected.

   Taking the maximum smoke peak also allows a missing peak in one smoke to be hidden by another smoke’s positive value.

   **Fix:** account for the maximum live allocation set, construction temporaries, batch dimensions, executables and host storage. Validate that production-sized live set on the target GPU. Chunking the operator-state calculation would make its bound simpler.

3. **BLOCKER — Staging binding remains incomplete. Item 21: NOT RESOLVED.**

   [verify_certificate():252](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:252) correctly checks the configuration hash, GPU selector, requested hours, host-memory string and the **listed** source hashes. All current listed source/configuration/smoke hashes matched my read-only recomputation.

   Remaining defects:

   - It never verifies the recorded smoke hashes or their accepted machinery gates.
   - It does not require a complete source manifest. Vendored runtime dependencies and the estimator itself are outside `SOURCES`.
   - It trusts `fits` without recomputing the time/memory inequalities or validating capacity.
   - GPU capacity is supplied by the certificate command; the smoke GPU-name prefix check does not distinguish A100 capacities.
   - Any configuration with `smoke: true` bypasses certification; its reduced workload is not validated.

   **Reproduced:** an in-memory copy of the current certificate still passed the verifier after replacing its smoke hash with `"STALE"`, setting capacity to 999 GB and setting the margin-adjusted projection to 1000 hours, while retaining the 30-hour request.

   **Fix:** require a complete dependency/input manifest, authenticate the smoke artifacts and their provenance, validate a bounded smoke schema, derive capacity from the allocation type, and recompute finite resource inequalities at staging.

4. **MAJOR — A6.2 is implemented for escalation pairs, but not for ladder candidates.**

   [c3run.py:245](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:245) correctly qualifies escalation fixed-sweep arms against their same-rule adaptive counterparts.

   However, [the ladder loop:337](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:337) runs **only `SEL`**, without `DIAG`. Its `rollouts()` still computes the old pooled-stationarity eligibility. Therefore production will not collect the adaptive trajectories needed to apply A6.2 to each candidate rule. Agreement with a converged-rule fixed-sweep trajectory cannot replace the required same-rule solver comparison.

   **Fix:** run both adaptive counterparts for every ladder rule on all 64 cases, persist A6.2 eligibility and solver distances, and require them during selection. Include these additional runs in the inventory and certificates.

   The inspected escalation implementation is otherwise correct:

   - `SEL[0]` maps to adaptive BE; `SEL[1]` maps to adaptive CN.
   - CN selection uses **K=2**.
   - Invalid adaptive counterparts make the selection axis unresolved.
   - The Gram contraction computes the physical field norm correctly: with `Lt = L.T`, the einsum evaluates coefficient differences multiplied by `L`, then normalizes by the physical initial-field norm and maximizes over evolved times.

5. **MAJOR — The 2D driver does access a test cohort’s descriptors.**

   [c2common.py:58](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2common.py:58) unconditionally calls `Q.cohort('test64')` and compares those descriptors with training cases. The vendor function regenerates the 64 descriptors from seed **20260916**.

   Thus “nothing reads or opens a test cohort” is **false under the requested strict gate**. I found no test reference-field loading or ROM scoring on that cohort in these production paths. The 3D configuration and loader use validation seed 923801.

   **Fix:** generate only the configured development/validation cohorts. Use an existing overlap certificate if a global overlap check is needed.

6. **MINOR — A6 is substantively justified, but several evidence statements need correction.**

   The smoke evidence supports replacing pooled stationarity with a verified solver-distance criterion and switching the representative CN arm to K=2:

   - W1024 BE at \(1.25\Delta t_0\): **2/40** nonstationary steps, yet coefficient distances to adaptive tight are approximately **4.85e−6 and 1.63e−6**.
   - W1024 CN K=1 at \(2.5\Delta t_0\): **13/20** nonstationary steps and worst coefficient distance **0.0149007**.
   - CN K=2 reduces that worst distance to **3.64e−5**; W512 gives **3.45e−5**. Corresponding adaptive-tight runs are verified.

   This is a reasoned amendment with an explicit replacement accuracy safeguard, provided finding 4 is fixed. It does not establish production-wide eligibility from two cases.

   Corrections needed:

   - A6.1’s BE distance “0” is incorrect.
   - A6.4’s example distances are not maxima “under every solver.” The recorded pairwise maxima are approximately **1.45e−7, 1.22e−8 and 1.76e−9**. All remain well below the bar.
   - A6.5 reverses the allocation comparison: requested **30/12 hours exceed**, rather than fall below, the certificate projections **18.77/6.73 hours**.
   - The opening `c3run.py` documentation still names CN K=1 before correcting itself later.

   **Fix:** correct these statements and retain the two-case, post-smoke qualification.

The configuration coverage otherwise matches the amended first-production scope: 2D has all six settings, five scheme-forms, seven steps, both anchor refinements and tolerance levels, FOM phases, per-setting panels and the common panel. The two 3D jobs cover all five settings, required escalation rules and conditional `gl80`, the full ten-rule ladder including `lat256`, and per-setting family panels. They use **923801 × 64** without a subset; timing seeds are explicitly **20261013** with deterministic panel offsets. The exact completed test-space sizes also lie below the estimator’s `4R′+64` proxy for these four widths.

No additional deterministic production-size crash was established. Resource safety remains unproven, and the ladder would produce insufficient qualification evidence.

**DO NOT SUBMIT.**