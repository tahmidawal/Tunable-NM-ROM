**NOT READY to stage the C2 smoke.** Audited HEAD `cc058e9dabe8aaeea9b058867b80e0a0b4367b41`. No files changed, no cluster activity, no banks built.

Paths below are relative to `experiments/jcp-fom2d/`.

| File | Verdict | Principal finding |
|---|---|---|
| `c2time.py` | **WRONG** | T-id bypasses identity checking; required executable release is absent. |
| `c2analyze.py` | **NEEDS-FIX** | Empty/deferred cells disappear; combined-workload pilots do not affect panel acceptance. |
| `make_c2configs.py` | **NEEDS-FIX** | Empty groups and selected-comparator metadata are not carried through to analysis. |
| `make_smkc2.py` | **NEEDS-FIX** | Requests combined pilots without an acceptance contract for their results. |
| `targets.py` (`front`) | **WRONG** | K2 completeness detection does not match the producer schema. |
| `p1core.py` | **CORRECT** | Byte-identical to P1 and qualify; current query construction agrees. |
| `p1common.py` | **CORRECT** | Byte-identical to P1 and qualify; current forms, budget and tolerances agree. |
| `fomrun2.py` callable path | **CORRECT** | FD includes solve and output-map dispatches; PS uses its compiled full-field program. Cache lifetime needs management by C2. |

Ranked fixes:

1. **P1 — T-id is a tautology, not a test of identity enforcement.**  
   `c2time.py:242–244,354–358` copies the correct reference, then bypasses `check()`, forces `hb=False`, and directly inserts the alias into `VETO`. Consequently, T-id still fires if the real hash comparison or veto propagation is broken.

   Corrupt the **alias’s reference**, execute the ordinary identity-checking path, and verify subject-level exclusion through the ordinary analysis path. Retain the control-only alias; that separation correctly protects the production subject.

2. **P1 — K2 can falsely declare a complete universe.**  
   `targets.py:134–141` searches for keys containing `numerically_qualified`. The actual producer, qualify’s `analyze.py:419–426`, stores rejected front candidates using **`acc_label`, `timing_label`, `usable`, and `complete38`**.

   I executed the reader with a producer-shaped entry having both numerical labels true and `usable=False`. It omitted that entry and emitted **`complete: true`**. This is an ordinary producer outcome, not forgery. If the retained members all fail, `c2analyze.py:149–152` can then incorrectly declare K2 fired.

   Read the actual labels and require a complete qualification inventory. Missing numerically qualified entries must keep K2 inconclusive. Their omission also matters when defining cross-mesh qualified populations.

3. **P1 — The registered memory-release policy is not implemented.**  
   `c2time.py:129–135` retains prepared inputs separately for every candidate and case. `fomrun2.py:105–135` retains every solver, output-map and PS executable. Nothing releases these caches during C2; `c2time.py:366` deletes only pilot input/output variables.

   At FD \(L=4096\), six inputs occupy **804,913,200 bytes per candidate**, even when several candidates share the same IC discretisation. These coexist with the approximately 68.7 GB bank, ROM settings, outputs and executable workspaces.

   Implement input sharing and an explicit executable/input release lifecycle consistent with A1.6, including pilots. The present code does not establish that the registered workload fits; I am **not claiming an observed OOM**.

4. **P2 — The combined-workload smoke can be labelled valid despite failed pilot evidence.**  
   `make_smkc2.py:37` requests the largest FD/PS pilots, but `c2time.py:360–372` checks only a strided subset for finiteness and unconditionally finishes. `c2analyze.py:98–101` ignores pilots entirely.

   Require every requested pilot to be present and valid, persist adequate full-output/solver evidence and memory measurements, and include those conditions in explicit smoke acceptance. A completed process is insufficient evidence that the combined workload passed.

5. **P2 — Empty and deferred groups disappear from the reported results.**  
   `make_c2configs.py:74–76,85–87` stores these only in `ent['empty']`. `c2analyze.py:112,132` iterates only `groups`, so H-comp omits those cells. Crossover can report **“no qualified mesh”** when the identity was qualified but its comparator was absent or deferred.

   Emit explicit cells for no match, deferred, population-too-small and invalid-selection outcomes. Preserve those statuses in crossover sequences. Also distinguish all-candidates-vetoed from ordinary “no candidate” at `c2time.py:300`.

6. **P2 — Selected-comparator annotations are incomplete.**  
   `make_c2configs.py:77–80` supplies edge information only when a candidate coincides with one of three C1 winners. Union groups receive no candidate metadata at `:89`. C2 may legitimately select another candidate. PS anchors and boundary labels are not propagated for that selected comparator.

   Attach annotations to every candidate and report the anchor for the **frozen C2 selection**, including adjacent-step validity/unavailability. Do not substitute the C1 winner’s annotation.

What checked out:

- **ROM query parity:** C2 calls the non-recording fused init–evolve–full-field-decode query. Current FRONT schedules use `nstep=800` in both paths; budget is 600, and form-specific production tolerances agree with `qacc.run`. GAL data is constructed when needed. Checkpoints also match. The fixed C2 buffer would need revisiting for future schedules requiring more than 800 steps, but that is not a current FRONT defect.
- **Stage mechanics:** arithmetic flanking-A normalization, two drift retries, persisted attempts, per-family top-three pooling, stage-1 common populations, tie-breaking, post-stage-1 veto filtering, frozen selection before fresh stage 2, and no stage-2 reselection are implemented correctly for the generated six-case configs.
- **A as a target:** generated identity labels give it its own stage-2 subject distinct from baseline label `A`.
- **Gates:** G-RR requires every target’s timing-case errors and ROM validity; A-valid is separate; missing G-FR timing-case errors fail closed. Both builders use the six dev6 case IDs and production ladder errors, including the recorded recovery key.
- **Controls:** T-dbl uses two separately dispatched, synchronized executions. T-fault does likewise inside A2, and analysis requires measured separation outside the drift band.
- **Estimator:** per-case medians, at least two valid repetitions, at least five common cases, ratio of case medians, and the paired 10,000-draw bootstrap are correct. CPU fixtures confirmed these thresholds and an exact constant-ratio interval.
- **Verdicts:** H-comp’s descriptive-bootstrap wording and K2’s “not shown to fire” branch follow A3.1/A5.1, provided the universe is correct.
- **Brackets:** 1×/2× classification, reversal detection and point-only handling work for available valid cells. Missing-cell reporting needs fix 5.
- **FOM callable:** a small CPU FD repeat produced bitwise-identical `(6,257,257)` outputs. The smoke’s PS \(N=32,\Delta t=0.01\) produced finite outputs and running maxima on all six dev6 cases. These checks do not establish GPU determinism.

**NOTED — forgery-only:** `c2time.py:153–164` hashes ROM coefficients/step evidence or the FOM lattice plus three full-field reductions. This is not a SHA256 of every full-field byte; deliberately constructed off-lattice substitutions can preserve it. Likewise, analysis records but does not independently verify the frozen-selection artifact/hash (`c2analyze.py:106,113`). I found no demonstrated honest-run identity failure from these mechanisms.

Fix **1, 3 and 4 before staging the smoke**; fix the remaining items before accepting production C2 conclusions.