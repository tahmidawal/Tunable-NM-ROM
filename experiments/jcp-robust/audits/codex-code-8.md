**Overall verdict: NEEDS-FIX. Acceptance code should not yet be used to accept smoke outputs or release production.**

Reviewed the code changes in `231a58b84`; HEAD was `ead0af79e` at the final check. No files changed, scratch created, jobs submitted, or SSH/cluster commands executed. No smoke result contents were inspected. CPU probes used disabled bytecode writing.

Paths below are relative to `experiments/jcp-robust/`.

1. **Round 7 #2 — RESOLVED — CORRECT — MINOR.**

   Shape failures now raise `MalformedEvidence`, rather than becoming ordinary numerical failures:
   
   - `select3d.py:71–74,141–152`: malformed candidate evidence becomes unavailable and stops selection.
   - `e2analyze3d.py:98–100,131–138`: malformed arrays invalidate overall integrity.
   - `e2analyze3d.py:147–153`: agreement checks now cover tight, tighter and successor distances, including their lengths.

   **Fix:** None for the original finding. The bank-binding defect in finding 5 is separate.

2. **Round 7 #5 — NOT RESOLVED completely — NEEDS-FIX — BLOCKING.**

   The original NaN-coefficient bypass is closed by `smoke_accept.py:261–262`. However, decoded distances still need not be finite for the evidence gate to pass.

   `smoke_accept.py:264–266` converts non-finite decoded distances to infinity. `rec_ok` subsequently checks certificate coverage only (`:268–271`), and the mandatory evidence check accepts any nonempty `d_rec` (`:272–273`). Qualification then returns a defined `own=False`, which passes `:282`.

   **CPU counterexample:** correctly shaped, finite coefficient arrays containing `1e200`, decoded through an identity matrix, overflowed the norm to infinity. The actual qualification function returned:
   ```
   (None, False, 'target not certified')
   ```
   Thus the evidence and “label defined” predicates can both pass despite non-finite recomputation and all validity flags being true. This is an extreme synthetic integrity test, not a claim about observed smoke data.

   **Fix:** Require finite decoder values, positive finite normalizers, and finite recomputed distances in `rec_ok`. Preserve acceptance of **finite**, numerically unqualified pilot outcomes.

3. **Round 7 #6 — RESOLVED — CORRECT — MINOR.**

   `e1analyze3d.py:74–77` raises on malformed coefficient shapes. The handler at `:247–249` records `integrity.ok=False`; `smoke_accept.py:116` requires that integrity gate. Distance agreement no longer skips non-finite recomputations (`e1analyze3d.py:132–137`).

   **Fix:** None for the original malformed-shape bypass. Missing downstream fields can make the wrapper terminate instead of emitting a complete rejection report; that is **MINOR**, since it does not accept the smoke.

4. **Round 7 #7 — NOT RESOLVED completely — NEEDS-FIX — BLOCKING.**

   The added comparison at `e2analyze.py:208–214` correctly rejects jobs whose **declared configuration bank** differs from the requested checkpoint/rotation, including the extra 1024² job.

   But the runtime identity remains unbound. `records()` passes recorded hashes into `e2bank.use()` (`e2analyze.py:30–31`), yet `e2bank.py:29–35` uses those hashes only for `SMOKE-RUNTIME`. For production banks, it verifies local files against configuration hashes while ignoring conflicting recorded checkpoint/rotation hashes. The incumbent branch performs neither comparison.

   **CPU counterexample:** the actual extracted loader accepted configured/local seed-1 hashes alongside recorded seed-2 hashes.

   Additionally, `e2analyze.py:182` loads the saved selection without checking its source job against `--sel`; bank checking the accuracy jobs does not bind that selection artifact.

   **Fix:** Require recorded hashes, configured hashes, local-file hashes and the requested bank identity to agree. Bind the saved selection to its source job and bank before using its rule choices.

5. **NEW — E2 3D can combine different seeds under one requested bank — WRONG — BLOCKING.**

   `e2analyze3d.py:75–91` loads `select3d-<b>.json`, merges the two supplied jobs, and initializes integrity without checking their `bank_seed` or checkpoint identity against `b` or the selection.

   `e2score3.install_union()` checks conflicts only for repeated setting names (`e2score3.py:20–29`). The registered job halves contain disjoint widths: W256/W512 versus W768/W1024. Consequently, a seed-1 first half and seed-2 second half can pass the union check and produce a cross-seed H-acc-r transition labelled as one bank.

   `select3d.py:44–59` likewise uses `b` as an output label without binding input jobs to it. The later timing comparison (`e2analyze3d.py:189–191`) does not close the accuracy/selection hole.

   **Fix:** Resolve the expected bank from `inputs/banks3d/s<b>/BANK.json`; require every selection, accuracy, scoring and timing input to match that identity and seed. Reject mixed-bank halves before merging.

6. **NEW — Reduced or duplicated seed populations can produce “seed-robust” — WRONG — BLOCKING.**

   A2.7 requires agreement among all three new seeds. `e2verdicts.py:95` accepts an arbitrary `--seeds` list, while `lab3()` requires only agreement among however many labels it receives (`:28–35`).

   **CPU counterexamples:** both `lab3(['pass'])` and `lab3(['pass','pass'])` returned `seed-robust`. Duplicate seed arguments can also repeat one bank’s label, because results are stored by seed but labels are reconstructed from the original list (`:98,106`).

   This matters when the registered budget deferral removes seed 3: its absence must not silently reduce the agreement requirement.

   **Fix:** Enforce distinct registered seed identities. Represent deferred/missing seeds as unavailable; fewer than three cannot establish seed robustness. Two genuinely distinct disagreeing seeds may still establish seed dependence.

7. **Smoke-control wiring and resource contract — CORRECT within the limits below — MINOR.**

   The explicit E1 controls, reproduction and target-certification gates remain required (`smoke_accept.py:86–125`). The resource contract checks committed job specifications, regenerated inventories and reservation identity (`:35–75`).

   E5 requires complete registered populations, doubled-work controls, identities, equivalence and reproduction (`e5analyze.py:123–163`), then explicitly requires all three injected/null fixtures and P-cross gates (`:722–749`). G-E5.2 correctly remains descriptive. The pure E5 drift-control and inventory tests passed.

   No additional blocking certificate/collection defect was established in this pass. Filesystem-writing E5 tests and smoke wrappers were not executed.

   **Fix:** None for these checks.

The smoke assessment is based on source inspection and synthetic probes, **not acceptance of actual outputs**:

| Smoke | Can complete, passing evidence reach ACCEPT? | Does every required failed/missing control or gate force rejection? |
|---|---|---|
| **smk1b** | Yes; no positive-path blocker identified. | Explicit controls/gates are required; round 7’s malformed-shape bypass is closed. |
| **smk2** | Yes, including A6.3’s legitimate “verdict not reached” path. | No additional smoke-specific control bypass established. Production provenance remains blocked by finding 4. |
| **smk3** | Yes, conditional on fetched bank, replay archive and accounting evidence. | **No:** non-finite decoded pilot evidence can pass; finding 2. |
| **smk4** | Yes. | Explicit E5 controls/gates are required, including the CPU drift test. |
| **smk5** | Yes. | Explicit H200 E5 controls/gates are required; the shared CPU drift test is exercised by smk4. |

**Four blocking findings remain: 2, 4, 5 and 6.**