
codex
The numerical fixes are effective. One quadrature-binding issue remains. These are code-readiness verdicts; upstream admission and the later R1f audit remain required.

- **Item 1 — resolved.** `ROM/rom_core.py:94–128` verifies both evaluated spaces; `:273–285,295` uses their actual Gram matrices for floors/errors. In-memory near-dependency checks yielded evaluated defects below \(8.7\times10^{-14}\) and identical-field errors below \(3.8\times10^{-8}\), consistent with norm-expansion roundoff.

- **Item 3 — resolved within evaluator scope.** `REF/r1de_eval.py:41–48,55–68,83–88,114–120` now checks production settings, settle-chain links, exact diagnostic times and agreement with the gate file’s accepted metrics. Held-out sealing remains intact.

- **Item 5 — partially resolved.** `ROM/rom_qcheck.py:42–46` binds trained projections; `:65,70–73` checks median/max floors and uses serializable Python booleans. Case/export binding is checked at `:48–51`. **However, it never compares the R4 summary’s `trained_bank_sha256` with the supplied trained evaluation file.** Consequently, rollout metrics from another trained bank on the same export/case can enter qualification. Compare those hashes on each side before accepting the R4 metrics. `rom_r4.py:82–83` already supplies the needed identity.

- **Item 6 — resolved.** `ROM/rom_r5.py:124–133` constructs FE cold cost from construction, setup and stepping, excluding persistence. `REF/transient.py:465,472` supplies setup time; the vendored copy is byte-identical.

- **Item 7 — resolved for the stated witnesses.** `ROM/rom_r5.py:112–119,137–164` adds force, pressure-drop and ROM pressure-field errors alongside velocity errors, retaining per-invocation measurements.

- **Item 9 — resolved.** `ROM/rom_core.py:209–214` adds the characteristic-scaled initial defect while preserving the original diagnostic.

- **Previous qcheck boolean defect — resolved**, `rom_qcheck.py:70–73`; failure serialization passed an in-memory check.

No new defect was identified in the inspected changes. The remaining trained-bank/R4 binding gap is part of item 5.

R1d/R1e (evaluator) READY  
R2 READY  
R3 READY  
R4 NOT READY  
R5 READY
tokens used
526,769
The numerical fixes are effective. One quadrature-binding issue remains. These are code-readiness verdicts; upstream admission and the later R1f audit remain required.

- **Item 1 — resolved.** `ROM/rom_core.py:94–128` verifies both evaluated spaces; `:273–285,295` uses their actual Gram matrices for floors/errors. In-memory near-dependency checks yielded evaluated defects below \(8.7\times10^{-14}\) and identical-field errors below \(3.8\times10^{-8}\), consistent with norm-expansion roundoff.

- **Item 3 — resolved within evaluator scope.** `REF/r1de_eval.py:41–48,55–68,83–88,114–120` now checks production settings, settle-chain links, exact diagnostic times and agreement with the gate file’s accepted metrics. Held-out sealing remains intact.

- **Item 5 — partially resolved.** `ROM/rom_qcheck.py:42–46` binds trained projections; `:65,70–73` checks median/max floors and uses serializable Python booleans. Case/export binding is checked at `:48–51`. **However, it never compares the R4 summary’s `trained_bank_sha256` with the supplied trained evaluation file.** Consequently, rollout metrics from another trained bank on the same export/case can enter qualification. Compare those hashes on each side before accepting the R4 metrics. `rom_r4.py:82–83` already supplies the needed identity.

- **Item 6 — resolved.** `ROM/rom_r5.py:124–133` constructs FE cold cost from construction, setup and stepping, excluding persistence. `REF/transient.py:465,472` supplies setup time; the vendored copy is byte-identical.

- **Item 7 — resolved for the stated witnesses.** `ROM/rom_r5.py:112–119,137–164` adds force, pressure-drop and ROM pressure-field errors alongside velocity errors, retaining per-invocation measurements.

- **Item 9 — resolved.** `ROM/rom_core.py:209–214` adds the characteristic-scaled initial defect while preserving the original diagnostic.

- **Previous qcheck boolean defect — resolved**, `rom_qcheck.py:70–73`; failure serialization passed an in-memory check.

No new defect was identified in the inspected changes. The remaining trained-bank/R4 binding gap is part of item 5.

R1d/R1e (evaluator) READY  
R2 READY  
R3 READY  
R4 NOT READY  
R5 READY
