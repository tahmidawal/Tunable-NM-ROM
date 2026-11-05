
codex
Revision 3 fixes several issues, but numerical error reporting, reference admission, and R5 timing remain partially unresolved. Inspection and reproductions were read-only and in memory.

1. **Partially resolved — orthonormality/error calculations.**  
   `ROM/rom_core.py:47–57` adds reorthogonalization and verification; `:227–233` rejects significant negative squared errors. However, verification uses `T.T @ gram @ T`, not the evaluated basis’s mass matrix (`:94,117`). An accepted synthetic case reported defect \(1.75\times10^{-13}\), while its evaluated defect was \(4.84\times10^{-10}\), producing **\(2.20\times10^{-5}\) error for identical fields**.  
   **Required:** verify/reorthogonalize the evaluated basis, including the basis-only floor space, or consistently use actual Gram matrices.

2. **Resolved — R5 metadata-slicing crash.**  
   `ROM/rom_r5.py:103–104,133` selects time-dependent arrays before slicing. Scalar bank metadata is excluded.

3. **Partially resolved — reference admission.**  
   `REF/r1de_eval.py:95–96` enforces the exact population; `:51,63–69` checks the final settle-state link, retained hashes and snapshot grid. Remaining gaps: diagnostic cadence is only a count (`:72`), preceding settle links are not verified (`:53–55`), and production checks omit scheme/refactor settings (`:41–44`). Uncertainty binding (`:102–104`) checks agreement between supplied comparisons, but does not bind their identities and metrics to the accepted gate evidence.  
   **Required:** complete those checks. R1f remains an independent, uncompleted acceptance gate.

4. **Resolved — held-out statistics exposure.**  
   `REF/r1de_eval.py:80–87` excludes cycle statistics/history processing for Re105 and exports quality checks only.

5. **Partially resolved — quadrature qualification.**  
   `ROM/rom_qcheck.py:31–60` checks coefficient modes, bank identity, floor changes and rollout metrics; `bank_train.py:98–101,168–204` provides evaluation-only replay and parameter digests. However, trained projection files are loaded without verifying their bank hash/time binding (`rom_qcheck.py:41`), unlike R4. Floor qualification checks only the median (`:55`), leaving the maximum used by the R2 floor rule unchecked. Bind the R4 result files to the corresponding case/model/export as well.

6. **Partially resolved — timing workload.**  
   `REF/transient.py:255–259` removes extra surface-force/energy work in timing mode, and the large snapshot buffer avoids mid-loop persistence. Nevertheless, `ROM/rom_r5.py:113–117` measures FE **cold** time around `s.run()`, which includes persistence at `REF/transient.py:572,477`. ROM cold timing excludes persistence (`rom_r5.py:131–137`).  
   **Required:** exclude persistence from both cold measurements or charge it symmetrically; keep cycle postprocessing outside integration timing.

7. **Partially resolved — timing witnesses/warm-up.**  
   `ROM/rom_r5.py:60–106` improves input binding; `:124–148` adds per-invocation velocity accuracy; `:154–163` adds repeated warm-ups. Startup Newton counts are corrected in `rom_core.py:194–198`. However, timed pressure/force/probe accuracy is not witnessed—only velocity. Include those outputs’ errors if their accuracy forms part of the timed comparison.

8. **Resolved for reporting; binding remains under finding 3.**  
   `ROM/rom_r4.py:55–57,118–126,145–154` requires reference admission, computes energy balance and reports uncertainty classifications. The energy terms/signs are consistent with the stated gradient-form balance.

9. **Resolved as a prospectively declared reporting policy.**  
   `ROM/rom_r4.py:94–97,116–117` flags the initial defect and retains the initial pressure sample, with an excluding-initial-time supplement. **New minor:** normalization by `norm(BL)` (`rom_core.py:192`) becomes misleading when BL is nearly zero. Report an additional characteristic-scaled defect.

10. **Resolved at implementation level.**  
    `ROM/rom_export.py:238` and `rom_project.py:32–33` require explicit orders; `rom_r4.py:87–88,163–164` reports dimensions and peak RSS. Actual production memory evidence remains to be collected.

**New defect — failed quadrature checks may not serialize.**  
`ROM/rom_qcheck.py:55,60–65`: when the floor criterion fails, `ok` can be a NumPy boolean, causing `json.dumps` to raise. Reproduced in memory. Convert to a Python `bool` so failed qualification evidence is retained.

R1d/R1e/R1f NOT READY  
R2 NOT READY  
R3 READY  
R4 NOT READY  
R5 NOT READY
tokens used
501,451
Revision 3 fixes several issues, but numerical error reporting, reference admission, and R5 timing remain partially unresolved. Inspection and reproductions were read-only and in memory.

1. **Partially resolved — orthonormality/error calculations.**  
   `ROM/rom_core.py:47–57` adds reorthogonalization and verification; `:227–233` rejects significant negative squared errors. However, verification uses `T.T @ gram @ T`, not the evaluated basis’s mass matrix (`:94,117`). An accepted synthetic case reported defect \(1.75\times10^{-13}\), while its evaluated defect was \(4.84\times10^{-10}\), producing **\(2.20\times10^{-5}\) error for identical fields**.  
   **Required:** verify/reorthogonalize the evaluated basis, including the basis-only floor space, or consistently use actual Gram matrices.

2. **Resolved — R5 metadata-slicing crash.**  
   `ROM/rom_r5.py:103–104,133` selects time-dependent arrays before slicing. Scalar bank metadata is excluded.

3. **Partially resolved — reference admission.**  
   `REF/r1de_eval.py:95–96` enforces the exact population; `:51,63–69` checks the final settle-state link, retained hashes and snapshot grid. Remaining gaps: diagnostic cadence is only a count (`:72`), preceding settle links are not verified (`:53–55`), and production checks omit scheme/refactor settings (`:41–44`). Uncertainty binding (`:102–104`) checks agreement between supplied comparisons, but does not bind their identities and metrics to the accepted gate evidence.  
   **Required:** complete those checks. R1f remains an independent, uncompleted acceptance gate.

4. **Resolved — held-out statistics exposure.**  
   `REF/r1de_eval.py:80–87` excludes cycle statistics/history processing for Re105 and exports quality checks only.

5. **Partially resolved — quadrature qualification.**  
   `ROM/rom_qcheck.py:31–60` checks coefficient modes, bank identity, floor changes and rollout metrics; `bank_train.py:98–101,168–204` provides evaluation-only replay and parameter digests. However, trained projection files are loaded without verifying their bank hash/time binding (`rom_qcheck.py:41`), unlike R4. Floor qualification checks only the median (`:55`), leaving the maximum used by the R2 floor rule unchecked. Bind the R4 result files to the corresponding case/model/export as well.

6. **Partially resolved — timing workload.**  
   `REF/transient.py:255–259` removes extra surface-force/energy work in timing mode, and the large snapshot buffer avoids mid-loop persistence. Nevertheless, `ROM/rom_r5.py:113–117` measures FE **cold** time around `s.run()`, which includes persistence at `REF/transient.py:572,477`. ROM cold timing excludes persistence (`rom_r5.py:131–137`).  
   **Required:** exclude persistence from both cold measurements or charge it symmetrically; keep cycle postprocessing outside integration timing.

7. **Partially resolved — timing witnesses/warm-up.**  
   `ROM/rom_r5.py:60–106` improves input binding; `:124–148` adds per-invocation velocity accuracy; `:154–163` adds repeated warm-ups. Startup Newton counts are corrected in `rom_core.py:194–198`. However, timed pressure/force/probe accuracy is not witnessed—only velocity. Include those outputs’ errors if their accuracy forms part of the timed comparison.

8. **Resolved for reporting; binding remains under finding 3.**  
   `ROM/rom_r4.py:55–57,118–126,145–154` requires reference admission, computes energy balance and reports uncertainty classifications. The energy terms/signs are consistent with the stated gradient-form balance.

9. **Resolved as a prospectively declared reporting policy.**  
   `ROM/rom_r4.py:94–97,116–117` flags the initial defect and retains the initial pressure sample, with an excluding-initial-time supplement. **New minor:** normalization by `norm(BL)` (`rom_core.py:192`) becomes misleading when BL is nearly zero. Report an additional characteristic-scaled defect.

10. **Resolved at implementation level.**  
    `ROM/rom_export.py:238` and `rom_project.py:32–33` require explicit orders; `rom_r4.py:87–88,163–164` reports dimensions and peak RSS. Actual production memory evidence remains to be collected.

**New defect — failed quadrature checks may not serialize.**  
`ROM/rom_qcheck.py:55,60–65`: when the floor criterion fails, `ok` can be a NumPy boolean, causing `json.dumps` to raise. Reproduced in memory. Convert to a Python `bool` so failed qualification evidence is retained.

R1d/R1e/R1f NOT READY  
R2 NOT READY  
R3 READY  
R4 NOT READY  
R5 NOT READY
