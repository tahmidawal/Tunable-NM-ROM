
codex
**DESIGN: RELEASE WITH CHANGES** — resolve the remaining prospective definitions below; label A4 as a post-failure extension.

**CODE: NOT READY** — before R1b, enforce the fixed window and consistent evidence across the convergence stencil.

References use `D` = [DESIGN-G5.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-cylinder-unsteady/experiments/cylinder-unsteady/DESIGN-G5.md), `REF` and `ROM` = the respective experiment directories. All inspection was read-only; the revised tests were inspected, not executed because they create files.

1. **Partially resolved — gate evaluator.**  
   `REF/r1_gates.py:27–98` now requires valid finite evidence, candidate-specific order stencils, startup checks and explicit R1c decisions. `REF/make_configs.py:113–123` includes startup16 at all production candidates. However, `r1_gates.py:14–15,23` still treats **100 times an algebraic residual tolerance as a field/force noise floor**, without calibration.  
   **Required:** justify that floor through a frozen tolerance-tightening control, or require order checks whenever differences exceed a demonstrated numerical floor. Also verify that stencil/startup records share the same initial-state hash and window; currently each JSON is accepted independently.

2. **Partially resolved — comparison completeness.**  
   `REF/compare.py:22–23,55–56,84–109` checks completion, matching initial specifications, snapshot cadence and force counts. But line 86 derives the expected count from arbitrary CLI endpoints: a shorter window still produces `valid=True`. It neither requires W=0.35 nor anchors `t0` to the initial state. Snapshot hashes are not checked at lines 25–27.  
   **Required before R1b:** enforce the declared start, W=0.35, 141 snapshots, exact eligible force-time grid and snapshot hashes. Bind the gate records to those checks.

3. **Partially resolved — reference uncertainty.**  
   `D:412–421` fixes cycle aggregation and introduces an unresolved classification. The linear extrapolation with factor two is a **heuristic estimate**, not an established bound. “Cycle statistics use the same recipe” leaves undefined the refinement differences for St, pressure drop and cycle maxima; `compare.py` supplies pointwise force gaps instead.  
   **Required:** define statistic-specific inputs and target mappings, and validate the propagation assumption over the full horizon—or leave the affected accuracy qualification unresolved.

4. **Resolved — trained-bank column bug.**  
   `ROM/rom_export.py:146–159,298–300`, `rom_project.py:25–38`, `rom_core.py:184–197` and `rom_r4.py:58–65` now separate bank projections, verify the evaluated-bank hash and reject mismatched saved times. The frozen-column substitution and rank≥8 indexing failure are removed.

5. **Resolved — initial-pressure calculation.**  
   `ROM/rom_core.py:158–165` solves the coupled acceleration/pressure system and records the initial continuity defect. `test_rom_core.py:90–96` checks its equations. Calling the pressure fully “consistent” remains conditional on the recorded initial state satisfying continuity.

6. **Resolved at design level — timing fairness.**  
   `D:426–434` defines a common duration, output cadence, cold/warm accounting, alternating repetitions and retained witnesses. The explicit limitation to **production-reference speed** adequately replaces the accuracy-matched-FOM claim. Implementation and output-cost equivalence still require review before R5.

7. **Partially resolved — quadrature qualification.**  
   `D:435–438` adds numerical thresholds and an unresolved outcome. However, independently orthonormalizing re-exports can rotate/sign-flip bases, making direct operator-array differences meaningless. POD8/POD24 rankings also lack the operator check specified only for POD16.  
   **Required:** compare operators in a common basis, or transport coordinates explicitly; cover every ranked arm and define handling of near-zero denominators.

8. **Resolved — embedding witness.**  
   `REF/transient.py:382–392` measures high-order velocity/pressure L2 transfer errors and stops above the frozen bound. `D:439–441` fixes both R1c arms to the original P3 S*. This resolves the missing witness; its successful execution remains pending.

9. **Resolved prospectively — retention and budget.**  
   `D:442–446` requires two checksum-matching snapshot copies before R1f and fixes a 25-hour ceiling with a pre-submission extension trigger. The collector alone still does not establish retention; R1f must verify the promised copies.

10. **Resolved, with one implementation caveat — extraction and phase support.**  
    `sources/extract_featflow.py:18,44–58` uses the shared procedure; all three `cycles.py` copies are byte-identical. The saved compilation confirms Table 4 and the CL-maximum pressure convention. `ROM/rom_r4.py:109–120` fixes phase-search support.  
    **Caveat:** `REF/cycles.py:65` uses endpoint-clamping `np.interp`; reject any 1996 pressure evaluation outside the available history instead of silently extrapolating.

**A4:** `D:383–392` explicitly raises the cap after observing two failures. This changes a stopping gate after data. Retaining failures is necessary, but subsequent success qualifies the **amended six-second protocol**, never the original four-second protocol.

**New blockers:** none beyond the unresolved R1b controls above. The quadrature coordinate issue and pressure-interpolation caveat must be closed before their dependent stages.
tokens used
172,939
**DESIGN: RELEASE WITH CHANGES** — resolve the remaining prospective definitions below; label A4 as a post-failure extension.

**CODE: NOT READY** — before R1b, enforce the fixed window and consistent evidence across the convergence stencil.

References use `D` = [DESIGN-G5.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-cylinder-unsteady/experiments/cylinder-unsteady/DESIGN-G5.md), `REF` and `ROM` = the respective experiment directories. All inspection was read-only; the revised tests were inspected, not executed because they create files.

1. **Partially resolved — gate evaluator.**  
   `REF/r1_gates.py:27–98` now requires valid finite evidence, candidate-specific order stencils, startup checks and explicit R1c decisions. `REF/make_configs.py:113–123` includes startup16 at all production candidates. However, `r1_gates.py:14–15,23` still treats **100 times an algebraic residual tolerance as a field/force noise floor**, without calibration.  
   **Required:** justify that floor through a frozen tolerance-tightening control, or require order checks whenever differences exceed a demonstrated numerical floor. Also verify that stencil/startup records share the same initial-state hash and window; currently each JSON is accepted independently.

2. **Partially resolved — comparison completeness.**  
   `REF/compare.py:22–23,55–56,84–109` checks completion, matching initial specifications, snapshot cadence and force counts. But line 86 derives the expected count from arbitrary CLI endpoints: a shorter window still produces `valid=True`. It neither requires W=0.35 nor anchors `t0` to the initial state. Snapshot hashes are not checked at lines 25–27.  
   **Required before R1b:** enforce the declared start, W=0.35, 141 snapshots, exact eligible force-time grid and snapshot hashes. Bind the gate records to those checks.

3. **Partially resolved — reference uncertainty.**  
   `D:412–421` fixes cycle aggregation and introduces an unresolved classification. The linear extrapolation with factor two is a **heuristic estimate**, not an established bound. “Cycle statistics use the same recipe” leaves undefined the refinement differences for St, pressure drop and cycle maxima; `compare.py` supplies pointwise force gaps instead.  
   **Required:** define statistic-specific inputs and target mappings, and validate the propagation assumption over the full horizon—or leave the affected accuracy qualification unresolved.

4. **Resolved — trained-bank column bug.**  
   `ROM/rom_export.py:146–159,298–300`, `rom_project.py:25–38`, `rom_core.py:184–197` and `rom_r4.py:58–65` now separate bank projections, verify the evaluated-bank hash and reject mismatched saved times. The frozen-column substitution and rank≥8 indexing failure are removed.

5. **Resolved — initial-pressure calculation.**  
   `ROM/rom_core.py:158–165` solves the coupled acceleration/pressure system and records the initial continuity defect. `test_rom_core.py:90–96` checks its equations. Calling the pressure fully “consistent” remains conditional on the recorded initial state satisfying continuity.

6. **Resolved at design level — timing fairness.**  
   `D:426–434` defines a common duration, output cadence, cold/warm accounting, alternating repetitions and retained witnesses. The explicit limitation to **production-reference speed** adequately replaces the accuracy-matched-FOM claim. Implementation and output-cost equivalence still require review before R5.

7. **Partially resolved — quadrature qualification.**  
   `D:435–438` adds numerical thresholds and an unresolved outcome. However, independently orthonormalizing re-exports can rotate/sign-flip bases, making direct operator-array differences meaningless. POD8/POD24 rankings also lack the operator check specified only for POD16.  
   **Required:** compare operators in a common basis, or transport coordinates explicitly; cover every ranked arm and define handling of near-zero denominators.

8. **Resolved — embedding witness.**  
   `REF/transient.py:382–392` measures high-order velocity/pressure L2 transfer errors and stops above the frozen bound. `D:439–441` fixes both R1c arms to the original P3 S*. This resolves the missing witness; its successful execution remains pending.

9. **Resolved prospectively — retention and budget.**  
   `D:442–446` requires two checksum-matching snapshot copies before R1f and fixes a 25-hour ceiling with a pre-submission extension trigger. The collector alone still does not establish retention; R1f must verify the promised copies.

10. **Resolved, with one implementation caveat — extraction and phase support.**  
    `sources/extract_featflow.py:18,44–58` uses the shared procedure; all three `cycles.py` copies are byte-identical. The saved compilation confirms Table 4 and the CL-maximum pressure convention. `ROM/rom_r4.py:109–120` fixes phase-search support.  
    **Caveat:** `REF/cycles.py:65` uses endpoint-clamping `np.interp`; reject any 1996 pressure evaluation outside the available history instead of silently extrapolating.

**A4:** `D:383–392` explicitly raises the cap after observing two failures. This changes a stopping gate after data. Retaining failures is necessary, but subsequent success qualifies the **amended six-second protocol**, never the original four-second protocol.

**New blockers:** none beyond the unresolved R1b controls above. The quadrature coordinate issue and pressure-interpolation caveat must be closed before their dependent stages.
