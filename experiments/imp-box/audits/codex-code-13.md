**Overall: NEEDS-FIX** at `1f8e20c65`. One round-12 compilation blocker remains.

- **Blocking — TRANS diagnostic still embeds the basis.** [diag.py:100](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/diag.py:100) wraps `make_shift_decoder` in ordinary `jax.jit`, capturing `Phi` through `dec`. CPU StableHLO confirms the basis becomes a literal. For W40_A3’s valid rank-96 configuration, that is **107.44 MiB**. This executes during time-stage component diagnostics. Pass decoder arrays explicitly or hoist the outermost callable. FOM component diagnostics also retain captured arrays at lines 117–128.

- **Fixed:** `Arm.build_on`, `run_cohort`, `synth_cohort`, and tensor/point diagnostic kernels. Small CPU cohort checks preserved fields and statistics exactly; the tested outer hoist had no captured executable constants.

- **G-HOIST comparison fixed; fixture incomplete.** Fields and statistics are compared, non-finite differences fail, and timing repetitions persist. Injected field NaNs, statistics NaNs, and finite statistics mismatches were rejected. However, [controls.py:175](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/controls.py:175) tests an independent lambda, so its mutation flag cannot detect regressions in the actual gate. This is a validation gap, not an additional demonstrated production correctness blocker.

No new correctness defect found in the reviewed changes. Checks were CPU-only, f64, highest precision; production compilation was not run. No files or LAB-LOG were written.