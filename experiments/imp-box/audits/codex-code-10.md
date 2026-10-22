**Overall: NEEDS-FIX** at `e566b0f535`. Three code defects and an unsupported competitor-strengthening claim remain.

1. **A10 — NEEDS-FIX; Q1 safeguards — CORRECT.**

   The diagnostic scripts support investigating a compiled-loop failure, but **“can only strengthen the FOM” is wrong**: removing N=12 also removes the reportedly valid adaptive configurations. N=10 and N=14 need not dominate N=12 in cost versus accuracy. A11 also introduces a dense N=12 implementation that has not been shown faulty. Retain functioning N=12 configurations; scope exclusions to demonstrated failures. See [DESIGN.md:672](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/DESIGN.md:672).

   Q1 replay is real: both FOM families and all reduced families rebuild with `cert=True`, compare replay outputs at ≤1e-12, and gate qualification. See [qual.py:28](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/qual.py:28) and [finalize.py:80](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/finalize.py:80). This detects disagreement between compiled programs, not faults shared by both.

   Evidence limitation: the checked tree contains diagnostic scripts, narrative and ledger entries, but no raw diagnostic results validating the quoted GPU errors. “Hopper code-generation fault” should remain a diagnosis supported by those records, not an independently verified conclusion of this audit.

2. **Dense discretisation — CORRECT; implementation propagation — NEEDS-FIX.**

   The grid, sine-analysis weights, negative sign and both derivatives match `pscore.nonlinear`. Small f64 CPU checks on an IC and random coefficients at every enabled N=8–128 gave relative differences up to **3.84e-14**, below the 1e-12 gate. The unconditional ≤1e-14 wording is too strong.

   Ordinary ladder rows, selection, certificate replay, paired timing, components, r_x evaluation, finer-lattice/coefficient diagnostics and test replay preserve `xform`. **Adaptive tolerance extensions do not:**

   - [ladders.py:118](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/ladders.py:118) preserves the configuration but omits `xform` from its key. FFT and dense extensions collide.
   - [timing.py:254](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/timing.py:254) deduplicates by N and reconstructs an FFT-default configuration. Executing the actual function with a mocked dense finalist produced an FFT extension.

   Because selection caches use these keys, overwriting a dense row with FFT can also associate cached outputs/qualification with a different implementation. Include `xform` in extension keys, deduplication and configurations throughout.

3. **G-XFORM — NEEDS-FIX.**

   It can reject finite errors, but **fails open on NaN**. Executing the actual control block with injected outputs produced:

   | Injection | Result |
   |---|---|
   | Identical outputs | Pass |
   | 1% scaling error | Fail |
   | Dense output entirely NaN | **Pass, `max_rel=0.0`** |

   Python’s `max(gx, NaN)` retains `gx`. Explicitly reject non-finite inputs/differences before reduction. Add a NaN mutation check and cover every enabled dense size; currently only 10, 24 and 64 are checked. See [controls.py:133](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/controls.py:133).

4. **Reporting/fairness — NEEDS-FIX.**

   A11 says FFT-only smoke speedups are not reported as results, but [make_report.py:52](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/make_report.py:52) automatically substitutes `smk2-*` for missing production cells. Their S values then enter the results table and export without an explicit smoke-only qualification. Exclude them from production results or segregate and label them unmistakably. Also display `xform` in FOM descriptions.

No files changed, jobs submitted, or log entries written. Checks were CPU-only; no scratch files were created.