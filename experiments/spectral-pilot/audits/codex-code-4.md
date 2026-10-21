**One MAJOR regression still prevents trust.** The other three round-3 MAJOR findings are fixed.

| Round-3 MAJOR finding | Disposition |
|---|---|
| 1. Screening pruning | **STILL-WRONG — MAJOR.** The original fixture now passes, but the replacement can discard an already fully timed winner; details below. |
| 2. Missing controls pass qualification | **FIXED.** In-memory `analyze.main()` fixtures confirmed that absent, partial, and failed required controls suppress qualification and speed ratios in both parts. An explicitly unavailable SW4 remains permitted as designed. |
| 3. B allowance and SW2 | **FIXED.** Missing/nonfinite space or time evidence returns `U=None`. The legitimate `(0.016, 0.004, 0.0002)` fallback retains `U=1e-6`. Fixed SW2 fixtures pass independently of observed differences. |
| 4. Invalid crossover and mixed-generation timing | **FIXED.** Invalid screens suppress `r_cross`; mismatched screen timestamps and invalid final timing suppress speed ratios in both parts. Verified through `analyze.main()` fixtures. |

**MAJOR — Screening can prune its own fully timed leader.**  
In [tutil.py:42](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/tutil.py:42), leaders receive full timing, but lines 44–46 apply single-call pruning to those leaders too and discard their measured arrays.

A deterministic fixture reproduced:

| Candidate | Single-call time | Subsequent time | Returned outcome |
|---|---:|---:|---|
| A | 10 | 1 | Pruned despite 15 full measurements |
| B | 11 | 2 | Pruned without full measurements |

A establishes the fully timed threshold of `3 × 1`, then fails that threshold against its own earlier single call. The returned candidate set is empty. Both timing consumers consequently select no finalist and can report “a method did not reach tau (censored),” although an accuracy-qualified, fully measured candidate exists.

Retain fully timed leaders unconditionally. Preserve their initial measurement arrays as well: those arrays determine pruning but currently never reach the saved screening record. This regression is not covered by the disclosed single-call pruning limitation.

**MINOR — Secondary native timing still discards repetitions.**  
A5 addendum 2 now discloses FOM-then-ROM batching: **ACCEPTED-AS-DISCLOSED-LIMITATION**. However, [time_a.py:297](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/time_a.py:297) still saves only native medians and their ratio. The earlier missing-repetition finding remains.

Checks used the specified interpreter, CPU JAX, disabled bytecode writes, and in-memory fixtures. I made no filesystem changes, including no lab-log append. `runs/A/A1/fom.json` changed externally during the review.

OVERALL: NEEDS-FIX