**Overall: NEEDS-FIX.** Audited `17a9698b8..cea76ac15`. Two blockers: remaining compilation paths embed large constants, and G-HOIST can falsely pass non-finite comparisons.

1. **CORRECT — `Hoisted` function semantics.**  
   [hoist.py:20](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/hoist.py:20) passes jaxpr constants in order, reconstructs the output tree, and delegates `_cache_size()` to the actual executable wrapper. Builders restore `info`. Nested TRANS hoists the inner query’s constants again correctly. CPU probes preserved trees, metadata and statistics; maximum field differences were `3.19e-16` for FOM and `6.94e-18` for ROM/TRANS. `vmap` preserves numerical behavior. “Same function” holds within floating-point roundoff, not bitwise equality.

2. **WRONG — large embedded constants remain on relevant paths.**
   - [timing.py:95](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/timing.py:95) wraps the hoisted TRANS inner query in an ordinary `jax.jit`, re-embedding its constants. This runs during qualification and time-stage diagnostics.
   - [cell.py:91](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cell.py:91) does the same with `jax.jit(jax.vmap(q))`.
   - [diag.py:208](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/diag.py:208) still embeds full nonlinear tensors; point-evaluation and component diagnostics also close over large arrays. These execute within the time stage.

   A CPU StableHLO probe confirmed that a hoisted array disappears from executable constants but returns under either ordinary outer JIT wrapper. Hoist the outermost callable or pass arrays explicitly through it; fix diagnostic kernels likewise.

3. **CORRECT — primary query timing contract.**  
   Hoisting uses an abstract descriptor, so it cannot precompute case-dependent work. IC projection, viscosity-dependent coefficients, evolution and decoding remain inside the compiled program. Warm-up precedes paired measurements; `_cache_size()` measures the wrapper actually invoked, and nonzero cache growth invalidates timing. No new compilation inside the primary measured calls was identified. The remaining compilation problems in item 2 affect stage completion, rather than moving case-dependent work outside measured queries.

4. **NEEDS-FIX — G-HOIST can falsely pass.**  
   [controls.py:158](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/controls.py:158) compares only field outputs; statistics are ignored. More critically, Python’s `max` can ignore a later NaN: the exact predicate accepts `max([0.0, NaN, 0.0]) <= 1e-12`. Thus a broken dense/adaptive/ROM/TRANS comparison can pass after a finite first entry.

   It covers all seven families explicitly registered by A16 and rejects finite discrepancies above threshold. Require every comparison to be finite and within tolerance, compare statistics too, and add a failing mutation. Its timing ratios also discard repetition arrays because `tmed` receives no label; retain those before using ratios for cross-cell interpretation.

5. **CORRECT — source-equivalence check fails closed.**  
   [teststage.py:140](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/teststage.py:140) checks the union of source names and requires exact registered old/new hashes plus committed equivalence-file content. Extracted-function probes accepted registered transitions and rejected absent, uncommitted and mismatched records. Adding `hoist.py` requires an explicit `null` predecessor. **The equivalence file is currently absent**, so older freezes will be refused until it is committed.

6. **CORRECT — restart restores the required archived state.**  
   [time.sh.in:4](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/bodies/time.sh.in:4) copies hidden markers and artifacts, verifies checksums, removes time/control completion markers and selection outputs, and runs only `time`, then `controls`. Restore failure prevents execution. Both local archive inventories contain the necessary references, representations and ladder records; omitted training arrays are unnecessary for these stages. Remote archive contents were not independently rechecked.

No repository or LAB-LOG writes, job submissions, or GPU checks were performed.