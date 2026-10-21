**The fixes improve the pilot, but qualification and candidate selection still have MAJOR defects.** This review was read-only. Small CPU checks used the specified interpreter with bytecode writes disabled; fixtures stayed in memory. No files were created, modified, or deleted.

The remaining findings are:

1. **MAJOR — The new screening changes the disclosed pruning rule and can discard the winner.**  
   [tutil.py:23](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/tutil.py:23) compares each candidate’s single measurement against the **minimum single measurement**, whereas A5 specifies comparison against the best **fully timed** query.

   A deterministic timing fixture reproduced:

   | Candidate | Single measurement | Subsequent measurements | Outcome |
   |---|---:|---:|---|
   | A | 1 | 10 | Retained |
   | B | 4 | 4 | Pruned |

   The selected candidate costs 10 despite an available candidate costing 4. Under the disclosed rule, B would survive comparison with A’s fully measured cost. A stable sentinel does not detect this candidate-specific anomaly.

   Call-level interleaving itself is **FIXED**, including synchronization and retained repetition arrays. The defect is the new pruning threshold. Restore the disclosed rule or explicitly amend and qualify this additional selection bias.

2. **MAJOR — Missing controls pass qualification.**  
   [analyze.py:80](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/analyze.py:80) filters out absent ranks before applying `all()`. An empty gate dictionary therefore passes both G4 and SW4. Part B’s `all(...)` at line 166 has the same empty-set behavior.

   This is reachable through normal checkpoint output: both ROM producers write results before reaching the control ranks. My actual-`main()` fixture with `gates={}` retained finite `U`, feasible sizes, and `S=2`.

   Explicit failed controls now block analysis correctly. **Absent required controls must also make qualification unresolved**, with an explicit policy for effective ranks below the prescribed control ranks.

3. **MAJOR — B’s allowance repair mishandles unavailable space evidence and its SW2 can reject a legitimate fallback.**  
   [bgate.py:19](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/bgate.py:19) correctly validates both time components before taking their maximum. However:

   - Space differences `(0.016, 0.004, NaN)` and finite time differences produce `U=1e-6`, rather than unavailable qualification under A4.
   - Missing time-component keys raise `KeyError`, despite the advertised “missing or non-finite” handling.
   - For finite differences `(0.016, 0.004, 0.0002)`, the original order gate fails, legitimately permitting numerical-reference fallback. Multiplying the finest difference by five **repairs** the order pattern, so `SW2_fired=False`; `fom_b.py` then removes the fallback allowance.

   SW2 must use a fixture guaranteed to violate the predicate. Multiplying an arbitrary observed difference by five is not such a fixture. Distinguish unavailable evidence from a finite, evaluated order-gate failure.

4. **MAJOR — Timing enforcement remains incomplete for crossover readouts and mixed-generation files.**  
   [analyze.py:109](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/analyze.py:109) computes `r_cross` whenever a screen exists, regardless of its validity. A fixture with an explicitly invalid screen still emitted:

   ```text
   r_cross = 5.656854249492379, status = "ok"
   ```

   The main speed ratio is correctly suppressed when the final file’s copied `screen_valid` flag is false. However, analysis never checks the currently loaded screen’s flag against that final file. An older valid final file plus a newer invalid screen still emitted `S=2` and used the newer cost model.

   Gate timing-derived crossover readouts too, and associate final timing with the exact screen it selected from.

Round-2 dispositions, following its 13-row inventory:

| # | Area | Round-3 disposition |
|---|---|---|
| 1 | `common.py` | **FIXED:** IC correction retained. **STILL-WRONG — MINOR:** `highest` precision is recorded, not enforced. |
| 2 | `romlib.py` | **FIXED:** rank cutoff and central dynamic-operator arguments retained. **STILL-WRONG — MINOR:** the categorical “no captured large constants” claim remains too broad. |
| 3 | `gen_a.py` | **FIXED:** analysis now carries the basis label. **STILL-WRONG — MINOR:** raw `accepted` and `tau_min_eligible` remain misleading; A timing files still omit the basis label. **ACCEPTED-AS-DISCLOSED-LIMITATION:** in-memory snapshots and omitted projections/benchmarks. |
| 4 | `pad_gate.py` | **FIXED:** downstream padding enforcement retained. **ACCEPTED-AS-DISCLOSED-LIMITATION:** missing projection and stage-time coverage, now qualified by the addendum. |
| 5 | `fom_a.py` | **FIXED:** spectral-floor D2 is now asserted in analysis for matching stored floors. **STILL-WRONG — MINOR:** missing reference keys raise instead of producing unresolved status. **ACCEPTED-AS-DISCLOSED-LIMITATION:** size-stopping heuristic. |
| 6 | `rom_a.py` | **FIXED:** explicit G4/SW4 failures block analysis; training label reaches analysis. **STILL-WRONG — MAJOR:** missing controls pass, finding 2. **ACCEPTED-AS-DISCLOSED-LIMITATION:** floors-only diagnostics. |
| 7 | `score.py` | **FIXED:** unavailable allowance takes precedence. |
| 8 | `time_a.py` | **FIXED:** call-level interleaving and main speed-ratio validity checks. **STILL-WRONG — MAJOR:** revised pruning and incomplete timing enforcement, findings 1 and 4. |
| 9 | `fd_l.py` | **FIXED:** low-resolution lattice correction retained. **ACCEPTED-AS-DISCLOSED-LIMITATION:** amended seeds. |
| 10 | `gen_b.py` / `bgate.py` | **FIXED downstream:** both time components are validated. **STILL-WRONG — MAJOR:** unavailable space evidence and SW2 behavior, finding 3. The old generator’s unsafe summary calculation remains. |
| 11 | `fom_b.py` | **FIXED:** prior RK4, IMEX-start and SW1 repairs retained. **ACCEPTED-AS-DISCLOSED-LIMITATION:** search heuristics. |
| 12 | `rom_b.py` | **FIXED:** explicit control failures now block analysis. **STILL-WRONG — MAJOR:** absent controls pass. **ACCEPTED-AS-DISCLOSED-LIMITATION:** omitted diagnostic evolutions/D3. |
| 13 | Diagnostics / B timing / accounting | **FIXED:** G7 enforcement, call-level interleaving, scoring-lattice initializer comparison, and B basis label in analysis. **ACCEPTED-AS-DISCLOSED-LIMITATION:** B native timing, H1, corner/finer-lattice diagnostics. **STILL-WRONG — MINOR:** complete per-configuration statuses and “not searched” accounting remain absent. |

The separately highlighted round-2 **analysis schema crash is FIXED**: Part A now passes extracted row lists into `min_size()`.

Its other additional findings remain:

- **STILL-WRONG — MINOR:** Part A native timing still discards repetition arrays and batches FOM before ROM; this deviation is not disclosed.
- **STILL-WRONG — MINOR:** G9 still raises on an empty effective-rank ladder.
- **ACCEPTED-AS-DISCLOSED-LIMITATION:** omitted pre-stage benchmarks and internal unit-budget enforcement. **STILL-WRONG — MINOR:** complete total-budget accounting remains absent.

No new numerical-kernel failure was established. The demonstrated failures concern which candidates survive, which evidence qualifies, and which timing conclusions are emitted.

OVERALL: NEEDS-FIX