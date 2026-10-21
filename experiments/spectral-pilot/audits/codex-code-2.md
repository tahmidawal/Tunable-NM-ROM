**The numerical fixes largely work, but qualification and timing still permit untrustworthy pilot conclusions.** A5 discloses several legitimate scope reductions; it does not disclose or repair the remaining gate and timing failures.

Read-only audit completed. No files were created, modified, or deleted. Checks used the specified Python environment, CPU JAX, and disabled bytecode writes.

Round-1 findings, in order:

| # | Area | Disposition and remaining severity |
|---|---|---|
| 1 | `common.py` | **FIXED:** high-resolution IC quadrature. Reproduced relative errors of `3.56e-14`, `4.10e-15`, and `4.10e-15` at N=384, 512, and 768. **STILL-WRONG — MINOR:** `highest` precision is recorded but not enforced. |
| 2 | `romlib.py` | **FIXED:** the rank cutoff; a rank-two input now returns two modes. **FIXED:** tensor slabs, packed operators, quadrature operators, and `make_rom_batch` pass their large numerical operators dynamically. The broader A5 claim of “no captured large constants” remains overstated; see below. |
| 3 | `gen_a.py` | **FIXED downstream:** `fom_a` enforces SW2. **STILL-WRONG — MINOR:** `accepted` still excludes padding/SW2, and `tau_min_eligible` still uses the superseded formula. **STILL-WRONG — MAJOR:** training qualification is not propagated through ROM timing and analysis. **ACCEPTED-AS-DISCLOSED-LIMITATION:** in-memory snapshots and missing written memory projections. Pre-stage time benchmarks and budget enforcement remain undisclosed omissions. |
| 4 | `pad_gate.py` | **FIXED:** `fom_a` consumes and enforces padding results. **ACCEPTED-AS-DISCLOSED-LIMITATION:** memory projections. **STILL-WRONG — MINOR:** stage duration and preflight time projection remain absent; A5’s statement that observed stage times are recorded is not true for this script. |
| 5 | `fom_a.py` | **FIXED:** missing padding, failed space/time/padding/SW1/SW2, and nonfinite allowance components yield `U=None`. **ACCEPTED-AS-DISCLOSED-LIMITATION:** two-successful-sizes stopping heuristic, now registered in A5. **STILL-WRONG — MAJOR:** spectral-floor D2 assertion remains absent. Missing space/time/SW2 keys still raise rather than explicitly producing unresolved status. |
| 6 | `rom_a.py` | **FIXED:** D2 covers ROM-T, ROM-Q, and sine-ROM; SW4 rejects zero/nonfinite nonlinear norms; G9 chooses an existing ladder rank. **STILL-WRONG — MAJOR:** G4/SW4 failures do not prevent qualification, and reference/training labels are not carried through. **ACCEPTED-AS-DISCLOSED-LIMITATION:** D1 and random-span results are floors only. |
| 7 | `score.py` | **FIXED:** `U=None` or nonfinite U takes precedence, returning unresolved even for invalid errors. Equality fixtures remain correct. |
| 8 | `time_a.py` | **FIXED:** whole-query pruning replaces the nonlinear-step estimate; thread checks include OpenMP/OpenBLAS/MKL; nonlinear repetitions are retained; final retry exists; native queries exist. **ACCEPTED-AS-DISCLOSED-LIMITATION:** A5’s one-repetition pruning heuristic. **STILL-WRONG — BLOCKER:** screening validity and failed final timing are not enforced. **STILL-WRONG — MAJOR:** screening is still batched by candidate, and gates remain unchecked. |
| 9 | `fd_l.py` | **FIXED:** `make_rk4(32, …, out_lattice=False)` works; explicit `lat=32` works. **ACCEPTED-AS-DISCLOSED-LIMITATION:** amended Part B seeds. |
| 10 | `gen_b.py` | **FIXED:** spawn pool, positive/finite order-difference guard, conditional Richardson wording, and explicit unestablished basis label. **STILL-WRONG — MAJOR:** time-gate finiteness is checked after an unsafe scalar `max`; SW2 is still absent and not disclosed as omitted. |
| 11 | `fom_b.py` | **FIXED:** RK4 lattice crash, IMEX `n=5`, and SW1a/SW1b execution/enforcement. **ACCEPTED-AS-DISCLOSED-LIMITATION:** size stopping and diffusion-based RK4 pruning. |
| 12 | `rom_b.py` | **FIXED:** G4 now compares independently evolved trajectories; SW4 guards zero/nonfinite norms and records rollout change; tensor assembly receives explicit matrices. **STILL-WRONG — MAJOR:** failed controls remain nonbinding. **ACCEPTED-AS-DISCLOSED-LIMITATION:** floors-only D1/random-span scope and omitted D3. |
| 13 | `diag_a.py`, `time_b.py`, cross-cutting | **FIXED:** synchronized warm-up, screening sentinel measurement, final retry, spread/outliers, and final scoring label. **STILL-WRONG — MAJOR:** G7 enforcement, screening validity/interleaving, and initialiser normalization. **STILL-WRONG — MINOR:** B H1 and native-output timing remain absent; basis labels disappear downstream. **ACCEPTED-AS-DISCLOSED-LIMITATION:** B corner/finer-lattice diagnostics and written memory projections. Explicit per-configuration statuses and complete “not searched” accounting remain incomplete. |

The findings that prevent trust are:

1. **BLOCKER — Sentinel failure still permits reported speedups.**  
   [time_a.py:256](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/time_a.py:256) and `time_b.py:190` merely record screening drift. Neither retries nor invalidates the screen, and final timing never checks it. A fresh final run cannot recover candidates discarded or misselected during a drifting screen.

   Final timing retries once, but after a second failure it retains numerical `S` values alongside `timing_valid=False`. [analyze.py:104](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/analyze.py:104) and its B counterpart consume those values without checking validity. Thus failed timing can still produce crossover and “ROM ≥2× faster” conclusions.

2. **MAJOR — Required controls remain informational.**  
   Both timing screens read ROM errors without checking `G4_pass` or available-but-failed SW4 fixtures. Part A never loads `diag.json` to enforce G7. The training label added to `fom.json` does not propagate into A timing or the final analysis.

   A5 says gates are enforced; that claim is only partly implemented. A failed mathematical-equivalence control can still supply a winning ROM.

3. **MAJOR — Screening still violates call-level interleaving.**  
   [time_a.py:222](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/time_a.py:222) times all five repetitions for each of three cases before switching candidates. Alternating candidate batches is the same defect identified in round 1. Part B still completes FOM-family screening before ROM screening.

   A5 authorizes the pruning heuristic, not this timing-protocol deviation. This matters because screening chooses the finalists.

4. **MAJOR — Newly identified analysis crash prevents Part A readouts.**  
   [analyze.py:95](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/analyze.py:95) passes `fom['sizes']` directly to `min_size()`. Its values are dictionaries containing `rows` and `seconds`; the helper expects lists of row dictionaries.

   A minimal actual-schema fixture reproduces:

   ```text
   TypeError: string indices must be integers, not 'str'
   ```

   Pass the extracted row lists, as the Part B caller already does. This is newly identified, not proven newly introduced.

5. **MAJOR — Part B’s nonfinite-time guard is incomplete.**  
   [gen_b.py:103](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/gen_b.py:103) computes `max(dt1024, dt512)` before validating both inputs. Python’s `max(1e-6, NaN)` returns `1e-6`; checking only the result can conceal an unavailable second gate and produce a finite allowance. Validate both components first. SW2 is also still missing from B.

6. **MAJOR — Part B initialiser selection still uses the wrong norm.**  
   `time_b.py` compares the Euclidean coefficient discrepancy against the full 512-grid IC norm. The registered metric uses the scoring lattice. Neither the numerator nor denominator is converted to that metric.

   Full-trajectory re-scoring protects final target accuracy, but it does not establish the registered `tau/20` initializer condition. It can change the selected initializer and therefore the priced query.

Additional implementation findings:

- **MINOR — Explicit-argument conversion is incomplete outside the central ROM path.** `common.make_fom` still captures `ICRule.SW` and `K2`; `romlib.ic_project` captures `Pr`; B’s direct rollout captures the large `Ij/Jj` index arrays. This does not establish a numerical failure, but A5’s categorical claim is inaccurate and production-size compiler validation remains outstanding.
- **MINOR — Native timing is only partially repaired.** Part A’s native query has correct semantics, but native repetitions are reduced immediately to medians and F/R calls are not interleaved. Part B has no native contract implementation. Neither omission is listed in A5.
- **MINOR — G9 still has an empty-ladder edge case.** Its new `max(...)` handles non-ladder effective ranks above the first rung, but raises if effective rank is below four.
- **MINOR — Budget controls remain unimplemented.** A5 explicitly waives written memory projections, not the required pre-stage time benchmarks, unit timeouts, or total budget accounting.

The small CPU regression checks found no numerical regression in the central NLOp refactor:

- Tensor versus direct spectral ROM rollout: maximum absolute difference `0.0` in the test.
- ROM-T and ROM-Q timing queries versus accuracy rollouts: native difference `7.29e-17`; decoded-field difference `1.25e-16`.
- RK4 at L=32: full-array and lattice outputs agree exactly.
- Missing allowance: finite, infinite, and NaN errors all classify unresolved.

These checks validate small numerical paths, not production-size compilation or timing reliability.

OVERALL: NEEDS-FIX