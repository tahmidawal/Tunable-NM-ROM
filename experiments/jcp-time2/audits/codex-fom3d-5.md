**fom1k: NOT CLEAN.**  
**b3d65: NOT CLEAN.**

Reviewed HEAD `a7ea6e1aa`. No files modified or jobs submitted; no lab-log append, per your instruction. Checks used source inspection and in-memory execution of the audit functions, including the complete 3D validator with the configured inventory and synthetic fields.

| Finding | Disposition | Round-5 assessment |
|---|---|---|
| **4** | **RESOLVED** | Cross-job loading now requires the accepted audit’s SHA-256 to match the exact result bytes. The existing `a1kfast` certificate matches its artifact. |
| **6** | **STILL-WRONG** | Per-step reconstruction substantially improves verification, but impossible diagnostics still pass. Details below. |
| **9** | **STILL-WRONG** | A11’s first causal statement is qualified, but A11.1 still concludes: “This is also the brief’s predicted CN ringing, seen in the full-order model.” A12.4 therefore overstates the wording repair. Nonblocking wording issue. |
| **13** | **STILL-WRONG** | Vendor eligibility and per-time rescoring are implemented, but iteration evidence and reason-histogram consistency remain unchecked; fractional reason codes pass. |
| **15** | **STILL-WRONG** | Repeated projection factorization is resolved by caching projections. Solver-evidence consistency remains incomplete under finding 6. |

The four round-4 acceptance failures:

| Acceptance failure | Disposition | Evidence |
|---|---|---|
| **FOM diagnostics** | **STILL-WRONG** | `fom_verified` returns `(True, True)` with contradictory `worst_lres`, fractional Newton counts, or infinite linear residuals on zero-Newton steps. |
| **ROM verification** | **STILL-WRONG** | `rom_steps_ok` returns `(True, True)` with impossible reason-4 evidence, fractional iteration counts, or corrupted residual-ratio padding. |
| **Vendor evidence** | **STILL-WRONG** | The complete 3D validator passes every check after replacing vendor reasons with `0.5`, histograms with negative counts, per-step iterations with NaNs, and `it_sum` with `999999`. |
| **Cross-job authentication** | **RESOLVED** | Digest binding closes the unchanged-job-ID/stale-certificate bypass. |

Specific recording-contract checks:

- **ROM indexing, maxima, tolerance and budget are aligned.** `make_evolve` writes step `k` before incrementing it; active entries are `[0:steps]`, with no initial-state diagnostic. The audit correctly reconstructs maxima starting from zero. `step_tolratio` uses the actual step tolerance: `tolf * scale` for LSPG, `tolf * ||Acn||` for GAL, with the recorded `+1e-300` denominator. The inclusive iteration bound of **600** matches the solver.
- **ROM padding and exit semantics are incomplete.** [audit_t2.py:91](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_t2.py:91) never checks `step_tolratio` padding. Iteration counts need not be integers; `int()` truncation lets fractional counts and aggregates agree. An all-reason-4 trajectory with `gn=1e100` and `tolratio=0.5` passes, although `make_lm` assigns reason 4 only when stationarity passes. Per-step acceptance is now reconstructed, but recording consistency is not fully enforced.
- **FOM padding and the 20-iteration bound are checked, but not integer counts or `worst_lres`.** [audit_fom.py:58](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:58) accepts fractional Newton counts through truncation and ignores the aggregate linear residual.
- **The zero-Newton premise differs from the current producer:** [fom2.py:66](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/fom2.py:66) initializes `lr=0` and records **zero**, not NaN, when no Newton iteration occurs. Even if NaN is deliberately allowed as an alternative “no linear solve” representation, the audit’s `isfinite(lres) | (nw == 0)` also admits infinity. This requires diagnostic consistency, not a new linear-tolerance acceptance criterion.
- **Vendor iteration evidence is saved but unused.** [audit_t3.py:141](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_t3.py:141) checks only `it_sum >= 25`; it never reconstructs that sum or `reasons`. The actual vendor uses three adaptive steps, each capped at 50 iterations, followed by 22 one-sweep steps. Those recorded counts can be checked directly.

**A13 is only partly honest.** Declaring the reference limitation and labelling H1-3D/H2-3D reference-dependent is appropriate. But [DESIGN.md:604](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/DESIGN.md:604) asserts that the reference’s time error “has the same sign” as the ROM’s BE time error. The cited distances do not isolate either time-error vector or establish that causal explanation. Say **“consistent with cancellation of temporal errors”**. Likewise, anchor discrepancy estimates time error relative to the validated discrete-ROM anchor, subject to its uncertainty. These are wording qualifications, not an additional submission blocker.

No additional independent blocking defect was established. Both holds remain the concrete, reproducible evidence-validation gaps above.