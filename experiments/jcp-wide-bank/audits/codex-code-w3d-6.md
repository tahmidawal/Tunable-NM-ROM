**No blocking implementation defect in `19d06be26`. A8 is acceptable as a smoke-informed amendment registered before J4, with s4e still required before staging J4.**

- **Propagation verified:** off-mesh, tensor, converged, check, validation/certification, timing, final-panel and reached-state Jacobian rollouts all inherit `adaptive_first=6`, across both banks and every J4 \(R'\).
- **G4 remains matched:** vendor and rule solvers both retain their default `adaptive_first=3`.
- **J2 is unchanged:** its config is identical to the parent commit and defaults to 3. No tolerance, trust, eligibility or convergence gate changed.

The checksum-verified **s4d result** confirms, for converged/check and the well-resolved rules:

- Case 1: reason-0 exits at zero-based indices **3, 4**; case 2: **3, 4, 5**. That is **5/100 steps**, specifically the fourth–sixth time advances.
- Worst refined error: **1.432525326%** for converged and **1.432525313%** for check, supporting **1.43%**.
- All certification steps for converged/check have reason 4. All recorded arms are finite, with no reason-3 exits.

**Minor wording qualification:** “every resolved rule” needs a definition. `gl24` also rounds to **1.43%**, but has **zero** non-stationary exits; its distance from converged is \(9.732\times10^{-4}\), outside the primary accuracy bar. Agreement in rounded refined error alone does not establish resolution.

**Comparison fairness:** uniform application prevents an explicit rank-dependent solver schedule. It does not prove solver-independent scaling: the schedule was chosen after observing the largest-rank smoke, and may benefit ranks differently. Describe J4 as using the amended common solver; do not directly compare its costs/errors with J2’s original schedule. The recorded fallback preserves unavailable verdicts rather than relaxing gates.

Approval covers the **s4e verification run and the stated A8 workflow**; s4e’s outcome is not yet verified. No files modified.

OK-TO-RUN: YES