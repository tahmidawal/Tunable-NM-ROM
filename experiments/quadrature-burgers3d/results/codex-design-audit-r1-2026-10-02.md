**Verdict: several fixes are correct, but production acceptance remains blocked.** Static review of `36538058e`; no files changed, jobs run, or network used. Paths below are relative to `experiments/quadrature-burgers3d/`.

| R1 item | Verdict | Evidence |
|---|---|---|
| **1. Lean FOM / memory** | **CORRECT implementation; NEEDS-RESTATEMENT of readiness** | `offmesh.py:224–264` preserves vendor Newton/BiCGStab arithmetic and output times, removes the full-state history, and passes eigenvalues explicitly. `refjob.py:75–84` asserts field/count equivalence. Actual 513-node peak-memory feasibility remains unverified here; recording a probe metric is not evidence that it passed. |
| **2. Reference acceptance / consumer** | **CORRECT for the in-job path** | `refjob.py:63–66,130–137` gates `.done` on residuals, caps and finiteness. `qpanel.py:553–563` checks checksum, acceptance, expected configuration, cohort and initial fields. **Offline consumption bypasses these checks**, below. |
| **3. Selection / exit classes** | **WRONG as a complete fix** | `select_q.py:75–107` correctly implements exit classes, no fallback, lexicographic ties and eligible `lat32768`/`gl32` agreement. But failed timing and continuum-resolution gates do **not** prevent selection. Invalid converged references still produce `smallest_m_reaching_converged` and control verdicts (`111–134`), contrary to promised withholding. |
| **4. Pipeline / held-out guard** | **WRONG** | Files exist, but important validation is absent. `refine_q.py:33–50` checks only reference seed—not `.done`, acceptance, expected configuration, checksum agreement or initial-field agreement. `audit_q.py:118–130` omits converged-reference validity from its supposedly equivalent selection implementation. `make_configs.py:38–43` accepts any existing selection file; `qpanel.py:108–109` checks its hash only **if supplied**, without requiring it for held-out seed 923901 or validating its contents. |
| **5. Timing** | **WRONG** | `qpanel.py:497–511` records missing groups as `None`, but filters them out of `all(...)`: empty groups still permit a pass. `select_q.py:30` propagates that pass. Repetition arrays and five warm-up calls are implemented (`523–531`). Sum/squared-sum checks improve detection but cannot establish full-field equality (`455–460`). |
| **6. G4** | **CORRECT** | `qpanel.py:230–242` exercises both configured ranks and asserts fields, coefficients, iteration counts, reasons and rejections. Still one case per rank, with shared LM code. |
| **7. Certification rollouts** | **CORRECT narrowly** | `qpanel.py:263–265` records exit counts and rejects nonfinite coefficients. Tensor-reached-state coverage is accurately restated. Finite but failed/nonstationary certification rollouts remain usable; recording reasons is not accepting convergence. |
| **8. Unresolved rho** | **WRONG** | `select_q.py:115–116` computes unresolved flags, but `130–131,145–146` independently emits win booleans and the aggregate win. An unresolved comparison can still be declared a win. |
| **9. Controls** | **NEEDS-RESTATEMENT** | Empirical-control interpretation is appropriate (`DESIGN.md:260–264`). However, `audit_q.py:134–144` computes differences directly; it does not inject corrupted inputs into the ordinary audit and verify rejection. This checks threshold sensitivity, not the complete detection path. |
| **10. Mesh invariance** | **NEEDS-RESTATEMENT** | `refine_q.py:85–97` correctly computes consecutive-mesh distances for common fixed arms. No field-distance acceptance threshold is specified, and neither the error-ratio verdict nor a separate selection-policy comparison is implemented. `DESIGN.md:265–267` overstates executable coverage. |
| **11. Cohort wording** | **CORRECT restatement** | `DESIGN.md:268–271` appropriately distinguishes the reused cohort and historical baseline. “Evaluated once” remains a procedural promise, not an enforced guard. |
| **12. Minor qualifications** | **CORRECT restatements** | `DESIGN.md:272–274` correctly limits Smolyak normalization, single-shift conclusions and reference-refinement interpretation. |

**`npcore` / refinement arithmetic:** no mathematical discrepancy found in the NumPy bank derivative, mode ordering, DST normalization, upwind stencil or quadrature scaling (`npcore.py:26–109`). Refined errors and coefficient-based cross-mesh distances are correct (`refine_q.py:55–78,93–95`). The failures are validation and verdict logic.

**Remaining BLOCKERS**

- Offline refined-reference acceptance can bypass all new producer/consumer safeguards.
- Selection must enforce gate validity and withhold unresolved/invalid-reference verdicts.
- Held-out execution needs a mandatory, semantically validated selection artifact; existence/hash alone is insufficient.
- Independent selection auditing must implement the same converged-reference validity rules.

**Remaining MAJORS**

- Empty timing groups still pass; aggregate checksums do not prove full-output equality.
- Certification convergence remains ungated.
- Mesh-invariance acceptance remains incomplete.
- Audit false passes remain possible: missing in-job refined comparisons default to zero (`refine_q.py:47`, `audit_q.py:74–78`); rho auditing checks sampled states against an upper bound rather than reproducing the reported maximum (`audit_q.py:105–111`).
- Production memory feasibility still requires actual probe evidence.

**Remaining MINORS**

- R1 claims dense quick-run milliseconds are reported, but dense arms have no measured query-ms summary (`qpanel.py:351–375,440`; `select_q.py:88`).
- Lean/vendor residual disagreement is recorded but not asserted (`refjob.py:82–84`).
- Probe warm-up synchronization is fixed; prescribed timing burn-in remains absent (`refjob.py:90–109`).