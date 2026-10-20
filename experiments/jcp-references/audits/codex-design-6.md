**Overall: REJECT at HEAD `ae0f1e5bc`.** A10.3 repairs the original numerical counterexamples, but the disclosure remains incomplete and the independent old-rule calculation still has a reproducible validity defect. The revised indicator is defensible **conditionally**; these tests do not establish actual PS–FD agreement.

No files were modified. Tests used NumPy with bytecode disabled; the auditor’s file reads were replaced in memory.

| Previous finding | Assessment of HEAD |
|---|---|
| **1. Chronology** | **CORRECT in A10.3(i).** It distinguishes new second-order references from existing first-order references and acknowledges the limits of repository evidence. However, A10’s heading, opening paragraph and A10.2 retain the broader, contradictory claims without explicit superseded labels. |
| **2. Indicator mathematics / floor loophole** | **CORRECT, conditional.** The revised floor retains the Richardson multiplier when available and explicitly assumes negligible remaining spatial error when convergence cannot establish it. The previous threshold counterexample is repaired. The fallback factor 10 is an allowance, not a bound for arbitrary undefined or slow convergence. |
| **3. Implementation agreement / validity** | **CORRECT for the reported reproductions.** Case 1, mixed floor/non-floor times, stale `finite=True`, and NaN epsilon now behave consistently. A separate arithmetic-overflow discrepancy remains below. |
| **4. First-order convergence / initial condition** | **NEEDS-RESTATEMENT.** A10.3(v) correctly limits PASS, but the original assertion that the initial fields differ “there” on the walls remains false. The wall-node difference is exactly zero. A10.3 does not explicitly correct that explanation or establish the cause of evolved first-order convergence. |
| **5. Old-rule disclosure** | **WRONG / incompletely repaired.** Early-return coverage, ordinary NaN rejection, independent old-rule comparison and the same-ladder report column are repaired. However, the auditor misses a non-finite computed discrepancy, and the report generator still omits the post-data chronology and floor assumption. |

**Reproductions against the current implementations**

I loaded all eight saved PS NPZs from `ps2dp0` and `ps3dp0`, used their canonical `n0`, and supplied synthetic accepted FD fields equal to the selected half-step PS field, with \(\epsilon_{\rm ref}=10^{-6}\).

| PS case | Recomputed \(\sigma_{PS}\) | Current rule, both implementations | Old rule, both |
|---|---:|---|---|
| 2D/0 | 1.454164703e−8 | PASS | UNRESOLVED |
| 2D/1 | 1.309303673e−12 | PASS | UNRESOLVED |
| 2D/2 | 5.863365719e−9 | PASS | UNRESOLVED |
| 2D/3 | 1.296228013e−12 | PASS | PASS |
| 3D/0 | 9.421344919e−9 | PASS | PASS |
| 3D/1 | 1.862209822e−10 | PASS | PASS |
| 3D/2 | 1.453538506e−10 | PASS | PASS |
| 3D/3 | 8.777627606e−10 | PASS | PASS |

Every auditor result had `match=True`. These PASSes are **synthetic functional tests**, not validation results.

Additional reproductions:

- **Original case-1 contradiction:** repaired. Its undefined \(q(t_1)\) remains, but both return PASS.
- **Mixed floor/non-floor times:** retaining case 1’s undefined floor time and introducing \(q=1\) at the other times gives PASS in both, `match=True`.
- **Floor loophole:** with \(q=0.50001\) and \(d_{23}/n_0=9\times10^{-13}\), both now give \(s_x=5.817566534\times10^{-12}\), matching the conditional Richardson estimate. The indicator is continuous across the floor threshold for this fixed order.
- **Undefined order below the floor:** at \(d_{23}/n_0=9\times10^{-13}\), both give \(s_x=9\times10^{-12}\). Undefined order above the floor is rejected.
- **Stale `finite=True` with a coarse-array NaN:** both reject. NaN middle-array or FD values are also rejected.
- **NaN epsilon:** both return `FD-INVALID`, including the old-rule verdict.
- **Gate, stability and FD-outcome failures:** old-rule verdicts are present and agree.

For initial-ladder unresolved cases, the auditor adds `-PENDING-ESCALATION`, matching the collector’s documented transformation; that is not a contradiction.

**Remaining and new defects**

1. **Independent old-rule validity still differs — WRONG.** Using actual 3D/0 PS arrays and a synthetic FD field whose positive-time entries are finite `1e200`, the squared norm overflows:

   - `verdict.crosscheck`: `UNRESOLVED`; old rule `UNRESOLVED`.
   - `audit_ref.audit_crosscheck`: `UNRESOLVED-PENDING-ESCALATION`; old rule **FAIL**; `match=False`.

   [audit_ref.py:317](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/audit_ref.py:317) checks finite `raw` but omits finite `xx`. This is an extreme-input robustness defect, not evidence of corruption in the saved PS data.

2. **Invalid normalization can produce an agreed false PASS — WRONG, defensive-validation gap.** Neither cross-check validates \(0<n_0<\infty\). With actual case-1 arrays, `n0=inf` produces PASS in both; `n0=-1` also produces PASS, with negative \(\sigma\). The saved canonical norms are valid, and the broader case audit separately checks normalization, so this does not invalidate the eight ordinary reproductions.

3. **Report disclosure remains inadequate — NEEDS-RESTATEMENT.** [make_report.py:260](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/make_report.py:260) now prints both verdicts and correctly labels “same ladder.” But it does not explain that A10 was adopted after inspecting PS self-convergence, disclose the floor assumption, or state A10.3’s restricted meaning of PASS. Its methods text still describes the resolution condition using “self-difference,” although the operative quantity includes the Richardson-adjusted spatial indicator.

4. **The initial-condition wording remains factually wrong.** Recomputed relative initial discrepancies are **0.001902430672** and **0.001583954007** for 2D cases 0 and 2; both have **zero wall-node discrepancy**. The text should describe interior reconstruction error associated with boundary incompatibility, without claiming that this alone proves the evolved first-order behavior.

Acceptance requires completing the disclosure and correcting the remaining validity inconsistency. The original case-1 contradiction and floor-threshold counterexample are no longer reasons to reject.