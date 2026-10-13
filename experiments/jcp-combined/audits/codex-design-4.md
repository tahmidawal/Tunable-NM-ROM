**All requested findings are RESOLVED at the design-contract level by A3.** No files were modified, including the lab log.

| Item | Status | Operational resolution |
|---|---|---|
| P14 | RESOLVED | A3.6 checks each finalist’s actual-solver reference against the finer rule on the full qualification population. |
| P16 | RESOLVED | A3.6 covers fully qualified 3D fallbacks and explicitly withholds qualification when the reference check fails. |
| P19 | RESOLVED | A3.2 separates the two selection objectives and defines an unbeaten set. |
| P20 | RESOLVED | A3.4 qualifies the BE adaptive probe; failure withholds its diagnostic while retaining the deployed-BE comparison. |
| P21 | RESOLVED | A3.5 supplies discriminating analytic fixtures and an absolute-defect gate. |
| N5 | RESOLVED | A3.6 closes the finalist reference-check population gap. |
| N7 | RESOLVED | A3.3 accounts for simultaneous perturbations of both vectors; algebra verified below. |
| N8 | RESOLVED | A3.4 supplies the missing comparator-probe contract. |
| N9 | RESOLVED | A3.1 separates numerical Pareto fronts from tolerant coverage; A3.2 makes selection set-based. |
| N10 | RESOLVED | A3.5’s concrete oblique and out-of-span fixtures have nonzero defects. |
| F1 | RESOLVED | Ordinary Pareto dominance avoids mutual deletion and order-dependent pruning. |
| F2 | RESOLVED | Separate feasible sets preserve cheapest qualifying versus accurate-at-equal-cost selection. |
| F3 | RESOLVED | All three guards are sufficient under the stated perturbation radius. |
| F4 | RESOLVED | Both specified analytic defects are correct; absolute values exceed the gate. |
| F5 | RESOLVED | Actual finalist and qualified-fallback finer-rule runs cover all 64 3D cases, with explicit failure handling. |

**H3 guard algebra — [A3.3](experiments/jcp-combined/DESIGN.md#amendment-a3-2026-10-13-after-auditscodex-design-3md-before-any-code)**

Write \(s=\langle\delta,\varepsilon\rangle\). For \(\|q\|\le U\),
\[
s_q=\langle\delta-q,\varepsilon+q\rangle
=s+\langle\delta-\varepsilon,q\rangle-\|q\|^2
\le s+U(D+E).
\]
Thus A3’s additional \(+U^2\) is unnecessarily conservative, but valid. Under (b′),
\[
s_q\le-0.3(D+U)(E+U)
\le-0.3\|\delta-q\|\,\|\varepsilon+q\|.
\]
Division gives the required cosine bound. Passing (b′) also excludes perturbed zero vectors: for \(U>0\), either \(D\le U\) or \(E\le U\) makes its left side positive using \(s\ge-DE\), so the guard cannot pass. At \(U=0\), the inherited zero-vector exclusion applies.

**(a′) is sensible and sufficient.** With \(E_q=\|\varepsilon+q\|\),
\[
E-U\le E_q\le E+U,\qquad
|e_{\rm CN}-E_q|\le|e_{\rm CN}-E|+U.
\]
Its two inequalities therefore imply (a) for every permitted perturbation. It is deliberately stringent: even when \(e_{\rm CN}=E\), it requires \(U\le(E-e_{\rm BE})/6\). A failed guard appropriately means unresolved robustness, not disproven cancellation.

Likewise, (c′) combines a lower bound on \(E_q^2\) with an upper bound on \(\|\delta-q\|^2\). These conclusions remain **indicator-based**, as A3 states, and concern the fixed nominal \(t^\ast\).

**B0 fixture calculations — A3.5**

Using unit normalizer and defect
\[
\Delta=\|v-y\|^2-\|(I-P)y\|^2-\|v-Py\|^2:
\]

- **(i):**
  \[
  \Delta=[1+(1-10^{-3})^2]-1-[1+10^{-6}]
  =-0.002.
  \]
- **(ii):** \(P'y=(1/2,0)\), so
  \[
  \Delta=2-\left(\tfrac14+1\right)-\tfrac14=\tfrac12.
  \]

Both pass the required absolute-defect threshold. Fixture (iii) has no numerical defect calculable from the document alone; its chosen real-data term and mismatched normalizer must produce the explicitly required computed defect before that control passes.

**READY.**