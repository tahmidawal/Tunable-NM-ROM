Audited `cc1253e89` read-only, CPU-only. No files changed or scratch written. Recomputed replication summaries and p1/m1 analyses; regenerated report/EXPORT in memory and all PNGs pixel-for-pixel. Sampled 36 coefficient trajectories against E0 test32.

| Part | Item | Verdict |
|---|---|---|
| A | Job, commit, H200, GPU backend, f64/highest, archive checksums | CORRECT |
| A | Seed 923901, cases 0–31, E0 test32 association | CORRECT |
| A | Bitwise CPU reproduction of cohort fingerprint | NEEDS-RESTATEMENT |
| A | Raw-row E-a/E-b and distance-array E-c summaries | CORRECT |
| A | Three ineligible W256 QN entries | CORRECT |
| A | Sampled coefficient decodes and ADT distances | CORRECT |
| A | All three winners tie-or-beat their comparator | CORRECT |
| A | Replication S: 2.03, 5.31, 16.6 | CORRECT |
| A | Matched-FOM holds, including lost W_C_W768/CC match | CORRECT |
| A | Immutable validation selection; nothing selected on 923901 | CORRECT |
| B | Title, status line, Mermaid, generated result numbers | CORRECT |
| B | Report/EXPORT/PNG numerical consistency | CORRECT |
| B | E0 qualification and screening/confirmation distinction in report | CORRECT |
| B | Finite-ladder upper-bound wording | CORRECT |
| B | Mesh-free/scoring-lattice contract; mesh-specific matching | CORRECT |
| B | Largest-mesh family/union H-comp outcomes; prominent K2 | CORRECT |
| B | Early spectral observation confirmed with cross-resolution qualification | CORRECT |
| B | Prohibited historical comparisons absent | CORRECT |
| B | Q1 export completeness and replication eligibility wording | WRONG |
| B | Inherited post-hoc labels and retraction disclosure | NEEDS-RESTATEMENT |
| B | m1 varying-mode policy disclosure | NEEDS-RESTATEMENT |
| B | Closing glossary completeness and accuracy | WRONG |
| B | LaTeX throughout mathematical prose | NEEDS-RESTATEMENT |
| B | Qualifications carried into standalone plots | NEEDS-RESTATEMENT |

Details for non-CORRECT items:

- **CPU cohort fingerprint:** CPU regeneration does not reproduce the archived GPU table hash. Active blob descriptors match E0; viscosity differs only for case 26 by \(1.39\times10^{-17}\). Sampled initial fields pass the scorer’s identity tolerances. This supports cohort identity, but not bitwise cross-platform reproduction. Sampled error and ADT-distance discrepancies were at most \(1.32\times10^{-16}\) and \(8.67\times10^{-19}\), respectively; this is not a numerical blocker.

- **Q1 export/scope:** [make_report.py:631](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/make_report.py:631) exports grid-arm `adt_gate` booleans without the numerical tight/tighter discrepancy or anchor provenance. Grid anchors are setting-wide checks on the deployed rule, not necessarily each exported arm’s rule. Furthermore, r1 contains no tighter trajectories: its reported eligibility recomputes E-a/E-b/E-c, whereas the glossary defines eligibility as full Q1 including tight/tighter agreement. Export the anchor numbers and their rule/cohort scope; distinguish replication E-a/E-b/E-c from a fresh Q1 certificate.

- **Inherited qualifications:** [report.md:426](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/report.md:426) discloses lane corrections and E0 A10, but omits the COMMON-BRIEF-required combined A10/A11 post-hoc labels and retracted A15. “Conditional on Step 2’s exploration” does not identify those specific qualifications.

- **m1 mode policy:** The report shows \(M\) changing—for M2, 2052 → 2049 → 2052 → 2052—but omits the required “deployment policy (mode set varies with mesh)” label and symmetric differences against the frozen mode list. Flat latency is supported for this policy; it is not a fixed-test-space experiment.

- **Glossary:** [report.md:482](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/report.md:482) leaves terms including \(M\), LSPG, CN-R, ETDRK4 and several gate/column meanings unexplained. Its universal “median/worst over validation cases” definition conflicts with the replication and subset-FOM tables. Its confirmation definition also reverses route-4 roles: those panels use A=ROM, B=FOM.

- **LaTeX:** Mathematical prose still contains expressions such as `median(e_a − η) ≤ median(e_b + η)` and Unicode inequalities outside math delimiters. Convert these in the generator; table identifiers can remain literal.

- **Standalone plots:** The screening Pareto plot says “E0” without its conditional-reference qualification. The mesh plot says “mesh-free” but omits scoring-lattice/prepared-input scope and an explicit warning that same-mesh timing curves are not matched-accuracy comparisons. The surrounding report supplies these qualifications; exported images do not.

**Part A: READY** — for the prescribed frozen replication, with the fingerprint caveat above.

**Part B: NOT READY** — numerical outputs reproduce, but certificate scope and required publication disclosures need correction.