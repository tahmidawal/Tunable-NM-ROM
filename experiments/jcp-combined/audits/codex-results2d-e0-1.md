**Overall: NEEDS-RESTATEMENT for provenance/reporting; the audited numerical results are CORRECT.** No files were created, modified, or appended.

| Item | Verdict | Finding |
|---|---|---|
| 1. Independent E0 rescoring | **CORRECT** | Decoded 18 arm/case combinations across all six settings on CPU, at the \(257^2\) nodes, and scored against both \(R_{ST}\) and \(R_S\) using E0’s loader. All 36 worst-error values agree within \(1.05\times10^{-17}\). The sample includes all three flagged S cases. |
| 2. H1/H2/H3/H5 and uncertainty | **CORRECT** | Paired counts and propagated median/max intervals reproduce the decisions. Independently checked H2 coverage and all 210 H5 matches/speed-ups. H3’s stored guards satisfy the prescribed formulas, including added \(\eta\). S is diagnostic only; flagged S case-kinds never enter hypothesis rankings. |
| 3. Budget gates and invariance | **CORRECT**, with reproduction qualification | All applicable gates pass in all six budget files. E0 reproduction is **not applicable**, certified by passing first-order reproduction—not a fresh E0 reproduction pass. Certificate, result, and archived frozen-selection hashes verify. All 10,488 compared reference-free per-case component arrays are exactly unchanged. |
| 4. Finalist failures | **CORRECT** | Both selected CN-R steps fail quadrature qualification at `fast`, `acc`, and R256. R128/R384/R512 pass. Converged-rule checks pass; the failures concern the deployed rules. |
| 5. Changed conclusions | **CORRECT numerically; NEEDS-RESTATEMENT in presentation** | Changed H1 conclusions and H5 matching are supported by E0 uncertainty separation. Frozen selection remains unchanged. The analysis’s recorded frozen-file path now has a different whole-file hash, although the exact historical version exists and its 2D selection matches exactly. |

The substantive first-order → E0 changes are:

| Conclusion | First-order | E0 |
|---|---|---|
| H1 R128→R256 | Pass | Pass: median −53.68%; 38 improved, none tied |
| H1 R256→R384 | Reference-limited | **Fail**: median improves only 5.16%, below 25%; 34 resolved, four tied |
| H1 R384→R512 | Reference-limited | **Does not pay off**: six improve, 22 worsen, ten tie; median worsens 0.70%, worst worsens 0.392 percentage points |
| H2 | Pass | Unchanged: 42/42 BE arms covered for both median and worst error; dial criterion passes |
| H3 | Cancellation not supported | Unchanged at `val32/19`; criterion (a) fails |
| H5 overall maximum | 17.7057× | Unchanged |
| H5 remeasured `acc` comparator | 3.7973×, BDF2 at 0.025 | **3.8934×**, CN-R at 0.01; H5 still passes |

The changed H1 decisions are resolved under the per-case \(2\epsilon_{\rm ref}\) rule and propagated aggregate intervals. For H5, **41 ST matches change**: every displaced comparator is now strictly worse than its ROM after interval propagation, with minimum separation \(4.17\times10^{-4}\). All selected E0 matches have separated intervals in the permitted direction. These speed-up changes arise from reference-dependent rematching, not new timing measurements.

The finalist quadrature distances confirm the failures against \(\tau=2.5\times10^{-4}\):

| Setting | CN-R at \(\Delta t_0\) | CN-R at \(2\Delta t_0\) |
|---|---:|---:|
| fast | \(7.607\times10^{-4}\) | \(7.183\times10^{-4}\) |
| acc | \(2.856\times10^{-4}\) | \(4.130\times10^{-4}\) |
| R256 | \(2.696\times10^{-4}\) | \(2.740\times10^{-4}\) |

Required restatements:

- Cite `checks/frozen-versions/FROZEN-SELECTION-306e226a.json` for the E0 analysis’s recorded selection hash. The current `FROZEN-SELECTION.json` differs at whole-file level; its 2D selection is unchanged.
- Preserve the distinction between **frozen hypothesis outcomes** and **finalist qualification**. Passing H2/H5 does not erase the deployed-rule failures.
- The current report’s 2D budget table still explicitly presents first-order, provisional numbers. It should not be described as the E0 budget table.
- Retain A13’s conditional accuracy language: \(\epsilon_{\rm ref}\) is a Richardson estimate, not a rigorous bound.