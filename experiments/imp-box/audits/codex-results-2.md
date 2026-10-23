**Overall verdict: NEEDS-RESTATEMENT.** The recorded arithmetic and stop outcomes survive. Remaining defects concern disclosure in the report and export.

Audited seven cells: **3,762 ladder rows, 234 selection entries, 895 attempts, 648 timing pairs and 128 defined S values**. CPU-only checks; no files or lab-log entries written.

| Item | Verdict | Finding |
|---|---|---|
| Selection statuses/finalists | **CORRECT for recorded evidence** | Recomputed counts, official error statistics, feasibility, qualification decisions, batch membership and finalists agree: **206 finalist, 12 arm-censored, 14 search-incomplete, 2 arm-unqualified** entries. The round-1 limitation remains: missing re-screen measurements prevent fully verifying ordering against unattempted candidates. It is now disclosed. |
| Paired timing and max–min S | **CORRECT** | All **648 pairs**, with 40 triplets each, reproduce their ratios and timing medians. Recorded identity, compilation and recomputed doubled-query predicates agree; **zero invalid pairs**. All **128 defined S values**, winning/limiting keys and exclusion lists reproduce. |
| A10/A11 FOM coverage | **CORRECT** | Every cell retains both implementations at all 11 dense-eligible sizes, except the explicitly excluded fixed-step FFT N=12. Dense fixed-step N=12 and both adaptive implementations remain. A3 restart ladders preserve the original ladder files as identical prefixes. |
| Stop-rule entries | **CORRECT** | Re-evaluation with `*:W40_A4,*:W28_A4` reproduces every evaluated field. SPEED-STOP: **NOT FIRED**, **95 matches**, **80 ≥0.5**, maximum **2.1437823699078313**, **27 missing/FOM-censored**. BANK-STOP: **NOT FIRED**; **3 above, 4 below, 8 no-affordable-rank, 3 missing**. |
| Controls, including G-XFORM/G-HOIST | **CORRECT in recorded evidence** | Reconstructed registries agree and all mandatory entries pass. G-XFORM covers all 11 sizes. A3 G-HOIST discrepancies are ≤**3.96×10⁻¹⁶**; saved timing ratios reproduce from repetition arrays. |
| Ladder-minimum and provisional labels | **CORRECT for main S entries** | **118 S entries** receive applicable minimum labels. Report and export identify provisional conditional-reference results. The round-1 omission is repaired. |
| Report/export numerical consistency | **CORRECT** | Regenerating with writes intercepted in memory reproduces both files **byte-for-byte**. All **24 bank equal-cost/floor entries** also reproduce. |
| Deferrals and A16 comparability disclosure | **NEEDS-RESTATEMENT** | The report contains no explicit deferral disclosure or G-HOIST timing ratios. Export includes the two wide A4 deferrals, but omits the broader deferred scope and A16 comparison evidence. |
| Incomplete-FOM labels in derived tables | **NEEDS-RESTATEMENT** | Main S entries carry these labels, but the bank-floor and crossing tables do not consistently propagate them. |

Three concrete reporting fixes remain:

- **Disclose the retained/deferred scope** from A14–A17, distinguishing deferred cells from pending work.
- **Add A16 timing evidence beside cross-cell comparisons.** Dense-FOM hoisted/embedded ratios are **0.898389 and 0.905893**—approximately 10% faster. Calling the entire change “within a few per cent,” as A17 does, needs correction.
- **Label incomplete-search diagnostics explicitly.** The [W40_A3 bank table](experiments/imp-box/report.md:84) prints three “above” verdicts that correctly become **missing** in the stop rule because adaptive-FOM selection is incomplete. The [crossing table](experiments/imp-box/report.md:210) similarly drops FOM-search qualifications. Export crossing records retain only reduced-arm status.

The largest recorded ratio, **3.318997407188012** for W40_A3 TRANS at 1% median, is correctly excluded from the stop-rule maximum because its FOM search is incomplete. **No numerical correction to S or either stop outcome is indicated.**