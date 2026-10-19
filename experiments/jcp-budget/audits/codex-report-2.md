Audited HEAD `f9cbe041f`. No repository writes, SSH, or submissions. Scratch usage: 11 MB. The existing `HEARTBEAT.json` modification was left untouched.

**Reproducibility confirmed:** `report.md`, `runs/EXPORT.json`, and all five PNGs regenerate byte-identically. All **20 tables** parse using the Markdown-it GFM table extension, with matching header/body column counts.

Per round-1 blocking fix:

| Item | Verdict | Evidence |
|---|---|---|
| 1. Correction provenance | **CORRECT** | Correction prose now cites DESIGN amendments instead of asserting the disputed ratios. Certificate comparisons are generated from resource JSONs and the ledger; cancelled runs are explicitly partial. Unsupported memory-pressure causation is removed. |
| 2. Lane statement scope | **CORRECT** | Report statements and EXPORT explicitly restrict adequacy to **W1, LSPG members, relative to the 1025² route**. GAL-CN is excluded explicitly. |
| 3. B7 populations and normalisation | **NEEDS-RESTATEMENT** | The revised 1024² plots correctly use 36 jointly eligible cases, show `n`, mark horizon-unqualified members, and include current-time normalisation. The three 2D T=0.5 tables also include both normalisations. However, **3D T=0.5 and 2D T=1 still omit that comparison**, despite DESIGN §5.3 and available `e_rel_t` arrays. See [make_report.py:365](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/make_report.py:365). |
| 4. B8 table rendering | **CORRECT** | All three problematic headers now use `abs(...)`, eliminating embedded delimiters. Parser verification passes. |
| 5. Glossary and mathematics | **WRONG** | Resolution thresholds and pair-specific ties are corrected, but definitions still contradict or omit implemented rules; details below. Mesh headings and the correction equation now use LaTeX. |
| 6. 3D plot labels | **CORRECT** | The regenerated image visibly defines the normalised vertical axis and identifies the form comparison as “at the tight-solver end.” |

Remaining glossary defects:

- **Dominance omits the 75% completeness gate.** Both [ana2d.py:442](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ana2d.py:442) and `ana3d.summarise3` return `inconclusive (incomplete)` below that fraction. Pair, growth, and sampling definitions also omit their minimum-population gates. “Stable” remains undefined; the code checks validity and energy relative to **initial energy**, not monotonicity between outputs.
- **“Non-rankable … enters no … table population” is false.** B8 descriptive columns aggregate all stored rows, including `val32|10` and `val32|31` at 256² and 1024². Sampling verdicts exclude them. Thus these columns and the sampling histogram have a different population from the verdicts. See [make_report.py:395](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/make_report.py:395).
- **Ingredient definitions conflate dimensions.** The glossary gives Q as Gauss 320² with a 640² check without restricting that definition to 2D. The 3D configuration uses `gl56` and `gl64`; its ingredient set is Q/T/S/F, without I or M. See [configs/b3d.json:81](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/configs/b3d.json:81).

Additional report-completeness defect: DESIGN §5.3 requests ROM/reference monotonicity-violation counts, median **and worst** floor trajectories, and Kendall’s τ. The report omits the counts and τ and plots only median floors. These are reporting omissions, not evidence that the saved numerical results are wrong.

The round-1 prohibited-comparison check remains **CORRECT**: no heat POD-128 comparison or cross-lane performance conclusion appears.

**NOT-READY**

1. Complete the registered B7 reporting: both error normalisations for 3D and T=1, ROM/reference monotonicity counts, worst floor trajectories, and Kendall’s τ, retaining eligibility and provisional labels.
2. Correct the glossary’s completeness/population gates and stability definition; distinguish 2D from 3D ingredients.
3. Explicitly label B8 descriptive populations—including excluded reference cases—and correct the claim that non-rankable cases enter no table population.