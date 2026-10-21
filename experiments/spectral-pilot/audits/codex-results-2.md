**The changed arithmetic reproduces, but interpretation and disclosure still need fixes.** Audit was read-only; no files were written.

Round-1 disposition:

| Item | Verdict | Round-2 finding |
|---|---|---|
| 1. Minimum sizes and speed ratios | FIXED | Independently reproduced from raw JSONs. |
| 2. Matched configurations | FIXED | Screen-selection/pruning disclosed; `Mq` and `Ninit` included. Final times reproduce as medians of per-case medians. |
| 3. Crossover diagnostic | STILL-WRONG | Cheaper-form arithmetic is correct, but the interpretation confuses rank ratios with timing ratios; glossary remains tensor-only. Details below. |
| 4. Censoring | FIXED | `cens.` now distinguishes unreached sizes from untimed entries. |
| 5. Control/allowance disclosure | FIXED | Coverage, G6/D2, diagnostics and unimplemented checks are disclosed. |
| 6. Timing protocol | FIXED | Unset precision and batched native timings disclosed. Spreads and outlier counts independently reproduce. |
| 7. Discretization discrepancy called a floor | FIXED | Replaced with appropriately qualified discrepancy wording. |
| 8. Break-even scope | FIXED | Explicitly component-based, mixed offline/online accounting. |
| 9. Family-width attribution | FIXED | Presented as motivation for a controlled experiment. |
| 10. B/AB causal decomposition | FIXED | Ratios labelled descriptive; high-order competitor performance declared unknown. |
| 11. Transfer/impossibility claims | FIXED | Conclusions scoped to this implementation and evaluated configurations; alternatives remain hypotheses. |
| 12. A4 omission and proposal | STILL-WRONG | Budget and matrix repaired; A4 rationale still does not follow from the cited rule. |
| 13. Strict local PILOT labels | STILL-WRONG | Numerical diagnostics paragraph and several table captions still rely on section-level labels. |

The independently recomputed **PILOT** crossover values match `ANALYSIS.json`. Columns below follow targets $\tau=(0.03,0.01,0.003,0.001)$:

| PILOT level | Cheaper-form $r_\times$, target order |
|---|---|
| A1 | 105.209055, 105.209055, 106.707297, 106.707297 |
| A2 | 110.646971, 124.118469, 164.132924, 164.132924 |
| A3 | 366.902164, 366.902164, 501.232435, unresolved |

At **PILOT $\tau=0.03$**, H-A2′ ratios reproduce as **0.912469, 1.735249, 1.046601**; tensor-only diagnostics reproduce as **1.295287, 2.471919, 3.075073**. The endpoint verdict remains **inconsistent**.

Remaining corrections:

- **Rank ratios are not evaluation-time ratios.** [Report line 149](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/report.md:149) compares whole-query speed ratios with “these per-evaluation ratios,” although the latter are $r_{\min}/r_\times$. Actual **PILOT cheapest-form ROM/FOM evaluation-time ratios at $r_{\min}$** are **0.808165, 1.915241, 1.028882**. Their orientation also reverses that of $S=t_{\mathrm{FOM}}/t_{\mathrm{ROM}}$. Remove the claimed cost attribution or support it with comparable timing quantities; acknowledging that decomposition was not performed does not establish what “carries the gap.”

- **Crossover reporting is incomplete and contradictory.** [Line 92](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/report.md:92) silently omits A3’s tightest-target unresolved crossover. Explicitly report it as unbracketed within the measured ladder. The glossary at line 213 still defines $r_\times$ using the tensor alone.

- **A4 omission remains inadequately justified.** [Line 190](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/report.md:190) says A3’s interrupted work subsequently ran separately, then invokes “all A3 stages finished” to explain skipping A4. A2.6 permits checkpoint splitting; a recovered timeout does not itself prohibit A4. State the actual stopping decision—remaining budget, unfinished stage, memory constraint, or discretionary stop—without claiming the rule compelled it.

- **Strict local labels remain incomplete.** The physical-diagnostics paragraph at [line 117](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/report.md:117) lacks PILOT. Several tables have PILOT only in their section headings rather than captions, contrary to round-1 item 13.

**Planning-budget arithmetic passes:** Part A′ is $12+48+12+15=87$ GPU-h; Part B′ is $10+20+24+8=62$ GPU-h. Total is **149 GPU-h**, with **29.8 GPU-h** contingency giving **178.8 GPU-h**. The printed approximations **150 / 180 GPU-h** are legitimate rounding, and planning estimates are clearly distinguished from measurements.

OVERALL: NEEDS-FIX