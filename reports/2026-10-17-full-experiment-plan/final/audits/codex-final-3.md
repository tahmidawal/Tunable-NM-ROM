OVERALL: NOT-CLEAN

The sampled numerical results agree with the lane JSONs. Most previous findings are fixed. Remaining issues are misleading explanatory wording and incomplete compliance with the provenance, glossary, and mathematical-formatting requirements.

I made no file changes, submitted no jobs, executed no generator, and used no scratch storage.

Below, `F` means `reports/2026-10-17-full-experiment-plan/final/`; lane paths mean `worktrees/<date>-jcp-<lane>/experiments/jcp-<lane>/`.

| Requested item | Verdict |
|---|---|
| 1. Independently recompute ≥40 numbers | **CORRECT** — no sampled numerical mismatch |
| 2. Provenance and hand-typed numbers | **NEEDS-RESTATEMENT** — mechanical checks pass; quotation exception does not satisfy the brief |
| 3. Verdict fidelity | **CORRECT** for verdict classifications and their tabulated scope; explanatory wording needs correction |
| 4. Retractions | **CORRECT** for the identified withdrawals and previously missing corrections |
| 5. Brief coverage | **NEEDS-RESTATEMENT** — sections present; glossary and formatting remain incomplete |
| 6. Generator correctness | **NEEDS-RESTATEMENT** — checked arithmetic and filters work; misleading prose remains |

**Remaining findings**

1. **NEEDS-RESTATEMENT — “none” conflates family qualification with individual-draw qualification.**

   [Report:258](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:258) defines “none” as “no qualifying member within the candidate list.” The table then reports none for R128M512 Sobol.

   But the owning JSON has qualifying point counts for **all five individual draws**: 16,384; 32,768; 32,768; 32,768; 16,384. The family result is none because qualification is intersected across draws at the **same candidate size**.

   Evidence:
   - `jcp-dims/runs/EXPORT.json:656`, `families_2d/R128M512/mstar/sobol`.
   - `jcp-dims/report.md:100` explicitly retains the per-draw range and zero censored draws.
   - [dstats.py:126](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/dstats.py:126) implements the all-draw intersection.
   - `F/make_final.py:823` supplies the incomplete explanation.

   State that “none” means **no candidate qualifies simultaneously across every required draw**. Retain the distinction from cross-family gate failures.

2. **WRONG — the general claim that kernel cost grows more slowly than operation counts contradicts the displayed fits.**

   [Report:89](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:89), generated at `make_final.py:659`, says cost grows more slowly than the operation count.

   Its own table shows:
   - 2D N: **1.17 versus declared 1**;
   - 2D J: **1.08 versus 1**;
   - 3D J: **1.03 versus 1**.

   Their displayed intervals are also above 1. The slower-growth description fits JTJ and Cholesky, not all kernels. The sentence additionally supplies a causal explanation immediately before saying the labels are not causal explanations.

   Replace it with a kernel-specific description of the observed deviations.

3. **NEEDS-RESTATEMENT — historical quotations remain outside the required number-level provenance contract.**

   [Report:5](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/final/report.md:5) now honestly discloses the exception. That fixes the previous overstatement, but does not satisfy [final-compilation.md:9](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/final-compilation.md:9), which requires each number’s JSON source path.

   `PROVENANCE.json:7679` onward records whole Markdown sections rather than individual numerical sources. For example, `report.md:400` quotes projected and measured job hours under one section-level record.

   These quotations are traceable documentary evidence, but are not independently reproducible numerical entries under the specified contract.

4. **NEEDS-RESTATEMENT — glossary completeness is still not achieved by redirecting readers elsewhere.**

   `F/report.md:388` says quotation-only terms are defined in the source lane’s glossary. That is insufficient for the requested self-contained glossary, and is not universally true: **Möbius sum** appears in the budget correction text but is not defined in that lane’s glossary.

   The compilation also leaves terms such as **hyperbolic-cross**, **Wilcoxon**, and **case-bootstrap** unexplained, despite using them outside quotations (`report.md:291`, `:40`, `:638`). The final glossary should explain these plainly.

5. **NEEDS-RESTATEMENT — the quotation-format exception is disclosed, but not authorized by the brief.**

   The H2q `$65^{3}$` expressions are fixed. However, copied mathematical expressions remain outside LaTeX—for example `r=R'` and the max/min inequality at `report.md:511–513`. The brief’s LaTeX requirement contains no quotation exception.

**Independent numerical checks — CORRECT**

I read lane JSONs directly without importing or running `make_final.py`. The following includes **114 explicitly checked rendered measurements**, plus independently recomputed counts, interval predicates, ratios, and ledger totals.

| Report evidence | Independent check |
|---|---|
| `F/report.md:80–87`, §3.1 | All **24** fitted exponents and interval endpoints agree. 2D exponents: **1.17403, 1.08417, 1.92488, 1.23508**. |
| `:111–118`, §3.2 | All **8** max/min ratios recomputed from mesh medians, plus **8** upper interval endpoints. Largest upper endpoint **1.0403859222**. |
| `:122–125`, §3.2 | All **16** millisecond timings agree after seconds-to-milliseconds conversion. |
| `:138–145`, §3.3 | All **16** median/worst percentage values agree. Qualification independently counts **98/112**. |
| `:41–44`, verdicts | H-acc-N: **21 pass, 4 inconclusive, 5 unqualified/inconclusive**. H-acc-r: **14/10/3** monotone/inconclusive/not-monotone through 384; **20/7** not-monotone/inconclusive through 512. |
| `:47–61`, §3.4 | Primary-scope counts reproduce UB **16 ≥2×, 2 <2×, 3 unresolved**; CC **19 <2×, 2 unresolved**; PS/union **21 <2×**, at each mesh. Union extrema: 2D **0.0493063–0.705517**, 3D **0.00785541–0.158239**. |
| `:62–68`, verdicts | Baseline domination recomputed from time/error interval endpoints: **8/8, 8/8, 2/4 qualified 3D, 5/5 diagnostic 3D**. H2q: **1.1491378, 2.9528017, 9.8741350**. |
| `:227–246`, §3.5 | All **20** baseline time ratios reproduce as medians of exported per-case ratios. Direction is correctly **baseline/ours**. |
| `:262–279`, §3.6 | Family outcomes and finite counts **9216, 4181, 24577, 16384, 28657, 32768, 32768** agree; interpretation caveat above. |
| `:291–318`, §§3.7–3.8 | Blocked square-cell verdicts, low-frequency comparison, error-budget labels, and LSPG-only sampling scope agree. Cross-mesh budget labels count **7 same, 3 inconclusive**. |
| `:324–328`, §3.9 | 2D qualified/stable/no-growth counts: **9/10/10**, **9/11/11**, **12/12/12**. The 3D qualification failure and separate provisional W3/W2 labels are preserved. |
| `:336`, §3.10 | Head **1.4402459%**; linear r=16 **10.1713102%**, cost **0.26480846×**; r=128 **1.0953379%**, cost **0.59100019×**. |
| `:341–347`, §§3.11–3.12 | Both unfinished lanes explicitly remain **pending**, with no scientific results invented. |
| `:355–360`, P1g | All **42** measurements agree, including separate E0/first-order spreads and historical-arm qualification labels. |
| `:372–373`, retractions | Pinned combined H5 independently gives **13.6447877× / 17.7057367×**. All six withdrawn-arm tight-solve distances and worst-case `val32\|19` agree. |
| `:587–600`, ledger | Ten lane spends and entry counts agree. Finished total **107.4971 GPU-h**; all-lane terminal-job total **143.9204 GPU-h**, correctly printed **107/144**. |

The lane-list check found **18/20 current git refs unchanged**. The running regimes lane advanced from `6ff7db995`/`938264527` to `6ec3a20fb`/`f1bc8150b`. This is a disclosed compile-time snapshot becoming older, not an arithmetic or provenance fabrication (`report.md:620`).

**Provenance, verdicts, and code checks**

- **CORRECT:** All **437 direct JSON-value records** resolve and match.
- **CORRECT:** All **61 checked disk-file hashes**, including generated figures, match. Both pinned historical git-blob hashes match; all **33 PNG hashes** match.
- **CORRECT:** The only wildcard-path exceptions are the documented optional `state`/`reaches_2x` branches in `s3.hcomp.counts`. Their defaults are now explained.
- **CORRECT:** The verdict classifications preserve primary LSPG scope, baseline batch separation, developmental/in-sample qualifications, finite-ladder upper bounds, deployment-mode policy, and H2q comparator qualification.
- **CORRECT:** Primary-scope filtering, baseline interval logic, time-ratio direction, growth-window separation, and ledger summation agree with the data.
- **CORRECT:** I inspected the family and union summary images. Their plotted content matches their intended datasets.
- **CORRECT:** Generator syntax parses in memory.
- **CORRECT:** Required major sections, Mermaid, both readings of k, dimensional/family H-comp, mesh-specific H-base, K2, deferred items, and merge status are present. No prohibited heat comparison or pre-September Burgers speed-up was found.

**Previous findings rechecked**

| Previous WRONG / NEEDS-RESTATEMENT item | Current verdict and evidence |
|---|---|
| Invalid growth key path | **CORRECT — fixed**; literal `W2/W1` paths resolve |
| Hand-typed 3D success count and failed-reference count | **CORRECT — fixed**; `report.md:210`, `:330` |
| Hand-typed worst-case identifier | **CORRECT — fixed**; `make_final.py:986–988` reads and registers each case |
| Derived paths, batch filters, optional keys | **CORRECT — fixed/documented**; `PROVENANCE.json:1438` |
| Universal E0 claim and P1g reference labels | **CORRECT — fixed**; `report.md:3`, `:353`, `:631` |
| Overbroad 2D baseline population | **CORRECT — fixed**; `report.md:25` |
| Evolution-only cost scope | **CORRECT — fixed**; `report.md:25`, `:33` |
| Historical P1g qualification | **CORRECT — fixed**; `report.md:357–360` |
| H2q glossary, timing contract, comparator qualification | **CORRECT — fixed**; `report.md:66–70`, `:646` |
| Head exponent exception | **CORRECT — fixed**; `report.md:97` |
| Baseline budget qualifications | **CORRECT — fixed**; `report.md:62–65` |
| Previously identified missing corrections | **CORRECT — included**; expanded §5.1, including log bullets 19124 and 19191 |
| False exhaustive-selector claim | **CORRECT — restated**; `report.md:388` explicitly limits the keyword search |
| Per-family figure and broken table pipes | **CORRECT — fixed**; `report.md:204`, `:227–246` |
| T=1 growth conflation | **CORRECT — fixed**; `report.md:322–328` |
| Ledger implied instantaneous consumption | **CORRECT — fixed**; `report.md:598` |
| Quotation provenance, glossary, mathematical formatting | **NEEDS-RESTATEMENT — remaining limitations described above** |

The named major withdrawals remain withdrawn. No sampled numerical result needs changing; the remaining corrections concern interpretation and compliance with the compilation brief.