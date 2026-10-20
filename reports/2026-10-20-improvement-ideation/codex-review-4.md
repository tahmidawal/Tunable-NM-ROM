VERDICT: CLEAN

Clean **at ideation-report scope**, with minor corrections below. No remaining defect warrants NOT-CLEAN; this is not approval of an executable experimental protocol.

`I` = `reports/2026-10-20-improvement-ideation`. Paths are repository-relative.

**Round-1 disposition:** findings **1, 3–21** are resolved substantively. **2** is resolved mathematically but retains contradictory prose; **22** is adequately addressed for ideation, with registration details still pending; **23** is substantially resolved, with minor provenance/compliance exceptions below.

**Verification:** all **232 `VALUES.json` entries reproduce**, and regeneration in memory matches `report.md` exactly. All **eight CPU analyses reproduce** within numerical tolerance. Output writes were intercepted in memory, bytecode disabled, and filesystem mutations prohibited. No files, Git state, GPU jobs or cluster jobs were changed.

The principal checks pass:

| Check | Recomputed result |
|---|---|
| Exact sine prefixes matching worst bank floors | **533 / 1308**, for r = 128 / 384 |
| Median per-case sine/bank dimension ratios | **1.011719 / 0.638021** |
| Actual median in-span errors: N0s / C1 / C2 | **0.755188% / 4.606862% / 13.593435%** |
| ETD assumed step costs | **0.835842 / 0.463826 ms** |
| Advection operation counts | **75,497,472 → 37,748,736 flops** |
| Full LSPG / Galerkin tensors | **1.811939 / 0.452985 GB** |
| Symmetry-packed tensors | **0.908329 / 0.227082 GB** |
| ECSW point ratios | **5.333333 / 10.666667** |
| Illustrative decode ratios | **0.809 / 1.98 / 1.73** |

Bank/sine comparisons share the lattice, fields and normalization. Their first-order-reference limitation is disclosed. Stability calculations reproduce; continuum/discrete selected-mode symmetric differences are **2 at M=512** and **4 at M=1536**, on L=256. Common normalization cancels correctly. The heuristic diffusion threshold is **0.001786725**, versus **0.001777340** using the RK4 negative-real intercept; neither establishes ETDRK4 stability.

OOD scoring retains **153 cases/configuration**, calibrates on **52**, excludes ineligible references and includes **two invalid runs** as alarms. In-sample calibration and exploratory indicator selection are disclosed.

ROM decode is inside the query in `worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/t2core.py:265–272`; spectral synthesis is inside `worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/psjax2d.py:61–68`. The timing boundaries are correct; the cross-job ratios remain estimates.

1. **[minor] Stale prose contradicts the corrected decomposition and endpoint interpretation.**

   **Evidence:** `I/report.md:276,379` still says “non-span part.” The recomputed **4.606862% / 13.593435%** quantities are **inside** the span, as correctly defined at line 105.

   Line **88** still concludes “the gap is not closed by convergence alone,” despite acknowledging that **1.538526%** is a projected-reference distance at **256²**, while **1.35%** is total error at **1024²**. The review record repeats the inference at line **391**.

   **Required fix:** replace “non-span” with “in-span”; remove the convergence conclusion pending the matched total-error comparison. These are residual wording defects, not failures of the revised numerical analysis.

2. **[minor] F15’s controls are not all equal-dimensional.**

   **Evidence:** `I/report.md:341` calls **1024 sines and bank 512** “equal-total-dimension controls” for a **1024-dimensional hybrid**. The bank control has **half** the candidate’s coefficient count. The sine control addresses the earlier sine-only-capacity objection, but comparison with bank 512 cannot isolate enrichment from increased bank capacity.

   **Required fix:** label bank 512 as an available, unmatched baseline. Either add a genuinely matched bank control later or restrict the inference to improvement over the available bank and equal-dimensional sines. Specify the hybrid’s bank/sine split.

3. **[minor — disclosed planning limitation] Some acceptance gates still need operational definitions and power checks.**

   **Evidence:** `I/report.md:305,309,331,353`. F13 names a valid embedded-method family but leaves controller constants and tolerance selection unspecified; F14f promises a fixed iteration limit and tolerance without values. Section 5 explicitly postpones freezing these details, which is acceptable for ideation.

   F12’s revised interval rule is valid, but **20 good cases cannot pass** its false-alarm requirement even with zero alarms: the one-sided 95% Clopper–Pearson upper endpoint is **0.139108**. At **32 good cases**, it is **0.089368**. At least **29 zero-alarm good cases** are needed.

   **Required fix before execution:** freeze controller/oracle settings and the bootstrap confidence level; distinguish inadequate statistical power from mechanism failure. The cited [Zoto–Bowman embedded pairs](https://arxiv.org/abs/2303.12139) support the proposed method family.

4. **[minor — optimization suggestion] F1 can absorb diffusion-coordinate transforms offline.**

   **Evidence:** `I/report.md:151–153,168–171`. Set \(z=V^\top R_m c\). Precompute
   \[
   B_z=BR_m^{-1}V,\qquad D_z=DR_m^{-1}V,\qquad P_z=\Psi QV.
   \]
   Then
   \[
   \dot z=-P_z^\top[(B_z z)\odot(D_z z)]-\nu Dz.
   \]
   No dense coordinate transform is required at each stage. Removing the assumed small-matrix charge changes the *same hypothetical* four-kernel sums to **0.624149 / 0.252133 ms**, before omitted overhead.

   **Required fix:** include this representation in F1’s implementation inventory and benchmark it. These are not predicted runtimes. The existing estimates are explicitly assumptions, so this omission does not invalidate the ranking or decode limitation.

5. **[minor] Report-rule compliance is substantial, not literal.**

   **Evidence:** title/status, LaTeX, coloured Mermaid, glossary and filename-departure explanation are present. However, `I/report.src.md:261` directly embeds the observed **96-case, six-time** wide-bank validation population; `report.md:261` calls it “96 … fields,” although the source describes **96 cases × six snapshots** (`worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md:113`).

   The hybrid ranking row at `I/report.md:126` also lacks the nearby first-order-reference qualification supplied in its detailed section.

   **Required fix:** generate the validation counts, say “96 cases,” and carry the `refdv` qualification into the ranking row.

The remaining ranking choices are judgments, not factual defects. Local hyper-reduction, residual sketching and matrix-oriented reduction now have appropriate consideration and prior art: [Grimberg et al.](https://arxiv.org/abs/2008.02891), [Balabanov–Nouy](https://arxiv.org/abs/1910.14378), and [Kirsten–Simoncini](https://arxiv.org/abs/2006.13289). The disclosed reference limitations, unqualified steep-regime runs, optimistic latency assumptions and unproved boundary hypothesis should remain explicit.