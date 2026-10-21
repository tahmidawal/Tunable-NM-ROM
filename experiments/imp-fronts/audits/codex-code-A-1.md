**Verdict: corrections are required before accepting Phase A outputs.** A2 has a demonstrated tail-computation error, and its registered bank-path control fails on the first selected case. Audited against Amendments 1–4; control verdicts below assess implementation, not whether the control passes.

Read-only throughout: no files changed, no jobs submitted, no JAX run, no scratch created. Small CPU checks used bytecode-disabled Python.

| Script | Verdict | Finding |
|---|---|---|
| [fcommon.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-fronts/experiments/imp-fronts/fcommon.py) | NEEDS-RESTATEMENT | Core loaders, lattice transforms and metrics are correct; enumeration and output-contract gaps below. |
| [a1_smear.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-fronts/experiments/imp-fronts/a1_smear.py) | WRONG | Non-finite residuals can pass G2; sensitivity omits gates and clipping; analysis trusts stale sets and manifests. |
| [a2_wall.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-fronts/experiments/imp-fronts/a2_wall.py) | WRONG | Tail cancellation corrupts statistics; availability denominator excludes skipped cases; empty populations crash. |
| [a3_trunc.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-fronts/experiments/imp-fronts/a3_trunc.py) | WRONG | Primary arithmetic is correct, but secondary verdict uses the wrong population and bypasses control failure. |
| [a4_hybrid.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-fronts/experiments/imp-fronts/a4_hybrid.py) | NEEDS-RESTATEMENT | Registered floor rule and controls are correct; QR rank policy and missing-evidence handling need repair. |
| `deps/burgers2d_film.py` | CORRECT | SHA-256 matches provenance and incumbent source: `c521b36a…18493e4`; default BE/upwind/Newton/BiCGStab implementation matches. |

1. **A2 tail computation — WRONG, demonstrated on real data** (`tails`, lines 64–71). Subtracting cumulative low-frequency energy from total energy loses small tails. For `dev6|0`, code reports $\tau_{9216}=10^{-14}$; direct summation of retained DST tail coefficients gives **$1.45976514\times10^{-8}$**. At 4096 it reports $7.99518\times10^{-8}$ versus $9.12060\times10^{-8}$. Thus both slopes and the decision ratio are affected. **Fix:** sum `C[:, K:]**2` directly, or use reverse cumulative sums, then clip.

2. **A2 population and unavailability — WRONG** (lines 139–149, 186–196). Invalid runs are recorded in `skipped`, correctly, but also remove their reference/projection diagnostics and shrink the half-available denominator to surviving rows. One surviving ROM can therefore satisfy “half” despite almost all in-range cases being unavailable. Zero survivors trigger `StopIteration`; no in-range survivors can trigger `None >= 3`. **Fix:** count eligible in-range cases before ROM filtering; maintain independent reference/projection/ROM availability; explicitly return INCONCLUSIVE for insufficient or empty populations. Report invalid and ineligible counts separately.

3. **A1 gates and sensitivity — WRONG** (lines 132–136, 184–190). `max(0., nan)` returns `0.`, so NaN residuals can pass G2 when saved fields are finite. The 256-grid branch checks neither residuals nor finiteness and uses unclipped floors in ratios. Generation checks only saved snapshots later, not all 51 generated fields. **Fix:** explicitly require finite residual arrays and all generated states, apply G2 to both resolutions, and clip/flag every sensitivity floor before division.

4. **A1 input binding — WRONG** (lines 58–59, 123–130, 219). Existing files are reused without validation; analysis prints the manifest but never verifies it, never checks embedded generator hashes, and never revalidates saved T6/X eligibility. **Fix:** verify file hashes, generator identity/settings, descriptors and current eligibility; validate the selected sets before analysis. Current inventory is clean: **52 files**, manifest matches, finite saved fields, maximum residual **$9.9898\times10^{-13}$**. These defects are latent here. Freeze/assert generator environment overrides such as `NEWTON_ITERS=8`.

5. **A1 primary $q$ — WRONG** (line 140). Uses $h=1/255$ for the primary 257-node fields; use **$1/256$**, with a separately named training-grid quantity if desired. Spearman correlation is unchanged by this constant scaling.

6. **A3 secondary reporting — WRONG** (lines 60, 88–90). `verdict_all_regimes` uses `everything`, including dev6/val32, instead of `all_regimes`; it also reports pass/STOP even when controls fail. Reproduction silently ignores missing `lane_e`. **Fix:** select the actual regime scope, propagate INCONCLUSIVE, require reproduction evidence for every scored regime case, and use Amendment 1’s wording: **“CPU truncation screen passed; F7 timing gate deferred.”**

7. **Numerical rank policies — NEEDS-RESTATEMENT.** A2 computes an unused `Q`, then projects with `lstsq(rcond=None)`: its effective cutoff is about $1.47\times10^{-11}$ rather than `F.orth`’s $10^{-13}$. This does **not** damage the checked incumbent: G384 has $\sigma_{\min}/\sigma_{\max}\approx0.999999999996$, and projection disagreement was $1.71\times10^{-14}$ relative. Nevertheless, use one explicit SVD policy for coefficients and projectors. A4’s unpivoted QR diagonal filtering is not rank-revealing: deleting a Q column can discard components of later retained input columns. Use SVD of the small QR factor, or an explicitly tested rank-revealing method; record rank and singular-value margins.

8. **Shared accounting/output contract — WRONG.** A1/A3/A4 omit excluded-reference counts; outputs omit the required conditional-Richardson qualification beside accuracy results. Empty A3/A4 populations raise `max()` errors rather than producing unavailable outcomes. Add explicit counts, reference qualification/metric metadata, and unavailable handling. `e0_cases()` should parse actual filenames rather than assume contiguous IDs from a count; reject unintended store overrides or record their resolved locations.

| Control / convention | Verdict | Assessment or concrete action |
|---|---|---|
| E0 loading and eligibility | CORRECT | `RS.load` verifies NPZ/JSON manifest hashes; `reference_status` supplies eligibility. |
| Regime loading and eligibility | CORRECT | Uses `meta['final'][2]`, including refined final levels, and the specified analysis configuration’s eligibility. |
| Lattice, DST-I, sine columns, walls | CORRECT | Orthonormal DST-I; stable $k_1^2+k_2^2$ ordering; product normalisation $2/256$; decoded walls zero. Tested sine Gram error $\le1.97\times10^{-14}$. |
| Error and floor metrics | CORRECT | Maximum over $t_1,\ldots,t_5$, fixed initial-field norm; explicit projection residual. |
| A1 G1 IC identity | CORRECT | Saved initial field compared against reference at $10^{-12}$; enforce before substantive analysis. |
| A1 G2 residual/finite gate | WRONG | NaN and sensitivity omissions described above. |
| A1 C-A1+ LOO upwind POD | CORRECT | Other-case six-time snapshots, top 128 vectors; T6 median threshold $\ge2$. |
| A1 C-A1− LOO reference POD | CORRECT | Same construction; threshold $\le1.25$. |
| A1 comparator resolution | CORRECT | Median of per-case positive/negative ratios, threshold $\ge2$. |
| A1 empirical own-field checks | CORRECT | Correct amended status: reported, not gating; primary floors clipped and flagged. |
| A1 exact-D fixture | CORRECT | Seed, nonzero all-ones $t_0$, $\alpha=1$, rank-index 5 sine, twice-orthogonalised union residual, swapped targets, prerequisites and $10^{-10}$ relative bar implemented. No full fixture replay claimed. |
| A1 descriptive decision | CORRECT | Median $\ge2$ plus at least five values $>1$; median $\le1.25$ negative; otherwise ambiguous; controls gate it. |
| A2 envelope identity | CORRECT | Measured relative error $6.87\times10^{-13}$ against $10^{-6}$. |
| A2 random-sine identity | CORRECT | Seed 7, 32 ranks below 2000, normal coefficients, interior Laplacian denominator; error $4.91\times10^{-14}$ against $10^{-8}$. |
| A2 gradient refinement | CORRECT | Three registered spacings and amended absolute/relative convergence test; finest value withheld when unconverged. |
| A2 bank-values path | CORRECT | Implements raw infinity-norm bound. **Actual first case `dev6|0` fails: gap 234.581 versus bound 1.40909. A2 must be INCONCLUSIVE.** Do not silently relax the registered tolerance. |
| A2 bank-path coverage | NEEDS-RESTATEMENT | Only the first surviving case is checked; freeze that fixture’s identity or specify broader coverage, rather than allowing eligibility/order to choose it. |
| A2 $\omega$ and band | CORRECT | Pooled high-pass energy, eight interior layers per wall; uniform interior share 0.121553. Zero/nonfinite total and sub-$10^{-24}$ tail are unavailable. Direct DST-tail synthesis would avoid subtraction roundoff. |
| A2 C-A2w+ | CORRECT | Perturbation alone, not reference plus perturbation; measured $\omega=0.813035\ge0.5$. |
| A2 C-A2w− | CORRECT | Centre Gaussian, width 0.03; measured $\omega=0.046884\le0.25$. |
| A2 RUN/DEFER thresholds | CORRECT | Uses AND: median tail ratio $\ge3$ and median wall share $\ge0.5$; population defects above still invalidate acceptance. |
| A3 full-sine identity | CORRECT | All 65025 modes, every scored case, $10^{-12}$ error-difference bar. |
| A3 constructed positive fixture | CORRECT | Seeded ranks 9217–20000, per-time norm $0.005n_0$, self-reference scoring, $0.001$ stop threshold. |
| A3 K16 stress / reproduction | CORRECT / WRONG | K16 correctly descriptive; reproduction must reject missing evidence, not skip it. |
| A4 containment | CORRECT | Checks every scored case/time against both constituent floors with $10^{-12}$ tolerance. |
| A4 null enrichment | CORRECT | Duplicate-bank SVD at $10^{-13}$; amended strict bank improvement includes $10^{-12}$ tolerance. |
| A4 high-frequency comparator | CORRECT | Correctly descriptive rather than gating after Amendment 1. |
| Sealed/OOD access | CORRECT | Inspected execution populations use only permitted cohorts; no sealed/OOD fields were read during this audit. |

Existing `checks/a3_trunc.json` and `checks/a4_hybrid.json` contain **pre-amendment control schemas**; they are not validation of the audited code. Regenerate only after corrections, preserving the disclosed draft-output status.