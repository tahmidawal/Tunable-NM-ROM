**NOT READY FOR BUDGET SMOKES.**

Reviewed read-only at `98fd6511d`, applying A1–A8. No files were modified, project modules imported, or GPU jobs run. AST-extracted code probes reproduced the exceptions and arithmetic failures below. Paths are relative to `experiments/jcp-combined/`.

**Disposition of the previous 16 findings**

| # | Status | Evidence and remaining fix |
|---|---|---|
| 1 | **RESOLVED** | `cluster/stage.py:23` includes `c2bud.py`; `:147` executes the configured driver. |
| 2 | **RESOLVED** | `freeze.py:64` preserves `x['f']`. The exact \(5/3\) factor satisfies both schedule assertions. The obsolete rounding helper at `:36` is unused. A similar **new 3D defect** appears below. |
| 3 | **RESOLVED** | `c3run.py:210–211` stores IDs per trajectory; `score.py:250–254` uses them to score mixed populations. Three total cases versus two diagnostic cases no longer causes the reported indexing failure. |
| 4 | **RESOLVED** | `c2bud.py:144–158` requires finite distances and verified paired rows; `:174–187` implements the successor and restores the original Gauss320 rule when both checks fail. |
| 5 | **PARTIALLY — MAJOR** | Dependency checks now protect many ablations (`budget2d.py:129–144`, `budget3d.py:138–151`), and B1 stops both drivers. However, 2D `e_tot`, `e_star`, `e_dyn` and their summaries remain available for invalid participating trajectories (`budget2d.py:153–162`); 3D `e_tot` is unconditional (`budget3d.py:153`). B2 does not check control-run validity (`c2bud.py:195–207`, `c3run.py:441–447`), and 2D B3 uses only available cases as its denominator (`budget2d.py:248–249`). **Fix:** apply validity dependencies to every trajectory-dependent quantity and require complete, valid control populations before declaring discrimination. |
| 6 | **PARTIALLY — MAJOR** | `budget3d.py:105–116,176–183` adds convergence and all-case finalist checks. But absent finer arrays are silently skipped, and intersections silently omit missing cases; an empty check can pass. A failed convergence check is not added to `withheld` (`:195–201`). The driver’s B2 baseline is now fixed-sweep BE, but the smoke still budgets `vendor` as baseline (`make_configs.py:102`, `configs/smk3bud.json:23–26`). **Fix:** require the complete expected comparison inventory, propagate failed convergence into attribution availability, and use the A8 fixed-sweep baseline consistently in the smoke. |
| 7 | **PARTIALLY — MAJOR** | `make_budget_configs.py:24–45` supplies structured arms and the frozen hash; staging checks that hash when present (`cluster/stage.py:68–70`). Grid reproduction remains optional and passes after **any positive number** of successful comparisons (`budget2d.py:270`, `budget3d.py:210`); missing arms/cases are skipped. A8 permits conditional fallback jobs, but no `c2fb`/`c3fb` generation or qualification path is implemented here. **Fix:** require full expected grid-arm/case coverage, require binding for production budget configs, and implement the conditional fallback path before any fallback is claimed qualified. |
| 8 | **PARTIALLY — MAJOR** | 2D now contains closure, triangle, \(t^\ast\), and per-case context diagnostics. Its reproduction loop has a new counting defect described below. The 3D analyzer still omits closure ratios, triangle diagnostics, the explicit orthogonal pair at \(t^\ast\), context classifications and component-resolution flags (`budget3d.py:147–151,169–194`); it also does not execute the two analytic B0 fixtures. **Fix:** complete the registered diagnostics and perform reproduction once per unique expected row. |
| 9 | **PARTIALLY — MAJOR** | Candidate replay verification and the per-case `0.1 * bar` threshold are corrected (`analyze3d.py:102–109`). But the comparator’s `solver_distance`/`solver_worst` remain numerical even when its probe fails verification or replay qualification (`:105–109`). **Fix:** explicitly mark that diagnostic unavailable under A3.4 while retaining comparison against deployed BE. |
| 10 | **PARTIALLY — BLOCKER** | H4 witnesses are now included in budget arms (`analyze3d.py:147–155`), but extracting an empty witness list raises `IndexError`. See new defect N2. |
| 11 | **RESOLVED under A8** | `analyze3d.py:86` consumes supplied scores. `:169–171` reports H6’s raw bars as **reference-limited (descriptive)**, as A8.4 now requires. The separate new-reference certification problem remains under item 15. |
| 12 | **PARTIALLY — MAJOR** | `select3d.py:120–126` uses summed median `tB`; missing control and qualifying-family timing now prevent selection. However, completeness checks count invocations without checking unique `(subject, case, repetition)` slots (`:121–123`). Duplicate case-0 measurements can substitute for missing case-1 measurements. Missing ladder rules are also skipped (`:78–79,103`). **Fix:** validate the exact Cartesian timing inventory and require the expected ladder evidence before selecting its smallest qualifying rung. |
| 13 | **PARTIALLY — MAJOR** | ROM row/archive equality is now checked (`score.py:162–169,285–294`), and a wholly missing FOM archive is rejected (`:174–175`). Neither path validates against the required run inventory: removing both a row and its array still escapes detection. FOM coverage remains archive-driven (`:179–198`). Paired non-finite invalid runs can still contribute to a passing gate (`:155–158,264–266`). **Fix:** derive expected keys/cases independently from the configured inventory, check FOM coverage bidirectionally, and distinguish invalid-run accounting from finite score certification. |
| 14 | **PARTIALLY — MAJOR** | Contract strings/times, actual checkpoints and rotation are checked (`score.py:67–73,118–122,241–242`). But the 2D descriptor check trusts the stored `physical_sha256` without hashing the actual `physical` array consumed for normalization. The 3D scorer regenerates descriptors without checking their identity against a pinned reference-cohort contract (`:230–234`). **Fix:** authenticate the actual descriptor data and bind the 3D seed/table/case identities to the registry. |
| 15 | **PARTIALLY — MAJOR** | The 2D frozen schema and original-reference certificate handoff are fixed (`analyze2d.py:78–82,159–163`). Aggregate uncertainty is still wrong: `:108–110` assumes propagated median half-width equals `median(eta)`. It generally does not. Also, `analyze3d.py:45–55` has neither certified-original-score nor frozen-selection inputs; unchecked new-reference scores remain rejected. **Fix:** propagate each arm’s lower/upper endpoints through median/max, and implement the A8.7 certification/frozen-selection handoff in 3D. |
| 16 | **RESOLVED** | `analyze2d.py:53–56` collapses equal `(cost,error)` points to the lowest ID. The isolated duplicate-point probe passes. |

For item 15, a concrete counterexample is:

- Errors A: `[0.001, 0.002, 0.100]`; errors B: `[0.003, 0.004, 0.100]`.
- Case uncertainties: `[0.00001, 0.010, 0.00001]`.
- Correct propagated median intervals are `[0.00099, 0.012]` and `[0.00299, 0.014]`: **overlapping**.
- The implementation uses threshold `0.00002` and treats the median difference `0.002` as resolved.

**New defects introduced or exposed by these changes**

1. **N1 — BLOCKER: the 2D budget analyzer crashes on the first arm.**

   In [budget2d.py:200](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget2d.py:200), `x, y = mx(...)` overwrites the outer arm dictionary `x`. Line 223 then calls `x.get('selected')`; lines 240–244 also expect that dictionary.

   **Reproduced:** `AttributeError: 'float' object has no attribute 'get'`. An unavailable component instead leaves `x=None`, with the same failure.

   **Fix:** use distinct names such as `arm`, `deployed_distance`, and `ideal_distance`. This blocks offline processing of the supplied `smk2bud`.

2. **N2 — BLOCKER: an ordinary H4 failure crashes analysis and prevents freezing.**

   [analyze3d.py:147](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/analyze3d.py:147) uses:
   ```python
   h4.get(k, {}).get('witnesses', [None])[0]
   ```
   An evaluated hypothesis with no witnesses contains `witnesses=[]`; the default does not apply.

   **Reproduced:** H4a with no witness and H4b with one witness raises `IndexError: list index out of range`.

   **Fix:** use `(h4.get(k, {}).get('witnesses') or [None])[0]`. Failed hypotheses must produce valid output and no witness.

3. **N3 — BLOCKER: the new 3D arm conversion loses exact timestep precision.**

   [analyze3d.py:152](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/analyze3d.py:152) reconstructs `dt` from the `.8g` display key. `freeze.py:89` and `make_budget_configs.py:41` then preserve that rounded value.

   **Reproduced through the actual conversion expression:**
   ```text
   fs|CN|0.016666667|K2 → dt=0.016666667
   horizon defect ≈ 5e-9
   output-spacing defect ≈ 1e-9
   ```
   Both exceed `fs3.py:175`’s `1e-12` schedule tolerance.

   **Fix:** recover the original exact spec from `cfg['grid']['dt_factors']`, or persist exact specs alongside keys. The supplied smoke uses `0.025`, so it misses this production failure.

4. **N4 — MAJOR: 2D reproduction necessarily fails after N1 is fixed.**

   The all-row reproduction loop at [budget2d.py:167](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget2d.py:167) is nested inside the arm loop at line 92. Every row is compared once **per arm**, but line 268 requires exactly `18 * nrows`.

   Both smoke settings have three arms. For the passing initial convergence-check path, the extracted inventory gives **108 rows per setting**: expected comparisons **1,944**, actual comparisons **5,832**.

   **Fix:** move reproduction outside the arm loop and verify unique expected row/time/reference identities.

5. **N5 — MAJOR: the new real-data B0 control can fail simply because the projection error is small.**

   [budget2d.py:159](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget2d.py:159) and `budget3d.py:159` perturb one normalization by only 1%. Their final gates require the **minimum over all evaluated cases/arms/references** to exceed `1e-6` (`budget2d.py:262–265`, `budget3d.py:203–206`).

   A3.5 specifies a **fixed real-data fixture**, not an all-case minimum. For a projection floor of `0.001`, the induced defect is only `1.97e-8`, despite the mismatch being applied correctly.

   **Fix:** pin one real-data fixture and a documented mismatch that produces the required defect; evaluate that control separately from cohort budget statistics.

6. **N6 — MAJOR: A8’s CPU requirement is not enforced by the offline entry points.**

   `score.py:306–309`, `select3d.py:41`, `analyze3d.py:56`, `analyze2d.py:90`, `budget2d.py:63` and `budget3d.py:49–50` proceed into JAX-backed code without setting or checking CPU execution. Direct invocation still depends entirely on the caller supplying `JAX_PLATFORMS=cpu`.

   This matters because A8 records an actual wrong-number GPU failure, and the selection/budget paths do not all require a passing independent score certificate.

   **Fix:** set CPU execution before importing JAX-backed dependencies and assert the backend; also enforce `highest` precision for these offline calculations.

The reviewed Python files parse, but that does not address the reproduced runtime failures. The current smoke configuration also leaves the fractional-step freeze failure untested.

**NOT READY FOR BUDGET SMOKES.**