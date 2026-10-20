**NOT READY FOR BUDGET SMOKES.**

Reviewed at `96d07566b`, against DESIGN A1–A9. No files modified, project modules imported, or GPU jobs launched. All nine requested files parse successfully. Reproductions below execute extracted source with synthetic inputs; they do not establish that either smoke will diverge.

Paths below are relative to `experiments/jcp-combined/`.

**Previous-audit dispositions**

| Item | Status | Evidence |
|---|---|---|
| 5 | **PARTIALLY** | The orthogonal pair now requires verified `star` (`budget2d.py:202`). Offline 3D B2 requires both trajectories valid (`budget3d.py:242`). However, 2D B2 discards entirely non-finite controls (`budget2d.py:284`); see N13 below. |
| 6 | **RESOLVED** | Required 3D convergence specs now come from `bud_specs`; missing either trajectory or required cases fails the check (`budget3d.py:114–134`). |
| 7 | **PARTIALLY** | Production budgets require `--grid` (`budget2d.py:70`, `budget3d.py:57`), and staging requires frozen binding (`cluster/stage.py:68–72`). The 3D expected grid population is configuration-derived (`budget3d.py:276–277`). **2D still derives it from surviving output** (`budget2d.py:313–317`). Conditional fallback generation remains absent (`make_budget_configs.py:24–45`). |
| 9 | **RESOLVED** | Finite tighter-replay sensitivity exceeding the threshold now suppresses solver distance, including the comparator’s diagnostic (`analyze3d.py:116–124`). |
| 13 | **PARTIALLY** | Grid coverage uses the configured inventory, but **budget case coverage still comes from archived case arrays** (`score.py:313–325`). Deleting a required trajectory, its case array and corresponding rows produces `inventory_mismatch=0` in an extracted-code probe. |
| 15 | **RESOLVED** | H2b now uses the endpoint-aware comparison (`analyze2d.py:302–304`). 3D reads and applies frozen selections and records their hash (`analyze3d.py:157–160`). |
| N7 | **PARTIALLY** | Per-time reproduction handles JSON nulls (`budget2d.py:110–117`, `budget3d.py:104–111`, `score.py:156–162,282–286`). Negative-control arithmetic still crashes on null aggregate scores; see N12. |
| N8 | **PARTIALLY** | Offline B2 now compares control against converged baseline (`budget2d.py:279–291`, `budget3d.py:239–247`), as A9 requires. The 2D completeness predicate remains incorrect; see N13. |
| N9 | **PARTIALLY** | The supplied smoke’s finalist-only case 2 is excluded from budget summaries (`budget3d.py:192–199`). Incomplete comparisons within the budget population still receive numerical “largest” classifications (`budget3d.py:210–214`); see N15. |
| N10 | **RESOLVED** | 3D evaluates baseline B3 and withholds time attribution when non-discriminating (`budget3d.py:238,248–257`). Missing-population certification remains a separate issue below. |
| N11 | **RESOLVED under A9.6** | Budget analyzers now implement reproduction and negative controls (`budget2d.py:95–130,308–312`; `budget3d.py:93–111,258–267,285–288`). `score2d` still cannot read `WB_*`, but A9 explicitly assigns budget G-score to the budget analyzers. Their remaining defects follow. |

**New failure paths**

**N12 — BLOCKER: the new budget negative controls crash on an invalid baseline’s serialized score.**

At [budget2d.py:127](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget2d.py:127), every control subtracts `r0['e_ST']`. At [budget3d.py:266](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget3d.py:266), the equivalent operand is `r0['e_ref']`. Neither handles `None`.

Executing those actual assignments with an invalid baseline’s JSON null produces, in both dimensions:

```text
TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
```

The ordinary reproduction loop now accepts the invalid pair, but the subsequent control crashes before the analyzer writes its result. `score.py:230–232,298–299` retains the same pattern.

Handle unavailable control scores explicitly and record failed/unavailable certification without crashing.

**N13 — MAJOR: 2D B2 can certify a mostly invalid control population.**

[budget2d.py:284](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget2d.py:284) skips an entirely NaN control trajectory before checking its row’s validity. This confuses an unrun case outside the check population with a required control that diverged.

Extracted lines 279–291, using the smoke’s three required check cases, one valid control distance of `0.001`, and two invalid all-NaN controls, return:

```text
per_case: [0.001]
complete: True
discriminating: True
```

A9.1 requires every control case valid. This result incorrectly permits quadrature attribution.

Construct the check population from configuration; retain missing/divergent required cases as invalid.

**N14 — MAJOR: budget G-score can pass after a result row disappears.**

Both analyzers iterate archive entries but silently skip absent JSON rows:

- `budget2d.py:103–105`
- `budget3d.py:98–100`

Their expected comparison counts then use the surviving `len(res['rows'])`, rather than an independently required inventory (`budget2d.py:304–311`; `budget3d.py:275,285–287`).

Executing the actual gate expressions with **100 archived trajectory/case entries but 99 JSON rows**, matching surviving comparisons and passing controls, returns:

```text
budget2d reproduction passed: True
budget3d reproduction passed: True
```

In 3D, removing a deployed-rule within-scheme-refinement row leaves that diagnostic unavailable without necessarily failing convergence or finalist checks. The missing evidence nevertheless receives passing G-score certification.

Require configured inventory, archive entries and JSON rows to agree, including case IDs and duplicate counts. This also closes the item-13 budget deletion bypass.

Separately, extracted `budget2d.py:313` confirms that removing one of three archived cases from both smoke settings shrinks expected grid comparisons **18 → 12**. A9.5 requires the configured population to remain fixed.

**N15 — MAJOR: invalid budget diagnostics still manufacture “largest component” classifications.**

At [budget3d.py:212](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget3d.py:212) and `budget2d.py:237`, unavailable components are discarded before choosing the maximum.

Executing the actual classification blocks with `delta_quad=1e-8` and every competing component unavailable returns:

```text
{'delta_quad': 1}
```

The finalist-only population bug is fixed, but a budget case with unverified ideal/time/solver trajectories still gets classified against an incomplete set. No comparison establishes quadrature as its largest component.

Count incomplete comparisons as unavailable; retain individual measured distances separately.

No additional unconditional startup crash was established in the supplied `c2bud`, `c3run` budget-mode, or staging paths. The demonstrated analyzer crash and false certification paths remain sufficient to reject readiness.

**NOT READY FOR BUDGET SMOKES.**