**Corrections remain necessary.** A2’s prior blockers are fixed; A3’s analysis failures are fixed. A1 input certification and report generation still need work.

Read-only audit: light, single-threaded, in-memory CPU probes only; no JAX, jobs, file changes, lab-log writes, or scratch created. Locations below are relative to `experiments/imp-fronts/`.

| Audit-2 finding | Verdict | Fixed? Evidence / remaining fix |
|---|---|---|
| A2 eligible-reference denominator | **CORRECT** | Yes. `a2_wall.py:141–180` enumerates references independently; missing ROMs and unavailable curvature retain reference/tail diagnostics. |
| A2 missing/empty fixture | **CORRECT** | Yes. `:215–219` marks the fixture unavailable without blocking the independently gated tail decision. |
| A1 sensitivity clipping | **CORRECT** | Yes. `a1_smear.py:253–259` preserves raw sensitivity floors and lists clipped bases. |
| A1 input binding, including demonstrated NaN descriptors | **WRONG** | Partial. `:147–155` rejects non-finite `nu/a`, wrong shapes/resolutions and recorded wrong settings. But absent settings remain accepted; `cx/cy/w` remain unchecked. See below. |
| A3 secondary scope / undefined verdict | **CORRECT** | Yes. `a3_trunc.py:96–111` defines both control-gated verdicts, uses one secondary key and the registered primary wording. |
| A3 missing/non-finite reproduction evidence | **CORRECT** | Yes. `:46–65` treats missing/non-finite recorded or recomputed regime errors as failed evidence. |
| Shared accounting / A1 empty handling | **NEEDS-RESTATEMENT** | Partial. A1 records G2 failures and emits an empty-population result, but that result omits provenance/accounting and breaks the report. A3 counts only archive-flagged invalid runs, not non-finite coefficients (`:41–48`). |
| A3 empty reductions / missing primary | **CORRECT** | Yes for analysis. `:54–65,87–106` avoids empty reductions and returns INCONCLUSIVE; report consumers remain incompatible. |
| A2 per-case curvature withholding | **CORRECT** | Yes. `a2_wall.py:226–232` removes reportable curvature and retains explicitly labelled unreportable diagnostics. |
| A2 effective validity / fixture validity | **CORRECT** | Yes. `:156–171,216` records rejection reasons and excludes invalid fixture runs. |
| Previously correct shared numerics / A4 | **CORRECT** | Direct tail sums, A1 primary `q`, A2 projection cutoff, Richardson formula and independent decision controls remain intact. A4’s SVD unions, tolerance and null-control gating match the amendments. |

Additional findings and required fixes:

- **WRONG — A1 accepts uncertified generator recipes.** An in-memory probe of `verify_inputs` returned `ok=True` with no saved settings (`a1_smear.py:153–168`). A clean analysis environment and small residuals cannot establish the historical Newton/linear-solver settings. Require persisted settings or independently verified generation provenance; otherwise mark the evidence uncertified.
- **WRONG — A1 sensitivity can certify the wrong IC.** Saved `cx/cy/w` are ignored, and G1 compares only the 257-node fields (`:149–152,182–201`). A finite, converged 256-node rollout for another centre/width passes validation and changes the reported sensitivity. Validate every descriptor and the native-grid sensitivity IC; reject non-finite descriptors.
- **NEEDS-RESTATEMENT — report rejects supported unavailable outcomes.** In-memory probes reproduced A1 `KeyError('controls')`, A3 missing-primary `KeyError`, and A4 empty-C1 `TypeError`. Fix `make_report.py:65–87,128,203–216,243–248,298–304` to render unavailable sections and exports safely. Existing saved checks also failed rendering at `:218` with missing `verdict_regime_scope`; regenerate or explicitly validate their schema.
- **WRONG — NaN becomes a reported infinity.** `make_report.py:24–28` renders `pct(NaN)` as `∞` (reproduced). Distinguish unavailable/NaN from genuine infinite error.
- **NEEDS-RESTATEMENT — numerical provenance is incomplete.** Tables generally derive measurements from checks, but `:255–259` hard-codes “24 of 38” and “3 %”; dimensions, thresholds and GPU accounting are also literals. Put the required numbers and historical evidence in checks JSON and render from those records.
- **NEEDS-RESTATEMENT — unconditional historical assertions.** `:255–261` always claims defects were fixed before reported runs and debugging outputs were discarded/regenerated. Source those assertions from run provenance. The A4 sine-versus-bank claim is correctly conditional (`:247–248`).
- **NEEDS-RESTATEMENT — report requirements only partly satisfied.** Title, interim status, Mermaid and glossary exist. Add inline provisional/conditional labels beside A2–A4 numerical tables; convert glossary equations at `:272–289` to LaTeX; define omitted terms/columns including β, Spearman, N0/Ap/Am/B1/B2 and containment violation.

No production numerical result or expensive bank decomposition was replayed.