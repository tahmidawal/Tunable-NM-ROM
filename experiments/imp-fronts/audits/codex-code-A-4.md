**No new WRONG finding in the audited fixes; several NEEDS-RESTATEMENT items remain.** Report generation still fails on supported unavailable outcomes.

Read-only review against audit #3 and DESIGN Amendments 1–8. Single-threaded, in-memory CPU probes only; no JAX, jobs, file changes, scratch files, or lab-log writes. Paths below are relative to `experiments/imp-fronts/`.

| Item | Verdict | Evidence / remaining fix |
|---|---|---|
| A1 generator assertions and persisted recipe | **CORRECT** | `a1_smear.py:48–70` asserts the registered settings and saves settings plus all five descriptors. |
| A1 rejects unpersisted settings | **CORRECT** | `:165–171` makes missing settings an explicit certification failure. |
| A1 descriptor and IC binding | **CORRECT** | `:148–177` checks `cx/cy/w` and recomputes the IC at both resolutions; downstream G2 rejects non-finite saved fields. The NumPy IC matches the incumbent formula. |
| A1 certification explanation | **NEEDS-RESTATEMENT** | `:184–187` still claims old files without settings are certified by hashes/environment/G2, contradicting rejection above. Replace this stale note. |
| A3 invalid-run accounting | **CORRECT** | `a3_trunc.py:41–43` counts both archive-invalid runs and non-finite coefficients before excluding them. |
| A2 retracted FD replay | **CORRECT** | `a2_wall.py:89–96,192–194,248–253` implements the original curvature-ratio comparison; replay results do not enter the current decision. |
| A2 replay disclosure scope | **NEEDS-RESTATEMENT** | The report calls this the original “control,” but only its curvature condition is replayed; the original also required a median β condition. Label it “curvature condition of the retracted control.” |
| Shared `run_commit` | **CORRECT** | `fcommon.py:174–189` adds commit and Python-code dirty status to newly written checks. Existing saved A3/A4 checks lack this metadata; it does not retroactively certify them. |
| NaN rendering | **CORRECT** | Probe: `pct(NaN)` → `n/a (NaN)`, `pct(None)` → `n/a`, `pct(+∞)` → `∞`. |
| Unavailable report/export handling | **NEEDS-RESTATEMENT** | Probes reproduce A1-empty `TypeError` at `make_report.py:343`, A3-missing-primary `KeyError` at `:348`, and A3-empty-scope `KeyError('worst')` at `:240–241`. Guard exports and render absent `worst` as unavailable. |
| Existing-check compatibility | **NEEDS-RESTATEMENT** | Saved A3 checks still lack `verdict_regime_scope`, required at `:243,347`. Regenerate checks or reject incompatible schemas before rendering. |
| Numbers sourced from JSON | **NEEDS-RESTATEMENT** | Historical counts and path gaps now come from JSON. Dimensions, thresholds, T6 denominator and GPU accounting remain literals. Persist these settings/accounting and render them from records. |
| Conditional historical disclosures | **NEEDS-RESTATEMENT** | `:294–296` unconditionally claims fixes and regeneration with persisted settings—even with A1 absent/unavailable. Gate the claim on recorded evidence. |
| Inline labels and glossary | **NEEDS-RESTATEMENT** | Main A2–A4 tables now carry conditional labels, and several glossary formulas use LaTeX. A2’s cohort table (`:218`) remains unlabelled; glossary expressions such as `s = a/ν` and `Δt = 0.005` remain outside math delimiters. |

No production numerical result or expensive decomposition was replayed.