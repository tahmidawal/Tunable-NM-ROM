**JOB GATE: SUBMIT `c3budW1024` and `c3budA`.** The round-1 selection blocker is resolved.

| Audit item | Verdict | Finding |
|---|---|---|
| A15 retraction and correction | **CORRECT** | Explicitly retracts the broader fast selection and restores H4a’s constraints. The `budget3d.py` correction concerns G-score case-swap/time-shift controls. |
| `analyze3d.py` | **CORRECT** | Fast-equal-accuracy selects from H4a’s candidates; none yields `null`/unavailable. Broader equal-accuracy choice is explicitly diagnostic. |
| Re-run `analysis3d-c3grid.json` | **CORRECT** | Independently recomputed selections agree. All five fast selections unavailable; H4a fails and H4b passes throughout. Numerical arm data, previous H4 fields, accurate selections and H6 are unchanged. |
| Re-frozen selection | **CORRECT** | New SHA `8d106e19…` binds the corrected analysis. Retracted `f47442d9…` is preserved byte-for-byte against commit `52779085e`. Non-solver freeze sections unchanged. |
| Both budget configs | **CORRECT** | Exactly match frozen arms, roles and selected flags; bound to the new SHA. Include BE baseline, H4b witness, accurate-equal-cost and CN diagnostics; no fast-selection role. Coincident roles correctly share an arm. |
| Both inventory certificates | **CORRECT** | Committed-source/config hashes and smoke gates verify. Resource projections reproduce exactly. |
| A14 fallback outcome | **CORRECT** | In-memory regeneration exactly reproduces `fallback2d.json`: all three fast fallbacks qualify; all three accurate fallbacks fail cost, with ratios **1.30865, 1.56126, 1.49389**. |
| A14.4 | **CORRECT** | Applicable grid reproduction covers **114 = 3 × 38** comparisons, zero failures. The missing GAL-CN diagnostic has no grid twin; the original **114/152, failed** record remains preserved. |

| Job | Allocation | Projected hours, including margin | Peak GB | Verdict |
|---|---|---:|---:|---|
| `c3budW1024` | H200 / 48 h / 200G | 38.1959 | 103.7275 | **SUBMIT** |
| `c3budA` | H200 / 48 h / 200G | 20.5132 | 104.7625 | **SUBMIT** |

The inherited **POST HOC quadrature-selection** and provisional reference labels remain applicable. No files written or jobs submitted.