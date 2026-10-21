All five round-2 issues are **FIXED**.

| Issue | Verification |
|---|---|
| Rank-versus-time interpretation | Rank ratios and FOM/ROM evaluation-time ratios are distinguished; unsupported cost attribution removed. |
| Unbracketed crossover | §6.1 explicitly identifies A3, τ=0.001 as unbracketed within the measured ladder. |
| Glossary definition | \(r_\times\) now uses the cheaper measured tensor/grid form; tensor-only is diagnostic. |
| A4 rationale | §8 states a discretionary stopping decision and budget priorities, without claiming checkpoint rules compelled omission. |
| Local PILOT labels | Previously flagged table captions and physical diagnostics now carry PILOT labels. |

Independently recomputed **PILOT FOM/ROM evaluation-time ratios**, using `time_screen.json`’s `cost_model` at the raw-data-derived \(N_{\min},r_{\min}\), taking the cheapest measured ROM form regardless of accuracy:

| Level | τ=0.03 | τ=0.01 | τ=0.003 | τ=0.001 |
|---|---:|---:|---:|---:|
| A1 | 1.237371 | 0.633863 | 0.533882 | 0.533882 |
| A2 | 0.522128 | 0.498047 | 0.517003 | 0.517003 |
| A3 | 0.971929 | unavailable | unavailable | unavailable |
| AB | 2.348690 | 3.486958 | 2.722622 | 2.046730 |

The new printed values **1.24, 0.52, 0.97** are correct. All three use ROM-Q with \(M_q=16\); the report correctly warns that these evaluation costs do not require accuracy qualification.

Raw feasible minima, cost-model repetition medians, and crossover calculations agree with the analysis. At A3’s tightest target, every measured cheapest-form ROM evaluation remains cheaper than the FOM evaluation, confirming the unbracketed status.

No new blocking error found. No files written.

OVERALL: RESULTS-CONFIRMED