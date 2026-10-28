# G2 case matrix and resource projection (draft)

Generated from `plan.py`; exact sizes and execution are not released. No G2 numerical result exists.

| Case | h/h0 | P velocity / pressure | Geometry | Element multiplier | DOF multiplier | Full-rule multiplier |
|---|---:|---|---:|---:|---:|---:|
| empty_channel_control | 1 | 3/2 | 3 | 1 | 1 | 1 |
| manufactured_h0 | 1 | 3/2 | 3 | 1 | 1 | 1 |
| dfg_h0 | 1 | 3/2 | 3 | 1 | 1 | 1 |
| manufactured_h1 | 0.5 | 3/2 | 3 | 4 | 4 | 4 |
| dfg_h1 | 0.5 | 3/2 | 3 | 4 | 4 | 4 |
| manufactured_h2 | 0.25 | 3/2 | 3 | 16 | 16 | 16 |
| dfg_h2 | 0.25 | 3/2 | 3 | 16 | 16 | 16 |
| dfg_p4_g4 | 0.25 | 4/3 | 4 | 16 | 29.82 | 16 |
| dfg_p3_g4 | 0.25 | 3/2 | 4 | 16 | 16 | 16 |
| dfg_solver_100x | 0.25 | 3/2 | 3 | 16 | 16 | 16 |
| dfg_quadrature_plus4 | 0.25 | 3/2 | 3 | 16 | 16 | 26.45 |
| upstream_extension | 0.25 | 3/2 | 3 | 17.45 | 17.45 | 17.45 |
| downstream_extension | 0.25 | 3/2 | 3 | 32 | 32 | 32 |
| symmetric_outlet_long_domain | 0.25 | 3/2 | 3 | 32 | 32 | 32 |

Absolute estimates are in the JSON when an explicit collected G1 baseline is supplied; refined counts and actual peak memory remain unmeasured. Multipliers are mesh-scaling estimates, not measured counts or memory guarantees. The existing adapter caps remain unchanged. A cap violation requires a reviewed resource-only amendment or separately qualified rule, never silent coarsening or quadrature weakening.

## Glossary

h/h0: proposed mesh-size ratio to the eventual frozen coarse level. P: finite-element polynomial degree. Geometry: curved-mapping polynomial order. DOF: one solver coefficient. Full rule: all positive mapped element quadrature points. Multiplier: estimated work/storage growth relative to the coarse case. Draft: no numerical execution is authorized.
