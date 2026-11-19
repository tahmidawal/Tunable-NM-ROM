# Burgers tree embeddings: field implementation and frozen tangent diagnostics

Packed field equivalence passes (largest change5.00891643e-09%). Stop this tangent ablation; retain the nonlinear representation and its scoped PDE solver failure.

The preceding three-case B2 setting passes physical accuracy and independent spatial, integration, weak-test, time-step and100x-precision checks. This batch tests one equivalent implementation and one specific architecture, under a finite one-GPU-hour allowance. It does not admit all-development PDE trajectories, deployable adaptation, a runtime win, native NM-ROM confirmation or final-test access. The original16.5% failure and all later representation results remain in [the representation report](2026-11-18-tree-embedding-representation.md).

## Equivalent field implementation

These are the unchanged decoder and fine weak PDE equations evaluated with fixed array shapes and runtime tree geometry. Padded arithmetic and storage are charged. The tolerance for field equivalence is0.00001%; physical accuracy and complete solver convergence are checked separately. These same-job component clocks **are diagnostic only**: each method is charged joint model setup, and cold initialization includes compilation. They do not establish a paired speed ratio or a cost improvement over the earlier job.

|Case|Method|Largest field change|Nonconverged steps|Padded field cache MiB|Cold initialization seconds|Diagnostic query seconds|
|---|---|---:|---:|---:|---:|---:|
|dev000|tree_direct24|5.00891643e-09%|0|33.497|6.785|50.850|
|dev000|global_stable960|2.06985338e-10%|0|124.990|2.389|27.005|
|dev003|tree_direct24|3.49513868e-09%|0|33.497|0.012|18.811|
|dev003|global_stable960|9.15699292e-13%|0|124.990|0.010|8.448|
|align002|tree_direct24|2.75719795e-09%|0|33.497|0.012|21.455|
|align002|global_stable960|0%|0|124.990|0.009|8.747|
## Frozen tangent architecture

This is a separately labelled direct-coefficient decoder: the derivative of each frozen child head at zero replaces its nonlinear output. The original nonlinear PDE optimizer remains stopped. Parent, shared weights,64 retained nodes and1152 adjustable variables are unchanged; all coefficients are jointly refitted with the complete physical overlap metric and no ridge/direction removal. Geometry is selected from the initial-field callable before reference access.

Worst physical projection error is0.56851064%; the actual4096 witness is0.56851065% atalign000, t=0.0. Maximum scaled stationarity is7.47044e-15. The representation gate is **False**. An unsuccessful screen is not admitted to PDE solving; a passing screen also requires the retained independent all-snapshot dense audit.

![Frozen nonlinear head and its tangent](b3-frozen-tangent.png)


Latest scheduler-ledger snapshot: 27476 elapsed GPU-seconds (7.632222h); full outstanding reservations bring the charge to27476 seconds against a cumulative ceiling of30789. Active owned jobs: none. Generated 2026-11-19T03:16:38.453377+00:00. Raw outputs and logs are retained with verified hashes before admission. Final-test solution fields remain unopened and the earlier campaign remains paused.

## Glossary

- **Tangent:** the decoder's derivative at zero child codes, used here as a fixed linear correction basis.
- **Curvature:** the change in a nonlinear decoder's directions as its codes move away from zero.
- **Projection:** an offline best-fit reconstruction that may use the reference field; it is not a PDE solve.
- **Stationarity:** the remaining physical objective gradient after fitting. Small values help distinguish limited representation from a poor fit.
- **Equivalence:** the returned field agrees with the unchanged compact implementation to the registered tolerance.
- **Cold initialization:** constructing/fitting the initial state, including compilation in this diagnostic.
- **MiB:**2^20 bytes. Padded cache memory includes zeros introduced to make array shapes independent of tree geometry.
- **Allocated time:** scheduler duration, including compilation, failures and idle portions; it is distinct from online query runtime.
