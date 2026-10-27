# Fixed larger-support POD construction

Final results for the single predeclared training-only support extension. The constructor failed its frozen acceptance predicates; no new-support trajectory validation was released.

| construction quantity | saved result |
|---|---:|
| target support | 16384 |
| terminal support | 13673 |
| candidate nodes | 32768 |
| relative training residual | 0.00262714686803 |
| banned nodes | 1099 |
| banned fraction | 3.353882% |
| maximum permitted banned fraction | 1.000000% |
| maximum off-support gradient | 2.15572911459e-07 |
| KKT gradient tolerance | 1.01705207429e-08 |
| QR drift quality passed | True |
| GPU-hours | 0.92 |

The global KKT certificate failed and the banned-node fraction exceeded its quality limit. The fixed path stopped without reaching its target or a certified earlier optimum. No retry, alternate support, new-support accuracy score or trajectory admission claim follows from this run.

The independent reviewer recomputed the residual and complete candidate-gradient vector in NumPy from the retained contribution matrix and terminal weights, corroborating the recorded failure. The exact matrix, target, candidate nodes, descriptors, terminal arrays and source capsule are retained with checksums. The large matrix is preserved locally outside Git and must survive any later worktree archival.

The separately qualified full-residual POD fallback remains available. A qualified hyper-reduced POD comparison remains unresolved. This failed constructor does not establish that POD loses to the coordinate bank.

## Glossary

- support: the grid nodes assigned positive fitting weights.
- candidate nodes: the fixed pool from which the greedy path could select support.
- relative training residual: the remaining fitting discrepancy divided by the target norm.
- banned nodes: candidates excluded after the frozen number of active-set drops.
- NNLS: nonnegative least squares, the offline fitting problem for positive weights.
- KKT certificate: a check that active and excluded candidate gradients satisfy first-order optimality conditions.
- off-support gradient: a candidate-gradient component at a node outside the terminal support.
- QR drift: disagreement in the updated matrix factorization used by the fit.
- GPU-hours: allocated GPU time, including setup and the stopped fit.
- full-residual POD: a reduced linear model whose online objective uses every grid residual row.
- hyper-reduced POD: a reduced linear model using weighted selected residual rows.
