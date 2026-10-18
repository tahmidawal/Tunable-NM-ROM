# Manuscript claim scope review

This is a source review of the current manuscript, with proposed wording corrections awaiting the manuscript owner. It changes no measured result.

| Location | Claim issue | Required scope |
| --- | --- | --- |
| `main.tex:91` | Overgeneralizes operator deployment choices and understates measured resolution-dependent accuracy. Existing resolution experiments already show accuracy/cost movement. | State the controlled comparison: at a fixed evaluation grid and inference procedure, each frozen operator contributes one operating point; resolution changes can change both numerical error and cost. |
| `main.tex:102` | Can imply separately trained models for different correction ranks, contradicting the frozen-decoder design. | Describe one frozen decoder with a prescribed test-count schedule; correction directions/solver assets are prepared offline but network weights do not retrain across the ladder. |
| `main.tex:142` | Unqualified guarantee inheritance is not established; reduced stability and approximation depend on projection and trial space. | Say linear projection ROMs restrict the full-order equations to a lower-dimensional trial space. |
| `main.tex:654` | The paper itself measures an effect of test count and records numerical failures under changed solver budgets; exclusivity exceeds evidence. | At a fixed grid and residual test space, correction rank changes the trial space. Describe tolerance and certified quadrature as numerical-cost controls in the measured accurate regime. |
| `main.tex:761` | Universal claim exceeds tested comparators and conflates a training-optimal affine fit with held-out worst-case optimality. | Limit to the tested Burgers affine/quadratic/POD comparators and achieved fits; do not claim superiority over every possible linear subspace. |
| `main.tex:691` | Categorical denial conflicts with the preceding measured physical error variation. Discretization limits physical gains, rather than making them identically absent. | Say correction rank chiefly changes reduction error in these panels, while discretization error limits the gain against the fine reference. |

The accompanying JSON pins the reviewed source and exact excerpts. Final disposition must identify the repaired source commit, retain historical wording in Git, and distinguish wording changes from any new empirical result.

## Glossary

- **Frozen decoder/operator:** a neural model whose trained weights stay unchanged during evaluation.
- **Trial space:** the set of solution fields the reduced model can represent.
- **Residual test space:** the smooth functions used to measure the PDE residual.
- **Projection / affine map:** restriction to a chosen lower-dimensional representation / a linear map with a constant offset.
- **Reduction error / discretization error:** discrepancy from the chosen discrete equations / discrepancy caused by approximating the continuous equations on a grid.
- **Disposition:** the recorded response to each review finding.
