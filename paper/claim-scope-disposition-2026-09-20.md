# Manuscript claim-scope disposition

This review repairs unsupported general statements while preserving every measured scientific value and the user-approved title. The refreshed three-dimensional appendix remains provisional development evidence.

| Finding | Disposition |
| --- | --- |
| operator resolution scope | State the controlled comparison: at a fixed evaluation grid and inference procedure, each frozen operator contributes one operating point; resolution changes can change both numerical error and cost. |
| scheduled ladder checkpoint | Describe one frozen decoder with a prescribed test-count schedule; correction directions/solver assets are prepared offline but network weights do not retrain across the ladder. |
| linear rom guarantees | Say linear projection ROMs restrict the full-order equations to a lower-dimensional trial space. |
| only accuracy control | At a fixed grid and residual test space, correction rank changes the trial space. Describe tolerance and certified quadrature as numerical-cost controls in the measured accurate regime. |
| any linear map | Limit to the tested Burgers affine/quadratic/POD comparators and achieved fits; do not claim superiority over every possible linear subspace. |
| reduction vs physical | Say correction rank chiefly changes reduction error in these panels, while discretization error limits the gain against the fine reference. |

The abstract has two explicit wording corrections: its opening now conditions an operator operating point on the evaluation grid and inference procedure and permits resolution-dependent error and cost; its scheduled ladder now identifies one frozen decoder with a prescribed test-count schedule. It is not recorded as unchanged. All numerical macro definitions and historical table cells are preserved.

The accompanying JSON records each exact old/new passage, reviewed-source hashes, and the abstract hashes. Repeated versions of the same overstatement in the introduction, related work and results are corrected too. No new literature result, numerical experiment, optimality certificate or final-test result is inferred from these edits.

## Glossary

- **Claim scope:** the conditions under which a statement is supported.
- **Frozen:** neural weights remain unchanged during evaluation.
- **Scheduled ladder:** correction ranks evaluated with a prescribed residual-test count at each rank.
- **Trial space / test space:** representable solution fields / functions used to measure the residual.
- **Affine comparator:** a fitted linear map with a constant offset, evaluated under the recorded protocol.
- **Reduction / discretization error:** discrepancy from the discrete equations / discrepancy due to approximating the continuous equations on a mesh.
- **Provisional development evidence:** measurements available during model selection and not promoted to final-test claims.
