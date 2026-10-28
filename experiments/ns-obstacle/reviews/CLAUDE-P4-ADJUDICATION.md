# P4 qualification consultation: adjudication in progress

Claude's completed independent response and original packet are retained unchanged. Independent Codex source/evidence checks resolve several apparent blockers already covered by the plan:

- All three p/geometry fields use the complete adopted P4 quadrature rule with common denominators and strict point validity. The finest h pair already uses the adopted P4 denominator. Earlier-pair normalization does not enter the conditional uncertainty constituent.
- Retained P3/g3 and P3/g4 solver kernels and dependency installer are byte-identical; native output helper work did not change these solves. A new solve solely because commit IDs differ is unnecessary. Their residual, trace, mass and Jacobian evidence is retained.
- The trace value is a conservative polynomial bound with an explicit roundoff allowance, not a clamped observed error. Raw estimates, coefficients and check residuals already exist. Generated screen labels/glossaries now identify the bound explicitly.
- The manufactured velocity contains degree-five terms and curved pullbacks; it is not exactly represented by P4. Existing P3 refinement provides convergence evidence. A new non-polynomial manufactured case is optional robustness, not an unmet frozen gate.
- Pointwise radial and normal-angle diagnostics can be derived from retained boundary arrays without changing thresholds. New generated geometry diagnostics retain those arrays and their provenance. Extended-domain remeshing must be disclosed; exact mesh appending was not a frozen requirement. Absolute cross-domain pressure is ineligible; common pole pressure drops are the relevant diagnostic.

The geometry statement is limited: the measured P3 geometry contrast is bounded by twice the maximum common-reference contrast. The factor two is shared with the empirical order/h estimate, not an additional independent geometry margin or a rigorous bound on P4 geometry interaction. Retain all three raw contrasts and label the unmeasured P4 interaction. No automatic P5/g5 or altered uncertainty formula is introduced.

Remaining work before admission: explicit resolved-noise and all-parameter coverage protocol, actual solver/quadrature/domain evidence, and full uncertainty assessment. Remedy branches require separate reviewed amendments if an existing scientific gate fails. No post-P4 source control or advisory review alone releases training.

Operational correction: the coordinator initially described immutable_lift_control as performing a Stokes solve, then corrected this after reading the source. It loads retained stokes.sol and evaluates three assembled residuals, with no solve/factorization. The reviewer's initial claim that Native04 lacked REPAIR.json was also retracted: its destination map already provided that file. The separate reference crossp01 omission was real and is preserved as a failed attempt.
