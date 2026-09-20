# Local implementation checks for the NS3D pilot

These are bounded local GPU smoke checks, not paper benchmark results. Scientific
verification and model training run on the cluster under the committed pilot.

`local_reference.json` preserves the failed original unscaled-flow resolution
screen. `local_reference_amended.json` checks the lower-amplitude benchmark on
one verification case. It cannot substitute for the complete cluster reference
gate. Neither is a neural-model result.

`local_operator.json` is the first implementation check before the amplitude
amendment. `local_operator_amended.json` includes the additional omitted-axis
negative control. The local field, manufactured-solution and independent RHS
checks pass. The original check was executed before the implementation commit;
it is retained as debugging history, not given a fabricated committed provenance.

`local_model.json` and `local_rollout.json` are the final bounded checks of the
committed learned bank/head implementation and complete query paths. Their tiny
training checkpoints are retained to reproduce the rollout smoke. These test
that differentiation, solenoidal bank projection, exact weak tensors, initial
fitting and actual evolution execute; their deliberately tiny optimization
budgets are not evidence of a trained model's accuracy or convergence.

`MANIFEST.json` records artifact hashes. Cluster source, configuration, submission
and follow-up commands are retained under `../runs/pilot01/`. Do not infer a paper
number or a runtime ratio from these smoke checks.

## Glossary

- **Smoke check:** a short implementation check with deliberately small inputs.
- **RHS:** the velocity's time derivative specified by the numerical PDE.
- **Solenoidal:** divergence-free.
- **Weak tensor:** cached quadratic advection integrated against smooth tests.
- **Reference gate:** a check that discretization error fits the declared budget.
- **Checkpoint:** stored learned parameters and latent states.
- **Provenance:** the source and configuration that produced an artifact.
