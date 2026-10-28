# Obstacle-flow runtime controls

These are provisional implementation controls. No obstacle reference, learned-bank result, POD comparison or speed claim is admitted by these tables.

The empty channel is an analytic Poiseuille control; its Stokes initialization already solves the steady problem, so it does not demonstrate nonlinear obstacle convergence.

| Channel control | Saved value | Declared threshold |
|---|---:|---:|
| Fluid area error | 1.63702508e-14 | 1e-05 |
| Velocity L2 error | 9.25320993e-15 | 1e-06 |
| Pressure RMS error / U² | 9.65223818e-15 | 1e-06 |
| Scaled free residual | 1.35614253e-14 | 1e-09 |
| Inlet/outlet mass imbalance | 1.69241315e-16 | 1e-07 |
| Scaled divergence | 6.04454502e-14 | 0.001 |
| Sampled essential trace / U | 1.12757482e-15 | 1e-10 |

The repaired 2D run passes 13/13 frozen initial-control checks. Independent review accepts this implementation milestone. These coarse values remain provisional for reference use; refinement and reference qualification are unfinished.

| Repaired control | Scaled residual | Scaled divergence | Sampled analytic trace / U | Flux imbalance |
|---|---:|---:|---:|---:|
| channel | 1.36206887e-14 | 6.03775987e-14 | 1.12757482e-15 | 0 |
| manufactured | 1.29926867e-11 | 1.66242319e-05 | 2.31245193e-06 | 4.55725609e-17 |
| dfg | 1.12005197e-15 | 0.0219928259 | 1.11022302e-15 | 5.07723944e-16 |

Manufactured flux uses all boundaries, since its inlet flux is nearly zero and its analytic wall data are not no-slip. Other cases use inlet/outlet imbalance. The sampled analytic trace is distinct from enforcement of the prescribed FE trace.

| Coarse benchmark force definition | Drag coefficient | Lift coefficient |
|---|---:|---:|
| reaction | 5.57978441083 | 0.0110331987994 |
| gradient_order12 | 5.56525027279 | 0.0165618635001 |
| gradient_order16 | 5.56525027279 | 0.0165618635017 |
| physical_symmetric_order12 | 5.57199095528 | 0.00857141857352 |
| physical_symmetric_order16 | 5.57199095528 | 0.00857141857506 |

The reaction and surface-force estimates disagree on this coarse mesh. All definitions are retained; selecting the most favorable estimate does not qualify a reference. The exact-coordinate pressure difference is 0.116873883245; its absolute gap from the stored benchmark anchor is 0.000646283725. The stricter reference tests remain unpassed.

The normal diagnostic uses a fresh mesh generated with the unchanged geometry recipe. It cannot reconstruct the missing mesh or vectors from the original failed run. The table compares the relative defect in $\int_{\partial\Omega}x\cdot n\,ds=2|\Omega|$.

| Normal evaluation path | Relative divergence-theorem defect |
|---|---:|
| Raw mapped boundary | 0.0175678828 |
| Ordinary boundary integration | 0.0175678828 |
| Adjacent-volume facet integration | 9.80909507e-15 |

Nested volume-facet status: `completed_diagnostic_not_admission`. Normal closure alone is insufficient to establish outward orientation.

The 3D geometry diagnostic completed, but its geometric identity checks fail. These failures block a 3D repair/flow release. Positive Jacobians at the saved interior samples do not rule out inversion elsewhere; curved-map folding remains an unmeasured hypothesis. No correction inferred from the 2D case is admitted here.

| 3D geometric diagnostic | Saved or directly derived value |
|---|---:|
| Fluid-volume error against analytic geometry | 3.50938343e-05 |
| Native exterior-facet divergence defect | 0.000489304609 |
| All-element-boundary divergence defect | 0.000403881548 |
| All-element minus exterior position-normal integral | 0.100506017 |
| Wall facet minus ordinary boundary area | -3.97903932e-13 |
| Wall facet minus ordinary position-normal integral | -0.575698245 |

| Allocation | Lane / attempt | Recorded state | Charged GPU-seconds |
|---|---|---|---:|
| 6566869 | 2d_reference / g1-smoke03 | FAILED | 33 |
| 6567287 | 2d_reference / g1-smoke04 | FAILED | 17 |
| 6567633 | 2d_reference / g1-smoke05 | FAILED | 16 |
| 6567998 | 2d_reference / g1-smoke06 | FAILED | 15 |
| 6610494 | 3d_reference / g1-smoke06 | FAILED | 29 |
| 6610978 | 2d_reference / g1-normal01 | COMPLETED | 25 |
| 6611259 | 3d_reference / g1-normal01 | COMPLETED | 28 |
| 6612209 | 2d_reference / g1-smoke07 | COMPLETED | 31 |
| 6613426 | 2d / g2-native01 | RUNNING | pending accounting |

Total recorded allocation: 194 GPU-seconds (0.0538888889 GPU-hours). Finite-element calculations run on CPU within these GPU allocations; this is resource accounting, not a GPU-acceleration measurement.

## Glossary

L2 error measures spatially integrated field error. RMS means root mean square. U is the declared characteristic velocity. The free residual measures the equations on unconstrained finite-element unknowns. Mass imbalance compares inlet and outlet flux. Divergence measures local incompressibility error. Essential trace is prescribed boundary velocity; a sampled maximum is not a continuous maximum certificate. A facet is a volume element face or edge. An outward-fluid normal points out of the fluid, including into an excluded obstacle. DFG names the established cylinder-in-channel benchmark. A manufactured control uses a chosen analytic solution and its matching forcing and boundary data. Weak reaction estimates force from the discrete equations; gradient and symmetric surface estimates integrate different stress forms and must satisfy the declared consistency checks. Drag and lift coefficients are normalized streamwise and transverse forces. POD is a weighted linear snapshot basis. A reference is an independently qualified numerical solution.
