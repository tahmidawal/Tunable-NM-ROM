# Frozen heat checkpoint: solver tolerance and compiled-query costs

These provisional development findings compare execution paths and solver stopping on the unchanged heat checkpoint. They measure runtime without new training; the nonlinear-head initial-state approximation gap remains separate.

Generated from `experiments/mr-heat2d/runs/pilot02/archive/outputs/results.json` and the native field audit. Scientific source `06f8ed98653e95bbf094c42212171d0db3519541`, GPU job `3350258`, `NVIDIA A100 80GB PCIe`. Frozen checkpoint SHA-256: `c60ebb97c62329c750f86504f0585678248ed1f20f2c054db304e1091bc969c3`.

[Standalone runtime and field-drift figure](figures/heat-runtime.svg).

The complete query transfers a full host initial field to the GPU, fits both initial starts, evolves the weak heat equations and returns all requested full fields to the host. The original modular strict control and efficient direct sine-transform FOM run in this same job. Timed order alternates, every paired block is preceded by clock burn-in, and compilation/reference/mesh setup are outside query costs.

## Frozen settings and evidence checks

| Setting | Value |
|---|---|
| Solver meshes / shared observation intervals | [64, 128] / 64 |
| Validation cases / repetitions / timed invocations | 4 / 7 / 504 |
| CN timestep / tolerance ladder | 0.025 / [1e-09, 1e-07, 1e-06, 1e-05] |
| Verified files / unique output and reference fields | 81 / 60 |
| Maximum saved-metric disagreement | 3.88578e-16 |
| Empirical reference-refinement estimate | 1.81003e-10 |
| Rigorous continuum error bound | Not established |

Every field is restricted to the same observation grid. Current-field relative, initial-field relative and absolute L2 errors are independently recomputed for every repetition. The empirical adjusted ratio is $(e+\delta)/(1-\delta)$; this refinement estimate is not a rigorous continuum bound. All original fresh reference and heat-advancement gates pass again.

## Paired query costs and unchanged-model accuracy

Times are cohort medians of within-case repetition medians. Ratios are the median across cases of each case's median paired-repetition ratio. The strict-control ratio compares the original modular strict arm with the listed method. All errors use the worst requested time and repetition for each case.

| Intervals | Method | Query ms | FOM / method | Strict control / method | Current error median / worst | Cases above 5% |
|---|---|---|---|---|---|---|
| 64 | fom_dst_exact_time | 1.19722 | 1 | 24.0805 | 0.000675205 / 0.000709211 | 0 / 4 |
| 64 | rom_modular_gtol1e-09 | 28.3231 | 0.0421416 | 1 | 0.0585754 / 0.0973564 | 3 / 4 |
| 64 | rom_compiled_gtol1e-09 | 27.2923 | 0.0435725 | 1.03551 | 0.0585754 / 0.0973564 | 3 / 4 |
| 64 | rom_modular_gtol1e-07 | 20.4582 | 0.0582387 | 1.36752 | 0.0585754 / 0.0973564 | 3 / 4 |
| 64 | rom_compiled_gtol1e-07 | 19.802 | 0.0606191 | 1.40938 | 0.0585754 / 0.0973564 | 3 / 4 |
| 64 | rom_modular_gtol1e-06 | 17.5896 | 0.0671605 | 1.57776 | 0.0585764 / 0.0973564 | 3 / 4 |
| 64 | rom_compiled_gtol1e-06 | 17.4532 | 0.0698346 | 1.60381 | 0.0585764 / 0.0973564 | 3 / 4 |
| 64 | rom_modular_gtol1e-05 | 14.6069 | 0.0808643 | 1.87879 | 0.0585779 / 0.0973565 | 3 / 4 |
| 64 | rom_compiled_gtol1e-05 | 14.1774 | 0.0847567 | 1.94677 | 0.0585779 / 0.0973565 | 3 / 4 |
| 128 | fom_dst_exact_time | 1.37571 | 1 | 20.1622 | 0.000168684 / 0.000177175 | 0 / 4 |
| 128 | rom_modular_gtol1e-09 | 28.8208 | 0.0499322 | 1 | 0.0585451 / 0.0973564 | 3 / 4 |
| 128 | rom_compiled_gtol1e-09 | 27.0112 | 0.0528193 | 1.05584 | 0.0585451 / 0.0973564 | 3 / 4 |
| 128 | rom_modular_gtol1e-07 | 20.4475 | 0.0692056 | 1.35705 | 0.0585451 / 0.0973564 | 3 / 4 |
| 128 | rom_compiled_gtol1e-07 | 19.6988 | 0.0718167 | 1.43559 | 0.0585451 / 0.0973564 | 3 / 4 |
| 128 | rom_modular_gtol1e-06 | 17.7691 | 0.0794282 | 1.54296 | 0.0585462 / 0.0973564 | 3 / 4 |
| 128 | rom_compiled_gtol1e-06 | 16.8944 | 0.0784496 | 1.63578 | 0.0585462 / 0.0973564 | 3 / 4 |
| 128 | rom_modular_gtol1e-05 | 14.6986 | 0.0910642 | 1.89052 | 0.0585477 / 0.0973565 | 3 / 4 |
| 128 | rom_compiled_gtol1e-05 | 14.2679 | 0.0959703 | 1.96503 | 0.0585477 / 0.0973565 | 3 / 4 |

## Execution parity and solver validity

Compiling the full query is an equivalent-execution claim only when fields, latent states and exact attempt/acceptance/reason counters agree for that tolerance. Every timed repetition is checked; a failed parity arm remains visible and cannot establish that claim.

| Intervals | Method | All solver checks pass | All parity checks pass | Maximum field drift from strict | Drift ceiling passes | Empirical eligible targets |
|---|---|---|---|---|---|---|
| 64 | fom_dst_exact_time | True | True | 0 | True | [0.1, 0.05, 0.01, 0.001] |
| 64 | rom_modular_gtol1e-09 | True | True | 0 | True | [0.1] |
| 64 | rom_compiled_gtol1e-09 | True | True | 0 | True | [0.1] |
| 64 | rom_modular_gtol1e-07 | True | True | 7.05817e-06 | True | [0.1] |
| 64 | rom_compiled_gtol1e-07 | True | True | 7.05817e-06 | True | [0.1] |
| 64 | rom_modular_gtol1e-06 | True | True | 3.82707e-05 | True | [0.1] |
| 64 | rom_compiled_gtol1e-06 | True | True | 3.82707e-05 | True | [0.1] |
| 64 | rom_modular_gtol1e-05 | True | True | 0.000347272 | True | [0.1] |
| 64 | rom_compiled_gtol1e-05 | True | True | 0.000347272 | True | [0.1] |
| 128 | fom_dst_exact_time | True | True | 0 | True | [0.1, 0.05, 0.01, 0.001] |
| 128 | rom_modular_gtol1e-09 | True | True | 0 | True | [0.1] |
| 128 | rom_compiled_gtol1e-09 | True | True | 0 | True | [0.1] |
| 128 | rom_modular_gtol1e-07 | True | True | 7.25728e-06 | True | [0.1] |
| 128 | rom_compiled_gtol1e-07 | True | True | 7.25728e-06 | True | [0.1] |
| 128 | rom_modular_gtol1e-06 | True | True | 3.97909e-05 | True | [0.1] |
| 128 | rom_compiled_gtol1e-06 | True | True | 3.97909e-05 | True | [0.1] |
| 128 | rom_modular_gtol1e-05 | True | True | 0.000347323 | True | [0.1] |
| 128 | rom_compiled_gtol1e-05 | True | True | 0.000347323 | True | [0.1] |

Eligibility here requires finite fields, selected initial and rollout gradients meeting the arm's declared tolerance, no budget exhaustion, empirical reference margin and compiled-path parity. It remains development evidence. The drift ceiling is checked separately on both field differences and changes in error, so nearly unchanged error cannot conceal a changed trajectory. A method can meet the broad target yet fail the stricter accuracy-preservation check.

| Intervals | Method | Initial fit ms | Evolution or direct FOM readout ms | Compiled device query ms | ROM readout ms | Median total initial / rollout attempts |
|---|---|---|---|---|---|---|
| 64 | fom_dst_exact_time | — | 0.341032 | — | — | 0 / 0 |
| 64 | rom_modular_gtol1e-09 | 6.50629 | 20.1623 | — | 0.202203 | 69 / 111.5 |
| 64 | rom_compiled_gtol1e-09 | — | — | 26.6389 | — | 69 / 111.5 |
| 64 | rom_modular_gtol1e-07 | 5.4142 | 14.7142 | — | 0.24853 | 55 / 78.5 |
| 64 | rom_compiled_gtol1e-07 | — | — | 19.2512 | — | 55 / 78.5 |
| 64 | rom_modular_gtol1e-06 | 4.71463 | 12.8354 | — | 0.298775 | 48.5 / 66.5 |
| 64 | rom_compiled_gtol1e-06 | — | — | 16.942 | — | 48.5 / 66.5 |
| 64 | rom_modular_gtol1e-05 | 4.19181 | 10.7878 | — | 0.349003 | 41 / 54 |
| 64 | rom_compiled_gtol1e-05 | — | — | 13.6247 | — | 41 / 54 |
| 128 | fom_dst_exact_time | — | 0.399352 | — | — | 0 / 0 |
| 128 | rom_modular_gtol1e-09 | 6.39623 | 20.1681 | — | 0.214906 | 69 / 112 |
| 128 | rom_compiled_gtol1e-09 | — | — | 26.1516 | — | 69 / 112 |
| 128 | rom_modular_gtol1e-07 | 5.33911 | 14.7699 | — | 0.275006 | 55 / 77.5 |
| 128 | rom_compiled_gtol1e-07 | — | — | 19.2566 | — | 55 / 77.5 |
| 128 | rom_modular_gtol1e-06 | 4.58361 | 12.8002 | — | 0.253536 | 48.5 / 66.5 |
| 128 | rom_compiled_gtol1e-06 | — | — | 16.6902 | — | 48.5 / 66.5 |
| 128 | rom_modular_gtol1e-05 | 4.10195 | 10.7573 | — | 0.290184 | 41 / 54 |
| 128 | rom_compiled_gtol1e-05 | — | — | 13.3698 | — | 41 / 54 |

The direct FOM combines propagation and inverse-transform field readout; its phase is retained in raw JSON. Compiled paths have one inseparable device-query phase. A dash means an unavailable separate phase. Initial attempt counts sum both starts; rollout counts sum every actual timestep.

## Scope and next accuracy question

No network weights, latent or bank dimensions, training cases or spatial family changed. Runtime changes therefore cannot remedy the head's initial-field approximation gap. A next bounded proposal would freeze the spatial bank and existing head architecture while comparing head/code refinement on the original training cohort against expanded training coverage. This training experiment has not been launched. Broader multi-bump heat coverage and independent final confirmation remain open; no merge occurred.

## Plain-language glossary

- **FOM / ROM / CN / LM:** full-grid sine solver / learned reduced solver / Crank–Nicolson timestep / damped nonlinear least-squares iteration.
- **Modular / compiled / checkpoint:** separate staged query calls / one compiled composition of those calls / saved unchanged neural weights.
- **Tolerance / solver validity / parity:** required normalized gradient smallness / meeting declared finite/convergence checks / agreeing fields, latents and counters.
- **Intervals / observation grid:** spatial cells along an axis / common locations where all meshes' errors are evaluated.
- **Query ms / paired ratio:** full host-input to host-output milliseconds / same-case, same-repetition cost ratio summarized across cases.
- **Current error / initial error / absolute L2:** discrepancy divided by current reference norm / initial norm / spatially integrated discrepancy alone.
- **Median / worst / cases above:** middle cohort value / largest cohort value / number of cases missing the stated error threshold.
- **Strict control / field drift:** original modular tight-tolerance arm / change in its field divided by current physical-reference norm.
- **Empirical adjusted ratio / rigorous bound:** observed error with a refinement estimate and denominator correction / proven upper error limit, unavailable here.
- **Readout / attempt / eligible target:** reconstructing requested fields / tried LM update / development target meeting the stated empirical and solver checks.
- **Source hash / provisional:** content-verification identifier / development finding awaiting independent review and confirmation.
