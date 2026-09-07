# Frozen fresh-wave decoder across spatial meshes

This generated report covers the bounded development pilot of actual frozen-decoder wave evolution and complete-query FOM comparisons. Numerical findings are provisional development evidence; the final cohort remains sealed.

Source `a02aacd9578f46a9bee0dcb093acd35e62d10739`, job `3349951`, device `NVIDIA A100 80GB PCIe`. Native result SHA-256: `d969c32607dacdd0d87e7a411eb8c426ed4dc98fcabe0d993261f753c10e86c5`.

The declared validation seed is `690602`, with indices `[0, 1]`; both boundaries use the same physical parameter draws across meshes. The frozen MLP has latent dimension `16` and bank/weak-test dimension `64`. The physical horizon is `2.4` with observation interval `0.05`. These are development cases, not independent-cohort confirmation.

The input contract is one host displacement field, one host velocity field and scalar wave speed. The output contract is both physical fields on the requested mesh at every observation time, copied to the host. Cold fitting, speed rescaling, evolution, and dense field output are included. There is no external forcing. QR conversion changes coefficient coordinates exactly and preserves the frozen physical decoder; all mesh-specific operator tables are rebuilt.

| Boundary | Intervals | Method | Step/CFL | Query median (s) | Physical error median | Physical error worst | Cases above target |
|---|---:|---|---:|---:|---:|---:|---|
| dirichlet | 256 | rom | 0.0025 | 3.30038 | 0.461654 | 0.553671 | 0.1: 2/2; 0.05: 2/2; 0.01: 2/2; 0.001: 2/2 |
| dirichlet | 256 | rom | 0.00125 | 6.56856 | 0.461654 | 0.55367 | 0.1: 2/2; 0.05: 2/2; 0.01: 2/2; 0.001: 2/2 |
| dirichlet | 256 | dst | 0 | 0.014024 | 0.00446922 | 0.00504742 | 0.1: 0/2; 0.05: 0/2; 0.01: 0/2; 0.001: 2/2 |
| dirichlet | 512 | rom | 0.0025 | 3.35427 | 0.460137 | 0.552043 | 0.1: 2/2; 0.05: 2/2; 0.01: 2/2; 0.001: 2/2 |
| dirichlet | 512 | rom | 0.00125 | 6.61684 | 0.460137 | 0.552043 | 0.1: 2/2; 0.05: 2/2; 0.01: 2/2; 0.001: 2/2 |
| dirichlet | 512 | dst | 0 | 0.0645338 | 0.00127174 | 0.00144462 | 0.1: 0/2; 0.05: 0/2; 0.01: 0/2; 0.001: 2/2 |
| absorbing | 256 | rom | 0.0025 | 3.22126 | 0.0757634 | 0.0765535 | 0.1: 0/2; 0.05: 2/2; 0.01: 2/2; 0.001: 2/2 |
| absorbing | 256 | rom | 0.00125 | 6.40747 | 0.0757634 | 0.0765535 | 0.1: 0/2; 0.05: 2/2; 0.01: 2/2; 0.001: 2/2 |
| absorbing | 256 | rk4 | 0.45 | 0.0949841 | 0.000805024 | 0.000874814 | 0.1: 0/2; 0.05: 0/2; 0.01: 0/2; 0.001: 0/2 |
| absorbing | 256 | rk4 | 0.225 | 0.18081 | 0.000804492 | 0.000874222 | 0.1: 0/2; 0.05: 0/2; 0.01: 0/2; 0.001: 0/2 |
| absorbing | 512 | rom | 0.0025 | 3.27431 | 0.0757295 | 0.0765435 | 0.1: 0/2; 0.05: 2/2; 0.01: 2/2; 0.001: 2/2 |
| absorbing | 512 | rom | 0.00125 | 6.46239 | 0.0757294 | 0.0765435 | 0.1: 0/2; 0.05: 2/2; 0.01: 2/2; 0.001: 2/2 |
| absorbing | 512 | rk4 | 0.45 | 0.297197 | 0.000178887 | 0.000195075 | 0.1: 0/2; 0.05: 0/2; 0.01: 0/2; 0.001: 0/2 |
| absorbing | 512 | rk4 | 0.225 | 0.526869 | 0.000178792 | 0.000194968 | 0.1: 0/2; 0.05: 0/2; 0.01: 0/2; 0.001: 0/2 |

Each error is the maximum over displacement, velocity and energy-state errors, themselves maximized over observation times. Displacement uses its initial norm; velocity and energy state use the initial phase-state energy scale. Case timings are medians of retained repetitions; the table then takes their cohort median. Repetitions do not increase the number of physical cases.

Physical-reference errors above use a common observation grid. Separate native-grid FOM discrepancies remain in the JSON. No qualified speedup is inferred from a raw time ratio when accuracy fails. Reference self-differences and absorber contraction are empirical, so normalized pair records leave a proven uncertainty bound unspecified.

| Boundary | Intervals | ROM step | Input (s) | Cold fit / speed (s) | Evolution (s) | Field output (s) | Nonstationary cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| dirichlet | 256 | 0.0025 | 0.00118518 | 0.0206496 | 3.26886 | 0.0102876 | 0 |
| dirichlet | 256 | 0.00125 | 0.00084423 | 0.0209014 | 6.53478 | 0.0103959 | 0 |
| dirichlet | 512 | 0.0025 | 0.00145253 | 0.020839 | 3.27502 | 0.0577669 | 0 |
| dirichlet | 512 | 0.00125 | 0.00137052 | 0.020697 | 6.54449 | 0.0502297 | 0 |
| absorbing | 256 | 0.0025 | 0.00126498 | 0.0205518 | 3.19162 | 0.00726845 | 0 |
| absorbing | 256 | 0.00125 | 0.000940863 | 0.0200143 | 6.37836 | 0.00719902 | 0 |
| absorbing | 512 | 0.0025 | 0.0014859 | 0.0197259 | 3.19383 | 0.059109 | 0 |
| absorbing | 512 | 0.00125 | 0.00130793 | 0.0212577 | 6.38202 | 0.059334 | 0 |

Cold fitting uses `800` iterations per multistart budget and checks selected-fit stationarity. The historical doubled-budget fitting-stability gate has not been rechecked. Dense cold fitting and requested field output grow with the mesh. Previous training costs are not remeasured, so no break-even query count is asserted.

| Boundary | Intervals | Case | Displacement step difference | Velocity step difference | Energy-state step difference | Refinement pass |
|---|---:|---:|---:|---:|---:|---|
| dirichlet | 256 | 0 | 3.65343e-07 | 8.20147e-07 | 9.9661e-07 | True |
| dirichlet | 256 | 1 | 4.78268e-07 | 7.59326e-07 | 1.03715e-06 | True |
| dirichlet | 512 | 0 | 3.66106e-07 | 8.20274e-07 | 9.95976e-07 | True |
| dirichlet | 512 | 1 | 4.80523e-07 | 7.60374e-07 | 1.04107e-06 | True |
| absorbing | 256 | 0 | 2.97591e-08 | 5.3507e-08 | 6.86374e-08 | True |
| absorbing | 256 | 1 | 5.55618e-08 | 9.88499e-08 | 1.29136e-07 | True |
| absorbing | 512 | 0 | 2.97711e-08 | 5.35419e-08 | 6.86786e-08 | True |
| absorbing | 512 | 1 | 5.55777e-08 | 9.89061e-08 | 1.29186e-07 | True |

The refinement verdict uses the original numerical-refinement target and compares the two declared ROM steps, with displacement and physical tangent velocity in the mesh's exact bank coordinates. This is separate from physical accuracy.

| Boundary | Case | Reference self / finest adjacent difference | Temporal difference | Spatial ratio | Conditional finest-reference estimate |
|---|---:|---:|---:|---:|---:|
| dirichlet | 0 | 3.18671e-11 | unavailable | unavailable | unavailable |
| dirichlet | 1 | 4.45075e-11 | unavailable | unavailable | unavailable |
| absorbing | 0 | 0.000162611 | 9.12954e-10 | 0.270704 | 6.03597e-05 |
| absorbing | 1 | 0.000194961 | 1.21058e-09 | 0.270786 | 7.23976e-05 |

For fixed walls, the reference is independently implemented continuum sine propagation, with spectral mesh self-refinement. For absorption, the reference solves the declared absorbing-boundary PDE, not an infinite-domain replacement. The conditional finest-mesh estimate is

$$\widehat e_f = \frac{r}{1-r}\,d_f + d_t,$$

where $d_f$ is the finest adjacent spatial difference, $r$ is the ratio of fine to coarse adjacent differences, and $d_t$ is the temporal pair difference. This assumes continued contraction and is not a rigorous bound.

| Absorbing intervals | ROM step | Case | Final displacement relative | Final velocity relative | Final energy-state relative | Final displacement absolute | Final reference energy fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0025 | 0 | 2.27032 | 2.26144 | 2.9296 | 0.000835041 | 4.99306e-06 |
| 256 | 0.0025 | 1 | 3.1257 | 1.97934 | 4.10823 | 0.000661065 | 1.73248e-06 |
| 256 | 0.00125 | 0 | 2.27032 | 2.26144 | 2.9296 | 0.000835041 | 4.99306e-06 |
| 256 | 0.00125 | 1 | 3.1257 | 1.97934 | 4.10823 | 0.000661065 | 1.73248e-06 |
| 512 | 0.0025 | 0 | 2.26879 | 2.26142 | 2.92808 | 0.000834478 | 4.99306e-06 |
| 512 | 0.0025 | 1 | 3.12595 | 1.97703 | 4.10705 | 0.00066112 | 1.73248e-06 |
| 512 | 0.00125 | 0 | 2.26879 | 2.26142 | 2.92808 | 0.000834478 | 4.99306e-06 |
| 512 | 0.00125 | 1 | 3.12595 | 1.97703 | 4.10704 | 0.00066112 | 1.73248e-06 |

Current-state normalization exposes errors when absorption leaves a small physical field. Absolute errors and the explicit zero/vanishing flags remain available in `summary.json` and native invocation records. Undefined zero-reference relative errors remain null. Phase and mean-field diagnostics are retained separately in the native result.

| Boundary | Intervals | New-mesh assembly including first compilation (s) | Stored bank/mass/operators (bytes) |
|---|---:|---:|---:|
| dirichlet | 256 | 4.68424 | 33878536 |
| dirichlet | 512 | 4.00139 | 135848456 |
| absorbing | 256 | 4.091 | 34411016 |
| absorbing | 512 | 4.20251 | 136913416 |

The assembly timer includes checkpoint loading, coordinate-network evaluation, QR conversion and discrete mass/stiffness/boundary-damping construction. It recurs for a new mesh. Physical-speed rescaling recurs inside each measured query. Storage above excludes neural weights and requested output arrays, whose hashes and sizes are recorded per invocation.

| Boundary | Intervals | Case | Unrestricted projected displacement error | Primary ROM initial displacement error | Primary ROM evolved displacement error |
|---|---:|---:|---:|---:|---:|
| dirichlet | 256 | 0 | 0.00279454 | 0.0184167 | 0.151039 |
| dirichlet | 256 | 1 | 0.00330664 | 0.028671 | 0.286717 |
| dirichlet | 512 | 0 | 0.00279454 | 0.0184167 | 0.150577 |
| dirichlet | 512 | 1 | 0.00332024 | 0.028671 | 0.285935 |
| absorbing | 256 | 0 | 0.00339792 | 0.0167865 | 0.0281323 |
| absorbing | 256 | 1 | 0.00422979 | 0.0240275 | 0.0387228 |
| absorbing | 512 | 0 | 0.00339521 | 0.0167865 | 0.0280925 |
| absorbing | 512 | 1 | 0.00422754 | 0.0240275 | 0.038715 |

The focused unrestricted-bank projection diagnostic measures the learned spatial span's mass projection of same-grid truth. It is not a dynamical trajectory or an equal-latent-dimensional comparator. Its energy error is not an optimal energy-norm lower bound. Primary ROM columns use the independently refined physical reference. No architecture search or new training was performed.

## Plain-language glossary

**Intervals** count grid cells per axis; fixed-wall boundary values are prescribed, while absorbing boundary values are evolved. **Boundary** identifies either fixed-zero reflecting walls (`dirichlet`) or the declared local absorbing condition. **Case** is a predetermined validation trajectory. **Validation** means development data excluded from training; the separate final cohort is sealed. **Frozen** means the network weights are reused without training.

**ROM** is the reduced wave solve using latent coordinates. **FOM** is the full spatially discretized wave solve. **DST** is an exact discrete sine transform propagator for the semidiscrete fixed-wall problem. **RK4** is a fourth-order explicit time integrator. **Step** is the ROM time-step size. **CFL** controls the FOM step relative to mesh spacing and wave speed. **Query** includes the complete stated input, initialization, evolution, output and transfer work. **Median** is the middle value; **worst** is the largest value. **Cases above target** counts failed or over-threshold trajectories, not timing repetitions.

**Latent dimension** counts reduced coordinates. **Bank** is the spatial feature family generated by the coordinate network. **Weak tests** project the wave equation onto bank features. **QR** is a matrix factorization used here only to change coefficient coordinates. **Tangent velocity** is the decoder derivative applied to latent velocity; **curvature** is its second derivative contribution. **Cold fit** initializes the latent state from the supplied physical fields. **Stationary** means the recorded local optimization derivative tests passed, not a proven global best fit.

**Displacement** is the wave field; **velocity** is its time derivative. **Energy-state error** combines displacement-gradient and velocity errors. **Initial normalized** divides by a fixed initial scale. **Current relative** divides by the field's norm at the observation time. **Absolute** is the unnormalized physical error. **Vanishing** flags a reference norm below the configured fraction of its initial scale; **zero** flags a numerically absent reference and makes its relative error undefined. **Energy fraction** divides current energy by its initial value. **Phase** compares modal oscillation angles. **Mean field** is the spatially integrated field.

**Same-grid discrepancy** compares ROM or FOM output with a tightly resolved solve of the identical spatial discretization. **Physical reference** uses an independently refined approximation to the declared continuum PDE. **Self-refinement** compares reference discretizations. **Contraction ratio** measures whether successive differences shrink. **Conditional estimate** assumes that shrinkage persists; **proven bound** would require stronger evidence. **Refinement pass** checks time-step sensitivity, not full physical accuracy. **Assembly** builds cached mesh-dependent operators. **Compilation** prepares the JAX executable. **Burn-in** stabilizes GPU activity before timing. **SHA-256** identifies exact artifact bytes.
