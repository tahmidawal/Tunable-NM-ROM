# Heat: frozen refined heads over a wider mesh ladder

This is the next bounded development proposal for the already approved heat worktree. It extends the audited head refinement across resolution and new development inputs; it contains no new results. Final cohorts remain sealed.

## Fixed models and physical family

Use both expanded-coverage endpoints from heat `pilot03`, with exact checkpoint hashes resolved from its archived JSON. Keep each endpoint's weights and training-code library unchanged, including the shared spatial bank. Preserve the restricted polynomial-boundary-factor Gaussian family, diffusivity, output times, weak modes, initial fitting and validated relaxed solver tolerance. Keep the Crank–Nicolson step fixed at the existing primary value; this experiment is about mesh transfer and fresh inputs.

The first cohort repeats the original four development draws at seed `790711`. Add eight independently seeded development draws at seed `790716`, using the same parameter ranges. Generate both sets on the cluster. Label cohorts separately; this is neither new training nor sealed final evaluation.

Requested mesh intervals are `64, 128, 256, 512, 1024`. Retain common observations at the original `64` intervals and also score the complete requested grid. Use independently checked spectral references at sufficient nested resolutions to support every requested grid; a prospective `1024, 2048` pair must pass the existing physical refinement checks. Preserve exact analytic initial fields and the distinction between spectral physical reference and discrete same-grid reference.

## Efficient full-order controls

Time the direct same-grid sine-transform solver and a coarse-grid envelope with solver intervals `16, 32, 64, 128`, restricted to choices no larger than the requested grid. Charge restriction of the supplied input, evolution, interpolation to every requested output node and full host output. Do not claim a crossover against only the requested-mesh solver when a cheaper interpolated FOM satisfies the same physical target.

Declare the handling of the initial output before submission, including whether a baseline reconstructs it from its coarse state or returns the supplied field directly. Save the actual returned initial field and charge all required output construction. Do not alter this policy after seeing errors.

## Measurement and preservation

Use three repetitions per configuration/case/mesh, GPU burn-in before timed blocks, and the existing full host-field input/output contract. Every cost and error must come from the same invocation. Persist timing arrays and first-repeat complete fields or an equally auditable exact representation with repeated full-field hashes. Record all convergence/budget/failure exits, setup costs, checkpoints, cohort draws, job and hardware metadata. Rebuild mesh-dependent operators; do not reuse incompatible quadrature or test tables.

Choose configurations by median of per-case timing medians, and report paired FOM/ROM ratios as medians of ratios of case medians. Report original and fresh cohorts independently as well as their union. Show failures and physical target misses, with empirical reference adjustment explicitly separated from a rigorous bound.

One unique `transfer04` directory in the existing heat paralab namespace, GPU partition and a two-hour hard cap. The owner must estimate memory, archive size and runtime from the existing implementation before staging. If the proposed envelope becomes expensive, reduce redundant controls or archival duplication with explicit accounting; retain both trained endpoints, the fresh cohort, complete-grid accuracy, and an efficient FOM envelope.

## Plain-language glossary

- **Frozen endpoint / bank / code library:** saved trained weights kept fixed / learned spatial features / stored training coordinates used to start the initial fit.
- **Intervals / mesh transfer / common grid:** cells per axis / evaluating the same weights at new spatial resolutions / shared physical observation locations.
- **Current-relative error / same-grid reference / spectral reference:** field discrepancy divided by the current truth norm / precise solution of the chosen grid equations / an independently refined sine-series physical reference.
- **Crank–Nicolson / weak mode / quadrature:** the existing time formula / smooth PDE test function / weighted physical sampling.
- **Coarse FOM envelope / restriction / interpolation:** cheapest qualifying full solver allowing a smaller internal grid / sampling the supplied input onto that grid / reconstructing outputs on the requested grid.
- **Repetition / case median / paired ratio:** repeated timing of one query / its middle timing / full-model time divided by reduced-model time for a matched physical case.
- **Development / final / empirical adjustment:** inputs used to assess and select the method / reserved independent confirmation / allowance based on observed reference refinement.
