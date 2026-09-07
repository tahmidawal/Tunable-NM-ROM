# Burgers time-step development study

This predeclares the bounded steps05 experiment. Numbers below specify the design; results will be generated in the existing multiresolution report after collection.

The rollout04 component profile identifies evolution as the largest reduced-query component. Test whether fewer time steps preserve the empirical physical accuracy gate with the same learned coordinate-separable decoder and fixed physical Gauss cold fit. The checkpoint, clipped Gaussian family, four cases from seed 7090702, latent dimension, weak modes, quadrature training, and efficient FOM configurations stay unchanged. Final cohorts remain unopened.

Use requested grids of 512 and 1024 intervals, common observations on 256 intervals, and six outputs at $t=0,0.05,\ldots,0.25$. Each mesh has four ROM configurations: Gauss cold budget 180, one start, evolution improvement threshold 0.01, and $\Delta t\in\{0.005,0.01,0.025,0.05\}$. These give 50, 25, 10 and 5 steps and align every output. The original step size is the within-job control. The evolution iteration cap stays 30; retain all budget, improvement, residual and failed exits without calling an improvement exit stationarity.

Retain every FOM grid no larger than the requested grid from 128, 256, 512 and 1024 intervals. Each has the same five time-step/Newton-tolerance pairs as rollout04: $(0.01,0.01)$, $(0.01,0.003)$, $(0.005,0.01)$, $(0.005,0.003)$ and $(0.0025,0.003)$, with linear tolerance 0.5. There are 43 declared configurations and 516 actual timed invocations: four cases and three repetitions each. Order is randomized from seed 89005; burn-in precedes each call. Record cost and errors from that same complete query, including dense input transfer and interpolation, cold fitting, evolution, dense decoding and host output. Setup and compilation are separate.

Regenerate the same six refined FOM settings, with finest 4096 intervals and time step 0.0003125; use adaptive Newton tolerance $10^{-11}$ and linear tolerance $10^{-9}$. Preserve residuals and three-level space/time comparisons. Each requested norm and the common observation norm has its own empirical margin. Eligibility requires every case's error plus margin to meet the target, the margin no larger than one tenth of that target, and observed refinement decrease. These are empirical development qualifications, not continuum error bounds. Additional tight same-grid FOM references cover all four ROM time steps.

Archive every invocation's common-grid field. Retain the full dense field from at least one repetition per configuration and case; each other repetition either has exactly the same full-field SHA256 or retains its own dense artifact. Keep fine references restricted to the finest requested grid so both requested-grid norms can be independently audited. Profile each ROM time step separately with staged/fused parity; component times remain diagnostic.

One A100 GPU job in its own steps05 directory has a two-hour limit, expected duration about 25–30 minutes from rollout04, and a bounded archive below 12 GB. Before submission: under-minute local timestep/alignment smoke, committed source provenance, queue/disk check, and mandatory GPU/f64/highest preflight. After completion: checksum collection, exact remote-directory removal, NumPy field audit and generated findings. No architecture, family, checkpoint or training change is included.

## Glossary

- **Interval:** one spatial grid spacing; a grid with $L$ intervals has $(L+1)^2$ nodes.
- **ROM / FOM:** reduced model / full grid solver.
- **Gauss cold fit:** fitting the initial latent state using fixed physical Gauss–Legendre points and charged interpolation of the supplied field.
- **Weak modes / quadrature:** smooth residual test functions / weighted evaluation samples used to approximate their integrals.
- **Budget / improvement exit:** configured iteration cap / stopping after a small relative improvement; neither proves a global optimum.
- **Common norm / requested-grid norm:** error measured on shared observation nodes / all returned nodes, respectively, normalized by the reference initial field norm.
- **Empirical margin:** refinement-based allowance whose validity is observed, not rigorously bounded.
- **Envelope:** the cheapest adequate measured FOM configuration under the declared per-case median cost convention.
- **SHA256 / staged-fused parity:** content hash / agreement of a split diagnostic query with the complete query.
