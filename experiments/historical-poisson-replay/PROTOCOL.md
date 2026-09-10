# Historical Poisson comparison replay

This cell restores the historical frozen-checkpoint, same-grid, device-resident
comparison requested on 2026-09-10. Results are provisional until the GPU logs,
source checksums and captured timed fields have been audited.

The source provenance map is `SOURCES.json`. Files in `vendor/` are byte copies
of the archived quadrature-free cell's staged dependencies; `in/` contains the
archived mesh-specific checkpoints. No model training occurs. The historical
quadrature-free ladder ends at 512 nodes per axis; 1024 is explicitly an extension
using the checkpoint from the older sampled-ROM/CG comparison.

The decoder is $u(z)=G h(z)$. With the original weak modes and source weighting,
the full-grid residual and its quadrature-free counterpart are

$$r_{\mathrm{full}}(z)=W_\lambda\Phi^T G h(z)-f_m,
\qquad r_{\mathrm{qf}}(z)=W_\lambda B h(z)-f_m,
\qquad B=\Phi^T G.$$

Both use the archived trust-LM implementation, the mean training code as the
starting point, the checkpoint's trust radius and iteration budget, and the
original two residual-reduction tolerances. Source projection uses the archived
dense sine products. Large mesh arrays are explicit compiled-function arguments
to avoid capturing them as constants; the mathematical operations are preserved.
The original full-grid residual is a control. The sampled EQ control is omitted;
this replay needs no quadrature fit.

The primary FOM is the archived unpreconditioned CG solver and its full original
tolerance ladder. The secondary FOM is the archived dense sine spectral solver.
All methods use the same grid, source and truth. The historical comparison selects
the cheapest CG tolerance whose cohort mean relative error is no greater than the
ROM's. Median and worst errors, all raw times and outlier counts are also retained.
This is a mean-error selection rule and is not represented as a worst-case bound.

Input sources are already on the GPU. The timed ROM query includes source
projection, nonlinear solving and decoding every interior output value to the
GPU. The timer ends after output completion. Input/output transfers, offline bank
construction and compilation are outside that timer. All subjects share one GPU
allocation and alternate forward/reverse order within each source. GPU burn-in
immediately precedes every source's block. Actual outputs from the final timed
repetition are retained without changing precision; these same outputs supply
reported errors, solver counters and QF/full identity gates. Every timing
repetition remains in JSON. An outlier is a repetition exceeding three times its
own source/subject median; outliers are counted and retained.

Both historical cohorts are regenerated from their recorded seeds on the GPU
job's host. Every mesh uses its own historical weights; this is not frozen-weight
mesh transfer. The original mesh label counts nodes including boundaries.

The local smoke run is a reduced structural check, not a research result. The
cluster allocation requires a GPU backend, float64 and highest matmul precision.
All code/checkpoint bytes are checked before execution. The owner commit and
content hashes are recorded directly; no cluster ancestor repository is consulted.

## Glossary

- **FOM / ROM:** full numerical solver / reduced numerical solver.
- **QF / EQ:** exact algebra without quadrature / a fitted empirical quadrature rule.
- **N, K, R, M:** grid nodes per axis including boundaries, latent coordinates,
  spatial bank features, and requested weak test modes.
- **Bank / head / checkpoint:** learned spatial functions, neural map from latent
  coordinates to their coefficients, and saved model weights and configuration.
- **Weak residual:** PDE equations projected onto smooth test functions.
- **Trust-LM:** the archived damped nonlinear least-squares solve with a maximum
  allowed latent step.
- **Tau / censored:** requested residual reduction / stopping without reaching
  that requested reduction, including iteration-budget exits.
- **CG / dense sine spectral:** iterative conjugate gradients / diagonal solution
  in the discrete sine basis evaluated by dense matrix products.
- **Device-resident query:** computation beginning and ending in GPU memory.
- **Held-out / fresh:** sources after the training portion of the original seeded
  draw / sources generated from the archived separate seed.
- **Mean, median, worst relative error:** average, middle and largest source error,
  each normalized by its own same-grid reference solution norm.
- **Outlier / raw repetition:** a retained unusually slow measured call / one
  measured call before statistical aggregation.
- **Extension / provisional:** newly tested beyond the old QF ladder / awaiting
  validation or broader scientific confirmation.
