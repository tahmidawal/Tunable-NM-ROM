# Burgers same-sample reconstruction screen

Status: prospective implementation protocol after explicit user approval of version B on 2026-10-27. No reconstruction result has been observed. This screen is developmental; it does not train new neural banks, open sealed cohorts, or evolve reconstructed states.

## Scope and ownership

Coordinator: existing `2026-10-27-burgers-decoder-interpolation` at consolidated fork `93f399faa`. Separate 2D/3D lanes `2026-10-27-burgers-recon2d` / `2026-10-27-burgers-recon3d`; namespaces `b2d-recon-20261027` / `b3d-recon-20261027`. Initial hard allocation ceilings are four GPU-hours per lane, including failures. Numerical execution requires independent design/code/exact-job review; review is an internal quality gate, not a further user approval requirement. No external campaign may be cancelled or modified. No branch merges without the user.

## Question and information contract

Each method consumes the identical point-sample array $y_c$ at one physical time and returns a fine field. Common grid, domain and zero-wall metadata are allowed. No exact IC descriptor, viscosity, reference normalization, earlier state, fine solution values or PDE correction enters reconstruction. The scorer alone may use reference fields/initial norms. The frozen spatial bank is linear in fitted coefficients; no nonlinear head is deployed.

Two separate inputs: A, direct nodal restriction of the finest stored E0 ST array; B, actual coarse central-FD/CN-with-Rannacher-startup FOM at the same time. B generation uses the same physical IC including the common recorded 3D scaling; these descriptors never enter reconstruction. No dynamics of a reconstructed field is performed. Ordinary interpolation cannot correct all coarse-solver bias; least-squares fits may smooth samples, and this must be labelled.

All arrays/scoring use physical amplitudes and x-slowest interior indexing; zero-wall observations supply no coefficient information. Outputs at t=0.05,0.10,0.15,0.20,0.25 are primary, with t=0 separate. L always denotes intervals per axis.

| Dimension/control | Lc | Lf | Bank | Prefixes | Exposed selection/report cohorts |
|---|---|---|---|---|---|
| 2D inherited finer-data bank | 32,64 | 256 | smooth-bank base R512, hidden1024x2 | 128,256,384,512 | dev6 / val32 |
| 2D training-resolution control | 128 | 256 | same base and coarse R512 | 128,256,384,512 | dev6 / val32 |
| 3D inherited multimesh bank | 16,32 | 64 | W1024, hidden2048x3 | 256,512,1024 | dev4 / val64 |

References are eligible stored ST outputs, not newly assumed fine-grid truth. Old test populations are not used for tuning; sealed/OOD are not loaded. Primary contrasts: dimension-specific actual-input B at Lc32 (2D) or Lc16 (3D). Other meshes, input A, and training controls are separately reported, not searched for a headline. All declared cases and failures remain in coverage denominators.

## Reconstruction methods

1. Tensor multilinear interpolation including zero walls.
2. Tensor cubic not-a-knot interpolation through all coarse nodes including zero walls. Tensor interpolation matrices depend only on geometry. No exact boundary derivative supplied.
3. Full tensor sine/DST-I interpolation, physically scaled, reproducing coarse interior nodes. Separately truncated sine (per-axis cutoffs floor((Lc-1)/2), floor(3(Lc-1)/4), Lc-1) and smooth spectral filtering with normalized fourth-power frequency penalty, lambda 1e-6,1e-4,1e-2,1. This is regularized reconstruction, not a PDE solve.
4. Frozen-bank weighted least squares and regularized fits. For each prefix independently, physically whiten using its fine-grid Gram (basis evaluations only, no target data), SVD the weighted sampled basis, and use relative sampled singular tolerance 1e-10. Fine Gram eigenvalue cutoff 1e-12 relative must be recorded, including removed directions. Ridge lambda 0,1e-10,1e-8,1e-6,1e-4,1e-2 penalizes fine physical field norm. Report effective ranks and the actual weighted coarse-to-fine operator amplification; raw basis-dependent condition numbers are supplementary.
5. Uncentered POD with declared continuous extension, matched coefficient counts and rank curves, same fit policies. Add covariance-weighted ridge using snapshot variances, with its penalty normalized in the fine physical metric. Means cannot be added invisibly. Strongest available comparator and matched-information comparator are separately labelled.

Uniform quadrature weights are Lc^-d and Lf^-d. A representation's geometry/factorization and output basis may be cached for repeated requests at fixed grids; this setup, including network feature evaluation, is measured separately. Per-query fitting and fine synthesis remain charged. No independently normalized fine basis is substituted without transforming coefficients.

## Training information and POD construction

2D base is trained at256 nodes, coarse at129 nodes using the same576trajectories/16384selected states. Regenerate the exact original upwind-BE values from seeds on cluster; verify state-pick/generator hashes and archived fingerprints, with separately documented numerical tolerances where exact compiler bytes differ. POD uses raw uncentered equal snapshot weighting, matching the global-MSE data term. Original model/rotation pairs are retained; no neural retraining. Coarse bank is labelled129-node solution-data-only with inherited fine-developed architecture/recipe.

3D bank uses1536trajectories,12times, and33/65/129-node groups;129group sees a fixed250047point subset. Regenerate the same data/masks and relative-state/equal-group weights. Two PODs:
- Matched-information extension:33node fields tensor-sine extended,65native identity,129selected values extended to Lf64 with fixed8-neighbor inverse-distance-squared interpolation, exact coincident values preserved. Geometry-only interpolation weights; no additional solution values. This extension can limit accuracy and is disclosed.
- Stronger multimesh comparator: same33/65groups, full129native fields directly restricted to Lf64, including values the bank did not receive. This extra information is explicit and prevents a bank advantage being obtained by weakening POD.
Normalize each state's contribution by the RMS of that group's original bank-sampled target (not the extended fine field), with equal state/group weights, subject to exact verification against training code. Fixed randomized POD uses oversampling128,two power iterations and recorded seed. Keep q1/q2 development reconstruction differences, orthonormality, capture residuals and independent probe diagnostics. It is approximate POD, not a certified global optimum. If approximation changes reconstructions materially, bank-over-best-POD claims are unavailable. No indefinite retry or unreported budget increase.

All training solutions are regenerated on cluster. Large regenerable training matrices may remain ephemeral after their outputs/fingerprints/provenance are verified; unique basis, fits, observations and audit evidence must be retained locally. Do not duplicate old archives or delete unique existing data.

## Selection and disclosure

Fine-informed methods: choose one configuration per dimension/input-source/coarse-grid on development only. Minimize median case-max error; within conditional reference ambiguity choose lower worst then smaller rank/regularization complexity. Every candidate remains in tables. The ordinary comparator is chosen on the same development cases, then frozen before validation. No per-case best-method selection is an admissible result.

Strict existing coarse-data control: rank384 and lambda0 with declared SVD tolerance fixed before data; POD same fixedrank/policy. Other prefixes are descriptive predeclared curves, not a finely selected coarse-only winner. Any fine-development-selected variant must be explicitly fine-selected. The full architecture/recipe history is exposed; no claim of a completely fine-information-free development process or fresh validation.

Before validation: freeze config IDs, software hashes, artifact hashes, all selectors and source manifests in a committed file. Verify disjoint train/dev/validation IDs/parameters from generators; do not inspect sealed data. A failure to reconstruct exact POD information prevents architecture attribution, not the ordinary-interpolation comparison.

## Coarse-input admission and reference uncertainty

CNR uses four BE half-steps replacing its first two CN steps. Time ladder dt=.0025,.00125,.000625,.0003125, stopping at the first consecutive pair with stage convergence and max evolved field gap <=1e-5 in fixed-reference-initial physical norm. Keep both fields and all stage summaries; no unstable/oscillatory cases silently removed. Reconstruct both final input levels because a fit can amplify their difference. Input-convergence failure is visible and prevents an accepted spatial-input interpretation; raw same-input reconstruction errors remain descriptive.

A primary claim is completion of the stored numerical array. Repeat A on the other saved reference levels to diagnose uncertain-input amplification. For fixed reconstruction operator T and restriction S, reference perturbations change its error vector by $(TS-I)\delta$; target-only margins do not cover this automatically. For fixed B observations, conditional E0 target margins apply, subject to inherited reference qualifications. No norm difference is a causal additive error split. Coarse/off-node/derivative metrics do not inherit the full-field uncertainty certificate.

## Measurements and acceptance

Primary field metric is fine-lattice L2 error divided by the fixed reference initial norm, max over five evolved times per case. Report per-time values, cohort median/worst, cases above1%, paired resolved wins/ties/losses, regressions and complete failure counts. Reference uncertainty is conditional Richardson, not a rigorous bound; common fixed-B pairwise differences require >2epsilon_ref to be called resolved. Cases, not times, are replication units.

Secondary: coarse-sample residual, off-observation-node error, oracle fine-reference projection floors (diagnostics only), singular spectra/effective ranks/weighted amplification, A-level and B-time-level sensitivity. Derivative accuracy is deferred because no qualified gradient reference exists.

All timed comparisons use the same allocation/device, f64/highest and GPU preflight. Prepared-query input/output boundary is identical across methods; state explicitly whether transfers are included and additionally measure common host-output cost if device-only primary timing is used. Burn in before each timed block. All timing repetitions (minimum3) are retained, with actual outputs or coefficients+independent reconstruction hashes from the timed invocation. Measure setup/cache/network evaluation/compilation separately. No cross-job wallclock comparison and no claim decoding is negligible.

One independent auditor must check algebra fixtures, data/provenance, exact job launch, returned fields/coefficients, selection and reporting. Require checksums before deleting only the collected remote job directory. All failed allocations remain in ledger. Each lane exports machine-readable JSON with final hashes and qualification scopes; coordinator report numbers are generated from it.

## Decision rule and limits

A suggested larger-campaign signal is ratio of validation median errors <=0.8 against the development-selected ordinary baseline, resolved wins on>=75% of all validationcases (ties count as non-wins), no resolved worsening of overall worst error, and disclosed paired-tail/cost behavior. Apply separately per dimension at its primary B mesh and frozen bank setting. These are practical screening thresholds, not predictions or statistical proof. Gains over matched/stronger regularized POD are needed before architecture-specific claims; approximate or differently informed POD limits that attribution. Any larger campaign/fresh nonsealed confirmation requires a new scoped decision. Dynamics stays separate.

## Glossary

- Bank: frozen spatial functions produced by the coordinate network; coefficients combine them into a field.
- Prefix/rank: number of retained spatial functions; different from a neural hidden-layer width.
- FOM: numerical solution of the full PDE on its stated mesh.
- POD: spatial basis fitted to a collection of solution snapshots; gappy fitting estimates its coefficients from samples.
- DST-I: sine transform matching the zero boundary values.
- Ridge/TSVD: controlled suppression of poorly observed coefficient directions.
- E0/ST: stored second-order space-and-time reference and its conditional uncertainty.
- Oracle: diagnostic fit allowed to see the fine answer; not a reconstruction competitor.
- IDW: interpolation from nearby supplied points weighted by inverse distance.
- Exposed: outcomes have already been available in earlier development; not a fresh test.

## Amendment A1 — coarse-input qualification before reconstruction results

The 2D preparation and 3D development smoke found some coarse inputs exceed the
fixed temporal-gap threshold at the final original rung. No reconstruction
outcomes informed this change. Add exactly one rung, dt=0.00015625, preserving
the 1e-5 threshold, every earlier field/status and all other equations. The
vendored reference solver bytes remain unchanged; set its stage-buffer capacity
`RC.NST=2000` explicitly in the adapter and record that override. In 2D the
supplement computes only originally unqualified inputs; 3D full preparation
includes the extra rung in the fixed ladder. Remaining failures stay visible
and descriptive; no automatic further ladder or threshold relaxation.

3D POD matches supplied training observations, not the bank's entire development
information: the bank also selected its checkpoint on historical bank-validation
data and optimized tail/Gram penalties. These distinctions prohibit a pure
architecture causal claim even with a converged POD computation.

Primary timing is one field per query. Interleave methods in a predetermined
order, with GPU burn at block starts and after long host-bound interruptions.
Every timed row retains at least three invocations, hashes, coefficients and
scores. If the fixed allocation ceiling requires limiting timing, retain all
predeclared accuracy curves while timing frozen selected configurations and
unregularized rank curves; untimed rows have no cost claim. This timing scope
must be fixed before validation, not chosen from validation results.
