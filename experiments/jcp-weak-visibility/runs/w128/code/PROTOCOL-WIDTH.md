# Frozen-bank width follow-up to the observed initial projection floor

This is a new post-pilot development experiment, authorized after the six-case weak-test study. It changes representation width as an explicitly separate variable. The accepted test-count-only study and all pointwise counterexamples remain frozen. The target remains the same-grid, same-step discrete Burgers solution, not continuum accuracy or a timing advantage.

## Fixed inputs and finite scope

Use the identical dn256b checkpoint, original rotation, zero Dirichlet boundaries, 256-interval grid, exact original sign-upwind advection, five-point diffusion, backward Euler step 0.005 and horizon 0.25. Regenerate all six original exposed development cases using the original 4-plus-2 drawing contract. Keep the same-host legacy bitwise guard and separate four-ULP viscosity portability witness. No neural training, new parameters, cases, test cohort or reference tolerance change is allowed.

The only candidate widths are the nested frozen prefixes $R\in\{128,256,512\}$. The original 512-function checkpoint is the upper limit. Each prefix uses its own exact physical mass projection of the same initial field; no reference reinitialization occurs afterward. This native initialization explicitly measures whether a larger representation fixes the observed initial floor. These arms do not isolate an initialization change from a representation change.

## Stage W0: projection-only screen

Build the full frozen bank once and evaluate all three prefixes on the same six qualified reference trajectories and all 51 original times. Regenerate both original and cancellation-repaired references, preserving the established tight/tighter, original-residual, Jacobian-action and nonexhausted-Newton-budget gates. Retain reference histories and parameters. Save each prefix's mass Gram eigenvalues, rank at relative thresholds $10^{-10}$, $10^{-12}$ and $10^{-14}$, coefficients, full error curves, initial errors and maximum-over-all-times projection errors.

Use an ordered QR factorization of the mass-normalized full bank and triangular solves for every original prefix coefficient; do not truncate a prefix at a rank threshold. Save the triangular factor, Gram spectrum and independently computed singular spectrum of each triangular prefix. The global QR orthogonality defect must be at most $10^{-10}$, the normal residual orthogonality defect at most $10^{-9}$, and coefficient-reconstructed fields must agree with the QR projection to at most $10^{-10}$ in the initial-field norm. Gram and QR rank counts must both be full at all three declared cutoffs before PDE admission.

The R128 projection curves must reproduce the accepted study to absolute $10^{-10}$ in the initial-field-normalized error after collection. Every wider projection must be no worse at each time than the preceding prefix up to absolute $10^{-10}$; this nesting property is a numerical check, not a novel empirical discovery. A failed rank or projection check is reported explicitly and prevents that prefix's PDE admission. R128 baseline failure stops the entire follow-up; R512 failure does not exclude R256 if R256 independently passes every gate. Evaluate QR orthogonality and projection controls separately for each prefix.

A larger prefix is admitted to W1 only if **both** its worst initial projection error and its worst projection error over all cases and times are at most 80% of their corresponding R128 values. The comparison uses all six fixed cases, and every reference must qualify. This 20% relative reduction threshold is fixed prospectively. Both prefixes may qualify; if neither does, stop after this screen. The checkpoint and ordering are never retrained or replaced.

## Stage W1: conditional equation solves

If at least one wider prefix passes W0, rerun the unchanged R128/M2048 baseline and every admitted prefix on all six cases. Test R256 with M2048 and M4096; test R512 with M2048 and M8192. The common M2048 contrast isolates width at a shared test space; each within-width richer-test contrast diagnoses visibility. M remains comfortably larger than the actual coefficient count, and there are no frame variables.

Use exact all-node quadrature. Retain the original production/tight/tighter stationarity thresholds, 200-proposal ceiling, returned-state stationarity checks and tight/tighter trajectory thresholds. Preserve all coefficients, solver histories, rejections and repetition arrays. No wall-clock comparison across jobs is permitted, and the dense diagnostic implementation is not an online cost claim.

Use the same independent observer for every width: smooth modes 8193 through 9216, outside all minimized test spaces. Its scaled and unscaled residuals supplement the original observer; they do not retrospectively replace the accepted test-count study's observer. The diagnostic kernel may carry zero rows up to this shared buffer size, as explicitly recorded in the existing implementation. All minimized rows retain the identical inverse-Helmholtz scaling.

At fixed projected-reference states and reached states, save raw-coordinate and mass-normalized tested-Jacobian spectra, retained ranks, coordinate-rescaling invariance, and direct-sine/automatic-differentiation/finite-difference action checks. Use all six cases and the same observation times as the pilot. Keep sign-switch counts and all derivative/operator diagnostics. Each physical initial tangent has dimension exactly R.

The W1 baseline must reproduce the accepted R128/M2048 maximum-error curves within absolute $10^{-8}$ and the same stationary trajectories within the existing tolerance. A candidate improves the primary worst-case accuracy only if every numerical gate passes, its worst maximum-over-all-times error is at most 80% of this baseline, and no individual case's primary maximum worsens by more than 2% relative. Report all common-test and richer-test outcomes, including failures and pointwise-time regressions. The primary metric includes initialization; evolved-only error and evolution-distance are explanatory post-pilot diagnostics, never replacements.

## Budget and stops

W0 uses one GPU with a ten-minute ceiling. If admitted, the baseline and each wider prefix use separate unique jobs, one active at a time, each no longer than 45 minutes. At most three W1 jobs are allowed. The aggregate maximum reservation for this follow-up is 145 GPU-minutes, plus the already charged 556 GPU-seconds, below the original three-GPU-hour lane ceiling. Expected work is substantially smaller based on the completed small-grid pilot, but the hard ceilings govern. No job may start while two other campaign jobs occupy the parent's slots.

Stop on failed reference/control gates, lost rank, no material projection improvement, inability to finish within the reservation, or completion of the finite width/test matrix. Preserve failures; do not add widths, train, open validation, change the operator or relax a threshold to obtain the desired pattern. Parent/reviewer clearance of this protocol and implementation precedes submission.

## Glossary

A prefix keeps the first ordered learned spatial functions. A projection floor is the best error available in their fixed linear span. Native initialization uses that prefix's own best fit of the input field. The Gram matrix measures physical sizes and overlaps of basis functions. Rank counts independent reachable field directions. A weak test is a smooth weighted equation average. An independent observer is a disjoint set of tests used only for diagnosis. An equation solve evolves the reduced coefficients without looking at the reference solution. Admission means satisfying every prospectively stated gate. An allocated GPU-minute counts the scheduler's actual reserved GPU time, including setup and failed attempts.
