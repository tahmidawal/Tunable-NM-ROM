# Accelerating the frozen heat nonlinear manifold solver

Prospective bounded development study of initializer and time-advancement changes. No performance conclusion is established by this design.

Keep the frozen nonlinear head, spatial bank and latent dimension. The original compiled weak Crank–Nicolson NMROM, free-coefficient linear control, same-grid spectral FOM, and interpolated coarse FOM are rerun on one GPU. All parameters, full input/output, paired timing repetitions and reference checks follow the preceding linear-bank study. Settings are declared in `config-nonlinear-fast.json`; final paper cases stay closed.

## Candidate changes

The adaptive initializer tries the nearest training code first. It tries the mean code only if the first fit is nonstationary, nonfinite, or fails the declared full-initial-field relative-error gate. The latter is computed exactly from the orthogonal projection, outside-bank residual, and reduced fitting residual. Record every attempted fit; a skipped second fit has reason `-1`. This changes the selection policy and may sacrifice some accuracy; the second start cannot simply be deleted because it improves difficult cases.

For nonlinear exponential projection, use the previously verified reduced linear heat propagator to predict coefficients over an observation interval. With orthogonal bank coordinates, $G=QR$, use the raw coefficient propagator $P_a=R^{-1}\exp(\Delta t L)R$. Then solve a nonlinear weak fitting problem:

$$z_{j+1}\approx\arg\min_z\|B h(z)-B P_a h(z_j)\|_2^2.$$

Every returned field remains $G h(z)$; free coefficients are only an intermediate predictor. A full-budget projection control isolates this change in time advancement. A bounded-correction control uses the declared smaller LM budget. Test both the original initializer and the adaptive initializer with bounded and full-budget projection. The smoke fixture showed a large error for bounded projection with adaptive initialization; retain it as a negative control and add the full-budget adaptive control before the development run. This is a new time integrator on the same nonlinear manifold, not an algebraic acceleration of exactly the old trajectory. It is not exact heat evolution merely because the predictor uses a matrix exponential.

Retain all finite outputs, residuals, gradients and termination reasons. A budget exit remains nonstationary even if its physical error meets a development target. Compare errors at every requested time, maximum error, initial error, energy decay, and speed against the rerun controls. The bounded variants do not silently invoke a fallback or use reference information. If a bounded variant fails, use the full-budget result to diagnose the loss rather than hide it. No test-case retraining or retrospective setting sweep.

## Glossary

- **NMROM / FOM:** nonlinear-manifold reduced model / full-grid model.
- **Bank / head / latent:** frozen spatial functions / frozen nonlinear coefficient map / compressed evolving coordinates.
- **Weak fitting:** matching smooth sine-weighted moments rather than pointwise PDE residuals.
- **Adaptive initializer:** a second starting guess is evaluated only when the declared first-fit gate fails.
- **Exponential projection:** linear reduced prediction followed by fitting back to the nonlinear decoder.
- **LM:** damped nonlinear least squares.
- **Budget / stationarity:** allowed correction attempts / satisfaction of the gradient stopping criterion.
- **GPU query:** supplied full accelerator input through blocked full accelerator output; host transfers are separately recorded.
- **Development target:** empirical error threshold on cases used for development, not final confirmation.
