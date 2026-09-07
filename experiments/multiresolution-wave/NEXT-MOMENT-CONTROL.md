# Proposed outgoing-wave moment controls

This is an unsubmitted future mechanism study. The completed [moment diagnostic](runs/dynamics02/analysis/ABSORBING-MOMENT.md) shows both a changed initial invariant and subsequent reduced-system drift; it does not establish that either defect alone causes the field errors.

Keep the existing learned spatial bank, absorbing data family, original training seed and the same two development cases. Keep the final cohort sealed. Use the existing requested meshes and observation times. No nonlinear head training or replacement spatial bank is part of this control.

The actual outgoing-wave invariant is

$$I=\ell^Tb+c\delta^Ta,\qquad \ell=G^TM1,\quad \delta=G^TB1.$$

Here $G$ is mass orthonormal, $B=MD_0$ includes the outgoing edge contributions, and the full-bank equations are $\dot a=b$, $\dot b=-c^2Ka-cDb$. Their invariant derivative is

$$\dot I=-c^2\ell^TKa+c(\delta^T-\ell^TD)b.$$

Use a factorial comparison so the initial-fit and evolution changes can be assessed separately:

| Arm | Spatial trial and test span | Initial state |
|---|---|---|
| Original bank | Existing learned bank | Existing mass projection |
| Constant added | Existing bank plus the exact constant direction | Mass projection into the enlarged span |
| Initial moment corrected | Existing learned bank | Original projection plus the correction below |
| Both changes | Existing bank plus the exact constant direction | Enlarged projection plus the correction below |

Construct the added direction from the mass-orthogonal remainder of the constant field. Retain the original learned directions and record the added direction, mass Gram matrix and conditioning. This increases the full linear control by one coordinate; it is not a square nonlinear least-squares objective. Test whether the new span contains the constant and whether both invariant-generator rows vanish to numerical precision.

For a given projected displacement and velocity, isolate the initial constraint with the minimum mass-norm velocity correction

$$b_{\mathrm{corrected}}=b+\frac{I_{\mathrm{input}}-I(a,b)}{\ell^T\ell}\ell.$$

The supplied physical input determines $I_{\mathrm{input}}$; no trajectory truth enters this correction. Record the initial displacement error, the added velocity norm and its energy-state error. Leave displacement unchanged within each span. All four arms use the same speed and actual reduced mass/operator conventions, and dense linear propagation remains charged to every query.

Report invariant error at initialization, drift from each arm's own initial value, time-max physical field errors, final current-relative errors, absolute errors, current truth norms, and mean displacement. Preserve full fields and coefficients. The primary mechanism comparisons are paired differences at the same mesh/case. A preserved invariant with unchanged or worse field error is a valid negative result.

Before any submission, review a concrete configuration and budget with the coordinator. A bounded initial panel would use the existing two meshes and two development cases, four arms and three repetitions, together with the same-job efficient RK4 controls and the unchanged independent reference/refinement protocol. Reserve one GPU job with a one-hour cap. Regenerate original seed data on the cluster; do not reopen the final cohort. This document authorizes no submission and claims no speed advantage.

## Plain-language glossary

- **Invariant / moment:** a global weighted quantity the exact discrete equations preserve.
- **Trial / test span:** fields the reduced model can represent / directions against which its equations are projected.
- **Mass projection / mass norm:** closest represented field under area integration / its weighted magnitude.
- **Constant direction:** the spatial field equal to one everywhere.
- **Factorial comparison:** combinations of two separate interventions, including each intervention alone.
- **Generator / conditioning:** the matrix defining linear evolution / sensitivity to numerical errors.
- **Current-relative / time-max / energy state:** error divided by the remaining truth norm / maximum across saved times / displacement-gradient and velocity norm.
- **Development / final cohort / repetition:** inputs already used to guide design / sealed confirmation inputs / repeated timing measurement.
