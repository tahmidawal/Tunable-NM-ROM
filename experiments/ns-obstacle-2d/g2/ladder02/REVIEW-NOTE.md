# Prospective ladder02 diagnostic amendment

The original ladder01 remains failed and retained. All six ladder02 cases use centered-difference steps `[1e-2,1e-3,1e-4]` instead of `[1e-4,1e-5,1e-6]`. No physical configuration, nonlinear solve, direction, row scaling, signal threshold or plateau/reducing acceptance condition changes. The fixed-quadrature residual is quadratic; the observed inverse-step error pattern supports cancellation but does not prove it.

Reference and perturbed coefficients, free mask, row scales, direction, raw assembled action, steps, plus/minus coefficients and raw residuals are saved before the check. This enables independent arithmetic replay. At the maximum 100000 DOFs, these arrays occupy less than 16 MiB; the 48 GiB working guard and 64 GiB allocation are unchanged. Source and exact single-job release remain required.
