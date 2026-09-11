# Fixed-encoder wave heads with phase supervision

Predeclared accuracy experiment preserving the true 32-coordinate nonlinear manifold and frozen 64-function spatial bank. No result is claimed before paired rollout validation. This is separate from the larger linear-bank/nonlinear-output hybrid.

Regenerate all original 64 training cases from their recorded seed with the original accepted fresh RK wave reference at 256 intervals and its original output times. Verify the training truth hashes and bank-coordinate initialization against the original lineage. The final paper cohort stays sealed.

The prior head learned independent codes for each training snapshot and used displacement reconstruction only. Those codes have no consistent fixed time derivative. The two new arms both use the same fixed affine training encoder:

$$z=P^+(a-\bar a),\qquad w=P^+b,$$

where $P$ and the center are the original training coefficient-PCA initialization extended to 32 coordinates. Thus the target code velocities differentiate the chosen code definition. This change in code supervision is explicit; the old deployed head remains the control.

Both new heads start from identical common-affine parameters, random hidden layers, zero nonlinear output weights, optimizer seed and minibatch sequence. Keep latent dimension, bank rank, network width, learning-rate schedule and 10000-step endpoint fixed. Do not select epochs by development-case error. One arm uses field reconstruction only; the other adds displacement-energy and tangent-velocity supervision:

$$L_u=\mathbb E\frac{\|h(z)-a\|_2^2}{\|u_0\|_M^2},$$

$$L_K=\mathbb E\frac{c^2[h(z)-a]^TK[h(z)-a]}{2E_0},\qquad
L_v=\mathbb E\frac{\|J(z)w-b\|_2^2}{2E_0}.$$

The field-only objective is $L_u$; the combined objective is $L_u+L_K+L_v$. The scales and speed are the individual training case's supplied initial physical quantities. The bank projection's unavoidable normal error is not trainable and is not subtracted from later full-field errors. Record each minibatch loss component independently for both arms. The bank, coordinate lift and mesh transfer stay unchanged.

Freeze both trained endpoints, then evaluate actual curvature-inclusive latent dynamics with the validated guarded Cholesky implementation and the same initial-fit contract. Test the original two opened development cases at 64 intervals before a separately frozen multiresolution/fresh-development confirmation. Compare the original deployed head, both new heads, original/faster latent settings, named iterative CG timestep/tolerance combinations and direct DST. Returned physical velocity remains $GJ(z)w$ from the actual latent evolution, not a separately predicted velocity output.

Each candidate requires all-state displacement/velocity/energy accuracy, stationary initial fitting, finite rank-valid evolution and halved-timestep consistency. A finite or accurate-looking trajectory with failed fitting/rank/temporal checks is retained as a diagnostic failure. If 10000 steps are insufficient, report that bounded endpoint; do not silently continue selected heads to a larger training budget. Complete-query timing includes every start, stage and full output. Preserve every timing repetition, loss/checkpoint, field/error artifact and source hash.

## Glossary

- **Fixed encoder:** immutable linear map from learned-bank coefficients to training latent codes.
- **Tangent velocity:** decoder Jacobian applied to the code's time derivative.
- **Phase supervision:** training that includes displacement energy and velocity, not only displacement values.
- **Initial physical scale:** normalization derived from the supplied initial field and energy of each case.
- **Matched training arms:** identical initialization and sample sequence with only the declared loss terms changed.
- **Latent dynamics:** time evolution in the 32 nonlinear coordinates and their 32 velocities.
- **Halfstep consistency:** agreement when the integration timestep is halved.
- **CG / DST:** iterative conjugate-gradient midpoint and direct discrete-sine-transform full-order controls.
- **Development cohort:** cases used to select or diagnose settings; distinct from the sealed final paper cohort.
