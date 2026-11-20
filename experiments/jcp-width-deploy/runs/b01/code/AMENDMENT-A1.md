# Reference arithmetic repair, registered after the second pilot

The second pilot completed all ROM arms, but development case 1 failed the predeclared tighter-reference Newton-budget check. Its unchanged exclusion is retained in `runs/p02/output/summary.json`, along with both reference trajectories. The next attempt reruns the identical two cases and test-space grid.

Only the arithmetic used to evaluate diffusion in the **comparison reference** changes. In exact arithmetic,

$$
u_{i-1,j}+u_{i+1,j}+u_{i,j-1}+u_{i,j+1}-4u_{i,j}
= (u_{i-1,j}-u_{i,j})+(u_{i+1,j}-u_{i,j})+(u_{i,j-1}-u_{i,j})+(u_{i,j+1}-u_{i,j}).
$$

The right side avoids subtracting large summed field levels. It is the same five-point finite-difference operator, with identical sign-upwind advection, boundary values, backward Euler time grid, seeds, initial state and linear preconditioner. All original nonlinear/linear tolerances and the maximum Newton budget are unchanged. The ROM kernels are unchanged. No case is removed or reclassified using its tiny old reference disagreement.

The new job regenerates both original and repaired reference trajectories from the registered seeds, including per-Newton residual histories. It checks the original residual at every returned repaired state. Admission requires the original residual to satisfy the unchanged relative $10^{-10}$ gate, paired original/repaired trajectories to differ by at most $10^{-8}$ relative to the initial field, and original/repaired primal residual differences normalized by the preceding-state norm to be at most $10^{-12}$. Jacobian actions at the first and final time on three predetermined directions must agree to relative $10^{-12}$. The existing tight/tighter agreement and nonexhausted Newton-budget gates remain necessary.

The sub-minute implementation fixture tests primal and Jacobian-action equivalence before the research driver executes. If the repaired reference still fails, retain the failure and stop numerical promotion; do not relax the reference tolerances or omit the affected case. Run p03 is limited to one GPU and 45 minutes within the unchanged three-hour lane ceiling.

Glossary: a reference is the separately checked comparison trajectory; arithmetic ordering changes the way a fixed formula is evaluated in finite precision; a Jacobian action is its directional sensitivity; a residual is equation mismatch; admission means passing every original qualification gate plus the additional equivalence checks.
