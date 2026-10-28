# A6 candidate: coefficient-metric POD and explicit velocity components

This is the implementation candidate for central A6 at `c08537e6f`, before FE data or training use. The policy is reviewed; exact source and native runtime review remain required. It replaces the quotient reconstruction and vector single-start eigenbank. The prior source and its failing synthetic evidence are preserved byte-for-byte under `legacy/97847a479/`. Neither a universal quotient-failure ratio nor a common component ratio across distinct eigenvalues is claimed.

Use deterministic column-pivoted mass-metric Gram–Schmidt with two orthogonalization passes on training coefficient snapshots, followed by the small-factor SVD. Rank is still determined from the original sampled singular spectrum: velocity cutoff $10^{-12}\sigma_1$; pressure-centered cutoff $10^{-12}\max(1,\sigma_{1,original})$. No extra retained modes, rank adjustment or tolerance retuning follows a failure. Modes are $Q_cU_R$, never $CV/\sigma$. Pressure remains constant first. The actual evaluated pressure Gram remains the inf-sup normalization.

Prospective numerical checks, all frozen before FE input:

- QR backward Frobenius error in the mass norm divided by the original rank-reference scale: at most $10^{-10}$.
- QR mass Gram and actual evaluated mode Gram spectral error from identity: at most $10^{-10}$.
- Full small-factor versus sampled singular-value absolute difference divided by the same original scale: at most $10^{-10}$; rank classifications must agree.
- At every eligible registered prefix, each training residual-norm difference between actual FE modes and an independent sampled-SVD prefix residual, divided by the original rank-reference scale: at most $10^{-10}$. Only aggregate weighted Frobenius error is POD-optimal. No comparison of unstable individual tail directions is substituted for these checks.

Cluster handling uses the full singular spectrum normalized by its leading singular value. Adjacent values join a cluster when their gap is at most $\max(64\epsilon,10^{-10}\max(s_i,s_j))$ in normalized units. Reject a cut through a cluster at the numerical-rank boundary, retained maximum or eligible registered prefix; pressure prefixes count the constant separately. Pivot scores within $64\epsilon$ times the maximum are numerical ties, resolved by the smallest frozen snapshot/FE-DOF index. This policy is amplitude invariant and records intrinsic tail-direction uncertainty.

Solve 48 scalar velocity-order H1 generalized eigenmodes plus one look-ahead; emit each scalar mode in the first then second physical component. Obtain maps from actual component coefficient views and verify local scalar DOF numbering, essential masks, diagonal scalar blocks and zero cross blocks. The matrix relative-Frobenius discrepancy tolerance is $10^{-12}$. Keep 32 pressure-order scalar candidates plus look-ahead. `gap_checked` metadata does not claim spectral completeness: unresolved scalar multiplicity/count remains explicit.

Analytic controls independently check the off-diagonal derivative convention with $(y,0)$ and $(0,x)$, unit/outward box normals, inward-fluid obstacle radial normals, and separate closure of the outer rectangle and obstacle. Dimensionless values/derivatives/weights are marked as already scaled; `physical_to_dimensionless` must not be applied to them again. Prefer one source-hashed Stokes lift at one declared viscosity, reused by all future training cases. Any retained legacy independent-lift path remains explicitly labelled.

Independent review and a genuine pinned-library integration control are still required. These source changes authorize no FE job, snapshot use or training.
