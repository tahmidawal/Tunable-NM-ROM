# Candidates

## 1. Dense quadratic tensor on the frozen bank

**Idea.** Precompute the current non-conservative operator directly:

`T[i,j,k] = sum_x Phi[x,i] G[x,j] (D- G)[x,k]`.

Then `q(h) = Phi^T N(Gh) = h^T T h`. In 2D, combine `D-_x G + D-_y G` into one tensor. Store `Q = T + T^T`; then `q = 0.5 h^T Q h` and `dq/dz = (Qh)(dh/dz)`.

**Exactness.** Exact to roundoff wherever every decoded center value is positive. For sign-upwind,

`N_up = u Dc u - (dx/2)|u|D2u`,

so fixed-backward differs only by `-dx 1[u<0] u D2u`. At exactly zero, values agree but Jacobians can differ because the code selects the forward branch. A prior N=256 CPU diagnostic found 6.9% negative decoded values, minimum `-1.2e-2`, yet on-manifold projected-advection errors were median about `1e-7`, maximum about `1e-4`; perturbed off-manifold states had much worse tails. Thus it is nearly oracle-quality here, but not formally sign-oracle-exact without an LM-iterate sign audit.

Build the tensor from `u D-u`, not `D-(u²/2)`: the latter is a different conservative FOM.

**Online cost.** Analytic residual plus Jacobian is roughly 0.10M flops for `M=R=32,K=8`, and 0.67–0.80M for `M=R=64,K=8–16`. Dense f64 storage is 256 KiB in 1D and 2 MiB in 2D; packed symmetry halves that, although packing may add a gather kernel. The optimized sampled implementation already uses two logical post-head reductions. Stock tensor code may use three; folding `A`, `Qh`, `h`, and their tangents can plausibly tie at two. Therefore this is not automatically faster.

**Offline cost.** `O(nMR²)`, once per bank/grid/test basis: about 134M FMA at 1D `n=4096`, 17B at 2D `256²`, and 275B at `1024²`. A blocked contraction is mandatory; materializing `n×R²` is unnecessary.

**What could go wrong.** Negative off-manifold LM candidates; an indexing/ghost-cell mismatch; an extra reduction launch erasing arithmetic savings. In 1D the likely gain is oracle accuracy and deletion of NNLS/stage 2, not speed over the 8–9 ms FOM.

**Smallest decisive experiment.** At the existing N=512 checkpoint, record signs at every accepted and rejected oracle LM candidate. Gate tensor versus direct fixed-backward `q`, residual, and Jacobian at `1e-12`. If no sign switches occur, require identical stops and rollout fields within `1e-9` of the sign oracle. If switches occur, label it approximate and require oracle-error drift below `1e-5` absolute. Time tensor, optimized NNLS-32, and oracle with paired repetitions.

## 2. Make exactness unconditional

**Idea.** For general-sign data, choose a polynomial discretization. Constant-alpha Rusanov gives a central quadratic flux plus fixed linear dissipation; the latter collapses through another `M×R` matrix. Alpha may be the trajectory’s known maximum initial amplitude. Centered or skew-symmetric fluxes are also quadratic.

A more invasive alternative is retraining with `G>=0` and `h>=0`, making fixed backward upwinding exact on this family.

**Exactness.** Rusanov is exact relative to its newly defined FOM for every sign pattern. It is not exact relative to the incumbent upwind truth. Nonnegative factorization preserves the incumbent stencil but may require substantially larger nonnegative rank. Clipping, softplus on the final field, or a smooth sign destroys finite polynomial tabulation.

**Online/offline cost.** Same tensor cost as candidate 1; Rusanov adds a negligible linear term. Both routes require new truth data or decoder training.

**What could go wrong.** Rusanov can over-diffuse; centered schemes can become unstable outside the current cell-Péclet regime; positivity constraints can raise the reconstruction floor.

**Smallest decisive experiment.** First compare eight Rusanov/centered trajectories at N=256 against their N=4096 self-convergence and the incumbent truth. Only if acceptable, retrain one decoder and demand tensor-to-new-oracle rollout parity at `1e-9`. Separately, a positivity arm must retain reconstruction near `3.7e-3` before any ROM test.

## 3. Compress the dense tensor

**Idea.** Fit symmetric CP or Tucker factors, or project the head outputs onto their leading `r` directions and store an `M×r×r` tensor. Node quadrature itself is a constrained CP representation; free factors may need lower rank.

**Exactness.** Exact only at untruncated rank; otherwise approximate, although its analytic Jacobian is exact for the compressed residual.

**Online cost.** Symmetric CP rank `L` costs approximately `O(L(R+M+MK))` and two or three reductions. Head projection costs one foldable linear map plus `O(Mr²)`.

**Offline cost.** Build the dense tensor, factor it, and certify values and gradients on training, perturbed, and actual LM states.

**What could go wrong.** Small value error can rotate `J^T r` or change an LM basin. At 2 MiB, compression may add kernels without saving useful bandwidth.

**Smallest decisive experiment.** Rank sweep against dense `Q`; require residual/Jacobian relative error below `1e-6`, gradient cosine above `0.9999`, and oracle rollout-error drift below `1e-5`. Adopt only if paired timing beats dense `Q`.

## 4. Product-closed spatial banks

**Idea.** Replace or re-express the learned bank in sine/Fourier modes. Products become analytic convolutions, `B` becomes trivial, and no grid contraction is needed offline.

**Exactness.** Exact for the chosen discrete polynomial operator and retained modes. With the incumbent backward stencil, half-grid phase shifts make the sine tensor substantially dense; classic sparse product closure appears with centered differentiation. Post-hoc re-expression is approximate.

**Online/offline cost.** Dense contraction remains comparable to candidate 1; the benefit is architectural simplicity and resolution transfer, not speed. A scratch diagnostic found sine-32’s free-coefficient projection floor below the learned bank’s, but that does not prove the nonlinear K=8 head will train equally well.

**What could go wrong.** Gibbs effects for thinner fronts, loss of bank/head co-adaptation, or needing enough modes to erase sparsity benefits.

**Smallest decisive experiment.** Train one fixed sine-bank `P=32` arm, then `P=48` only if needed. Require held reconstruction no worse than `3.7e-3` and tensor rollout matching its own oracle near `6e-3`.

## 5. Lift non-polynomial pieces

**Idea.** For sign-upwind introduce `v=|u|`:

`N = u Dc u - (dx/2) v D2u`.

Decode `v = Gv hv(z)` with the same latent variable and tabulate the resulting bilinear tensor. The same pattern applies to auxiliary fields for exponential, rational, or constitutive nonlinearities.

**Exactness.** Learned approximation unless the lifted relation is enforced exactly. Here its error is multiplied by `dx`; for an O(1) nonlinearity, the lift’s decoder floor directly becomes residual error.

**Online/offline cost.** One extra head and tensor contraction per lift, usually one or two additional reductions. Offline work includes lifted snapshots, training, and full-grid certification.

**What could go wrong.** Moving kinks in `|u|`, inconsistent auxiliary fields off-manifold, and extra kernels. It is unnecessary for the current positive family.

**Smallest decisive experiment.** Use one deliberately sign-changing 1D family. The lifted ROM must match its sign-upwind oracle within twice the oracle’s own error and preserve gradient cosine above `0.999`.

## 6. Learn the projected nonlinear term

**Idea.** Train `q(h)=Phi^T N(Gh)` directly. Prefer the exact quadratic tensor plus a learned sign-switch correction; use a generic MLP only for black-box nonlinearities. Train on values and Jacobians from on-manifold codes, trust-region perturbations, and actual rejected LM candidates.

**Exactness.** Learned approximation. A quadratic-regression scratch check already showed fat off-manifold tails, despite good median error.

**Online/offline cost.** A small MLP adds roughly 3–5 launches per residual/Jacobian evaluation, potentially 15–20% to this launch-bound rollout. Teacher generation remains full-grid offline.

**What could go wrong.** Untrusted gradients, extrapolation, and basin changes. A low training loss or NNLS-style fit metric is insufficient.

**Smallest decisive experiment.** Only after the plain tensor demonstrably fails on sign-changing data: require held-out gradient cosine above `0.999`, rollout within `1e-3` relative of the oracle, and no more than 20% timing overhead.

# Ranking

1. Dense fixed-backward tensor on the existing positive family.
2. Constant-alpha polynomial FOM when general-sign robustness is required.
3. Tensor compression, but only after dense 2D profiling shows a real cost.
4. Fixed trigonometric bank as a longer-term simplification experiment.
5. Lifted auxiliary fields for unavoidable non-polynomial terms.
6. Learned projected residual; quadratic regression only for black-box operators.

# What I would do first

Build the blocked 1D symmetric tensor and perform the sign/value/Jacobian audit at N=512 before optimizing anything. The decisive fork is simple:

- If all actual LM candidates remain positive, this is a genuinely sample-free, oracle-exact Burgers residual; require `1e-9` rollout parity.
- If candidates switch sign but rollout drift remains below `1e-5` absolute, adopt it only as a clearly labelled fixed-stencil approximation.
- If the tails alter stops or rollout accuracy, do not add a learned correction yet; test a polynomial FOM for the intended future data class.

Then port the unchanged construction to 2D, combine both advection axes into one `64³` tensor, and compare it within one paired job against the optimized `m=256` rule. That is where sample-free oracle accuracy could also become a meaningful speed result.