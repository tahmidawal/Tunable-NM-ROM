# Heat in three spatial dimensions: current NM-ROM pilot

Prospective design committed before the first scientific job. This is a development pilot, with no final-test cases opened and no performance result assumed.

## Problem and reference

Solve $u_t=\nu\Delta u$ on the unit cube with zero Dirichlet walls, $\nu=0.02$, and initial fields $64A\prod_{j=1}^3 x_j(1-x_j)\exp(-\|x-c\|^2/(2w^2))$. Draw centers independently in $[0.35,0.65]^3$, width in $[0.10,0.15]$, amplitude in $[0.8,1.2]$. Each case consumes one fixed-length random row. Draws are generation metadata only; online queries receive the sampled initial field. Save times $0,0.1,\ldots,0.5$. All axes are spatial; time is a separate output axis. $N$ means intervals, with $(N-1)^3$ interior unknowns.

The same-grid reference is exact exponential propagation of the seven-point discrete Laplacian using a three-axis orthonormal DST-I. Continuum eigenvalues with nested finer-grid input and restriction give a separately reported empirical physical reference. Require transform, stencil, analytic eigenmode and weak-operator identities within $10^{-11}$; a nontrivial rollout must advance beyond its first step. Reference refinement must be below $10^{-4}$ relative current-state error before physical comparisons are accepted; otherwise physical columns remain provisional. Keep initial, evolved-time, all-time, current-norm and initial-norm errors separate.

## Model and training

Train the current learned coordinate bank $G_\theta(x)$ (Fourier-feature MLP with boundary mask) with free training coefficients, then a nonlinear coefficient head $h_\psi(z)$ (MLP plus linear skip) in the complete field metric. No POD initialization or substitute bank. The first job uses $N=32$, bank rank $R=64$, latent sizes $K=8,16$, 128 training trajectories and 16 disjoint validation trajectories. Separate training/data/minibatch seeds are in config. The final seed is reserved and unused. Train the bank for 6000 minibatch updates and each head for 8000 updates; persist loss arrays and resumable optimizer state every 1000 steps. These are bounded training endpoints, not claims of convergence. Validation chooses between endpoints using worst evolved current-relative error, with median and failure counts retained. The complete training snapshot matrix and parameter table are regenerated on cluster from the recorded seeds.

Construct one nested correction basis from training head reconstruction residuals in the field metric. The online decoder is $D_q(z,y)=G(h(z)+C_qy)$; prefixes $q=0,8,16,32$ are measured, plus the separately labelled unrestricted $R$-coefficient endpoint. Use a fixed $M=256$ lowest-frequency sine test space in the principal ladder. Linear corrections are eliminated exactly using QR; each nonlinear solve is over $K$ coordinates. Recover and carry the previous complete coefficients into the next time step, including corrections. Time stepping uses the current heat weak Crank--Nicolson formulation with $\Delta t=0.025$ and a half-step diagnostic. Preserve attempted/accepted iteration counts and stationarity. At $q=R$ the head is redundant: do not impose a fictitious head-rank condition.

Build cold-start quadrature using nonnegative least squares on decoder-output snapshots, with a nominal candidate support around $4M$, and certify its moment error against dense contraction on validation fields. The exact linear weak operator is assembled offline. Query inputs and dense requested outputs are always charged. A failed quadrature certificate is explicitly a failed sampled arm; retain a labelled dense diagnostic to identify sampling error, without claiming mesh-independent initialization. No strong-form collocation is used.

## Controls, timing and acceptance

Compare same-job direct DST propagation, unrestricted learned-bank weak evolution, POD-Galerkin at $K$, $K+q$ and $R$, and the frozen-head correction ladder. POD uses exactly the training snapshots available to the bank. Full-bank and POD projections include the input contraction and full output. Use both the identical weak CN rule and an exact reduced linear semigroup where appropriate, clearly labelled. Operator baselines are the next job: 3D FNO and U-Net first, then DeepONet/Transolver as time permits, on the same training/validation data and output contract.

An additional full-bank Galerkin semigroup supplies a symmetric, dissipative linear-bank control. Best-found nonlinear field fits use truth only in separately labelled representation diagnostics; their codes never initialize an online query. The principal bank/head training remains based solely on training snapshots. At each mesh, POD is rebuilt from that mesh's regenerated training fields; the neural bank/head remain frozen across meshes, and this distinction is disclosed.

Burn in the GPU immediately before each timed block. Randomize method order per case/repetition; persist every paired timing/error invocation, complete selected fields, iteration arrays, hardware/precision/seed/source hashes and setup/training costs. Report median and worst error, median time, failure and timing-outlier counts. No cross-job timing ratios. Accuracy targets of 5%, 2% and 1% are reported as descriptive thresholds, not selection gates moved after inspection. A representation advantage, a monotone rank ladder and a runtime advantage are separate conclusions. A head that loses to linear controls is retained honestly.

## Execution and next decisions

Exclusive worktree `2026-09-20-paper-h3d`, base `02ff0f1f18db37d8589b28e90c3a93dae5bc5a88`; cluster namespace `paper_h3d_20260920`, unique directories per attempt. One single-GPU job at once, first walltime at most two hours. Local JAX smokes under one minute through `jaxrun`. All arithmetic f64/highest; mandatory GPU preflight. Source is content-hash pinned before staging; data is regenerated, results checksum-collected, and only the completed verified remote attempt is removed. Root owns the canonical lab log and paper reports. No merges or pushes.

After inspecting bank floor, head fits, stationarity and learning curves, prioritize the responsible limitation, record any amended exploratory settings before running them, and leave final cases unopened until all configurations freeze. Larger meshes, independent training seeds and all operator families remain outstanding after this pilot.

## Glossary

- **Bank / head:** learned spatial functions / nonlinear map from small latent vectors to bank coefficients.
- **$R$, $K$, $q$:** spatial-bank rank, head latent dimension, and additional solved correction dimension.
- **$M$, $m$:** number of smooth weak tests and number of quadrature samples.
- **Weak form:** PDE residual integrated against smooth test functions.
- **DST / CN:** discrete sine transform / Crank--Nicolson time discretization.
- **POD:** linear basis obtained from training-snapshot singular vectors.
- **NNLS / quadrature:** nonnegative least-squares weight fit / weighted approximation of an integral or grid sum.
- **Current-relative / initial-relative:** error divided by the reference norm at that time / by its initial norm.
- **Stationarity:** small objective gradient; not proof of a globally best representation.
- **Development / final:** data available for tuning / untouched data reserved for evaluation after selection freezes.
- **Same-grid / physical reference:** exact evolution of the discrete spatial operator / finer approximation to the continuum PDE.
