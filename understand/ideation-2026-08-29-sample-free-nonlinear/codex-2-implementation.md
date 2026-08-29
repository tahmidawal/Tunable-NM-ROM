I completed the report, but the read-only sandbox rejected the sole authorized write to `codex-2-implementation.md`. I created no files and ran none of the prohibited commands. The full report is preserved below.

# Sample-free Burgers residual: implementation-realistic assessment

## Candidates

### 1. Dense symmetric tensor with fixed backward upwinding

**Idea.** Build it immediately after `G_int`, `Phi_j`, and `A_j`. In 1D, pad `G_int` with the same ghost-zero rows as `upwind_adv_field_1d`, form `DmG=(G-G_left)/dx`, and accumulate spatial blocks of

`T[m,j,k] = sum_x Phi[x,m] G[x,j] DmG[x,k]`.

Never materialize an `n x R^2` array. Accumulate blocked `einsum('xm,xj,xk->mjk',...)`; save dense `Q=T+T.swapaxes(1,2)` and discard `T`. Then `q(h)=Phi^T N(Gh)=0.5*sum_j (Qh)[:,j]*h[j]`.

In 2D, reshape and ghost-pad `G_int` consistently with `stencil_indices`, set `DG=(G-G_xminus)/dx+(G-G_yminus)/dx`, and build one combined tensor. Both code axes therefore need one online contraction, not two.

**Exactness.** This is algebraically exact for the full-grid oracle wherever every decoded center is strictly positive. At `u=0` the residual value still agrees, but its Jacobian may not: JAX’s `where(c>0,...,forward)` selects the forward branch at zero. Any negative decoder undershoot switches the stencil and breaks exactness.

The exact sign-upwind map is piecewise quadratic across all hyperplanes `G[x,:]h=0`; finding its region without evaluating every grid row is itself grid work. Also, the current `u_i(u_i-u_{i-1})/dx` is not the conservative flux difference `(u_i^2-u_{i-1}^2)/(2dx)`; a flux tensor would not match this oracle.

Let `H=dh/dz`, `v=Qh`, `L=(1+dt*nu*lambda)[:,None]*A`, and `w=(1+dt*nu*lambda)^(-alpha)`. The exact formulas are

`r = w*(Lh-prev+0.5*dt*sum_j(v[:,j]*h[j]))`

`dr/dz = w[:,None]*((L+dt*v)@H)`.

“`2Th`” is valid only for a symmetric `T`; the `Q` formula handles the nonsymmetric bank/stencil tensor exactly.

**Online cost.** Compute `A@[h,H]`, `v=Qh`, then `v@[h/2,H]`: about 102k f64 flops for 1D `M=R=32,K=8`. For 2D `M=R=64`, both axes included, it is about 0.67M flops at `K=8` or 0.80M at `K=16`; the current `m=256`, five-point sampled path is roughly 1.84M/3.48M.

Kernel reality is less favorable. The optimized sampled path has two logical post-head reductions: merged `[A;G_st]@[h,H]`, then `Phi_q^T@[N,dN]`, with the switch fused. Stock tensor code needs three: `A@[h,H]`, `Q@h`, and `v@[h/2,H]`.

Thus it adds one reduction kernel per initial/candidate residual-plus-Jacobian evaluation—approximately one per LM attempt under `nocond`. It removes node loads and switch work, but no guaranteed launch. A custom fused Pallas/Triton two-stage contraction could break even at two kernels, but “one tensor kernel” requires measurement.

**Offline cost and memory.** The blocked contraction costs about `2*n*M*R^2` flops: 0.27G at 1D `n=4094`; about 0.55T at 2D `N=1024`. Rebuild it for every decoder, grid, and `M`.

Dense `Q` is 256 KiB in 1D and 2 MiB in 2D. Separate 2D axis tensors would occupy 4 MiB and are unnecessary. Packed symmetry is 132 KiB/1.016 MiB, but gathers may cost more than the saved memory. Current 2D `G_st+Phi_q+A` storage is about 0.78 MiB.

**Wiring.** First add a tensor `r_w` beside `full_r_w` in `sep_b1d_scale.py`; its existing `make_device` provides a correctness arm using AD.

For speed, attach `Q` to `b1d_fast_common.Setup` and add a tensor branch to `make_device_fast` returning the analytic joint `r,JT` above. This matters: `sep_b1d_scale.make_device` and `blat_common._finish_ops` internally call `jacfwd(r_w)`. Changing only `rJ_ex` in `sep_burgers_exlin.py` does not change the LM kernel.

In 2D, retain `prev_of_ex=A@h`, replace sampled `u_and_N_ex/Phi_q.T@Nu`, return `A@H` as the third `rJ` output, and either make `_finish_ops` consume the supplied analytic `rJ` or use a local fast LM step.

**What could go wrong.** Tiny negatives defeat exactness; changed summation order prevents bit identity; the extra launch can erase the 1D flop saving; an unblocked 2D build attempts a roughly 32 GiB temporary.

**Smallest decisive experiment.** At the existing N=512 checkpoint, log `min(u)` and every `u<=0` at accepted and rejected oracle LM candidates. Compare tensor/oracle `q`, residual, and analytic Jacobian to relative `1e-12`.

Then run the unchanged eight-trajectory oracle protocol: require identical stop reasons, rollout-error difference at most `1e-9`, and latent deviation within the existing fast-path parity envelope. One switched point falsifies exact sign-oracle equivalence.

### 2. Guarantee positivity or choose a polynomial FOM

**Idea and exactness.** A nonnegative bank and nonnegative head coefficients preserve `u=Gh`, retain exact linear projections, and make backward upwinding fixed, but require retraining and may need larger nonnegative rank.

The less invasive fallback is a separately labelled modified FOM using conservative Burgers flux with constant-alpha Rusanov. That flux is quadratic plus fixed linear dissipation, so it tabulates into `Q` plus an `M x R` matrix. State-dependent `alpha=max|u|` is not polynomial.

**Cost, offline work, and failure.** Tensor cost is unchanged and the linear matrix is negligible. Positivity can hurt reconstruction; constant alpha can over-diffuse and changes truth. Neither should be silently substituted for the current oracle.

**Smallest decisive experiment.** Compare a fixed-backward oracle with the existing sign oracle on eight N=512 rollouts. If error differs beyond `1e-9`, label it a method change. For Rusanov, regenerate one truth/ROM cell, demand tensor-to-modified-oracle parity at `1e-9`, and separately report drift from incumbent truth.

### 3. Compress Q; redesign the bank only later

**Idea, exactness, and cost.** After dense `Q` works, fit symmetric CP/Tucker factors. Rank-`L` symmetric CP evaluates `q=A_c[(Bh)^2]` and its Jacobian in `O(L(R+M+MK))` with 2–3 reductions; it is exact only if factorization error is roundoff.

A Fourier/trigonometric product-closed bank could yield sparse analytic convolutions, but the present boundary-masked neural bank is not closed, so that is a new decoder experiment.

**Offline work, failure, and test.** Compression can preserve values while rotating `J^T r`. Sweep rank against saved dense `Q`; gate `q`, `J`, and `J^T r`, then require oracle rollout-error parity at `1e-9`.

Since dense 2 MiB already fits cache, adopt compression only for measured runtime. A product-closed bank must first match current reconstruction and oracle rollout at N=128.

### 4. Learned projection or lifting

**Idea and exactness.** Train a small `h -> Phi^T N(Gh)` map, preferably a quadratic baseline plus learned correction, on full-grid values and Jacobians from training, perturbed, and actual LM-candidate states. It is sample-free but approximate.

Lifting `u+`, `u-`, or `|u|` only moves the sign nonlinearity into closure constraints unless new reduced lifted coordinates are learned.

**Cost, offline work, failure, and test.** A two-layer MLP costs `O(width*(R+M))` but typically adds 2–4 GEMM/activation kernels plus derivative work, poor for this launch-bound loop. Oracle labelling is grid-bound and extrapolation can change LM basins.

Test on held-out rejected as well as accepted candidates, then the same eight rollouts. Require gradient cosine near one, identical stops, and `1e-9` rollout-error parity to call it lossless.

## Ranking

1. Dense symmetric fixed-backward tensor, conditional on a strict sign audit.
2. Constant-alpha polynomial flux, explicitly a modified-FOM study.
3. CP/Tucker only after dense 2D profiling; closed banks are longer-term.
4. Learned projection/lifting for nonlinearities that resist polynomialization.

## What I would do first

Build blocked 1D `Q` and add value/Jacobian/sign gates before optimization. If every evaluated decoded state is positive and parity passes, wire analytic `r,JT` into `make_device_fast`; otherwise stop calling the route exact and measure fixed-backward versus sign-oracle drift. Only then port it to 2D and test whether fewer flops beat the likely extra reduction launch.

Summary 1/3: The dense symmetric tensor is exact and small only within a fixed-stencil sign region.  
Summary 2/3: Stock JAX likely adds one reduction launch per r+J despite cutting 2D arithmetic by about 3–4×.  
Summary 3/3: Gate every LM candidate against the oracle first; use polynomial flux only as an explicit modified FOM.