# Separable-decoder first cell (2026-08-22)

Design + audited spec: `reports/2026-08-22-separable-eq-decoder-design.md` on main.

**Model** (pure neural, NO POD anywhere): u(x;z) = bc(x) * ( g(x)^T h(z) );
g = Fourier-feature MLP (2->128->128->r), h = MLP head (k->128->128->r) + linear
skip — nonlinear in z. Auto-decoder training (joint Adam over g, h, per-state
codes), f64, warmup-cosine, fixed data-scale constant folded into the features.

**Cells** (one cluster job, sequential): Poisson-2D and Burgers-2D at N=64,
(K,R) in {(8,32),(16,64)}, M=4K, m=4M=16K NNLS-EQ grid nodes, 40k train steps,
seed 0. Incumbent solvers only: ctol_tol.lm_tau_poisson (trust 1.0x radius) and
blat_common lm_step_jit/rollout_jit (trust 0.01x radius).

**Arms**: `meshfree` (network evaluated in-loop, incumbent path) vs `cached`
(feature banks G_q / stencil G_st cached once; no spatial network in the
compiled iteration). GATE 0 asserts the two arms' weak residual/Jacobian agree
<= 1e-12 relative — the discretization (incl. the FOM sign-upwind stencil) is
never changed, only how it is evaluated. Cost and error come from the same
invocation; FOM CG ladder / reference Newton rollout timed in the same job.

**Smokes** (local GB10, N=16): Poisson gate0 dev 0.0, solve error = oracle
error (manifold-limited); Burgers gate0 dev 1.6e-15, 50/50 steps, no blowups.

**Job**: Slurm 2802238, /cluster/tufts/paralab/tawal01/sepdec/sepdec_r1,
stage manifest-checked both directions.
