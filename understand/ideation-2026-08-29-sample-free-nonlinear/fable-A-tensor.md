# Sample-free Burgers residual — angle A: the precomputed quadratic tensor and the sign switch

Code read (read-only): `b1d_common.py` (`upwind_adv_field_1d`, `tridiag_jac`), `sep_b1d_scale.py`
(`full_r_w`, gate F), `b1d_fast_common.py` (`make_device_fast`, lean/onepass path),
`sep_burgers_exlin.py` (2D `u_and_N_ex`). Two facts from the code that matter below:
the FOM is the **non-conservative** form `N = c * ux` with the switch on the *centre* value
`c = u[i]` (both axes in 2D switch on the same `c`), and the oracle Jacobian is `jacfwd`
through `where`, i.e. the switch treated as constant. No GPU runs were done.

## Candidates

### 1. Fixed-backward-difference tensor T⁻ on the existing FOM (coordinator's candidate, sharpened)

**Idea.** Split the FOM's stencil exactly. With `D⁰` central, `Lap` the 3-point Laplacian:
`D⁻u = D⁰u − (dx/2) Lap u`, `D⁺u = D⁰u + (dx/2) Lap u`, so pointwise

    N(u) = u·D^up u = u·D⁰u − (dx/2)·|u|·Lap u .

Sign-upwind is "central plus |u|·dx/2 artificial viscosity"; `|u|` is the only non-polynomial
piece. On a nonnegative field `|u| = u` and `N = u·D⁻u` exactly, quadratic. Define

    T[i,j,k] = Σ_x Phi[x,i] G[x,j] (D⁻G)[x,k]     (M×R×R, on the interior grid, ghost zero at the
                                                   wall is automatic because bc(0)=0)
    Phi^T N(Gh) = h^T T h ,   2D: D⁻ := D⁻_x + D⁻_y, same size tensor.

**Exactness (worked out).** `h^T T⁻ h − Phi^T N_oracle(Gh) = −dx · Phi^T[ 1[u≤0] ⊙ u ⊙ Lap u ]`.
So T⁻ equals the full-grid oracle *to machine precision on every field with min u ≥ 0*, and on
a field with undershoot ε in a region of width ℓ the discrepancy is `O(ε²/ℓ)` per point (u is
O(ε) there and `dx·Lap u` is O(ε·dx/ℓ²)·... in practice ≤ 2|u||Du| ≈ ε²/ℓ), against
`|N| ≈ a²/w ≈ 10` in the blob. With the measured decoder floor (3.7e-3 rel-L2 → pointwise
ε ~ 1e-3 to 1e-2) the relative residual perturbation is 1e-6 to 1e-4, and it is a *consistent*
perturbation (same operator the FOM uses on positive data), so its effect on rollout error is
second order — far below the 4.5–6e-3 oracle rollout error. The FOM truth is positive by the
max principle, so on this family "the FOM with a fixed D⁻" **is** the FOM: T⁻ is exact for the
truth-generating operator and only differs from the ROM's own oracle on ROM undershoots.
Exactness class: exact on the positivity cone; O(ε²) consistent elsewhere.

**Solver safety of the wrong branch.** T⁻ on a negative pocket amounts to local viscosity
`ν + u·dx/2` (negative where u<0). Net viscosity stays positive iff `u > −2ν/dx`, i.e.
`u > −2.5` at N=128, ν=0.01, and `−82` at N=4096. No LM iterate on this family comes near
that, so T⁻ cannot turn anti-diffusive mid-solve.

**Jacobian (exact formula).** With `H = ∂h/∂z` (R×K, already produced by `linearize`):

    ∂r/∂z = W [ (1 + dt ν Λ) B + dt ( T₍₁₎(h) + T₍₂₎(h) ) ] H,
    T₍₁₎(h)[i,j] = Σ_k T[i,j,k] h_k   (∂ through the u slot),  T₍₂₎(h)[i,k] = Σ_j T[i,j,k] h_j  (Du slot).

With the symmetrized tensor `T₍₁₎+T₍₂₎ = 2 Σ_k T_sym[i,j,k] h_k`. This coincides with the
oracle's `jacfwd`-through-`where` Jacobian wherever u>0 (the switch is constant there), so LM
iteration counts and stop reasons should match the oracle arm, not just the final error.
`jax.linearize` on the bilinear form gives it for free — no hand-coding needed.

**Symmetric savings.** Only the (j,k)-symmetric part enters `h^T T h`; store
`T_sym` as `M × R(R+1)/2`: 1D 32×528 = 16.9k f64 (135 KB) instead of 32.8k; 2D 64×2080 =
133k (1.06 MB) instead of 262k. Online: `p = vech(hhᵀ)` (off-diagonal doubled, 528 mults)
then one GEMV. The flux-form tensor (below) is symmetric by construction.

**Flux form?** Build T from whatever the FOM actually is. The FOM is `u·D⁻u`, and the flux
form `D⁻(u²/2) = ((u_i+u_{i−1})/2)·D⁻u` differs by `(dx/2)(D⁻u)²` — a different O(dx)
scheme (conservative; correct shock speeds, but at ν ≥ 0.01 the layers are resolved so it
hardly matters). If the FOM is ever switched to flux form the tensor is
`T[i,j,k] = ½ Σ_x (D⁻ᵀPhi)[x,i] G[x,j] G[x,k]` (summation by parts: derivative moves onto the
sine test mode, giving near-cosine weights) — same size, symmetric, same cost. Do **not** build
the flux tensor against the current non-conservative truth; gate F would fail by O(dx).

**Online cost (per residual, f64).**

| | 1D (R=M=32, m=32) | 2D (R=M=64, m=256) |
|---|---|---|
| sampled NNLS rule | 3mR + mM ≈ 4k flops; tables 4k entries; kernels: gather-einsum, where, product, matvec | 5mR + mM ≈ 98k flops; 98k entries (0.8 MB) |
| full-grid oracle | 2nR + stencil: 8k (n=128) … 262k (n=4096) | 2nR: 0.5M (N=64) … 8.4M + 33 MB traffic (N=256) |
| **tensor T_sym** | 34k flops, 17k entries (135 KB), kernels: vech (fused elementwise) + 1 GEMV | 266k flops, 133k entries (1 MB, L2-resident), same 2 kernels |

Jacobian via `Q(h) = 2 T_sym·h` (MR² flops) then `Q·H` (MRK), or batched linearize (K× the
residual). Everything is ≪ 1 μs on an A100; the rollout stays launch-bound at ~15 ms, so in 1D
expect **parity, not speedup** — the win is exactness with no NNLS, no stage 2, no m to tune,
and strict n-independence (oracle-quality at n=4096 with no grid work). In 2D the tensor is
~3× the flops and 1.3× the memory of the m=256 rule but with fewer kernels (no stencil gather,
no `where`), so it should be at parity with m=256 while matching the oracle. Compression
(Tucker/CP) is unnecessary below R ≈ 200; T is a sum of n rank-1 terms with a fast-decaying
HOSVD spectrum in the Phi index, so it would compress if ever needed (cubic lifts, R ≳ 256).

**Offline cost.** Build once per (bank, grid): apply the stencil to the bank columns, then
`einsum('xi,xj,xk->ijk')` chunked over k (the n×R² intermediate is 2 GB at N=256 2D, so chunk):
1D n=4096: 134 MFLOP; 2D N=256: 17 GFLOP; 3D N=64 with R=96: ~150 GFLOP. Seconds on any GPU,
minutes on CPU. No training. Also `T` at different n converge to the continuous
`∫ φ_i g_j g_k' dx`, which is the mesh-free limit of the ROM.

**What could go wrong.** (i) Undershoots larger than assumed (check: min of `G h(z)` along
test trajectories; the bound above tells you the tolerance to expect). (ii) Ghost/index
convention mismatch between `D⁻G` and `upwind_adv_field_1d` (gate it). (iii) f32 anywhere:
T entries for high sine modes are small differences of large numbers; keep the build and
contraction in f64. (iv) Sign-changing data — see 2/3.

**Smallest decisive experiment.** (a) CPU gate: at the gate-F states and every oracle-arm z
along the test trajectories, `|h^T T h − Phi^T N(Gh)| / |Phi^T N|` must be ≤ 1e-13 on states
with min u ≥ 0 and ≤ the printed bound `dx·‖1[u≤0] u Lap u‖` otherwise. (b) In
`make_device_fast` (branch `X_v is None`) replace `Nu = upwind(G_int@hz); Pq = Phi_j.T@Nu`
by the T_sym contraction; run oracle and tensor arms on the same test set at N=128, 512,
4096. Pass criterion: rollout rel-L2 equal to the oracle arm to ≤ 1e-5 absolute (i.e.
4.5e-3 vs 4.5e-3 ± 1e-6 expected), identical LM iteration counts/stop reasons, e2e ms ≤
oracle and ≈ NNLS-32. One A100 job, < 10 min. (c) 2D: same swap in `sep_burgers_exlin.py`
(`Phi_q.T @ Nu` → tensor with `D⁻_x + D⁻_y`), compare to the full-grid and m=256 arms.

### 2. Make the FOM polynomial (general-sign data), then tensor is exact everywhere

Three sub-options, all "consistent with a modified FOM", all need truth regeneration
(minutes) **and** a stage-1 retrain (snapshots change by O(dx)):
- **Global Lax–Friedrichs/Rusanov, constant α ≥ max|u| ≈ 1.5:** `N = D⁰(u²/2) − (α dx/2) Lap u`.
  Quadratic + linear; the artificial-viscosity part folds *into the exact linear term* as
  `+(α dx/2) Λ B h` — the ROM just sees `ν + α dx/2`. Extra diffusion is 5.9e-3 at N=128
  (6–60 % of ν), 7e-4 at N=1024; only acceptable if the truth uses the same scheme.
- **Envelope Rusanov, α(x) = max over training snapshots of |u(x)|:** frozen spatial profile,
  so `−(dx/2) α(x) Lap u` is linear → one extra M×R matrix `Phi^T diag(α) Lap G`. Much less
  diffusive than global LF where u is small; monotone on the training family. Chicken-and-egg
  (α from upwind snapshots, then regenerate) and a nonstandard scheme — mention, don't lead.
- **Central / skew-symmetric FOM `⅓(u u_x + (u²/2)_x)`:** purely quadratic, second order,
  energy-conserving. Cell Péclet `u dx/ν ≤ 1.2` at N=128 so it is non-oscillatory on this
  family; note the current upwind FOM carries numerical viscosity `u dx/2` ≈ 60 % of ν at
  N=128, ν=0.01. Principled, sign-agnostic, and arguably a better FOM. Cost: regen + retrain.
Decisive experiment: after regen/retrain, the tensor arm must equal the new oracle to 1e-13
at all states (there is no branch left).

### 3. Split form on the existing FOM: exact T⁰ plus a sampled O(dx) remainder

`N = u D⁰u − (dx/2)|u| Lap u`: tabulate the central term exactly; sample only
`Phi^T[|u| Lap u]` with the NNLS/learned nodes. The sampled term is O(dx)·‖N‖, so sampling
error drops by a factor dx relative to sampling all of N. Exactness: exact + O(dx·sampling
error) for any sign pattern, on the unmodified FOM. Cost: T_sym plus the current m-node
machinery (kernel count ≈ current). Useful only if sign-changing data on the current FOM is
required; on positive data it is strictly dominated by 1. Decisive test: NNLS-16 arm with the
split residual should drop from 2.2e-2 toward the oracle 5e-3.

### 4. Learned / polynomial surrogate `h ↦ Phi^T N(Gh)` (R→M map, no points)

For Burgers this is dominated: the exact map is quadratic and candidate 1 *is* the surrogate
with zero fitting error. Keep it for genuinely non-polynomial N (Bratu `e^u`, `|u|` on
sign-changing data): degree-d polynomial in h gives an `M × C(R+d,d)` table (1D d=3: 209k;
2D d=3: 3M entries, 24 MB — borderline), or a small MLP trained Sobolev-style against the
teacher residual **and** its Jacobian (the stage-2 `teacher()` already emits both). Failure
modes: extrapolation of LM iterates off the training manifold in h, and an untrusted
Jacobian; both must be gated by the tensor's exactness test where a polynomial truth exists.
Kramer–Willcox lifting fixes `e^u`-type terms exactly (w = e^u ⇒ quadratic); it does not
polynomialize `|u|`, which is candidate 2's job.

### 5. Product-closed (trig) bank / sparse analytic tensors — not worth it

A pure sine bank makes T sparse and analytic (discrete orthogonality), but it throws away the
learned bank, and a dense 32³ or 64³ table is already trivially small. No gain at R ≤ 96.

## Ranking

1. **T⁻ tensor on the existing FOM/data (1).** Exact on the truth's positivity cone, O(ε²)
   on ROM undershoots, no training, no NNLS, kernel count ≤ current, one-job experiment.
2. **Polynomial FOM + exact tensor (2, central/skew-symmetric variant).** The honest
   general-sign answer; costs regen + stage-1 retrain — do it when sign-changing data or 2D/3D
   families demand it.
3. **Split form (3).** The bridge if the upwind FOM must be kept for sign-changing data.
4. **Surrogate/polynomial-in-h (4).** Only for non-polynomial nonlinearities.
5. **Trig closure / compression (5).** Shelve.

## What I would do first

Build `T_sym` from the frozen 1D bank (ten lines: `DG = (G − shift(G))/dx`, chunked einsum,
symmetrize), run the CPU exactness gate against `Phi^T upwind(Gh)` on the test-trajectory
states, and print the fraction of (step, point) pairs with u ≤ 0 together with the bound
`dx·‖1[u≤0] u Lap u‖`. If the gate says ≤ 1e-13 on positive states and ≤ 1e-5 elsewhere, swap
the contraction into the `X_v is None` branch of `make_device_fast` and run oracle-vs-tensor at
N ∈ {128, 512, 4096} on one A100: matching rollout error, identical LM counts, and parity
timing settles it. Then port to 2D against the m=256 arm, where the tensor should be the first
sample-free rule that is both oracle-exact and no slower than the sampled rule.
