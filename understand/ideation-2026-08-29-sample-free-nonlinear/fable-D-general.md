# Sample-free nonlinear residuals — angle D: generality beyond quadratic Burgers, and the contrarian view

Agent D. Read `b1d_common.py`, `b1d_fast_common.py` (lean residual + LM body), `sep_burgers_exlin.py` (2D upwind), `PROFILE.md`, `OPTIM-NOTES.md`. No jobs run.

## 0. Two facts that reframe the whole question

**Fact A — the sampled rule is already a tensor.** Because u = G h,

    Phi^T N(G h) = sum_x Phi[x,:] (G[x,:]·h) ((D G)[x,:]·h)   = sum_x  Phi[x,:] ⊗ G[x,:] ⊗ (DG)[x,:]  contracted with h ⊗ h.

The full-grid oracle is T written as a CP decomposition of rank n whose factor vectors are the *rows* of Phi, G, DG. An m-node EQ rule is the same tensor truncated to CP rank m with weights w and factors constrained to be grid rows. The dense T (M R^2) is the "rank R^2" exact form. So oracle / nodes / dense T / free-CP are one family, differing only in how T is stored and contracted — the debate is about storage and kernel count, not about kinds of approximation.

**Fact B — the sign-upwind stencil decomposes exactly as centered + O(dx) dissipation.**

    D^- u = D^c u - (dx/2) D2 u,   D^+ u = D^c u + (dx/2) D2 u
    ⇒  N_upwind(u) = u·D^c u  −  (dx/2) |u| · D2 u          (exact identity, D2 = 3-point Laplacian)

The only non-polynomial piece is |u|, and it multiplies an O(dx) *artificial viscosity* term. Everything below follows from these two identities.

## Candidates

### D1. Dense tensor T from the frozen bank (coordinator's candidate), positive data

- **Idea.** T[i,j,k] = Σ_x Phi[x,i] G[x,j] (D^- G)[x,k], M×R×R. Residual q(h) = h^T T h; Jacobian (T_sym h)(dh/dz). 2D: D^-_x + D^-_y go in the *same* T (one tensor, not two).
- **Exactness.** Exact = oracle wherever the decoded field is > 0. At undershoot points (c<0) the FOM stencil would flip to D^+; the discrepancy per point is −c·dx·D2u with |c| = undershoot ≲ 1e-3·a in the tails, so the residual differs from the oracle's by ~1e-6 relative, not 1e-9. Note the oracle is not "more right" there: the FOM never sees a negative field. Prediction: rollout error matches the oracle arm to ≤1e-5 absolute, *not* bitwise.
- **Online.** 1D: M R^2 = 32k flops, 262 kB read; fold T into the existing lean matmul ([A ; T.reshape(MR,R)] @ h, then one (M,R)·h fusion) → **2 fusions** vs 3 for the lean sampled path (matmul, where/mult, PhiqT@). 2D R=64: 262k flops, 2 MB; R=96: 590k, 4.7 MB. Jacobian: T_sym h is (M,R) then one (M,R)@(R,K) — 2 kernels for all K tangents (the sampled path's linearize+vmap does comparable).
- **Offline.** One einsum: n·M·R^2 = 8 MFLOP (1D), 17 GFLOP (2D n=65k) — seconds. Memory trivially fits L2 (A100 40 MB) up to R≈280 at M=64.
- **What could go wrong.** Only the undershoots; and any latent visited *during* LM iterations (off-manifold) with larger negative lobes — but the FOM-consistent answer there is undefined anyway. Also: T from `u·D^- u` (current non-conservative FOM) ≠ T from `D^-(u²/2)` (they differ by (dx/2)(D^-u)²); build whichever the FOM uses. Flux form has the bonus that summation by parts moves D onto the analytic sines ((D^-)^T Phi_k = −D^+ Phi_k = half-grid cosines, exact, no boundary term since Phi and u²/2 vanish at walls) so T is symmetric in (j,k) and never differentiates the trained bank.
- **Decisive experiment (CPU-feasible, <1 min).** Load the committed N=256 checkpoint; build T; over all Z_tr and the 8 test codes report max‖h^T T h − Phi^T N_upwind(G h)‖/‖Phi^T N‖ and the count of grid points with c<0. Then drop q(h) into `make_full_rw` and rerun the oracle-arm rollout: **must land within 1e-5 of the oracle's 4.5–6e-3, and must beat NNLS-16/learned-16 (2.2e-2/1e-2) with the same 2-fusion residual.**

### D2. Lifting / polynomialization (Kramer–Willcox style) for sign changes and other nonlinearities

- **Idea.** For sign-changing data use Fact B: N = u D^c u − (dx/2) v·D2 u with v = |u|. Decode v with its own bank/head, v = G_v h_v(z) (shared latent, trained in stage 1 on |u_FOM| snapshots); then Phi^T(v·D2 u) = h_v^T T_v h, T_v[i,j,k] = Σ Phi G_v (D2 G). General non-polynomial N (exp u, 1/u, u^p): auxiliary fields w = f(u) with their own banks make every term multilinear in (h, h_w, …) → one tensor per term; in the LM least-squares framing the lifted consistency equations simply add residual rows (2M equations, K unknowns — no over-determination problem, unlike Galerkin lift&learn).
- **Exactness.** *Learned approximation* — v = G_v h_v ≠ |G h| exactly; the error enters the residual directly (unlike node sampling). For the upwind case it is benign: the lifted term is O(dx)-weighted, so a 10% error in v is a 10% error in the artificial viscosity, ≈ 3e-4 absolute at N=128 — below the 3.7e-3 decoder floor. For O(1) nonlinearities (exp, rational) the auxiliary decoder floor is the ROM floor.
- **Online.** Same as D1 per term (+1 head eval, +1 contraction per lifted term; ~+1–2 fusions). Offline: one more stage-1 field per lift (cheap in 1D, ~2× stage-1 in 2D), one einsum per tensor.
- **What could go wrong.** |u| has a kink at a *moving* zero crossing — a shock-like decoder-floor problem (fine here because of the dx weight, bad for O(1) lifts). Off-manifold z during LM: h_v extrapolates exactly as h does — no new failure class. Smooth sign (tanh(u/ε)) is still non-polynomial and gains nothing; positivity clamps are non-polynomial too. Don't.
- **Decisive experiment.** 1D, one sign-changing family (e.g. IC = a·sin(2πx)·blob, N=256): train (G,h) + (G_v,h_v); compare rollout of [D1-style T + T_v lift] vs the upwind oracle on that data. Pass = within 2× oracle error. Cost: one stage-1 run.

### D3. Higher-order tensors and when compression is needed (cubic/quartic; 2D/3D at R=64–96)

| term / dims | dense size (f64) | read/eval @1.5 TB/s | verdict |
|---|---|---|---|
| quadratic, 1D M=32,R=32 | 262 kB | 0.2 µs | dense, no compression |
| quadratic, 2D M=64,R=64 / 96 | 2 / 4.7 MB | 1.4 / 3 µs | dense (≤ one launch) |
| quadratic, 3D M=128,R=128 | 16 MB | 11 µs | dense, borderline (≈2 launches) |
| cubic, 1D 32·32³ | 8 MB | 5 µs | dense OK; symmetric (Veronese) 1.5 MB |
| cubic, 2D 64·64³ | 134 MB | 90 µs | **loses** — compress |
| quartic, 2D 64·64⁴ | 8.6 GB | — | must compress |

Launch cost ≈ 5–10 µs (measured ~0.14–0.26 ms per LM iteration over ~20–25 fusions), so dense storage is free until M·R^d·8 B ≳ 10 MB. Compression options, in order of preference:

1. **Head-PCA Tucker (exploits the small head).** h(z) over the training codes spans only ~K'≈K..2K effective directions (K=8). Take P = top-K' left singular vectors of H = [h(z_tr)], store T' = T ×₂P ×₃P… (M·K'^d: cubic 2D = 64·16³ = 262k) and contract with h' = P^T h. Exact iff h ∈ span P; the SVD tail of H is the error bound — a 1-second check. Adds one (R×K') matvec (1 fusion, foldable into the head's last layer). This is the right default for cubic+ and for 3D.
2. **Symmetric/Veronese storage**: R(R+1)(R+2)/6 monomials (5.5× less at R=32) — but forming monomials is an extra gather kernel; only worth it when it drops T below L2.
3. **Free-CP (ALS on T)**: rank r storage r(M+2R), 2–3 fusions. This is exactly "learned quadrature with free node features" — same three-kernel structure as m nodes but unconstrained factors, so it can reach rank r < m at equal error. Gradient exact given the factors. Only worth it if head-PCA fails (h not low-dimensional).
4. Randomized sketching of T: unnecessary at these sizes — T is built exactly in seconds.

Fourier/trig banks "closed under products" (sparse analytic T): would require retraining a bank that is not an MLP over RFF, whose floor is unknown and probably worse (the MLP bank beats POD because it is *not* trig), and the bc(x) mask breaks closure anyway. Not worth it while dense T costs 2 MB.

### D4. Devil's advocate — is the m-node rule the right design after all?

Kernel and byte counts per residual evaluation (XLA fusions, estimated from the lean path):

| path | fusions | bytes read (1D N=256) | bytes (2D N=256, R=64, m=256) | error class |
|---|---|---|---|---|
| oracle (full grid) | 3–4 | 8·n·R = 65 kB | **33 MB (G itself)** | exact |
| m nodes (lean) | 3 | 33 kB | 0.8 MB | sampling: 5e-3 (m=32) … 2e-2 (m=16) |
| dense T | 2 | 262 kB | 2 MB | exact (mod undershoot) |
| head-PCA Tucker | 3 | ~20 kB | ~0.1 MB | ‖(I−PPᵀ)h‖-bounded |

Quantified verdict:
- **1D: no speed story anywhere.** Oracle, nodes and T all sit at the launch floor (~15 ms/traj vs FOM 8–9 ms); the entire sampling apparatus buys ~1 fusion. The tensor cannot make the ROM faster than the sampled rule — it makes it *exact* and deletes NNLS, node training, the m sweep, and the checkpoint-dependent node fragility noted in the exlin memory.
- **2D at R=64–96, M=64:** T (2–5 MB) ≈ nodes (0.8 MB) ≪ oracle (33 MB, the true cost the nodes were invented to avoid). Tensor ties on kernels, wins on error (removes the ~1e-2 sampled error of tight/learned nodes).
- **The tensor loses** when (i) a term is cubic+ in 2D/3D without a low-dimensional head (then Tucker/CP is needed and CP *is* a node rule with free factors); (ii) R ≳ 200 at M=64 (T leaves L2); (iii) N is non-polynomial with no cheap lift (limiters, WENO, exp) — sampling evaluates *any* N pointwise for free and *shares* the node evaluations across all terms, whereas every polynomial term needs its own tensor; (iv) coefficients vary in x (κ(x) u u_x adds a tensor index). So: nodes are the right design for generic nonlinearity; tensors are the right design for polynomial (or liftable) nonlinearity — which includes every Burgers/NS-type convective term.
- Batching favours T further: T is read once per vmapped batch; node features scale per trajectory.

### D5. Choose the FOM to be polynomial (sample-free by construction)

- **Idea.** Replace sign-upwind with a polynomial flux: centered `u D^c u`, skew-symmetric `(1/3)(u D^c u + D^c(u²/2))` (discretely energy-conserving → backward Euler unconditionally energy-dissipative), or global Lax–Friedrichs/Rusanov `D^c(u²/2) − (α dx/2) D2 u`. The LF dissipation is a *Laplacian*, so in the weak form it collapses through Λ B h — the whole residual becomes `W[B(h−h_prev) + dt(hᵀTh + (ν + α dx/2) Λ B h)]`, the Poisson path plus one contraction. α = max|u₀| per trajectory (known from the IC by the max principle) keeps LF monotone and still exact.
- **Exactness.** Exact w.r.t. its own FOM, for *any* sign pattern, no undershoot issue, no lift.
- **Accuracy vs the sign-upwind FOM at the resolutions in play.** Cell Péclet Pe = a·dx/ν, dx = 1/(N−1), a ≤ 1.5: N=128 → Pe ∈ [0.12, 1.18]; N=256 → [0.06, 0.59]; N=512 → [0.03, 0.30]; N=4096 → ≤0.04. **Pe < 2 for every case in the family**, so centered differencing is non-oscillatory (M-matrix) everywhere — there is no stability reason to upwind. Meanwhile the upwind FOM carries ν_num = |u|dx/2 ≈ 0.0059 at N=128 (a=1.5): a **59 % excess viscosity at ν=0.01** (30 % at N=256, 6 % at ν=0.1) — an O(dx) truth error far larger than the 4e-3 ROM floor. Centered is O(dx²) and closer to the PDE; LF is *more* diffusive than upwind in the tails (α ≥ |u|) but uniformly so, and identical in order. Consequence: the upwind and centered "truths" differ by O(10 %) at N=128, ν=0.01 — you cannot swap the FOM and grade against old data; regenerate (tridiagonal Newton, seconds) and grade the ROM against its own FOM, which is legitimate since the FOM is a modelling choice. The paper's comparison to prior upwind numbers changes; the resolution-convergence story improves.
- **Online/offline cost.** Same as D1 minus the undershoot caveat. Data regen: seconds (1D), minutes (2D).
- **What could go wrong.** Newton on the centered Jacobian (non-symmetric tridiagonal) — fine at Pe<2; the u ≥ 0 max principle no longer strictly holds for centered (does for LF). If the family is later pushed to Pe>2 (coarser N, lower ν), use skew-symmetric or LF, not plain centered.
- **Decisive experiment (CPU, minutes).** N=256, seed 0/1: generate the 8 test trajectories with upwind, centered, skew-symmetric, LF; also at N=4096 as PDE reference. Report ‖u_scheme,N − u_scheme,4096‖ (self-convergence) and ‖u_centered − u_upwind‖/‖u‖ at N=128/256. Expect centered ≫ upwind in accuracy at N≤256 and the cross-scheme gap ≈ 0.05–0.2 at ν=0.01. If so, retrain stage 1 on centered/skew data (floor should be unchanged or slightly better — smoother fields) and run the D1 tensor ROM: must equal the *new* oracle to 1e-9 (now truly exact) and sit at floor + ~1e-3.

## Ranking

1. **D1 + flux-form T (now, positive data)** — exact up to undershoots, 2 fusions, removes stage 2 entirely; decisive test costs a minute.
2. **D5 (choose a polynomial FOM)** — makes D1 exactly exact for any data and *improves the truth*; the Péclet numbers say upwinding was never needed for this family. Do it as the follow-up because it changes the dataset and the published comparison.
3. **D3 head-PCA Tucker** — the mechanism that keeps D1 viable for cubic terms and 3D; check the SVD of H first (free).
4. **D2 lifting** — needed only for O(1) non-polynomial N or if one insists on sign-upwind with sign-changing data (where it works because the lifted term is O(dx)).
5. **Learned surrogate h → Phi^T N(G h)** — unstructured regression with gradient trust and off-manifold issues; strictly dominated by D1/D3 for polynomial N and by D2 otherwise. Skip.
6. Trig banks / sketched T — no need at these sizes.

## What I would do first

Write one CPU script (no GPU): load the N=256 checkpoint, build T (both `u·D^-u` and `D^-(u²/2)` forms), check h^T T h against Phi^T N_upwind(Gh) on every training/test code (expect ≤1e-6 rel, discrepancy located at undershoot points only), then run the T residual through the existing `make_full_rw`/LM rollout and confirm it reproduces the oracle arm's 4.5–6e-3 to ≤1e-5. In the same script, generate the 8 test trajectories with the centered and skew-symmetric FOM and print the upwind-vs-centered gap and the self-convergence to N=4096 — that single table decides whether D5 becomes the new default FOM. Then state plainly in the lab log: in 1D the tensor changes exactness and simplicity, not speed; the speed comparison vs the tridiagonal FOM is unchanged.

## Glossary

- **Bank G / head h**: frozen spatial shapes (n×R) and latent-to-coefficient MLP; u = G h(z).
- **T**: precomputed M×R×R table giving Phi^T(u·u_x) as a quadratic form in h.
- **Oracle**: nonlinear term summed over the full grid — the best any node rule can do.
- **m nodes / NNLS**: quadrature points and nonnegative weights approximating the oracle.
- **CP / Tucker**: tensor decompositions as sum of rank-1 terms / small core with factor matrices.
- **Lift**: introduce an auxiliary field w = f(u) so a non-polynomial term becomes polynomial in (u, w).
- **Péclet number Pe = u·dx/ν**: ratio of advection to diffusion at the cell scale; centered differencing is non-oscillatory for Pe < 2.
- **ν_num**: artificial viscosity implicitly added by upwinding, |u|dx/2.
- **Fusion / kernel**: one GPU launch; the ROM's cost is the number of these, not flops.
