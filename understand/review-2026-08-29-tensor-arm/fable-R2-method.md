# Referee report R2 — methodological legitimacy and novelty

**Claim under review:** `reports/2026-08-29-b1d-tensor-sample-free-burgers.md` (+ `TENSOR-NOTES.md`, the 08-29 synthesis, 08-28 presentation notes). Read-only review; nothing was run.

## Verdict: MAJOR REVISION (as a method paper: reject; as a negative/engineering note inside a larger paper: minor)

The experiment is executed carefully and the report is honest about its own limits. The problem is what it is presented as. Stripped of vocabulary, the result is: *a Galerkin ROM whose decoder is linear in a 32-vector and whose PDE nonlinearity is quadratic has a precomputable reduced quadratic operator, and on a family where the FOM's upwind switch never fires, that operator reproduces the full-grid projection.* Both halves are textbook. What is new is small and is not isolated by the experiment design.

---

## 1. Novelty

**Not new.** Precomputing $\Phi^\top(u\odot Du)$ as an $M\times R\times R$ tensor for a quadratic nonlinearity is the standard intrusive construction of POD-Galerkin for Burgers (Kunisch–Volkwein 2001/02), and is the explicit motivating example in Chaturantabut–Sorensen 2010 for *why* DEIM exists (the $O(r^3)$ tensor is the thing DEIM avoids when $r$ is large). The quadratic-bilinear line (Benner–Breiten 2015; Benner–Gugercin–Willcox 2015 survey), OpInf (Peherstorfer–Willcox 2016) and Lift & Learn (Kramer–Willcox 2019) all take exactly this tensor as the object of study; OpInf differs only in *learning* $H$ rather than projecting it, and this project does the *intrusive* (projection) version, which predates OpInf. The "exact linear terms via Laplacian eigenmodes as test functions" is spectral Petrov–Galerkin; the "ladder" (linear → quadratic → degree-$p$ tensor $M\times R^p$) is the standard polynomial-nonlinearity observation, not a contribution.

**The actual delta**, which the report never states: the decoder is $u=G\,h(z)$ with $h$ a nonlinear $8\to32$ map, so the ROM is a **32-dimensional linear-subspace Galerkin ROM restricted to an 8-dimensional learned submanifold in coefficient space**. That is the same structure as quadratic-manifold Galerkin/OpInf (Barnett–Farhat 2022; Geelen–Wright–Willcox 2023), where $h(z)=[z;\,z\otimes z]$ and the reduced operators are precomputed in the enlarged coefficient space; here $h$ is an MLP instead of a Kronecker map. The paper should be positioned there: "learned bank + MLP coefficient map; precomputed reduced operators live in bank-coefficient space; no hyper-reduction needed for polynomial FOMs." Relative to Lee–Carlberg 2020 (nonlinear manifold, needs GNAT), the whole point is that linearity-in-$h$ *buys back* precomputability. That is a legitimate design observation, but it is a design choice, not a result.

**"Sample-free / quadrature-free" is misleading.** The community's term is "exact projection of a polynomial nonlinearity; hyper-reduction not required." Calling it sample-free implies it competes with sampling *as a quadrature*; it does not — it is the closed form sampling approximates. The tensor costs $M R^2$ flops per residual, equivalent to sampling at $m \approx MR^2/(R+M) = 512$ nodes; that it "ties NNLS-32" in wall time is a launch-latency artefact (see §5), and in 2D with the $R=512$ bank recorded in the lab log the tensor is $MR^2 = 128\cdot 512^2 \approx 3.4\times10^7$ flops (268 MB in f64) against $\sim1.6\times10^5$ for $m=256$ sampling — a 200× flop deficit. The synthesis's "64³, 2 MB" 2D estimate assumes a bank the project does not have.

## 2. Is the comparison fair — what is actually demonstrated?

On a positive decoded field the tensor *is* the oracle's arithmetic regrouped, so gate T0 ($8\times10^{-15}$) is a unit test of the implementation, not a finding. The only empirical content is: **undershoots below zero at 60–68 % of LM candidates do not change rollout error beyond $10^{-6}$.** That is a real, if modest, robustness statement.

The NNLS-32 and learned-32 columns add nothing: the oracle already beat NNLS-32 by the identical 1–9 % in the 08-27 ladder (5.01 vs 5.48, 6.17 vs 6.24, …), so "tensor beats NNLS-32" is "oracle beats NNLS-32" restated, and the report itself says these are seed-noise. Delete the claim or give it error bars. "Identical stop-reason histograms" is weaker than it sounds: ~99 % of solves end in **stall** (relative decrease $<10^{-3}$), not tolerance. Residual mismatches of $10^{-4}$ relative are below what a solver that stops at $10^{-3}$ relative progress can see. Identical histograms therefore say "both stalled at the same place", not "same solution to tolerance".

## 3. The sign-upwind caveat — sound argument or luck?

The argument is sound but unbounded. At a point with $u<0$ the two stencils differ by $u\,\Delta x\,\Delta_h u$; with $|u|\le 8\times10^{-3}$ on the tails (where $\Delta_h u$ is also small) the product is small, and the measured mismatch falls $1.2\times10^{-6}\to2.3\times10^{-7}$ over $N=128\to1024$, consistent with $O(\Delta x)$. But this is a property of *this* decoder's undershoot amplitude on *this* smooth family, not of the method. A referee will demand:

1. An a-priori bound $\|q_T-q_{or}\|\le \Delta x\sum_{u_x<0}|\Phi_x||u_x||\Delta_h u_x|$ evaluated along the rollout and compared to the measured mismatch (cheap, from the audit JSONs).
2. A numerical assertion that the *truth* is non-negative on every snapshot (`min U ≥ 0` over train and test) — it is asserted by max-principle argument only; implicit first-order upwind is monotone so it should hold, but report the number.
3. A sign-changing family (e.g. $a\sim U(-1.5,1.5)$ or $u_0=\sin 2\pi x$): the tensor must *fail* there, and the paper needs that failure on the table to make the "positivity cone" scope credible rather than convenient.
4. An ablation with a worse decoder (fewer training steps, larger undershoot) to show where the argument breaks.

Note also the physics framing: the FOM is first-order, non-conservative $u\,D^-u$ with ~60 % artificial viscosity at $N=128$ (synthesis). A JCP referee will ask why one builds an exact reduced operator for a discretisation that is itself the crudest available, and will point out the obvious fix — a centred/skew-symmetric FOM makes the residual polynomial for *any* sign, which the synthesis knows and defers.

## 4. Is the decoder floor making the question moot?

Largely yes, and the project has already said so: "at $m=M$ the quadrature barely binds in 1D (+2..+10 % over the oracle)". Floor 3.7e-3, oracle 5.0e-3, NNLS-32 5.5e-3. Every rule within 10 % of each other sits on a floor set by the 8-dim head. The tensor is exact at a budget where exactness is worth ≤10 %; where quadrature *does* bind ($m=16$, 4–5× worse) the tensor has no counterpart because its cost is fixed at the $m\approx512$ equivalent. So the experiment cannot show the tensor mattering; only a decoder with floor $\ll$ quadrature error (or 2D at $m=256$, where the 08-25 ladder found quadrature error in the linear terms — now exact anyway) could.

## 5. Cost claims

"Cost parity" in a regime the project itself describes as launch-bound (≈0.5 ms per LM step for 32×32 matvecs; FOM 8–9 ms/trajectory; ROM 22–29 ms) is content-free: every arm is measuring kernel count. The 10–20 % gain over the oracle is fewer kernels, not fewer flops in $N$. The wider ladder being run can only show that the tensor is $N$-independent — true by construction ($MR^2$ flops, no $N$) — while the oracle grows once $N\gtrsim16$k (from the FOM-cost addendum). That is predictable and does not need a job. What *would* be credible: a flop/byte table per residual evaluation (oracle $\propto N(R+M)$, sampled $m(R+M)$, tensor $MR^2$), a 2D paired job with the real bank size, and a matched-accuracy comparison in a regime where a kernel is not the unit of cost (batched vmap, or CPU).

## 6. Generality claims in the synthesis

- **Polynomial PDEs, degree $p$ → $M R^p$ tensor:** supported, standard, with the storage/flop cost growing as $R^p$ (already $R=512$ in 2D makes $p=2$ marginal, $p=3$ impossible without compression).
- **Head-PCA Tucker compression:** plausible, untested; note the SVD of head outputs is checked nowhere.
- **Lifting $v=|u|$:** *not* a lift in the Kramer–Willcox sense ($|u|$ has no polynomial auxiliary dynamics; $v_t=\mathrm{sign}(u)u_t$ carries the same switch). It is a learned surrogate for a non-smooth term and should be labelled as such.
- **Centred/LF FOM route:** correct and standard (LF dissipation is a Laplacian), but it changes the truth; speculation until run.
- **2D/3D "could tie $m=256$ on cost":** unsupported; see the flop count in §1.

## 7. Statistical rigour

Single seed, one family, 8 test trajectories. The tensor–oracle *algebraic* agreement needs no seeds; the undershoot magnitude (and hence the $10^{-6}$ figure) and every percentage against sampling do. Minimum to publish: ≥3 decoder seeds × ≥32 test trajectories with paired differences and CIs; one sign-changing family (expected failure); the a-priori bound of §3; one 2D cell at the real $R$.

---

## Required changes

1. Reposition: intrusive precomputed quadratic operator in bank-coefficient space for an MLP-restricted linear ROM; cite Kunisch–Volkwein, Chaturantabut–Sorensen, Benner–Breiten, Peherstorfer–Willcox, Kramer–Willcox, Barnett–Farhat, Geelen–Wright–Willcox, Lee–Carlberg. Drop "sample-free" as a method name; keep it at most as a description.
2. Remove or error-bar the "beats NNLS-32 by 1–9 %" claim (it is the oracle's margin).
3. Add the a-priori undershoot bound vs measured mismatch; report `min U_truth`.
4. Add a sign-changing family and show the failure.
5. Replace "cost parity" with a flop/byte table; state explicitly the tensor is the $m\approx512$ flop equivalent.
6. Correct the 2D storage/flop estimate to the actual $R$; either run the 2D cell or drop the 2D cost speculation.
7. Multi-seed, ≥32 trajectories, paired CIs.
8. State that the stop-reason criterion is dominated by stall and what that implies.

## What is genuinely solid

- Implementation correctness: $Q$ built in two orders to $10^{-15}$; algebraic identity on all 8192 states to $10^{-14}$; T0 exact on the positivity cone.
- The sign audit at *every* LM candidate, accepted and rejected — this is the right way to test a "conditionally exact" claim and most papers do not do it.
- The honest decomposition $N_{\text{upwind}} = uD^cu - \tfrac{\Delta x}{2}|u|\Delta_h u$, which correctly locates the only non-polynomial piece.
- Provenance (jobs, commits, checksums, GPU assertions, generated tables) is exemplary.
- The report's own scope statement ("non-negative data only; not bit-exact; no 1D speedup") is accurate; the problem is the headline, not the fine print.
