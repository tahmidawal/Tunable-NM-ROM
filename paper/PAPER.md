# Tunable Nonlinear-Manifold ROMs from One Trained Decoder

*Anonymous submission to ICLR 2027. Every number below is generated from run records by `gen_tables.py`; tables are inlined from `tables-md/` behind an HTML comment naming their id; **[PENDING: …]** marks a lane that has not landed.*

## Abstract

A learned PDE surrogate is fixed once trained: what it can represent cannot be
changed at inference. We show a nonlinear-manifold reduced-order model whose
reachable set is chosen at run time, turning one trained decoder into a family
of operating points. The decoder
writes the solution as a fixed spatial bank times a small neural head. At inference
the solver may add $q$ extra bank directions on top of the head's output: $q$ sets
accuracy, while offline-certified quadrature rules and the solver tolerance set
cost. Three ingredients make this work: a bank that evaluates per node, so cached
cost is flat in mesh size; a linear skip in the head, so the solve starts reliably;
and the split between bank, head and corrections, which lets the rank dial between
the head's compression and the bank's floor. On 2D Burgers the family is monotone
and meets a bar fixed before any run, with the test count held constant so the rank
alone is the control, and it beats the classical linear reduced model at matched
latent dimension. It does not win outright: a well-tuned full-order solver, and
neural operators trained on the same data, are both cheaper and more accurate
here, and we report that. On Poisson and heat the family collapses to a linear
model, which says when the nonlinear manifold is worth having. We also show that
quadrature rules must be validated on states the solver actually reaches, not on
their fitting residual.

## 1 Introduction

<!-- section sources: b-qxm analysis.json; b-eqtop summary.json; mesh-ladder json; b-panel summary.json -->

A projection-based reduced-order model (ROM) solves the discretised equations in
a few coordinates and so inherits the solver's controls; a linear ROM exposes an
obvious accuracy–cost family through its basis rank
(Benner et al., 2015). A *nonlinear-manifold* ROM
(Lee & Carlberg, 2020; Kim et al., 2022) trades that rank for a learned decoder
whose few coordinates reach further than the same number of linear modes, and
in doing so appears to give the family up: the decoder is fixed once trained,
and the solver's remaining controls (tolerance, iteration budget) move cost far
more than they move accuracy. This paper recovers an accuracy control for a
nonlinear-manifold ROM without retraining, gives it a structural meaning, and is
precise about what it is worth.

The decoder writes the solution as a fixed spatial *bank* $G$ of $R$
functions times a small neural *head* $h_\theta$ of $k$ coordinates, and at
inference the solver may add $q$ fixed linear directions,
$u=G(h_\theta(z)+C_q y)$. The rank $q$ moves the reachable set from
the head's image ($q=0$) to the bank's whole span ($q=R$), where the model is a
linear ROM with no nonlinear iteration: a control with a structural meaning.
The paper's three parts are the control, the architecture that admits it, and
the condition under which it is a trade; the losses are reported as losses.

**Contributions.**

1. **A run-time accuracy control from one trained decoder, and a
matched-dimension result.** With the test count $M$ held fixed so that $q$ is
the only control, the Burgers $256^2$ ladder is monotone, every rung
converges, and its rungs span $2.44\times$ in error for
$5.16\times$ in cost inside one allocation (§5.2),
meeting a bar fixed before any run. Certified quadrature rules cut the cost
about $5\times$ at four-decimal-identical error, and the tolerance is a
further cost lever (§5.1). At matched latent dimension
$k=16$ inside one bank, the neural head reaches $2.5629 %$
where the best linear map reaches $56.9296 %$ and POD-16
$61.6503 %$ (§5.4). The family does not beat
a tuned full-order solver or a well-trained neural operator at $256^2$
(§5.1, §5.7).
2. **The architectural choices that admit the control.** A bank that
evaluates per node, so the cached reduced cost is flat while the mesh grows
$264\times$ (§5.5, measured); a
linear skip in the head, chosen so that the decoder Jacobian keeps a
latent-independent component for the cold start (§3.1,
a design choice we do not ablate); and the bank/head/correction split with
exact elimination of the corrections where the residual is linear
(§3.2, measured).
3. **Where the control is not worth having.** Where the weak residual
is nonlinear in the coefficients (Burgers) the corrections cost a growing
nonlinear solve and the ladder is a trade. Where it is linear (Poisson, heat,
waves) the corrections are eliminated exactly, the top rung is the linear
model and also the cheapest point, and the collapse is confounded with a weak
bank: on those cells the learned bank is three to four times worse than a POD
basis of the same rank, while on Burgers the two are comparable
(§5.6). We characterise where the nonlinear manifold is not
worth having; no cell in this paper shows it non-dominated at matched cost.
4. **Two reporting practices.** Every result is reported as bank floor,
best-found and solved error, so each lever's effect lands on the layer it acts
on (§3.5); and quadrature rules are certified on states
the solver actually reaches, never by their fitting residual, which we show
does not predict the held-out error (§5.3).

**What we do not claim.** No speedup over an efficient full-order
solver in 2D (none of the $27$ reduced subjects in the
same-allocation panel is non-dominated); no accuracy superiority over neural
operators (a U-Net and a Transolver beat the ROM); no frontier over POD at
every rank (POD-512 beats the best dense rung on the evolved metric); no cost
ratio across jobs; no convergence theory; and every number is single-seed,
development-cohort evidence (§6).

## 2 Related work and what is new here

<!-- section sources: none (prose only) -->

**Linear and nonlinear manifold ROMs.**
Projection on a linear subspace (Sirovich, 1987; Benner et al., 2015) is limited on advection-dominated problems by the
slow decay of the Kolmogorov width (Cohen & DeVore, 2015; Ohlberger & Rave, 2016).
Nonlinear manifold ROMs replace the subspace with the image of a learned
decoder: Lee & Carlberg (2020) introduced convolutional-autoencoder manifolds
with least-squares Petrov–Galerkin projection (Carlberg et al., 2011), and
Kim et al. (2022) made the decoder evaluable on a sample mesh, which is
what makes hyper-reduction possible on a nonlinear manifold. Our trial manifold
is of this family; the decoder is separable (a coordinate network supplying a
spatial bank, composed with a nonlinear coefficient map), and it is that
factorisation, not the projection, that we rely on. The projection is
least-squares Petrov–Galerkin with a fixed test space and we claim no novelty
for it (§3.3). Quadratic manifolds
(Geelen et al., 2022; Barnett & Farhat, 2022; Weder et al., 2024)
and tensorial POD ( Stef anescu & Sandu, 2014) precompute polynomial reduced
operators; our linear terms are preassembled the same way and that construction
is not new.

**Hybrid linear–nonlinear reduction.**
Adding a nonlinear correction to a linear basis is a family of its own.
Quadratic manifolds (Geelen et al., 2022; Barnett & Farhat, 2022) are a
linear part plus a fixed quadratic map of the same coordinates;
NN-augmented projection ROMs (Barnett et al., 2023) keep a small linear
basis and let a network close the gap to a larger one; POD-DL-ROM
(Fresca & Manzoni, 2022) composes a POD trunk with a learned head;
adaptive and hierarchical bases enlarge a linear basis online by low-rank
updates (Peherstorfer & Willcox, 2015) or by splitting basis vectors
(Carlberg, 2015). Our model is the reverse composition: the
nonlinear head sits inside the linear bank, and the linear part is the
*correction*. What is new is narrow and we state it precisely: the
correction directions are a nested prefix chosen offline once, so a rank is
selected at run time without any refit; where the residual is linear the
corrections are eliminated through a projector that does not depend on the
latent code, so the nonlinear iteration stays $k$-dimensional at every $q$ and
the $q=R$ rung is a linear solve with no iteration at all; and the ladder is
measured against POD, the free bank and full-order controls in one allocation.
Not new: residual PCA, augmenting a manifold with linear directions, variable
projection (Golub & Pereyra, 1973).

**Neural operators.**
FNO (Li et al., 2021), DeepONet (Lu et al., 2021) and the operator-learning
literature (Kovachki et al., 2023), U-Net backbones
(Ronneberger et al., 2015) as the standard strong baseline of PDEBench
(Takamoto et al., 2022), and Transolver (Wu et al., 2024) learn a
map between function spaces and evaluate it in one pass; they are non-intrusive
and apply where no solver exists. We do not treat the two as substitutes: we
hold the data and split fixed across an FNO, a U-Net and a Transolver at an
equal wall budget and report the result whichever way it falls. It falls
against us (§5.7).

**Hyper-reduction and its certification.**
We use empirical quadrature in the non-negative least-squares form of
Hern'andez et al. (2017), with the LP variant (Yano & Patera, 2019),
energy-based mesh sampling (Farhat et al., 2014; Chapman et al., 2017),
GNAT (Carlberg et al., 2013) and DEIM (Chaturantabut & Sorensen, 2010) as
the nearby alternatives, unchanged. What we add is a rule about how a fitted
rule may be accepted: not by its fit residual on the snapshots it was built
from, but by its error on states the solver actually visits, with the fit built
on such states; at the top rungs ($q\ge128$) the number of fit states binds
before the node count does (§5.3).

**Classical solvers.**
On separable constant-coefficient cells the discrete sine transform is a direct
solve (Swarztrauber, 1977) faster than every reduced model we measure, and
preconditioned Krylov methods (Saad, 2003) are the iterative comparator;
we name the direct solver wherever it applies and report it in every job. We
claim no new projection or hyper-reduction algorithm.

## 3 Method

<!-- section sources: b-eqtop summary.json (bars) -->

We reduce a discretised PDE by restricting its state to the image of a frozen
learned decoder and solving, in a small number of coordinates, a least-squares
problem built from a fixed set of smooth weak tests. The decoder has three parts
whose separation is the point of the paper: a spatial *bank*, a neural
*head*, and a set of linear *correction* directions that the solver may
switch on at run time. §3.1 defines the decoder,
§3.2 the correction ladder, §3.3 the
solve that is actually run, §3.4 the offline operators and
the quadrature rules with their certification, §3.5 what is
fixed and what is chosen at run time, and §3.5 the error
decomposition every result is reported in. Per-PDE derivations, solver constants
and exit codes are in Appendix A.
Figure 2 (appendix) shows the data flow and, by colour, when
each quantity is fixed.

### 3.1 Setting and trial manifold

<!-- section sources: b-eqtop summary.json (bars) -->

All problems are posed on $[0,1]^2$ with homogeneous Dirichlet data and
discretised by second-order finite differences on a uniform grid of $N$
intervals per axis; the unknown is the vector of interior nodal values
$u\in\mathbb{R}^{n}$, $n=(N-1)^2$, and $A$ is the negative discrete Laplacian.
Four sizes must be kept apart: $R$ is the width of the spatial bank, $k$ the
latent dimension of the head, $M$ the number of retained weak tests, and $m$ the
number of quadrature nodes where sampling is used. The trial manifold is

$$
u(z) \;=\; G\,h_\theta(z),
  \qquad
  G\in\mathbb{R}^{n\times R},\quad
  h_\theta:\mathbb{R}^{k}\to\mathbb{R}^{R},\quad k\le R\ll n,
$$

<!-- equation (1) -->

with the bank built once per mesh from a random-Fourier-feature coordinate
network $g_\phi$ (Tancik et al., 2020) and a smooth vanishing
factor that enforces the boundary condition exactly at every resolution,

$$
G_{x,:} = \mu(x)\,g_\phi(x)^{\top},
  \qquad
  \mu(x) = 16\,x_1(1-x_1)\,x_2(1-x_2).
$$

<!-- equation (2) -->

The head is a two-layer SiLU MLP plus a linear skip,
$h_\theta(z)=\varphi_\theta(z)+W^{\top}z$, so its Jacobian
$Dh_\theta=D\varphi_\theta+W^{\top}$ retains a latent-independent component; the rank
of $J_{u}=G Dh_\theta$ is at most $\min\{k,\operatorname{rank}G\}$ and the
skip makes equality generic without proving it; the elliptic solver checks the
numerical rank of the projected Jacobian at every query and counts a failed
check as an invalid solve (the “valid” column of Table 10).
The skip is a design choice, adopted so that the cold solve has a
latent-independent descent direction; we do not ablate it in this paper. There is no
encoder in the deployed path: latent codes come from an auto-decoder fit at
training time and from a least-squares fit against the supplied input at query
time (§3.3).

Two consequences of (1) organise everything that follows. The
manifold lies inside the $R$-dimensional span of the bank, so the bank's linear
projection error is a floor on the achievable field error whatever the head or
the solver does; nonlinearity in $h_\theta$ buys a smaller *solved* dimension,
not an escape from the span. And all $x$-dependence factors through $G$, so
the decoder restricted to any node set $S$ is the cached matrix $G_{S,:}$
times $h_\theta(z)$: per-node evaluation costs $O(|S|R)$ whatever $n$ is,
which is what makes both node sampling and exact operator preassembly available
from one decoder.

### 3.2 The correction ladder

<!-- section sources: b-eqtop summary.json (bars) -->

The deployed model augments the head's output with $q$ fixed directions
$C_q\in\mathbb{R}^{R\times q}$ and solves for the latent code and the correction
coefficients together,

$$
u(z,y) \;=\; G\big(h_\theta(z) + C_q\,y\big),
  \qquad z\in\mathbb{R}^{k},\; y\in\mathbb{R}^{q},\; 0\le q\le R .
$$

<!-- equation (3) -->

The columns of $C_q$ are chosen offline once per checkpoint by a principal
component analysis of the head's own residual on the training codes (the part of
the bank-projected training states the head fails to reach), ordered so that
$C_{q}$ is a prefix of $C_{q'}$ for $q<q'$; the ladder is nested and no rung
needs retraining. Three rungs have names. At $q=0$ the model is the frozen
head. At $q=R$ the head is irrelevant—any $h_\theta(z)$ can be cancelled
by $y$—and the model is the linear reduced model on the bank's span, solved
without a nonlinear iteration. In between, $q$ moves the reachable set from the
head's image towards the bank's span, and the solved dimension from $k$ to
$k+q$.

How $y$ is solved determines the cost. When the weak residual is linear in the
coefficients (Poisson, heat) the inner minimisation over $y$ is a linear least
squares with a projector that does not depend on $z$, so $y$ is eliminated
exactly and the nonlinear iteration stays $k$-dimensional at every $q$
(Appendix A.1, variable projection with a constant
projector (Golub & Pereyra, 1973)). When the residual is quadratic in the
coefficients (Burgers) the elimination is not closed-form; we damp the
$(z,y)$ blocks separately inside one Levenberg–Marquardt step
(*block-damped* variable projection): in the same job a joint LM on
$(z,y)$ at $q=64$ leaves $6$ budget exits and plain
variable projection is an order of magnitude slower, while the block-damped
step exits stationary everywhere at the same error
(Table 7). The test count $M$ must exceed $k+q$; the headline ladder holds $M$
fixed across rungs so that $q$ is the only control (§5.2).

### 3.3 Weak residual, objective, and solver

<!-- section sources: b-eqtop summary.json (bars) -->

Let $P\in\mathbb{R}^{M\times n}$ collect the $M$ lowest tensor-product sine
vectors of the square, orthonormal in the discrete inner product; they are the
eigenvectors of the five-point operator, $PA=\LambdaP$, and this
identity is asserted numerically in every job. With $r(\cdot)$ the fully
discrete residual of the PDE at hand, the reduced problem solved online is

$$
(z^{\star},y^{\star})
  =
  \operatorname*{arg\,min}_{z,\,y}\;
  \tfrac12\,\lVert r_{w}(z,y) \rVert_2^2,
  \qquad
  r_{w}(z,y) = \Lambda_\star^{-1}\,P\,r\!\big(u(z,y)\big)
  \in\mathbb{R}^{M},
  \qquad M>k+q ,
$$

<!-- equation (4) -->

with a diagonal row scaling $\Lambda_\star$ stated per PDE in
Appendix A; it changes which solution is selected and is part of
the method. This is a least-squares Petrov–Galerkin condition with an
explicit, latent-independent test space (Carlberg et al., 2011; Lee & Carlberg, 2020),
not tangent Galerkin, and overdetermined rather than square.
Each attempt solves the damped normal system
$(H+\lambda \operatorname{diag}(\operatorname{diag} H)) \delta=-g$ with $H=J_{}\TJ_{}$, $g=J_{}^{\top}r_{w}$
from forward-mode differentiation through $h_\theta$ (Bradbury et al., 2018), and
accepts the step only if it is finite, within a fixed trust radius, and strictly
decreases the residual: damped Levenberg–Marquardt with a monotone test
(Marquardt, 1963), no line search.
Three exit families are recorded and never conflated: the scale-free
stationarity measure

$$
\eta(z,y)
  =
  \frac{\lVert J_{}^{\top}r_{w} \rVert_2}{\lVert J_{} \rVert_{F}\;\lVert r_{w} \rVert_2}
  \le\eta_{\mathrm{tol}},
$$

<!-- equation (5) -->

a residual threshold, and the stalls (budget exhausted, vanishing step, damping
limit, non-finite value); outputs produced under a stall are reported as
early-stopped, never as converged. Stationarity certifies a first-order
critical point of (4) on the frozen manifold and nothing
more: not a global minimum, nothing about the $n-M$ untested residual
components, no bound on the field error. For an arm whose reduced fit is
*attainable* (a POD basis fitted to its own initial condition, the free
bank) the residual falls to round-off and (5) is $0/0$, so
a separate completion status accepts an attained fit and the stricter flag is
printed beside it. No query uses the solution it predicts: the elliptic solve
starts from the cached training code nearest the projected source; the Burgers
query fits $(z,y)$ to the supplied initial field on a fixed
$48\times48$ Gauss–Legendre rule independent of the query mesh, and each time
step is warm-started at the previous code. The interpolation of the input is
charged to the query.

### 3.4 Precomputed operators, quadrature, and certification

<!-- section sources: b-eqtop summary.json (bars) -->

For a fixed linear operator the tested residual is exactly precomputable:
$P(A u-f)=B (h_\theta(z)+C_q y)-b$ with
$B=PAG=\Lambda PG\in\mathbb{R}^{M\times R}$ assembled offline
by two one-dimensional sine transforms, with no sampling anywhere. Linear terms
are therefore never approximated, on any PDE here. For Burgers the advection
$P N(G h)$ with the sign-upwind stencil $N$ is the only term that
resists preassembly (a fixed tensor $T_{iab}$ exists only where the upwind
switch is inactive, Appendix A.3); it is evaluated either
*densely* on every interior node, which costs $O(nR)$ per residual, or on
an *empirical quadrature* rule of $m$ nodes with non-negative weights
(Hern'andez et al., 2017; Yano & Patera, 2019), which costs $O(mR)$ through the
cached $5m\times R$ stencil block.

A rule is fitted offline by non-negative least squares
(Lawson & Hanson, 1974) on decoder outputs at $n_{\rm fit}$ stored codes,
with a hard cap of $m$ nodes (Appendix A.4); changing $m$
re-solves the fit, rules are not nested, and deployment selects among stored
rules. *A rule is never certified by its NNLS fit residual*; we certify it
by the held-out relative error of the projected advection term,

$$
\rho(u)=\frac{\lVert \sum_{i\in\mathcal S}w_iP_{:,i}N_i(u)-P N(u) \rVert_2}{\lVert P N(u) \rVert_2},
  \qquad
  \rho_{\max}\le0.116\ \text{(primary bar)},\quad \rho_{\max}\le0.06\ \text{(tight)},
$$

<!-- equation (6) -->

over states the solver actually reaches on trajectories disjoint from the fit
and evaluation cases. The primary bar is the held-out $\rho$ of the incumbent $q{=}0$ rule at the state carrying the first-interval penalty, measured in the q-diag cell and adopted as the primary bar in the q-ridge design before any certification job ran; the tight bar was declared before b-eqtop ran. Second, the fit states must be reachable states too, and at the top rungs
($q\ge128$) their *number* binds before the node count does
(§5.3).

### 3.5 What is fixed, what is chosen, and how error is reported

<!-- section sources: b-eqtop summary.json (bars) -->

Every operating point uses one frozen artefact per PDE: the weights $\theta$,
the bank $G$ and $R$, the latent dimension $k$, the directions $C_q$, the
time discretisation and the query contract are identical across the family.
Selected at run time are (i) the correction rank $q$, a prefix of the stored
directions; (ii) the quadrature, dense or one of the stored certified rules;
(iii) the stopping tolerance and attempt budget. The first two select among
offline-prepared artefacts; only the third is a pure run-time number. The test
count $M$ is a design parameter of the weak problem and is held fixed along the
headline ladder; where earlier cells raised it with $q$ we say so and report
the effect separately (§5.2). Nothing is re-optimised at query
time, and no operating point is reported without its exit reasons.

Every panel reports three numbers, never one. The *bank floor* is the
relative error of the orthogonal projection of the truth onto
$\operatorname{range}G$, what no head and no solver can beat. The
*best-found* error is the smallest error any point of the augmented
manifold $\{G(h_\theta(z)+C_q y)\}$ attains for that case, found by a
multistart oracle and therefore an upper bound on the true minimum. The
*solved* error is what the deployed iteration returns. Their differences
assign the deployed error to representation, reduction and solver, and a lever
can only move the layer it acts on: $q$ moves the reduction layer, tolerance
and budget the solver layer, nothing at inference the floor.

The construction provides no accuracy certificate, no escape from the bank, and
no convergence theory; $G$, $P$, $B$ and the rules are rebuilt at each
resolution even though $\theta$ transfers, and that setup is charged
separately.

## 4 Experimental setup

<!-- section sources: b-panel summary.json; tuning02 output-index.json -->

**Problems.** Table 4 (appendix) specifies each PDE,
domain, discretisation, mesh, reduced sizes, reference and cohort. Burgers 2D
at $256^2$ with one frozen checkpoint is the hero; Poisson, heat and the
reflective wave are the linear cases; the L-shaped Poisson domain is the cell
where no fast transform exists.
**Metrics.** Burgers errors are relative $L^2$ trajectory errors against
the same job's converged full-order solve on the same grid (removing the
mesh's own discretisation error, $4.0265 %$ at $256^2$
against a $4096^2$ reference), maximised over cases and output times. Two
maxima are reported everywhere and neither is chosen: worst over *all*
times, which includes $t=0$ and is bounded below by the decoder's compression
of the supplied field, and worst over the *evolved* times $t>0$, with the
$t=0$ compression as its own column. Poisson and wave errors are against the
exact discrete solution.
**Timing.** Every cost is the median over retained repetitions of a
completed device computation, timed in one Slurm allocation on one GPU after a
burn-in, with device synchronisation around every region, arms interleaved in a
recorded random order, float64 and highest matmul precision. Device query
(input and outputs resident) and complete host-to-host query are both reported.
*No ratio is ever formed across jobs or GPUs.* Offline setup is charged
once per mesh and reported separately.
**Comparators** are always in the same job: Newton with a modal Helmholtz
preconditioner at several tolerances (Burgers); the direct sine transform and
tuned conjugate gradients (Poisson, heat, waves); sparse direct and IC(0)-PCG
(L-shape); POD-LSPG through the same weak objective, tests, solver and stopping
rule; neural operators on the identical dataset and split with recorded index
hashes, an equal wall budget and the same selection rule.
**Provenance.** Every table names its job id, GPU and checkpoint here or
in Table 3; every solve carries its exit reason; every error
is recomputed from saved fields by an independent NumPy audit; every number is
emitted by one generator from those records.

## 5 Results

<!-- section sources: b-panel summary.json; b-qxm analysis.json; b-eqtop summary.json + report; head-ablation abl01/pabl01-audit.json; b-head-train eval01-audit.json; mesh-ladder json; b-speed report; p-linear summary.json + verdicts.json; heat linear-bank report (2026-09-10); w-ladder summary.json; lshape summary.json; no-second summary.json -->

Every number below is a single training seed on opened development cases, one
checkpoint per PDE; costs are compared only inside one allocation, and where two
numbers come from different jobs the text compares accuracy only. The losses
come first.

### 5.1 Nothing reduced is on the frontier at $256^2$

<!-- section sources: b-panel summary.json; b-qxm analysis.json; b-eqtop summary.json + report; head-ablation abl01/pabl01-audit.json; b-head-train eval01-audit.json; mesh-ladder json; b-speed report; p-linear summary.json + verdicts.json; heat linear-bank report (2026-09-10); w-ladder summary.json; lshape summary.json; no-second summary.json -->

Table 1 and Figure 1 come from one
allocation on one A100 with $36$ timed subjects: the
correction ladder with dense and certified-quadrature advection at two
tolerances, POD-LSPG at six ranks, the unrestricted bank, the trained FNO, and
eight full-order Newton settings, all against the same converged reference.
On (GPU ms, worst *evolved* error) **$0$ of the
$27$ reduced subjects are non-dominated** once the full-order
controls are included, and likewise on the *all-times* metric
($0$ of $27$). The most accurate reduced
subject is POD-512 at $0.2184 %$ and
$2928.9$ ms; Newton at tolerance $10^{-3}$, $\Delta t=0.005$,
reaches $0.0489 %$ at $31.4$ ms, and
the FNO, timed in the same allocation, $7.4164 %$ at
$7.2$ ms. This is the evidence for the abstract's sentence that the
model does not beat an efficient full-order solver in 2D.

Within the reduced subjects: the dense ladder $q=0,\dots,256$ is monotone on
both metrics with every rung converged, spanning $3.64\times$ in
evolved error for $14.62\times$ in cost, but it raises
$M=4(K+q)$ with $q$ (§5.2 separates the two). Certified
quadrature is a cost lever at equal error: at $q=0$ the rule costs
$57.9$ ms against $284.8$ ms dense
($4.92\times$) for $1.8891$ against
$1.8890 %$, and at $q=128$ $247.0$ against
$1223.2$ ms ($4.95\times$) for
$0.8925$ against $0.8930 %$: about
$5\times$ cheaper at four-decimal-identical error. Loosening the tolerance from
$10^{-6}$ to $10^{-3}$ removes a further
$26$–$23 %$ at negligible error
change. The EQ ladder of this panel is *not* monotone at the top ($q=256$:
$1.0361 %$ against $0.8925 %$
at $q=128$) because those rungs carried rules certified only on the secondary
bar; §5.3 shows the fix (cross-job spread of an identical cell:
$14 %$, §6).

The classical baseline: on the *evolved* metric POD-512 beats the best dense
rung on both axes ($0.2184 %$ at $2928.9$ ms
against $0.5194 %$ at $4164.6$ ms); on the *all-times*\
metric it is dominated by the unrestricted bank and the reduced-only frontier is
$7$ certified-EQ rungs plus that endpoint, with no
POD rank on it. Both are printed because the metric decides. The POD ranks
$k'\ge32$ carry a “strict” flag of no only because their initial fit is
attainable at round-off ($0/0$ stationarity); each sits on its projection floor
(solved/best-found $\le1.00577$), under a completion rule
pre-registered for this panel and post-dating the head-ablation job.

**Table 1.** The correction ladder at $256^2$ in one allocation (job
3780638, NVIDIA A100 80GB PCIe, checkpoint incumbent (gate checkpoint_unchanged; hash in T2, tuning row)): dense advection and
the certified rule at two tolerances, worst error over evolved and over all
times (%), median GPU ms, and the $t{=}0$ compression bounding the all-times
metric. The $q=128$ and $256$ rules here certify on the secondary bar only;
primary rules are in Table 2. Every subject of the job is in
Table 13.

<!-- table: T03_tunability -->
| $q$ | $M$ | dense: evolved % | all % | ms | EQ, tol $10^{-6}$: evolved % | all % | ms | EQ, tol $10^{-3}$: evolved % | all % | ms | $t{=}0$ % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 64 | 1.8890 | 2.5629 | 284.8 | 1.8891 | 2.5629 | 57.9 | 1.8898 | 2.5628 | 42.6 | 2.5629 |
| 16 | 128 | 1.3985 | 2.4806 | 363.0 | 1.4270 | 2.4806 | 79.7 | 1.4272 | 2.4806 | 60.4 | 2.4806 |
| 32 | 192 | 1.2336 | 2.3534 | 442.6 | 1.2493 | 2.3534 | 95.7 | 1.2497 | 2.3534 | 72.3 | 2.3534 |
| 64 | 320 | 1.0843 | 2.1489 | 637.3 | 1.2275 | 2.1489 | 119.1 | 1.2278 | 2.1489 | 88.5 | 2.1489 |
| 128 | 576 | 0.8930 | 1.8116 | 1223.2 | 0.8925 | 1.8116 | 247.0 | 0.8926 | 1.8116 | 189.0 | 1.8116 |
| 256 | 1088 | 0.5194 | 0.9053 | 4164.6 | 1.0361 | 1.0361 | 700.4 | 1.0324 | 1.0324 | 429.4 | 0.9053 |

![Figure 1](figures/fig_tunability_family.png)

**Figure 1.** The family. Left: worst evolved error against the correction rank for
the two fixed-$M$ ladders and the scheduled ladder (b-qxm; errors comparable
across its three jobs, costs not, so no cost axis). Centre: the $256^2$
same-allocation panel (job 3780638), correction rungs against POD ranks,
the free bank, the FNO and the full-order controls, filled markers converged,
the non-dominated set as a step line. Right: the primary-rule EQ ladder against
its dense twins (job 3780164, provisional, replication
3783811 pending). No cost is compared across panels.

### 5.2 Rank against test count: the family at fixed $M$

<!-- section sources: b-panel summary.json; b-qxm analysis.json; b-eqtop summary.json + report; head-ablation abl01/pabl01-audit.json; b-head-train eval01-audit.json; mesh-ladder json; b-speed report; p-linear summary.json + verdicts.json; heat linear-bank report (2026-09-10); w-ladder summary.json; lshape summary.json; no-second summary.json -->

The crossed grid (Table 8) runs $q\in\{0,\dots,256\}$ against fixed
$M\in\{256,1088\}$, the schedules $M\in\{4,8,16\}(K+q)$ and bridge cells, a
saturation sweep in $M$ and a solver control, in three jobs whose shared cells
agree to $10^{-9}$. Two fixed-$M$ ladders were run and both are reported. At
$M=256$ the ladder $q=0,\dots,128$ inside one job spans
$1.22\times$ in error for $2.27\times$
in cost and **fails** the pre-registered bar (monotone, at least three
non-dominated points, at least $2\times$ on both axes, nothing early-stopped).
**At $M=1088$ the ladder $q=0,64,128,256$, inside one job, is
monotone, every rung converged, and spans $2.44\times$ in evolved error
for $5.16\times$ in cost with $4$ non-dominated points**, and
passes. The choice of $M=1088$ as the headline was made after the crossed grid
was run, by the lane's decision rule (the largest fixed $M$ that holds every
rung to $q=256$), not pre-registered; at $M=256$ the $q=256$ rung is
under-determined ($M<K+q$) and cannot exist. The scheduled ladder spans
$3.64\times$; its log-span is $69.0 %$ rank
and $31.0 %$ test count along the corner path, and on the
balanced sub-grid the rank main effect carries $85.1 %$ of the
variance in log error against $6.1 %$. The test count does
$84, 77, 40, 10, 5 %$ of the work at successive rungs: it buys the bottom
of the ladder and the rank buys the top, and the absolute $M$ needed grows with
$q$ (at $q=256$ the $M$ effect is still $2.03\times$). The
solver switch at $K+q>64$ is inert (relative difference
0.00e+00{}).

### 5.3 Quadrature rules must be certified on reachable states

<!-- section sources: b-panel summary.json; b-qxm analysis.json; b-eqtop summary.json + report; head-ablation abl01/pabl01-audit.json; b-head-train eval01-audit.json; mesh-ladder json; b-speed report; p-linear summary.json + verdicts.json; heat linear-bank report (2026-09-10); w-ladder summary.json; lshape summary.json; no-second summary.json -->

A rule's NNLS fit residual is a property of its fitting set, not a certificate
(Figure 3): the static-snapshot rule at $q=256$
fits to 1.5e-04{} and reaches $\rho_{\max}=0.462$
on held-out reachable states, four times the primary bar. Certified by (6) on states the dense solver visits, the top-rung
blocker turns out to be the number of fit states, not the node count: rules
fitted on 14 and 8 states at $q=128$ and $256$ reach
$\rho_{\max}=0.1908$ and $0.1678$
at $m=2048$ and fail, while 64-state rules at the same $m$ reach
$0.0669$ and $0.1074$ and pass
(Table 19); at $q=64$ a 25-state rule passes where a 64-state
re-draw does not, so this is a statement about $q\ge128$. Rebuilt with the cheapest primary-certified rule per rung
and timed in one allocation, the EQ ladder is monotone on the evolved metric
(yes{}), every rung converged, running
$1.8891\to0.5389 %$ at
$59.1\to722.2$ ms, $0.18$–$0.21\times$
the same-job dense cost; at $q=256$ the 8-state rule gave
$1.0361 %$ where the 64-state rule gives $0.5389 %$
against dense $0.5194 %$ (Table 2). The rules are not free: the primary ladder's rules took
$1.7$ h of NNLS in total (Table 6). **All
of this is one draw**: a second draw at $q=64$ moved $\rho_{\max}$ from
$0.0531$ to $0.1220$ and flipped the
flag, so the draw is first-order; job $3783811$ measures that spread,
and until then the ladder is “certified in this draw”.

**Table 2.** **Provisional.** The EQ ladder with the cheapest primary-certified
rule per rung, timed in one allocation (job 3780164, A100 80 GB),
with its same-job dense twins. Each certified flag is one draw; replication
job $3783811$ is pending.

<!-- table: T09_eq_ladder -->
| $q$ | $m$ | $\rho_{\max}$ | primary | tight | EQ evolved % | EQ all % | EQ ms | dense evolved % | dense ms | EQ/dense cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1024 | 0.0153 | yes | yes | 1.8891 | 2.5629 | 59.1 | 1.8890 | 292.9 | 0.202 |
| 16 | 1024 | 0.0935 | yes | no | 1.4270 | 2.4806 | 80.6 | — | — | — |
| 32 | 1024 | 0.0533 | yes | yes | 1.2493 | 2.3534 | 97.7 | — | — | — |
| 64 | 1024 | 0.0531 | yes | yes | 1.2275 | 2.1489 | 120.6 | 1.0843 | 624.9 | 0.193 |
| 128 | 2048 | 0.0669 | yes | no | 0.8936 | 1.8116 | 246.9 | 0.8930 | 1190.5 | 0.207 |
| 256 | 2048 | 0.1074 | yes | no | 0.5389 | 0.9053 | 722.2 | 0.5194 | 3915.1 | 0.184 |

### 5.4 The head earns its place per dimension, not per millisecond

<!-- section sources: b-panel summary.json; b-qxm analysis.json; b-eqtop summary.json + report; head-ablation abl01/pabl01-audit.json; b-head-train eval01-audit.json; mesh-ladder json; b-speed report; p-linear summary.json + verdicts.json; heat linear-bank report (2026-09-10); w-ladder summary.json; lshape summary.json; no-second summary.json -->

At matched $k=16$ in one frozen bank per PDE, the head is compared with the
optimal rank-$k$ affine map inside the same bank, a quadratic map, the
unrestricted bank, and POD-LSPG at ranks $8$–$128$, through the same weak
objective, tests, solver and stopping rule (Table 15,
Table 16). **The matched-dimension result is the claim**: on
Burgers the head reaches $2.5629 %$ worst same-grid error against
$56.9296 %$ for the best linear map, $31.8276 %$ for
the quadratic one and $61.6503 %$ for POD-16. Larger POD ranks
are a different question and the answer depends on the metric: on the
all-times metric no POD rank up to $k'=128$ matches the head (POD-128
$10.1198 %$), while on the evolved metric in the panel
job POD-128 reaches $1.9464 %$ against the $q=0$ rung's
$1.8890 %$ and POD-256 $0.7109 %$. On
Poisson at $1024^2$ the head wins per dimension ($6.0927 %$
against $17.2966 %$ linear, $20.0548 %$ POD-16)
and loses per millisecond (POD-128: $4.3452 %$ at
$5.930$ ms against the head's
$5.843$ ms), because that query is dominated by projection and
decode. In three layers (Table 9): on Burgers the floor is
$0.3918 %$, best-found $2.5447 %$, solved
$2.5629 %$; the solver layer costs $0.018$
percentage points and the reduction layer sits
$6.5\times$ above the floor. The head is the binding
layer, and training alone does not close it: no retrained head on the same bank
beats the incumbent's best-found $2.5447 %$
(Table 29, with the caveat that the like-for-like retrain does
not reproduce the incumbent), and on Poisson a wider head moves the
best-found/floor ratio from $4.18$ to $2.76$
and no further (Table 23).

### 5.5 Cached cost is flat in the mesh; nothing crosses over

<!-- section sources: b-panel summary.json; b-qxm analysis.json; b-eqtop summary.json + report; head-ablation abl01/pabl01-audit.json; b-head-train eval01-audit.json; mesh-ladder json; b-speed report; p-linear summary.json + verdicts.json; heat linear-bank report (2026-09-10); w-ladder summary.json; lshape summary.json; no-second summary.json -->

With one frozen checkpoint per PDE transferred across $64$–$1024$ intervals,
the cached reduced solve costs $44.16\to43.58$ ms
on Burgers while the unknowns grow $264\times$
(Table 21); the complete query grows
$1.033\times$ through dense input and output. Against the
cheapest same-job full-order arm meeting the target there is no crossover at
any rung (FOM/ROM 0.265–0.495{} on Burgers,
0.063–0.112{} against the direct transform on Poisson). A kernel-level port at bit-level parity makes the
frozen Burgers query $1.520x$ faster at $256^2$, short of its
$2\times$ target, because the query is kernel-count bound
(Table 28).

### 5.6 Where the control collapses: linear PDEs

<!-- section sources: b-panel summary.json; b-qxm analysis.json; b-eqtop summary.json + report; head-ablation abl01/pabl01-audit.json; b-head-train eval01-audit.json; mesh-ladder json; b-speed report; p-linear summary.json + verdicts.json; heat linear-bank report (2026-09-10); w-ladder summary.json; lshape summary.json; no-second summary.json -->

When the residual is linear in the coefficients the corrections are eliminated
exactly and the ladder has a different shape. Poisson at $1024^2$ ($R=512$,
$K=32$; Table 10): the rungs $q=0,\dots,256$ run
$3.1495\to0.9648 %$ at a flat
$6.4$–$6.9$ ms, and the top
rung $q=R$, the full-bank QR solve, lands on the floor at
$0.7421 %$ and is the *cheapest* point at
$4.42$ ms; the pre-registered degenerate-curve criterion
passes (at $256^2$ only its literal cost-span clause fails, because the top
rung is $3.17\times$ cheaper than the middle rungs).
Solver effort buys nothing at $q=0$: floor $0.7421$,
best-found $3.1139$, solved
$3.1495 %$ ($M=129$; the $M=257$ solve in
Table 9 differs in the fourth digit). The direct transform is
exact at $3.248$ ms. Heat at
$1024^2$ (an earlier cell of the same family, Table 11): the bank
evolved linearly reaches $1.68 %$ at
$0.56$ ms against the head's
$4.56 %$ at $12.3$ ms, with the
direct solve at $1.03$ ms. The reflective wave
(Table 22): the top rung, the bank under exact modal propagation
with no head, is not strictly the most accurate rung (strict reading:
no{}) but matches the $q=32$ rung within the
pre-registered integrator tie band of $0.020$ pp at every
mesh ($256^2$: $5.121$ against
$5.104 %$), and is
$158$–$42\times$
cheaper than the cheapest head rung; the head ladder is monotone in $q$
(yes{}), cost is not.

**The collapse is confounded with a weak bank, and we say so.** On
Poisson POD $k'=512$ reaches $0.1838 %$ against
the learned bank's floor of $0.7421 %$, and on waves POD
$k'=64$ reaches $1.502 %$ against the bank's
$5.121 %$ at the same cost: on both linear cells the
learned bank is three to four times worse than a POD basis of the same rank.
On Burgers, in the panel job, the free bank and POD-512 are comparable as
banks ($0.6027 %$ against $0.6125 %$ on the
all-times metric). The exact elimination is a property of the linear residual; the size of the
collapse is at least as much a property of these banks. No reduced arm beats
the direct solve on any of the three. On the L-shaped domain (no fast transform) the bank floor is
$0.6650 %$ on the development cohort and
$1.6155 %$ on the selection cohort, corner enrichment
buys only $1.067\times$, and every head sits
$2.24$–$5.77\times$ above its floor
(Table 30); its solve layer has not run.

### 5.7 Neural operators on the same data

<!-- section sources: b-panel summary.json; b-qxm analysis.json; b-eqtop summary.json + report; head-ablation abl01/pabl01-audit.json; b-head-train eval01-audit.json; mesh-ladder json; b-speed report; p-linear summary.json + verdicts.json; heat linear-bank report (2026-09-10); w-ladder summary.json; lshape summary.json; no-second summary.json -->

A PDEBench-style U-Net and a Transolver were trained on exactly the FNO's data,
split, reference, metric, wall budget and selection rule
(Table 12). Worst error on the matched eight-case cohort:
U-Net-small $1.4712$, U-Net-medium $1.5189$,
Transolver-refine $1.5224$, U-Net-refine $1.7110$,
**ROM $1.8671$**, FNO-large $2.4829$, efficient full-order
solver $0.9978 %$: $4$ operator arms beat the
ROM and every FNO capacity is worse than it. The earlier finding that the ROM
is more accurate than a neural operator was an artefact of the family chosen
and is withdrawn. On cost, only the FNO was timed against the ROM in one
allocation, and there it dominates the ROM on both axes ($7.4164 %$ at
$7.2$ ms, §5.1); the U-Net and the Transolver beat
the ROM on accuracy and were not timed against it, so no cost statement is made
for them. Every operator number is a lower bound (7 of 8{}
U-Net/Transolver arms and 4 of 4{} FNO arms were still improving at
their budget); on Poisson the gap is larger (Table 27).
What survives is the within-reduced-model claim and the mechanism.

## 6 Limitations

<!-- section sources: no-second summary.json; b-eqtop summary.json; b-qxm analysis.json -->

**Single seed, single checkpoint.** Every number rests on one training
seed and one checkpoint per PDE. The three-seed repetition and the
sealed-cohort evaluation are running; that lane has recorded an amendment to
its fidelity gate, from $10^{-9}$ to $10^{-3}$ relative, made before any of its
numbers were read.
**Development cohort.** Burgers numbers are on six opened cases (32
held-out for the tuning study), wave and Poisson on eight and twelve; on the
L-shape the 461-source validation worst is two to three times the 32-source
development worst.
**2D only, and nothing harder than viscous Burgers is solved**: the
L-shape cell has no solve layer yet and Navier–Stokes no reduced model.
**No cold-start comparison against a shallow masked-autoencoder ROM**; the
classical comparison is POD-LSPG through our own solver, and the linear skip is
a design choice we do not ablate.
**Operator numbers are lower bounds**: 7 of 8{}
U-Net/Transolver arms and 4 of 4{} FNO arms were still improving at
their wall budget, and none was tuned per family; a larger budget beats the
ROM by more.
**Quadrature certification is one draw**: a second draw at $q=64$ moved
$\rho_{\max}$ from $0.0531$ to
$0.1220$ and flipped the flag; until job
$3783811$ lands the EQ ladder is “certified in this draw”.
**The headline ladder was chosen after the fact and quotes a dearer
baseline**: $M=1088$ was selected after the crossed grid ran, the $M=256$
ladder fails the same bar, and the fixed-$M$ $q=0$ rung costs
$848.0$ ms against $290.6$ ms for the scheduled
rung (different jobs). **Cross-job spread** of an identical cell is
$14 %$, the size of the tolerance saving.
**Offline cost is not free**: the primary EQ ladder's rules took
$1.7$ h of NNLS in total (Table 6).
**No theory**: no convergence guarantee, no quadrature-error bound on
unseen states, no continuum error bound.

**Pending cells.** Not in this draft, and no result sentence depends
on them: the $1024^2$ panel (\gen{pending: b-panel 1024$^2$ (job 3783817)}); seeds and the sealed cohort
(\gen{pending: b-seeds}); the L-shape solve layer (\gen{pending: lshape solve jobs 3784662/3/4}); the Navier–Stokes
reduced model (\gen{pending: ns2d phases 2–3}); the operators' seed and precision controls
(\gen{pending: no-second ctrl01 (job 3783831)}). The resolution control has landed and fired its
pre-registered clause (§5.7).

## 7 Conclusion

<!-- section sources: b-qxm analysis.json -->

One trained decoder exposes a monotone accuracy–cost family at run time
through the number of bank directions the solver may add to the head's output:
with the test count fixed, that rank alone spans $2.44\times$ in error
for $5.16\times$ in cost on 2D Burgers inside one allocation, and
certified quadrature moves cost at fixed accuracy. The trade
exists where the residual is nonlinear in the coefficients; where it is linear
the top rung is a linear model and the cheapest point, confounded with a weak
bank. None of this beats a tuned full-order solver or a
well-trained neural operator in 2D; the case rests on the mechanism and its
condition, not on winning.

## Reproducibility statement

<!-- section sources: none (prose only) -->

Every table and every number in the prose is generated by one script
(`paper/gen_tables.py`) from the machine-readable outputs of the runs
named in Appendix C; the script records the SHA256 of every
file it reads. Each run's job id, GPU, source commit and checkpoint hash are in
Table 3. Solver definitions, stopping rules and exit-reason
codes are in §3.3 and Appendix A; timing and
cohort protocol in §4. An anonymised repository with the
generator, the provenance registry, the lane summaries and the audit JSONs, and
the checkpoints by hash, accompanies the submission.

## AI use statement

<!-- section sources: none (prose only) -->

Language-model assistants were used to draft and edit prose, to write the table
generator and the figure scripts, and to audit run records; every number in the
paper comes from the recorded run outputs through that generator and none was
produced by a language model. The authors take full responsibility for the
content.

## References

*Rendered from `refs.bib` at build time; citations above are (author, year).*

---

# Appendices

## A Method details

<!-- section sources: none (prose only) -->

### A.1 Elliptic instance: Poisson

<!-- section sources: none (prose only) -->

For $A u=f$ we take $\Lambda_\star=\Lambda$, so the weak residual is

$$
r_{w}(z,y)
  \;=\;\Lambda^{-1}P(Au-f)
  \;=\; B_0\,\big(h_\theta(z)+C_q y\big) - b_0,
  \qquad
  B_0=PG,\quad b_0=\Lambda^{-1}P f .
$$

<!-- equation (7) -->

Because the retained tests are orthonormal and $A u^{\star}=f$, this residual
is exactly the tested error $P(u-u^{\star})$, so the minimised
objective is the squared discrete $L^2$ error of the reduced state in the
retained modes; this holds only for the modes $P$ retains.

**Analytically eliminated corrections.** Let $B_0C_q=Q\mathcal{R}$ be
a thin QR factorisation. $Q$ does not depend on $z$, so the inner
minimisation over $y$ has the closed form
$\mathcal{R}y=Q^{\top}(b_0-B_0h_\theta(z))$ and the minimised value is

$$
\min_{y}\lVert B_0(h_\theta(z)+C_q y)-b_0 \rVert
  \;=\;
  \lVert B_\perp h_\theta(z)-b_\perp \rVert,
  \qquad
  B_\perp=(I-QQ^{\top})B_0,\quad b_\perp=(I-QQ^{\top})b_0 .
$$

<!-- equation (8) -->

The LM iteration runs on $z$ alone with the deflated operator, and $y$ is
recovered by one triangular solve. Because the projector is constant the
elimination introduces no approximation and $B_\perpDh_\theta$ is the exact
Jacobian of the eliminated residual; the only dropped derivative anywhere in
the elliptic solve is the head curvature. At $q=R$ the deflated operator is
zero, the latent problem is degenerate by construction, and the model is the
linear least-squares solve $\min_y\lVert B_0 y-b_0 \rVert$; the timed top rung is that
QR solve, not an LM iteration on a zero Jacobian. We report the reduced
stationarity of the deflated problem, the augmented stationarity of the full
problem, the backward error of the triangular recovery and the numerical rank
of $B_0C_q$.

### A.2 Parabolic instance: heat

<!-- section sources: none (prose only) -->

The semi-discrete heat equation $\dot u=-\kappaA u$ is advanced with
Crank–Nicolson,
$(I+\tfrac{\Delta t\kappa}{2}A)u^{n+1}=(I-\tfrac{\Delta t\kappa}{2}A)u^{n}$,

and the reduced step substitutes the manifold into the fully discrete equation
*before* projecting. With $u^{n}=u(z_n)$, projecting with
$P$ and row-scaling by $\Lambda_\star=I+\tfrac{\Delta t\kappa}{2}\Lambda$
gives

$$
r_{w,n}(z)
  \;=\;
  B_0\,h_\theta(z) \;-\; D\,B_0\,h_\theta(z_n),
  \qquad
  D=\operatorname{diag}\!\left(\frac{1-\tfrac{\Delta t\kappa}{2}\lambda_i}
                      {1+\tfrac{\Delta t\kappa}{2}\lambda_i}\right),
$$

<!-- equation (9) -->

and $z_{n+1}=\operatorname*{arg min}_{z}\lVert r_{w,n}(z) \rVert_2$, warm-started
at $z_n$. Two points are structural. The right-hand side is built from
the *decoded current state* at every step: carrying the quantity the
previous step already made small freezes the recursion and reproduces a
one-step solution to round-off. And $z_{n+1}$ enters through $h_\theta$, so
the step is a nonlinear least-squares problem, not a linear solve.
At $q=R$ the step is linear in the coefficients and the whole trajectory is the
exact modal propagation of the bank coefficients, with no head and no
iteration; this is the “top rung” of the heat and wave ladders. The deployed
path factors the damped normal matrix by Cholesky; a variant assembles the Gram
matrix $S=B_0^{\top} B_0$ offline and forms $H=Dh_\theta^{\top} SDh_\theta$ directly, which is
exact and is reported as an algebraic ablation of the same step.

### A.3 Nonlinear advection: Burgers

<!-- section sources: none (prose only) -->

The full-order problem is $u_t+u(u_x+u_y)=\nu\Delta u$ with the non-conservative
sign-upwind stencil

$$
N(u)_{ij} \;=\; u_{ij}\,\big(\delta_x u + \delta_y u\big)_{ij},
  \qquad
  (\delta_x u)_{ij} = N\!\cdot\!
  \begin{cases}
    u_{ij}-u_{i-1,j}, & u_{ij}>0,\\[2pt]
    u_{i+1,j}-u_{ij}, & u_{ij}\le 0,
  \end{cases}
$$

<!-- equation (10) -->

$\delta_y$ analogous and switching on the same centre value, ghost zeros on all
four walls, and backward Euler in time,
$r_n(u)=u-u^{n}+\Delta t\big(N(u)-\nu\Delta_h u\big)$. Projecting and row
scaling by $\Lambda_\star=I+\Delta t \nu\Lambda$ gives, with
$c=h_\theta(z)+C_q y$,

$$
r_{w,n}
  \;=\;
  \big(I+\Delta t\,\nu\Lambda\big)^{-1}
  \Big[\,
    B_0c-B_0c_n
    \;+\;\Delta t\,\big(\,\widehat{N}(c)+\nu\Lambda B_0c\,\big)
  \Big],
$$

<!-- equation (11) -->

where the diffusion term is exact and the row scaling is the diagonal of the
linearised implicit operator in the modal basis, the modal form of the
Helmholtz preconditioner the full-order Newton solver uses. Only
$\widehat{N}(c)\approxP N(G c)$ requires a choice: dense evaluation at
every interior node, the sampled rule
$\widehat{N}(c)=\sum_{i\in\mathcal S}w_iP_{:,i}N_i(G c)$ with the sign
switch retained at the retained nodes, or the precomputed quadratic form
$\tfrac12 c^{\top} Q c$ with $Q$ the symmetrised tensor
$T_{iab}=\sum_xP_{ix}G_{xa}(DG)_{xb}$, which is an identity only where
$u>0$ at every stencil node of every trial state the solver visits. The last is
an algebraic route we verified in-job (residual and Jacobian parity below
$10^{-11}$ relative on sign-satisfying states) and do not time in this paper.
Time stepping uses a fixed substep count per output interval; each step is
warm-started at the previous code, and the linear extrapolation is used instead
when, and only when, its residual norm is smaller.

### A.4 Empirical quadrature: the fit and the counts

<!-- section sources: none (prose only) -->

A candidate node set $\mathcal C$ is drawn once with a fixed seed. For each of
$n_{\text{fit}}$ stored codes $c^{(\ell)}$ we evaluate the decoded state, form
the per-node contributions $N_i(G c^{(\ell)})$ and the exact full-grid
target $P N(G c^{(\ell)})$, and solve

$$
\min_{w\ge 0}\;
  \lVert \,\mathcal{G}\,w-\beta\, \rVert_2,
  \qquad
  \mathcal{G}_{(\ell,i),\,c}=P_{i,c}\,N_c\big(G c^{(\ell)}\big),
  \qquad
  \beta_{(\ell,i)}=\big[P N(G c^{(\ell)})\big]_i ,
$$

<!-- equation (12) -->

rows normalised by their Euclidean norms, support grown greedily by the
Lawson–Hanson criterion with an exact non-negative refit after each addition
and a hard cap of $m$ nodes, padded to the requested count if the support
saturates. Three counts are separated: the requested node count $m$, the
achieved support, and the number of fit states $n_{\text{fit}}$. In the
certification cell the fit states are reachable states (states of the dense
solver's own converged per-step trajectory on fit trajectories), the held-out
states are 512 reachable states per rung from certification trajectories
disjoint from both the fit and the evaluation cases, and $\rho$ of
(6) is evaluated on those. A deployed rule is a stored triple: node
indices, weights, and the cached $m\times5\times R$ bank block.

### A.5 Solver constants and exit codes

<!-- section sources: none (prose only) -->

Damping starts at $\lambda_0=10^{-6}$ (Poisson, Burgers) or $10^{-4}$ (heat);
$\lambda\leftarrow\max(\lambda/3,10^{-12})$ on acceptance and
$\min(10\lambda,\lambda_{\max})$ on rejection; the trust radius is taken from the
spread of the training codes. Exit codes differ per cell and are reported per
cell. Burgers: 4 stationarity, 1 residual, 2 tiny step, 3 rejected at
$\lambda\ge10^{14}$, 0 budget. Poisson: 6 stationarity, 2 relative residual, 1
stall, 3 damping limit, 5 non-finite initial value, 0 budget. Heat: 1
stationarity, 2 tiny step, 3 damping limit, 4 non-finite, 0 budget. The heat
criterion divides by $\lVert J_{} \rVert_{F}$ only, applied to a residual already
normalised by the target norm; Poisson and Burgers use (5)
directly. The completion rule that accepts an attained initial fit
(§3.3) was pre-registered in the panel cell's design before
the job ran; the stricter flag is printed beside it in Table 13.

### A.6 What the method is, stated once

<!-- section sources: none (prose only) -->

For a reader comparing this construction with related descriptions, the
following hold for every result here. Dirichlet enforcement is a smooth
polynomial factor folded into the bank, with zero lift, and covers homogeneous
data only. The projection is least-squares Petrov–Galerkin with a fixed test
space, not tangent Galerkin; the two differ even for a linear trial space,
since $V^\top A^\top(AVz-b)=0$ and $V^\top(AVz-b)=0$ are different conditions.
The heat scheme is Crank–Nicolson and the reduced step is a nonlinear
least-squares problem, not a linear system in $z_{n+1}$; no latent mass
matrix is formed and no latent ODE is integrated. The quadrature fit targets
the weak advection term with $M$ rows per snapshot; the requested node count
$m$, the achieved support and the fit-state count $n_{\text{fit}}$ are three
different numbers. No conjugate-gradient solve is applied to the projected
operator. There is no encoder in the deployed path. The update is damped
Levenberg–Marquardt, not undamped Gauss–Newton. The Poisson ground truth is
the verified discrete solution, not an analytic field.

## B Architecture

<!-- section sources: none (prose only) -->

![Figure 2](figures/architecture.png)

**Figure 2.** Architecture and data flow. Purple: trained once and frozen for every
result. Blue: assembled offline once per mesh. Orange: computed inside the timed
query. Green: chosen at run time among artefacts that already exist. Grey:
supplied, or a comparator.

## C Provenance

<!-- section sources: every lane; tables/provenance.json -->

**Table 3.** Provenance of every result table: job id, GPU, source commit and
checkpoint. The SHA256 of every machine-readable file the generator read is in
`tables/provenance.json`. Every job asserted `jax_backend=gpu`,
float64 and highest matmul precision before doing any work.

<!-- table: T02_provenance -->
| table | lane | job id(s) | GPU | commit | checkpoint |
|---|---|---|---|---|---|
| T3, T5 | b-panel | 3780638 | NVIDIA A100 80GB PCIe | 89df5cd18b01… | incumbent (gate checkpoint_unchanged; hash in T2, tuning row) |
| T4 | b-qxm | G1 = 3780175 (NVIDIA A100 80GB PCIe), G2 = 3780177 (NVIDIA A100-PCIE-40GB), S1 = 3780178 (NVIDIA A100-PCIE-40GB) | per job | ed431edb498a | 18f0266ae6f04542… |
| T6a, T7 | head-ablation (Burgers) | 3711424 | NVIDIA A100 80GB PCIe | 2718bd320dce… | 18f0266ae6f04542… |
| T6b, T7 | head-ablation (Poisson) | 3711736 | NVIDIA A100 80GB PCIe | 6759adcc9d85… | a128e7635c31… |
| T8 | fixed-checkpoint tuning | 3712269 | NVIDIA A100 80GB PCIe | d2b93bd4f56d… | 18f0266ae6f0… |
| T9 | b-eqtop | bet101 = 3780164 (NVIDIA A100 80GB PCIe, commit ace8936e42b3…); bet201 = 3780165 (NVIDIA A100 80GB PCIe, commit ace8936e42b3…) | see job list | see job list | 18f0266ae6f04542… |
| T10 | mesh-ladder (Burgers) | 3711388 | NVIDIA A100-PCIE-40GB | 521cdced6f4d… | 18f0266ae6f0… |
| T10 | mesh-ladder (Poisson) | 3711389 | NVIDIA A100 80GB PCIe | 521cdced6f4d… | a128e7635c31… |
| T11a | w-ladder | $64^2$: job 3780447 (NVIDIA A100 80GB PCIe, commit 0bb3cc86); $256^2$: job 3783805 (NVIDIA A100-PCIE-40GB, commit 2655bb01); $1024^2$: job 3780450 (NVIDIA A100 80GB PCIe, commit 0bb3cc86) | per job | per job | frozen-math SHA asserted in job |
| T11d | heat linear bank (2026-09-10) | 3511417 | NVIDIA A100-PCIE-40GB | 73fdaa88eb75… | expanded_seed790715 (frozen) |
| T11b, T11c | p-linear | $256^2$: job 3780692 (NVIDIA A100-PCIE-40GB, commit b43a437d7360); $1024^2$: job 3783813 (NVIDIA H200, commit 3e411b5ac59d); head capacity job 3783883 | per job | per job | R=512/K=32 checkpoint (pbh02 primary) |
| T14 | no-second | 3780138, 3780139, 3780625 (+ FNO 3710846, 3702464) | A100 80GB PCIe | c4f8b045 / 339c026b | operator checkpoints hash-verified in job |
| T15 | b-speed | 3745655 (spd01), 3745656 (fine01), 3745913 (comp01) | A100 80GB PCIe | 8fdfbb08 / 94399dd6 | 18f0266ae6f04542… |
| T16 | b-head-train | 3745912 (training), 3749074 (evaluation) | A100-PCIE-40GB | 0f0c56f7 / 2b9e7ee7 | trained checkpoints hashed in archive |
| T18 | lshape | 3783786 | NVIDIA A100 80GB PCIe | 1086ccefdcb5… | 7 heads + bases Git-tracked |
| T12, T13 | b-seeds | **[PENDING: pending: b-seeds]** | — | — | — |

## D Full tables

<!-- section sources: every lane (see each table comment) -->

**Table 4.** Problem specification. Cohort and reduced sizes are read from the run
configurations where recorded; the sealed final cohorts have not been opened
for any cell.

<!-- table: T01_problems -->
| PDE | equation, domain, boundary | meshes | time stepping | reduced sizes | reference | cohorts |
|---|---|---|---|---|---|---|
| Burgers 2D | $u_t+u(u_x+u_y)=\nu\Delta u$, $(0,1)^2$, $u\|_{\partial\Omega}=0$ | $256^2$ (ladder 64–1024) | $\Delta t=0.005$, backward Euler, sign-upwind | $K=16$, $R=512$ | refined $ 4096^2$, $\Delta t=0.00015625$ | 6 development cases; 32 held-out (tuning); sealed cohort unopened |
| Poisson 2D | $-\Delta u=f$, $(0,1)^2$, $u\|_{\partial\Omega}=0$ | $256^2$, $1024^2$ | none (elliptic) | $K=16$, $R=128$ (incumbent); $K=32$, $R=512$ | exact discrete (DST); 2048$^2$ refinement | 12 development sources |
| Heat 2D | $u_t=\kappa\Delta u$, $(0,1)^2$ | $64^2$–$1024^2$ | Crank–Nicolson | $k=8$, $R=32$ | exact modal | 12 development cases (earlier cell, job 3511417) |
| Wave 2D (reflective) | $u_{tt}=c^2\Delta u$, $(0,1)^2$, $u\|_{\partial\Omega}=0$ | $64^2$, $256^2$, $1024^2$ | RK4 on the manifold; exact modal propagation for the bank | $K=32$, $R=64$ | direct DST | 8 development cases |
| Poisson, L-shape | $-\Delta u=f$, $(0,1)^2\setminus[\tfrac12,1)^2$ | $256^2$, $512^2$ | none | $K\in\{16,32\}$, $R\in\{256,512,514\}$ | sparse direct (SuperLU) | 3072 / 256 / 32 sources (train / selection / development) |

**Table 5.** Sampled problem families, transcribed from the generator sources named
in the last column (code constants, not run outputs).

<!-- table: T01b_spec -->
| PDE | equation and boundary | sampled family (transcribed from the generator source) | source |
|---|---|---|---|
| Burgers 2D | $u_t+u(u_x+u_y)=\nu\Delta u$ on $(0,1)^2$, $u=0$ on $\partial\Omega$ | $u_0=a\exp(-\lvert x-c\rvert^2/2w^2)$, $c_i\sim U(0.15,0.85)$, $w\sim U(0.05,0.20)$, $a\sim U(0.5,2)$; $\nu\sim\log U(0.01,0.1)$; outputs $t\in\{0.05,\dots,0.25\}$ | `experiments/mr-burgers2d/engines.py:params_draw` |
| Poisson 2D | $-\Delta u=f$ on $(0,1)^2$, $u=0$ on $\partial\Omega$ | $f=a\exp(-\lvert x-c\rvert^2/2w^2)$, $c_i\sim U(0.15,0.85)$, $w\sim\log U(0.02,0.1)$, $a\sim U(0.5,2)$ | `multistage-precision/ms_parametric.py:sample_params` |
| Poisson, L-shape | same source family on $(0,1)^2\setminus[\tfrac12,1)^2$ | as above; sources centred in the removed quadrant rejected | `experiments/lshape` |
| Heat 2D | $u_t=\kappa\Delta u$ on $(0,1)^2$, $\kappa=0.02$ fixed | polynomial-boundary Gaussian family (single_bc_poly_gaussian_v1); Crank–Nicolson $\Delta t=0.025$ | heat linear-bank report, job 3511417 |
| Wave 2D (reflective) | $u_{tt}=c^2\Delta u$ on $(0,1)^2$, $u=0$ on $\partial\Omega$ | compact bump $\times$ Gaussian: half-widths $s_i\sim U(0.36,0.42)$, centre $c_i\sim U(s_i{+}0.025,\,1{-}s_i{-}0.025)$, amplitude $\sim U(0.7,1.3)$, $\sigma_i\sim U(0.12,0.16)$, advective velocity $v_i\sim U(-0.5,0.5)$ (zero every fourth case); speed $c\sim U(0.85,1.15)$ | `experiments/multiresolution-wave/audit_dynamics.py:parameter_rows` |

**Table 6.** Offline cost per stage, from the lane records: head training on the
retrained arms (the incumbent's own training time was not recorded), the
correction-direction fit, and the NNLS fit of each rule the primary EQ ladder
ran. Every rule is refit when $m$ changes; nothing is nested.

<!-- table: T17_offline_cost -->
| offline stage | seconds | amortised over | source |
|---|---|---|---|
| head training, like-for-like retrain of the incumbent recipe (4608 traj., 200k steps) | 387 | once per checkpoint | b-head-train job 3745912 (A100-PCIE-40GB) |
| head training, selected retrained arm | 1459 | once per checkpoint | b-head-train job 3745912 (A100-PCIE-40GB) |
| correction directions $C_q$ (PCA of the head residual; flattened LM fit of the best-found codes) | 1242 | once per checkpoint, all $q$ | cheap-corrections job 3734098 |
| certified EQ rule, $q=0$, $m=1024$ (reachable, qrg304:reachable) | 72 | once per rung and rule | b-eqtop job 3780164 |
| certified EQ rule, $q=16$, $m=1024$ (reachable, qrg304:reachable) | 147 | once per rung and rule | b-eqtop job 3780164 |
| certified EQ rule, $q=32$, $m=1024$ (reachable, qrg304:reachable) | 155 | once per rung and rule | b-eqtop job 3780164 |
| certified EQ rule, $q=64$, $m=1024$ (reachable, qrg304:reachable) | 160 | once per rung and rule | b-eqtop job 3780164 |
| certified EQ rule, $q=128$, $m=2048$ (fs64, this_job:reachable) | 2920 | once per rung and rule | b-eqtop job 3780164 |
| certified EQ rule, $q=256$, $m=2048$ (fs64, this_job:reachable) | 2650 | once per rung and rule | b-eqtop job 3780164 |

**Table 7.** How the corrections are solved on Burgers, one job (job
3734098, NVIDIA A100 80GB PCIe): joint LM on $(z,y)$, block-damped, and plain
variable projection at $q=64$ and $128$. Same error where all converge; the
block-damped step is the one kept.

<!-- table: T19_solver_variants -->
| arm | $M$ | worst all-times % | budget exits | all stationary | GPU ms |
|---|---|---|---|---|---|
| $q{=}64$, joint LM on $(z,y)$ | 320 | 2.1489 | 6 | no | 1119.7 |
| $q{=}64$, block-damped | 320 | 2.1489 | 0 | yes | 619.2 |
| $q{=}64$, plain variable projection | 320 | 2.1489 | 0 | yes | 9527.9 |
| $q{=}128$, plain variable projection | 256 | 1.8116 | 297 | no | 47566.9 |
| $q{=}128$, block-damped | 256 | 1.8116 | 0 | yes | 825.5 |

![Figure 3](figures/fig_eq_certification.png)

**Figure 3.** **Provisional.** NNLS fit residual on the fitting set against
held-out $\rho_{\max}$ on reachable states for every fitted quadrature rule of
the certification cell (jobs 3780164, 3780165), with the primary and tight bars.
The fit residual does not predict the held-out error; hollow squares are rules
fitted on static snapshots. One draw per rule; replication 3783811\
pending.

**Table 8.** Rank against test count (dense advection, budget 600, single seed). Both
fixed-$M$ ladders are shown; the $M=256$ ladder fails the bar and the
$M=1088$ ladder passes it, and $M=1088$ was chosen after the grid was run.
The scheduled ladder's cells span two jobs so its costs are not printed. Jobs G1 = 3780175 (NVIDIA A100 80GB PCIe), G2 = 3780177 (NVIDIA A100-PCIE-40GB), S1 = 3780178 (NVIDIA A100-PCIE-40GB); commit
ed431edb498a; checkpoint 18f0266ae6f04542….

<!-- table: T04_rank_vs_tests -->
| ladder | $q$ | $M$ | worst evolved % | GPU ms |
|---|---|---|---|---|
| fixed $M=256$ (job 3780177) | 0 | 256 | 1.2710 | 406.2 |
| fixed $M=256$ (job 3780177) | 64 | 256 | 1.1255 | 645.6 |
| fixed $M=256$ (job 3780177) | 128 | 256 | 1.0418 | 921.8 |
| fixed $M=1088$ (job 3780177) | 0 | 1088 | 1.2657 | 848.0 |
| fixed $M=1088$ (job 3780177) | 64 | 1088 | 1.0593 | 1359.4 |
| fixed $M=1088$ (job 3780177) | 128 | 1088 | 0.8711 | 1886.5 |
| fixed $M=1088$ (job 3780177) | 256 | 1088 | 0.5194 | 4377.9 |
| scheduled $M=4(K+q)$ | 0 | 64 | 1.8890 | — |
| scheduled $M=4(K+q)$ | 16 | 128 | 1.3985 | — |
| scheduled $M=4(K+q)$ | 32 | 192 | 1.2336 | — |
| scheduled $M=4(K+q)$ | 64 | 320 | 1.0843 | — |
| scheduled $M=4(K+q)$ | 128 | 576 | 0.8930 | — |
| scheduled $M=4(K+q)$ | 256 | 1088 | 0.5194 | — |

**Table 9.** Three layers of deployed error, worst over each cell's development
cases: what the bank can express, what the augmented manifold can reach
(multistart oracle, an upper bound), what the deployed iteration returned. The
p-linear $q>0$ oracle was mis-scaled and is retracted; only its $q=0$ oracle
appears.

<!-- table: T07_three_layers -->
| cell | arm | bank floor % | best-found % | solved % | best/floor | solved $-$ best (pp) | source |
|---|---|---|---|---|---|---|---|
| Burgers $256^2$ | neural head, $q{=}0$ | 0.3918 | 2.5447 | 2.5629 | 6.49 | 0.018 | head-ablation |
| Poisson $1024^2$, $R{=}128$ | neural head | 2.3148 | 6.0926 | 6.0927 | 2.63 | 0.000 | head-ablation |
| Poisson $256^2$, $R{=}512$ | neural head $K{=}32$, $q{=}0$ | 0.7459 | 3.1211 | 3.1217 | 4.18 | 0.001 | p-linear |
| Poisson $1024^2$, $R{=}512$ | neural head $K{=}32$, $q{=}0$ | 0.7421 | 3.1139 | 3.1146 | 4.20 | 0.001 | p-linear |
| wave $256^2$ | head, $q{=}0$ | 2.4571 | 8.0772 | 11.3314 | 3.29 | 3.254 | w-ladder |
| wave $256^2$ | head, $q{=}32$ | 2.4571 | 2.4571 | 5.1038 | 1.00 | 2.647 | w-ladder |
| L-shape $256^2$ | `head_sdf_R512_K32` | 0.7766 | 3.1802 | 3.1814 | 4.09 | 0.001 | lshape (untimed) |
| L-shape $256^2$ | `head_sdf_R512_K16` | 0.7766 | 3.8607 | 3.8608 | 4.97 | 0.000 | lshape (untimed) |

**Table 10.** Poisson ladder at $256^2$ and $1024^2$ (one job per mesh:
$256^2$: job 3780692 (NVIDIA A100-PCIE-40GB, commit b43a437d7360); $1024^2$: job 3783813 (NVIDIA H200, commit 3e411b5ac59d); never compare costs across meshes; $q=512$ via LM is degenerate
by construction and its QR twin is the timed top rung).

<!-- table: T11b_poisson -->
| mesh | subject | $M$ | worst % | median % | total ms | device ms | valid | non-dom. (all) | non-dom. (reduced) |
|---|---|---|---|---|---|---|---|---|---|
| $256^2$ | $q{=}0$ | 129 | 3.1567 | 0.5080 | 9.214 | 6.703 | 36/36 | no | no |
| $256^2$ | $q{=}32$ | 257 | 2.4699 | 0.4437 | 9.537 | 6.893 | 36/36 | no | no |
| $256^2$ | $q{=}64$ | 384 | 2.0808 | 0.4127 | 9.819 | 7.347 | 36/36 | no | no |
| $256^2$ | $q{=}128$ | 640 | 1.5497 | 0.3195 | 10.219 | 7.536 | 36/36 | no | no |
| $256^2$ | $q{=}256$ | 1153 | 0.9689 | 0.1315 | 9.658 | 7.121 | 36/36 | no | no |
| $256^2$ | $q{=}512$ | 2177 | 0.7459 | 0.0580 | 117.854 | 114.990 | 0/36 | no | no |
| $256^2$ | linear top rung ($q{=}R$, QR) | 2177 | 0.7459 | 0.0580 | 3.221 | 1.693 | 36/36 | no | yes |
| $256^2$ | free bank (LM) | 2177 | 0.7789 | 0.0597 | 23.295 | 21.416 | 36/36 | no | no |
| $256^2$ | POD $k'{=}32$ | 129 | 9.9912 | 2.7371 | 3.766 | 2.094 | 36/36 | no | no |
| $256^2$ | POD $k'{=}64$ | 257 | 5.7375 | 0.6105 | 4.062 | 2.309 | 36/36 | no | no |
| $256^2$ | POD $k'{=}128$ | 512 | 2.5117 | 0.1060 | 4.825 | 3.231 | 36/36 | no | no |
| $256^2$ | POD $k'{=}256$ | 1025 | 0.8165 | 0.0134 | 6.171 | 4.566 | 36/36 | no | no |
| $256^2$ | POD $k'{=}512$ | 2048 | 0.1855 | 0.0014 | 10.002 | 8.340 | 36/36 | no | yes |
| $256^2$ | direct DST | — | 0.0000 | 0.0000 | 2.526 | 0.227 | 36/36 | yes | no |
| $256^2$ | CG $10^{-2}$ | — | 0.1527 | 0.0554 | 23.818 | 22.290 | 36/36 | no | no |
| $256^2$ | CG $10^{-4}$ | — | 0.0012 | 0.0003 | 32.698 | 30.962 | 36/36 | no | no |
| $256^2$ | CG $10^{-6}$ | — | 0.0000 | 0.0000 | 38.046 | 36.330 | 36/36 | no | no |
| $1024^2$ | $q{=}0$ | 129 | 3.1495 | 0.5079 | 6.394 | 3.941 | 36/36 | no | no |
| $1024^2$ | $q{=}32$ | 257 | 2.4641 | 0.4437 | 6.592 | 4.108 | 36/36 | no | no |
| $1024^2$ | $q{=}64$ | 384 | 2.0760 | 0.4128 | 6.745 | 4.240 | 36/36 | no | no |
| $1024^2$ | $q{=}128$ | 640 | 1.5459 | 0.3195 | 6.875 | 4.393 | 36/36 | no | no |
| $1024^2$ | $q{=}256$ | 1153 | 0.9648 | 0.1315 | 6.849 | 4.319 | 36/36 | no | no |
| $1024^2$ | $q{=}512$ | 2177 | 0.7421 | 0.0580 | 43.220 | 40.632 | 0/36 | no | no |
| $1024^2$ | linear top rung ($q{=}R$, QR) | 2177 | 0.7421 | 0.0580 | 4.424 | 1.830 | 36/36 | no | yes |
| $1024^2$ | free bank (LM) | 2177 | 0.7751 | 0.0596 | 13.021 | 10.662 | 36/36 | no | no |
| $1024^2$ | POD $k'{=}32$ | 129 | 9.9816 | 2.7357 | 3.696 | 1.318 | 36/36 | no | no |
| $1024^2$ | POD $k'{=}64$ | 257 | 5.7282 | 0.6099 | 3.970 | 1.571 | 36/36 | no | no |
| $1024^2$ | POD $k'{=}128$ | 512 | 2.5019 | 0.1058 | 4.397 | 2.095 | 36/36 | no | no |
| $1024^2$ | POD $k'{=}256$ | 1025 | 0.8119 | 0.0133 | 5.262 | 2.908 | 36/36 | no | no |
| $1024^2$ | POD $k'{=}512$ | 2048 | 0.1838 | 0.0014 | 6.972 | 4.572 | 36/36 | no | yes |
| $1024^2$ | direct DST | — | 0.0000 | 0.0000 | 3.248 | 0.237 | 36/36 | yes | no |
| $1024^2$ | CG $10^{-2}$ | — | 0.0722 | 0.0297 | 62.498 | 60.017 | 36/36 | no | no |
| $1024^2$ | CG $10^{-4}$ | — | 0.0004 | 0.0001 | 85.988 | 83.437 | 36/36 | no | no |
| $1024^2$ | CG $10^{-6}$ | — | 0.0000 | 0.0000 | 103.642 | 101.260 | 36/36 | no | no |

**Table 11.** Heat 2D, an earlier cell of the same decoder family (job
3511417, NVIDIA A100-PCIE-40GB, commit 73fdaa88eb75…; $k=8$, $R=32$, twelve
development cases, three repetitions; the finest three meshes of a five-rung
ladder). The linear evolution of the full bank is the $q=R$ rung.

<!-- table: T11d_heat -->
| mesh | arm | worst physical % | worst same-grid % | GPU ms | host ms |
|---|---|---|---|---|---|
| $64^2$ | linear bank, exact modal evolution ($q{=}R$, no head) | 1.675754 | 1.675754 | 0.116167 | 1.146627 |
| $64^2$ | nonlinear head ($k{=}8$) | 4.559260 | 4.554246 | 12.091245 | 12.540239 |
| $64^2$ | same-grid direct solve | 0.089749 | 0.000000 | 0.184825 | 0.557447 |
| $64^2$ | coarse ($16^2$) direct solve, interpolated | 2.585355 | 2.623291 | 0.176126 | 0.518843 |
| $256^2$ | linear bank, exact modal evolution ($q{=}R$, no head) | 1.675830 | 1.675830 | 0.137731 | 2.326213 |
| $256^2$ | nonlinear head ($k{=}8$) | 4.555681 | 4.555390 | 11.888083 | 13.147866 |
| $256^2$ | same-grid direct solve | 0.005602 | 0.000000 | 0.209627 | 1.438594 |
| $256^2$ | coarse ($16^2$) direct solve, interpolated | 2.578601 | 2.581196 | 0.190585 | 1.400465 |
| $1024^2$ | linear bank, exact modal evolution ($q{=}R$, no head) | 1.675830 | 1.675830 | 0.561659 | 17.681166 |
| $1024^2$ | nonlinear head ($k{=}8$) | 4.555479 | 4.555461 | 12.318828 | 36.325427 |
| $1024^2$ | same-grid direct solve | 0.000350 | 0.000000 | 1.034563 | 25.068664 |
| $1024^2$ | coarse ($16^2$) direct solve, interpolated | 2.577910 | 2.578073 | 0.313116 | 24.364979 |

**Table 12.** Neural operators on the shared Burgers data: capacity, epochs, whether
the arm was still improving at its budget, worst and median error on the
matched eight-case cohort and on the 32 held-out cases. Accuracy is comparable
across jobs; timing never is, and none is shown. Single seed, one mesh, one
3000 s budget per capacity.

<!-- table: T14_operators -->
| arm | family | params | epochs | still improving | 8-case worst % | val-32 worst % | val-32 median % | job |
|---|---|---|---|---|---|---|---|---|
| `same_nt1e-2_dt005` | FOM | — | — | — | 0.9978 | — | — | `3702709` |
| `same_nt1e-6_dt005` | FOM | — | — | — | 1.3811 | — | — | `3702709` |
| `same_nt1e-4_dt005` | FOM | — | — | — | 1.3811 | — | — | `3702709` |
| `unet-small` | U-Net | 4,368,389 | 2327 | yes | 1.4712 | 5.5937 | 1.3963 | `3780138` |
| `unet-medium` | U-Net | 7,763,461 | 1963 | yes | 1.5189 | 3.9622 | 1.3169 | `3780138` |
| `tsol-refine` | Transolver | 3,108,240 | 1628 | no | 1.5224 | 9.3183 | 1.3591 | `3780139` |
| `unet-refine` | U-Net | 7,763,461 | 1962 | yes | 1.7110 | 7.5176 | 1.0806 | `3780138` |
| `same_nt1e-2_dt01` | FOM | — | — | — | 1.8642 | — | — | `3702709` |
| `rom` | ROM | — | — | — | 1.8671 | — | — | `3702709` |
| `coarse_half_dt005` | FOM | — | — | — | 1.8993 | — | — | `3702709` |
| `unet-large` | U-Net | 17,462,021 | 908 | yes | 2.3176 | 4.2340 | 1.3642 | `3780138` |
| `fno-large` | FNO | 17,877,317 | 692 | yes | 2.4829 | 6.3825 | 1.8054 | `3710846` |
| `tsol-small` | Transolver | 3,108,240 | 1628 | yes | 2.5273 | 6.8213 | 1.4522 | `3780139` |
| `tsol-medium` | Transolver | 6,952,208 | 586 | yes | 2.8498 | 9.5338 | 2.3846 | `3780139` |
| `fno-medium` | FNO | 5,780,213 | 1198 | yes | 2.8882 | 6.0423 | 1.7638 | `3710846` |
| `fno-small` | FNO | 1,193,125 | 1741 | yes | 2.9350 | 7.4812 | 1.9552 | `3710846` |
| `tsol-large` | Transolver | 12,322,960 | 882 | yes | 3.0988 | 6.3953 | 3.0690 | `3780139` |
| `fno-refine` | FNO | 17,877,317 | 688 | yes | 3.2660 | 7.1747 | 1.8810 | `3710846` |
| `coarse_quarter_dt01` | FOM | — | — | — | 3.6398 | — | — | `3702709` |

**Table 13.** Every timed subject of the $256^2$ same-allocation panel (job
3780638). “conv.” is the pre-registered completion rule that accepts
an attained initial fit; “strict” is the earlier rule that does not.

<!-- table: T05_panel_all -->
| subject | family | $q$ / $k^\prime$ | $M$ | quad. | evolved % | all % | $t{=}0$ % | vs ref % | GPU ms | host ms | conv. | strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `q0_M256_dense_g1em06` | rom | 0 | 256 | dense | 1.2710 | 2.5629 | 2.5629 | 4.0637 | 341.1 | 343.4 | yes | yes |
| `q0_M64_dense_g1em06` | rom | 0 | 64 | dense | 1.8890 | 2.5629 | 2.5629 | 4.5575 | 284.8 | 287.4 | yes | yes |
| `q0_M64_eqcert_g0p001` | rom | 0 | 64 | eq | 1.8898 | 2.5628 | 2.5628 | 4.5521 | 42.6 | 45.1 | yes | yes |
| `q0_M64_eqcert_g1em06` | rom | 0 | 64 | eq | 1.8891 | 2.5629 | 2.5629 | 4.5529 | 57.9 | 60.4 | yes | yes |
| `q16_M128_dense_g1em06` | rom | 16 | 128 | dense | 1.3985 | 2.4806 | 2.4806 | 4.1065 | 363.0 | 365.6 | yes | yes |
| `q16_M128_eqcert_g0p001` | rom | 16 | 128 | eq | 1.4272 | 2.4806 | 2.4806 | 4.1136 | 60.4 | 62.9 | yes | yes |
| `q16_M128_eqcert_g1em06` | rom | 16 | 128 | eq | 1.4270 | 2.4806 | 2.4806 | 4.1145 | 79.7 | 82.2 | yes | yes |
| `q32_M192_dense_g1em06` | rom | 32 | 192 | dense | 1.2336 | 2.3534 | 2.3534 | 4.0814 | 442.6 | 445.3 | yes | yes |
| `q32_M192_eqcert_g0p001` | rom | 32 | 192 | eq | 1.2497 | 2.3534 | 2.3534 | 4.0805 | 72.3 | 74.8 | yes | yes |
| `q32_M192_eqcert_g1em06` | rom | 32 | 192 | eq | 1.2493 | 2.3534 | 2.3534 | 4.0818 | 95.7 | 98.0 | yes | yes |
| `q64_M320_dense_g1em06` | rom | 64 | 320 | dense | 1.0843 | 2.1489 | 2.1489 | 4.1013 | 637.3 | 639.7 | yes | yes |
| `q64_M320_eqcert_g0p001` | rom | 64 | 320 | eq | 1.2278 | 2.1489 | 2.1489 | 4.0836 | 88.5 | 90.9 | yes | yes |
| `q64_M320_eqcert_g1em06` | rom | 64 | 320 | eq | 1.2275 | 2.1489 | 2.1489 | 4.0855 | 119.1 | 121.9 | yes | yes |
| `q128_M576_dense_g1em06` | rom | 128 | 576 | dense | 0.8930 | 1.8116 | 1.8116 | 4.0797 | 1223.2 | 1225.7 | yes | yes |
| `q128_M576_eqcert_g0p001` | rom | 128 | 576 | eq | 0.8926 | 1.8116 | 1.8116 | 4.0791 | 189.0 | 191.6 | yes | yes |
| `q128_M576_eqcert_g1em06` | rom | 128 | 576 | eq | 0.8925 | 1.8116 | 1.8116 | 4.0809 | 247.0 | 249.7 | yes | yes |
| `q256_M1088_dense_g1em06` | rom | 256 | 1088 | dense | 0.5194 | 0.9053 | 0.9053 | 4.0391 | 4164.6 | 4167.1 | yes | yes |
| `q256_M1088_eqcert_g0p001` | rom | 256 | 1088 | eq | 1.0324 | 1.0324 | 0.9053 | 4.0322 | 429.4 | 431.9 | yes | yes |
| `q256_M1088_eqcert_g1em06` | rom | 256 | 1088 | eq | 1.0361 | 1.0361 | 0.9053 | 4.0332 | 700.4 | 703.0 | yes | yes |
| `q0_M64_eqcert_g1em06_fastL4` | fast | 0 | 64 | eq | 1.8891 | 2.5629 | 2.5629 | 4.5529 | 39.1 | 41.7 | yes | yes |
| `pod16_M64_dense` | pod | 16 | 64 | dense | 28.7250 | 61.6503 | 61.6503 | 61.6503 | 46.9 | 49.4 | yes | yes |
| `pod32_M128_dense` | pod | 32 | 128 | dense | 18.7995 | 47.0681 | 47.0681 | 47.0681 | 81.7 | 84.2 | yes | no |
| `pod64_M256_dense` | pod | 64 | 256 | dense | 7.0835 | 19.8156 | 19.8156 | 19.8156 | 148.2 | 150.8 | yes | no |
| `pod128_M512_dense` | pod | 128 | 512 | dense | 1.9464 | 10.1198 | 10.1198 | 10.1198 | 337.2 | 339.6 | yes | no |
| `pod256_M1024_dense` | pod | 256 | 1024 | dense | 0.7109 | 3.7698 | 3.7698 | 4.0362 | 927.8 | 930.8 | yes | no |
| `pod512_M2048_dense` | pod | 512 | 2048 | dense | 0.2184 | 0.6125 | 0.6125 | 4.0266 | 2928.9 | 2932.9 | yes | no |
| `free512_M1024_dense` | free | 512 | 1024 | dense | 0.4471 | 0.6027 | 0.6027 | 4.0423 | 2542.3 | 2545.9 | yes | no |
| `fno-large` | fno | — | — | — | 7.4164 | 7.4164 | 0.0000 | 5.7495 | 7.2 | 7.5 | — | — |
| `dense_tight` | fom | — | — | — | 0.0000 | 0.0000 | 0.0000 | 4.0265 | 60.8 | 63.3 | — | — |
| `fft_tight` | fom | — | — | — | 0.0000 | 0.0000 | 0.0000 | 4.0265 | 89.3 | 92.0 | — | — |
| `nt1e-2_dt005` | fom | — | — | — | 3.7127 | 3.7127 | 0.0000 | 2.4737 | 15.8 | 18.3 | — | — |
| `nt1e-2_dt01` | fom | — | — | — | 3.1999 | 3.1999 | 0.0000 | 3.6168 | 9.3 | 11.4 | — | — |
| `nt1e-3_dt005` | fom | — | — | — | 0.0489 | 0.0489 | 0.0000 | 4.0399 | 31.4 | 33.7 | — | — |
| `nt1e-3_dt01` | fom | — | — | — | 1.5179 | 1.5179 | 0.0000 | 5.1761 | 20.0 | 22.4 | — | — |
| `nt1e-4_dt005` | fom | — | — | — | 0.0338 | 0.0338 | 0.0000 | 4.0320 | 36.7 | 39.4 | — | — |
| `nt1e-4_dt01` | fom | — | — | — | 1.5109 | 1.5109 | 0.0000 | 5.1552 | 27.9 | 30.2 | — | — |

**Table 14.** Fixed-$q$ sweeps in the test count $M$ (dense; errors comparable
across the three b-qxm jobs, costs not shown). The last column restricts the
span to at least two tests per unknown.

<!-- table: T04b_fixed_q -->
| $q$ | $M$ sweep | worst evolved % | monotone | span | best $M$ | span, $M\ge 2(K{+}q)$ |
|---|---|---|---|---|---|---|
| 0 | 64, 128, 256, 512, 1088, 2048, 4096 | 1.8890 / 1.4695 / 1.2710 / 1.2662 / 1.2657 / 1.2652 / 1.2649 | yes | 1.49$\times$ | 4096 | 1.49$\times$ |
| 16 | 128, 192, 256, 512, 1088 | 1.3985 / 1.2690 / 1.2305 / 1.2204 / 1.2204 | yes | 1.15$\times$ | 1088 | 1.15$\times$ |
| 32 | 192, 256, 320, 384, 768, 1088 | 1.2336 / 1.1869 / 1.1717 / 1.1665 / 1.1634 / 1.1629 | yes | 1.06$\times$ | 1088 | 1.06$\times$ |
| 64 | 128, 256, 320, 512, 576, 640, 1088, 1280, 2048, 4096 | 1.8032 / 1.1255 / 1.0843 / 1.0647 / 1.0633 / 1.0619 / 1.0593 / 1.0588 / 1.0594 / 1.0620 | no | 1.70$\times$ | 1280 | 1.06$\times$ |
| 128 | 256, 576, 1088, 1152 | 1.0418 / 0.8930 / 0.8711 / 0.8689 | yes | 1.20$\times$ | 1152 | 1.03$\times$ |
| 256 | 544, 1088, 2176 | 0.7566 / 0.5194 / 0.3736 | yes | 2.03$\times$ | 2176 | 2.03$\times$ |

**Table 15.** Head ablation, Burgers $256^2$ (job 3711424,
NVIDIA A100 80GB PCIe). Same-grid error against the converged full-order solve;
“vs ref” against the $4096^2$ reference.

<!-- table: T06a_head_burgers -->
| arm | dim. | bank floor % | best-found % | solved same-grid % | vs ref % | GPU ms | stationary |
|---|---|---|---|---|---|---|---|
| (a) neural head, EQ | 16 | 0.3918 | 2.5447 | 2.5629 | 4.5546 | 47.6 | yes |
| (a) neural head, dense | 16 | 0.3918 | 2.5447 | 2.5629 | 4.5575 | 281.7 | yes |
| (b) linear map (fit to head outputs) | 16 | 0.3918 | 56.9296 | 56.9296 | 56.9296 | 19.9 | yes |
| (b) linear map (fit to truth) | 16 | 0.3918 | 61.5226 | 61.5226 | 61.5226 | 19.7 | yes |
| (c) quadratic map | 16 | 0.3918 | 31.8276 | 31.8276 | 31.8276 | 28.0 | yes |
| (e) POD-LSPG $k'{=}16$ | 16 | 61.6503 | 61.6503 | 61.6503 | 61.6503 | 46.0 | yes |
| (e) POD-LSPG $k'{=}32$ | 32 | 47.0681 | 47.0681 | 47.0681 | 47.0681 | 80.7 | yes |
| (e) POD-LSPG $k'{=}64$ | 64 | 19.8147 | 19.8147 | 19.8156 | 19.8156 | 144.1 | no |
| (e) POD-LSPG $k'{=}128$ | 128 | 10.1189 | 10.1189 | 10.1198 | 10.1198 | 328.3 | no |
| (d) unrestricted bank ($R{=}512$) | 512 | 0.3918 | 0.3918 | 0.6027 | 4.0423 | 2417.6 | no |
| FOM Newton, loose | — | — | — | 3.7127 | 2.4737 | 15.4 | — |
| FOM Newton, tight (reference) | — | — | — | 0.0000 | 4.0265 | 88.7 | — |

**Table 16.** Head ablation, Poisson $1024^2$, $R=128$ (job 3711736,
NVIDIA A100 80GB PCIe).

<!-- table: T06b_head_poisson -->
| arm | dim. | bank floor % | best-found % | solved % | query ms |
|---|---|---|---|---|---|
| (a) neural head | 16 | 2.3148 | 6.0926 | 6.0927 | 5.843 |
| (a$+$) head $+$ 32 eliminated corrections | 16 | — | — | 4.6670 | 7.317 |
| (b) linear map (head outputs) | 16 | 2.3148 | 17.2966 | 17.2966 | 5.163 |
| (b) linear map (truth) | 16 | 2.3148 | 21.1648 | 21.1648 | 5.329 |
| (c) quadratic map | 16 | 2.3148 | 16.6302 | 16.6303 | 5.774 |
| (e) POD-LSPG $k'{=}8$ | 8 | 22.3196 | 22.3196 | 22.3196 | 4.633 |
| (e) POD-LSPG $k'{=}16$ | 16 | 20.0548 | 20.0548 | 20.0548 | 4.659 |
| (e) POD-LSPG $k'{=}32$ | 32 | 11.8589 | 11.8589 | 11.8589 | 5.050 |
| (e) POD-LSPG $k'{=}64$ | 64 | 7.7397 | 7.7397 | 7.7398 | 5.170 |
| (e) POD-LSPG $k'{=}128$ | 128 | 4.3213 | 4.3213 | 4.3452 | 5.930 |
| (d) unrestricted bank ($R{=}128$) | 128 | 2.3148 | 2.3148 | 2.3274 | 5.995 |
| direct DST | — | — | — | 0.0007 | 4.341 |

**Table 17.** Solver knobs at a fixed Burgers checkpoint on 32 held-out cases (job
3712269, NVIDIA A100 80GB PCIe, checkpoint 18f0266ae6f0…). Loosening the
tolerance from $10^{-8}$ to $10^{-3}$ removes $35.6 %$ of the
query for a change from $6.712$ to $6.701 %$
worst error; a cap of 2 is a cliff ($90.3 %$). The full-order
control at tolerance $10^{-4}$ beats the best ROM setting on both axes.

<!-- table: T08_solver_knobs -->
| setting | worst % (32 held-out) | GPU ms | early-stopped |
|---|---|---|---|
| EQ $m{=}256$, tol $10^{-6}$, cap 180 | 7.245 | 51.0 | 0/96 |
| EQ $m{=}512$, tol $10^{-8}$ | 6.712 | 64.7 | 0/96 |
| EQ $m{=}512$, tol $10^{-3}$ | 6.701 | 41.6 | 0/96 |
| EQ $m{=}512$, cap 2 (early-stopped) | 90.324 | 28.4 | 96/96 |
| FOM Newton $10^{-2}$, $\Delta t{=}0.01$ | 6.807 | 9.9 | 0/96 |
| FOM Newton $10^{-2}$, $\Delta t{=}0.005$ | 35.357 | 17.4 | 0/96 |
| FOM Newton $10^{-4}$, $\Delta t{=}0.005$ | 6.171 | 22.4 | 0/96 |
| FOM Newton $10^{-6}$, $\Delta t{=}0.005$ | 6.172 | 88.3 | 0/96 |
| FOM $128^2$, $\Delta t{=}0.005$ | 9.849 | 21.2 | 0/96 |
| FOM $64^2$, $\Delta t{=}0.01$ | 16.095 | 14.0 | 0/96 |

**Table 18.** Cold-start iteration-cap sweep on Poisson at $1024$ intervals
(local GB10 diagnostic; only ratios transfer). From the nearest training code
the converged answer is reached at a small cap; from $z=0$ the cap traces a
monotone curve that looks like an accuracy–cost frontier, every point of
which is dominated by starting well.

<!-- table: T08b_cold_start -->
| start | LM cap | worst % | median % | median iters | ms (GB10, ratios only) |
|---|---|---|---|---|---|
| nearest | 1 | 13.55 | 4.29 | 1 | 20.18 |
| nearest | 2 | 6.80 | 1.29 | 2 | 25.68 |
| nearest | 4 | 6.09 | 1.24 | 4 | 33.83 |
| nearest | 6 | 6.09 | 1.24 | 5 | 40.58 |
| nearest | 8 | 6.09 | 1.24 | 5 | 38.67 |
| nearest | 10 | 6.09 | 1.24 | 5 | 38.23 |
| nearest | 12 | 6.09 | 1.24 | 5 | 38.92 |
| nearest | 20 | 6.09 | 1.24 | 5 | 38.47 |
| nearest | 300 | 6.09 | 1.24 | 5 | 70.17 |
| zero | 1 | 91.88 | 74.88 | 1 | 20.60 |
| zero | 2 | 91.88 | 74.88 | 2 | 25.46 |
| zero | 4 | 91.88 | 74.88 | 4 | 33.43 |
| zero | 6 | 91.88 | 54.37 | 6 | 45.09 |
| zero | 8 | 30.03 | 5.96 | 8 | 52.23 |
| zero | 10 | 8.19 | 2.10 | 10 | 61.98 |
| zero | 12 | 6.09 | 1.46 | 12 | 71.31 |
| zero | 20 | 6.09 | 1.24 | 14 | 85.48 |
| zero | 300 | 6.09 | 1.24 | 14 | 80.89 |

**Table 19.** **Provisional.** Cheapest rule certified on each bar per rung,
with its fit-state count, and the parent lane's rule re-certified on the same
held-out states. Jobs bet101 = 3780164 (NVIDIA A100 80GB PCIe, commit ace8936e42b3…); bet201 = 3780165 (NVIDIA A100 80GB PCIe, commit ace8936e42b3…). Draw replication $3783811$
pending.

<!-- table: T09b_eq_certification -->
| $q$ | cheapest primary-certified rule | cheapest tight-certified rule | best parent-lane rule (re-certified): primary? |
|---|---|---|---|
| 0 | this_job/std, $m{=}512$, 64 states, $\rho_{\max}{=}0.0641$ | this_job/rhow64, $m{=}622$, 64 states, $\rho_{\max}{=}0.0491$ | $m{=}1024$, $\rho_{\max}{=}0.0153$, yes |
| 16 | this_job/std, $m{=}512$, 64 states, $\rho_{\max}{=}0.0879$ | this_job/std, $m{=}1024$, 64 states, $\rho_{\max}{=}0.0118$ | $m{=}2048$, $\rho_{\max}{=}0.0452$, yes |
| 32 | qrg304/reachable, $m{=}1024$, 42 states, $\rho_{\max}{=}0.0533$ | qrg304/reachable, $m{=}1024$, 42 states, $\rho_{\max}{=}0.0533$ | $m{=}1024$, $\rho_{\max}{=}0.0533$, yes |
| 64 | qrg304/reachable, $m{=}1024$, 25 states, $\rho_{\max}{=}0.0531$ | qrg304/reachable, $m{=}1024$, 25 states, $\rho_{\max}{=}0.0531$ | $m{=}1024$, $\rho_{\max}{=}0.0531$, yes |
| 128 | this_job/fs64, $m{=}2048$, 64 states, $\rho_{\max}{=}0.0669$ | this_job/rhow64, $m{=}2048$, 64 states, $\rho_{\max}{=}0.0472$ | $m{=}2048$, $\rho_{\max}{=}0.1908$, no |
| 256 | this_job/fs64, $m{=}2048$, 64 states, $\rho_{\max}{=}0.1074$ | — | $m{=}2048$, $\rho_{\max}{=}0.1678$, no |

**Table 20.** **Provisional.** Every quadrature rule: NNLS fit residual beside
the held-out $\rho$, so the anti-correlation stays visible. Bars: primary
$\rho_{\max}\le0.116$, tight $\rho_{\max}\le0.06$.

<!-- table: T09c_eq_rules_full -->
| $q$ | fit arm | $m$ | population | NNLS rel. fit | $\rho_{\max}$ | $\rho_{95}$ | primary | job |
|---|---|---|---|---|---|---|---|---|
| 0 | reachable | 1024 | qrg304:reachable | 4.58e-05 | 0.0153 | 0.0088 | yes | `3780164` |
| 0 | reachable | 1024 | qrg304:reachable | 4.58e-05 | 0.0153 | 0.0088 | yes | `3780165` |
| 0 | reachable | 1521 | qrg304:reachable | 7.83e-06 | 0.0261 | 0.0083 | yes | `3780164` |
| 0 | reachable | 1521 | qrg304:reachable | 7.83e-06 | 0.0261 | 0.0083 | yes | `3780165` |
| 0 | static | 256 | qrg304:static | 6.85e-03 | 0.2407 | 0.2073 | no | `3780164` |
| 0 | static | 256 | qrg304:static | 6.85e-03 | 0.2407 | 0.2073 | no | `3780165` |
| 0 | fs64 | 1024 | this_job:reachable | 4.70e-05 | 0.0424 | 0.0154 | yes | `3780165` |
| 0 | fs64 | 1546 | this_job:reachable | 8.92e-06 | 0.0177 | 0.0091 | yes | `3780165` |
| 0 | rhow64 | 609 | this_job:reachable | 2.24e-04 | 0.0686 | 0.0550 | yes | `3780165` |
| 0 | rhow64 | 622 | this_job:reachable | 2.24e-04 | 0.0491 | 0.0328 | yes | `3780165` |
| 0 | std | 256 | this_job:reachable | 4.03e-03 | 0.4018 | 0.3674 | no | `3780165` |
| 0 | std | 512 | this_job:reachable | 4.07e-04 | 0.0641 | 0.0545 | yes | `3780165` |
| 0 | std | 1024 | this_job:reachable | 4.35e-05 | 0.0255 | 0.0087 | yes | `3780165` |
| 0 | std | 1526 | this_job:reachable | 9.17e-06 | 0.0135 | 0.0032 | yes | `3780165` |
| 0 | std | 1528 | this_job:reachable | 8.90e-06 | 0.0124 | 0.0038 | yes | `3780165` |
| 0 | std | 1530 | this_job:reachable | 9.07e-06 | 0.0128 | 0.0040 | yes | `3780165` |
| 16 | reachable | 1024 | qrg304:reachable | 1.90e-04 | 0.0935 | 0.0273 | yes | `3780164` |
| 16 | reachable | 1024 | qrg304:reachable | 1.90e-04 | 0.0935 | 0.0273 | yes | `3780165` |
| 16 | reachable | 2048 | qrg304:reachable | 1.92e-05 | 0.0452 | 0.0041 | yes | `3780164` |
| 16 | reachable | 2048 | qrg304:reachable | 1.92e-05 | 0.0452 | 0.0041 | yes | `3780165` |
| 16 | static | 512 | qrg304:static | 1.54e-03 | 0.1279 | 0.0523 | no | `3780164` |
| 16 | static | 512 | qrg304:static | 1.54e-03 | 0.1279 | 0.0523 | no | `3780165` |
| 16 | fs64 | 1024 | this_job:reachable | 1.79e-04 | 0.0751 | 0.0232 | yes | `3780165` |
| 16 | fs64 | 2048 | this_job:reachable | 1.59e-05 | 0.0148 | 0.0043 | yes | `3780165` |
| 16 | rhow64 | 946 | this_job:reachable | 2.30e-04 | 0.0618 | 0.0174 | yes | `3780165` |
| 16 | rhow64 | 972 | this_job:reachable | 2.33e-04 | 0.0603 | 0.0098 | yes | `3780165` |
| 16 | std | 256 | this_job:reachable | 3.09e-02 | 0.4133 | 0.3926 | no | `3780165` |
| 16 | std | 512 | this_job:reachable | 1.99e-03 | 0.0879 | 0.0723 | yes | `3780165` |
| 16 | std | 1024 | this_job:reachable | 2.42e-04 | 0.0118 | 0.0108 | yes | `3780165` |
| 16 | std | 2048 | this_job:reachable | 2.31e-05 | 0.0070 | 0.0031 | yes | `3780165` |
| 16 | std | 2556 | this_job:reachable | 8.89e-06 | 0.0021 | 0.0015 | yes | `3780165` |
| 16 | std | 2581 | this_job:reachable | 8.67e-06 | 0.0020 | 0.0014 | yes | `3780165` |
| 32 | reachable | 1024 | qrg304:reachable | 2.89e-04 | 0.0533 | 0.0226 | yes | `3780164` |
| 32 | reachable | 1024 | qrg304:reachable | 2.89e-04 | 0.0533 | 0.0226 | yes | `3780165` |
| 32 | reachable | 2048 | qrg304:reachable | 2.66e-05 | 0.0668 | 0.0120 | yes | `3780164` |
| 32 | reachable | 2048 | qrg304:reachable | 2.66e-05 | 0.0668 | 0.0120 | yes | `3780165` |
| 32 | static | 768 | qrg304:static | 6.27e-04 | 0.1482 | 0.0214 | no | `3780164` |
| 32 | static | 768 | qrg304:static | 6.27e-04 | 0.1482 | 0.0214 | no | `3780165` |
| 32 | fs64 | 1024 | this_job:reachable | 3.80e-04 | 0.0696 | 0.0324 | yes | `3780165` |
| 32 | fs64 | 2048 | this_job:reachable | 4.21e-05 | 0.0417 | 0.0036 | yes | `3780165` |
| 32 | rhow64 | 1024 | this_job:reachable | 3.50e-04 | 0.0374 | 0.0308 | yes | `3780165` |
| 32 | rhow64 | 1208 | this_job:reachable | 2.40e-04 | 0.0325 | 0.0280 | yes | `3780165` |
| 32 | std | 256 | this_job:reachable | 8.60e-02 | 0.6639 | 0.6101 | no | `3780165` |
| 32 | std | 512 | this_job:reachable | 4.14e-03 | 0.2339 | 0.1944 | no | `3780165` |
| 32 | std | 1024 | this_job:reachable | 3.40e-04 | 0.0481 | 0.0329 | yes | `3780165` |
| 32 | std | 2048 | this_job:reachable | 3.23e-05 | 0.0073 | 0.0049 | yes | `3780165` |
| 32 | std | 2628 | this_job:reachable | 1.23e-05 | 0.0058 | 0.0029 | yes | `3780165` |
| 32 | std | 2675 | this_job:reachable | 1.24e-05 | 0.0062 | 0.0021 | yes | `3780165` |
| 64 | reachable | 1024 | qrg304:reachable | 7.12e-04 | 0.0531 | 0.0230 | yes | `3780164` |
| 64 | reachable | 1024 | qrg304:reachable | 7.12e-04 | 0.0531 | 0.0230 | yes | `3780165` |
| 64 | reachable | 2048 | qrg304:reachable | 7.54e-05 | 0.1120 | 0.0172 | yes | `3780164` |
| 64 | reachable | 2048 | qrg304:reachable | 7.54e-05 | 0.1120 | 0.0172 | yes | `3780165` |
| 64 | static | 1280 | qrg304:static | 2.43e-04 | 0.1439 | 0.0235 | no | `3780164` |
| 64 | static | 1280 | qrg304:static | 2.43e-04 | 0.1439 | 0.0235 | no | `3780165` |
| 64 | fs64 | 1024 | this_job:reachable | 9.19e-04 | 0.1303 | 0.0332 | no | `3780165` |
| 64 | fs64 | 2048 | this_job:reachable | 1.14e-04 | 0.1248 | 0.0032 | no | `3780165` |
| 64 | rhow64 | 1024 | this_job:reachable | 8.60e-04 | 0.1039 | 0.0364 | yes | `3780165` |
| 64 | rhow64 | 1563 | this_job:reachable | 2.48e-04 | 0.0518 | 0.0238 | yes | `3780165` |
| 64 | std | 256 | this_job:reachable | 1.84e-01 | 0.7670 | 0.7632 | no | `3780165` |
| 64 | std | 512 | this_job:reachable | 1.28e-02 | 0.3106 | 0.2780 | no | `3780165` |
| 64 | std | 1024 | this_job:reachable | 6.97e-04 | 0.1220 | 0.0262 | no | `3780165` |
| 64 | std | 2048 | this_job:reachable | 6.75e-05 | 0.0583 | 0.0074 | yes | `3780165` |
| 64 | std | 3054 | this_job:reachable | 1.18e-05 | 0.0213 | 0.0022 | yes | `3780165` |
| 64 | std | 3128 | this_job:reachable | 1.11e-05 | 0.0129 | 0.0023 | yes | `3780165` |
| 128 | reachable | 1024 | qrg304:reachable | 1.62e-03 | 0.2786 | 0.2246 | no | `3780164` |
| 128 | reachable | 1024 | qrg304:reachable | 1.62e-03 | 0.2786 | 0.2246 | no | `3780165` |
| 128 | reachable | 2048 | qrg304:reachable | 1.04e-04 | 0.1908 | 0.0518 | no | `3780164` |
| 128 | reachable | 2048 | qrg304:reachable | 1.04e-04 | 0.1908 | 0.0518 | no | `3780165` |
| 128 | static | 2048 | qrg304:static | 7.16e-05 | 0.2194 | 0.0677 | no | `3780164` |
| 128 | static | 2048 | qrg304:static | 7.16e-05 | 0.2194 | 0.0677 | no | `3780165` |
| 128 | fs64 | 2048 | this_job:reachable | 3.52e-04 | 0.0669 | 0.0214 | yes | `3780164` |
| 128 | fs64 | 2560 | this_job:reachable | 1.82e-04 | 0.0292 | 0.0111 | yes | `3780164` |
| 128 | rhow64 | 2048 | this_job:reachable | 3.54e-04 | 0.0472 | 0.0176 | yes | `3780164` |
| 128 | rhow64 | 2319 | this_job:reachable | 2.52e-04 | 0.0277 | 0.0102 | yes | `3780164` |
| 128 | std | 1024 | this_job:reachable | 1.49e-03 | 0.2793 | 0.1615 | no | `3780164` |
| 128 | std | 2048 | this_job:reachable | 7.98e-05 | 0.2373 | 0.0520 | no | `3780164` |
| 128 | std | 2560 | this_job:reachable | 3.14e-05 | 0.2040 | 0.0308 | no | `3780164` |
| 128 | std | 3055 | this_job:reachable | 1.40e-05 | 0.1193 | 0.0197 | no | `3780164` |
| 128 | std | 3088 | this_job:reachable | 1.35e-05 | 0.1091 | 0.0183 | yes | `3780164` |
| 128 | std | 3128 | this_job:reachable | 1.33e-05 | 0.1487 | 0.0161 | no | `3780164` |
| 256 | reachable | 1024 | qrg304:reachable | 4.01e-02 | 0.5731 | 0.5370 | no | `3780164` |
| 256 | reachable | 1024 | qrg304:reachable | 4.01e-02 | 0.5731 | 0.5370 | no | `3780165` |
| 256 | reachable | 2048 | qrg304:reachable | 3.08e-04 | 0.1678 | 0.0957 | no | `3780164` |
| 256 | reachable | 2048 | qrg304:reachable | 3.08e-04 | 0.1678 | 0.0957 | no | `3780165` |
| 256 | static | 2048 | qrg304:static | 1.49e-04 | 0.4621 | 0.4275 | no | `3780164` |
| 256 | static | 2048 | qrg304:static | 1.49e-04 | 0.4621 | 0.4275 | no | `3780165` |
| 256 | fs64 | 2048 | this_job:reachable | 2.41e-03 | 0.1074 | 0.0878 | yes | `3780164` |
| 256 | fs64 | 2560 | this_job:reachable | 9.35e-04 | 0.0647 | 0.0314 | yes | `3780164` |
| 256 | rhow64 | 2048 | this_job:reachable | 1.44e-03 | 0.1299 | 0.0836 | no | `3780164` |
| 256 | rhow64 | 2560 | this_job:reachable | 6.23e-04 | 0.1018 | 0.0380 | yes | `3780164` |
| 256 | rhow64 | 3072 | this_job:reachable | 3.37e-04 | 0.0963 | 0.0229 | yes | `3780164` |
| 256 | std | 1024 | this_job:reachable | 1.61e-02 | 0.4977 | 0.4450 | no | `3780164` |
| 256 | std | 2048 | this_job:reachable | 1.75e-04 | 0.2958 | 0.1822 | no | `3780164` |
| 256 | std | 2560 | this_job:reachable | 5.84e-05 | 0.1956 | 0.1052 | no | `3780164` |
| 256 | std | 3072 | this_job:reachable | 2.32e-05 | 0.1774 | 0.0722 | no | `3780164` |
| 256 | std | 3370 | this_job:reachable | 1.45e-05 | 0.1308 | 0.0685 | no | `3780164` |
| 256 | std | 3410 | this_job:reachable | 1.43e-05 | 0.1761 | 0.0836 | no | `3780164` |

**Table 21.** Frozen-checkpoint mesh ladder (Burgers job 3711388 on
NVIDIA A100-PCIE-40GB, checkpoint 18f0266ae6f0…; Poisson job
3711389 on NVIDIA A100 80GB PCIe, checkpoint
a128e7635c31…). “efficient FOM” is the cheapest same-job full-order
arm meeting the 5 % target on that mesh.

<!-- table: T10_mesh_ladder -->
| PDE | intervals | unknowns | ROM cached ms | ROM device ms | ROM host ms | ROM worst vs ref % | ROM vs same-grid FOM % | efficient FOM | FOM ms | FOM/ROM | ROM meets 5% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Burgers | 64 | 3,969 | 44.2 | 48.5 | 50.4 | 10.8570 | 5.5018 | `fom_same_nt1e-2` | 12.9 | 0.265 | no |
| Burgers | 128 | 16,129 | 44.7 | 46.5 | 48.1 | 6.5431 | 2.8939 | `fom_same_nt1e-2` | 14.9 | 0.320 | no |
| Burgers | 256 | 65,025 | 44.7 | 48.2 | 50.6 | 4.5546 | 2.5629 | `fom_same_nt1e-2` | 15.7 | 0.325 | yes |
| Burgers | 512 | 261,121 | 45.1 | 47.4 | 53.7 | 3.7168 | 3.2836 | `fom_same_nt1e-2` | 23.5 | 0.495 | yes |
| Burgers | 1024 | 1,046,529 | 43.6 | 50.1 | 76.8 | 3.8847 | 3.8562 | `fom_coarse4_nt1e-4` | 22.2 | 0.443 | yes |
| Poisson | 64 | 3,969 | 2.1 | 2.3 | 3.2 | 6.1119 | 6.1282 | `dst` | 0.1 | 0.063 | no |
| Poisson | 128 | 16,129 | 2.1 | 2.2 | 3.2 | 6.1107 | 6.1149 | `dst` | 0.1 | 0.065 | no |
| Poisson | 256 | 65,025 | 2.1 | 2.2 | 3.3 | 6.1106 | 6.1116 | `dst` | 0.2 | 0.067 | no |
| Poisson | 512 | 261,121 | 1.9 | 2.3 | 3.8 | 6.1106 | 6.1108 | `dst` | 0.2 | 0.079 | no |
| Poisson | 1024 | 1,046,529 | 2.0 | 3.0 | 6.6 | 6.1106 | 6.1106 | `dst` | 0.3 | 0.112 | no |

**Table 22.** Reflective 2D wave, one job per mesh ($64^2$: job 3780447 (NVIDIA A100 80GB PCIe, commit 0bb3cc86); $256^2$: job 3783805 (NVIDIA A100-PCIE-40GB, commit 2655bb01); $1024^2$: job 3780450 (NVIDIA A100 80GB PCIe, commit 0bb3cc86)); costs
comparable only within a mesh. The top rung is the learned bank evolved by
exact modal propagation with no head.

<!-- table: T11a_waves -->
| mesh | arm | $q$ / $k^\prime$ | worst energy-state % | $t{=}0$ % | GPU ms | complete ms |
|---|---|---|---|---|---|---|
| $64^2$ | `head_q0` | 0 | 11.2055 | 5.5314 | 184.535 | 185.575 |
| $64^2$ | `trained_nested40` | 8 | 6.8128 | 4.7478 | 200.232 | 201.072 |
| $64^2$ | `nested_q8` | 8 | 5.9379 | 3.7715 | 199.289 | 200.375 |
| $64^2$ | `nested_q16` | 16 | 5.0053 | 2.5994 | 1977.402 | 1978.383 |
| $64^2$ | `nested_q32` | 32 | 4.9161 | 2.0345 | 2500.060 | 2501.038 |
| $64^2$ | `linear_bank64` | 64 ($=R$) | 4.9323 | 2.0345 | 1.168 | 1.986 |
| $64^2$ | `pod_k40` | $k'{=}40$ | 4.8825 | 4.2685 | 1.190 | 1.989 |
| $64^2$ | `pod_k64` | $k'{=}64$ | 1.4893 | 1.4709 | 1.202 | 2.018 |
| $64^2$ | `pod_k128` | $k'{=}128$ | 0.8668 | 0.4932 | 1.252 | 2.052 |
| $64^2$ | `dst` | — | 0.0000 | 0.0000 | 3.391 | 4.286 |
| $64^2$ | `rk4_fom` | — | 0.0612 | 0.0000 | 5.511 | 6.865 |
| $64^2$ | `cg_1e-06` | — | 0.5745 | 0.0000 | 109.836 | 111.350 |
| $256^2$ | `head_q0` | 0 | 11.3314 | 5.5585 | 183.516 | 198.619 |
| $256^2$ | `trained_nested40` | 8 | 6.8764 | 4.7840 | 197.587 | 213.401 |
| $256^2$ | `nested_q8` | 8 | 6.0627 | 3.8045 | 197.389 | 212.596 |
| $256^2$ | `nested_q16` | 16 | 5.1661 | 2.6270 | 1989.463 | 2004.700 |
| $256^2$ | `nested_q32` | 32 | 5.1038 | 2.0731 | 2520.177 | 2536.035 |
| $256^2$ | `linear_bank64` | 64 ($=R$) | 5.1208 | 2.0731 | 1.428 | 16.659 |
| $256^2$ | `pod_k40` | $k'{=}40$ | 4.9963 | 4.2844 | 1.365 | 16.639 |
| $256^2$ | `pod_k64` | $k'{=}64$ | 1.5018 | 1.4817 | 1.399 | 16.657 |
| $256^2$ | `pod_k128` | $k'{=}128$ | 0.9873 | 0.5122 | 1.482 | 16.773 |
| $256^2$ | `dst` | — | 0.0000 | 0.0000 | 4.356 | 19.703 |
| $256^2$ | `rk4_fom` | — | 0.0016 | 0.0000 | 24.346 | 40.083 |
| $256^2$ | `cg_1e-06` | — | 0.6018 | 0.0000 | 142.511 | 158.115 |
| $1024^2$ | `head_q0` | 0 | 11.3388 | 5.5602 | 192.462 | 451.994 |
| $1024^2$ | `trained_nested40` | 8 | 6.8789 | 4.7863 | 207.832 | 467.228 |
| $1024^2$ | `nested_q8` | 8 | 6.0692 | 3.8066 | 206.632 | 468.808 |
| $1024^2$ | `nested_q16` | 16 | 5.1763 | 2.6287 | 2003.425 | 2264.489 |
| $1024^2$ | `nested_q32` | 32 | 5.1164 | 2.0755 | 2529.445 | 2789.801 |
| $1024^2$ | `linear_bank64` | 64 ($=R$) | 5.1335 | 2.0755 | 4.562 | 265.785 |
| $1024^2$ | `pod_k40` | $k'{=}40$ | 5.0030 | 4.2854 | 4.183 | 263.883 |
| $1024^2$ | `pod_k64` | $k'{=}64$ | 1.5010 | 1.4824 | 4.232 | 264.860 |
| $1024^2$ | `pod_k128` | $k'{=}128$ | 0.9888 | 0.5135 | 5.879 | 265.409 |
| $1024^2$ | `dst` | — | 0.0000 | 0.0000 | 17.103 | 277.324 |
| $1024^2$ | `rk4_fom` | — | 0.0000 | 0.0000 | 958.007 | 1174.433 |
| $1024^2$ | `cg_1e-06` | — | 0.6079 | 0.0000 | 1043.368 | 1278.837 |

**Table 23.** Poisson head capacity on the frozen $R=512$ bank (job
3783883, NVIDIA A100-PCIE-40GB): width helps, depth alone does not, and
no arm reaches a $2\times$ ratio to the floor. The re-run of the primary
recipe reproduces it exactly.

<!-- table: T11c_head_capacity -->
| head | best-found dev. % | best-found / floor | solved $1024^2$ % | total ms |
|---|---|---|---|---|
| `K32_w128_L2` | 3.1212 | 4.18 | 3.1146 | 12.93 |
| `K32_w256_L2` | 2.2652 | 3.04 | 2.2600 | 12.84 |
| `K32_w128_L3` | 3.0625 | 4.11 | 3.0558 | 13.23 |
| `K32_w256_L3` | 2.3614 | 3.17 | 2.3571 | 13.39 |
| `K64_w128_L2` | 2.5625 | 3.44 | 2.5587 | 13.72 |
| `K64_w256_L3` | 2.0553 | 2.76 | 2.0549 | 14.12 |
| `K32_w128_L2_x3` | 2.6428 | 3.54 | 2.6360 | 12.97 |

**Table 24.** Per-rung mean and spread over three training seeds, monotone-seed
count (lane b-seeds).

<!-- table: T12_seeds -->
**[PENDING: pending: b-seeds]**

**Table 25.** Development against sealed worst error per rung, difficulty-normalised
(lane b-seeds).

<!-- table: T13_sealed -->
**[PENDING: pending: b-seeds sealed cohort]**

**Table 26.** The operators' own inference-time knob: each validation-selected
checkpoint evaluated at coarser grids and prolonged back (job 3787189,
NVIDIA A100 80GB PCIe), with the interpolation floor (a perfect operator's error at that
rung) and the pre-registered gates ($\ge1.5\times$ own speed at $\le2\times$
own error). Speedups are within one model's own curve, same job.

<!-- table: T14c_resolution -->
| operator | rung | worst evolved % | interp. floor % | device ms | speed vs 256 | error vs 256 | both gates |
|---|---|---|---|---|---|---|---|
| `fno-large` | 256 | 6.38 | 0.00 | 7.35 | — | — | ref. |
| `fno-large` | 128 | 7.30 | 0.23 | 3.46 | 2.12 | 1.14 | yes |
| `fno-large` | 64 | 9.82 | 0.92 | 3.44 | 2.14 | 1.54 | yes |
| `fno-large` | 32 | 13.85 | 4.02 | 3.33 | 2.21 | 2.17 | no |
| `tsol-refine` | 256 | 9.32 | 0.00 | 11.20 | — | — | ref. |
| `tsol-refine` | 128 | 15.29 | 0.23 | 8.15 | 1.37 | 1.64 | no |
| `tsol-refine` | 64 | 35.18 | 0.92 | 8.09 | 1.38 | 3.78 | no |
| `tsol-refine` | 32 | 75.13 | 4.02 | 8.04 | 1.39 | 8.06 | no |
| `unet-refine` | 256 | 7.52 | 0.00 | 5.90 | — | — | ref. |
| `unet-refine` | 128 | 38.13 | 0.23 | 2.35 | 2.51 | 5.07 | no |
| `unet-refine` | 64 | 60.18 | 0.92 | 2.31 | 2.55 | 8.01 | no |
| `unet-refine` | 32 | 67.65 | 4.02 | 2.30 | 2.56 | 9.00 | no |

**Table 27.** U-Net against FNO on the Poisson operator-screen dataset (physical
reference). The U-Net arms hit the epoch cap while still improving; the FNO
early-stopped.

<!-- table: T14b_operators_poisson -->
| arm | family | params | median % | worst % | job |
|---|---|---|---|---|---|
| `fno-large` | FNO | 17,876,673 | 2.03 | 29.37 | `3702464` |
| `fno-medium` | FNO | 5,779,729 | 2.35 | 28.28 | `3702464` |
| `fno-small` | FNO | 1,192,801 | 2.64 | 32.47 | `3702464` |
| `unet-large` | U-Net | 17,461,393 | 1.95 | 8.31 | `3780625` |
| `unet-medium` | U-Net | 7,763,041 | 1.68 | 6.02 | `3780625` |
| `unet-small` | U-Net | 4,368,073 | 2.40 | 12.80 | `3780625` |

**Table 28.** Speed at bit-level parity (jobs 3745655 (spd01), 3745656 (fine01), 3745913 (comp01)): fastest arm whose
fields agree with the incumbent to $10^{-12}$ with identical iteration counts
and exit reasons. The pre-registered $2\times$ target fails at every mesh.

<!-- table: T15_speed -->
| attempt | intervals | arm | incumbent ms | arm ms | GPU speedup | incl. host | parity | 2$\times$ target |
|---|---|---|---|---|---|---|---|---|
| spd01 | 256 | `L4` | 46.856 | 30.824 | 1.520x | 1.478x | 2.565e-13 | FAIL |
| fine01 | 512 | `L4` | 46.928 | 31.106 | 1.509x | 1.454x | 5.053e-13 | FAIL |
| fine01 | 1024 | `L4` | 49.283 | 33.544 | 1.469x | 1.310x | 6.951e-13 | FAIL |
| comp01 | 256 | `C1` | 46.627 | 30.723 | 1.518x | 1.482x | 2.562e-13 | FAIL |
| comp01 | 1024 | `C1` | 49.804 | 33.282 | 1.496x | 1.408x | 7.129e-13 | FAIL |

**Table 29.** Training study on the Burgers head (jobs 3745912 (training), 3749074 (evaluation)): data
density, objective, latent dimension and a joint bank arm, each evaluated
through the unchanged head-ablation machinery. $0$ of
$26$ arms pass the pre-registered success criterion. The
like-for-like retrain does not reproduce the incumbent, so the density verdict
is measured below baseline.

<!-- table: T16_training -->
| arm (EQ query) | $K$ | bank floor % | best-found % | solved all % | solved evolved % | GPU ms | conv. |
|---|---|---|---|---|---|---|---|
| incumbent (4608 traj., $K{=}16$) | 16 | 0.3918 | 2.5447 | 2.5629 | 1.9002 | 48.7 | yes |
| 128 traj., $K{=}16$ | 16 | 0.3918 | 12.6496 | 12.6496 | 7.2529 | 45.4 | yes |
| 512 traj. | 16 | 0.3918 | 6.8202 | 6.8204 | 3.7153 | 39.3 | yes |
| 2048 traj. | 16 | 0.3918 | 3.5570 | 3.5574 | 1.8752 | 46.6 | yes |
| 4608 traj. (like-for-like retrain) | 16 | 0.3918 | 3.9616 | 3.9710 | 1.9522 | 59.1 | yes |
| 128 traj., $K{=}32$ | 32 | 0.3918 | 8.8489 | 8.8496 | 6.2351 | 80.7 | yes |
| 2048 traj., $K{=}32$ | 32 | 0.3918 | 3.1132 | 3.1137 | 2.4882 | 54.5 | yes |
| 128 traj., weak-residual term | 16 | 0.3918 | 14.2164 | 14.2165 | 7.5935 | 43.9 | yes |
| 128 traj., trajectory term | 16 | 0.3918 | 12.8237 | 12.8238 | 7.8543 | 42.8 | yes |
| 128 traj., code-smoothness term | 16 | 0.3918 | 13.6497 | 13.6505 | 13.4135 | 36.8 | yes |
| selected: 2048 traj., $K{=}32$, weak term | 32 | 0.3918 | 2.8289 | 3.1275 | 3.1275 | 55.1 | yes |
| joint bank$+$head, $R{=}512$ | 32 | 2.8517 | 5.0383 | 5.0471 | 2.4399 | 46.3 | yes |

**Table 30.** L-shaped Poisson (job 3783786, NVIDIA A100 80GB PCIe): bank floors
by boundary factor and rank, and the head layer at $256^2$. The solve layer is
$\gen{pending: lshape solve jobs 3784662/3/4}$.

<!-- table: T18a_lshape_bank -->
| bank | floor, dev. $256^2$ % | floor, dev. $512^2$ % | floor, common $256^2$ % |
|---|---|---|---|
| `smooth_R256` | 1.5113 | 1.5057 | 3.1933 |
| `smooth_R512` | 0.7123 | 0.7093 | 1.9344 |
| `sdf_R256` | 1.5697 | 1.5626 | 3.2101 |
| `sdf_R512` | 0.7766 | 0.7698 | 1.6155 |
| `enrich_R512` | 0.6676 | 0.6650 | 1.7028 |
| `smooth_ff128s2_R512` | 2.5438 | 2.5388 | 3.8411 |

<!-- table: T18b_lshape_head -->
| head arm | best-found % | weak solve % (untimed) | best-found / bank floor |
|---|---|---|---|
| `head_enrich_R512_K16` | 3.8508 | 3.8510 | 5.77 |
| `head_sdf_R256_K16` | 3.5177 | 3.5181 | 2.24 |
| `head_sdf_R512_K16` | 3.8607 | 3.8608 | 4.97 |
| `head_sdf_R512_K32` | 3.1802 | 3.1814 | 4.09 |
| `head_smooth_R256_K16` | 3.4919 | 3.4928 | 2.31 |
| `head_smooth_R512_K16` | 3.5924 | 3.5931 | 5.04 |
| `head_smooth_ff128s2_R512_K16` | 6.7961 | 6.7970 | 2.67 |

## E Glossary

<!-- section sources: b-eqtop summary.json (bars) -->

Written for a reader who knows none of this project's vocabulary.

- **Bank, head, latent code ($G$, $h_\theta$, $z$; $R$, $k$)** — The fixed spatial functions the decoder combines (width $R$); the small network that maps $k$ latent numbers to their coefficients; the $k$ numbers solved for per query.
- **Correction rank $q$, rung, ladder, $C_q$** — The number of fixed extra bank directions the solver may add to the head's output; one value of $q$; the sequence of rungs; the matrix of directions, a nested prefix chosen offline. $q=0$ is the head alone, $q=R$ the linear model on the bank.
- **Test count $M$, weak tests, row scaling** — How many sine test functions the residual is averaged against; the functions; the diagonal weights on those averaged equations. All three change which solution is selected.
- **$m$, EQ, rule, NNLS, fit states, reachable states** — Quadrature node count; empirical quadrature, evaluating a sum over all nodes by a weighted sum over $m$ of them; one fitted instance; the non-negative least-squares fit; the states used to fit a rule; states the solver actually visits, as opposed to stored snapshots.
- **$\rho$, primary and tight bars, certified** — The rule's relative error on the projected advection term on held-out reachable states; $\rho_{\max}\le0.116$ and $\le0.06$; passing a bar. Never the NNLS fit residual.
- **Dense** — Evaluating the advection term at every interior node, exactly.
- **Evolved vs all-times metric, $t{=}0$ compression** — Worst error over output times $t>0$; over all output times including $t=0$; the error at $t=0$ alone, which is the decoder's reconstruction of the supplied initial field and bounds the all-times metric from below.
- **Same-grid, reference, physical error** — Error against the converged full-order solve on the same mesh (isolates the reduction); against a much finer solve (includes the mesh's own discretisation error); the cell's declared accuracy metric.
- **Three layers: bank floor, best-found, solved** — Error of the best linear projection onto the bank (what nothing at inference can beat); the smallest error any point of the augmented manifold attains, found by a multistart oracle (an upper bound on the true minimum); the error the deployed iteration returns.
- **Stationary, budget exit, early-stopped, converged, strict** — The solve met the normalised-gradient tolerance; it hit its iteration cap; any output produced under a cap or stall; the pre-registered completion rule that also accepts an attained initial fit whose residual is at round-off; the earlier rule that does not.
- **Non-dominated, frontier, error span, cost span** — A point nothing else beats on both axes; the set of such points; ratio of largest to smallest error among a ladder's non-dominated converged points; the same for cost. The tunability bar: monotone, at least three non-dominated points, at least $2\times$ on both spans, nothing early-stopped.
- **Same-allocation, same-job, cross-job** — Two measurements inside one cluster allocation on one GPU, which may be divided; two from different allocations, which may not.
- **Development cohort, held-out, sealed cohort** — Cases opened and used for diagnosis; cases never used for selection; cases to be opened once, at the end, with every choice frozen (unopened for every cell here).
- **Pre-registered** — Declared in a design document before the job ran; every bar, gate and falsification clause in this paper was.
- **POD-LSPG, $k'$, DST, CG, Newton** — The classical linear reduced model on a snapshot basis of rank $k'$ solved through the same weak objective; the direct discrete sine transform solve, exact for separable constant-coefficient operators on a rectangle; conjugate gradients; the full-order nonlinear iteration for Burgers.
- **FNO, U-Net, Transolver** — Three neural-operator families; each is one trained model giving one accuracy–cost point.
- **Checkpoint, frozen, incumbent** — Saved network weights; unchanged for every result; the one Burgers checkpoint every cell shares.
- **Provisional, pending** — A number that is real but rests on one draw, one seed or an unfinished replication; a placeholder for a run that has not landed.

