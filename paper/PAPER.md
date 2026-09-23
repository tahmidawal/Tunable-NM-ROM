# Tunable Non-linear Manifold ROMs for Elliptic, Parabolic, and Hyperbolic PDEs via Matrix-Free Petrov--Galerkin Projection

*Anonymous submission to ICLR 2027. Every number below is generated from run records by `gen_tables.py`; tables are inlined from `tables-md/` behind an HTML comment naming their id; **[PENDING: …]** marks a lane that has not landed.*

*Status for the reader (generated 2026-09-22 21:28; this block is removed before submission).*
*Populated tables (95): T00, T01, T01b, T02, T02b, T02c, T03, T03b, T03c, T03m, T03mb, T03mc, T04, T04b, T04m, T05, T05b, T05c, T05m, T06a, T06b, T07, T08, T08b, T09, T09b, T09c, T09c, T09d, T10, T11a, T11b, T11c, T11d, T11e, T11f, T11g, T11h, T11i, T12, T12b, T13, T13b, T14, T14b, T14c, T14d, T15, T16, T17, T18a, T18b, T18c, T18d, T18m, T19, T20, T20b, T21, TC, TC, TC, TC, TC, TC, TC, TC, TC, TH, TH, TH, TH, TH, TH, TH, TH, TH, TH, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR, TR. Populated does not mean final: the three-dimensional appendix is provisional development evidence.*
*Pending cells: none. Active experiment status is recorded in the canonical LAB-LOG.md; this manuscript uses a frozen evidence snapshot.*
*The sealed cohort (T13, b-seeds job 3804465) is the headline for the scheduled ladder; T12 is the development-cohort seed table; the two top EQ rungs are single-draw rules, never certified.*
*Open decisions for the user: (1) the headline Burgers metric, worst over evolved times or worst over all times, both printed everywhere, and now decisive for §5.1 at 1024², where reduced rungs are non-dominated on the evolved metric only because the t=0 compression bounds all-times; (2) sign-off on the abstract's new opening two sentences (resolution-knob framing), which are provisionally accepted and unchanged in this pass.*
*Abstract: the readability pass's Abstract A (<= 250 words) with first-use glosses; the previous 397-word abstract and Abstract B (~200 words, one line shorter) are in ABSTRACT-2026-09-17.md for the user to pick.*
*Changed in this pass: b-panel closed (bpn301 replaces bpn101 at 256², bpn203 adds 1024²); L-shape closed at 512² and now in the abstract; b-qxm pin at 4b9723e8 dropped after the lane committed its regeneration (no number in §5.2 moved); three seeds landed on the development cohort; Figure 2 moved into §3.2 beside the equation it draws; the sealed cohort landed (be9415ab): sealed values replace the development q=0 incumbent value as the headline, the wrong-branch cold start at q=0 is a stated failure mode.*

## Abstract

Neural operators such as Fourier Neural Operators and DeepONets
typically deliver one (accuracy, speed) operating point per trained
model; moving it generally requires retraining. We present a
nonlinear-manifold reduced-order model (NM-ROM) for Poisson, heat and
viscous Burgers equations that exposes a deployment-time accuracy–speed
tradeoff from one trained decoder, set by
the correction rank $q$ (accuracy) and the solver tolerance, budget and
empirical-quadrature rule (cost). The framework combines a frozen coordinate-network spatial
bank with a nonlinear head and nested correction directions; exact
Dirichlet enforcement in the bank; least-squares
Petrov–Galerkin projection of the discrete residual onto fixed weak
tests, with every linear operator preassembled; matrix-free JAX
evaluation; and NNLS empirical quadrature for Burgers advection,
validated on held-out reached states but not confirmed on independent re-draws at $4096^2$. Across Poisson and heat in two and
three dimensions, and viscous Burgers in two, at meshes up to $4096^2$,
the frozen model keeps its accuracy as the two-dimensional mesh is
refined, so its speedup over the named iterative solvers reaches $\nHiresPoissonAccSFortyNinetySix\times$ at
$\nHeadPoissonAccErr %$ error against conjugate gradients on Poisson,
$\nHeatWideAccS\times$ at $\nHeatWideAccErr %$ against Crank–Nicolson
CG on a sealed held-out heat cohort, and $\nBurgDevAccSFortyNinetySix\times$ at
$\nBurgDevAccErrFortyNinetySix %$ on Burgers against Newton–BiCGStab
($\nBurgHoldAccSFortyNinetySix\times$ at
$\nBurgHoldAccErrFortyNinetySix %$ on held-out cases); at matched latent
dimension its fast setting is more accurate than a reproduced
shallow-masked-autoencoder NM-ROM and POD-LSPG. The named solvers remain
more accurate; three-dimensional heat meets its accuracy target but
beats CN–CG only with a batched fit at $128^3$, and three-dimensional Burgers, Navier–Stokes and full-state waves miss
their targets.

## 1 Introduction

<!-- section sources: b-qxm analysis.json; b-eqtop summary.json; mesh-ladder json; b-panel summary.json -->

Neural operators for partial differential equations (PDEs)
(Li et al., 2021; Lu et al., 2021; Kovachki et al., 2023) typically
deliver one (accuracy, wall-clock) operating point per trained model,
with no deployment-time knob to trade compute for accuracy; moving the
point generally requires retraining. Full-order methods (FOMs) solved
with iterative solvers such as conjugate gradients
(HestenesStiefel1952CG, ?) expose this knob through their tolerance,
but their cost grows with the mesh. Reduced-order models (ROMs) approach
the problem from the opposite direction: linear projection-based ROMs
(Benner et al., 2015; Sirovich, 1987) can require many basis
functions for moving fronts (Cohen & DeVore, 2015), and
nonlinear-manifold ROMs (NM-ROMs) learn a more compact representation
(Lee & Carlberg, 2020; Kim et al., 2022), but their online nonlinear
solve can offset the benefit of fewer unknowns. The key open question is thus:
*can one trained NM-ROM give a deployment-time accuracy–cost
choice, keep its accuracy as the mesh is refined, and be faster than
named iterative full-order solvers while being more accurate than prior
nonlinear-manifold ROMs?*

This paper answers that question largely affirmatively for
two-dimensional Poisson, heat and viscous Burgers. We present an NM-ROM
whose distinguishing property is a deployment-time accuracy–cost choice
from a single trained model. It represents the solution by a learned
spatial bank multiplied by a nonlinear map of latent variables,
augmented with nested, precomputed correction directions: activating
more directions enlarges the trial space without changing the learned
weights. The correction rank is the accuracy control; stopping
tolerances and empirical quadrature set the cost of the solve. All
online coefficients come from the weak PDE residual, without access to
the reference solution. The accurate error
(Table 1)
stays near its coarse-mesh level up to $4096^2$ while the speedup over
the named solver grows, and on held-out Burgers cases the model is far
more accurate than a reproduced NM-ROM and POD-LSPG
(Table 2). Three-dimensional heat meets its
accuracy target but is faster than CN–CG only with a batched fit at
$128^3$; it is not yet so for three-dimensional Burgers, Navier–Stokes
or the full wave state (Table 3).

**Contributions.**

1. **A tunable NM-ROM with a deployment-time accuracy/speed
tradeoff from a single trained model.**
Nested correction directions provide discrete deployment settings that
change approximation capacity without retraining. We isolate correction
rank at fixed residual test count and separately evaluate a prescribed
test-count schedule across seeds and held-out cases.
2. **Architectural choices that make this tradeoff achievable.**
A learned spatial bank, a nonlinear head with a linear skip, and nested
corrections support compact latent solves, boundary enforcement and
node-local evaluation.
3. **A matrix-free JAX implementation that makes the framework
practical.** With precomputed linear operators, matrix-free
differentiation and validated empirical quadrature, query time grows
more slowly with the mesh than the named solvers': the accurate setting reaches
$\nHeadPoissonAccErr %$ on Poisson at
$\nHiresPoissonAccSFortyNinetySix\times$ the speed of CG, and
$\nBurgDevAccErrFortyNinetySix %$ on Burgers at
$\nBurgDevAccSFortyNinetySix\times$ that of Newton–BiCGStab, at $4096^2$
in the same allocation.

## 2 Related Work

<!-- section sources: none (prose only) -->

**Linear and non-linear manifold ROMs.**
Linear ROMs approximate solutions in a fixed subspace
(Benner et al., 2015; Sirovich, 1987); moving fronts can
require a large basis (Cohen & DeVore, 2015; GreifUrban2019, ?).
Nonlinear-manifold ROMs combine learned representations with residual
projection (Lee & Carlberg, 2020; Kim et al., 2022), including quadratic
manifolds (Geelen et al., 2022; Barnett & Farhat, 2022), neural-field
representations (KimWenLeeChoiCNFROM2024, ?) and adaptive bases
(Peherstorfer & Willcox, 2015; Carlberg, 2015).
Least-squares Petrov–Galerkin projection
(Carlberg et al., 2011) and preassembly of reduced operators
( Stef anescu & Sandu, 2014; Weder et al., 2024) are established
components. We differ from these by a coordinate-network bank with an
exact boundary factor, a head with a linear skip, and nested corrections
selected after training that give deployment-time settings from one
trained model, as runtime-tunable networks do
(YuSlimmable2019, ?; Cai2020OnceForAll, ?).

**Neural operators and hyper-reduction.**
FNO (Li et al., 2021), DeepONet (Lu et al., 2021), U-Net
(Ronneberger et al., 2015; Takamoto et al., 2022) and Transolver
(WuTransolver2024, ?) learn solution maps. Their evaluation resolution
or inference procedure can also change accuracy and cost; our mechanism
instead changes the trial space used to solve the PDE. Empirical
quadrature (Hern'andez et al., 2017; Yano & Patera, 2019) reduces nonlinear
residual evaluation through non-negative weights. We retain that
construction, testing its error on states reached by the solver and
through independent re-draws. The main iterative linear-system
comparator is conjugate gradient (HestenesStiefel1952CG, ?).

## 3 Methodology

<!-- section sources: none (prose only) -->

\edef\restoredisplays{
\noexpand\abovedisplayskip=\the\abovedisplayskip
\noexpand\belowdisplayskip=\the\belowdisplayskip
\noexpand\abovedisplayshortskip=\the\abovedisplayshortskip
\noexpand\belowdisplayshortskip=\the\belowdisplayshortskip}
\setlength{\abovedisplayskip}{1pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{1pt plus 1pt minus 1pt}
\setlength{\abovedisplayshortskip}{1pt plus 1pt minus 1pt}
\setlength{\belowdisplayshortskip}{1pt plus 1pt minus 1pt}

To answer the question of whether a single trained model can expose a
deployment-time accuracy/speed tradeoff, we construct an NM-ROM by
turning a learned manifold into a practical PDE solver through three
components. First, we define a trial manifold whose decoder enforces
Dirichlet conditions exactly and carries nested correction directions
(§3.1). Second, we project the full-order
residual onto that manifold by least-squares Petrov–Galerkin
projection onto fixed weak tests, leaving a small least-squares problem
in the latent code and the correction coefficients
(§3.2). Third, we preassemble every linear term
exactly, and apply Empirical Quadrature to the Burgers advection
(§3.3). The architecture in §3.4
is then designed to support these requirements in practice.
Throughout, $u \in \mathbb{R}^{n}$ is the full-order state on a uniform grid
of $N$ intervals per axis and $A \in \mathbb{R}^{n \times n}$ the discrete
negative Laplacian. Four sizes recur: $R$ spatial functions in the bank,
$k$ latent coordinates, $q$ correction directions and $M$ weak tests,
with $k+q\ll M\ll n$ and $m$ the number of quadrature nodes. The
formulas are the scalar Dirichlet case. Appendix A
derives the remaining equations, and Figure 1 is the
decoder.

### 3.1 Trial Manifold with Exact Dirichlet Enforcement

<!-- section sources: none (prose only) -->

A solution is a fixed set of spatial functions with coefficients that a
small network produces. The bank $G\in\mathbb{R}^{n\times R}$ holds
those $R$ functions, one column each, sampled at the mesh nodes and
then frozen. The head $h_\theta:\mathbb{R}^{k}\to\mathbb{R}^{R}$ turns $k$ latent
coordinates $z$ into the $R$ coefficients. The decoder is their
product, $\mathcal{D}(z)=Gh_\theta(z)$, with
$k\le R\ll n$. There is no encoder. A query chooses $z$; it does
not encode a field.
Each column of the bank is a random-Fourier-feature coordinate network
$g_\phi$ (Tancik et al., 2020), read at the node coordinates
and multiplied by a factor that vanishes on the boundary. On the square,

$$
G_{x,:} = \mu(x)\,g_\phi(x)^{\top},
  \qquad
  \mu(x) = 16\,x_1(1-x_1)\,x_2(1-x_2).
$$

<!-- equation (1) -->

Because $\mu$ is zero on the boundary $\Gamma$, every column of
$G$ is zero there, so every trial state and every derivative
$\partial u/\partial z$ vanishes on $\Gamma$ at any resolution.
Homogeneous Dirichlet data are therefore enforced by the decoder, and
no penalty term is required. The cube uses a product of the same
kind (§3.4). An L-shaped domain is included only as an
ill-conditioned Poisson check.

**Nested correction directions.**

The head reaches only the states its $k$ coordinates produce. After
training we widen that set with $q$ fixed directions in coefficient
space, the columns of $C_q\in\mathbb{R}^{R\times q}$. The trial state is

$$
u(z,y) \;=\; G\big(h_\theta(z) + C_q\,y\big),
  \qquad z\in\mathbb{R}^{k},\; y\in\mathbb{R}^{q},\; 0\le q\le R .
$$

<!-- equation (2) -->

The weights $y$ are not trained; the solver chooses them at the query.
The columns are computed once from training fields. For each field,
$\eta\in\mathbb{R}^{R}$ is the bank coefficients that represent it and
$z^{\star}$ is the code that brings $h_\theta(z^{\star})$ closest to
$\eta$. The miss is $\eta-h_\theta(z^{\star})$, and the columns of
$C_q$ are the leading principal components of these misses
(Appendix A). They are nested: $C_q$ is the first $q$
columns of any larger matrix, and raising $q$ does not retrain a
weight. At $q=0$ the solve uses the head alone, in $k$ unknowns. At
$q=R$ the head drops out and the trial states fill the span of the
bank, whose projection error is a floor under every $q$.

### 3.2 Projecting the PDE onto the Manifold

<!-- section sources: none (prose only) -->

Each PDE gives a discrete residual $r(u)\in\mathbb{R}^{n}$, zero at
the discrete solution. A trial state has only $k+q$ free numbers, so
$r$ cannot be driven to zero. We ask instead that it be small
against $M$ fixed directions. The rows of $P\in\mathbb{R}^{M\times n}$
are those weak tests. On the Dirichlet square they are the
low-frequency tensor-product sine vectors, eigenvectors of the
five-point operator, $PA=\LambdaP$. Substituting
(2) into $r$ and testing gives the problem solved
at every query,

$$
(z^{\star},y^{\star})
  =
  \operatorname*{arg\,min}_{z,\,y}\;
  \tfrac12\,\lVert r_{w}(z,y) \rVert_2^2,
  \qquad
  r_{w}(z,y) = \Lambda_\star^{-1}\,P\,r\!\big(u(z,y)\big)
  \in\mathbb{R}^{M},
  \qquad M>k+q .
$$

<!-- equation (3) -->

The diagonal $\Lambda_\star$ puts the rows on a common scale
(Appendix A). This is least-squares Petrov–Galerkin
projection with fixed tests (Carlberg et al., 2011; Lee & Carlberg, 2020),
and $M>k+q$ makes the system overdetermined. What changes from one PDE
to the next is whether $y$ can be removed before the iteration.

**Elliptic (Poisson).**
The residual is $r(u)=A u-f$. Testing and rescaling leave a
residual that is linear in the coefficients,

$$
r_{w}(z,y)
  \;=\;
  B_0\,\big(h_\theta(z)+C_q y\big) - b_0,
  \qquad
  B_0=PG,\quad b_0=\Lambda^{-1}P f .
$$

<!-- equation (4) -->

Because the residual is linear in $y$, the best $y$ for a fixed
$z$ has a closed form. A thin QR factorisation
$B_0C_q=Q\mathcal{R}$, independent of $z$, gives
$\mathcal{R} y=Q^{\top}(b_0-B_0h_\theta(z))$
(Golub & Pereyra, 1973), and Levenberg–Marquardt iterates on
$z$ alone. Raising $q$ enlarges the trial space and leaves the
iteration in $k$ unknowns. At $q=R$ the latent problem drops out and
(4) is one linear solve for $y$
(Appendix A.1).

**Parabolic (heat).**
The semi-discretisation $du/dt=-\kappa A u$ is advanced by
Crank–Nicolson. The manifold is substituted into the fully discrete
step before projection. At $q=0$ each step solves

$$
z_{n+1}
  \;=\;
  \operatorname*{arg\,min}_{z}\;
  \lVert B_0\,h_\theta(z) - D\,B_0\,h_\theta(z_n) \rVert_2,
$$

<!-- equation (5) -->

where $D$ scales each test by its Crank–Nicolson factor. With
corrections, $h_\theta(z)+C_q y$ replaces $h_\theta(z)$ and
$y$ is removed by the same QR step. The next step receives the
corrected coefficients. At $q=R$ the step is the linear recurrence
(10). Reflective waves are integrated explicitly
in the latent coordinates (Appendix A.4).

**Hyperbolic (Burgers).**
For $u_t+u(u_x+u_y)=\nu\Delta u$, with a sign-upwind stencil and
backward Euler, write $c=h_\theta(z)+C_q y$. The tested step is

$$
r_{w,n}
  \;=\;
  \big(I+\Delta t\,\nu\Lambda\big)^{-1}
  \Big[
    B_0(c-c_n)
    +\Delta t\big(\widehat N(c)+\nu\Lambda B_0 c\big)
  \Big].
$$

<!-- equation (6) -->

Diffusion is the preassembled linear term. Advection
$\widehat N(c)$ is nonlinear in $c$, so $y$ stays among the unknowns.
One Levenberg–Marquardt step damps the $z$ block and takes an
undamped Gauss–Newton step in $y$. That split removes the jointly
damped configuration's $6$ budget exits
(Table 23). The upwind stencil is in
Appendix A.3.

**Solver and exits.**
Poisson and heat iterate $z$ after $y$ is eliminated; Burgers
iterates $(z,y)$. A damped Levenberg–Marquardt step
(Marquardt, 1963) is kept only when the residual decreases. A stall
is reported as early-stopped, separately from stationarity and from a
residual threshold. Every solve starts from stored training codes
(Appendix A.6).

### 3.3 Hyper-reduction

<!-- section sources: none (prose only) -->

The manifold cuts the unknowns from $n$ to $k+q$, but a tested term
such as $P N(G c)$ still visits every node, so one residual
still costs $O(n)$. Linear terms are tested once, offline,
$P(A u-f)=B(h_\theta(z)+C_q y)-b$, where the
$M\times R$ matrix $B=\LambdaPG$ is assembled by the
discrete transform of the problem. A linear term is then exact, and
the online residual does not return to the mesh.
Burgers advection is the term that remains. It is evaluated at every
node, at $O(nR)$ per residual, or replaced by an empirical quadrature
rule (Hern'andez et al., 2017; Yano & Patera, 2019) whose non-negative
weights solve $\min_{w\ge 0}\lVert \mathcal{G}w-\beta \rVert_2$
(Lawson & Hanson, 1974), one weight per node, shared by every
state (Appendix A.5). A rule is confirmed only on
re-draws of states the solver reaches, against a bar fixed before
certification (Table 22), and never on its NNLS fit
residual.

### 3.4 Model Architecture

<!-- section sources: none (prose only) -->

Three properties of the solve motivate the decoder in
Figure 1. Every query starts from a stored training code
and descends along the head's Jacobian, which motivates a linear skip.
Quadrature evaluates the decoder only at a sparse set of nodes, and
one model serves every mesh, which motivates a coordinate-network
bank. One trained model must offer several accuracy settings, which
motivates nested corrections.

**Linear skip, for the latent solve.**
The head splits into a shallow network and a linear map,
$h_\theta(z)=\varphi_\theta(z)+W^{\top}z$,
with $\varphi_\theta$ an MLP of two hidden SiLU layers. The skip keeps
a Jacobian component independent of $z$, so the first step from
a stored code has a fixed linear direction.

**Coordinate-network bank, for node-local evaluation.**
The factor in (1) is a product on the cube; periodic
Navier–Stokes drops it and applies a solenoidal projection. Because
$g_\phi$ reads a coordinate, one node is
decoded alone, and a quadrature rule caches the bank on its nodes.

**Nested corrections, for accuracy.**
$C_q$ holds the leading directions of the training misses in
§(1), nested in $q$, and (3)
solves their weights $y$. One frozen head supplies both columns of
Table 1.

![Figure 1](figures/architecture.png)

**Figure 1.** The decoder. $z$ splits into a linear skip and a two-layer
network, which sum to $h(z)$; $y$ adds $C_q y$; the bank rebuilds the
field and vanishes on the boundary. There is no encoder. Online path:
Figure 2.

**What is fixed and what is solved.**

Training fixes the bank, the head, the directions $C_q$ and the
tests for every row of Table 1. Each query solves
$z$ and $y$. Deployment chooses $q$, the test count $M$, the
quadrature and the stopping rule, and leaves the weights fixed.

## 4 Implementation

<!-- section sources: none (prose only) -->

\subsection{Matrix-free Projected Operators via JAX}

The matrix $B=\LambdaPG$ of §3.3 is built once per mesh, so the
online residual of a linear term is the product
$B(h_\theta(z)+C_q y)-b$. The remaining derivative is the Jacobian of the head. Forward-mode
automatic differentiation (Bradbury et al., 2018) supplies it, and the
reduced Jacobian in $(z,y)$ is

$$
Dh_\theta\, v \;=\; \mathrm{JVP}(h_\theta,z,v),
  \qquad
  B\,[\,Dh_\theta \;\; C_q\,].
$$

<!-- equation (7) -->

Each accepted step solves the damped normal equation directly. A linear
PDE does not return to the grid, and the projected operator is not
passed to a Krylov method. Burgers advection reads the cached stencil
on the quadrature nodes; runs above $1024^2$ assemble the Jacobian in
(7) analytically (Appendix A.3).

\subsection{Training Protocol}

Training is an auto-decoder on training data only, in two stages.
The coordinate network $g_\phi$ is fitted to the training states through
(1) and frozen, which fixes the bank. The head is then
fitted to the bank coefficients of those states, with one stored code
per training state, and $C_q$ is built from that head's misses
(§(1)). Widths, cohorts and offline costs are in
Appendix D.

## 5 Experimental Setup and Benchmark Problems

<!-- section sources: none (prose only) -->

\restoredisplays

**Problems and references.**
In two dimensions: Poisson on the square, heat, viscous Burgers and
reflective waves, with an L-shaped Poisson domain as an ill-conditioned
check; in three: Poisson, heat,
Burgers and incompressible Navier–Stokes. Meshes and cohorts are in
Table 11, the sampled families in Table 12
and the three-dimensional configurations in Table 20.
Burgers provides the correction-rank study.

**Accuracy.**
We report worst relative $L^2$ error over the stated cases and requested
times, in percent. Poisson uses the steady reference norm and heat the
current reference norm; Burgers and Navier–Stokes use the initial-field
norm over evolved times; waves use the energy-state error over
displacement and velocity. Same-grid error measures reduction
error relative to the converged discrete solution. Error against a
refined reference also includes discretisation error; the two are never
interchanged. Settings for a held-out (final) cohort are fixed before its
cases are accessed; all other cohorts are development cohorts.

**Timing and speedup.**
Every accuracy–runtime pair comes from the same solver invocation, in
double precision, with GPU burn-in, synchronised timing and medians of
retained repetitions (protocol: Appendix D.1). GPU
query time covers initialisation, the solve or rollout and the requested
device outputs; complete-query time adds host transfers and is labelled
separately. Training and compilation are offline costs. Speedup is
$S=T_{\mathrm{FOM}}/T_{\mathrm{method}}$ with both times from the same
allocation; each table names the FOM algorithm and shows its error.
Solves that miss their stopping rule are reported as such and do not
enter a speedup.

**Baselines.**
Linear problems are compared with conjugate gradients (CG), including the
L-shaped check and the CG solve inside each Crank–Nicolson (CN) heat
step. Burgers is compared with Newton–BiCGStab and Navier–Stokes with
its CNAB2 time integrator. The named FOM of a row is that solver at its
fastest tested setting that is at least as accurate as the accurate
NM-ROM setting.

## 6 Numerical Experiments and Results

<!-- section sources: none (prose only) -->

Table 1 reports the headline comparison against the
named iterative solvers; the subsections that follow analyse the settings
behind it and which knob to turn.

### 6.1 Accuracy and Speed against Full-Order Solvers

<!-- section sources: none (prose only) -->

**Table 1.** NM-ROM against the named full-order solver at every measured
problem and mesh. **Both NM-ROM columns of a row come from one
frozen model**: *fast* is the uncorrected setting ($q=0$) and
*accurate* the selected accurate rank (Table 13); no network is
retrained between them. Error is the worst relative $L^2$ error over the
cohort (%). Speedup is the FOM's median time divided by the NM-ROM's in
the same GPU allocation, **bold** where the NM-ROM is faster. The
FOM is the named iterative solver at its fastest tested setting that is at
least as accurate as the accurate setting. Each ratio divides two times from one job; for the
$4096^2$ Burgers fast column both come from a later job with a wider
grid of settings, where the rule picks \nBhFastFom. The FOM column
reports the accurate setting's comparator. Marker definitions, the heat evolved-time errors and the
quadrature labels are in Appendix D.1. Dashes mark
settings not measured. Times, settings and complete-query speedups:
Table 13.

<!-- table: TH_headline -->
| Problem | Mesh | Accurate err. (%) | Accurate speedup | Fast err. (%) | Fast speedup | FOM err. (%) | FOM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson 2D (development) | 256^2 | 0.97 | **3.13×** | 3.16 | **3.33×** | 0.15 | CG, rtol 10^{-2} |
| Poisson 2D (development) | 1024^2 | 0.96 | **13.9×** | 3.15 | **15.2×** | 0.072 | CG, rtol 10^{-2} |
| Poisson 2D (development) | 2048^2 | 0.96 | **72.8×** | 3.15 | **74.3×** | 0.049 | CG, rtol 10^{-2} |
| Poisson 2D (development) | 4096^2 | 0.96 | **116×** | 3.15 | **117×** | 0.45 | CG, rtol 10^{-1} |
| Poisson, L-shape 2D (development) | 256^2 | 2.13 | **3.84×** | 3.86 | **4.04×** | 0.64 | CG, rtol 10^{-2} |
| Poisson, L-shape 2D (development) | 512^2 | 2.12 | **6.07×** | 3.85 | **6.18×** | 0.38 | CG, rtol 10^{-2} |
| Poisson, L-shape 2D (development) | 1024^2 | 2.20 | **8.38×** | 3.85 | **8.59×** | 1.05 | CG, rtol 3{\times}10^{-2} |
| Poisson, L-shape 2D (development) | 2048^2 | 2.20 | **17.0×** | 3.85 | **17.1×** | 0.76 | CG, rtol 3{\times}10^{-2} |
| Heat 2D (development; earlier checkpoint, one setting) | 64^2 | — | — | 4.56 | 0.22× | 1.59 | CN–CG, rtol 10^{-2} |
| Heat 2D (development; earlier checkpoint, one setting) | 256^2 | — | — | 4.56 | 0.61× | 0.93 | CN–CG, rtol 10^{-2} |
| Heat 2D (development; earlier checkpoint, one setting) | 1024^2 | — | — | 4.56 | **4.85×** | 0.77 | CN–CG, rtol 10^{-2} |
| Heat (wide bank) 2D (sealed held-out; opened once) | 1024^2 | 0.49 | **1.59×** | 1.36 | **1.57×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank) 2D (sealed held-out; opened once) | 2048^2 | 0.49 | **9.74×** | 1.36 | **9.59×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank) 2D (sealed held-out; opened once) | 4096^2 | 0.49 | **36.0×** | 1.36 | **35.4×** | 0.47 | CN–CG, \Delta t=0.05, rtol 10^{-2} |
| Heat (wide bank, batched fit) 2D (sealed held-out; opened once) | 1024^2 | 0.49 | **13.9×** | 1.36 | **13.0×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank, batched fit) 2D (sealed held-out; opened once) | 2048^2 | 0.49 | **60.0×** | 1.36 | **57.0×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank, batched fit) 2D (sealed held-out; opened once) | 4096^2 | 0.49 | **103×** | 1.36 | **101×** | 0.47 | CN–CG, \Delta t=0.05, rtol 10^{-2} |
| Burgers 2D (development) | 256^2 | 0.51^{s} | 0.043× | 1.89 | 0.79× | 0.049 | Newton–BiCGStab, tol 10^{-3} |
| Burgers 2D (development) | 512^2 | 0.55^{x} | 0.068× | 2.14 | **1.31×** | 0.052 | Newton–BiCGStab, tol 10^{-3} |
| Burgers 2D (development) | 1024^2 | 0.59^{d} | 0.0037× | 2.29 | **2.02×** | 0.034 | Newton–BiCGStab, tol 10^{-4} |
| Burgers 2D (development; 6 cases, arm chosen here) | 2048^2 | 0.87^{\ell} | 0.99× | 2.37 | **4.16×** | 0.049 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers 2D (development; 6 cases, arm chosen here) | 4096^2 | 0.60^{w} | **4.87×** | 2.41 | **10.10×** | 0.050 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers (held-out cases) 2D (held-out; 64 cases, one timing repetition) | 2048^2 | 1.31^{\ell} | **1.43×** | 9.03 | **5.63×** | 0.13 | Newton–BiCGStab, tol 10^{-3} |
| Burgers (held-out cases) 2D (held-out; 64 cases, one timing repetition) | 4096^2 | 1.33^{w} | **5.38×** | 9.03 | **10.10×** | 0.14 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers, confirmed rule 2D (development; re-drawn quadrature rule) | 512^2 | 0.56^{v} | 0.091× | 2.14 | 0.75× | 0.050 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers, confirmed rule 2D (development; re-drawn quadrature rule) | 1024^2 | 0.59^{v} | 0.32× | 2.29 | **1.39×** | 0.048 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers (earlier model) 2D (development; earlier model, stalled exits permitted) | 1024^2 | — | — | 3.91 | **1.63×** | 2.39 | Newton–BiCGStab, relaxed |
| Poisson 3D (accepted final) | 32^3 | 0.26 | 0.94× | 1.40 | 0.99× | 0.16 | CG, rtol 10^{-2} |
| Poisson 3D (accepted final) | 64^3 | 0.26 | **1.33×** | 1.39 | **1.38×** | 0.11 | CG, rtol 10^{-2} |
| Poisson (dev. sources) 3D (development) | 128^3 | 0.16 | **6.75×** | 0.55 | **6.98×** | 0.075 | CG, rtol 10^{-2} |
| Poisson (dev. sources) 3D (development) | 256^3 | 0.16 | **23.2×** | 0.55 | **23.7×** | 0.049 | CG, rtol 10^{-2} |
| Heat (new bank) 3D (accepted final) | 32^3 | 0.11 | 0.057× | 2.00 | 0.13× | 0.078 | CN–CG, \Delta t=0.025, rtol 10^{-4} |
| Heat (new bank) 3D (accepted final) | 64^3 | 0.11 | —^{n} | 2.00 | 0.21× | 0.081 | CN–CG, \Delta t=0.025, rtol 10^{-4} |
| Heat (new bank) 3D (accepted final) | 128^3 | 0.11 | 0.28× | 2.00 | 0.62× | 0.082 | CN–CG, \Delta t=0.025, rtol 10^{-4} |
| Heat (new bank, batched fit) 3D (accepted final) | 32^3 | 0.11 | 0.62× | 2.00 | 0.57× | 0.078 | CN–CG, \Delta t=0.025, rtol 10^{-4} |
| Heat (new bank, batched fit) 3D (accepted final) | 64^3 | 0.11 | 0.94× | 2.00 | 0.85× | 0.081 | CN–CG, \Delta t=0.025, rtol 10^{-4} |
| Heat (new bank, batched fit) 3D (accepted final) | 128^3 | 0.11 | **2.22×** | 2.00 | **2.10×** | 0.082 | CN–CG, \Delta t=0.025, rtol 10^{-4} |

**Linear problems.**
On Poisson the accurate setting reaches $\nHeadPoissonAccErr %$ error,
$\nHeadPoissonAccS\times$ faster than CG at $\nHeadPoissonMesh$ and
$\nHiresPoissonAccSFortyNinetySix\times$ at $4096^2$ with the same
error; the corrections cost little speed because they are eliminated
analytically (§3.2). On the L-shaped check the
accurate setting is $\nHeadLshapeAccS\times$ faster at
$\nHeadLshapeAccErr %$. The earlier
heat checkpoint crosses CG between $256^2$ and $1024^2$ ($\nHeadHeatFastS\times$
at $1024^2$). A wider heat bank, evaluated once on a sealed held-out
cohort at every mesh, reaches $\nHeatWideAccErr %$ over all times and is
$\nHeatWideAccS\times$ faster than CN–CG at $4096^2$
($\nHeatWideBatchedAccS\times$ with the batched fit). In three dimensions the held-out Poisson cohort gives
$\nHeadPoissonThreeAccErr %$ at $\nHeadPoissonThreeAccS\times$. A new three-dimensional
heat bank, evaluated once on a sealed cohort of \nHeatNewCases cases,
meets the 1 % all-times target at every mesh (accurate
$\nHeatNewAccErr %$); with Crank–Nicolson stepping it stays slower
than CN–CG ($\nHeatNewCnAccS\times$ at $128^3$) and only the batched fit
overtakes it, at $128^3$ ($\nHeatNewBfAccS\times$) and not at $32^3$ or
$64^3$.

**Burgers.**
The reduced solve costs the same at every mesh, so the fast setting
overtakes Newton–BiCGStab by $1024^2$ ($\nHeadBurgersFastS\times$
faster at $\nHeadBurgersFastErr %$ error). The accurate
setting reaches $\nHeadBurgersAccErr %$ at $256^2$ and, on development
cases, is slower than the FOM through $2048^2$. This includes the rules
confirmed on independent re-draws at $512^2$ and $1024^2$ (rows marked
$^{v}$). The dense $1024^2$ panel row's stored $q=128$ rule gives
$\nHeadBurgersMidErr %$ at $\nHeadBurgersMidS\times$. At $4096^2$ the accurate setting chosen on six development cases
reaches $\nBurgDevAccErrFortyNinetySix %$ and is
$\nBurgDevAccSFortyNinetySix\times$ faster than the fastest Newton setting
at least as accurate. On 64 held-out cases the same setting gives
$\nBurgHoldAccErrFortyNinetySix %$ at
$\nBurgHoldAccSFortyNinetySix\times$.
The lattice quadrature rule of those rows is validated on held-out
reached states but not confirmed on independent re-draws: it passes five
draws and fails the confirmation draw (Appendix D.1).
The fast setting gives $\nBhFastS\times$ on both cohorts against the
fastest tested Newton–BiCGStab setting at least as accurate as itself.

**Resolution.**
Table 1 shows that the NM-ROM query time grows more
slowly with the mesh than the iterative FOM's: in every series the
fast-setting speedup rises under the rule that selects each row's
comparator (the fastest tested setting at least as accurate as the
accurate NM-ROM, or as the fast setting for the $4096^2$ Burgers fast
column), and the selected comparator may change with the mesh. The
speedup also rises against one fixed tight setting where a series timed
one (Table 13).

**Table 2.** Other nonlinear-manifold ROMs on the shared Burgers 2D family at
$256^2$ and $512^2$: worst and median evolved same-grid error over 32
validation cases (never used for any choice) and compiled-query memory,
one allocation. Latent dimension is matched for
the baselines and our fast setting; our accurate setting solves
$k+q=272$ unknowns. The shallow masked-autoencoder NM-LSPG of
Kim et al. (2022) passed its reproduction gate on the third of
three pre-registered attempts; its hyper-reduction was not reproduced,
and its published encoder width does not fit our devices, so at $512^2$
its width-capped encoder is training-time limited
(Table 16, which also
states the data-matched row and the residual path: every row here, ours
included, uses a dense residual, so no times are given). The
convolutional-autoencoder NM-ROM (Lee & Carlberg, 2020) was not run.

<!-- table: TH_nmrom_baselines -->
| Method | Unknowns | 256^2 worst (%) | 256^2 median (%) | 256^2 MB | 512^2 worst (%) | 512^2 median (%) | 512^2 MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kim et al. NM-LSPG | 8 | 163.93 | 43.34 | 2480 | 175.71 | 55.48 | 9959 |
| Kim et al. NM-LSPG | 16 | 145.40 | 43.87 | 2567 | 174.99 | 58.56 | 10511 |
| Kim et al. NM-LSPG | 32 | 127.51 | 43.61 | 2742 | 252.33 | 56.72 | 10879 |
| Kim et al. NM-LSPG, data-matched | 16 | 120.33 | 38.71 | 2567 | — | — | — |
| POD-LSPG | 8 | 55.52 | 22.72 | 18 | 55.82 | 24.08 | 71 |
| POD-LSPG | 16 | 41.53 | 10.45 | 30 | 42.01 | 13.83 | 122 |
| POD-LSPG | 32 | 24.78 | 5.65 | 55 | 25.63 | 7.74 | 222 |
| This work, fast (q=0, k=16) | 16 | 6.79 | 0.66 | 625 | 7.72 | 0.69 | 2368 |
| This work, accurate (q=256, k=16) | 272 | 0.88 | 0.089 | 1769 | 1.05 | 0.080 | 6891 |

On 32 validation Burgers cases at $256^2$ and $512^2$
(Table 2) our fast setting, at matched latent
dimension, reaches worst errors of
$\nBaseOursFastWorst %$, against $\nBaseKimRange %$ for the
reproduced shallow masked-autoencoder NM-ROM and $\nBasePodRange %$
for POD-LSPG; the accurate setting, at 272 solved unknowns, reaches
$\nBaseOursAccWorst %$.

**Where the method currently fails.**

Table 3 collects the three problems that miss their
targets; they are excluded from Table 1. On
three-dimensional Burgers the corrections cut the held-out worst error
from $\nFailBurgersQzero %$ to $\nFailBurgersAcc %$, but the FOM is
faster and the refined-reference check fails, so only same-grid
reduction error is established. On three-dimensional
Navier–Stokes the corrections lower the worst error only from
$\nFailNsQzero %$ to $\nFailNsAcc %$; \nFailNsFailing of
\nFailNsCases held-out cases miss the $\nFailNsTarget %$ target. A follow-up
diagnosis attributes this to representation: the family is a translating
vortex orbit that a fixed bank must rebuild, and a shift-tracking linear
POD ROM, a different model, meets the target on a fresh held-out cohort
($\nNsGrokWorst %$ worst, \nNsGrokOver of \nNsGrokCases over) at
$\nNsGrokS\times$ CNAB2. Four neural operators trained at $32^3$ on the
same data and wall budget beat the NM-ROM on both of these problems, and
are behind it on Poisson 3D; none of them transfers to $64^3$
(Appendix C). On reflective waves the fast setting beats CG, but its
energy-state error, which includes velocity, is $\nFailWaveQzero %$;
corrections bring it to $\nFailWaveAcc %$, still above CG's, at a cost
that removes the speedup. The three-dimensional solves evaluate the
residual densely.

**Table 3.** Problems on which the method misses its target: Navier–Stokes
$\nFailNsTarget %$ per held-out case; wave, an energy-state error within that of CG; Burgers 3D, faster than the
FOM at its accuracy (no numeric error target was recorded). Error is the
worst same-grid relative $L^2$ error (%) over evolved times; the wave
error is the energy-state error over displacement and velocity.
Speedup is against the named FOM in the same allocation.

<!-- table: TH_failures -->
| Problem | Setting | NM-ROM err. (%) | FOM err. (%) | Speedup | FOM |
| --- | --- | --- | --- | --- | --- |
| Burgers 3D, $32^3$ (final) | $q=0$ | 16.26 | 2.31 | 0.038× | Newton–BiCGStab |
| Burgers 3D, $32^3$ (final) | $q=192$ | 4.40 | 2.31 | 0.023× | Newton–BiCGStab |
| Navier--Stokes 3D, $32^3$ (final) | $q=0$ | 20.34 | 2.47 | 0.0025× | CNAB2 |
| Navier--Stokes 3D, $32^3$ (final) | $q=256$ | 18.92 | 2.47 | 0.0007× | CNAB2 |
| Wave 2D, $1024^2$ (dev.) | $q=0$ | 11.34 | 4.29 | **3.03×** | midpoint–CG |
| Wave 2D, $1024^2$ (dev.) | $q=32$ | 5.12 | 4.29 | 0.23× | midpoint–CG |

**Table 4.** Correction-rank tunability at $4096^2$. Each problem block is one
frozen model; every speedup is the median GPU query time of *one*
named full-order setting (the *FOM* row, the fastest at least as
accurate as the block's most accurate setting, in the same allocation)
divided by the setting's (ms). Error is the worst same-grid relative
$L^2$ error (%), over evolved times for Burgers and heat. Burgers
development (6 cases) and held-out (64 cases, never used for selection)
share settings; heat uses the sealed held-out cohort, Poisson
development sources. On Burgers $q$ and $M$ change together
($M\approx4(k+q)$, $M=544$ an alternative at $q=256$; fixed-$M$ ladder:
Table 17), $q>0$ with the lattice rule marked
$^{w}$ in Table 1. Each block divides one FOM time, its most accurate
setting's comparator, measured with the rungs in the same job; the
Burgers $q=0$ ratio is therefore not Table 1's
fast-column ratio ($\nBhFastS\times$), which uses a later job's wider grid.
Heat and Poisson vary $q$ only.
$^{\star}$ looser stopping tolerance; $^{\dagger}$ quadrature rule
failed its held-out check.

<!-- table: TH_tunability -->
| Setting | Dev. err (%) | Dev. ms | Dev. speedup | Held-out err (%) | Held-out ms | Held-out speedup |
| --- | --- | --- | --- | --- | --- | --- |
| Burgers 2D (6 development / 64 held-out cases) |  |  |  |  |  |  |
| q=0, M=64 | 2.41 | 40.7 | 12.9× | 9.03 | 40.6 | 13.2× |
| q=128, M=576 | 1.09 | 77.4 | 6.77× | 2.43 | 74.6 | 7.21× |
| q=256, M=544 | 0.88 | 112 | 4.67× | 2.93 | 105 | 5.11× |
| q=256, M=1088 | 0.60 | 127 | 4.12× | 1.33 | 113 | 4.77× |
| q=256, M=1088, looser tol.\star | 0.60 | 108 | 4.87× | 1.33 | 100 | 5.38× |
| q=256, M=2176\dagger | 0.46 | 132 | 3.96× | 1.13 | 126 | 4.25× |
| FOM: Newton–BiCGStab, tol. 3×10-3 | 0.050 | 524 | 1× | 0.14 | 537 | 1× |
| Heat 2D, wide bank (16 sealed held-out cases) |  |  |  |  |  |  |
| q=0 | –- | –- | –- | 0.84 | 32.2 | 146× |
| q=32 | –- | –- | –- | 0.22 | 31.7 | 149× |
| q=96 | –- | –- | –- | 0.082 | 24.4 | 193× |
| FOM: CN–CG, \Delta t=0.025, rtol 10-6 | –- | –- | –- | 0.045 | 4713 | 1× |
| Poisson 2D (12 development sources) |  |  |  |  |  |  |
| q=0 | 3.15 | 16.7 | 117× | –- | –- | –- |
| q=64 | 2.08 | 16.9 | 116× | –- | –- | –- |
| q=128 | 1.55 | 16.9 | 116× | –- | –- | –- |
| q=256 | 0.96 | 16.9 | 116× | –- | –- | –- |
| FOM: CG, rtol 10-1 | 0.45 | 1955 | 1× | –- | –- | –- |

**Settings.**

Moving between the two settings of a row never requires retraining, only
new deployment arguments: this is what we mean by deployment-time
tunability (times and settings in Table 13;
fixed-$M$ evidence in Table 17).

### 6.2 Which Knob to Turn

<!-- section sources: none (prose only) -->

Correction rank is the accuracy control (with a smaller fixed test space
the same ranks give a $1.22\times$ error reduction
from $q=0$ to $256$, so the test count is held fixed along the ladder).
Quadrature and the stopping tolerance are cost controls: they lower the
runtime at little or no change in error. The iteration cap is a
safeguard, not a control; a truncated solve fails (Appendix
Table 18 and Table 19).

On Burgers at $4096^2$, changing the correction rank trades accuracy for cost. Table 4 compares every setting with one
shared Newton–BiCGStab setting
($\nTuneBurgDevSpeedMin$–$\nTuneBurgDevSpeedMax\times$ on development
cases). Table 1's fast column is a different comparison:
$\nBhFastS\times$ on both cohorts, against the setting the rule picks for
it on a later job's wider grid. On linear problems the
corrections are eliminated analytically, so accuracy improves at nearly
constant cost.

The multi-seed sealed ladder is in Appendix E.

**Limitations.**

(i) Speedups use the named iterative full-order solvers at the stated
tolerances; direct and spectral solvers and coarser discretisations are
outside this study's scope. The named FOM is more accurate in every row.
For heat at $4096^2$, the linear-bank baseline ($\nHeatLinErr %$ in $\nHeatLinMs$ ms) is
faster than every NM-ROM setting, as in three dimensions
($\nHeatNewLinErr %$ at $128^3$).
(ii) A wider Burgers bank cuts the $256^2$ projection floor from
$\nBhFloorOld %$ to $\nBhFloorNew %$ on its selection cases. With confirmed quadrature, it is less accurate on 64 held-out cases
($\nBhBankErr %$ at $\nBhBankS\times$) than the model in
Table 1. Its settings with unconfirmed quadrature are in
Appendix D.1. The correction subspace and weak test
space, rather than the bank, limit accuracy here.
Development accuracy does not transfer to held-out cases
(the unrestricted solve in the 512-function bank reaches only
$\nBurgBankFloorConfirm %$).
Poisson errors lie near the bank floor ($\nHiresFloorSquare %$ square,
$\nHiresFloorCube %$ cube); the L-shape ($\nHiresLshapeAccErr %$ against
a floor of $\nHiresLshapeFloor %$) is limited by the head.
(iii) The fixed-test-space rank study uses one checkpoint. The
multi-seed and sealed study varies $M=4(k+q)$, so it is not a replication.
FOM comparisons do not isolate the nonlinear head's contribution
over the bank's linear span.
(iv) Quadrature validation covers finitely many reached states,
not a global certificate; no rule is used in three dimensions.
Reduced solves can converge to a wrong branch. Lower same-grid error
does not remove discretisation error.
(v) Unmarked cohorts are small development cohorts; exceptions are marked
final or held-out. Timings report medians without dispersion.

## 7 Conclusion and Future Work

<!-- section sources: none (prose only) -->

We built an NM-ROM whose distinguishing property is a deployment-time
accuracy–cost choice from one trained model, through nested corrections
that change its capacity after training; it keeps its accuracy under
mesh refinement on
two-dimensional Poisson, heat and Burgers, so its speedup over the named
iterative solvers grows with resolution: at $4096^2$ its accurate setting
reaches $\nHeadPoissonAccErr %$ at $\nHiresPoissonAccSFortyNinetySix\times$
on Poisson, $\nHeatWideAccErr %$ at $\nHeatWideAccS\times$ on held-out
heat and $\nBurgHoldAccErrFortyNinetySix %$ at
$\nBurgHoldAccSFortyNinetySix\times$ on held-out Burgers, far more
accurate than a reproduced NM-ROM and POD-LSPG; its $4096^2$ rule is not
confirmed on re-draws, and the confirmed variant is slower than the FOM.
The named solvers remain
more accurate; three-dimensional heat meets its accuracy target but is
faster than CN–CG only with a batched fit at $128^3$; and
three-dimensional Burgers, Navier–Stokes and the full wave state miss
their targets (Table 3);
future work is a better three-dimensional bank, a cheaper high-rank
solve and quadrature in three dimensions.

## Reproducibility statement

<!-- section sources: none (prose only) -->

Every table and prose number is generated from hash-pinned run records
identifying source, checkpoint, settings, cohort, allocation and audit
(Appendix D); failed settings are kept.

## AI use statement

<!-- section sources: none (prose only) -->

Language-model assistants supported method and experiment development,
implementation, analysis, auditing and drafting. Recorded solver runs
produced the numbers and the generators reproduce them. The authors are
responsible for the methods, results and text.

\enlargethispage{\baselineskip}

## References

<!-- section sources: none (prose only) -->

See `bib-inline.tex` and `main.pdf`; citations in the text are author–year keys from that file.

---

# Appendices

\setcounter{table}{0}\setcounter{figure}{0}
\makeatletter\@addtoreset{table}{section}\@addtoreset{figure}{section}\makeatother
\renewcommand{\thetable}{\Alph{section}.\arabic{table}}
\renewcommand{\thefigure}{\Alph{section}.\arabic{figure}}

## A Method details

<!-- section sources: none (prose only) -->

This appendix writes the reduced problem (3) out for
each PDE: what the residual is, how the time step enters, and what is
solved at each query. Poisson, heat and Burgers below are fully discrete
weak least-squares problems; the wave arm (§A.4) is an
explicit second-order latent integration and is not an instance of
(3). Formulas are for the scalar two-dimensional
Dirichlet case; three-dimensional and periodic-vector scope is stated in
Appendix D.1.

### A.1 Elliptic instance: Poisson

<!-- section sources: none (prose only) -->

For $A u=f$ we take $\Lambda_\star=\Lambda$. Testing the residual
and rescaling the rows leaves (4).
Because the retained tests are orthonormal and $A u^{\star}=f$, this residual
is exactly the tested error $P(u-u^{\star})$, so the minimised
objective is the squared discrete $L^2$ error of the reduced state in the
retained modes; this holds only for the modes $P$ retains.

**Analytically eliminated corrections.** Because the residual is
linear in $y$, the best $y$ for a given $z$ can be written down
instead of iterated for. Let $B_0C_q=Q\mathcal{R}$ be a thin QR
factorisation. $Q$ does not depend on $z$, so the inner
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

The reduced heat step is one nonlinear least-squares problem per time
step. The semi-discrete equation $\dot u=-\kappaA u$ is advanced with
Crank–Nicolson,
$(I+\tfrac{\Delta t\kappa}{2}A)u^{n+1}=(I-\tfrac{\Delta t\kappa}{2}A)u^{n}$,

and the reduced step substitutes the manifold into the fully discrete equation
*before* projecting. With $u^{n}=u(z_n)$, projecting with
$P$ and row-scaling by $\Lambda_\star=I+\tfrac{\Delta t\kappa}{2}\Lambda$
gives (5),
where $D=\operatorname{diag}\!\left((1-\tfrac{\Delta t\kappa}{2}\lambda_i)/(1+\tfrac{\Delta t\kappa}{2}\lambda_i)\right)$,
warm-started at $z_n$. Two points are structural. The right-hand side is built from
the *decoded current state* at every step: carrying the quantity the
previous step already made small freezes the recursion and reproduces a
one-step solution to round-off. And $z_{n+1}$ enters through $h_\theta$, so
the step is a nonlinear least-squares problem, not a linear solve.
With corrections, $c_n=h_\theta(z_n)+C_q y_n$ replaces $h_\theta(z_n)$:

$$
(z_{n+1},y_{n+1})=\operatorname*{arg\,min}_{z,y}
  \lVert B_0\big(h_\theta(z)+C_q y\big)-D\,B_0\,c_n \rVert_2 ,
$$

<!-- equation (9) -->

where $y$ is eliminated as in (8) and the *complete*
previous coefficients $c_n$, corrections included, are carried into the next
step. At $q=R$ the head is redundant and the step is the linear
least-squares recurrence

$$
c_{n+1}=B_0^{+}D\,B_0\,c_n,
$$

<!-- equation (10) -->

with $B_0^{+}$ the pseudo-inverse ($M\ge R$; the numerical rank of $B_0$ is
checked) and $c_0$ fitted to the supplied initial field. This endpoint is
the weak Crank–Nicolson propagation of the bank coefficients, with no head
and no iteration. It is the “top rung” of the heat ladder and the
linear-bank baseline of the text. The deployed
path factors the damped normal matrix by Cholesky; a variant assembles the Gram
matrix $S=B_0^{\top} B_0$ offline and forms $H=Dh_\theta^{\top} SDh_\theta$ directly, which is
exact and is reported as an algebraic ablation of the same step.

### A.3 Nonlinear advection: Burgers

<!-- section sources: none (prose only) -->

Burgers is the only problem whose tested residual stays nonlinear in the
coefficients, so nothing is eliminated in closed form. The full-order
problem is $u_t+u(u_x+u_y)=\nu\Delta u$ with the non-conservative
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

<!-- equation (11) -->

$\delta_y$ analogous and switching on the same centre value, ghost zeros on all
four walls, and backward Euler in time,
$r_n(u)=u-u^{n}+\Delta t\big(N(u)-\nu\Delta_h u\big)$. Projecting and row
scaling by $\Lambda_\star=I+\Delta t \nu\Lambda$ gives (6),
with $c=h_\theta(z)+C_q y$.
The diffusion term is exact and the row scaling is the diagonal of the
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
when, and only when, its residual norm is smaller; the $4096^2$ rows also try a
quadratic predictor under the same rule. The runs above $1024^2$ assemble
the Jacobian analytically instead of by forward-mode products and clip the
step; in the settings listed in Table 13 they solve the
normal system by Cholesky factorisation. In every Burgers arm above $1024^2$
except the $2048^2$ development accurate arm, the damping $\lambda$ is carried
from one step to the next rather than reset to $\lambda_0$.

**Block-damped step.**
With $J_{}=[J_{z} J_{y}]$ the Jacobian of
(6) in $(z,y)$, obtained in one forward-mode pass,
each iteration solves

$$
\Big(J_{}\TJ_{}+\lambda\,D_{z}+\varepsilon\,D_{y}\Big)
  \begin{bmatrix}\deltaz\\ \delta y\end{bmatrix}=-J_{}\operatorname{tr}_{w,n},
  \qquad \lVert \deltaz \rVert\le\Delta ,
$$

<!-- equation (12) -->

where $D_{z}=\operatorname{diag}(\operatorname{diag}(J_{z}\TJ_{z}),0)$ carries the
Levenberg–Marquardt damping on the latent block only,
$D_{y}=\operatorname{diag}(0,I)$ with a fixed small ridge $\varepsilon$, and $\Delta$ is
the $q=0$ trust radius. The correction step is therefore an undamped
Gauss–Newton step on the current linearisation, not throttled by a radius
calibrated for the latent code. The step is accepted only if the residual
decreases; otherwise $\lambda$ grows as in §A.6.
Table 23 compares it with joint damping and with
variable projection.

### A.4 Second-order instance: reflective waves

<!-- section sources: none (prose only) -->

For $u_{tt}=c^2\Delta u$ the bank is mass-orthonormal and the $R$ weak
tests are the bank itself, so projection is Galerkin with stiffness
$K=G^{\top} M_hAG$, $M_h$ the diagonal mass matrix. Writing the coefficients as
$a(z,y)=h_\theta(z)+C_q y$ and differentiating twice in time,
$\ddot a=Dh_\theta \ddotz+h_\theta”(z)[\dotz,\dotz]+C_q\ddot y$,
the projected equation $\ddot a=-c^2Ka$ becomes a linear least-squares
problem for the joint acceleration,

$$
\big[\,Dh_\theta(z)\;\;C_q\,\big]
  \begin{pmatrix}\ddotz\\ \ddot y\end{pmatrix}
  = -\Big(c^2K\,a+h_\theta''(z)[\dotz,\dotz]\Big),
$$

<!-- equation (13) -->

whose curvature term $h_\theta”[\dotz,\dotz]$ is a
forward-over-forward derivative of the head. The first-order system in
$(z,y,\dotz,\dot y)$ is advanced with classical RK4 at a fixed
step, one solve of (13) per stage; there is no implicit
residual minimisation and no stationarity exit. Initial coordinates and
velocities are obtained from the supplied displacement and velocity. At $q=R$
the head is dropped and $\ddot a=-c^2Ka$ is propagated exactly through the
eigendecomposition $K=V\Lambda V^{\top}$,
$a(t)=V\big(\cos(\omega t) \hat a_0+\omega^{-1}\sin(\omega t) \hat b_0\big)$,
$\omega=c\sqrt{\Lambda}$. The iterative full-order comparator uses implicit
midpoint with CG. The error reported in Table 3 is the
energy-state error over displacement and velocity.

### A.5 Empirical quadrature: the fit and the counts

<!-- section sources: none (prose only) -->

A rule is judged by how well its weighted sum over $m$ nodes reproduces
the tested advection term of the full grid. That relative error, on one
state, is

$$
\rho(u)=\frac{\lVert \sum_{i\in\mathcal S}w_iP_{:,i}N_i(u)-P N(u) \rVert_2}{\lVert P N(u) \rVert_2},
  \quad
  \rho_{\max}\le0.116\ \text{(primary bar)},\quad \rho_{\max}\le0.06\ \text{(tight)},
$$

<!-- equation (14) -->

over the held-out reachable states described below. The primary bar is
the held-out $\rho$ of the original model's $q=0$ rule at the state that
carries its largest first-interval error, measured in an earlier diagnostic
study and fixed before any certification job ran; the tight bar was fixed
before the EQ-ladder job of Table 22. A candidate node set $\mathcal C$ is drawn once with a fixed seed. For each of
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

<!-- equation (15) -->

rows normalised by their Euclidean norms, support grown greedily by the
Lawson–Hanson criterion with an exact non-negative refit after each addition
and a hard cap of $m$ nodes, padded to the requested count if the support
saturates. Three counts are separated: the requested node count $m$, the
achieved support, and the number of fit states $n_{\text{fit}}$. In the
certification study the fit states are reachable states (states of the dense
solver's own converged per-step trajectory on fit trajectories), the held-out
states are 512 reachable states per rung from certification trajectories
disjoint from both the fit and the evaluation cases, and $\rho$ of
(14) is evaluated on those. A deployed rule is a stored triple: node
indices, weights, and the cached $m\times5\times R$ bank block.

### A.6 Solver constants and exit codes

<!-- section sources: none (prose only) -->

The representation ablations separate three error quantities: the
*bank floor*, the projection error onto $\operatorname{range}G$,
which no head or solver can beat; the *best-found* error, the
smallest any point of the augmented manifold attains, located by a
multistart oracle; and the *solved* error, what the deployed
iteration actually returns.

Write $J_{}$ for the Jacobian of $r_{w}$ in $(z,y)$. Each
Levenberg–Marquardt attempt solves
$(H+\lambda \operatorname{diag}(\operatorname{diag} H)) \delta=-g$ with $H=J_{}\TJ_{}$ and
$g=J_{}^{\top}r_{w}$, and accepts $\delta$ only if the residual strictly
decreases. The stationarity measure of §3.2 is scale free: it
divides the gradient norm by the sizes of the Jacobian and the residual,

$$
\eta(z,y)=\frac{\lVert J_{}^{\top}r_{w} \rVert_2}{\lVert J_{} \rVert_{F}\,\lVert r_{w} \rVert_2}
  \;\le\;\eta_{\mathrm{tol}} ,
$$

<!-- equation (16) -->

so one tolerance $\eta_{\mathrm{tol}}$ serves every mesh and every rank.
The elliptic solve starts from the cached training code nearest the
projected source; the heat and Burgers queries start from the nearest
codes and fit $(z,y)$ to the supplied initial field.

Damping starts at $\lambda_0=10^{-6}$ (Poisson, Burgers) or $10^{-4}$ (heat);
$\lambda\leftarrow\max(\lambda/3,10^{-12})$ on acceptance and
$\min(10\lambda,\lambda_{\max})$ on rejection; the trust radius is taken from the
spread of the training codes. Exit codes differ per problem and are reported per
problem. Burgers: 4 stationarity, 1 residual, 2 tiny step, 3 rejected at
$\lambda\ge10^{14}$, 0 budget. Poisson: 6 stationarity, 2 relative residual, 1
stall, 3 damping limit, 5 non-finite initial value, 0 budget. Heat: 1
stationarity, 2 tiny step, 3 damping limit, 4 non-finite, 0 budget. The heat
criterion divides by $\lVert J_{} \rVert_{F}$ only, applied to a residual already
normalised by the target norm; Poisson and Burgers use (16)
directly. An initial fit that meets its stopping rule is accepted as the
starting state of the time loop; this acceptance rule was fixed before the
evaluation jobs ran, and complete exit flags remain in the source records.

## B Architecture and online data flow

<!-- section sources: none (prose only) -->

Figure 2 shows the offline preparation and the online solve
of every NM-ROM query in the paper.

![Figure 2](figures/architecture.png)

**Figure 2.** NM-ROM from training to prediction, read left to right. Offline
(blue), training fixes the spatial bank $G$, the head $h_\theta$, the
correction directions $C_q$ and the weak tests; these never change
again. Online (orange), the PDE inputs do two things: they set the
starting reduced coordinates $(z_0,y_0)$, taken from stored training
codes, and they enter the weak residual directly. The reduced solve then
adjusts the $k+q$ unknowns $(z,y)$ until that residual is small, and the
reconstruction returns $u=G[h_\theta(z)+C_q y]$ at the requested
times. Time-dependent problems repeat the solve once per step.
Initialisation and the reduced solver are PDE-specific, and for linear
PDEs the correction coefficients are eliminated analytically rather than
iterated. Empirical quadrature (EQ) is an optional way to evaluate the
residual, used for two-dimensional Burgers; the three-dimensional and
wave solves evaluate it densely. Correction rank changes what the model
can represent; EQ changes only how the residual is evaluated. Baselines
are separate methods, not stages of this pipeline.

## C Neural operators in three dimensions

<!-- section sources: none (prose only) -->

Four neural operators were trained on the same data, split and wall budget
as the three-dimensional models of this paper and evaluated in the same
allocation: an FNO (Li et al., 2021), a U-Net (Ronneberger et al., 2015),
a DeepONet (Lu et al., 2021) and a Transolver (WuTransolver2024, ?).
Table 5 and Table 6 report worst
same-grid error, median GPU query time and the speedup over the same
full-order control used elsewhere in the row's problem. They win where
our three-dimensional models fail: on Burgers 3D the best of them,
\nOpBurgBestName, reaches $\nOpBurgBestErr %$ at $\nOpBurgBestS\times$
Newton–BiCGStab against $\nFailBurgersAcc %$ for the NM-ROM, and the
fastest is $\nOpBurgFastS\times$; on Navier–Stokes \nOpNsBestName
reaches $\nOpNsBestErr %$ against $\nFailNsAcc %$. On Poisson 3D the
corrected NM-ROM is the more accurate, and no operator is faster than CG
(at most $\nOpPoissonFastS\times$); on heat the best operator error is
$\nOpHeatBestErr %$, with no CG measurement in that record and so no
speedup. Five facts frame these numbers.

First, every operator here is trained and evaluated at $32^3$. The same
trained operators transfer badly to $64^3$: worst errors of
$\nOpTransferLo$–$\nOpTransferHi %$ across the four families,
recovered to $\nOpNativeLo$–$\nOpNativeHi %$ only by training on the
native grid and interpolating (for the FNO, also by physical padding).
The top of that recovered range is again DeepONet. The fine-mesh rows of Table 1 are
therefore NM-ROM against the named solvers alone; no operator was trained
at those meshes.

Second, DeepONet is the weakest arm everywhere here (worst
$\nOpDeepWorst %$), and in three dimensions it is limited by the
budget: every arm stopped on its step budget, recorded with its exit
reason in Table 9. We do not read it as an
architecture ceiling in either dimension.

Third, the operators produce the requested output times directly, in one
pass. They are not autoregressive rollouts, so the set of output times is
fixed when they are trained.

Fourth, each arm's stop reason is recorded with its budget
(Table 9): every three-dimensional arm there
stopped on its step budget rather than by early stopping, while the
two-dimensional arms above train with an early-stopping patience. At
equal wall time the float32 U-Net and Transolver completed more epochs
than the float64 FNO, which favours them.

Fifth, an operator's own evaluation resolution can trade accuracy for
cost, as §2 says; the comparison here fixes one
evaluation setting per trained operator.

**Table 5.** Three-dimensional linear problems at $32^3$, $\nOpCasesDev$
development cases: worst same-grid relative $L^2$ error, median GPU query
time and speedup over the problem's full-order control. Poisson uses the
steady error and CG at rtol $10^{-2}$ as the control; the heat record has
no CG measurement, so no heat speedup is inferred.

<!-- table: TR_3d_linear -->
| Method | Error (\%) | GPU ms | $S$ | Error (\%) | GPU ms | $S$ |
| --- | --- | --- | --- | --- | --- | --- |
| NM-ROM, uncorrected | 0.589 | 2.450 | 1.06$\times$ | 6.013 | 34.771 | --- |
| NM-ROM, corrected | 0.161 | 2.632 | 0.986$\times$ | 1.379 | 114.343 | --- |
| POD | 0.037 | 0.232 | 11.2$\times$ | 1.244 | 0.189 | --- |
| FNO | 0.237 | 11.409 | 0.227$\times$ | 0.395 | 10.638 | --- |
| U-Net | 0.193 | 4.099 | 0.633$\times$ | 0.318 | 4.373 | --- |
| DeepONet | 1.807 | 2.599 | 0.998$\times$ | 21.769 | 3.889 | --- |
| Transolver | 0.530 | 4.135 | 0.627$\times$ | 1.483 | 6.282 | --- |
| FOM | 0.140 | 2.594 | 1$\times$ | --- | --- | --- |

**Table 6.** Three-dimensional nonlinear problems at $32^3$: Burgers on its
accepted final cohort ($\nOpCasesFinal$ cases, worst evolved error) and
Navier–Stokes on a development cohort. Speedups divide the named
full-order solver of the same problem in the same allocation.

<!-- table: TR_3d_nonlinear -->
| Method | Error (\%) | GPU ms | $S$ | Error (\%) | GPU ms | $S$ |
| --- | --- | --- | --- | --- | --- | --- |
| NM-ROM, uncorrected | 16.263 | 219.161 | 0.038$\times$ | 13.535 | 1253.205 | 0.0025$\times$ |
| NM-ROM, corrected | 4.399 | 364.677 | 0.0229$\times$ | 12.534 | 4622.722 | 0.000677$\times$ |
| POD | 2.561 | 310.989 | 0.0268$\times$ | 23.732 | 30.460 | 0.103$\times$ |
| FNO | 2.378 | 6.072 | 1.37$\times$ | 0.470 | 8.903 | 0.352$\times$ |
| U-Net | 2.451 | 2.115 | 3.94$\times$ | 2.044 | 1.934 | 1.62$\times$ |
| DeepONet | 20.575 | 2.578 | 3.23$\times$ | 26.933 | 4.261 | 0.735$\times$ |
| Transolver | 2.017 | 3.489 | 2.39$\times$ | 2.691 | 3.361 | 0.931$\times$ |
| FOM | 2.312 | 8.337 | 1$\times$ | 0.628 | 3.130 | 1$\times$ |

**Two dimensions at $256^2$.**
The Burgers operators were also timed against the NM-ROM, POD and the
full-order solver in one allocation (job \nOpsTwoDJob,
\nOpsTwoDCases development cases, \nOpsTwoDReps retained repetitions
each; Table 8). The fast operators are quicker than the
NM-ROM and than the full-order setting the rule matches to them
(\nOpsTwoDOpFaster of \nOpsTwoDOpArms operator arms beat it, up to
$\nOpsTwoDOpFastS\times$), and an order of magnitude less accurate:
$\nOpsTwoDOpErrLo$–$\nOpsTwoDOpErrHi %$ worst evolved error, against
$\nOpsTwoDAccErr %$ for the accurate NM-ROM setting and
$\nOpsTwoDFastErr %$ for the fast one. On these \nOpsTwoDCases cases, and against this
panel's own reference, every operator arm's error also exceeds the
discretisation error of the mesh itself, $\nOpsTwoDDisc %$: on this
cohort they do not resolve the discrete solution they were trained on.
That comparison belongs to this cohort and this
reference, and it does not generalise in either respect. The operator
percentages here are scored against a converged solve on the same grid
in the same job, while $\nOpsTwoDDisc %$ is what that grid itself
costs against a finer reference, so the two were never subtractable.
Cohort alone moves an arm by more than the gap: the same
\nOpsBarBelowValName checkpoint scores $\nOpsBarSameMatch %$ on the
matched eight cases and $\nOpsBarSamePanel %$ here, a factor of
$\nOpsBarSameGap$. Against the same number,
\nOpsBarBelowValName is lower on the 32 validation cases of
Table 7 ($\nOpsBarBelowValErr %$ worst), and on the
matched eight cases all \nOpsBarBelowMatch of the \nOpsBarPubArms
published U-Net, Transolver and FNO arms are lower. A like-for-like
re-measurement on the validation cases is in progress. No
NM-ROM or POD arm here is faster than its comparator either; the
accurate setting meets the campaign's $\le 1 %$ accuracy bar and misses
its $\ge 5\times$ speed bar ($\nOpsTwoDAccS\times$), which is what
Table 1 already shows for accurate Burgers below
$4096^2$.

Two asymmetries are declared and left in place, both favouring the
operators: their complete-query scope is not charged the upload of the
input field, and the fields their errors are scored on come from an
extra, untimed query. The FNO error reproduces the earlier panel case by
case (difference $\nOpsTwoDFnoGap$), and no timing was ever divided
across jobs. This lane also could not reproduce an earlier
validation-selection pathology: here the selected arms are the better
ones. The two cohorts are different splits with no case in common, so
that is cohort-specific and refutes nothing.

**How much data each side was given.**
The operators were trained on \nParOpTraj trajectories; the model they
are compared against was fitted on \nParRomTraj. That gap is an
artefact of what one training sample costs on each side, not a decision
about how much data each method was allowed, and it runs against the
operators' interest rather than for it. Both sides draw initial
conditions from the same Gaussian-bump family through the same generator
and are scored at the same six output times, but the operators' training
targets were held to the stricter reference: each of their cases is a
$4096$-interval solution at $\Delta t=1.5625{\times}10^{-4}$ restricted
to the $256$-interval evaluation grid, costing a median of
$\nParOpSec$ s of A100 time per trajectory, while each trajectory
behind the bank and the head was solved directly on the $256$-interval
grid at $\Delta t=0.005$ for about $\nParRomSec$ s, a factor of roughly
$\nParCostRatio$. At that rate the operators' \nParOpTraj cases are
what about $\nParOpHours$ GPU-hours buys, and matching
\nParRomTraj trajectories at the same fidelity would have cost about
$\nParFullHours$ A100-hours. The two sides also count data
differently: an operator takes one input–output pair per trajectory and
is supervised on the \nParPerCase evolved fields, so \nParOpTraj
cases are \nParOpStates supervised states, while the head is fitted
per state and sees \nParRomStates of the \nParStatesAvail states its
\nParRomTraj trajectories contain. The $\nParTrajRatio\times$
trajectory count should therefore not be read as a data advantage on
equal terms. The trajectory and state counts on our side, and the
$\nParRomSec$ s, are quoted from job \nParRomJob's own record and
were not re-derived from anything else in this repository; every other
figure here is re-derived from the pinned source.

**DeepONet in two dimensions.**
DeepONet was the one operator named in our abstract that had never been
trained on this 2D family. It now has been, on the same data, split,
metric, optimiser, seed and wall budget as the U-Net, Transolver and FNO
arms (job \nDonJob; Table 7). Its \nDonArms arms
reach $\nDonMeanLo$–$\nDonMeanHi %$ mean validation error, against
$\nDonOtherMeanLo$–$\nDonOtherMeanHi %$ for the selected arms of the
other three families, and $\nDonMatchLo$–$\nDonMatchHi %$ worst error
on the matched eight-case cohort where the NM-ROM reaches
$\nDonRomMatch %$. A persistence control, which returns the supplied
field at every output time, scores $\nDonPersistMean %$, so these arms
are about $\nDonVsPersist\times$ better than doing nothing.

Four qualifications make this weak evidence about DeepONet, and we state
them rather than drop the row; the paper's claim is about tunability,
not about beating a baseline. All \nDonArms arms ended by early
stopping, not on the wall budget, so the budget did not bind and their
errors are not a budget-imposed floor. The lane's “still improving”
flag is vacuous at this patience: with patience \nDonPatience it could
only be set above about 5000 epochs, against a cap of \nDonEpochCap,
and the training loss was still falling in all four histories, so this
is not early stopping at convergence. Training used \nDonTrainCases
cases, at $\nDonTrainRmsLo$–$\nDonTrainRmsHi %$ training error
against the validation errors above, so a data-limited reading is live
and this lane does not separate it from a capacity reading; three
coupled capacities are not a capacity sweep. And no speed number from
this lane is admissible, so none is claimed. As in three dimensions,
this is not an architecture ceiling: no DeepONet-specific schedule or
hyperparameter search was run.

**Table 7.** DeepONet on Burgers 2D at $256^2$ (job \nDonJob), beside the
selected arm of each other family, the NM-ROM on the matched cohort and
a persistence control. Fixed-initial relative error over the six
requested output times, on the 32 validation cases and on the eight
matched cases. Accuracy only: this lane takes no timing, and accuracy is
comparable across these jobs while timing is not. “Ended by” is the
recorded stop reason of each arm.

<!-- table: TR_deeponet2d -->
| Arm | Ended by | Val. mean (%) | Val. median (%) | Val. worst (%) | Matched-8 worst (%) |
| --- | --- | --- | --- | --- | --- |
| DeepONet, small (selected) | early stopping | 14.79 | 11.75 | 54.74 | 16.02 |
| DeepONet, medium | early stopping | 14.90 | 12.73 | 51.79 | 18.98 |
| DeepONet, refined | early stopping | 15.76 | 12.46 | 58.98 | 21.37 |
| DeepONet, large | early stopping | 18.22 | 14.80 | 60.35 | 26.63 |
| U-Net, refined | wall budget | 1.35 | 1.08 | 7.52 | 1.71 |
| Transolver, refined | wall budget | 1.95 | 1.36 | 9.32 | 1.52 |
| FNO, large | wall budget | 2.28 | 1.81 | 6.38 | 2.48 |
| NM-ROM (same cohort) | --- | --- | --- | --- | 1.87 |
| Persistence control | --- | 64.68 | 65.52 | 90.65 | 75.00 |

**Table 8.** Burgers 2D at $256^2$ in one allocation (job \nOpsTwoDJob):
worst evolved and all-times same-grid error, median GPU query time, the
full-order setting the rule matches to each row, and the speedup over
it. Operator arms are the trained sizes of each family; the last three
rows are the full-order settings the rule selects. Complete-query times
add host transfer and are in the lane record.

<!-- table: TR_ops256 -->
| Arm | Family | Evolved (%) | All times (%) | GPU ms | FOM by the rule | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| NM-ROM, fast | NM-ROM | 1.89 | 2.56 | 38.6 | nt1e-3_dt01 | 0.50× |
| NM-ROM, accurate | NM-ROM | 0.51 | 0.91 | 465 | nt1e-3_dt005 | 0.07× |
| POD-LSPG, 256 modes | POD-LSPG | 0.71 | 3.77 | 898 | nt1e-3_dt005 | 0.03× |
| POD-LSPG, 512 modes | POD-LSPG | 0.22 | 0.61 | 2770 | nt1e-3_dt005 | 0.01× |
| FNO | FNO | 7.42 | 7.42 | 7.3 | nt1e-2_dt01 | 1.18× |
| U-Net, small | U-Net | 5.14 | 5.14 | 9.4 | nt1e-2_dt01 | 0.92× |
| U-Net, medium | U-Net | 4.76 | 4.76 | 5.8 | nt1e-2_dt01 | 1.48× |
| U-Net, large | U-Net | 4.49 | 4.49 | 10.0 | nt1e-2_dt01 | 0.86× |
| U-Net, refined | U-Net | 4.55 | 4.55 | 5.9 | nt1e-2_dt01 | 1.48× |
| Transolver, small | Transolver | 4.90 | 4.90 | 11.0 | nt1e-2_dt01 | 0.78× |
| Transolver, medium | Transolver | 5.96 | 5.96 | 12.8 | nt1e-2_dt01 | 0.67× |
| Transolver, large | Transolver | 5.83 | 5.83 | 15.3 | nt1e-2_dt01 | 0.57× |
| Transolver, refined | Transolver | 4.46 | 4.46 | 11.1 | nt1e-2_dt01 | 0.78× |
| nt1e-2_dt01 | FOM | 3.20 | 3.20 | 8.7 | --- | 1× |
| nt1e-3_dt01 | FOM | 1.52 | 1.52 | 19.4 | --- | 1× |
| nt1e-3_dt005 | FOM | 0.049 | 0.049 | 30.7 | --- | 1× |

**Table 9.** Recorded training budgets of the neural operators: parameters,
step budget, steps completed, the step selected by validation, batch,
learning rate, seed and exit reason. Every arm exits on its step budget.

<!-- table: TC_development_training -->
<!-- Generated from hash-pinned campaign evidence. -->
| PDE | Operator | Parameters | Budget | Done | Selected | Batch | LR | Seed | Exit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Burgers 3D | fno | 1771431 | 20000 | 20000 | 16100 | 2 | 0.001 | 920310 | steps |
| Burgers 3D | unet | 89431 | 20000 | 20000 | 17500 | 2 | 0.001 | 920311 | steps |
| Burgers 3D | deeponet | 1861959 | 30000 | 30000 | 30000 | 2 | 0.001 | 920312 | steps |
| Burgers 3D | transolver | 564392 | 30000 | 30000 | 29400 | 2 | 0.001 | 920313 | steps |
| Heat 3D | fno\_w16\_m6 | 1771349 | 20000 | 20000 | 19500 | 4 | 0.001 | 920321 | steps |
| Heat 3D | unet\_w8 | 89197 | 20000 | 20000 | 20000 | 4 | 0.001 | 920322 | steps |
| Heat 3D | deeponet\_r128\_w16 | 857749 | 20000 | 20000 | 20000 | 4 | 0.001 | 920323 | steps |
| Heat 3D | transolver\_w48\_s32 | 562840 | 20000 | 20000 | 20000 | 4 | 0.001 | 920324 | steps |
| Poisson 3D | fno\_w24\_m8 | 9440953 | 30000 | 30000 | 26750 | 2 | 0.0005 | 920450 | steps |
| Poisson 3D | unet\_w16 | 354577 | 30000 | 30000 | 28750 | 2 | 0.0005 | 920451 | steps |
| Poisson 3D | deeponet\_r128\_w16 | 791697 | 30000 | 30000 | 29000 | 2 | 0.001 | 920452 | steps |
| Poisson 3D | transolver\_w48\_s32 | 561272 | 30000 | 30000 | 29750 | 2 | 0.001 | 920453 | steps |
| Navier--Stokes 3D | deeponet | 6790319 | 32000 | 32000 | 23200 | 2 | 0.001 | 202609208 | steps |
| Navier--Stokes 3D | transolver | 569064 | 32000 | 32000 | 32000 | 2 | 0.001 | 202609209 | steps |
| Navier--Stokes 3D | fno | 4196511 | 8000 | 8000 | 7100 | 2 | 0.001 | 202609205 | steps |
| Navier--Stokes 3D | unet | 89287 | 8000 | 8000 | 7900 | 2 | 0.001 | 202609206 | steps |

## D Experimental configuration and reproducibility

<!-- section sources: none (prose only) -->

Table 11 and Table 12 identify the two-dimensional
studies and Table 20 the three-dimensional ones. Within each study,
training precedes evaluation and the selected bank, head and correction
directions remain frozen. Deployment changes reduced coordinates, not
network weights. Recorded training configurations and available offline
costs are retained with the source evidence; unrecorded training costs
are not inferred.
\input{tables/TH_training}

**Table 10.** Training configurations of the frozen checkpoints, transcribed from
the committed configuration and code files of each study (paths and
SHA256 in `evidence/training-configs-2026-09-21`).

<!-- table: TH_training_all -->
| Model | Stage | Network | Optimiser, learning rate | Steps | Batch | Training data |
| --- | --- | --- | --- | --- | --- | --- |
| Burgers 2D (k=16, R=512) | bank | 128 Fourier features (scale 4), width 1024, 2 layers | AdamW, wd 10^{-5}, warm-up + cosine 10^{-3}\to10^{-5} | 300000 | all states \times 4096 points | 16384 states of 576 trajectories |
|  | head | width 512, 2 layers | Adam, warm-up + cosine 10^{-3}\to10^{-5} | 200000 | 4096 states | 131072 states of 4608 trajectories |
| Poisson 2D (k=32, R=512) | bank | 64 Fourier features (scale 4), width R, 2 layers | Adam, four phases at 10^{-3}, 3{\times}10^{-4}, 3{\times}10^{-4}, 10^{-4} | 100000, 40000, 30000, 30000 | 64 sources \times 4096 points | 2611 fit / 461 validation sources |
|  | head | width 128, 2 layers | Adam, 10^{-3} | 150000 (full batch) | all | as bank |
| Poisson 2D (k=16, R=128) | bank + head | 64 Fourier features, g and h width 128, 2 layers | Adam; staged from an R=64 model (bank, head, joint at 3{\times}10^{-4}, 3{\times}10^{-4}, 10^{-4}) | 40000, 30000, 20000 after widening | 64 sources \times 4096 points | 512 training sources |
| L-shape (k=16, R=512) | bank / head | 64 Fourier features (scale 4), 2 layers; head width 128, 2 layers | as Poisson k=32 | 100000, 40000, 30000, 30000; head 150000 | 64 sources \times 4096 points | 2611 fit / 461 validation sources |
| Poisson 3D (k=16, R=128) | bank / head | 64 Fourier features (scale 1.5), widths 256; head width 256 + skip | Adam, cosine (floor 0.03); 10^{-3} bank, 5{\times}10^{-4} head | 150000 / 100000 | 64 states \times 2048 points / 128 states | 512 training / 16 validation fields |
| Heat 3D (k=32, R=128) | bank / head | 32 Fourier features (scale 1.5); head width 256 + skip | Adam, cosine (floor 0.03); 10^{-3} bank, 5{\times}10^{-4} head; clip 1, refit every 10000 | 150000 / 150000 | 64 states \times 2048 points / 128 states | 512 training / 16 validation trajectories |

\input{tables/TH_solver_details}

**Three-dimensional heat, new bank.** The bank of the Heat (new
bank) rows ($R=\nHeatNewR$) was trained with a code-free
variable-projection trainer that eliminates the coefficients exactly;
its projection floor is $\nHeatNewFloor %$. Its arms were fixed by a
rule registered before the sealed cohort was opened. A second model
($R=\nHeatNewBR$, $k=\nHeatNewBK$, $q=\nHeatNewBQ$), chosen after speed results had been seen,
reaches $\nHeatNewBErr %$ on the same sealed cohort at $128^3$,
$\nHeatNewBCnS\times$ (Crank–Nicolson) and $\nHeatNewBBfS\times$ (batched
fit) against the rule's CN–CG setting; it is not used in
Table 1.

**Table 11.** Problem specification. Cohort and reduced sizes are read from the run
configurations where recorded. The Burgers sealed cohort has been opened
and is reported in Table 21; other rows describe their
recorded development and validation cohorts.

<!-- table: T01_problems -->
| PDE | equation, domain, boundary | meshes | time stepping | reduced sizes | reference | cohorts |
|---|---|---|---|---|---|---|
| Burgers 2D | $u_t+u(u_x+u_y)=\nu\Delta u$, $(0,1)^2$, $u\|_{\partial\Omega}=0$ | $256^2$ (ladder 64–1024) | $\Delta t=0.005$, backward Euler, sign-upwind | $k=16$, $R=512$ | refined $ 4096^2$, $\Delta t=0.00015625$ | 6 development cases; 32 validation; 64 held-out at $2048^2$, $4096^2$; sealed cohort opened once |
| Poisson 2D | $-\Delta u=f$, $(0,1)^2$, $u\|_{\partial\Omega}=0$ | $256^2$, $1024^2$ | none (elliptic) | $K=16$, $R=128$ (incumbent); $K=32$, $R=512$ | exact discrete solution; 2048$^2$ refinement | 12 development sources |
| Heat 2D | $u_t=\kappa\Delta u$, $(0,1)^2$ | $64^2$–$1024^2$ | Crank–Nicolson | $k=8$, $R=32$ | refined-grid reference ($1024^2$/$2048^2$ pair); error includes discretisation | 12 development cases, 3 repetitions; measured in job 3529772; checkpoint lineage: earlier cell, job 3511417 |
| Wave 2D (reflective) | $u_{tt}=c^2\Delta u$, $(0,1)^2$, $u\|_{\partial\Omega}=0$ | $64^2$, $256^2$, $1024^2$ | RK4 on the manifold; exact modal propagation for the bank | $k=32$, $R=64$ | direct DST | 8 development cases |
| Poisson, L-shape | $-\Delta u=f$, $(0,1)^2\setminus[\tfrac12,1)^2$ | $256^2$, $512^2$ | none | $K\in\{16,32\}$, $R\in\{256,512,514\}$ | converged discrete solution | 3072 / 256 / 32 sources (train / selection / development) |

**Table 12.** Sampled problem families, transcribed from the generator sources named
in the last column (code constants, not run outputs).

<!-- table: T01b_spec -->
| PDE | equation and boundary | sampled family (transcribed from the generator source) | source |
|---|---|---|---|
| Burgers 2D | $u_t+u(u_x+u_y)=\nu\Delta u$ on $(0,1)^2$, $u=0$ on $\partial\Omega$ | $u_0=a\exp(-\lvert x-c\rvert^2/2w^2)$, $c_i\sim U(0.15,0.85)$, $w\sim U(0.05,0.20)$, $a\sim U(0.5,2)$; $\nu\sim\log U(0.01,0.1)$; outputs $t\in\{0.05,\dots,0.25\}$ | `experiments/mr-burgers2d/engines.py:params_draw` |
| Poisson 2D | $-\Delta u=f$ on $(0,1)^2$, $u=0$ on $\partial\Omega$ | $f=a\exp(-\lvert x-c\rvert^2/2w^2)$, $c_i\sim U(0.15,0.85)$, $w\sim\log U(0.02,0.1)$, $a\sim U(0.5,2)$ | `multistage-precision/ms_parametric.py:sample_params` |
| Poisson, L-shape | same source family on $(0,1)^2\setminus[\tfrac12,1)^2$ | as above; sources centred in the removed quadrant rejected | `experiments/lshape` |
| Heat 2D | $u_t=\kappa\Delta u$ on $(0,1)^2$, $\kappa=0.02$ fixed | polynomial-boundary Gaussian family (single_bc_poly_gaussian_v1); Crank–Nicolson $\Delta t=0.025$ | checkpoint `expanded_seed790715` (lineage: linear-bank cell, job 3511417); printed CG measurements: job 3529772 |
| Wave 2D (reflective) | $u_{tt}=c^2\Delta u$ on $(0,1)^2$, $u=0$ on $\partial\Omega$ | compact bump $\times$ Gaussian: half-widths $s_i\sim U(0.36,0.42)$, centre $c_i\sim U(s_i{+}0.025,\,1{-}s_i{-}0.025)$, amplitude $\sim U(0.7,1.3)$, $\sigma_i\sim U(0.12,0.16)$, advective velocity $v_i\sim U(-0.5,0.5)$ (zero every fourth case); speed $c\sim U(0.85,1.15)$ | `experiments/multiresolution-wave/audit_dynamics.py:parameter_rows` |

### D.1 Headline settings, times and timing protocol

<!-- section sources: none (prose only) -->

Table 1 markers.
$^{f}$ held-out final cohort (all others development);
$^{r}$ error against a refined reference, which includes discretisation
error (all others same-grid); $^{t}$ all output times including
$t=0$; $^{e}$ evolved output times only; $^{c}$ complete-query time
with host transfers (all others GPU query; heat was measured in GPU query
only). Heat (wide bank) is a separately trained frozen model with a
128-function bank ($q=32$ accurate); over evolved times its accurate
error is $\nHeatWideAccEvolved %$ ($\nHeatWideBatchedAccEvolved %$
batched; the batched fit: Table 13); Heat (new
bank) ($q=\nHeatNewQ$ accurate; Table 20): at most $\nHeatNewAccEvolved %$
($\nHeatNewBatchedAccEvolved %$ batched) over evolved times;
$^{n}$ \nHeatNewNonstat solve (one step of one case) missed its
stationarity rule, so no speedup is given. Burgers accurate
quadrature: $^{s}$ stored rule of job 3780164
($m=\nEqcLaneRuleM$; not the $q=256$ rule of Table 22),
not confirmed on re-draws: it fails the confirmation draw at $256^2$;
$^{x}$ its nodes with weights refit at $512^2$
($m=\nEqcRowRuleMFiveTwelve$), one held-out draw, not re-drawn (with
unrefit weights they pass \nEqcScaledPassFiveTwelve of
\nEqcScaledDrawsFiveTwelve re-draws); $^{\ell}$ $63{\times}63$
lattice rule at $2048^2$, thin margin: it clears the re-draw bar by only
$\nEqcLatMargin$ ($\rho_{\max}=\nEqcLatConfRho$ vs. $\nEqcBar$);
$^{w}$ the same rule at $4096^2$: five held-out draws pass and the
confirmation draw does not; $^{v}$ confirmed by the pre-registered
re-draw procedure ($1024^2$: thin margin,
$\rho_{\max}=\nEqcConfRhoTenTwentyFour$; none passes at $256^2$);
$^{d}$ dense residual. The earlier Burgers model has one setting and
may exit on a stall; its rule-selected FOM is the relaxed Newton setting
($10^{-2}/0.5$), and its tight-setting ratio is in
Table 13 with the other rows' tight ratios; the
FOM setting can differ between meshes of a series. $^{h}$ 64 held-out
cases never used for selection (not the sealed final cohort); at $2048^2$
the development-chosen $M=544$ setting was not run on them, so the
$M=1088$ setting is shown. Poisson (dev. sources) rows use development
sources of a separate run and are not the held-out cohort.

Table 13 gives the settings and measured times
behind every row of Table 1. Every cost and error come
from the same invocation. Timings use GPU warm-up and synchronisation,
double precision and the highest matrix-multiplication precision; all
repetitions, including outliers, enter the median, and no time is
borrowed from another allocation. Requested full fields and
initialisation are charged; host transfer is included only where the
table says complete query. For the held-out Burgers rows, an independent
restricted-grid recomputation of the errors disagreed with the full-grid
value by more than 5 % on $\nBurgGateBadTwentyFortyEight/\nBurgGateRowsTwentyFortyEight$
($2048^2$) and $\nBurgGateBadFortyNinetySix/\nBurgGateRowsFortyNinetySix$
($4096^2$) case–arm rows; per-arm cohort worsts agree within
$\nBurgGateWorstPct %$ and the full-grid recomputation is exact. For Burgers at $256^2$ no quadrature rule passed the pre-registered
re-draw procedure (its pick failed the confirmation draw), so
Table 1 has no confirmed-rule row there; a post-hoc
follow-up rule with an exact first time step passes all
\nEqcFollowDraws draws of two jobs and gives $\nEqcFollowErr %$ at
$\nEqcFollowS\times$ against the same Newton–BiCGStab rule. For Burgers at $4096^2$, the lattice quadrature rule used in the accurate
rows passes checks on five held-out trajectory samples but fails on the
independent confirmation sample ($\rho_{\max}=\nBhJzeroRho$ against the
threshold $\nEqcBar$). Restricting the same sample to time step 2 onward
gives $\rho_{\max}=\nBhJzeroRhoKtwo$, locating the failure in the first
steps. The rule is validated on held-out states but not confirmed on
independent re-draws.

Using the exact residual at the first time step and the same quadrature
rule thereafter passes all five checks and the confirmation check
($\rho_{\max}=\nBhJoneRho$). The error is unchanged at the reported
precision. Median query times are $\nBhJoneMsDev$ ms on the six
development cases and $\nBhJoneMsHold$ ms on the 64 held-out cases.
The corresponding speedups against Newton–BiCGStab are
$\nBhJoneSDev\times$ and $\nBhJoneSHold\times$.
These queries cost $\nBhJoneCostDev\times$ and $\nBhJoneCostHold\times$
as much as queries using quadrature at every step in the same job
(Table 13). This variant is slower than the
full-order solver on both cohorts.
No confirmed form of this accurate lattice rule is also faster. In the
wider bank, the settings whose quadrature is not confirmed reach
$\nBhSubErrLo$–$\nBhSubErrHi %$ error at
$\nBhSubSLo$–$\nBhSubSHi\times$ the speed of Newton–BiCGStab.
The fast setting's smaller rule passes the confirmation check and is
faster than the full-order solver, at higher error.

For Burgers at $4096^2$, copying the six
double-precision output fields to the host adds about
$\nBurgHostGapMs$ ms to every arm. The Poisson and three-dimensional heat CG comparators are
unpreconditioned; heat CG is warm-started from the previous time level. The Heat2D rows use checkpoint
`expanded_seed790715`; Table 11 separates that
checkpoint's lineage from the job that produced the printed
measurements.

**Table 13.** Supporting data for Table 1, including the full-order setting the rule selected for each row: the two settings
of each frozen model, median times (ms), timing scope and evidence status
(allocations: Appendix D). “EQ” and “dense” name the Burgers residual
evaluation; “single” marks models measured at one setting. Where a
run recorded both timing scopes, the speedups in the scope not used by
Table 1 are listed; a series never mixes scopes. The
same column carries, for Burgers, the ratios against the tight Newton
setting: that setting is far more accurate than the NM-ROM, so the rule
never selects it and those ratios are context, not headline numbers.
$^{b}$ this column's time and its comparator were both measured in the
later job named beside the FOM setting, so the ratio stays within one
job.
For heat that column gives the speedups against the tighter
CN–CG setting (rtol $10^{-6}$). The heat batched fit solves each output time independently against
exactly propagated test moments, exploiting the linear, autonomous
structure of the heat equation (eigenfunction tests). Rows from the
paired-CG record list the one full-order setting that record retains,
itself chosen by the same rule.
$^{\ast}$ not in Table 1: a re-measurement of the row
above it, a development-source run at a mesh whose held-out row is in
Table 1, or the confirmed variant of the
$4096^2$ Burgers lattice rule with an exact residual at the first step.
For the $4096^2$ Burgers fast speedup in Table 1, the
numerator is the time for \nBhFastFom, rather than the Newton time
printed here.

<!-- table: TH_headline_times -->
| Problem | Mesh | Accurate | Fast | Accurate ms | Fast ms | FOM setting | FOM ms | Timing | Other scope, or the tighter (not rule-admissible) FOM: acc. / fast | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson 2D | 256^2 | q=256 | q=0 | 7.12 | 6.70 | CG, rtol 10^{-2} | 22.29 | GPU query | — | development |
| Poisson 2D | 1024^2 | q=256 | q=0 | 4.32 | 3.94 | CG, rtol 10^{-2} | 60.02 | GPU query | — | development |
| Poisson 2D | 2048^2 | q=256 | q=0 | 5.47 | 5.35 | CG, rtol 10^{-2} | 397.80 | GPU query | 28.7× / 28.8× (complete query) | development |
| Poisson 2D | 4096^2 | q=256 | q=0 | 16.92 | 16.74 | CG, rtol 10^{-1} | 1954.80 | GPU query | 32.7× / 33.2× (complete query) | development |
| Poisson 2D^{\ast} | 4096^2 | q=256 | q=0 | 17.17 | 17.01 | CG, rtol 2{\times}10^{-1} | 1779.89 | GPU query | 27.4× / 27.6× (complete query) | development |
| Poisson, L-shape 2D | 256^2 | q=64 | q=0 | 3.03 | 2.88 | CG, rtol 10^{-2} | 11.64 | complete query | — | development |
| Poisson, L-shape 2D | 512^2 | q=64 | q=0 | 4.80 | 4.72 | CG, rtol 10^{-2} | 29.13 | complete query | — | development |
| Poisson, L-shape 2D | 1024^2 | q=128 | q=0 | 5.81 | 5.67 | CG, rtol 3{\times}10^{-2} | 48.70 | complete query | 13.7× / 14.1× (GPU query) | development |
| Poisson, L-shape 2D | 2048^2 | q=128 | q=0 | 18.75 | 18.66 | CG, rtol 3{\times}10^{-2} | 318.22 | complete query | 29.8× / 30.2× (GPU query) | development |
| Heat 2D | 64^2 | — | single | — | 11.23 | CN–CG, rtol 10^{-2} | 2.50 | GPU query | — | development |
| Heat 2D | 256^2 | — | single | — | 11.34 | CN–CG, rtol 10^{-2} | 6.94 | GPU query | — | development |
| Heat 2D | 1024^2 | — | single | — | 12.21 | CN–CG, rtol 10^{-2} | 59.18 | GPU query | — | development |
| Heat (wide bank) 2D | 1024^2 | q=32 | q=0 | 27.33 | 27.77 | CN–CG, \Delta t=0.05, rtol 10^{-3} | 43.51 | GPU query | 4.70× / 4.62× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank) 2D | 2048^2 | q=32 | q=0 | 27.97 | 28.42 | CN–CG, \Delta t=0.05, rtol 10^{-3} | 272.55 | GPU query | 28.8× / 28.3× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank) 2D | 4096^2 | q=32 | q=0 | 31.68 | 32.19 | CN–CG, \Delta t=0.05, rtol 10^{-2} | 1139.83 | GPU query | 149× / 146× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank, batched fit) 2D | 1024^2 | q=32 | q=0 | 3.13 | 3.36 | CN–CG, \Delta t=0.05, rtol 10^{-3} | 43.51 | GPU query | 41.0× / 38.3× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank, batched fit) 2D | 2048^2 | q=32 | q=0 | 4.54 | 4.78 | CN–CG, \Delta t=0.05, rtol 10^{-3} | 272.55 | GPU query | 177× / 169× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank, batched fit) 2D | 4096^2 | q=32 | q=0 | 11.03 | 11.33 | CN–CG, \Delta t=0.05, rtol 10^{-2} | 1139.83 | GPU query | 428× / 416× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Burgers 2D | 256^2 | q=256, M=1088, EQ m=2560 | q=0, M=64, EQ m=1024 | 746.02 | 40.36 | Newton–BiCGStab, tol 10^{-3} | 31.79 | GPU query | — | development |
| Burgers 2D | 512^2 | q=256, M=1088, EQ m=2438 | q=0, M=64, EQ m=922 | 783.33 | 40.49 | Newton–BiCGStab, tol 10^{-3} | 52.92 | GPU query | — | development |
| Burgers 2D | 1024^2 | q=256, M=1088, dense | q=0, M=64, EQ m=934 | 22053.85 | 40.27 | Newton–BiCGStab, tol 10^{-4} | 81.31 | GPU query | — | development |
| Burgers 2D | 2048^2 | q=256, M=544, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 124.15 | 29.48 | Newton–BiCGStab, tol 3{\times}10^{-3} | 122.76 | GPU query | 0.99× / 2.01× (complete query); 6.07× / 25.5× (vs. tight Newton) | development |
| Burgers 2D | 4096^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 107.51 | 40.18^{b} | Newton–BiCGStab, tol 3{\times}10^{-3}; fast: Newton–BiCGStab, tol 10^{-3}, \Delta t=0.01, 406.0\,ms (job 4153483) | 523.85 | GPU query | 2.18× / 2.68× (complete query); 30.4× / 80.3× (vs. tight Newton) | development |
| Burgers (held-out cases) 2D | 2048^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 114.91 | 29.19 | Newton–BiCGStab, tol 10^{-3} | 164.43 | GPU query | 1.28× / 2.49× (complete query); 6.48× / 25.5× (vs. tight Newton) | held-out |
| Burgers (held-out cases) 2D | 4096^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 99.96 | 39.71^{b} | Newton–BiCGStab, tol 3{\times}10^{-3}; fast: Newton–BiCGStab, tol 10^{-3}, \Delta t=0.01, 401.2\,ms (job 4153483) | 537.39 | GPU query | 2.28× / 2.74× (complete query); 32.3× / 79.6× (vs. tight Newton) | held-out |
| Burgers, exact first step 2D^{\ast} | 4096^2 | q=256, M=1088, EQ lattice m=3969, first step exact | — | 5308.00 | — | Newton–BiCGStab, tol 3{\times}10^{-3} | 523.76 | GPU query | — | confirmed rule |
| Burgers, exact first step 2D^{\ast} | 4096^2 | q=256, M=1088, EQ lattice m=3969, first step exact | — | 6557.77 | — | Newton–BiCGStab, tol 3{\times}10^{-3} | 536.84 | GPU query | — | confirmed rule |
| Burgers, confirmed rule 2D | 512^2 | q=256, M=1088, EQ lattice m=3969, first step exact | q=0, M=64, EQ m=1024 | 194.97 | 23.62 | Newton–BiCGStab, tol 3{\times}10^{-3} | 17.74 | GPU query | — | development |
| Burgers, confirmed rule 2D | 1024^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 108.58 | 25.00 | Newton–BiCGStab, tol 3{\times}10^{-3} | 34.69 | GPU query | — | development |
| Burgers (earlier model) 2D | 1024^2 | — | single | — | 41.68 | Newton–BiCGStab, relaxed | 68.04 | GPU query | — | development |
| Burgers (earlier model) 2D^{\ast} | 1024^2 | — | single | — | 41.68 | Newton–BiCGStab, tight | 605.75 | GPU query | — | development |
| Poisson 3D | 32^3 | q=96 | q=0 | 2.63 | 2.49 | CG, rtol 10^{-2} | 2.47 | GPU query | — | accepted final |
| Poisson 3D | 64^3 | q=96 | q=0 | 3.36 | 3.22 | CG, rtol 10^{-2} | 4.46 | GPU query | — | accepted final |
| Poisson (dev. sources) 3D^{\ast} | 64^3 | q=96 | q=0 | 1.43 | 1.38 | CG, rtol 10^{-2} | 2.88 | GPU query | 1.57× / 1.56× (complete query) | development |
| Poisson (dev. sources) 3D | 128^3 | q=96 | q=0 | 1.86 | 1.80 | CG, rtol 10^{-2} | 12.55 | GPU query | 2.70× / 2.66× (complete query) | development |
| Poisson (dev. sources) 3D^{\ast} | 128^3 | q=96 | q=0 | 1.90 | 1.76 | CG, rtol 10^{-2} | 12.40 | GPU query | 2.65× / 2.70× (complete query) | development |
| Poisson (dev. sources) 3D | 256^3 | q=96 | q=0 | 6.08 | 5.95 | CG, rtol 10^{-2} | 140.87 | GPU query | 4.18× / 4.21× (complete query) | development |
| Heat (new bank) 3D | 32^3 | q=288 | q=0 | 58.16 | 25.41 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 3.33 | GPU query | — | accepted final |
| Heat (new bank) 3D | 64^3 | q=288 | q=0 | 58.69 | 25.45 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 5.23 | GPU query | — | accepted final |
| Heat (new bank) 3D | 128^3 | q=288 | q=0 | 62.03 | 28.25 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 17.44 | GPU query | — | accepted final |
| Heat (new bank, batched fit) 3D | 32^3 | q=288 | q=0 | 5.36 | 5.88 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 3.33 | GPU query | — | accepted final |
| Heat (new bank, batched fit) 3D | 64^3 | q=288 | q=0 | 5.54 | 6.14 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 5.23 | GPU query | — | accepted final |
| Heat (new bank, batched fit) 3D | 128^3 | q=288 | q=0 | 7.84 | 8.30 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 17.44 | GPU query | — | accepted final |

**Table 14.** Heat at high resolution, accuracy: one frozen model per block;
worst same-grid relative $L^2$ error over all output times and over
evolved times, accurate / fast setting. CN: reduced Crank–Nicolson
steps; batched fit: each output time fitted independently to exactly
propagated test moments (linear autonomous problems with eigenfunction
tests only). The sealed 2D cohort was opened once and gives every
wide-bank row of Table 1; the development row ran on
another GPU (A100) and is shown for reference. Wide bank: $R=128$, $k=8$;
earlier: the $R=32$ checkpoint of the Heat rows; 3D, earlier bank: the
three-dimensional model that preceded the new bank of
Table 1 (it misses the 1 % all-times target;
$256^3$ is the first 16 final cases, not a mesh trend).

<!-- table: TH_heat_hires -->
| Model | Mesh | Cohort (cases) | Stepping | Acc. / fast | Err. all times (%) | Err. evolved (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Wide bank | 1024^2 | development (12) | CN | 32 / 0 | 0.24 / 0.56 | 0.17 / 0.46 |
| Wide bank | 1024^2 | sealed (16) | CN | 32 / 0 | 0.49 / 1.36 | 0.22 / 0.84 |
| Wide bank | 1024^2 | sealed (16) | batched fit | 32 / 0 | 0.49 / 1.36 | 0.21 / 0.84 |
| Wide bank | 2048^2 | sealed (16) | CN | 32 / 0 | 0.49 / 1.36 | 0.22 / 0.84 |
| Wide bank | 2048^2 | sealed (16) | batched fit | 32 / 0 | 0.49 / 1.36 | 0.21 / 0.84 |
| Wide bank | 4096^2 | sealed (16) | CN | 32 / 0 | 0.49 / 1.36 | 0.22 / 0.84 |
| Wide bank | 4096^2 | sealed (16) | batched fit | 32 / 0 | 0.49 / 1.36 | 0.21 / 0.84 |
| Earlier (R=32) | 2048^2 | development (12) | CN | 24 / 0 | 1.68 / 4.56 | 1.11 / 4.56 |
| Earlier (R=32) | 4096^2 | development (12) | CN | 24 / 0 | 1.68 / 4.56 | 1.11 / 4.56 |
| 3D, earlier bank | 128^3 | final (64) | CN | 96 / 0 | 1.93 / 3.18 | 0.75 / 1.54 |
| 3D, earlier bank | 128^3 | final (64) | batched fit | 96 / 0 | 1.93 / 3.18 | 0.72 / 1.36 |
| 3D, earlier bank | 256^3 | final, 1st 16 (16) | CN | 96 / 0 | 1.27 / 2.02 | 0.55 / 0.91 |
| 3D, earlier bank | 256^3 | final, 1st 16 (16) | batched fit | 96 / 0 | 1.27 / 2.02 | 0.50 / 0.86 |

**Table 15.** Heat at high resolution, timing for the rows of
Table 14: median GPU query times (ms), the FOM chosen
by the rule of Table 1 and its speedup, and the speedup
against CN–CG at $\Delta t=0.025$, rtol $10^{-6}$.

<!-- table: TH_heat_hires_times -->
| Model | Mesh | Cohort (cases) | Stepping | ms acc. / fast | FOM ms | FOM (Table 1 rule) | Speedup | vs CN–CG 1e-6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wide bank | 1024^2 | development (12) | CN | 31.2 / 36.9 | 66.8 | \Delta t=0.05, rtol 10^{-3} | 2.14 / 1.81× | 6.33 / 5.35× |
| Wide bank | 1024^2 | sealed (16) | CN | 27.3 / 27.8 | 43.5 | \Delta t=0.05, rtol 10^{-3} | 1.59 / 1.57× | 4.70 / 4.62× |
| Wide bank | 1024^2 | sealed (16) | batched fit | 3.1 / 3.4 | 43.5 | \Delta t=0.05, rtol 10^{-3} | 13.9 / 13.0× | 41.0 / 38.3× |
| Wide bank | 2048^2 | sealed (16) | CN | 28.0 / 28.4 | 272.5 | \Delta t=0.05, rtol 10^{-3} | 9.74 / 9.59× | 28.8 / 28.3× |
| Wide bank | 2048^2 | sealed (16) | batched fit | 4.5 / 4.8 | 272.5 | \Delta t=0.05, rtol 10^{-3} | 60.0 / 57.0× | 177 / 169× |
| Wide bank | 4096^2 | sealed (16) | CN | 31.7 / 32.2 | 1139.8 | \Delta t=0.05, rtol 10^{-2} | 36.0 / 35.4× | 149 / 146× |
| Wide bank | 4096^2 | sealed (16) | batched fit | 11.0 / 11.3 | 1139.8 | \Delta t=0.05, rtol 10^{-2} | 103 / 101× | 428 / 416× |
| Earlier (R=32) | 2048^2 | development (12) | CN | 15.9 / 8.9 | 173.1 | \Delta t=0.1, rtol 10^{-2} | 10.9 / 19.5× | 51.1 / 91.4× |
| Earlier (R=32) | 4096^2 | development (12) | CN | 18.0 / 10.8 | 1014.6 | \Delta t=0.1, rtol 10^{-2} | 56.5 / 93.7× | 267 / 442× |
| 3D, earlier bank | 128^3 | final (64) | CN | 35.1 / 16.5 | 7.0 | \Delta t=0.1, rtol 10^{-2} | 0.20 / 0.43× | 0.83 / 1.78× |
| 3D, earlier bank | 128^3 | final (64) | batched fit | 5.2 / 4.0 | 7.0 | \Delta t=0.1, rtol 10^{-2} | 1.36 / 1.77× | 5.67 / 7.40× |
| 3D, earlier bank | 256^3 | final, 1st 16 (16) | CN | 37.0 / 21.0 | 173.3 | \Delta t=0.05, rtol 10^{-4} | 4.69 / 8.27× | 8.96 / 15.8× |
| 3D, earlier bank | 256^3 | final, 1st 16 (16) | batched fit | 9.6 / 8.4 | 173.3 | \Delta t=0.05, rtol 10^{-4} | 18.1 / 20.5× | 34.7 / 39.3× |

**Table 16.** Supporting data for Table 2: every row
of that table with its residual path and median GPU query time from the
same allocation, plus the exploratory hyper-reduced Kim et al. rows,
whose hyper-reduction did not pass its reproduction gate. Our rows use
the unoptimised dense reference path (no empirical quadrature, stopping
tolerance $10^{-6}$, $M=4(k+q)$ tests); our optimised query is timed only
in Table 1, in other jobs, and no ratio across jobs is
formed. The shallow masked-autoencoder NM-LSPG of Kim et al. (2022)
passed its reproduction gate, on their own 2D Burgers benchmark (their
Sec. 6.2), on the third of three pre-registered attempts
(\nBaseGateAct activation; median $\nBaseGateMedian %$ against a
$\nBaseGateBar %$ bar; published ${<}\nBaseGatePublished %$). Its
published encoder width ($2n$) needs $\nBaseEncoderNeedGB$ GB to train at
$256^2$ and $\nBaseEncoderNeedGBFiveTwelve$ GB at $512^2$, beyond the
$\nBaseDeviceGB$ and $\nBaseDeviceGBFiveTwelve$ GB devices; at $512^2$
its encoder, capped at width $\nBaseKimWidth$, trained only
$\nBaseKimEpochs$ epochs in its $\nBaseKimWall$ s budget, so those errors
partly reflect a training-time limit. The data-matched row is trained on our
bank's \nBaseDataMatched trajectories.

<!-- table: TH_nmrom_baselines_appx -->
| Mesh | Method | Residual path | k | Worst (%) | Median (%) | GPU ms (this job) | MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256^2 | Kim et al. NM-LSPG | dense | 8 | 163.93 | 43.34 | 1301 | 2480 |
| 256^2 | Kim et al. NM-LSPG | dense | 16 | 145.40 | 43.87 | 2384 | 2567 |
| 256^2 | Kim et al. NM-LSPG | dense | 32 | 127.51 | 43.61 | 3846 | 2742 |
| 256^2 | Kim et al. NM-LSPG, data-matched | dense | 16 | 120.33 | 38.71 | 3886 | 2567 |
| 256^2 | POD-LSPG | dense | 8 | 55.52 | 22.72 | 138 | 18 |
| 256^2 | POD-LSPG | dense | 16 | 41.53 | 10.45 | 259 | 30 |
| 256^2 | POD-LSPG | dense | 32 | 24.78 | 5.65 | 519 | 55 |
| 256^2 | This work, fast (q=0, k=16) | dense (reference path) | 16 | 6.79 | 0.66 | 262 | 625 |
| 256^2 | This work, accurate (q=256, k=16) | dense (reference path) | 272 | 0.88 | 0.089 | 3746 | 1769 |
| 256^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 8 | 115.60 | 51.78 | 65 | 2452 |
| 256^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 16 | 202.87 | 52.51 | 126 | 2506 |
| 256^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 32 | 122.68 | 41.98 | 475 | 2794 |
| 512^2 | Kim et al. NM-LSPG | dense | 8 | 175.71 | 55.48 | 746 | 9959 |
| 512^2 | Kim et al. NM-LSPG | dense | 16 | 174.99 | 58.56 | 1497 | 10511 |
| 512^2 | Kim et al. NM-LSPG | dense | 32 | 252.33 | 56.72 | 2059 | 10879 |
| 512^2 | POD-LSPG | dense | 8 | 55.82 | 24.08 | 314 | 71 |
| 512^2 | POD-LSPG | dense | 16 | 42.01 | 13.83 | 617 | 122 |
| 512^2 | POD-LSPG | dense | 32 | 25.63 | 7.74 | 383 | 222 |
| 512^2 | This work, fast (q=0, k=16) | dense (reference path) | 16 | 7.72 | 0.69 | 404 | 2368 |
| 512^2 | This work, accurate (q=256, k=16) | dense (reference path) | 272 | 1.05 | 0.080 | 6269 | 6891 |
| 512^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 8 | 273.27 | 53.08 | 30 | 9813 |
| 512^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 16 | 185.17 | 58.52 | 62 | 10099 |
| 512^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 32 | 194.47 | 56.00 | 138 | 10569 |

**Table 17.** Burgers 2D correction settings from one frozen model at $256^2$,
test count fixed at $M=1088$, dense residual, one allocation. Errors are worst
evolved same-grid percentages on development cases; times are paired
median GPU query times. All rows meet the stopping rule.

<!-- table: TR_correction_main -->
| Correction rank $q$ | Relative $L^2$ error (%) | GPU query time (ms) |
|---|---|---|
| 0 | 1.2657 | 848.0 |
| 64 | 1.0593 | 1359.4 |
| 128 | 0.8711 | 1886.5 |
| 256 | 0.5194 | 4377.9 |

**Table 18.** Which knob to turn, measured on Burgers 2D from frozen models.
Rank: fixed-$M$ ladder at $256^2$ (Table 17). EQ:
dense time over EQ time at $q=0$ in one allocation per mesh. Tolerance
and cap: 32 validation cases (Table 24),
stopping tolerance $10^{-8}\!\to\!10^{-3}$ and iteration cap 2.
$^{\ast}$ Jobs 3789570 and 3789572;
Table 19 is another allocation.

**Table 19.** Dense and empirical-quadrature (EQ) residual evaluation on
Burgers 2D at $256^2$. Error is worst evolved relative $L^2$ (%); time is
median GPU milliseconds from one allocation (job 3780164, the
EQ-ladder job of Table 22; the $256^2$ row of
Table 1 is job 3789570 and Table 17
is the fixed-$M$ ladder job, each with its own measured error); the last column is dense
time over EQ time. Dashes: no paired dense measurement. Rules at
$q=0, 16, 32$ pass every independent reconstruction;
higher-rank rules pass some (Table 22).

<!-- table: TR_figure1_table -->
| Correction rank $q$ | Dense error (%) | Dense ms | EQ error (%) | EQ ms | Dense / EQ |
|---|---|---|---|---|---|
| 0 | 1.8890 | 292.87 | 1.8891 | 59.07 | 4.96$\times$ |
| 16 | — | — | 1.4270 | 80.61 | — |
| 32 | — | — | 1.2493 | 97.70 | — |
| 64 | 1.0843 | 624.93 | 1.2275 | 120.63 | 5.18$\times$ |
| 128 | 0.8930 | 1190.49 | 0.8936 | 246.87 | 4.82$\times$ |
| 256 | 0.5194 | 3915.14 | 0.5389 | 722.21 | 5.42$\times$ |

**Table 20.** Three-dimensional configurations, read from the run records.
Domains are the unit cube with homogeneous Dirichlet data, except
Navier–Stokes, which is periodic. Burgers and Navier–Stokes appear in
Table 3.

<!-- table: TH_config3d -->
| Problem | Mesh | Cases | Correction ranks | Named FOM | Residual |
| --- | --- | --- | --- | --- | --- |
| Burgers 3D | 32^3 | 32 final | 0, 192 | Newton–BiCGStab, \Delta t=0.01 | dense |
| Poisson 3D | 32^3, 64^3 | 64 final | 0, 32, 96 (k=16) | CG, rtol 10^{-2}, no preconditioner | dense |
| Heat 3D (new bank) | 32^3, 64^3, 128^3 | 64 sealed final (all times) | 0, 288 (k=32, R=320) | CN–CG, setting per row by the rule; named \Delta t=0.025, rtol 10^{-6} | exact (linear) |
| Navier–Stokes 3D | 32^3 periodic | 32 final | 0, 32, 64, 128, 256 (k=64, R=1536, M=2048) | CNAB2, \Delta t=0.01 | dense |

Three-dimensional Poisson ($-\Delta u=f$, Gaussian sources), heat
($u_t=\kappa\Delta u$, Gaussian initial fields) and scalar Burgers
($u_t+u(u_x+u_y+u_z)=\nu\Delta u$, backward Euler, sign-upwind) are posed
on the unit cube with zero Dirichlet walls and extend the operators of
Appendix A across three axes. Navier–Stokes uses
periodic vector fields, an incompressibility projection and the CNAB2
integrator; its FOM is not a CG solve. All four three-dimensional problems are evaluated on held-out final
cohorts whose settings were frozen beforehand.

\subsection{Source records}
The accompanying source package retains the full experiment archive,
including failed configurations, retractions, training records, checkpoints,
case-level fields, timing repetitions and independent audits. The PDF
contains the evidence needed for its stated claims rather than every
experiment in that archive. Table generators and frozen evidence identify
the displayed values through these machine-readable manifests:

- `tables/provenance.json`: original two-dimensional studies.
- `tables/rewrite-provenance.json`: paired CG comparisons.
- `tables/burgers-iterative-provenance.json`: the earlier
fine-grid Burgers tight and relaxed controls.
- `tables/headline-provenance.json`: every row of
Table 1, Table 13, Table 3 and
Table 20 and every row of Table 4, with the
commit and SHA256 of each source record.

Each manifest records source hashes; its corresponding evidence retains
solver configuration, checkpoint identity and allocation metadata. Development
selection and final evaluation are distinct. Solver exit flags document
numerical stopping, not a proof of global optimality.

\input{tables/TH_jobs}

## E Validation of the correction and quadrature studies

<!-- section sources: none (prose only) -->

A separate ladder with $M=4(k+q)$, evaluated once on a sealed cohort,
gives monotone error reduction for all 4 checkpoints
(top-rank errors $0.59$–$0.68 %$, Table 21); 3 of 4
meet the pre-registered secondary criterion (settings spanning at least
$2\times$ in error and in cost). Two stricter pre-registered checks fail:
the original model's sealed-to-development ratio at $q=0$ (one sealed case
converges to a wrong branch, $10.1120 %$) and universal
convergence (\nSealedUnconvergedPlain; Table 21).

**Table 21.** The sealed cohort (job 3804465, NVIDIA A100-PCIE-40GB),
opened once after every choice was frozen. Top: per rung of the dense
$M=4(k+q)$ ladder, the original model's sealed worst evolved error and device
time, the seed mean $\pm$ sample standard deviation on the sealed and the
development cohort, the pre-registered sealed-over-development ratio of seed means
(raw and difficulty-normalised) and the converged count over the
4 checkpoints. Bottom: per-checkpoint verdicts. The
original model's $q=0$ rung is a single sealed case converged to a wrong branch
(gradient exit, zero budget exits); the pre-registered ratio criterion fails
at $q=0$ for the original model alone (5.35) and the
convergence criterion fails on \nSealedUnconvergedPlain. Top-rank sealed errors per checkpoint: \nSealedTopPerCheckpoint.

<!-- table: T13_sealed -->
| $q$ | $M$ | original model sealed % | ms | seeds sealed % | seeds development % | sealed / dev | normalised | converged |
|---|---|---|---|---|---|---|---|---|
| 0 | 64 | 10.1120 | 375.1 | 2.8023 $\pm$ 0.6357 | 2.0002 $\pm$ 0.4986 | 1.40 | 1.41 | 4 of 4 |
| 16 | 128 | 1.4617 | 480.8 | 1.7305 $\pm$ 0.1297 | 1.5988 $\pm$ 0.0995 | 1.08 | 1.09 | 4 of 4 |
| 32 | 192 | 1.4458 | 540.5 | 1.5044 $\pm$ 0.1422 | 1.3727 $\pm$ 0.1314 | 1.10 | 1.14 | 4 of 4 |
| 64 | 320 | 1.3252 | 754.5 | 1.3528 $\pm$ 0.1099 | 1.2130 $\pm$ 0.0580 | 1.12 | 1.13 | 3 of 4 |
| 128 | 576 | 1.0964 | 1460.4 | 1.0461 $\pm$ 0.0619 | 0.9562 $\pm$ 0.0264 | 1.09 | 1.17 | 4 of 4 |
| 256 | 1088 | 0.6789 | 6725.5 | 0.6188 $\pm$ 0.0345 | 0.4987 $\pm$ 0.1164 | 1.24 | 1.20 | 4 of 4 |

<!-- table: T13b_sealed_verdicts -->
| checkpoint | job | monotone (evolved) | monotone (all-times) | every setting converged | error span | cost span | secondary criterion |
|---|---|---|---|---|---|---|---|
| original | `3804465` | yes | yes | yes | 14.89$\times$ | 17.93$\times$ | yes |
| first seed | `3804465` | yes | yes | yes | 3.52$\times$ | 13.25$\times$ | yes |
| second seed | `3804465` | yes | yes | no | 4.81$\times$ | 13.17$\times$ | no |
| third seed | `3804465` | yes | yes | yes | 5.22$\times$ | 14.62$\times$ | yes |

**Table 22.** The EQ ladder with the cheapest rule passing the primary bar in its
draw per rung, timed in one allocation (job 3780164, A100 80 GB),
with its same-job dense twins, and the construction status from the four-draw
replication (job 3783811): confirmed = every re-draw passes, marginal
= only the listed fraction does. Errors are as measured with these rules
(SHA256-identified); they are not properties of the construction above
$q=32$. Ladder rows are read from the study's final summary record.

<!-- table: T09_eq_ladder -->
| $q$ | $m$ | fit states | $\rho_{\max}$ (this draw) | construction status | EQ evolved % | EQ all % | EQ ms | dense evolved % | dense ms | EQ/dense cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1024 | 64 | 0.0153 | confirmed (3 of 3 re-draws) | 1.8891 | 2.5629 | 59.1 | 1.8890 | 292.9 | 0.202 |
| 16 | 1024 | 64 | 0.0935 | confirmed (3 of 3 re-draws) | 1.4270 | 2.4806 | 80.6 | — | — | — |
| 32 | 1024 | 42 | 0.0533 | confirmed (2 of 2 re-draws) | 1.2493 | 2.3534 | 97.7 | — | — | — |
| 64 | 1024 | 25 | 0.0531 | marginal (2 of 6 draws pass) | 1.2275 | 2.1489 | 120.6 | 1.0843 | 624.9 | 0.193 |
| 128 | 2048 | 64 | 0.0669 | marginal (4 of 5 draws pass) | 0.8936 | 1.8116 | 246.9 | 0.8930 | 1190.5 | 0.207 |
| 256 | 2048 | 64 | 0.1074 | marginal (1 of 5 draws pass) | 0.5389 | 0.9053 | 722.2 | 0.5194 | 3915.1 | 0.184 |

**Table 23.** How the corrections are solved on Burgers, one job (job
3734098, NVIDIA A100 80GB PCIe): joint LM on $(z,y)$, block-damped, and plain
variable projection at $q=64$ and $128$. Same error where all converge; the
block-damped step is the one kept.

<!-- table: T19_solver_variants -->
| arm | $M$ | worst all-times % | budget exits | all stationary | device ms |
|---|---|---|---|---|---|
| $q{=}64$, joint LM on $(z,y)$ | 320 | 2.1489 | 6 | no | 1119.7 |
| $q{=}64$, block-damped | 320 | 2.1489 | 0 | yes | 619.2 |
| $q{=}64$, plain variable projection | 320 | 2.1489 | 0 | yes | 9527.9 |
| $q{=}128$, plain variable projection | 256 | 1.8116 | 297 | no | 47566.9 |
| $q{=}128$, block-damped | 256 | 1.8116 | 0 | yes | 825.5 |

**Table 24.** Solver-side controls on Burgers 2D, 32 validation cases,
one allocation: stopping tolerance, EQ node count $m$ and iteration cap,
with full-order settings for scale. Error: worst over
cases and output times of the $L^2$ error against the refined reference,
relative to the initial-field norm, so the full-order rows show their
discretisation error.

<!-- table: T08_solver_knobs -->
| setting | worst % (32 validation) | device ms | early-stopped |
|---|---|---|---|
| EQ $m{=}256$, tol $10^{-6}$, cap 180 | 7.245 | 51.0 | 0/96 |
| EQ $m{=}512$, tol $10^{-8}$ | 6.712 | 64.7 | 0/96 |
| EQ $m{=}512$, tol $10^{-3}$ | 6.701 | 41.6 | 0/96 |
| EQ $m{=}512$, cap 2 (early-stopped) | 90.324 | 28.4 | 96/96 |
| FOM Newton $10^{-2}$, $\Delta t{=}0.01$ | 6.807 | 9.9 | 0/96 |
| FOM Newton $10^{-2}$, $\Delta t{=}0.005$ | 35.357 | 17.4 | 0/96 |
| FOM Newton $10^{-4}$, $\Delta t{=}0.005$ | 6.171 | 22.4 | 0/96 |
| FOM Newton $10^{-6}$, $\Delta t{=}0.005$ | 6.172 | 88.3 | 0/96 |

**Reading the validation tables.**
Correction rank $q$ is the number of added coefficient directions;
$M$ is the number of weak test modes and $m$ the quadrature node count.
A scheduled ladder changes $M$ with $q$; the main fixed-test-space study
holds $M$ constant. A sealed cohort is opened only after choices are frozen.
Confirmed quadrature passes every independent reconstruction of the rule;
marginal means some reconstructions fail. These statuses concern rule
construction, not just the error of one selected trajectory. Same-grid
error and error against a refined reference measure different quantities.
