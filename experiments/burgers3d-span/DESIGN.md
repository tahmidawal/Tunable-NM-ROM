# DESIGN — burgers3d-span: a Burgers 3D row with the paper's current method

Pre-registered 2026-09-23 (EDT), before any ROM result of this lane exists. Branch `exp/2026-09-23-burgers3d-span`
(forked from `exp/2026-09-20-paper-b3d` @ `37f92a5d`), local commits only, never pushed. Cluster namespace
`/cluster/tufts/paralab/tawal01/b3dspan_20260923/`, one sub-directory per job. Amendments are appended at the end,
dated, and never rewrite this text.

## 1. Diagnosis of the earlier failure (b3d004/b3d007, 33³, 2.43 % at 380 ms vs Newton–BiCGStab 8.7 ms)

Measured from the retained b3d007 invocation records (`worktrees/2026-09-20-paper-b3d/.../b3d007/collected/out/seed0`):

- **Iteration counts were not the problem.** Every ROM/POD arm used 2.0–2.9 LM iterations per step (100–145 per
  50-step query; initial fits 4–12). POD-64, with only 64 unknowns and two iterations per step, still took 124 ms.
- **Each iteration was dense in the mesh.** The weak residual was evaluated on all 29 791 interior nodes: decode
  $G c$ (29 791 × 256), the upwind stencil on the full grid, and an explicit dense test matrix
  $\Phi^\top$ (29 791 × 642) applied to the advection. The Jacobian was `jax.jacfwd` through that whole pipeline, i.e.
  $k+q$ (up to 224–256) forward tangents each costing a full-grid decode, stencil and a 29 791 × 642 product. There was
  no hyper-reduction ("There is no empirical-quadrature arm. This is dense weak projection over all interior grid
  points", HANDOFF.md). Cost per iteration ≈ 1.2 ms (64 unknowns) to 3 ms (256 unknowns); ×100–145 iterations → the
  124–390 ms.
- **Per-query overheads:** a 400-budget dense initial LM fit, and 51 dense output fields.
- **The mesh was the cheapest possible for the FOM.** At 33³ the exact-Helmholtz-preconditioned Newton–BiCGStab takes
  0.17–0.35 ms per step. No reduced solver with ≥ 2 sequential LM iterations per step can beat that by much on a GPU.
- **Accuracy was capped by the representation.** The retained bank (R = 256, trained on 33³ fields only) has worst
  projection error 4.2 % (training) / 2.9 % (validation) and 5.5 % on the b3d007 final cohort; the free-bank solve was
  4.4 % worst evolved. The K = 32/64 heads were 9–16 %. This bank has never seen a 65³ or 129³ field; the 65³/129³
  discrete solutions differ from the 33³ ones by up to 3.6 % (b3d007 physical-reference records: 65³ vs 129³ restricted
  to the 33³ nodes), so it cannot be the model for a 32³/64³/128³ row.

**Consequences for the design.** (i) Retrain the bank (the brief allows it when the existing one is inadequate; it is).
(ii) Train one model on all three meshes' training fields, so the same frozen model serves every mesh. (iii) Remove
every full-grid pass from the LM iteration: linear terms exact and precomputed; advection by a precomputed rule on a
mesh-independent number of unknowns/tests/nodes; analytic Jacobian; one factorisation per iteration. (iv) Six dense
outputs (t = 0, 0.05, …, 0.25), as the paper's 2D Burgers rows. (v) Initial fit = one exact least-squares projection
(span) or a nearest-code head fit on R'-dimensional coordinates.

## 2. Problem, meshes, cohorts

Equation, family, stepping: exactly the paper-b3d FOM (vendor `b3d_common.py`, sha256 `c5e36b4c…`): 1–3 positive
masked Gaussian blobs, amplitude A ∈ [0.5, 2], ν log-uniform in [0.01, 0.1], sign-upwind advection switching on the
centre value, 7-point Laplacian, ghost-zero walls, backward Euler Δt = 0.005, 50 steps to T = 0.25. **Meshes: n = 33,
65, 129 nodes per axis including the walls (32³, 64³, 128³ cells; 31³, 63³, 127³ unknowns).** The FOM discretisation is
not changed; the ROM solves the FOM's discrete backward-Euler problem through its tested residual.

| cohort | seed (`build_param_table`) | size | use |
|---|---|---|---|
| training | 923001 | 384 | bank, ordering, heads (all three meshes) |
| bank-validation | 923051 | 48 | bank / head checkpoint selection and bank floors only |
| ROM validation | 923101 | 16 | the settings rule (§6) |
| certification draws 1–5 | 923201–923205 | 8 each | quadrature certificates (§5) |
| confirmation draw | 923206 | 16 | quadrature certificates (§5) |
| **sealed held-out** | **923401** | **32** | opened once, frozen settings only (§7) |

All are fresh seeds never used by any lane. Every job regenerates its fields from these seeds on the cluster.

## 3. The model (one training job; two bank widths trained in parallel as insurance)

`train.py`. Coordinate bank $G(x)=\mu(x)\,s\,g_\phi(x)$, $\mu=64\prod x_i(1-x_i)$ (exact Dirichlet factor), 128 random
Fourier features (scale 2), SiLU MLP width 512 × depth 3. **Code-free variable-projection training** (heat3d-bank
`train_vp.py`, which fixed that lane's divergent auto-decoder): the loss is the exact projection error of the
normalised training snapshots onto span(G), via a thin QR, as a POD-weighted mean (top 1536 POD modes per mesh group) +
0.5 × a power-4 tail over a random minibatch + 1e-3 × a span-invariant whitening term. **Three point groups**: all
interior nodes of the 33-node grid; all interior nodes of the 65-node grid; a fixed random subset (250 047 nodes, seed
923002) of the 129-node interior grid. Training snapshots: the 384 training trajectories at steps
{0,1,2,4,7,10,15,20,25,30,40,50}, generated per mesh by the same Newton–BiCGStab (ntol 1e-8, ltol 1e-9). Checkpoint =
the lowest worst bank-validation projection error over the three groups.

Ordering (paper App. A.1): thin QR of the bank on the 65-node grid in the $h^3$ metric, $G=Q_GR_G$; rows
$R_Ga_i/\lVert u_i\rVert$ for every training snapshot of every mesh ($a_i$ = its least-squares bank coefficients);
SVD → $V$; $T=R_G^{-1}V$; $\hat G = GT$. Training data only.

Heads: $h(z)=\mathrm{MLP}(z)+Wz$ (width 512, depth 3), auto-decoder on the ordered coefficients of every training
snapshot (all meshes), relative loss + 0.1 × squared relative loss; K ∈ {32, 64}; checkpoint = lowest worst
best-found error on the bank-validation cohort. Library for the nearest-code start = the training codes and their head
outputs.

Bank widths: job `trR512` (R = 512) and job `trR384` (R = 384) run concurrently with identical recipes. **Rule (fixed
now):** the model is R = 512 unless its job fails, its bank is ill-conditioned (condition > 1e8 on the 65-node grid), or
its worst full-grid bank-validation floor at R' = R exceeds the R = 384 model's at any mesh; then R = 384. Recorded
diagnostics: bank floors at every R' of the ladder on the full native grids of all three meshes (bank-validation
cohort, six output times), POD reference floors, and the old b3d004 bank's floor on the same fields.

## 4. The reduced query (paper §3, fast LM path of the 2D lane)

`common.make_query`. Per mesh, offline and from the frozen model only: $\hat G$ on the interior nodes, its thin QR
$\hat G=QR_q$ (nested: the prefix $Q_{:,:R'}$, $R_q[:R',:R']$ is the QR of $\hat G_{R'}$; $Q$ stored as column blocks at
the ladder edges so an arm reads exactly R' columns), $A=\Phi^\top\hat G$ ($M_{\max}\times R$, by 3D DST; $\Phi$ = the
lowest sine modes by discrete eigenvalue, shell-completed as in paper-b3d), and the advection rule tables.

Unknowns: span $u=\hat G_{R'}c$ ($R'$ unknowns); head $u=\hat G_{R'}h(z)_{:R'}$ ($K$ unknowns). Tests: span
$M = 4R'$, head $M = 4K$ (paper convention $M = 4\times$ unknowns), shell-completed. Scaled tested residual
$r = S\odot(Ac - Ac_{\rm prev} + \Delta t(\mathrm{adv}(c) + \nu\lambda\odot Ac))$, $S=(1+\Delta t\nu\lambda)^{-1}$.

Advection rules (each certified, §5):
- **`tensor`**: on nonnegative states the FOM's sign-upwind stencil is the backward difference, and
  $a(u)=u\odot D^-u$ is an exact quadratic form of the coefficients; $T_{mij}=\Phi_m^\top(\hat G_i\odot D^-\hat G_j)$ is
  precomputed per mesh (as the paper does for Navier–Stokes). The paper-b3d family is positive and backward Euler with
  this upwind scheme keeps the FOM solution nonnegative; decoded ROM states can dip slightly below zero where the
  solution is ≈ 0, where the FOM stencil switches. That discrepancy is exactly what $\rho$ measures against the true
  sign-upwind advection, so the tensor is admitted only as a certified rule, like EQ. Nested: arm $(R',M)$ uses
  $T[:M,:R',:R']$.
- **`lat16`**: empirical quadrature on the uniform lattice $x=k/16$ ($15^3=3375$ nodes, equal weights $s^3$,
  $s=(n-1)/16$) with the FOM's own sign-upwind stencil evaluated on the cached 7-point bank rows — the paper's 2D
  "lattice EQ" rule, the paper's method for Burgers.

LM (the 2D fast path): analytic Jacobian ($J = A + \Delta t\,S\odot J_{\rm adv}$, times $J_h$ for the head), one
$(r,J)$ evaluation per iteration, Cholesky of $J^\top J+\lambda\,\mathrm{diag}(J^\top J)$, step clipped to a trust radius
(0.05 × the RMS spread of the training coefficient vectors, resp. codes), damping carried between steps ($\lambda_0 =
10^{-6}$, ÷3 / ×10), predictor = the smallest residual among the current state and its linear/quadratic
extrapolations, scale-free stopping $\lVert J^\top r\rVert\le\eta\lVert J\rVert_F\lVert r\rVert$ with $\eta=10^{-3}$,
50-iteration step budget. Initial state: span — exact least squares $c_0=R_q^{-1}Q^\top u_0$ ($O(NR')$, one pass);
head — nearest library code in field distance, then LM on $\lVert R_q h(z)_{:R'} - Q^\top u_0\rVert$ (200 iterations,
η = 1e-6). Output: six dense fields $Q(R_qc)$ at t = 0, 0.05, …, 0.25.

**Arms (identical at every mesh; R = 512 model; for R = 384 drop the 512 entries and use 384 as the top):**

| family | R' | rule | η |
|---|---|---|---|
| span | 512, 384, 256, 192, 128, 96, 64, 32 | tensor | 1e-3 |
| span | 512 | tensor | 1e-6 (tolerance check) |
| span | 512, 256, 128, 64 | lat16 | 1e-3 |
| head K = 32 | 512, 256, 128 | tensor | 1e-3 |
| head K = 64 | 512 | tensor | 1e-3 |
| head K = 32 | 512 | lat16 | 1e-3 |
| control `bad` | span 128, lat16 with weights × 0.5 | certification only; must FAIL | — |

## 5. Quadrature certificate (paper App. A, eq. rho)

For every arm, at every mesh, the deployed query is run on the certification populations (5 draws × 8 trajectories,
then the confirmation draw of 16). On every state it reaches that backward Euler evaluates the advection at
(k = 1…50; the paper's current definition, the initial state excluded),
$\rho = \lVert \mathrm{rule}(u) - \Phi^\top a_{\rm upwind}(u)\rVert / \lVert\Phi^\top a_{\rm upwind}(u)\rVert$ over the
arm's M tests, the exact side by the FOM's sign-upwind stencil on every interior node and a 3D DST. **Confirmed iff
$\rho_{\max}\le 0.116$ (`\nEqtopBar`) on each of the 5 draws AND on the confirmation draw.** Never the fit residual.
The minimum decoded value on reached states is recorded (positivity). The `bad` control must not be confirmed.

## 6. Pre-registered settings rule (validation cohort 923101, per mesh)

- Error of an invocation: $\max_{t\in\{0.05,\dots,0.25\}}\lVert u(t)-u_{\rm ref}(t)\rVert_2/\lVert u_{\rm ref}(0)\rVert_2$ on
  every interior node; reference = same-grid Newton–BiCGStab, Δt = 0.005, ntol 1e-10, ltol 1e-11, recomputed in each
  job. Arm error = worst over cases (and repetitions).
- Eligible: certificate confirmed (§5), every output finite, no LM exit with reason 3 (damping exhausted / nonfinite).
  Budget exits (reason 0) are reported; an arm with any budget exit on more than 1 % of its steps is ineligible.
- **accurate** = the eligible arm with the smallest worst evolved error.
- **fast** = the eligible arm with the smallest median GPU time whose worst evolved error ≤ **5 %** (the paper's
  convention for problems without an earlier default setting, as its Navier–Stokes fast column; fixed now).
- **FOM** = the fastest tested full-order setting (median GPU ms) whose worst evolved error ≤ the accurate arm's; both
  speedups of the mesh divide that one time. If no FOM setting is that accurate, the most accurate FOM setting is named
  and the speedups are labelled as against a less accurate FOM.
- FOM grid (Newton–BiCGStab, same allocation): Δt ∈ {0.005, 0.01} × ntol ∈ {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4} ×
  ltol ∈ {0.5, 0.1}, plus Δt = 0.025 × ntol ∈ {1e-2, 1e-3} × ltol 0.5 (26 settings); ntol relative to the previous
  state's norm, ltol the BiCGStab relative tolerance, at most 20 Newton iterations per step.
- Ineligible arms remain in every table, marked.

## 7. Sealed held-out evaluation

After the three validation panels and their certificates are collected, audited and the selection is committed as
`selection.json` (sha256 recorded in the lab log), one job per mesh runs the selected accurate and fast arms and the
whole FOM grid on the sealed cohort 923401 (32 cases), once. The FOM of the held-out row is chosen by the same rule on
the held-out errors (mechanical). Both validation and held-out rows are reported; the held-out rows are the headline.

## 8. Timing protocol (every panel job)

GPU query scope: supplied dense initial interior field resident on the GPU → six dense GPU output fields,
`block_until_ready` on both sides; compilation excluded (every program is compiled and run once per case untimed
first). 2 s GPU burn-in. Order **A–B–A**: phase A1 = every ROM arm × every case × 3 repetitions in a seeded random
order; phase B = every FOM setting × every case × 3 repetitions, random order; phase A2 = as A1. A cool-down of
min(1 s, previous time) follows any invocation ≥ 0.5 s. ROM time = median over all 6 × cases invocations; FOM time =
median over its 3 × cases. Gates (fixed now; a failing gate marks the mesh's rows *provisional*, it does not drop or
re-time them): **drift** — per ROM arm, median(A2)/median(A1) ∈ [1/1.10, 1.10]; **neighbour** — within A1∪A2, each
invocation normalised by its (arm, case) median; mean normalised time of invocations whose predecessor arm has median
≥ 4× this arm's, divided by the mean for the other invocations, ≤ 1.10 pooled over arms. Timing ratios never cross
allocations.

## 9. Gates and audits

In-job: GPU backend (`jax_backend=gpu`), f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`; cohort table hashes; model file
hashes; $A$ by DST vs explicit $\Phi$ columns ≤ 1e-12; tensor vs direct $\Phi^\top(u\odot D^-u)$ at random coefficients
≤ 1e-10; $\hat G=QR_q$ reconstruction ≤ 1e-12; **parity** — (a) trial-space: projection of every reference field onto
span($\hat G$) vs span($G$) (unrotated) ≤ 1e-10 relative, (b) the head K = 32 lat16 arm at R' = R with the rotated
bank vs the unrotated bank with $T$ folded into the head's last layer and skip: fields ≤ 1e-10 relative and identical
iteration counts (33 and 65 nodes); reference residuals ≤ 1e-9 relative; every invocation finite.

Independent NumPy audit (`audit.py`, run locally on the collected files): recompute every reported error from the saved
restricted fields (15³ lattice x = k/16, all arms, cases, times) and compare with the in-job restricted errors
(≤ 1e-12), recompute the full-grid errors of case 0 for the reference-audit arms from saved full fields (≤ 1e-10);
recompute $\rho$ in NumPy (NumPy bank features, NumPy upwind, explicit sine tests) for the arg-max state of every arm
at 33 nodes and a random subset at 65 nodes (≤ 1e-8); backward-Euler reference residual in NumPy for case 0 at steps
9→10 (33 and 65 nodes); recompute the selection from the recorded numbers; two injected controls must be detected (a
swapped case, a perturbed error).

## 10. Stopping rule, compute, disk

If by **2026-09-24 20:00 EDT** no eligible arm at 65 nodes on the validation cohort is both within the fast bar (5 %)
and faster than the §6 FOM of that mesh, stop: report the negative result with the diagnosis and do not open the
sealed cohort. Disk: no full 3D field is saved for every case; saved per (arm, case): six fields on the 15³ lattice;
full fields only for case 0 of the reference and four audit arms (span R'=R tensor, span 128 tensor, head K32 R tensor,
FOM Δt 0.005 ntol 1e-3 ltol 0.5); training snapshots never leave the job. GPUs: A100-80GB / H100 / H200 only. The
paralab free space is checked before submission.

## R1 — revisions after the independent design audit (2026-09-23, before any job; override §§3–10 where they differ)

Codex read-only audit: `checks/codex-design-audit-2026-09-23.txt` (2 blockers, 10 majors, 1 minor). Disposition:

1. **Non-finite / unconverged FOM data (blocker) — fixed.** Training data, references and FOM rows record finiteness
   and the Newton status; training data and references abort unless every field and residual is finite and every step
   meets its tolerance within the 20-iteration cap. FOM grid rows that hit the cap are flagged (they stay comparators
   only if finite; a capped row is marked).
2. **Comparator of the fast setting (blocker) — partly accepted.** The brief fixes the paper rule (both speedups of a
   mesh divide the time of the fastest FOM at least as accurate as the *accurate* setting); that stays the primary
   Table-1 number. Every arm is additionally reported against its *own* matched FOM (the fastest FOM at least as
   accurate as that arm). **The stopping rule (§10) uses the stricter own-matched comparator**: a setting passes only
   if it is within 5 % and faster than the fastest FOM setting at least as accurate as itself.
3. **Head metric off the ordering grid (major) — fixed.** Head training, validation and the nearest-code search now
   use each group's own field metric (the triangular factor of the ordered bank on that group's points); targets are
   trained at unit scale and the scale is folded exactly into the last layer and skip.
4. **Held-out FOM selection (major) — both reported.** Primary: the §6 rule applied mechanically on the held-out
   errors (it can only strengthen the FOM). Beside it: the FOM setting chosen on validation, frozen.
5. **Tensor wording and certificate contract (major) — accepted.** The tensor is a *certified backward-advection
   surrogate* of the sign-upwind tested advection, exact on nonnegative states; it does not solve the sign-upwind
   objective exactly. Certified states are the accepted step endpoints $u_1,\dots,u_{50}$ of the deployed solver
   (LM trial states and Jacobians are not certified); a non-finite $ho$ fails. The bar 0.116 is the paper's
   transferred empirical threshold, not an error bound. Added control arm `exact` (span, R' = R, dense sign-upwind
   tested advection on every node with its exact Jacobian; untimed, never eligible) on the validation cohort at 33
   and 65 nodes: the field difference between the rule arms and `exact` measures what the rule changes in the solution.
   The minimum decoded value on certified states is reported (negative-state evidence).
6. **Stalls (major) — fixed.** LM exit reasons 0 (budget) and 2 (tiny step without stationarity) are non-stationary
   exits. Eligible iff no reason-3 exit, non-stationary exits on ≤ 1 % of all evolution steps, all outputs finite.
   Head initial fits are reported (iterations, exit, gradient) but do not gate.
7. **Bank-loss wording (major) — accepted.** The loss is a truncated-POD surrogate of the mean projection error
   (top 1536 modes per group; the discarded tail energy is recorded) plus the tail term, with a conditioning
   regulariser that is not span-invariant.
8. **Model-selection evidence (major) — fixed.** Full-grid bank floors use all 48 bank-validation trajectories; the
   condition number of the bank on the 65-node grid (singular values of $R_G$) is recorded and used by the §3 rule.
   Tie or both-fail: R = 512 if finite and conditioned, else R = 384.
9. **Training memory (major) — accepted as a risk.** Both training jobs run on H200 (141 GB), 240 GB host memory.
10. **Tensor setup cost (major) — accepted, measured.** Setup times are logged and reported; they are offline and not
    charged to the query (as every precomputed table in the paper).
11. **Audit coverage (major) — strengthened.** Saved restricted fields use an OFFSET lattice $x=(2k+1)/32$ ($16^3$
    nodes, disjoint from the EQ lattice); full fields of case 0 for the reference and the audit arms at every mesh;
    the NumPy $ho$ audit includes a sign-mutation control (downwind stencil) that must disagree.
12. **FOM-phase timing effects (major) — fixed.** The neighbour gate is also applied within phase B (FOM invocations
    normalised by their (setting, case) median). If any timing gate fails at a mesh, the timed phases are repeated
    once in a new job; if a gate fails again, that mesh's speedups are labelled provisional and cannot satisfy the
    stopping rule.
13. **Shell completion (minor) — every arm asserts its actual M ≥ requested and M ≤ the table's M.**

## Glossary

- **Bank / ordered bank / R'**: the learned spatial functions; the same functions rotated once so the first R' carry the
  most training energy; the number of those columns a query uses.
- **Span / head**: a query that solves for R' coefficients directly / for K latent coordinates through the network head.
- **M, tests**: the number of sine test functions the residual is projected on.
- **Rule, tensor, lat16, ρ**: how the nonlinear advection term is evaluated; a precomputed quadratic form; a uniform
  lattice quadrature; the relative error of the rule's tested advection against the exact one.
- **Worst evolved error**: over cases, the largest error at the five output times after t = 0, relative to the norm of
  the initial field.
- **FOM**: the full-order Newton–BiCGStab solver on the same mesh.
- **Validation / sealed held-out**: cases used to choose settings / cases opened once after the choice is frozen.
- **A–B–A**: ROM timing block, FOM timing block, ROM timing block, in one GPU allocation.

## A1 — training compute (2026-09-23 17:25 EDT, before any ROM result; no test data seen)

The first training attempts failed or were too slow, none produced a model: 4238911/4238912 and 4239869/4239874
ran out of device memory (fused validation, then a 42 GiB step temporary beside the device-resident snapshots; fixed
by chunked validation and host-side minibatches), and 4241058/4241062 ran at roughly 3 s per bank step on an A100
(all 530k points of the three groups in every step), i.e. about 10 h for the declared 12 000 steps — past the
stopping-rule date. Amended recipe (both widths): each bank step uses a fresh random subset of 32 768 points per
group (the 33-node group keeps all 29 791), for the POD-mode mean term and the minibatch tail term alike — a
stochastic estimate of the same loss; checkpoint selection still evaluates the full point sets. 8 000 bank steps
(was 12 000), 40 000 head steps (was 60 000). Everything else in §3 and R1 is unchanged. The failed attempts' logs
are kept in `runs/failed/`.

## A2 — fixed-sweep fast path and a two-step-size span (2026-09-23 ~18:45 EDT; AFTER the first validation panels)

This amendment is written after seeing the first validation panels (`val33v1`/`val65v1`/`val129v1`, jobs
4244520/4244524/4244527, and their certificates) and is labelled as such. What they showed: the tensor rule is
certified everywhere (ρ ≤ 1e-2), the span ladder is monotone, but at 64³ no eligible arm within 5 % is faster than
its own matched FOM (span R'=128 4.15 % at 21.7 ms vs `fom_dt0.01_nt0.01_lt0.1` 1.18 % at 11.6 ms): the per-step cost
is (i) the LM while loop's host round trips and kernel launches (≈0.18 ms/step even at R'=32) and (ii) three to four
full reads of the tensor per step. The held-out cohort is untouched.

Added arms (validation panels re-run in full, new job directories `*A2`, every v1 arm re-timed in the same
allocation; certificates run for the new arms): span, tensor rule, **fixed sweeps** — the same scaled residual, LM
step (Cholesky, damping, clip, accept-if-decrease) and predictor, but exactly 3 sweeps on the first two steps (no
extrapolation history) and 1 sweep afterwards, the time loop unrolled (no while loop), the tensor read once for all
three predictor candidates and once after the sweep. Stationarity is measured after the last sweep and reported; the
§6/R1 eligibility (non-stationary exits ≤ 1 % of steps) applies unchanged. For R' ∈ {512, 384, 256, 192, 128, 96, 64}
and backward-Euler steps Δt ∈ {0.005, 0.01} (the FOM grid has the same Δt knob; the reference stays Δt = 0.005).
Selection (§6 with R1) is applied to the A2 panels, which contain every v1 arm; the v1 panels are reported as the
first validation panels. Nothing else changes (bar, rules, FOM grid, cohorts, gates, stopping rule).

## A3 — adaptive start for the fixed-sweep path (2026-09-23 ~19:15 EDT; AFTER the A2 validation panels at 33/65)

Written after seeing `val33x`/`val65x` (jobs 4244595/4244597) and their certificates; labelled as such; held-out
untouched. At 64³ the A2 fixed-sweep arms did not change errors (identical to the LM arms to 4 digits) and cut time
by ~20–30 %, but the stopping rule still fails: the cheapest arm within 5 % that beats its own matched FOM would be the
Δt = 0.01 fixed-sweep R'=128 arm (3.94 %, 9.37 ms vs `fom_dt0.01_nt0.01_lt0.1` 11.68 ms), which is **ineligible**
under R1 item 6 because 6 of its 400 steps (1.5 %) end non-stationary — all at step 0 (after 3 sweeps) or step 2
(first quadratic extrapolation). Added arms `fsa`: identical to `fs1` except that the first three steps use the
adaptive LM of the original arms (while loop, the same stopping test, budget 50), R' ∈ {512,…,64}, Δt ∈ {0.005, 0.01}.
The A3 panels contain every v1 and A2 arm again, timed in one allocation per mesh; selection (§6, R1, eligibility
unchanged) is applied to the A3 panels. The A2 rerun at 129 (`val129y`) is cancelled as superseded. No rule, bar,
cohort, FOM setting or gate changes. This is the last amendment before the stopping-rule date; whatever the A3
panels show is applied as is.

## A4 — correction of the selection's certificate inputs (2026-09-23 ~19:30 EDT; held-out not seen)

The first frozen selection (`checks/selection-superseded-fsa-only-certs.json`) read certificates only from the A3
certification jobs, which certified the new `fsa` arms only; every other arm was therefore marked ineligible for
"no certificate", although its certificate exists (v1 jobs 4244521/4244525/4244529 for the LM/lat16/head arms, A2
jobs for the `fs1` arms). `select.py` now merges the certificates of all certification jobs of a mesh by arm name.
Re-applying the unchanged rule: the fast arms and the FOM are unchanged at every mesh; the accurate arm becomes the
LM arm at η = 1e-6 (33, 65 nodes) and the `fs1` Δt = 0.005 arm (129 nodes), whose errors equal the previous `fsa`
accurate arms' to 7 digits (e.g. 0.9852577 % vs 0.9852582 % at 65) — solver-tolerance noise — but which are slower
at 33/65. The rule is applied literally (no tie-break was pre-registered). The first held-out attempt
(jobs 4246320/4246321/4246323) had already started with the superseded selection; it was cancelled ~45 s after
start, before any result was read (one quick-run line had been logged at 33 nodes; its logs are archived unread in
`runs/failed/`). The held-out jobs are resubmitted with the corrected selection and, as a labelled extra arm, the
superseded accurate arm (so both choices are reported on the sealed cohort).
