# lshape — Poisson on the L-shaped domain: the separable decoder against sparse direct and PCG

Pre-registered before any implementation. Amendments are dated in place, appended as
§A1, §A2, …, and never rewrite a criterion.

Worktree `worktrees/2026-09-17-lshape`, branch `exp/2026-09-17-lshape`, forked from
`exp/2026-09-16-p-bank-head` at `266dea9d`. Cluster namespace
`/cluster/tufts/paralab/tawal01/lshape_20260917/`. Paper table T18.

---

## 1. The question

Reviewer GwrW asked for "Poisson in a domain with a corner" as the less well-conditioned
case, and observed that on the square the direct DST solver "significantly outperforms" the
method. On the L-shaped domain the DST does not apply (the operator is not separable), the
solution carries the $r^{2/3}$ re-entrant-corner singularity, and the fair full-order
comparators are a sparse direct solve and preconditioned conjugate gradients. The question
this cell answers, per mesh:

> Once a sparse direct solve and PCG at several tolerances are placed on the same
> error–cost plane as the separable-decoder ROM (head at $q=0$, its correction ladder) and
> POD-LSPG, **is any reduced model non-dominated, and at which mesh?**

Everything else here — the FOM verification, the boundary-factor question, the bank and
head layers — exists to make that plane trustworthy.

## 2. Domain, discretisation and the full-order model

**Domain.** $\Omega = (0,1)^2 \setminus [\tfrac12,1)^2$: the unit square minus its
upper-right quadrant, re-entrant corner at $(\tfrac12,\tfrac12)$, interior angle
$3\pi/2$. Homogeneous Dirichlet data on all six boundary segments.

**Mesh.** The $N$-interval square grid $x_{ij} = (i/N, j/N)$ with $N$ even, so that the
corner is a node. Interior (unknown) nodes are
$\{(i,j): 1 \le i,j \le N-1,\ \neg(i \ge N/2 \wedge j \ge N/2)\}$, in lexicographic
($i$-major) order; their count is $n = (N-1)^2 - (N/2)^2$:

| $N$ | 32 (smoke) | 64 | 128 | 256 | 512 | 1024 (reference) |
|---|---:|---:|---:|---:|---:|---:|
| $n$ | 705 | 2945 | 12033 | 48641 | 195585 | 784385 |

**Operator.** The 5-point finite-difference Laplacian, $A = -\Delta_h$, with every
neighbour outside $\Omega$ (including the nodes on the two inner segments
$\{x_1=\tfrac12, x_2 \ge \tfrac12\}$ and $\{x_2=\tfrac12, x_1\ge\tfrac12\}$) set to
zero. $A$ is a principal submatrix of the square's 5-point matrix and is therefore
symmetric positive definite.

**Operator verification gate (G-FOM), run in every job before any other number, on every
mesh used, and required to pass:**

* **G-FOM-1 symmetry:** $\max_{ij}|A_{ij}-A_{ji}| \le 10^{-12}$ (it is exactly $0$ by
  construction; the gate is the check that the construction is what it says).
* **G-FOM-2 independent assembly:** the stencil assembly (index arithmetic on the masked
  grid, `lsh_core.assemble_stencil`) and an independent assembly (Kronecker-product
  square Laplacian `kron(I,T)+kron(T,I)` restricted to the interior-node index set,
  `lsh_core.assemble_kron`) agree entrywise: `(A1 - A2).nnz == 0`.
* **G-FOM-3 positive definiteness:** the smallest eigenvalue from the same
  shift-invert Lanczos that produces the test modes is $> 0$, and at $N \ge 128$ it lies
  within $1\,\%$ of the known first Dirichlet eigenvalue of this L-shape,
  $\lambda_1 = 4 \times 9.6397238 = 38.5589$ (Trefethen–Betcke value for the side-2
  L-shape, scaled by $1/L^2$ with $L=\tfrac12$).
* **G-FOM-4 reference accuracy:** the sparse-direct reference solution satisfies
  $\|Au-f\|_2/\|f\|_2 \le 10^{-12}$ for every case on every mesh.
* **G-FOM-5 solver agreement:** the GPU conjugate-gradient solve at its tightest
  tolerance and the CPU ILU-PCG at its tightest tolerance agree with the sparse-direct
  reference to $\le 10^{-8}$ relative on every case.
* **G-FOM-6 square fidelity (local, before the first job):** with the geometry switched to
  the plain square at 255 intervals, this cell's sparse-direct field solve and QR-based bank
  projection floor, applied to the parent lane's incumbent checkpoint (`r128_joint.pkl`,
  `bc_poly` factor), reproduce the parent's audited per-case development bank floors
  (`artifacts/pbh01/result.json`, `D1_bank_floor_development`, 255 intervals; worst
  $2.3227106776681678\,\%$) to $\le 10^{-9}$ relative on all 12 cases.

**Why there is no DST.** The DST diagonalises the square's operator because it is a
Kronecker sum of two 1-D operators with known eigenvectors; the L-shape operator is a
principal submatrix of that sum and is not itself a Kronecker sum, so no fast transform
diagonalises it. This is exactly why the reviewer asked for this case.

**Full-order comparators, all timed in the same job as every reduced model, host
$(N+1)^2$ nodal source array in, host $(N+1)^2$ nodal solution array out (zeros outside
$\Omega$), the identical contract every ROM subject is timed under:**

| subject | where | what is timed |
|---|---|---|
| `fom_splu` | CPU (SciPy SuperLU, `scipy.sparse.linalg.splu`, COLAMD ordering) | gather, one `lu.solve`, scatter; the factorisation is offline and its seconds are reported separately |
| `fom_cg_gpu_r{1e-2,1e-4,1e-10}` | GPU (JAX, matrix-free 5-point stencil on the masked grid, `lax.while_loop`) | device put, CG to relative residual $\le$ rtol, device get; iteration count recorded |
| `fom_pcg_ilu_cpu_r{1e-2,1e-10}` | CPU (SciPy `cg` with `spilu(A, drop_tol=1e-4, fill_factor=10)` as preconditioner) | gather, PCG to rtol, scatter; the ILU factorisation is offline; iteration count recorded |

CHOLMOD / `sksparse`, UMFPACK and `pyamg` are not installed in the cluster venv (checked
2026-09-17), so SuperLU is the sparse direct solver and ILU-PCG is the preconditioned
iterative comparator; a true IC(0) is not available in SciPy and is not hand-rolled. The
diagonal of $A$ is constant on interior nodes, so Jacobi-PCG coincides with plain CG; the
plain GPU CG is therefore the "Jacobi" rung and is labelled as such.

**Reference solution.** `fom_splu` followed by two steps of iterative refinement, untimed,
verified by G-FOM-4. Every reported error is against this same-grid reference. A
"physical" error against the 1024-interval sparse-direct solution restricted to nested
nodes is also recorded, together with the successive-mesh differences, which document the
reduced FD convergence rate the corner singularity imposes.

## 3. Source family and cohorts

The parent lane's Gaussian family, $f(x) = a\exp(-|x-c|^2/2w^2)$ with
$c \sim U[0.15,0.85]^2$, $\log w \sim U(\log 0.02, \log 0.1)$, $a \sim U(0.5, 2)$,
drawn by `core.source_params(seed, count)` (each parameter column drawn at its own
length, so a draw of length $m$ is not a prefix of a draw of length $m' > m$). The source
is **restricted to $\Omega$**: only values at interior nodes of $\Omega$ enter the
equation. **Draws whose centre lies in the removed quadrant ($c_1 \ge \tfrac12$ and
$c_2 \ge \tfrac12$) are rejected**, because a source whose mass sits outside the domain
gives a solution driven by the Gaussian tail alone, which is a numerically meaningless
case for a *relative* error. Rejection keeps the family's conditional distribution; the
draw length and the accepted prefix are recorded for every cohort:

| cohort | construction | count | role |
|---|---|---:|---|
| training | `source_params(0, 4608)`, centre in $\Omega$, first 3072 | 3072 | bank and head training |
| — fit split | seeded 85 % (`split_seed = 20260916`) | 2611 | the only data any optimizer sees |
| — internal validation | remaining 15 % | 461 | head reporting |
| common selection | `source_params(20260916, 512)`, centre in $\Omega$, first 256 | 256 | **bank selection** (one fixed cohort, equal size for every arm — the parent's amendment-4 rule) |
| development | `source_params(20260917, 64)`, centre in $\Omega$, first 32 | 32 | every reported solve number; selects nothing |

Pairwise disjointness of all three cohorts is asserted in code (`np.allclose` on the
4-vectors); a failure aborts the job. Final paper cohorts stay sealed.

Training mesh $N=256$ ($n = 48641$). Evaluation meshes $N \in \{64, 128, 256, 512\}$.
Bank floors are reported at $N = 256$ and $512$.

## 4. Decoder: the boundary factor on the L-shape

The decoder is the parent's separable form $u(x;z) = \mathrm{bc}(x)\,g_\phi(x)^\top h_\theta(z)$
with $g_\phi$ the random-Fourier-feature MLP ($n_{\rm ff}=64$, scale 4, 2 hidden layers of
width $R$), $h_\theta$ an MLP with a linear skip (2 hidden layers of width 128), SiLU,
float64. Only the boundary factor changes; the parent's
$\mathrm{bc}(x) = 16x_1(1-x_1)x_2(1-x_2)$ is square-specific (it does not vanish on the
two inner segments).

**Pre-registered primary factor — `smooth` (R-function product).** With
$a = \tfrac12 - x_1$, $b = \tfrac12 - x_2$,

$$\mathrm{bc}_{\rm s}(x) \;=\; 16\,x_1(1-x_1)\,x_2(1-x_2)\;\psi(x),\qquad
\psi(x) \;=\; a + b + \sqrt{a^2+b^2}.$$

$\psi$ is Rvachev's R-disjunction of the half-planes $x_1 < \tfrac12$ and $x_2 < \tfrac12$:
$\psi \ge 0$, $\psi = 0$ exactly on $\{a\le 0, b \le 0\}$ — the closed removed quadrant,
hence the two inner segments and the corner — and $\psi > 0$ everywhere in $\Omega$. It is
$C^\infty$ away from the corner and Lipschitz at it. Its normal derivative on each inner
segment is nonzero (on $x_1=\tfrac12$, $x_2>\tfrac12$: $\psi \approx a + a^2/2|b|$), so the
factor vanishes **linearly** on every straight wall, which is the behaviour of the true
solution there; at the corner it vanishes like $r$. **Why it is primary:** it is smooth
wherever the solution is smooth, so the bank inherits no interior kinks.

**Declared secondary factor — `sdf` (signed distance).** $\mathrm{bc}_{\rm d}(x) =
4\,\mathrm{dist}(x,\partial\Omega)$, closed form as the minimum over the six boundary
segments (the two inner segments reduce to the corner point for $x$ in the lower-left
quadrant). It vanishes exactly on $\partial\Omega$ and nowhere inside, is Lipschitz, but
has **gradient kinks along the medial axis inside $\Omega$** (the bisectors), where the
product $\mathrm{bc}_{\rm d}\,g$ cannot be smooth for any smooth $g$. **Pre-registered
expectation:** a worse bank floor than `smooth` at equal $R$, because of those interior
kinks. Both factors are run, as the brief asks, because both are cheap.

**Declared diagnostic arm — `enrich`.** The `smooth` bank at $R=512$ plus two fixed
columns, the classical corner singular functions cut off at the outer walls,

$$s_j(x) \;=\; 16\,x_1(1-x_1)\,x_2(1-x_2)\; r^{2j/3}\sin\!\bigl(\tfrac{2j}{3}(\varphi-\tfrac{\pi}{2})\bigr),\quad j=1,2,$$

with $(r,\varphi)$ polar coordinates about the corner, $\varphi\in(\tfrac\pi2, 2\pi)$
in $\Omega$, each column normalised to unit mean square on the training mesh. They vanish
on all of $\partial\Omega$ and carry exactly the $r^{2/3}$ behaviour a smooth product cannot.
The stored rank is $R+2 = 514$. **Its purpose is diagnostic:** if the bank floor of `smooth`
is limited by the corner, `enrich` will show it by a floor drop at nearly identical cost.
It is a legitimate bank (fixed known functions, no data), so it *can* be selected by the
bank rule.

**Declared iteration arm — `smooth_ff128s2`.** The brief's lever (a): $n_{\rm ff}=128$
Fourier features at scale 2 (lower bandwidth, more features), `smooth` factor, $R=512$.

**Bank arms (six), all at $S=3072$, $K=16$ during bank training, the parent's four-phase
recipe (J1 100000 @ $10^{-3}$, B 40000 @ $3\times10^{-4}$, H 30000 @ $3\times10^{-4}$,
J2 30000 @ $10^{-4}$; minibatch 64 sources × 4096 nodes; orthogonality weight $10^{-4}$;
warmup 0.02; cosine to 1 %; seeds 20260916101–104), so that the only differences are the
factor, $R$ and the feature count:**

| arm | factor | $R$ | $n_{\rm ff}$ / scale |
|---|---|---:|---|
| `smooth_R256` | smooth | 256 | 64 / 4 |
| `smooth_R512` | smooth | 512 | 64 / 4 |
| `sdf_R256` | sdf | 256 | 64 / 4 |
| `sdf_R512` | sdf | 512 | 64 / 4 |
| `enrich_R512` | smooth + 2 singular columns | 514 | 64 / 4 |
| `smooth_ff128s2_R512` | smooth | 512 | 128 / 2 |

**Reported for every bank arm** at $N=256$ and $512$: worst and median bank projection floor
on the fit split (256-source seeded subsample), the validation split, the common selection
cohort and the development cohort; numerical rank and condition number; the jointly
trained head's best-found on validation and development.

**Bank selection rule.** Lowest **worst floor on the common selection cohort at $N=256$**;
ties by median, then smaller $R$, then table order. The development cohort selects nothing.

**Bank target.** Worst development floor at $N=512$ **below 1.0 %** (the parent's bar; the
parent's selected square bank reached 0.7419 %).

## 5. Head layer

With the selected bank frozen the reconstruction loss separates through the thin QR exactly
as in the parent (§5 of `p-bank-head/DESIGN.md`), so heads are fitted full batch in $R$
dimensions. The parent measured the weak-residual term neutral and the code-smoothness term
harmful, so **the head objective here is plain reconstruction only**,
$\mathcal L_{\rm rec}$, 150000 full-batch Adam updates at $10^{-3}$ with cosine decay,
fresh codes $Z\sim 0.1\,\mathcal N(0,I)$, seed 20260916201.

**Head arms:** on the selected bank $K\in\{16,32\}$ — the two **primaries**; on every
other bank $K=16$ — the declared comparison heads (`factor swap`, `enrichment`,
`ff128s2`), so the boundary-factor question is answered at the head layer and not only at
the bank floor. Each head arm also gets a nested correction basis of $q_{\max}=128$
directions (right singular vectors of its normalised training residuals in the QR metric,
verified orthonormal in the field metric to $10^{-8}$).

**Reported per head arm** at $N=256$: training error at the stored codes, best-found on
validation and development (multistart LM, 8 starts, budget 400, the parent's oracle), the
ROM's own untimed weak solve on the development cohort (LM budget 300, gtol $10^{-6}$),
and the three-layer decomposition (bank floor ≤ best-found ≤ solved).

**Head target.** Worst development best-found within **1.2×** the same bank's floor (the
parent's bar; the parent missed it at 4.2–4.5×, and it is kept so the two cells are
comparable). A miss is reported as a miss, and the brief's iteration levers (a) features,
(b) $K=32$, (c) factor swap are all already declared arms above, so a failure to train
produces findings, not a rescue.

## 6. Weak residual and test modes

The test family is the faithful analogue of the parent's sine modes: the $M=257$ lowest
eigenpairs $(\Phi,\Lambda)$ of $A$ on each mesh (shift-invert Lanczos on the SuperLU
factor; eigen-residual $\|A\Phi-\Phi\Lambda\|_F/\|\Phi\Lambda\|_F$ recorded). The residual
is formed with the operator, not the eigen-identity:

$$r(z) = B\,h_\theta(z) - f_m,\qquad B = \Lambda^{-1}\Phi^\top A\,G,\qquad
f_m = \Lambda^{-1}\Phi^\top f,$$

so it is exact for whatever $\Phi$ the eigensolver returns; $\Lambda^{-1}$ is row scaling.
Online, $f_m$ is a dense $M\times n$ matrix–vector product: the square's skinny
sine-product projection does not exist here, and this cost is charged inside every reduced
query. Solver: `arms.make_stationary_lm`, budget 300, gtol $10^{-6}$, trust radius = radius
of the training-code cloud, nearest-training-code initialiser (in the projected residual),
Gauss–Jordan step for $\le 64$ unknowns and pivoted LU above — the parent's arm (a).

**Correction ladder $q\in\{0,32,64,128\}$, exact analytic elimination.** With
$C\in\mathbb R^{R\times q}$ the first $q$ directions, $L = BC = Q_qR_q$,
$B_\perp = (I - Q_qQ_q^\top)B$, $f_\perp = (I-Q_qQ_q^\top)f_m$: solve
$\min_z\|B_\perp h_\theta(z) - f_\perp\|$ by the same LM, recover
$y = R_q^{-1}Q_q^\top(f_m - Bh_\theta(z))$, output $G(h_\theta(z)+Cy)$. One solver for
every arm (a deviation from the parent's $q=32$ path, which used `kernel_solver`; recorded).
**The $q=R$ rung** ("free bank") is constructible only when $M > R$; at $M=257$ that is the
$R=256$ banks only. For $R=512$ the rung is reported as *not constructible at $M=257$* and
the untimed bank floor stands in for its error.

**POD-LSPG** at $k'\in\{16,32,64,128,256\}$ from the 3072 training snapshots regenerated on
each evaluation mesh (exact Gram, blockwise), same residual $B_{k'} = \Lambda^{-1}\Phi^\top AV_{k'}$,
identity head, same solver.

## 7. The solve jobs

Three solve jobs, one attempt directory each: `N ∈ {64, 128}` together (both tiny), `N=256`,
`N=512`. (The brief says one job per mesh; folding 64 and 128 into one job costs nothing in
timing isolation — every comparison is same-job — and keeps one more job for iteration under
the 8-job cap.) Per mesh, in one process on one GPU:

* subjects: the two primaries at $q\in\{0,32,64,128\}$ (+ $q=R$ where constructible), the
  four comparison heads at $q=0$, POD-LSPG at five ranks, the six full-order comparators;
* 32 development cases × 3 timed repetitions, randomised subject order per (repetition,
  case), 0.1 s GPU burn before every invocation, compile warm-up excluded, every
  repetition's timing retained, one output field per (subject, case) stored (repetitions
  are bit-identical and deduplicated by hash);
* per invocation: same-grid error vs the sparse-direct reference, physical error vs the
  restricted 1024-interval reference, complete-query ms (host in → host out), device ms
  (fused GPU interval) or solver ms (CPU solve interval), LM/CG iterations, exit reason,
  normalised-gradient stationarity;
* untimed: bank floor and best-found for every checkpoint on the 32 cases at that mesh.

## 8. Pre-registered pass/fail

**The deliverable is the non-dominated set**, defined per mesh on the plane
(worst same-grid error over the 32 cases, median complete-query ms over cases and
repetitions): subject $A$ dominates $B$ if $\mathrm{err}_A \le \mathrm{err}_B$ and
$\mathrm{cost}_A \le \mathrm{cost}_B$ with at least one strict. The set is reported with
and without the full-order comparators, on complete-query ms (primary) and on
device/solver ms (secondary), and at every mesh. **The headline statement is whether any
reduced model is in the set once `fom_splu` and the PCG rungs are included, and at which
mesh.** A reduced model that is dominated by `fom_splu` is reported as dominated; no
speedup over any full-order solver is claimed unless it appears in that set.

**Cell success (all four, jointly), evaluated at $N=256$ on the 32 development cases:**

1. a frozen primary with worst solved same-grid error $< 5.0\,\%$ (the parent's square cell
   reached 3.11 % on 12 cases at $K=32$; 32 cases and a singular family justify a looser bar,
   set before any number exists);
2. every one of its 32 solves at $q=0$ exits stationary (reason 4);
3. it beats POD-LSPG at the matched rank $k'=K$ on worst error;
4. the G-FOM gates all pass on every mesh.

Partial success is reported as partial. **Honesty clauses:** report whether POD-LSPG at
$k'=8K$ matches the neural head; report `fom_splu` cost and error at every mesh, next to
the ROM numbers; report the bank and head targets as pass/fail on their own terms; if the
selection rule and the development ranking disagree, say so.

**Falsification.** The working hypothesis is that the R-function factor lets the separable
bank reach, on the L-shape, a floor within $2\times$ the parent's square floor
($0.7419\,\% \Rightarrow < 1.5\,\%$), and that the corner is not what limits it. It is
falsified if every `smooth` arm's development floor at $N=512$ exceeds $1.5\,\%$ **and**
`enrich_R512` improves on `smooth_R512` by more than $1.5\times$ — that reading says the
corner singularity, not the factor, binds, and it would be reported as the cell's negative
result. If `enrich` does *not* help while the floor is still above the bar, the limit is
elsewhere (coverage or the feature family) and is reported as open.

## 9. Resources and process

* Cluster jobs: cap 8; planned 4 (`lsh01` train, `lsh02`–`lsh04` solve), 4 in reserve for
  iteration. One job per attempt directory; `squeue` before and after every submit; A100
  first, then H100/H200/L40S after 3 h pending; `pax007` excluded; `gpu` partition only;
  `jax_backend=gpu` asserted; float64; matmul precision `highest`; `--mem 180G`.
* Local GB10: sub-minute smokes at $N=32$ through the slot helper only, plus the G-FOM-6
  square fidelity gate.
* Every trained checkpoint and every correction basis is Git-tracked under
  `experiments/lshape/checkpoints/`. Archives are checksum-collected, chunked, Git-tracked,
  then the exact remote attempt directory is deleted.
* An independent NumPy/SciPy audit (`lsh_audit_np.py`, imports neither JAX nor any driver)
  recomputes every reported error from the retained fields, re-derives every bank floor from
  the saved weights, re-runs the operator gates and re-checks every correction basis.
* The report is generated by `reports/generate_lshape.py` from the run JSONs into
  `reports/2026-09-1x-lshape.md` and `reports/summary.json`. No number is typed by hand.
* Codex audits DESIGN.md before `lsh01` and the report after the last job.

---

### Amendments

**2026-09-17, §A1 — pre-job audit record (before `lsh01`).** (a) The Codex audit required by
the lane protocol could not run: the Codex account usage limit is exhausted until 2026-09-19
11:33 (`reports/codex-design-audit.md` is absent; the attempt is logged in the scratchpad). A
fresh-context read-only Claude review agent was launched as a substitute and was itself killed
by the API session limit before reporting. The self-audit that was done instead, and is
recorded here: a CPU-only NumPy check of the geometry ($n$ matches $(N-1)^2-(N/2)^2$ at
$N\in\{32,64,128,256\}$), the two operator assemblies (entrywise identical, exactly
symmetric), the first eigenvalue (relative difference to the Trefethen–Betcke value
$3.5\times10^{-3}, 1.7\times10^{-3}, 7.6\times10^{-4}, 3.2\times10^{-4}$ at those meshes —
converging at the reduced rate the corner imposes), the cohort acceptance counts (3447 of 4608,
386 of 512, 50 of 64 centres in $\Omega$), both boundary factors (exactly zero on every
boundary node, positive inside), the closed-form SDF against a brute-force segment distance
(max abs difference 0), and the singular columns. (b) **One bug found and fixed before any
job:** the singular columns used `atan2` with $\varphi<0 \mapsto \varphi+2\pi$, which leaves
the inner segment $x_2=\tfrac12, x_1>\tfrac12$ at $\varphi=0$ where $\sin(\tfrac23(0-\tfrac\pi2))\ne0$;
the mapping is now $\varphi\le0\mapsto\varphi+2\pi$ and the columns vanish on that segment to
$4\times10^{-17}$. Interior nodes never sit on that ray, so no result would have moved, but
the boundary claim in §4 would have been false as coded. (c) G-FOM-6 passed locally:
worst relative difference to the parent's audited square floors $5.30\times10^{-11}$ (JAX path)
and $5.08\times10^{-11}$ (pure NumPy path) against $10^{-9}$; sparse-direct vs DST parity
$6.1\times10^{-14}$. (d) The $N=32$ training smoke ran end to end and its NumPy audit passed
every check. Nothing in §§1–9 changes.

**2026-09-17, §A2 — the CPU preconditioned comparator is IC(0)-PCG, not ILU-PCG. Found by the
solve smoke, before any solve job.** §2 declared `fom_pcg_ilu_cpu_r*` with SuperLU's `spilu`
as the preconditioner and said a true IC(0) "is not hand-rolled". The $N=64$ smoke showed why
that was wrong: SuperLU's ILU is a threshold ILU with partial pivoting and a column
permutation, so the preconditioned operator is **not symmetric** and conjugate gradients has
no convergence guarantee with it — at $N=64$ it ran to the 100000-iteration cap at a
relative error of $1.6\times10^{-2}$ (at $N=32$ it converged in 4 iterations, which is how
the flaw hid). The comparator is replaced by a zero-fill incomplete Cholesky, IC(0),
computed by the classical recurrence on the 5-point pattern (`lsh_core.ic0_factor`: no
common predecessors exist in the zero-fill pattern, so $L_{kk}^2 = A_{kk} - L_{kW}^2 -
L_{kS}^2$), applied through two SuperLU triangular solves in natural order; its
pattern residual $\|(LL^\top - A)\circ\mathrm{pat}(A)\|_{\max}$ is recorded per mesh. The
subject is named `fom_pcg_ic0_cpu_r{1e-2,1e-10}` and everything else in §2 stands. Measured
on the local box at $N=512$ (loaded, for iteration counts only): IC(0)-PCG 189 iterations at
$10^{-2}$ and 519 at $10^{-10}$ against 1746 for plain CG at $10^{-10}$.

**2026-09-17, §A3 — `lsh01` ABORTED on a mis-scaled numerical gate; the gate is rescaled and
the job is rerun in full. No criterion changes.** Job `3780148` (A100, `pax144`, 59:56
elapsed) completed the operator gates, all six bank arms, the bank selection and seven of the
eight head arms, then died on the eighth (`head_smooth_ff128s2_R512_K16`) at
`assert binfo['orthogonality_error'] < 1e-8` with a measured $1.8455\times10^{-8}$. The
threshold was inherited from the parent cell, where the correction basis has $q_{\max}=32$
columns; §5 of this design raised $q_{\max}$ to 128 without rescaling it. The quantity is
$\|W^\top W - I\|_F$ over a $q\times q$ matrix, so at fixed per-entry round-off the absolute
Frobenius norm grows like $q$; per column the measured value is
$1.8455\times10^{-8}/\sqrt{128} = 1.63\times10^{-9}$, i.e. round-off, and the basis was fine.
**The gate is now on the per-column value** $\|W^\top W - I\|_F/\sqrt{q} < 10^{-8}$
(`basis_orthogonality_limit` in the config), and the absolute norm and the maximum entry
deviation are both recorded beside it. The independent audit keeps its own, separate absolute
bound of $10^{-6}$ on $W = G\,\mathrm{directions}$ formed on the full grid, which accumulates
more than the driver's QR-metric form; both are conditioning checks, not scientific criteria.
**`lsh01` is retracted as an attempt** — its partial `result.json` is archived for the record
and **no number from it is reported** — and the identical job is rerun as `lsh02` with the
rescaled gate. Nothing else in §§1–9 changes, and no criterion, cohort, seed or arm is
altered, so `lsh02` reproduces `lsh01`'s seven completed arms as well as finishing the eighth.

**2026-09-17, §A4 — the independent audit's bank-floor gate is absolute, not relative; found by
the audit of `lsh02`, after the job, and no reported number moves.** The audit of `lsh02` failed
one check, `bank_floors_reproduced`, at a worst **relative** difference of
$5.2360\times10^{-6}$ against its inherited $10^{-8}$ bound. The bound, not the job, was wrong,
and the diagnosis is recorded here in full because the check is the cell's only independent
verification of the floors.

*What was measured.* (i) Two equally valid NumPy routes to the same floor — QR and SVD of the
same feature matrix $G$ — agree to $1.14\times10^{-14}$ relative, so the floor is **not**
cancellation-limited and is well determined in f64 on one machine. (ii) The driver-versus-audit
difference is **not** uniform across arms: $2.5\times10^{-10}$ (`smooth_R256`),
$8.4\times10^{-10}$ (`smooth_R512`), $3.9\times10^{-9}$ (`enrich_R512`) and
$5.2\times10^{-6}$ for `smooth_ff128s2_R512` alone. (iii) It tracks each bank's conditioning:
$\mathrm{cond}(G) = 1.3\times10^{4},\,2.4\times10^{5},\,2.0\times10^{4},\,2.3\times10^{5},\,
1.1\times10^{6},\,8.9\times10^{9}$ over the six arms, and perturbing $G$ by one $\varepsilon$
per entry inside NumPy moves the floor by $8\times10^{-15}$ at the smallest condition number
and $1.0\times10^{-10}$ at the largest. The observed difference is a near-constant
$1.4\times10^{4}$–$2.0\times10^{5}$ multiple of that single-$\varepsilon$ band at **every** arm,
across four decades.

*What that means.* The driver evaluates the random-Fourier features and the MLP on the GPU and
the audit re-evaluates them on the CPU; the two `exp`/`sin`/`cos` and matmul-reduction paths
differ by $\approx10^{-12}$ relative — the cross-machine 1-ulp effect the lane protocol already
lists as a paid-for landmine — and each bank's $\mathrm{cond}(G)$ multiplies it. A fixed
relative bound on this quantity therefore measures conditioning, not correctness, and the one
arm it convicts is the worst arm in the cell on every other axis.

*The change.* The deciding criterion is now **absolute and tied to the reporting precision**:
the difference may not move the last digit of a floor quoted to four decimals in percent, i.e.
$\le 10^{-6}$ in the ratio (`FLOOR_ABS_LIMIT`). Measured worst absolute difference
$4.7233\times10^{-8}$, a factor $21$ inside the bound, and a factor $1.8\times10^{4}$ smaller
than the closest gap in the bank ranking ($8.7\times10^{-4}$, `sdf_R512` against
`enrich_R512` on the common cohort). The tight relative number is still computed, still
reported in `detail`, and is never dropped; $\mathrm{cond}(G)$ is now recorded per arm beside
it. **No floor, no ranking, no selection and no criterion changes**, and `lsh02` is not
retracted: the audit now passes all nineteen checks, including `bank_selection_reproduced`
and `head_stored_code_errors_reproduced` (worst $1.44\times10^{-10}$, still under the
untouched $10^{-8}$ relative bound). The audit script is not staged to the cluster, so the
three solve jobs submitted before this patch are unaffected by it.

**2026-09-17, §A5 — the final-report Codex audit is replaced by a written self-audit.** §9
requires Codex to audit the report after the last job. The Codex account's usage limit is
exhausted until 2026-09-19 11:33 (coordinator-verified), which is after this lane's reporting
window. As in §A1 the substitute is a written self-audit, `reports/self-audit-lsh02.md`,
listing each claim, the `result.json` field it rests on, and the check that was run against it.
If the lane is still open after 2026-09-19 11:33 the Codex audit is run and appended.

**2026-09-17, §A6 — a fifth job carries the $q=R$ free-bank rung for the primaries at $N=256$;
it runs at $M=1024$ test modes, not the $M=513$ the coordinator named, because $M=513$ is
ill-posed on the real banks.** The coordinator's decision (2026-09-17, after the interim
report): the paper's structural claim is that on linear PDEs the top rung is a linear reduced
model and the cheapest point, and the L-shape is the cell where the direct sine-transform
competitor is removed, so the $q=R$ rung is needed here; one more solve job at $N=256$ with the
same six full-order comparators and the POD ranks in the same allocation, so its cost is
comparable to the other rungs *in that job*. Four things were done before staging it.

*(a) The free rung runs no LM.* On that rung $C=I$, so $B_\perp=(I-Q_qQ_q^\top)B$ is zero to
round-off and no nonlinear unknown remains: every coefficient comes from the exact elimination
$y = R_q^{-1}Q_q^\top f_m$. The LM's normalised gradient on a zero operator is a round-off cosine,
so it would climb its damping ladder to exit reason 3 on every query and inflate the cost of what
is by construction a linear reduced model — which would have made the claim under test look false
for an implementation reason. `make_rom_kernel(..., free=True)` skips it; the rung reports
0 iterations, and its stationarity is `full_stationarity`, the gradient of the *full* residual,
which the kernel computes for every rung regardless. `arm_setup` records `free_rung` and
`nonlinear_unknowns = 0`. `lsh03`–`lsh05` never reach this branch ($M=257<R=512$ for every
primary), so the science they were staged with is unchanged; the byte difference in `lsh_solve.py`
between those three attempts and this one is exactly this branch plus the two bookkeeping fields.

*(b) Local smoke of the branch (`checks/smoke-free-rung-summary.json`, `checks/smoke-config-free-rung.json`).*
$N=32$, $M=40$ against the smoke primaries' $R=32$, run without `--smoke` because that flag pins
$M=16$: the rung fires on both primaries, 0 iterations, exit 4, full stationarity
$1.8\times10^{-13}$, `linear_rank` $=32=R$, byte-identical output from the $K=4$ and $K=8$ heads
(as it must be: the rung is head-independent), the cheapest reduced subject in the run, and the
independent NumPy solve audit passes every check. Its *error* on that 705-node toy bank (worst 97.5 %
against a 41 % floor) is not evidence about the real cell, but it is what prompted (c).

*(c) The untimed free-rung error on the real `lsh02` banks as a function of $M$
(`checks/free_rung_M_sweep.py`, one CPU shift-invert eigensolve at $k=1024$, eigen-residual
$7.9\times10^{-14}$; every prefix $M$ is the same lowest-eigenpair set).* The rung is the
least-squares solution of $Bc=f_m$ over all $R$ coefficients and is head-independent, so this is
the number the job will report, obtained ahead of it:

| bank | $R$ | $M$ | free rung worst / median | bank floor worst | worst / floor | $\mathrm{cond}(B)$ |
|---|---:|---:|---:|---:|---:|---:|
| `sdf_R512` | 512 | 513 | 6.2782 % / 1.8691 % | 0.7766 % | 8.08x | 6.35e+06 |
| `sdf_R512` | 512 | 640 | 0.8640 % / 0.3230 % | 0.7766 % | 1.11x | 3.26e+05 |
| `sdf_R512` | 512 | 768 | 0.8042 % / 0.3184 % | 0.7766 % | 1.04x | 2.62e+05 |
| `sdf_R512` | 512 | 1024 | 0.7791 % / 0.3170 % | 0.7766 % | 1.00x | 2.38e+05 |
| `smooth_R512` | 512 | 513 | 8.4852 % / 1.2033 % | 0.7123 % | 11.91x | 4.10e+06 |
| `smooth_R512` | 512 | 640 | 0.8227 % / 0.1399 % | 0.7123 % | 1.16x | 3.38e+05 |
| `smooth_R512` | 512 | 768 | 0.7340 % / 0.1252 % | 0.7123 % | 1.03x | 2.77e+05 |
| `smooth_R512` | 512 | 1024 | 0.7161 % / 0.1225 % | 0.7123 % | 1.01x | 2.52e+05 |
| `enrich_R512` | 514 | 515 | 8.7530 % / 0.8862 % | 0.6676 % | 13.11x | 2.38e+07 |
| `enrich_R512` | 514 | 640 | 0.8204 % / 0.1364 % | 0.6676 % | 1.23x | 1.89e+06 |
| `enrich_R512` | 514 | 768 | 0.6930 % / 0.1249 % | 0.6676 % | 1.04x | 1.47e+06 |
| `enrich_R512` | 514 | 1024 | 0.6719 % / 0.1202 % | 0.6676 % | 1.01x | 1.28e+06 |

At $M=R+1$ the system is one equation over square: bank content outside the span of the lowest
513 eigenmodes is unconstrained, $\mathrm{cond}(B)$ is $10^{6}$–$10^{7}$, and the top rung sits
8–13× above its own floor. From $M\approx768$ it is within 4 % of the floor on every bank and at
$M=1024$ it *is* the floor (1.00×, 1.01×, 1.01×). **The job therefore runs at $M=1024$.** The
coordinator's requirement — every subject in the same allocation so the comparison is within-job —
holds at any $M$; what does *not* hold is cross-job cost comparability: with $f_m=Pf$ a dense
$M\times n$ product charged inside every reduced query, no cost in this job may be set beside a
cost from the $M=257$ jobs, and the report will label the two blocks separately. The deviation
from the named $M=513$ is recorded here as a deviation; if the coordinator wants the $M=513$
measurement timed as well, the untimed answer is the first row of each block of the table above,
and a timed one is a second config away.

*(d) A concern this exposes about the queued jobs.* POD-LSPG at $k'=256$ in `lsh03`–`lsh05` is
solved against $M=257$ test modes — the same one-over-square structure. If it looks poor there,
that is the test-space count, not the POD basis; this job carries POD at all five ranks under
$M=1024$ and is the control that separates the two readings. `lsh03`–`lsh05` are **not** resubmitted
(they are queued; the science is never changed between submits).

*Carried into the next regeneration of the report, on the coordinator's instruction:* (i) the
validation-versus-development worst-case gap (461 cases: 10.7–16.7 %; 32 cases: 3.5–6.8 %) goes
**beside the headline error**, not in a caveat section; (ii) the report generator keys timed
subjects by `(mesh, name)` and would let a second $N=256$ job overwrite `lsh04` — it must key by job
as well and present the $M=257$ and $M=1024$ blocks as separate, non-comparable tables; (iii) a
$K=64$ head is a reasonable use of a spare job only after the solve jobs land, and is not
submitted now. Cluster jobs after this one: five used, three in reserve.

**2026-09-17, §A7 — `lsh05` (N=512) aborted on G-FOM-4; the gate is round-off-floor-aware and
the job is resubmitted once as `lsh07`. The reference field itself does not change.** Job
`3784664` (A100, `pax049`) passed the GPU preflight and the G-FOM-1/2/3 operator gates at
$N=512$ (`lambda1_relative_difference` $1.32\times10^{-4}$) and then died at 1 m 15 s on
`assert resid <= cfg['reference_residual_limit']` with $\mathrm{resid} = 1.0661\times10^{-12}$
against the declared $10^{-12}$, on development case 8. It is not a staging, config-key or
memory failure: the staged manifest verified, `jax_backend=gpu` was printed and the paralab
share was at 92 %, not full.

*What was measured (CPU-only NumPy, `checks/smoke-a7-reference-gate.json` and the diagnosis in
the session scratchpad; the independent audit module reproduces the driver's reference field to
max absolute difference exactly $0$ at $N\in\{32,64,256\}$).* The residual **stagnates**: at
$N=512$ case 8 it is $2.486\times10^{-12}$ after the direct solve and
$1.064,\,1.063,\,1.067,\,1.069 \times10^{-12}$ after one, two, three and four steps of
iterative refinement. No number of refinements reaches $10^{-12}$, because the *evaluation* of
$\|Au-f\|$ in f64 has a round-off floor
$\varepsilon\,\|A\|_\infty\|u\|_2/\|f\|_2$, and the measured residual sits at a
mesh-independent $0.156\times$ that floor at **both** $N=256$ ($0.1562$) and $N=512$
($0.1556$). The floor is $1.99\times10^{-12}$ at $N=256$ and $7.95\times10^{-12}$ at $N=512$:
it grows like $\|A\|_\infty \sim N^2$. A fixed absolute bound on this quantity is therefore a
mesh-dependent gate in disguise — $5\times$ above the floor at $N=256$, $0.13\times$ below it
at $N=512$, i.e. unreachable there by any f64 algorithm. The cluster and the GB10 differ on the
failing case by $2\times10^{-15}$ absolute ($1.0661$ vs $1.0640\times10^{-12}$), the
already-paid-for 1-ulp cross-machine effect.

*The change, which is the smallest one that changes no science.* `FOM.reference` returns the
measured floor as a third value; nothing about the reference field, the factorisation or the
two refinement steps changes, and the audit module computes its own floor the same way. G-FOM-4
now accepts
$$\mathrm{resid} \;\le\; \max\bigl(\texttt{reference\_residual\_limit},\;
\texttt{reference\_residual\_roundoff\_factor}\cdot\varepsilon\,\|A\|_\infty\|u\|_2/\|f\|_2\bigr),$$
with the factor set to $1.0$, and both the floor and the applied limit are recorded per case in
`references`. The gate is **unchanged at $10^{-12}$ wherever the floor is below it** — every
mesh already run: measured worst $4.70\times10^{-15}$ ($N=32$), $1.87\times10^{-14}$ ($N=64$),
$7.42\times10^{-14}$ ($N=128$, `lsh03`) and $2.97\times10^{-13}$ ($N=256$, `lsh04`, where the
limit rises only to at most $1.99\times10^{-12}$ and the measured value is $6.7\times$ inside
it) — and at $N=512$ it becomes $7.95\times10^{-12}$ against a measured $1.07\times10^{-12}$, a
$7.4\times$ margin. Every reported error in this cell is $\ge 10^{-4}$ relative, eight orders
above any of these numbers, so no error, ranking or criterion moves. `lsh03`, `lsh04` and
`lsh06` are **not** re-run and are unaffected; their staged copies of `config-solve.json` and
`lsh_core.py` preserve exactly what they executed. The independent audit's own
`reference_residual_below_limit` check takes the same rule and now also reports
`worst_reference_residual_over_limit`; re-audited after the patch, all three completed jobs
still pass all thirteen checks.

*Resubmission.* One resubmit only, as attempt `lsh07`, identical in every scientific respect to
`lsh05` — same config file, same $M=257$, same cohorts, seeds, arms, ladder, POD ranks,
comparators, repetitions and timing contract, same A100 request and `--mem 180G`. If it fails
again the cell reports $N\in\{64,128,256\}$ and stops. Cluster jobs after `lsh07`: seven used,
one in reserve.

**2026-09-17, §A8 — the Codex audit of the final report is again replaced by a written
self-audit.** The coordinator re-verified on 2026-09-17 that the Codex usage limit holds until
2026-09-19 11:33, after this lane's reporting window. As in §A1 and §A5 the substitute is
`reports/self-audit-2026-09-17-solves.md`, which lists every claim in the regenerated report,
the `result.json` field it rests on, and the check run against it, and which additionally
records the independent NumPy audit verdicts for `lsh03`, `lsh04` and `lsh06`.

**2026-09-17, §A9 — the free rung is head-independent to round-off, not byte-identically; §A6's
claim is corrected. No reported number moves.** §A6(b) recorded, from the $N=32$, $R=32$ local
smoke, "byte-identical output from the $K=4$ and $K=8$ heads (as it must be: the rung is
head-independent)". On the real $N=256$, $R=512$ banks in `lsh06` the two heads' output fields
are **not** bitwise identical. The mathematics is unchanged and the conclusion is unchanged: with
$C=I$ the recovered coefficients are
$h_\theta(z) + R_q^{-1}Q_q^\top(f_m - Bh_\theta(z))$, and since $Q_qR_q = BC = B$ the term
$R_q^{-1}Q_q^\top Bh_\theta(z)$ equals $h_\theta(z)$ *in exact arithmetic*, so the head cancels.
In f64 that cancellation is inexact and what survives depends on $h_\theta(z)$, which differs
between the $K=16$ and $K=32$ heads. Measured on all 32 development cases
(`checks/verify_report_2026-09-17.py`, check `free_rung_is_head_independent_to_roundoff`): worst
relative field difference between the two heads $4.06\times10^{-14}$, worst difference in the
reported same-grid error $4.86\times10^{-17}$, i.e. the two subjects agree on every reported digit
(worst $0.7790957\,\%$ for both). The three timed repetitions of one subject **are** bitwise
identical, as the per-invocation `field_sha256` values show, which is why the smoke's stronger
claim was not caught earlier: at $R=32$ the cancellation happened to be exact. The report now
states the measured agreement instead of asserting byte-identity.

**2026-09-17, §A10 — the report generator keys timed subjects by job and by test-mode block, and
the $M=257$ and $M=1024$ results are presented as separate tables.** §A6's carry-forward (ii) is
implemented: `reports/generate_lshape.py` builds `aggs` under the key
`(job_id, mesh, subject)`, groups solve jobs into blocks by `config.requested_modes`, and
**raises** if two jobs in one block report the same `(mesh, subject)` — the silent overwrite that
would have let `lsh06` replace `lsh04`'s $N=256$ numbers. The guard was tested against the real
JSONs by passing `lsh04`'s `result.json` twice; it aborts with the colliding subject named.
Every headline table, every per-mesh solve section, every clause lookup and every figure panel is
now resolved through the block index, each `summary.json` row carries `test_modes` beside
`job_id` and `source_sha`, and the headline carries an explicit statement that costs may not be
compared across blocks (the dense $M\times n$ projection is charged inside every reduced query).
Carry-forward (i) is implemented as well: the validation-versus-development gap is generated from
`lsh02`'s `head_arms` and printed immediately below the headline verdict, not in a caveat
section. Carry-forward (iii), the $K=64$ head, is **not** run: the solve jobs were the priority
and `lsh07` consumed the seventh of eight jobs.
