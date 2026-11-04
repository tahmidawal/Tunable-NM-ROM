# Exactly divergence-free, exactly no-slip neural bank for the steady DFG 2D-1 cylinder — pre-registration

Lane `exp/2026-11-03-cyl-divfree` (worktree `worktrees/2026-11-03-cyl-divfree`), forked from
obstacle-rom `57680db12` (code-only mirror `e2d4b08bf`). Cluster namespace
`/cluster/tufts/paralab/tawal01/cyl-divfree-20261103/` (obstacle namespaces are read only).
This file is committed **before any cluster job**; changes after the first job are listed in
section 9 with their reason.

## 1. Question

Run 09 (scanned-coordinate neural bank, model `645d0300…`, job 6661438) passes all seven
development targets with velocity error $3.62\times10^{-4}$ / $3.49\times10^{-4}$ (Re 16.875 / 23.125)
but divergence $3.8\times10^{-3}$ / $4.5\times10^{-3}$, i.e. ~250× the POD-4 value. Its modes are
trained to POD-mode and gradient targets with a *penalty* on the divergence trace, so
incompressibility is approximate, and the residual error concentrates in a thin ring at the
cylinder wall. Stage 2 asks: **does building incompressibility and no-slip into the
architecture fix divergence without losing velocity accuracy?**

Fork-point check: `57680db12` contains the complete ns-obstacle-rom source used by run 09
(`pilot2d.py`, `accuracy_v2/*`, `spaces.py`, …). `pilot2d.source_digest()` at this commit is
`867c4e2c…`, equal to the `source_sha256` of the preparation release consumed by run 09, so
the unchanged loaders accept the existing prepared inputs. No fallback base was needed.

## 2. Stage 1 — projection floor (measurement, no training)

The existing tool is `pilot2d.projection_diagnostic`: weighted-$L^2$ least squares of
$u_{\mathrm{ref}}-u_{\mathrm{lift}}$ onto the model's velocity trial space, with the same
quadrature weights `model.rule.weights` as the full-rule velocity metric (`metrics.field_errors`),
then scored by the same `score_state`. (Its squared residual equals the squared velocity
numerator of the scored metric, so it *is* the best velocity fit in the metric's norm.)
Run 09 already invoked it inside job 6661438 (`representation_diagnostic` in `status_007.json`).
Stage 1 reports those saved numbers (via `extract_baselines.py` → `BASELINES.json`) and
re-measures them in this lane's first job (`run09_replay`), adding the neural-only 4-column
floor (without the 4 FE supremizer columns). Because the projection is orthogonal in the metric
norm, the solve-induced part is $\sqrt{e_{\mathrm{solve}}^2-e_{\mathrm{floor}}^2}$.

## 3. Architecture (one designed architecture)

Dimensionless coordinates $X=x/D$, $D=0.1$ m: channel $[0,22]\times[0,4.1]$, cylinder centre
$C=(2,2)$, radius $R=0.5$. For mode $k=1..4$:

$$\psi_k(X)=W(X)\,N_k(X)+c_k\,\chi(X),\qquad u_k=\partial_Y\psi_k,\quad v_k=-\partial_X\psi_k .$$

* **Wall factor** $W=\prod_{j} f_j^2$, $f_j=1-e^{-d_j/\ell}$, $\ell=1$, with distances
  $d_j\in\{X,\;Y,\;4.1-Y,\;|X-C|-R\}$ (inlet, both walls, cylinder). $W$ and $\nabla W$ vanish on
  those boundaries, so $\nabla(WN_k)=0$ there: the mode velocity is exactly zero on the inlet,
  walls and cylinder (the inlet profile stays in the unchanged FE lift). $W\neq0$ on the
  outlet (do-nothing boundary). $f_j$ is the run-09 envelope factor, squared.
* **Cylinder flux freedom** $\chi(\rho)=(1-(\rho/s)^2)^3$ for $\rho=|X-C|-R<s$, else $0$,
  $s=0.8$. $\chi=1$, $\chi'=0$ on the cylinder; $\chi\equiv0$ at distance $\ge0.8D$, which
  is short of every other boundary (nearest is 1.5 D). Without it every mode would have
  $\psi=0$ on the cylinder and could not change the flux split above/below the
  (off-centre) cylinder. $c_k$ are 4 trainable scalars.
* **Velocity gradient** is the Hessian of $\psi_k$ (forward-over-forward autodiff),
  symmetrised explicitly, so $\nabla\cdot u_k=H_{YX}-H_{XY}=0$ bitwise.
* **Network** $N:\mathbb R^{29}\to\mathbb R^4$, tanh MLP 29–160–160–160–4 (≈57.5k parameters,
  comparable to run 09's frozen parent plus four heads). Inputs: the 21 run-09 features
  (scaled coordinates, Fourier $\sin/\cos(\pi f z)$, $f\in\{1,2,4,8\}$, three cylinder Gaussians)
  plus **wall-aware cylinder features**: $\tanh(\rho/2)$, $\cos\theta$, $\sin\theta$,
  $\cos2\theta$, $\sin2\theta$, $e^{-\rho/\delta}$ for $\delta\in\{0.1,0.3,1\}$.
* Online: unchanged — same 4 neural velocity modes **plus** the same 4 FE supremizers, same
  FE pressure space, lift, fixed test bank and exact-tensor weak least-squares solve
  (`pilot2d.build_model('bank',4,…)`, `Model2D.query`, zero initial state).

Local smoke (`smoke_bank.py`, random weights): boundary velocity $\le7\times10^{-15}$ (inlet,
walls, cylinder), outlet velocity non-zero, $\max|\nabla\cdot u|=0$, Hessian vs. central
finite differences $6\times10^{-9}$ relative, exact save/load.

## 4. Training (same recipe as the run-09 lineage)

Objective, with **the run-09 frozen normalisations** (asserted equal to the run-09 parent
metadata to $10^{-12}$):

$$J=\sum_{r=1}^4 m_r\Big(\tfrac{e^v_r}{n^v_r}+\tfrac{e^g_r}{n^g_r}+\tfrac{e^{\mathrm{tr}}_r}{n^g_r}\Big)+1.0\cdot T,$$

$e^v,e^g$ weighted $L^2$ errors to the TRAIN POD mode values and dimensionless gradients on the
full volume rule (1 036 497 points), $e^{\mathrm{tr}}$ the supervised divergence-trace term
(weight 1.0 as in run 09; constant here, since the bank is solenoidal — it is reported so that
$J$ is directly comparable with run 09's selected value $3.10\times10^{-4}$), $T$ the
$\sigma^2$-weighted cylinder traction error over the global TRAIN traction energy (weight 1.0,
805 cylinder points). Teachers through the unchanged `accuracy_v2.teacher.load` and
`boundary_teacher.load`. TRAIN only; development and validation are not read.

Optimizer: Adam 6000 steps, batch 8192, lr $10^{-3}$ with the `train.py` cosine schedule, full
measurement every 500; then full-batch L-BFGS (history 10, Armijo with 20 halvings — the
`train.py` algorithm) from the best measured Adam iterate, until $\|\nabla J\|_\infty\le10^{-10}$,
line-search failure, or the training deadline (7200 s of allocation, 600 s export reserve).
Selection: lowest full TRAIN objective. Export is re-loaded and the objective replayed.
Seed 3112026. f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, GPU preflight.

## 5. Pre-declared success (development data only)

On **both** development cases (Re 16.875, 23.125), from `gates.py` on `EVALUATION.json`:

* **G1** all seven original targets pass with unchanged thresholds (velocity ≤ 1 %, pressure ≤ 2 %,
  drag ≤ 2 %, transverse ≤ 0.002, divergence ≤ 0.005, mass ≤ 0.001, boundary trace ≤ $10^{-6}$),
  solver status `stationary`/`root_converged`, inf-sup gate passed;
* **G2** divergence ≤ 10 × POD-4 divergence, POD-4 taken as the smaller of the recorded
  same-solver weak-LS and mixed-Galerkin values (limits $1.507\times10^{-4}$ / $2.134\times10^{-4}$);
* **G3** velocity error ≤ run 09's ($3.617\times10^{-4}$ / $3.494\times10^{-4}$).

Success = G1 ∧ G2 ∧ G3 on both cases. Validity precondition: the same job's `run09_replay`
reproduces run 09's recorded velocity error to $10^{-8}$ relative (harness equivalence).
Velocity is also reported against POD-4 (both solvers) and the bank's own projection floor.
Note: the coordinator brief quotes "POD-4 mixed Galerkin 0.0103 % / 0.0076 %"; in the saved JSON
those are the POD-4 **weak-LS** values; mixed Galerkin is $4.6\times10^{-6}$ / $5.4\times10^{-6}$.
Both are reported.

## 6. Jobs and budget

Hard cap 4 GPU-hours (allocated elapsed × GPUs), at most 2 allocations at once, no reference
generation (reuse `bank-fit01/inputs/prepared`, `development-materialize02` population,
`fine-preparation01` probes, all read-only).

| job | content | time limit |
|---|---|---|
| E1 | Stage 1 + harness: run09 replay, POD-4 weak-LS, untrained divfree bank (pipeline check only) | 40 min |
| T1 | train divfree bank | 2 h 10 min |
| E2 | evaluation of T1 bank (+ run09 replay + POD-4) and gates | 40 min |
| (repair) | at most one, see §7 | ≤ remaining budget |

## 7. Stop rule and the single permitted repair

One designed architecture plus at most one documented repair, chosen from development data
and TRAIN diagnostics and recorded here (section 9) **before** it is submitted. Pre-identified
candidates: (a) training stopped by the deadline, not by stationarity, and G3 fails → continue
L-BFGS warm from the selected parameters with the remaining budget; (b) the bank's own
projection floor already exceeds run 09's solve error → capacity/feature change.
If the repair fails, record and stop. All failures retained.

## 8. Not claimed

No validation access, no speed-up claim, no physical admission; development-only diagnostic.

## 9. Amendments after the first job

(none yet)
