# hires-burgers — DESIGN (pre-registered 2026-09-20, before any GPU job)

Lane of the 2026-09-20 speed-and-accuracy campaign; binding contract
`reports/2026-09-20-speed-accuracy-campaign-protocol.md`. Worktree
`worktrees/2026-09-20-hires-burgers`, branch `exp/2026-09-20-hires-burgers`, forked from
`exp/2026-09-17-b-panel` @ `25434a27`. Cluster namespace
`/cluster/tufts/paralab/tawal01/hires_b_20260920/`, one directory per job. Budget: at most 2
running and 8 total GPU jobs.

## 1. Question

One frozen Burgers 2D checkpoint (`sep_hfit_dense_mid_N256_dense.pkl`, $K=16$, $R=512$,
trained at $256^2$), transferred without retraining to $2048^2$ and $4096^2$ intervals:

1. Does the corrected accuracy survive transfer above $1024^2$? (First job, first printed lines.)
2. Is there an **accurate rung** — corrections on, worst same-grid evolved relative $L^2$ error
   $\le 1\,\%$ (stretch $\le 0.5\,\%$) — that is $\ge 5\times$ faster than the named
   Newton–BiCGStab full-order model in the same allocation, stalled exits counted and shown?
3. Where does the accurate rung's time go, and how far can it be driven down at unchanged accuracy?

## 2. Lane success bar (copied from the protocol, not adjustable)

At the largest mesh reached, the accurate setting has worst same-grid error $\le 1\,\%$ and
$S = T_{\mathrm{FOM}}/T_{\mathrm{ROM}} \ge 5$ against the named FOM in the same allocation. The
fast setting ($q=0$) is reported beside it. Missing the bar is a result.

**Error metric (inherited from b-panel's audit, unchanged):** for case $c$ and output time $t_k$,
$\epsilon_k = \lVert u_k - u^{\mathrm{tight}}_k\rVert_2 / \lVert u_0 \rVert_2$ on the full $L$-grid,
with $u^{\mathrm{tight}}$ the same-job `fft_tight` solve (Newton tol $10^{-6}$, BiCGStab tol
$10^{-8}$, $\Delta t = 0.005$). *Evolved* = $\max_{k\ge1}$, *all-times* = $\max_{k\ge0}$ (the
$k=0$ term is the decoder's compression of the supplied initial field, ≈3.9 % at $1024^2$; it is
reported beside every evolved number and never hidden). Worst = max over the six development cases.
The bar is scored on **evolved** error, as the lane brief states; the all-times verdict is printed
next to it.

**Comparators, all in the same allocation, all named:** `fft_tight` (tight);
`nt1e-4_dt005` (relaxed, passing: 0.034 % at $1024^2$); `nt1e-3_dt005`; `nt1e-3_dt01`,
`nt1e-4_dt01`, `nt1e-2_dt01`, `nt1e-2_dt005` (the cheaper controls; several are *less* accurate
than the ROM and are shown for that reason). The headline speedup is quoted against **both** the
tight and the relaxed-passing solve, and separately against the *fastest tested FOM whose error
is $\le$ the ROM's*. Coarse-grid FOM ($L/2$, $L/4$, same I/O scope through `engines.make_fom(...,
target=L)`) is scored against a refined reference (§5). No comparator is softened or dropped
after the fact; a FOM with stalled Newton steps is shown with its stall count.

## 3. What is brought in, and from where

| piece | source | status at the fork point |
|---|---|---|
| b-speed kernels (`fuse` 1.25×, `lean` fold 1.15×; arm `C1`) | `exp/2026-09-16-b-speed` @ `47c178d2`, `experiments/b-speed/{fast,ladders}.py` | **already present byte-identical** as `experiments/b-panel/speed/{fast,ladders}.py` (checked with `diff -q` 2026-09-20); $q=0$ only |
| b-eqtop certified rules | `exp/2026-09-17-b-eqtop` @ `8542c604`, `certified-rules/*.npz` | **already present byte-identical** in `experiments/b-panel/inputs/rules-eqtop/` (six SHA256 equal) |
| directions $C$ (`directions_qtd02.npz`) | b-panel inputs | present |

b-speed's kernel covers $q=0$ only. The accurate rung needs $q>0$, so this lane adds `hfast.py`
(§6) and parity-gates it in-job against the audited `topfix.make_query(..., 'base')`.

## 4. Quadrature rules and their certificate

A rule approximates $\Phi^\top a(u)$ by $\sum_j w_j \Phi(x_j) a(u)(x_j)$. **A rule is certified
only by held-out $\rho$ on reachable states, never by the NNLS fit residual** (standing rule):
$\rho(u) = \lVert P_q^\top a - \Phi^\top a\rVert / \lVert \Phi^\top a\rVert$, primary bar
$\rho_{\max} \le 0.116$, tight $0.06$ (both inherited from q-ridge/b-eqtop, unchanged).

Candidate rules per rung $(q, M)$, all re-certified **at each mesh**:

- `xfer`: b-eqtop's support mapped to the same physical points, weights refit by NNLS on 64 fit
  states with per-state scaling (objective $\sum_s\rho_s^2$, b-eqtop's `rhow`);
- `lat64` / `lat32`: uniform interior sub-lattice with equal weights $(L/s)^2$ — no fit, no draw,
  exact for products of combined wavenumber below $2s$, maps to every mesh with $s \mid L$. Local
  exploration at $256^2$ on one development case (`explore/lattice_rho.py`, not a result): median
  $\rho$ 2.5e-4 (NNLS rule 1.0e-3) but $\rho_{\max}$ 0.126 vs 0.082, both attained at the *initial*
  state $w_0$; on evolved states the lattice is ≈3–5× tighter. So `lat64` may fail the inherited
  bar on $w_0$ alone — that is reported as a failure, with $\rho$ excluding $w_0$ shown beside it;
- control `bad`: q-ridge's $q=256$, $m=2048$ rule, which b-panel showed regresses at $1024^2$
  (1.87 % vs dense 0.59 %). **It must fail the certificate and show the regression here**, or the
  certificate is not discriminating (controls-must-fail rule).

**Populations (deviation from b-panel A3, declared):** fit and held-out states come from the
audited dense query at a population mesh $L_p = 512$ (8 fit trajectories → 64 states, 8 disjoint
held-out trajectories → all 408 states), decoded at the target mesh — bank coefficients are
mesh-free, and the dense query at $2048^2$ for 16 trajectories × 3 rungs is not affordable. To
close the gap this leaves, every deployed rule is **also** certified on the states its own query
visits at the target mesh on the held-out trajectories ("deployed population"). A rule is
`certified_primary` only if both populations meet the bar. If no rule certifies at a rung, the
rung falls back to the dense truth arm and the report says so.

## 5. Arms per job

ROM: $q=0$ ($M=64$) fast; accurate rungs $(q,M) \in \{(128,576), (256,544), (256,1088)\}$ —
$(256,544)$ because b-qxm measured 0.757 % at $256^2$ with half the tests (cheaper Jacobian,
easier rule). Each at evolution tolerance $10^{-3}$ and $10^{-6}$, budget 600, exits recorded.
Dense truth arm per rung (exact advection through the Φ-free separable projection, same solver),
once per case, to separate quadrature error from representation error. FOM arms of §2. Coarse
FOM at $L/2$, $L/4$. Refined reference: $4L$ at $\Delta t/16$ when it fits (8192² for $L=2048$);
$2L$ at $4096^2$ if $4L$ does not fit — labelled. Reference phase runs last under a deadline.

Timing: §"Non-negotiable mechanics" of the protocol — one allocation, randomised order, 0.25 s
burn-in before every invocation, `block_until_ready` both sides, 5 retained repetitions, medians,
identical I/O scope (dense GPU input → six dense GPU fields; host-inclusive time beside it), GPU
model and UUID recorded.

## 6. Speed loop (every change parity-gated and timed before/after in the same job)

`hfast.py`: `fold` ($AC$, $G_5C$ offline), `ajac` (analytic structured Jacobian instead of
$K+q$ forward tangents through the $R$-wide bank), `fuse` (one $(r,J)$ evaluation per LM
iteration), `hoist` (QR of $R_cC$ offline), `nodiag` (drop the redundant per-step two-Jacobian
stationarity diagnostic; the LM's own exit gradient is the same quantity), `decfuse`, optional
`chol`. **Parity gate:** fields vs the audited base arm at the same rule and tolerance, relative
$\le 10^{-9}$, integer iteration and exit vectors compared and reported; an arm that fails is
reported as its own labelled arm with its directly measured error. Profile: initialise / evolve /
decode separately, per-iteration cost, `evalJ`, Gram, solve micro-timings, iterations/step,
damping retries. Later hypotheses (each its own SPEED-LOG row): fewer iterations from a better
predictor; smaller $M$ at fixed $q$; Cholesky; f32 decode (labelled precision arm, gate 1e-12 on
integers + stated field tolerance); batched steps.

## 7. Controls and stop rules

- GPU preflight exit 42; `jax_backend=gpu` in log; f64; `JAX_DEFAULT_MATMUL_PRECISION=highest`.
- Φ-free operator parity vs `engines.modes` and $\Phi^\top G$ at $L_p$: $\le 10^{-12}$, asserted.
- Evaluation cohort SHA256 equals b-panel's (`108f12dc…`). Final cohort stays sealed.
- Repetition outputs identical. Every FOM row carries converged/stalled step counts.
- Control rule `bad` must fail. A threshold that cannot fail is reported as untested.
- Stop: bar met at the largest mesh with a certified rule and audited; or 8 jobs spent; or no
  remaining idea. No tuning on the six cases beyond choosing among pre-declared arms; the rung
  choice for the headline is made by the rule "cheapest certified arm with evolved error ≤ 1 %".

## 8. After each job

Checksum-verified pull, independent NumPy recomputation of errors from saved fields (restricted
fields for every arm and case, full fields for the audit case), remote directory deleted,
`summary.json` with source hashes committed (no field arrays, nothing > 50 MB), SPEED-LOG,
HANDOFF and the lab log (under `flock`) updated.

## Glossary

**FOM** full-order model: backward-Euler upwind finite differences, Newton with FFT-preconditioned
BiCGStab. **q** number of linear correction directions added to the neural head's output.
**M** number of sine test functions in the weak residual; **m** number of quadrature nodes.
**EQ** empirical quadrature: the advection term sampled at $m$ nodes instead of all $n$.
**ρ** relative error of the sampled weak advection term on a state. **Reachable states** states
the ROM query actually visits. **Evolved error** worst error over output times after $t=0$.
**Stalled exit** an LM step that ended on budget, tiny step or rejection rather than on the
stationarity or residual test; for the FOM, a time step whose Newton residual missed its tolerance.
