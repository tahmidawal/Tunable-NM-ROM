# Can the Burgers (nonlinear) residual be made sample-free? — synthesis of a seven-agent ideation

**Status: ideation + CPU spot-checks only. No GPU job, no training, no cluster run. Nothing
here is a result; the numbers marked [cpu-check] are provisional, single-checkpoint
(N=256, K=8, R=32, seed 0) diagnostics run by the agents on the local box, CPU only, and
must be reproduced in a gated job before being quoted.** The seven raw reports, the brief
they were given, and the four CPU scripts live in
`understand/ideation-2026-08-29-sample-free-nonlinear/`.

Agents: four Claude Fable 5 agents with assigned angles (A: tensor + sign handling,
B: structured banks, C: learned point-free surrogates, D: generality + devil's advocate)
and three OpenAI Codex runs (1: general survey, 2: implementation realism against the
actual code, 3: deliberate skeptic). All read the same brief and the same code
(`worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder/`).

## The answer in plain words

**Yes, for Burgers — and it can be exact, not approximate.** Poisson is sample-free because
its residual is linear in the field and the field is linear in the head output $h$. Burgers'
advection $u\,u_x$ is *quadratic* in the field, hence quadratic in $h$, and a quadratic in
$R=32$ numbers is a fixed table with $M \times R \times R$ entries:

$$
\Phi^\top\big(u \odot D u\big) = \sum_{j,k} h_j\,h_k\,T_{:,j,k},
\qquad
T_{ijk} = \sum_x \Phi_{xi}\,G_{xj}\,(DG)_{xk}.
$$

$T$ is built once on the full grid from the frozen bank (the same move as $B=\Phi^\top G$,
one index deeper), and online the advection term is one contraction $h^\top T h$ — no
sample points, no NNLS weights, no stage-2 node training. All seven agents rank this first.

The one obstacle is the FOM's **sign-upwind** stencil (look left where $u>0$, right where
$u<0$), which is not polynomial. Agents A and D found the clean decomposition

$$
N_{\text{upwind}}(u) = u\,D^{c}u \;-\; \tfrac{\Delta x}{2}\,|u|\,\Delta_h u,
$$

i.e. sign-upwind is *central difference plus an* $O(\Delta x)$ *artificial viscosity whose
only non-polynomial piece is* $|u|$. On non-negative fields $|u|=u$ and the whole term is
quadratic. Our 1D data (positive Gaussian blobs, zero walls, viscous Burgers) is
non-negative by the max principle, so on the truth the backward-difference tensor $T^-$ is
**exactly** the full-grid oracle. It can only deviate where the *decoded* ROM field
undershoots zero.

## What the agents measured [cpu-check], all on the committed N=256 checkpoint

Three agents (B, C, and the D script design) independently checked $T^-$ against the true
sign-upwind full-grid projection $\Phi^\top N(Gh)$:

| quantity | value | source |
|---|---|---|
| $h^\top T^- h$ vs oracle term, restricted to points with $u>0$ | $8\times10^{-15}$ (exact) | B, C |
| same, all decoded training states (relative) | median $1.4\times10^{-7}$, mean $2\times10^{-6}$, max $6\times10^{-5}$ – $1.6\times10^{-4}$ | B, C |
| fraction of decoded grid points with $u<0$ (min $-1.2\times10^{-2}$) | 6.9 % | B, C |
| at $z + 0.05\,\mathcal N(0,I)$ (beyond the LM trust radius 0.017) | median $1.8\times10^{-7}$, p95 $2.4\times10^{-3}$, max $1.3\times10^{-2}$ | C |
| Jacobian $\partial q/\partial z$ vs oracle: min gradient cosine | 0.9996 | C |
| for comparison: NNLS-32 held-out mismatch / cosine | 6.3 % / 0.91 | C (from run JSONs) |
| quadratic *regression* on the bank (OpInf-style), on-manifold | median $3\times10^{-6}$, max $3\times10^{-4}$; fat tails off-manifold | C |
| projection floor of the truth onto 32 learned bank shapes | $2.7\times10^{-4}$ | B |
| projection floor onto the first 32 sine modes / POD-32 | $3.4\times10^{-5}$ / $1.4\times10^{-5}$ | B |
| decoder floor (K=8 head over the bank) | $3.7\times10^{-3}$ | committed |

Two consequences the group drew:

1. The sign-switch error is ~300× smaller than the sampled rule that *already matches the
   oracle rollout*. On this data the tensor is oracle-quality with nothing fitted.
2. **The learned bank is not where the accuracy lives.** Plain sines span the truth better
   than the learned 32 shapes; the $3.7\times10^{-3}$ floor comes from the 8-dimensional
   head. So a fixed sine bank with the same nonlinear head might lose nothing — and then
   $B$ is the identity, $T$ is analytic, and Burgers is quadrature-free in exactly the
   Poisson sense. One training job decides this.

## Where the agents disagree, and how to settle each

| issue | one side | other side | settle by |
|---|---|---|---|
| Is $T^-$ "exact"? | A, B, D: exact on the positivity cone; $10^{-7}$ in practice; label "oracle to 5 digits" | Codex-2/3: not exact unless *every* LM candidate state (accepted and rejected) is positive; a single switched point falsifies the exact claim | Audit signs at every LM candidate in a rollout. Expect: "exact" fails formally, agreement to ~$10^{-5}$ in rollout error holds. Label accordingly. |
| Kernel count | A, D: 2 fusions vs 3 for sampling — fewer kernels | Codex-2/3: the optimized sampled path already fuses to 2 reductions; stock tensor code needs 3; in 1D expect parity or slight regression | Paired timing in one job. In 1D the tensor is about exactness and simplicity, **not speed**; the FOM comparison is unchanged. |
| Is it worth it at all? | Fable A–D: deletes NNLS, the $m$ sweep, stage 2, and checkpoint-dependent node fragility; strict $n$-independence | Codex-3: keep sampling as production default — it handles any pointwise flux/limiter/coefficient and 2D sampling already does not bind at $m=256$ | Coordinator's view: in 1D the tensor is a cleaner story, not a faster one; in 2D it is the first sample-free rule that could tie $m=256$ on cost *while matching the oracle* — that is worth one paired job. Sampling stays the general tool for non-polynomial physics. |
| Jacobian formula | "$2\,T h$" | Codex-2: $T$ is *not* symmetric (bank vs differenced bank); use $Q = T + T^{(j\leftrightarrow k)}$, $q = \tfrac12 h^\top Q h$, $\partial q/\partial z = (Qh)\,\partial h/\partial z$ | Codex-2 is right; use $Q$. |
| Build from $u\,D^-u$ or flux form $D^-(u^2/2)$? | flux form is symmetric and moves $D$ onto the analytic sines | the current FOM is non-conservative $u\,D^-u$; a flux tensor differs by $\tfrac{\Delta x}{2}(D^-u)^2$ and would fail gate F | Build from what the FOM actually is. Flux form only if the FOM changes. |

## Beyond positive data and beyond quadratic

For sign-changing data or other PDEs the routes, ranked by group consensus:

1. **Choose a polynomial FOM** (centered / skew-symmetric / global Lax–Friedrichs with
   constant $\alpha$). The LF dissipation is a Laplacian and folds into the existing
   $\Lambda B h$ term; the residual becomes the Poisson path plus one contraction, exact for
   any sign. Agent D computed the cell Péclet numbers of our family ($u\,\Delta x/\nu \le
   1.2$ at N=128, $\le 0.04$ at N=4096): **upwinding was never needed for stability here**,
   and it adds ~60 % artificial viscosity at N=128, $\nu=0.01$. Caveat from every agent:
   this *changes the truth* (upwind vs centered differ by $O(10\,\%)$ at N=128) and every
   prior comparison; data regeneration and a stage-1 retrain are required. **A user
   decision, not something to do unasked.**
2. **Split form on the existing FOM** (A): tabulate $u\,D^c u$ exactly, sample only the
   $O(\Delta x)$ term $|u|\,\Delta_h u$. Sampling error shrinks by a factor $\Delta x$ with
   the current node machinery. The bridge if sign-upwind must be kept.
3. **Lift** $v=|u|$ with its own bank/head (D, Codex-1): the lifted term carries a
   $\Delta x$ weight, so a 10 % error in $v$ is far below the decoder floor. General
   pattern for $e^u$, rational terms, etc. Learned approximation; needs a sign-changing
   family to test.
4. **Learned correction on top of the tensor** (C): $q = h^\top T h + c(h)$ with $c \equiv 0$
   on positive states, trained with the existing gradient-matching loss and certified by
   gradient cosine, never by fit residual. Only for non-polynomial physics; a
   from-scratch surrogate is rejected by every agent (judged against $10^{-7}$, adds
   3–5 kernels, untrusted Jacobian).
5. **Higher-order terms / 3D** (D): dense storage is free until $M R^d \cdot 8\,\text{B}
   \gtrsim 10$ MB (2D quadratic at R=64: 2 MB, fine; 2D cubic: 134 MB, compress). The
   compression of choice is **head-PCA Tucker**: $h(z)$ spans ~$K$–$2K$ directions, so
   project to them and store $M \times K'^d$. Check the SVD of the head outputs first (free).
6. **Product-closed (trig) banks** (B): dense $T$ is already one kernel, so sparsity buys
   nothing; the value of a sine bank is analyticity, $B=I$, and $n$-independence — see
   the finding above. Splines only for FEM FOMs or non-Dirichlet BCs.

## Proposed experiment plan (not launched — for review)

**E1 — CPU audit, minutes, local.** From the committed N=256 (then 512) checkpoint: build
$T^-$ in two accumulation orders (f64, blocked — never materialize $n\times R^2$; Codex-2
notes the 2D naive temporary is ~32 GiB), record every accepted *and rejected* LM candidate
of the oracle rollout, and compare tensor vs oracle in $q$, $J$, $J^\top r$, and sign
pattern at each. Also report the contraction condition $\sum_{jk}|T_{ijk}h_jh_k|/|q_i|$
(Codex-3's cancellation concern).

**E2 — one A100 job, ~10 min.** Seventh arm `tensor` beside `oracle` / `base_tight` /
`nodes_tight` in `sep_b1d_scale.py` (AD path first), then the analytic $r$, $J$ in
`make_device_fast` for timing. N = 256, 512, 4096 from the committed checkpoints, no
training. Pass criteria, in order of strictness: (i) identical stop-reason histograms;
(ii) rollout error equal to the oracle arm to $\le 10^{-5}$ absolute (the honest bar given
the undershoots; $10^{-9}$ only if E1 finds no switched point); (iii) paired e2e time
$\le$ `base_tight`.

**E3 — one training job (optional, decides the bank question).** Stage 1 with a fixed
sine bank ($P=32$; $P=48$ if needed) and the same head/data/seed. Pass: held recon
$\le 3.7\times10^{-3}$; then the analytic-$T$ rollout must match its own oracle.

**E4 — 2D port.** One combined tensor for $D^-_x + D^-_y$ (64³, 2 MB) in
`sep_burgers_exlin.py`, against the full-grid and $m=256$ NNLS arms in one paired job.
This is the only place the tensor could also be a cost result.

**FOM-choice study (user decision first).** Generate the 8 test trajectories with upwind,
centered, skew-symmetric and LF at N=256 and a N=4096 reference; report self-convergence
and cross-scheme gaps. Only then decide whether to change the truth.

## Glossary

- **Bank $G$ / head $h$** — the two halves of the decoder: $R$ spatial shapes tabulated on
  the grid, and the small network mapping the latent $z$ to $R$ mixing amounts; $u = G h(z)$.
- **Test modes $\Phi$, $M$** — the sine functions the residual is projected onto.
- **$B = \Phi^\top G$** — the precomputed table that makes every linear term exact.
- **$T$ / $Q$** — the precomputed 3-index table for the quadratic advection term; $Q$ its
  symmetrized form used for the Jacobian.
- **Oracle** — the advection term summed over the full grid; the best any sampled rule can do.
- **Sign-upwind / $D^-$ / $D^c$** — the FOM's stencil that switches with the sign of $u$;
  the fixed backward difference; the central difference.
- **Undershoot** — decoded field values slightly below zero where the truth is $\ge 0$.
- **LM candidate** — a latent state tried by the Levenberg–Marquardt solver, whether the
  step was accepted or rejected.
- **Péclet number** — $u\,\Delta x/\nu$; centered differencing is non-oscillatory below 2.
- **Lift** — introduce an auxiliary field ($v=|u|$) so a non-polynomial term becomes polynomial.
- **Head-PCA Tucker** — compress $T$ by projecting $h$ onto the few directions it actually spans.
- **[cpu-check]** — a provisional number from an agent's local CPU diagnostic; not a result.
