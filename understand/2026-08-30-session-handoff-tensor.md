# Session handoff — 2026-08-27 → 2026-08-30 (Poisson QF, 1D/2D Burgers tensor, deck)

Read this first when resuming. It is a pointer document: every number lives in a report whose
tables are generated from run JSONs; nothing here should be quoted without opening the report.
`LAB-LOG.md` ("Where things stand" + the 2026-08-27…30 chronology) is the canonical record.

## What is true now (one paragraph each)

**Poisson 2D is quadrature-free.** Residual $r(z)=W\odot[\Lambda B h(z)-\Phi^\top f]$ with
$B=\Phi^\top G$ built once; equals the full-grid residual to ~1e-13 at every solution, fastest
path, no NNLS fit. Adopted as the Poisson default.
Report: `reports/2026-08-27-b1d-node-screening-and-poisson-qf.md`.

**1D Burgers is sample-free via the precomputed quadratic tensor** ("exact projection of a
polynomial nonlinearity, no hyper-reduction needed" — the standard POD-Galerkin/OpInf quadratic
operator; novelty is only the MLP-head-over-learned-bank setting). $T_{ijk}=\sum_x
\Phi_{xi}G_{xj}(D^-G)_{xk}$, $Q=T+T^{(jk)}$ ($T$ is not symmetric), advection $=\tfrac12 h^\top Qh$.
Exact where the decoded field is $\ge 0$ (gate T0 ~1e-15); on this non-negative family the tensor
arm equals the full-grid oracle's rollout error to ≤1.2e-6 (N=128) … ≤6e-10 (N≥16 384) with
identical stop histograms; deviation is FIRST order in the decoded undershoot × Δx × curvature
(bounded). Not distinguishable from NNLS-32 at n=8 (the "1–9 % better" claim was retracted).
Latent solve flat in N (one-GPU ladder exponent −0.08, N=128–4096); at 16k→65k the O(N) oracle
climbs 21→33 ms, tensor stays ~14 ms (cross-job ratio, not an exponent). Cost within ~7 % of the
sampled rule; no 1D speedup (launch-bound). f64 required.
Report: `reports/2026-08-29-b1d-tensor-sample-free-burgers.md` (five adversarial reviews applied;
reviews archived in `understand/review-2026-08-29-tensor-arm/`).

**2D Burgers, small bank (K=16, R=64, M=64): same result.** One 64³ (2 MB) tensor for both axes;
truth asserted ≥0 at every N. Tensor = oracle to 2e-6…8e-5 with identical stop histograms and LM
counts at N=64/256/512/1024; tensor/NNLS-256 error 0.996–1.003 (sampling was not binding at
m=4M); latent solve 20–28 ms vs NNLS 20–29 ms while the oracle grows 33→74→97→670 ms; paired
vs matched FOM 0.42×/0.58×/0.98×/3.45× (sampled rule 3.31×) — the 2D crossover is unchanged.
Decoded undershoots are larger in 2D (15–17 % of points) → parity 1e-4…1e-6. **R=512 headline
bank: head-PCA compression FAILED** (499/512 directions needed; T0 4.3e-5) — that configuration
stays sampled.
Report: `reports/2026-08-30-b2d-tensor-ladder.md`.

**Figures exist for both** (fields, pointwise errors, |tensor−oracle| on its own scale, error
curves, 2D cross-sections): `reports/figs/b1d-tensor-*` (script
`reports/fig_2026-08-30-b1d-tensor-fields.py`, local, reproduces the job numbers) and
`reports/figs/b2d-tensor-*` (five of 24; all on the 2D branch under `runs/b2dtensor/figs/`,
script `make_figs.py`).

**Study deck:** `reports/2026-08-29-study-deck-linear-vs-nonlinear-residuals.{tex,pdf}`, 31
slides: architecture, per-case training table, post-stage-1 pipeline diagram, linear →
quadrature-free, nonlinear → sampled + stage 2, the tensor rung (math, stencil/positivity,
diagram, 1D/2D results, constant-time, reviews, scope), four figure slides, summary, glossary.
Build: `latexmk -pdf`; every slide was rendered and checked; keep it that way (render pages
after edits — three slides overflowed at first each time).

**Ideation synthesis** (seven agents; what to do for sign-changing data, non-polynomial
physics, higher degree, compression): `understand/2026-08-29-sample-free-nonlinear-residual-synthesis.md`
(with the R=512 correction note) and raw reports in `understand/ideation-2026-08-29-sample-free-nonlinear/`.

## Branches / worktrees (all pushed to origin; none merged)

| worktree | branch | what | state |
|---|---|---|---|
| `worktrees/2026-08-27-b1d-poissonqf` | `exp/2026-08-27-b1d-poissonqf` | 1D testbed, node screening, Poisson QF, scaling ladder, rollout optimization | finished (`124aa05`) |
| `worktrees/2026-08-29-b1d-tensor` | `exp/2026-08-29-b1d-tensor` (from b1d-poissonqf) | 1D tensor arm, audit, constant-time ladder, N=16k/65k decoders | finished (`4f96247`) |
| `worktrees/2026-08-29-b2d-tensor` | `exp/2026-08-29-b2d-tensor` (from b1d-tensor) | 2D tensor ladder, R=512 stretch, figures | finished (`b9a3931`) |

Cluster: all namespaces (`b1dqf/`, `b1dtensor/`, `b2dtensor/`) deleted; queue empty at handoff.
Local: no jaxrun jobs running.

**Merge decisions are pending with the user** (rule: ask, never merge unasked). Nine finished
experiment branches now sit unmerged (the six listed in earlier lab-log entries plus the three
above). The b2d-tensor branch is a superset of b1d-tensor which is a superset of b1d-poissonqf,
so "merge b2d-tensor" would carry all three.

## Key code (all in `experiments/separable-decoder/` of the respective worktree)

- 1D: `b1d_common.py` (grid, upwind stencil, tridiagonal FOM, trainer), `b1d_fast_common.py`
  (optimized rollout; `make_device_fast(su, X_v, w_v, opt, Q=None)` — `Q` switches the tensor
  branch), `b1d_tensor_common.py` (`build_T` blocked, `symmetrize`), `sep_b1d_tensor.py` (arms +
  gates), `sep_b1d_ladder.py` (one-GPU ladder), `b1d_tensor_audit.py` (E1 sign audit).
- 2D: `b2d_tensor_common.py`, `sep_b2d_tensor.py` (third residual path beside `full` and `ex`
  in the incumbent solver; gates STEP/ROLL bit-identical), `sep_burgers_exlin.py` (incumbent),
  `runs/b2dtensor/make_figs.py`, `gen_tables.py`.
- Cluster: `cluster/{stage,push,pull}_{tensor,b2dtensor}.sh`, `run_*.sbatch` (pin
  `--constraint=a100-80G` for timing; H200 + `--mem 160G+` for N=1024 2D training).
- Report generators on main: `reports/gen_2026-08-27-b1d-nodes-and-poisson-qf.py`,
  `gen_2026-08-29-b1d-tensor.py`, `gen_2026-08-30-b2d-tensor.py` (they call the branch generators).

## Open experiments, in the order I would run them

1. **Multi-seed + more trajectories** (the reviewers' first demand): ≥3 decoder seeds ×
   ≥32 test trajectories, 1D N=512 and 2D N=256, arms oracle/NNLS/tensor, paired CIs. Cheap
   (1D minutes, 2D ~1 h/seed for training). Decides whether any tensor-vs-NNLS statement can
   ever be made; also gives the undershoot statistics across seeds.
2. **Sign-changing family** (expected FAILURE of the plain tensor — needed to make the
   "non-negative data" scope credible): e.g. 1D $a\sim U(-1.5,1.5)$ or $u_0=\sin 2\pi x$; then
   the split form (tabulate the central part, sample only the $|u|\Delta_h u$ term) or the
   polynomial-FOM route. The FOM-choice study (centered / skew / Lax–Friedrichs vs upwind;
   Péclet ≤ 1.2 on this family) changes the truth — **user decision first**.
3. **Fixed sine bank stage-1** (agent B's finding: 32 sines span the truth 8× better than the
   learned bank; the 3.7e-3 floor is the K=8 head): train with $G=$ first-$P$ sines, $P=32$
   then 48; pass = held recon ≤ 3.7e-3. If it passes, $B=I$ and $T$ is analytic — Burgers
   becomes quadrature-free in the Poisson sense. One job.
4. **R=512 compression the other way:** CP/Tucker of $T$ itself (not of the head), or a dense
   270 MB table on an H200 just to measure, or retrain the headline cell with R≈128 (the
   n256_j3 K=32/R=128 checkpoint exists — tensor 128³ = 16 MB, feasible dense).
5. **Assert min(truth) ≥ 0 numerically in the 1D driver** (the 2D driver does; 1D relied on the
   max-principle argument) and add per-step ‖u_T − u_or‖ to the comparison JSON (R3's ask).
6. Port the rollout optimizations to the 2D exlin solver (older open item; kernel-count wins).

## Landmines learned this session (not in CLAUDE.md yet)

- `codex exec` in a background shell hangs on stdin — launch with `< /dev/null`; the
  `-s read-only` flag blocks its own output write on this box (bwrap) — omit it and put the
  guardrails in the prompt.
- `rsync --delete` on re-stage wipes `logs/` of failed attempts — the 2D push script now
  excludes `logs/`; the 1D one does not.
- Gate C at N=16 384 trips a fixed 1e-12 tripwire from ε/Δx roundoff — `GATE_C_TOL` knob.
- The inherited 2D trainer closes its jit over the 17 GB snapshot array (captured-constant
  warning; ran on H200) — fix before any N=1024 2D retrain on an A100.
- Two different A100 models (40 GB / 80 GB) in the pool: pin `--constraint=a100-80G` for
  timing, or compare only within a job.
- Beamer: after every deck edit render the touched pages (`pdftoppm -f N -l N`) — overflow
  warnings in the log are the only signal, and titles that wrap are the usual cause.
- Never say "arms interleaved" unless the loop order really is reps → trajectory → arm (AB/BA);
  the 2D driver does it, the 1D four-job driver did not.
