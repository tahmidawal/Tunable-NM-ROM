# DESIGN — jcp-smooth-bank (C3): a bank trained for its derivatives

Pre-registered 2026-10-08, before any training or evaluation job. Lane `exp/2026-10-08-jcp-smooth-bank` (worktree
`worktrees/2026-10-08-jcp-smooth-bank`, forked from `exp/2026-10-01-ns3d-coordnet-bank` @ `21c175a1b`), mirrored to
`origin/codeonly/exp/2026-10-08-jcp-smooth-bank` by `sync_github.sh` after every commit. Cluster namespace
`/cluster/tufts/paralab/tawal01/jcpsmooth/`, one sub-directory per job, never reused, deleted after a checksum-verified
pull. Lane concurrent-job cap: **one Slurm job at a time** (§9 explains how one job uses several GPUs).
Convention: nothing above an *amendment* is edited after a GPU number exists; amendments are appended with a date.

## 1. Question

The off-mesh ROM decodes the bank $u(x)=G(x)c$ **and its analytic gradient** at the points of a classical rule to
evaluate the tested advection
$$N_a(c)=L\sum_q w_q\,\psi_a(x_q)\,u(x_q)\,\big(u_x+u_y\big)(x_q),\qquad \psi_a=2\sin(a_1\pi x)\sin(a_2\pi y).$$
The bank was trained on **values only**. Two hypotheses:

- **H1 (Sobolev).** Adding a gradient-matching term to the training loss lowers the bank's gradient error on
  validation states and lowers the ROM error floor (the end-to-end error of the converged-quadrature rollout).
- **H2 (smoothness).** A smoother bank (smaller random-Fourier-feature scale $\sigma$) needs fewer quadrature points
  to reach the $\rho$ acceptance bar, at some cost in representation accuracy. The trade-off between the error
  floor and the points needed is mapped, one point per bank.

Plus one control asked by the paper plan (F2 / 06-codex-audit): **3d coarse-data control** — a bank trained only on
coarse-mesh data — to address the objection that the off-mesh gain exists only because "the bank saw finer data".

### What theory predicts, and what it does not (03-theory §1 with the 06-codex-audit corrections)

Tensor Gauss–Legendre with $p$ points per axis converges geometrically, at a rate set by the size of the Bernstein
ellipse on which the integrand $\psi_a f$, $f=u(\mathbf 1\cdot\nabla u)$, is analytic (Theorem 1.2, CORRECT
qualitatively; its displayed constant and the explicit strip-width lower bound of Lemma 1.1 are WRONG and are **not
used here**). Two regimes (Corollary 1.3, qualitative only):

1. *test-frequency limited*: nothing converges before $p$ is of the order of the largest per-axis test frequency
   $a_{\max}$ ($a_{\max}\approx\sqrt{4M/\pi}$: about 25 for $M=512$ (`fast`) and about 44 for $M=1536$ (`acc`)).
   This onset is the same for every bank, so **smoothness cannot move it**;
2. *decoder limited*: beyond the onset the error falls like $\varrho^{-2p}$ with $\varrho$ set by the bank's
   analyticity. A smoother bank should steepen this tail. The fitted $\varrho$ is an *effective convergence
   parameter*, not a certified strip width (audit, NEEDS-RESTATEMENT).

Measured context (2D lane report, $1024^2$, worst over reached states, continuum target): `fast` Gauss $32^2$ 0.11,
$48^2$ 3.4e-2, $64^2$ 2.3e-2, $128^2$ 6.5e-3; `acc` Gauss $48^2$ 0.18, $64^2$ 1.9e-2. Hari's bank (DECODERS.md,
$M=320$): $32^2$ 7.8e-2, $48^2$ 4.5e-4, $64^2$ 3.9e-6. At comparable onset, our tail is 2–4 orders slower. This
lane tests whether the slow tail is the bank. The **pre-registered expectation** is therefore modest on the
acceptance bar 0.116, which sits near the onset for `fast` (Gauss $32^2$ already gives 0.11) and just past it for
`acc` ($64^2$): a smoother bank can at best move `acc` from $64^2$ to about $48^2$ and `fast` very little. Its
effect should show mainly at the tighter bar 0.01 and in the tail slope. A negative or small result is a valid
outcome and will be reported as such.

## 2. Provenance of the frozen bank, and what is held fixed

The frozen 2D bank `dn256b` (the 2D lane's model; `experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl`,
sha256 `18f0266a…`) was produced in three steps (traced from the run JSONs and code on this branch):

| step | code | what it did |
|---|---|---|
| 1. joint bank + head + codes training | `experiments/separable-decoder/sep_burgers_r3.py` → `sep_solvers.train_autodecoder_v2`, Slurm 2835788 | $K=16$, $R=512$; RFF $n_{ff}=128$, $\sigma=4.0$ (default, no override), $B\sim\sigma\,\mathcal N(0,1)^{2\times128}$; $g$: 2 SiLU layers of width 1024; head $h$: 2 SiLU layers of width 256 + linear skip; Dirichlet factor $16x(1-x)y(1-y)$; loss = global relative MSE + $10^{-4}$ feature-Gram orthonormality; AdamW lr $10^{-3}$ warmup-cosine to $10^{-5}$, weight decay $10^{-5}$ on MLP weights; EMA 0.999 (raw vs EMA chosen at end); 300 000 steps, per step all 16 384 training states and a random subset of $p_{\rm sub}=4096$ points, the last 10 000 steps on all points; seed 0; f64 |
| training data | `burgers2d_film.make_rollout(256)` via `sep_burgers_r3.build_data_lean` | the canonical seed-0 draw of 576 trajectories (Gaussian bumps $c_x,c_y\in U(.15,.85)$, $w\in U(.05,.20)$, $a\in U(.5,2)$, $\log\nu\in U(\log .01,\log .1)$), backward Euler $\Delta t=0.005$, 50 steps, first-order sign-upwind advection, **256 nodes per axis including the boundary ($h=1/255$)**; 16 384 of the 29 376 states picked by seed 0 (every state with $t\le5$ steps, the rest at random); all $254^2$ interior points |
| 2. head refit (`hfit`, bank frozen) and dense coefficient extraction | `sep_coeff_extract.py`, `sep_hfit.py` (`runs/dn256b`) | head refitted on 131 072 extracted states; **the bank $G$ is unchanged** |
| 3. importance rotation | `burgers-bank-knob/make_rotation.py` (`exp/2026-09-25-burgers2d-test`) | $G=Q_GR_G$ on the $256$-node training mesh; SVD of the per-row-normalised $R_G h(z_i)$ over the 131 072 training codes; $T=R_G^{-1}V_s$, $L=V_s^{\mathsf T}R_G$; first $R'$ rotated columns kept |

Original training cost: 300 000 steps in 2.70 h on one A100-80GB (round-3 summary, the sibling job 2835789 of the
same recipe); estimated $\le3$ h per value-only bank and $\le4$ h per Sobolev bank (§9).

**Held fixed for every retrained bank** (only the training recipe differs): the generator, the 576-trajectory seed-0
draw, the seed-0 state pick, all interior points, $K=16$, $R=512$, $n_{ff}=128$ (except where named), widths,
optimiser, schedule, steps, EMA, weight decay, orthonormality term, seed 0 for parameters/codes/point subsampling.
The trainer is a byte-for-byte copy of `train_autodecoder_v2` with one optional added term (§3.1); with $\lambda=0$
the loss is unchanged.

**Ordering for retrained banks.** The deployed rotation used the head's coefficient vectors $h(z_i)$ at 131 072
extracted codes (step 2), which needs a head refit we do not repeat. For every bank — the frozen one included — this
lane uses the **same construction with the head coefficients replaced by the bank's own least-squares
coefficients of the training states** ($c_i=\arg\min\|Gc-u_i\|$ on the $254^2$ interior of the training mesh, over the bank's own
16 384 training states; for the coarse-data bank, its own coarse training states): rows $(R_Gc_i)^{\mathsf T}/\|R_Gc_i\|$, SVD, $T=R_G^{-1}V_s$. The head approximated exactly
these coefficients, so the ordering criterion (training energy per state) is the same; the frozen bank is evaluated
with **both** the deployed `rotation_R512` (replication of the 2D lane) and the lane's rotation (the like-for-like
comparator). The linear-rung trust radius and the initial-fit candidate codes, which the 2D lane took from
$h(Z_{\rm tr})$, are taken from the same $c_i$.

**Settings.** Only the linear rungs: `acc` ($R'=384$, $M=1536$) and `fast` ($R'=128$, $M=512$), with the 2D lane's
solver unchanged (`qcore.make_linear_query`). The `head` setting is **out of scope**: it needs a head per bank, and
the 2D lane found it non-mesh-invariant for every arm (B2 1.91), so it cannot isolate a bank effect.

## 3. Arms

### 3.1 Sobolev term (3a)

$$\mathcal L=\underbrace{\frac{\operatorname{mean}_{s,p}(\hat u_{sp}-u_{sp})^2}{\operatorname{mean}(u^2)}}_{\text{unchanged value loss}}
+\lambda\,\frac{\operatorname{mean}_{s\in S_g,p}\big\|\nabla\hat u_{sp}-D_hu_{sp}\big\|^2}{\operatorname{mean}\|D_hu\|^2}
+10^{-4}\,\text{orth},$$
with $\nabla\hat u$ the bank's exact spatial gradient (forward-mode, as in the ROM) times the head coefficients,
$D_h$ the **second-order central difference on the 256-node training mesh** (neighbours of an interior node are
always on the mesh; boundary values are the exact zeros), evaluated at the same $p_{\rm sub}$ random points as the
value term and on a random subset $S_g$ of 2048 of the 16 384 states per step (all points in the last 10 000 steps),
i.e. an unbiased estimate of the full gradient loss at about 1/8 of its cost. The denominator is a constant computed
once on all training states.

*Why finite differences and not a spectral derivative.* The training data are a first-order upwind solution on this
mesh: their sine series converges only algebraically (the fields have nonzero second normal derivative at the walls,
and the upwind solution has grid-scale structure near steep fronts), so a spectral derivative at the training
resolution would amplify exactly the grid-scale content we want the bank *not* to learn, and would add Gibbs ringing
at the walls. Central differences are local, have no wall artefact, and their truncation error
$\tfrac{h^2}{6}u'''$ is about $(h/w)^2/6\lesssim2\times10^{-3}$ relative for the narrowest bump ($w=0.05$,
$h=1/255$), well below the bank's gradient error (to be measured). They are also the derivative the same data's
discretisation is consistent with.

Pre-registered $\lambda$ ladder: $\lambda\in\{0.01,\ 0.1,\ 1\}$. Round 1 runs $\lambda=0.1$; round 2 completes the
ladder.

### 3.2 Smoothness sweep (3b)

Feature scale $\sigma\in\{4\ (\text{current}),\ 2\ (\text{half}),\ 1\ (\text{quarter})\}$, everything else fixed
(same seed, so $B$ is the same Gaussian draw rescaled). No spectral penalty in round 1 (one lever at a time).

### 3.3 Rounds

| round | id | recipe | purpose |
|---|---|---|---|
| 1 | `base` | $\sigma=4$, $\lambda=0$ | retrain of the original recipe: like-for-like comparator and training-noise control |
| 1 | `sob01` | $\sigma=4$, $\lambda=0.1$ | H1 |
| 1 | `sig2` | $\sigma=2$, $\lambda=0$ | H2 |
| 1 | `sig1` | $\sigma=1$, $\lambda=0$ | H2 |
| 2 | `sob001`, `sob1` | $\sigma=4$, $\lambda\in\{0.01,1\}$ | λ ladder completion (always run) |
| 2 | `comb` | best $\sigma$ (by the H2 rule, §6) with best $\lambda$ (by the H1 rule) | 3c, **only if** H1 passes and some $\sigma<4$ passes H2; otherwise not run |
| 2 | `coarse` | `base` recipe on data generated at **128 nodes per axis** ($h=1/127$), same trajectories, same pick | 3d |
| 2 | `base_s1` | `base` with seed 1 | seed-variance control (round 1 only measures same-seed rerun noise) |

Round 2 is decided by the rules of §6 applied to the round-1 validation results; its arms and their order are fixed
here.

## 4. Cohorts and references

- **Selection and all verdicts:** `dev6 ∪ val32` (38 cases; the 2D lane's development/validation cohorts, generated
  from their recorded seeds by `qcore.cohort`), disjoint from the training draw (asserted).
- `test64` is **not** run in this lane (no selection may touch it; replication on test is not needed for a bank
  comparison). No new cohorts are created.
- References: the 2D lane's refined references at $8192^2$ restricted to the 257² shared nodes, ST ($\Delta t/16$)
  and S ($\Delta t$, space-only), job 4734273, pulled archive `worktrees/2026-10-01-quadrature-study/experiments/quadrature-study/runs/refdv/archive/output`,
  sha256-checked against its manifest exactly as `qstudy.py` does. They are first-order in time
  (ST) and first-order upwind-limited where unrefined: **every end-to-end accuracy number is PROVISIONAL**, and S is
  reported beside ST everywhere.

## 5. Metrics (all on `dev6 ∪ val32`, evaluation mesh $L=1024$ intervals per axis, f64, highest precision)

Per bank, per setting ($R'$ = 384 or 128 rotated columns):

1. **Projection error floor** $e_{\rm val}$: $L^2$ least-squares projection of the S reference fields (6 output times
   × 38 cases, on the $255^2$ interior of the 257² shared nodes) onto the rotated span; relative $L^2$ error, worst
   and median over the 228 states (the $t=0$ states included: the initial fit uses them). Secondary: the same against
   the training-mesh FOM (`burgers2d_film`, 256 nodes) of the 38 cases.
2. **Gradient error** $e_{\nabla}$: of the same projections, $\|\nabla(G'c)-D_hu\|/\|D_hu\|$ with $D_h$ central
   differences on the 257² grid ($h=1/256$) of the S reference, interior nodes; worst and median.
3. **$\rho$ ladders on reached states.** Population: the bank's own continuum rollout `gref` (Gauss $640^2$) states
   $k=1..50$ on all 38 cases (1900 states). Target: Gauss $640^2$; target convergence check Gauss $768^2$ vs $640^2$
   with bar $10^{-5}$ (the 2D lane's G6; a bank failing it has no valid ladder). Rules: Gauss
   $p^2$, $p\in\{8,16,24,32,40,48,56,64,80,96,128,160,192,256\}$, Fibonacci
   $\{987,1597,2584,4181,6765,10946,17711,28657,46368\}$. $\rho$ = the 2D lane's definition (relative 2-norm error of the
   $M$-vector of tested advections against the target; `qstudy` phase-2 code, copied). Reported: worst and median
   per rung, and the same restricted to the first 320 tests (Hari's $M$; a diagnostic of the test-band effect).
4. **Points needed** $m^*_{\rm G}(b)$, $m^*_{\rm F}(b)$: the smallest Gauss / Fibonacci rung whose **worst** $\rho$ over the
   population is $\le b$, for $b\in\{0.116\ (\text{primary, the 2D lane's bar}),\ 0.06,\ 0.01\}$; also a log-linear
   interpolation between bracketing rungs (secondary, labelled interpolated).
5. **Tail rate** $\hat\varrho$: least-squares slope of $\log(\text{median }\rho)$ vs $2p$ over the Gauss rungs with
   $p\ge a_{\max}$ ($p\ge 32$ for `fast`, $p\ge48$ for `acc`) and $\rho>10^{-12}$; an effective parameter only.
6. **Spectrum / effective bandwidth.** For 64 states of the population (fixed seed) and for the first $R'$ rotated
   columns: values on a $256\times256$ tensor Chebyshev–Gauss grid on $[0,1]^2$, 2D Chebyshev coefficients
   $a_{jk}$ by DCT; envelope $E_j=\max_{\max(j',k)= j}|a_{j'k}|/\max|a|$. Reported: the effective degree
   $n_\varepsilon=\min\{j:E_{j'}\le\varepsilon\ \forall j'\ge j\}$ for $\varepsilon\in\{10^{-4},10^{-8}\}$, and the
   fitted geometric rate of $E_j$ over the decaying range. Computed for $u$ and for the integrand $f=u(u_x+u_y)$
   (without $\psi$, which is common to every bank). Chebyshev, not sine, coefficients: the fields vanish on the walls
   but their second normal derivatives do not, so sine coefficients decay algebraically for **every** bank and would
   hide the analyticity that governs Gauss (§1).
7. **End-to-end** (rollouts, 50 steps, 6 output fields, the 2D lane's metrics): worst over 38 cases of the evolved
   ($t>0$) relative error at the 257² shared nodes against ST and S (**PROVISIONAL**), and the distance from the
   bank's own `gref` rollout. Arms per bank and setting: `gref`, Gauss $\{32,48,64,96\}^2$, Fibonacci
   $\{1597,4181,6765\}$, and `lat64` (the deployed $63^2$ mesh lattice, for the replication gate). The **ROM error
   floor** of a bank is its `gref` worst error.
8. **Cost.** Query seconds (fused LM rollout + decode at $1024^2$) for each bank at its own $m^*_{\rm G}(0.116)$ and
   $m^*_{\rm G}(0.06)$ rules and for `base` and the frozen bank at theirs, on one GPU in one job: every subject compiled
   and warmed, then A–B–A order over the 6 `dev6` cases × 3 repetitions with a 0.1 s burn before each call, medians
   reported. Also LM iterations per rollout. Times are never compared across GPU types or jobs.
9. **Training cost**: steps, wall seconds, GPU type, per bank.

## 6. Pass/fail bars and decision rules (validation cohorts only)

All comparisons are against `base` (same pipeline, same seed). **Noise rule:** an effect on a metric counts only if
it is larger than twice the `base`–frozen difference on that metric (same seed, same recipe, a different run and
ordering path); in round 2 the `base`–`base_s1` difference is added as the seed-variance yardstick and every round-1
verdict is re-read against it (an amendment records the outcome).

- **H1 passes** for a $\lambda$ if, in **both** settings, $e_\nabla$ (median) falls by $\ge20\%$, $e_{\rm val}$
  (median) does not rise by more than 10%, **and** in at least one setting the ROM error floor (`gref` worst vs S)
  falls by $\ge5\%$ relative while not rising by more than 2% in the other. Best $\lambda$ = the passing $\lambda$ with
  the lowest mean (over settings) `gref` worst-vs-S error.
- **H2 passes** for a $\sigma$ if $m^*_{\rm G}(0.06)$ or $m^*_{\rm G}(0.01)$ falls by a factor $\ge1.5$ in at least one
  setting with no increase in the other. Its **cost** is read on the frontier: $e_{\rm val}$ worst and the ROM error
  floor. A **clear 2D winner** (gate for any 3D work and for the round-2 `comb`) is a bank passing H1 or H2 whose ROM
  error floor (worst vs S) is within 10% of `base` in both settings. Best $\sigma$ = the passing $\sigma$ with the
  largest $m^*$ reduction.
- **3d reading** (`coarse`): the `coarse` bank's off-mesh ROM error floor vs S is compared with (i) `base`'s and (ii)
  the FOM's own error vs S at 128 and 256 nodes (FOM run in the same job, the engines discretisation at $L=128$ and
  $L=256$). If the `coarse` bank's ROM beats the 128-node FOM, the off-mesh accuracy is not only "having seen finer
  data"; if it is no better than the 128-node FOM, the objection stands for this bank. Reported as measured, both
  references, provisional.

## 7. Gates and controls (each must be checked to fire at one real mesh before any verdict)

| id | check | must |
|---|---|---|
| R1 | frozen bank + deployed rotation, `lat64` population: `acc` Gauss $64^2$ and `fast` Gauss $64^2$ worst $\rho$ at $1024^2$ | reproduce the 2D lane (1.9e-2, 2.3e-2) to 2 significant figures, and its `gref` and `lat64` worst ST errors to 0.01 pp |
| R2 | trainer copy with $\lambda=0$ (`base`) | the loss is bit-identical to the original's at $\lambda=0$ by construction (code diff audited); if the original job's training log is available, `base` reproduces its logged `rel-MSE` at steps 1 and 5 000 to 3 significant figures; the end-of-training `base`–frozen differences are the noise yardstick of §6 |
| C1 | `ctrl` Gauss $8^2$ | **fails** the 0.116 bar for every bank (worst $\rho>1$ in the 2D lane) |
| C2 | Chebyshev bandwidth estimator on synthetic fields: Gaussian bumps $w=0.2$ vs $0.05$ | must order them ($n_\varepsilon$ larger for 0.05) |
| C3 | the same estimator on a field with a kink ($\lvert x-\tfrac12\rvert\times$bump) | must **fail** to report a geometric rate (fitted rate flagged non-geometric: $E_j$ decays slower than any $\varrho>1.02$) |
| C4 | target convergence G6 per bank: Gauss $768^2$ vs $640^2$ worst $\rho$ | $\le10^{-5}$, else that bank's ladder is invalid |
| C5 | gradient-error metric on a field that lies exactly in a bank's span ($G'c$ for a population $c$, sampled on the 257² grid) and on the same field plus grid-scale noise of relative size $10^{-3}$ | clean: $e_{\rm val}\le10^{-10}$ and $e_\nabla$ = the FD truncation only ($\le10^{-2}$); noisy: $e_\nabla$ must be $\ge10\times$ the clean value (the metric must see grid-scale roughness) |
| C6 | rollout sanity | every rollout finite; LM exits recorded; a bank whose `gref` rollout fails (non-finite, or budget exits on more than 5% of steps) is reported as failed, not as an accuracy number |

## 8. Deliverables

`DESIGN.md` (this), lane code (`smoothtrain.py`, `bankeval.py`, `qcore_lane.py`, `cluster/stage.py`), `audits/`
(every Codex audit), `runs/<job>/` (pulled outputs, manifests), `make_report.py` → `report.md` (generated numbers,
provisional labels, glossary), and PNG plots in `plots/`:
`frontier.png` (error floor vs points needed, one point per bank, both settings, bars 0.116/0.06/0.01),
`ladders.png` (worst and median $\rho$ vs $m$, per bank, Gauss and Fibonacci),
`e2e.png` (end-to-end worst error vs ST and S and query cost, new banks vs `base` and frozen),
`spectra.png` (Chebyshev envelopes per bank).

## 9. Jobs and GPU-hour estimate

The lane cap is one Slurm job at a time. A job may request several GPUs on one node and run one independent process
per GPU (one bank per GPU, each in its own output sub-directory); timing comparisons are made only inside one process
on one GPU.

| job | GPUs | content | estimate |
|---|---|---|---|
| `tr1` | 4 × A100-80GB | round-1 training: `base`, `sob01`, `sig2`, `sig1` (one per GPU); each then writes its LS training coefficients and rotation | ≈4 h wall, ≈14 GPU-h |
| `ev1` | 5 × A100-80GB | evaluation (§5.1–5.7) of frozen(deployed rotation), frozen(lane rotation), `base`, `sob01`, `sig2`, `sig1` at $1024^2$ — one bank per GPU (frozen's two rotations share one GPU); then §5.8 timing of all banks on GPU 0 | ≈3 h wall, ≈12 GPU-h |
| `tr2` | ≤5 × A100-80GB | round 2 (§3.3) | ≈4 h wall, ≈18 GPU-h |
| `ev2` | ≤5 × A100-80GB | evaluation of round 2 (+ FOM at 128/256 for 3d) and a re-timing with `base` | ≈3 h, ≈12 GPU-h |

Total ≈56 GPU-h. 3D (only after a clear 2D winner) needs its own amendment.
