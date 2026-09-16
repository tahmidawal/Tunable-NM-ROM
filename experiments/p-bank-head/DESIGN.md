# p-bank-head — how far the Poisson bank floor and the head can be pushed

Pre-registered before any implementation. Amendments are dated in place and never
silently rewrite a criterion.

Worktree `worktrees/2026-09-16-p-bank-head`, branch `exp/2026-09-16-p-bank-head`,
forked from `exp/2026-09-14-head-ablation` at `fee3231a`.
Cluster namespace `/cluster/tufts/paralab/tawal01/p_bank_head_20260916/`.

---

## 1. The question

On Poisson 2D, with the **same separable architecture**, how far can the two
representation layers be pushed, and does the solved answer follow?

The incumbent frozen checkpoint (`runs/correction_accuracy10/checkpoints/r128_joint.pkl`,
$K=16$, $R=128$, 512 training sources at a $255$-interval mesh) has a three-layer
decomposition at $1024^2$ on the 12 development sources, taken from `artifacts/pabl01/result.json`:

| layer | quantity | value |
|---|---|---:|
| 1 representation — bank | worst bank projection floor | 2.314808 % |
| 1 representation — head | worst best-found over codes | 6.092634 % |
| 2+3 reduction and solver | worst solved same-grid | 6.092722 % |

Two facts follow and they set the whole design.

* **The solver is not the problem.** Solved $-$ best-found $= 0.00009$ pp. The weak
  Levenberg–Marquardt solve already finds the best point the head's image contains,
  and the 2026-09-15 zero-start diagnostic showed a single basin from every start.
* **The head, not the bank, is the binding layer.** The bank can reach 2.31 % and the
  head's $K=16$ image reaches only 6.09 %. The source family has **four** parameters
  $(c_x, c_y, w, a)$, so the solution manifold is four-dimensional; $K=16$ is four times
  the intrinsic dimension. The head is therefore not information-limited. It is
  **fit-limited**: either underfit on its own training set, or unable to interpolate
  between training codes on unseen sources.

That is the concrete form of the 2026-09-11 finding *"a better bank does not yet improve
the trained online decoder"* (widening $R$: $64 \to 128$ moved the bank floor
$6.489 \to 4.355$ % on the 42-source cohort but moved online worst error the wrong way,
$7.280 \to 7.554$ %). This cell must diagnose **why** before it chooses a head objective.

## 2. Layers, latitude and what is held fixed

Latitude: **training data amount, objective, $K$, $R$**.
Explicitly NOT in scope: new coordinate-feature families, new head architectures,
new solvers, new initializer policies, new test-mode families, new quadrature.

Held fixed everywhere: the separable decoder
$u(x;z) = \mathrm{bc}(x)\,g(x)^\top h_\theta(z)$ with $\mathrm{bc}$ the hard polynomial
boundary factor, $g$ a random-Fourier-feature MLP ($n_{\rm ff}=64$, scale 4, 2 hidden
layers), $h$ an MLP with a linear skip (2 hidden layers of width 128), SiLU, float64;
the weak residual, $M=257$ retained sine tests, the nearest-training-code initializer,
LM budget 300, stationarity tolerance $10^{-6}$, the output contract, and
`JAX_DEFAULT_MATMUL_PRECISION=highest`.

**One architectural consequence of $R$, recorded as a deviation.** The bank columns are
the outputs of $g$'s last affine layer, so a rank-$R$ bank needs last-hidden width
$\ge R$. The incumbent satisfies this with equality ($g_{\rm hidden}=128=R$). This cell
therefore sets $g_{\rm hidden} = R$, which reproduces the incumbent rule at $R=128$ and is
the minimal setting at which $R=512$ is attainable at all. The **head** width stays at the
incumbent 128 for every arm; only $h$'s output layer widens with $R$, as it must.
`widen_bank.widen` cannot be used to reach $R=512$ from the incumbent (its complement is
capped at $g_{\rm hidden}+1 = 129$ columns), so the $R=512$ banks are trained from scratch
and, for comparability, so are the $R=128$ banks.

### 2.1 Notation

At an $n$-interval mesh with $N_{\rm int} = (n-1)^2$ interior nodes:

* $G(\theta) \in \mathbb{R}^{N_{\rm int} \times R}$ — the cached bank, $G_{ij} = \mathrm{bc}(x_i) \tilde g_j(x_i)$.
* $G = Q_G R_G$ — thin QR, so $\|G d\|_2 = \|R_G d\|_2$ and coefficient least squares in
  the $R_G$ metric is exactly field-metric least squares.
* $\Phi \in \mathbb{R}^{N_{\rm int} \times M}$ — the $M=257$ retained orthonormal sine test modes,
  $\Lambda$ their diagonal eigenvalues.
* $B = \Phi^\top G \in \mathbb{R}^{M \times R}$, $\;f_m = \Lambda^{-1}\Phi^\top f$.
* The ROM solves, in $z$ alone,
  $$z^\star(f) \;=\; \arg\min_{z \in \mathbb{R}^K} \tfrac12\bigl\| B\,h_\theta(z) - f_m \bigr\|_2^2 ,$$
  by damped Levenberg–Marquardt from the nearest training code, and returns
  $u = G\,h_\theta(z^\star)$.

### 2.2 The three layers, as measured

For a source $s$ with same-mesh FD truth $u_s$:

$$\underbrace{\min_{c\in\mathbb R^R}\frac{\|Gc-u_s\|}{\|u_s\|}}_{\text{bank floor}}
\;\le\;
\underbrace{\min_{z\in\mathbb R^K}\frac{\|Gh_\theta(z)-u_s\|}{\|u_s\|}}_{\text{head best-found}}
\;\le\;
\underbrace{\frac{\|Gh_\theta(z^\star)-u_s\|}{\|u_s\|}}_{\text{solved}}$$

reported as the worst over the 12 development sources. The middle term is a seeded
multistart LM search, so it is an **upper** bound on the true head floor, exactly where
tightness would favour the head.

## 3. Cohorts and seeds

| cohort | construction | count | role |
|---|---|---:|---|
| training | `core.source_params(0, S)` | $S \in \{192, 768, 3072\}$ | bank and head training |
| — fit split | seeded 85 % of the above (`split_seed = 20260916`) | $0.85S$ | the only data the optimizer sees |
| — internal validation | the remaining 15 % | $0.15S$ | **every selection in this cell** |
| development | `core.source_params(7090703, 6)` $\cup$ `core.source_params(7090732, 6)` | 12 | reporting only, never selection |

`core.source_params(0, S)` draws each of the four parameter arrays at length $S$, so the
three training cohorts are **independent draws from the same family, not nested prefixes**.
$S=192$ is exactly the cohort `pabl01` builds its POD arms from, which keeps the lowest
rung an exact match to the incumbent comparison. The incumbent checkpoint itself was
trained on 512 sources (a prefix of a 576 draw at seed 0), so 512 sits between the
192 and 768 rungs; the incumbent is carried through as a **frozen control**, never retrained.

Disjointness of training and development parameters is asserted in code
(`assert not any(np.allclose(t, s) ...)`, the `pabl01` test) and the assertion failing
aborts the job. Final paper cohorts stay sealed; no new case is opened.

Training mesh: 255 intervals (the incumbent's `training_nodes = 256`), $64516$ interior nodes.
Evaluation meshes: 64, 256, 1024 intervals.
References: `pilot.reference` at 2048 intervals, restricted by nested-node injection.

## 4. Layer BANK — the sweep

Six arms, $R \in \{128, 512\}$ $\times$ $S \in \{192, 768, 3072\}$, all at $K=16$ with the
reconstruction-only objective, all with one fixed schedule and one fixed seed so the only
differences are $R$ and $S$.

Training is the staged recipe the accepted Poisson checkpoints use
(`staged_training.train_phase`), preceded by a from-scratch joint warm phase because there
is no inherited checkpoint to continue:

| phase | trainable | updates | lr |
|---|---|---:|---:|
| J1 joint | $B, g, h, h_{\rm lin}, Z$ | 100000 | $10^{-3}$ |
| B bank | $B, g, Z$ | 40000 | $3\times10^{-4}$ |
| H head | $h, h_{\rm lin}, Z$ (exact QR field metric) | 30000 | $3\times10^{-4}$ |
| J2 joint | $B, g, h, h_{\rm lin}, Z$ | 30000 | $10^{-4}$ |

Minibatch 64 sources $\times$ 4096 points, orthogonality weight $10^{-4}$, warmup fraction
0.02, cosine decay to 1 % of the peak — all the incumbent's values. Equal update counts
across arms, so larger $S$ buys coverage and not compute; realised exposures are recorded.

**Reported for every bank arm**, at 255 and 1023 intervals: the worst and median bank
projection floor on the fit split, on the internal-validation split, and on the 12
development sources; the bank's numerical rank and condition number; the head's worst
best-found on the internal-validation split and on the 12 development sources.

**Pre-registered bank selection rule.** The selected bank is the arm with the lowest
**worst internal-validation bank projection floor at 1023 intervals**. Ties are broken by
median, then by smaller $R$, then by smaller $S$. Development numbers are reported but do
not select.

**Pre-registered bank target.** Worst bank projection floor on the 12 development sources
at $1024^2$ **below 1.0 %**.

## 5. Layer HEAD — objective and the sweep

With the selected bank's $g$-track **frozen**, the reconstruction loss separates exactly:

$$\|G h_\theta(z) - u\|_2^2 \;=\; \bigl\|R_G h_\theta(z) - Q_G^\top u\bigr\|_2^2 \;+\; \underbrace{\|u - Q_GQ_G^\top u\|_2^2}_{\text{constant in }\theta,z},$$

so the head is fitted full-batch in $R$ dimensions with no spatial grid in the loop.
Write $T_s = Q_G^\top u_s$, $\;\pi_s^2 = \|u_s - Q_GQ_G^\top u_s\|_2^2$, $\;\nu_s = \|u_s\|_2$.

**Reconstruction term** (the incumbent objective, relative field metric):

$$\mathcal{L}_{\rm rec}(\theta, Z) \;=\; \frac{1}{S_{\rm fit}}\sum_{s}\frac{\bigl\|R_G h_\theta(z_s) - T_s\bigr\|_2^2 + \pi_s^2}{\nu_s^2}.$$

**Weak-residual term** — *exactly* what the ROM minimises online, and linear in the
coefficients $h_\theta(z)$, so it costs one $M\times R$ matrix–vector product per source:

$$\mathcal{L}_{\rm weak}(\theta, Z) \;=\; \frac{1}{S_{\rm fit}}\sum_{s}\frac{\bigl\|B\,h_\theta(z_s) - f_{m,s}\bigr\|_2^2}{\|f_{m,s}\|_2^2}, \qquad f_{m,s} = \Lambda^{-1}\Phi^\top f_s .$$

**Code-smoothness term in parameter space.** Let
$\hat p_s = \bigl(\tfrac{c_x-0.5}{0.35}, \tfrac{c_y-0.5}{0.35}, \tfrac{\log w - \log 0.045}{0.8}, \tfrac{a-1.25}{0.75}\bigr)$
be the family's own normalised descriptor and $\mathcal N$ the symmetric $\kappa=8$
nearest-neighbour graph of the fit split in $\hat p$. Then

$$\mathcal{L}_{\rm sm}(Z) \;=\; \frac{1}{|\mathcal N|}\sum_{(s,t)\in\mathcal N}\frac{\|z_s-z_t\|_2^2}{\|\hat p_s-\hat p_t\|_2^2 + \varepsilon}\;\Bigm/\;\frac{1}{S_{\rm fit}}\sum_s\|z_s\|_2^2 ,\qquad \varepsilon = 10^{-12},$$

a scale-free Dirichlet energy of the code map over the parameter domain. It pushes the
training codes onto a smooth image of the 4-dimensional parameter domain so that $h$ has
to interpolate rather than memorise. **Scope note, important:** $\hat p$ is an *offline
training-time* quantity only. The deployed query still receives nothing but the nodal
source field; no Gaussian descriptor ever reaches the online path. This is stated because
`core.py` carries the standing invariant "no Gaussian descriptors reach a neural model",
which is about inference and is preserved here.

**Total head objective:**

$$\mathcal{L}(\theta,Z) \;=\; \mathcal{L}_{\rm rec} \;+\; \beta_w\,\mathcal{L}_{\rm weak} \;+\; \beta_s\,\mathcal{L}_{\rm sm}.$$

**Head sweep.** $K \in \{16, 32\}$ $\times$ $(\beta_w, \beta_s) \in \{0,1\}\times\{0, 10^{-3}, 10^{-2}\}$
— twelve arms, each 150000 full-batch Adam updates at $10^{-3}$ with cosine decay,
fresh codes $Z \sim 0.1\,\mathcal N(0,I)$, one fixed seed. The $(\beta_w,\beta_s)=(0,0)$
arm is the incumbent objective and is the control within this layer.

**Pre-registered head selection rule.** For each $K$ separately, the primary is the arm
with the lowest **worst internal-validation best-found error at 255 intervals**. Ties by
median, then by smaller $\beta_w$, then by smaller $\beta_s$. Development numbers are
reported but do not select.

**Pre-registered head target.** Worst best-found on the 12 development sources at
$1024^2$ within **1.2×** of the same checkpoint's bank projection floor.

## 6. Diagnosis of the 2026-09-11 bank-versus-head finding

Run on the **incumbent** checkpoint before anything is trained, and repeated for every
trained checkpoint, so the diagnosis is a measured quantity and not a story:

| id | quantity | what a positive reading means |
|---|---|---|
| D1 | bank floor, worst/median, on the 512 incumbent training sources and on the 12 development sources | how much room the bank leaves |
| D2 | head error **at the stored training codes** on the training sources | fit quality where the optimizer actually looked |
| D3 | head **best-found** on the training sources (multistart LM in $z$) | the head manifold's own distance to its training data |
| D4 | **code-refit gain** $=$ D2 $-$ D3 | if large, the saved codes were not converged — an optimization failure |
| D5 | head best-found on the **development** sources | the head manifold's distance to unseen data |
| D6 | **generalisation gap** $=$ D5 $-$ D3 | if large, the head interpolates badly — a data/regularisation failure |
| D7 | solved $-$ D5 | if large, a solver failure |
| D8 | per-development-source best-found against its distance to the nearest fit-split parameter $\hat p$ (Spearman) | whether D6 is coverage-driven |

The three failure modes are mutually exclusive as stated, and the objective chosen in §5
is the one each mode implies: D4 large $\Rightarrow$ optimization (longer/cleaner code
training); D6 large $\Rightarrow$ coverage and smoothness ($S$, $\mathcal L_{\rm sm}$);
D4 and D6 both small with D3 large $\Rightarrow$ capacity ($K$) or objective mismatch
($\mathcal L_{\rm weak}$). All four levers are in the sweep, so the cell measures the
diagnosis and then reports which lever the data said to pull.

## 7. Layer SOLVE — the frozen evaluation

Run in **one job**, with the **unchanged** head-ablation Poisson machinery: arm (a)'s
query kernel from `head-ablation/poisson_ablation.py` (skinny sine-product projection,
nearest-training-code start, 257 tests, `make_stationary_lm` with budget 300 and
gtol $10^{-6}$, dense nodal output), at 64, 256 and 1024 intervals on the same 12
development sources, 3 timed repetitions, randomised subject order, GPU burn-in before
every timed invocation, every repetition array retained.

Subjects, all in the same job on the same GPU:

| subject | description |
|---|---|
| `a_neural@incumbent` | the incumbent checkpoint — **the control** |
| `a_neural_q32@incumbent` | incumbent plus its retained 32 eliminated corrections |
| `a_neural@<new K=16>` | the selected $K=16$ checkpoint |
| `a_neural_q32@<new K=16>` | with 32 eliminated corrections from a basis built in-job from its own training residuals |
| `a_neural@<new K=32>`, `a_neural_q32@<new K=32>` | the same for $K=32$ |
| `e_pod{16,32,64,128}` | classical POD-LSPG rebuilt from the `pabl01` 192-source snapshots |
| `e_podN{16,32,64,128}` | classical POD-LSPG rebuilt from the selected bank's own training snapshots |
| `d_freebank@<ckpt>` | unrestricted bank coefficients, only where $M > R$ |
| `dst_direct` | the direct discrete sine transform full-order solve |

Reported per subject and mesh: worst and median **same-grid** error (against the same-mesh
FD-DST solution) and **physical** error (against restricted 2048-interval truth), median
total query ms and device ms, stationary-exit counts, LM iterations, and the three-layer
decomposition per checkpoint.

**Fidelity gates, all before any verdict.**

* **G1, local.** The evaluation path run with the incumbent checkpoint reproduces
  `pabl01`'s `a_neural` per-case physical errors at 64 intervals to $\le 10^{-9}$ relative.
* **G2, in-job.** The same reproduction at 64, 256 and 1024 intervals against the staged
  `pabl01` reference scalars, $\le 10^{-9}$ relative.
* **G3, in-job.** The streaming POD implementation, run on the `pabl01` 192-source cohort,
  reproduces `pabl01`'s `e_pod{8,16,32,64,128}` per-case physical errors to $\le 10^{-9}$.
* **G4, in-job.** Every trained bank has full numerical rank $R$ at every evaluated mesh.
* **G5, in-job.** Every correction engine has full linear rank $q$.
* **G6, audit.** A NumPy-only audit that imports neither the driver nor JAX recomputes
  every reported error from the retained fields.

## 8. Pre-registered success and honesty clauses

**Success** (all four, jointly):

1. A **frozen** checkpoint with worst solved **same-grid** error $< 2.0\,\%$ at $1024^2$ on
   the 12 development sources;
2. every one of its 12 solves exits stationary (reason 4, normalised gradient $\le 10^{-6}$);
3. median total query cost within $1.5\times$ the incumbent's at $1024^2$ in the same job;
4. it beats POD-LSPG at the matched rank $k' = K$ on worst error.

Partial success is reported as partial; a missed clause is stated as a miss.

**Honesty clauses, pre-registered:**

* Report whether POD-LSPG at $k' = 8K$ still matches the neural head; if it does, say so
  in the headline.
* Report the direct DST cost and error. It will be faster. Say so.
* **Do not claim a speedup.** No ratio against any full-order solver is a headline in this
  cell.
* Report the bank and head targets of §4 and §5 as pass or fail on their own terms even if
  the solve clause fails.
* Report the training compute of every arm; "more data" arms are not given more updates.
* If a selection rule and the development ranking disagree, say so.

**Falsification.** The cell's working hypothesis is that the head, not the bank, binds,
and that coverage plus the weak term move it. It is falsified if, on the selected bank,
every head arm's development best-found stays above $1.2\times$ the bank floor while the
bank floor itself improves — that is the 2026-09-11 finding reproduced at larger scale,
and it would be reported as such, as a negative result, without a rescue arm.
It is also falsified, differently, if $S$ and the objective move nothing at all
($<5\,\%$ relative change in worst development best-found across all twelve head arms),
which would say the limit is the head's function class and therefore outside this cell's
latitude.

## 9. Resources and process

* Cluster jobs: **hard cap 3**; two planned (`pbh01` training + diagnosis, `psol01` solve),
  one held in reserve. One job per attempt directory, `squeue` before and after every
  submit, A100 preferred, `pax007` excluded, `gpu` partition only, `jax_backend=gpu`
  asserted, float64, `highest` matmul precision.
* Local GB10: sub-minute `jaxrun` smokes only; four sibling agents share the box.
* Every trained checkpoint is Git-tracked in this worktree.
* Archives are checksum-collected, chunked, Git-tracked, then the exact remote attempt
  directory is deleted.
* The report is generated from the run JSONs by a script in `reports/`. No number in it is
  typed by hand.

---

### Amendments

**2026-09-16, amendment 1 — the incumbent's training cohort, and a retraction inside `pbh01`.**
`pbh01` was submitted with `incumbent_training_count = 512` and therefore computed the
incumbent's *training-side* diagnostics D2, D3, D4, D6 and D8 against
`core.source_params(0, 512)`. That is **not** the incumbent's training set: the accepted
2026-09-11 lineage trained on `core.source_params(0, 576)[:512]`, and because every
`source_params` call draws each of the four parameter arrays at its own length, the two
cohorts share nothing. The symptom was unmissable — D2 came back at 2091 % worst — which is
how it was caught. **Those five `pbh01` values for the incumbent are retracted.** D1 on the
development cohort, D5 and D7 never touch the training parameters and stand unchanged, and
nothing in the bank or head sweep is affected, because those arms generate and use their own
cohorts consistently. The five retracted quantities are recomputed on the correct cohort in
`checks/incumbent-diagnosis/`, a bounded local GB10 diagnostic, and the config and driver are
corrected for the record. No cluster job was spent on the correction.

**2026-09-16, amendment 2 — cohort identity is checked to a tolerance, not by hash.**
While recovering the incumbent cohort it emerged that `core.source_params` is **not
bit-reproducible between the local GB10 and the Tufts cluster**: the Gaussian-width column,
$w=\exp(\mathcal U(\log 0.02, \log 0.1))$, differs by one ulp (max $1.39\times10^{-17}$)
between the two NumPy builds, while the other three columns agree bitwise. The 2026-09-11
assertion `sha(training) == checkpoint['training_draw_sha256']` passed only because it ran on
the cluster. Both audits therefore compare regenerated cohorts to a tolerance and record both
hashes instead of requiring them to agree. The difference is numerically irrelevant at the
tolerances this cell uses: the cross-job fidelity gate on `pabl01` arm (a) passes at
$2.96\times10^{-15}$ against a $10^{-9}$ tolerance.

**2026-09-16, amendment 3 — one extra reported checkpoint and one extra POD rung.**
(a) The selected bank arm carries its own jointly trained head. It is evaluated in the solve
job and reported as `bank_arm_head`, but it is **not** a pre-registered primary and cannot be
selected; it exists so the frozen-bank head sweep cannot hide a regression against the head
its own bank was trained with. (b) The POD rank set is run as
$k' \in \{8, 16, 32, 64, 128\}$ rather than $\{16, 32, 64, 128\}$; $k'=8$ is free, is one
of the rungs `pabl01` ran, and gives one more fidelity gate. Neither change relaxes a
criterion.


**2026-09-16, amendment 4 — the bank selection rule is replaced, because the pre-registered
one compares statistics over different-sized samples. Declared before any number on the new
cohort was computed.**

Section 4's rule selects on *the worst internal-validation floor*, but each arm's validation
split is a different cohort with a different size (29 sources at $S=192$, 115 at $S=768$, 461
at $S=3072$). The maximum over a sample is not a size-fair statistic: the worst of 461 draws
from a family is systematically larger than the worst of 29, so the rule as written is biased
towards small $S$ for reasons that have nothing to do with the bank. The first two arms of
`pbh01` make the size of the effect visible — `bank_R128_S192` reports validation worst
3.588 % against development worst 4.943 %, while `bank_R128_S768` reports validation worst
5.230 % against development worst 1.861 % — so the rule would have preferred the worse bank.
This is a flaw in the design, not a result.

**Replacement rule, fixed here before any floor on the new cohort was evaluated.** Define one
**common selection cohort**, `core.source_params(20260916, 256)` — a fresh seed, disjoint by
construction from every $S$-cohort (seed 0) and from the development cohort (seeds 7090703 and
7090732), and asserted disjoint in code. Every bank arm is scored on *that same* cohort, so the
maximum is comparable across arms. The selected bank is the arm with the lowest **worst floor
on the common selection cohort at 255 intervals**; ties by median, then smaller $R$, then
smaller $S$. The mesh is 255 rather than 1023 because the floor is measured to be
mesh-independent to three significant figures (incumbent 2.3227 % at 255 against 2.3152 % at
1023; `bank_R128_S192` 4.9435 % against 4.9349 %) and because a rank-512 bank at 1023 intervals
is a 4.3 GB array that does not belong on the shared local box.

Nothing else changes: the development cohort still selects nothing, the head selection rule of
section 5 is untouched, and every pre-registered target and success clause stands as written.
Both rules' picks and both rankings are reported side by side, and the `pbh01` head sweep — which
ran on the arm the original rule chose — is reported as run, whatever the replacement rule says.
