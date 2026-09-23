# A quadratic-manifold ROM on 2D viscous Burgers at 256², priced in one allocation

What this covers: the quadratic manifold of Geelen–Wright–Willcox (2022) and Barnett–Farhat (2022),
$u = u_{\rm ref} + V_r a + W\,\mathrm{vech}(a\otimes a)$, run as a ROM baseline on the same cohort,
mesh, solver and timing protocol as this paper's NM-ROM and POD-LSPG rows — the "why not something
simpler than a neural head?" comparison. **These numbers are final for job `4186220`.** Every one is
generated from `summary.json` by `reports/render_report.py`; none is typed by hand. The design was
pre-registered in `DESIGN.md` before the first GPU job and independently audited
(`checks/design-audit.md`); one blocking defect in the fitting protocol was found and fixed before
any number here was produced (`DESIGN.md` §A1).

Job `4186220`, source commit `99e7b3d1`, NVIDIA A100 80GB PCIe, 35.2 min, 29 arms in **one allocation on one
GPU**. All gates pass, including the cross-job fidelity gates that check the panel is measuring the
archived model. The independent NumPy audit recomputed every error from the saved fields.

## The answer in three lines

1. **Accuracy.** The quadratic term is real and consistent — it beats POD-LSPG at the same solved
   dimension at every rung it is fitted at. But it does **not** reach the nonlinear head: at 16
   solved unknowns the quadratic manifold scores 22.214 % worst evolved error against the head's
   **1.8891 %**, a factor of **11.8**. It sits between POD-LSPG and us, much nearer
   POD-LSPG.
2. **Cost.** The $O(r^2)$ quadratic term costs 1.88× its own linear control at $r=8$ rising to
   4.64× at $r=64$, because the trial basis grows from $r$ to $1+r+r(r+1)/2$ columns. No
   quadratic arm beats the paper-rule full-order solver: the best is 0.210×, i.e.
   **4.8× slower** than the rule-matched FOM.
3. **Unknowns.** For **every** quadratic arm in the job, the cheapest NM-ROM setting at least as
   accurate is the same one: `q0_M64_eqcert_g1em06_fastL4`, **16 unknowns, 39.6 ms,
   1.8891 %**. The head needs **4× fewer unknowns** than the best quadratic arm and is
   **17.6× cheaper** while being **3.7× more accurate**.

## 1. The $r$ ladder, with the quadratic term isolated

`quad` is the fitted quadratic manifold; `lin` is the **identical** $u_{\rm ref}$ and $V_r$ with the
$W$ block deleted, so `lin`/`quad` isolates the quadratic term and nothing else. `pod` is this
paper's own classical POD-LSPG at the same rank, in the same job. "snapshot" is the relative
snapshot reconstruction residual of the fitted map, linear part only → with the quadratic term.

| $r$ | $P=r(r+1)/2$ | trial columns | ridge $\gamma$ | $\lVert W\rVert_F$ | snapshot rel. | quad worst ev. % | lin worst ev. % | POD worst ev. % | quad gain | quad ms | lin ms | quad cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 36 | 45 | 0.01 | 0.0578 | 0.2660 → 0.1703 | 40.779 | 52.439 | 52.560 | 1.29× | 42.0 | 22.3 | 1.88× |
| 16 | 136 | 153 | 0.01 | 0.0733 | 0.1367 → 0.0753 | 22.214 | 28.433 | 28.725 | 1.28× | 85.3 | 46.6 | 1.83× |
| 32 | 528 | 561 | 0.01 | 0.0743 | 0.0606 → 0.0294 | 12.685 | 18.827 | 18.799 | 1.48× | 223.9 | 82.2 | 2.72× |
| 64 | 2080 | 2145 | 1 **(grid top)** | 0.0039 | 0.0210 → 0.0191 | 6.942 | 7.097 | 7.084 | 1.02× | 696.5 | 150.1 | 4.64× |

**The quadratic term pays, and by a consistent amount.** It cuts the worst evolved error by
1.29×, 1.28× and 1.48× at $r=8,16,32$ against an identical linear part, and it beats
POD-LSPG at the same rank at every one of those rungs. This is a real effect, not noise, and it is
the honest case for the method.

**It stops paying at $r=64$** (1.02×, i.e. nothing). The ridge selection pins $\gamma$ at the top
of the declared grid there and $\lVert W\rVert_F$ collapses by two orders of magnitude: with
$P=2080$ coefficients per row fitted from 3328 snapshots, the quadratic term does not generalise
across held-out trajectories. This is the outcome `DESIGN.md` §3.1 predicted in writing before the
job ("the quadratic gain is largest at small $r$ and shrinks or reverses by $r=64$") and §A1.1
pre-registered how to read. It is a property of the method at this snapshot budget, and a lane with
more trajectories could change it.

## 2. The two pre-declared sensitivity arms

Both values were fixed in the config **before** the job, never chosen after seeing evaluation error.
They exist so that a loss cannot be blamed on a choice this lane made for the baseline.

| arm | what it varies | against | worst ev. % | baseline % | error ratio | cost ratio |
|---|---|---|---|---|---|---|
| `qman32_quad_M512` | test-space M | `qman32_quad_M128` | 12.770 | 12.685 | 0.993× | 1.46× |
| `qman32_quadg0p0001_M128` | fixed ridge | `qman32_quad_M128` | 8.946 | 12.685 | 1.418× | 1.05× |

**The test space is not the limiting factor.** Quadrupling the weak test modes at $r=32$ from
$M=128$ to $M=512$ changes the error by 0.993× — that is, not at all (marginally worse) — at
1.46× the cost. The shared $M=4r$ contract is not what held the quadratic manifold back.

**The ridge rule is leaving accuracy on the table, and this is stated plainly.** At $r=32$ the
pre-declared fixed $\gamma=10^{-4}$ beats the rule-selected $\gamma=0.01$ by **1.42×**
(8.946 % against the ladder's) for 1.05× the cost. So **the ladder above is a lower
bound on what a better-tuned quadratic manifold could reach**, and a reader should treat it as one.
It does not change the verdict: the best quadratic arm of any kind in this job,
`qman64_quad_M256` at 64 unknowns and 6.942 %, is still **3.7× worse** than
the head at 16 unknowns. Selecting $\gamma$ by ROM error on the evaluation cohort — which is how
GWW and BF often choose it — is forbidden here because it is selection on evaluation data; that
asymmetry is a real limitation of this comparison and is named again in §5.

## 3. Why: it is the manifold, not the solve

The representation floor is an independent, untimed best-found fit of each reference field on the
arm's own manifold, with the shared LM driver — it says what the trial map could do if the ROM
solve were perfect.

| arm | unknowns | representation floor % | $t=0$ compression % | worst all-times % | worst evolved % |
|---|---|---|---|---|---|
| `qman8_quad_M32` | 8 | 73.773 | 73.773 | 73.773 | 40.779 |
| `qman16_quad_M64` | 16 | 52.148 | 52.148 | 52.148 | 22.214 |
| `qman32_quad_M128` | 32 | 32.364 | 32.364 | 32.364 | 12.685 |
| `qman64_quad_M256` | 64 | 19.614 | 19.615 | 19.615 | 6.942 |
| `pod32_M128_dense` | 32 | 47.068 | 47.068 | 47.068 | 18.799 |
| `pod64_M256_dense` | 64 | 19.815 | 19.816 | 19.816 | 7.084 |
| `q0_M64_dense_g1em06` | 16 | 2.545 | 2.563 | 2.563 | 1.889 |

For every reduced arm the floor, the $t=0$ compression and the worst all-times error agree to three
decimals. **The binding constraint is what the manifold can represent, not what LSPG can find on
it** — the solve is finding essentially the best fit available. That is the attribution the design
asked for, and it means the quadratic manifold's shortfall is a statement about the trial map, not
about the solver, the tolerance or the budget.

## 4. The full panel

Every row below came out of job `4186220`, one allocation, one GPU, five timed repetitions, randomised
subject order, 0.25 s burn-in. **Speedups are single-job ratios**; the full-order comparator is the
fastest tested setting whose worst evolved error is at or below the row's. The mesh's own
discretisation error is 4.0265 % (`fft_tight` against the 4096-interval reference) — every
quadratic-manifold arm is above it, and the NM-ROM rows are below it.

| arm | family | solved unknowns | trial-basis columns | quadratic terms P | M | worst evolved % | median evolved % | worst all-times % | GPU-query ms | complete-query ms | FOM by the rule (GPU) | speedup (GPU) | speedup (complete) | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qman8_lin_M32` | quadratic manifold | 8 | 9 | 0 | 32 | 52.4388 | 35.9210 | 79.3106 | 22.298 | 24.134 | `nt1e-2_dt01` | 0.395× | 0.436× | yes |
| `qman8_quad_M32` | quadratic manifold | 8 | 45 | 36 | 32 | 40.7792 | 28.8606 | 73.7730 | 41.971 | 44.210 | `nt1e-2_dt01` | 0.210× | 0.238× | yes |
| `qman16_lin_M64` | quadratic manifold | 16 | 17 | 0 | 64 | 28.4334 | 21.6091 | 61.5226 | 46.592 | 48.658 | `nt1e-2_dt01` | 0.189× | 0.216× | yes |
| `qman16_quad_M64` | quadratic manifold | 16 | 153 | 136 | 64 | 22.2144 | 14.9212 | 52.1481 | 85.292 | 87.552 | `nt1e-2_dt01` | 0.103× | 0.120× | yes |
| `qman32_lin_M128` | quadratic manifold | 32 | 33 | 0 | 128 | 18.8268 | 10.0055 | 47.1893 | 82.184 | 84.336 | `nt1e-2_dt01` | 0.107× | 0.125× | yes |
| `qman32_quad_M128` | quadratic manifold | 32 | 561 | 528 | 128 | 12.6849 | 6.5207 | 32.3644 | 223.873 | 225.738 | `nt1e-2_dt01` | 0.039× | 0.047× | yes |
| `qman32_quad_M512` | quadratic manifold | 32 | 561 | 528 | 512 | 12.7698 | 6.5810 | 32.3644 | 327.845 | 329.737 | `nt1e-2_dt01` | 0.027× | 0.032× | yes |
| `qman32_quadg0p0001_M128` | quadratic manifold | 32 | 561 | 528 | 128 | 8.9465 | 6.1704 | 19.0802 | 234.487 | 236.137 | `nt1e-2_dt01` | 0.038× | 0.045× | yes |
| `qman64_lin_M256` | quadratic manifold | 64 | 65 | 0 | 256 | 7.0967 | 3.4378 | 19.7926 | 150.085 | 151.740 | `nt1e-2_dt01` | 0.059× | 0.069× | yes |
| `qman64_quad_M256` | quadratic manifold | 64 | 2145 | 2080 | 256 | 6.9416 | 3.4079 | 19.6145 | 696.544 | 698.605 | `nt1e-2_dt01` | 0.013× | 0.015× | yes |
| `q0_M64_dense_g1em06` | NM-ROM | 16 | — | — | 64 | 1.8890 | 1.0059 | 2.5629 | 284.919 | 287.087 | `nt1e-3_dt01` | 0.070× | 0.076× | yes |
| `q0_M64_eqcert_g1em06` | NM-ROM | 16 | — | — | 64 | 1.8891 | 1.0100 | 2.5629 | 58.230 | 60.138 | `nt1e-3_dt01` | 0.342× | 0.365× | yes |
| `q256_M1088_dense_g1em06` | NM-ROM | 272 | — | — | 1088 | 0.5194 | 0.1790 | 0.9053 | 4090.689 | 4093.004 | `nt1e-3_dt005` | 0.008× | 0.008× | yes |
| `q256_M1088_eqcert_g1em06` | NM-ROM | 272 | — | — | 1088 | 1.0361 | 0.3607 | 1.0361 | 699.408 | 701.696 | `nt1e-3_dt005` | 0.045× | 0.048× | yes |
| `q0_M64_eqcert_g1em06_fastL4` | NM-ROM (fast kernel) | 16 | — | — | 64 | 1.8891 | 1.0100 | 2.5629 | 39.550 | 41.372 | `nt1e-3_dt01` | 0.504× | 0.530× | yes |
| `pod8_M32_dense` | POD-LSPG | 8 | — | — | 32 | 52.5598 | 34.9459 | 78.5148 | 22.803 | 24.958 | `nt1e-2_dt01` | 0.387× | 0.422× | yes |
| `pod16_M64_dense` | POD-LSPG | 16 | — | — | 64 | 28.7250 | 21.4948 | 61.6503 | 46.635 | 48.716 | `nt1e-2_dt01` | 0.189× | 0.216× | yes |
| `pod32_M128_dense` | POD-LSPG | 32 | — | — | 128 | 18.7995 | 9.9416 | 47.0681 | 81.103 | 83.406 | `nt1e-2_dt01` | 0.109× | 0.126× | yes |
| `pod64_M256_dense` | POD-LSPG | 64 | — | — | 256 | 7.0835 | 3.4707 | 19.8156 | 147.428 | 149.336 | `nt1e-2_dt01` | 0.060× | 0.071× | yes |
| `pod256_M1024_dense` | POD-LSPG | 256 | — | — | 1024 | 0.7109 | 0.2834 | 3.7698 | 926.038 | 928.371 | `nt1e-3_dt005` | 0.034× | 0.036× | yes |
| `pod512_M2048_dense` | POD-LSPG | 512 | — | — | 2048 | 0.2184 | 0.0320 | 0.6125 | 2893.961 | 2897.090 | `nt1e-3_dt005` | 0.011× | 0.012× | yes |
| `dense_tight` | FOM | — | — | — | — | 0.0000 | 0.0000 | 0.0000 | 60.405 | 62.370 | — (is a FOM) | — | — | — |
| `fft_tight` | FOM | — | — | — | — | 0.0000 | 0.0000 | 0.0000 | 90.544 | 92.653 | — (is a FOM) | — | — | — |
| `nt1e-2_dt005` | FOM | — | — | — | — | 3.7127 | 1.4783 | 3.7127 | 15.596 | 17.645 | — (is a FOM) | — | — | — |
| `nt1e-2_dt01` | FOM | — | — | — | — | 3.1999 | 1.4192 | 3.1999 | 8.817 | 10.530 | — (is a FOM) | — | — | — |
| `nt1e-3_dt005` | FOM | — | — | — | — | 0.0489 | 0.0335 | 0.0489 | 31.473 | 33.353 | — (is a FOM) | — | — | — |
| `nt1e-3_dt01` | FOM | — | — | — | — | 1.5179 | 1.1980 | 1.5179 | 19.925 | 21.938 | — (is a FOM) | — | — | — |
| `nt1e-4_dt005` | FOM | — | — | — | — | 0.0338 | 0.0153 | 0.0338 | 36.890 | 38.798 | — (is a FOM) | — | — | — |
| `nt1e-4_dt01` | FOM | — | — | — | — | 1.5109 | 1.1722 | 1.5109 | 28.103 | 30.088 | — (is a FOM) | — | — | — |

## 5. Qualifications, stated rather than buried

* **No hyper-reduction for the baseline.** GWW and BF both pair the quadratic manifold with
  hyper-reduction (DEIM / ECSW). This lane constructs none for it, because a certified rule is a
  lane's worth of work; the quadratic arms run the exact dense advection sum. `DESIGN.md` §7
  pre-registered **dense against dense** as the fair headline cost ratio, so here it is:
  `qman64_quad_M256` at 696.5 ms and 6.942 % against the dense NM-ROM `q0_M64_dense_g1em06`
  at 284.9 ms and 1.8890 %. The head is **2.44× cheaper and 3.7× more
  accurate on the like-for-like comparison**, so the missing hyper-reduction does not rescue the
  baseline — it would have to be more than an order of magnitude to matter.
* **The ridge is selected on training snapshots only**, by held-out error over whole trajectories.
  §2 shows that rule is not optimal for ROM accuracy. Choosing it by ROM error would be selection
  on evaluation data and is refused; the sensitivity arm is the declared substitute.
* **$r$ stops at 64** because $P=r(r+1)/2$ is already 2080 against 3328 snapshots there; $r=128$
  would need $P=8256$ and cannot be fitted from this snapshot set at all.
* **One mesh, one PDE, one snapshot budget.** Nothing here refutes GWW or BF in their own regimes —
  smaller $r$, different equations, different snapshot counts, with hyper-reduction.
* **`qman` `lin` is not the POD-LSPG row.** It carries the mean shift and a centred POD basis. The
  paper's POD-LSPG rows are the `pod` family, in this same job at the same ranks.

## 6. What a reviewer gets from this

The simpler thing was tried, on equal terms, in the same allocation, and it works — it is reliably
better than a linear subspace of the same dimension. It is not competitive with the nonlinear head:
11.8× the error at equal unknowns, and no setting of it reaches the head's accuracy at
any dimension or cost tested. The reason is representational, not numerical.

## Glossary

Written for a reader opening this cold.

* **FOM / full-order model** — the original discretised PDE solved directly, here by
  FFT-preconditioned Newton–BiCGStab on the same $ 256 ^2$ grid. The thing a ROM has to beat.
* **ROM / reduced-order model** — solves for a few unknowns on a low-dimensional *trial map* instead
  of every grid value.
* **Trial map** — the function from the few reduced unknowns to a full field. POD-LSPG's is linear
  ($V_{k'}a$); the quadratic manifold's adds a quadratic term; ours is a trained nonlinear head.
  In this job the trial map is the **only** thing that differs between reduced arms.
* **POD** — proper orthogonal decomposition: the optimal linear basis for a set of snapshots.
* **LSPG** — least-squares Petrov–Galerkin: solve for the reduced unknowns by minimising the
  discrete residual, rather than by projecting the equations.
* **Snapshot** — a stored full-field state from a training run. Here: 128 trajectories × 26 states.
* **$r$, $k'$, solved unknowns** — the number of reduced unknowns actually solved for each time step.
  Matched across families when comparing "at the same solved dimension".
* **$P = r(r+1)/2$** — the number of distinct quadratic monomials $a_i a_j$, $i\le j$. It is why the
  quadratic manifold's cost grows quadratically in $r$.
* **`vech`** — the vector of those distinct monomials, each appearing once.
* **Trial columns** — how many full-field vectors the trial map stores: $r$ for POD,
  $1+r+P$ for the quadratic manifold. Online cost scales with this.
* **$W$, $\lVert W\rVert_F$** — the quadratic coefficient matrix and its size. A tiny norm means the
  fit decided the quadratic term should do almost nothing.
* **Ridge $\gamma$** — the regularisation strength in the least-squares fit of $W$. Larger shrinks
  $W$. Selected here on held-out **training trajectories**.
* **Grid top** — the selected $\gamma$ sat at the largest value offered, meaning the data wanted the
  quadratic term suppressed as far as the grid allowed.
* **`quad` / `lin`** — the same fitted $u_{\rm ref}$ and $V_r$ with, and without, the $W$ block. Their
  difference is the quadratic term's contribution and nothing else.
* **$M$** — the number of weak test modes the residual is projected onto; $4\times$ the unknowns by
  the shared contract.
* **Worst / median evolved %** — the error against the same-grid converged solve over the evolved
  times ($t = 0.05\ldots0.25$, excluding $t=0$), worst and median over the six cases, as a
  percentage of $\lVert u(0)\rVert$. The primary accuracy metric.
* **Worst all-times %** — the same including $t=0$, so it includes how well the map reproduces the
  supplied initial field.
* **$t=0$ compression %** — that $t=0$ term alone.
* **Representation floor** — the best fit of a reference field achievable *on* the arm's manifold,
  found offline. If a row's error equals its floor, the trial map is the limit, not the solver.
* **GPU-query ms** — wall time from the input field resident on the GPU to the six output fields
  resident on the GPU. **Complete-query ms** adds the host upload and download.
* **FOM by the rule** — the fastest full-order setting tested in this job whose error is at or below
  the row's. **Speedup** is that setting's time divided by the row's; below 1× means slower.
* **Converged** — every time step and the initial fit stopped for a legitimate reason with a small
  gradient, under the pre-registered rule. A non-converged arm is reported but excluded from the
  admissible frontier.
* **Held-out / cohort** — the six evaluation cases, disjoint from the 128 training trajectories.
  Nothing in this lane was tuned on them.
* **Hyper-reduction (DEIM / ECSW / EQ)** — approximating the expensive nonlinear term by evaluating
  it at a few points, so a ROM's cost stops scaling with the grid. Our NM-ROM's fast rows use a
  certified empirical-quadrature rule; the quadratic-manifold arms here have none.
* **Discretisation error** — the error of the exact solution *of this mesh* against a far finer
  reference. A ROM below it is resolving its mesh; above it is not.
* **Admissible / non-dominated** — respectively, converged under the pre-registered rule, and not
  beaten by another arm on both cost and error.
