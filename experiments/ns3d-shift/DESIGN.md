# NS3D shift-aware decoder: solve the translation online

Development-only design. No number in this file is a measurement. The final
cohort is a **new** sealed seed, named below, drawn only if the development
stage passes its bar. Seeds 202609203 and 202609211 are closed and are never
read here.

## Question

Periodic 3D Navier-Stokes at $N=32$ in this project is a translation orbit of
localized vortices. A fixed-span bank must rebuild every shifted state, and
`ns3d-grok` diag01 (job 4139559, development seed 202609202) measured the cost:
a rank-64 POD fit on the **uncentred, translation-augmented** training snapshots
floors at **53.860 %** evolved worst, 16/16 over the 5 % target, while a
**separately fitted centered** POD of the same rank, evaluated with an *oracle
per-time* translation, floors at **0.128 %**. Those are two different banks, not
one bank with and without an evaluation shift; the comparison is between what a
fixed span can do and what a shift-aware span can do at equal rank. Grok's fix,
a ROM that re-centres on the vortex centroid
of its own state each step, reached 4.799 % worst on sealed seed 202609211 but
at **0.468x** CNAB2, because moving a *stored* basis costs a 9222-term Fourier
phase sum every step against a 3.0 ms CNAB2 trajectory.

This experiment asks whether the translation can instead be an **online unknown
of the reduced least-squares solve**, with no basis ever moved:

> Let $u(x,t) = v(x - c(t), t)$ with $v$ in a fixed centered bank. Solve the
> reduced coefficients $a$ and the frame $c$ **together** from the same weak
> residual on the fixed Fourier test space. Is $c$ identifiable without an
> oracle centroid, how well conditioned is the augmented Jacobian, and is the
> result both under 5 % and faster than CNAB2?

## The mechanism, and why it should be free

The user's framing is $u(x) = g_\varphi(x - c)\,h_\theta(z)$ with $c$ solved
online. Evaluating the decoder at shifted coordinates is the *pointwise* view of
that idea. Its **co-moving (freezing) form** is algebraically identical and
costs nothing per step, which is the form this experiment implements.

Substituting $u(x,t) = v(x-c(t),t)$ into $\partial_t u = \mathcal P[N(u)] + \nu\Delta u$
and using that $N$, $\Delta$ and the Leray projector $\mathcal P$ all commute
with translation gives, in the moving frame,

$$\partial_t v \;=\; \dot c\cdot\nabla v \;+\; \mathcal P[N(v)] \;+\; \nu\Delta v .$$

So with $v = G a$ in a **fixed** centered bank $G$ and the project's fixed
solenoidal Fourier tests $\Phi$, the per-step weak residual is the existing one
plus a single extra term that is *linear* in the frame increment
$\delta = c^{n+1}-c^{n}$:

$$r(a,\delta) \;=\; \frac{A\,(a^{n+1}-a^{n}) \;-\; \Delta t\big(\mathsf T(\bar a,\bar a) - \nu\lambda\,A\bar a\big) \;-\; \sum_{d=1}^{3}\delta_d\, D_d\,\bar a}{1 + \tfrac12 \Delta t\,\nu\lambda},
\qquad \bar a=\tfrac12\!\left(a^{n+1}+a^{n}\right),$$

with $A=\Phi^{\mathsf T}G$, $\mathsf T_{mjk}=\langle\phi_m, G_j\times\operatorname{curl}G_k\rangle$
and $D_d=\Phi^{\mathsf T}\partial_d G$ all precomputed **offline and never
touched again**. The unknown vector is $(a,\delta)\in\mathbb R^{r+3}$, solved by
the same damped Levenberg-Marquardt used everywhere in this repository.

This is the project's usual implicit-midpoint weak residual (nonlinear term at
the midpoint, Crank-Nicolson diffusion), not the FOM's CNAB2; that was already
true of `ns3d_rom.make_run` and setting $\delta\equiv 0$ recovers it exactly.
"Algebraically identical" refers to the continuous substitution: applying the
midpoint rule after the change of coordinates is not the same discrete scheme as
applying a laboratory-frame midpoint rule to $G(x-c)a$.

Two consequences worth stating before any measurement:

1. **Nothing is shifted at run time.** The 9222-term phase sum that made the
   tracker slow does not appear. The per-step cost is the rank-$r$ weak solve
   plus three columns.
2. **A coordinate network is not what makes the shift cheap.** The freezing form
   is free for *any* fixed bank, POD included. If this is confirmed, the
   pointwise-decoder motivation is weaker than it looked, and the mechanism, not
   the architecture, is the result. The coordinate-network bank is still
   measured (arm F below) so the claim is tested rather than asserted.

## Arms

All development arms use training seed 202609201, development seed 202609202,
$N=32$, $T=0.2$, six output times, truth CNAB2 at $\Delta t = 0.001$, errors
relative $L^2$ against $\lVert u_0\rVert_2$, exactly as diag01.

Floors (representation only, no time stepping of the reduced model):

| id | arm | what it measures |
|---|---|---|
| A0 | fixed POD, ranks 32/64/128 | reproduces the 53.9 % baseline as a control |
| A1 | centered POD, oracle per-time shift, ranks 32/64/128/237 | representation ceiling of any shift-aware bank |
| F | coordinate-network bank, $R=64$, trained on centered snapshots | whether the learned pointwise bank reaches A1. **Deferred**: run only after the POD mechanism is decided, so a training budget is not spent on a mechanism that has not earned it. |

Solved ROMs (all in the centered rank-$r$ bank, same LM, same tests, no oracle):

| id | arm | frame |
|---|---|---|
| B0 | $\delta \equiv 0$ after the initial centering | diagnostic: an *initially centered, frozen-frame* ROM. Not a must-fail control. |
| B1 | $\delta$ solved from the residual, LM damping only | the idea |
| B2 | $\delta$ solved with the phase-condition gauge row | the idea with the degeneracy removed |
| C | centroid tracker, grid form (Grok diag04) | the arm to beat, re-run paired on development |
| E | CNAB2 at $\Delta t \in \{0.002,0.004,0.005,0.01,0.02\}$ | the FOM comparator |

The initial condition is legal for every solved arm: $c^0$ is the energy
centroid of the *given* $u_0$ and $a^0 = G^{\mathsf T}\,\mathrm{shift}(u_0,-c^0)$
with $G$ orthonormal (asserted). Those operations are inside the timed query. No
later truth is read by B0/B1/B2/C, and the runner signature for those arms
contains no truth argument at all.

There is **no solved oracle-$\delta$ arm.** Prescribing the true centroid path is
not a guaranteed ceiling for a finite bank, it would need per-step increments the
six saved frames do not supply, and interpolating them silently would be a
fabricated arm. The oracle appears only as the A1 *projection* floor, which is
itself conditional on the prescribed shifts and is not a global optimum over all
shifts.

**Gauge (B2).** $(a,\delta)$ are degenerate *to the extent that* the translation
tangents $\partial_d G a$ lie inside $\operatorname{span}(G)$; for a finite
centered bank that is approximate, so the degeneracy is measured, not assumed.
B2 appends three rows $w_g\,a^{n\mathsf T} S_d\,(a^{n+1}-a^{n}) = 0$ with
$S_d = (\partial_d G)^{\mathsf T}G$, the discrete form of the standard
orthogonality phase condition $\langle\partial_d v^n, v^{n+1}-v^{n}\rangle=0$.
Spectral differentiation on a periodic box makes $S_d$ **skew-symmetric**
(asserted numerically), so $a^{n\mathsf T}S_d\Delta a = \bar a^{\mathsf T}S_d\Delta a$
and using $a^n$ introduces no first-order lag. These rows are a **penalty, not an
exact constraint**: $w_g$ is swept on development and the realised phase
violation is recorded per step, so a weight that merely lets the condition drift
is visible.

**The gauge frame is not the centroid frame.** $\operatorname{centroid}(u) = c +
\operatorname{centroid}(Ga) \bmod 1$, so a valid solution may carry a $c$ that
differs from the true centroid while the reconstructed field is right. Every
accuracy number is measured on reconstructed *physical fields*; the frame
diagnostic compares $\operatorname{centroid}(\hat u)$ with
$\operatorname{centroid}(u_{\text{true}})$, never $c$ with a truth centroid.

## Pass bar

Pre-registered, unchanged from the project target and from the arm to beat:

- **Accuracy:** evolved worst relative $L^2 \le 5\,\%$ with **0 development
  cases over 5 %**, at a rank whose $\delta$ is solved online.
- **Speed:** paired median ROM query **faster** than the comparator CNAB2 query,
  where the comparator is the fastest tested CNAB2 step whose evolved worst is
  no larger than the ROM's. The arm to beat is 4.799 % at 0.468x.
- **Identifiability:** the shift carries information the coefficient update
  cannot absorb. Nonzero $\delta$ columns do **not** establish this, and gauge
  rows can inflate an augmented singular-value ratio by their weight alone, so
  the reported quantity is the singular spectrum of the **deflated** block
  $(I - J_a J_a^{\dagger})\,J_\delta$ at development states, alongside the plain
  spectrum of $\partial r/\partial(a,\delta)$ with and without the gauge, the
  column scalings used, and the numerical rank of the translation tangents
  $\partial_d G a$ inside $\operatorname{span}(G)$. Pre-registered threshold:
  the smallest singular value of the deflated block must exceed $10^{-6}$ times
  the largest singular value of $J_a$ for the frame to count as identifiable
  from the residual rather than merely chosen by the gauge.

Meeting accuracy but not speed is a partial result and is reported as such; it
does **not** license the sealed draw.

## Controls and integrity

- Every reduced arm is checked at frame 0 against the explicit orthogonal
  projection of $u_0$ into the shifted bank; a gap above $10^{-8}$ aborts.
- $\mathsf T$, $A$, $D_d$ and the diffusion identity
  $\Phi^{\mathsf T}\Delta G = -\operatorname{diag}(\lambda)A$ are each validated
  against a direct full-grid evaluation on random coefficients to $10^{-9}$
  before use; failure aborts. $G^{\mathsf T}G=I$ is asserted to $10^{-10}$.
- The checks must not validate the same formula twice. Independent of the
  residual code: $S_d + S_d^{\mathsf T}=0$ (skew symmetry); the analytic column
  $J_\delta[:,d] = -D_d\bar a/(1+\tfrac12\Delta t\,\nu\lambda)$ against
  `jacfwd`; fractional-shift equivariance of the reconstruction; and a
  **finite-difference shift check**: $\partial_{c_d}\,\Phi^{\mathsf T}\mathrm{shift}(Ga, cn)$
  evaluated by central differences of the FFT shift helper must equal $-D_d a$,
  which is what fixes the sign convention independently of the residual code.
  End to end, a $\delta\equiv 0$ trajectory must reproduce
  `ns3d_rom.make_run(..., linear=True)` in the centered frame to $10^{-10}$, and
  a complete query must be **equivariant**: the query on
  $\mathrm{shift}(u_0,s)$ must equal the shift of the query on $u_0$, including
  for an $s$ that crosses the torus boundary.
- `test_modes` is required to return complete cos/sin pairs at matched
  polarization ($M$ a multiple of 4), so the least-squares norm is translation
  invariant on the test space; $M \ge r+16$.
- The **must-fail control is A0**, the fixed uncentred bank, which reproduces
  53.9 % at rank 64. B0 is a diagnostic, not a must-fail control: it is an
  *initially centered, frozen-frame* ROM and may legitimately do well, in which
  case online translation is less necessary than the diagnosis claims and that
  is itself the finding.
- Convergence is reported, not assumed: per-step LM iterations, termination
  reasons (budget / tolerance / tiny step / rejection / stationary), residual
  norms, realised phase violation, and a finite-field check. A returned
  trajectory whose steps mostly exit on budget is reported as unconverged.
- Projection inequality, at the **same** predicted shift: the orthogonal
  projection of the truth into the shifted bank cannot be worse than the ROM
  field in that bank. A violation aborts.
- Errors are recomputed from saved development fields in an independent NumPy
  script before the remote directory is deleted.
- f64 everywhere, `JAX_DEFAULT_MATMUL_PRECISION=highest`, mandatory
  `jax_backend=gpu` preflight (exit 42), `gpu` partition, one directory per job
  under `/cluster/tufts/paralab/tawal01/ns3dshift_20260922/`, `squeue` checked
  before and after each submit, budget **<= 1 running, <= 6 total** GPU jobs.
- Timing: same job, same allocation as the errors it is compared with, GPU
  burn-in before each block, `block_until_ready`, >= 5 retained repetitions,
  medians, no ratio taken across jobs. **Complete queries** are timed, from
  $u_0$ to the six laboratory-frame output fields, so the initial centering and
  projection and every output reconstruction are inside the measurement, and the
  FOM and the ROM produce the same output contract. Diagnostic SVDs, operator
  builds and error reductions are outside the timed call. The errors quoted
  against a timing come from the outputs of a timed invocation.

## Development ladders, fixed now

Rank $r \in \{32, 64, 128\}$; tests $M \in \{96, 292, 1024\}$ (multiples of 4 so
each wave's two polarizations and its cos/sin pair are complete; 292 is the
audit's recommendation, about four times the 67 unknowns at $r=64$); ROM step
$\Delta t \in \{0.01, 0.005\}$; gauge weight $w_g \in \{0, 0.1, 1, 10\}$; LM budget
$\in \{20, 60\}$ with `gtol` $10^{-7}$. The selected setting is the one with the
smallest development **median** among those meeting the accuracy bar, ties broken
by median time, and it is then re-checked against the largest $M$ in the ladder
so a small residual in poorly observed modes cannot fake solver success.

## Stop rules

1. If A1 does not collapse relative to A0 at rank 64 (it is expected to, from
   diag01), the idea is wrong and the experiment stops at the floor table.
2. If B1 and B2 both fail to converge -- LM exiting on budget or rejection for
   most steps, or reconstructed fields diverging -- the experiment stops and
   reports that the frame is not recoverable from this weak residual with this
   solver. Failure to converge is reported as a solver/formulation result, not
   as proof that the frame is unidentifiable; the deflated-Jacobian measurement
   is what speaks to identifiability. There is no fallback to an oracle arm.
3. If the accuracy bar passes but the paired speedup is below 1.0, the
   experiment stops after reporting the cost breakdown. No sealed draw.
4. The multi-structure extension ($u=\sum_j g(x-c_j)a_j$) is **not** triggered by
   this experiment. If A1 is already below 5 % with one shift, a solved
   trajectory above 5 % points at the dynamics, the gauge, the test space, the
   discretisation or the solver -- not at the single-frame representation -- so
   adding independently shifted structures would not be the indicated fix. It is
   a separate design with its own relative-shift interaction terms.
5. GPU budget exhausted: stop and report.

## Audit disposition

Independently audited by `codex exec -m gpt-6-astra` before the first job
(reports: `results/codex-design-audit-pass1.md` and `-pass2.md`, two passes of the
same prompt; the second is the more detailed and the numbering below follows it).
Findings 4, 6, 7 and 14 were raised as blockers and all four are accepted and
fixed above: the gauge frame is no longer
conflated with the centroid frame and no stop rule compares $c$ to a truth
centroid; identifiability is now measured by the deflated block
$(I-J_aJ_a^{\dagger})J_\delta$ with a pre-registered threshold; B0 is demoted
from a must-fail control to a diagnostic and A0 becomes the must-fail control;
the solved oracle arm D is removed entirely; and the multi-structure trigger is
withdrawn as a fallback and restated as a separate design question. Accepted should-fixes: complete cos/sin test pairs, $S_d$ skew
symmetry, orthonormality assertion on the bank, independent (non-self-referential)
residual and Jacobian checks including a manufactured translating field, the
projection inequality, explicit convergence reporting, ladders fixed in advance,
complete-query timing with burn-in, and the corrected description of the two
different banks behind 53.860 % and 0.128 %. The audit's cost pessimism (findings
11, 12) is accepted as the reason the **first** job carries a paired cost profile
and is allowed to end the experiment on its own. Arm F (coordinate-network bank)
is deferred to after the POD mechanism is decided, per finding 14.

## Sealed cohort

Drawn only if the accuracy **and** speed bars both pass on development.
Seed **202609221**, 32 cases, disjointness against training seed 202609201
(512 cases), development seed 202609202 (16), and the closed seeds 202609203
(32) and 202609211 (32) checked by rounded parameter row before generation.
Opened once, with rank, $\Delta t$, $M$, gauge weight and LM budget frozen from
the development stage and recorded in this file as an amendment before submit.

## Glossary

- **Frame / shift $c$:** the translation of the vortex structure on the torus.
- **Centered bank:** POD (or learned) basis fit on snapshots each moved so its
  own energy centroid sits at the origin.
- **Oracle shift:** a translation read from the true field. A representation
  ceiling, never a model.
- **Freezing / co-moving form:** solving for the field in a frame that travels
  with the structure, with the frame speed as an extra unknown.
- **Phase condition / gauge:** the extra equation that fixes the split between
  "move the frame" and "change the coefficients".
- **Weak residual / LSPG:** the reduced equation tested against a fixed set of
  smooth solenoidal Fourier functions and solved in least squares.
- **Evolved worst:** for each case take the worst error over the output times
  after $t=0$; then take the worst over cases.
