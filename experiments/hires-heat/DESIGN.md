# hires-heat lane — design (written before any GPU job)

Lane of the 2026-09-20 speed-and-accuracy campaign (`reports/2026-09-20-speed-accuracy-campaign-protocol.md` on main). Branch `exp/2026-09-20-hires-heat`, forked from `exp/2026-09-20-paper-h3d` @ 230c5410. Cluster namespace `/cluster/tufts/paralab/tawal01/hires_h_20260920/`, one directory per job. No results in this file.

## Question

Can the heat NM-ROM (learned spatial bank $G(x)\in\mathbb R^{N\times R}$, nonlinear head $h(z)$, $q$ linear bank corrections) be **accurate ($\le 1\%$ worst same-grid relative $L^2$) and $\ge 5\times$ faster than the named CN-CG full-order model** at meshes above those already audited: 2D at $2048^2$ and $4096^2$, 3D at $128^3$, using frozen weights evaluated on the finer grid? First sub-question: does *corrected* accuracy survive frozen-weight transfer above $1024^2$?

## Method

Decoded state $u = G\,(h(z) + D_q y)$, $D_q$ = first $q$ correction directions (SVD of training-only field-orthonormal reconstruction residuals; for the audited 2D checkpoint, which has none stored, they are built by the same rule from its regenerated training set, `core.sep2d_directions`). Each solve minimises $\lVert A(h(z)+D_q y)-t\rVert_2$; because $y$ enters linearly it is eliminated exactly (variable projection): with $AD_q=Q_qR_q$, solve the $K$-dimensional problem $\min_z\lVert (I-Q_qQ_q^\top)(A h(z)-t)\rVert$ by damped LM and recover $y=R_q^{-1}Q_q^\top(t-Ah(z))$. Port of `experiments/paper-h3d/rom.py`; parity with its published final-cohort numbers at $N=32$ was reproduced locally to 1e-12 (q0 1.5616 %, q96 0.76178 %, linear bank 0.72370 %, evolved-time metric).

$A$ for time steps is the weak matrix $a=\Phi^\top G$ on $M$ low sine tests (eigenfunctions of the discrete Laplacian, eigenvalues $\lambda_m$), computed by DST of bank columns (gated against explicit tests to 1e-10 where affordable). Initial fit: `field` ($A$ = triangular factor of $G$, target $Q^\top u_0$, the audited 2D contract) or `moments` ($A=a$, target $\Phi^\top u_0$, the paper-h3d contract).

Stepping arms (all labelled in every table):

- `cn`: Crank–Nicolson weak step, $t=\frac{1-\Delta t\nu\lambda/2}{1+\Delta t\nu\lambda/2}\odot a c_{\text{prev}}$, $\Delta t=0.025$ (20 solves). Baseline; same scheme as the named FOM.
- `exactseq`: exact semidiscrete weak factor $e^{-\nu\lambda\,0.1}$ between outputs (5 solves).
- `direct`: each output time fitted independently to $e^{-\nu\lambda t_j}\odot\Phi^\top u_0$ (5 batched solves, no sequential dependence). Valid only because the PDE is linear and autonomous and the tests are eigenfunctions; this is the same structure the DST control exploits, and is stated wherever the arm is reported.

## Arms in every comparison job (one allocation, same GPU, same I/O scope)

1. NM-ROM fast ($q=0$) and accurate ($q>0$ ladder), stepping arms above.
2. **Named FOM:** same-grid Crank–Nicolson + unpreconditioned warm-started CG, $\Delta t=0.025$, rtol $10^{-6}$ (the user-selected comparator of 2026-09-10).
3. Fastest tested FOM with error $\le$ the ROM's: CN-CG grid $\Delta t\in\{0.025,0.05,0.1\}\times$ rtol (listed in each config); selected by the summariser after the job, per ROM arm, on the same metric.
4. Coarse-grid FOM at matched accuracy: inject $u_0$ to $n_c$ intervals, same CN-CG there, multilinear interpolation to the fine grid; error against the continuum-spectral reference.
5. Direct/transform control: exact same-grid DST propagation (`dst_exact_CONTROL`).
6. Linear solve in the learned bank (`linear_bank_*_BASELINE`) — a labelled baseline, **not** the NM-ROM.

## Metrics and pass bar (pre-registered)

Primary error: worst over cases of the maximum over **all six output times including $t=0$** of current-relative $L^2$ error against exact same-grid semidiscrete propagation. Reported beside it: the evolved-times-only value (paper-h3d convention; its 0.754 % headline excludes the $t=0$ fit, which is 1.9 % with the moments initial fit), and the error against the continuum-spectral reference (empirical refinement check $<10^{-4}$, not a bound).

Bar per lane protocol, at the largest mesh reached: accurate setting primary error $\le 1\%$ (stretch $0.5\%$) **and** $S=T_{\text{named FOM}}/T_{\text{ROM}}\ge 5$, medians of $\ge 5$ retained synchronised repetitions after burn-in, GPU query scope (supplied full field on device → six full fields on device). The ratio against arm 3 is reported beside it and is the one that matters scientifically; missing either is reported as a miss.

Known in advance: the audited 2D checkpoint has $R=32$, and its free-coefficient solve has 1.68 % worst error attained at $t=0$ — so its ladder **cannot** reach 1 %. Job 1 measures that ladder and its transfer; job 2 trains a wider bank ($R=128$, paper-h3d recipe ported to 2D, new seeds 791000/791001 for train/validation) as a new, labelled checkpoint. Timing cohorts stay the 12 audited development draws (seeds 790711, 790716) in 2D and the 16 development draws (seed 920311) in 3D. A sealed 2D cohort (seed 791099, 16 draws) is opened once, in the last 2D job, after all choices are frozen. No selection on it.

## Controls that must fail / gates

- q-ladder monotonicity and the $R$-limit: at $q=R-K$ the ROM must approach the linear-bank error (checked at a real mesh in the local smoke: 1.046 % vs 1.031 %).
- Weak-matrix DST identity vs explicit tests $<10^{-10}$; reference refinement $<10^{-4}$; all fields finite f64; `jax_backend=gpu` or exit 42.
- Every optimisation: parity against the unoptimised arm in the same job (fields $\le 10^{-8}$ relative for iterative-path changes that alter floating-point order, integers reported; exact-algebra changes $\le 10^{-12}$) and same-job before/after timing. Logged in `SPEED-LOG.md`.
- Non-stationary LM exits and unconverged CG solves are counted and reported; an arm with failures is not selected as comparator.
- Independent NumPy/SciPy audit (`audit.py`): reference recomputed with `scipy.fft.dstn` on the full grid, errors recomputed from saved fields (full fields for two cases where $N\le 3\times10^5$; strided sub-grid fields elsewhere, tolerance 5 % relative on the error value, stated as a restricted audit).

## Stop rules and budget

$\le 8$ GPU jobs, $\le 2$ running; H200 + 240 G for $\ge 2048^2$ / $\ge 128^3$. Stop when the bar is met at the largest mesh with the sealed cohort opened once, or the budget is spent. $256^3$ only if everything else is done. Planned: J1 audited-checkpoint ladder 256→4096 (transfer answer); J2 wide-bank training + 256/1024 panel; J3 3D 64/128 profile panel; J4+ profile→fix→re-measure and the sealed 2D job.

## Addendum 1 (2026-09-20, before job `h2d-final04`; written after jobs 4051290 and 4054869 were audited)

Frozen for the sealed 2D job: checkpoint `inputs/wide2d` (bank R=128 + head K=8, trained in job 4054869 on seeds 791000/791001 only; hashes in `inputs/wide2d/SHA256SUMS`), $M=256$ tests, meshes 1024²/2048²/4096², arms exactly as listed in `configs/h2d-final04.json`. Roles declared now: **fast** = $q=0$, **accurate** = $q=32$ (genuinely nonlinear: $K+q=40<R$), **top rung** = $q=96$. The K=16 head (better validation error) is deliberately not used: K=8 was the declared model of `h2d-wide02.json` and no selection is made after seeing panel numbers. Speed variants (`field_chain`, `…_tol1e-4_chol`, `…_direct_tol1e-4_chol`) are accepted per SPEED-LOG rule only if, on the 12 development cases, their worst all-times error is within 2 % (relative) of the `cn` arm with the same $q$ and they have no more failing solves; acceptance is decided on development cases and then merely reported on the sealed cohort. The sealed cohort (seed 791099, 16 draws) is opened once, here, for every declared arm; the bar verdict is read from it at 4096² for the accepted fastest variant of the accurate arm, with the `cn` arm beside it. No retuning afterwards.
