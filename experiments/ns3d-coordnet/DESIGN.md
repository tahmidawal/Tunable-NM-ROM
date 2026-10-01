# NS3D: one coordinate-network bank inside the co-moving frame (pre-registration)

Written before any GPU job of this lane. Amendments are appended as §A1, §A2, …; nothing above
them is edited afterwards. No number in this file is a measurement of this lane; every number
quoted is from a parent lane and says so.

Branch `exp/2026-10-01-ns3d-coordnet-bank` (local; never pushed — mirrored code-only by
`sync_github.sh`), forked from `exp/2026-09-25-t2-ns3d-test` @ `acf1af87`. Cluster namespace
`/cluster/tufts/paralab/tawal01/nscoord_20261001/<job>/`, one directory per job, never reused.
At most about 10 cluster jobs in total.

## 1. Question

The paper's Navier–Stokes 3D rows use a **rank-64 POD bank of energy-centroid-centred
snapshots inside a co-moving frame, with a separate POD bank and head per mesh**
(`experiments/ns3d-shift-head`, report `reports/2026-09-23-ns3d-shift-head.md`). The paper's
method section describes the bank as a coordinate network shared by every mesh. This lane tests
whether **one** trained coordinate-network bank, sampled unchanged at $32^3$, $64^3$ and $96^3$,
can replace the POD bank inside the same co-moving frame, with the same $k=8$ head recipe, the
same tests, the same time stepping and the same solver.

## 2. What is inherited unchanged

From `ns3d-shift` / `ns3d-shift-head` (code imported, not copied):

- Co-moving form $u(x,t)=v(x-c(t),t)$, $v=G\,a$; frame increment $\delta=c^{n+1}-c^n$ solved
  online. Offline tensors $A=\Phi^{\mathsf T}G$, the quadratic advection tensor
  $\mathsf T_{mjk}=\langle\phi_m, g_j\times(\nabla\times g_k)\rangle$, and
  $D_d=\Phi^{\mathsf T}\partial_d G$ (`shift_rom.build_operators_fast`, validated in-job).
- Tests: the $M=292$ lowest divergence-free Fourier modes.
- Step: implicit midpoint, $\Delta t=0.02$, 10 steps, 3 damped Gauss–Newton sweeps (Marquardt
  $10^{-6}$, Cholesky), analytic Jacobian, linear-extrapolation start
  (`head_rom.make_head_run`, `shift_rom.make_frozen_run`).
- Head recipe: $k=8$, auto-decoder, two hidden SiLU layers of width 256 plus a linear skip,
  weighted-PCA initialisation, full-batch Adam 100 000 steps, warm-up + cosine
  $10^{-3}\to10^{-5}$, code penalty $10^{-6}$, best-found codes; training data = centred bank
  coefficients of 512 training-seed trajectories × 21 frames, cases $\equiv 7 \pmod 8$ held out
  as validation; encoder = nearest stored code + 12 variable-projection sweeps
  (`head_rom.init_head/train_head/best_codes`, head seed 20260923 as the parent).
- Truth and FOM: CNAB2 pseudo-spectral, 2/3 dealiasing, truth $\Delta t=0.001$, $T=0.2$, six
  output times; CNAB2 comparator ladder {200,100,80,70,60,50,40,20,10} steps.
- Error metric: per case, relative $L^2$ against $\lVert u_0\rVert$; **evolved worst** = worst
  over output times after $t=0$, then worst over cases.

Parent development numbers this lane must be compared with (from the parent jobs, not
re-measured here unless stated): POD-64 oracle-shift floor 0.125 % / 0.129 % / 0.129 % at
32/64/96; paper head $k=8$ evolved worst 0.151 % / 0.152 % / 0.153 % at 32/64/96 (jobs 4198840,
4198090, 4198101); 96³ head at 6.67× CNAB2.

## 3. The coordinate-network bank

$$g_\phi:\mathbb T^3\to\mathbb R^{3\times R},\qquad g_r(x)=\nabla\times\psi_r(x),\qquad
\psi_\phi(x)=\mathrm{MLP}\big([\sin 2\pi k\cdot x,\ \cos 2\pi k\cdot x]_{k\in\mathcal K}\big)\in\mathbb R^{3\times R}.$$

- **Periodicity.** $\mathcal K$ = integer wave vectors with $\lVert k\rVert_\infty\le k_{\max}$
  (one of each $\pm$ pair). Every feature is 1-periodic, so $g_\phi$ is periodic by
  construction; no boundary factor. The MLP uses SiLU.
- **Divergence-free: vector potential + curl (chosen).** $\nabla\cdot\nabla\times\psi=0$
  exactly in the continuum, for every column and every $\phi$, so the bank carries no gradient
  part that the solenoidal tests would not see. Rejected alternative: a direct MLP output
  Leray-projected per mesh. That is also mesh-consistent for band-limited columns, but its
  gradient part is invisible to the training loss only if the projection is inside the loss, and
  then the bank has no closed-form derivatives. With the curl, derivatives are analytic (autodiff)
  and the per-mesh projection below is a near-identity clean-up whose size is reported.
- **Sampling at mesh $n$.** $G^{\rm raw}_n = n^{-3/2}\,[g_\phi(x_i)]_{i}$ at the $n^3$ grid nodes,
  in the $(3,n,n,n)$ field layout. The $n^{-3/2}$ makes the Euclidean Gram the periodic
  trapezoid rule for the $L^2(\mathbb T^3)$ Gram, which is exact for trigonometric polynomials
  below the Nyquist frequency — so a band-limited bank has the same Gram on every mesh.
- **The bank at mesh $n$ (used by everything downstream):**
  $$G_n=\mathrm{L\ddot owdin}\big(P_n\,G^{\rm raw}_n\,T\big),$$
  $P_n$ = the FOM's discrete space (2/3 mask + spectral Leray, `ns3d_fom.project_field`), $T$ the
  frozen ordering (below), Löwdin $=B(B^{\mathsf T}B)^{-1/2}$, the closest orthonormal matrix
  with the same column order (needed because the encoder is an orthogonal projection and the
  operator build asserts orthonormality to $10^{-10}$). Reported per mesh, as the measure of
  "one bank": the energy fraction $P_n$ removes, $\lVert B^{\mathsf T}B-I\rVert_{\max}$ before
  Löwdin, and the relative change Löwdin makes. A pre-Löwdin deviation above 0.05 aborts the
  job; above $10^{-3}$ it is flagged against bar (c).
- **Derivatives.** $D_d$ is built spectrally from the sampled $G_n$ (as for the POD bank). It is
  checked against autodiff: $\Phi_{\rm sub}^{\mathsf T}(\partial_d g_\phi\,T S_n)$ by forward-mode
  JVPs of the curl at the grid nodes ($S_n$ the Löwdin factor, $\Phi_{\rm sub}$ 8 random tests;
  $\Phi^{\mathsf T}P_n=\Phi^{\mathsf T}$ because the tests are solenoidal and inside the mask).
  The bank job also compares autodiff and spectral derivatives of the raw sampled columns at
  32³ and 64³, and the autodiff divergence at random points. Disagreement is limited only by
  aliasing, so these also measure how band-limited the bank is. Reported; flagged above
  $10^{-4}$.
- **Cross-mesh consistency.** The bank job restricts $G_{64}$ and $G_{96}$ spectrally to $32^3$
  and compares them with $G_{32}$ column by column (reported).

### Training (stage 1, one allocation per bank)

- **Data, training seed 202609201 only, pooled across meshes.** 512 trajectories; trajectory
  $c$ is generated at mesh $\{32,64,96\}[c \bmod 3]$ with the truth CNAB2, six frames each
  (3072 states). Each frame is centred on its own energy centroid at its own mesh, then
  spectrally resampled onto a $48^3$ training grid (32³ data zero-padded; 64³/96³ data truncated
  at $|k_i|\le 23$, the dropped energy reported). Weight $w_i=1/\overline{u_0^2}$ of the state's
  trajectory (the error metric's normalisation).
- **Objective** (variable projection, i.e. the coefficients are eliminated exactly):
  $$\mathcal L(\phi)=1-\frac{\lVert P_{G(\phi)}Y\rVert_F^2}{\lVert Y\rVert_F^2},$$
  $P_G$ the orthogonal projector on $\mathrm{span}\,G^{\rm raw}_{48}(\phi)$, and
  $Y=U_KS_K$ the weighted snapshot matrix compressed to its leading $K=512$ singular pairs. This
  equals the weighted mean squared projection error of the 3072 training states up to the
  truncated tail, which is reported. Its minimum over all rank-$R$ subspaces is the weighted POD
  of the same data (Eckart–Young); that optimum is reported beside the trained value and its
  floors are measured too ("same-data POD"), to separate data choice from network
  representation.
- **Optimiser.** Full batch on all $48^3$ nodes, f64, Adam with global-norm clip 1, warm-up 1000
  steps then cosine $10^{-3}\to10^{-6}$. The number of steps is fixed from a measured step time
  and a wall budget of 5 h (18 000 s), so the schedule always completes; the calibration and
  step count are recorded.
- **Architectures and ranks** (one bank per job):

| job | $R$ | arch | width × depth | $k_{\max}$ | data |
|---|---:|---|---|---:|---|
| `bank_r64a` | 64 | A | 512 × 4 | 4 | centred |
| `bank_r64b` | 64 | B | 768 × 3 | 6 | centred |
| `bank_r128` | 128 | A | 512 × 4 | 4 | centred |
| `bank_r256` | 256 | A | 512 × 4 | 4 | centred |
| `bank_ff64` | 64 | A (1.5 h) | 512 × 4 | 4 | **uncentred** (control) |

  At $R=64$, A vs B is chosen by **final training loss only** (no development data).
- **Ordering** (paper appendix "Ordering the bank"), training states only, on the training
  grid: $B=P_{48}G^{\rm raw}_{48}=Q_GR_G$; rows $R_Ga_i/\lVert u_i\rVert=Q_G^{\mathsf T}u_i/\lVert u_i\rVert$
  stacked into $\mathcal A=USV^{\mathsf T}$; $T=R_G^{-1}V$. The bank file stores $\phi$ and $T$;
  nothing else is ever fitted to it.
- **Floors in the bank job**, at 32³/64³/96³ on the 16 development cases (seed 202609202),
  oracle shift (truth centroid per frame, as the parent's `oracle_shift_errors`): the bank at
  $R$ and its ordered prefixes $R'\in\{8,16,32,48,64,128,192\}\cap[1,R]$; the same-data POD at
  $R$; the no-centring projection of the same bank (control); and, in `bank_r64a` only, the
  parent's POD-64 rebuilt by the parent recipe at each mesh (must reproduce the parent's floors).

## 4. ROM per mesh (stage 2, one allocation per mesh, development)

The selected bank (rule in §6) is frozen (file + sha256) and every mesh job samples it. In each job:

1. **POD reference arm = the paper's model**: the parent's POD-64 bank rebuilt and gated
   against the stored probe ($\le10^{-8}$), with the **paper's frozen $k=8$ head for that mesh**
   (`ns3d-operators/frozen/h32`, `h64`, `ns3d-shift-head/frozen` — byte-identical to the heads
   of jobs 4198840/4198090/4198101). Its per-case development errors, and those of the parent's
   linear arm, must reproduce the parent jobs to $\le10^{-8}$ absolute (**reproduction gate**).
2. **Coordnet arms**: $G_n$ from the frozen file; operators validated; head $k=8$ trained on
   $G_n^{\mathsf T}$(centred states) of the 512 × 21 training states at this mesh with the
   recipe of §2 (the bank is shared; the head is per mesh, exactly as the paper's POD rows); the
   span ladder $R'\in\{R,\dots,64,48,32,16,8\}$ of the ordered bank. All at the ladder setting
   ($\Delta t=0.02$, 3 sweeps) — **nothing is tuned for the coordnet bank**.
3. **Controls**: coordnet head with the frame frozen ($\delta\equiv0$) — must fail; coordnet head
   fixed-sweep driver vs the generic LM reference on 4 cases ($\le10^{-6}$).
4. **CNAB2 ladder**; fields of every reduced arm (f64) and CNAB2 row (f32) saved for the audit.
5. **Timing**: GPU burn-in; sentinels (coordnet head, POD head, widest coordnet span) solo;
   every arm and every CNAB2 setting in one randomised interleaved block, 7 repetitions after 2
   burn-in calls, synchronised; sentinels solo again. Gate: each sentinel's in-block and end
   medians within $[1/1.10,\,1.10]$ of its first solo median. Timed outputs must reproduce the
   accuracy pass's errors.
6. **Audit**: `verify_coordnet.py` (NumPy only, separate process) recomputes every reported
   error from the saved fields and must first reject a perturbed copy (must-fail control); then
   the fields are deleted.

FD-CG is not run (not needed for the bars; the parent's CNAB2 comparison is the efficiency
comparison).

## 5. Pre-registered bars (development cases only)

- **(a) Floor.** The selected $R=64$ coordnet bank's oracle-shift evolved-worst floor at each
  mesh, against POD-64 at the same mesh (bank job `bank_r64a`; mesh jobs recompute both).
  PASS if coordnet ≤ 1.5 × POD-64 at every mesh. Also reported: the smallest trained $R$ (64,
  128, 256, dedicated banks) whose floor is ≤ the POD-64 floor at every mesh ("$R$ needed to
  match"; "> 256" if none).
- **(b) Head rollout at 96³.** Coordnet head $k=8$ development evolved worst ≤ 0.25 % (paper POD
  head 0.153 %), at ≥ 5× CNAB2, where the speedup is the fastest stable CNAB2 setting (all
  fields finite, evolved worst ≤ 100 %) whose evolved worst is no larger than the arm's, divided
  by the arm's median, both from that job's interleaved block.
- **(c) Flat across meshes from ONE frozen bank.** The same bank file (sha256 checked in every
  job); $\max/\min$ over {32³, 64³, 96³} of the coordnet head's development evolved worst ≤ 1.25,
  and the same ratio for the bank floor ≤ 1.25; pre-Löwdin orthonormality deviation ≤ $10^{-3}$
  at every mesh.

## 6. Decisions fixed now

- **Selection of the ROM bank.** Among the dedicated banks (64 → the A/B pick by training loss,
  128, 256) and their ordered prefixes, the one with the **fewest columns** whose development
  floor is ≤ 0.20 % at all three meshes (ties: the dedicated bank). If none reaches 0.20 % but
  one reaches 0.25 %, the smallest such, flagged as unlikely to meet (b).
- **Stop rule.** If the selected $R=64$ floor is > 3 × POD-64 at 96³ **and** no trained bank or
  prefix reaches ≤ 0.25 % at all meshes, the lane stops after stage 1: (a) is reported as a
  negative result, (b) and (c) are reported as not reachable (the head cannot beat its bank's
  floor), and no mesh or test job is run.
- **Fixed-frame control must fail on real data.** `bank_ff64` (trained on uncentred states, same
  recipe) projected without any shift must have an evolved-worst floor > 5 % at every mesh (by
  Eckart–Young it cannot beat the uncentred POD-64 floor of 53.9 % at 32³ measured by ns3d-grok
  diag01). The no-centring projection of the centred bank and the frame-frozen head rollout must
  also exceed 5 %. If any of them passes, the setup is wrong and nothing is reported as a result
  until it is explained.
- **Seeds.** Training 202609201; development 202609202 (16 cases) for every choice. Seeds
  202609203 and 202609211 are never read (only their parameter rows, for the disjointness check).
  The **test cohort** is the t2-ns3d-test cohort, seed 202609221, 32 cases, opened **once** by one
  job that evaluates all three meshes with every setting frozen in
  `frozen/frozen_settings.json` (bank file sha256, the three per-mesh coordnet heads from the
  development jobs, ladder setting, span list), committed before submission. The test job also
  reruns the paper's frozen POD heads and gates them against the errors previously recorded on
  that cohort (ns3d-test t32a/t64a via t2-ns3d-test configs; shift-head b2_heldout96), and
  checks rounded-row disjointness against training (512), development (16), 202609203 (32) and
  202609211 (32).

## 7. Verification

- Codex audit of this file and the code before any GPU job; of the final report after.
  Outputs under `results/`.
- Every gate and control is checked on a real mesh before any verdict is read.
- In-job NumPy audit (above) plus a local re-run of `verify_coordnet.py` is not possible after
  the fields are deleted; the in-job one is the audit of record, and the bank-job floors are
  recomputed locally from the pulled bank file at 32³ (`results/local_floor_check.json`).
- The report `reports/2026-10-01-ns3d-coordnet-bank.md` is generated from the run JSONs by a
  script; no number is typed by hand.

## Glossary

- **bank $G$** — the fixed spatial basis the reduced state lives in ($3n^3\times R$ at mesh $n$).
- **coordinate network / coordnet** — a neural network evaluated at a point $x$; here it returns
  $R$ vector fields at once, so sampling it on any grid gives a bank.
- **vector potential, curl** — the network outputs $\psi$; the bank is $\nabla\times\psi$, which
  is divergence-free.
- **POD** — proper orthogonal decomposition: the best rank-$R$ linear basis for a set of
  snapshots in the mean-square sense.
- **centred / co-moving frame / $\delta$** — each state is shifted so its energy centroid is at
  the origin; the ROM solves the shift increment $\delta$ every step.
- **oracle-shift floor** — the error of the best reconstruction of a true state in the bank
  after centring it with its true centroid; no solve can beat it.
- **ordering $T$, prefix $R'$** — a rotation of the bank so its columns are sorted by how much
  training data they carry; $R'$ = number of leading columns used.
- **Löwdin** — the symmetric orthonormalisation $B(B^{\mathsf T}B)^{-1/2}$.
- **head $h$, $k$** — small network mapping a $k$-number code to bank coefficients.
- **span arm** — the linear reduced model in the first $R'$ ordered columns.
- **evolved worst / median** — per case, worst relative $L^2$ error (vs $\lVert u_0\rVert$) over
  output times after $t=0$; then worst / median over cases.
- **CNAB2** — the pseudo-spectral full-order solver (Crank–Nicolson viscous, Adams–Bashforth-2
  advection), used for the truth ($\Delta t=0.001$) and as the speed comparator.
- **development / test** — cases used for every choice / cases evaluated once with everything
  frozen.
