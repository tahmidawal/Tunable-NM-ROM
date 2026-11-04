# NS3D: can a properly trained coordinate-network bank reach the POD-64 floor? (pre-registration)

Written and committed before any GPU job of this lane. Amendments are appended as §A1, §A2, …;
nothing above them is edited afterwards. No number here is a measurement of this lane; every
number quoted comes from the parent lane `exp/2026-10-01-ns3d-coordnet-bank` and is labelled.

Branch `exp/2026-11-03-ns3d-pod-match` (sparse worktree `worktrees/2026-11-03-ns3d-pod-match`),
forked from the current fork point `93f399faa` (`exp/2026-10-20-jcp-consolidated`, which contains
the parent lane's code). It is never pushed directly; `sync_github.sh` mirrors it code-only to
`origin/codeonly/exp/2026-11-03-ns3d-pod-match`. Cluster namespace
`/cluster/tufts/paralab/tawal01/ns3d-pod-match-20261103/<job>/`: one directory per job, never reused.

## 1. Question and why it is open

The parent lane froze a 30-minute pilot bank (curl of a periodic integer-Fourier-feature MLP,
$R=64$, trained by the variable-projection (VP) objective, no band-limit penalty). Its development
oracle-centroid floor is 1.004 / 1.243 / 1.257 % at $32^3/64^3/96^3$, against POD-64 0.125 / 0.129 /
0.129 % (parent jobs). Its training loss was $5.19\times10^{-5}$, against $3.73\times10^{-7}$ for the
optimum of the same objective (the weighted POD of the same data): about 140× short. The head
($k=8$) error was 1.01–1.02× its bank floor at every mesh, so the bank is the whole bottleneck. With the POD bank, the same
head, frame and solver give 0.15 % worst. The planned longer, band-limited and larger-$R$ fits
(parent §A3) were cancelled before evaluation (parent §A4).

**Question.** Within the same network family, does a bank trained toward convergence reach a
projection floor within 1.5× of POD-64 at every mesh while reading the same on every mesh? If
it does, does the unchanged head/frame/solver then give a ROM close to the POD model's accuracy?

## 2. Inherited unchanged

Imported from the parent lanes, not copied (`experiments/ns3d-coordnet/coordnet.py`, `bankio.py`,
`rom_mesh.py`, `verify_coordnet.py`; `ns3d-shift`, `ns3d-shift-head`, `ns3d-grok/diag_floor.py`,
`ns3d/ns3d_fom.py`):

- **Network family.** $g_\phi=\nabla\times\psi_\phi$, $\psi_\phi=\mathrm{MLP}([\sin 2\pi k\cdot x,\cos 2\pi k\cdot x]_{k\in\mathcal K})$, integer $\mathcal K$ with $\lVert k\rVert_\infty\le k_{\max}$; periodic and
  divergence-free by construction. **Both arms use the pilot's architecture A** (width 512,
  depth 4, SiLU, $k_{\max}=4$, about 1.26 M parameters). The frequencies are never optimised:
  the optimisers see only the layer weights.
- **Training data** (train seed 202609201 only): 512 trajectories, trajectory $c$ at mesh
  $\{32,64,96\}[c\bmod 3]$, truth CNAB2, six frames, each centred on its own energy centroid,
  spectrally resampled to the $48^3$ training grid; trajectory weight $1/\overline{u_0^2}$. The
  weighted snapshot matrix is compressed to its leading $K=512$ singular pairs $Y=U_KS_K$.
- **Bank at mesh $n$**: $G_n=\mathrm{L\ddot owdin}(P_n\,G^{\rm raw}_n\,T)$, $T$ = the paper's QR+SVD
  ordering on the training grid (training states only).
- **Floors**: oracle-centroid projection floor, evolved worst over the 16 development cases (seed
  202609202), `diag_floor.oracle_shift_errors`; POD-64 = the parent recipe (128 training cases × 6
  frames at mesh $n$), rebuilt in-job and gated against the parent records (relative gap
  $\le10^{-6}$: 0.12472927 % / 0.12891179 % / 0.12890898 %).
- **ROM** (only if a bank passes, §6): `rom_mesh.py` unchanged — $k=8$ head per mesh trained with
  the shift-head recipe, ladder setting ($\Delta t=0.02$, 3 Gauss–Newton sweeps), span ladder,
  the paper's frozen POD heads as the reference arm with the reproduction gate, frame-zero control,
  LM parity, CNAB2 ladder and interleaved timing with the neighbour gate, in-job NumPy audit
  (`verify_coordnet.py`).

## 3. Arms (one bank job each, one GPU each)

Both arms: $R=64$, architecture A, init seed 20261001 (the pilot's), f64,
`JAX_DEFAULT_MATMUL_PRECISION=highest`, full batch on all $48^3$ nodes, and the same **staged
optimiser**:

1. **Adam**: global-norm clip 1, warm-up 1000 steps then cosine $10^{-3}\to10^{-6}$; the step count is
   fixed in-job from a measured step time and a 10 800 s (3 h) budget, so the schedule completes.
2. **L-BFGS** (`optax.lbfgs`, memory 20, zoom line search up to 30 steps) on the **same
   objective**, from the Adam endpoint, until a 6 000 s deadline. It also stops early if the
   objective changes by less than $10^{-10}$ relative over 8 consecutive calls of 25 iterations. A
   call that ends nonfinite or higher than it started is reverted, and the stage ends.

The bank of record is **the end of the L-BFGS stage** (no selection between stages). The
Adam-stage bank's $R=64$ floors are measured as a diagnostic only.

**(a) A3 recipe as designed.** Objective (parent §A3):
$$\mathcal L_a(\phi)=1-\frac{\lVert P_{G}Y\rVert_F^2}{\lVert Y\rVert_F^2}+\lambda\,\frac{\lVert G-\Pi_{10}G\rVert_F^2}{\lVert G\rVert_F^2},\qquad\lambda=1,$$
$\Pi_{10}$ = keep Fourier modes with $\max_i|k_i|\le10$ on the $48^3$ grid (the 2/3 space of $32^3$).

**(b) POD-guided fit.** The lesson from the 2D cylinder campaign: fitting raw snapshots failed,
while fitting POD modes and their gradients worked. Targets are the weighted POD of **the same
pooled training data**: $U_R$, the leading 64 left singular vectors of the weighted snapshot
matrix, i.e. the optimum of the arm-(a)/parent VP objective. They are re-orthonormalised by QR
with signs kept, which does not change any leading prefix. Column $r$ of the network is regressed
onto mode $r$, which keeps the importance ordering, with energy weights
$w_r=s_r^2/\sum_{j\le64}s_j^2$:
$$\mathcal L_b=\sum_r w_r\lVert G_r-U_r\rVert^2
+\frac{\sum_r w_r\lVert g_r(Q)-f_r(Q)\rVert^2}{\sum_r w_r\lVert f_r(Q)\rVert^2}
+\frac{\sum_r w_r\lVert\nabla g_r(Q)-\nabla f_r(Q)\rVert^2}{\sum_r w_r\lVert\nabla f_r(Q)\rVert^2}.$$
- $G$ is the network sampled on the $48^3$ nodes. $f_r$ is the exact trigonometric interpolant of
  mode $r$; the modes are trigonometric polynomials on the training grid with no Nyquist-plane
  energy, which is checked as a hard gate. $Q$ is 8192 fixed uniform random off-grid points
  (seed 20261103). $\nabla g$ is the network's autodiff gradient; $\nabla f$ is spectral.
- Hard gates before training: the interpolant reproduces the modes at 256 grid nodes to
  $\le10^{-10}$ relative, and its gradient equals the FFT derivative there to $\le10^{-9}$.
- Why these weights: since $Y=U_KS_K$, for any bank, $\mathcal L_{\rm VP}\le \text{optimum}+c\,\sum_r w_r\lVert G_r-U_r\rVert^2$
  with $c=\sum_{r\le64}s_r^2/\sum_{r\le K}s_r^2\le1$. The first term is therefore an upper bound on
  the VP excess. The off-grid value and gradient terms constrain the network between the
  training nodes, which is what makes a bank read the same on every mesh.
- No band penalty is used: the targets carry the data's own content above $|k|=10$ (reported).

**(c) R = 128: not run.** The two arms are budgeted at about 12 of the 15 GPU-hours (§7),
leaving room only for the ROM jobs and one preflight resubmission. Arm (c) is recorded as
*not run (budget)* and is not added later in this lane.

**Recorded for every stage of both arms:** the training objective and its parts; the exact
VP objective (NumPy SVD, no Gram, no ridge) of the raw and the $P_{48}$-projected bank, beside the
Eckart–Young optimum and their ratio; the bank's energy fraction above $|k|_\infty=10$; for arm (b)
the per-column relative regression errors (grid, off-grid); the curves, step times, L-BFGS
iterations, line-search evaluations and stop reason. Also recorded at every mesh: the
spectral content of the POD targets, the derivative check (autodiff vs spectral, $32^3$ and
$64^3$), cross-mesh column consistency, the floors of the ordered prefixes $R'\in\{8,16,32,48\}$,
the no-centring control (must exceed 5 %), and the same-data POD floor.

## 4. Gates (pre-registered; development cases only)

For the bank of record of each arm, at $n\in\{32,64,96\}$:

- **G1 floor.** coordnet $R=64$ floor $\le 1.5\times$ the in-job POD-64 floor, **at every mesh**.
- **G2 flat across meshes.** $\max_n/\min_n$ of the coordnet $R=64$ floor $\le1.25$.
- **G3 one bank.** Pre-Löwdin deviation $\lVert B^{\mathsf T}B-I\rVert_{\max}\le10^{-3}$ at $64^3$ and
  $96^3$ ($B=P_nG^{\rm raw}_nT$). The $32^3$ value is reported separately and not gated: the parent
  showed that a $32^3$ grid cannot resolve detail that the finer grids keep.
- **A bank passes** if G1, G2 and G3 all pass. The driver computes the gates; the report
  generator recomputes them from `summary.json`.
- **Validity gates (hard):** POD-64 reproduced at every mesh; POD-target checks (arm b); the
  frequencies unchanged; no-centring control $>5$ %. A job failing a hard gate yields no verdict.

## 5. Selection and the stop rule

- **If neither arm passes:** stop. Record everything, run no ROM job, and do not keep tuning.
- **If exactly one passes:** that bank goes to §6.
- **If both pass:** the one with the lower $\max_n$ (floor / POD-64 floor) goes to §6, with ties
  going to arm (b). Only one bank goes to the ROM.
- Nothing is chosen from development ROM results; the ROM is evaluated once per mesh.

## 6. ROM (only for a passing bank)

The bank of record is frozen as `banks/<arm>/bank_selected.npz` (byte copy of the job's
`bank_final_ordered.npz`, sha256 in the configs). Three development jobs (`dev32`, `dev64`,
`dev96`) run `rom_mesh.py` **unchanged**, with configs identical to the parent's `dev32/64/96.json`
except for the bank file and its sha256. That covers: a head $k=8$ retrained per mesh with the
unchanged recipe (seed 20260923), the development cohort (16 cases, seed 202609202), the
frame-zero control (must exceed 5 %), LM parity, POD reference reproduction, the CNAB2
comparator ladder with interleaved timing, and the in-job audit. Reported, with these bars
fixed now:

- **R1 (POD-level accuracy):** coordnet head development evolved worst $\le1.5\times$ the paper POD
  head's in the same job, at every mesh.
- **R2 (head vs its bank):** head $\le1.5\times$ its own bank's $R=64$ floor at every mesh (the
  parent's bar b′).
- **R3 (speed):** at $96^3$ the head is $\ge5\times$ faster than its CNAB2 comparator, defined as
  in the parent.
- **R4 (flat):** head $\max/\min$ over meshes $\le1.25$.

**The test cohort (seed 202609221) and every other closed seed (202609203, 202609211) are not
opened.** The drivers refuse closed seeds. Opening a second test needs the user's decision.

## 7. Budget and jobs

Hard cap: **15 GPU-hours** in total, counted as allocated wall time × GPUs (from `sacct`),
failures included. At most 2 allocations at once, one GPU each. `gpu` partition only;
A100-80GB (`--constraint=a100-80G`; an H100-80G is acceptable after a preflight failure).

| job | what | wall limit | expected |
|---|---|---:|---:|
| `bank_a` | arm (a) | 6 h 40 min | ≈ 6 h |
| `bank_b` | arm (b) | 6 h 40 min | ≈ 6 h |
| `dev32/64/96` | ROM, only if a bank passes | 0.5 / 0.5 / 1 h | ≈ 0.6 h total |

The two bank jobs run concurrently. If the running total plus the next job's wall limit would
exceed 15 h, that job is not submitted.

## 8. Verification and records

- An independent review of this design and the code before any job. Codex is quota-blocked
  until 2026-11-04 16:37, so by coordinator instruction the review is done by a fresh Claude
  subagent and, by a later coordinator note, also by Grok (cursor-agent), each given only the design and the code. Full outputs are kept in `reviews/`,
  and a Codex re-audit is owed when the quota returns. The changes it causes are appended as §A1
  before any job.
- Each job writes `OUTPUTS.sha256`. `pull.sh` verifies the checksums, `jax_backend=gpu` and the
  exit records before deleting the remote directory. Bank parameter files (about 10 MB each)
  are committed.
- Report `reports/2026-11-03-ns3d-pod-match.md` on this branch, generated by
  `reports/build_2026-11-03-ns3d-pod-match.py` from the pulled JSON. A dated entry is appended
  to the canonical `LAB-LOG.md`.

## Glossary

- **bank $G$**: the fixed spatial basis ($3n^3\times R$) the reduced state lives in.
- **coordinate network / coordnet**: a network evaluated at a point $x$, returning $R$ vector
  fields, so sampling it on any grid gives a bank.
- **POD-64**: the best rank-64 linear basis (mean-square) of a snapshot set. The parent's
  per-mesh POD-64 is the reference; the **same-data POD** is that of this lane's pooled
  $48^3$ training data, the optimum of the VP objective.
- **VP objective**: the fraction of the (weighted, compressed) training data outside the bank's
  span, with the coefficients eliminated exactly.
- **Eckart–Young optimum**: the smallest VP objective any rank-$R$ subspace can reach.
- **band penalty**: the bank's energy above $|k|_\infty=10$ on the training grid.
- **oracle-centroid floor**: projection error of a true state after centring it on its true
  energy centroid; evolved worst = worst over output times $t>0$, then over cases.
- **Löwdin, pre-Löwdin deviation**: symmetric orthonormalisation; how far the sampled,
  projected, ordered bank is from orthonormal before it.
- **head $k=8$**: small network mapping 8 numbers to bank coefficients.
- **CNAB2**: the pseudo-spectral full-order solver and speed comparator.
- **development / test cohort**: cases used for checks here / cases kept closed.
- **L-BFGS**: a quasi-Newton optimiser using curvature from recent gradients.
