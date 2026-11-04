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

## A1 (2026-11-03, before any GPU job) — changes after the two independent reviews

Reviews: `reviews/grok-design-review.md` (Grok 4.7 xhigh via cursor-agent, read-only) and
`reviews/claude-design-review.md` (a fresh Claude subagent), both on commit `2e8506d79`, both
given only the design and the code. They replace the Codex audit, which is quota-blocked until
2026-11-04 16:37 (coordinator instruction). **A Codex re-audit is owed.** Neither review found a
mathematical error in the objectives, the POD targets, the interpolant, the VP-excess bound or
the gate formulas. No gate threshold changes. All changes below were made before any job.

1. **L-BFGS line search** (Grok #5). The zoom line search now uses
   `initial_guess_strategy="one"`, as `optax.lbfgs`'s own default does. The code previously used
   the SGD setting `"keep"`.
2. **L-BFGS reverts** (Claude #7, #8). A call that ends higher than it started now reverts to
   the best iterate evaluated inside that call, and then the stage ends, instead of discarding
   the whole call. A rise of at most $10^{-15}$ absolute counts as a round-off stall, and the
   previous iterate is kept. The stall test is now a decrease $\le\max(10^{-10}\cdot\text{value},10^{-15})$
   over 8 consecutive calls. An exception inside the stage keeps the last accepted iterate and
   is flagged.
3. **Reference floors first** (Claude #10). The development cases, the parent POD-64 floors
   with their hard reproduction gate, and the same-data POD floors are computed at all three
   meshes **before training**. A failure exits with status 3 within minutes. A missing
   reproduction reference is a hard failure (Claude #13).
4. **Order of measurement** (Claude #12). After training, the gate inputs of the bank of record
   are measured and written first ($R=64$ floor and pre-Löwdin deviation at every mesh), then
   the gates. After that come the descriptive extras: prefixes, the no-centring control,
   cross-mesh columns, derivative checks, and the Adam-stage diagnostic. A diagnostic failure
   cannot kill the job.
5. **Recovery mode** `--floors-only <dir>`. If a job dies after training, the saved
   `bank_final.npz` is re-evaluated by the same code without retraining. If that file is
   missing, the order of fallback is `bank_lbfgs_partial.npz`, then `bank_adam.npz`. The
   result records the file used and its sha256. A recovery run is not a rerun of training.
6. **No verdict on a hard failure** (Claude #13). If any hard validity gate fails, `gates.verdict`
   is `null`; otherwise it is `pass` or `fail`. `make_rom_configs.py` requires `verdict=="pass"`.
7. **Off-grid term in chunks** (Grok #11). The arm-(b) off-grid value and gradient term is
   evaluated in chunks of 2048 points (`lax.map`). It equals the unchunked value to round-off
   (checked locally). The Claude reviewer's XLA memory analysis gives about 33 GB of
   temporaries for arm (b) and 28 GB for arm (a) at full scale.
8. **Pinned launch** (Grok #10, Claude #11). `cluster/launch.sh` holds the exact submissions:
   `gpu:1`, `--constraint=a100-80G` (default in `submit.sh`; an explicit override
   `GPU_CONSTRAINT=h100-80G` only after a preflight failure), memory 160G, wall 6 h 40 min for
   the bank jobs; ROM 0 h 45 / 0 h 20 / 0 h 15 at $96^3/64^3/32^3$ (the parent's measured elapsed
   times were 17 / 6 / 3 min).
9. **Provenance** (Claude #16). `submit.sh` refuses to stage an `experiments/` tree with
   uncommitted changes. `summary.json` also hashes `diag_floor.py`, `ns3d_fom.py` and the config.
10. **Timeout manifest** (Claude #17). `job.sbatch` traps the Slurm `TERM` warning
    (`--signal=B:TERM@300`) and still writes `OUTPUTS.sha256`. Bank mode prints
    `VERIFY_EXIT=NA`.
11. **G3 interpretability** (Claude #15; reporting only, G3 unchanged). The target spectral
    content now includes the cut $\max_i|k_i|>15$, which is outside $P_{48}$. At every mesh the
    G3 statistic is also computed for the same-data POD itself (resampled, $P_n$, its own $T$),
    as a control.
12. **Stop rule and budget wording** (Claude #19).
    - An arm whose job fails a hard gate counts as **not passing** and is not rerun. The
      exceptions are failures before training starts (GPU preflight, node or infrastructure
      faults), which may be resubmitted once, charged to the budget.
    - For the 15 GPU-hour rule, running jobs count at their **wall limit** until `sacct`
      reports final elapsed times.
    - At most 2 allocations at once, **including the ROM jobs**. The ROM jobs are submitted in
      the order $96^3$, $64^3$, $32^3$.
    - Worst case: $2\times6.67+0.75+0.33+0.25=14.67$ h, within the cap.
13. **Text** (Claude #21). The target spectral content is recorded once, on the $48^3$ grid
    (not "at every mesh"). The VP ratio to the optimum is reported for the raw bank; the
    projected value is reported beside it.

## A2 (2026-11-04, after `bank_a`/`bank_b`, before any further job) — a G3-aware continuation of arm (a)

**User authorization of 2026-11-04 (via the coordinator).** The 15 GPU-hour cap is lifted.
Continue until a bank passes the pre-registered gates or there is a well-evidenced diagnosis of
why the architecture cannot. The gates are **unchanged**: G1 ≤1.5× POD-64 at every mesh, G2
≤1.25, G3 ≤1e-3 at 64³/96³ as amended. Every attempt gets an amendment before its job, and every
failure is kept (`LEDGER.md`). Test and closed seeds stay closed. At most 2 allocations at once.

**Evidence so far** (jobs 6846165/6846174, development floors only; `runs/bank_*/output/summary.json`).
- Arm (a): floor 1.35/1.44/1.44× POD-64 (G1 pass); max/min 1.105 (G2 pass); pre-Löwdin
  6.7e-3/6.8e-3 at 64³/96³ (**G3 fail**). VP objective 1.47× its optimum; L-BFGS stopped at its
  deadline while still decreasing.
- Arm (b): G1 fail (1.75/1.87/1.87×), G2 pass, G3 fail (9.6e-3/9.7e-3).
- The G3 control (§A1 item 11) reads the same-data POD the same way. It gives 9.1e-3 at 64³/96³
  and 0.607 at 32³. Its modes carry up to 0.80 % of their energy at $\max_i|k_i|>15$, which
  $P_{48}$, and hence the frame $T$, discards but $P_{64}$/$P_{96}$ keep. G3 therefore cannot be
  met by fitting the data alone. A bank passes G3 only if its ordered directions carry almost no
  energy outside the $P_{48}$ space at the finer meshes, and arm (a)'s band penalty (relative to
  total energy) does not control that per direction: the raw bank's singular-value ratio is
  8.4e-6, so weak directions amplified by $T$ keep their residue.

**Change (objective, therefore independently reviewed before the jobs).** Warm start from
`bank_a`'s bank of record (`runs/bank_a/output/bank_final.npz`, sha256 `ef37ea38…`), same
architecture A and $R=64$, no Adam stage. L-BFGS (as in §3/§A1) on
$$\mathcal L_{a2}=\mathcal L_a+\mu\,\lVert B_{64}^{\mathsf T}B_{64}-I\rVert_F^2,\qquad
B_{64}=P_{64}\,G^{\rm raw}_{64}\,R_{48}^{-1},$$
where $P_{48}G^{\rm raw}_{48}=Q_{48}R_{48}$ (QR). This is exactly the G3 matrix at 64³ up to the
ordering rotation $V$, which leaves the Frobenius norm unchanged. It is computed from the bank
and the grids only, with no development data, using the same $P_n$ (`ns3d_fom.project`); its
value on the `bank_a` bank matched the pipeline (Frobenius² 3.9e-4; max entry 7.3e-3 in this
frame, against the pipeline's 6.7e-3 in the ordered frame). The 64³ network samples are
rematerialised per 32 768-point chunk to bound memory.

- **Two jobs, concurrent, one GPU each:** `bank_a2_mu1` ($\mu=1$) and `bank_a2_mu001`
  ($\mu=10^{-2}$).
  - Both put the penalty well above the current VP excess (≈1.7e-7) until the statistic is
    small; $\mu=1$ favours G3 more strongly, at a possible cost in G1.
  - L-BFGS budget 21 600 s (6 h); wall limit 8 h. Everything else as in §3/§A1: floors, gates,
    controls, recovery mode.
- **Bank of record:** the end of each job's L-BFGS stage.
  - If exactly one passes G1–G3, it goes to the ROM (§6).
  - If both pass, the one with the lower $\max_n$ floor/POD-64 goes, ties to $\mu=10^{-2}$.
  - If neither passes, the next attempt is decided from the evidence and amended before any job.
- G3 is still evaluated by the unchanged pipeline (`mesh_bank`) at 64³ and 96³. Penalising 64³
  does not guarantee 96³; the report will say so if they differ.

## A3 (2026-11-04, after `dev32` 6858232 failed in 33 s, before its rerun) — the inherited 32³ sanity abort

`rom_mesh.py` (unchanged from the parent) aborts when the sampled bank's pre-Löwdin deviation
exceeds `one_bank_orthonormality_gate`. The parent configs set that to 0.05. This was a parent
sanity guard, not a gate of this lane: §4 deliberately leaves 32³ ungated.

The selected bank reads 0.607 at 32³, and the same-data POD control reads 0.607 too
(`runs/bank_a2_mu001`). This is inherent to the data, not a defect of the network: up to 53 % of
a POD mode's energy lies at $\max_i|k_i|>10$, which the 32³ FOM space cannot hold. At 32³ the
Löwdin clean-up changes the bank by 9.8 % overall and up to 76 % in its worst column. The
oracle floor at 32³ is still 1.325× POD-64, passing G1.

So `dev32` died before producing any number. Its logs are kept in `runs/failed_preflight_dev32/`,
and it is charged in `LEDGER.md`.

**Change.** `dev32b` reruns the 32³ development ROM with `configs/dev32b.json` = `configs/dev32.json`
with only `one_bank_orthonormality_gate` raised to 1.0. Everything else is unchanged: same bank,
head recipe, cohort, controls, timing and audit. The 0.607 is reported beside the result.

No gate, objective or architecture changes, so no new review is required (authorization rule 6).
The R1–R4 readings at 32³ are reported with this disclosure. 64³ and 96³ ran under the
parent's 0.05 guard and passed it (7.0e-5, 3.1e-4).
