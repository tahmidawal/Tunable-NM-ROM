# burgers-compare-hires — DESIGN (pre-registered 2026-09-23, before the first GPU job)

Lane `exp/2026-09-23-burgers-compare-hires`, worktree `worktrees/2026-09-23-burgers-compare-hires`, forked from
`exp/2026-09-22-quadratic-manifold` @ `440d6332`. Cluster namespace `/cluster/tufts/paralab/tawal01/bcmp_20260923/`,
one directory per job, never reused. Budget: at most 2 of this lane's jobs running at once; 8 jobs total.
Numbers wanted by the evening of 2026-09-24 EDT (paper deadline 2026-09-25 AOE).

## 1. Question

The paper's "comparison with other methods" table for 2D viscous Burgers exists only at $256^2$, where the
full-order model (FOM) costs 20–31 ms and every reduced model, ours included, is slower than it. Our NM-ROM's
query cost is flat in the mesh (≈40 ms fast setting at $256^2$ and at $4096^2$) while the FOM's grows. This lane
re-runs the whole comparison at **$1024^2$ (priority)** and **$2048^2$ (stretch)**: in ONE allocation per mesh,
on the same six development cases, every method's worst / median evolved same-grid error, median GPU ms, and a
speedup against **the fastest tested FOM setting at least as accurate as that arm, from the same job**.

## 2. Fixed inputs (nothing is retrained on our side)

- Checkpoint `experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl`
  (SHA256 `18f0266a…`, $K=16$, $R=512$, trained at $256^2$). Directions `directions_qtd02.npz`. q=0 rule
  `rules-eqtop/rule_q0_m1024_qrg304_reachable.npz` (b-eqtop, transferred with weights × $(L/256)^2$, `scaled`).
- Cohort dev6 = `params_draw(7090702,4)` + `params_draw(911702,2)`, asserted against the b-panel hash
  `108f12dc…` (cluster only — the GB10 NumPy differs by 1 ulp in `exp`).
- Error: $\max_{k\ge1}\lVert u_{\rm arm}(t_k)-u_{\rm ref}(t_k)\rVert_2/\lVert u_0\rVert_2$ over the five evolved output
  times, reference `fft_tight` (Newton–BiCGStab, FFT-Helmholtz preconditioner, $\Delta t=0.005$, ntol $10^{-6}$,
  ltol $10^{-8}$) solved **in the same job at the same mesh**. Worst and median over the six cases.
- Timing scopes (unchanged from b-panel / repanel): GPU ms = input resident on the GPU → six output fields
  resident on the GPU, synchronised both sides; median pooled over (case × repetition), 5 retained repetitions
  per case. Complete ms (host upload + download) reported alongside.

## 3. Arms

| family | arms | notes |
|---|---|---|
| NM-ROM fast | `q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2` | optimised path; 16 unknowns |
| NM-ROM accurate | `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2` | optimised path (Cholesky, clip, damping carry-over, quadratic predictor, fused Jacobian); 272 unknowns. `lat64` j=0: certified (thin) at $1024^2$ by burgers-eqcert bc1024; at $2048^2$ confirmed in bc2048b (thin, 0.1156 vs 0.116) but **marginal** on other trajectories — labelled so, not re-certified here |
| NM-ROM accurate, robust rule | `q256_M1088_lat64_g0p01_fast_chol_clip_lamcarry_pred2_x1` | one exact first step; the robust certificate (bc1024 0.0358/0.0498; bc2048b 0.051). Secondary row. Slow (exact first step), so it runs in the slow phase |
| POD-LSPG | ranks 16, 64, 256, 512, $M=4k$, dense residual | POD of training snapshots at the target mesh |
| quadratic manifold (GWW / Barnett–Farhat) | $r=16,32,64$, $M=4r$, dense residual | same snapshots; ridge by **trajectory-split** holdout (`qman.fit` semantics; the column split was a caught leak and must not return) |
| neural operators | FNO (`fno-large` config), U-Net (`unet-refine`), Transolver (`tsol-refine`), DeepONet (`don-small`) | trained **at the target mesh** (§4); plus `fno-large-256` = the published $256^2$-trained FNO evaluated at the target mesh (FNO is discretisation-agnostic; zero-shot, labelled) |
| FOM grid | 7 audited (`nt*`, `fft_tight`) + 8 `lean_*` Newton–BiCGStab settings — the union repanel timed | every candidate, including the tight reference, is eligible for the rule |

**Unknowns** column: NM-ROM $K+q$; POD $k$; quadratic manifold $r$; FOM $(L-1)^2$; operators have no online
unknowns (parameter count reported instead).

### 3.1 POD / quadratic-manifold fitting

- Training snapshots: `params_draw(0, 128)` (the panel's `train_seed 0`, 128 trajectories), solved at the target
  mesh with `engines.make_fom(L, 0.005, None, .25, .005)` at ntol $10^{-9}$, ltol $10^{-7}$, every 2nd step
  (26 states per trajectory, 3328 snapshots) — exactly `ladder.generate_snapshots`, but written into one
  preallocated host array (the concatenation would double a 111 GB array at $2048^2$). Asserted disjoint from dev6.
- POD: the same uncentred method of snapshots as `ablation.pod_basis`; quadratic manifold: the same centred
  construction and ridge grid as `qman.fit`, **computed in row blocks** so the $n\times N_s$ snapshot matrix never
  has to sit on the device. Same algebra, different summation order: the streamed fit must reproduce
  `qman.fit` / `pod_basis` to $10^{-9}$ relative on a small mesh before it is used (checked locally, and
  recorded in `checks/`). Ridge grid $\{0,10^{-10},10^{-8},10^{-6},10^{-4},10^{-2},1\}$, 20 % of trajectories held
  out, seed 20260922 (qmn102's values).
- The dev6 cohort is never used for any fit or selection.

## 4. Operator training (jobs `t1024`, `t2048`)

- Data: the pinned 128-case training index (`5333584b…`) and 32-case validation index (`468b9e70…`) of the
  $256^2$ operator panel — same ids, seeds and descriptors, read from the index (never re-drawn) — regenerated
  **at the target mesh** with the panel's own reference setting `fft_tight` (`opdata.py`). So an operator is
  trained on exactly the field it is graded against.
- Configurations: the four validation-selected arms of the $256^2$ panel (ops-timing-panel opt201 roles), taken
  verbatim from their recorded configs. **Deviation, stated:** the capacity screen and the refinement step are not
  repeated at the target mesh; the selected configuration is carried over. The ops-tune-grid / ops-tune-deeponet
  tuned configurations do not exist yet (their jobs are still running), so none is used.
- Budget: the $256^2$ protocol's **equal wall budget, 3000 s per arm**, early stopping on validation. At
  $1024^2$ an epoch costs ~16× more than at $256^2$ (~64× at $2048^2$), so this budget buys far fewer epochs than
  the $256^2$ arms had (U-Net 1962, Transolver 1628 epochs). Epochs completed are reported next to every operator
  row; a budget-bound operator is under-trained **relative to its $256^2$ self**, and the report says so.
- Memory: the optimiser batch stays at 8; it is evaluated in micro-batches (gradient accumulation, halved only on
  a CUDA OOM, recorded) — same update up to summation order.
- f64 I/O everywhere; FNO network f64; U-Net / Transolver / DeepONet networks f32, as at $256^2$.

## 5. Panel job per mesh (`p1024`, `p2048`) — ONE allocation, phases in this order

1. **Truth**: dev6 at the mesh with `fft_tight`; cohort hash asserted.
2. **Phase F (fast panel)**: the two NM-ROM rows + all 15 FOM settings, quick run (fields saved), then a randomised,
   burn-separated (0.25 s), synchronised timed panel, 5 repetitions × 6 cases. **No arm slower than ~1 s is in
   this phase** — that is the fix for repanel's order-effect failure (a 68 s arm slowed its successors by up to 50 %).
3. **Phase S (slow arms)**: the snapshot fits, then POD-LSPG, quadratic-manifold and robust-rule NM-ROM arms, **one
   arm at a time**: build, quick run (fields saved), burn, 5 repetitions × 6 cases, release. OOM drops the arm and is
   reported (never silently).
4. **Phase B (bracket)**: every Phase F subject re-timed (2 repetitions × 6 cases, randomised). Drift gate below.
5. **Phase O (operators)**: separate PyTorch process in the same allocation, same GPU; 20 untimed burn-in queries
   then 5 × 6 timed, synchronised; GPU name gated equal to the JAX phase's.
6. NumPy audit of the saved fields (remote, then again locally after collection).

**Speedups** use Phase F's FOM medians for every arm (one comparator table per mesh). An arm with no tested FOM at
least as accurate says so instead of printing a ratio.

## 6. Gates (a failing gate is reported, never worked around)

| gate | bar |
|---|---|
| backend / precision | `jax_backend=gpu`, x64, `highest`; torch CUDA for Phase O |
| cohort | dev6 hash equals b-panel's |
| truth converged | every `fft_tight` step at ntol |
| repetition identical | every timed JAX repetition's full-field SHA256 equals its quick run's |
| ≥ 5 repetitions | every (arm, case) in Phases F, S, O |
| **no order effect (Phase F)** | per arm, samples normalised by the arm's per-case median; median of those after a slow predecessor vs after a fast one within **5 %**. Two splits: (a) predecessor ≥ 1 s (repanel's definition; vacuous if no such arm), (b) predecessor slower than the arm's own per-case median ×3. Arms with < 5 samples in either group are not judged. **Control:** the same code on repanel's br1024 invocations must FAIL (it did in repanel), checked before this gate is trusted |
| bracket drift | Phase B median within 10 % of Phase F median for every rule-selected FOM comparator and both NM-ROM rows |
| trajectory-split ridge | every quadratic-manifold fit records `split='by trajectory'` and a selected ridge inside the grid |
| streamed fit parity | streamed POD / QM fit reproduces `pod_basis` / `qman.fit` to $10^{-9}$ on the local check |
| operator integrity | $t_0$ returned bitwise; saved field hash = timed field hash; GPU name matches the JAX phase |
| independent audit | `audit_cmp.py` (NumPy only) recomputes every error from the saved full fields and matches the job to $10^{-9}$; **control:** a field perturbed by $10^{-3}$ relative must be flagged |
| checkpoint unchanged | decoder SHA256 before = after |

## 7. What is reported, and what would be a null

Per mesh: method, unknowns, worst %, median %, GPU ms, the FOM the rule chose, speedup — generated from the
summary JSON by a script. If a baseline beats us on any column it is printed as such. A mesh whose panel job fails
its gates is reported as failed, with the reason, not replaced by numbers from another job.

## 8. Jobs

| attempt | what | GPU | notes |
|---|---|---|---|
| `t1024` | operator data + training at $1024^2$ | A100-80G | ~4 h |
| `t2048` | operator data + training at $2048^2$ | A100-80G or H200 | ~5 h; if it fails or runs out of time, $2048^2$ has no operator rows and says so |
| `p1024` | panel at $1024^2$ | H200 preferred (A100-80G acceptable) | after `t1024` |
| `p2048` | panel at $2048^2$ | H200 | after `t2048` |

Retries get a new attempt name and directory.

## A1 (2026-09-23 ~02:40 EDT, before any panel job) — bank-span arms, by coordinator direction

The paper drops the correction directions $C_q$. Our model's settings become (a) the head only ($q=0$, the
"fast" arm already here) and (b) a **bank-span solve**: $u = G\,T_{:,:R'}\,a$ with $a\in\mathbb R^{R'}$ the LM
unknowns, $T$ the importance rotation of the frozen bank (SVD of $G\Sigma^{1/2}$, $\Sigma$ the second moment of the
training head coefficients — training codes only), taken verbatim from the sibling lane
`worktrees/2026-09-23-burgers-bank-knob/experiments/burgers-bank-knob/inputs/rotation_R512.npz`
(SHA256 `51149166…`, produced by its `make_rotation.py`; copied to `inputs/`, hash-checked in-job). Added to Phase S
of every mesh:

- $R'\in\{512,384,256,128\}$, $M=4R'$ weak tests, advection through the uniform lattice rules `lat64` and `lat128`,
  gtol $10^{-6}$; implemented as the ORIGINAL bank with the linear head $h(a)=T_{:,:R'}a$, so the rule stencils,
  test projection and `arms`' LM/initializer are unchanged (output decoded through the row-blocked bank).
- **Held-out ρ under truncation, re-measured in-job for every bank-span arm**: on the states its own query visits for
  8 held-out trajectories (`params_draw(20260921,56)` rows 0–7, burgers-eqcert's population source, asserted
  disjoint from dev6 and training), $\rho=\lVert P_q^\top a(\text{nodes})-\Phi^\top a\rVert/\lVert\Phi^\top a\rVert$ per
  state; any arm with $\rho_{\max}>0.116$ is **marked** in its row. ρ is never certified by an NNLS fit residual
  (there is no fit: lattice rules).
- The $q=256$ arms stay, labelled "reference only".

## A2 — independent pre-job audit (subagent; Codex cannot run on this box), dispositions

No blocker. Accepted and fixed before any panel job:
1. **Order gate** normalised by a per-case median that contained the slowed samples, so it could not see a slowdown on
   arms whose samples mostly follow slow arms (the NM-ROM fast arm, the fast FOM comparators). Replaced by a
   within-case paired estimator (median over cases of median log-time after slow minus after fast). **Control
   changed (deviation from §6, recorded here):** the pre-registered control "must fail on br1024" is withdrawn —
   br1024's recorded failure is a case-mix artefact (all after-slow samples sit inside their own case's timing range;
   paired gap on br1024 ≤ 1.2 %), so that dataset has no effect to detect. The control is now an injected +6 %
   slowdown on real rows (br1024, and this job's own Phase F rows in `audit_cmp.py`), which every judged role arm and
   rule-chosen FOM must individually fail (`checks/order-gate-control.json`: recovered gaps 5.8–6.6 %).
2. The operator cohort is written right after the truth solve and `cmp.py` failure no longer aborts the operator phase
   and the audit.
3. A per-arm deadline (`phase_s_deadline_seconds`): a slow arm whose projected 5×6 block would overrun is dropped
   and reported, never shortened.
4. A partially timed or dropped arm prints no GPU ms and no speedup.
5. Report prints rule status, operator training record (mesh, epochs, stop reason) and the zero-shot marker in a
   notes column; a mesh with failed gates carries a banner.
6. Minor: failed operator trainings become notes; grid-parity gate fails if no pair was produced; missing bracket
   timing fails the drift gate; the robust-rule arm is released after its block; panel collection streams with
   rsync (no second remote copy of tens of GB).
Not changed: complete-ms for operators excludes an upload (inherited; favours the operators; GPU-ms is the
headline column).

## A3 (2026-09-23 ~06:55 EDT) — 2048^2 operator retry runs concurrently with the panel job

`t2048` (4196041) trained U-Net (62 epochs) and Transolver (24 epochs) at $2048^2$; FNO and DeepONet ran out of CUDA
memory on its A100-80G at micro-batch 1 (≥ 8 GiB reserved-but-unallocated: fragmentation). Retry `t2048s` (FNO +
DeepONet only, same configs and 3000 s budgets, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; an H200 retry
`t2048r` sat unscheduled because every H200 was allocated and was cancelled before it ran). To save ~2 h, `p2048` is
submitted concurrently: after its JAX phases it waits (bounded) for `t2048s`, re-hashes each checkpoint against the
SHA256 the training job itself recorded, and times it in Phase O like every other operator
(`cluster/late_ops.py`); the audit checks the timed checkpoint against that record. An arm that does not arrive in
time gets no row and is reported as not run.

## A4 (2026-09-23 ~07:00 EDT) — FNO at 2048^2 trained inside the p2048 H200 allocation

`t2048s` confirmed that the f64 FNO cannot train at $2048^2$ on an A100-80G even at micro-batch 1 with
`expandable_segments` (a 32 GiB request with 55 GiB in use). Every H200 is allocated to other lanes, so rather than
a third training job, `p2048` (an H200 job) trains `fno-large` itself after its JAX phases (same data generator,
config, 3000 s budget; nothing is timed while it trains), then times it in Phase O. DeepONet still comes from
`t2048s` via A3. The first `p2048` submission (4206695) was cancelled while PENDING, before it ran.

## A5 (2026-09-23 ~10:25 EDT) — p2048b's JAX process killed at the host-memory limit; clean rerun p2048c

`p2048b` (4207177) ran Phases F and S through the quadratic manifold at $r=32$, then its `cmp.py` process was
killed by the 240 GB cgroup during the $r=64$ fit. Cause: `ladder.sha_array` hashes via `.tobytes()`, which copied the
72 GB bank on top of the 111 GB snapshot matrix and the bank itself. So `p2048b` has no $r=64$ arm, no Phase B bracket,
no completion flag, and cannot be the accepted $2048^2$ panel; it is kept as a record, and its in-job FNO training
(A4) and operator phase still ran. Fix: `cmp.sha_array` hashes a byte view in 256 MB chunks (identical digest, checked).
`p2048c` reruns the whole $2048^2$ panel in one H200 allocation, unchanged otherwise; the FNO it times is
`p2048b`'s in-job checkpoint, picked up through the A3 path (re-hashed against the SHA256 its training recorded).
No number from `p2048b` is quoted as a result.
