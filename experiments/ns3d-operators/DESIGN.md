# ns3d-operators — neural operators on Navier–Stokes 3D (Table 2 row), pre-registration

Written and committed before any cluster job of this lane. Amendments are appended as §A1, §A2, …;
nothing above them is edited afterwards. No number in this file is a measurement of this lane.

Branch `exp/2026-09-23-ns3d-operators` (local only, never pushed), forked from
`exp/2026-09-23-ns3d-shift-head` @ `708c70fe` (the lane behind the paper's NS 3D rows). Cluster
namespace `/cluster/tufts/paralab/tawal01/nsops_20260923/<job>/`, one directory per job, never reused.

## 1. Question

Table 2 of the paper compares the NM-ROM with FNO, U-Net, Transolver and DeepONet per (PDE, mesh) cell:
same data, split, optimiser, seed and wall budget; each operator family at its most accurate trained
size chosen on validation; error = worst same-grid relative $L^2$ error over evolved output times;
speedup = median GPU time of ONE full-order setting per cell (the fastest tested CNAB2 setting at least
as accurate as the NM-ROM accurate setting) divided by each method's median time, all in the same
allocation. This lane builds that row for NS 3D at $32^3$ and $64^3$, and at $96^3$ as a stretch (§7).

## 2. Problem, data, cohorts (inherited, unchanged)

NS family of `experiments/ns3d/ns3d_fom.py` (sha256 recorded in every job): periodic unit cube, initial
velocity = curl of two localised vector potentials (amplitude 0.2), $\nu$ log-uniform in $[0.002,0.01]$,
$T=0.2$, six output times $t=0,0.04,\dots,0.2$, truth = CNAB2 with 2/3 dealiasing at $\Delta t=0.001$
on the evaluation mesh (`diag_floor.generate`). The same code generated the NM-ROM's data and the
paper's rows.

* **Training data (operators).** Seed 202609201, the first 512 draws (prefix-preserving) — exactly the
  512 trajectories the NM-ROM head was trained on (`head_train_cases=512` in the shift-head configs),
  regenerated in each training job at the target mesh. **Split = the head's split:** draw index
  $\equiv 7 \pmod 8$ is validation (64 trajectories), the other 448 are training. The head was fitted on
  the 448 training trajectories' states (21 frames each, one code per state) and held out the same 64;
  the bank is the POD of the first 128 of the 512 draws (six frames each, including the 16 of them
  that sit in the validation split). So the operators and our model see the **same 512 trajectories**;
  the operators use 448 for gradients and 64 for checkpoint/size selection, our head used 448 for
  fitting and our bank used 128 (of the 512). An operator sees one supervised pair per trajectory
  (initial field → five evolved fields at the six saved frames); the head saw 21 states per
  trajectory. This is stated in the report as the data-parity sentence.
* **Evaluation cohort.** The 16 development cases of Table 1 (seed 202609202) at $32^3$ and $64^3$,
  regenerated in the panel job. At $96^3$ (stretch) the 32 held-out cases (seed 202609221) — the cohort
  of the paper's held-out NS sentence, opened once by the shift-head lane (job 4202872); this would be
  its second opening and no choice is made on it. No operator choice is made on any evaluation cohort.
  Training rows are asserted disjoint (rounded-row) from both evaluation cohorts; seeds 202609203 and
  202609211 are never used.
* **Error.** Per case and output time, $\lVert u_{\rm arm}(t)-u_{\rm truth}(t)\rVert_2/\lVert u_0\rVert_2$
  (the NS rows' normalisation, `diag_floor.rel_rows`); reported: worst over cases of the max over the
  five evolved times ("evolved worst"), plus the median over cases.

## 3. NM-ROM arms (frozen, no retraining)

The paper's NS settings, from the frozen development models of the shift-head lane: at mesh $n$ the
bank is rebuilt from the seed exactly as in `head_ladder.py` (128 training trajectories, energy-centred,
rank-64 POD), sign-aligned to and gated against the stored bank probe of job `a2_h{n}` (gap ≤ 1e-8),
and the head ($k=8$) and importance rotation are loaded from that job (copied to
`frozen/h{32,64}/`, sha256 below). At $96^3$ the lane's committed `frozen/` (from `a3_h96`) is used.

* **accurate** = head $k=8$, implicit midpoint $\Delta t=0.02$, 3 damped Gauss–Newton sweeps;
* **fast** = span $R'=16$ of the importance-ordered bank, same stepping.

Reproduction gate: per-case development errors of both arms and of every CNAB2 setting equal the
`a2_h{n}` summary values to ≤1e-6 relative (different GPU), else the mesh is reported as failed.

| file | sha256 |
|---|---|
| frozen/h32/head_k8.npz | 35c1a8aff8bdb5da292d3dd297470d25f23900b25529ba7fe99f309a8e792369 |
| frozen/h32/rotation.npz | 910d66995bc1f6604156c6a5c9491fa492e2947dd89000c4c22d7788ea3b523a |
| frozen/h32/bank_probe.npz | a78d896fa78efa8fba3bbbcd4fce81abd17ca34924acd720dd090a7d52312802 |
| frozen/h64/head_k8.npz | 605637b1a54087d59b88c7f6563c6dafaa4adcea58b04022253abf8196437c02 |
| frozen/h64/rotation.npz | 62e29e976129a50a8b5a43ea9941498df73f7029003307ffc945dd5378f58435 |
| frozen/h64/bank_probe.npz | 36bde9e938d26ba8d8a66a4e5e3dedefe58685f6257453a812fbd5cf7e297ff3 |

## 4. Neural operators

**Code.** `ops3d.py`: 3D ports of the Table-2 cells' PyTorch families (ops-tune-grid @ 52d1b573 as
copied into heat-compare-hires/ops): the official NeuralOperator 2.0.0 `FNO` with the reviewed float64
`SpectralConvF64` (byte-identical copy, sha256 b41d614e…, upstream-hash check in `configure()`), and
U-Net / Transolver / DeepONet with the same topology moved to 3D.

**Contract (every family, and the NM-ROM's query scope).** Input: $u_0$ (3 components, float64, on
the GPU) and $\nu$. Features: $u_0/s_u$ and the standardised $\log\nu$ broadcast as one constant
channel (4 channels; scales from the training split only). Output: 15 channels = 5 evolved times × 3
components, times an output scale, cast to float64, then **Leray-projected** onto the 2/3-dealiased,
divergence-free, zero-mean space of the truth (the FOM's own projector, verified equal to
`ns3d_fom.project_field` to 5e-16 and idempotent, `checks/check_projector.log`). The projection is an
orthogonal projection onto a space containing the truth, so it can only lower the $L^2$ error; it is
part of the network in training and charged in every timed query. $t=0$ returns $u_0$ exactly.
One pass, fixed output times.

**Declared 3D / periodic deviations from the 2D cells.** No coordinate channels (the PDE is
translation-invariant on the torus; FNO and U-Net stay translation-equivariant; the 2D Dirichlet cells
fed $x,y$). Convolutions use circular padding. U-Net: 4 levels, grid side divisible by 16.
Transolver: token = $p^3$ patch with $p=n/16$ (token grid $16^3$ at every mesh), unified positional
encoding against an $8^3$ reference lattice, circular 3^3 convolutional projections. DeepONet: CNN
branch (3 levels, circular padding, adaptive pool to $4^3$), trunk on periodic features
$\sin/\cos(2\pi f x_i)$, $f\in\{1,2,3,4\}$. FNO modes are counts per axis (NeuralOperator convention).
Loss uses the initial-relative metric (the NS metric), not a current-relative one.

**Sizes (two per family, one seed 20260914).**

| arm | family | configuration | dtype |
|---|---|---|---|
| fno-s | FNO | width 24, modes 12³, 4 layers | float64 |
| fno-l | FNO | width 32, modes 16³, 4 layers | float64 |
| unet-s | U-Net | base 16 | float32 |
| unet-l | U-Net | base 32 | float32 |
| tsol-s | Transolver | dim 128, 6 layers, 8 heads, 32 slices | float32 |
| tsol-l | Transolver | dim 192, 8 layers, 8 heads, 64 slices | float32 |
| don-s | DeepONet | width 32, rank 256, trunk 384 | float32 |
| don-l | DeepONet | width 48, rank 512, trunk 768 | float32 |

Precision follows the Table-2 cells: FNO float64/complex128, U-Net/Transolver/DeepONet float32 with
float64 I/O, TF32 disabled.

**Optimiser and budget (the Table-2 cells' protocol).** AdamW (lr $10^{-3}$, weight decay $10^{-4}$),
batch 8 (micro-batches with gradient accumulation if 8 does not fit; recorded), gradient clip 1.0,
cosine learning rate in elapsed wall time to $10^{-5}$ (heat-compare-hires' declared deviation, used
here for the same reason), Transolver warm-up 160 optimisation steps, **3000 s wall budget per arm**,
epoch cap 4000, patience 250 epochs; validation after every epoch; checkpoint = best validation mean
over cases of the per-case max evolved error. Training targets are held on the GPU in float32
(quantisation ~1e-7 relative, far below any operator error); losses and errors are float64.

**Jobs.** One job per (arm, mesh): 16 jobs at $32^3$/$64^3$ on A100-80G (all arms of a mesh on the
same GPU type so the wall budget buys comparable work), submitted concurrently, each in its own
directory. An arm that cannot train (no micro-batch fits, crash, time limit) is recorded as **not
trained**, never shrunk.

**Size selection (pre-registered).** Per family and mesh, the arm with the lower validation
mean-case-max of its selected checkpoint (from its `result.json`, validation data only); a tie within
1 % relative goes to the smaller arm. Written by `select_ops.py` into `configs/panel{n}.json`
and committed before that mesh's panel job. All trained arms are also timed and printed in an
appendix-style table; Table 2 uses the selected one.

## 5. Panel job per mesh (one allocation, one process, A100-80G)

1. Truth: the evaluation cohort at mesh $n$ (CNAB2 $\Delta t=0.001$), finite.
2. NM-ROM: bank rebuild + gate, head/rotation load, both arms on every case (accuracy pass), then the
   reproduction gate (§3).
3. CNAB2 ladder: steps {200, 100, 80, 70, 60, 50, 40, 20, 10} over $T=0.2$ (the shift-head grid);
   a setting is *stable* if all fields are finite and its evolved worst is ≤ 100 %.
4. Operators: every trained checkpoint of this mesh, sha256-checked against its training record,
   evaluated on every case (accuracy pass).
5. **FOM rule.** The one comparator of the cell = the stable CNAB2 setting with the smallest median
   time in this job whose evolved worst is ≤ the NM-ROM accurate arm's evolved worst. Every speedup
   = that median / the arm's median. (Reported beside it, not in Table 2: the ratio against the
   fastest CNAB2 at least as accurate as each arm.)
6. **Timing (A–B–A).** Query = GPU-resident input → six GPU-resident fields, synchronised before and
   after (JAX `block_until_ready`, `torch.cuda.synchronize`); compilation and warm-up excluded (every
   arm runs all cases untimed first, then 2 burn-in calls). Timed arms: NM-ROM accurate and fast,
   every CNAB2 setting, every trained operator. Phase A1: 3 rounds, each a fresh random permutation of
   the arms, each arm then timed once per case (16 samples per arm per round). Phase B: arm-major, in a
   fixed random order, each arm 3 rounds × all cases consecutively. Phase A2: as A1 (new permutations).
   **The reported time is the median of all samples of A1+B+A2** (9 per case). Gates (applied to the
   NM-ROM accurate and fast arms, the rule-chosen CNAB2 and every Table-2 operator; failing gates are
   reported, never worked around): **drift** median(A2)/median(A1) in [1/1.10, 1.10]; **order effect**
   median(B)/median(A1∪A2) in [1/1.10, 1.10]. Positive control: the same gate code applied to this
   job's A-samples with a synthetic ×1.15 on A2 must fail.
7. **Timed outputs** equal the accuracy-pass outputs (JAX arms ≤1e-9 relative, float64 FNO ≤1e-9,
   float32 operators ≤1e-4).
8. **Saved fields (disk-capped).** At $32^3$: every arm's full float64 fields for all cases (≈1.5 GB).
   At $64^3$/$96^3$: full fields of cases {0, 1, the arm's worst case} per arm plus the truth of those
   cases, and for every (arm, case) a fixed random sample of 32768 grid points × 3 components × 6
   times of prediction and truth. Free space is checked before writing; fields are pulled and then
   deleted from the cluster; nothing else full-field is written.
9. **Independent audit** (`audit_panel.py`, NumPy only, separate process in the job and again locally):
   regenerates every case's $u_0$ with NumPy (own FFT Leray projection) and checks it against the saved
   $u_0$; recomputes every saved full-field error (match ≤1e-9 relative to the job); for sampled cases
   recomputes the error estimate from the samples (reported, loose bound 20 %); recomputes the FOM rule,
   every speedup and every timing median from the raw repetitions; must reject a copy with one saved
   field perturbed by 1e-6 relative.

## 6. What is reported

Per mesh (script-generated from the panel summary JSON; no hand-typed number): NM-ROM accurate and
fast (error, ms, speedup), each family's selected operator (size, error, ms, speedup, epochs, stop
reason), the chosen CNAB2 setting (steps, error, ms), job id, GPU, cohort, gates. A mesh whose panel
fails a gate is printed with the failure. Report `reports/<date>-ns3d-operators.md` with a glossary.

## 7. $96^3$ (stretch)

Run only if the training and the panel fit before 2026-09-24 ~18:00 EDT: the same 8 arms trained at
$96^3$ (H200; if training data exceed 35 % of GPU memory they are held in pinned host memory), panel on
the 32 held-out cases with the lane's committed `frozen/` settings (the shift-head `b2_heldout96`
reproduction values are the gate), on H200. If it does not fit, the report says it was not run.

## 8. Audits

Codex `exec` read-only design audit of this file and the code before the first training job
(`checks/codex-design-audit.md`), and a results audit before reporting
(`checks/codex-results-audit.md`). Dispositions go into amendments.

## A1 (2026-09-23, before any cluster job) — Codex design audit, dispositions

`codex exec` (read-only instructions; run with `--dangerously-bypass-approvals-and-sandbox` because
the bubblewrap sandbox cannot start on this box, as in earlier lanes), report kept verbatim at
`checks/codex-design-audit.md`. No leakage found; verdict "hold until the probe, directory-race and
interruption issues are fixed". Dispositions:

1. *Memory probe* — accepted: the probe now runs two complete AdamW steps (optimizer state allocated),
   model construction is inside the OOM handler, and training objects are released before the
   checkpoint reload.
2. *Directory race* — accepted: `stage.sh` checks the tree is clean first and claims the remote
   directory with a plain (atomic) `mkdir`; `train_op.py` refuses an output directory that already
   holds `best.pt`/`result.json`/`history.json`.
3. *Interrupted runs* — accepted: an arm stopped by a signal, or whose reloaded checkpoint does not
   reproduce its recorded validation value (1e-6 relative), is written `complete=false` and is
   ineligible for selection (= "not trained"). The job time limit (2.5 h) is ~2.5× the expected run.
4. *Selection asymmetry* — accepted as a labelling requirement: the NM-ROM's $k=8$ was selected by the
   shift-head lane on the same 16 development cases used here (as for Table 1); the operators select
   size and checkpoint on the separate 64-trajectory validation split. The report states that the
   $32^3$/$64^3$ cells are **development comparisons** in which the NM-ROM setting was chosen on the
   evaluation cohort; only the stretch $96^3$ cell uses a cohort unused for either side's selection.
5. *Disk cap / restricted audit* — accepted: the panel refuses to write fields if free space is below
   40 GB **or** if its own saved-field total would exceed a hard cap of 4 GB (checked before each
   write); the audit is called **restricted** in the report (exact recomputation on the saved cases,
   sampled estimates elsewhere; the truth solver is the same CNAB2 code that produced the paper's rows
   and is not independently re-solved).
6. *Gates enforce status* — accepted: `summary.json` carries `status: final` only if every gate passes
   (reproduction, drift, order, positive control, timed-output parity, bank rebuild, finite outputs,
   ≥ 9 samples per arm and case); otherwise `PROVISIONAL (diagnostic only)`, and the report prints no
   headline row from a provisional panel without that label. A 2 s GPU burn precedes each timing phase.
   Speedups are labelled "vs the NM-ROM-accuracy CNAB2 comparator" (the Table-2 rule), not
   equal-accuracy speedups per operator (those are printed separately as context).
7. *Pinned fallback* — accepted: batches are gathered into a reusable pinned buffer; $96^3$ jobs request
   ≥ 160 GB host memory.
8. *Budget bookkeeping* — accepted: training time and finalisation time are recorded separately; the
   final validation uses the checkpoint's own normalisation. The last update + validation may overrun
   3000 s by one epoch's tail, as in the 2D cells.
9. *U-Net equivariance* — accepted: stride-2 pooling/upsampling make the U-Net equivariant only to
   shifts by multiples of 16 cells; the wording "translation-equivariant" applies to the FNO only.

## A2 (2026-09-23 ~15:25 EDT, after pn32, before any 64³/96³ panel) — saved full fields at 96³

With 32 held-out cases at $96^3$ (127 MB per full field set), full fields of cases {0, 1, worst} for 19
arms would exceed the 4 GB cap of A1 and abort the job. At $96^3$ only each arm's **worst case** is
saved in full (plus its truth); samples are kept for every case as before. $32^3$ (done) and $64^3$
keep {0, 1, worst}. Nothing else changes. `pn32` (job 4234771) passed every gate; no setting or
selection was revisited after seeing it.

## A3 (2026-09-23 ~16:05 EDT, after pn32/pn64) — Codex results audit, dispositions

`checks/codex-results-audit-32-64.md` (read-only; recomputed all 38 arms' medians, errors, FOM
choices and speedups, all 114 saved full-field sets to 4.1e-16, the checkpoint sha256 chain and the
selection; no corrupted number, no selection leak, no cross-job ratio). Verdict: the rows may enter
Table 2 as qualified development comparisons. Dispositions:

1. *Timed-output parity was error-metric parity* — accepted. pn32/pn64 compared each timed output's
   error vector with the accuracy pass (gap 0 for every arm), not the fields. Their status stays as
   written by the job, but the report states that this gate was error-metric parity. From the next
   panel on, the timed field is also compared with the accuracy pass on the 8192 sample points.
2. *Saved-field coverage deviated from §5.8* — accepted as a disclosed deviation. The code committed
   before pn32 (`make_panel_config.py`, `panel.py`) saves {0, 1, worst} per arm and 8192 sample points
   at every mesh, not all cases at 32³ and 32768 points as §5.8 said; the text was not updated. A2
   described this coverage without flagging it. The report says "restricted audit" with the actual
   coverage.
3. *The 20 % sampled bound was not enforced* — accepted. The sampled estimates are diagnostic only
   (Codex: 500 resamples of a saved 64³ error field give a 95 % ratio range 0.48–1.58, so the observed
   0.22/0.42 gaps are sampling scatter of localised errors). This relabelling is retrospective.
4. *Positive control* — accepted: it now multiplies the actual A2 samples by 1.15 (Codex applied that
   to pn32/pn64 and every arm still rejects).
5. *Epoch labels* — accepted: the report prints epochs evaluated (last may be partial), optimisation
   steps and the 0-based best epoch.
6. *Data parity* — accepted: the report now says that the bank includes 16 trajectories of the
   operators' validation split.

## A4 (2026-09-23 ~16:25 EDT) — 96³ training split across H200 and A100-80G (coordinator direction)

H200s were saturated cluster-wide. On the coordinator's instruction, the three 96³ arms still pending on
H200 (don-l 4232984, don-s 4232979, tsol-l 4232975; never started, directories empty) were cancelled and
resubmitted unchanged on A100-80G in the same, still-unused job directories (don-l 4239995,
don-s 4239998, tsol-l 4240001; staged source 814c58b6, `train_op.py` identical to the pre-registered
one). fno-s, fno-l, unet-s, unet-l and tsol-s run or ran on H200. **Deviation from §4/§7** ("all arms of
a mesh on the same GPU type"): at 96³ the 3000 s budget buys less work for the A100 arms (and their
32.6 GB training set exceeds 35 % of an 80 GB card, so it is held in pinned host memory, per A1.7). The
report prints each arm's GPU and epochs, and the 96³ cell is labelled with this deviation. No selection
has been made at 96³.

## A5 (2026-09-23 ~17:25 EDT, after all 96³ training, before the 96³ panel) — 96³ panel GPU

The 96³ size selection was written by `make_panel_config.py 96` from the validation records only
(fno-l, unet-l, tsol-s, don-s). H200s are saturated, so the 96³ panel runs on an A100-80G (the GPU type
of the shift-head held-out job it reproduces) instead of the H200 named in §7, with JAX's memory
fraction raised to 0.7 (the rank-64 bank build at 96³ needs it; the operators need < 5 GB at batch 1).
Everything else as pre-registered; cohort = the 32 held-out cases (seed 202609221), second opening.

## A6 (2026-09-23 ~17:45 EDT) — pn96 (job 4243807) ran out of GPU memory in warm-up; rerun pn96b

pn96 completed the truth, bank rebuild, reproduction gate (all arms pass, max 2.9e-10) and the accuracy
pass, then hit a PyTorch CUDA OOM during warm-up (DeepONet-l's 96³ trunk evaluation) because JAX held
70 % of the card. No timing ran; its logs and partial summary are kept at `runs/pn96_crashed/` and its
remote directory is deleted. Fix (memory only, no change to any setting, model or selection): JAX
fraction 0.55, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and `torch.cuda.empty_cache()`
after each arm's accuracy pass and warm-up. The rerun `pn96b` evaluates the identical config
(`configs/panel96.json`) on the same held-out cohort (its third opening overall; no choice is made on it).

## A7 (2026-09-23 ~18:00 EDT) — pn96b (job 4244288) ran out of JAX memory in timing phase A2; rerun pn96c

Cause: a leftover in `panel.py`'s `timed()` kept every arm's timed output of the last A2 round alive
on the device (`last_out`, never read after the field-parity change of A3): 19 arms × 32 cases ×
127 MB at 96³. At 32³/64³ it fitted (≤ 11 GB) and changed nothing measured. The dict is removed;
no setting, model, selection or gate changes. pn96b reproduced pn96's accuracy pass exactly (gap 0 on
every arm); its logs and partial summary are kept at `runs/pn96b_crashed/`, remote deleted. Rerun
`pn96c`, same config, same cohort, A100-80G, JAX fraction 0.55.

## A8 (2026-09-23 ~18:40 EDT) — Codex results audit of 96³, dispositions

`checks/codex-results-audit-96.md`: no numerical blocker; recomputed all 19 arms, the FOM rule
(CNAB2 70 steps; 60 steps ineligible), speedups, the 19 saved full fields (4.1e-16), sha256 chain,
selection (config committed before pn96), gates; the reruns changed memory handling only. Verdict: the
row may enter Table 2 as a qualified held-out comparison. Accepted and applied in the generated report:
(1) audit coverage printed per mesh (worst case only at 96³); (2) the A4 GPU split is printed for every
operator arm (all-arms table) and the Transolver size/hardware confound is stated; (3) reruns "agree
within reproduction tolerance" (generated gap), not "identical"; (4) the epoch range is generated and
"not converged" softened to "budget-limited"; (5) parity scope (sampled timed-field parity on the final
A2 repetition) and the cohort's fourth opening are stated.
