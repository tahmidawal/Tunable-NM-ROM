# ops-all / b3d — neural operators for the Burgers 3D rows of Table 1 (64³, 128³, 256³)

Group C of `OPS-ALL-PROTOCOL.md`. Written before any cluster job of this group. No tuning; later changes are
appended as dated amendments.

## 1. Source of the Table 1 rows (corrected)

The brief named lane `2026-09-23-burgers3d-span`, but the Table 1 Burgers 3D rows (1.01/1.46/2.20 % accurate,
3.37/4.26/3.77 % fast, FOM 0.92/1.39/1.52 %) and the job ids in the brief (held-out 4253861/4253865/4253867, validation
4249201/4249210/4249218) belong to its successor **`2026-09-23-burgers3d-retry`** (branch `exp/2026-09-23-burgers3d-retry`
@ `642ab587`, lab log 2026-09-24 entry, `selection.json` sha256 `c342bf72…`). All code, model and configs are taken
from there (the settings in the brief match it exactly). Model M2 (`bank.pkl` sha256 `6687f259…`), span-only.

* Meshes: 65/129/257 nodes per axis (64³/128³/256³ cells), homogeneous Dirichlet unit cube, T = 0.25, outputs
  t = 0, 0.05, …, 0.25.
* NM-ROM accurate / fast (frozen): 64³ span R′=512 Δt .005 / R′=192 Δt .01; 128³ same; 256³ R′=512 Δt .01 /
  R′=256 Δt .01 (fsc solver, f64 tensor table).
* FOM: Newton–BiCGStab, the lane's 26-setting grid (Δt ∈ {.005,.01,.025} × Newton/linear tolerances); Table 1
  settings 64³ Δt .005 ntol 1e-3; 128³ and 256³ Δt .01 ntol 1e-2 (ltol 0.1).
* Evaluation cohort: the sealed held-out cohort 923901 × 32 (already opened once by the lane; no choice is made on it
  here). Truth = same-grid Newton–BiCGStab Δt .005, ntol 1e-10, ltol 1e-11.
* Error: per case max over the five evolved times of ‖u − u_ref‖₂ / ‖u_ref(0)‖₂ on all interior nodes; reported worst
  and median over the 32 cases.

## 2. Operator data

Same generator and training seed as the NM-ROM (seed 923701, prefix-preserving first N draws; data solver Newton–
BiCGStab Δt .005, ntol 1e-8, ltol 1e-9 — the NM-ROM's `data_ntol/ltol`), regenerated at the target mesh inside the job.
Checkpoint selection on the NM-ROM's own bank-validation seed 923751 (training-side data; never an evaluation cohort).
Rows asserted disjoint from 923901 × 32 and 923801 × 64.

| mesh | N train (923701 prefix) | N validation (923751 prefix) | reason |
|---|---|---|---|
| 64³ | 1536 (all the NM-ROM used) | 64 | — |
| 128³ | 1536 | 64 | host memory 77 GB f32 |
| 256³ | 256 | 32 | host memory (256 × 403 MB f32) within `--mem 240G`; attempted once |

## 3. Operator contract

Families and code: `ops3d.py` + `spectral_conv_f64.py` copied unchanged from `experiments/ns3d-operators` (official
NeuralOperator 2.0.0 FNO with the reviewed float64 spectral conv; U-Net / Transolver / DeepONet 3D ports), wrapped by
`opsb3d.py`. **Dirichlet handling:** the operator grid is G³ with G = n − 1 = 64/128/256: the interior nodes plus the
wall plane at index 0 (exactly zero). Its periodic extension places the wall at both ends of every axis, so the FFT of
the FNO and the circular padding of the convolutions see the Dirichlet domain's own boundary; no extra padding. The
wall plane of the output is discarded. Inputs (5 channels): u0 / u_scale, standardised log ν (constant channel),
coordinates x, y, z (the problem is not translation invariant). Output: 5 channels = the five evolved times, times an
output scale; t = 0 returns u0 exactly. Loss = mean squared initial-relative error; FNO float64, others float32 with
float64 I/O, TF32 off.

## 4. Sizes (one per family, no tuning) and training

Nearest lane with validation-selected 3D sizes = ns3d-operators at 64³ (`configs/panel64.json`): fno-l (width 32,
modes 16, 4 layers), unet-l (base 32), tsol-l (dim 192, 8 layers, 8 heads, 64 slices, patch G/16), don-s (width 32,
rank 256, trunk 384). Used at 64³ and 128³. At 256³ the smallest sizes (protocol): fno-s (24/12/4), unet-s (16),
tsol-s (128/6/8/32), don-s. Optimiser/budget unchanged from ns3d-operators: AdamW 1e-3 / wd 1e-4, batch 8 with
micro-batching if needed, clip 1, cosine-in-wall-time to 1e-5, Transolver warm-up 160, **3000 s per network**, patience
250, seed 20260914; checkpoint = best validation mean case-max. An arm that cannot train is recorded as not trained.

## 5. Jobs

One allocation per cell (3 GPUs total, the group cap): `cluster/cell.sbatch` trains the four arms sequentially (data
generated once), then runs `panel_b3d.py` in a fresh process in the same allocation. `panel_b3d.py` = the lane's
`panel3.py` plus: operator arms evaluated on every case after the FOM grid (checkpoint sha256-checked) and timed in
the A1/A2 phases shuffled together with the NM-ROM arms (same GPU-resident-input → six GPU-resident fields contract,
`torch.cuda.synchronize`), a reproduction check of the NM-ROM worst errors against the Table 1 job, and no full-field
audit dumps. GPU: 64³/128³ on A100-80G (only one H200 was free at submission), 256³ on H200 (`--mem 240G`). Table 1's
own timings were on H200, so absolute ms at 64³/128³ differ from Table 1; every ratio here is same-job.

FOM rule (paper): fastest FOM setting in this job whose worst error ≤ the NM-ROM accurate arm's worst error.

## A1 (2026-09-24 ~13:05 EDT, before any result) — 256³ split into two allocations

The H200 queue estimate for the one-allocation 256³ cell was ~22:40 EDT, so `c257` (4286267) was cancelled while pending.
U-Net/Transolver/DeepONet train on an A100-80G (`t257`, train-only); the H200 job then attempts FNO and runs the panel,
reading the other checkpoints from `t257`. The panel (every timing) is still one H200 allocation.

## A2 (2026-09-24 ~15:40 EDT) — DeepONet 256³ retried on the H200

don-s ran out of memory at micro-batch 1 on the A100-80G (`t257`). The protocol requires the 256³ attempt on an H200, so
the pending H200 job `p257` (4286473, never ran) was cancelled and resubmitted as `p257b` with arms don-s, fno-s. No
setting was changed. U-Net/Transolver 256³ results from `t257` are unchanged and were not examined on any evaluation case.

## A3 (2026-09-24 ~16:40 EDT) — cohort identity gate

The c65 panel stopped at `assert expected_cohort_sha256` before any NM-ROM/operator evaluation: the parameter-table
sha256 includes `s_star`, a peak computed on the GPU, and differs between H200 (Table 1) and A100/GB10. The initial
fields regenerated on the GB10 equal Table 1's saved audit-lattice initial fields bitwise (64³ and 128³). The gate is now:
the sha256 is recorded (`cohort.sha256_matches_table1`), and the panel asserts that the u0 (≤1e-12 abs) and all six
reference frames (≤1e-7 rel) on the 16³ audit lattice equal Table 1's saved `reference_restricted.npz` of the same mesh
(copied into `configs/table1_refs/`, sha256 equal to the Table 1 job's OUTPUTS manifest). The code was rsynced into the
c129 job dir during its training stage (before its panel stage) and into the pending p257b dir; 64³ panel rerun as the
panel-only job `pn65` reading the c65 checkpoints.

## A4 (2026-09-24, after the fact) — 256³ panel failed; not resubmitted

p257b (4292776) finished training don-s and fno-s on the H200, then the panel ran out of GPU memory building the
257-node NM-ROM tables under the JAX memory fraction 0.72 chosen for the shared JAX+PyTorch process (Table 1 used 0.92).
Recorded as the cell's result; not resubmitted, on the coordinator's instruction.
