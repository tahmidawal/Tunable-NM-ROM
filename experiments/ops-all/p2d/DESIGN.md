# ops-all / p2d — neural operators for the Poisson 2D and L-shape Table 1 cells

Written before any cluster job of this group. Protocol: `../../../OPS-ALL-PROTOCOL.md` (binding). Amendments are
appended as §A1, §A2, … and nothing above them is edited afterwards.

## Cells

| problem | meshes | Table 1 source jobs | NM-ROM accurate / fast | named FOM (Table 1) | cohort |
|---|---|---|---|---|---|
| Poisson 2D (square) | 256², 1024², 2048², 4096² | 4199318 (256–2048²), 4199321 (4096²) | span $R'=512$ / span $R'=128$ | CG rtol 3e-2 (256², 1024²), 1e-1 (2048², 4096²) | 12 development sources (seeds 7090703 ×6, 7090732 ×6) |
| Poisson L-shape | 256², 512², 1024², 2048² | 4204384, 4199770, 4201792, 4205072 | span $R'=128$ / span $R'=64$ | CG rtol 3e-2 | 32 development sources (seed 20260917) |

Error = worst (and median) same-grid relative $L^2$ over the cohort; reference = DST-I (square) / SuperLU with
refinement (L-shape), exactly the lanes' references. Time = GPU query (`fused_device_seconds`), the scope of both
series in Table 1.

## Operators (no tuning)

One configuration per family, the validation-selected published ones used by the nearest 2D high-resolution lane
(heat-compare-hires @ 9a5baf9a, taken from ops-tune-grid @ 52d1b573): FNO `large` (width 64, 32 modes, 4 layers,
float64), U-Net `medium` (base 32, float32), Transolver `large` (dim 256, 8 layers, patch $n/64$, float32),
DeepONet `small` (float32). At 4096² the smallest existing size per family: FNO `large` (the only FNO size),
U-Net `small` (base 24), Transolver `small` (dim 128), DeepONet `small`.

Contract (`ops/pops.py`): input = normalised nodal source + $x$, $y$ channels; output = the solution times the fixed
domain mask (boundary, and the removed quadrant for the L-shape); loss = mean squared relative $L^2$. AdamW
(lr 1e-3, wd 1e-4), batch 8 (gradient accumulation where it does not fit), clip 1.0, wall-time cosine lr, seed
20260914, **3000 s wall budget per network, one network per job**. Checkpoint = best mean validation error on the
first 16 draws of the NM-ROM's validation split. Never the evaluation cohort (disjointness asserted).

Training data = the NM-ROM's own training cohort and seed (square: `source_params(0, 3072)`; L-shape:
`cohort(0, 4608, 3072)`), its fit split (`fit_validation_split(3072, 20260916, 0.15)`, 2611 draws), regenerated at
the target mesh on the cluster. Square targets are the exact DST solve per batch (free), so every square network
sees all 2611 draws. L-shape targets need a CG solve per draw (batched GPU CG to relative residual 1e-9), so the
first N fit draws are used: N = 2611 at 256² and 512², 1024 at 1024², 512 at 2048² (declared; within 3000 s the
large-mesh networks see fewer samples than that anyway).

## Panel (one job per cell, one GPU)

`panel.py` builds, in one process on one UUID-guarded GPU: the two Table 1 NM-ROM settings with the source lanes'
code copied verbatim from the staged `code/` directories of the Table 1 jobs (`nmrom_sq/` = pbkH/pbkI,
`nmrom_ls/` = l2048b); the lanes' CG grid (square rtol {0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.03, 0.01, 1e-3, 1e-4};
L-shape {0.7, 0.5, 0.3, 0.2, 0.1, 0.03, 0.01, 0.003, 1e-3, 1e-4}; tight tolerances dropped at 4096² for time); and the
four operator checkpoints. Timing A–B–A (poisson-bank-knob A5): A1 = NM-ROM + operators randomised within each
case with 0.1 s GPU burn-in before every call; B = CG; A2 = A1. Table time = median over A1 ∪ A2.

Gates / checks reported (not tuned against): NM-ROM worst errors reproduce the Table 1 lane values to 1e-6
relative; CG converged; NM-ROM/CG fields deterministic; drift A2/A1 within 1.10; device UUID unchanged. Operator
fields are compared across repetitions (max relative deviation recorded; cuDNN may be non-deterministic).

Per cell we report: FOM (named Table 1 setting) ms/err, the fastest CG at least as accurate as the NM-ROM accurate
setting, NM-ROM accurate/fast ms/err, and per operator worst %, median %, median GPU ms, epochs, stop reason — or the
failure (OOM etc.). Every ratio is same-job.

## GPUs

Training ≤ 2048²: `a100-80G|h100-80G|h200-141G`, except FNO at 2048² (heat measured a 93 GB peak at micro-batch 1)
and everything at 4096²: H200, `--mem 240G`. Panels: A100/H100/H200 80 GB+ up to 2048², H200 at 4096² (69 GB bank).
At most 5 p2d jobs queued or running at once (`cluster/submit.sh` refuses a sixth).

## A1 (2026-09-24 ~13:05 EDT, after the first 4 submissions, before any result) — per-user GPU limit; co-scheduling

The first wave (s256f/u/t running, s256d pending) showed the account's Slurm QOS cap `QOSMaxGRESPerUser`: about
12 GPUs for the whole account, shared by all four ops-all groups, so this group gets ~3 GPUs, not 5. At one network
per GPU (32 training jobs ≈ 32 GPU-hours) the set cannot finish in the ~9 h window. Change, made before any result:
networks that fit together are **co-scheduled on one GPU as separate processes**, each with its own full 3000 s wall
budget and a hard per-process memory share (`--mem-share`, `torch.cuda.set_per_process_memory_fraction`):
L-shape 256² and 512²: all four on one GPU (share 0.24); 1024² (both problems): FNO+Transolver and U-Net+DeepONet
(share 0.5); 2048²: FNO alone (H200), U-Net alone, Transolver+DeepONet (share 0.5); 4096²: one per H200. Square 256²
was already submitted one network per GPU and stays so. A co-scheduled network gets the same wall budget but a
share of the GPU's throughput, so fewer epochs; every result records `co_scheduled_on_same_gpu` and its epochs, and
the report flags it. This can only disadvantage the operators. `l256f` (job 4286516) was cancelled while PENDING,
its remote directory deleted and the name retired. Panel jobs are never co-scheduled.

## A2 (2026-09-24 ~15:25 EDT, before any 2048²/4096² result) — no H200 available; 4096² and FNO-2048² on ≥80 GB cards

`s4096f` (4290709) and `s4096u` (4290832) sat PENDING on `h200-141G` with Slurm start estimates of 21:40 today and
09:45 on 09-26 (all 32 H200s in use), past this group's window. Both were cancelled while PENDING, remote directories
deleted, names retired. The 4096² networks (restaged as `s4096fa`, `s4096ua`, `s4096t`, `s4096d`), FNO at 2048²
(`s2048f`, `l2048f`) and the 4096² panel now use `a100-80G|h100-80G|h200-141G`. The Table 1 4096² job (4199321) itself
ran on an A100 80GB. A network that cannot fit on the card it lands on (expected for FNO at 4096², and likely for FNO
at 2048², measured 93 GB at micro-batch 1 in heat) records the out-of-memory failure as its result; that result is
"out of memory on an 80 GB card", not a statement about a 141 GB H200.

## A3 (2026-09-24 ~15:35 EDT) — micro-batch probe leaked memory; three jobs restaged

`s4096fa` (4292145) logged its micro-batch probe with 65 GB still held from the failed larger attempts, so its
"micro-batch 1 does not fit" was not a clean test. The probe now runs `gc.collect()` + `empty_cache()` after each
failed attempt and logs the allocated memory. Jobs staged with the leaky probe that had not finished probing were
cancelled and restaged: `s2048u` (4292114) → `s2048ub`, `s4096ua` (4292146) → `s4096ub`, `s4096t` (4292219) →
`s4096tb`; `s4096fa` is superseded by `s4096fb`. Networks already training (s256*, l256a, l512a, s1024a/b,
l1024a/b) were not affected in validity: at worst the leak made them pick a smaller micro-batch (gradient
accumulation keeps the effective batch at 8), i.e. slower epochs. Recorded per network (`micro_batch`).

## A4 (2026-09-24 ~15:40 EDT) — FNO at 2048² on 80 GB cards: activation checkpointing (memory only)

`s2048f` (4292408, A100 80GB) found no micro-batch that fits: micro-batch 1 needs more than 80 GB (heat measured 93 GB
on an H200), and no H200 is available (A2). The job's out-of-memory record is kept (`runs/s2048f-oom-plain-4292408`).
FNO at 2048² (square and L-shape) is rerun as `s2048fc` / `l2048fc` with activation checkpointing of each FNO block
(`torch.utils.checkpoint`, blocks recomputed in the backward pass). The arithmetic is identical — checked locally:
gradients and outputs of a checkpointed and a plain FNO with the same weights agree exactly (max difference 0.0) —
so this is not a model or hyperparameter change; it only trades memory for recomputation (fewer epochs in 3000 s).
Not applied at 4096², where the plain FNO needs well over 100 GB before the blocks.

## A5 (2026-09-24 ~16:25 EDT, before any L-shape 2048² job) — L-shape 2048² training set 256 draws

The batched CG data generation at L-shape 1024² took 1988 s for 1024 draws (two co-scheduled processes each
generating their own copy). Extrapolated to 2048² (≈2× iterations, 4× unknowns) 512 draws would take over an hour per
process before the 3000 s training budget starts. To finish inside the window the L-shape 2048² networks use the
first **256** fit draws (was 512) and a 4.5 h job limit. The training budget is unchanged. Declared; it can only
disadvantage the operators.

## A6 (2026-09-24 ~18:25 EDT) — coordinator rule: one network per GPU; dedicated rerun of the square 1024² cell

The coordinator ruled that each GPU allocation trains exactly one network (no co-scheduling), because co-trained
operators end budget-starved. Jobs of this group that shared a GPU: `l256a` (4 networks), `l512a` (4), `s1024a`,
`s1024b`, `l1024a`, `l1024b`, `s2048b`, `l2048b` (2 each; `l2048b` was already running and is not cancelled). Their
results stay in the record, flagged ᶜ with per-network epochs/steps; no further co-scheduled job is submitted. With
the group cap now 2 GPUs and ~3 h left, the square 1024² cell is rerun one network per GPU (`s1024rf/ru/rt/rd`,
same configs, data, seed and budget) with its own panel (`s1024rp`), as cell `s1024r`. Both square 1024² rows are
reported, labelled; the rerun is the protocol-compliant one. No other cell can be redone in the window.

## A7 (2026-09-24 ~19:00 EDT) — L-shape 2048² panel: XLA autotuning out of memory on an 80 GB card

`l2048p` (4301745, A100 80GB) failed during the NM-ROM bank reduction (`lsh_core.reduce_bank`, P·(AG)): XLA's GEMM
autotuner tried to allocate a 12 GB profiling buffer beside the 2049² bank and ran out of memory. The Table 1 job
(4205072) ran on an H200. Rerun as `l2048q` with `XLA_FLAGS=--xla_gpu_autotune_level=0` (default kernels, no
profiling buffers), same code otherwise. The NM-ROM reproduction gate (worst error vs Table 1 to 1e-6) checks that this
does not change the NM-ROM results.

## A8 (2026-09-24 ~19:40 EDT) — L-shape 2048² panel: bank held on the host during `reduce_bank`

`l2048q` (4303032, A100 80GB, autotuning off) still ran out of device memory in `lsh_core.reduce_bank` (an 18 GB
allocation for P·(AG) beside the 12.9 GB device bank). `reduce_bank` converts the bank to a host array anyway, so
the panel now passes the host copy and frees the device copy first, then re-uploads it for the rotation (same
functions, same arithmetic; locally the NM-ROM errors are bit-identical). Rerun as `l2048r`. The reproduction gate
checks the Table 1 errors again.

## A9 (2026-09-24 ~20:25 EDT) — L-shape 2048² panel: second attempt with the JAX memory pool capped

`l2048r` (4303797, A100 80GB) completed with every gate passing, but the FNO and DeepONet queries ran out of memory in
warm-up: JAX's allocator keeps its build-time peak (bank build + `reduce_bank`) reserved, leaving too little for the
PyTorch operators on an 80 GB card. `l2048r` stays the record for the NM-ROM, CG, U-Net and Transolver rows. One more
panel, `l2048s`, caps the JAX pool at 60 % of the card (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`) so the operators have
about 30 GB. If it completes with all gates, it replaces `l2048r` as the cell's panel (one job for all ratios).
Otherwise `l2048r` stands and FNO/DeepONet at L-shape 2048² are reported as "query out of memory on an 80 GB card
beside the NM-ROM state".
**A9 outcome:** `l2048s` (4305415) ran out of JAX memory in `reduce_bank` (18 GB allocation) under the 60 % cap, so
the NM-ROM state alone needs more than ~48 GB there. `l2048r` (4303797) is the L-shape 2048² panel. FNO and DeepONet
at L-shape 2048² trained (4 and 9 epochs) but their queries do not fit beside the NM-ROM state on an 80 GB card.
