# ops-all / g2d — neural operators for Burgers 2D 512², 4096² and Heat 2D 4096²

Group D of `OPS-ALL-PROTOCOL.md` (read it; this file only records g2d's choices). Written 2026-09-24
before any g2d job. Namespace `/cluster/tufts/paralab/tawal01/opsall_20260924/g2d/<job>/`, jobs `g2d_*`,
at most 3 g2d jobs running or pending (`cluster/submit.sh` refuses a fourth). Sources: `COPIED-FROM.json`
(every file is a byte-exact `git show` of a pinned commit; `scripts/assemble.py`). Changes to copied
files are listed at the end of this file.

## Cells and what each Table 1 row is

| cell | Table 1 source | cohort | NM-ROM accurate / fast | Table 1 FOM |
|---|---|---|---|---|
| Burgers 512² | burgers2d-speed `b512`, job 4241035 (`selection-512.json`) | dev6 (`params_draw(7090702,4)+params_draw(911702,2)`, hash `108f12dc…`) | `R384_lin_M1536_lat64_g0p01_fast_chol_clip_lamcarry_pred2__eng` / `R128_lin_M512_lat64_g0p01_fast_chol_clip_lamcarry_pred2__eng` (engineered solver, default mode) | Newton–BiCGStab `lean_nt3e-3_l3e-3_dt005`, graphs mode |
| Burgers 4096² | burgers-bank-knob `bk4096b`, job 4197473 (paper row = its k≥j+1 sensitivity table) | dev6 | `R384_lin_M1536_lat64_g0p001_fast_chol_clip_lamcarry_pred2` / `R128_lin_M512_lat64_g0p001_fast_chol_clip_lamcarry_pred2` (bankknob.py solver) | `lean_nt3e-3_l3e-3_dt005` |
| Heat 4096² | heat-bank-knob `h2d`, job 4197350 | sealed held-out 16 (seed 791099) | `lin_R128_cn` / `lin_R48_cn` (span of the ordered wide2d bank, CN) | CN–CG Δt 0.025 rtol 1e-3 |

Note: the coordinator named burgers-bank-knob for 512² too; the printed 512² row (0.19 %/0.60×, 1.75 %/1.59×)
is burgers2d-speed's job 4241035 (same model, same settings rule, engineered solver), so that lane's code
and arms are used at 512².

Error metric: Burgers — worst over dev6 of $\max_{k\ge1}\lVert u(t_k)-u_{\rm ref}(t_k)\rVert_2/\lVert u_0\rVert_2$,
reference `fft_tight` solved in the same job; heat — worst over the 16 cases of the max over all six
times of the current-relative error against the exact same-grid DST flow (heat-compare-hires metric).

## Operators (no tuning)

Configurations verbatim from the neighbouring meshes' operator panels (the only size trained there,
hence also the "smallest trained size" for the 4096² attempts): Burgers `fno-large`, `unet-refine`,
`tsol-refine`, `don-small` (burgers-compare-hires `specs/train.json`); heat `fno`, `unet`, `transolver`,
`deeponet` (heat-compare-hires `configs/ops`). AdamW, config seed, 3000 s wall budget, checkpoint =
best validation mean-case-max on the training family's validation split. Training data = the source
lanes' operator training sets regenerated at the target mesh (Burgers: pinned 128/32 index, `fft_tight`;
heat: seeds 791000×512 / 791001×16, exact flow on the fly). One training job per network, except
Burgers 4096², where the 150 GB f64 training set cannot be written to the shared disk (~370 GB free
for four groups): one H200 job generates it once into host memory and trains the four networks
one after another, each with its own 3000 s budget (declared deviation).

## Panel job per cell (one GPU allocation)

1. The source lane's NM-ROM/FOM driver, trimmed to the two Table 1 NM-ROM arms and the FOM grid
   (Burgers: `b2speed.py` at 512², `bankknob.py` at 4096²; heat: heat-bank-knob `hbk_run.py` with only
   `lin_R128_cn`, `lin_R48_cn`, its CN–CG grid and the DST control; then heat-compare-hires `panel.py`
   with only its operator blocks).
   Same truth solve, same timing contract as the source job.
2. Operators timed in a second process in the same allocation (Burgers: `ops/optime.py`, 20 burn-in,
   5 reps × 6 cases; heat: inside `panel.py` as in heat-compare-hires). Errors scored with the same
   formula against the same in-job reference.
3. Speedup = the fastest tested FOM at least as accurate as the NM-ROM accurate setting (same job),
   divided by each method's median GPU ms.

A family that cannot train or run (OOM, time) is recorded with its error message as the result. No
number from another mesh or another job fills a cell.

## Deviations / modifications to copied files

(appended as they happen)
- `b2speed.py`, `bankknob.py`: one added line after the truth solve, `opcohort.write(...)` (`scripts/opcohort.py`, the
  `opcohort` block of burgers-compare-hires `cmp.py`), so operators are scored against the job's own `fft_tight`
  arrays. Nothing else in either driver is changed (sha256 before/after in `COPIED-FROM.json`).
- Configs (all in `configs/`, generated from the source configs by the snippets recorded in their `g2d_note`/`purpose`):
  `pb512.json` = burgers2d-speed `config-512.json` with only the two selected `__eng` arms, certificates skipped
  (the source job certified them; same model, same rule), parity pairs dropped; FOM grid, both compile modes, timing
  phases (A-B-A, 5/3 reps) unchanged. `pb4096.json` = burgers-bank-knob `config-h64a.json` (the lane's own A100-80G /
  hold-out mechanics: bank in 32 row blocks, certificates skipped, timed invocations not re-scored, `fft_tight`,
  `lean_tight`, `lean_nt1e-4_dt005` untimed) with the dev6 cohort and only the two Table 1 linear arms.
  `ph4096_hbk.json` = heat-bank-knob `h2d.json` restricted to 4096², the held-out cohort, R' ∈ {128, 48} linear CN
  rungs; `ph4096_ops.json` = heat-compare-hires `pn2048.json` with only the operator blocks at 4096².
- Operator timing is a second process after the NM-ROM/FOM driver in the same allocation (as in burgers-compare-hires
  Phase O and the heat panel's operator blocks). Burgers operators are timed by `ops/optime.py` (20 burn-in, 5 reps per
  case) and scored by `scripts/opscore.py`.
- Burgers 4096² training: `scripts/train_mem.py` (in-memory data, ops/train.py `train()` unchanged; see its docstring).

## Expanded mandate (2026-09-24 ~17:15 EDT, coordinator; recorded before any expanded job)

Goal: cost to reach a target worst error {10, 5, 1, 0.5} % per method, Burgers 2D 256²–4096², Heat 2D 1024²–4096².
Heat 256²/512²: the heat-bank-knob lane ran only 1024²/2048²/4096² (h2d.json `meshes`), so there is no Table 1
NM-ROM/FOM row at 256²/512²; skipped and reported.

- **Group cap 6.**
- **Operator size ladders.** Burgers adds `unet-large` (base 48), `unet-b64` (base 64) and `tsol-large` (dim 256), all
  ops-tune-grid published configs, plus `fno-w96f32` (ops-tune-grid `fno/width96` trained in float32 through a new
  `F32FNO` wrapper: upstream SpectralConv in complex64, f64 features in and f64 output out). At 256² the
  ops-timing-panel opt201 ladder already exists (13 checkpoints: unet small/medium/refine/large, tsol
  small/medium/refine/large, don small/medium/refine/large, fno-large), so only `unet-b64` and `fno-w96f32` are
  trained there. At 512² every size is trained by g2d: the four base sizes again, because the earlier g2d 512²
  checkpoints were deleted after pb512 (disk rule), plus the four larger rungs. At 1024²/2048² the existing
  compare-hires base checkpoints are reused. Heat adds `unet-large`, `unet-b64` and `fno-w96f32` at 1024² and
  2048². The heat transolver is already the `large` config and heat DeepONet is last priority, so neither gets a
  larger rung. Base FNOs stay float64 (the checkpoints that already exist). All new trainings: AdamW, config seed,
  3000 s, best validation checkpoint. Burgers ones use `scripts/train_mem.py` (data in host memory, never on disk).
- **One new panel per cell** (`pl<n>` Burgers, `phl<n>` heat):
  - Burgers: `bankknob.py` with burgers-bank-knob `config-<n>.json`, keeping arm families (a), the head at every
    R', and (b), the span at every R'. Certificates are skipped. Every FOM setting is timed except the `fft_tight`
    reference. Then every operator size is timed with `optime.py` and scored with `opscore.py`. Checkpoints are
    verified by SHA256 against the training record (`scripts/stage_ops.py`).
  - Heat: `hbk_run.py` with every linear-span R' (128, 96, 64, 48, 32, 16) in CN plus the head-only
    `nmrom_R128_q0_cn` and the full CN–CG grid, then `panel.py` with every operator size.
  - At 256²/512², bankknob's span arms at `g0p01_x1` (exact first step) are the bk jobs' arms, not burgers2d-speed's
    engineered Table 1 path. The Table-1 engineered numbers at 512² are in `pb512`.
- Burgers 4096² retry `t4096rt` (tsol-refine, don-small) with `XLA_PYTHON_CLIENT_ALLOCATOR=platform`: in b4096tc both
  hit CUDA OOM after the JAX data-generation pool and the U-Net run had used part of the 80 GB.
- `ph4096` (narrow heat panel) was cancelled after ~1 min of running and deleted; it was superseded by `phl4096`.
  No number from it exists. `pb4096` failed with a JAX OOM in the truth solve (XLA fraction 0.90 on an A100-80G).
  The `pl4096` body uses 0.95.
- Outputs: `cost_points.json` (coordinator schema + `operator_settings` + `cost_to_target`) from
  `scripts/summarize_cost.py`; auto-regenerated by `cluster/autocollect.sh`.

## Heat extension (coordinator, ~19:50 EDT)

- Heat 2D 256² and 512² are full cells like the other heat cells: frozen wide2d model, the hbk span ladder, head-only,
  CN, the same 16 held-out cases, the CN–CG grid, and operators trained at the mesh. Each has 6 networks, one per
  job: `fno`, `unet`, `transolver`, `deeponet`, `unet-large`, `fno-w96f32`. `unet-b64` is left out for time; at
  Burgers 512² it was worse than `unet-refine` under the 3000 s budget. There is no Table 1 row at these meshes, and
  `cost_points.json` notes it.
- Heat 8192² (`phl8192`, H200, 240G): the NM-ROM ladder and CN–CG grid. Transolver is the only operator known to fit
  at 4096², at micro-batch 8 on an A100-80GB, so it is attempted once (`h8192tran`, H200). The other families are
  "not attempted: out of memory already at 4096²" (FNO and DeepONet at micro-batch 1). The U-Net fitted at 4096²
  only at micro-batch 1 on an H200, which is about 4× short of what 8192² would need.
- Coordinator rule (~18:25): one network per GPU allocation. Every g2d job follows it except `b4096tc` (before the
  rule). That job trained its networks one after another, not concurrently: FNO OOM at start; U-Net got the full
  3000 s (12 epochs); Transolver and DeepONet then OOMed at start. The retry is split into `t4096ts` and `t4096dn`.
- 2026-09-24 ~20:00: Heat 8192² dropped by the user (coordinator); never submitted.
- ~21:00 `phl4096` (4303352): the hbk part finished, but the operator process OOMed at the first U-Net. `panel.py`
  imports JAX, and g2d's submit script did not set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (heat-compare-hires' did), so
  JAX reserved 90 % of the GPU. This is a g2d bug, and every heat panel body now sets the variable. Before I looked at
  the failure, I deleted the remote 4096² U-Net and Transolver checkpoints (a g2d mistake), so both are retrained
  (`h4096u2` on an H200, `h4096t2` on an A100-80GB, same configs) and the whole cell reruns as `phl4096b`. None of
  phl4096's numbers are mixed with phl4096b's.
- ~21:35 coordinator: group cap raised to 10.
- ~22:10 user (coordinator): Burgers 2D only. Heat jobs cancelled with scancel (squeue before/after checked):
  4308349 h4096u2, 4308141 h2048ff32, 4307977 h2048ub64, 4307919 h2048ul, 4307409 h512ff32, 4307420 h512deep,
  4307200 h512ul. All were RUNNING, their outputs were never used, and their remote dirs were deleted. The finished h512fno/unet/tran
  trainings are unused and deleted. phl512, phl2048 and phl4096b were never submitted. Heat stays at the two completed
  cells, 256² and 1024². Freed GPUs go to Burgers 2048²/4096². At 4096², `unet-large` and `fno-w96f32` are
  attempted once each (`t4096ul`, `t4096ff32`, A100-80GB, platform allocator), next to `t4096ts` and `t4096dn`.
- ~22:45 Burgers 4096² trainings on A100-80GB, each in a fresh process with the platform allocator: `tsol-refine`
  (patch 4 = 1024² tokens), `don-small`, `unet-large` and `fno-w96f32` all hit CUDA OOM at micro-batch 1. The only
  4096² operator is `unet-refine` from b4096tc (12 epochs).
- ~23:00 `pl4096` (4309401, A100-80GB): the driver OOMed building the M=2048 test operators of the R'=512 span arm
  (bank 69 GB). Its opcohort had already been written, so its U-Net was timed, but there is no NM-ROM/FOM, so the job
  is not used. **Rule, fixed before either new job produced a number:** `pl4096b` (4309865, H200, full ladder) is the
  4096² panel if it completes. `pl4096c` (4309893, A100-80GB, the same ladder without the R'=512 span arm) is used
  only if pl4096b fails. Ratios are never mixed across the two.
- ~23:10 `pl4096c` (4309893, A100-80GB) also OOMed in the driver's test-operator build (5.5 GiB, bankknob line 254), even without the R'=512 arm. It is unused. `pl4096b` (H200) got past the quick phase.
