# MASTER RESULTS RECORD — NeurIPS 2026 submission 25242
**Snapshot taken 2026-07-31, before anything is written into the rebuttal.**

Everything below traces to a results JSON on the Tufts cluster under
`/cluster/home/tawal01/nmrom/Tunable-NM-ROM/results/raw/`. **1,739 result JSON exist;
195 are already quarantined in `_superseded*` directories and are excluded from every
table here.**

Companion files in this directory:
- `LADDER-RESULTS.md` — the full Burgers-2D write-up (ladder + addendum + corrections)
- `ladder_frontier.png` — accuracy vs solve time, four resolutions, all four methods
- `pareto_speedup.png` — accuracy vs speedup Pareto frontier, per resolution
- `pareto_summary.md` — the merged endpoint table with caveats
- `ladder_tables.md` — per-rung raw tables
- `PROVENANCE-summary-table.txt` — file path for every number in the cross-PDE table

---

# 🔴 BLOCKING — 11 speedups in the cross-PDE summary table cannot be traced to any file

Every **accuracy** value in the cross-PDE table matches a result JSON. **Eleven of the
thirty-three speedups do not** — not in a live directory, and not in a superseded one.
They were searched for at 6% tolerance across all 1,739 files.

| cell | method | rel-L2 (traceable) | table speedup | speedups actually in the JSONs |
|---|---|---|---|---|
| Poisson-2D N=64 | NM-ROM best-acc | 5.73e-4 ✓ | **0.6×** | 0.866 |
| Poisson-2D N=128 | NM-ROM best-speed | 6.19e-2 ✓ | **19.6×** | 0.023, 12.6 |
| Poisson-2D N=128 | FNO | 3.30e-3 ✓ | **4.1×** | 2.6, 3.07 |
| Poisson-2D N=256 | NM-ROM best-speed | 4.01e-2 ✓ | **40.7×** | 0.068, 24.6 |
| Poisson-2D N=256 | FNO | 3.33e-3 ✓ | **16.7×** | 10.1 |
| Poisson-3D N=32 | NM-ROM best-acc | 8.41e-3 ✓ | **0.8×** | 0.104, 0.202, 0.232, 0.521 |
| Poisson-3D N=32 | NM-ROM best-speed | 1.54e-1 ✓ | **5.0×** | 3.29 |
| Poisson-3D N=32 | FNO | 4.30e-3 ✓ | **1.3×** | 0.876 |
| Poisson-3D N=64 | NM-ROM best-acc | 7.57e-3 ✓ | **0.6×** | 0.517, 0.991, 1.08 |
| Wave-2D N=64 | NM-ROM best-acc | 4.71e-3 ✓ | **0.5×** | 0.689 |
| Wave-2D N=64 | FNO | 2.18e-1 ✓ | **3.7×** | **101** |

The Wave-2D N=64 FNO row is the starkest: the table says **3.7×**, the only JSON with that
accuracy says **101×** — a 27× discrepancy, in the *conservative* direction, which suggests
the table may hold a re-timed value whose JSON was never written or has been lost.

**This is the same error class the project already found once** ("a posted accuracy value,
Heat-2D N=128 = 3.75e-3, was found to exist in no result file at all"). Every one of these
must be re-derived from a run whose JSON is on disk, or withdrawn, **before the table is used.**

The 22 rows that DO match are listed with their file paths in `PROVENANCE-summary-table.txt`.

---

# 1. CROSS-PDE SUMMARY TABLE (as supplied), with Burgers-2D appended

Burgers-2D rows are from tonight's ladder and are fully traceable. Speedup is against the
**fastest measured gate-passing FOM at that cell**. Other rows carry the caveat above.

| cell | FOM | NM-ROM best-acc | NM-ROM best-speed | FNO (tuned) |
|---|---|---|---|---|
| Heat-3D N=128 | 122 ms | 1.83e-1 @ 4.12× | 1.96e-1 @ 3.91× | no data |
| Heat-3D N=64 | 25 ms | 7.26e-2 @ 0.48× | 7.67e-2 @ 1.30× | no data |
| Heat-3D N=32 | 13 ms | 9.60e-2 @ 0.52× | (same point) | no data |
| Heat-2D N=128 | 38 ms | 8.64e-3 @ 1.08× | 1.27e-2 @ 1.21× | no data |
| Heat-2D N=64 | 16 ms | 5.01e-3 @ 0.38× | 1.09e-2 @ 0.46× | no data |
| Poisson-2D N=64 | 4.8 ms | 5.73e-4 @ 0.6× ⚠ | 1.11e-1 @ 7.5× | 2.42e-3 @ 1.5× |
| Poisson-2D N=128 | 14.9 ms | 6.90e-4 @ 1.3× | 6.19e-2 @ 19.6× ⚠ | 3.30e-3 @ 4.1× ⚠ |
| Poisson-2D N=256 | 32.7 ms | 1.32e-3 @ 2.6× | 4.01e-2 @ 40.7× ⚠ | 3.33e-3 @ 16.7× ⚠ |
| Poisson-3D N=32 | 4.2 ms | 8.41e-3 @ 0.8× ⚠ | 1.54e-1 @ 5.0× ⚠ | 4.30e-3 @ 1.3× ⚠ |
| Poisson-3D N=64 | 7.3 ms | 7.57e-3 @ 0.6× ⚠ | 2.43e-1 @ 8.1× | 5.11e-3 @ 1.4× |
| Wave-2D N=64 | 10.7 ms | 4.71e-3 @ 0.5× ⚠ | 2.66e-2 @ 2.0× | 2.18e-1 @ 3.7× ⚠ |
| Wave-2D N=128 | 25.2 ms | 4.94e-3 @ 1.5× | 3.01e-2 @ 7.8× | 2.24e-1 @ 9.4× |
| Wave-2D N=256 | 53.4 ms | 1.63e-2 @ 3.3× | 4.36e-2 @ 16.0× | 2.27e-1 @ 20.5× |
| **Burgers-2D N=64** | **176 ms** | **1.34e-3 @ 3.6×** | **1.93e-1 @ 20.6×** | **1.45e-2 @ 35×** |
| **Burgers-2D N=128** | **769 ms** | **3.08e-3 @ 1.0×** | **1.56e-1 @ 30.3×** | **1.30e-2 @ 161×** |
| **Burgers-2D N=256** | **1064 ms** | **5.00e-3 @ 0.36×** | **1.41e-1 @ 99.0×** | **1.18e-2 @ 174×** |
| **Burgers-2D N=512** | **2588 ms** | **2.11e-2 @ 3.8×** | **6.62e-2 @ 41.8×** | **2.73e-2 @ 315×** |

⚠ = speedup not traceable to any JSON (see BLOCKING section).

**Why Burgers looks so different:** FNO inference is ~3–8 ms on *every* PDE. What changes is
the FOM it divides by. Poisson/Wave FOMs are 4.8–53 ms, so FNO gains only 1.3–20×. The
Burgers FOM is 176 ms–2.6 s (50 backward-Euler steps, each a Newton solve), so FNO gains
35–315×. **"Faster than FNO" is true on Poisson and Wave and false on Burgers** — it cannot
be a blanket claim.

---

# 2. BURGERS-2D LADDER — tonight's work (fully traceable)

Full write-up: `LADDER-RESULTS.md`. Headlines:

- **Four rungs, four methods, one shared dataset per rung.** Data at
  `/cluster/tufts/paralab/tawal01/b2data/burgers2d_n{64,128,256,512}/`, generated float64,
  stored float32; sha256 recorded for every artifact.
- **Tunable frontier from ONE stored artifact per rung** (`pod_k1024.npz`, sliced at
  deployment): cost range 29× → 234× as resolution grows.
- **We beat tuned FNO on accuracy at every rung** while staying 3.3–7.8× cheaper than the FOM
  — but FNO is 17–83× faster in absolute wall-clock.
- **POD-Galerkin+DEIM has no usable operating point anywhere**: diverges at 9/18/14/4 points
  per rung, best surviving point 30–52% error, worsening with resolution.
- **Out of distribution we beat FNO on all four axes**: our worst degradation 3.3× against
  FNO's 16.7×; on `nu_high` the ranking reverses 22×.
- **The nonlinear chart is data-starved, not wrong**: POD+MLP gain over the POD floor climbs
  as n_train^0.39 with no plateau (1.50 → 3.57 at k=64 over 8× data), and the matched-solve
  ordering reaches parity at n_train=4000, k=256.
- **float64 is FASTER than float32** for the Burgers FOM: 3.66 s/trajectory vs 163.69 s at
  N=512, and float32 misses `NEWTON_RTOL=1e-10` on 50/50 steps at every rung.

## Errors found and corrected tonight

1. **Sixth denominator error, committed by the orchestrator.** The k-extension jobs recorded
   speedups against the shipped FOM at N=64/128 where `incr_1e-2` is faster. "k=512 beats FNO
   at 4–6×" is really **1.3–1.6×** there.
2. **`incr_1e-2` reverses with resolution**: 5.2× faster at N=64, 3.3× at N=128, **16.6×
   slower** at N=256, **~62× slower** at N=512 — and it passes the convergence gate every
   time. 32 inflated ratios (up to 17,480×) withdrawn in place.
3. **A fifth instance that depends on the test set, not the mesh**: on the `nu_high` OOD axis
   at N=128, `incr_1e-2` is 7.72× slower, giving 25.57× instead of 3.18×.
4. **The N=512 cell was not method-matched** — our frontier used a different parameter draw
   (worth 1.9×) from the one FNO scored. Closed by re-running on FNO's exact trajectories.
5. **A `horizon2x` indexing bug** in the orchestrator's own follow-up: `run_rom_pod.py`
   hardcodes `NUM_STEPS=50` and strides by 51, but that axis stores 101 rows per trajectory.
   Two records ~20× too large, withdrawn in place.
6. **Home directory over quota** (28.6 GB) killed 8 jobs and produced one corrupt dataset.
   Data moved to paralab; `~/nmrom/b2data` is now a symlink.

## Reproducibility findings that must travel with any number

- **rel-L2 reproduces to the last stored digit** across four GPU models (a100-PCIE, a100-SXM4,
  h200, l40s); per-trajectory agreement ~1e-15.
- **Speedups do not.** The same configuration measured **4.27× on an A100-PCIE and 0.70× on an
  L40S** — a 6.4× swing, because the ROM's dense f64 normal equations collapse on a card with
  ~1/64 fp64 throughput while the memory-bound FOM gets slightly faster. **Any posted speedup
  must name its GPU.**
- Collocation-draw spread is small: 6.6% (max−min)/median over 5 seeds.

## Known method/description mismatch — needs a framing decision

The Burgers-2D lane runs a **frozen linear POD chart** with **uniform-random collocation and
uniform weights** — verified in `burgers/scripts/run_rom_pod.py` and `solver/pod_solver.py`,
and stated in the driver's own docstring ("plain-random EQ; NNLS not used here"). The paper
describes a nonlinear manifold ROM with NNLS-based empirical quadrature. Three independent
lines confirm the learned decoder does not help here at n_train=500 (ViT-CP diverges; the
matched N=64 ablation has frozen beating learned-CP; POD+MLP loses by 1.87–4.34×).

A wrong speedup can be corrected in a discussion thread. A mischaracterised method cannot.

---

# 3. WHAT IS STILL OPEN

| item | status |
|---|---|
| **11 untraceable speedups in the cross-PDE table** | 🔴 BLOCKING — re-derive or withdraw |
| **Independent adversarial audit** | never run; stopped at user request to free compute |
| Heat lane has **no FNO/DeepONet data at all** (5 cells) | gap in the cross-PDE comparison |
| POD-DeepONet has no FOM reference at N=512 | no speedup for that cell |
| 86 torch rows carry a speedup with no protocol stamp | `device_sync`/`backend` unverifiable |
| n_train=8000 confirmation of the nonlinear crossover | data generating (seeds 5007–5014) |
| Batched + fp32 ROM to close the FNO wall-clock gap | in progress (Agent K) |
| N=512 uses n_train=200 vs 500 elsewhere | within-cell comparisons unaffected |
| Single seed per operator-baseline configuration | unquantified variance |

---

# 4. WHERE EVERYTHING LIVES

```
cluster results   /cluster/home/tawal01/nmrom/Tunable-NM-ROM/results/raw/
  burgers ladder    b2ladfront_* b2ladkext* b2ladpodg* b2ladfno_* b2laddeon_*
                    b2ladood* b2ladoodk_* b2ladh2xI_* b2podmlp_* b2datascale_*
  superseded        _superseded*  (195 files, excluded from every table)
cluster data      /cluster/tufts/paralab/tawal01/b2data/burgers2d_n{64,128,256,512}/
                    train.npz  val.npz  pod_k256.npz  pod_k1024.npz
                    b2scale/    (the n_train = 1000/2000/4000 scaling study)
helper scripts    /cluster/home/tawal01/nmrom/lad/       (build_pod_snap, gen_val, podmlp, ...)
slurm             /cluster/home/tawal01/nmrom/slurm/     (b2*.slurm)
logs              /cluster/home/tawal01/nmrom/slurm-logs/
local analysis    scratchpad/lad/  (synth.py, pareto.py, LADDER-CONTEXT.md,
                    ORCHESTRATOR-NOTES.md, ood_tables.md, frontier/podg/fno/deeponet tables)
```

---

# 🔴 BLOCKING #2 (added 2026-07-31, after Agent K) — the FNO accuracy comparison was NOT like-for-like, and the error favours us

Every posted Burgers-2D NM-ROM accuracy at N=64/128/256 is a mean over **3–5 validation
trajectories** (`run_rom_pod.py` defaults to `--num-trajectories 5`; the N=256 arms used 3).
The FNO rows they were compared against are means over **50**. Verified directly from the
JSONs:

| rung | our trajectories | FNO trajectories |
|---|---|---|
| N=64 | **5** | 50 |
| N=128 | **5** | 50 |
| N=256 | **3** | 50 |
| N=512 | 20 | 20 ✓ (the only matched rung) |

`train.npz`'s and `val.npz`'s `U_val` are **bit-identical** at N=128 (max abs diff 0.0), so
this is purely a test-set-SIZE mismatch, not a different sample.

**Measured effect at N=128, k=512, n_eq=8192** (Agent K, job 2107929, A100-PCIE-40GB — the
same device the N=128 FNO row was measured on):

| test set | our rel-L2 |
|---|---|
| 5 trajectories (as posted) | 9.642e-3 |
| **50 trajectories (matched)** | **2.4336e-2** |

Against FNO's 1.3043e-2 over the same 50: **FNO is 1.87× MORE accurate, not 1.35× less.**

## What this invalidates

**"We beat FNO on accuracy at every rung" does not survive as stated.** It was true only on
the 3–5 trajectory subsets. Affected: `LADDER-RESULTS.md` ADDENDUM §A and §B, correction §C2,
`pareto_summary.md`, the "ours, beating FNO" column of the merged table, and the orange stars
in `pareto_speedup.png`. **N=512 is unaffected** — it was matched at 20 vs 20, and there ours
is 2.108e-2 against FNO's 2.730e-2, still a genuine win.

Re-evaluation over all 50 trajectories at N=64/128/256 is running (`b2match50.slurm`,
job 2111033 onward). **Until it lands, quote no FNO accuracy comparison except at N=512.**

This is the project's own error class — comparing numbers computed on different test sets —
and it is the seventh instance found. It was found internally and goes against us.

---

# ADDITIONAL FINDINGS FROM AGENT K (inference speed)

## The launch-bound premise was wrong
`H = JᵀWJ` is **98% of the per-build flops**. The solve runs at 16–22% of A100 fp64 peak
(0.954 ms/GN-iteration measured against 19.5 TFLOP/s peak) — that is *not* the launch-bound
signature seen in POD-DeepONet, so batching's whole ceiling is ~2×, not 10×.

## Achieved: 2.12×, against a ~32× target

| step | rel-L2 (50 traj) | s/traj | cumulative |
|---|---|---|---|
| published (B=1, f64, tol 1e-6) | 2.4336e-2 | 0.22350 | 1.00× |
| + vmap batch 32 | 2.4336e-2 | 0.14094 | 1.59× |
| + gn_rel_tol 1e-4 | 2.4336e-2 | **0.10536** | **2.12×** |
| + float32 | 2.4337e-2 | 0.15709 | 1.42× (worse) |

Batching is **exact, verified not assumed**: batched-vs-sequential max state deviation
8.6e-15 in f64, and per-step GN iteration counts identical (dIt = 0 at every batch size).
Amortisation saturates by B=20.

Like-for-like vs FNO at N=128 (both fp32-standard, batch 32, same A100): ours 2.4336e-2 @
0.10536 s vs FNO 1.3043e-2 @ 0.004781 s — **22.0× slower** (was 46.7×).

## float32 does not help on an A100, but helps enormously on other cards
A100's fp64 *tensor* peak (19.5 TF/s) equals its fp32 non-tensor peak, so fp32 buys at most
1.6× there and at the published tol=1e-6 it **cannot converge at all** (1598/1600 steps at the
cap; the gradient norm hits the fp32 roundoff floor). On a GB10 the same matched comparison
gives fp32 **14.7×** faster than fp64. `tensorfloat32` reaches 21.8 TF/s but **changes the
answer by 0.6–1.1%** and fails the gate — not usable.

## Both algorithmic levers are negative
- **Quasi-Newton** (H + Cholesky once per BE step): GN iterations 2.27 → **7.40**, 358/2500
  steps at the cap, never converges, and is *slower* at B=32. The iteration rise more than
  cancels the 3× per-iteration saving.
- **NNLS empirical quadrature**: all six configurations **diverge** (rel-L2 0.404–1.86, every
  arm at the cap). On held-out states the EQ quadrature reproduces the full-grid reduced
  gradient only 1.03–1.59× better than uniform-random of the same size. **Uniform-random at
  2048 nodes converges cleanly** (5.154e-2). Random collocation beat this EQ implementation at
  every node count tried. Caveat: this was a newly written gradient-matching sketched builder,
  not the repo's `eq/nnls.py` path, so it is evidence about *this* implementation.

## Eighth denominator-trap instance — now batch-size dependent
Batched FOM: shipped 0.9084 → 0.42146 s/traj, but `incr_1e-2` 0.35643 → **0.52095** (gets
slower with batching). The fastest gate-passing variant is `incr_1e-2` at B=1/5/20 and
**shipped at B=32**. The driver now selects per batch size automatically.

## Cluster note
The `normal` QOS scheduled nothing for this account all session (900+ jobs pending
cluster-wide). All of Agent K's completed work ran under `--qos=interactive`
(1 GPU, 64 GB, 4 h, one job at a time). N=64/256/512 jobs were queued and never ran.
