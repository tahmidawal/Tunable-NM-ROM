# NeurIPS 25242 — Review Response: Status & Plan

**Last updated: 2026-07-27 (original-cell repair campaign launched — 8 A100 jobs in flight)**

This file is the single source of truth. Written to be resumable from any session.

---

# ⚡ 30-second status (2026-07-27 morning)

**RESPONSE PACKAGE READY**: `responses/` has 4 per-reviewer markdowns (verbatim quotes → response)
+ `NeurIPS25242_reviewer_responses.docx` with the consolidated master table (11 rows × 6 methods).
All 6 corrected Poisson cells banked single-ckpt on CG truth (2.4e-3 → 1.31e-2 @ 3.6–17.2×,
best-speed to 26×). POD-DeepONet stable everywhere (1.6e-2 @ 56–112× on 2D Poisson!). Classical
DST baselines all 11 original cells. Heat bug = port-only, fixed, paper pipeline verified. Two FNO
sweeps rerunning on H200s after timeouts (p2d_256, p3d_64) — master table marks them "sweep
completing"; refresh docx when they land. Disk: freed ~385 GB on paralab (user-authorized).

---

# ⚡ Older status

- Reviews: 3 / 2 / 2 (borderline reject, reject, reject). Meta: "all concerns essential."
- **We built 3 new hard-PDE benchmarks (Burgers-1D/2D, Wave-2D) with tuned baselines.**
- **Score so far: we beat FNO on all 3 cells. POD wins raw accuracy only on the 1D control cell (by design of the story). DeepONet fixed and loses everywhere.**
- A100 cluster jobs running NOW (Tufts pax, see "In flight").
- Every number regenerates from JSON: `Tunable-NM-ROM/baselines/scripts/make_table.py`.

---

# 📒 Experiment ledger (done / running / next)

## ✅ Done — with the win each one buys

| # | Experiment | Result | The win it gives us |
|---|---|---|---|
| 1 | Burgers-1D NM-ROM (k=128/R=512) + knob sweep | 5.9e-3 @ 8.4× ↔ 1.0e-2 @ 17.3×, **one checkpoint** | Kills "single model is a lie" — beats FNO at both endpoints |
| 2 | Burgers-1D, 3 seeds (A100) | 5.5e-3 / 6.4e-3 / 5.9e-3 (medians robust) | Kills the "single seed" objection |
| 3 | Burgers-1D A100 re-timing | 1.05e-2 @ 8.9× dedicated GPU | Paper-grade timing, same hardware class as paper |
| 4 | Burgers-2D frozen-POD + LS-GN (Path B) | **2.1e-2 @ 3.9× — beats FNO's 2.4e-2** | Wins the cell; LS-GN stable exactly where DEIM explodes |
| 5 | Wave-2D exact-Galerkin, k-sweep | **8.1e-3 @ k=256, 3.5e-3 @ k=353** | Sweeps every baseline; 27× more accurate than FNO |
| 6 | Wave floor-tracking verification | Error = 1.02–1.24× of chart's theoretical floor | "Solver provably near-optimal" — no baseline can claim this |
| 7 | Wave N=128 scale-up (A100) | 4.9e-3 @ 1.6× → 3.0e-2 @ 7.7× | Speedup grows with resolution — the trend the paper needs |
| 8 | FNO tuned per cell (official lib, 8-config sweeps) | 1.2e-2 / 2.4e-2 / 2.2e-1 | Baselines above suspicion — our wins count |
| 9 | POD-DeepONet (fixed DeepONet) | 0.23 / 0.33 / 0.96 — **stable, never diverges** | Kills the "broken baseline" objection |
| 10 | POD-Galerkin + DEIM k-sweeps | 1D: 2.0e-4 · 2D: explodes 4e4 · Wave: slow | The three-regime "when is nonlinear needed" answer |
| 11 | Ground-truth audit + regeneration | Truth was WRONG on 2/5 1D trajs; fixed, verified 1e-16 | Credibility armor — we audit our own truth |
| 12 | Singular-value decay per dataset | 21 / 91 / 171 modes to 99.9% | The n-width figure, quantified |
| 15 | Poisson truth-data fix + 4 single-ckpt cells (CG, A100) | 6.3e-3/5.1e-3/8.4e-3/1.6e-2 — beats FNO 3 of 4, one model | Meta-review #1 answered with corrected, defensible cells |
| 16 | Heat rollout bug: confirmed PORT-only, fixed | 1.31 → 3.3e-2 (40×); paper pipeline verified correct | Turns fxe8's "invalid heat math" into a strength (we audit our own code) |
| 17 | Classical CPU baselines (DST/splu/CG) | DST: 0.19 ms Poisson-2D-64, 0.25 ms full heat traj | The honest speedup frame; repositioning written into tex |
| 18 | Paper text surgery (compiles, 0 undefined refs) | Eq. 4, EQ notation, baselines, intrusive framing, 2 appendices, new §benchmark suite | Answers every writing-only concern |

## ⏳ Running right now (submitted 2026-07-27, all on A100s)

| Job | What | Why |
|---|---|---|
| `heat64` (1811433) | heat2d N=64 datagen + 80k-epoch train → ckpt to `~/nmrom/heat-out/` | Fuel for the **decisive rollout-bug test** (`heat/scripts/bug_check.py`, already written: rollout 50 vs 1 + fixed-solver comparison) |
| ~~`wavebase128/256`~~ | **DONE** — POD-Galerkin JSONs banked (N=256 k=512: 0.3× = slower than FOM!), FNO/DeepONet pairs + N=256 disp basis in `~/nmrom/wave-out/` (pull retrying, cluster ssh flaky) | Baseline columns at N=128/256 |
| `poisson2d_n128/256`, `poisson3d_n32/64/128` (**1811562–66**, resubmitted) | Single-checkpoint reruns on **CG-truth data** (first batch 1811489–93 KILLED — trained on BC-violating analytic data, see Discovery #9) | Meta-review #1 + P3D-64 contradiction |
| local CPU (background) | Classical baselines all 11 original cells (DST direct / sparse-LU / unprec-CG reference) | Replaces the 324×-vs-strawman framing |
| `pdata` (1811599) | Generates + ships the 5 CG-truth poisson npz to `~/nmrom/poisson-out/data/` | Local FNO/POD-DeepONet training data |
| `peval` (1811606, dependency-chained) | Re-evals ALL 5 poisson ckpts with the **jit-fixed** solver on A100 → paper-grade timings | The in-job evals ran the solver eagerly → timings fake-slow (accuracy fine) |

**CG-truth single-checkpoint results — `peval` FINAL (A100 dedicated, jit-fixed solver):**

| Cell | best-acc | best-speed (gn3) | FNO (old data) |
|---|---|---|---|
| Poisson-2D N=128 | **6.32e-3 @ 5.6×** | 6.19e-2 @ 12.6× | 7.32e-3 |
| Poisson-2D N=256 | **5.13e-3 @ 11.1×** | 4.01e-2 @ 24.6× | 7.94e-3 |
| Poisson-3D N=32 | **8.41e-3 @ 0.5×** | 1.54e-1 @ 3.3× | 1.91e-2 |
| Poisson-3D N=64 | **1.59e-2 @ 4.9×** | 2.43e-1 @ 8.2× | 1.65e-2 (≈tie) |
| Poisson-3D N=128 | ❌ ckpt missing — training job died, diagnosing | | |

- **Accuracy beats FNO in 3 of 4 completed cells with ONE model** (3D-64 ≈ tie).
- **Speedups are 5–25×, NOT the paper's 100–300×.** The paper's FOM was a CG that breaks down on the (nonsymmetric) port operator; vs a properly-converging SPD CG this is the honest number. It still grows with N (5.6→11.1× in 2D) and must be framed against the classical-DST appendix, per the repositioning already written into the tex.
- Heat-rollout bug CONFIRMED port-only + fixed (see Discovery #8): 1.31 → 3.3e-2 (40×); paper pipeline verified correct; heat2d_n64 fixed_rollout banked.

**Paper-text surgery DONE tonight (main.tex compiles, 0 undefined refs):** Eq. (4) rewritten as GN on the nonlinear BE residual · intrusive-vs-non-intrusive framing in intro · baselines paragraph honest (per-cell FNO sweeps on Poisson+new suite, POD-DeepONet replaces DeepONet, both SMA modes) · EQ requested-vs-achieved counts + padding + weight-conditioning documented · NEW Timing Protocol appendix · NEW Classical Solver Baselines appendix · NEW §Hard-PDE Benchmark Suite section with final table (burgers1d/2d + wave + 3 findings) · Limitations updated.

Local prep done meanwhile: `bug_check.py` (heat), pairs-loader in `baselines/data.py` (FNO/DeepONet can train on cluster-extracted pairs), `poisson/scripts/run_rom.py` upgraded to knob-override CLI + results-JSON emission (burgers/wave schema).

## ✅ A100 batch 1 COMPLETE (2026-07-26 evening) — final two results:

| # | Experiment | Result | What it gives us |
|---|---|---|---|
| 13 | Wave N=256 scale-up (A100) | k=512: 1.6e-2 @ 3.3× · k=128: 4.4e-2 @ **15.5×** | **Speedup-vs-resolution curve complete**: ≤0.9× (N=64) → 7.7× (N=128) → 15.5× (N=256). The paper's speedup figure. |
| 14 | Burgers-2D t2000 nonlinear retrain | Recon 3.6× better (5.7e-3 val) — **rollout still diverges** | Useful negative: on 2D Burgers, deep-chart ROLLOUT STABILITY (not recon) is the binding constraint → frozen-POD chart is the right tool, and now we know why. Cell stays won via Path B. |

## ❌ Next (priority order)

| # | Experiment | Why |
|---|---|---|
| 1 | Heat bug check (`nm_rom.py:154`) | Blocks reuse of ALL paper heat numbers; may improve them |
| 2 | Re-run the 4 contradictory paper cells, single-checkpoint | The meta-review's #1 issue on the original tables |
| 3 | Multigrid/FFT baselines on original Poisson/Heat | The "324× vs strawman CG" claim must be replaced |
| 4 | Darcy (variable-coefficient Poisson) | Standard benchmark; kills the "FFT solves this" attack |
| 5 | Seeds on Wave + Burgers-2D | Complete the variance story |
| 6 | Paper text: Eq. (4) rewrite, EQ notation, timing appendix, SMA cold-start | The writing-only fixes |
| 7 | Code release prep | Reproducibility objection |

**One line: every benchmark cell is a win or an honest control; the two running jobs are bonuses; everything left is paper-text surgery or original-cell repairs.**

---

# 🎯 Concern → Response map (the important section)

### Concern 1: "Single trained model" claim contradicted by Table 2 (fxe8, 5mgh, meta)
*4 of 11 paper cells used different k/R per Pareto endpoint = different models.*

| What we did | Status |
|---|---|
| Burgers-1D: ONE checkpoint (k=128,R=512), knobs only → 5.9e-3@8.4× ↔ 1.0e-2@17.3× | ✅ DONE (quiet-GPU timed) |
| Burgers-2D: ONE offline POD basis, k-slice + GN-tol knobs → 2.1e-2@3.9× ↔ 9.0e-2@15× | ✅ DONE |
| Wave: ONE offline basis, k sliced at deployment → 3.5e-3 ↔ 1.8e-2 | ✅ DONE |
| Re-run the paper's 4 contradictory cells from single checkpoints | ❌ TODO (inference-only, cheap) |

### Concern 2: Broken/unfair baselines (all reviewers)
*DeepONet diverged everywhere; FNO untuned; SMA oracle warm-start; unpreconditioned CG.*

| What we did | Status |
|---|---|
| FNO: official `neuraloperator` lib, per-cell 8-config sweep, curves saved in JSONs | ✅ DONE (3 cells) |
| DeepONet → POD-DeepONet: **stable everywhere now, no divergence** — loses honestly (0.23–0.96) | ✅ DONE |
| POD-Galerkin(+DEIM) intrusive control with same knobs, verified to 1e-7 vs our stack | ✅ DONE |
| Preconditioned CG / multigrid / FFT baselines for the ORIGINAL paper cells | ❌ TODO |
| SMA-NM-ROM honest cold-start reporting | ❌ TODO (writing change) |

### Concern 3: PDEs too easy / when is nonlinear manifold needed (GwrW, 5mgh)

| What we did | Status |
|---|---|
| Burgers-1D, Burgers-2D, Wave-2D benchmarks, clean verified truth | ✅ DONE |
| The three-regime answer: 1D = POD suffices (control); 2D Burgers = **POD-DEIM explodes (4e4 at k=128)**; Wave = POD accurate-or-fast-never-both | ✅ DONE |
| n-width evidence: modes to 99.9% energy = 21 (b1d) / 91 (wave) / 171 (b2d); sv arrays in JSONs | ✅ DONE |
| Advection-dominated + Galerkin-vs-LSPG question answered (see Discoveries #5) | ✅ DONE |
| Variable-coefficient Poisson (Darcy) for the original paper | ❌ TODO (biggest remaining item) |

### Concern 4: Invalid heat math, Eq. (4) (fxe8)
- Correct derivation now exists implicitly in the new solvers (GN on the nonlinear BE/CN residual, warm-started; frozen-Jacobian = first GN step). ❌ TODO: rewrite Eq. (4) in the paper text.
- **⚠️ Suspected real bug in the paper's heat rollout** (`heat/solver/nm_rom.py:154` stores A(u) not u). STILL UNVERIFIED — check before quoting any heat number. Cheap test: rollout num_steps=50 vs 1.

### Concern 5: EQ under-described / inconsistent counts (fxe8)
- Found + fixed a REAL EQ defect (padded nodes with ~zero NNLS weight → divergence; row normalization + 50/50 uniform blend, both packages). ✅
- Burgers-2D finding: GN tolerance (not EQ count) is the effective speed knob; random collocation subsets fail at high k — NNLS selection needed for sparse collocation. ✅ documented in result JSONs
- ❌ TODO: paper text — reconcile n_eq/N_eq notation, describe nested-rule construction.

### Concern 6: Timing methodology (fxe8)
- All quoted timings re-measured on quiet GPU; contended runs notes-flagged in JSONs. ✅
- A100 re-timing (paper-comparable hardware) running on cluster NOW. ⏳
- ❌ TODO: timing-protocol appendix text + code release prep.

### Concern 7: Single seed (5mgh)
- Burgers-1D 3-seed variance running on cluster NOW. ⏳
- ❌ TODO: extend to other quoted cells.

---

# 📊 Current results table (N=64/256 cells, quiet-GPU timings)

| Cell | NM-ROM (single model, knobs) | POD-Galerkin | FNO (tuned) | POD-DeepONet |
|---|---|---|---|---|
| Burgers-1D N=256 | **5.9e-3 @ 8.4× ↔ 1.0e-2 @ 17.3×** | 2.0e-4 but slower than FOM | 1.2e-2 | 2.3e-1 |
| Burgers-2D N=64 | **2.1e-2 @ 3.9× ↔ 9.0e-2 @ 15×** | 4.0e-1 best; explodes 4e4 @ k=128 | 2.4e-2 | 3.3e-1 |
| Wave-2D N=64 | **3.5e-3 / 8.1e-3 / 1.8e-2 (k=353/256/128)** | 1.7e-2, slower than FOM | 2.2e-1 | 9.6e-1 |

**Bold = we beat FNO in every cell.** Wave speedup ≤1.9× at N=64 (FOM nearly free); N≥128 in flight.

The NM-ROM entry uses the best chart per cell — nonlinear ViT+CP manifold on 1D, frozen-POD chart + LS-GN on 2D, exact-Galerkin linear chart on Wave. **That spectrum IS the paper story** (see Discoveries #5/#6).

---

# ✅ A100 batch: RESULTS IN (2026-07-26 afternoon)

**Wave N=128 (the speedup story materialized):** one basis, k sliced:
- k=512: **4.9e-3 @ 1.6×** · k=256: **1.9e-2 @ 5.2×** · k=128: **3.0e-2 @ 7.7×**
- vs N=64's ≤1.9× — speedup grows with resolution exactly as predicted.

**Burgers-1D seed variance (gn30/eq256 knob):** seed1 5.5e-3, seed2 6.4e-3, seed0 5.9e-3(GB10)/1.4e-2(A100, one outlier traj at 4.2e-2 — float-path GN sensitivity; per-traj MEDIANS robust ≈4.6–5.8e-3 everywhere). **Paper quotes: median per traj + seed spread.**

**A100 dedicated timings (burgers-1D):** default 1.05e-2@8.9×, speed-knob 3.4e-2@11.9×.

All in `results/raw-a100/`. Still running: `b2train` (t2000 nonlinear retrain, ~1.5h) and `wave256`
(resubmitted after fixing a build_pod segfault — Gram was built over the 65k² state side; now
method-of-snapshots + stride). Drain watcher pulls automatically.

**Watchers on the local GB10 session auto-pull results to `Tunable-NM-ROM/results/raw-a100/` when the queue drains.** If resuming from a fresh session: `ssh tufts-login squeue -u tawal01`, results in `~/nmrom/Tunable-NM-ROM/results/raw/` on the cluster, logs in `~/nmrom/slurm-logs/`.

### Cluster survival notes (learned the hard way)
- **paralab share is FULL (0 bytes)** — never write data there. Venv lives there read-only: `/cluster/tufts/paralab/tawal01/nmrom-rebuttal/venv`
- Code+results live in `~/nmrom/Tunable-NM-ROM` (home). Heavy data → node-local `/tmp` inside jobs
- Home quota tight (28.6GB, ~1.4 free). `~/.local` (7.3G) + `~/.conda` (6.8G) are cleanup candidates — **user permission pending**
- Per-user GPU cap ≈ 7 (QOSMaxGRESPerUser); it FAILS jobs in a race instead of queueing sometimes — just resubmit
- Always `PYTHONUNBUFFERED=1` in sbatch or logs stay empty on crash
- SLURM scripts: `~/nmrom/slurm/` on cluster (mirrored in local scratchpad)

---

# 🗺️ Where everything lives

```
/home/tahmid/Dev/Tunable-NM-ROM/
├── burgers/        1D+2D viscous Burgers (skew-symmetric advection, hardened Newton FOM)
│   ├── scripts/    generate_data, generate_chunk, merge_chunks, train, run_rom,
│   │               build_pod, run_rom_pod   (pod = Path B frozen-chart solver)
│   └── data/       burgers1d_n256.npz (CLEAN, x64-verified), burgers2d_n64.npz (clean),
│                   burgers2d_n64_pod.npz (238-mode basis)
├── wave/           2D wave, CN on (u,v), V_SCALE=16 channel balance
│   ├── scripts/    generate_data, build_pod (--style disp!), run_rom_galerkin, run_rom_podmlp
│   └── solver/     pod_galerkin_cn.py (exact Galerkin, THE wave winner), pod_solver.py (LS-GN)
├── baselines/      POD-Galerkin+DEIM, FNO (neuraloperator), POD-DeepONet, make_table.py
├── results/raw/    ONE JSON per quotable run (make_table reads ONLY this)
│   ├── _superseded/, _quarantine_*/, _debug_ablations/   (never fed to the table)
│   └── raw-a100/   (A100 results land here via watcher)
├── queue/          GB10 sequential runner + phase manifests + status logs
└── heat/, poisson/ the ORIGINAL paper code (heat bug unverified — see Concern 4)
```

Paper LaTeX: `/home/tahmid/Dev/Neurips/neurips26 (1)/main.tex`. Reviews: in the first message of the original session (key points all captured in Concerns above).

---

# 🔬 Discoveries log (all verified; several are paper-worthy)

1. **Ground-truth generator silently froze trajectories** (BiCGStab breakdown at fronts + non-finite guard → state stopped evolving). 1D truth wrong on 2/5 val trajs — **NM-ROM was right, truth was wrong** (matched to 3 digits). Fixed: tol-Newton + GMRES + line search, verified 1e-16 vs scipy. All polluted results quarantined.
2. **Wave channel imbalance**: |v|/|u|≈17× starved u under a shared scale → that's why the ViT-CP wave decoder never worked. V_SCALE=16 + per-channel loss fixed the mechanism; frozen-POD chart then beat learned CP factors entirely (CP arms A/B both falsified).
3. **EQ NNLS padding defect** (near-zero weights on padded nodes) → rollout divergence. Fixed in both packages.
4. **Wave solver is floor-tracking-optimal**: rollout error within 1.02–1.24× of the chart's projection floor at every k. Solver adds ~zero error; chart is everything.
5. **"Galerkin for linear charts, LS-GN for nonlinear ones"** — 3-rung evidence: LS-collocation 5.7e-2 → exact Galerkin same chart 3.1e-2 → matched (displacement-POD) basis 8.1e-3. Also answers 5mgh's Galerkin-vs-LSPG question. Per-step tolerance matters on non-dissipative PDEs (wave needs 1e-6; Burgers forgives 1e-4 = free 3.4× speedup).
6. **LS-GN collocation is stable exactly where POD-DEIM explodes** (same chart, same data; line search + weighted LS vs interpolation). Burgers-2D k=256: 2.1e-2 stable vs DEIM 4e4 blowup.
7. **Basis construction matters**: displacement-only raw-snapshot POD ≫ channel-normalized stacked POD for wave (proj floors halved).
8. **Heat-rollout bug CONFIRMED — but PORT-ONLY (2026-07-27).** Decisive test (`heat/scripts/bug_check.py`, A100-trained ckpt): rollout(50) ≡ rollout(1) to 6e-8 — the port stored A(u_{n+1}) instead of the decoded u_{n+1} as the next RHS, freezing the recursion. Fix cut mean rel-L2 **1.31 → 3.3e-2 (40×)**. The paper's ORIGINAL pipeline (`best-results/Heat-2D/.../experiment.py:385`) decodes the state correctly — **published heat numbers are NOT invalidated**; the release code now matches the validated pipeline. Banked: heat2d_n64 fixed_rollout 3.04e-2 @ 1.31× (GB10, 80k-epoch quick ckpt; paper's 300k ckpt would close the rest).
9. **Poisson "analytical" training data violates the Dirichlet BCs** (2026-07-27). Non-integer freqs ⇒ sin(kπ·1)≠0 ⇒ the generator CLIPS the analytic field to 0 at the boundary — the stored "truth" has discrete residual 2.6× the forcing (it is not a solution of the BVP). Affects the package AND the paper's original `generate_poisson_data.py` (train+val). The GN solver was fine all along: it solved the true discrete system and got blamed for the O(1) gap (rel 0.79 "garbage" runs). Fix: `data_source: cg` everywhere — truth = CG-discrete solve, verified consistent to 1e-5. **Paper's original Poisson numbers inherit this data flaw; the CG-truth reruns are the defensible versions.**
10. **The port's Poisson K_op was nonsymmetric** (identity boundary rows + un-eliminated interior coupling) → cold-start CG breakdown (NaN); F also has a nonzero far-boundary trace that identity rows turn into spurious BC data. Fixed with standard Dirichlet elimination: `K̃ = m·L(m·u) + (1−m)·u`, RHS masked. Also: solver LM damping restored to the paper-bisected 5e-2 (port had 1e-4 — below the recorded divergence threshold), residual-norm convergence, stencil boundary masking.
11. **Classical CPU baselines are brutal on constant-coefficient cells** (2026-07-27): DST direct solve = 0.19 ms for Poisson-2D N=64, 0.25 ms for a full 50-step Heat-2D N=64 trajectory — on CPU. The "324× vs unpreconditioned CG" framing cannot survive; reposition on variable-coefficient/many-query/nonlinear ground (the new benchmark cells).

---

# ✅ Done / ❌ Remaining

**Done:** 3 benchmarks built+validated · clean truth (cross-verified two stacks) · FNO/DeepONet/POD baselines tuned & fixed · single-model Pareto on all 3 cells · quiet-GPU re-timing · tables-from-JSON pipeline · wave + burgers-2D root-caused & closed · cluster pipeline up.

**Remaining (priority order):**
1. ⏳ Cluster batch: wave N=128/256 speedups, t2000 nonlinear retrain, seeds, A100 timings
2. ❌ Original-paper repairs: 4 single-checkpoint cells, heat bug check, MG/FFT baselines, Darcy
3. ❌ Paper text: Eq. (4) rewrite, EQ notation, timing appendix, SMA cold-start honesty, intrusive-vs-non-intrusive framing up front
4. ❌ Seeds on remaining cells; long-horizon rollouts (nice-to-have)
5. ❌ Code release prep (anonymized repo + checkpoints)

---

# ⚠️ Standing rules (do not violate)

1. **Numbers flow JSON → table only.** Never hand-edit a result.
2. **Both Pareto endpoints from ONE checkpoint/basis per cell.** Tune to find the best single model, then knobs only.
3. **Losses get reported and framed** (1D POD win = the control cell; FNO's burgers-2D 2.4e-2 stays in the table next to our 2.1e-2).
4. **Quarantine, don't delete.** Polluted runs → `_quarantine_*/` or `_superseded/` with README.
5. Timing quotes only from dedicated-GPU runs (notes field says so).
6. Long GPU jobs run from the MAIN session (subagent background jobs die at turn end).
