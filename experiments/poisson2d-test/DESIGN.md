# poisson2d-test — the Poisson 2D Table 1 settings on held-out TEST sources (pre-registration)

**Written and committed before any test job ran.** Branch `exp/2026-09-25-poisson2d-test`, forked from
`exp/2026-09-23-poisson-bank-knob` @ `06546331`. Cluster namespace `/cluster/tufts/paralab/tawal01/p2test_20260925/`.

## Question

Paper Table 1 reports the Poisson 2D (unit square) rows at $256^2$, $1024^2$, $2048^2$, $4096^2$ on the twelve
development sources only (`poisson-bank-knob`, `reports/summary.json` sha256
`f19d01721a0ba999518eeb3f05ee49ac4f0c10f52ca43313d35ce4c08940f67b`, jobs 4199318 and 4199321). Do the same frozen
settings give the same worst error and the same speedup over conjugate gradients on sources that were never used for
training, bank selection or any setting choice?

## Frozen model and settings (nothing is trained, tuned or selected here)

- Model: p-linear `primary_K32` ($K=32$, $R=512$), `experiments/p-linear/checkpoints/primary_K32.pkl` sha256
  `0c195e99c4fbc47c7906d84df885d12bc5452401a5c97eec3727b7ffba70be4f`, basis sha256
  `d629fb764295b5fad2252078993335ea7f7d3ea242584d7cef32dad21824fcd5`.
- Nested-bank rotation: `experiments/poisson-bank-knob/runs/prep.npz` sha256
  `43767c771a42f7e51a9c3017815e5d558c0022c90c1cd01d1e6dc5f92d6f86c7` (built from the training fit split only).
- **Accurate setting = `R512_linear`** (bank-span linear rung, $R'=512$). **Fast setting = `R128_linear`** ($R'=128$).
  These are the arms the parent's pre-registered rule selected at every mesh; they are fixed here by name and are
  **not** re-selected on the test cohort.
- **FOM (per mesh, one per row):** the fastest tested unpreconditioned CG tolerance, by CG-phase median GPU-query
  time, whose worst test-cohort same-grid error is $\le$ the worst test-cohort error of `R512_linear`, from the same job.
  CG tolerance ladder = the parent's: rtol $\{0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.03, 0.01, 10^{-3}, 10^{-4}\}$, maxiter 200000.
  If no tested CG qualifies, the row has no speedup (reported as missing, not extrapolated).
- Speedup $=$ FOM GPU-query ms / arm GPU-query ms, both from the same job. Arm ms = median `fused_device_seconds` over
  phases A1 $\cup$ A2; FOM ms = its CG-phase median. Complete-query (`total_seconds`) ratios are reported alongside.
- Error = relative $L^2$ same-grid error against the DST-I solve on the same mesh; the row value is the **worst** over
  the test cohort (median also reported).

## Test cohort

- **Fresh draw:** `C.source_params(2026092501, 32)` — the same generator (`ms_parametric.sample_params`: centre in
  $[0.15,0.85]^2$, log-uniform width in $[0.02,0.1]$, amplitude in $[0.5,2]$) as the training and development sources.
  Seed `2026092501` appears nowhere else in the repository (grep of every worktree's `.json/.py/.md`, 2026-09-25).
- The driver asserts, before any solve, that no test source is within $10^{-6}$ ($L^\infty$ in parameter space) of: all
  3072 training draws (seed 0), the twelve development sources (7090703×6, 7090732×6), and the other recorded Poisson 2D
  cohorts `common256` (20260916×256), `fresh256` (20260921×256), `confirm256` (20260922×256), bank-floor extra
  training draws (20260920×16384) and three small older cohorts (7090702, 911702, 777). Because the sampler draws
  column by column, a different count from the same seed is not a prefix; these checks are a sanity net, the real
  guarantee is the unused seed.
- **Why not `fresh256`.** It exists (bank-floor lane, seed 20260921) and was never used to choose a Table-1 setting,
  but (1) the bank floor of this exact R=512 bank on it (0.9594 % worst) is already computed and recorded in the lab
  log, and the accurate arm's error *is* the bank floor, so its accurate-arm outcome is not blind; (2) the bank-floor
  lane used the large cohorts in its promotion gate P1 (for other banks); (3) 256 fields per subject at $4096^2$ are
  34 GB each, infeasible to store for the full audit. A fresh draw costs nothing.
- **Why 32 and not 64.** The parent's audit re-solves every (subject, case) field in NumPy, so every field is stored
  on the cluster until audited: at $4096^2$, 17 subjects × 32 × 134 MB = 73 GB; 64 sources would be 146 GB on a
  share that is 93 % full (370 GB free, shared with other lanes). 32 keeps the audit at full coverage.
- Local smoke runs use a separate throwaway draw (seed 2026092599, 3 sources, $128^2$), never the test cohort.

## Timing protocol (the parent's A5 A–B–A design, unchanged)

Phase A1 = every ROM arm (5 reps × 32 sources, randomised within each source), phase B = every CG tolerance
(3 reps × 32, randomised), phase A2 = A1 again; device sync + 5 s cooldown + 2 s dummy kernel between phases,
0.1 s GPU burn-in before every invocation, UUID-guarded single GPU, host f64 source in → host f64 field out for every
subject, f64 and `JAX_DEFAULT_MATMUL_PRECISION=highest`, GPU preflight `jax_backend=gpu`. Stage profile, 3 reps.

## Deviations from the parent job (all fixed now, none chosen from results)

1. **Arm set reduced** to what the question needs: the linear-rung ladder $R' \in \{512,384,256,128,64,32\}$
   (includes both Table-1 arms; gives the $R'$ knob on test for free), `R512_q0` (head-only at the full bank: the
   parent's fast-bar reference), and `orig_q0` (unrotated parent kernel, parity baseline) at $\le 2048^2$. Dropped: the
   $C_q$ reference arms (removed from the paper, A5) and head-only arms at $R'<512$. Consequence: the random
   neighbour context of each arm differs from the parent's; the neighbour and drift gates are re-measured here.
2. **Parity** is checked for $q=0$ only (`orig_q0` vs `R512_q0`, $\le 10^{-10}$) at $256^2$–$2048^2$; there is no
   `R512_q256` arm.
3. **One job per mesh** (four concurrent jobs, own directories) instead of 256/1024/2048 in one allocation.
   Constraint `a100-80G`, the card type of every development row (A100 80GB PCIe), so test and development
   milliseconds are on the same card model. Speedups are still within-job only.
4. **Truth fields kept on the host during the bank build** (`p2t_core.build_banks_host`: the parent function with
   three asserted textual edits). The parent's $4096^2$ peak was 77.7e9 of 80.8e9 bytes with 12 fields; 32 fields on
   the device would not fit. Bank, contraction and $R_G$ are unchanged; only the floor diagnostic reads the truth.

## Gates (the parent's, reported per mesh)

parity ($\le 2048^2$), determinism (every repetition byte-identical), CG converged, within-phase neighbour
($\le 1.10$), drift A2/A1 in $[1/1.10, 1.10]$, profile = fused to $10^{-10}$, device guard, and the independent NumPy
audit (`pbk_audit_np.py`: dense-sine DST-I truth, every recorded error to $10^{-8}$, re-hash, two negative controls)
**PASS**. As in the parent, a timing-gate failure does not remove a row: the row is labelled provisional with the
failing subjects named. An audit FAIL, a CPU backend or a missing `jax_backend=gpu` voids the row.

## Rerun policy (fixed now)

A job that dies before completing (OOM, node or `cuInit` failure, disk full, time limit) may be rerun **identically**
(same config, same seed) under a new attempt name, and the failed attempt is reported. A job that completes is never
rerun to change a number. No improvement is tuned on test sources: any change of setting must be selected on the
twelve development sources only, frozen here as a dated amendment before its single test evaluation, and every test
evaluation, including worse ones, is reported. None is planned.

## Outputs

`reports/summary.json` and `reports/<date>-poisson2d-test-vs-dev.md`, both generated by `make_report.py` from the
pulled `result.json`/`audit.json` (sha256 recorded) and the parent `summary.json`; no hand-typed numbers.
Development values are read from the parent `summary.json` (`meshes[i].table1`), which matches paper Table 1:
$256^2$ 0.75 / 13.4× / 2.31 / 17.5× / 0.55; $1024^2$ 0.74 / 27.9× / 2.31 / 68.5× / 0.22; $2048^2$ 0.74 / 54.1× / 2.31 /
174× / 0.61; $4096^2$ 0.74 / 106× / 2.31 / 358× / 0.45 (accurate err % / speedup / fast err % / speedup / FOM err %).
