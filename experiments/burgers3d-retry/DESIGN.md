# DESIGN — burgers3d-retry: a second attempt at a Burgers 3D row

Branch `exp/2026-09-23-burgers3d-retry`, forked from `exp/2026-09-23-burgers3d-span` @ `2c9e7b39`, local commits only,
never pushed. Cluster namespace `/cluster/tufts/paralab/tawal01/b3dretry_20260923/`, one directory per job. Brief:
`reports/2026-09-23-burgers3d-retry-handoff.md` (main). This file is written in two parts: §§1–3 (diagnosis, probes,
cohorts) before any ROM result of this lane; §§4–9 (the pre-registered protocol) are committed before any validation
panel is run. Amendments are appended, dated, and never rewrite earlier text.

## 1. Starting point (lane 1, `burgers3d-span`, negative on its sealed cohort)

Held-out (seed 923401, 32 cases, opened; **never used here for any choice**): accurate span $R'=512$
1.32 / 2.39 / 3.48 % at 0.02 / 0.03 / 0.21× Newton–BiCGStab (32³/64³/128³); fast 8.29 / 8.31 / 7.89 % (> 5 % bar).
Validation (16 cases) under-estimated the held-out worst error by 2.1–2.7×.

What limits it (from lane 1's records only):

1. **Accuracy is the linear floor of the bank, not the solver.** Evolved error ≈ the projection floor of the ordered
   bank at the same $R'$ (validation 128³: span 512 1.50 % vs floor 2.2 %; span 192 3.7 % vs floor 5.0 %). The R = 512
   bank equals POD of its own training data on bank-validation fields (POD 512: 0.86 / 1.61 / 2.36 % worst; bank:
   0.93 / 1.63 / 2.20 %), i.e. the coordinate network is not the bottleneck — the *linear n-width* of this family and
   the amount of training data are.
2. **The nonlinear head generalises badly** (training error ≈ 0.05 %, bank-validation best-found 21–25 %): 384
   trajectories of a ~17-parameter family (1–3 blobs × centre/width/amplitude, ν, time) is sparse for an auto-decoder.
3. **Cost.** The tensor rule costs $O(M R'^2)$ with $M = 4R'$; at $R'=512$ the table is 4.3 GB and the fixed-sweep path
   read it twice per step (predictor + sweep) → ≈ 300 ms on an A100. The full-order solver is very cheap at 32³/64³
   (7.7 / 11.6 ms at its paper-rule setting) and grows ≈ 5× per mesh doubling (61 ms at 128³).
4. **Validation too small.** 16 cases cannot estimate a worst case over 32.

## 2. Probes (diagnostic only; probe seeds 923601 / 923651, disjoint from every cohort of both lanes)

Run before any design choice below; results are recorded in §2.1 when collected.

- `pod65`, `pod129`, `pod257h` (jobs 4246732, 4246733, 4246816): POD validation floors for nested training sizes
  (96 … 1536 trajectories; 96 … 384 at 257 nodes) on each mesh's point group, and the frozen lane-1 bank's floor on the
  same fields. Question: does more data lower the floor, and how does the floor grow with the mesh?
- `fineh` (4246813): full-grid floor of the frozen bank at 129 and 257 nodes on 8 probe cases, and Newton–BiCGStab GPU
  times of six settings at 129 and 257 nodes (H200). Question: is 256³ a regime where a reduced model can win?
- `fscchkh` (4246812): equivalence and speed of the cached-predictor solver (`common2.make_query_fsc`) and its float32
  table variant against lane 1's `make_query_fs`, at 33/65/129 nodes on 4 probe cases.
- Training candidates `trw512h` / `trw1024h` (4246814 / 4246815): the lane-1 recipe on 4× the data (1536 training
  trajectories, seed 923701; bank-validation 923751 × 96), bank MLP width 512 / 1024, 12 000 bank steps; head variants
  K ∈ {16, 32, 64} and K = 32 with an L2 code penalty 1e-3 / 1e-2; the lane-1 bank's floors on the same bank-validation
  fields are recorded in the same job (the model-choice rule of §4 compares them).

Local runs on the GB10 are not used for any number: the box returned NaN from f64 GEMM/QR non-deterministically on
2026-09-23 (the same code gives finite tables on the cluster); every check runs on the cluster.

### 2.1 Probe results (collected 2026-09-23 ~22:30 EDT; files in `checks/probe-*.json`)

**Solver path (`fscchkh`, H200, 4 probe cases per mesh).** The cached-predictor path reproduces lane 1's fixed-sweep
path to ≤ 3.1e-15 relative (fields) at every mesh and R′ tested, with identical stationarity; the float32 table changes
fields by ≤ 1.2e-8 relative. Median GPU ms (65 nodes): R′ = 512, Δt = 0.005: 124.6 (lane-1 path) → 80.6 (cached) →
52.8 (cached + f32); Δt = 0.01: 66.1 → 46.0 → 29.8; R′ = 256, Δt = 0.01: 14.5 → 11.9 → 9.7.

**Full-order cost and bank floor by mesh (`fineh`, H200, 8 probe cases, frozen lane-1 bank, full grids).**

| nodes | FOM Δt .01 ntol 1e-2 (ms / worst %) | FOM Δt .005 ntol 1e-3 | FOM Δt .025 | bank floor (evolved, worst %) R′ = 512 / 256 / 192 / 128 |
|---|---|---|---|---|
| 129 | 27.5 / 1.08 | 67.4 / 0.64 | 18.5 / 4.24 | 1.09 / 2.29 / 3.09 / 4.75 |
| 257 | 200.2 / 1.10 | 491.7 / 0.67 | 132.5 / 4.59 | 1.34 / 2.68 / 3.67 / 5.31 |

The FOM costs 7.3× more at 257 than at 129 nodes; the bank's floor rises by only ~20 %. 256³ is therefore the regime
where a reduced model can win both columns; 64³ (11.6 ms on an A100) cannot win the accurate column at any R′ ≥ 384.

**Data scaling of the linear floor (`pod65`, 96 probe-validation cases incl. t = 0).** POD worst floor at R = 512 vs
number of training trajectories: 96 → 6.65 %, 192 → 4.97 %, 384 → 4.40 %, 768 → 4.46 %, 1536 → 3.64 %; RMS
0.79 → 0.41 %. The frozen lane-1 bank on the same fields: 3.27 % worst / 0.49 % RMS. **The worst-case linear floor is
set by the family's n-width (a few hard cases), not by the amount of data**: 4× data lowers it by ~0.8 points. More
data mainly helps the RMS and, possibly, the head (auto-decoder generalisation).

## 3. Cohorts of this lane (fresh seeds, never used by lane 1 or by the probes)

| cohort | seed | size | use |
|---|---|---|---|
| training | 923701 | 1536 | bank, ordering, heads (candidates M1/M2) |
| bank-validation | 923751 | 96 | model choice (§4), head choice, floors |
| ROM validation | 923801 | 64 | the settings rule |
| certification draws 1–5 | 923811–923815 | 8 each | quadrature/tensor certificates |
| confirmation draw | 923816 | 16 | certificates |
| **sealed held-out** | **923901** | **32** | opened once, frozen settings only |

The validation cohort is 4× lane 1's (64 vs 16) so that its worst case is a less optimistic estimate of a 32-case
worst case.

## 4. The model (rule fixed before the candidates' bank-validation floors are read)

Candidates: **M0** = lane 1's frozen R = 512 bank (`burgers3d-span/inputs/model_R512`, bank.pkl sha256 `85b14417…`);
**M1** = `trw512h`; **M2** = `trw1024h` (§2). All three have R = 512 and are ordered by lane 1's recipe on their own
training data. Measure: worst full-grid projection floor at R′ = 512 on the bank-validation cohort 923751 (96 cases,
six output times) at each of 33/65/129 nodes, computed inside the training jobs (M0's by `compare_bank` on the same
fields). **Rule: the model is the candidate with the smallest maximum over the three meshes of that floor.** A candidate
whose job did not complete, or whose 65-node Gram condition exceeds 1e8, is excluded.

Heads: only for M1/M2 (M0's heads are 21–25 % and are not used). Among the chosen model's head variants, the one with
the smallest bank-validation best-found worst error is the head; head arms enter the panels **only if that error is
≤ 10 %** (twice the fast bar), otherwise the lane is span-only, as lane 1 was.

## 5. Meshes and hardware

65, 129 and 257 nodes per axis (64³, 128³, 256³ cells). 32³ is dropped: lane 1 showed the FOM at 32³ is cheaper
than any reduced query that meets the bar, and the probes (§2.1) say the question is decided at larger meshes.
**Every panel, certification and held-out job runs on an H200** (the 257-node tables need ~80 GB; one GPU type keeps
all rows comparable). Tables are built by `tables.py` (bank rows + Gram Cholesky; same projection as QR, Gram
condition recorded). Query: `panel3.py`, timing contract, gates and A–B–A protocol unchanged from lane 1 §8/R1 12.

## 6. Arms (identical at every mesh)

| family | R′ | solver | Δt | table |
|---|---|---|---|---|
| span | 512, 384, 256, 192, 128, 96, 64 | `fsc` (cached predictor, adaptive first 3 steps, then 1 sweep) | 0.005, 0.01 | f64, f32 |
| head (if §4 admits it) | 512, 256 | `fsh` (adaptive first 3 steps, then 1 sweep) | 0.005, 0.01 | f64 |

Tests M = 4R′ (span) / 4K (head), shell-completed; η = 1e-3; trust 0.05 × RMS spread (lane 1). 28 span arms
(+ 4 head arms). No lat16 / LM-while-loop arms: `fsc` equals lane 1's fixed-sweep path to 3e-15 (§2.1), and lat16
failed its certificate at R′ = 512 in lane 1.

FOM grid: lane 1's 26 Newton–BiCGStab settings, same allocation.

## 7. Certificates

Every arm, every mesh: the deployed query on draws 923811–923815 (8 each) and the confirmation draw 923816 (16);
ρ on every accepted state (k ≥ 1) over the arm's M tests, exact side = the FOM's sign-upwind stencil on every
interior node and a 3D DST; the float32-table arms are certified with the float32 contraction as deployed.
Confirmed iff ρ_max ≤ 0.116 on each of the six draws. Minimum decoded value recorded.

## 8. Pre-registered settings rule (validation cohort 923801 × 64, per mesh) — lane 1 §6 + R1, unchanged

- Error of an invocation: $\max_{t\in\{0.05,\dots,0.25\}}\lVert u(t)-u_{\rm ref}(t)\rVert_2/\lVert u_{\rm ref}(0)\rVert_2$
  on every interior node, reference = same-grid Newton–BiCGStab Δt = 0.005, ntol 1e-10, ltol 1e-11. Arm error = worst
  over the 64 cases.
- Eligible: certificate confirmed, every output finite, no reason-3 exit, non-stationary exits ≤ 1 % of all steps.
- **accurate** = eligible arm with the smallest worst error. **fast** = eligible arm with the smallest median GPU
  time whose worst error ≤ **5 %**.
- **FOM (paper rule)** = the fastest FOM setting whose worst error ≤ the accurate arm's; both columns divide it.
  **own-matched** = the fastest FOM setting at least as accurate as that arm. Both reported.
- Selection is mechanical (`select.py` of lane 1, unchanged logic) and committed as `selection.json` with its
  sha256 in the lab log before the held-out jobs are staged.

## 9. Sealed held-out, audits, reporting

After §8 is committed: one H200 job per mesh runs the frozen accurate and fast arms and the whole FOM grid on the
sealed cohort 923901 × 32, once (FOM chosen by the same rule on the held-out errors; the validation-frozen FOM beside
it). No setting, arm list, bar or rule changes after the held-out cohort is opened; if a held-out job fails for a
technical reason it is rerun unchanged and the failure is recorded.

Audits: `audit3.py` (NumPy, no JAX) recomputes every restricted-lattice error of every panel from saved fields
(≤ 1e-12), full-grid case-0 errors of the audit arms (≤ 1e-10), ρ in NumPy for the arg-max certified state of every arm
at 65/129 nodes and of the selected arms plus 4 random arms at 257 nodes (memory), the selection from the recorded
numbers, and two must-fail controls. A `codex exec` design audit of this file and the code before the validation
panels; a `codex exec` results audit before the report. Every change after a result is seen is a dated, labelled
amendment. The report is generated from the JSONs by `report3.py`.

Deadline: report to the user by 2026-09-24 20:00 EDT whatever the outcome.

## R1 — revisions after the independent design audit (2026-09-23 ~20:45 EDT, before any validation panel; override §§4–9)

Codex read-only audit: `checks/codex-design-audit-2026-09-23.txt` (2 blockers, 5 majors, 2 minors). Disposition:

1. **Success must require both frozen settings (blocker) — fixed.** `select3.py` adds `lane_success`: on the held-out
   cohort, the frozen accurate and fast arms are both eligible, fast ≤ 5 %, both paper-rule speedups > 1, every timing /
   determinism gate passed, and a FOM at least as accurate as the accurate arm exists. The verdict of the lane is
   stated per mesh against this gate; the own-matched speedup is reported beside it.
2. **Certificates bound to the evaluated model (blocker) — fixed.** `select3.py` asserts, for every certification
   job merged, identical model sha256s, exactly the six prescribed draws, and identical deployed arm specs
   (kind, R′, solver, Δt, head, K, η, M, trust).
3. **Float32 table (major) — removed.** All arms use the float64 table; the lane's method is f64 end to end. (The
   float32 variant's cached-predictor contraction is not exactly the certified operator; dropping it costs ~1.5× at
   R′ = 512 per §2.1 and removes the question.)
4. **Head initial fit (major) — fixed.** A head arm whose initial fit ends with reason 3 or non-finite on any case is
   ineligible.
5. **Timing gates (major) — fixed.** GPU burn-in (2 s) before every timed phase and 0.2 s after every cool-down; the
   neighbour gates are two-sided, ∈ [1/1.10, 1.10]; a determinism gate (timed outputs equal the quick run on the audit
   lattice to ≤ 1e-12) joins the timing gates. Full-output parity is not re-checked inside timing (the sampled guarantee
   on the 16³ lattice is what is pre-registered).
6. **Audit coverage (major) — fixed.** `audit3.py --selection` recomputes accurate, fast and the paper-rule FOM from
   the recorded numbers and must match `selection.json`; at 257 nodes the NumPy ρ audit covers the arg-max-ρ state of
   the selected arms plus 4 other distinct arms chosen with a fixed seed; at 65/129 every arm.
7. **257-node feasibility (major) — measured.** Rehearsal `reh257b` (probe cohort, frozen lane-1 bank) records table
   build time and runs to completion before any 257-node validation job is submitted; its figures are added below.
8. **Condition number (minor) — fixed.** Recorded as `bank_condition` = √(λmax/λmin) of the Gram and
   `gram_condition` = λmax/λmin; the panel aborts if the Gram condition exceeds 1e8.
9. **Candidate cutoff (minor) — fixed.** Candidates M1/M2 count only if their training job has completed by
   **2026-09-24 03:00 EDT**; a technical failure before then may be resubmitted once unchanged. The model choice is
   written mechanically to `checks/model-choice.json` with every candidate's floors.

Arms after R1 (every mesh): span `fsc`, R′ ∈ {512, 384, 256, 192, 128, 96, 64} × Δt ∈ {0.005, 0.01} (14 arms), plus the
4 head arms if §4 admits a head.

**R1-7 rehearsal figures (`reh257b`, job 4246870, H200, frozen lane-1 bank, probe cohort 923651 × 4; diagnostic).**
Tables at 257 nodes: bank rows 9 s, Gram + A 16 s, tensor 276 s; all table gates ≤ 6.6e-15. Four span arms and two FOM
settings ran to completion; timing gates pass (drift 1.006, neighbour 1.077 / 1.000, determinism 0). Median ms: span
R′ = 512 Δt .005 (f64) 128.2; R′ = 192 Δt .01 25.7; FOM Δt .01 ntol 1e-2 199.6. Feasible on one H200.
