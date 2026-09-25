# t2-ns3d-test — Table 2 Navier–Stokes 3D rows at $32^3$/$64^3$ on the 32 held-out test cases (pre-registration)

Written and committed **before any evaluation of this lane**. Amendments are appended as §A1, §A2, …;
nothing above them is edited afterwards. No number in this file is a measurement of this lane.

Branch `exp/2026-09-25-t2-ns3d-test` (local only, never pushed), sparse worktree
`worktrees/2026-09-25-t2-ns3d-test`, forked from `exp/2026-09-23-ns3d-operators` @ `766c3247`.
Cluster namespace `/cluster/tufts/paralab/tawal01/t2ntest_20260925/<job>/`, one directory per job,
never reused.

## 1. Question

Paper Table 2 prints the NS 3D cells at $32^3$ and $64^3$ from the ns3d-operators panels pn32 (job
4234771) and pn64 (job 4237885), both on the **16 development cases** (seed 202609202) on which the
NM-ROM setting $k=8$ was chosen. The $96^3$ cell (pn96c, job 4244423) is on the 32 held-out test cases
(seed 202609221). This lane re-evaluates the $32^3$ and $64^3$ cells on those same **32 test cases**,
with the same models, settings, rule and protocol as pn96c, so that all three NS cells of Table 2 are
test-case numbers.

## 2. What is frozen (no retraining, no retuning)

* **Code.** The ns3d-operators driver and audit, byte-identical (this lane adds no model or panel code):

| file | sha256 |
|---|---|
| experiments/ns3d-operators/panel.py | de091eb1bb5640c35ab347e56cbae06bf83097aeb71e4ed38f43b60600e3f55f |
| experiments/ns3d-operators/audit_panel.py | dd4c6878452f2ca8bd00db9ad2a8e4b8a8446a4f38dbfbe18c155d2766d21bd4 |
| experiments/ns3d-operators/ops3d.py | 2ced1f72c75f62f7c221baca92e18c56fb33c263928a83dab5f09cdfaf85e676 |
| experiments/ns3d-operators/spectral_conv_f64.py | b41d614e07ae55678fc667a19e4d3eaad2d9ed5a952748368522324cd62bb31f |
| experiments/ns3d/ns3d_fom.py | e4ffdbe6ec97dc77c457d4a92ac596cd9bfffa01a37e286f5f1a09358b7daff7 |
| experiments/ns3d-shift/shift_rom.py | 22921d34d486bc3939f8ad3dfdae220390e78d57bc4ca9e88812290007bda4a3 |
| experiments/ns3d-shift-head/head_rom.py | 6d07ccfa7cbd64e01efc29091777a6159c35d07416d5e5a03fb9e08a4812204c |
| experiments/ns3d-grok/diag_floor.py | fc951362f184192535cefd6b18a08467746f210bc74f9acd559d41c7b294db8c |

  `panel.py` is the version that ran pn96c (A3 sampled field parity, A7 memory fix). pn32/pn64 ran an
  earlier revision (error-metric timed parity only); nothing measured differs in kind.

* **NM-ROM** (the Table 2 settings): accurate = head $k=8$, implicit midpoint $\Delta t=0.02$, 3 damped
  Gauss–Newton sweeps; second setting = span $R'=16$ of the importance-ordered bank, $\Delta t=0.02$, 3
  sweeps (the setting printed in Table 2 at every NS mesh: 2.96 % / 3.24 %). Bank rebuilt from the seed
  in the job and gated against the stored probe (gap ≤ 1e-8); head and rotation from `frozen/h{32,64}`:

| file | sha256 |
|---|---|
| ns3d-operators/frozen/h32/head_k8.npz | 35c1a8aff8bdb5da292d3dd297470d25f23900b25529ba7fe99f309a8e792369 |
| ns3d-operators/frozen/h32/rotation.npz | 910d66995bc1f6604156c6a5c9491fa492e2947dd89000c4c22d7788ea3b523a |
| ns3d-operators/frozen/h32/bank_probe.npz | a78d896fa78efa8fba3bbbcd4fce81abd17ca34924acd720dd090a7d52312802 |
| ns3d-operators/frozen/h64/head_k8.npz | 605637b1a54087d59b88c7f6563c6dafaa4adcea58b04022253abf8196437c02 |
| ns3d-operators/frozen/h64/rotation.npz | 62e29e976129a50a8b5a43ea9941498df73f7029003307ffc945dd5378f58435 |
| ns3d-operators/frozen/h64/bank_probe.npz | 36bde9e938d26ba8d8a66a4e5e3dedefe58685f6257453a812fbd5cf7e297ff3 |

* **Operators.** All 8 trained checkpoints per mesh from the ns3d-operators training jobs (tr32_*:
  4232609–4232670, tr64_*: 4232677–4232722, all A100-80G, 3000 s budget), `best.pt` read from the
  ns3d-operators worktree (gitignored there), sha256 checked against each training record by
  `make_config.py`, again after upload (`CHECKPOINTS.sha256`, in the job before any work) and by
  `panel.py`. The **Table 2 size per family is the ns3d-operators pre-registered selection on the
  64-trajectory validation split** (copied unchanged from `configs/panel{32,64}.json`): 32³ fno-l,
  unet-l, tsol-s, don-s; 64³ fno-l, unet-l, tsol-l, don-s. It is not revisited on test. All 8 are
  evaluated and timed and every one is reported.

| mesh | arm | sha256 |
|---|---|---|
| 32 | fno-s | bf2820ad620c5bc37be6404e0f7b3f618aab68b1161c47f474a28fa6e1ba7ab8 |
| 32 | fno-l | 2225733b7f88ad1bc300946a8a87889f2196cc560a2d4601b44d1b94c7fdc528 |
| 32 | unet-s | cd867e790ccb02d8962c781b2118a8862ce187ff83fcd6a8ceadebbb289824d4 |
| 32 | unet-l | 110dde74c8580a232a40c0d824403ecd3599b9d7710158f61ffb26c1f2c069c4 |
| 32 | tsol-s | 967e140a2359f6f228190a4de3eb84ffa60b8f2139978ef23c5b322e20386462 |
| 32 | tsol-l | 2a7c15ef8e59681e656b5184c23a9f2be9e14a607944469010d5686c3622fa8c |
| 32 | don-s | 52509d640ff61f35b6908fcd0e10e2548717d56e54ee8f7fb9869ed4ec326c0f |
| 32 | don-l | ba6c55cec0bfdab9e3ba9dab8c66a8a8a589afae0add512337fc1d702b382356 |
| 64 | fno-s | 6262d206da383f70754e77d46579567f12aee721d3468b18fe5afc68f363710c |
| 64 | fno-l | f02f113666fc777f76e8e8ec007c59c950714d8f03ca2fe83feadef72666f5cb |
| 64 | unet-s | 71cb47358f400c48f17e33bf8a95eb7ce6d14ebcca04b1b2759b5c5bd935d9fc |
| 64 | unet-l | 62da0b4f186dce7e216dbf8780fe89af6cca340a3d172bccf95c8dea799eac94 |
| 64 | tsol-s | 2d56ffcdfd3abea32c8c9d1f1c934d28a6973ae7e50a5de1eb0f4167964c7923 |
| 64 | tsol-l | 4ef70d67b3f2280b63e150ca4628074799094a9a4a1e8acdc6dfaa65fb7589ee |
| 64 | don-s | ca6fec93627e8c7378493ff6fa9b641e4feed0102794b55c3d1dd8901112c8f8 |
| 64 | don-l | 0dcbb47bd589b5cdeaff5652e356e98c5cb549cd1a8d6cb632feebe99712fec9 |

  **DeepONet size caveat (pre-declared).** Table 2's caption says each family is shown "at its most
  accurate trained size"; the rule actually applied is validation selection, and at 64³ it picked
  don-s although don-l was more accurate on the development cases (51.61 % vs 51.75 %, lab log
  2026-09-24 open item 2). The report prints both DeepONet sizes (and both sizes of every family) with
  the validation pick marked, plus which size is more accurate on test; it does not choose a
  Table 2 size on test.

* **Configs.** `configs/t2test{32,64}.json`, written by `make_config.py` from
  `ns3d-operators/configs/panel{32,64}.json` (sha256 f8ae80ff… / bc4ee8ea…) with only: evaluation cohort
  → seed 202609221, 32 cases; checkpoint paths → the job directory; `full_cases` = [0, 1] (+ each
  arm's worst case, as pn32/pn64); the reproduction reference → the ns3d-test lane's test jobs on the
  same cohort (§4.2). Everything else (CNAB2 ladder {200,100,80,70,60,50,40,20,10}, sample seed,
  timing seed, rounds 3, burn calls 2, field cap 4 GB, min free 40 GB) unchanged.

| config | sha256 |
|---|---|
| configs/t2test32.json | 5376c53a8faa50585cde65461bfbd7dee9134a6a5afdbb6641e0746c681ce4ae |
| configs/t2test64.json | 704deffc92f7a37aeb571fc2b7aa337eca08b2e70b120d969e8127849d46635b |

## 3. Cohort

The 32 held-out test cases, seed 202609221 (the $96^3$ Table 2 cohort; same parameter draws at every
mesh). At $32^3$/$64^3$ this cohort was opened once before, by ns3d-test (jobs 4319322 / 4319329,
NM-ROM and CNAB2 only; no choice was made on it). This lane is its second opening at those meshes and
the first for the operators. No choice of any kind is made on it here. The panel asserts it is disjoint
(rounded rows) from the 512 training draws (seed 202609201).

## 4. Protocol (= pn96c)

1. **Jobs.** One job per mesh (`t2t32`, `t2t64`), concurrently, each in its own directory, A100 80GB
   (`--gres=gpu:a100:1 --constraint=a100-80G`, the GPU type of pn32/pn64/pn96c and of ns3d-test),
   `gpu` partition, `JAX_DEFAULT_MATMUL_PRECISION=highest`, f64, GPU preflight (`jax_backend=gpu`,
   exit 42 otherwise), JAX memory fraction 0.5 (pn32/pn64). `squeue` before and after each submit.
   Before them, one smoke job (`smk32`, `--smoke`: seed 7, 2 cases, reduced ladder, 1 round; never
   the test cohort) checks staging and paths; its numbers are not reported.
2. **In each job** (`panel.py`): truth (CNAB2 $\Delta t=0.001$) → bank rebuild + gate → accuracy pass of
   every arm (NM-ROM ×2, CNAB2 ×9, operators ×8) → **reproduction gate**: per-case test errors of the
   NM-ROM arms and every finite CNAB2 setting equal the ns3d-test values (t32a job 4319322, summary
   sha256 d7a425b6…; t64a job 4319329, sha256 beb7e075…) to ≤ 1e-6 relative → A–B–A timing (A1
   interleaved 3 rounds, B arm-major 3 rounds, A2 interleaved 3 rounds; 9 samples per arm per case;
   2 s GPU burn before each phase; reported time = median of all samples) → gates.
3. **FOM rule (Table 2).** The one comparator per cell = the stable CNAB2 setting with the smallest
   median time in the job whose worst evolved error is ≤ the NM-ROM accurate arm's. Every speedup =
   that median / the arm's median, same job. The per-arm "matched CNAB2" ratio is printed as context.
4. **Gates** (on NM-ROM ×2, the rule FOM and the 4 selected operators): reproduction, drift
   median(A2)/median(A1) and order median(B)/median(A1∪A2) in [1/1.10, 1.10], positive control
   (×1.15 on A2 must fail), timed-output parity (errors and sampled fields vs the accuracy pass;
   1e-9 f64, 1e-4 f32 operators), coverage, finite, bank rebuild, FOM found. `status: final` only if all
   pass; otherwise the panel is PROVISIONAL and printed as such, never re-run to pass.
5. **Audit.** `audit_panel.py` (NumPy, independent, restricted) in the job and again locally after pull:
   u0 regeneration, saved full-field errors (cases 0, 1, worst per arm) to ≤ 1e-9, timing medians, FOM
   rule, speedups, perturbed-copy rejection.

## 5. What is reported

`reports/summary.json` and a generated report (`make_report.py`, from the pulled summaries only, with
their sha256): per mesh, every method's worst evolved error %, median ms, speedup vs the cell FOM; the
FOM setting, its error and ms; job id, GPU, gates; the test values side by side with the current
development values of Table 2 (pn32/pn64, summary sha256 3c87f34b… / 5f55eb9a…, read from this tree).
Both operator sizes per family. Every evaluation run by this lane (including the smoke and any failed
job) is listed. The paper is not edited.

## 6. If something fails

A crash is recorded and archived; a rerun with an identical config is allowed only for an
infrastructure cause (OOM, node failure, preflight 42), disclosed as an amendment with the cohort's
opening count. A failed gate is reported, not worked around.
