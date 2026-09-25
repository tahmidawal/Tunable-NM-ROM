# ns3d-test — Navier–Stokes 3D Table 1 rows at $32^3$ and $64^3$ on held-out test cases

Pre-registration, written and committed **before any test job**. Amendments are appended as §A1,
§A2, …; nothing above them is edited afterwards. No number in this file is a measurement of this
lane; the development values quoted in §6 are copied from the records named beside them and are
regenerated from JSON by `make_report.py`.

Branch `exp/2026-09-25-ns3d-test` (sparse worktree, local only, never pushed), forked from
`exp/2026-09-23-ns3d-operators` @ `766c3247`. Cluster namespace
`/cluster/tufts/paralab/tawal01/ns3test_20260925/<job>/`, one directory per job, never reused.

## 1. Question

The paper's Navier–Stokes 3D rows of Table 1 at $32^3$ and $64^3$ are measured on the 16
development cases (seed 202609202) on which $k=8$ and the fast setting were chosen. $96^3$ already
has a test run (32 cases, seed 202609221, jobs 4202872 and 4244423). What are the same rows — the
frozen development models, unchanged — on the 32 held-out test cases at $32^3$ and $64^3$?

## 2. Frozen models (no retraining, no choice)

Per mesh $n\in\{32,64\}$ the model is the one of development job `a2_h{n}` of the ns3d-shift-head
lane (32³: job 4198840, 64³: job 4198090), which produced the paper's development rows:

* **bank** $G$: centred rank-64 POD of 128 training trajectories (seed 202609201), rebuilt in the
  job by the recipe of `head_ladder.py`/`panel.py`, sign-aligned to and gated against the stored
  bank probe (relative gap $\le 10^{-8}$);
* **head** $k=8$ and the **importance rotation** $V$: loaded, not retrained.

| file (under `experiments/ns3d-operators/frozen/`) | sha256 | identical to |
|---|---|---|
| h32/head_k8.npz | 35c1a8aff8bdb5da292d3dd297470d25f23900b25529ba7fe99f309a8e792369 | `a2_h32/output/head_k8.npz` |
| h32/rotation.npz | 910d66995bc1f6604156c6a5c9491fa492e2947dd89000c4c22d7788ea3b523a | `a2_h32/output/rotation.npz` |
| h32/bank_probe.npz | a78d896fa78efa8fba3bbbcd4fce81abd17ca34924acd720dd090a7d52312802 | `a2_h32/output/bank_probe.npz` |
| h64/head_k8.npz | 605637b1a54087d59b88c7f6563c6dafaa4adcea58b04022253abf8196437c02 | `a2_h64/output/head_k8.npz` |
| h64/rotation.npz | 62e29e976129a50a8b5a43ea9941498df73f7029003307ffc945dd5378f58435 | `a2_h64/output/rotation.npz` |
| h64/bank_probe.npz | 36bde9e938d26ba8d8a66a4e5e3dedefe58685f6257453a812fbd5cf7e297ff3 | `a2_h64/output/bank_probe.npz` |

`make_config.py` refuses to write a config if any frozen copy differs from the `a2_h{n}` output.

## 3. Frozen settings

Written by `make_config.py {32,64}` into `configs/test32.json` (sha256
`71354a29f94d87e13d29566a8b9ce06e2782a786c0f7c2dc5b1051e2a85e0f51`) and `configs/test64.json`
(sha256 `4e4d786e9713ae9626a8b598a9344c47f67b58a51166565ded9316e023f1c9f3`). Solver constants are
the development ones: damping $10^{-6}$, 12 encoder sweeps, $T=0.2$, truth CNAB2 $\Delta t=0.001$,
six output times.

| arm (name in the JSON) | model | $\Delta t$ / sweeps | role |
|---|---|---|---|
| `nmrom_accurate_head_k8` | head $k=8$ in the moving frame | 0.02 / 3 | **accurate** (Table 1, as at $96^3$) |
| `nmrom_fast_span16_dt0.04_it2` | span $R'=16$ of the ordered bank | 0.04 / 2 | **fast A** — current development Table 1 fast setting |
| `nmrom_fast_span16_dt0.02_it3` | span $R'=16$ | 0.02 / 3 | **fast B** — the setting used on the $96^3$ test cases |
| `span{64,48,32,8}_dt0.02_it3` | span $R'$ | 0.02 / 3 | span ladder (with fast B) |
| `span{64,48,32,8}_dt0.04_it2` | span $R'$ | 0.04 / 2 | span ladder (with fast A) |
| `cnab2_s{200,100,80,70,60,50,40,20,10}` | spectral CNAB2 | $0.2/\text{steps}$ | FOM step ladder |

Both fast variants are reported; neither is chosen on test data. The span ladders are reported, not
used for any choice.

## 4. Cohorts

* **Test:** seed **202609221**, the first 32 draws — the cohort of the $96^3$ test run. Checked in
  the job to be row-disjoint (parameters rounded to 9 digits) from training 202609201 (512 draws),
  development 202609202 (16) and the closed cohorts 202609203 (32) and 202609211 (32); the job
  aborts on any overlap.
* **Was it ever used for a choice at $32^3$/$64^3$?** Searched every config, script, summary and
  report of the ns3d, ns3d-shift, ns3d-shift-head and ns3d-operators lanes and all worktrees
  (2026-09-25): at $32^3$ and $64^3$ the seed appears only as a *closed* seed in the shift-head
  development configs (`h32.json`, `h64.json`: never drawn) and in the operators' training-row
  disjointness check (`train_op.py`: parameter rows only, no fields, no choice). No field of this
  cohort has ever been generated at $32^3$ or $64^3$, and $k=8$, the span width $R'=16$ and both
  fast step settings were chosen from development records only. **Caveat, stated in the report:**
  the parameters are mesh-independent, and this same cohort has already been evaluated four times at
  $96^3$ (jobs 4201281, 4202872, 4243807/4244288/4244423 — see the ns3d-shift-head and
  ns3d-operators lanes), and its $96^3$ results were seen before the Table-1 fast setting
  (Δt 0.04 / 2 sweeps) was picked on 2026-09-24 from the development records. That pick used the
  development records only (cheapest development setting under 5 %); it is disclosed, not hidden.
  A fresh seed is therefore not drawn.
* **Development:** seed 202609202, 16 cases, regenerated in the same job, used only for the
  reproduction gate (§5.1) and for same-job, same-protocol development context timing (§5.3).

## 5. Job per mesh (one allocation, one process, A100-80G, `gpu` partition)

`test_panel.py --config configs/test{n}.json`, from its own job directory; both meshes submitted
concurrently. GPU preflight (`jax_backend=gpu` or exit 42), `JAX_DEFAULT_MATMUL_PRECISION=highest`,
float64 throughout.

1. **Reproduction gate (development cohort).** Every arm of §3 and every CNAB2 setting on the 16
   development cases; the per-case error rows must equal the `a2_h{n}` summary rows to
   $\le 10^{-6}$ relative (they were measured on a different GPU). Any arm without a reference, or
   any gap above tolerance, fails the gate.
2. **Test accuracy pass.** Every arm on the 32 test cases. Saved: full fields of the worst case per
   arm (and of cases 0, 1 at $32^3$) with their truth, plus 8192 fixed random grid points of every
   (arm, case). Disk cap 4 GB, minimum 40 GB free.
3. **Timing (A–B–A, the ns3d-operators protocol of job 4244423).** Query = GPU-resident $u_0$ → six
   GPU-resident fields, synchronised. Warm-up excluded (accuracy pass + 2 burn-in calls per case). A1:
   3 rounds, each a fresh random permutation of the arms, each arm once per case; B: arm-major, 3
   rounds × all cases; A2: as A1. Reported time = median of all 9 samples per case (288 per arm).
   Run first on the test cohort (the reported numbers), then with the identical protocol on the
   development cohort (context only).
4. **Table-1 FOM rule, per cohort:** the stable CNAB2 setting (all fields finite, evolved worst
   $\le 100\,\%$) with the smallest median time **in the same job** whose evolved worst is $\le$ the
   accurate arm's evolved worst on that cohort. Every speedup = that median / the arm's median, same
   job, same cohort. Also reported: every arm against the fastest CNAB2 at least as accurate as
   that arm.
5. **Gates** (on the test cohort; a failing gate is reported, never worked around; `status` is
   `final` only if all pass): reproduction (1); bank rebuild; drift median(A2)/median(A1) and order
   median(B)/median(A1∪A2) both in $[1/1.10, 1.10]$ for accurate, fast A, fast B and the rule FOM;
   positive control (×1.15 on A2 must fail for every arm); timed-output parity (the last A2 round's
   outputs reproduce the accuracy pass: error rows and sampled fields, $\le 10^{-9}$); coverage (288
   positive samples per arm); finite reported arms; a rule FOM exists.
6. **Independent audit** `audit_test.py` (NumPy only, in the job and again locally): regenerates
   every test $u_0$ with its own Leray projection; recomputes every saved full-field error
   ($\le 10^{-9}$); sampled estimates (diagnostic); recomputes every timing median, drift/order value,
   the FOM rule and every speedup from the raw repetitions, for both cohorts; must reject a copy with
   one saved field perturbed by $10^{-6}$.

## 6. Reporting

`make_report.py` writes `reports/summary.json` (sha256 of every input) and
`reports/2026-09-25-ns3d-test.md` from the pulled JSONs only. Per mesh, in Table 1 format:
accurate worst % and speedup, fast A and fast B worst % and speedup, FOM worst % and setting,
ms per arm, job id, GPU; beside them the development values of the paper (fast-block timing of
`a2_h{n}`, recomputed from that JSON with the same rule) and the same-job A–B–A development values.
Also: medians over cases, cases over 5 %, the span ladders.

The development protocol differs from the test protocol (a2: fast-block median of case 0, 7
repetitions; test: A–B–A over all cases). The report states this, and gives the same-job,
same-protocol development ratios so that the test–development comparison does not mix protocols.

## 7. Integrity rules

* No setting is changed after seeing a test number. The test cohort is opened once per mesh by the
  job of §5. A crash before timing is fixed only in plumbing and rerun with the identical config;
  the rerun must reproduce the first attempt's recorded rows; both are reported (as in the $96^3$
  lanes).
* An improvement, if any is wanted after the test run, must be selected on the development cases (or
  a validation split), frozen here as an amendment, and then evaluated on the test cohort once more;
  every test evaluation is reported, including worse ones.
* No hand-typed number in the report; same-job ratios only; results pulled with checksums; cluster
  job directories deleted after the pull.

## Glossary

* **accurate / fast setting** — the two Table 1 operating points of one frozen model: the head
  ($k=8$ unknowns + 3 frame unknowns) and the span of the first $R'=16$ ordered bank columns.
* **span $R'$** — solve in the span of the first $R'$ columns of the importance-ordered POD bank.
* **Δt / sweeps** — reduced time step and fixed Gauss–Newton iterations per step.
* **CNAB2** — the pseudo-spectral full-order solver (Crank–Nicolson viscous, Adams–Bashforth 2
  advection); "steps" is its step count over $T=0.2$.
* **evolved worst** — per case the maximum relative $L^2$ error (relative to $\lVert u_0\rVert$)
  over the five output times after $t=0$, then the maximum over cases.
* **FOM rule** — the one full-order comparator per cohort: fastest stable CNAB2 at least as accurate
  as the accurate setting.
* **development / test cohort** — the 16 cases on which settings were chosen / 32 cases never used
  for any choice at these meshes.
* **A–B–A** — interleaved, arm-major, interleaved timing phases; drift and order gates compare them.

## A1 (2026-09-25 ~03:25 EDT, after the test jobs) — execution record, no change of settings

Both test jobs ran once, concurrently, from their own directories with the frozen configs of §3
(commit `2bdb5f7e`): `t32a` = job 4319322 and `t64a` = job 4319329, both A100 80GB PCIe (node pax105,
shared with other lanes' jobs on other GPUs), `jax_backend=gpu`, every gate passed, in-job audit
passed and rejected its perturbed control. The test cohort was opened once per mesh; no rerun, no
second evaluation. Pulled with checksum verification; the cluster directories and the namespace
were deleted. The local re-run of `audit_test.py` (§5.6) agrees with the in-job audit up to
round-off (u0 regeneration gap 6.8e-16 vs 6.4e-16); it is stored as `runs/<job>/audit_local.json`
(it first overwrote `output/audit.json` by mistake; the in-job file was restored from a copy and
the pulled manifest re-verified). The report generator gained a generated caveat for a mesh
whose Table-1 FOM differs between cohorts (32³: CNAB2 40 steps on test, 50 on development).
