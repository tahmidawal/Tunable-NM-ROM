# Post-closeout timing re-measurement and ROM input-preparation cost: protocol

Written and committed on 2026-11-15, after the in-family sealed Burgers quadrature closeout
(`exp/2026-11-15-jcp-sealed-closeout`, freeze `F1.json` at Git `6d5cef5ebf19`) and **before any
measurement in this lane**. Status: registered. Any later change is added at the end as a dated amendment with its reason.

This lane closes two gaps that the closeout left:

- **(a)** In 2D only 512² and 2048² passed both lattice timing checks, so the lattice-vs-accuracy-matched-spectral
  time ratio exists for 2 of 5 2D meshes (and all three 3D meshes).
- **(b)** The ROM's input preparation was not timed (the full-order solver's was), so no end-to-end ratio exists.

## 1. Does timing-only re-measurement on the opened sealed inputs count as a second opening?

**Decision: yes, in the sense that matters, so this lane does not use the sealed inputs at all. It times
already-exposed validation inputs instead and labels every number that way.**

The registered procedure and freeze say:

- `PLAN-v2.md` §F2: "Open the sealed cohorts **once**", contingent on the F1 freeze; §3: "No new sealed or reserved-OOD
  cohort is touched before §F."
- `DECISIONS.md` (hashed in F1): "No new reference generation, replacement of failed cases, **reopening for tuning**, or
  OOD case access." "No further use of these cohorts will be called prospective." The timing checks (A2.4: timed outputs
  agree with scored outputs; within-block and main-vs-confirm agreement to 10 percent) were registered **as part of the one
  opening**, and "A failed timing check is reported separately from accuracy." "A failed infrastructure run may restart
  the same immutable configuration." "Never silently substitute another case, model, solver or timing."
  "Stale cross-job times are never a comparator."
- The opening is a registry transaction (`published.Opened`, an exclusive lock bound to the approved review, the
  registry under `jcp-opening-records/`). Regenerating the sealed in-family inputs means running that machinery again.

Reasoning. A timing-only rerun with frozen settings would not decrypt anything new, would not touch out-of-family data
and would not re-score accuracy. But (i) it would go back through the opening registry, which the plan authorises once;
(ii) the timing checks were registered outcomes of that single opening, and the three 2D meshes **failed** them; a
timing drift failure is not an infrastructure failure, so the only registered restart clause does not cover a rerun; and
(iii) re-timing the same sealed cases until the gate passes is optional stopping on a registered gate. So a rerun on the
sealed inputs would be a second use of the opening that the registration does not authorise. The closeout's timing
verdicts on the sealed population therefore **stand as recorded** (2D: 2 of 5 qualified; 3D: 3 of 3).

What this lane does instead:

- **Inputs.** Already-exposed **validation** cohorts of the same family and the same meshes: 2D `val32`
  (`qcore.cohort('val32')`, draw seed 20260927) and 3D `val64` (seed 923801, `C.table(923801, 64)`). Both were the
  historical selection cohorts (`PLAN-v2.md` §3), so they are exposed by construction. Cases 0–5 of each, the **same
  number** (six) as the closeout timing block.
- **Structural guard.** The pinned `refstore` loader refuses the sealed (`sealed128`, `sealed64`) and OOD cohorts
  (`SealedCohortError`, DESIGN A3.8). This lane never imports `published.py`, never sets `JCP_OPEN_REGISTRY` or
  `JCP_APPROVED_REVIEW_SHA256`, and never reads `jcp-opening-records/`. No out-of-family population is touched.
- **Label.** Every number from this lane is "post-closeout, validation inputs, timing only, not prospective, not part of
  the sealed confirmation". The manuscript may not merge these with the sealed-panel ratios without that label.
- **No accuracy selection.** The pinned driver scores the six validation cases against the E0 references as part of its
  normal pipeline (and its timed-output checks need those outputs). Those errors are recorded only as a sanity check that
  the frozen pipeline reproduces; nothing is selected, tuned or ranked on them.

## 2. Diagnosis of the closeout failures (existing data, read before this protocol was written)

`diagnosis/closeout_drift.py` reads only the timing records of the eight archived closeout panels (no case contents) and
writes `diagnosis/closeout-drift.json`. Its findings (numbers in that JSON):

- The closeout gate is **panel-wide**: the worst repetition-block drift over *every* timed subject in the job (31 rule
  arms + 13 full-order solvers in the main block; the selected arm, the dense anchor, a decode and all full-order solvers
  in the confirmation block).
- In all three failing 2D panels the failing subjects were **not** the compared pair:
  - 256² (A100): only `decode|gauss256`, a ~0.24 ms subject, in confirmation block 3 (timer/launch jitter on a
    sub-millisecond kernel).
  - 1024² (A100): `PS_N12`, `PS_N16`, `PS_N20`, `PS_N24`, `PS_N32`, all in **confirmation block 0**, about 10–14% slow,
    then normal: a warm-up transient at the start of the confirmation block (after the confirmation compile/warm pass
    and the 30 s burn).
  - 4096² (H200): only `PS_N24`, again in **confirmation block 0**.
- The compared pair (`rom_lat|fib987`, `fom|PS_N48`) had worst block drift 1.030, 1.083 and 1.090, and main/confirmation
  fold ≤1.094, on 256², 1024² and 4096² respectively: within 10%. This is a **post-hoc** reading and does not change
  the registered panel-wide verdict on the sealed population. It does say what to control: first-block warm-up after a
  phase change, and sub-millisecond subjects in a panel-wide gate.

## 3. What is measured (all eight meshes)

| Dim | Mesh (closeout name) | Frozen lattice rule | Matched spectral solver (closeout selection, not re-selected) | GPU type (same as closeout job) |
|---|---|---|---|---|
| 2D | 256² | fib987 | PS_N48 | A100 80GB PCIe |
| 2D | 512² | fib987 | PS_N48 | A100 80GB PCIe |
| 2D | 1024² | fib987 | PS_N48 | A100 80GB PCIe |
| 2D | 2048² | fib987 | PS_N48 | A100 80GB PCIe |
| 2D | 4096² | fib987 | PS_N48 | H200 |
| 3D | 64³ (`d3m65`) | lat4096 | PS_N16 | A100 80GB PCIe |
| 3D | 128³ (`d3m129`) | lat4096 | PS_N20 | A100 80GB PCIe |
| 3D | 256³ (`d3m257`) | lat4096 | PS_N20 | H200 |

Part (a) is the three 2D meshes 256², 1024², 4096². Part (b) is all eight. One job per mesh measures everything for that
mesh in the same allocation (prepared-input query, both preparations and fused end-to-end calls), so every ratio is
same-job. The closeout's sealed prepared-input ratios are shown beside, labelled, never mixed into a ratio.

### Frozen settings

Each job config is derived in code (`timing.py`) from the byte-identical closeout config `vendor/configs/d{dim}m{mesh}.json`,
whose SHA256 is checked against `F1.json` (itself checked against `F1-REVIEW.json`). The pinned upstream source is the
closeout's staged `plateau-source.tar.gz` (SHA256 `dfe3b4c9…`, equal to F1 `staged_upstream_source_tar_sha256`), and every
upstream file is checked against `upstream.json` before import (as the closeout's `panel.py` does). Allowed differences
from the closeout config, and nothing else:

- `cohort`: `val32` (2D) or `val64` (3D); `cases`: `[0..5]`; `audit_cases`: `[0, 2]` (2D, unchanged).
- `arms`: only the frozen lattice spec (fib987 / lat4096, copied verbatim from the closeout config); `conv_arm` and
  `anchor_check_arm` set to that arm (they only drive bookkeeping and the in-job refstore check). 3D always adds the
  `tensor` arm (upstream behaviour).
- `foms`: only the closeout-selected matched spectral solver, its config entry copied verbatim.
- `timing`: the upstream driver's own main/confirmation panels are cut to 1 repetition each with `initial_burn` 0. They
  serve only as extra warm-up and are **not** used for any result.
- The in-job selection hook (`PL.cheapest_plateau_rule`) is replaced, as in the closeout's `panel.py`, by a function that
  returns the frozen lattice arm; that hook is also where this lane's registered measurement panel runs, inside the same
  process, using the driver's own query closures, prepared inputs and output checks.

Solver, quadrature, tolerances, time step, step budget, model weights, rules, float64 and highest matmul precision are
untouched.

### Subjects (six per mesh)

| Subject | Timed work (one invocation, synchronised with `block_until_ready`) |
|---|---|
| `rom_lat` | lattice query from the prepared initial field: projection, reduced solve, lattice decode (as closeout) |
| `fom` | matched spectral query from prepared sine coefficients, output on the scoring lattice (as closeout) |
| `rom_prep` | ROM input preparation: physical descriptor → initial field on the mesh, on the device. 2D: `jnp.asarray(qcore.e.initial(L, desc))`; 3D: `jnp.asarray(C.initial_interior(n, tab, j))`. These are exactly the expressions the pinned driver uses to make the ROM input |
| `fom_prep` | full-order input preparation, the closeout definition: 2D `fomrun2.prepare` (`psjax2d.ic`, composite Gauss–Legendre sine coefficients); 3D `psjax.ic(tab, j, N)`; result on the device |
| `rom_e2e` | `rom_prep` then `rom_lat` in one timed call (fused end to end) |
| `fom_e2e` | `fom_prep` then `fom` in one timed call (fused end to end) |

Same definition for both preparations: from the stored physical descriptor (already in host memory) to the prepared
input resident on the GPU, wall clock (`time.perf_counter`), synchronised. Descriptor generation (the cohort draw) is
excluded for both. The closeout measured the FOM preparation once per case (first call); here both preparations are
warm, repeated measurements, so they are comparable to each other. No preparation code is optimised.

### Schedule (pre-declared longer warm-up)

1. Warm-up: every subject on every case, **3 passes** (closeout: 1), then a **60 s** GPU burn (closeout: 30 s), using
   the pinned driver's own burn kernel.
2. **Discarded block**: one full shuffled repetition, timed and stored, excluded from all statistics (new).
3. Main block: **3** shuffled repetitions (closeout: 3).
4. Re-warm (1 pass over every subject and case) and a **60 s** burn, then a **discarded** shuffled repetition (new;
   targets the confirmation-block-0 transient diagnosed in §2).
5. Confirmation block: **5** shuffled repetitions (closeout: 5).

Each repetition is a random permutation of subject × case (seed 20261105, the closeout timing seed) with the closeout's
0.1 s burn before every invocation. Recorded diagnostics: `nvidia-smi` sampling of SM/memory clocks, temperature, power,
utilisation and throttle reasons once per second for the whole job; host load average at every invocation; node
occupancy (`squeue -w <node>`) and `nvidia-smi` at job start.

### Gates (the closeout's criteria, applied per subject)

For each subject, both blocks:

- **Output agreement**: every timed invocation's output agrees with the driver's untimed output for that case to
  relative max-norm ≤1e-12 (queries; same checks as the closeout driver) or is bitwise equal (preparations).
- **Within-block drift**: max(max block median / block-set median, block-set median / min block median) ≤1.10 (the
  closeout `block_drift`), separately for main and confirmation.
- **Main vs confirmation**: max(main median / confirmation median, inverse) ≤1.10.

Qualification:

- **Prepared-input ratio** qualified iff `rom_lat` and `fom` both pass every gate. This is the closeout's comparison.
- **End-to-end ratio** qualified iff `rom_lat`, `fom`, `rom_prep` and `fom_prep` all pass, **and** the fused subjects
  `rom_e2e`, `fom_e2e` pass, **and** each fused confirmation median lies within 10% of the component sum
  (component-sum vs fused-query consistency, `PLAN-v2.md` §3 group-A control).
- The panel-wide gate (all six subjects) is reported too. It is equivalent to "all six subjects pass".

### Ratios (confirmation block, as the closeout)

- Prepared-input ratio = median(`rom_lat`, 30 confirmation invocations) / median(`fom`, 30).
- End-to-end ratio (primary) = [median(`rom_prep`) + median(`rom_lat`)] / [median(`fom_prep`) + median(`fom`)].
- End-to-end ratio (fused, consistency) = median(`rom_e2e`) / median(`fom_e2e`).
- Per-case paired ratios (median over 5 repetitions per case), with minimum/median/maximum, and IQR and min/max of every
  subject, are reported. A ratio above one favours the full-order solver.

### Stopping rule

One job per mesh. A job that does not complete (preflight exit 42, node or OOM failure, wall time) may be resubmitted
once with the identical committed config, and the failure is listed. A completed job that fails a timing gate is
**reported as failing and not rerun**. Every mesh is reported, passing or failing.

### Must-fail control

`test_gates.py` injects a 1.5× slowdown into one confirmation repetition of one subject in a synthetic record and
requires the gate code to fail it; it also requires a doubled-work fused subject to fail the consistency check. The test
must pass before any result is reported.

## 4. What results can and cannot support

- They can give a same-job, stability-checked lattice-vs-matched-spectral ratio on 256², 1024² and 4096² **on validation
  inputs**, and an end-to-end ratio on all eight meshes on validation inputs.
- They cannot convert the closeout's failed sealed timing checks into passes, and are not prospective.
- Accuracy matching was established by the closeout on the sealed cohort; this lane does not re-establish it on
  validation inputs and does not re-select the full-order solver.

## Glossary

- **Sealed / opened cohort**: physical test cases held back until a single authorised opening; after that, "opened".
- **Validation cohort**: cases used historically to select settings; already exposed.
- **Lattice query (`rom_lat`)**: the reduced-order model's full query, returning fields on the scoring lattice.
- **Matched spectral solver**: the fastest full-order pseudospectral configuration that the closeout found to match
  or beat the lattice ROM's eligible median and worst error on the sealed cohort.
- **Prepared input**: the solver's starting data already on the GPU (initial field for the ROM, sine coefficients for
  the spectral solver). **Input preparation** produces it from the physical case description.
- **Block drift**: how far the median of one repetition block strays from the median over all blocks of that subject.
- **Main / confirmation block**: the two timed phases; confirmation must agree with main.
- **End-to-end**: preparation plus query.
