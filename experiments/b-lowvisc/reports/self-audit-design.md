# Self-audit of `DESIGN.md`, in place of the required Codex audit

The lane protocol requires an independent audit of this design by Codex (`gpt-6-astra`, a
different model family) before the first cluster job. The coordinator's notice in
`LANE-PROTOCOL.md` records, verified, that `codex exec` returns a usage-limit refusal until
2026-09-19 11:33, and instructs lanes not to block on it. This document is the substitution
recorded in `DESIGN.md` §A1. **It is written by the same model that wrote the design and is
therefore strictly weaker than the audit it replaces.** The Codex audit of the finished report
will be run after 2026-09-19 11:33 if the lane is still open, and appended as a dated addendum.

Each row names a claim the design makes, the artifact it rests on, and the check actually run.

| # | claim | rests on | check run | result |
|---|---|---|---|---|
| 1 | The incumbent viscosity family is $\exp U(\log 10^{-2},\log 10^{-1})$, not the $[10^{-3},10^{-2}]$ the brief assumed | `experiments/mr-burgers2d/engines.py:25`; `experiments/b-seeds/deps/burgers2d-coord-rom/burgers2d_film.py:181` | both read; `grep` over the tree for every other `log(0.01)`/`log(0.1)` draw site (13 files) shows the same bounds everywhere | confirmed; recorded as `DESIGN.md` §A2 |
| 2 | The low-viscosity cohort is the incumbent cohort with $\nu$ divided by exactly ten and nothing else changed | `lv_common.params_draw` | recomputed with NumPy for the six development cases: columns 0–3 `array_equal` True; $\max\|\nu_{hi}/\nu_{lo}-10\|/10 = 3.55\times10^{-16}$ | confirmed; also gated in job and re-checked by `audit_gate.py` (`descriptors_bitwise_identical_across_families`, `viscosity_ratio_is_exactly_ten`) |
| 3 | The comparator POD floors are not hand-typed | `make_configs.py` reads `comparators/bpn301-summary.json`, rows with `family=pod`, `metric=best_found_percent`, `job_id=3780638` | `config-gate.json` regenerated from the JSON; the six values printed by the generator | confirmed |
| 4 | The comparator files are verbatim copies of b-panel's | `comparators/PROVENANCE.json` | SHA256 recorded for each with source branch `exp/2026-09-17-b-panel` and commit `f3c5fbed` | confirmed |
| 5 | The full-order grid, meshes, time step, cohort seeds, reference mesh and snapshot settings are the incumbent panel's | `make_configs.py` reads them out of `comparators/bpn301-config-256.json` | `config-gate.json` diffed field by field against the parent for every copied key | confirmed |
| 6 | `gate.py` runs end to end and every reported number can be recomputed independently | `checks/smoke64-result.json`, `checks/smoke64-audit.json` | 64-interval smoke on the GB10 through `jaxrun`; `audit_gate.py` then reports **71 checks, 0 failed**, including errors recomputed to 0.0 absolute, POD floors recomputed by the Gram-eigenvector route to 0.0 percentage points, orthonormality $1.4\times10^{-14}$, medians to 0.0 | confirmed |
| 7 | The POD floor definition is the same quantity the panel's `best-found` column reports | the panel's own table: `pod512_M2048_dense` best-found 0.6090 %, worst-all-times 0.6125 % — a static reconstruction bound in the fixed-initial normalisation | **not yet confirmed by measurement.** This is the risk the design carries into the gate job, which is why `incumbent_pod_floors_reproduce_bpn301` is a *reported* gate at $10^{-3}$ relative and why §5.3 says no comparison is read if it fails | open, gated |
| 8 | Leg (b) is the weaker leg | order-of-magnitude estimate: at $\nu=10^{-3}$, $\Delta t\,\nu\,\lambda_{\max}\approx2.6$ against $\approx262$ at $\nu=10^{-1}$, with the advective part $\Delta t\,\vert u\vert L\approx5$ | arithmetic only; written into §2 as a prior **before** the measurement so the result cannot be presented as expected either way | recorded |
| 9 | The $L=256$ grid is marginal for the low-viscosity family | first-order upwind numerical viscosity $\vert u\vert h/2$ vs the drawn $\nu$ | computed for the six cases: $2.1$–$3.7\times10^{-3}$ numerical against $1.4$–$9.5\times10^{-3}$ physical; cell Reynolds 1.3–4.2 against the incumbent's 0.13–0.42 | confirmed; written into §3 as a stated cost, with falsification clause F4 and a mesh probe |
| 10 | A finer evaluation mesh is out of budget | the decoder is a coordinate decoder and evaluates at any mesh, but the bank and head must be trained on data from the evaluation regime; b-seeds stage B picks 131072 states, which at $L=1024$ is $1.1\times10^{11}$ entries | arithmetic; recorded in §3 as the reason $L=256$ is the training and panel mesh | confirmed |
| 11 | The pass criterion is the review's criterion, not a weaker one | `review-round1.md` finding 10(iii) and the closing paragraph; `2026-09-17-b-panel.md` "0 of the 27 reduced subjects are non-dominated" | criterion P written to require non-dominance against **every** same-job full-order control and **every** POD rank, all converged, with the reduced-only frontier reported but explicitly not the criterion | confirmed |
| 12 | The convergence rule is the panel's, not a new one | `b-panel` DESIGN §5, reproduced in its report's "converge under this lane's rule and not under the stricter one" section | adopted verbatim in §6 by reference; the stricter flag is reported beside it | confirmed |
| 13 | Fixed $M$ is the honest control for the ladder | `worktrees/2026-09-17-b-qxm/experiments/b-qxm/reports/2026-09-17-b-qxm.md` | cited in §5 stage 3 | **not yet read in full**; the b-qxm report is read before the panel configuration is written, and any correction enters as an amendment | open |
| 14 | No cost ratio crosses jobs | `config-gate.json` carries b-panel's costs only under `comparator_fom_*` with a note forbidding division | both families are re-timed in this job, so every ratio the lane states is within one allocation | confirmed |
| 15 | The sealed cohort is never touched before the last job | §4 | `gate.py` writes `final_cohort_unopened: true` and the audit checks it; the sealed draw is not referenced by any staged file | confirmed |

## Weaknesses this self-audit can see in its own design

1. **Row 7 is the single point of failure.** If the panel's `best-found` is not a static
   projection bound, the incumbent floors will not reproduce and leg (a) has no validated
   baseline. Mitigation: the gate reports the deviation per rank rather than aborting, so a
   near-miss is diagnosable from the job's own output.
2. **Leg (a) and the under-resolution caveat are entangled.** If POD degrades, part of the
   degradation may be numerical-diffusion structure rather than physical sharpening. The design
   cannot separate them at $L=256$; the mesh probe measures the full-order side of that question
   but not the POD side. A finer-mesh POD probe (POD floors at $L=512$ and $L=1024$ from
   trajectories generated at those meshes, no training) would separate them and is **not** in
   the current gate job. If leg (a) passes, adding it to the panel job is the cheapest way to
   close the objection, and that is recorded here as the first follow-up.
3. **The design does not pre-register what happens if G-a passes but the retrain fails TR-like
   reproduction bars.** b-seeds has a training-reproduction bar; this lane has none, because it
   is training on different data and has no reference number to reproduce. The bank floor and
   best-found are reported as measurements with no bar attached, and F2 is phrased as a
   comparison of degradation ratios rather than absolute values for that reason.
4. **Three timed repetitions on six cases is the campaign's protocol but is not a variance
   estimate.** All repetitions are retained and reported; no confidence interval is claimed.
