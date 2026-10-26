# Burgers 3D refinement: evidence reconciliation and targeted numerical controls

Status: proposed design, awaiting independent review. No new scientific GPU result exists.
The implementation base is consolidated commit `93f399faad67f94ea2b56fe25f13003143e64694`;
the newer improvement lanes are sources of labelled evidence, not implicitly merged code.

## Questions and hypotheses

1. H-inventory: the available evidence supports a conditional accuracy–cost frontier once
   solver qualification, quadrature qualification, reference population, output contract and
   timing allocation are kept separate. A missing certificate is unknown, not a pass.
2. H-Q: a wide-bank fixed-sweep CN trajectory may agree with a tighter nonlinear solve while
   changing under quadrature refinement. Its error cannot be attributed solely to nonlinear
   iteration count. Test quadrature successors with the same bank, tests, stepper and cases.
3. H-S: the wide-bank QN-BDF2 large-step failures may come from incomplete solution of the
   reduced equations. Test the same history and rule with tighter adaptive solves, refresh
   frequency, extra work and changed trust radius separately. A trajectory-level improvement
   alone does not distinguish trust clipping from iteration budget.
4. H-T: after convergence in quadrature and nonlinear work, step refinement may lower E0
   error. A solver improvement must retain its qualification and include its additional cost.
5. H-N: reduced evolution with fixed bank, tests, quadrature and fixed output points may be
   mesh independent, while arbitrary sampled input and full-field output grow with the mesh.

## Stage R: recovery before new compute

Produce a deterministic JSON inventory and per-case records from the quadrature, E0,
combined, speed3d, qualify, dims, budget, robust and POD-baseline lanes. Preserve raw-run and
config paths, SHA256, source commits, job/GPU identities, bank identity, r, M, m, scheme,
step, nonlinear policy, seed, exact ordered case ids, scoring reference, eligibility,
uncertainty, saved coefficients/fields, timing repetitions and output contract.
Use separate observation ids when the same reduced configuration was run in several jobs.
Do not silently overwrite a failed observation with a later success.

Join evidence by full configuration identity and explicit case ids, never array position
alone. Keep certified, failed, unverified, missing and historical evidence visible.
The existing dims aggregate and width-only export cover different scopes; retain the
explicit-M variants. Do not manufacture seconds by combining a ratio and an anchor from
different jobs. Cross-job curves use dimensionless within-job ratios or work counts.

Re-score saved unsealed coefficient trajectories on the approved E0 ST lattice where the
bank, output times, normalisation and case identity can be verified. Compare the recomputed
per-case errors with archived scores. E0 uncertainty is conditional Richardson evidence,
not a rigorous bound. Publish reference eligibility masks, not only pooled statistics.
Historical test cases are replication evidence only; selection uses dev/validation.
Sealed and reserved OOD stores remain closed, including directory globbing through them.
Report the strongest qualified alternatives separately by median error, worst error and
within-job cost; there need not be a single best baseline.

## Stage D: first bounded experiment set

No retraining. Reuse the M2 and W1024 frozen banks with content-hash provenance.
Use only seed 923801 validation cases, regenerated from seed in each cluster job.

- D-Q: W1024 fixed-sweep CN, step 0.01, one sweep, default test policy; verify the existing
  gl56 state with gl64 and gl80 successors. The full-population gl64 trajectory already
  exists: reuse it and gl56 after checking exact identities; only the missing gl80 is new.
  First run a one-case memory/time pilot. Continue
  on the archived first-eight check population and the validation cases with greatest
  archived successor discrepancies. Report this as diagnostic subset evidence. A production
  qualification claim requires the complete registered validation population and tighter
  checks; the diagnostic subset cannot qualify the full configuration.
- D-S: W768 and W1024 QN-BDF2, step 0.025, gl40, refresh period four. Replay archived failed
  cases 5, 30 and 43 where present, plus case 0 as a fixed ordinary control. First reproduce
  the original trajectory. At the identical saved pre-step history compare the production
  solve with a fresh-Jacobian adaptive tight and tighter solve. Change trust radius, refresh
  frequency and iteration budget one at a time. Persist residual, fresh stationarity,
  conditioning, damping, clips, accepted/rejected steps, Jacobian/residual evaluations and
  coefficient histories. Both trust and extra work rescuing a step is mixed evidence.
  The historical `make_qn_evolve(hist, k0)` resets the cached solver carry and forces a fresh
  matrix at restart; coefficient history alone does not reproduce the original QN state.
  Either capture and restore the complete original carry (matrix/factor/J-norm, damping,
  last-refresh index, previous work/failure and predictor), with an exact-reproduction
  control, or explicitly label these as fresh-restart solves. The default implementation
  uses labelled fresh restarts for solution comparison and full-trajectory one-factor
  arms for refresh-policy causality. Never call a fresh restart the cached baseline.
- D-T: only on a bank/rule passing the available D-Q and nonlinear controls, compare step
  0.025, 0.0125 and 0.00625 with verified adaptive solves. Hold initialisation and outputs
  fixed. Reuse saved exact matches; do not rerun already complete controls.
- D-POD: the historical 3D ECSW/Q-DEIM comparisons did not gate hyper-reduction convergence.
  Reuse the frozen frequent pick `ecsw|512|k8|BDF2|0.016666667|common|central` and its archived
  basis/support. Compare the same-rank no-hyper-reduction model and increased support on the
  same seed/cases, scheme, step, initialisation and output times. Save coefficient histories,
  own-state nonlinear discrepancies and paired timings. A small subset is a diagnostic;
  qualified comparison needs the complete shared validation population. Do not rerun the
  historical tuning grid. Historical observed domination is retained with its missing
  hyper-reduction qualification disclosed.
  Freeze the original k8 support and existing archived k16 support (if available) before
  replay; no re-selection using validation results. If the k16 rule is unavailable,
  direct k8 versus no-HR still measures convergence, and successor support stays missing.
  For each case evaluate all six stored fields using the same full-grid mass norm and
  fixed initial-reference norm as the bank comparison. The Q2-POD field gate is max
  evolved-output distance <=2.5e-4 for k8 versus no-HR and, when available, k8 versus k16
  and k16 versus no-HR. The exact no-HR comparison closes convergence directly rather
  than treating a larger support as truth. On every saved own-state coefficient vector,
  also evaluate the common full-grid projected nonlinear vector and each weighted-vector
  approximation: rho = ||N_HR-N_full||/||N_full||, with rho=0 when both vanish and infinity
  when only the denominator vanishes. Require rho<=0.116; state this as a new conservative
  comparator-admission rule, not a predicate satisfied by the historical campaign.
  Because the historical method weights the whole residual, persist the analogous full
  tested-residual and normal-equation gradient discrepancies too. Use the original sampled
  Q1 solver checks and label full-operator discrepancies separately. Any instability,
  nonfinite value, missing case or failed field/rho check blocks qualified HR admission.

The first numerical stage is capped at 2 GPU-hours including failed pilots. The complete
initial campaign is capped at 12 GPU-hours; every extension requires a written amendment
justified by the measured pilot, with the same total cap unless explicitly revised.
The cap is a resource limit, not permission to claim completion with missing evidence.
No broader sweep is released before the diagnostic results audit.
Allocation: at most 8 GPU-hours for the solver/quadrature lane and 4 for the POD qualification
lane, including at most 1 GPU-hour initial smoke per lane. Each uses a separate dated worktree
from the consolidated fork and namespace. The coordinator's evidence/report worktree runs no
scientific GPU jobs. Reviewers are read-only and must not mutate lane artifacts.

## Stage F: conditional frontier validation

After Stage R/D review, freeze a small manifest spanning the qualified median/worst/cost
extremes and any diagnosed repaired wide-bank candidate. Existing evidence supplies the
width, test-size and quadrature ladders; new jobs fill only specific missing certificates.
Preserve all unsuccessful candidates. Give POD the same integration choices and tuning
opportunities before any comparative improvement claim. If an ETD/IMEX arm is added, first
verify the weak reduced equations and run small consistency/order controls; no such arm is
promised. Existing finite-difference and spectral comparisons retain their historical
scope unless paired anew at matched error, population, reference and output contract.

Measure initialisation, evolution, Jacobian/residual evaluations, nonlinear iterations,
decode, full query, offline setup and memory separately. Cost and error are taken from the
same invocation; save all repetitions, validity and coefficient fingerprints. Burn in
before every timing block, block asynchronous work, randomise paired A–B–A panels and
retain the hardware identity. Certification work is distinct from deployable query time.
Mesh validation fixes reduced settings and test modes: existing intervals 64–384 provide
historical evidence; any repaired candidate is rechecked at 64, 128 and 256 intervals if
the pilot fits the cap. Fixed-point and full-field output receive separate tables.

## Qualification, controls and stop rules

Retain the original Q1/Q2 predicates without weakening them after seeing outcomes:
pooled uncertified-step fraction at most 0.01, fresh stationarity at most 1e-3,
production-to-tight rollout distance at most 2.5e-4 on every case,
tight-to-tighter at most 2.5e-5; quadrature successor distance at most 2.5e-4
and own-state advection rho at most 0.116, with the finer rule itself checked.
These thresholds are design constants. Record scope limitations where original predicates
differ, and retain every validity/anchor check from the original lane.

Controls: reproduce saved trajectories; independently recompute selected weak residuals,
Jacobians and scores; under-resolved quadrature must remain detectable; reject missing
cases, nonfinite values, wrong references and wrong output schedules. Controls address
realistic execution failures; do not spend repeated rounds on fabricated artifacts.
Stop a cell on failed GPU preflight, precision mismatch, provenance failure, memory risk,
nonfinite output, missing expected files or an exceeded measured resource bound.
Checkpoint completed cases so a timeout cannot erase the inventory. A failed cell remains
in the ledger; inspect queue/state/output before any restart.

## Independent audit gates and execution

An independent Codex reviewer must approve design before numerical implementation, code
and small controls before production, and exact job/configuration before submission.
Review results and interpretation before publication; repair blocking issues and re-audit.
Review verdicts and fixes are committed with source hashes. No claims derive solely from
this design or from a reviewer statement.

Worktree: `worktrees/2026-10-26-b3d-refine`, branch `exp/2026-10-26-b3d-refine`.
Cluster namespace: `/cluster/tufts/paralab/tawal01/b3d-refine-20261026/`.
Each job has its own child directory. Use gpu partition, f64, highest matmul precision,
GPU preflight, recorded seed/config/commit/GPU/job, direct staging to paralab, queue checks
before and after submit, checksummed collection and backed-up manifests. Delete no unique
data; remove a remote job directory only after verified complete local collection.
Never use a bare scancel or broad process kill. Back every commit up with the code-only
mirror based on local 93f399faa and remote 5a64899fd.

## Deliverables and reporting rules

Generated inventory, source manifest, frozen configurations, job ledger, audit records,
reproducible raw outputs and generated report tables/plots: accuracy versus total query
cost, cost/work versus mesh, accuracy versus width, quadrature convergence and error budget.
Plot only measured comparable points; a missing comparison is an explicit open item.
The Research Update gains a clearly labelled refinement section while preserving its
saved user-edited slide order and existing slide bytes. Regenerate its local viewer/PDF.
Record completed, failed, deferred and blocked items in a handoff and canonical lab log.
Do not merge branches. Ask for a merge decision when lanes finish.

## Glossary

- Bank: frozen spatial functions used to represent a solution.
- r: number of bank coefficients; M: number of smooth weak tests; m: integration points.
- N / L: mesh nodes / intervals; these differ in the historical 3D scripts.
- Weak residual: PDE imbalance integrated against smooth tests.
- Q1 / Q2: nonlinear-solver / quadrature qualification checks.
- E0 ST: space/time-refined reference, with conditional numerical uncertainty.
- QN / CN / BDF2 / ETD / IMEX: reused-Jacobian solver; Crank–Nicolson; second-order
  backward differentiation; exponential / implicit–explicit integration.
- Tight / tighter: independently verified more stringent nonlinear solves.
- POD: standard data-derived linear reduced basis; hyper-reduction samples its nonlinear work.
- Frontier: settings not beaten in both error and cost under the same query contract.
- Diagnostic subset: selected difficult cases, insufficient for a whole-population claim.
