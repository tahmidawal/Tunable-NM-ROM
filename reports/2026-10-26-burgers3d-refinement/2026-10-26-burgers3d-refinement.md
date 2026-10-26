# Burgers 3D refinement: recovered evidence, numerical gaps and controlled replays

This report reconciles existing evidence and the bounded refinement campaign. The historical
numbers are conditional estimates against approved E0 ST references; new numerical results
are accepted only with their independent audit status. This generated version does not
promote missing or subset evidence into full-validation qualification.

## Recovered state

The current experimental base is `93f399faad67f94ea2b56fe25f13003143e64694`. The inventory contains
684 speed-screening summary records, 390 detailed dimension-study records
and 62 later quadrature-round records, plus 2930 scored
observations and 143571 per-case records. These are different record types,
not a count of independent trained models or newly run experiments.

The nominal-width export reports 83 qualified entries;
the detailed configuration map contains 93; its broader qualified union
contains 101. Explicit test-space variants and later round-only
candidates explain why a single unlabelled count loses evidence. Raw score/coefficient case
ids are checked; intended registered populations are not reconstructed for every old schema.
The inventory exposes unknown fields and unsupported supporting schemas explicitly.

### Reference population and uncertainty

| Cohort | ST eligible / stored | Conditional uncertainty range | Selection allowed |
| --- | --- | --- | --- |
| dev4 | 4/4 | 1.37861e-05–9.15484e-05 | True |
| test32 | 32/32 | 1.15645e-05–0.000102639 | False |
| val64 | 64/64 | 8.91068e-06–0.000115885 | True |

Only development and validation populations can inform selection. Historical test observations
remain replication evidence. Sealed and reserved out-of-distribution cohorts remain closed.
Reference uncertainty is a conditional Richardson estimate, not a rigorous error bound.

### There is no single strongest bank configuration

| Purpose | Qualified historical setting | Median error (%) | Worst error (%) |
| --- | --- | --- | --- |
| Best observed median | `W768\|gl40\|fs\|CN\|0.01\|K1` | 0.157965 | 7.16593 |
| Best observed worst | `W512k8\|lat16384\|fs\|CN\|0.01\|K1` | 0.205405 | 3.13723 |

These full-validation figures select different objectives. A low median can coexist with a
large error on a difficult case. The test-space-expanded setting retains its explicit name;
it is not silently pooled with a nominal test-space configuration.

## Cost and accuracy on the exact saved timing workload

The timing-job coefficients were re-decoded directly with a NumPy evaluator and scored
against the approved reference fields. Its exact case ids are [0, 1, 2, 3]; it is
a historical workload, not the full validation population. All original timing repetitions,
payload identity links and source hashes are retained. Its timing source is job
`6056422` on `NVIDIA H200`. Costs include mesh input projection, reduced
evolution and full-field output under that job's query contract; diagnostic certification
and offline setup are not query work.

| Setting | Median full query (ms) | Median error (%) | Worst error (%) | Cases above 1% error |
| --- | --- | --- | --- | --- |
| `W256\|lat8192\|qn\|BDF2\|0.025\|k4` | 12.46 | 2.2531 | 2.9126 | 4 |
| `W256\|gl24\|qn\|CN\|0.025\|k4` | 14.26 | 0.76661 | 1.9338 | 1 |
| `W256\|gl24\|fs\|CN\|0.0125\|K1` | 16.11 | 0.38643 | 1.9268 | 1 |
| `W256\|gl24\|fs\|CN\|0.01\|K1` | 19.22 | 0.36431 | 1.934 | 1 |
| `W256M2052\|lat16384\|fs\|CN\|0.01\|K1` | 24.05 | 0.3649 | 1.8979 | 1 |
| `W512\|lat16384\|qn\|BDF2\|0.025\|k4` | 27.88 | 2.0546 | 2.9023 | 4 |
| `W512\|lat32768\|qn\|CN\|0.025\|k4` | 40.23 | 0.62926 | 0.97058 | 0 |
| `M2_512\|gl32\|qn\|BDF2\|0.025\|k4` | 41.8 | 2.0546 | 2.9024 | 4 |
| `M2_512\|lat32768\|qn\|BDF2\|0.025\|k4` | 42.66 | 2.0546 | 2.9024 | 4 |
| `M2_512\|gl32\|qn\|BDF2\|0.025\|kinf` | 43.42 | 2.0546 | 2.9024 | 4 |
| `W512\|lat16384s2\|fs\|CN\|0.01\|K1` | 43.56 | 0.15233 | 0.63143 | 0 |
| `W512\|lat16384s1\|fs\|CN\|0.01\|K1` | 43.6 | 0.15233 | 0.63029 | 0 |
| `W512\|lat16384s4\|fs\|CN\|0.01\|K1` | 43.62 | 0.15233 | 0.63079 | 0 |
| `W512\|lat16384s3\|fs\|CN\|0.01\|K1` | 44.11 | 0.15233 | 0.6329 | 0 |
| `M2_512\|gl32\|qn\|BDF2\|0.016666667\|k4` | 52.71 | 1.093 | 1.4818 | 3 |
| `M2_512\|gl32\|fs\|BE\|0.01\|K1` | 61.35 | 1.8884 | 1.9768 | 4 |
| `W512k8\|lat16384\|fs\|CN\|0.01\|K1` | 75.35 | 0.15095 | 0.57893 | 0 |
| `W768\|gl40\|fs\|CN\|0.01\|K1` | 291 | 0.13441 | 0.3448 | 0 |

![Accuracy versus total query cost on the exact historical timed population](figs/timed_frontier.png)

The full-validation accuracy and this timed subset are displayed separately. Combining the
former with the latter would hide the population difference and is not used as a new frontier.

## What the existing controls already distinguish

![Quadrature convergence with inherited thresholds](figs/quadrature_convergence.png)

Wide-bank fixed-sweep CN can pass nonlinear qualification while failing quadrature successor
agreement. Wide-bank large-step QN-BDF2 can instead pass quadrature but fail every-case
distance to a verified tight solve. Pooled stationarity alone can miss a small number of
important failed steps. Trust clipping is an observed counter; it is not by itself a cause.

Restarting from coefficient history resets the historical QN cache. Fresh-restart controls
are labelled separately; full-trajectory one-factor controls assess refresh-policy changes.
The completed quadrature trajectories are reused, and only missing successor evidence is run.

![Accuracy versus bank width with fixed tests and quadrature](figs/accuracy_width.png)

Open markers retain failed or unverified qualification. A decreasing raw error is not enough
to claim a qualified width dial. The square test-space control failed historically, so the
original registered causal test-space verdict remains blocked even where expansion helped.

![Error-budget interventions on their historical population](figs/error_budget.png)

Each bar is a median intervention distance on that member's own cases. The packages interact;
these bars are not additive contributions to total error. The representation floor and
time-discretisation package bind in different settings. The archived cohort limits remain.

## Mesh and input-regime dependence

![Historical same-allocation mesh cost and reported nonlinear work](figs/mesh_cost_iterations.png)

This mesh panel uses the mesh-free input/scoring-lattice contract and the historical deployment
mode policy, which may change the test list with the mesh. It establishes neither fixed-M
cost nor full-field output cost. Only the appropriately qualified curves support numerical
claims; unqualified curves remain visible. Full output and arbitrary mesh-sampled input
must be measured separately. Historical Jacobian counts are not reconstructed from iteration
counts where the original counter is absent.

![Descriptive in-range viscosity and steepness associations](figs/in_range_regime.png)

The regime plot uses only the existing validation population and the same E0 reference.
Amplitude divided by viscosity is a nominal steepness proxy. These associations are not
one-factor causal tests and do not qualify new steep-front or viscosity-extrapolation regimes.
The earlier two-dimensional steep-front failures cannot be silently promoted to 3D results.

## Three different comparisons

The historical confirmed ROM-to-ROM comparisons are:

| Earlier comparator | Speed ratio | Comparator Q2 | Qualification scope |
| --- | --- | --- | --- |
| C_old | 1.1491 | pass | qualified vs qualified comparator |
| C_W768 | 2.9528 | pass | qualified vs qualified comparator |
| C_W1024 | 9.8741 | fail | qualified configuration vs the deployed (unqualified) comparator |

They do not establish improvement over POD or a full solver. The historical POD headline
pooled diagnostic and qualified bank rows, and its hyper-reduction convergence was reported
rather than gated. Its measured accuracy/cost evidence is retained, with this qualification
gap stated explicitly. The new comparator lane tests the saved supports against the complete
unsampled reduced objective before any qualified comparison.

The earlier spectral result remains evidence against a general bank speed advantage on the
box. No new speed ratio versus a spectral or finite-difference solver is inferred from these
different jobs, outputs or populations. A repaired solver must face the same accuracy,
output and hardware contract, and POD receives the same numerical improvements.

## New bounded experiment results

### Solver lane

Audit/status: `diagnostic subset; consult per-job independent audit verdicts`. Scope: validation diagnostics; no new frontier or comparator win claim.

| Diagnostic setting/case | Fresh uncertified steps | E0 max error (%) | Raw median query (ms) |
| --- | --- | --- | --- |
| W1024\|baseline\|case30 | 1 | 3.900657 | 218.96 |
| W1024\|extra\|case30 | 0 | 3.900657 | 290.64 |
| W1024\|refresh\|case30 | 0 | 3.900678 | 227.05 |
| W1024\|trust\|case30 | 1 | 3.900657 | 216.1 |
| W1024\|strict\|case30 | 0 | 3.900666 | 300.03 |

These are selected-case diagnostics. A repaired certificate is not a full-trajectory Q1 pass or a physical-accuracy improvement. Raw timings do not establish comparative speed without the declared paired acceptance checks.

At `W1024|case30|step7`, the returned Jacobian condition number is 10.8825. The fresh tight solve uses 3 iterations and reaches stationarity 3.02833e-09; its field distance to the returned baseline step is 0.00344831. This is a fresh restart from identical coefficient history, not restoration of the original cached solver state.

Machine export and hashed raw evidence: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-solver/experiments/b3d-solver/EXPORT.json`. No cross-job speed-up is inferred.

### Pod lane

Audit/status: `READY_FOR_DIAGNOSTIC_PUBLICATION`. Scope: diagnostic selected validation subset; not a full-population accuracy/cost frontier.

| Frozen support | Maximum full-gradient discrepancy | Full-objective uncertified steps | New HR admission |
| --- | --- | --- | --- |
| k8 | 0.01542368 | 60/60 | False |
| k16 | 0.001233156 | 6/60 | False |

| Validation case | k8 to k16 distance | k16 to no-HR distance | k8 to no-HR distance |
| --- | --- | --- | --- |
| 0 | 0.001788379 | 0.000132795 | 0.001785279 |
| 5 | 0.003440759 | 0.0003461965 | 0.00366978 |
| 30 | 0.0028835 | 0.0002130666 | 0.002896786 |
| 43 | 0.002307133 | 0.0001234245 | 0.002315305 |

These failures disqualify the frozen supports under the new conservative whole-objective contract. They do not establish that the bank beats POD. The post-completion printing failure and recovered checkpoint are retained in the execution record.

Machine export and hashed raw evidence: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-pod/experiments/b3d-pod/runs/EXPORT.json`. No cross-job speed-up is inferred.


## Completed, failed, deferred and open

- Completed recovery: canonical fork, separate historical populations/references, explicit
  qualification scopes, source hashes, saved coefficients and timing repetition evidence.
- Completed analysis: exact historical timed-workload re-scoring and generated scientific plots.
- Failed historical controls remain visible: under-resolved quadrature, incomplete nonlinear
  convergence and square test-space controls; post-hoc selections and retracted combined
  selection claims remain labelled in the source inventory.
- Numerical diagnostics and any frozen full-validation follow-up retain independent design,
  code, job and results gates. See lane exports and the job ledger for current completion.
- Deferred until supported: a qualified full-population repaired frontier, qualified POD
  hyper-reduction comparison, paired fastest-full-solver comparisons, fixed-mode scaling of
  a repaired candidate and separate full-field output/offline costs.
- No bank retraining, sealed-cohort opening or branch merge is implied.

## Reproduction and provenance

Run `inventory.py`, `rescore_timing.py`, then `make_report.py` with the repository's absolute
Python environment. These analysis scripts use CPU NumPy and saved trajectories. They do
not rerun the historical solver grid. `data/SOURCES.json` pins historical sources;
`PROVENANCE.json` pins this report's direct inputs, and `VALUES.json` retains extracted
numbers and figure inputs. Compressed inventory and per-case JSON lines are backed up on
the prescribed code-only mirror. Full numerical reruns use each lane's frozen manifest.

## Plain-language glossary

- Bank: frozen spatial functions that turn a short vector into a field; r is their count.
- M: number of smooth functions used to test the PDE; m: number of integration points.
- Mesh intervals: divisions along each spatial axis; nodes include the two boundary endpoints.
- E0 ST: approved space/time-refined reference; its uncertainty is conditional on convergence assumptions.
- Error: largest output-time field difference divided by the reference initial-field norm.
- Median / worst: middle / largest case error; they answer different accuracy questions.
- Outlier count: number of cases exceeding the table's stated error threshold.
- Full query: input projection, reduced time evolution and requested output decoding.
- Q1: nonlinear-solver validity, stationarity and agreement with tighter solves.
- Q2: quadrature successor and own-state nonlinear agreement, with the finer rule checked too.
- Qualified: satisfies both checks on the required population; unknown evidence is not a pass.
- CN / BDF2 / BE: Crank–Nicolson / second-order backward differentiation / backward Euler.
- FS / QN: fixed full-Jacobian correction sweeps / a solver that reuses its small normal matrix.
- Jacobian: derivative of the reduced residual; stationarity measures whether further correction remains.
- Trust radius / damping: controls on correction size / regularisation of a nonlinear correction.
- POD / ECSW: a standard linear reduced basis / weighted sampling of its residual equations.
- No-HR: evaluate the complete reduced objective without weighted spatial sampling.
- Frontier: settings that no other admissible setting beats in both cost and the chosen error metric.
- Diagnostic subset: selected cases for explaining a failure, not a whole-population performance estimate.
- Post hoc / retracted: selected after observing data / no longer an accepted earlier claim.
- Q/T/S/F/e_P: quadrature / time / nonlinear solve / reduced-form intervention / representation floor.
- Mesh-free: use method-specific input and output points; full-field output requests every mesh value.
- Conditional / provisional: valid only with stated numerical assumptions / awaiting required evidence.
