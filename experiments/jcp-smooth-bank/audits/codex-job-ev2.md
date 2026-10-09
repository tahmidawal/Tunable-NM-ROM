**NO-GO as staged:** grid alignment and scoring are correct, but the new FOM comparators lack an acceptance gate and saved fields for independent verification.

| Item | Verdict | Finding |
|---|---|---|
| Payload hashes | **CORRECT** | All **117** staged manifest entries pass; **21** source/config files match HEAD. All **12** tr2 artifacts, **4** tr1b/base artifacts, and **4** tr2 large-file rotations match their manifests. |
| References | **CORRECT** | All **77** staged reference files match refdv’s output manifest; all **76** `f257` array hashes pass. References are marked accepted. |
| Variant mapping | **CORRECT** | Five specified banks and rotations are mapped correctly; base is re-evaluated and included in paired timing. |
| Batch execution | **CORRECT** | Five A100s on one GPU-partition node; f64/highest precision; per-device preflight; individual waits with failure propagation; timing runs on device 0 after all workers succeed. Shell syntax passes. |
| FOM grid/time alignment | **CORRECT** | `linspace` nodes exactly match reference nodes. Reference restriction strides are **2/1** for 129/257 nodes; common-grid FOM strides are **1/2**. `snaps[::10]` gives **0, .05, .10, .15, .20, .25**. |
| A1.7/A2.6 scoring | **CORRECT** | ROM and FOM common-grid errors use the same 129-node restriction and initial-field norm; evolved maxima exclude time zero. Own-grid FOM errors are recorded separately. |
| FOM acceptance | **WRONG** | [bankeval.py:530](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/bankeval.py:530) merely records finiteness/residuals. Neither collection nor coarse-control reporting rejects failed comparators. Require finite snapshots/residuals and an explicit residual threshold; the training generator’s existing acceptance threshold is **1e-8**. |
| FOM audit artifacts | **WRONG** | Comparator fields and per-step residual arrays are discarded. Save them so errors and convergence can be independently checked, consistent with A1.8. |
| Resources/provenance | **NEEDS-RESTATEMENT** | ev2 allocates up to **50 GPU-hours**, with no 80GB constraint. A6 explicitly updates ev1 only. Staged commit `d271c6393` precedes HEAD `aba760049`’s staging records; source hashes agree. |

Resolve the two FOM issues and restate ev2’s resource allowance before submission. No repository or lab-log files modified.