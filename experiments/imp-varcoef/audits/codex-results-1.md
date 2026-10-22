- **CORRECT — decisive candidates.** All seven frozen test candidates are valid on all 32 pooled cases and pass the recorded qualification thresholds. Recomputed worst errors:

  | Target | PSI | ETD | FDMG | CCv |
  |---|---:|---:|---:|---:|
  | 3% | 1.575507% | 1.550524% | 1.235377% | No selected candidate |
  | 10% | 6.257625% | 8.889542% | 6.544793% | 7.491006% |

- **CORRECT — stage-1 selection.** All recorded selections reproduce from triplets. Union shortlists pool the family shortlists. ETD N48/mid wins both decisive targets: Δt=0.003125 and 0.0125 respectively. Frozen-selection copies and hashes agree.

- **CORRECT — stage-2 inference.** Both decisive FOMs and ROMs have six slots × three valid repetitions. Independently recomputed anchor-normalised costs and 2,000-draw paired-slot intervals match exactly:

  | Target | FOM cost | ROM cost | G | 95% interval |
  |---|---:|---:|---:|---:|
  | 3% | 0.280742 | 0.326956 | 0.858653 | [0.799044, 0.899606] |
  | 10% | 0.136499 | 0.147859 | 0.923172 | [0.877392, 0.942848] |

- **CORRECT — ROM consistency and controls.** In-job/inherited cost ratios are 0.990516 and 0.981935, within ±20%; qualification remains inherited. T-dbl fires (median 1.981891), T-fault rejects all three drift attempts, and T-id rejects its corrupted alias without vetoing the production source.

- **NEEDS-RESTATEMENT — timing validation is partial.** G-RR=true and A-valid=true, but **G-FR=false**: three 1%-target candidate queries and their kernels fail error reproduction. The decisive winners pass, so this does not invalidate their gate calculation. Do not describe all timing subjects as validated.

- **NEEDS-RESTATEMENT — ladder lineage.** L2 reruns overwrite some L1 records; candidate summaries retain prequalification pooled statistics. For example, PSI N64’s worst error is 0.006728204447 in candidates versus 0.006728231633 in final L2 runs. Qualification thresholds and decisive winners are unaffected; exact numerical reuse is not established.

- **CORRECT — outcomes and contrast isolation.** Post-hoc A10 gives **FAIL**; the registered rule gives **UNRESOLVED**. C=10 is the A7.4 deferred dev-only contrast, with no validation ladder or timing selection. Its unresolved-family entries do not leak into C=100. The 1% target has no qualified ROM denominator.

- **CORRECT — execution evidence.** All three production logs show H200, `jax_backend=gpu`, f64/highest checks, and completion; stderr files are empty.

- **CORRECT — staged test job.** Manifest and bindings match. Exactly seven permitted keys comprise the frozen union/family winners at 3% and 10%, C=100 only. References cover test cases 0–15 at the amended levels. Certificate independently reproduces **0.200576 GPU-h**; reservation and batch limit are **0.5 h**. This validates staging, not future execution.

- **NOTED — forgery-only:** deliberate file rewriting or intentional test replay, per A6.4.

**Overall verdict: CLEAN — no WRONG items; restatements above required.**