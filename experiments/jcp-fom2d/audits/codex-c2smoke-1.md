**SMOKE ACCEPTED** for proceeding to C2 production. No blocking fix found. Acceptance uses the recorded runtime validity/identity checks, with the evidence limits noted below.

I independently recomputed the numerical gates, selection and bootstrap from the [archived result](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/runs/smkc2/archive/output/c2time_smkc2/result.json).

| Check | Audit finding |
|---|---|
| G-RR | Six ROM errors reproduce P1; maximum difference **2.134e-16**. Independently rescored P1’s hash-verified saved fields against E0 on CPU. All six runtime validity checks pass. |
| G-FR | All 24 errors reproduce the accepted ladder smoke; maximum difference **3.799e-16**. |
| A-valid | All six scoring runs record valid, finite outputs. |
| Timing identities | All stage-0 checks and all **60** stage-1/2 triplet identity checks pass. No production subject vetoed. |
| Timing validity | **24/24** stage-1 and **36/36** stage-2 triplets valid, with no retries. Recomputed drift range: **0.953806–1.045503**. |
| T-dbl | Recomputed median **1.9812376153**, inside [1.8, 2.2]. Pinned code dispatches and synchronizes two separate executions. |
| T-fault | Recomputed drifts **2.014846, 1.963525, 1.967782**; all outside the band, final triplet invalid after two retries. |
| T-id | Corrupted-reference alias fails through the ordinary identity check and is the sole vetoed subject. Production CC subject remains unaffected. |

Stage-0 medians and stage-1 selection costs reproduce exactly:

| Candidate | Stage-0 seconds | Stage-1 normalized cost |
|---|---:|---:|
| PS N32, dt=0.01 | 0.0121079544 | 0.120514699 |
| CC L256, dt=0.01, t2 | 0.0349895549 | 0.367088358 |
| UB L512, dt=0.005, t2 | 0.0872550383 | 0.937894023 |
| PS N256, dt=0.0005 | 0.1443074611 | 1.454900033 |

The CC shortlist contains CC alone; the union correctly pools all four candidates from the per-family shortlists. Both common populations contain all six cases. Winners are CC and PS N32 respectively. No ties occur; the implemented tie order is stage-0 cost, then lexical key.

Frozen-selection contents and SHA-256 match the result. Its archived timestamp, **07:17:24 UTC**, agrees with the stage-1 log; the pinned runner writes it before entering stage 2.

Using only stage-2 durations, I recomputed per-case median normalized costs, their ratio, and **10,000 paired case-bootstrap draws, seed 0**. Both cells match the analysis exactly:

| Cell | S = cost(FOM)/cost(ROM) | 95% lower | 95% upper |
|---|---:|---:|---:|
| CC joint | 1.412722177474 | 1.309401251379 | 1.835364446759 |
| Union joint | 0.467043418811 | 0.456575208007 | 0.477185745083 |

All three combined-workload pilots pass their recorded evidence checks. UB/CC have **400/102 stages**, maximum Newton count **3**, and worst residuals **4.015e-11 / 2.666e-11**, below t0’s **1e-10** threshold. PS records finite output and running maximum. Peak memory is **96.592 GB**, below **138.102 GB**.

The archive records **10 compile events for 9 executable keys**, including the expected PS N256 rebuild after release. The log confirms H200, `jax_backend=gpu`, x64 and highest precision, with empty stderr and normal completion. Output checksums and the copied result match. The analyzer’s post-submission missing-cell-status correction does not affect these cells.

**NOTED — evidence/forgery limits:** hashes, ROM solver traces and complete pilot residual arrays are not retained; those checks cannot be independently replayed from this archive. Runtime predicates were inspected in the pinned code. Output fingerprints are not full-byte field hashes, and frozen-file chronology is supported by code/log/timestamp consistency rather than tamper-proof attestation.

**Smoke-only, no claim:** these timings suggest a modest ROM advantage over CC but a loss to the spectral comparator—consistent with no union-wide 2× ROM advantage for this target. They establish neither production C2 outcomes nor K2; the smoke universe is incomplete.

No files changed, no cluster operations performed, and no lab-log entry written.