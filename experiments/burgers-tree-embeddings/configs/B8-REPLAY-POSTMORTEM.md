# Retained replay component postmortem

Prospective allocation cap: one GPU,10 minutes. Run compare_residual_replay.py on the cluster against committed verified retention manifests for b8treereplay (FAILED) and b8treepde (COMPLETED). No PDE solve rerun, architecture change, tolerance relaxation or modification of retained evidence. No final-test fields.

Job7387152 completed the continuous500-step solve and all five requested4096 outputs, then failed the bitwise initial-code criterion before component field checks. Verified trajectory hashes: forward b5e52f3db737c01a5e3cd460cda36ee8b82bb2a4bd166d11c6ea545340829f3f; replay 4d84854e22a9905d8dae2f140d0907670fefdfb2d4f7fc8026e51ce5ccb21db7. Stored-array audit found initial maximum absolute code difference2.6922908347160046e-15, L2 difference9.243727087245648e-15, relative code norm difference1.7552620991731516e-15;93 of96 coordinates differed. Both initial fits used8 iterations and8 accepted updates. This is consistent with floating-point-scale reproducibility differences, not proof of their precise runtime cause. Initial-field equivalence still requires model evaluation on the cluster.

Recover the skipped, originally specified component diagnostics: all501 integration-grid physical fields normalized by current forward field L2, and all five actual4096 outputs normalized by stored current-time reference L2, each<=1e-8. Verify unchanged physical stopping rules and all step convergence. Record iteration/acceptance differences. The new report always keeps original_overall_gate_passed=false, exact_replay_passed=false and passed=false even when physical_field_components_passed=true. No retroactive exact-replay pass and no paired timing claim.

Future replay wrapper exceptions now persist explicit failed_exception status and details rather than leaving status running. This change affects only future executions and does not alter the retained failed attempt.

## Stored15000-update family audit

Read-only audit of b8childgn/tree_015000.pkl verified SHA256 059799b616debd402b2b59efc94d1c9fcae2d2337dee80673f62f321d746ab13 against the run's15000 history entry. No model evaluation was used.

The72 selection fits have39 unfinished:24/24 ridge,12/24 two-pulse,3/24 primary. Ridge worst independent errors are12.3768589% uniform and10.9767924% prescribed at b8_dev006,t0. Their stations are1.2644e-3/1.9142e-3 and field updates6.0468e-4/2.3707e-3, with95/104 accepted steps out of120 continuation iterations. These are actively progressing unfinished fits, not converged capacity floors. Later ridge worst errors:4.64%/4.51% at.125,2.47%/2.78% at.25.

Two-pulse worst errors are3.07371%/2.91857%; primary worst2.16149%/1.57531%. All primary.125/.25 fits converge and have error<=.413%. This evidence prioritizes optimization diagnosis on expanded ridges, while leaving the current30000-update job unchanged. It does not establish architecture capacity failure or remove the independent representation gates.
