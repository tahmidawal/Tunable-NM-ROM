**No WRONG findings.** Arithmetic and staged configuration checks pass; two claims need tighter wording.

| Item | Verdict | Finding |
|---|---|---|
| Reference judges, ε_ref, Péclet | **CORRECT** | Recomputed from recorded per-case statistics: all 64 cases accepted; all predicate decisions agree. |
| Reference controls | **CORRECT** | Both jobs’ six fixtures each fail exactly their intended predicate; recomputed `controls.all_fire = true`. |
| GPU logs | **CORRECT** | Both logs contain `jax_backend=gpu`, `x64+highest ok`, and `ALL-DONE`. |
| Registered A5.5 | **CORRECT** | Dev 0 and 2 fail at both contrasts; dev 1 and 3 pass. Both registered masks have zero eligible cases and are undecidable: **UNRESOLVED**. |
| A10 arithmetic | **CORRECT** | Independent recomputation agrees with `elig.py a10()`: all eight pass; both masks have 32/32 eligible cases. |
| A10 post-hoc disclosure | **CORRECT** | DESIGN and implementation explicitly label adoption after reference results. |
| A10 certification language | **NEEDS-RESTATEMENT** | The sum-of-differences “tail bounded” claim is an assumed error indicator, not a demonstrated bound. Small adjacent differences alone cannot exclude common bias or a later unresolved tail. |
| A10 reporting obligation | **CORRECT** | Reporting registered **UNRESOLVED** alongside the explicitly post-hoc measured gate is sufficient disclosure, provided the latter remains conditional and is never presented as registered confirmation. |
| Audit reproducibility | **NEEDS-RESTATEMENT** | These are recomputations from JSON statistics. Local reference lattice arrays are absent, so underlying norms, direction statistics and mutated control fields were not independently reconstructed. |

Recomputed reference ranges:

| Contrast | ε_ref minimum–maximum | Péclet minimum–maximum |
|---|---:|---:|
| 100 | 2.2316668e-5–1.3498417e-4 | 0.119409–1.426090 |
| 10 | 2.1223837e-5–1.1386246e-4 | 0.037761–0.450997 |

A10’s `distance + U_xc`, each below **2.5e-4**:

| Contrast | Dev 0 | Dev 1 | Dev 2 | Dev 3 |
|---|---:|---:|---:|---:|
| 100 | 3.6051081e-5 | 7.5355906e-5 | 1.4108934e-4 | 8.3678878e-5 |
| 10 | 3.3196371e-5 | 6.7423616e-5 | 1.0174790e-4 | 7.9610088e-5 |

A10 preserves the intended resolution **conditionally**: if `U_xc` bounds the cross-check’s error, the triangle inequality supports the reference resolution. Its arithmetic does not establish that premise.

| Staged ladder item | Verdict | Finding |
|---|---|---|
| Resources | **CORRECT** | GPU partition, H200, 128G, 3.75 h. Certificate ×1.5 = **3.6103223 h**, below the reservation; certificate config hash matches. |
| Keys | **CORRECT** | Exactly **329 of 338** registered keys: only the nine `ccv|L512|…` keys removed, as A7.4 specifies. |
| Reference directories | **CORRECT** | Both specified cluster reference output directories are configured; loader checks reference-field hashes against eligibility. Remote file presence was not checked. |
| Eligibility binding | **CORRECT** | Config hash, eligibility bytes and `.FROZEN` agree. Both masks’ source-result hashes also agree. |
| Execution controls | **CORRECT** | `stop_after_val=true`, C10 dev-only, `max_tries=8`; implementation honors these settings. |
| Budget | **CORRECT** | **0.577 + 5.710 = 6.287 GPU-h < 8**. Substituting lad1’s full reservation gives **7.627 GPU-h**, still within cap. Later reserves remain projections requiring their prescribed certificates. |
| Staging manifest | **CORRECT** | All **168** entries match their corresponding local files. |

**NOTED — forgery only:** matching hashes establish consistency, not independent authenticity of deliberately rewritten records.

No files changed, jobs submitted, or lab-log entries written.

**Overall verdict: CLEAN (no WRONG; NEEDS-RESTATEMENT items above).**