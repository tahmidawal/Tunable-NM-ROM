**Overall: READY for the design; implementation enforcement remains partial.** With A4 > A3 > A2 > A1 > base precedence, I found no remaining WRONG design item permitting an unsupported pass or mandating rejection of a correct implementation.

| Item | Verdict | Evidence |
|---|---|---|
| A4.1 design | **RESOLVED** | Requires identity across retries, fused flanks and all doubled-work executions; any subject identity failure blocks timing claims independently of drift exclusions. [DESIGN.md:656](experiments/jcp-p1-scaling/DESIGN.md#L656)–665. |
| Split triplets and retries | **RESOLVED in code** | Checks A/B/A identities on each attempt; `all_ok` covers every attempt and enters `valid`. [p1time.py:221](experiments/jcp-p1-scaling/p1time.py#L221)–229. |
| Fused flanks | **RESOLVED in code** | Both A flanks and fused B enter `hok`, which gates `valid`. [p1time.py:231](experiments/jcp-p1-scaling/p1time.py#L231)–237. |
| Doubled-work flanks | **Coverage resolved; enforcement pending** | `identity_ok` now checks `o1, oa, ob, o2`, but is only recorded: this driver does not invalidate the control or block claims. [p1time.py:259](experiments/jcp-p1-scaling/p1time.py#L259)–262. |
| Injected-fault option | **Implemented; demonstration pending** | Selected subjects receive deliberately wrong reference hashes. [p1time.py:131](experiments/jcp-p1-scaling/p1time.py#L131)–133. No smoke demonstration was executed in this audit. |
| Whole-design reread | **No additional WRONG item identified** | A2 removes the unjustified component-time gate; A3 limits the scaling claim and specifies trajectory identity; A4 supplies explicit failure consequences. |

The round-4 blocker is therefore **resolved as a design requirement, but not yet demonstrated end to end**. `analyze.py` is absent: its mandated subject-wide blocking, doubled-control invalidation, and injected-fault “verdict not reached” outcome remain acceptance checks for the separate code audit. This requires implementation and verification, not design restatement.