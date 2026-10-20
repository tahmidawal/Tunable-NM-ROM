**Verdict: SUBMIT.**

1. **CORRECT — memory veto.** The actual front-analysis block rejects a nonempty `memory_rejected`, peak above `120e9`, or missing/null/non-finite peak. Exactly `120e9` remains accepted, matching A17.7.

2. **CORRECT — fixture and report fields.** Re-running the original 121 GB rejection fixture now produces `front_65.timing_valid=false`, `cost=null`, and both `state` and `state_worst` equal to `"missing"` for every timed point. All eight rejection variants passed. The report exposes invalidity; the detailed memory reason remains in the source result.

3. **CORRECT — no regression found.** Previously audited subjects, pre-ladders, bank identity, job configuration and timing logic are unchanged. Fixtures confirm ordinary costs/fronts still work and duplicate keys, bad hashes, unregistered cases/repetitions, drift and insufficient repetitions still suppress costs.

4. **CORRECT — refreshed certificate.** `checks/resources-e2f0.json` matches current HEAD `2864f16ad` and its recorded code commit `1c29ec6b6` under the certificate’s code-tree hash. Config, jobspec, inventory and timing hashes match committed files. Independently regenerated inventory and projection: **0.824708 h**; the **1.5 h** limit satisfies the required margin.

No files changed, arrays copied, jobs submitted, or cluster/SSH commands run.