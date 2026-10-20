Audited **405e43c9e**, read-only. No files changed, jobs submitted, SSH used, or arrays copied.

1. **NEEDS-FIX — solver change correct; non-eligibility enforcement incomplete.**  
   [refrun.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ref/refrun.py:192) preserves the base tight retry, all judges and outcome paths, and records `tight_after_refine`. Only the refined-triple tight retry gains the flag condition; skipping it retains `unresolved-floor`.

   However, [anaref.py:271](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/anaref.py:271) requires **window** acceptance, not whole-case acceptance. I executed the production eligibility block with a whole-case `unresolved-floor`, an accepted W2 window, and passing other gates: it returned **`eligible=True`**. `ana7.py:214` consumes that flag unchanged. This violates A9’s terminal, non-rankable rule.

2. **CORRECT — partial collections, within available local evidence.**  
   `ref2b` contains only cases **7–9**, with **9 promoted files**; `ref2c` contains only **20–30**, with **33 promoted files**. Both have `PARTIAL` markers and `complete=false`. Every completed case has JSON, NPZ and fine-file entries with matching result-manifest hashes. All locally available output hashes passed. No unfinished-case file appears in either promotion manifest. Fine files remain remote; their verification is supported by the hash-checking promotion code and recorded success markers, not independently rechecked here.

3. **CORRECT — coverage and no-overwrite compatibility.**  
   `ref2a`: dev6 **0–5**, val32 **0–6**; `ref2b`: **7–9**; `ref2c`: **20–30**; `ref2r`: **10–19, 31**. Each requested case occurs exactly once, with no manifest-derived destination overlap or gap.

4. **CORRECT — staging and certificates match the declared estimate.**  
   Both jobs have **156 source files matching HEAD**, and all **160 staged manifest entries** pass. Config hashes match certificates; both flags are false.

   | Job | Recomputed projection | A9 reserve | Sbatch limit |
   |---|---:|---:|---:|
   | ref2r | 3.4651 h | 2.4486 h | 5.25 h |
   | ref3b | 4.2635 h | 2.8199 h | 6.50 h |

   Limits cover 1.5× projections. Both request H200, 240G host memory, GPU partition, isolated paralab directories, f64/highest precision and GPU preflight. Refinement memory estimates are **39.65 GiB / 104.49 GiB**, below the configured approximately **118.33 GiB** allocation. These remain estimates: the preserved base-tight-then-refine path can still run the new refinement level at tight tolerance.

5. **CORRECT — no additional submission blocker identified from local evidence.** Live queue, disk and remote state were outside this audit.

**ref2r: FIX-FIRST. ref3b: FIX-FIRST.**

1. Enforce A9’s whole-case terminal exclusion in eligibility while preserving E0-based W1 scoring. Add a regression check for `unresolved-floor` with an otherwise accepted W2 window, then commit and restage both jobs against the corrected HEAD.