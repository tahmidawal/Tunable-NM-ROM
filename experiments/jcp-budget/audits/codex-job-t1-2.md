Audited HEAD `3e276a1a8`, locally only; no SSH or submissions.

| Item | Verdict | Evidence |
|---|---|---|
| Blocker 1: terminal failures | **CORRECT — closed** | `anaref.py:148–156` rejects every failed T=1 whole-case outcome. All 3,456 tested T=0.5 predicate combinations match round 1 behavior. |
| Blocker 2: eligibility provenance | **CORRECT — closed** | `ana7.elig_for()` takes W1/W2 eligibility and tier from T=0.5, requiring T=1 prefix acceptance and non-terminal status. W3 uses T=1. Independent truth-table checks pass; growth still uses one T=1 realization. |
| Blocker 3: runtime reference gate | **CORRECT — withdraw prior requirement** | `b2run.run_b7()` reads no reference files; reference loading is confined to optional B8. `ana7.load_newref()` checks JSON/NPZ hashes against the manifest, identity, and agreement with the eligibility record. E0 references use their own manifest-verifying loader. |
| Blocker 4: B7-only analysis | **CORRECT — closed** | `ana7.py:653,667,686` conditions B8 coverage and fine recomputation on B8 configuration. Pipeline includes the B7-only invocation. Existing T=0.5 B8 behavior is preserved. |
| A12 cache | **CORRECT** | Verification precedes `np.load`; corrupted and unlisted fixtures fail before loading. Committed listing paths map correctly to `<cache>/<run>/output/<step>/<file>`; 1024 selects nine files. Existing local and P1 loading paths retain their behavior. Remote availability was not tested. |
| Staging/resources | **CORRECT** | Each job verifies all **162 manifest entries** and **158 HEAD source files**. Certificates and configuration hashes match; certificates are unchanged from round 1. Limits remain **10 / 6.5 / 8.75 hours**, each ≥1.5× projection. H200, `gpu`, 240G, isolated directories, f64/highest and GPU preflight remain configured. |
| Other blockers | **CORRECT — none found** | No additional submission-blocking defect identified. |

Audit-process exception: I accidentally executed the regression script’s write to `checks/test_anaref_a9.json`. Its bytes remain identical to HEAD, but the write violated your instruction. No LAB-LOG, main, or other-worktree writes occurred.

**ref2t1: SUBMIT**  
**ps2t1: SUBMIT**  
**b1024t1: SUBMIT**

Blocking fixes: **none**.