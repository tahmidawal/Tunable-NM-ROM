Audit of `19ad2c8a0`, local files only. No files changed, no SSH, no submissions.

| Item | Verdict | Evidence |
|---|---|---|
| 1. Registered design | **NEEDS-FIX** | FD configs cover dev6 0–5 and val32 0–9,19; PS covers {0,19}. B7 uses sub16, excludes 19, and its 11 members exactly equal the union of `b1024a/b`. Both FD refinement flags are false. However, failed T=1 references can remain eligible, and T=1 analysis uses the wrong source for W2 eligibility. |
| 2. Horizon and layout | **CORRECT** | FD, PS and ROM paths propagate T=1 and produce 21 outputs, spaced 0.05 apart. Windows are indices 1–5, 6–10, 11–20. References use `T1/`; promotion preserves that directory and refuses overwrites. PS supports the registered separate escalation attempt at T=1. |
| 3. Staging/resources | **CORRECT**, with estimates | Each job’s 161 manifest entries verify; all 157 provenance sources match HEAD. Certificate config hashes match. Scripts request H200, 240G, `gpu`, isolated paralab paths, f64/highest precision and GPU preflight. Limits/projections are 10/6.622, 6.5/4.330 and 8.75/5.783 hours—all exceed 1.5×. The FD output buffer is 10.497 GiB; the cited 17.62 GiB reference smoke and 87.46 GiB larger-mesh ROM pilot leave substantial headroom below 128.62 GiB. These are supporting estimates, not measured T=1 peaks. |
| 4. Runtime reference verification | **WRONG** | `b1024t1` never reads references. Its config contains only B7; reference loading belongs to B8. Missing or corrupted cluster T=1 references therefore cannot stop this job. |
| 5. Other blockers | **NEEDS-FIX** | `ana7` unconditionally requires B8 outputs; this B7-only job necessarily fails coverage. Following the pipeline’s `--fine` invocation also raises a missing-`b8` configuration error. |

The A1.5 prefix gate **is enforced**: `anaref` compares matched levels over the first eleven outputs using the G-R2 bar; a missing or failed prefix makes T=1 eligibility false. `ana7` consumes that eligibility.

**Job decisions:** `ref2t1` **FIX-FIRST**; `ps2t1` **SUBMIT**; `b1024t1` **FIX-FIRST**.

Blocking fixes only:

1. **Enforce terminal T=1 failure in eligibility.** [anaref.py:148](experiments/jcp-budget/anaref.py#L148) excludes only A9 *refined* floor failures. A read-only synthetic check returned `eligible=True` for a T=1 whole-case `unresolved-order` failure with accepted W3. Reject failed T=1 cases regardless of refinement, while preserving the E0 W1 policy.
2. **Preserve A1.5’s W2 provenance.** [ana7.py:621](experiments/jcp-budget/ana7.py#L621) loads eligibility exclusively at the job horizon. For T=1, use T=0.5 W2 eligibility/validation status and T=1 W3 eligibility, retaining the registered same-realisation growth calculation.
3. **Add the requested runtime reference gate to B7.** [b2run.py:244](experiments/jcp-budget/b2run.py#L244) needs a preflight checking every required sub16 T=1 JSON/NPZ against the cluster `REFS-MANIFEST.sha256`, failing on missing files or hash disagreement. Do not require fine files: these FD configs intentionally omit them. Restage the affected job.
4. **Support B7-only analysis.** [ana7.py:629](experiments/jcp-budget/ana7.py#L629) must require B8 files only when B8 is configured; guard the `--fine` path similarly and correct the T=1 pipeline invocation.