**SUBMIT.** The round-3 defect is fixed. No remaining submission blocker was found.

1. **CORRECT — configuration.** All **19 groups / 33 arm instances** match the amended design: four head sizes × three seeds with `min_pass=2`; fixed-M=128 groups for K=4,8,16; incumbent; all eight linear rungs; fixed-M linear r=4,8,16. K=32/r=32 reuse follows A7. The ladder includes 16 through 320, selectable maximum 192, sub16/dev6 populations, and BE at dt=0.005.

2. **CORRECT — inputs.** All **12 compact-head SHA256 hashes** match the accepted training archive’s `compact_sha256` fields. Every compact head and the incumbent loaded through `hcommon.Head`, including source-checkpoint and bank checks. Candidate-code shapes agree with each K.

3. **CORRECT — staging and sbatch.** All **140 manifest entries** verify. All **136 provenance entries** match staged files, working sources, and committed blobs at `74a042add28d2745cc870304561df97a01c853d0`. GPU partition, A100-80G, absolute cluster venv, GPU preflight exit 42, x64, highest precision, and namespace-local outputs are present. Sbatch, reservation, and certificate agree on **11.25 hours**. Production release is recorded.

4. **NEEDS-RESTATEMENT — resource certainty, unchanged.** Independently recomputed projection: **7.377805 hours**, with the rounded 1.5× reservation of **11.25 hours**. It includes the full selectable ladder and check-rule scaling through 320², with measured dimension allowance **1.627563×**. This is a plausible estimate, not a proven worst-case runtime bound; the certificate correctly labels it accordingly. Memory remains plausible, but the smoke’s **17.25 GB** device peak does not establish production’s peak for the largest rules and three-head groups.

5. **CORRECT — freeze path and round-3 fix.** The structural checker now accepts correctly recorded null-rho failures. Read-only, in-memory tests through **both actual checker functions** accepted:
   - An immediately passing trail.
   - Null-distance/nonfinite-`w0` failure followed by a passing rung.
   - Genuine nonfinite-rho failures followed by a passing rung.
   - An exhausted trail with genuine nonfinite rho.

   Saved solves and diagnostics support offline selection recomputation and the `analyze.rules()` freeze gate. One nonblocking test-coverage limitation: the committed regression tests exercise null-rho exhaustion only structurally; this audit additionally exercised it through both gates.

6. **CORRECT — integrity, with recovery limitation.** Atomic replacement protects individual solve archives and result JSON. They are not a transactional pair, and no resume path exists; interruption can require rerunning. Incomplete output cannot pass the complete-result freeze gate.

**Required fixes: none before submission.** Retain the resource-estimate qualification. Extending committed combined-gate tests to cover genuine nonfinite-rho selection and exhaustion is recommended.

No files were changed, no generators or jobs were run, and no lab-log entry was written.