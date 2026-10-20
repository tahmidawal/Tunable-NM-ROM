**SUBMIT for all three jobs, individually through the cap-enforcing submit script. Only one lane slot is currently available.**

| Audit item | q3r4 | m3a | m3b |
|---|---|---|---|
| 1. Staging, manifest, commit, sbatch, namespace | CORRECT | CORRECT | CORRECT |
| 2. DESIGN/config/member-list compliance | CORRECT | CORRECT | CORRECT |
| 3. Analyzer inputs, A4 outputs, width coverage | CORRECT | CORRECT | CORRECT |
| 4. Memory and certificate plausibility | CORRECT | CORRECT | CORRECT |
| 5. Ledger and submission caps | CORRECT | CORRECT | CORRECT |
| **Overall** | **SUBMIT** | **SUBMIT** | **SUBMIT** |

- **Provenance:** all 163 source files per attempt match their Git blobs at `e658ce23911949955fddacf217100de34cdb06a8`; all staged manifest hashes pass. Separate `jd_*` names and paralab directories, H200, GPU preflight, f64, highest precision, and output checksums are configured correctly.
- **q3r4:** exact set equality with qualify’s `threeD.exposed_untested_after_last_round`: 16 members, no omissions or extras. Every `q+`/`q++` link is correct, including lat65536 → lat131072 → lat262144. The under-resolved control is present.
- **m3a/m3b:** their 18/12 ladders exactly match the corresponding `MANIFEST-cells3d.json` entries. All three specs run on 64 cases, with tight ADT anchors throughout and tighter anchors on cases 0–7. Comparator Q2 comes from the prescribed BE ladders.
- **Analysis contract:** production and ADT storage keys match `analyze3d.py`; case identifiers, `dist_next`, successor validity, and six-state `rho` are supplied. Per-spec and width-intersection $m^*$ follow A1.5/A1.7. Coverage is computed from qualified configurations; the jobs cannot guarantee that every width passes.
- **Memory:** smk3 exercised W1024 at $M=4097$, including gl48→gl56→gl64 and lat131072→lat262144 with ADT queries. It also exercised the heavy three-rule round member. Peak was **52.646 GB**, below the **120 GB** acceptance ceiling. Production cases execute sequentially; this is credible memory evidence, not a guaranteed production peak.
- **Certificates:** independent recomputation gives **1.2040 / 4.3220 / 6.6677 h**. Config hashes match; limits **2 / 6.5 / 10.25 h** satisfy the required 1.5× margin. These remain estimates: the round model does not separately itemize `q++` or control work.
- **Ledger:** validation passes. Spent + reserved + pending core = **47.4803 / 55 GPU-h**, leaving **7.5197 h**. Live queue verification found `f2a` and `f2br1` running, with five account jobs total. Submit **one** attempt now; subsequent attempts must wait for capacity. The submit wrapper enforces lane cap 3 and account cap 8.

No files changed, no jobs submitted, and no scratch copies created.