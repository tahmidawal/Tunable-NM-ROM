Audited commit `342e6c077775aedbbd27d28c353d2003bcbfa42c`. **The job definitions and certificates pass; downstream analysis needs fixes.** No files changed, jobs submitted, SSH/cluster commands run, or scratch files created.

1. **CORRECT — selection population and ladders.**  
   Configs match §3.3 as amended by A1.8–A1.9 and A2.6:
   - `a`: widths 256/512; `b`: 768/1024.
   - Three specs: FS-CN 0.01, FS-CN 0.0125, QN-BDF2 0.025.
   - Gauss: gl16/20/24/32/40/48/56/64/80; lattice: 8192/16384/32768/65536/131072.
   - Cohort 923801, all 64 cases; tighter cases 0–7.
   
   Bank-0/bank-1 configs are identical after removing bank identity and attempt/mode labels. Each job has **11,424 unique executions**, matching its saved inventory.

2. **CORRECT — seed-1 bank binding.**  
   `W1024s1` resolves to `inputs/banks3d/s1/bank.pkl`, SHA256:
   `0a4116f5fbca5099163f4122acd30cf5d2530043373968ff05a8385eefc408d5`.
   
   This matches `BANK.json`, both training checksum manifests, and the local bank bytes. Config and job-spec `cluster_inputs` point to the **e2t3d1 archive**. [stage.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/cluster/stage.py:133) copies and verifies it before execution; `c3common.load_bank` verifies again. Seed-1 selection settings never resolve to incumbent W1024. The context additionally loads the historical M2 timing reference, but does not substitute it for selection settings.

3. **CORRECT — output interfaces and bank-aware scoring.**  
   Driver coefficient keys, `cases|` mappings, ADT keys, certificates, and ladder records match the readers. `e2analyze3d.py` checks the two job roles, selection job IDs, coefficient-archive hash, scorer identity, decoder hashes, and E0 provenance. `e2score3.py` installs the requested bank mapping and clears decoder caches. The copied `d3run.py`, `c3common.py`, and `s3ctx.py` are byte-identical to the local dims versions.

4. **NEEDS-FIX — target-unverified handling is incomplete in 3D.**  
   [e2analyze3d.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2analyze3d.py:229) uses `cell['qualified']` for diagnostic selected slots instead of preserving their available own-family qualification, and does not propagate A2.5 qualifiers through downstream rows. [e2verdicts.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2verdicts.py:132) collects qualifiers only for 2D.

   Furthermore, a target-unverified search returning `none within ladder` becomes definitive `absent` in `keys_3d`, allowing a target-dependent selection verdict into robustness counts. Preserve target certification explicitly, exclude such selection labels, and retain labelled successor-Q2/error results.

5. **NEEDS-FIX — registered downstream summaries are missing.**  
   `e2analyze3d.py` computes H-acc-r only for gl64, omitting the additional selected-rule comparisons described by A1.9. [e2verdicts.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2verdicts.py:155) explicitly skips 3D spread-ratio computation; its comment does not supply the missing cross-seed calculation. Both fixes can use the planned saved outputs.

6. **CORRECT — A13.4 timing arithmetic and limits.**  
   Independently reproduced all per-rule source means/maxima and verified source hashes. Setup is **2,213.613 s**. Unmeasured gl80 classes equal 2×gl64; W1024 uses measured gl80 values.

   | Jobs | Projection, including 1.25× | Limit | Limit/projection |
   |---|---:|---:|---:|
   | e2q0a, e2q1a | 1.823194 h | 2.75 h | 1.50834 |
   | e2q0b, e2q1b | 4.801016 h | 7.25 h | 1.51010 |

   Dims m3b measured **2.843944 h**; this timing model projects **3.831223 h** for its actual inventory. The model is a defensible estimate, not a runtime guarantee: dims substitutes BE 0.01 for the new CN 0.0125 spec, and retrained-bank adaptive costs remain unmeasured.

7. **CORRECT with evidence limitation — memory.**  
   Heavy-cell smk3c peaked at **79.178 GB device memory**, below the 120 GB acceptance bar; m3b peaked at **45.697 GB**. Rung tables are released as the ladder advances. H200 device capacity is supported by these measurements. **240G host memory appears adequate for evaluation**, but I found no local evaluation MaxRSS measurement establishing its exact headroom. The 273.9 GiB training peak is a different workload and cannot certify evaluation memory.

8. **CORRECT — certificate binding and training-spec equivalence.**  
   All four certificate code-tree, job-spec, config, inventory, and timing hashes match. The certificate commit predates HEAD, but its bound code-tree hash is identical. Current e2t3d1 equals its executed job spec; e2t3d2/3 differ only in seed and corresponding attempt/output/completion names.

**Verdicts:** `e2q0a` **FIX-FIRST**; `e2q0b` **FIX-FIRST**; `e2q1a` **FIX-FIRST**; `e2q1b` **FIX-FIRST**.

The blockers are findings **4–5**, not the selection jobs or timing limits. Fix the downstream analysis and refresh certificates after the code-tree changes.