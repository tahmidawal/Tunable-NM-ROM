**Verdicts: g1 SUBMIT; g2 SUBMIT; g3 SUBMIT; f1 SUBMIT.** No blocking honest-path defect found at staged HEAD `6f61b238a35f4e55282e28cc60223bd406eaedd3`.

| Job | Inventory checked | Certificate ×1.5 / allocation | Verdict |
|---|---|---:|---|
| g1 | M2_512: 11,808 case-runs; W256: 14,880 | 2.99 / 24 h | **SUBMIT** |
| g2 | W512: 11,808 case-runs; W768: 8,736 | 8.49 / 24 h | **SUBMIT** |
| g3 | W1024: 8,736 case-runs | 4.04 / 24 h | **SUBMIT** |
| f1 | 17 FD configurations; 15 PS configurations | 0.92 / 12 h | **SUBMIT** |

Case-run counts include the deployed-rule tighter ADT checks, but exclude replay and timing calls.

The configs cover all 64 validation cases, seed **923801 only**. Settings, deployed/reduced rules, schemes, solver combinations, adaptive starts, and all four steps—including `0.05/3`—match the amended design. K1–K3 are collected on deployed rules. Tight ADT covers every case/configuration; tighter ADT covers cases 0–7 on deployed rules, with finalist reduced-rule checks correctly deferred to p1. The f1 ladder matches A1.9, calibrates all three tolerance levels on cases 0–3, rejects an unconverged tight reference, and collects warmed, hash-checked timing repetitions.

All four staged manifests verified without mismatches. Staged source/config provenance matches HEAD; staged imports and dry-run inventories succeeded. Batch scripts request **H200, gpu partition, 200G**, the stated hours, GPU preflight exit 42, x64/highest, E0 environment, and `REF1_PATH`; outputs and caches reside under each job’s namespace. The relevant execution code is unchanged from the accepted smoke.

**Resource qualification:** I reproduced the certificate numbers, but would describe them as estimates with allocation headroom, not established upper bounds. `resources.py` uses median smoke timings and assumes only one-quarter of cases need replay; it does not explicitly budget replay compilation, tighter-ADT compilation, or all FOM compilation/scoring/I/O costs. Smaller-rule and narrower-bank proxies also do not bound nonlinear iteration counts. These are non-blocking here given the allocations: the smoke grid completed in 2,075.6 seconds with approximately 12.18 GB peak device memory, and the L=384 pilot completed in 353.3 seconds with approximately 24.59 GB peak. Neither demonstrates full production runtime, but the evidence supports feasibility.

Unexercised production combinations remain: W256/W512/W768, several reduced rules, ROM step 0.0125, the full FOM ladder, and the 64-case workload. I verified that all requested rules exist and load, bank spread entries exist, and step canonicalization supports the new step. Cases run sequentially; expanding to 64 does not create a 64-member solver batch. I found no deterministic failure requiring another smoke.

The outputs retain the trajectories, certification, ADT distances, replay records, scores, and timing arrays needed for **H1.1–H1.3, H1.4 screening, H2 screening, and subsequent H4.3 matching**. No missing evidence currently forces a rerun. Final H1.4/H2 claims still require the pre-registered direct confirmation panels; f1 supports matching but does not itself establish matched speedups.

A4.5’s remaining analysis, freeze, matcher/K10, and confirmation/qualification decisions remain safely deferrable **until before their respective decisions**. K4’s stored check fires; K9 is accepted here as your supplied evidence, not independently reproduced.

No repository, git, or lab-log changes; no submissions or GPU use.