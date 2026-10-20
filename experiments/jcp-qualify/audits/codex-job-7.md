**Overall: FIX-FIRST — wait for a lane slot.** Staged job passes the audit, but live `squeue` shows all three permitted slots occupied: 5414290, 5416294, 5416492.

- **Selection PASS:** exact ordered match to `front3d.round_next`: 20 untested members from 436 candidates after removing all 17 round-1 failures.
- **Rules/populations PASS:** registered successors and check rules; cohort 923801, all 64 cases; q++ cases 0–7.
- **Provenance PASS:** all 141 staged source files match COMMIT `632b824e9`; all 145 manifest hashes pass; no extra staged source files.
- **Sbatch PASS:** GPU partition, H200, 1:45 limit, f64/highest precision, GPU preflight, isolated paralab directory.
- **Certificate PASS:** smk3 upper envelope 118.8 s; projection 1.125333 h; 1.5× rounded limit 1.75 h. Config hash matches.
- **Budget PASS:** 2.6405 spent + 12.5 reserved + 1.75 proposed = **16.8905 < 35 GPU-h**. No pending core reservation; defer item 3 applies.
- **Control omission ACCEPTABLE:** round-1 control fired (distance 0.00430295 > 0.00025); driver unchanged, analysis retains its gate. No per-round repetition requirement.

**SUBMIT once a slot opens and normal submission checks pass.** No files changed.