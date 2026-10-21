1. **Source lane corrected.** The brief pointed at `burgers3d-span`; the Table 1 Burgers 3D rows and the brief's job ids
   are from `burgers3d-retry` (@642ab587). All NM-ROM/FOM code, the model and the configs come from there.
2. **Cohort opened again.** The 32 sealed held-out cases (seed 923901) were already opened by Table 1; this is a later
   opening. No choice of any kind is made on them (sizes fixed before, checkpoints chosen on seed 923751).
3. **Hardware differs from Table 1 at 64³/128³.** Table 1 timed on H200; these panels ran on A100-80G (one H200 free at
   submission). NM-ROM and FOM ms therefore differ from Table 1; every ratio here is same-job.
4. **256³ split into two allocations.** Operators U-Net/Transolver/DeepONet were trained on an A100-80G (`t257`) because
   the H200 queue estimate was ~9.5 h; the FNO attempt and the panel ran on an H200 (`p257b`; the panel failed, see 11). DeepONet (don-s) ran out of memory at micro-batch 1 on the
   A100-80G in `t257` and was re-attempted from scratch on the H200 in `p257b` (the first H200 submission `p257` was
   cancelled while pending, never ran). The 3000 s budget buys
   less work on an A100 than on an H200. 
5. **Training data size.** 1536 trajectories (all the NM-ROM used) at 64³/128³; only the first 256 at 256³ (host
   memory). Validation = the NM-ROM's bank-validation seed 923751 (64 cases; 32 at 256³), not a split of 923701.
6. **No tuning.** One size per family (ns3d-operators 64³ validation selection at 64³/128³; smallest sizes at 256³), one
   seed, 3000 s each. Sizes/recipes were chosen for Navier–Stokes, not Burgers.
7. **Dirichlet handling.** Operators act on the (n−1)³ grid that includes one zero wall plane per axis (periodic
   extension = the Dirichlet walls); coordinates are input channels. No extra padding.
8. Operator training targets use the NM-ROM's data tolerance (ntol 1e-8) while scoring uses the tight reference
   (ntol 1e-10); the difference is far below any operator error.
9. **64³ panel rerun.** The panel of job c65 stopped at the cohort-fingerprint assertion (GPU-dependent sha256, see
   DESIGN A3) before evaluating anything; the 64³ panel is the separate panel-only allocation `pn65` using the c65
   checkpoints. Cohort identity is verified by the reference fields matching Table 1's saved lattice fields.
10. **FOM rule moves at 128³.** Loosely terminated Newton–BiCGStab is hardware sensitive: at 128³ the Table 1 FOM
    setting (Δt .01, ntol 1e-2, ltol 0.1) scores 1.468 % worst on this A100 vs 1.39 % in Table 1's H200 job, just above
    the NM-ROM accurate arm (1.464 %, reproduced exactly), so the paper rule picks Δt .01 ntol 1e-3 in this job. Both
    ratios are reported; Table 1's own speedups are not reproduced on A100 hardware.
11. **256³ FAILED (not resubmitted, coordinator instruction).** H200 job `p257b` (4292776, 2 h 05 min) regenerated the
    data, trained don-s (micro-batch 1, 8 epochs) and fno-s (micro-batch 1, 17 epochs) to the 3000 s budget, then the
    panel died while building the NM-ROM tables (`tables.build_tables`, bank row block): JAX RESOURCE_EXHAUSTED allocating
    16 GiB. Cause: this panel process capped JAX at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.72` (≈101 GB) to leave room for the
    PyTorch operators; Table 1's 256³ job built the same tables with 0.92 (≈130 GB). Nothing was evaluated on the held-out
    cohort at 256³; the 256³ lines in the table are training/validation records only. The same-process JAX+PyTorch
    design cannot fit at 257 nodes without changes (e.g. building tables before PyTorch with a higher JAX fraction).
