# valR320b (job 4143180) — collection note

- `cluster/collect.sh` (staged audit.py) passed checksums and `run_exit=0` but reported audit failures: all 642 (K16) / 846 (K32) were
  the 5 % strided sub-grid representativeness check on arms with error <= ~0.5 % (15^3 sub-grid of a fine-scale residual); exact
  recomputation max discrepancy 7.7e-16 (sub-grid) / 6.1e-16 (full field); independent 100k random-node full-grid estimate within 2.8 %.
- audit.py changed (disclosed, commit below): `passed` = exact recomputation + random-node check; sub-grid representativeness reported
  separately. Re-audited with the lane's current audit.py -> passed=True for both panels; summary.json regenerated.
- Remote dir removed manually after these checks (collect's --remove would have used the staged, older audit).
- Trained bank/heads copied to `inputs/vp_R320/` (committed; small) for the final job.
