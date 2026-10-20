Training-only NNLS fixture for amendments A17/A18 (CPU, local). `mk.py` builds the ECSW contribution matrix G (1,920 ×
32,768) of the 2D L = 256, r = 16 dry-run cell from training draws only (24 snapshot trajectories, the A2.8 ECSW
training states); G.npz is not committed (regenerate with mk.py; its sha256 is in G.sha256). `t_old.py` (pinned to the pre-A17 `nnls.py` of lane commit c57098e00) +
`old_algorithm_t.log`: the pre-A17 algorithm (normal-equation refit, 2 % batches): 6,444 iterations / 7,563 bans to
256 points. `t4.py` + `t4.log`: numpy reference single-index Lawson–Hanson with exact least squares. `t7.py` + `t7.log`:
the A17 algorithm (exact QR refit, single-index entry). Logs record iterations, supports and training residuals.
