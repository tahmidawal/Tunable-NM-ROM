### burgers-heldout — milestone 1: new frozen Burgers model `cpod512` built (bh1, job 4139115); bh2 (1024² selection) queued

Worktree `worktrees/2026-09-21-burgers-heldout` (branch `exp/2026-09-21-burgers-heldout`), namespace `bheld_20260921`. Resumed after the pause; DESIGN.md pre-registered and Codex-audited (Codex cannot run a shell here; files inlined; blocker A-1 accepted: hold64 was opened by earlier lanes, so an untouched cohort fresh64 = params_draw(20260929,64) is added; sel32 = params_draw(20260927,32) is the selection cohort; both checked disjoint from all training draws and cohorts).

**Model.** Rank-512 mesh-free bank inside bank-floor's `cat1024` span: field-metric POD of the 131072 extracted training states (per-state weighting, chosen on sel32 among raw/state/u0), folded into ONE separable parameter set (parity 2.7e-13), Gram trace matched to the incumbent. Worst evolved projection floor at 256² in the error metric (/‖u0‖): sel32 incumbent inc512 0.513 %, cat1024 0.131 %, **cpod512 0.183 %** (dev6 0.094 / 0.030 / 0.037 %). Head: incumbent recipe unchanged (sep_coeff_extract + sep_hfit_run arm mid from scratch), fresh-test oracle mean 0.434 % vs the incumbent's own run 0.520 %. Directions: qtd02 rule, rank 512, orthonormality 3.1e-13. All gates passed (Gram identity, truth ≤1e-8, parity, orthogonality). Model/directions git-ignored under `experiments/burgers-heldout/ckpt/bh1/`, SHA256 in the committed `CKPT-MANIFEST.json`; remote dir deleted.

**Wrong / retracted.** The first local smoke failed the compressed-bank orthogonality gate (1e-5) because the in-span POD whitened with a Cholesky of the Gram (cond(G)² ≈ 5e9); replaced by a thin-QR whitener before any job. No GPU number retracted.

**Running / next.** bh2 = job 4142941 (1024², dev6+sel32, rung ladder q0…q384, lat64 certificates + lat16 control, floors at the mesh, 3 reps): applies the pre-registered selection rule. Then bh3 (4096² dev6+hold64) ∥ bh4 (4096² fresh64). 2 of 8 jobs used.

