No BLOCKER or MAJOR findings. One smoke-coverage limitation remains.

1. **MINOR — 2D has no cases outside the convergence-check set.** [smk2bud.json:81](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/configs/smk2bud.json:81) selects `dev6/{0,2}` and `val32/0`, with `check_val32=1`. All three therefore receive every convergence-check trajectory. The smoke exercises selected-arm checks, but cannot distinguish their all-case coverage from ordinary check-set coverage. Adding another validation case while retaining `check_val32=1` would cover that production distinction.

Verified:

- **N16 fixed**, including the staged copy. [c2bud.py:186](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2bud.py:186) reruns `conv_check()` after restoring the original pair. Executing the extracted control flow with mocked outcomes—initial pass, successor pass, and double failure—produced **108 unique rows per setting**, with **zero inventory mismatches or duplicates** in every branch.
- **N17 fixed.** [budget3d.py:220](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget3d.py:220) guards empty finalist reductions. The extracted missing-trajectory fixture returns `cases=0`, null worst distances, and `qualified=False`, without raising.
- **Staging/provenance passes.** All **47 / 33 manifest entries** verify. Staged Python files match both pinned commit `ead593720e5c8fc4e9439fcfe82ee6795ea2804c` and current source. Config differences from the commit are precisely the reference-path rewrites.
- **Both sbatches pass inspection and shell syntax checks:** `gpu` partition; A100 with `a100-80G` constraint / H200; absolute cluster venv; highest precision and x64; GPU preflight exits 42 on CPU; logs, output, temporary files and caches under their respective `jcp-comb/<attempt>` directories.
- **Runtime project imports are covered**, including transitive vendor dependencies and required banks/rules. Unstaged 2D imports in the shared 3D core occur in functions unused by 3D budget mode.
- **2D phase coverage:** convergence check, reachable successor and restored-pair branches, B1, `gauss8` control, budget ablations, ideal and qualification runs, selected-arm checks.
- **3D phase coverage:** deployed/converged/finer roles, B1, `lat256` control, ideal and qualification runs. Extracted scheduling produces **77 unique rows**, exactly matching analyzer inventory; selected trajectories cover cases **0,1,2** on all three rules, while ordinary budget runs cover **0,1**.
- **Analyzer compatibility:** keys and case IDs agree; `--allow-smoke` permits analysis without production grid reproduction. Staged references match registry hashes and expected metadata/shapes; local registry reference paths exist.

No unconditional startup crash or silent evidence error was established for these configurations. This was static inspection plus extracted, mocked control-flow checks—not GPU execution or analysis of completed outputs. No files or lab-log entries were modified.

**SUBMIT**