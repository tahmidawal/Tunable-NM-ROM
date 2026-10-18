# Burgers 3D retry — handoff prompt for a separate session

This file is a self-contained brief for a new session that tries to get a usable Burgers 3D row for the ICLR 2027 paper (deadline 2026-09-25 AOE). It records the state after the first attempt (lane `burgers3d-span`, which came out negative on held-out cases) as of 2026-09-23 evening.

---

## Prompt (paste into the new session)

You are continuing the Tunable NM-ROM project in `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude`. Your job is to try to get a **Burgers 3D** result that holds up: an accurate setting and a fast setting from one frozen model, each faster than Newton–BiCGStab in the same GPU job, on sealed held-out cases. The ICLR deadline is **2026-09-25 AOE**. Report to the user by **2026-09-24 20:00 EDT** at the latest, whether the result is positive or negative.

**Read first:**
- `CLAUDE.md` (repo rules: Tufts cluster, GPUs, one job per directory, lab log, never push).
- The top of `LAB-LOG.md`, plus its `burgers3d-span` entries.
- `worktrees/2026-09-23-burgers3d-span/experiments/burgers3d-span/`: `DESIGN.md` (amendments A1–A4), `PROGRESS.md`, and `reports/2026-09-23-burgers3d-span-accuracy-speed.md` with its summary JSON (sha256 `b355f86f…`).
- The paper's method, `paper_latex/main.tex` §3, to see what the method is. **Do not edit the paper.** The main session owns it.

**What the first attempt found:** branch `exp/2026-09-23-burgers3d-span` @ `2c9e7b39`, local only.
- **Model.** One frozen coordinate-network bank, R = 512, trained on fields from all three meshes (33/65/129 nodes), rotated once into an ordered bank. Its best possible (projection) error at full width is 0.93 / 1.63 / 2.20 % worst.
- **Advection term.** A precomputed quadratic tensor. It matches the solver's upwind stencil only where the state is nonnegative, and it is certified with ρ ≤ 1e-2 against the bar of 0.116. The lattice quadrature fails its certificate at R′ = 512.
- **Heads.** The trained nonlinear heads are poor (21–25 %), so every setting is a linear span.
- **Held-out result (32 sealed cases, A100-80G, worst evolved error), negative:**
  - accurate, span R′ = 512: 1.32 / 2.39 / 3.48 % at 0.02 / 0.03 / 0.21× Newton–BiCGStab (32³ / 64³ / 128³);
  - fast, span R′ = 96 / 128 / 192 with Δt = 0.01: 8.29 / 8.31 / 7.89 %, above the 5 % bar;
  - speed of the fast setting: 1.05 / 1.19 / 3.01× under the paper's rule, 0.64 / 0.76 / 2.01× against a full-order setting matched to its own error.
  - Validation underestimated the held-out worst case by 2–3×.
- **Diagnosis:**
  - **(i) Cost.** Tensor advection costs O(M·R′²) per iteration, so the accurate setting is slow, while the 3D meshes are small enough that the full-order solver is very fast (7.7–61 ms).
  - **(ii) Accuracy.** The error is limited by the bank's floor, and the bank generalises poorly from the training data.
- **Old run (b3d004, 33³).** 380 ms, because every iteration did full-grid decode and advection with no quadrature.

**Ideas, in the order I'd try them. You decide; justify the choice in a new DESIGN.md before any timed job.**
1. **Larger meshes with the same frozen model** (256³, possibly 192³). The full-order cost grows steeply while the reduced cost barely changes, so the speedup should grow. Check accuracy there: the bank's floor rises with the mesh.
2. **A cheaper accurate setting.** Use a certified quadrature for advection at large R′ instead of the full tensor, or pick a smaller R′ that still beats the full-order solver. Consider the time step and the fixed-sweep solver as knobs, fixed in advance.
3. **Better generalisation.** Retrain the bank on more trajectories or more varied training data. This is the real accuracy fix, but it is slow, so do it only if it fits the deadline.
4. **Fix the nonlinear head.** Diagnose why it trains to 21–25 %. The NS head works well, so compare with `worktrees/2026-09-23-ns3d-shift-head`.

**Integrity rules (non-negotiable):**
- **Make a fresh sealed held-out cohort** with a new seed, never used by the first attempt. The first attempt's 32 held-out cases (seed 923401) have been opened and must not be used for any choice.
- **Pre-register the settings rule in DESIGN.md and commit it before timing.** Use the paper's rule: accurate = the most accurate tested setting; fast = the cheapest setting within a bar fixed in advance (e.g. 5 %). Choose settings on a validation cohort, then open the held-out cohort once.
- **Measure speedups in the same job.** Our model and the Newton–BiCGStab ladder run in one allocation with A–B–A timing and the drift and order-effect gates (≤ 1.10). The paper-rule speedup is against the fastest full-order setting at least as accurate as the accurate setting; also report the own-matched speedup.
- **Certify every quadrature or tensor rule** on held-out reached states (ρ ≤ 0.116, excluding the initial state).
- **Audit and review.** Run an independent NumPy audit of every error, and a `codex exec` design audit plus a results audit.
- **Record every change made after seeing results** as a labelled amendment, and report negative results plainly.

**Mechanics:**
- **Worktree:** work in a new worktree, `worktrees/2026-09-23-burgers3d-retry`, on branch `exp/2026-09-23-burgers3d-retry`, forked from `exp/2026-09-23-burgers3d-span` @ `2c9e7b39`. Create it as a narrow sparse checkout, for example:
  ```
  git worktree add --no-checkout -b exp/2026-09-23-burgers3d-retry worktrees/2026-09-23-burgers3d-retry exp/2026-09-23-burgers3d-span
  cd worktrees/2026-09-23-burgers3d-retry
  git sparse-checkout set --no-cone '/*' '!/experiments/mr-heat2d/runs/' '!/experiments/paper-b3d/runs/' '!/experiments/b-panel/' '!/experiments/cheap-corrections/' '!/experiments/q-ridge/' '!/experiments/head-ablation/' '!/experiments/b-ladder-top/'
  git checkout
  ```
  Read the trained bank and inputs from the first lane read-only (`worktrees/2026-09-23-burgers3d-span`), and copy only the small files you need, recording their sha256.
- **Cluster namespace:** `/cluster/tufts/paralab/tawal01/b3dretry_20260923/`, one job directory per job. Run `squeue` before and after every submit.
- **GPUs:** there is a 10-GPU per-user cap on the account, and H200s are often full. Use A100-80G where memory allows, and run independent pieces in parallel.
- **Disk:**
  - The local disk is about 99 % full (roughly 50 GB free), so keep pulls small.
  - The shared paralab disk overflowed on 2026-09-23. Never save full 3D fields for every case: subsample, keep at most a couple of full fields for the audit, and check free space before large writes.
- **Git:** commit locally, never `git push`.
- **Lab log:** append to `LAB-LOG.md` with `flock /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/.lablog.lock sh -c 'cat entry.md >> LAB-LOG.md'`.
- **Scope:** don't touch `paper_latex/` or main.

**Deliver:**
- **Per mesh:**
  - the accurate and fast settings, each with its setting, worst and median error, GPU ms, paper-rule speedup and own-matched speedup;
  - the full-order setting chosen, with its error and ms;
  - the cohort, job id, GPU, and whether every gate passed.
- **Records:** the generated report path (glossary at the end), the summary JSON with its sha256, the branch and final commit, and confirmation that the cluster namespace is emptied.
- **Everything that failed or was changed after seeing results.**
- **A one-line verdict:** is there a Burgers 3D row worth putting in Table 1, or not?

---

## Glossary

- **Span R′:** a solve in the first R′ columns of the importance-ordered bank. It is linear in the coefficients.
- **Accurate / fast:** the two settings reported per mesh, chosen by the pre-registered rule.
- **Paper-rule speedup:** the time of the fastest Newton–BiCGStab setting at least as accurate as the *accurate* setting, divided by each setting's time, in the same job.
- **Own-matched speedup:** the same, but against the fastest full-order setting at least as accurate as *that* setting.
- **Held-out / validation:** sealed cases opened once, versus cases used to choose settings.
- **ρ:** the quadrature error certificate on reached states. The bar is 0.116.
- **A–B–A:** a timing design in which our model runs, then the full-order solver, then our model again, in one job.
