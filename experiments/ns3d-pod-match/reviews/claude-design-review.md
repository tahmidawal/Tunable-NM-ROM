# Design/code review — independent fresh Claude subagent (read-only)

Reviewer: a fresh Claude general-purpose subagent with no context from the lane author, run 2026-11-03 on commit 2e8506d79, given only the design and the code (prompt: `design-review-prompt.txt`; it also ran CPU-only smoke and XLA memory-analysis checks in the scratchpad). Substitutes for the Codex audit (Codex quota-blocked until 2026-11-04 16:37) by coordinator instruction. **A Codex re-audit is owed when the quota returns.** Its final message, extracted verbatim from its transcript, follows.

---

The pre-registration is **not ready** yet. I found no mathematical or logic error. The objectives, POD targets, interpolant conventions, optimiser wiring, seed handling and gate formulas are all correct. What is missing is protection against losing a whole 6-hour bank job to a failure the code only notices at the end, plus two gaps in the DESIGN that leave decisions open.

## Independent review: ns3d-pod-match pre-registration (DESIGN.md + train_bank2.py + configs + cluster scripts)

Reviewed tree: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-03-ns3d-pod-match`, HEAD `2e8506d79`. Paths below are relative to `experiments/`.

**Checks I ran** (CPU only, `JAX_PLATFORMS=cpu`, no bytecode written to the repo, outputs only in the scratchpad):
- **Both smoke runs finish:** `train_bank2.py --smoke` completes with exit 0 for both arms (status `final`).
  - Arm (b): interpolant value check 1.7e-15, gradient check 1.9e-15, Nyquist energy about 1e-31, QR column change 1.8e-15.
  - L-BFGS ran and was recorded: 130 iterations, 132 line-search evaluations, stop reason `deadline`.
- **Compile-time memory estimate** at full scale (48³ grid, width 512, R=64, 8192 off-grid points), from XLA `memory_analysis` on CPU:

  | Arm | Adam step temp | L-BFGS call temp (25 iterations, zoom) |
  |---|---:|---:|
  | (a) | 27.5 GB | 27.6 GB |
  | (b) | 33.1 GB | 33.2 GB |

- **Spectral content of training-seed states:** I generated three training-seed trajectories at 64³. At t=0.2 the energy fraction above |k|∞=10 is 5e-7 to 7e-6, and at or above 16 it is 7e-10 to 6e-8.
- **Parent pilot timings:** 303 ms per Adam step for architecture A. Everything outside training took about 380 s, but the pilot measured floors at 32³ only.

---

### Findings

**1. Arm (a) objective. `train_bank2.py:118-127`, `coordnet.py:140-153, 230-241`. CORRECT.**
- The code computes 1 − ‖P_G Y‖²/‖Y‖² by Cholesky with a 1e-12 relative ridge. The ridge can only raise the loss.
- The band term is λ‖G − Π₁₀G‖²/‖G‖² with the mask max_i|k_i| ≤ 10 on the 48³ grid and λ=1 (`bank_a.json`). This matches DESIGN §3(a).
- The exact NumPy SVD value is reported at every stage (`train_bank2.py:330-331`).

**2. Arm (b) objective. `train_bank2.py:128-141`. CORRECT.**
- Grid term: Σ w_r‖G_r − U_r‖². G is sampled with the n^{-3/2} scaling and U has unit columns, so Σ w_r‖U_r‖² = 1 and the three terms start on the same O(1) scale.
- The off-grid value and gradient terms are normalised by nF and ndF (`:300-301`), as written in the DESIGN.
- The weights are `ev[:R]/ev[:R].sum()`, and `ev` holds the s² eigenvalues of the weighted Gram (`:275`). This matches w_r = s_r²/Σs².

**3. `trig_interpolant` conventions. `train_bank2.py:69-99`. CORRECT.**
- **Coefficients:** c = fftn(norm="forward")·nt^{3/2}, so f(x) = Σ c_k e^{2πik·x} = nt^{3/2}·(the interpolant of the sampled column). Because the bank is G = nt^{-3/2}·g(x_i), this f is in the network's units.
- **Axis ordering:** the einsum chain ("rcabd,pd" → "rcabp,pb" → "rcap,pa") pairs axes (a, b, d) with x[:,0], x[:,1], x[:,2]. This agrees with `grid_points` (meshgrid with ij indexing) and with the (3, n, n, n) layout produced by `make_sample_fn`.
- **Gradient factor:** 2πik_d on axis d, with x in [0,1). This is the same convention as `ns3d_fom.geometry` (k = 2π·modes) and as the JVP-based `CN.velocity_gradient`.
- **Output layout:** (3_e, P, 3, R), the same as `velocity_gradient`. The reshape of `spectral_derivative` output at `:286-287` ((3_e, R, 3, n³) → (3_e, 256, 3, R)) is right.
- **Nyquist:** because 48 is not one of the data meshes, `BI.resample` always drops the Nyquist planes, so the gate at `:295` is satisfied by construction (about 1e-31).

**4. Upper bound in DESIGN.md:90-93. CORRECT.**
- Write ‖(I−P_G)Y‖² = Σ_k s_k²‖(I−P_G)u_k‖².
- For k ≤ R, ‖(I−P_G)u_k‖ ≤ ‖u_k − G_k‖ because G_k lies in span(G). For k > R the factor is at most 1.
- Dividing by Σ_{k≤K} s_k² gives L_VP ≤ optimum + c·Σ w_r‖G_r − U_r‖², with c = Σ_{r≤64}s²/Σ_{r≤K}s² ≤ 1.
- This applies to the raw 48³ bank, which is what `vp_exact_raw` reports. It does not bound the floors on other meshes; the DESIGN does not claim that it does.
- Optional fix: assert `vp_exact_raw − optimum ≤ c·grid·(1+1e-8)` inside `certify` as a cheap bug trap.

**5. POD target construction. `train_bank2.py:233-263`. CORRECT.**
- Since Y = Xw^T V = U_K S_K, the line `Yn[:, :R] / column norms` gives exactly u_r.
- QR with diag(R) > 0 keeps every leading prefix's span and stays within the 1e-6 gate.
- The result is the weighted POD (rows scaled by √(1/ū₀²)) of the same pooled 48³ data. The optimum `ev[R:K].sum()/ev[:K].sum()` is the matching Eckart–Young value.
- Accuracy note: the Gram route squares the conditioning. In the pilot, ev₆₄/ev₁ ≈ 2.7e-8, so the expected eigenvector error is about 1e-8 to 1e-7. That passes the gate but is not far from it, and `orthonormality_before_qr` is recorded, so a failure would be visible.
- Wording, DESIGN.md:77: U_R is the optimum of the VP **part** of arm (a) and of the parent objective. It is not the optimum of 𝓛_a once the band penalty is added. Fix the sentence in §A1.

**6. Staged optimiser. `train_bank2.py:351-401` (Adam) and `:404-467` (L-BFGS). CORRECT, with two NEEDS-CHANGE items below.**
- **Adam:**
  - Calibration uses the same clipped Adam chain and two 500-step chunks; the 0.97 reserve is applied.
  - `warmup_cosine_decay_schedule(0, 1e-3, 1000, steps, 1e-6)`: here `decay_steps` includes the warm-up, so the schedule ends exactly at `steps`. Matches §3.1.
- **L-BFGS API:** `value_and_grad_from_state`, then `update(..., value=, grad=, value_fn=)`, then `tree_get(st, "info").num_linesearch_steps`. This is the documented optax 0.2.8 pattern, and the smoke run exercises it.
- **Revert logic:** on rejection the code breaks before reassigning `layers`/`lstate` (`:438-444`).
- **Stall rule:** matches §3.2.
- **Deadline:** checked between calls, so one call can overrun. The worst case is about 25 × 31 evaluations, or 6–8 min at arm (b)'s cost.
- **Compile-time constants:** every jit receives `data` as an argument (`:353`, `:411`, `:312-313`). Only the 48³ boolean `amask` (about 110 kB, `:120`) and `k` (`:78`) are captured. That is harmless.

**7. NEEDS-CHANGE: a revert discards the whole call. `train_bank2.py:441-443`.**
- If the last of 25 iterations raises the objective, the 24 good iterations before it are lost and the stage ends.
- Fix: carry `(best_q, best_v)` through the scan (one extra 1.26M-parameter copy) and revert to the best iterate inside the call. Record this in §A1 as "reverted to the best iterate of the call".

**8. NEEDS-CHANGE (low priority): arm (a)'s stall threshold is below what float64 can resolve.**
- `projection_loss` returns 1 − x. The rounding floor is about 1e-16 absolute, while the stall rule tests (prev − cur) ≤ 1e-10 × prev. Near the target of about 4e-7, that threshold is about 4e-17.
- As a result, a converged arm (a) will most likely stop with "objective increased … reverted" rather than "stalled". The stop label becomes noise-driven, though the bank of record is still well defined.
- Fix: either compute the fit as ‖Y − Q Qᵀ Y‖²/‖Y‖² from a Cholesky-based Q (the same objective, better relative precision), or state in §A1 that an increase of at most 1e-15 absolute counts as a stall.

**9. CORRECT: the Fourier frequencies cannot change.**
- B is a separate, non-differentiated argument everywhere (`:116-146`, `:353`, `:411`).
- `save_params` and `load_params` call `check_frequencies` (`bankio.py:298-316`), and the reloaded `bank_final.npz` is checked at `:472`.
- The check at `:465` compares `Bf` to itself and is therefore tautological, but harmless.

**10. NEEDS-CHANGE (important for the budget): the hard POD-64 reproduction gate is only evaluated after about 5 h. `train_bank2.py:528-546`.**
- The parent POD-64 floors do not depend on the bank. They are computed inside the floor loop, after all training. A reproduction miss, or a host out-of-memory at 96³, would therefore cost a whole 6h40 allocation with no verdict.
- The 96³ step holds 128 cases × 6 frames × 21 MB ≈ 16.3 GB in host RAM.
- This code path has only ever reproduced the 32³ reference (pilot gap 1.9e-10). 64³ and 96³ have never been run here.
- Fix: compute `parent_pod_R64` for all three meshes (and generate the development cases) **before** stage 3, and exit with code 3 if the gate fails. The values are deterministic, so reordering does not change any number.

**11. NEEDS-CHANGE: host memory, GPU type and wall limit are not fixed for submission. `submit.sh:5, 37`; DESIGN §7.**
- `--mem`, `--gres` and `--constraint=a100-80G` are free arguments, and the constraint only applies if `SBATCH_EXTRA` is set.
- Peak host memory is about 16–25 GB in the data stage (the states list plus `np.stack` plus `X*√w`, with X = 8.15 GB) and about 20 GB in the 96³ floors.
- Fix:
  - State the exact submit line in §A1, e.g. `--gres=gpu:1 --mem=128G --time=06:40:00 --constraint=a100-80G`.
  - Make the a100-80G constraint the default in `submit.sh`, with an explicit override for the H100 fallback.

**12. NEEDS-CHANGE: there is no recovery path if the job dies in the floor stage.**
- Partial checkpoints are written (`:392-394`, `:454-456`), and `bank_final.npz` and `bank_adam.npz` exist before the floors. But there is no mode that reruns only stages 6–8 from them.
- Fix:
  - Add `--floors-only <dir>`, which loads the two banks and recomputes data X (needed by `order_bank`), the ordering and the floors. That turns a late failure into a roughly 1 h job instead of a 6h40 rerun.
  - Move the Adam-stage diagnostic floors (`:497-507`) after all of the final bank's gate inputs, so a timeout cuts the diagnostic, not the verdict.

**13. NEEDS-CHANGE: `all_passed` ignores hard failures. `train_bank2.py:585-589`.**
- The DESIGN says "a job failing a hard gate yields no verdict", but `gates.all_passed` is computed regardless.
- Fix: set `gates["verdict"] = None if hard else gates["all_passed"]`, and have the report generator read `verdict`.
- Also: if a mesh is missing from `parent_pod_floor_reference`, the reproduction gate is skipped silently (`:540-541`). Make that a hard failure.

**14. CORRECT: the gates are computed as the DESIGN states. `train_bank2.py:574-586`.**
- G1: ratio to the in-job `parent_pod_R64` at every mesh, ≤ 1.5.
- G2: max/min of the coordnet R=64 floor, ≤ 1.25.
- G3: `orthonormality_before_lowdin` is the max-abs of BᵀB − I with B = P_n G_raw T (`bankio.py:339-341`), gated at 64³ and 96³ only, with 32³ reported.
- The no-centring control is hard at 5% (`target_relative=0.05`, `:519-520`).
- The POD-target checks are hard (`:292-296`).

**15. Flag (no threshold change proposed): G3 may be in tension with arm (b).**
- T comes from the P₄₈-projected bank (2/3 mask, |k| < 16). On finer meshes, B = P_n G_n T also picks up the bank's energy in 16 ≤ |k| < 21.3 at 64³, and up to 32 at 96³. So the deviation at 64³/96³ is roughly the per-column energy fraction outside the P₄₈ space.
- Arm (b) regresses onto POD modes that keep the data's content up to |k| = 23, with no band penalty. Whole snapshots carry only about 1e-8 of their energy at |k| ≥ 16, but low-energy POD modes (r near 64) could carry a much larger fraction.
- A perfect arm-(b) fit could therefore fail G3 for reasons unrelated to bank quality.
- Fix (reporting only):
  - Add a cut at 15 (that is, ≥ 16) to `spectral_content` (`:294`).
  - Compute G3's statistic on the same-data POD itself (resampled U, P_n, and T from its own P₄₈ ordering) as a control, so a G3 failure can be interpreted.

**16. NEEDS-CHANGE: provenance gaps. `submit.sh:30`; `train_bank2.py:180-184`.**
- `COMMIT.txt` is HEAD, but the working tree is not checked for cleanliness.
- Only 4 files are hashed into `summary.json`. Not covered are `diag_floor.py`, `ns3d_fom.py`, `shift_rom.py`'s imports, and the config.
- Fix:
  - Refuse to submit if `git status --porcelain -- experiments/` is non-empty.
  - Add `ns3d-grok/diag_floor.py`, `ns3d/ns3d_fom.py` and the config file to `report["files"]`.

**17. CORRECT, with one NEEDS-CHANGE: cluster scripts.**
- **Correct:**
  - The GPU preflight prints `jax_backend=…` and exits 42 on anything other than gpu (`job.sbatch:18`), and the driver also refuses a non-GPU backend (`:172`).
  - Each job gets its own directory, and `submit.sh:9` refuses to reuse one.
  - All `.py`/`.json`/`.sbatch`/`.sh` files of the imported packages are staged. The smoke run confirmed the import closure: `shift_rom` imports `ns3d_rom`, `ns2d_rom` and `ns3d_model`, all inside staged directories.
  - `OUTPUTS.sha256` is written after the run, including on failure.
  - `pull.sh` verifies the manifest, requires `jax_backend=gpu`, `RUN_EXIT=0` and `status=='final'` before deleting the remote directory, and keeps it otherwise.
- **NEEDS-CHANGE: a Slurm time-limit kill skips the manifest.** `pull.sh:13` then aborts. That is safe, since the remote is kept, but the job is unaudited. Fix: `#SBATCH --signal=B:TERM@300` plus a `trap` that writes `OUTPUTS.sha256`.
- Minor: bank mode prints `VERIFY_EXIT=0` unconditionally (`job.sbatch:44`), which is misleading. Print `VERIFY_EXIT=NA`.

**18. CORRECT: seeds.**
- The driver reads only `train_seed` (202609201) and `dev_seed` (202609202). It refuses either if it appears in `closed_seeds` [202609203, 202609211, 202609221] (`:167-169`).
- The other seeds (init 20261001, off-grid points 20261103, internal RNGs 3 and 5) are not cohort seeds.
- `F.parameters` keeps prefixes, so the 128-case parent POD is the parent recipe.
- Optional: also assert `dev_seed == 202609202`.

**19. NEEDS-CHANGE: stop rule and budget wording. DESIGN §5 and §7.**
- **Hard-gate failures:** §5 does not say what happens to an arm whose job fails a hard gate (no verdict). State that it counts as "not passing" and is not rerun, except for a preflight or infrastructure failure that happens before training starts.
- **Running jobs in the budget rule:** §7's "running total plus the next job's wall limit" does not say whether a still-running job counts at its wall limit or at its elapsed time. Two concurrent banks at their limits are 13.3 h, which leaves 1.7 h. That is less than the 2 h of ROM wall limits (0.5 + 0.5 + 1). State that running jobs count at their wall limit until `sacct` is final.
- **Concurrency:** the "at most 2 concurrent allocations" rule must also apply to the three ROM jobs. Nothing enforces it in code.

**20. CORRECT, with a moderate margin: GPU memory and wall time.**
- **GPU memory:** about 28–33 GB of XLA temp per step for both stages and both arms, plus about 1.5 GB of arguments. This is well under the 60 GB default preallocation on an A100-80GB.
  - The off-grid gradient term at 8192 points costs about 5.6 GB over arm (a).
  - The `trig_interpolant` peak is a few GB, with about 6 TFLOP of complex einsum (minutes).
- **Wall time** for arm (b), estimated: data about 5 min, targets about 5–10 min, calibration plus compiles about 10–15 min, Adam 10,800 s, L-BFGS 6,000 s plus at most 8 min, floors at three meshes about 20–40 min. That is about 5.4–5.9 h against a 6h40 limit.
  - This is consistent with the DESIGN's "≈ 6 h", but only 45–75 min of margin is left. Findings 10 and 12 are what protect against an overrun.

**21. Minor wording mismatches with the DESIGN. NEEDS-CHANGE (text only).**
- §3 says the POD targets' spectral content is recorded "at every mesh". It is recorded once, on 48³.
- The VP ratio to the optimum is recorded for the raw bank only, not for the P₄₈-projected one (`:335`).
- Either align the text in §A1 or add the projected ratio.

---

### Overall verdict

**Not ready.** Nothing in the experiment's logic needs to be redone. Every must-fix item protects the budget or settles a decision that would otherwise be made after seeing results, and none changes a pre-registered threshold.

Must-fix before submission, with the changes recorded as §A1:
1. **#10:** compute the parent POD-64 floors, and their hard reproduction gate, before training, so a failure costs minutes rather than about 6 h.
2. **#11:** fix and record the exact submit line (`--mem` ≥ 96–128G, `--time=06:40:00`, `--constraint=a100-80G` as the default).
3. **#12:** add a floors-only recovery mode, and move the Adam-stage diagnostic floors after the final bank's gate inputs.
4. **#13:** void the gates when a hard failure occurs; make a missing reproduction reference a hard failure.
5. **#19:** say how a hard-gate failure counts in the stop rule, and how running jobs count in the 15 GPU-hour rule.
6. **#16:** add a clean-tree check at submission and hash the remaining imported files and the config.

Recommended but not blocking: #7 (revert to the best iterate within a call), #8 (stall precision for arm a), #15 (spectral cut at 15 and a POD G3 control, so a G3 outcome can be interpreted), #17 (manifest on timeout), #21 (text).

The smoke outputs and the two check scripts I used are in `/tmp/claude-1002/-home-tahmid-Dev-Tunable-NM-ROM/5c91747e-5a65-57ce-b940-e6e2e90f5699/scratchpad` (`smoke_a/`, `smoke_b/`, `mem.py`, `spec.py`). No repository file was modified.