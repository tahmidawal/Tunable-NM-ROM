# Design/code review — Grok (grok-4.7-xhigh via cursor-agent, --mode ask, read-only)

Reviewer: xAI Grok 4.7 xhigh through the Cursor Agent CLI, run 2026-11-03 on commit 2e8506d79, given only the design and the code (prompt: `design-review-prompt.txt`). Substitutes for the Codex audit (Codex quota-blocked until 2026-11-04 16:37). **A Codex re-audit is owed when the quota returns.** Full output verbatim below; raw JSON kept as `grok-design-review.raw.json`.

---

I'll review the pre-registration and the driver against the imported dependencies, read-only, and cite file and line for each finding.The design and driver are in. Next I’ll check the imported math (sampling, POD, spectral derivatives, L-BFGS) against what the driver actually calls.The objectives and the L-BFGS call are the parts most likely to be wrong. I’ll check the trigonometric interpolant and the Optax L-BFGS pattern against the installed library.The mathematics of both objectives, the POD targets, and the gates match the pre-registration. The run is not ready to launch: the L-BFGS line search is not the one `optax.lbfgs` uses, and the job can be killed after training by an undersized allocation or by arm (b)’s off-grid gradient.

## Findings

**1. Arm (a) objective — CORRECT.** Severity: none.
`L_a` is the variable-projection loss plus `λ=1` times the fraction of energy with `max_i|k_i|>10`. Training calls `projection_loss` (`experiments/ns3d-coordnet/coordnet.py:140-153`) and `lowpass` with `band_mask` (`coordnet.py:230-241`), wired at `experiments/ns3d-pod-match/train_bank2.py:118-127`. `band_kcut=10` and `band_weight=1` are in `configs/bank_a.json:63-64`. The mask `max_i|k_i|≤10` is the integer 2/3-rule space of `32^3` (`|k|<32/3`). The training loss adds a relative ridge of `1e-12` inside the Gram matrix; the certificate written beside it is the unridged SVD loss (`coordnet.py:129-137`, `train_bank2.py:330-337`). At float64 that ridge sits far below the `1e-10` stall test.

**2. Arm (b) objective, scaling, and the VP bound — CORRECT.** Severity: none.
The grid term is `∑_r w_r‖G_r−U_r‖²` with `w_r=s_r²/∑_{j≤64}s_j²` (`train_bank2.py:132-141`, weights at `275`). `G` is `n^{-3/2}g(x_i)` (`coordnet.py:107`). The POD mode `φ` is Euclidean-unit in that same raveled layout, so `φ(x_i)` is `O(n^{-3/2})` and the matching network field is `g=n^{3/2}φ`. The off-grid targets are that field: `trig_interpolant` multiplies forward-normalized FFT coefficients by `nt**1.5` (`train_bank2.py:77`) and rebuilds with `exp(+2πi k·x)` on axes `(x,y,z)` in `ij` order (`85-96`). The gradient multiplies by `2πi k` on the matching axis. `spectral_derivative` uses the same factor, `1j·(2π k)` (`experiments/ns3d-shift/shift_rom.py:30-34`, `experiments/ns3d/ns3d_fom.py:19-25`). The node check compares both value and gradient after the same `nt**1.5` scaling (`train_bank2.py:284-292`). Value and gradient shapes match `CN.velocity` `(P,3,R)` and `CN.velocity_gradient` `(3,P,3,R)`.

The stated bound holds. With `Y=U_K S_K`, `‖φ_j‖=1`, and `G_j` in the column span,

`L_VP − optimum ≤ (∑_{j≤64} s_j² / ∑_{j≤K} s_j²) · ∑_r w_r‖G_r−φ_r‖²`.

The coefficient is at most 1, so the regression term alone upper-bounds the VP excess. It bounds the raw sampled bank, which is the VP loss being trained. The off-grid terms are extra and are outside that inequality, as `DESIGN.md:90-93` says.

**3. POD targets — CORRECT.** Severity: none.
`Y = Xwᵀ V_{:K}` has orthogonal columns of norm `s_r=√ev_r` (`train_bank2.py:234-240`). Column-normalizing `Yn[:,:R]` recovers those singular vectors. QR with `sign(diag(R))` (`255-259`) preserves every leading prefix span. Individual columns move only by the non-orthogonality, and a column change above `1e-6` is a hard failure before training (`262-263`, `305-310`). Snapshot weights are `1/mean(u₀²)` on the native mesh, then one global rescaling (`211-224`); that rescaling cancels in `w_r` and in the VP loss. This is the weighted POD of the pooled training data, which is the Eckart–Young optimum `ev[R:K]/ev[:K]` (`250`).

**4. Adam schedule — CORRECT.** Severity: none.
Global-norm clip 1, then Adam. `warmup_cosine_decay_schedule(0, 1e-3, 1000, steps, 1e-6)` (`train_bank2.py:376-378`) matches Optax 0.2.8, where `decay_steps` is the full length including warmup. The step count is `0.97 × 10800 / measured_step`, rounded down to a multiple of 500 (`371`), with a hard abort if that is under `4×1000` (`372-373`). Calibration times a steady chunk, not the compile (`363-370`).

**5. L-BFGS line search starts from the previous step size — NEEDS-CHANGE.** Severity: blocker.
`DESIGN.md:62-63` specifies `optax.lbfgs` with a zoom line search of at most 30 steps. In Optax 0.2.8, `optax.lbfgs`’s own default line search sets `initial_guess_strategy='one'` (`optax/_src/alias.py:2634-2637`). The driver replaces that line search and leaves the function default `"keep"` (`optax/_src/linesearch.py:1306`, `train_bank2.py:405-407`). `"keep"` is the SGD setting; the same docstring says quasi-Newton searches should start at step size 1. A search that exhausts 30 steps can still take a trial point that increases the objective. The driver then reverts and ends the whole L-BFGS stage (`train_bank2.py:441-443`). The bank of record would be whatever iterate was last accepted, possibly the Adam weights, for a reason the pre-registration does not describe.

The rest of the L-BFGS call matches the Optax 0.2.8 contract: `value_and_grad_from_state`, `update(..., value=, grad=, value_fn=)`, `tree_get(state, "info").num_linesearch_steps`, and a scan of 25 iterations (`410-423`). A nonfinite or increasing call is not committed (`438-443`). The stall rule is eight calls in a row whose decrease is at most `1e-10` relative (`449-452`). The deadline is checked before each call, so the last call can run past 6000 s (`435`).

**6. Large arrays are jit arguments — CORRECT.** Severity: none.
`Y` `(331776×512, ~1.4 GB)`, `U`, `Q`, `Fq`, and `dFq` live in the `data` dict and are arguments of both jitted steps (`train_bank2.py:267-273`, `298-301`, `353-360`, `410-412`). `value_and_grad` differentiates only `layers`. The line-search closure `lambda q: loss_fn(q, B, data)` closes over those traced arguments, not over a host array captured from outside the jit. The band mask closed over by arm (a) is `48³` booleans. The evaluation helper `make_sampler` closes over the `48³` grid coordinates (`coordnet.py:111-115`); that array is about 2.5 MB and is not the training loss.

**7. Frequencies, checkpoints, and exits before floors — frequencies CORRECT; floor gap NEEDS-CHANGE.** Severity: major for the floor gap.
`B` is not in the Adam or L-BFGS state. Gradients are taken only with respect to `layers` (`train_bank2.py:356`, `417`). `save_params` calls `check_frequencies` (`bankio.py:38-39`). The final check raises if `B` differs from `frequency_set` (`train_bank2.py:465-466`); `not np.max(...) == 0.0` is `not (max == 0)`.

Weights are checkpointed every 900 s (`392-394`, `454-456`). Floors are written only after both stages, one mesh at a time (`565-566`). These paths reach `summary.json` with no floors and no gates: Adam nonfinite (`390-391`), the step-time abort (`372-373`), a rank-deficient SVD inside `certify` (`coordnet.py:133-134`), a singular Gram inside Löwdin (`bankio.py:64-65`), and a Slurm kill. There is no `SIGTERM` handler. A hard-gate failure before training is different: it writes `status=failed-gates` and exits 3 on purpose (`305-310`).

**8. Gates — CORRECT.** Severity: none.
G1 is `evolved_worst` of the final `R=64` bank over the in-job parent POD-64, required `≤1.5` at every mesh (`train_bank2.py:575-579`). `evolved_worst` ignores `t=0` (`diag_floor.py:72-83`). G2 is `max/min` of those three floors, `≤1.25` (`580-581`). G3 is `‖BᵀB−I‖_max` before Löwdin (`bankio.py:80-84`), gated only for meshes `≥64`, with `32` stored and not gated (`582-584`). The reference values in `configs/bank_a.json:48-52` are the fractions corresponding to the percentages in `DESIGN.md:47`. The reproduction test is `abs(got−want)/want ≤ 1e-6` (`train_bank2.py:540-546`). No-centring must have `evolved_worst > 0.05` (`517-520`). Arm (b) node checks are `≤1e-10` and `≤1e-9` (`292-293`), and any Nyquist energy above `1e-20` aborts (`295-296`). Prefixes `{8,16,32,48}` plus `R=64`, derivative checks at `32` and `64`, cross-mesh consistency, and the same-data POD floor are all recorded. `gates.all_passed` is only G1–G3 (`585`). A hard failure still sets `status=failed-gates` and exit 3 (`588-594`), which is the pre-registered “no verdict.” The future report script has to read `status` and `failures`, not `all_passed` alone. The thresholds themselves are well-posed; the Nyquist cut `1e-20` is above float64 roundoff for a linear combination of snapshots whose Nyquist bins were written as exact zeros.

**9. Seeds, stop rule, budget — seeds and stop rule CORRECT; budget contingency NEEDS-CHANGE.** Severity: minor.
Train seed `202609201` and development seed `202609202` are the only cohort seeds read (`train_bank2.py:199`, `486`, `528`). Closed seeds `202609203`, `202609211`, `202609221` are refused if they are used as either of those (`167-169`). The off-grid seed `20261103` is not one of them. Gate numbers are fixed in `DESIGN.md:111-148` before any result. The bank stop rule is unambiguous: neither passes → stop; one passes → that bank; both pass → smaller `max_n(floor/POD-64)`, ties to arm (b) (`DESIGN.md:123-129`). Nothing in the driver picks the better of Adam and L-BFGS.

Two full wall limits are `6 h 40 min × 2 = 13 h 20 min`, under the 15 h cap, so the concurrent pair is allowed. The three ROM wall limits sum to 2 h. If both bank jobs are charged their full limits, the remaining `1 h 40 min` cannot hold all three ROM limits. `DESIGN.md:165-166` then drops whichever job is submitted next. That order is not pre-registered, so which mesh is skipped would be chosen after seeing the clock. At the table’s expected `≈6 h` each, `12 + 2` still fits.

**10. Cluster scripts — machinery CORRECT; resources not pinned — NEEDS-CHANGE.** Severity: blocker for the resource pin.
`job.sbatch:10` sets float64 and `JAX_DEFAULT_MATMUL_PRECISION=highest`. `job.sbatch:18` prints `jax_backend=gpu` and exits 42 otherwise, before training. `submit.sh:9` refuses an existing job directory. Staging rsyncs the Python of every imported package onto the paralab path (`submit.sh:11-18`), and `squeue` is printed before and after (`31-43`). `pull.sh:15-28` checks the manifest, refuses missing non-field artifacts, and requires `jax_backend=gpu`. A bank job is deleted only when `RUN_EXIT=0` and `summary.status=="final"` (`36-37`, `48-52`); otherwise the remote directory stays.

The sbatch file does not set `--constraint=a100-80G` or `--mem`. Both arrive only if the caller exports `SBATCH_EXTRA` and passes memory. The `96³` parent-POD array is `128×6×3×96³×8 ≈ 15.2 GiB` (`diag_floor.py:166`, `train_bank2.py:528-534`), and the development trajectories add about `1.9 GiB` at the same time. A 32 GB allocation dies in that floor, after Adam and L-BFGS. 64 GB covers the array; 128 GB covers it with the XLA compile of the line-search scan.

**11. Step memory and the 6 h 40 min wall — arm (a) fits; arm (b) is unmeasured — NEEDS-CHANGE.** Severity: major.
Arm (a) is the pilot architecture on the full `48³` batch. That pilot ran on an A100 80 GB at `0.303 s/step` (`experiments/ns3d-coordnet/runs/pilot/logs/4708239.out:15`, summary `seconds_per_step` `0.303`). The added band FFT is one transform of 64 fields of size `48³`. At that step time, Adam’s scheduled steps are about 3 h, L-BFGS is capped at 6000 s plus one call (about 5 minutes at this step time), and data plus three meshes of floors are a small fraction of an hour. The nominal arm (a) job fits in 6 h 40 min with on the order of an hour to spare.

Arm (b) puts `CN.velocity_gradient` on all 8192 points inside `value_and_grad` (`train_bank2.py:138-140`), in the same step as the full `48³` backward. The only existing caller chunks that gradient, and only in forward mode, at 4096 points (`bankio.py:97-108`). The nested reverse-over-forward is not chunked. If the step exceeds about `2.6 s` (`0.97×10800/4000`), calibration aborts at `train_bank2.py:372-373` and writes no floors, after data generation and two chunks of 500 steps. A GPU OOM is a process kill and also writes no floors.

## Verdict

**Not ready.** No mathematical error turned up in either objective, the `n^{3/2}` trigonometric scaling, the VP-excess bound, the weighted POD targets, or G1/G2/G3.

Must-fix before any GPU allocation, and record the code change as an amendment rather than editing the frozen design text:

1. Pass `initial_guess_strategy="one"` into the zoom line search so the stage is `optax.lbfgs` as pre-registered.
2. Pin `--constraint=a100-80G` and a host `--mem` of at least 64 GB (128 GB is the comfortable figure) in the launch. A 32 GB host allocation is killed while building the `96³` POD floor, after training.
3. Run a short GPU smoke of one arm (b) step, full `48³` batch plus the 8192-point gradient, before the 6 h 40 min allocation. Launch arm (b) only if that step fits on 80 GB and is comfortably under `2.6 s`.
