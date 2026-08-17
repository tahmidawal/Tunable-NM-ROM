# Adversarial review of the follow-up harness (Codex, 2026-08-17)

`codex exec` (OpenAI Codex CLI, GPT-class model, independent of the agent that wrote the
code) was run **read-only** over both follow-up worktrees before the cluster fan-out, briefed
on: timing-protocol validity (warm-up, device sync, identically compiled FOM, one-GPU
sequencing), seed handling (training vs data randomness), POD comparability at each k, EQ
refit correctness when m or M changes, and "anything else that would make a number wrong".
The full verbatim report is reproduced below the disposition table.  Every MUST item was
either applied or answered in writing; nothing was silently dropped.

Scope note: the review covers BOTH worktrees, so this file is archived identically in
`experiments/poisson2d-rom-objective/` and `experiments/burgers2d-rom-latent-stepping/`.

## Disposition — MUST FIX

| # | finding | disposition |
|---|---|---|
| 1 | `fu_common.lm_jit_solver` copied the *rollout* LM (`blat_common._finish_ops.lm_step_jit`), not the *IC-fit* LM (`ms_autodecoder.lm_solve`): a 1e-12 vs 1e-13 step threshold, and a stall test that could fire on a REJECTED step. The "jitted IC fit" was therefore not the same algorithm as the Python one it is benchmarked against. | **APPLIED.** `followup/fu_common.py` rewritten as an exact `lax.while_loop` port of `lm_solve`: same initial Jacobian, same damping schedule, same acceptance test, both stopping tests applied **only after an accepted step** (`rel_dec < 1e-12` or `‖dz‖/(1+‖z_old‖) < 1e-13`), same lambda-saturation aborts, and the same attempt / accepted / rejected / residual-eval / Jacobian-eval accounting. `fu_timing.py` records `z_rel_diff` between the two solvers' latents in every row. |
| 2 | The N-ladder used an **unnormalised parameter distance** (and omitted viscosity) to pick the nearest-IC cold start, while `blat_rom.py` uses the nearest **initial field** in L2. Different basin ⇒ different accuracy and timing. | **APPLIED.** `fu_common.nearest_train_ic(n, u0)` regenerates the training ICs at each N and picks the field-L2 nearest, exactly as `blat_rom.py` does. The chosen index and distance are recorded per row. |
| 3 | `speedup_end_to_end_python_ic` added the Python IC time to a rollout that was started from the **jitted** IC latent. | **APPLIED.** Renamed `speedup_end_to_end_python_ic_composed` and emitted as `null` unless the two solvers agree (`z_rel_diff <= IC_Z_TOL = 1e-6`); the agreement flag is in the JSON. |
| 4 | The median-of-7 protocol was not applied uniformly (Python IC 3/1, Poisson preprocessing 3/1) yet those numbers entered composed speedups. | **APPLIED.** Every timed component in both harnesses now uses `TIME_REPS` (7) repetitions after `TIME_WARM` (2) warm-ups. |
| 5 | Poisson had **no** per-iteration timing versus k and no per-solve timing across the m/M ladders; `pro_colloc.py`'s `secs` field includes lazy EQ fitting, first-call compilation, 16 solves and error evaluation and must not be used as a cost. | **APPLIED.** `poisson .../followup/fu_timing.py` gained `MODE=k` and `MODE=m` (new cells `pt_k`, `pt_m`), both jitted, warm-up + sync + median of 7, reporting iterations, attempts, termination reasons and s/iteration. Burgers gained the equivalent cost cell `bt_m`. `secs` is used nowhere. |
| 6 | "Iterations-to-tolerance" is not measurable: the reference LM has no absolute residual tolerance. | **APPLIED (option B, as offered).** Reported as **iterations to termination** with the termination-reason histogram; an optional invariant stop `‖r‖ <= REL_TOL·‖f_m‖` exists (`REL_TOL`, default 0 = the reference rule) and the value used is recorded in the manifest. |
| 7 | Poisson N ladder fitted the EQ weights once on the N=64 `dx²` rule but recomputed the source projection with each N's grid rule — two different discretisations of the same continuum integral on the two sides of the residual. | **APPLIED.** The EQ weights are now **refit on each N's grid** inside the N loop, so both sides use the same grid rule at every N. (This also matches what the Burgers ladder already did.) |
| 8 | `pro_colloc.eq_weights` drew fresh snapshot indices and latent perturbations on every `(M, m, pool)` cache miss, so an m or M ladder changed the EQ training set together with m or M, and grid vs meshfree were fitted on different draws. | **APPLIED, opt-in.** New `EQ_FIXED_SNAPS` (default **0** = the frozen round-1/round-2 behaviour, so the archived results stay reproducible); **all follow-up cells set it to 1**, which draws indices, perturbations and the row subset from a fixed stream so every ladder point and both pools see the same snapshots. The flag is recorded in each JSON manifest. |
| 9 | `speedup_solve_only` was labelled as ROM-vs-FOM wall clock although input preprocessing (O(n)) and full-field reconstruction were not included; the Burgers `MODE=k` end-to-end omitted the 51-slice decode. | **APPLIED.** Poisson reports `speedup_solve_only`, `speedup_with_preprocess` and `speedup_end_to_end` (solve + preprocessing + full-field decode), each timed under the same protocol; Burgers `MODE=k` end-to-end now includes the decode. The tables state which is which. |
| 10 | The Burgers N ladder logged the FOM Newton residual but never applied `build_data`'s 1e-8 abort — a risk at N=256 with the fixed-8-iteration Newton baseline. | **APPLIED.** `test_traj` raises if the residual is non-finite or `> FOM_RES_TOL` (1e-8). The JSON is saved after every N, so an abort at a fine mesh keeps the coarser rows. Poisson got the analogous CG guard (`FOM_RES_TOL = 1e-10`). |
| 11 | The timing tool loaded checkpoints without `blat_rom.py`'s config / data-fingerprint guards. | **APPLIED.** `load_ck` applies the same six config assertions; `MODE=k` additionally asserts every checkpoint's stored data fingerprint against the regenerated data and warns if a checkpoint's POD basis differs from the first one's. |
| 12 | The multi-seed comment claimed `TRAIN_SEED` varies the latent initialisation; in the Burgers trainer the latents are initialised **deterministically** from the top-K POD coefficients. | **APPLIED (claim corrected, not the code).** Burgers: the seed varies the FiLM network initialisation and the minibatch / collocation-point order only — stated in `blat_train_ad.py`, `fu_cells.sh` and the README. Poisson's `train_autodecoder_stage0` *does* draw the latents from the seeded key, so there the claim stands and is stated separately. |
| 13 | `fu_pod.py` implements only full-grid objectives while the coordinate k cells also emit EQ rows; comparing them would violate "same quadrature". | **APPLIED (labelling + enforcement in the summary).** The k-ladder figure's primary comparison is coordinate **full grid** vs POD **full grid**; the coordinate EQ row is plotted dashed and named as hyper-reduced. `fu_pod.py`'s docstring states the restriction, and it now records `rank`, `cond` and `square_system` per k so a square/rank-deficient Petrov-Galerkin point is labelled rather than plotted as a fair comparison. |

## Disposition — SHOULD FIX (applied)

- **Per-row EQ diagnostics.** Both NNLS fitters now report `row_rel_median`, `row_rel_p95`
  and `row_rel_max` alongside the global `rel_fit`, in the JSON and in the log line, so a
  single badly-fitted mode cannot hide behind a small norm.
- **s/Jacobian vs s/attempt.** The Burgers timer runs one extra **untimed** Python-loop
  rollout with the identical step kernel to recover LM attempt counts, and reports
  `s_per_attempt` next to `s_per_jacobian_eval` (both amortised over the rollout, labelled).
- **`fu_pod` conditioning.** `cond`, `rank` and a square/underdetermined flag per k and M.
- **Poisson `INIT`.** The manifest option is now honoured (`mean` or `nearest`).
- **Launchers.** `launch.sh` validates the cell name and recreates the remote cell directory
  from scratch, so leftovers from an earlier run can neither break the post-`scp` checksum
  comparison nor be mistaken for this run's output.
- **Staging provenance.** The batch script now echoes a `git_dirty` hash of the experiment
  directory next to `git_commit`.
- **`fu_cells.sh` wave dispatch** is an explicit `wave1` / `wave2` with an error on anything
  else (a typo previously selected wave 2).
- **Seed aggregation.** `fu_summarize.py` (both PDEs) aggregates exactly seeds 0/1/2 with
  `ddof=1`, prints the per-seed values next to the mean ± std, and states that the POD
  control is a deterministic function of the training snapshots (std 0 by construction).
- **Blow-ups.** The Burgers tables carry the blow-up count per cell; accuracy statistics are
  over completed rollouts only (as in the frozen round) and the count is printed beside them.
- **Reproducibility wording.** The `TRAIN_SEED` default is described as reproducing the frozen
  runs' **weights and latents**; the pickle itself differs because the config dict now carries
  `train_seed`.

## Disposition — NOT applied, with reasons

- **Degenerate eigenshells in `blat_common.test_modes`.** Codex is right that taking exactly
  M modes from a stable sort splits a degenerate `(kx,ky)/(ky,kx)` pair at M = 16, 128, 256
  (Poisson's `mode_mask` keeps complete shells; Burgers' does not). Changing it would change
  every frozen Burgers number and break comparability with the archived tables, for an effect
  that is a fraction of one test mode out of M. **Kept as-is and recorded as a caveat**; the
  retained mode count is reported in every row.
- **Pinning one GPU model for the multi-seed cells.** The multi-seed arm compares **errors**,
  which are hardware-independent at f64; no timing is taken from those cells. Pinning would
  cost queue time for no claim. **Documented instead.**

## Verbatim Codex report
## MUST FIX

- `B/followup/fu_common.py:48,60-62` does not reproduce the LM used by `blat_common.fit_ic`. The reference `ms_autodecoder.lm_solve` checks a normalized step threshold of `1e-13` only after an accepted step (`2026-08-14-multistage-precision/.../ms_autodecoder.py:106-115`); the new solver uses `1e-12` and can stop on a rejected tiny step. It copies the rollout LM, not the IC-fit LM, so the claimed like-for-like speedup can change the result and iteration count. Port `lm_solve` exactly into `lax.while_loop`, including its acceptance, damping, stopping, reason, residual-evaluation, Jacobian-evaluation, and attempt accounting.

- `B/followup/fu_timing.py:145-150,214-218` does not use the same nearest-IC initialization as the reference. The reference chooses the nearest training initial field by L2 distance (`blat_rom.py:185-191`); the follow-up chooses unnormalised parameter distance and omits viscosity. This can select a different basin and change both accuracy and timing. Generate the training ICs at each tested `n` and use the exact field-distance rule.

- `B/followup/fu_timing.py:177-180` reports a false Python-IC end-to-end speedup. `time_rom` is run only from `z_j`, but `speedup_end_to_end_python_ic` adds the Python fit time to that jitted-IC rollout. Time a separate rollout from `z_py`, or suppress the Python end-to-end number unless per-start results and final latents agree within a declared tolerance.

- The mandated median-of-7 protocol is not applied consistently. `B/followup/fu_timing.py:153` times the Python IC fit with 3 repetitions and 1 warm-up, while the jitted fit uses 7/2. `A/followup/fu_timing.py:183-186` similarly uses 3/1 for preprocessing, yet includes that number in `speedup_with_preprocess` at `:209`. Use the same two warm-ups and seven synchronized measurements for every component entering a speedup.

- Poisson has no valid k-, m-, or M-ladder timing. `A/followup/cluster/fu_cells.sh:62-78` runs `pro_colloc.py` and `fu_pod.py`; `pro_colloc.py:206-266` records a coarse total that includes lazy EQ fitting (`:216,227`), first-call compilation, 16 solves, and error evaluation. `A/followup/fu_timing.py` only handles fixed K. Thus “per-iteration cost versus k” and “per-step time versus m/M” are not measured. Add dedicated jitted timing arms with matching-shape warm-ups, synchronization, median 7, and both attempts and Jacobian counts. Do not use `row["secs"]`.

- The Poisson harness cannot currently support “iterations-to-tolerance versus k.” `A/followup/fu_timing.py:118-149` has only relative-decrease/step stopping and no absolute residual tolerance, while no k timing exists. Either introduce and report an invariant tolerance or rename the result to iterations-to-stall/termination and record termination reasons.

- `A/followup/fu_timing.py:82,157-160,171-187` mixes quadrature discretizations across N. Decoder-side EQ targets are fitted once using the N=64 `dx²` rule, while the source term is recomputed with each N’s grid rule. At N≠64 the residual is therefore “N=64 decoder projection minus N-grid source projection.” Reusing a meshfree rule is legitimate only if both sides target one fixed continuum integral. Fit against a fixed, sufficiently accurate continuum reference and compute source projections by that same convention; alternatively refit both sides per N if the target is each discrete FOM.

- `A/pro_colloc.py:114-150` redraws the EQ snapshot indices and latent perturbations for every `(M,m,pool)` cache miss. Consequently the m and M ladders change the EQ training set along with m or M, and grid-versus-meshfree comparisons use different snapshot draws. Precompute one indexed/perturbed latent set outside `eq_weights` and reuse it for every ladder point and both pools.

- `A/followup/fu_timing.py:191-214` measures only the latent solve as the flat ROM line. Required source preprocessing is O(n), and full-field reconstruction is not timed at all. `speedup_solve_only` therefore cannot be described as ROM-vs-FOM end-to-end wall-clock. Either explicitly limit the claim to the reduced nonlinear solve or time median-7 preprocessing and matched-grid decoding and include both in the reported wall-clock. `B/followup/fu_timing.py:224-232` has a similar mislabel in MODE=k: `speedup_end_to_end_jit_ic` omits the 51-slice decode.

- `B/followup/fu_timing.py:55-63,128-135` accepts an unconverged FOM trajectory in MODE=n. It logs the residual but never applies `build_data`’s `1e-8` abort. This is especially risky at N=256 with the fixed-eight-Newton baseline. Abort on non-finite or excessive residual. Apply an analogous guard to Poisson’s computed CG residual at `A/followup/fu_timing.py:179-182`.

- `B/followup/fu_timing.py:47-52,119-127,205-212` loads checkpoints without the configuration and data-fingerprint checks present in `blat_rom.py:100-117`. A wrong N, BC mode, architecture, training recipe, or training draw can silently enter the N/K timing. Validate every checkpoint; in MODE=k also assert all stored fingerprints and POD bases agree before reusing the first checkpoint’s `V`.

- `B/blat_train_ad.py:87-95,135-145` does not vary latent initialization: it is deterministic standardized POD coefficients. Yet `B/followup/cluster/fu_cells.sh:31-33` claims `TRAIN_SEED` changes latent initialization. If latent-init randomness is part of the declared multi-seed protocol, add a seeded POD-init perturbation and rerun all three seeds under that protocol. Keep the perturbation opt-in so the default frozen run remains reproducible, and give even seed 0 a distinct filename when it is enabled. Otherwise remove the latent-init claim and explicitly define the seed sweep as network initialization plus minibatch sampling only.

- `A/followup/fu_pod.py:80-85` implements only full-grid weak objectives, while the coordinate k cells also emit `nnls` and `nnlsoff` results (`A/followup/cluster/fu_cells.sh:55-63`). Comparing an EQ coordinate row against `fu_pod` would violate “same quadrature.” Enforce full-coordinate versus full-POD in the summary/plotting code, or add a POD EQ path using a clearly specified common quadrature protocol.

## SHOULD FIX

- EQ diagnostics can hide a bad row. `A/pro_colloc.py:147-173`, `A/followup/fu_timing.py:84-104`, and `B/blat_common.py:481-504` report only global residual norms after row scaling. Add per-row absolute/relative residual maximum, median, p95, and grouping by snapshot/quantity/mode; abort or warn on a declared threshold.

- `B/blat_common.py:368-375` takes exactly M entries from a stable sort and cuts degenerate eigenshells. At N=64, M=16, 128, and 256 split an `(kx,ky)/(ky,kx)` pair, making the result depend on flattening order and breaking x/y symmetry. Retain complete eigenshells, record the actual `M′`, and use `M′` consistently in timing and rank checks.

- `A/followup/fu_pod.py:83-85` does not record numerical rank or condition number. The direct least-squares answer is the same unregularized minimizer as LM when the matrix has full column rank; otherwise its minimum-norm convention can differ from a damped LM started elsewhere. Record rank/condition and cross-check direct versus LM. For the required M=64, k≤32 ladder the system is overdetermined; the extra k=64/M=64 point is square and should be excluded or labelled.

- `A/followup/fu_timing.py:203-206` and `B/followup/fu_timing.py:92-97` divide total rollout time by Jacobian evaluations, but total work also contains one residual-only evaluation per LM attempt. This is an amortized number, not the isolated cost of a Jacobian iteration. Preserve attempts through the device scan and report both `seconds/attempt` and `seconds/Jacobian`, or benchmark a fixed-work one-attempt kernel.

- Multi-seed jobs are not hardware-controlled: both staging scripts request a generic GPU (`A/.../fu_cells.sh:27-29`, `B/.../make_cell.sh:26-28`), while the B seed-0 checkpoint was trained on an H100. This violates the strict interpretation that only training randomness varies. Pin one GPU model or run all three seeds sequentially in one same-device job.

- Neither harness aggregates exactly seeds 0/1/2 into the promised mean ± sample standard deviation. Add a CPU-only summarizer that validates seed IDs, checkpoint configurations, data fingerprints, test-set identity, row keys, and completeness, then uses `ddof=1`. POD is deterministic here, so its seed standard deviation should be explicitly reported as zero rather than recomputed from nominally different controls.

- `B/blat_rom.py:61-77` returns a finite survivor-only `traj_rel_mean` even if some rollouts blow up. Counts are available, but downstream seed aggregation could overlook them. Make the aggregate null/failing unless `n_completed == n_total`, or define and document a failure penalty.

- `A/followup/fu_timing.py:51,161,190` records `INIT` but always uses the mean latent. Implement the requested initializer or remove the option from the manifest.

- Both launchers overlay an existing remote cell directory (`A/.../launch.sh:13-17`, `B/.../launch.sh:13-17`). A rerun with old outputs/logs will fail checksum verification. After confirming no queued job, recreate the exact cell directory or stage into a fresh run-ID directory.

- Current B staging copies working-tree files but logs only HEAD (`B/.../make_cell.sh:13,39`); the reviewed worktree presently has uncommitted changes in `blat_train_ad.py`, `blat_rom.py`, and `fu_cells.sh`. Refuse to stage a dirty tree or record a dirty marker plus a diff/hash. Also make `B/.../fu_cells.sh:60` an explicit `elif wave2`; a typo currently selects wave2.

- Adding `train_seed` to checkpoint metadata means the default serialized pickle is not literally bit-for-bit identical to the frozen artifact, although the numerical RNG path is unchanged. Phrase the guarantee as bit-identical weights/latents, or compare model arrays rather than file hashes.

## NOTES

- Timing synchronization is otherwise sound. Poisson’s FOM and ROM timed functions block at `A/followup/fu_timing.py:182,191-194`; Burgers blocks FOM, rollout, IC, and decode at `B/followup/fu_timing.py:88-94,138-140,157-170,199-201,220-223`. Matching-shape warm-ups occur before timed repetitions.

- Both N ladders run all N sequentially in one process and request one GPU. The FOM baselines are the same jitted implementations used for truth generation: Poisson’s CG at `A/followup/fu_timing.py:175-182`, and the exact returned Burgers rollout at `B/followup/fu_timing.py:55-63,137-140`.

- Seed separation itself is sound. A builds data through `mp.SEED` and uses `TRAIN_SEED` only at `A/followup/fu_train.py:51-57`; B builds fixed seed-0 training data/fresh `TEST_SEED` test data and uses `TRAIN_SEED` only for network initialization and minibatch/point sampling (`B/blat_train_ad.py:58-65,93-94,135-145`). EQ and evaluation RNGs remain fixed. Current non-default checkpoint and ROM filenames are seed-qualified in both worktrees.

- POD data and metrics are comparable. A’s basis uses exactly the first `N_TRAIN` interior snapshots and evaluates the same first 16 held-out fields in full-grid relative L2 (`A/followup/fu_pod.py:44-66`). B’s `V` is built from every training trajectory/time snapshot (`B/blat_train_ad.py:58-70`), and POD/coordinate paths share `make_step_ops`/`make_weak_ops`. A’s direct weak solve at `fu_pod.py:83-85` is algebraically the same full-grid α=1 minimizer when full rank.

- Cache keys do prevent incorrect reuse within their present processes. A keys on `(m, weak, M, offgrid)` (`pro_colloc.py:101-112`); B’s evaluation key includes decoder kind/rank, weak kind, M, m, and pool (`blat_rom.py:202-220`), while follow-up timing additionally includes N (`B/followup/fu_timing.py:66-81`). B therefore refits EQ for every N, M, m, pool, and decoder rank as required.

- Reusing one Poisson meshfree rule across N is conceptually defensible because the decoder and continuum sine modes are mesh-independent; the defect is specifically that the fitted decoder target is an N=64 grid approximation while the source target changes quadrature with N.

- I found no unintended f32 path in the Stage-2 numerics: both common modules enable x64, POD host arrays are f64, and cluster preflights reject CPU execution. Burgers’ stored true-parameter `z` is f32 upstream, but it is not used by the auto-decoder ROM solve.

- Current shell files pass `bash -n`; command quoting, relative code/output paths, one-cell directories, and GPU preflight structure are otherwise sound. No files were modified and no Python, GPU work, training, submission, or remote command was run.