Read-only audit of `962be67ed`; no files changed, GPU jobs run, or pushes made. References are relative to `experiments/ns3d-coordnet/`. Items 1–10 correspond to pass 1.

1. **FIXED — Fourier frequencies.** `train_bank.py:179` zeros `B` before clipping and Adam; both optimizer states start fresh (`195`, `211`). Zero moments remain zero, and zero leaves contribute nothing to the clipping norm. CPU check confirmed exact preservation. Save/load assertions exist at `bankio.py:39,55`.

2. **PARTIAL — objective certification.** Raw/projected QR losses and a singular-value ratio are recorded (`train_bank.py:230–235`). **NEW:** `coordnet.py:131–135` uses every QR column even when the bank is rank-deficient, including arbitrary orthogonal completion directions. CPU counterexample: reported loss **0**, actual loss **1**. **Fix:** use a rank-revealing SVD projector or reject deficient banks; persist singular values and explicit ridge bias.

3. **NOT FIXED — cross-mesh equivalence gate.** `train_bank.py:335–341` still only records column discrepancies; no threshold blocks acceptance. **Fix:** define a finite discrepancy threshold and propagate failure into the bank acceptance status.

4. **FIXED — resampling validation.** `bankio.py:22–25` validates even cubic meshes and explicitly identifies the same-size Nyquist-preserving shortcut. Put that exception in the docstring too.

5. **PARTIAL — Löwdin/prefix guards.** Full normalization followed by `Gc[:, :prefix]` and `lowdin_S[:, :prefix]` is correct (`rom_mesh.py:178–185`); `Tc @ lowdin_S` gives the matching derivative columns (`211`). Eigenvalue and entrywise-abort guards exist (`bankio.py:64,83`). Spectral distortion is reported but not gated (`82,90`). **Fix:** gate spectral/column distortion and require integer `1 <= prefix <= parent_rank`; oversized and negative prefixes currently slice silently.

6. **PARTIAL — weighting, calibration, parent reproduction.** Native-mesh normalization and projected POD references are fixed (`train_bank.py:112,288,323`). Calibration now includes clipping, but uses constant-rate Adam and a separate compilation (`193–212`). **NEW:** the `4*warmup` minimum (`203`) can override the entire time budget; calibration, certification, and downstream floors remain outside that allowance. Parent reproduction failure merely produces `status="flagged"` and exits zero (`305–311,351–354`). **Fix:** budget all phases, reject infeasible warmup budgets, and make reproduction failure a nonzero hard failure.

7. **PARTIAL — ROM gates/freezing.** Comparator selection is **correct**: minimum same-job median among stable, sufficiently accurate CNAB2 rows, with eligible rows persisted (`rom_mesh.py:485–502`). Timing bounds, prefix support, and listed frozen-setting checks are implemented (`134–137,463–464`). Remaining holes:
   - Timed NaNs can disappear through `max(agree, nan)` (`474`); LM `max(gaps)` can also hide later NaNs (`411–412`). Require explicit finiteness.
   - Derivative disagreement only flags, and NaN passes its comparison (`216–217`).
   - Candidate arrays remain captured by parent `head_floor_errors` (`../ns3d-shift-head/head_rom.py:319–326`), called at `rom_mesh.py:295,315`. Pass them explicitly.
   - Horizon/truth-generation settings remain mutable (`90–91,149,199`). Freeze the complete evaluation contract.

8. **PARTIAL — independent verifier.** Required filenames/dtypes match actual rollout, frame-zero, and CNAB2 saves (`verify_coordnet.py:33–43`; `rom_mesh.py:367,393,427`). Missing files now fail. **Residual holes, CPU-reproduced:** NaN summary statistics pass through `max` (`verify_coordnet.py:62–63`), and wrongly shaped claimed errors can broadcast (`56–61`). Floors and most reported statistics remain unaudited. **Fix:** validate exact error-array shape, finite statistics, all claimed reductions, and expected arm identities from configuration; add floor evidence.

9. **PARTIAL — decisions/design.** §A2 corrects the lower-bound claim, supplies stop outcomes/tiebreaks, and discloses prior test exposure (`DESIGN.md:291–306`). Contradictory claims remain at `182–190,225–226`; development selection conditioning is not clearly labelled, and no consolidated validator enforces bars (a)–(c). **Fix:** replace superseded wording and implement deterministic selection/bar validation.

10. **PARTIAL — cluster propagation/staging/pull.**
    - **Exit 3 works:** it still runs verification and makes the job fail (`job.sbatch:28–37,45–51`).
    - **NEW shell hole:** invoking `one_mesh ... || RC=1` suppresses `errexit` throughout the function; failed hashing/cleanup at `34–35` need not affect its return. Check those commands explicitly.
    - `pull.sh:30–31` accepts any successful-looking exit record without requiring terminal `ALL_EXIT=0`, every expected test mesh, or passing `verify.json` contents. Bank mode supplies synthetic `VERIFY_EXIT=0` (`job.sbatch:43`), including flagged bank results. Require mode-specific complete records and validated summaries/audits.
    - Nested manifests are checked when present, but missing ones are explicitly tolerated (`pull.sh:19,25`). Require the expected manifest set.
    - Explicit bank staging is fixed (`submit.sh:20–25`), but newly uses forbidden system `python3` (`21`). Use the mandated absolute interpreter, verify staged hashes, and preserve a complete input manifest; `COMMIT.txt` alone remains insufficient (`30`).

11. **NEW — local floor checker is not configuration-faithful.** `local_floor_check.py:59–72` defaults case count and hardcodes timestep/horizon; `81–90` always centres and expects centred-bank keys, so the uncentred control is unsupported. Failed comparison still exits zero (`93–96`). **Fix:** derive settings and mode from the run config and exit nonzero on disagreement.

12. **Pass-1 additional findings:** rank 192 is fixed (`make_configs.py:44`); compression/pilot/probe disclosures are added (`DESIGN.md:304–307`). Streaming verification remains unimplemented: fields accumulate through all rollouts and timings before auditing (`rom_mesh.py:367,427`; `job.sbatch:31`).

Python/shell syntax checks passed. Validation used only short CPU and shell checks.