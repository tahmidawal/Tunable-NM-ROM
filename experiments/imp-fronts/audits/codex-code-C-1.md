**Verdict: WRONG — not ready for admission.** Audited commit `36027235a`; concurrent uncommitted fixes are distinguished below.

- **CORRECT:** Arm (c)’s formula implements Amendments 19–20: odd, 2-periodic, seed-1 `|round(2B0)|`, fixed frequencies, zero gradient, no weight decay. Recipe changes are confined to architecture.
- **CORRECT:** Incumbent feature arithmetic remains unchanged; ARM manifest hashes match. ROM `sep_common` and NumPy reference copies match their training counterparts.
- **CORRECT:** `archrun` pins the intended module for rendered selection jobs; `fpolicy` pins it before engine imports.
- **WRONG:** Committed `phaseb/floors_b.py` never installs NumPy architecture dispatch; scoring inherits this. Synthetic probe: `Kc` discarded, relative value mismatch **1.02015**. Concurrent working-tree patch passes that probe.
- **WRONG:** `sep_common.load_pkl()` accepts conflicting `cfg['arch']`/`Kc`; reproduced with a malformed checkpoint. NumPy/bankguard validation does not protect every training/rotation consumer.
- **CORRECT:** `test_arm_c.py` implements real identity checks, polynomial rejection, and Amendment 21’s reference-normalised K1 control; unavailable evidence fails.
- **WRONG:** `test_arm_c_eval.py:190–192` instead uses maximum wall curvature divided by the **decoded** Laplacian RMS. Its gradient comparison also uses combined sampled-gradient RMS, not separate lattice-component RMS.
- **CORRECT:** SM-c persists paired timing repetitions; `phase_times.py smokec` implements Amendment 21–22’s split projection and 16,440-second cutoff.
- **WRONG:** Committed `resources.train_c_item()` expects a different schema and obsolete projection; it rejects the producer’s output. A concurrent uncommitted repair exists.
- **CORRECT:** E10-s changes the factor set, preserves policy transitions, caps rounds at 11, and never requests a half solve at 0.125. Promotion requires tight certification.
- **WRONG:** E10-s projection uses smoke wall time ÷ cases × 40 × 1.5; smoke disables anchors, and the estimator neither adds anchor work nor implements the registered K1 path-distribution extension.
- **CORRECT:** E8/E10 ledger charges and caps are separated.
- **WRONG:** `admit_e10()` trusts certificate booleans without checking projected cost or binding smoke evidence. A certificate with `projected=99` and three true flags was admitted.
- **CORRECT:** Every-step rho uses the r2acc `Setting`/nonlinearity/`P.rho` path. Artifact hash matches; all **171** prod trajectories are covered; saved output states match exactly.
- **NEEDS-RESTATEMENT:** **17,771** includes 171 initial fits: **17,600 accepted-step states**. Recorded maxima are 0.0386336 overall, 0.0147703 FC1; validation error 2.50e-9. The script records validation errors but never enforces a threshold.

CPU tests: **89 passed in 28.47 s**; requested shell/render syntax checks passed. Full-size `test_arm_c.py` was inspected, not executed. No repository writes, jobs, SSH, or lab-log changes; scratch peaked at 760 KB and was deleted.