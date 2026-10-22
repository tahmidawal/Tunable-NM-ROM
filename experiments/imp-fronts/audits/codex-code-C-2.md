**Verdict: WRONG — E10 admission/verdict gates still need fixes.** HEAD `5932bd434`.

- **CORRECT:** Prior floor/scoring dispatch defect fixed; synthetic arm-(c) decoding matches the reference.
- **CORRECT:** Checkpoint architecture guards fixed, including training/rotation consumers; all 16 guard checks pass.
- **CORRECT:** Evaluator tests now use reference-normalised wall RMS and separate lattice-gradient component RMS.
- **CORRECT:** `train_c_item()` accepts the producer’s schema and split projection; enforces 16,440 seconds.
- **CORRECT:** E10 projection now extends K1 paths through 0.125 and includes tighter anchors.
- **CORRECT:** Prior admission probe fixed: oversized/nonfinite projections and missing or altered smoke evidence are rejected.
- **CORRECT:** State counts corrected to 171 initial + 17,600 accepted-step states; rho producer now enforces validation tolerance.
- **CORRECT:** Arm manifest hashes match; full arm-(c) identities, fixed-frequency gradients and K1 control pass.
- **WRONG:** `rom/evidence.py:186–197` accepts the wrong frozen control fixture. A replay-consistent `gauss160`-only control passes although the addendum pins `gauss256`. Wrong rotation hash and precision also pass. Enforce fixture, rotation and precision before admission.
- **WRONG:** `phaseb/assemble.py:150` trusts an unbound `{"passed":true}`. Probe produces E10-s **CONTINUE** without verified control evidence; use the hash-bound verifier.
- **WRONG:** `phaseb/assemble.py:158` ignores rho validation. Setting `validation.passed=false` and relative error to 1 still produces E10-q **STOP**; require validated evidence before deciding.

CPU verification: **169 pytest tests passed in 32.55 s**; standalone `test_arm_c.py`: **0 failures**; `resources.py --selftest`: **OK**.
No repository/main/lab-log writes, jobs or SSH. Scratch occupied **7 MB** before deletion and was deleted; pre-existing HEARTBEAT modification remains.