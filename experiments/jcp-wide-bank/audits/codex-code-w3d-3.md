All requested items **PASS**.

| Item | Verdict |
|---|---|
| A6 implementation | Second-order FD on `lat4096[:32]`; fourth-order FD on the converged rule’s first 32 points. Both enforce `<1e-6`; converged-point FD2 remains diagnostic only. |
| Stencil and step | Correct: `[-f(2h)+8f(h)-8f(-h)+f(-2h)]/(12h)`, with `h=1e-5`. Shifting all coordinates checks the `(1,1,1)` directional derivative returned by `point_blocks`. |
| Rules available | Verified both NPZ hashes, duplicate consistency, and required keys for all four configs. Each loads vendored `lat4096` as float64, shape `(4096,3)`, with no `g1_rule` override. J2/smoke use `gl48`; J4/smoke4 use `gl56`. |
| Wrong derivative rejected | A read-only probe injecting a 1% derivative scaling error produced approximately `9.90e-3` on every checked gate, comfortably failing. This remains a sampled directional check, not proof against every possible derivative bug. |
| Change scope | The requested `w3d.py` diff changes only G1 evaluation, its diagnostics, and enforcement. G2–G4 and downstream computation are unchanged. HEAD also adds the A6 DESIGN text. |

A short, in-memory GPU probe with f64 and highest precision confirmed:

| Bank | `lat4096` FD2 | Converged FD4 |
|---|---:|---:|
| M2, R=512 | `1.32e-7` | `2.54e-13` (`gl48`); `4.16e-13` (`gl56`) |
| W1024, R=1024 | `2.17e-7` | `1.36e-12` (`gl56`) |

**The `1e-6` bar is achievable.** At this step size, `eps/h ≈ 2.22e-11`; the fourth-order stencil’s absolute-weight factor gives approximately `3.33e-11 × |f|` absolute roundoff, then divided by the derivative scale for this relative metric. The measured errors confirm ample margin.

No files modified; no lab-log append, per your read-only instruction.

OK-TO-RUN: YES