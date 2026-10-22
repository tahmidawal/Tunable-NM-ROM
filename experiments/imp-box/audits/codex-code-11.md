**Overall: READY — code audit at `6378e77fb`.** Round-10 blockers are closed. This is code readiness, not acceptance of GPU results.

1. **A10 and Q1 safeguards — CORRECT, with an evidence limitation.**  
   The addendum now excludes only fixed-step FFT N=12; adaptive FFT and dense N=12 remain, alongside added N=10/14 configurations. It no longer claims those additions dominate the excluded rows. See [ladders.py:90](experiments/imp-box/ladders.py#L90).

   Every qualified finalist—FOM and reduced—rebuilds with `cert=True`, replays outputs, and gates agreement at ≤1e-12. These are distinct compiled certificate programs. They detect disagreement, but cannot guarantee detection of faults shared by both programs.

   **Limitation:** diagnostic scripts, narrative and accounting are present; raw diagnostic outputs are absent from this checkout. The reported H200 failure supports the scoped workaround, but I cannot independently verify its numerical evidence or establish compiler code generation as the definitive cause.

2. **Dense discretisation and propagation — CORRECT.**  
   `nonlinear_dense` matches the FFT discretisation: identical padded grid, sine-analysis weights, negative sign, and both spatial derivatives. CPU f64 checks using an IC and random coefficients at every enabled dense size, N=8–128 including 12, gave maximum relative disagreement **4.66e-14**, below 1e-12. This is discretisation equivalence, not bitwise equality.

   `xform` survives ladder rows, selection, Q1 replay, paired timing, components, r_x cost, finer-lattice/coefficient diagnostics, and test evaluation. Adaptive extensions now preserve both configuration and distinct keys. Executing the extension function with mocked FFT and dense finalists produced two correctly labelled extensions.

3. **G-XFORM — CORRECT.**  
   It covers every enabled dense size and participates in mandatory control acceptance. Testing its actual comparison function yielded:

   | Input | Outcome |
   |---|---|
   | Identical outputs | Pass |
   | 1% scaling error | Fail |
   | All-NaN output | Fail |
   | Infinite output | Fail |

4. **Reporting and other changed code — CORRECT within scope.**  
   Production reports no longer substitute smoke cells. FOM descriptions expose `xform`; exports include FOM status and qualifications. Ledger entries sum to the recorded **1.64 GPU-hours**. No additional blocking defect found.

The A10/A11 addendum supersedes the earlier “can only strengthen” and ≤1e-14 wording; those earlier statements should not be quoted independently.

No files changed, jobs submitted, or lab-log entries written. Checks were small and CPU-only; no scratch files were created.