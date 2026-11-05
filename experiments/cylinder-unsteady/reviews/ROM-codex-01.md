
codex
The basic operator signs and units are consistent. In-memory checks matched direct-quadrature convection and inertia-inclusive reaction forces within \(10^{-14}\). The training architecture, D-scaled gradients, weighting, Adam schedule, child objectives and training-only selection match the registered recipe. However, the following issues prevent full release.

References use `ROM` and `REF` for their respective experiment directories. Readiness below concerns implementation; successful upstream admission remains mandatory.

1. **Blocker — reported errors can be wrong after imperfect orthonormalization.**  
   `ROM/rom_core.py:67–78` retains Gram eigenvalues down to \(10^{-12}\) relative, without verifying the resulting basis is numerically orthonormal. Yet `ref_coeffs():188–190`, `floors():203–206` and `errors():217–219` assume an identity mass matrix.  
   An in-memory retained near-dependency produced a mass-identity defect of \(6.8\times10^{-7}\) and **reported velocity error \(4.77\times10^{-4}\) for exactly identical fields**. This can distort floors and rank selection.  
   **Required:** reorthogonalize and verify the retained basis, or use its actual Gram matrix consistently for projection and norm expansions. Reject unexplained negative squared errors rather than indiscriminately clipping them.

2. **Blocker — R5 crashes on current projection metadata.**  
   `ROM/rom_r5.py:98` slices every projection entry with `v[:1]`. `rom_export.py:299–300` writes `bank_frozen_sha256` as a zero-dimensional array, which cannot be sliced that way.  
   **Required:** select and slice only time-dependent projection arrays.

3. **Blocker for reference admission — R1e checks metadata counts, not a complete qualified population.**  
   `REF/r1de_eval.py:28–44,61–74` does not verify snapshot bytes, hashes, exact 681-time coverage, diagnostic cadence, production configuration or the required five-case population. A supplied subset containing Re100 can receive an overall pass. `:65–71` also accepts a successful last settle without verifying state linkage to recording or cumulative settle duration.  
   **Required:** validate the exact population, configuration, actual arrays/time grids, hashes and settle-to-record provenance. R1f must independently reconstruct the measurements and verify both retained copies; the supplied evaluator does not implement that audit.

4. **Major — held-out reference statistics are exposed before R5.**  
   `REF/r1de_eval.py:47–50,73` exports full cycle means and histories for every supplied case, including held-out Re105. These include frequency and amplitudes beyond the permitted reference-quality admission outputs.  
   **Required:** retain held-out histories sealed; expose only the specified quality checks before R5. The projection guards themselves are correct in `ROM/rom_export.py:295–297` and `rom_project.py:25–27`.

5. **Major — quadrature qualification is incomplete.**  
   `ROM/rom_qcheck.py:26–37` compares raw operators, which correctly avoids orthonormal-basis rotations. It does **not** check the required R2 floor changes or R4 primary-metric changes. It also does not establish that both exports use identical FE coefficient modes and identical trained parameters; supplied evaluated-bank files carry no enforced cross-rule model identity.  
   **Required:** bind the frozen functions across rules, provide evaluation-only replay of the selected trained parameters on order 14, and implement all registered qualification thresholds before ranking.

6. **Major — R5 compares different timed workloads.**  
   `ROM/rom_r5.py:83–107` times the full reference driver against an in-memory reduced rollout. The reference additionally computes surface forces and kinetic energy every step (`REF/transient.py:254–258`), buffers full FE states, and performs persistence/cycle processing. The ROM’s reconstruction is separated, and equivalent persistence is absent.  
   **Required:** isolate equivalent integration/output contracts for both methods and report persistence separately or charge it symmetrically. Explicitly distinguish reduced-state output from reconstructed velocity/pressure output.

7. **Major — R5 lacks accuracy and identity witnesses for timed invocations.**  
   `ROM/rom_r5.py:53–79` does not verify snapshot hashes, case/configuration identity or trained-projection binding as R4 does. `:105–107,130–131` records checksums and repeatability, but no accuracy against the reference for each timed result. Warm-ups at `:111–114` precede potentially lengthy builds and all subsequent blocks, rather than occurring immediately before each block.  
   **Required:** bind inputs, retain same-invocation errors and output witnesses, and implement the registered warm-up protocol. Include all startup Newton iterations; `rom_core.py:169–172` currently retains only the final substep’s count.

8. **Major — R4 reports unqualified target passes and omits energy balance.**  
   `ROM/rom_r4.py:123–126` labels targets met without consuming reference uncertainty or assigning the required “resolved under the linear-propagation estimate” classification. No reduced physical energy balance is computed. `REF/r1de_eval.py:92–103` reads uncertainty inputs and gate results without validating their mutual identity or requiring passing upstream gates.  
   **Required:** bind the uncertainty inputs, propagate resolution classifications into R4/R5 reporting, and implement the registered energy diagnostic.

9. **Major — initial-pressure consistency is conditional on an unchecked constraint defect.**  
   `ROM/rom_core.py:159–170` starts from an unconstrained L2 projection, solves the differentiated continuity system, then enforces continuity during startup. It records \(Ba_0+B_L\), but does not qualify it. A substantial defect therefore causes an unreported startup constraint correction and pressure impulse.  
   **Required:** predeclare how this defect is qualified and reported. Do not silently replace the registered initial projection with a different initialization after results.

10. **Minor — production configuration, dimensions and memory need explicit bookkeeping.**  
    `ROM/rom_export.py:27,238` and `rom_project.py:32–33` still default to P3/order10. Require an explicit, validated P4/order12 production manifest and exact recorded-window selection. Pressure construction correctly retains the constant; supremizers correctly impose homogeneous essential boundaries while leaving the outlet free. However, `rom_core.py:89` calls the **velocity** dimension `d`; total unknowns are `d+dp`, reaching \(3r\) only without rank loss. Report both dimensions and retained ranks. Exported volume tables alone occupy approximately **2.86 GB per million points**, before Arm copies, snapshot matrices and training buffers; retain measured peak-memory evidence.

No additional scaling discrepancy was found in the frozen-bank span, pressure normalization, supremizer sign, force denominator, gradient layout, or saved-time mapping. R2’s floor thresholds and R3 rank-selection rule are implemented correctly, subject to finding 1.

R1d/R1e/R1f NOT READY  
R2 NOT READY  
R3 READY  
R4 NOT READY  
R5 NOT READY
tokens used
457,677
The basic operator signs and units are consistent. In-memory checks matched direct-quadrature convection and inertia-inclusive reaction forces within \(10^{-14}\). The training architecture, D-scaled gradients, weighting, Adam schedule, child objectives and training-only selection match the registered recipe. However, the following issues prevent full release.

References use `ROM` and `REF` for their respective experiment directories. Readiness below concerns implementation; successful upstream admission remains mandatory.

1. **Blocker — reported errors can be wrong after imperfect orthonormalization.**  
   `ROM/rom_core.py:67–78` retains Gram eigenvalues down to \(10^{-12}\) relative, without verifying the resulting basis is numerically orthonormal. Yet `ref_coeffs():188–190`, `floors():203–206` and `errors():217–219` assume an identity mass matrix.  
   An in-memory retained near-dependency produced a mass-identity defect of \(6.8\times10^{-7}\) and **reported velocity error \(4.77\times10^{-4}\) for exactly identical fields**. This can distort floors and rank selection.  
   **Required:** reorthogonalize and verify the retained basis, or use its actual Gram matrix consistently for projection and norm expansions. Reject unexplained negative squared errors rather than indiscriminately clipping them.

2. **Blocker — R5 crashes on current projection metadata.**  
   `ROM/rom_r5.py:98` slices every projection entry with `v[:1]`. `rom_export.py:299–300` writes `bank_frozen_sha256` as a zero-dimensional array, which cannot be sliced that way.  
   **Required:** select and slice only time-dependent projection arrays.

3. **Blocker for reference admission — R1e checks metadata counts, not a complete qualified population.**  
   `REF/r1de_eval.py:28–44,61–74` does not verify snapshot bytes, hashes, exact 681-time coverage, diagnostic cadence, production configuration or the required five-case population. A supplied subset containing Re100 can receive an overall pass. `:65–71` also accepts a successful last settle without verifying state linkage to recording or cumulative settle duration.  
   **Required:** validate the exact population, configuration, actual arrays/time grids, hashes and settle-to-record provenance. R1f must independently reconstruct the measurements and verify both retained copies; the supplied evaluator does not implement that audit.

4. **Major — held-out reference statistics are exposed before R5.**  
   `REF/r1de_eval.py:47–50,73` exports full cycle means and histories for every supplied case, including held-out Re105. These include frequency and amplitudes beyond the permitted reference-quality admission outputs.  
   **Required:** retain held-out histories sealed; expose only the specified quality checks before R5. The projection guards themselves are correct in `ROM/rom_export.py:295–297` and `rom_project.py:25–27`.

5. **Major — quadrature qualification is incomplete.**  
   `ROM/rom_qcheck.py:26–37` compares raw operators, which correctly avoids orthonormal-basis rotations. It does **not** check the required R2 floor changes or R4 primary-metric changes. It also does not establish that both exports use identical FE coefficient modes and identical trained parameters; supplied evaluated-bank files carry no enforced cross-rule model identity.  
   **Required:** bind the frozen functions across rules, provide evaluation-only replay of the selected trained parameters on order 14, and implement all registered qualification thresholds before ranking.

6. **Major — R5 compares different timed workloads.**  
   `ROM/rom_r5.py:83–107` times the full reference driver against an in-memory reduced rollout. The reference additionally computes surface forces and kinetic energy every step (`REF/transient.py:254–258`), buffers full FE states, and performs persistence/cycle processing. The ROM’s reconstruction is separated, and equivalent persistence is absent.  
   **Required:** isolate equivalent integration/output contracts for both methods and report persistence separately or charge it symmetrically. Explicitly distinguish reduced-state output from reconstructed velocity/pressure output.

7. **Major — R5 lacks accuracy and identity witnesses for timed invocations.**  
   `ROM/rom_r5.py:53–79` does not verify snapshot hashes, case/configuration identity or trained-projection binding as R4 does. `:105–107,130–131` records checksums and repeatability, but no accuracy against the reference for each timed result. Warm-ups at `:111–114` precede potentially lengthy builds and all subsequent blocks, rather than occurring immediately before each block.  
   **Required:** bind inputs, retain same-invocation errors and output witnesses, and implement the registered warm-up protocol. Include all startup Newton iterations; `rom_core.py:169–172` currently retains only the final substep’s count.

8. **Major — R4 reports unqualified target passes and omits energy balance.**  
   `ROM/rom_r4.py:123–126` labels targets met without consuming reference uncertainty or assigning the required “resolved under the linear-propagation estimate” classification. No reduced physical energy balance is computed. `REF/r1de_eval.py:92–103` reads uncertainty inputs and gate results without validating their mutual identity or requiring passing upstream gates.  
   **Required:** bind the uncertainty inputs, propagate resolution classifications into R4/R5 reporting, and implement the registered energy diagnostic.

9. **Major — initial-pressure consistency is conditional on an unchecked constraint defect.**  
   `ROM/rom_core.py:159–170` starts from an unconstrained L2 projection, solves the differentiated continuity system, then enforces continuity during startup. It records \(Ba_0+B_L\), but does not qualify it. A substantial defect therefore causes an unreported startup constraint correction and pressure impulse.  
   **Required:** predeclare how this defect is qualified and reported. Do not silently replace the registered initial projection with a different initialization after results.

10. **Minor — production configuration, dimensions and memory need explicit bookkeeping.**  
    `ROM/rom_export.py:27,238` and `rom_project.py:32–33` still default to P3/order10. Require an explicit, validated P4/order12 production manifest and exact recorded-window selection. Pressure construction correctly retains the constant; supremizers correctly impose homogeneous essential boundaries while leaving the outlet free. However, `rom_core.py:89` calls the **velocity** dimension `d`; total unknowns are `d+dp`, reaching \(3r\) only without rank loss. Report both dimensions and retained ranks. Exported volume tables alone occupy approximately **2.86 GB per million points**, before Arm copies, snapshot matrices and training buffers; retain measured peak-memory evidence.

No additional scaling discrepancy was found in the frozen-bank span, pressure normalization, supremizer sign, force denominator, gradient layout, or saved-time mapping. R2’s floor thresholds and R3 rank-selection rule are implemented correctly, subject to finding 1.

R1d/R1e/R1f NOT READY  
R2 NOT READY  
R3 READY  
R4 NOT READY  
R5 NOT READY
